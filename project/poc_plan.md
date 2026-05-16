# Proof of Concept Plan: NeRF Reconstruction with Custom Adam

## 1. Goal of the PoC

Produce a single recognisable rendered novel view of a 3D scene (the NeRF-Synthetic "Lego" bulldozer) using:

- A compact NeRF model and volume-rendering pipeline you implement yourselves.
- An Adam optimizer you implement yourselves on top of PyTorch's autograd.

If by the end of the PoC you can show a side-by-side of a training photo and a rendered novel view that look like the same Lego from different angles, the full project will work. If not, you know what to fix before committing.

**Time budget: 4 to 6 hours total**, split between the two students.

---

## 2. What is in scope (and what is deliberately not)

### In scope for the PoC

- One synthetic scene (Lego), low resolution (200x200 pixels).
- One tiny NeRF model (a few hundred lines of code).
- One optimizer (Adam) implemented from scratch.
- One loss function (L2).
- One training run with a fixed seed.
- A few rendered novel views and a loss curve.

### Deliberately out of scope (left for the full project)

- COLMAP and self-captured scenes.
- Multiple optimizers and loss formulations.
- Hyperparameter sweeps, multiple seeds, statistical comparisons.
- Improvements (multi-scale training, adaptive sampling, learning-rate restarts).
- Gaussian Splatting comparison.
- High-resolution training, hierarchical sampling, view-dependent effects.

The PoC is the smallest experiment that proves the optimization pipeline is sound.

---

## 3. Success criteria

The PoC succeeds if **all three** of these hold:

1. The training loss decreases by at least two orders of magnitude over training.
2. The rendered novel view is recognisable as the Lego bulldozer (not noise, not a blurry blob).
3. PSNR on a held-out test view is above 20 dB.

If any of these fails, debug before committing to the full project.

---

## 4. Environment setup (30 minutes)

```bash
cd /home/logus/env/iscte/co/project
mkdir -p poc
cd poc
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib imageio tqdm
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Expected last line: `CUDA: True`.

If CUDA is not available, stop and resolve before proceeding (WSL2 plus CUDA can require driver setup).

---

## 5. Get the data (5 minutes)

We will use the **Tiny NeRF preprocessed dataset** (`tiny_nerf_data.npz`), published by Mildenhall alongside the Tiny NeRF tutorial. It contains 106 images of the Lego scene at 100x100 resolution, the corresponding camera-to-world matrices, and the focal length, all packaged into a single ~10 MB numpy file.

The file is hosted by the NeRF authors at UC San Diego (this is the URL used by the official Tiny NeRF notebook in the `bmild/nerf` repository):

```bash
mkdir -p data
cd data
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
cd ..
```

Verify:

```bash
python -c "import numpy as np; d = np.load('data/tiny_nerf_data.npz'); print({k: d[k].shape for k in d.files})"
# Expected output (something like):
# {'images': (106, 100, 100, 3), 'poses': (106, 4, 4), 'focal': ()}
```

If the shapes match (around 100 images, 100x100x3, plus a 4x4 pose per image and a scalar focal length), you're set.

---

## 6. PoC notebook structure

Create `poc/poc_nerf.ipynb`. The notebook is organised into the cells below.

### Cell 1: Imports and configuration

```python
import json, os, math
import torch
import torch.nn as nn
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

device = "cuda"
DATA_DIR = "data/nerf_synthetic/lego"
RES = 200             # downsampled resolution
N_SAMPLES = 64        # samples per ray
BATCH_RAYS = 1024
N_ITERATIONS = 5000
LR = 5e-4
```

### Cell 2: Data loading

A function that reads `transforms_train.json`, loads each image, downsamples to `RES`, and returns `(images, poses, focal_length)`. This is dataset-specific boilerplate, roughly 30 lines.

### Cell 3: Ray generation

Given a camera pose and intrinsics, produce ray origins and directions for every pixel. Roughly 15 lines.

```python
def get_rays(H, W, focal, pose):
    """Return ray origins and directions for every pixel in an HxW image."""
    i, j = torch.meshgrid(
        torch.arange(W, device=device).float(),
        torch.arange(H, device=device).float(),
        indexing="xy",
    )
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], dim=-1)
    rays_d = (dirs @ pose[:3, :3].T)
    rays_o = pose[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d
```

### Cell 4: Positional encoding

Maps a 3D point to a higher-dimensional vector using sines and cosines at increasing frequencies. This is what lets the small MLP fit high-frequency detail.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.freqs = 2.0 ** torch.arange(num_freqs).float().to(device)
        self.out_dim = 3 + 3 * 2 * num_freqs

    def forward(self, x):
        encs = [x]
        for f in self.freqs:
            encs += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(encs, dim=-1)
```

### Cell 5: NeRF MLP

A small MLP taking the encoded point and producing a density (non-negative scalar) and an RGB colour (in [0, 1]).

```python
class TinyNeRF(nn.Module):
    def __init__(self, enc_dim, width=128, depth=4):
        super().__init__()
        layers = [nn.Linear(enc_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.density = nn.Linear(width, 1)
        self.rgb = nn.Linear(width, 3)

    def forward(self, x):
        h = self.trunk(x)
        sigma = torch.relu(self.density(h))[..., 0]
        c = torch.sigmoid(self.rgb(h))
        return sigma, c
```

### Cell 6: Volume rendering

Sample `N_SAMPLES` points along each ray, query the model, composite with the volume-rendering integral. Roughly 25 lines.

```python
def render_rays(rays_o, rays_d, model, encoding, near=2.0, far=6.0, N=N_SAMPLES):
    t = torch.linspace(near, far, N, device=device)
    delta = (far - near) / N
    t = t + (torch.rand(rays_o.shape[0], N, device=device) - 0.5) * delta
    pts = rays_o[:, None] + rays_d[:, None] * t[:, :, None]
    sigma, c = model(encoding(pts.reshape(-1, 3)))
    sigma = sigma.reshape(rays_o.shape[0], N)
    c = c.reshape(rays_o.shape[0], N, 3)
    deltas = torch.cat([t[:, 1:] - t[:, :-1], torch.full_like(t[:, :1], 1e10)], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    w = T * alpha
    rgb = (w[..., None] * c).sum(dim=1)
    return rgb
```

### Cell 7: Custom Adam optimizer

Implemented from scratch. This is one of the central CO contributions of the project.

```python
class MyAdam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.addcdiv_(m / bc1, v.div(bc2).sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
```

### Cell 8: Training loop

```python
encoding = PositionalEncoding(num_freqs=10).to(device)
model = TinyNeRF(enc_dim=encoding.out_dim).to(device)
opt = MyAdam(list(model.parameters()) + list(encoding.parameters()), lr=LR)

images, poses, focal = load_lego_train(DATA_DIR, RES)  # from Cell 2

losses = []
for it in tqdm(range(N_ITERATIONS)):
    img_idx = np.random.randint(len(images))
    target = images[img_idx]
    pose = poses[img_idx]
    rays_o, rays_d = get_rays(RES, RES, focal, pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    target_flat = target.reshape(-1, 3)

    pix = torch.randint(0, RES * RES, (BATCH_RAYS,), device=device)
    pred = render_rays(rays_o[pix], rays_d[pix], model, encoding)
    loss = ((pred - target_flat[pix]) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
```

### Cell 9: Plot the loss curve

```python
plt.figure(figsize=(8, 4))
plt.semilogy(losses)
plt.xlabel("Iteration"); plt.ylabel("L2 loss (log)")
plt.title("Training loss with custom Adam on Tiny NeRF")
plt.show()
```

Expected: the curve drops by roughly two orders of magnitude.

### Cell 10: Render a novel view

```python
# Load a held-out test pose
test_poses = load_test_poses(DATA_DIR)
pose = test_poses[0]
rays_o, rays_d = get_rays(RES, RES, focal, pose)

# Render in batches to avoid OOM
with torch.no_grad():
    pred = render_in_batches(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), model, encoding, batch_size=4096)
img = pred.reshape(RES, RES, 3).cpu().numpy()

plt.figure(figsize=(6, 6))
plt.imshow(np.clip(img, 0, 1))
plt.title("Novel view rendered by Tiny NeRF (custom Adam)")
plt.axis("off")
plt.show()
```

Expected: a recognisable Lego bulldozer from a viewpoint not in the training set.

### Cell 11: Side-by-side check

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(load_test_image(DATA_DIR, 0)); axes[0].set_title("Ground truth")
axes[1].imshow(np.clip(img, 0, 1));            axes[1].set_title("Rendered")
for a in axes: a.axis("off")
plt.show()
```

Compute PSNR between the two:

```python
mse = ((img - gt) ** 2).mean()
psnr = -10 * np.log10(mse)
print(f"PSNR: {psnr:.2f} dB")
```

Expected: PSNR above 20 dB.

---

## 7. Work split between the two students

The PoC is small enough that you can pair-program it in one sitting, but if you prefer to split:

- **Student A:** Cells 1, 2, 3, 8 (data loading, ray generation, training loop).
- **Student B:** Cells 4, 5, 6, 7 (positional encoding, MLP, volume rendering, custom Adam).
- **Both:** Cells 9, 10, 11 (visualisations) and joint debugging.

---

## 8. Common pitfalls and how to handle them

- **Loss does not decrease.** Check that the positional encoding is being applied. Check that learning rate is in the range 1e-4 to 1e-3.
- **Rendered image is grey or uniform.** Density values may be saturating. Verify the ReLU on density and the sigmoid on colour are present and applied correctly.
- **Training is very slow.** Lower `RES` to 100 or `N_SAMPLES` to 32 for the first debug run. Once it converges at low resolution, scale up.
- **CUDA out of memory.** Lower `BATCH_RAYS` or use rendering-in-batches for novel views.
- **Loss decreases but image is blurry.** Increase iterations to 10000, or add more positional encoding frequencies (12 instead of 10).

---

## 9. After the PoC

If the success criteria are met, you have demonstrated:

1. A working volume-rendering pipeline you implemented.
2. A working custom Adam that successfully trained the model.
3. A novel-view rendering that proves the optimization recovered the scene.

From here, the full project just adds: more optimizers (SGD, momentum, Nesterov, AdamW), more losses (L1, SSIM, hybrids), comparison plots across seeds and learning rates, the planned improvements (LR restarts, adaptive ray sampling, multi-scale training), and the Gaussian Splatting comparison runs. None of these add fundamentally new infrastructure; they reuse the pipeline you built in the PoC.

If the PoC fails at any step, the debugging effort itself is informative: it tells you what part of the pipeline needs the most attention in the full project. Either way, you finish the PoC knowing whether the plan is realistic.
