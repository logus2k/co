import os, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric

DEVICE       = "cuda"
SCENE_DIR    = "/home/logus/env/iscte/co/data/nerf_synthetic/lego"
N_SAMPLES    = 64
LR           = 5e-4
SEED         = 0
NEAR, FAR    = 2.0, 6.0
MLP_WIDTH    = 256
MLP_DEPTH    = 8
MLP_SKIP     = 4               # concat encoded input before this layer (standard NeRF)
N_TEST_VIEWS = 3
TARGET_PSNR  = 25.0

SANITY       = os.environ.get("SANITY", "0") == "1"
N_ITERATIONS = 3000 if SANITY else 40000
EVAL_EVERY   = 1000 if SANITY else 2000
CONFIGS      = [(200, 4096)] if SANITY else [(200, 4096), (800, 4096)]
print(f"MODE: {'SANITY CHECK' if SANITY else 'FULL RUN'}  |  {N_ITERATIONS} iters  |  configs {CONFIGS}", flush=True)

def load_split(scene_dir, split, max_views=None):
    with open(os.path.join(scene_dir, f"transforms_{split}.json")) as f:
        meta = json.load(f)
    frames = meta["frames"][:max_views] if max_views else meta["frames"]
    imgs, poses = [], []
    for fr in frames:
        rgba = np.asarray(Image.open(os.path.join(scene_dir, fr["file_path"]+".png")), dtype=np.float32)/255.0
        imgs.append(rgba[..., :3]*rgba[..., 3:4] + (1.0-rgba[..., 3:4]))
        poses.append(np.asarray(fr["transform_matrix"], dtype=np.float32))
    images = torch.from_numpy(np.stack(imgs)); poses = torch.from_numpy(np.stack(poses))
    focal = 0.5*images.shape[2]/np.tan(0.5*float(meta["camera_angle_x"]))
    return images, poses, float(focal)

def downscale(images, res):
    x = images.permute(0,3,1,2)
    x = F.interpolate(x, size=(res,res), mode="area")
    return x.permute(0,2,3,1).contiguous()

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.freqs = 2.0 ** torch.arange(num_freqs).float().to(DEVICE)
        self.out_dim = 3 + 3*2*num_freqs
    def forward(self, x):
        encs = [x]
        for f in self.freqs:
            encs += [torch.sin(f*x), torch.cos(f*x)]
        return torch.cat(encs, dim=-1)

class NeRFMLP(nn.Module):
    """Standard NeRF MLP: feedforward with a skip connection that re-injects the
    encoded input before layer `skip`. This is what makes depth 8 trainable."""
    def __init__(self, enc_dim, width=256, depth=8, skip=4):
        super().__init__()
        self.skip = skip
        self.layers = nn.ModuleList()
        for i in range(depth):
            in_dim = enc_dim if i == 0 else (width + enc_dim if i == skip else width)
            self.layers.append(nn.Linear(in_dim, width))
        self.density = nn.Linear(width, 1)
        self.rgb = nn.Linear(width, 3)
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                h = torch.cat([h, x], dim=-1)
            h = torch.relu(layer(h))
        return torch.relu(self.density(h))[..., 0], torch.sigmoid(self.rgb(h))

def get_rays(H, W, focal, pose):
    i, j = torch.meshgrid(torch.arange(W, device=DEVICE).float(),
                          torch.arange(H, device=DEVICE).float(), indexing="xy")
    dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], dim=-1)
    return pose[:3,3].expand(H,W,3), dirs @ pose[:3,:3].T

def render_rays(rays_o, rays_d, model, encoding, near=NEAR, far=FAR, N=N_SAMPLES):
    t = torch.linspace(near, far, N, device=DEVICE)
    delta = (far-near)/N
    t = t + (torch.rand(rays_o.shape[0], N, device=DEVICE)-0.5)*delta
    pts = rays_o[:,None] + rays_d[:,None]*t[:,:,None]
    sigma, c = model(encoding(pts.reshape(-1,3)))
    sigma = sigma.reshape(rays_o.shape[0], N); c = c.reshape(rays_o.shape[0], N, 3)
    deltas = torch.cat([t[:,1:]-t[:,:-1], torch.full_like(t[:,:1], 1e10)], dim=-1)
    alpha = 1.0 - torch.exp(-sigma*deltas)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha+1e-10], dim=-1), dim=-1)[:,:-1]
    return ((T*alpha)[...,None]*c).sum(dim=1)

class MyAdam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params=[p for p in params if p.requires_grad]
        self.lr,self.b1,self.b2,self.eps=lr,b1,b2,eps
        self.m=[torch.zeros_like(p) for p in self.params]
        self.v=[torch.zeros_like(p) for p in self.params]; self.t=0
    @torch.no_grad()
    def step(self):
        self.t+=1; bc1,bc2=1-self.b1**self.t,1-self.b2**self.t
        for p,m,v in zip(self.params,self.m,self.v):
            if p.grad is None: continue
            g=p.grad
            m.mul_(self.b1).add_(g,alpha=1-self.b1)
            v.mul_(self.b2).addcmul_(g,g,value=1-self.b2)
            p.addcdiv_(m/bc1, v.div(bc2).sqrt().add_(self.eps), value=-self.lr)
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.detach_(); p.grad.zero_()

@torch.inference_mode()
def render_full(model, encoding, pose, focal, H, W, chunk=8192):
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o, rays_d = rays_o.reshape(-1,3), rays_d.reshape(-1,3)
    out=[render_rays(rays_o[i:i+chunk], rays_d[i:i+chunk], model, encoding)
         for i in range(0, rays_o.shape[0], chunk)]
    return torch.cat(out).reshape(H,W,3)

def evaluate(model, encoding, test_images, test_poses, focal, H, W):
    ps,ss=[],[]
    for gt_t,pose in zip(test_images,test_poses):
        pred=render_full(model,encoding,pose,focal,H,W).clamp(0,1).cpu().numpy()
        gt=gt_t.cpu().numpy(); mse=float(((pred-gt)**2).mean())
        ps.append(-10*np.log10(max(mse,1e-10)))
        ss.append(float(ssim_metric(pred,gt,channel_axis=-1,data_range=1.0)))
    return float(np.mean(ps)), float(np.mean(ss))

print("Loading Lego...", flush=True)
train_full, train_poses_cpu, native_focal = load_split(SCENE_DIR, "train")
test_full,  test_poses_cpu,  _            = load_split(SCENE_DIR, "test", max_views=N_TEST_VIEWS)
native_W = train_full.shape[2]

summary=[]
for res,batch in CONFIGS:
    cov = batch*N_ITERATIONS/(len(train_full)*res*res)
    print(f"\n=== {res}x{res}  batch={batch}  (coverage {cov:.2f}x) ===", flush=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    train_images=downscale(train_full,res).to(DEVICE)
    test_images=downscale(test_full,res).to(DEVICE)
    train_poses=train_poses_cpu.to(DEVICE); test_poses=test_poses_cpu.to(DEVICE)
    H=W=res; focal=native_focal*res/native_W
    torch.manual_seed(SEED); np.random.seed(SEED)
    encoding=PositionalEncoding(10).to(DEVICE)
    model=NeRFMLP(encoding.out_dim, width=MLP_WIDTH, depth=MLP_DEPTH, skip=MLP_SKIP).to(DEVICE)
    n_params=sum(p.numel() for p in model.parameters())
    print(f"  NeRFMLP width={MLP_WIDTH} depth={MLP_DEPTH} skip@{MLP_SKIP}: {n_params:,} params", flush=True)
    opt=MyAdam(list(model.parameters()), lr=LR)
    curve=[]; eval_time=0.0
    torch.cuda.synchronize(); t0=time.time()
    for it in range(1,N_ITERATIONS+1):
        idx=np.random.randint(len(train_images))
        rays_o,rays_d=get_rays(H,W,focal,train_poses[idx])
        rays_o,rays_d=rays_o.reshape(-1,3),rays_d.reshape(-1,3)
        target=train_images[idx].reshape(-1,3)
        pix=torch.randint(0,H*W,(batch,),device=DEVICE)
        pred=render_rays(rays_o[pix],rays_d[pix],model,encoding)
        loss=((pred-target[pix])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it%EVAL_EVERY==0:
            torch.cuda.synchronize(); te=time.time()
            psnr,ssim=evaluate(model,encoding,test_images,test_poses,focal,H,W)
            torch.cuda.synchronize(); eval_time+=time.time()-te
            wall=time.time()-t0; curve.append((it,wall,psnr,ssim))
            print(f"  iter {it:6d}  PSNR {psnr:6.2f}  SSIM {ssim:.3f}  elapsed {wall:7.1f}s", flush=True)
    torch.cuda.synchronize(); total=time.time()-t0
    per_iter_ms=1000*(total-eval_time)/N_ITERATIONS
    vram=torch.cuda.max_memory_allocated()/1e6
    best_p=max(c[2] for c in curve); best_s=max(c[3] for c in curve)
    summary.append((res,batch,cov,per_iter_ms,vram,best_p,best_s,total,n_params))
    del train_images,test_images,model,encoding,opt; torch.cuda.empty_cache()

print("\n\n========== SUMMARY (NeRFMLP w/ skip, width 256 depth 8) ==========")
print(f"{'res':>5} {'batch':>6} {'cov':>6} {'ms/it':>7} {'VRAM_MB':>9} {'bestPSNR':>9} {'bestSSIM':>9} {'run_s':>8} {'params':>9}")
for (res,batch,cov,ms,vram,bp,bs,tot,npar) in summary:
    print(f"{res:>5} {batch:>6} {cov:>5.2f}x {ms:>7.2f} {vram:>9.1f} {bp:>9.2f} {bs:>9.4f} {tot:>8.1f} {npar:>9,}")
if not SANITY:
    print("\nReference (small MLP width=128 depth=4, batch 4096, 40k iters):")
    print("  200x200: best PSNR 23.08 / SSIM 0.856   |   800x800: best PSNR 21.68 / SSIM 0.786")
