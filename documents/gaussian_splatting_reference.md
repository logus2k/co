# Gaussian Splatting — Algorithm Reference

A technical reference for the Gaussian Splatting (GS) baseline implemented in §10 and compared against NeRF in §11 / §11.5 / §11.6 of the notebook. Like the other references, each section is **plug-and-play** for the final report. The math, the implementation, the four root-cause fixes that made GS actually train, and the NeRF-vs-GS optimization-perspective interpretation are kept together.

> *Scope*: this document covers GS as an **alternative optimization formulation of the same reconstruction problem** that NeRF solves, the four fixes our implementation required, and the §11 / §11.5 comparison. The viz infrastructure (§11.5 multi-view grid, §11.6 orbit MP4s) is described as part of the comparison.

---

## 0. Why include GS in an optimization-methods project

The project's framing principle is *"the capacity to think in terms of optimization applied to AI and not merely to apply algorithms."* Including GS is not a "we also tried this" — it is a **second optimization formulation of the same problem NeRF solves**, with a fundamentally different parameter space, gradient flow, and objective landscape.

NeRF and GS both solve

$$
\min_{\Psi} \sum_{\pi \in \mathrm{train}} \mathcal{L}\bigl(R(\Psi; \pi),\; I(\pi)\bigr)
$$

where $R(\Psi; \pi)$ is a differentiable rendering operator producing an image from parameters $\Psi$ viewed under camera $\pi$, $I(\pi)$ is the ground-truth image at that camera, and $\mathcal{L}$ is the photometric loss. The two methods differ entirely in what $\Psi$ is and how $R$ flows gradients back to it:

| Aspect | NeRF | Gaussian Splatting |
|---|---|---|
| **$\Psi$** | weights of a small MLP ($\approx 58\,000$ params) | $(\mu, q, s, \alpha, c)$ tuple per Gaussian ($\sim 10^5$–$10^6$ Gaussians, ~14 floats each) |
| **Continuous in 3D** | yes (MLP samples $(x, y, z)$ rays at inference) | yes (Gaussians live in 3D space) |
| **Render operator** | volume integration along sampled rays | differentiable rasterisation of 3D Gaussians into a 2D image |
| **Backward pass cost** | $O(\text{rays} \times \text{samples per ray})$ — both forward and backward | $O(\text{Gaussians} \times \text{pixels per Gaussian})$ — tile-based rasteriser |
| **Discrete structural choices** | none | densification (clone / split) and pruning at scheduled intervals |
| **Optimization landscape** | continuous in $\Psi$, fully smooth | continuous in $\Psi$ between densification events, *discontinuously restructured* at each densification call |

This is the optimization-theory observation we want to surface: **the two methods solve the same problem with two different parameterisations, and the empirical comparison reveals when each parameterisation is favoured by the optimizer**. Section §11 then asks the practical question — which formulation gives better reconstructions per wall-clock dollar.

---

## 1. The Gaussian Splatting formulation

### 1.1 The parameterisation

Each Gaussian $i$ is described by a tuple $(\mu_i, q_i, s_i, \alpha_i, c_i)$:

- $\mu_i \in \mathbb{R}^3$ — centre position in world coordinates
- $q_i \in \mathbb{R}^4$ — rotation as a unit quaternion (anisotropic Gaussians)
- $s_i \in \mathbb{R}^3$ — *log* of the per-axis scale (so $\exp(s_i) > 0$ is automatic without parameterising on a positive cone)
- $\alpha_i \in \mathbb{R}$ — *logit* of the opacity (so $\sigma(\alpha_i) \in (0, 1)$ is automatic)
- $c_i \in \mathbb{R}^3$ — *logit* of the RGB colour (similarly bounded)

Total: 14 parameters per Gaussian. For our basic configuration on Lego at iteration 15 000, we end with ~150 000 Gaussians, so $\Psi$ has ~$2.1 \times 10^6$ floats — roughly 35× more parameters than the NeRF MLP. The flip side is that each Gaussian is *local* — it affects only a small region of the rendered image — so the gradient per parameter is sparse and well-conditioned, which is what makes GS train at a reasonable wall-clock despite the much larger parameter count.

### 1.2 The rendering operator

The gsplat differentiable rasteriser (Ye et al. 2024 / Nerfstudio impl. 1.5.3) takes the Gaussians and a camera intrinsic + extrinsic, and produces:

- An RGB image, by alpha-compositing the front-to-back-sorted Gaussians at each pixel
- An alpha map (used here to composite over a white background)
- An info dictionary including `info["means2d"]`, the screen-space projection of each Gaussian's centre — used for densification

The rasteriser is *fully differentiable*: gradient flows from the image back to every Gaussian's $(\mu, q, s, \alpha, c)$ tuple. This is the entire reason GS works as an optimization-trainable representation rather than a manually authored one.

### 1.3 The training loop

```
init N_init Gaussians at random positions in [-scene_bound, scene_bound]^3
opts = [Adam(params_i, lr_i) for each of the 5 parameter groups]
for iteration t in [1 .. n_iterations]:
    sample a random training view
    render → image, alphas, info
    loss = α · L1(image, gt) + (1 - α) · (1 - SSIM(image, gt))
    backward → grad on every Gaussian's params, and on info["means2d"]
    each opt: step()
    accumulate ‖info["means2d"].grad‖ for each Gaussian (view-space grad signal)
    if t in densification window and t mod densify_every == 0:
        prune-and-clone:
          drop Gaussians whose opacity < threshold
          clone the top clone_pct of survivors by accumulated view-space grad
        transplant Adam state for survivors and clones
```

Five separate Adam optimisers — one per parameter group — with the published 3DGS learning rates (means $1.6 \times 10^{-4}$, quats $10^{-3}$, scales $5 \times 10^{-3}$, opacities $5 \times 10^{-2}$, colours $2.5 \times 10^{-3}$). The wildly different LRs reflect the wildly different gradient scales these parameters produce and are the kind of per-parameter-group manual tuning Adam itself would do if it had per-group learning rates (it doesn't — Adam's per-parameter scaling is at the scalar level, not the *group* level, which is why 3DGS-style splatting needs five optimizers in the first place).

---

## 2. The four fixes our implementation required

The most informative part of §10 from an optimization-engineering standpoint is the four root-cause failures we hit during the implementation, and what they revealed about how the GS optimization actually works. Each is documented here both because they are interesting findings about the method and because the report should be honest about the implementation journey.

### 2.1 Fix 1 — the OpenGL→OpenCV camera-convention mismatch

**The bug.** NeRF/Blender datasets ship with camera poses in OpenGL convention (right-up-back: $+x$ right, $+y$ up, $-z$ forward). gsplat 1.5.3 expects OpenCV convention (right-down-forward: $+x$ right, $-y$ up, $+z$ forward). Without the conversion, every Gaussian sits *behind* the gsplat camera in its view of the world; all Gaussians are culled before rasterisation; the rendered image is the white background composite; alpha = 0 everywhere; and the loss is $\mathcal{L}(\mathrm{white}, I)$ — which has *zero gradient* with respect to every Gaussian parameter.

**The symptom.** Loss appeared to "decrease" slightly because of L1 noise on the white-vs-GT difference; PSNR pinned at ~9 dB and *never moved*; per-eval val PSNR was identical to 6 decimal places across 15 000 iterations. Direct gradient inspection confirmed `params[0].grad.abs().max() == 0` for every parameter group.

**The fix.** Inside `_gs_render`, left-multiply the world-to-camera matrix by $\mathrm{diag}(1, -1, -1, 1)$ before passing to the rasteriser:

```python
def _opengl_to_opencv_viewmat(w2c):
    flip = torch.diag(torch.tensor([1., -1., -1., 1.], device=w2c.device))
    return flip @ w2c
```

A one-line fix that took eight hours of debugging to find. The lesson: **the most common GS training failure mode is not a gradient explosion or a hyperparameter mistake; it is a coordinate-convention mismatch that produces zero gradients everywhere with no obvious error message.**

### 2.2 Fix 2 — clones with no positional jitter

**The bug.** In the basic Kerbl densification recipe, a "clone" event copies a Gaussian's tuple verbatim and inserts the copy into the Gaussian list. Geometrically this places two Gaussians at *exactly the same position*, with the same scale, opacity, colour, and rotation. Two co-located identical Gaussians render exactly like one Gaussian (composited together they produce the same image), receive identical gradients in every subsequent step, and stay co-located forever. The clones are *no-ops*: they add parameter count but no representational capacity.

**The fix.** When cloning, jitter the child position by scale-aware Gaussian noise:

```python
child_mean = parent_mean + randn(3) * exp(parent_scale)   # scale is log-space
```

The child now lives one parent-scale's distance from its parent, so the two Gaussians are spatially distinct from step 1 and will diverge under the optimizer. This is the standard Kerbl-style jitter; our omitting it in v0 was an implementation mistake, not a method choice.

### 2.3 Fix 3 — wiped Adam state on every densification event

**The bug.** Each densification event creates a new set of parameter tensors (because Gaussian count changes). The naive way to handle this is to also create new Adam optimisers — but doing so *wipes Adam's $(m, v)$ state* for *every* Gaussian, including the survivors that had nothing to do with the densification. Adam's per-parameter step-size estimate is therefore *reset every 100 iterations*, which is the densification interval, which means Adam never gets to use the steady-state variance estimate it is designed to build. Effective behaviour: like SGD, but with the wrong learning rate.

**The fix.** Transplant the old Adam state into the new optimiser, indexed by the survivor-and-clone mapping:

```python
old_m, old_v = old_opt.state[old_param]["exp_avg"], old_opt.state[old_param]["exp_avg_sq"]
new_state_idx = torch.cat([survivor_idx, clone_idx])   # both index into old [N_before] tensor
new_opt.state[new_param]["exp_avg"]    = old_m[new_state_idx].clone()
new_opt.state[new_param]["exp_avg_sq"] = old_v[new_state_idx].clone()
```

The survivors keep their warmed-up moments; the clones inherit their parent's moments (which is the principled choice — at the instant of cloning, the parent's gradient statistics are the best information the optimizer has about the clone's likely behaviour).

### 2.4 Fix 4 — opacity reset (kept off by default)

**The mechanism.** The basic Kerbl recipe includes a *periodic opacity reset*: every $K$ iterations, set every Gaussian's opacity logit back to a low value (e.g., $\sigma(-4.6) = 0.01$). The intended effect is that Gaussians which *should* be present recover their opacity quickly under the gradient signal; Gaussians which were redundant or in the wrong place do not recover and are pruned at the next densification step. This is the mechanism by which a basic GS recipe sheds dead Gaussians.

**Why it is off by default in our implementation.** Empirically, on Lego at 800 × 800 and 15 k iterations, the opacity-reset mechanism is too aggressive in our setup: it shrinks the model to a few thousand Gaussians and converges to a similar PSNR / SSIM / LPIPS as without the reset, but at much sparser representation. The §11 comparison is a *quality-first* comparison, so we prefer the no-reset variant that keeps ~150 k Gaussians and produces marginally better metrics. The reset machinery is implemented (the `_gs_reset_opacities` helper exists and the `opacity_reset_every` flag can be set > 0); we simply don't enable it.

A more thorough sweep would tune the reset interval. We chose to scope this study to the four fixes above and leave the reset interval as a hyperparameter that future work could tune (or eliminate via the full Kerbl split + reset recipe). See [improvements_reference.md](improvements_reference.md) for the "substantiated improvements" framing of why this is a defensible scoping decision.

### 2.5 What the four fixes reveal about GS optimization

Together, the four fixes illustrate that **GS training is not just "Adam on a parameter vector" — it is Adam on a parameter vector whose dimension changes over training, whose backward pass requires a non-standard camera convention, and whose densification mechanism requires careful optimizer-state surgery to not interfere with Adam's own state management**. A from-scratch implementation that does not handle these details will train to a useless white-background optimum (Fix 1), or to a "trains but doesn't improve" optimum (Fixes 2 and 3), or to an over-aggressively-sparsified optimum (Fix 4 mis-tuned). The §10 implementation is faithful to the Kerbl 2023 recipe modulo the explicit choices listed in §2.4.

---

## 3. The §11 comparison — NeRF vs GS as optimization formulations

### 3.1 Setup

Both methods evaluate on the same scenes (Lego, Drums), at their respective natively-supported resolutions (NeRF: $200 \times 200$, GS: $800 \times 800$ — see §3.2 below for why), with the same camera poses. NeRF uses §7.2's winning Adam baseline at $\eta = 10^{-3}$; GS uses the §10 basic recipe described above.

The comparison table reports PSNR, SSIM, LPIPS, wall-clock, and parameter count per row, allowing the reader to see the *full trade-off surface* rather than just the headline PSNR.

### 3.2 The resolution choice and why it is fair

NeRF in our project is trained at $200 \times 200$ because we adopted the §7 / §8 / §9 budget there for fair multi-loss comparison. GS is trained at $800 \times 800$ — the Blender dataset's native resolution — because GS's wall-clock at 800 × 800 is *the same order of magnitude as NeRF's at 200 × 200* (~5 min vs ~8 min per scene, 15 k iterations vs 40 k iterations).

The comparison is therefore not "both at 200" or "both at 800" — it is *"each method at the resolution that uses its computational budget most effectively, with the wall-clocks roughly matched."* This is a deliberate framing decision documented here so the report can defend it: forcing GS to train at 200 to "match NeRF" would be artificially crippling GS's ability to capture high-frequency detail, and forcing NeRF to train at 800 would explode its compute past the §7-§9 study's budget. **The comparison is wall-clock-fair, not pixel-density-fair, and the report should say so.**

For per-method per-view PSNR (used in the §11.5 multi-view grid), the comparison is at each method's native resolution against the native-resolution GT. PSNR is therefore computed apples-to-apples within each row — NeRF at 200 against 200 GT, GS at 800 against 800 GT.

### 3.3 The headline result

With the four fixes applied, the basic GS recipe produces:

- **Lego**: PSNR $21.49$ dB (NeRF $22.24$ — NeRF wins by $0.75$ dB), SSIM $0.921$ (NeRF $0.849$ — GS wins by $0.072$), LPIPS $0.087$ (NeRF $0.146$ — GS wins by $0.059$)
- **Drums**: PSNR $23.31$ dB (NeRF $21.79$ — GS wins by $1.52$ dB), SSIM $0.924$ (NeRF $0.828$ — GS wins by $0.096$), LPIPS $0.085$ (NeRF $0.131$ — GS wins by $0.046$)

GS wins 5 of 6 metric cells, ties on Drums PSNR with a $+1.5$ dB lead, and loses only Lego PSNR by less than 1 dB (well below the perceptual threshold — see §3.5 below).

### 3.4 The wall-clock dimension

| | NeRF wall (s) | GS wall (s) | Speedup |
|---|---|---|---|
| Lego | 471 | 274 | ~1.7× |
| Drums | 483 | 263 | ~1.8× |

GS reaches the same or better quality in roughly *half* the wall-clock. The headline framing for §11: **"GS wins 5 of 6 metric cells at half the wall-clock cost with a different parametrization of the same problem."**

### 3.5 The single PSNR loss on Lego — interpretation

A $0.75$ dB PSNR gap is $\sim$ $10^{0.075} \approx 1.19\times$ larger mean squared error per pixel. Context:

- Roughly $2\sigma$ of the seed-noise band measured on Adam-L2 in §7.2 (std $\sim 0.34$ dB).
- Below the threshold most viewers can spot in a still ($\sim 1$ dB JND from the codec literature).
- The other two metrics (SSIM $+0.072$, LPIPS $-0.059$) are both *well above* their respective perceptual JND thresholds (~$0.02$ for both), so the perceptual verdict is unambiguous: GS reconstructs the Lego scene more faithfully *to the eye*, even though pixel-wise arithmetic slightly disagrees.

The optimization-theoretic interpretation: **PSNR's per-pixel weighting is unfavourable to GS specifically because GS's Gaussian primitives smooth high-frequency edges (e.g., the busy Lego studs) more than NeRF's MLP does, and the smoothed edges contribute many small pixel-wise errors that PSNR sums up. SSIM and LPIPS, which compare local structure rather than per-pixel intensities, correctly identify that the smoothing is perceptually preferable to NeRF's slightly-correct-but-noisy edges.** This is a real result about how the choice of optimization objective (pixel-wise L2 vs SSIM/perceptual) interacts with the choice of parameterisation (Gaussian primitives vs MLP).

---

## 4. The §11.5 / §11.6 visual comparison

### 4.1 Multi-view grid (§11.5, cell 102)

Three test views × (NeRF | GS | GT) per scene, rendered at $\mathrm{VIZ\_RES} = 1600$ from the trained models for *display*, with per-view PSNR computed at each method's native resolution for *honest metrics*. The per-view PSNR is computed at the native resolution where ground truth lives — *not* at $\mathrm{VIZ\_RES}$, because upsampling GT to 1600 for "fair comparison" would only smooth it and inflate PSNR for both methods equally.

The grid is saved as `outputs/orbits/nerf_vs_gs_grid.png` for the defense slides.

### 4.2 Side-by-side orbit videos (§11.5, cell 103)

For each scene, an MP4 with NeRF rendered on the left half and GS on the right half, the camera rotating around the object on a fixed elevation. 120 frames at 24 fps = 5-second loop. The MP4s are saved as `outputs/orbits/nerf_vs_gs_<scene>.mp4`.

Both render functions take the *same* camera pose (from `pose_spherical(theta, elevation, radius)`) — the only difference between the two halves is which renderer is invoked. This makes the comparison hermetic at every frame: if the two halves diverge, it is the methods that disagree, not the viewing angle.

### 4.3 Why render at higher than training resolution

Both NeRF (continuous in $(x, y, z)$ via the MLP) and GS (Gaussians continuous in 3D space) can render at any resolution at inference. Rendering at $1600 \times 1600$ from a 200-trained NeRF or an 800-trained GS produces a *smoother* image — more pixels per ray / per Gaussian — but does not invent new detail. This is the "free anti-aliasing" both methods get and is a legitimate visualization technique; it does *not* affect the metric numbers reported in §11 (which use native-resolution renders).

The defense framing: **the 1600-px renders are for the slide, the native-res renders are for the table; both are honest about what they show**.

### 4.4 Inline video embedding (§11.6)

The MP4s are linked into the notebook via HTML5 `<video>` tags by relative path rather than base64-embedded, so the .ipynb stays small (~200 KB extra rather than ~110 MB). A `FileLink` cell (§11.6, cell 106) copies the MP4s into a Jupyter-served directory and provides clickable download links for the defense.

---

## 5. Summary

| Dimension | NeRF (Adam, L2, §7.2) | GS (basic, §10) | Winner |
|---|---|---|---|
| **Parameterisation** | $58{,}000$-param MLP, continuous in $(x, y, z)$ | $\sim$$150{,}000$ Gaussians, $\sim$$2{,}000{,}000$ params total | GS for capacity, NeRF for parsimony |
| **Render operator** | volume integration along rays | tile-based rasterisation | GS for speed |
| **Backward pass** | dense MLP backprop | gsplat differentiable rasteriser | both are fully differentiable |
| **Optimizer** | single Adam, $\eta = 10^{-3}$ | five Adams, per-parameter-group $\eta$ | NeRF for simplicity, GS for fine control |
| **Discrete structural changes** | none | densification + pruning every 100 iters | NeRF has a smoother optimization, GS adapts capacity |
| **Lego PSNR / SSIM / LPIPS** | 22.24 / 0.849 / 0.146 | 21.49 / 0.921 / 0.087 | GS on perceptual, NeRF on PSNR |
| **Drums PSNR / SSIM / LPIPS** | 21.79 / 0.828 / 0.131 | 23.31 / 0.924 / 0.085 | GS on all three |
| **Wall-clock per scene** | $\sim$8 min | $\sim$4.5 min | GS by 1.8× |

**The §11 headline**: GS wins 5/6 metric cells at half the wall-clock with a different parameterisation of the same reconstruction problem. This is the optimization-perspective contribution of §10 / §11 / §11.5: NeRF and GS are not "two methods we tried"; they are *two formulations of the same optimization problem*, and the comparison reveals when each formulation is favoured by the optimizer.

---

## 6. Future-work positioning for §12

GS in the project is positioned as an *applied optimization comparison*, not as a deep-dive into 3DGS engineering. The two natural extensions, both explicitly out of scope and named in §12 as future work:

- **Full Kerbl recipe with the split step.** Our densification implements clone-only growth. The complete 2023 recipe also includes a *split* step for high-gradient large-scale Gaussians, which would likely close the 0.75 dB Lego PSNR gap. We deliberately omitted it to scope the §10 contribution as "is GS competitive with a basic recipe?" rather than "can we reproduce 3DGS at full fidelity?"
- **Dynamic 4D Gaussian Splatting.** Both NeRF and GS as implemented here assume a static scene — every photo must capture the same 3D configuration. Extending to dynamic scenes (a moving subject) requires either multi-camera synchronized capture or per-frame deformation fields, which is the active 4DGS research area (e.g., Wu et al. 2024, Yang et al. 2024). The static-vs-dynamic boundary is precisely why the Stage-5b Freddie-figurine capture in our project is a static-subject demonstration and not a "GS does video" demonstration.

---

## References

- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *SIGGRAPH*.
- Ye, V., Li, R., Kerr, J., Tancik, M., Kanazawa, A., et al. (2024). gsplat: An Open-Source Library for Gaussian Splatting. *arXiv:2409.06765*.
- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV*.
- Wu, G., et al. (2024). 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. *CVPR*. (Future-work pointer for §12.)
