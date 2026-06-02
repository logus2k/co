# Presentation Guide — Optimization for 3D Scene Reconstruction

*Walkthrough script for the Phase 2 defence (16-18 June 2026). Companion to `src/nerf_tutorial2_v4_AC.ipynb` and `project/tutorial2_plan.md`. This is not the report; it is what the two presenters should consult while walking the panel through the work.*

---

## How to read this guide

Every section maps to a section of the notebook in order. Within each section:

- **Say** — the narrative. Paraphrase it; do not read verbatim.
- **Show** — a small code excerpt the presenter can point at as evidence.
- **Point** — the rationale tied to the project objectives or the rubric.
- `[FILL: ...]` markers are values to drop in after the final Run All completes; the surrounding sentences are already written.

The five project objectives the rubric explicitly evaluates are:

1. **Formulate** an optimization problem in the AI context.
2. **Select and implement** optimization methods.
3. **Analyse** convergence, stability, performance.
4. **Compare** approaches and **justify** choices.
5. **Propose duly substantiated improvements**.

Each section below labels which objective it serves.

---

## 1. Opener (90 seconds)

**Say:** "We chose 3D scene reconstruction from 2D photographs as our optimization application — a continuous, non-convex, high-dimensional, stochastic problem and a textbook case of where the classical analytical optimization machinery of Module 1 stops being applicable. We implemented a NeRF (Neural Radiance Field) from scratch, five gradient optimizers from scratch, four loss formulations, and contrasted NeRF against an open-source 3D Gaussian Splatting baseline on the same scenes and metrics. About 130 training runs across two scenes and three seeds. One of our headline findings is methodological: a fair comparison of first-order optimizers requires per-method learning-rate tuning — at a shared rate the Adam-vs-SGD gap is 12.77 dB; at each method's own best rate the gap collapses to 1.82 dB. Almost 11 dB of the apparent advantage was a learning-rate-mismatch artefact, not the optimizer."

**Point:** All five objectives hit in the opener. *Formulate* (3D recon as continuous optimization), *select & implement* (5 from-scratch optimizers and 4 losses, plus the tools we chose for Bayesian HP search and GS), *analyse* (convergence / stability / quality), *compare and justify* (the optimizer comparison), *propose improvements* (4 ablations).

---

## 2. Sections 1–2 — Problem and formulation

*Objective: Formulate.*

**Say:** "We minimise the average per-pixel error between rendered and observed views over the parameters $\theta$ of a small MLP. The optimum satisfies the first-order condition $\nabla_\theta \mathcal{L} = \mathbf 0$ — the same condition Module 1 introduced for unconstrained optimization. What is different here is that two features of the problem make Module 1's analytical machinery inapplicable. First, $|\theta| \approx 58{,}000$ — the Hessian-based classification Module 1 used to distinguish minima from saddles has on the order of $3 \times 10^9$ entries; we cannot form it. Second, the loss is non-convex (the MLP non-linearities composed with the rendering operator), so there is no closed-form solution of $\nabla_\theta \mathcal{L} = \mathbf 0$. The project is therefore about how iterative first-order methods *substitute* for the analytical machinery that fails at our scale."

**Show:** The objective from §2 of the notebook:

$$\min_\theta \;\; \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}\big(R(\theta;\pi_i),\, I_i\big)$$

**Point:** This is the *Formulate* objective. We do not simply apply an algorithm; we explicitly characterise the problem's properties (non-convex, high-dimensional, stochastic) and they motivate every method that follows.

---

## 3. Section 3 — NeRF model and renderer

*Objective: Select and implement.*

**Say:** "The differentiable scene representation has five parts: data loading (we read `nerf_synthetic` `transforms_*.json`), ray generation (one ray per pixel), positional encoding (sin/cos at exponentially increasing frequencies — lifts 3D coordinates into a high-frequency feature space so a small MLP can fit detail), the MLP itself (4 layers of width 128 — about 58 thousand parameters, chosen by measurement in Stage 0), and the volume renderer — the operator $R(\theta;\pi)$ in the formulation. Everything is from scratch; the only external dependencies are PyTorch autograd and image utilities."

**Show:** The volume-rendering integral, in code:

```python
def render_rays(rays_o, rays_d, model, encoding, near=2.0, far=6.0, N=64):
    t = torch.linspace(near, far, N, device=DEVICE)
    delta = (far - near) / N
    t = t + (torch.rand(rays_o.shape[0], N, device=DEVICE) - 0.5) * delta
    pts = rays_o[:, None] + rays_d[:, None] * t[:, :, None]
    sigma, c = model(encoding(pts.reshape(-1, 3)))
    sigma = sigma.reshape(rays_o.shape[0], N)
    c = c.reshape(rays_o.shape[0], N, 3)
    deltas = torch.cat([t[:, 1:] - t[:, :-1], torch.full_like(t[:, :1], 1e10)], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], dim=-1),
        dim=-1)[:, :-1]
    return ((T * alpha)[..., None] * c).sum(dim=1)
```

**Point:** The implementation is intentionally small enough to inspect end to end. Every gradient that flows backwards through this is something the presenters can explain. This is what makes the comparison in §7 fair — there is no library code we could not trace.

---

## 4. Section 4 — Five optimizers from scratch

*Objective: Select and implement.*

**Say:** "We implemented five gradient-based optimizers from scratch on top of PyTorch autograd: SGD, SGD with classical momentum, SGD with Nesterov accelerated gradient, Adam, and AdamW. All five expose the same minimal `zero_grad()` / `step()` interface, so the training loop is optimizer-agnostic. All five are *first-order* — they iteratively approach $\nabla_\theta \mathcal{L} = \mathbf 0$ but none forms the Hessian. Adam's $\hat v$ buffer is a *diagonal* approximation of curvature, a tractable surrogate for the second-order information Module 1's analytical theory uses directly. That trade-off — replacing analytical second-order with iterative first-order plus diagonal-curvature estimates — is the central theme."

**Show:** Adam's `step()` is the one to point at — it has the most going on in the fewest lines:

```python
class MyAdam:
    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None: continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.addcdiv_(m / bc1, v.div(bc2).sqrt().add_(self.eps), value=-self.lr)
```

**Point:** This is the heart of "Select and implement." Every numerical convention — bias correction, eps placement, in-place ops — is ours. The "from scratch" commitment in Tutorial #1 was scoped exactly to these five optimizers because they are *the subject of study*. (This is the answer to "why Optuna from a library but Adam from scratch?" — see the anticipated questions.)

---

## 5. Section 5 — Loss functions

*Objective: Select and implement.*

**Say:** "Four loss formulations: L2 (mean squared error — the PSNR-aligned baseline), L1 (mean absolute error — less sensitive to outlier pixels), SSIM (structural similarity — a perceptual measure of local luminance, contrast, structure), and a weighted L1+SSIM combination (the form used in the Gaussian Splatting paper). A fifth, perceptual L2 plus LPIPS, appears in §9 as an improvement candidate."

"SSIM is a *spatial* measure — it cannot be computed from scattered pixels — so when a loss requires spatial structure the harness samples a contiguous $64 \times 64$ patch of rays instead of independent pixels. The whole SSIM is hand-rolled with an 11x11 Gaussian-windowed local mean, variance, and covariance computed via `F.conv2d`. We did not use a library SSIM."

**Show:** The patch-sampling branch in `run_experiment`:

```python
if cfg.patch_size > 0:
    P = min(cfg.patch_size, H, W)
    top  = int(torch.randint(0, H - P + 1, (1,)).item())
    left = int(torch.randint(0, W - P + 1, (1,)).item())
    rows = torch.arange(top, top + P, device=DEVICE)
    cols = torch.arange(left, left + P, device=DEVICE)
    pix = (rows[:, None] * W + cols[None, :]).reshape(-1)
else:
    pix = torch.randint(0, H * W, (cfg.batch_rays,), device=DEVICE)
```

**Point:** A `loss_needs_patch(name)` registry routes SSIM-based losses to the patch path automatically. The harness is loss-agnostic at the call site.

---

## 6. Section 6 — Experimental setup

*Objectives: Select and implement + Analyse.*

**Say:** "Three things to flag in this section."

**(a) Stage 0 was measured, not guessed.** "We ran four investigations: a resolution sweep (100/200/400/800), a batch/iteration scale-up, an MLP-capacity test (small 128x4 vs large 256x8 with skip), and an SGD-vs-Adam separability check at 200 and 400. From the data: per-iteration cost is *resolution-independent* (NeRF samples a fixed ray batch per step); reconstruction quality drops with resolution at fixed budget (coverage penalty). We picked 200x200, MLP 128x4 (~58k params), and a fixed 40,000-iteration budget — every choice traces to a measured number."

**(b) Train / validation / test splits.** "Straight from `nerf_synthetic` `transforms_*.json`. 100 train; 5 val (selection during training); 10 test (final reported metrics only). LPIPS uses the cached AlexNet backbone, weights under `data/models`."

**(c) The harness.** "A `RunConfig` dataclass describes a run completely; `run_experiment(cfg)` does training + evaluation + on-disk JSON-plus-`.pt` caching, with `reuse=True` so identical configurations never recompute. About 130 runs over the project — caching is what made the experiment matrix tractable."

**Show:** The cache key:

```python
@dataclass
class RunConfig:
    optimizer: str = "adam"
    loss:      str = "l2"
    scene:     str = "lego"
    seed:      int = 0
    lr:        float = 5e-4
    n_iterations: int = 40000
    # ... + Stage-3/Stage-6 fields ...

    def run_id(self):
        """Deterministic short id. New (post-v3) fields are stripped before
        hashing when at their defaults, so existing cached runs still match."""
        d = asdict(self)
        out = {k: d[k] for k in self._LEGACY_HASH_FIELDS if k in d}
        # include new fields only when active
        ...
        digest = hashlib.sha1(json.dumps(out, sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.scene}_{self.optimizer}_{self.loss}_s{self.seed}_{digest}"
```

**Point:** Measurement-first is itself a methodological choice. The backwards-compatible `run_id` is what lets us add features (Stage 3, Stage 6) without invalidating earlier caches.

---

## 7. Section 7 — Optimizer comparison (the headline result)

*Objectives: Analyse + Compare and justify.*

### 7.1 LR sweep

**Say:** "We swept the same eight-point log-uniform learning-rate grid for every optimizer — methodologically fair, no method gets a wider search than another. Adam and AdamW peak at $\eta = 10^{-3}$ around 23 dB and diverge for $\eta \ge 10^{-1}$. The SGD family (plain SGD, momentum, Nesterov) peaks at $\eta = 3 \times 10^{-1}$, the upper edge of the grid, with maxima of roughly 19.5 / 21.2 / 22.0 dB. The two families want learning rates separated by **300×** — any comparison at a shared LR would either run Adam in divergence or run SGD essentially unmoved."

### 7.2 Multi-seed comparison — the headline number

**Say:** "Thirty runs: five optimizers × two scenes (Lego, Drums) × three seeds × the full 40,000-iteration budget, each at its own §7.1-best LR with LPIPS computed at the end. Pooled across scenes and seeds:"

| Optimizer | test PSNR (dB) | test SSIM | test LPIPS |
|---|---|---|---|
| **Adam** | **22.01 ± 0.34** | **0.839** | **0.139** |
| AdamW | 22.00 ± 0.38 | 0.839 | 0.141 |
| Nesterov | 21.74 ± 0.28 | 0.829 | 0.159 |
| momentum | 21.68 ± 0.22 | 0.825 | 0.163 |
| SGD | 20.19 ± 0.47 | 0.765 | 0.272 |

**The sound bite to land:** "Stage 0's separability check used a *shared* LR of $5 \times 10^{-4}$ and measured a **12.77 dB** Adam-vs-SGD gap. Allowing each optimizer its own best LR from §7.1 collapses the gap to **1.82 dB**. Almost 11 dB of the apparent Adam advantage was a learning-rate-mismatch artefact, not a property of the optimizers themselves."

**Point:** This is the project's strongest *Compare and justify* finding. Memorise the phrase: "11 dB was mismatch; 1.82 dB is method."

### 7.3 Qualitative side-by-side + orbit MP4

**Say:** "The §7.3 grid renders the same held-out Lego view from each optimizer's trained model — visual story matches the numbers. SGD's reconstruction is visibly blurrier; this is the gap LPIPS sees (0.272 vs Adam's 0.139, almost 2×) more clearly than PSNR does (~9% pixel gap)."

"The orbit MP4 (`outputs/orbits/lego_adam_l2_s0_*.mp4`) demonstrates that the trained model is a coherent 3D representation — renderable from poses it never saw during training. That is the operational test that the optimization succeeded."

---

## 8. Section 8 — Loss comparison + the Optuna sub-study

*Objectives: Compare and justify + Select and implement (HP optimization).*

### 8.2 Loss comparison at the §7-best optimizer

**Say:** "Four losses, all at the §7-best optimizer (Adam at $\eta = 10^{-3}$), on the same scenes and seeds as §7.2. SSIM-based losses use a 64x64 patch sampler giving exactly the same 4,096 rays per step as the pixel-wise losses use."

| Loss | PSNR | SSIM | LPIPS |
|---|---|---|---|
| **L2** | **22.01 ± 0.34** | 0.839 | 0.139 |
| L1 | **14.08 ± 5.87** ⚠️ | 0.700 | 0.528 |
| SSIM | 21.70 ± 0.55 | **0.860** | **0.119** |
| L1+SSIM | 21.75 ± 0.51 | **0.860** | **0.118** |

**The two findings:**

1. "L2 wins PSNR (its aligned metric); the SSIM-based losses win SSIM and LPIPS. **Textbook trade-off** — different metrics reward different objectives. A single number would have hidden it."
2. "L1 collapsed — its mean is 14 dB with std 5.87 because at least one seed left the basin of convergence at the shared $10^{-3}$ LR. This is the same methodological story §7 made for optimizers, applied at the level of *losses* — a fair comparison requires per-method LR tuning. §8.3 fixes it for L1."

### 8.3 Optuna refinement for L1 — the methodologically important sub-study

**Say:** "We use Optuna's **TPE sampler** with **median pruner** to find L1's best LR. The objective trains L1 on Lego, seed 0, for up to 10,000 iterations and reports validation PSNR every 1,000 iterations. The pruner is what makes the study substantially cheaper than the grid in §7.1 — divergent LRs declare themselves at iteration ~2,000, the pruner kills the trial, and the compute saving scales with the fraction of bad trials."

**Show:** The objective function is the educational moment:

```python
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    cfg = RunConfig(optimizer="adam", lr=lr, loss="l1",
                    scene="lego", seed=0,
                    n_iterations=10000, eval_every=1000)
    def cb(it, val_psnr):
        trial.report(val_psnr, step=it)
        if trial.should_prune():
            raise optuna.TrialPruned()
    r = run_experiment(cfg, with_lpips=False, save=False, reuse=False,
                       step_callback=cb, verbose=False)
    return r.best_val_psnr if math.isfinite(r.best_val_psnr) else float("-inf")
```

**Three things to land:**

1. **Project-rubric alignment.** "Tutorial #1's requirements list explicitly names *'Hyperparameter optimisation'* and *'Bayesian optimisation'* as application areas. Using Optuna here addresses both, with the right configuration (TPE for adaptive sampling + median pruner for early termination of bad trials)."

2. **The "library vs from-scratch" position.** "Our 'from scratch' commitment was scoped specifically to the five gradient optimizers — they are the *subject of study*. Optuna is a *tool we select* to address one of the rubric's named application areas without distracting from the main subject."

3. **The two-level optimization story.** "The gradient optimizers of §4 optimize the model parameters $\theta$. Optuna's TPE optimizes the learning rate $\eta$ — the hyperparameter of those inner optimizers. Two layers of optimization, both addressed, both justified. This is the §12 closing line."

**Results after refinement:** L1 best LR `[FILL: lr from Optuna study]`; refined L1 row in the §8 table: PSNR `[FILL: mean ± std]`, SSIM `[FILL]`, LPIPS `[FILL]`.

---

## 9. Section 9 — Improvement ablations

*Objective: Propose duly substantiated improvements.*

**Say:** "Four candidates from Tutorial #1 §5, each evaluated as an isolated ablation against the §7.2 baseline (Adam + L2 + uniform sampling + $\eta = 10^{-3}$). All ablations change exactly one component so any effect is attributable to its cause."

The four:

- **A. Learning-rate restarts (SGDR).** Cyclic cosine LR with cycle doubling. Motivation: escape sharp local minima in the non-convex landscape.
- **B. Perceptual loss** (`l2_perc`). L2 augmented with a deep-feature distance through the cached LPIPS AlexNet backbone, weighted at 0.1. Motivation: pixel-wise losses are poorly aligned with human perception.
- **C. Multi-scale (coarse-to-fine) training.** Begins at 64×64, doubles to 100×100 at iter 10,000 and to 200×200 at iter 20,000. Motivation: smoother low-resolution objective acts as a warm start.
- **D. Adaptive view sampling.** $p_i \propto (e_i + \varepsilon)^{\alpha}$ where $e_i$ is a running EMA of per-image error. Motivation: ill-conditioning in pixel space — focus compute on under-reconstructed views.

**Results** (pooled across scenes and seeds):

| | PSNR | SSIM | LPIPS | wall (s) |
|---|---|---|---|---|
| baseline | 22.01 ± 0.34 | 0.839 | 0.139 | 480 |
| SGDR | 21.89 ± 0.34 | 0.834 | 0.145 | 458 |
| **perceptual** | 21.45 ± 0.30 | 0.827 | **0.096 ± 0.005** ⭐ | 551 |
| multi-scale | 21.99 ± 0.34 | 0.837 | 0.141 | 456 |
| adaptive views | 21.94 ± 0.24 | 0.836 | 0.142 | 492 |

**The headline:** "Perceptual loss is the only improvement that moves the needle — LPIPS drops from 0.139 to 0.096, about **31 % better perceptual quality**, at a cost of 0.5 dB PSNR. This is *exactly* the trade-off Tutorial #1 §5 motivated as a reason to attempt the improvement. The other three — SGDR, multi-scale, adaptive view sampling — are statistically indistinguishable from baseline (within ±0.15 dB; their std bands overlap the baseline's)."

**Point:** The "Propose improvements" objective is fulfilled with measurement, not assertion. A defensible ablation result includes the methods that *did not* help — that is what makes it science rather than advocacy. One of four is a clear win; three are not.

---

## 10. Section 10 — Gaussian Splatting baseline

*Objectives: Select and implement + Compare and justify.*

**Say:** "Tutorial #1 §3.1 committed to comparing NeRF against an open-source 3D Gaussian Splatting implementation — the proposal explicitly noted that *'the contribution of our project is to study how the optimization behaviour compares; reimplementing the custom CUDA rasteriser is out of scope'*. We use **gsplat 1.5.3** (Nerfstudio's PyTorch wrapper) — exactly as anticipated. Our `run_gs_experiment` returns the same `RunResult` shape as the NeRF runs, so §11 compares apples-to-apples on PSNR / SSIM / LPIPS computed over the same held-out test views."

"Per-parameter Adam learning rates from the published 3DGS paper, L1+SSIM loss reusing our hand-rolled `ssim_value`. Result file saved to `outputs/runs_gs/<run_id>.json` so it never collides with the NeRF cache under `outputs/runs/`."

**What v4 adds (basic densification):**

**Say:** "The published Kerbl et al. (2023) recipe has several mechanisms that account for its sub-pixel quality: spherical-harmonic colour, gradient-based densification (clone + split), and periodic opacity reset. v4 adds the densification half. Every 100 iterations, in the window [500, 0.7 × total_iters], we accumulate per-Gaussian view-space gradient norms via `info["means2d"].grad`, clone Gaussians whose mean accumulated gradient exceeds a threshold (the optimizer can then drive the copy elsewhere), and prune Gaussians whose $\sigma(\text{opacity})$ has fallen below a threshold. The per-parameter Adam optimizers are rebuilt after each densification step — Adam state resets, the simplest correct handling of a varying parameter count."

**Show:** The densification call site:

```python
if (densify and densify_from < it <= densify_until
        and it % densify_every == 0):
    params, grad_accum, grad_count, n_kept, n_cloned = (
        _gs_densify_and_prune(params, grad_accum, grad_count,
                              densify_grad_threshold,
                              prune_opacity_threshold))
    opts = _gs_make_optimizers(*params)
```

**What we deliberately did NOT implement, and why:**

- No spherical-harmonic up-sampling — caps quality on shiny / view-dependent surfaces but keeps the implementation small.
- No Gaussian splitting — so very large Gaussians cannot subdivide spatially; only cloning + scale relaxation drive refinement.
- No periodic opacity reset — low-opacity Gaussians are pruned rather than revived.

"Each of these is a Kerbl-et-al. mechanism that contributes to published 3DGS's 30 dB scores. We name them in §10.2 of the notebook so the §11 comparison can be honest about the gap."

**Results with densification (v4):** Lego PSNR `[FILL: GS Lego densified PSNR]`, Drums PSNR `[FILL]`, both with `[FILL: final Gaussian count]` Gaussians at convergence, ~`[FILL]` seconds of wall-time per scene.

**Point:** The library decision here is consistent with the from-scratch commitment for the gradient optimizers — Tutorial #1 explicitly anticipated this split. Using gsplat is "select and implement" applied to the right tool at the right level.

---

## 11. Section 11 — NeRF vs Gaussian Splatting

*Objective: Compare and justify.*

**Say:** "Head-to-head on the same scenes (Lego, Drums), the same held-out test views, the same evaluation metrics, and the two efficiency dimensions Tutorial #1 promised — training time and parameter count."

| | NeRF best (§7.2 Adam) | GS basic + densification (§10) |
|---|---|---|
| PSNR | 22.01 dB | `[FILL]` |
| SSIM | 0.839 | `[FILL]` |
| LPIPS | 0.139 | `[FILL]` |
| Parameter count | ~58,000 | ~`[FILL: n_gauss × 14]` |
| Training time (per scene) | ~480 s | ~`[FILL]` s |

**The framing to land:**

"NeRF and Gaussian Splatting sit at very different points on the parameter-count axis — NeRF is *implicit and compact* (a small MLP), GS is *explicit and large* (an explicit cloud of millions of primitives). The comparison is not *which is better in the abstract*; it is *which way to spend the compute budget*. If GS still trails NeRF on PSNR despite densification, we explain why — the simplifications listed in §10. If GS catches NeRF on perceptual metrics, that is the explicit-representation advantage showing up where it should — on view-dependent appearance."

**Point:** Tutorial #1 promised this comparison; we delivered it on the same scenes and metrics with a working GS implementation we wrote, plus a published one we used and adapted. Both decisions defensible.

---

## 12. Section 12 — Conclusions (the closing)

*All objectives, plus the methodological frame.*

**Say (the closing line):**

"The project addresses optimization at two levels.

The **inner layer** is five from-scratch gradient methods — SGD, momentum, Nesterov, Adam, AdamW — acting on the model parameters $\theta$ of a NeRF. The principal *Compare and justify* finding is the 12.77 → 1.82 dB collapse of the Adam-vs-SGD gap when each optimizer is tuned at its own learning rate; almost 11 dB of the apparent advantage was a methodological artefact rather than a property of the optimizers themselves.

The **outer layer** is a Bayesian / TPE hyperparameter search acting on the *learning rate* $\eta$ of the inner layer. We apply it to L1 in §8.3 and show that the same per-method-tuning lesson §7 made for optimizers applies one level up — L1's apparent gap in §8.2 was a learning-rate mismatch in disguise.

The four *propose improvements* candidates — SGDR, perceptual loss, multi-scale, adaptive view sampling — were evaluated as isolated ablations. **One** of the four (perceptual loss) is a clear win: ~31 % perceptual-quality gain on LPIPS at a 0.5 dB PSNR cost — the *intended* trade-off. The other three are statistically indistinguishable from baseline. Including those negative results is what makes the ablation honest.

Module 1's analytical first-order condition $\nabla_\theta \mathcal{L} = \mathbf 0$ is the literal target of every gradient method used here. Module 1's Hessian-based classification of critical points — infeasible at $|\theta| \sim 10^5$ — is replaced in this project by empirical evidence across seeds and scenes. That substitution, made deliberately and justified, is what the project demonstrates."

---

## 13. Anticipated questions (defence prep)

### Q: Why Optuna from a library when Adam is from scratch?

**A:** "The 'from scratch' commitment in Tutorial #1 was scoped specifically to the five gradient optimizers because they are *the subject of study*. Optuna is a *tool we select* to address one of the rubric's named application areas — Bayesian / hyperparameter optimization — without pulling the project away from its committed subject. The same logic applies to using `gsplat` for the GS baseline (the proposal explicitly permits a library there) and `lpips` for the LPIPS metric. Reimplementing TPE would have been ~200 lines that would not compete with Optuna for quality, and it would not have answered any question the project posed."

### Q: Why did L1 diverge at the shared learning rate?

**A:** "L1's gradient is *sign-based* — its magnitude is constant regardless of how far off the prediction is. Adam at $10^{-3}$ — a rate calibrated for L2's quadratic gradient that shrinks near the minimum — produces too large a step in L1's late phase. §8.3 is the fix: per-loss LR via Optuna with the median pruner. The result *after* refinement is what we report alongside the §8.2 result."

### Q: Why does the simplified GS underperform NeRF even with densification?

**A:** "We deliberately omitted three Kerbl-et-al. mechanisms: spherical-harmonic up-sampling (view-dependent colour), Gaussian splitting (so very-large Gaussians cannot subdivide spatially), and periodic opacity reset (so stuck-low-opacity Gaussians are pruned rather than revived). Each contributes to published 3DGS's 30+ dB scores. With all three added back, GS would close most of the gap. Our framing in §10 is explicit about which subset of the recipe we implemented; we did not over-claim."

### Q: Why a fixed 40,000-iteration budget rather than running each optimizer to its own convergence?

**A:** "Stage 0 measured this. At 200×200, batch 4096, Adam is essentially converged by 10,000 iterations; SGD is still improving at 40,000. We deliberately fixed the budget so the comparison answers *'which optimizer reaches the best quality in a fixed compute budget'* — an anytime-performance question. Running each to its own convergence would conflate the optimizer comparison with a budget question."

### Q: Why 200×200 rather than full 800×800?

**A:** "Empirical Stage 0 finding. Per-iteration cost is resolution-*independent* (NeRF samples a fixed ray batch per step), but reconstruction quality drops with resolution at fixed budget — 1024 rays cover ~10 % of a 100×100 image but only ~0.16 % of an 800×800 image. The SGD-vs-Adam separability check confirmed that 200×200 distinguishes optimizers fully: the Adam-minus-SGD gap was 12.77 dB at 200×200 vs 12.00 dB at 400×400. Reducing resolution did not compress the methodology, and it kept the ~100-run experiment matrix feasible."

### Q: Why only Lego and Drums?

**A:** "They are a difficulty gradient within `nerf_synthetic` — Drums is the harder scene (more complex geometry, less uniform texture) — and the two-scene mean-±-std validates that our optimizer ranking is consistent across scenes. Tutorial #1 §6 also committed to a self-captured COLMAP scene. The loader is in place — `load_scene_splits` dispatches on a `'colmap:'` prefix to a COLMAP-format reader we wrote — and the self-captured run is one of the remaining items in `tutorial2_plan.md`."

### Q: Did you consider implementing TPE from scratch?

**A:** "Yes — explicitly. We chose to use Optuna because the *Select and implement optimization methods* objective is satisfied by selecting the right tool at the right level. The from-scratch detour for TPE would have been a substantial distraction from the project's stated subject — the five gradient methods — and would not have produced a TPE that competes with the library. We made the choice deliberately, with both options on the table."

### Q: Why did the perceptual loss improve LPIPS but lose PSNR?

**A:** "They measure different things. PSNR is pixel-MSE in decibels; LPIPS is the deep-feature distance through a pretrained network trained on human similarity judgements. The perceptual loss explicitly optimises an objective closer to LPIPS, so trading 0.5 dB of pixel fidelity for ~31 % perceptual-quality improvement is the *intended* behaviour — and it is exactly the trade-off Tutorial #1 §5 motivated as a reason to attempt this improvement."

### Q: How did you validate the harness before running the experiment matrix?

**A:** "Two layers. First, a CPU smoke test that exec'd every definition cell and exercised the factory wires. Second — after we learned from a hard lesson during the first overnight run — a tiny CPU *training step* through every loss × every improvement combination, because a factory returning a callable is *not* the same as the training path working. The first layer caught syntax / wiring; the second catches interaction bugs like the one where LPIPS buffers were created inside an inference-mode context and then refused to back-propagate. After v4 was assembled we also ran a small GPU dry-run end to end before launching the full Run All."

### Q: What is the role of caching?

**A:** "Every `run_experiment(cfg)` is keyed by a stable hash of the config — `run_id()` strips new-feature fields when they are at default, so v4 features did not invalidate v3 caches. Results are written to disk as JSON + `.pt`. The full experiment matrix is therefore *fully resumable* across kernel restarts, VS Code restarts, or session resumption. About 130 runs accumulated; only ~85 minutes of fresh GPU work remained when v4 was finalised."

---

## How to use this guide on the day

1. **Walk Sections 1–12 in order.** Do not dwell on §3 / §4 / §5 implementation unless asked; point at the live code and move on.
2. **Land §7's sound bite** explicitly: *"11 dB was mismatch; 1.82 dB is method."* That is the strongest single result.
3. **Land §9's perceptual-loss finding** as a second sound bite: ~31 % LPIPS gain at -0.5 dB PSNR, the *intended* trade-off.
4. **Pre-empt §10's simplifications** — name SH, splitting, opacity reset proactively; do not let the panel ask first.
5. **Close on the two-level optimization story** in §12. That is the closing line.

The notebook is the implementation. This guide is the script. Read the live code on the day; quote the markdown for the narrative.
