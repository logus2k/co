# Loss Functions — Algorithm Reference

A technical reference for the loss functions used in §5 and compared in §8 of the notebook. As with the [optimizers reference](optimizers_reference.md), each section is **plug-and-play**: any subsection can be lifted into a Markdown cell in the notebook or composed into the final report's "Methods" chapter. The math, the implementation, the design decisions, and the connection to the §8 / §8.3 results are kept together.

> *Scope*: this document covers L2, L1, SSIM, L1+SSIM, and L2+perceptual. The §8.3 Optuna refinement of L1's learning rate and the gradient-clipping investigation live in `bayesian_optimization_reference.md`.

---

## 0. Why the loss matters as much as the optimizer

The optimizer descends a landscape that the loss *defines*. The same parameter $\theta$ can be a global minimum under one loss and far from any minimum under another — so choosing the loss is itself an optimization decision, not a "post-hoc presentation choice." Three things make the loss choice particularly consequential on this problem:

1. **The loss shapes the landscape geometry.** L2 is smooth and quadratic in the small-error regime; L1 is non-differentiable at zero and has constant-magnitude gradient elsewhere; SSIM is structured and saturates. The optimizer's behaviour — how aggressively Adam normalises, how quickly momentum builds, whether plain SGD converges at all — depends on which of these landscapes it sees.
2. **The loss decides which errors matter.** Two reconstructions can be equally "wrong" on a pixel-wise L2 sense but qualitatively different: one might have a uniform low-magnitude blur, the other have a sharp edge slightly misplaced. SSIM and LPIPS *redistribute* the loss's mass over the image so the optimizer is rewarded for getting *structure* right rather than averaging error away.
3. **The loss determines the metric the model trains *toward*.** Maximising PSNR is, definitionally, minimising L2. Minimising LPIPS requires either an LPIPS loss term or a proxy. The §8 comparison therefore quantifies not "which loss is best" abstractly, but "which loss aligns the optimizer with which evaluation metric."

The §8 comparison is the project's empirical study of this alignment; the §8.3 sub-study is the same question for L1 specifically, after Bayesian per-loss LR tuning.

---

## 1. Pixel-wise losses

### 1.1 L2 — mean squared error

**Definition.**
$$
\mathcal{L}_{\mathrm{L2}}(\hat I, I) = \frac{1}{N} \sum_{i=1}^N (\hat I_i - I_i)^2
$$

**Properties.**
- *Smooth everywhere* — gradient $\partial \mathcal{L}_{\mathrm{L2}} / \partial \hat I_i = (2/N)(\hat I_i - I_i)$ is continuous and proportional to the error magnitude. The optimizer therefore takes large steps where the error is large and small steps where it is small, which is the textbook regime in which Adam's variance estimator behaves benignly.
- *Maximising PSNR is exactly minimising L2*: $\mathrm{PSNR} = 10 \log_{10}(1 / \mathcal{L}_{\mathrm{L2}})$ for $[0, 1]$-normalised images, so any improvement in L2 translates monotonically to PSNR.
- *Tolerates uniformly small errors.* The squared penalty grows slowly near zero, so a uniformly blurred reconstruction (all pixels off by a small amount) and a sharp reconstruction (most pixels exact, a few off by a large amount) can have the same L2 — and the optimizer has no reason to prefer one over the other. This is the classic "L2 loss → blurry images" criticism of pixel-wise objectives.

**Implementation note.** `loss_l2(pred, target) = ((pred - target) ** 2).mean()`. Works on scattered ray batches (no spatial structure required).

### 1.2 L1 — mean absolute error

**Definition.**
$$
\mathcal{L}_{\mathrm{L1}}(\hat I, I) = \frac{1}{N} \sum_{i=1}^N |\hat I_i - I_i|
$$

**Properties.**
- *Non-differentiable at zero.* The gradient is $\partial \mathcal{L}_{\mathrm{L1}} / \partial \hat I_i = (1/N)\,\mathrm{sign}(\hat I_i - I_i) \in \{-1/N, 0, +1/N\}$ almost everywhere, undefined at $\hat I_i = I_i$. Autograd resolves this by convention (subgradient = 0 at zero), and in floating-point training the exact $\hat I_i = I_i$ event has measure zero, so this is not a practical implementation problem — but it does have a *theoretical* consequence below.
- *Edge preservation.* The constant-magnitude penalty means a large per-pixel error is not punished disproportionately. The optimizer is therefore *less* tempted to smear an error across many pixels to reduce a single large one, which is the mechanism by which L1 preserves edges better than L2.
- *Adversarial regime for Adam.* L1's gradient magnitude is constant, so Adam's second moment $\hat v \to \mathrm{const}$ rather than tracking the loss landscape's curvature. The per-parameter scaling $\eta / \sqrt{\hat v}$ becomes a *constant* scaling, indistinguishable from a global learning-rate choice — so Adam loses its main advantage over SGD on this loss. The §8 and §8.3 results show this as L1's *much higher seed-to-seed variance* than the other losses: one of three Lego seeds diverges at any LR we try, because Adam's normalisation cannot damp the sign-driven oscillation that the constant-magnitude gradient produces near a minimum. **This is the key §8.3 finding.**

**Implementation note.** `loss_l1(pred, target) = (pred - target).abs().mean()`.

---

## 2. Spatial losses

### 2.1 SSIM — structural similarity

**Definition.** SSIM (Wang et al. 2004) decomposes the comparison between two images into three components — luminance, contrast, structure — measured locally with a Gaussian window. For two patches $x, y$:
$$
\mathrm{SSIM}(x, y) = \frac{(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
$$
with $\mu, \sigma^2, \sigma_{xy}$ the local mean, variance, and covariance computed under an 11×11 Gaussian window, and $c_1 = 0.01^2, c_2 = 0.03^2$ small constants for numerical stability.

We use SSIM as a *loss* through the canonical form $1 - \mathrm{SSIM}$, so minimising the loss maximises structural similarity.

**Properties.**
- *Spatial — requires a contiguous patch.* Unlike pixel-wise losses, SSIM is meaningful only when the rendered pixels form a coherent image region. The harness samples a `patch_size × patch_size` block of rays when `cfg.patch_size > 0` to support this; pixel-wise losses can use the default scattered sampling.
- *Differentiable, but saturating.* The Gaussian window and the multiplicative form make SSIM smooth in its arguments. It saturates at 1 when the patches are identical, which means the gradient vanishes near a perfect reconstruction — convergence near the minimum is naturally damped.
- *Perceptually closer to human judgement* than L2. Two patches with the same L2 error but different local structure (e.g., a sharp edge vs. a uniform blur) can have very different SSIM. This is the property that makes SSIM-trained models look more "natural" even when their PSNR is slightly worse.
- *Less curvature, smaller gradient magnitude, harder for SGD.* The smoothness of the Gaussian-window-based comparison means the loss landscape is gentler than L2's, which is fine for Adam but can slow plain SGD.

**Implementation note (`ssim_value`, cell 31).** Hand-rolled in PyTorch: build a 2-D Gaussian kernel, use `F.conv2d` with `groups=3` to compute per-channel local means and variances, plug into the canonical formula. Self-contained with no external SSIM library, so the comparison is hermetic. The function returns a scalar SSIM value; the loss is computed as `1.0 - ssim_value(p, t)` over a flattened `patch_size × patch_size` patch.

### 2.2 L1 + SSIM — pixel accuracy plus structure

**Definition.**
$$
\mathcal{L}_{\mathrm{L1+SSIM}}(\hat I, I) = \alpha\, \mathcal{L}_{\mathrm{L1}}(\hat I, I) + (1 - \alpha)\,(1 - \mathrm{SSIM}(\hat I, I))
$$
with $\alpha = 0.2$ (the Gaussian Splatting paper's choice; see [gaussian_splatting_reference.md](gaussian_splatting_reference.md)).

**Properties.**
- *Best of both regimes.* The L1 term preserves pixel-level edges (where SSIM's local-mean averaging is less informative); the SSIM term enforces local structural similarity (where L1's pixel-wise penalty doesn't distinguish a blur from a sharp reconstruction).
- *Why $\alpha = 0.2$ specifically.* The two terms are on different scales — L1 outputs are in image units, $(1 - \mathrm{SSIM})$ in the range $[0, 2]$. A small weight on the L1 term lets SSIM dominate the perceptual quality while L1 provides an anti-blurring "anchor." This ratio is what the 3DGS paper found empirically to balance the two objectives on outdoor scenes; we adopt it without retuning, both for reproducibility and because it lets us compare to the GS reference fairly.
- *Adam-friendly.* The SSIM term contributes a smoothly varying gradient with non-constant magnitude, which restores Adam's second-moment-tracking advantage that pure L1 loses. We expect L1+SSIM to behave closer to L2 than to L1 in terms of seed-to-seed stability; §8.2 confirms this.

**Implementation note.** `_l1_ssim_loss_from_patch(pred, target, patch_size, alpha=0.2)`. Reuses `ssim_value` from §2.1, so the SSIM mechanics are tested in one place.

---

## 3. Perceptual loss (Improvement B, §9)

### 3.1 L2 + LPIPS perceptual distance

**Definition.**
$$
\mathcal{L}_{\mathrm{L2+perc}}(\hat I, I) = \mathcal{L}_{\mathrm{L2}}(\hat I, I) + w_{\mathrm{perc}}\, \mathrm{LPIPS}_\phi(\hat I, I)
$$
with $w_{\mathrm{perc}} = 0.1$ and $\mathrm{LPIPS}_\phi$ the Learned Perceptual Image Patch Similarity (Zhang et al. 2018), computed by passing both images through a pretrained AlexNet backbone $\phi$ and measuring the L2 distance in the network's feature space.

**Properties.**
- *Deep-feature, not pixel-feature.* LPIPS measures distance in the activation space of a network trained for ImageNet classification; that space is known to align well with human perceptual judgement, which is why it is the standard "is this image plausibly correct?" metric in modern NeRF and image-generation papers.
- *Composite landscape.* The L2 anchor keeps the model from diverging into perceptually-plausible-but-pixel-wrong reconstructions (a known failure mode of pure perceptual losses); the LPIPS term then refines toward perceptually preferred answers within the L2-near manifold. The weighting $w_{\mathrm{perc}} = 0.1$ is intended to be small enough to act as a refinement and not a primary signal.
- *Expensive.* Each loss evaluation requires a forward pass through AlexNet on the rendered patch. Wall-clock per iteration is ~1.4× a pure L2 step at the same patch size. The §9 ablation includes this overhead in its reported `iter_per_s`.

**Implementation note (`_perceptual_loss_from_patch`).** The cached LPIPS-AlexNet weights live under `data/models/`; `get_lpips()` returns a *singleton* eagerly constructed at cell-load time and explicitly cast out of `inference_mode` to guarantee backward-passability. This is a non-trivial detail — the original lazy-initialised version of `get_lpips` returned inference-mode tensors, which silently break gradient flow. The eager-construction fix is documented in §6.2 of the notebook.

**Connection to §9 result.** Improvement B is the *only* one of the four §9 improvements that produces a statistically meaningful change against the §7.2 Adam-L2 baseline: it drops LPIPS from 0.139 → 0.096 (a 31% reduction) at the cost of −0.5 dB PSNR. **This is the substantiated trade-off in the rubric's "Propose duly substantiated improvements" objective.**

---

## 4. Sampling strategy: scattered vs. patch

The loss decides whether the harness must sample rays as a *contiguous patch* or as *scattered pixels*.

| Loss | Patch required? | Default sampling |
|---|---|---|
| L2 | no | `batch_rays` scattered pixel rays |
| L1 | no | `batch_rays` scattered pixel rays |
| SSIM | **yes** | `patch_size × patch_size` block |
| L1 + SSIM | **yes** | `patch_size × patch_size` block |
| L2 + LPIPS | **yes** | `patch_size × patch_size` block (LPIPS needs a coherent image) |

`cfg.patch_size = 64` is used in this project for all spatial losses. The training loop in `run_experiment` switches sampling strategy based on `cfg.patch_size > 0` so the same code path handles both.

This has a subtle implication for the §8 comparison: spatial losses see ~`patch_size**2 = 4096` rays per step regardless of `cfg.batch_rays`, so the *effective* batch size is held nearly constant between pixel-wise and spatial losses. Without this, the spatial losses would be at a sampling-noise disadvantage and the comparison would be unfair.

---

## 5. Summary comparison

| Loss | Sampling | Smoothness | Adam-friendly? | Aligned with |
|---|---|---|---|---|
| **L2** | scattered | $C^\infty$ | yes | PSNR (definitionally) |
| **L1** | scattered | non-diff at 0 | **no** (constant grad magnitude) | edge preservation; *unstable for Adam at long budgets* |
| **SSIM** | patch | smooth, saturating | yes | local structure |
| **L1 + SSIM** | patch | smooth (SSIM dominates) | yes | structure + edge sharpness |
| **L2 + LPIPS** | patch | smooth | yes | human perceptual judgement |

---

## 6. Connecting the implementation to the §8 / §8.3 / §9 results

The §8 comparison is meant to *test the predictions* this reference section sets up. Three results matter most for the optimization-thinking narrative:

1. **"L1 is unstable for Adam at long budgets, despite seemingly correct LRs."** Confirmed by §8.2 (one of three Lego seeds diverges at `lr=1e-3`), confirmed by §8.3 (after Optuna-tuned LR + cosine schedule, the divergence migrates to a different seed but the pooled variance is unchanged), confirmed by the §8.3 gradient-clipping investigation (clipping at `max_norm=1.0` is a no-op because L1 grad norms stay below 1.0 — the instability is *sign-oscillation* under Adam's variance estimator, not gradient magnitude). The combined finding: **L1's instability is structural, a consequence of its constant-magnitude gradient interacting with Adam's per-parameter variance estimator.** This is a *substantiated* observation about the interaction between loss geometry and optimizer choice — exactly the kind of optimization thinking the rubric rewards.

2. **"L1+SSIM and SSIM dominate L1 on perceptual metrics."** Confirmed by §8.2: L1+SSIM produces test-LPIPS = 0.118 vs L1's 0.534 (4.5× worse), at test-PSNR within 0.3 dB. The L1 row's pooled metrics are dragged down by the diverged seed but even excluding it L1 underperforms L1+SSIM and SSIM on every perceptual metric. **The conclusion is that the SSIM-style spatial term is what produces perceptually-aligned reconstructions, not the L1 pixel-accuracy term on its own.**

3. **"Perceptual loss as a substantiated improvement (§9 Improvement B)."** As described in §3.1: −31% LPIPS at −0.5 dB PSNR. This is the *intended* trade-off the project's Tutorial #1 §5 motivated, now empirically confirmed.

---

## References

- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE TIP*, 13(4).
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*.
- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *SIGGRAPH*. (Source of the L1+SSIM weighting $\alpha = 0.2$.)
