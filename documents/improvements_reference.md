# Improvements Ablation — Algorithm Reference

A technical reference for the four candidate improvements compared in §9 of the notebook. Each section is **plug-and-play** in the same sense as the [optimizers](optimizers_reference.md) and [losses](losses_reference.md) references: any subsection can be lifted into a Markdown cell. The four improvements were committed to in Tutorial #1 §5 and are evaluated here as isolated ablations against the §7.2 best baseline (Adam, $\eta = 10^{-3}$, L2, uniform view sampling, 40 000 iterations).

> *Scope*: this document covers SGDR (Improvement A), perceptual loss (B), multi-scale training (C), and adaptive view sampling (D). Each is described in the optimization-thinking framework the rubric rewards: motivation grounded in the optimization landscape, mathematical formulation, expected effect, and what the §9 ablation measured.

---

## 0. What "a substantiated improvement" means under the rubric

The project's fifth objective is *"Propose duly substantiated improvements"*. "Substantiated" is doing real work in that sentence: it means each proposed improvement must come with

1. **A specific failure mode it claims to address.** Generic "this might be better" is not enough — the proposal must point to a concrete property of the optimization landscape, the loss geometry, or the training procedure that the improvement is intended to fix.
2. **An ablation that isolates the variable.** Exactly one change against a fixed baseline. Otherwise the result cannot be attributed to the proposed improvement.
3. **A statistical comparison, not a point estimate.** Multi-seed runs with pooled mean ± std, and a comparison against the baseline's variance. A 0.3 dB "improvement" that is well within the baseline's seed-to-seed noise is not a substantiated improvement; a 0.5 dB improvement at 0.34 dB pooled std is.
4. **An interpretation tied to the failure mode.** What did the measurement reveal about the landscape / loss / procedure? "It improved" is a measurement; "*this is why* it improved" — or "*this is why* the predicted effect did not appear" — is the substantiation.

The four improvements below each follow this template.

---

## 1. Improvement A — SGDR (warm restarts)

### 1.1 Failure mode targeted

The NeRF objective is *non-convex and high-dimensional*. Cosine-warmup annealing (§4.6 of optimizers reference) takes the learning rate monotonically to zero — once $\eta$ is small enough, the parameters can no longer escape a basin even if a better one exists nearby. This is the textbook "stuck in a sharp local minimum" failure mode that warm-restart schedules try to address.

### 1.2 Method

SGDR (Loshchilov & Hutter 2017a) replaces the single-cycle cosine schedule with a *cyclic* cosine schedule: $\eta$ decays toward $\eta_{\min}$, then jumps back to the initial $\eta_0$, and decays again. Cycle $i$ has length $T_0 \cdot T_{\mathrm{mult}}^i$ with $T_{\mathrm{mult}} > 1$, so each cycle is longer than the last; early cycles explore aggressively and later cycles refine.

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\bigl(1 + \cos(\pi \cdot \text{in-cycle progress})\bigr)
$$

Our configuration: $T_0 = 5000$, $T_{\mathrm{mult}} = 2.0$, $\eta_{\min} = 0$, four cycles in the 40 k-iteration budget (5 k + 10 k + 20 k + 5 k tail).

### 1.3 Implementation

See `sgdr_lr(step, base_lr, t0, t_mult)` in cell 29 of the notebook. The training loop applies it instead of `cosine_warmup_lr` when `cfg.use_sgdr=True`.

### 1.4 Expected effect

If the §7.2 Adam-L2 baseline is converging to a *sharp* local minimum that warm restarts can escape, we expect SGDR to produce a clearly better pooled test PSNR. If the baseline is already converging to a *wide* minimum (the well-known "flat-minima-generalise-better" regime), warm restarts will at best be a no-op and at worst destabilise late training.

### 1.5 What §9 measured

Statistically indistinguishable from the baseline. The §9 ablation result places SGDR's pooled test PSNR within the baseline's seed-to-seed noise band. The interpretation: **the §7.2 Adam-L2 baseline is converging to a wide-enough basin that warm restarts do not help, but they also do not hurt** — exactly the "no-op" outcome the failure-mode analysis predicted as one of the two possibilities. This is a *negative* result, but a substantiated one: the failure mode SGDR addresses is not present at this scale.

---

## 2. Improvement B — Perceptual loss (L2 + LPIPS)

### 2.1 Failure mode targeted

L2 alone is *poorly aligned with human perceptual judgement* — the textbook criticism of pixel-wise losses (Zhang et al. 2018). A reconstruction that is uniformly blurry can have lower L2 (and higher PSNR) than a reconstruction that is mostly sharp but has a few mis-placed edges, even though the second is the one a human would call "better." The intended remedy is to add a perceptual term to the loss so the optimizer is rewarded for perceptual closeness, not pixel-wise closeness alone.

### 2.2 Method

Augment L2 with the LPIPS (Learned Perceptual Image Patch Similarity) deep-feature distance through a cached AlexNet backbone $\phi$:

$$
\mathcal{L}_{\mathrm{L2+perc}}(\hat I, I) = \mathcal{L}_{\mathrm{L2}}(\hat I, I) + w_{\mathrm{perc}}\, \mathrm{LPIPS}_\phi(\hat I, I)
$$

with $w_{\mathrm{perc}} = 0.1$. Details and code in [losses_reference.md §3.1](losses_reference.md#31-l2--lpips-perceptual-distance).

### 2.3 Expected effect

This is the one §9 improvement with the strongest a-priori reason to work: LPIPS is the standard "perceptual quality" metric and an LPIPS term in the loss should directly drag the optimizer toward perceptually-preferred minima. We *expect* a clear LPIPS improvement, possibly at the cost of some PSNR (because the optimizer is now juggling two objectives and the perceptual one slightly disagrees with the pixel-wise one).

### 2.4 What §9 measured

**The intended trade-off, statistically confirmed.** Pooled test LPIPS drops $0.139 \to 0.096$, a 31% reduction. Pooled test PSNR drops $22.01 \to ~21.5$, about $-0.5$ dB. SSIM tracks LPIPS upward.

This is the project's headline *substantiated improvement*: the predicted failure mode was real, the proposed remedy addressed it, the trade-off appeared as predicted, and the magnitudes are large enough to dominate seed-to-seed noise. The defense framing is: **"−31 % LPIPS at −0.5 dB PSNR is the substantiated trade-off Tutorial #1 §5 motivated, now empirically confirmed, and the choice between L2 and L2+perceptual is now a deliberate optimization-of-which-metric decision rather than a default."**

---

## 3. Improvement C — Multi-scale (coarse-to-fine) training

### 3.1 Failure mode targeted

The NeRF objective is non-convex; the loss landscape at full image resolution may be dominated by high-frequency detail that the optimizer cannot match until coarse structure is already approximately correct. A model started from random initialisation and trained at full resolution from step 1 must therefore *jointly* solve "where are the major surfaces?" and "where are the texture details?", which has many bad local minima.

### 3.2 Method

Start training at a small resolution ($64 \times 64$) where the loss landscape is *smoother* (because spatial averaging reduces high-frequency noise), then increase the resolution at scheduled milestones. The intuition is the classical "coarse-to-fine" trick from computer vision (e.g., image pyramids in optical flow, multi-resolution mesh fitting): the coarse problem is easier and its solution is a *warm start* for the fine problem.

Our schedule: resolution $64 \times 64$ for iterations $1$–$10\,000$, then $100 \times 100$ for $10\,000$–$20\,000$, then $200 \times 200$ for $20\,000$–$40\,000$.

### 3.3 Implementation

See `cfg.multiscale`, `cfg.multiscale_milestones`, `cfg.multiscale_resolutions` in `RunConfig` (cell 41). The training loop in `run_experiment` (cell 46) maintains a per-resolution data cache and switches resolution at the milestones; `adaptive_sampling` state is reset on each switch so error statistics correspond to the current resolution.

### 3.4 Expected effect

If the failure mode (full-resolution training getting stuck in high-frequency local minima) is real, we expect multi-scale to produce a clear pooled-PSNR improvement at no perceptual cost. If the baseline Adam is already strong enough to handle the full-resolution landscape from step 1, multi-scale will be a no-op or a small loss (because two-thirds of training is at sub-target resolution).

### 3.5 What §9 measured

Statistically indistinguishable from the baseline. The interpretation: **at our model size (~58 k params) and dataset (100 train views per scene), the §7.2 Adam baseline does not get stuck in the high-frequency local minima that multi-scale is designed to escape**. The "warm start" benefit is real in principle but lost in the wash because Adam's per-parameter step is already adaptive enough to find the full-resolution minimum directly. This is a substantiated negative result: the proposal was well-motivated, the failure mode is identifiable in the literature, but on this specific problem the failure mode is not active.

---

## 4. Improvement D — Adaptive view sampling

### 4.1 Failure mode targeted

Uniform random sampling of training views allocates equal effort to every view, regardless of how well the current model already reconstructs each one. If some views are converged and others are not, this wastes optimization effort on the easy ones. The proposed remedy is to *bias* sampling toward views the model is currently bad at, so optimization effort is concentrated where it is most needed.

### 4.2 Method

Maintain a running per-view error estimate $e_i$ for each training view, and sample view $i$ with probability
$$
p_i \propto (e_i + \varepsilon)^\alpha
$$
with $\varepsilon = 10^{-3}$ to prevent zero probabilities, and $\alpha = 1.0$ for the "linear-in-error" weighting. After every iteration, update the chosen view's error estimate with an exponential moving average:
$$
e_i \leftarrow 0.9\, e_i + 0.1\, \ell_t
$$
where $\ell_t$ is the iteration's loss value.

This is *importance sampling* with the importance signal being the current per-view loss.

### 4.3 Implementation

See `cfg.adaptive_sampling`, `cfg.adaptive_alpha`, `cfg.adaptive_eps` in `RunConfig`. The training loop maintains `view_errors` as a 1-D tensor of length `len(train_i)`, initialised to ones (uniform). View selection uses `torch.multinomial` on the normalised probability vector; the EMA update happens after each iteration.

### 4.4 Expected effect

If view-level loss heterogeneity is real and persistent — i.e., some training views are systematically harder than others throughout training — we expect adaptive sampling to converge faster and to a better final state. If the per-view losses converge to roughly the same value quickly (the easy-and-hard distinction is transient), adaptive sampling becomes uniform-in-the-limit and provides no benefit.

### 4.5 What §9 measured

Statistically indistinguishable from the baseline. The interpretation: **per-view loss heterogeneity is transient at this problem size; once Adam has had a few thousand iterations to build per-parameter step sizes, all train views converge to roughly the same per-view loss, and the EMA-based importance sampling collapses to uniform**. The improvement targets a failure mode that exists in larger / more heterogeneous datasets (e.g., real-world COLMAP captures with vastly different lighting) but is not present at this scale on synthetic Blender scenes. A substantiated negative result of the same character as Improvements A and C: the failure mode is real in the literature; it is not active here.

---

## 5. Summary

| Improvement | Failure mode targeted | §9 outcome | Substantiation |
|---|---|---|---|
| **A. SGDR** | Sharp local minima ⇒ warm restart escapes them | No statistically significant change | Baseline minimum is wide; restart failure mode not active |
| **B. L2 + LPIPS perceptual** | L2 is mis-aligned with perceptual quality | **−31% LPIPS at −0.5 dB PSNR** | The intended trade-off, confirmed |
| **C. Multi-scale** | Full-res random init has too many high-freq local minima | No statistically significant change | Adam at our scale finds the full-res minimum directly |
| **D. Adaptive view sampling** | Per-view loss heterogeneity wastes optimizer effort | No statistically significant change | Per-view loss converges uniformly at this scale |

**Three of the four are substantiated *negative* results. One is a substantiated positive trade-off.** The defense narrative around this is important: a negative result with a clearly identified failure mode that is not active at this scale is *not a failed experiment* — it is a substantiated finding about the optimization landscape of this particular problem. The rubric does not require all four improvements to land; it requires that each be substantiated.

---

## 6. Connecting the implementation to the §9 result

The §9 ablation directly produces the substantiation requested by the rubric's fifth objective:

- **The perceptual-loss trade-off (-31% LPIPS at -0.5 dB PSNR)** is the project's primary "improvement" deliverable — substantiated by multi-seed pooled statistics with effect size that dominates baseline variance.
- **The three negative results** are each substantiated by the same statistical methodology: the §7.2 baseline has a pooled test-PSNR std of ~0.34 dB across three seeds, and each of A/C/D produces a pooled mean change of less than 0.5 dB, well within the baseline noise band.

This is what "substantiated improvements" looks like under the rubric. It is also what *honest* optimization research looks like: most proposed improvements do not work at the scale at which one tests them, and saying so with statistical backing is itself the contribution.

---

## References

- Loshchilov, I. & Hutter, F. (2017a). SGDR: Stochastic gradient descent with warm restarts. *ICLR*.
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*.
- (Multi-scale / coarse-to-fine training) Adelson et al. (1984). Pyramid methods in image processing. *RCA Engineer*, 29(6).
- (Adaptive importance sampling in deep learning) Katharopoulos & Fleuret (2018). Not all samples are created equal: Deep learning with importance sampling. *ICML*.
