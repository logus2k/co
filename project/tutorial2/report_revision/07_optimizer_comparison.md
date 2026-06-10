# 7. Optimizer Comparison

This section compares the five optimizers of Section 4 head-to-head under the methodology of Section 6. Two complementary results are produced: a learning-rate sweep that identifies the right rate for each method, and a multi-seed comparison that runs every optimizer at its best rate across multiple seeds and scenes. The sweep is essential because Adam and SGD want learning rates several orders of magnitude apart on this NeRF objective.

## 7.1 Learning-rate sweep

The same logarithmic grid is tested for every optimizer: η ∈ {10⁻⁴, 3 × 10⁻⁴, 10⁻³, 3 × 10⁻³, 10⁻², 3 × 10⁻², 10⁻¹, 3 × 10⁻¹}, on a reduced 10,000-iteration budget because the relative ordering is established well before convergence. The sweep runs on Lego with a single seed; multi-seed averaging is reserved for Section 7.2. For each optimizer the rate maximising the best validation PSNR during the run is selected.

**Figure 1** - Validation PSNR as a function of learning rate, one curve per optimizer, evaluated on the Lego scene.

[Insert the same Figure 1 image used in the Word baseline.]

The sweep produces the textbook two-family pattern. Adam and AdamW peak at η = 10⁻³ (best validation PSNR around 23 dB) and diverge for η ≥ 10⁻¹, where the per-parameter adaptive scaling makes a large global rate catastrophic. The SGD family peaks at η = 3 × 10⁻¹, the upper edge of the grid, with maxima of 19.5, 21.2, and 22.0 dB for SGD, momentum, and Nesterov. Selected rates passed to Section 7.2: η_adam = η_adamw = 10⁻³ and η_sgd = η_momentum = η_nesterov = 3 × 10⁻¹.

The methodological lesson is that the well-tuned learning rates for the two optimizer families are separated by roughly 300x. Any comparison using a single shared rate would either run Adam in its divergence regime or run SGD effectively unmoved.

## 7.2 Multi-seed optimizer comparison

With each optimizer's rate fixed by Section 7.1, every optimizer is run at its selected rate across three seeds, two scenes (Lego, Drums), and the full 40,000-iteration budget, with LPIPS enabled. Five optimizers × two scenes × three seeds gives 30 runs.

**Figure 2** - Validation PSNR over training iterations, averaged across seeds, one curve per optimizer, one subplot per scene.

[Insert the same Figure 2 image used in the Word baseline.]

**Table 2** - Wall-clock seconds to first reach 20 dB validation PSNR, mean ± standard deviation across seeds and scenes.

| Optimizer | Mean | Std |
|---|---|---|
| SGD | 169.56 | 12.63 |
| Momentum | 117.92 | 68.38 |
| Nesterov | 87.04 | 45.95 |
| Adam | 35.93 | 13.63 |
| AdamW | 36.13 | 12.89 |

Adam wins narrowly but consistently. Pooled across the two scenes and three seeds, its mean test PSNR is 22.01 ± 0.34 dB; AdamW essentially ties at 22.00 ± 0.38 dB; Nesterov reaches 21.74 dB, momentum 21.68 dB, plain SGD trails at 20.19 dB. The ranking is identical on both scenes, and the per-scene seed standard deviations of 0.05 to 0.40 dB are roughly an order of magnitude smaller than the Adam-to-SGD gap, so the ordering is reliable rather than seed noise.

The headline methodological finding follows from comparing this multi-seed result to the scoping-study separability check. That earlier check ran Adam and SGD at a shared learning rate of 5 × 10⁻⁴ and measured a 12.77 dB gap in Adam's favour. Once each method is given its own learning-rate-sweep-selected rate, the gap collapses to 1.82 dB. Almost eleven dB of the apparent Adam advantage is a learning-rate-mismatch artefact rather than a property of the optimizer. A fair comparison of first-order methods requires per-method learning-rate tuning; without it, the comparison measures the mismatch instead of the method.

The remaining gaps are themselves informative. Momentum and Nesterov give SGD roughly 80% of Adam's quality lift, rising from 20.19 dB to about 21.7 dB. This confirms that the smoothed first-moment estimate the SGD family already computes recovers most of the benefit Adam's adaptive scaling provides on this problem; the additional 0.3 dB Adam contributes comes from its diagonal second-moment estimate acting as a coarse curvature surrogate. Nesterov's lookahead correction over plain momentum is marginal at +0.06 dB.

LPIPS sharpens the ranking that PSNR softens. The pixel-level PSNR gap from SGD to Adam is about 9%, but the perceptual-distance gap is almost twice that: LPIPS 0.272 versus 0.139. SGD reaches reconstructions that are roughly correct on average but visually degraded in ways pixel metrics underweight, namely the high-frequency detail the perceptual network is sensitive to. This is the case for reporting all three metrics: a single number would have hidden the perceptual gap.

Throughput is essentially flat at 80 to 90 iterations per second across all five methods, so the comparison is one of quality at a fixed compute budget rather than of cheaper compute.
