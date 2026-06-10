# 8. Loss Function Comparison

This section compares four loss formulations, L2, L1, SSIM, and the weighted combination L1+SSIM, under the winning optimizer from Section 7 (Adam at η = 10⁻³) and the same 40,000-iteration budget. SSIM and L1+SSIM consume contiguous 64 × 64 patches of rays (4,096 per gradient update, the same total batch size as the pixel-wise losses); L1 and L2 use scattered pixel sampling. The sampling mode is selected automatically from the loss, so the comparison is sample-budget-fair: every row sees the same number of pixels per gradient update, only the geometric arrangement differs. Each loss is run on Lego and Drums across three seeds.

## 8.1 Results

**Figure 5** - Validation PSNR over training iterations, averaged across seeds, one curve per loss formulation, one subplot per scene.

[Insert the same Figure 5 image used in the Word baseline.]

L2 leads on PSNR, the spatial losses lead on perceptual metrics, but L1 diverges. L2 produces the highest pooled test PSNR (22.01 ± 0.34 dB, identical to the Section 7.2 baseline since L2 is the Section 7.2 loss). SSIM and L1+SSIM essentially tie L2 on PSNR (21.70 and 21.75 dB) while winning on LPIPS by a wide margin (0.119 and 0.118 versus L2's 0.139). The two spatial losses are statistically indistinguishable from each other on every metric: the α = 0.2 weight on L1 in L1+SSIM is small enough that SSIM dominates the gradient signal. The Section 11 comparison with Gaussian Splatting (which trains under L1+SSIM at the same α = 0.2) uses this row as its NeRF-side loss baseline, so the comparison is fair on the loss axis as well as the optimizer axis.

L1 collapses to 14.11 ± 5.85 dB. This is not a measurement of "L1 produces a 14 dB reconstruction"; it is a measurement of three Lego seeds producing about 21 dB and three Drums seeds collapsing to about 10.8 dB (the "predict the mean colour" baseline). The 5.85 dB pooled standard deviation reflects that approximately one in three training runs diverges. Section 8.2 probes this directly.

## 8.2 Refining the L1 learning rate with Bayesian optimization

The loss comparison forced L1 to run at the learning rate selected for L2. There is no reason to expect a single rate to be appropriate for L1's sign-based gradients as well, so this section runs a dedicated learning-rate sweep for L1 using Bayesian optimization. The technique is the Tree-structured Parzen Estimator (TPE), the canonical Bayesian global-optimization method for hyperparameter spaces, with a median pruner that terminates below-median trials after a warm-up. The study runs twelve trials with a log-uniform prior η in [10⁻⁶, 10⁻²], the full 40,000-iteration budget per trial, and a pruner warm-up at one-fifth of that budget. Because the loss comparison also identified L1's sign-based gradient as a candidate failure mode, every trial additionally runs with a cosine warm-up schedule and gradient clipping at maximum norm 1.0.

The TPE search found η* = 3.83 × 10⁻⁴ as the best single-seed Lego rate, with best-validation PSNR reaching 22.83 dB. The multi-seed validation tells a different story. The pooled refined L1 test PSNR is 14.11 ± 5.85 dB, statistically identical to the unrefined Section 8.1 baseline of 14.11 ± 5.91 dB. The cosine schedule and gradient clipping migrated which Lego seed diverges, but the pooled mean and variance are unchanged.

Direct gradient-norm inspection during training reveals why clipping was a no-op: L1's gradient L²-norm stays around 0.1 throughout training, well below the max_norm = 1.0 threshold. The instability is therefore not a magnitude phenomenon, which is what the standard "gradient clipping fixes Adam-driven instability" recipe is built to address. The cause lies on a different axis.

L1's gradient is (1/N) · sign(Î − I), which has constant magnitude per pixel. Adam's second-moment estimate stays approximately constant rather than tracking landscape curvature, the per-parameter step η/√v̂ becomes a constant scaling, but the sign of the step alternates rapidly near a minimum because constant-magnitude steps cannot settle the way gradient-magnitude steps can. Some seeds fall into the resulting sign-oscillation regime, others do not. The failure mode is sign-oscillation under Adam's variance estimator, not gradient magnitude, which is exactly why magnitude-based remedies cannot fix it.

The substantiated finding is that L1 is structurally incompatible with Adam at long iteration budgets on this problem class, and no learning-rate, schedule, or clipping intervention recovers stability. For applications where L1 is the desired loss, the practical recommendation is to use an optimizer without second-moment normalisation, plain SGD with momentum.
