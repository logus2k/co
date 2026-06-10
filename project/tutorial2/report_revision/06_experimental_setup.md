# 6. Experimental Setup

This section fixes what every experiment in the project holds constant. Section 6.1 selects the rendering resolution, MLP architecture, and iteration budget by measurement; Section 6.2 defines the splits and metrics; Section 6.3 describes the training procedure.

## 6.1 Scoping study

The main comparison matrix has roughly 100 training runs once optimizers, losses, seeds, and scenes are crossed. Two settings apply uniformly to every run, the rendering resolution and the MLP architecture, and together they determine both the total compute the matrix consumes and the quality regime in which the optimizer differences will be visible. The scoping study fixed both empirically on the Lego scene before the main matrix was committed.

Per-iteration training cost is essentially independent of rendering resolution because the NeRF step samples a fixed batch of rays per iteration. Reconstruction quality, however, decreases with resolution at a fixed iteration budget: a 1024-ray batch covers about 10% of a 100 × 100 image and about 0.16% of an 800 × 800 image, so higher resolutions are under-trained at the chosen budget. A dedicated SGD-vs-Adam separability check confirmed that 200 × 200 distinguishes the optimizers fully: the Adam − SGD best-PSNR gap was 12.77 dB at 200 × 200, marginally larger than the 12.00 dB measured at 400 × 400. Rendering resolution is selected as 200 × 200.

A larger MLP (8 layers of width 256 with the standard NeRF skip connection, about 494,000 parameters) improved best PSNR by only about 0.35 dB over the small MLP (4 layers of width 128, about 58,000 parameters), at roughly 3.7x the per-iteration cost. A plain 8-layer feed-forward MLP without the skip connection failed to train. The small MLP is selected; absolute PSNR is not the objective of an optimizer-comparison study. The iteration budget is fixed at 40,000 per run, so the comparison measures which optimizer reaches the best state within a fixed compute budget.

## 6.2 Splits and evaluation metrics

Each nerf_synthetic scene ships with three disjoint pose sets, which are used directly as the train, validation, and test splits. Validation supports convergence tracing and the learning-rate sweep of Section 7; the test set is used only for the final reported metrics.

Reconstruction quality is measured three ways: PSNR (decibels, higher better; a logarithmic transform of mean squared error), SSIM ([0,1], higher better; structural similarity comparing local luminance, contrast, and structure), and LPIPS ([0,1], lower better; a learned perceptual distance through a pretrained AlexNet backbone). Reporting all three is the project's empirical analogue of Module 1's analytical critical-point classification: a non-convex stochastic optimum cannot be proven globally optimal, but it can be confirmed operationally good if held-out PSNR, SSIM, and LPIPS are consistent across seeds and scenes.

## 6.3 Experimental methodology

Every comparison is built from individual training runs. A run is fully described by six choices: optimizer, loss, scene, random seed, learning rate, and iteration budget. The training procedure is the same across every section: it constructs the model, optimizer, and loss from the configuration, runs the stochastic training loop with periodic validation, evaluates on the held-out test set, and records the complete iteration-level history together with the final test metrics. Holding this procedure fixed is what makes the comparison fair: optimizers, losses, scenes, and seeds differ only in the component under study. Trained model weights persist alongside the metrics, so any saved model can be rendered from a previously unseen viewpoint, the operational test that the optimization succeeded.
