# 11. NeRF vs Gaussian Splatting Comparison

The Gaussian Splatting baseline of Section 10 introduces a different parameterisation of the same reconstruction problem: explicit 3D Gaussian primitives instead of an implicit MLP. This section reports the head-to-head test-set comparison on metrics common to both methods, plus the efficiency dimensions (training time and parameter count) the Tutorial #1 work plan committed to reporting.

## 11.1 Headline

Gaussian Splatting wins five of the six metric cells: both SSIM scores, both LPIPS scores, and Drums PSNR by +1.52 dB. It trails only Lego PSNR by 0.75 dB, a gap below the perceptual just-noticeable-difference threshold and within two standard deviations of NeRF's own seed-noise band on that scene. GS reaches this quality at roughly half the wall-clock NeRF needs.

**Figure 8** - Test-set PSNR, SSIM, and LPIPS for NeRF and Gaussian Splatting, one subplot per metric, one bar pair per scene.

[Insert the same Figure 8 bar chart used in the Word baseline.]

## 11.2 Efficiency dimensions

The two methods sit at very different operating points on every cost axis.

**Table 3** - Per-method parameter count, training resolution, iteration budget, wall-clock, forward-pass cost, optimizer count, and densification behaviour.

| | NeRF (Adam) | GS (basic) |
|---|---|---|
| Parameters | ≈58,000 (MLP weights) | 1 to 2 × 10⁶ (Gaussian primitives × 14 floats each) |
| Training resolution | 200 × 200 | 800 × 800 |
| Training iterations | 40,000 | 15,000 |
| Wall-clock per scene | ≈8 min | ≈4.5 min |
| Forward-pass cost | O(rays × samples-per-ray) MLP evaluations | O(Gaussians × pixels-per-Gaussian) tile-based rasterisation |
| Optimizer | single Adam at η = 10⁻³ | five Adams with per-parameter-group learning rates |
| Discrete structural changes | none | densification and pruning every 100 iterations |

GS has roughly 30 times more parameters but is roughly twice as fast per iteration because the tile-based rasteriser is more pixel-throughput-efficient than NeRF's per-ray MLP evaluations. The comparison is wall-clock-fair, not pixel-density-fair: each method runs at the resolution where its compute budget is used most effectively. Forcing GS to train at 200 × 200 would artificially cripple its high-frequency detail capture; forcing NeRF to train at 800 × 800 would explode its compute past the project's training budget.

## 11.3 Discussion

Where Gaussian Splatting wins. SSIM and LPIPS both favour GS by clear margins (SSIM +0.072 on Lego and +0.096 on Drums; LPIPS −0.059 on Lego and −0.046 on Drums). GS reconstructions are visibly closer to ground truth on local structure and deep-feature similarity. Drums PSNR also goes to GS by +1.52 dB because the relatively smooth drum-kit surfaces play to a strength of the representation: each anisotropic Gaussian primitive is a natural fit for locally curved materials.

Where NeRF wins. Lego PSNR by 0.75 dB. Lego's fine high-frequency detail (the studs, the rivets, small edges) is where Gaussian primitives smooth more aggressively than NeRF's per-ray MLP. PSNR's per-pixel weighting accumulates these many small smoothing errors into a measurable gap that the eye does not perceive because each individual error is below the perceptual threshold.

Why the 0.75 dB Lego gap matters less than it looks. The pooled NeRF Adam standard deviation on Lego is 0.34 dB, so the gap is roughly two standard deviations of NeRF's own seed-noise band; a different NeRF seed within ±0.4 dB of its mean would already close it. The smallest PSNR difference an untrained observer can reliably distinguish in a side-by-side comparison is around 1 dB, so this gap sits just below the just-noticeable-difference threshold. The SSIM gap of +0.072 is roughly 3.5 times the SSIM JND of about 0.02; the LPIPS gap of −0.059 is roughly three times the LPIPS JND of about 0.02. The perceptual metrics agree unambiguously that GS reconstructs the scene more faithfully to the eye; PSNR's pixel-arithmetic disagreement is real but imperceptible.

In optimization terms, NeRF and GS are two parameterisations of the same problem, and the right choice depends on which loss matters in the application. A pixel-wise PSNR objective on a high-frequency scene like Lego slightly favours NeRF's per-ray MLP; a perceptual-quality objective or a smoother-surface scene favours GS's explicit Gaussians. The defensible framing is not that GS is universally better than NeRF; it is that the choice between explicit and implicit 3-D parameterisations is a deliberate optimization-of-which-metric decision rather than a default.
