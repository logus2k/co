# 10. Gaussian Splatting Baseline

This section implements a 3D Gaussian Splatting (GS) baseline on Lego and Drums, so the NeRF-vs-GS contrast of Section 11 can be drawn on identical data and metrics. The implementation builds on the differentiable rasteriser from gsplat v1.5.3 (Nerfstudio), a PyTorch implementation of the Kerbl et al. 2023 algorithm.

## 10.1 GS as an alternative parameterisation of the same problem

From an optimization perspective, NeRF and GS are two parameterisations of the same reconstruction problem; both solve $\min_\theta \sum_\pi \mathcal{L}(R(\theta; \pi), I(\pi))$ for a differentiable rendering operator $R$. NeRF makes θ the ~58,000 weights of a small MLP, with gradients flowing through the volume-integration renderer. GS makes θ the per-Gaussian tuples (centre position, scale, rotation, opacity, colour) of roughly 10⁵ explicit primitives, with gradients flowing through a tile-based rasteriser. Section 11's comparison is therefore a comparison of two optimization formulations, not of two engineering choices.

Training uses the L1+SSIM loss with α = 0.2 (the published 3DGS choice, matching Section 8's L1+SSIM row), five Adam optimisers in parallel with the per-parameter-group learning rates of the original paper, 30,000 initial Gaussians placed uniformly in the scene cube, and 15,000 iterations at the dataset's native 800 × 800 per scene.

## 10.2 Four implementation details required for correct GS training

A from-scratch implementation has to handle four properties of the optimization problem that are not explicit in the published pseudocode. Each one, if left unhandled, produces an identifiable failure mode rather than a near-miss, which is itself informative about how GS training behaves.

- **OpenGL → OpenCV camera convention.** Without conversion, every Gaussian sits behind the rasteriser's camera, all are culled before rasterisation, the loss has zero gradient with respect to every Gaussian parameter, and validation PSNR stays at ~9.5 dB across every evaluation point with no obvious error message. Fix: left-multiply the world-to-camera matrix by diag(1, −1, −1, 1) before passing it to the rasteriser.

- **Scale-aware positional jitter on clones.** Naive clone densification copies a parent's tuple verbatim, placing two Gaussians at exactly the same position; they receive identical gradients in every step and stay co-located forever, adding parameter count without representational capacity. Fix: jitter the child position by scale-aware Gaussian noise, μ_child = μ_parent + N(0, exp(s_parent)), so the two Gaussians are spatially distinct from step one.

- **Adam state transplant across densification.** Naive densification creates new parameter tensors and therefore new Adam optimisers, wiping the first- and second-moment estimates every 100 iterations so Adam never accumulates the steady-state variance estimate it is designed to build. Fix: transplant the old moments and step counter into the new optimiser, indexed by the survivor-and-clone mapping; survivors keep their warmed-up moments and clones inherit their parent's.

- **Opacity reset off by default.** The basic Kerbl recipe periodically resets all opacities to force redundant Gaussians to re-earn presence. Empirically too aggressive on Lego at 800 × 800 (shrinks the model to a few thousand Gaussians and converges to similar PSNR / SSIM / LPIPS at much sparser representation), so the reset machinery is left disabled and gradient-based opacity erosion produces the pruning signal organically. The reset remains a documented hyperparameter for future work.

The configuration fed into Section 11 is the four fixes applied, opacity reset off, L1+SSIM with α = 0.2, 15,000 iterations at 800 × 800. Wall-clock is approximately 4.5 minutes per scene.
