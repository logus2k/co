# **Computational Optimization - Tutorial #1: Work Plan**

**Project title:** Optimization for 3D Scene Reconstruction: A Comparative Study of Optimizers, Losses, and Representations for NeRF and 3D Gaussian Splatting

**Group:** António Cruz, Duarte Cabrita

**Submission date:** May 18, 2026

---

## 1. Problem Identification

3D scene reconstruction from a small set of 2D photographs is a foundational task in computer vision and AI. Given a collection of images captured from different viewpoints, the goal is to recover a representation of the underlying scene that allows new, previously unseen views to be rendered with high fidelity. The task underpins applications across robotics, augmented and virtual reality, autonomous driving, virtual production, and digital heritage preservation.

Two recent approaches have defined the state of the art:

- **Neural Radiance Fields (NeRF)**, introduced in 2020, represents the scene as a continuous function parameterised by a small multi-layer perceptron that maps a 3D point and a viewing direction to a colour and a density.
- **3D Gaussian Splatting**, introduced in 2023, represents the scene as a large collection of explicit 3D Gaussian primitives, each with a position, scale, rotation, opacity, and colour.

Both approaches frame scene reconstruction as a continuous, non-convex optimization problem: find the parameters of the chosen representation such that, when rendered from the known camera viewpoints, they best reproduce the captured photographs.

This project formulates and studies that optimization problem in depth. We are interested not only in producing visually convincing 3D reconstructions, but in understanding how the choice of optimizer, loss function, regularisation, and representation affects convergence behaviour, training stability, and final reconstruction quality.

---

## 2. Optimization Problem Formulation

Let `θ` denote the parameters of the chosen scene representation:
- For NeRF, `θ` is the set of weights and biases of a small MLP.
- For 3D Gaussian Splatting, `θ` is the collection of 3D Gaussian positions, scales, rotations, opacities, and colours.

Let `R(θ; π)` denote the differentiable rendering operator: given the parameters `θ` and a camera pose `π`, it produces a synthesised image. Let `{(I_i, π_i)}_{i=1..N}` be the captured photographs and their associated camera poses (recovered by Structure-from-Motion).

The reconstruction problem is then:

```
min_θ   (1/N) Σ_i  L( R(θ; π_i),  I_i )   +   λ · Ω(θ)
```

where `L` is a per-image loss that measures the discrepancy between the rendered and observed images, and `Ω(θ)` is a regularisation term on the parameters with weight `λ ≥ 0`. Because the rendering operator is differentiable with respect to `θ`, the problem admits gradient-based optimization, although the loss landscape is non-convex and high-dimensional.

The formulation choices we will make and defend explicitly:

- **The image loss `L`.** Candidates include L1, L2, SSIM, and convex combinations of these. Each has known effects on which image details are preserved and how robust the optimization is to outliers.
- **The regularisation term `Ω(θ)`.** Candidates include weight decay (for the NeRF MLP), and density / opacity regularisers and adaptive density control heuristics (for Gaussian Splatting).
- **The view- and ray-sampling strategy.** Which rays or which views are presented to the optimizer at each iteration. Uniform random, importance sampling by current reconstruction error, and curriculum schedules are options.
- **The parameterisation choices.** Positional encoding frequency for NeRF; initial Gaussian distribution and density-control thresholds for Gaussian Splatting.

These choices are part of what the project will study, not fixed in advance.

---

## 3. Optimization Methods to be Used

We will implement and compare methods along two axes: the choice of scene representation, and the choice of optimizer.

### 3.1 Scene Representations

**NeRF, implemented from scratch.** We will write a compact implementation of the NeRF representation and the volume rendering integration in PyTorch. The implementation will be small (on the order of a few hundred to a few thousand lines of clear Python), so that every optimization choice remains under our control. This is the heart of the optimization study.

**3D Gaussian Splatting, using a published reference implementation.** We will use an open-source Gaussian Splatting implementation as a contemporary point of comparison. Reimplementing the custom CUDA rasteriser is out of scope; the contribution of our project is to study how the optimization behaviour of NeRF compares with that of Gaussian Splatting on the same scenes, and how our optimization choices on the NeRF side affect the gap.

### 3.2 Optimizers (applied to NeRF)

We will implement the following optimizers ourselves on top of PyTorch's automatic differentiation, and compare them on the NeRF reconstruction problem:

- Stochastic Gradient Descent (SGD)
- SGD with classical momentum
- SGD with Nesterov accelerated momentum
- Adam
- AdamW (Adam with decoupled weight decay)
- A learning-rate schedule (cosine annealing or warmup-then-decay) on top of the best of the above

Writing the optimizers from scratch is essential: it ensures the project is a genuine study of computational optimization rather than a benchmarking exercise of PyTorch's library internals.

### 3.3 Loss Formulations

We will compare:

- L1 loss between rendered and observed pixels
- L2 loss
- SSIM loss
- A weighted combination of L1 and SSIM (as used in the Gaussian Splatting paper)

---

## 4. Analysis Plan

For each combination of representation, optimizer, and loss, we will measure and report:

**Convergence behaviour:**
- Training loss as a function of iteration count
- Rendered-image PSNR and SSIM on held-out views as a function of iteration count
- Wall-clock time to reach a fixed target quality

**Stability:**
- Sensitivity to random initialisation across multiple seeds
- Sensitivity to learning rate (sweeps)
- Behaviour on scenes of different difficulty (textured vs textureless, dense vs sparse view coverage)

**Final quality:**
- PSNR and SSIM on held-out test views
- Qualitative side-by-side comparison of rendered novel views

All numerical comparisons will be averaged across at least three seeds, and we will report variability alongside means.

---

## 5. Planned Improvements

Based on the initial comparison results, we will investigate the following improvements:

- **Adaptive view sampling.** Weight the sampling of training rays by current reconstruction error rather than uniformly. Expected to accelerate convergence on difficult regions.
- **Multi-scale (coarse-to-fine) training.** Train initially at lower image resolution and progressively increase. Expected to improve recovery of coarse structure and reduce time-to-convergence.
- **Learning-rate restarts.** Periodically reset the learning rate to a higher value to escape sharp local minima in the non-convex landscape.
- **Perceptual loss.** Augment the pixel-wise loss with a feature-space loss based on a pretrained network. Expected to improve perceptual quality even when PSNR is similar.

Each improvement will be evaluated against the best-performing baseline configuration and reported as an ablation.

---

## 6. Dataset, Compute, and Software

**Datasets.** We will capture our own dataset of one or two real-world scenes using a smartphone (approximately 30 to 60 images per scene), processed through COLMAP for Structure-from-Motion pose estimation. We will additionally evaluate on at least one standard scene from the NeRF synthetic dataset, to allow comparison with published numbers.

**Compute.** All training and evaluation will run on a local NVIDIA RTX 4090 (24 GB VRAM). A compact NeRF trains in tens of minutes to a few hours per scene on this hardware; Gaussian Splatting trains in a few minutes. The full sweep of optimizer and loss combinations is comfortably achievable within the project timeline.

**Software.** Python with PyTorch as the primary framework. Custom optimizer code and volume-rendering code will be written from scratch for NeRF. An open-source reference implementation will be used for Gaussian Splatting. COLMAP will be used for camera calibration.

---

## 7. Deliverable Format

We will submit the project as **Option A: Jupyter Notebook**, combining the mathematical formulation, the Python implementation, the training and evaluation loops, the convergence plots, the qualitative image comparisons, and the discussion. The notebook will be organised to allow reproducible re-running of all reported experiments.

---