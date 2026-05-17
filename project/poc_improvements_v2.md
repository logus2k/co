# PoC Improvements: Baseline for Further Development

## Purpose

This document captures what the current PoC (`src/nerf_poc.ipynb`) has demonstrated, what remains to be developed before the full project is compliant with the five project objectives, and the prioritised order in which to tackle the remaining work.

Scope reminder: the May 18 submission is the 3-4 page work plan ([work_plan_tutorial1.md](work_plan_tutorial1.md)). The PoC notebook is private verification that the pipeline works; it does not have to be fully compliant by itself. The list below describes the path from current PoC state to full-project deliverable.

---

## What the PoC has demonstrated

- A complete training-to-rendering pipeline runs end to end on a 4090.
- `MyAdam` is implemented from scratch on top of PyTorch autograd and successfully trains the model. This is the central CO contribution at PoC stage.
- Training loss converges cleanly (roughly 1.2 to 1.5 orders of magnitude over 5000 iterations) with no instability or divergence.
- PSNR on a held-out view is 23.94 dB on the Tiny NeRF Lego scene.
- Novel-view synthesis works: a 60-frame orbit produces recognisable renders at viewpoints never seen during training, confirming the model learned a coherent 3D representation rather than memorising images.

These results are sufficient to confirm that the planned full project is feasible.

---

## Gaps against each project objective

### Objective 1: Formulate an optimization problem

Missing from the notebook (present in the work plan):
- Markdown cells stating the loss function, the parameters being optimised, the rendering operator, and the formulation choices (loss, regularisation, view sampling, encoding frequency).
- Explicit statement that the problem is continuous, non-convex, and gradient-based.

The work plan has the mathematical formulation. The notebook should mirror it in plain Markdown so the deliverable is self-contained.

### Objective 2: Select and implement appropriate optimization methods

Missing:
- Only one optimizer (Adam) is implemented from scratch. The work plan commits to SGD, SGD with classical momentum, SGD with Nesterov momentum, Adam, and AdamW, plus a learning-rate scheduler.
- Each additional optimizer is roughly 10 to 40 lines and should be written by hand, not imported from `torch.optim`.

### Objective 3: Analyze convergence, stability, and performance

Missing:
- Reproducibility: no `torch.manual_seed(...)` or `np.random.seed(...)` calls. Should be set in the globals cell.
- Stability: single seed per configuration. The work plan promises at least three seeds per condition with mean and standard deviation reported.
- Hyperparameter sensitivity: no learning-rate sweep. Each optimizer has its own useful range and the comparison only makes sense once each method has been individually tuned.
- Time measurement: no wall-clock instrumentation. The course requirements include scalability and efficiency considerations; per-iteration time and time-to-target-quality should be logged.
- Convergence plots: only the training loss is plotted. Held-out PSNR (and SSIM) as a function of iteration is more informative because it tracks generalisation, not just training fit.

### Objective 4: Compare different approaches and justify methodological choices

Missing entirely. The PoC has one method; the full project needs at least the following comparisons:
- Optimizer comparison: five custom optimizers against each other on the same scene, same seed budget, same iteration budget.
- Loss formulation comparison: L1, L2, SSIM, and a weighted L1+SSIM combination.
- Representation comparison: NeRF (custom implementation) against 3D Gaussian Splatting (reference implementation) on the same scene.

The Gaussian Splatting comparison is a substantial piece of work in its own right: it has its own training pipeline, its own rendering kernels, and its own evaluation conventions. It is not "wrapping existing code in loops". Plan for it as a separate workstream of multiple days.

### Objective 5: Propose well-founded improvements based on results

Missing:
- No "Discussion and Future Work" section in the notebook.
- The work plan lists adaptive view sampling, multi-scale (coarse-to-fine) training, learning-rate restarts, and a perceptual loss. Each should be implemented, run against the best baseline configuration, and reported as an ablation with numbers.

### Evaluation metrics

Currently only PSNR is reported.
- SSIM should be added (one line with `skimage.metrics.structural_similarity` or a torch-native equivalent).
- LPIPS (perceptual distance using a pretrained network) is a stronger perceptual metric and aligns with the planned improvements story; the `lpips` package is a one-line install.
- Wall-clock time per iteration and total time-to-quality should be reported alongside the quality metrics.

### Data and dataset hygiene

- Train/test split currently holds out exactly one image. For the full project, hold out at least 5 to 10 views; use them all in the evaluation; report mean and std across them.
- The Tiny NeRF dataset is preprocessed to 100x100. For the full project the standard NeRF synthetic dataset (800x800 with proper JSON-described splits) is the right substrate. The PoC pipeline migrates to it with modest changes to the data-loading cell.

### Notebook structure and reporting

- Every code section should be preceded by a short Markdown cell stating what it does and why.
- After each experiment, a brief Markdown analysis cell should interpret the result (not just show the plot).
- Plots should have axis labels, titles, and legends without exception. Loss curves should use log-y. Convergence curves should overlay multiple methods for easy comparison.
- A final "Discussion and Future Work" section should summarise findings and tie them back to the formulation choices defended at the top.

---

## Prioritised work plan

Items are tagged by leverage per hour. Higher-leverage items are roughly "cheap to add, high impact on grading or clarity"; lower-leverage items are substantial workstreams.

### Tier 1: cheap, high-impact (do before or right after the work-plan submission)

1. Add `torch.manual_seed(0)` and `np.random.seed(0)` to the globals cell. Reproducibility groundwork for everything that follows.
2. Add Markdown formulation cells at the top of the notebook and short Markdown intros to each code section. Most of the prose can be pulled from the work plan.
3. Add SSIM alongside PSNR in the evaluation cell.
4. Add `time.time()` brackets around the training loop and the rendering loop. Report iterations per second and total time.
5. Add a final "Discussion and Future Work" Markdown cell listing the planned improvements from the work plan.

### Tier 2: medium-effort, project-essential (start the week after submission)

6. Implement `MySGD` (with optional momentum and Nesterov) from scratch alongside `MyAdam`.
7. Implement `MyAdamW` (Adam with decoupled weight decay) from scratch.
8. Add a learning-rate scheduler (cosine annealing or warmup-then-decay).
9. Refactor the training loop to take any of the custom optimizers as a parameter, so comparisons are runs of the same loop with different `opt` objects.
10. Migrate from `tiny_nerf_data.npz` to the full NeRF synthetic dataset; load via `transforms_*.json`. Keep the Lego scene as primary, add at least one more for cross-scene comparison.
11. Hold out a real test set of 5 to 10 views; aggregate metrics across them.

### Tier 3: large workstreams (allocate dedicated time)

12. Run the optimizer-comparison experiment: five methods on at least two scenes, three seeds each, with LR pre-tuned per method. Produce convergence-vs-iteration and convergence-vs-time plots overlaying all methods.
13. Run the loss-formulation experiment: L1, L2, SSIM, L1+SSIM, with the best optimizer from step 12.
14. Set up the 3D Gaussian Splatting baseline using an open-source reference implementation. Train on the same scenes, evaluate on the same held-out views, with the same metrics.
15. Implement and evaluate the planned improvements: adaptive view sampling, multi-scale training, LR restarts, and a perceptual loss. Each as an ablation against the best baseline configuration.

### Tier 4: nice to have, dependent on time

16. LPIPS in addition to SSIM.
17. `torch.compile(model)` plus fp16 inference for faster rendering during development (already drafted; can be folded in when convenient).
18. WebGL viewer for the Gaussian Splatting output ("wow-factor" demo for the presentation).

---

## What is deliberately not in this document

- A claim that the PoC "is missing" anything required for the May 18 submission. The May 18 deliverable is the work plan; the notebook is for internal verification.
- A claim that custom CUDA kernels or CuPy are needed. PyTorch is already running on CUDA; performance work belongs at the `torch.compile` / fp16 / batching level.
- A schedule with dates. The schedule belongs in the work plan, not here. This document is a feature list, prioritised.

---

## Suggested update cadence

This document should be edited as items are completed, with a date stamp on each completion. When all Tier 1 items are done, mark Tier 1 as complete and move on. When Tier 3 starts, this file becomes the authoritative source on remaining work; the work plan can stay as the historical record of intent.
