# Tutorial 2 Execution Plan

## Purpose

This is the single execution source of truth for Phase 2 of the Computational Optimization project: every task from the current state to the delivered Tutorial 2 report and defence. It supersedes the tier list in `poc_improvements_v2.md` (which was written before the Phase 1 submission and is now historical).

Phase 2 is worth 60% of the course grade. Submission deadline: 11 June 2026. Online defence: 16-18 June 2026.

The committed scope is fixed by the submitted proposal (`computational_optimization_tutorial1.pdf`): NeRF and a volume renderer from scratch, five custom optimizers, four loss formulations, a Gaussian Splatting baseline, and four candidate improvements, delivered as a Jupyter Notebook plus a 15-20 page report derived from it.

---

## Current Status

**Done:**
- Stage 0 scoping benchmark: configuration locked (resolution 200x200, small MLP 128x4, 40,000-iteration budget). See Section 6.1.
- `src/nerf_tutorial2_v2_AC.ipynb` (the live notebook; previous versions are frozen) restructured into the 12-section target layout below (57 cells after the §7.1 inserts).
- All five optimizers implemented from scratch and split into Sections 4.1-4.6, plus the cosine-warmup LR schedule (Stage 2).
- L1 and L2 loss functions and the `make_loss` registry (Section 5).
- Stage 1 experiment harness: `nerf_synthetic` loader, train/val/test split, `RunConfig`/`RunResult`, the parameterized `run_experiment`, on-disk JSON logging, and the LPIPS metric. Validated end to end on CPU and confirmed on GPU via the §6.4 smoke test.
- §7.1 Learning-Rate Sweep on Lego (40 runs, 10,000-iter budget, 2026-05-27): Adam/AdamW peak at lr=1e-3 (~23 dB); momentum/Nesterov peak at lr=3e-1 (~21-22 dB); plain SGD peaks at lr=3e-1 (~19.5 dB) and is still climbing at the grid edge, so an upward extension for the SGD family at lr ∈ {1.0, 3.0} is pending.

**In flight:**
- §7.2 multi-seed optimizer comparison RUNNING as of 2026-05-27 (30 runs: 5 optimizers × {Lego, Drums} × 3 seeds × 40,000 iters; ~3 hours on a clean GPU; resumable via the per-run config-hash cache under `outputs/runs/`).

**Not done (remaining Phase 2):**
- SSIM and L1+SSIM losses and the patch-based ray sampling they need (Stage 3, surfaced in Section 8).
- Loss comparison (Section 8) and NeRF vs Gaussian Splatting comparison (Section 11).
- Gaussian Splatting baseline (Section 10).
- The improvements (at least one as a full ablation, Section 9).
- The written analysis prose: the §7.1 and §7.2 Interpretation cells, plus all of Sections 8-12.

§7 is fully fleshed out (intro, §7.1 LR sweep, §7.2 multi-seed comparison cells in place); Sections 8-12 still exist as titled placeholders.

---

## Target Notebook Structure

The notebook is the single source for both deliverables (Option A directly, Option B via the team's exporter). Root sections use a single `#`; subsections use `##`. Tables of numbers are produced as code-cell output (`pandas` DataFrames), never hardcoded Markdown.

```
# 1. Introduction
# 2. Problem Formulation
# 3. NeRF: Model and Differentiable Renderer
   ## 3.1 Data Loading
   ## 3.2 Ray Generation
   ## 3.3 Positional Encoding
   ## 3.4 NeRF MLP
   ## 3.5 Volume Rendering
# 4. Optimizers
   ## 4.1 SGD
   ## 4.2 SGD with Momentum
   ## 4.3 SGD with Nesterov Accelerated Gradient
   ## 4.4 Adam
   ## 4.5 AdamW
   ## 4.6 Learning-Rate Schedule
# 5. Loss Functions
# 6. Experimental Setup
# 7. Optimizer Comparison
# 8. Loss Function Comparison
# 9. Proposed Improvements
# 10. Gaussian Splatting
# 11. NeRF vs Gaussian Splatting Comparison
# 12. Conclusions and Future Work
```

This layout is in place as of 2026-05-21. The PoC's qualitative tools (orbit animation, render-by-index) were not carried into the restructured notebook: they depend on a trained model and remain in `src/nerf_poc.ipynb`; they will return as a qualitative showcase in Section 7 once the experiments produce trained models. Sections 7 to 12 currently exist as titled placeholders.

---

## Execution Stages

Each task has a checkbox. Mark `[x]` and date it on completion.

### Stage 0: Scoping benchmark and decision — COMPLETE (2026-05-21)

The resolution, MLP, and iteration budget were decided from measurement, not guessed. The investigation comprised four runs: a resolution sweep (100/200/400/800), a batch/iteration scale-up, an MLP-capacity test (small 128x4 vs large 256x8-with-skip), and an SGD-vs-Adam separability check at 200 and 400. Full code: `src/resolution_benchmark.ipynb` and the benchmark scripts in `src/benchmarks/`.

**Decision:**
- **Resolution: 200x200.** Best quality-per-compute; avoids the high-resolution coverage penalty; and the separability check confirmed it distinguishes optimizers fully — the Adam minus SGD gap was 12.77 dB at 200x200 versus 12.00 dB at 400x400, refuting the concern that 200 might be an uninformative regime.
- **MLP: small, width 128, depth 4 (~58k parameters).** The large MLP (256x8 with skip, 494k parameters) gained only ~0.35 dB for ~3.7x the per-iteration cost; not worth it for an optimizer-comparison study.
- **Iteration budget: fixed 40,000 iterations.** Phase 2 is an anytime-performance test at a fixed compute budget, not an open-ended run to each optimizer's own convergence.
- **Batch: 4096 rays.** Scene set: 2-3 `nerf_synthetic` scenes (Lego primary; 1-2 more selected in Stage 1).

**Key measured findings:** per-iteration training cost is resolution-independent (NeRF samples a fixed ray batch); reconstruction quality decreases with resolution at a fixed budget (coverage penalty); a plain 8-layer MLP fails to train without a skip connection; compute is not the binding constraint (a ~100-run matrix is a few hours at any resolution).

**Carry-forward for Phase 2:** the separability check ran SGD and Adam at the same learning rate (a clean control); the real Phase 2 comparison must give each optimizer its own learning rate via LR sweeps, or SGD will look broken rather than slow.

The Stage 0 writeup is in `src/nerf_tutorial2_v2_AC.ipynb` (Section 6.1).

### Stage 1: Experiment harness — COMPLETE (2026-05-21)

- [x] Migrate from `tiny_nerf_data.npz` to the full `nerf_synthetic` dataset; loader reads `transforms_*.json`. (`load_scene`, Section 3.1)
- [x] Implement a proper train / validation / test split (hold out 5-10 views, used for all reported metrics). (`load_scene_splits`, `N_VAL_VIEWS=5`, `N_TEST_VIEWS=10`, Section 6.2)
- [x] Refactor the training loop into a function taking `(optimizer, loss, scene, seed, lr, n_iterations)`. (`run_experiment` + `RunConfig`, Section 6.3)
- [x] Add a per-run config object and on-disk logging of loss history, PSNR/SSIM history, and timings, so runs are not lost. (`RunConfig` / `RunResult`, JSON files under `outputs/runs/`)
- [x] Add LPIPS as an evaluation metric alongside PSNR and SSIM. (`get_lpips` + `evaluate`, Section 6.2)

The harness was validated end to end on CPU and then confirmed on GPU via the §6.4 smoke test (2026-05-27): a 300-iter Adam run reached val PSNR 14.00 dB (rising), 48.7 iter/s, LPIPS computed, JSON logging confirmed.

### Stage 2: Optimizers — COMPLETE (2026-05-21)

- [x] Implement `MySGD` from scratch. (Section 4.1)
- [x] Implement SGD with classical momentum. (`MySGD` momentum mode, Section 4.2)
- [x] Implement SGD with Nesterov accelerated gradient. (`MySGD` nesterov mode, Section 4.3)
- [x] Keep `MyAdam` (already done); refactor to the shared optimizer interface. (Section 4.4)
- [x] Implement `MyAdamW` (decoupled weight decay) from scratch. (Section 4.5)
- [x] Implement the cosine-annealing-with-warmup learning-rate schedule. (`cosine_warmup_lr`, Section 4.6; wired into `run_experiment` via `use_schedule`)
- [x] Sanity-check each optimizer on one short run before the full sweep. (all five ran through the harness in the CPU validation; the GPU sanity run at the real config is folded into the Stage 4 LR sweep.)

All five optimizers share the `zero_grad()` / `step()` interface and are selected by name through `make_optimizer`.

### Stage 3: Loss functions

- [x] L2 (already done); refactor to the shared loss interface. (Section 5, `loss_l2`)
- [x] Implement L1. (Section 5, `loss_l1`)
- [ ] Implement SSIM loss. (Section 8 — needs patch-based ray sampling, since SSIM is a spatial measure that cannot be computed from scattered pixels)
- [ ] Implement the weighted L1 + SSIM combination. (Section 8, follows SSIM)

### Stage 4: Core experiments

- [x] **Learning-rate sensitivity sweep per optimizer.** (Section 7.1; 40 runs at 10k iters on Lego, 2026-05-27. `sweep_learning_rates` + `select_best_lr`. SGD-family upward extension at lr ∈ {1.0, 3.0} pending — their curves are still climbing at the 3e-1 grid edge.)
- [ ] **Optimizer comparison:** 5 optimizers, >= 3 seeds, fixed loss, on the chosen scenes. (Section 7.2 — RUNNING as of 2026-05-27: 30 runs across {Lego, Drums} × {seed 0, 1, 2} at 40k iters with LPIPS.)
- [ ] Loss comparison: 4 losses, >= 3 seeds, with the best optimizer from the previous step. (Section 8; depends on Stage 3 SSIM/L1+SSIM.)
- [ ] Produce comparison tables as `pandas` DataFrames (code-cell output). (§7.2 cell 55 produces the optimizer-comparison tables once the run completes.)
- [ ] Produce convergence plots overlaying methods (loss-vs-iteration and quality-vs-iteration, log axes, labelled). (§7.2 cell 56 produces the validation-PSNR overlay with seed-std bands.)

### Stage 5: Gaussian Splatting baseline

- [ ] Select and set up an open-source 3D Gaussian Splatting reference implementation.
- [ ] Train it on the same scenes as NeRF.
- [ ] Evaluate with the same metrics (PSNR, SSIM, LPIPS) on the same held-out views.
- [ ] Record training time and parameter count for the efficiency discussion.

### Stage 6: Improvements

- [ ] Choose which improvement(s) to implement, guided by the bottleneck observed in Stage 4 (the proposal commits to attempting all four; at minimum one must be a full ablation).
- [ ] Implement and run the chosen improvement(s): adaptive view sampling, multi-scale training, learning-rate restarts (SGDR), perceptual loss.
- [ ] Evaluate each as an isolated ablation against the best baseline configuration.

### Stage 7: Analysis and report writing

- [ ] Write the Introduction and Problem Formulation sections (can be adapted from the proposal).
- [ ] Write a Markdown intro before every code section and a Markdown interpretation after every experiment.
- [ ] Write the NeRF vs Gaussian Splatting comparison section (Section 11).
- [ ] Write Conclusions and Future Work (Section 12), tying findings back to the formulation.
- [ ] Verify all plots have titles, axis labels, and legends; all tables are DataFrame output.

### Stage 8: Deliverables and defence

- [ ] Restart-and-run-all the notebook from a clean kernel; confirm reproducibility.
- [ ] Export the notebook to Word with the team's converter; save as PDF (Option B).
- [ ] Clean up the PDF: remove unnecessary code and output, fix formatting.
- [ ] Final check: notebook (Option A) and report (Option B) both coherent and consistent.
- [ ] Prepare the defence presentation for the 16-18 June window.

---

## Conventions

**Exporter conventions (required for the report build):**
- Root sections use a single `#`; subsections use `##`, `###`.
- Tables of numbers are `pandas` DataFrames shown as code-cell output, not hardcoded Markdown tables.

**Notebook quality conventions:**
- Every code section is preceded by a Markdown cell explaining what it does and why.
- Every experiment is followed by a Markdown cell interpreting the result, not just showing the plot.
- Plots always have a title, axis labels, and a legend. Loss curves use a log y-axis. Convergence curves overlay the compared methods.
- Markdown prose is written as report prose, since it becomes the report body verbatim.

**Computational optimization conventions:**
- All five optimizers are implemented from scratch on top of PyTorch autograd; no `torch.optim` for the compared methods.
- All compared methods run under identical conditions: same data sampling, same initialisation, same iteration and seed budgets.

---

## Risks

- **Compute budget.** The run matrix (5 optimizers + 4 losses + LR sweeps, >= 3 seeds, 2-3 scenes) is 100+ training runs. Stage 0 must keep resolution and iteration count low enough that this is feasible. This is the single biggest scheduling risk.
- **Gaussian Splatting setup.** New infrastructure with its own dependencies and rendering kernels. Start Stage 5 early and in parallel with Stage 4, not at the end.
- **Report writing volume.** Option B is 15-20 pages of narrative. Writing the Markdown interpretation cells as the experiments are run (Stage 7 interleaved with Stages 4-6) avoids a writing crunch before the deadline.
- **Windows teammate.** One machine has an RTX 3050 on Windows without Triton; `torch.compile` is disabled there via the existing fallback. Heavy experiment runs should target the 4090.

---

## Critical Path

Stage 0 → Stage 1 → Stage 2 → Stage 4 is the spine. Stages 3, 5, and 6 can overlap once the harness (Stage 1) exists. Stage 7 (writing) should be interleaved with Stages 4-6, not deferred. Stage 5 (Gaussian Splatting) is the highest-risk item and should be started early.
