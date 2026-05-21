# Stage 0 Scoping Benchmarks

Scripts behind the Stage 0 configuration decision (Phase 2 of the Computational
Optimization project): rendering resolution, MLP size, and iteration budget.
The decision and the measured results are written up in Section 6.1 of
`../nerf_tutorial2.ipynb`.

- `bench_800.py`  — resolution sweep (100 / 200 / 400 / 800) and the batch /
  iteration scale-up.
- `bench_mlp.py`  — MLP-capacity test, first attempt (plain 8-layer MLP; this
  configuration failed to train, density collapsed to zero).
- `bench_mlp2.py` — MLP-capacity test, corrected (large MLP with the standard
  NeRF skip connection vs the small 128x4 MLP).
- `bench_sep.py`  — SGD-vs-Adam separability check at 200x200 and 400x400.

Each script uses absolute dataset paths and was run from the project venv
(`.venv_co`). The companion notebook `../resolution_benchmark.ipynb` holds the
resolution-sweep exploration in notebook form.
