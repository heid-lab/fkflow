# Paper runs

Everything behind the transition-state results of the paper: the sampling runs,
the Sella + IRC checks on the steered and unsteered samples, and the inputs of
the tables. All paths are relative to `chiflow/`.

| File | Purpose |
| --- | --- |
| `make_manifest.py` | defines every run and check; writes `runs.tsv` and `checks.tsv` |
| `runs.tsv` | one row per sampling run: idx, study, block, run, seed, n, surfaces (Sella + IRC surfaces, `-` for none), Hydra overrides |
| `checks.tsv` | one row per Sella + IRC chunk: idx, surface, run, data_run, chunk, n_chunks; run `reference` are the RDB7 reference TSs |
| `gpu_worker.sh` | SLURM array that works through `runs.tsv` |
| `saddle_worker.sh` | SLURM array that works through `checks.tsv` as the runs finish |
| `force_worker.sh` | SLURM array: forces of the generated and the reference TSs on g-xTB and AIMNet2-rxn (`force_check.py`) |
| `aggregate.sh`, `aggregate.py` | Sella + IRC summaries of all checks, then the result CSVs and `tables.md` (mean ± std over seeds) |
| `smoke.sh` | one run of each configuration on 4 reactions, and Sella + IRC on 2 |
| `stereo_counts.csv` | per test reaction: tetrahedral stereocentres, reference TS failing the chirality check, atoms |

## Studies

Unless a block varies it, every run uses the 51-point time grid (30 steps on
[0, 0.8], 20 in the steering window [0.8, 1]), noise σ = 0.3 (`sqrt_window`, off
from t = 0.95) and, for FK, the `difference` schedule with resampling after every
step in the window and λ = 100. IS uses the `sum` schedule with one resampling
event after the last step; best of N is the same with σ = 0. Five seeds each.

| Study | Potential | Blocks |
| --- | --- | --- |
| A | chirality | A1 GoFlow / best of N / IS / FK × N = 1–64; A2 schedule × λ; A3 resampling interval, σ, window start; A4 the original manuscript's configuration |
| B | g-xTB force + chirality + E/Z | B1 main arms; B2 composition; B3 λ; B4 resampling interval, window start; B5 N = 8–128 |
| C | AIMNet2-rxn force (ensemble mean + spread) + chirality + E/Z | as B, and C6 without the spread term |

B is checked with Sella + IRC on g-xTB, C on AIMNet2-rxn. GoFlow and FK with
N = 30 (A1) and the N = 30 arms of B1 and C1 (FK, IS, best of N) are checked on
both; the other A runs are not checked.

## Requirements

- The trained GoFlow checkpoint at `artifacts/paper/flow3/checkpoints/flow3-paper.ckpt` (Zenodo).
- The processed data at `data/RDB7/processed_data/` (Zenodo, or `preprocessing.sh`).
- `uv sync --group aimnet` for AIMNet2-rxn and Sella.
- The g-xTB build of xtb 6.7.1, as `GXTB_BIN` or `xtb` on the `PATH`.

## Running

From `chiflow/`:

```bash
python scripts/paper/make_manifest.py     # only after changing the definitions
sbatch scripts/paper/smoke.sh             # minutes; check the log for "smoke: 0 failure(s)"
sbatch scripts/paper/gpu_worker.sh        # 8 GPU workers
sbatch scripts/paper/saddle_worker.sh     # 4 CPU nodes; submit after gpu_worker.sh
sbatch scripts/paper/force_worker.sh      # 2 CPU nodes; after gpu_worker.sh has finished
sbatch scripts/paper/aggregate.sh         # after all workers; writes results/tables.md
```

Results go to `$PAPER_OUT` (default `./paper`): `runs/<run>/` (samples,
`test_results.csv`, logs), `saddle/<surface>/<run>/{sella,irc}/`,
`force/<surface>/` and `results/`. The workers can be stopped and
resubmitted; the comments at the top of `gpu_worker.sh` explain how to release
the runs of a killed worker.

Sampling on a GPU is not bitwise reproducible from run to run (atomic additions
in the message passing, amplified by resampling); repeated runs agree in their
statistics, not in every geometry.
