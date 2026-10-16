# ChiFlow: FK-steered flow matching for transition states

Transition-state (TS) generation for chemical reactions from the reactant and
product graphs, with Feynman-Kac (FK) steering of a trained flow-matching model
([GoFlow](https://chemrxiv.org/engage/chemrxiv/article-details/6850098f3ba0887c33dbd713))
at inference time. The steering potentials are:

| Potential | Config (`+fk_potentials=...`) | Steers towards |
| --- | --- | --- |
| chirality | `chirality` (default) | the R/S configuration of the reactant and product stereocentres |
| E/Z | part of `*_ez` | the E/Z configuration of their double bonds |
| g-xTB force | `gxtb_force`, `gxtb_force_chirality`, `gxtb_force_chirality_ez` | small forces on the g-xTB surface (a saddle point) |
| AIMNet2-rxn force | `aimnet_force`, `aimnet_force_chirality`, `aimnet_force_chirality_ez` | small forces on the AIMNet2-rxn surface (ensemble mean, optionally plus the ensemble spread) |

Potentials in one config are summed. Generated TSs can be checked with a Sella
TS optimisation and an IRC on g-xTB or AIMNet2-rxn (`sella_check.py`,
`irc_check.py`).

## Installation

Requires [uv](https://docs.astral.sh/uv/). Python 3.12.6 is pinned in `pyproject.toml`.

```bash
cd chiflow/
uv sync                                    # CUDA 12.8 (Linux or Windows, x86_64)
uv sync --no-default-groups --group cpu    # or: CPU only
uv sync --group aimnet                     # add AIMNet2-rxn and Sella (to either of the above)
```

`torch-scatter`, `torch-sparse`, `torch-cluster` and `torch-spline-conv` are
compiled against a specific torch build and come from the PyG wheel index. For
another CUDA version, change the `pyg-cu128` index URL in `pyproject.toml` to the
matching [PyG index](https://data.pyg.org/whl/) and run `uv lock`.

**g-xTB.** The g-xTB potential and checks call the g-xTB build of xtb 6.7.1
([grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb)). Point `GXTB_BIN` at the
binary or put it on the `PATH` as `xtb`. Every g-xTB calculation is a single
process with one thread, so give a steering job as many CPUs as particles you
want evaluated in parallel.

**AIMNet2-rxn** is downloaded from Hugging Face at a pinned revision on first use
(set `HF_HOME` to choose the cache).

## Data

Training and evaluation use [RDB7](https://zenodo.org/records/13328872)
([Spiekermann et al.](https://www.nature.com/articles/s41597-022-01529-6)),
11,926 gas-phase reactions with reactant, TS and product geometries at
ωB97X-D3/def2-TZVP. The raw files are in `data/RDB7/raw_data/`.

```bash
bash preprocessing.sh    # writes data/RDB7/processed_data/{data.pkl,feat_dict.pkl}
```

Preprocessing keeps every reaction; nothing is filtered. For each reaction it
stores, in atom-map order:

- the reactant, TS and product coordinates (`pos`, one slice each);
- atom features of reactant and product (aromaticity, formal charge,
  hybridisation, H count, valence, degree, ring membership; one-hot);
- the union of the reactant and product bonds as edges, typed by the bond type
  on each side;
- the tetrahedral stereocentres of reactant and product, with their neighbours
  in CIP order and the R/S tag (chirality potential and chirality errors);
- the non-aromatic double bonds of reactant and product, with their
  substituents in CIP order and the E/Z tag (E/Z potential);
- the RDKit molecules and the reaction SMILES.

The paper uses a random 80/10/10 split, `data/RDB7/splits/random_split.pkl`
(1,194 test reactions). `split_preprocessed.py --random` does not seed its
shuffle, so a regenerated split differs; use the published one (Zenodo, below).

## Trained model and published data

Zenodo: **DOI to be added** — the trained GoFlow checkpoint (place it at
`artifacts/paper/flow3/checkpoints/flow3-paper.ckpt`), the processed data and
split, the generated samples and Sella/IRC results of the paper, and the DFT
inputs and outputs of the DFT validation set.

## Usage

Everything runs through `flow_train.py` (Hydra; configs in `configs/`). The paper
model is `experiment=flow3`.

**Training**

```bash
uv run python flow_train.py experiment=flow3 train=true test=true
```

**Inference, GoFlow without steering** (the sample closest to the median of 30 samples):

```bash
CKPT=artifacts/paper/flow3/checkpoints/flow3-paper.ckpt
uv run python flow_train.py experiment=flow3 train=false test=true \
  ckpt_path=$CKPT custom_model_weight_path=$CKPT \
  model.num_steps=51 +model.window_num_steps=20 +model.steering_t_start=0.8 \
  model.num_samples=30 +model.inference_sampling_method=median
```

**Inference, FK steering** (the paper's recipe; here with the g-xTB force,
chirality and E/Z potentials):

```bash
uv run python flow_train.py experiment=flow3 train=false test=true \
  ckpt_path=$CKPT custom_model_weight_path=$CKPT \
  model.num_steps=51 +model.window_num_steps=20 \
  +model.steering_t_start=0.8 +model.steering_t_end=0.95 \
  +model.noise_schedule=sqrt_window +model.noise_t_start=0.8 +model.noise_t_end=0.95 \
  +model.steering_base_variance=0.3 \
  model.num_samples=30 +model.inference_sampling_method=fk_steering \
  +model.fk_potential_scheduler=difference +model.resample_freq=1 \
  +model.fk_steering_temperature=100 \
  +fk_potentials=gxtb_force_chirality_ez
```

Results go to `logs/<task_name>/runs/<time>/` (`test_results.csv`: RMSE, RMSE
with reflection, distance MAE, chirality errors, ESS per reaction) and the
generated geometries to `samples_all.pkl` (set
`+callbacks.test_and_save_after_training.predictions_path=<dir>`).

**Steering parameters** (full descriptions in the `FlowModule` docstring,
`flow_matching/flow_module.py`):

| Parameter | Meaning |
| --- | --- |
| `num_samples` | number of particles (GoFlow: samples; the one closest to their median is kept) |
| `fk_steering_temperature` | λ, strength of the tilt p(x) exp(−λ U(x)) |
| `fk_potential_scheduler` | FK potential schedule: `difference`, `sum`, `harmonic_sum`, `max` |
| `resample_freq` | resample after every k-th step in the steering window (1000: only after the last step, i.e. importance sampling) |
| `steering_t_start`, `steering_t_end` | steering window: resampling events only in [t_start, t_end], plus after the last step |
| `steering_base_variance` | noise level σ of the SDE (a standard deviation); 0 gives the ODE |
| `noise_schedule`, `noise_t_start`, `noise_t_end` | shape and window of σ(t) |
| `window_num_steps` | steps placed in [steering_t_start, 1]; the rest go to [0, steering_t_start] |

**Sella and IRC checks**

```bash
PYTHONPATH=. uv run python sella_check.py run <out> 0 1 calculator=gxtb data=<samples.pkl>       # reference TSs
PYTHONPATH=. uv run python sella_check.py run <out2> 0 1 calculator=gxtb reference_from=<out> fk=<samples.pkl>
PYTHONPATH=. uv run python irc_check.py run <irc> 0 1 calculator=gxtb data=<samples.pkl> fk=<out2>
PYTHONPATH=. uv run python sella_check.py summarize <out2>
PYTHONPATH=. uv run python irc_check.py summarize <irc> scripts/paper/stereo_counts.csv
```

The module docstrings explain the criteria: first-order saddle, same saddle as
the reference TS, and IRC ends matching the RDB7 reactant and product graphs
with the right stereo.

## Reproducing the paper

All run settings of the paper are defined in one place,
[`scripts/paper/make_manifest.py`](scripts/paper/make_manifest.py). The SLURM
arrays in `scripts/paper/` run the sampling, the Sella + IRC checks and the force
evaluations, and an aggregation job writes the tables; see
[`scripts/paper/README.md`](scripts/paper/README.md).

## Project structure

```
chiflow/
├── configs/             Hydra configs (model, data, FK potentials, experiments)
├── flow_matching/       FlowModule (flow matching, FK steering) and the potentials
├── gotennet/            GotenNet representation, data module, callbacks
├── utils/               dataset construction, chirality and E/Z extraction
├── data_processing/     reaction-core classification (alternative splits)
├── scripts/paper/       the paper's runs and checks
├── flow_train.py        training and inference entry point
├── preprocessing.py     RDB7 -> data.pkl
├── split_preprocessed.py  train/val/test splits (random, reaction core, barrier height)
├── sella_check.py       Sella TS optimisation of generated TSs
├── irc_check.py         IRC from the Sella saddles, graph and stereo comparison
├── calculators.py       ASE calculators for AIMNet2-rxn and g-xTB
├── force_check.py       forces of generated or reference TSs on g-xTB or AIMNet2-rxn
└── eval_aimnet_force.py AIMNet2-rxn forces of generated TSs (GPU)
```
