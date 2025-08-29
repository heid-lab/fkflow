# FK Steering Hypercube Benchmark

This package implements the FK (Feynman-Kac) steering benchmark on the mode isolation problem described in (TODO: Add paper ref). It provides a complete framework for training Schrödinger Bridge models and running FK steering experiments.


## Setup

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Navigate to the hypercube directory:
   ```bash
   cd hypercube/
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

This will automatically create a virtual environment and install all required dependencies including PyTorch, torchcfm, pandas, numpy, and tqdm.

## Usage

### 1. Train Models

Before running experiments, you need to train the Schrödinger Bridge flow and score models:

```bash
# Train models for dimensions 3-15 (default)
./train_models.sh
```

Or train specific dimensions manually:
```bash
uv run train.py --dimensions "5-10" --base_epochs 4000 --batch_size 1024 --lr 0.01 --sigma 2.0 --seed 42
```

**Training Parameters:**
- `--dimensions`: Range (e.g., "3-15") or list (e.g., "5,7,9") of dimensions
- `--base_epochs`: Base epochs (scaled by dimension, so dim d trains for d×base_epochs)
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--sigma`: Noise level for Schrödinger Bridge
- `--seed`: Random seed for reproducibility

Trained models are saved to `trained_models/sb_models_dim{d}.pt`.

### 2. Run Experiments

#### Quick Start - All Experiments
```bash
# Run all benchmark experiments
./hypercube_exp_run_all.sh
```

This runs three main experiments:
1. **Dimension Scaling**: Performance across dimensions 3-15
2. **Particle Scaling**: Performance vs number of particles  
3. **Inference Curves**: Detailed analysis with multiple particle counts

#### Individual Experiments

**Experiment 1: Dimension Scaling**
```bash
./hypercube_exp1.sh
```
Tests dimensions 3-15 with both indicator and distance potentials, comparing FK steering vs importance sampling.

**Experiment 2: Particle Scaling**  
```bash
./hypercube_exp2.sh
```
Tests particle counts 16-1024 on dimension 8 with different potential types.

**Experiment 3: Inference Curves**
```bash
./hypercube_exp3.sh
```
Generates full performance curves across dimensions with multiple particle sizes.

#### Manual Inference

For custom experiments:

```bash
# Run inference on specific dimensions
uv run inference.py --dimensions "5-8" --num_runs 10 --particles 128 --temperature 1.0

# Benchmark particle counts on specific dimension
uv run benchmark.py --dimension 5 --particles "32,64,128,256" --num_runs 10
```

**Key Parameters:**
- `--dimensions`: Dimensions to test (range or comma-separated)
- `--num_runs`: Number of independent runs per configuration
- `--particles`: Number of particles for FK steering
- `--temperature`: FK steering temperature (higher = more aggressive steering)
- `--sigma`: SDE noise level
- `--resample_freq`: Resampling frequency (lower = more FK-like, higher = more IS-like)
- `--potential_type`: "indicator" or "distance" potential function
- `--num_steps`: Number of SDE integration steps

## Results

All results are saved as CSV files in the `results/` directory:

```
results/
├── dimension_scaling/          # Experiment 1 results
│   ├── dim_indicator_fk.csv
│   ├── dim_indicator_importance.csv
│   ├── dim_distance_fk.csv
│   └── dim_distance_importance.csv
├── particle_scaling/           # Experiment 2 results
│   ├── particles_indicator_fk.csv
│   ├── particles_indicator_importance.csv
│   ├── particles_distance_fk.csv
│   └── particles_distance_importance.csv
└── inference_curves/           # Experiment 3 results
    ├── curve_indicator_32p.csv
    ├── curve_indicator_64p.csv
    └── ...
```

Each CSV contains summary statistics (mean ± std) and individual run data is saved with `_runs.csv` suffix.

## Key Metrics

The benchmark evaluates two main metrics:

1. **Success Rate**: Percentage of samples reaching the target region [0,∞)^d
2. **Wasserstein-2 Distance**: Distance to the target Gaussian distribution (lower is better)

## File Structure

```
hypercube/
├── core.py              # Core FK steering implementation and utilities
├── train.py             # Model training script  
├── inference.py         # Main inference script for dimension scaling
├── benchmark.py         # Particle count benchmarking script
├── utils.py             # Analysis and visualization utilities
├── pyproject.toml       # Package configuration and dependencies
├── README.md           # This documentation
├── train_models.sh     # Training automation script
├── hypercube_exp*.sh   # Experiment automation scripts
├── trained_models/     # Saved model checkpoints (created after training)
└── results/           # Experiment results (created after running experiments)
```

## Development

## Citation

If you use this benchmark in your research, please cite the associated paper:

```bibtex
[Citation details to be added]
```