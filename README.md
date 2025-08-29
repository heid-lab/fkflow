# Feynman-Kac Flow Matching


This repository contains the research code for **Feynman-Kac Flow Matching**, a novel approach for steering flow-based generative models towards desired regions using Feynman-Kac formalism. Our method provides an effective alternative to importance sampling for guided generation in flow matching models.

## Overview

Feynman-Kac Flow Matching enables controlled generation by incorporating potential functions during the sampling process, allowing models to generate samples that satisfy specific constraints or preferences. The repository demonstrates this approach across three domains:

- **Synthetic 2D distributions** - Theoretical validation on toy problems
- **Hypercube geometry** - Scalable benchmark for mode isolation 
- **Chemical reaction modeling** - Real-world application to molecular conformations

The code is organized into three main components:

## Repository Structure

### 1. `hypercube/` - FK Steering Hypercube Benchmark

Complete implementation of the FK steering benchmark on hypercube geometry problems.

**Setup:**
```bash
cd hypercube/
uv sync                    # Install dependencies
./train_models.sh          # Train models (dimensions 3-15)
./hypercube_exp_run_all.sh # Run all experiments
```

**Features:**
- Schrödinger Bridge model training
- FK steering vs importance sampling comparison
- Dimension scaling analysis (3-15D)

See [`hypercube/README.md`](hypercube/README.md) for detailed documentation.

### 2. `synthetic/` - Guided Flow Experiments on 2D Distributions 

Implementation of FK steering on synthetic 2D distributions, adapted from ["On the Guidance of Flow Matching"](https://arxiv.org/abs/2502.02150).

**Features:**
- 2D toy distribution experiments (moons, spirals, circles, etc.)
- FK steering comparison with guidance
- Pure deterministic flow implementation

**Setup:**
```bash
cd synthetic/
conda env create -f environment.yml
conda activate guided_flow
pip install -e .
bash script/train_cfm.sh    # Train base models
```

See [`synthetic/README.md`](synthetic/README.md) for detailed setup and notebook usage.

### 3. `chiflow/` - Chiral Flow Matching for Chemical Reactions

Flow matching with FK steering for generating chemical reaction transition states with correct chirality, trained on the RDB7 dataset.

**Key Features:**
- Chemical reaction transition state generation
- Chirality-aware flow matching with specialized potential functions
- RDB7 dataset preprocessing pipeline

**Setup:**
```bash
cd chiflow/
uv sync
bash preprocess_extract_rxn_core.sh    # Extract reaction cores
bash preprocess_create_splits.sh       # Create dataset splits  
bash preprocessing.sh                   # Process RDB7 data
uv run python flow_train.py +experiment=flow3  # Train model
```

**Paper Reproduction:**
```bash
# FK Steering (main method)
uv run python flow_train.py +experiment=flow3 \
  model.inference_sampling_method=fk_steering \
  model.steering_base_variance=0.3 \
  model.fk_steering_temperature=0.4 \
  train=false test=true ckpt_path=path/to/model.ckpt
```

See [`chiflow/README.md`](chiflow/README.md) for detailed documentation.

## Quick Start

We recommend starting with the **hypercube benchmark** as it provides the most complete and reproducible implementation:

```bash
cd hypercube/
uv sync                      # Install dependencies
./train_models.sh            # Train Schrödinger Bridge models (dimensions 3-15)
./hypercube_exp_run_all.sh   # Run all benchmark experiments
```

This will:
1. Train flow and score models for dimensions 3-15 
2. Run dimension scaling experiments comparing FK steering vs importance sampling
3. Generate particle count optimization results
4. Save CSV results to `hypercube/results/` 

### Alternative Entry Points

- **Synthetic 2D experiments**: Interactive Jupyter notebooks for visualization
  ```bash
  cd synthetic/ && conda env create -f environment.yml
  ```

- **Chemical reaction modeling**: Requires RDB7 dataset setup
  ```bash
  cd chiflow/ && uv sync
  ```

## Core Method

Feynman-Kac Flow Matching extends standard flow matching by incorporating potential functions during generation:

1. **Potential Functions**: Define regions of interest or constraints (e.g., chirality, target modes)
2. **FK Reweighting**: Use Feynman-Kac formalism to bias sampling towards high-potential regions
3. **Particle Resampling**: Periodically resample particles based on accumulated weights


## Requirements

### System Requirements
- Python 3.8+ (Python 3.11 recommended for synthetic experiments)
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU recommended for training (particularly for ChiFlow)
- 16GB+ RAM for chemical experiments
- conda (for synthetic experiments only)

### Key Dependencies
- **Core**: PyTorch, torchdiffeq, torchcfm
- **Chemical**: RDKit, PyTorch Geometric, Lightning
- **ML/Optimization**: Hydra, Weights & Biases (optional)
- **Analysis**: NumPy, pandas, matplotlib

## Publication

This code accompanies the paper:

```bibtex
[Citation details to be added]
```




## License

[License information to be added]

## Acknowledgments

- **GoFlow** [Galustian et al.](https://chemrxiv.org/engage/chemrxiv/article-details/6850098f3ba0887c33dbd713)
- **RDB7 Dataset**: [Spiekermann et al.](https://www.nature.com/articles/s41597-022-01529-6)
- **Guided Flow Framework**: ["On the Guidance of Flow Matching"](https://arxiv.org/abs/2502.02150)
- **Flow Matching**: [torchcfm library](https://github.com/atong01/conditional-flow-matching) and associated papers
