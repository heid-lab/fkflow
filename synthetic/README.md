# Synthetic Guided Flow Experiments Setup

This directory contains synthetic 2D distribution experiments adapted from the paper ["On the Guidance of Flow Matching"](https://arxiv.org/abs/2502.02150) (ICML 2025).

## Overview

The synthetic experiments demonstrate guided flow matching on 2D distributions, comparing various guidance methods including FK steering. The setup follows the original paper's implementation with additional FK steering capabilities integrated into the notebook for comparison.

## Setup and Installation

### Prerequisites

- Python 3.11
- conda (recommended)

### Installation

**Follow the original paper's setup exactly:**

1. Navigate to the synthetic directory:
   ```bash
   cd synthetic/
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate guided_flow
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Running Experiments

### 1. Train Base Models

First, train the base flow matching models:

```bash
bash script/train_cfm.sh
```

For optimal transport variant:
```bash
bash script/train_cfm_ot.sh
```


### 2. Run Notebook Experiments

The main experiments are conducted through Jupyter notebooks:

#### Available Notebooks

1. **`fig.ipynb`** - Main figures from the paper
   - Demonstrates various guidance methods on 2D distributions
   - Includes FK steering alongside other guidance approaches

2. **`vals.ipynb`** - Full benchmarks 
   - Benchmarks over multiple samples


## FK Steering Integration

### Key Addition: `fk_steering.py`

This module provides FK steering functionality adapted for the synthetic experiments:

- **Pure deterministic flow** (no SDE terms, unlike hypercube implementation)
- **Multiple resampling strategies** (multinomial, stratified, systematic, residual)
- **Event-based potential scheduling** (sum, harmonic_sum, difference)
- **Notebook compatibility** with existing guided flow interface

### Usage in Notebooks

The FK steering can be used within the notebooks alongside other guidance methods:

```python
# Example usage in notebook cells
from fk_steering import evaluate_fk, create_fk_guide_cfg

# Configure FK steering parameters
fk_params = {
    'num_samples': 8,
    'fk_steering_temperature': 1.0,
    'fk_potential_scheduler': 'sum',
    'resample_method': 'stratified',
    'resample_freq': 2
}

# Generate trajectory with FK steering
trajectory = evaluate_fk(x0_sampler, x1_sampler, model, x1_dist, fk_params, ode_cfg)
```


## File Structure

```
synthetic/
├── guided_flow/           # Core guided flow implementation (from original paper)
├── notebooks/            
│   ├── fig.ipynb         # Main paper figures (adapted from original fig.ipynb)
│   ├── vals.ipynb        # Validation experiments (adapted from original fig.ipynb)
│   └── fk_steering.py    # FK steering implementation 
├── script/               # Training scripts
│   ├── train_cfm.sh      # Train base CFM models
│   └── train_cfm_ot.sh   # Train OT-CFM models
└── environment.yml       # Conda environment specification
```



## Citation

Original paper:
```bibtex
@inproceedings{
  feng2025on,
  title={On the Guidance of Flow Matching},
  author={Feng, Ruiqi and Yu, Chenglei and Deng, Wenhao and Hu, Peiyan and Wu, Tailin},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=pKaNgFzJBy}
}
```

FK steering implementation adapted for this experimental framework.

