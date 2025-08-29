# ChiFlow: Flow Matching with FK Steering for Chemical Reactions

Flow Matching with Feynman-Kac Steering for generating chemical reaction transition states with correct chirality.

## Setup

### Prerequisites
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   or via `pip`
   ```bash
   pip install uv
   ```

2. **Clone and navigate to the chiflow directory**:
   ```bash
   cd chiflow/
   ```

3. **Create environment and install dependencies**:
   ```bash
   uv sync -n
   uv add torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.6.0+cu124.html --no-build-isolation -n
   ```


## Dataset and Preprocessing

ChiFlow is trained and evaluated on the open-source [RDB7 database](https://zenodo.org/records/13328872) by [Spiekermann et al.](https://www.nature.com/articles/s41597-022-01529-6). 

### Prerequisites for Preprocessing

Ensure you have the raw RDB7 data files:
- Raw `.csv` and `.xyz` files should be located in `data/RDB7/raw_data/` directory

### Preprocessing Steps

Not yet supported. Currently only works with the preprocessed files as they are in the Repository.

## Usage

### Training a Model

Run training with a specific experiment configuration:
```bash
uv run python flow_train.py test=false
```

Available model sizes:
- `flow1` - Large model
- `flow2` - Medium model  
- `flow3` - Small model

### Paper Reproduction: Key Experimental Runs

To reproduce the main results from the paper, run these specific configurations:

#### 1. Flow3 with Median Sampling (Baseline)
```bash
uv run python flow_train.py \
  model.num_steps=50 \
  model.num_samples=50 \
  +model.inference_sampling_method=median \
  train=false test=true \
  ckpt_path=path/to/trained/model.ckpt
```

#### 2. Flow3 with FK Steering (Main Method)
```bash
uv run python flow_train.py  \
  model.num_steps=50 \
  model.num_samples=50 \
  +model.inference_sampling_method=fk_steering \
  +model.steering_base_variance=0.3 \
  +model.fk_steering_temperature=0.4 \
  +model.resample_freq=10 \
  train=false test=true \
  ckpt_path=path/to/trained/model.ckpt
```

### Configuration Details

The main configuration is in `configs/train.yaml`. Key settings:

- **Data**: Set to `rdb7`
- **Logger**: Choose `wandb` for Weights & Biases logging or `null` for no logging
- **Trainer**: Choose `gpu` or `cpu`
- **Training/Testing**: Set `train: true` and `test: true` to both train and test

In `configs/model/flow.yaml`:
- Set `n_atom_rdkit_feats`: 24 for rdb7

### FK Steering Parameters

Key parameters for Feynman-Kac steering:
- `steering_base_variance`: Controls the base diffusion variance (paper uses 0.3)
- `fk_steering_temperature`: Temperature for potential-based reweighting (paper uses 0.4)  
- `resample_freq`: Frequency of particle resampling (paper uses 10)
- `fk_potential_fns`: List of potential functions (default: chiral_potential)

### Data Requirements

Before running, ensure your data directory contains:
- `train.csv` and `train.xyz`
- `val.csv` and `val.xyz` 
- `test.csv` and `test.xyz`

### Outputs

All model outputs, logs, and results will be saved within the `chiflow/` directory structure to keep everything contained.

## Project Structure

```
chiflow/
├── configs/          # Hydra configuration files
├── data_processing/  # Data preprocessing utilities
├── flow_matching/    # Core flow matching implementation
├── gotennet/        # GotenNet model architecture
├── utils/           # Utility functions
├── flow_train.py    # Main training script
└── pyproject.toml   # Project dependencies
```
