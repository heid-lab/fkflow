#!/bin/bash

# Training script for FK Hypercube models (dimensions 3-15)
# This trains Schrödinger Bridge flow and score models needed for the experiments

set -e # Exit on any error

echo "=== FK Hypercube Model Training ==="
echo "Training models for dimensions 3-15..."
echo "Models will be saved to: trained_models/"
echo

# Create trained_models directory
mkdir -p trained_models

# Training parameters
DIMENSIONS="3-15"
BASE_EPOCHS=1000 # Scaled by dim, so in dim d will train for d*BASE_EPOCHS
BATCH_SIZE=1024
LEARNING_RATE=0.01
SIGMA=2.0
SEED=42

echo "Training parameters:"
echo "  Dimensions: $DIMENSIONS"
echo "  Base epochs: $BASE_EPOCHS (scaled by dimension)"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Sigma: $SIGMA"
echo "  Seed: $SEED"
echo "  Device: auto-detected"
echo

# Get start time
START_TIME=$(date)
echo "Started at: $START_TIME"
echo "=============================================="
echo

# Run training
uv run train.py \
    --dimensions "$DIMENSIONS" \
    --base_epochs $BASE_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --sigma $SIGMA \
    --seed $SEED \
    --device auto

# Get end time
END_TIME=$(date)
echo
echo "=============================================="
echo "Training completed!"
echo "Started:  $START_TIME"
echo "Finished: $END_TIME"
echo

# List trained models
echo "Trained models:"
if [ -d "trained_models" ]; then
    ls -la trained_models/sb_models_dim*.pt | wc -l | xargs echo "  Total models:"
    echo "  Files:"
    for model_file in trained_models/sb_models_dim*.pt; do
        if [ -f "$model_file" ]; then
            basename "$model_file" | sed 's/^/    /'
        fi
    done
else
    echo "  No trained_models directory found"
fi

echo
echo "Ready to run experiments with ./run_all_experiments.sh"
