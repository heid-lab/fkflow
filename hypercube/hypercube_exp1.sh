#!/bin/bash

# Experiment 1: Dimension Scaling (5-15)
# Fixed parameters: 50 steps, temp 1.0, sigma 2.0, 32 particles
# Variable: dimensions 3-15, resample frequency (3 vs 50) i.e. FK or IS, potential type (indicator vs distance)

set -e # Exit on any error

# Create results directory
RESULTS_DIR="results/dimension_scaling"
mkdir -p "$RESULTS_DIR"

echo "=== FK Hypercube Experiment 1: Dimension Scaling ==="
echo "Results will be saved to: $RESULTS_DIR"
echo

# Common parameters
DIMENSIONS="3-15"
NUM_STEPS=50
TEMPERATURE=1.0
SIGMA=2.0
PARTICLES=32
SEED=42
NUM_RUNS=10

# Experiment 1a: Indicator potential with FK sampling (resample freq 3)
echo "Running 1a: Indicator potential with FK sampling..."
uv run inference.py \
    --dimensions "$DIMENSIONS" \
    --num_steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --sigma $SIGMA \
    --particles $PARTICLES \
    --resample_freq 3 \
    --potential_type indicator \
    --log_weight 20 \
    --seed $SEED \
    --num_runs $NUM_RUNS \
    --output "$RESULTS_DIR/dim_indicator_fk.csv"

echo "✓ Completed 1a"
echo

# Experiment 1b: Indicator potential with Importance sampling (resample freq 50)
echo "Running 1b: Indicator potential with Importance sampling..."
uv run inference.py \
    --dimensions "$DIMENSIONS" \
    --num_steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --sigma $SIGMA \
    --particles $PARTICLES \
    --resample_freq 50 \
    --potential_type indicator \
    --log_weight 20 \
    --seed $SEED \
    --num_runs $NUM_RUNS \
    --output "$RESULTS_DIR/dim_indicator_importance.csv"

echo "✓ Completed 1b"
echo

# Experiment 1c: Distance potential with FK sampling (resample freq 3)
echo "Running 1c: Distance potential with FK sampling..."
uv run inference.py \
    --dimensions "$DIMENSIONS" \
    --num_steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --sigma $SIGMA \
    --particles $PARTICLES \
    --resample_freq 3 \
    --potential_type distance \
    --weight_scale 20 \
    --exponent 1.0 \
    --seed $SEED \
    --num_runs $NUM_RUNS \
    --output "$RESULTS_DIR/dim_distance_fk.csv"

echo "✓ Completed 1c"
echo

# Experiment 1d: Distance potential with Importance sampling (resample freq 50)
echo "Running 1d: Distance potential with Importance sampling..."
uv run inference.py \
    --dimensions "$DIMENSIONS" \
    --num_steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --sigma $SIGMA \
    --particles $PARTICLES \
    --resample_freq 50 \
    --potential_type distance \
    --weight_scale 20 \
    --exponent 1.0 \
    --seed $SEED \
    --num_runs $NUM_RUNS \
    --output "$RESULTS_DIR/dim_distance_importance.csv"

echo "✓ Completed 1d"
echo

echo "=== Experiment 1 Complete ==="
echo "All results saved to: $RESULTS_DIR"
echo "Files created:"
echo "  - dim_indicator_fk.csv (+ _runs.csv)"
echo "  - dim_indicator_importance.csv (+ _runs.csv)"
echo "  - dim_distance_fk.csv (+ _runs.csv)"
echo "  - dim_distance_importance.csv (+ _runs.csv)"
