#!/bin/bash

# Experiment 2: Particle Scaling
# Fixed parameters: dim 8, 50 steps, temp 1.0, sigma 2.0
# Variable: particles 32-1024, resample frequency (3 vs 50), potential type (indicator vs distance)

set -e # Exit on any error

# Create results directory
RESULTS_DIR="results/particle_scaling"
mkdir -p "$RESULTS_DIR"

echo "=== FK Hypercube Experiment 2: Particle Scaling ==="
echo "Results will be saved to: $RESULTS_DIR"
echo

# Common parameters
DIMENSION=8
NUM_STEPS=50
TEMPERATURE=1.0
SIGMA=2.0
PARTICLES_LIST="16,32,64,128,256,512,1024"
SEED=42
NUM_RUNS=10

# Experiment 2a: Indicator potential with FK sampling (resample freq 3)
echo "Running 2a: Indicator potential with FK sampling..."
uv run benchmark.py \
    --dimension $DIMENSION \
    --particles "$PARTICLES_LIST" \
    --num_steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --sigma $SIGMA \
    --resample_freq 3 \
    --potential_type indicator \
    --log_weight 20 \
    --seed $SEED \
    --num_runs $NUM_RUNS \
    --output "$RESULTS_DIR/particles_indicator_fk.csv"

echo "✓ Completed 2a"
echo

# Experiment 2b: Indicator potential with Importance sampling (resample freq 50)
echo "Running 2b: Indicator potential with Importance sampling..."
uv run benchmark.py \
    --dimension $DIMENSION \
    --particles "$PARTICLES_LIST" \
    --num_steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --sigma $SIGMA \
    --resample_freq 50 \
    --potential_type indicator \
    --log_weight 20 \
    --seed $SEED \
    --num_runs $NUM_RUNS \
    --output "$RESULTS_DIR/particles_indicator_importance.csv"

echo "✓ Completed 2b"
echo

# Experiment 2c: Distance potential with FK sampling (resample freq 3)
echo "Running 2c: Distance potential with FK sampling..."
uv run benchmark.py \
    --dimension $DIMENSION \
    --particles "$PARTICLES_LIST" \
    --num_steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --sigma $SIGMA \
    --resample_freq 3 \
    --potential_type distance \
    --weight_scale 20 \
    --exponent 1.0 \
    --seed $SEED \
    --num_runs $NUM_RUNS \
    --output "$RESULTS_DIR/particles_distance_fk.csv"

echo "✓ Completed 2c"
echo

# Experiment 2d: Distance potential with Importance sampling (resample freq 50)
echo "Running 2d: Distance potential with Importance sampling..."
uv run benchmark.py \
    --dimension $DIMENSION \
    --particles "$PARTICLES_LIST" \
    --num_steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --sigma $SIGMA \
    --resample_freq 50 \
    --potential_type distance \
    --weight_scale 20 \
    --exponent 1.0 \
    --seed $SEED \
    --num_runs $NUM_RUNS \
    --output "$RESULTS_DIR/particles_distance_importance.csv"

echo "✓ Completed 2d"
echo

echo "=== Experiment 2 Complete ==="
echo "All results saved to: $RESULTS_DIR"
echo "Files created:"
echo "  - particles_indicator_fk.csv (+ _runs.csv)"
echo "  - particles_indicator_importance.csv (+ _runs.csv)"
echo "  - particles_distance_fk.csv (+ _runs.csv)"
echo "  - particles_distance_importance.csv (+ _runs.csv)"
