#!/bin/bash

# Experiment 3: Inference Curves
# Full inference curves for dimensions 3-15 with particle sizes 32, 64, 128, 256
# Tests both indicator and distance potentials with FK sampling (resample freq 3)

set -e # Exit on any error

# Create results directory
RESULTS_DIR="results/inference_curves"
mkdir -p "$RESULTS_DIR"

echo "=== FK Hypercube Experiment 3: Inference Curves ==="
echo "Results will be saved to: $RESULTS_DIR"
echo

# Common parameters
DIMENSIONS="3-15"
NUM_STEPS=50
TEMPERATURE=1.0
SIGMA=2.0
RESAMPLE_FREQ=3 # FK sampling only
SEED=42
NUM_RUNS=10
PARTICLE_SIZES=(32 64 128 256)

echo "Parameters:"
echo "  Dimensions: $DIMENSIONS"
echo "  Particle sizes: ${PARTICLE_SIZES[*]}"
echo "  Steps: $NUM_STEPS"
echo "  Temperature: $TEMPERATURE"
echo "  Sigma: $SIGMA"
echo "  Resample frequency: $RESAMPLE_FREQ (FK sampling)"
echo "  Seed: $SEED"
echo "  Runs per configuration: $NUM_RUNS"
echo

echo "=============================================="
echo

# Loop through particle sizes
for PARTICLES in "${PARTICLE_SIZES[@]}"; do
    echo "Testing $PARTICLES particles..."

    # Experiment 3a: Indicator potential with various particle sizes
    echo "  Running 3a-$PARTICLES: Indicator potential, $PARTICLES particles..."
    uv run inference.py \
        --dimensions "$DIMENSIONS" \
        --num_steps $NUM_STEPS \
        --temperature $TEMPERATURE \
        --sigma $SIGMA \
        --particles $PARTICLES \
        --resample_freq $RESAMPLE_FREQ \
        --potential_type indicator \
        --log_weight 20 \
        --seed $SEED \
        --num_runs $NUM_RUNS \
        --output "$RESULTS_DIR/curve_indicator_${PARTICLES}p.csv"

    echo "  ✓ Completed indicator with $PARTICLES particles"

    # Experiment 3b: Distance potential with various particle sizes
    echo "  Running 3b-$PARTICLES: Distance potential, $PARTICLES particles..."
    uv run inference.py \
        --dimensions "$DIMENSIONS" \
        --num_steps $NUM_STEPS \
        --temperature $TEMPERATURE \
        --sigma $SIGMA \
        --particles $PARTICLES \
        --resample_freq $RESAMPLE_FREQ \
        --potential_type distance \
        --weight_scale 20 \
        --exponent 1.0 \
        --seed $SEED \
        --num_runs $NUM_RUNS \
        --output "$RESULTS_DIR/curve_distance_${PARTICLES}p.csv"

    echo "  ✓ Completed distance with $PARTICLES particles"
    echo
done

echo "=============================================="
echo "Results saved to: $RESULTS_DIR"
echo "Files created:"

# List all generated files
for PARTICLES in "${PARTICLE_SIZES[@]}"; do
    echo "  ${PARTICLES} particles:"
    echo "    - curve_indicator_${PARTICLES}p.csv (+ _runs.csv)"
    echo "    - curve_distance_${PARTICLES}p.csv (+ _runs.csv)"
done
