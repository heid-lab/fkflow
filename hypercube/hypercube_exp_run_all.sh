#!/bin/bash

# Master script to run all FK Hypercube experiments
# This runs both dimension scaling and particle scaling experiments

set -e # Exit on any error

echo "=== FK Hypercube Experiments - Master Script ==="
echo "This will run all experiments and save results to results/"
echo

mkdir -p results/

# Get start time
START_TIME=$(date)
echo "=============================================="
echo

# Run Experiment 1: Dimension Scaling
echo "Starting Experiment 1: Dimension Scaling..."
./hypercube_exp1.sh
echo "Experiment 1 completed"
echo

# Run Experiment 2: Particle Scaling
echo "Starting Experiment 2: Particle Scaling..."
./hypercube_exp2.sh
echo "Experiment 2 completed"
echo

# Run Experiment 3: Inference Curves
echo "Starting Experiment 3: Inference Curves..."
./hypercube_exp3.sh
echo "Experiment 3 completed"
echo

# Summary
echo "Started at: $START_TIME"
echo "Finished at: $END_TIME"
