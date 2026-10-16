#!/bin/bash
#SBATCH --job-name=fkflow_paper_aggregate
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#
# Sella + IRC summaries of every check, then the result CSVs and tables
# (aggregate.py), after gpu_worker.sh, saddle_worker.sh and force_worker.sh have
# finished.
#
#   sbatch scripts/paper/aggregate.sh  (from chiflow/)
set -euo pipefail

export PAPER_OUT=${PAPER_OUT:-$PWD/paper}
export PYTHONPATH=. OMP_NUM_THREADS=1
.venv/bin/python scripts/paper/aggregate.py summarize
.venv/bin/python scripts/paper/aggregate.py collect
