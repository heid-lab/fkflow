#!/bin/bash
#SBATCH --job-name=fkflow_paper_force
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#
# Forces on the generated TSs of every run of the paper (force_check.py), on
# g-xTB and on AIMNet2-rxn, and on the RDB7 reference TSs (run "reference").
# Each array task takes a whole CPU node and runs SLOTS (surface, run) pairs side
# by side, CPUS each, claimed from one queue as in gpu_worker.sh (claims under
# $OUT/claims/force). Needs the finished runs of gpu_worker.sh.
# Outputs: $OUT/force/<surface>/<run>.tsv.
#
#   sbatch scripts/paper/force_worker.sh  (from chiflow/)
set -uo pipefail

OUT=${PAPER_OUT:-$PWD/paper}
MANIFEST=scripts/paper/runs.tsv
SLOTS=${SLOTS:-8}
CPUS=$((128 / SLOTS))
export PYTHONPATH=. OMP_NUM_THREADS=$CPUS SLURM_CPUS_PER_TASK=$CPUS  # per slot

TASK=${SLURM_ARRAY_TASK_ID:-0}
CLAIMS=$OUT/claims/force
mkdir -p "$CLAIMS" "$OUT/logs" "$OUT/force/gxtb" "$OUT/force/aimnet"
exec > "$OUT/logs/force_task${TASK}_${SLURM_JOB_ID:-local}.out" 2>&1
echo "task $TASK on ${SLURM_JOB_NODELIST:-$(hostname)}: $SLOTS slots of $CPUS CPUs"

# the queue: per surface the reference TSs (from any run's pickle), then every run
units() {
  for surface in gxtb aimnet; do
    echo "$surface reference"
    tail -n +2 "$MANIFEST" | cut -f4 | sed "s/^/$surface /"
  done
}

slot() {
  local S=$1
  while read -r SURFACE RUN; do
    mkdir "$CLAIMS/${SURFACE}_$RUN" 2>/dev/null || continue
    local TSV=$OUT/force/$SURFACE/$RUN.tsv
    if [[ $RUN == reference ]]; then
      local ARGS=(reference=1 "$OUT/runs/goflow_N30_s1/samples_all.pkl")
    else
      local ARGS=("$OUT/runs/$RUN/samples_all.pkl")
    fi
    echo "slot $S: $SURFACE $RUN $(date '+%F %T')"
    if .venv/bin/python force_check.py "$TSV" calculator="$SURFACE" "${ARGS[@]}" < /dev/null; then
      touch "$CLAIMS/${SURFACE}_$RUN/done"
    else
      echo "!!! slot $S: $SURFACE $RUN failed" | tee "$CLAIMS/${SURFACE}_$RUN/failed"
    fi
  done < <(units)
}

for ((s = 0; s < SLOTS; s++)); do
  slot "$s" &
  sleep 2
done
wait
n_failed=$(find "$CLAIMS" -name failed | wc -l)
echo "task $TASK: done, $n_failed unit(s) failed overall"
[[ $n_failed -eq 0 ]]
