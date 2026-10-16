#!/bin/bash
#SBATCH --job-name=fkflow_paper_gpu
#SBATCH --partition=GPU-l40s,GPU-a40,GPU-a100s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-7
#
# Sampling runs of the paper (runs.tsv, from make_manifest.py), pulled from one
# shared queue by the array tasks. Each task claims the lowest unclaimed row,
# runs it and comes back for the next, until the queue is empty. A claim is a
# mkdir, which is atomic, so no two tasks run the same row; the queue order of
# runs.tsv (main tables first, whole seeds) is therefore also the finishing order.
#
# A run writes samples_all.pkl, test_results.csv and the Hydra logs to
# $OUT/runs/<run>/, then DONE (or FAILED with the exit code). saddle_worker.sh
# starts the Sella + IRC checks of a run when its DONE appears.
#
# Resuming: a claim without a done marker belongs to a task that was killed
# (wall time, node loss). With no task running, delete those claims and submit again:
#   find $OUT/claims/runs -mindepth 1 -maxdepth 1 -type d '!' -exec test -e {}/done \; -exec rm -r {} +
#
# The 16 CPUs are for the g-xTB potential (one xtb process per molecule); any GPU
# with sm_80 or newer works (the AIMNet neighbour list is built for sm_80).
#
#   sbatch scripts/paper/gpu_worker.sh  (from chiflow/)
set -uo pipefail

OUT=${PAPER_OUT:-$PWD/paper}
MANIFEST=scripts/paper/runs.tsv
CKPT=artifacts/paper/flow3/checkpoints/flow3-paper.ckpt

TASK=${SLURM_ARRAY_TASK_ID:-0}
CLAIMS=$OUT/claims/runs
mkdir -p "$CLAIMS" "$OUT/logs"
exec > "$OUT/logs/gpu_task${TASK}_${SLURM_JOB_ID:-local}.out" 2>&1
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST"; exit 1; }
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT"; exit 1; }
echo "task $TASK on ${SLURM_JOB_NODELIST:-$(hostname)}, GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

n_failed=0
while IFS=$'\t' read -r IDX STUDY BLOCK RUN SEED N SURFACES OVERRIDES; do
  [[ $IDX == idx ]] && continue
  CLAIM=$CLAIMS/$(printf '%05d' "$IDX")
  mkdir "$CLAIM" 2>/dev/null || continue
  DIR=$OUT/runs/$RUN
  mkdir -p "$DIR"
  rm -f "$DIR/DONE" "$DIR/FAILED"
  echo "=== $IDX $RUN ($STUDY $BLOCK) $(date '+%F %T')"
  # shellcheck disable=SC2086  # OVERRIDES is a list of Hydra overrides
  .venv/bin/python flow_train.py experiment=flow3 $OVERRIDES \
    callbacks.test_and_save_after_training.compute_pymatgen_heuristic_rmsd=true \
    +callbacks.test_and_save_after_training.predictions_path="$DIR" \
    callbacks.test_and_save_after_training.runs_stats_path="$DIR" hydra.run.dir="$DIR/hydra" \
    data.num_workers=0 train=false logger=csv task_name="$RUN" \
    ckpt_path="$CKPT" custom_model_weight_path="$CKPT" < /dev/null > "$DIR/run.log" 2>&1
  rc=$?
  # the run must have used its overrides: particles and grid as in the manifest
  STATS=$(tail -1 "$DIR/stats.csv" 2>/dev/null | awk -F, '{print $(NF-1)+0, $NF+0}')
  if [[ $rc -eq 0 && $STATS != "$(grep -oP 'model.num_steps=\K[0-9]+' <<< "$OVERRIDES" || echo 50) $N" ]]; then
    echo "!!! $RUN: stats.csv has num_steps num_samples = $STATS, not as in $MANIFEST"
    rc=99
  fi
  if [[ $rc -eq 0 && -f $DIR/samples_all.pkl ]]; then
    cp "$DIR"/hydra/test_results.csv "$DIR"/ 2>/dev/null
    touch "$DIR/DONE"
    echo "--- $RUN ok $(date '+%F %T')"
  else
    echo "$rc" > "$DIR/FAILED"
    n_failed=$((n_failed + 1))
    echo "!!! $RUN failed (rc=$rc), see $DIR/run.log"
  fi
  touch "$CLAIM/done"
done < "$MANIFEST"
echo "task $TASK: queue empty, $n_failed run(s) failed"
[[ $n_failed -eq 0 ]]
