#!/bin/bash
#SBATCH --job-name=fkflow_paper_saddle
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1000G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-3
#
# Sella + IRC checks of the paper (checks.tsv, from make_manifest.py). Each array
# task takes a whole CPU node and runs SLOTS chunks side by side, two CPUs each.
# A slot claims the next unclaimed chunk whose sampling run has finished (DONE
# from gpu_worker.sh), runs sella_check.py on it and then irc_check.py on the
# saddles that chunk found, and comes back for the next. Chunks of runs that are
# still sampling are skipped and picked up on a later pass, so the checks start
# while the GPU runs are going.
#
# The RDB7 reference TSs are optimised once per surface (run "reference", first
# in the queue); every other check reuses them (reference_from) and waits until
# they have started. Outputs: $OUT/saddle/<surface>/<run>/{sella,irc}/.
#
# Resuming works as for gpu_worker.sh, with the claims under $OUT/claims/checks.
#
#   sbatch scripts/paper/saddle_worker.sh  (from chiflow/)
set -uo pipefail

OUT=${PAPER_OUT:-$PWD/paper}
MANIFEST=scripts/paper/checks.tsv
SLOTS=${SLOTS:-64}
POLL=300  # s between passes while chunks wait for their sampling run
export PYTHONPATH=. OMP_NUM_THREADS=2 SLURM_CPUS_PER_TASK=2  # per slot; sella_check reads the latter

TASK=${SLURM_ARRAY_TASK_ID:-0}
CLAIMS=$OUT/claims/checks
mkdir -p "$CLAIMS" "$OUT/logs"
exec > "$OUT/logs/saddle_task${TASK}_${SLURM_JOB_ID:-local}.out" 2>&1
TOTAL=$(($(wc -l < "$MANIFEST") - 1))
echo "task $TASK on ${SLURM_JOB_NODELIST:-$(hostname)}: $SLOTS slots, $TOTAL chunks in the queue"

sampling_active() {
  # any GPU worker of this campaign still queued or running
  [[ -n $(squeue -u "$USER" -h -n fkflow_paper_gpu -o %i 2>/dev/null) ]]
}

run_chunk() {
  local SURFACE=$1 RUN=$2 DATA_RUN=$3 CHUNK=$4 N_CHUNKS=$5
  local PKL=$OUT/runs/$DATA_RUN/samples_all.pkl DIR=$OUT/saddle/$SURFACE/$RUN
  local REF=$OUT/saddle/$SURFACE/reference
  mkdir -p "$DIR/logs"
  local LOG=$DIR/logs/chunk$(printf '%02d' "$CHUNK").log
  if [[ $RUN == reference ]]; then
    .venv/bin/python sella_check.py run "$DIR/sella" "$CHUNK" "$N_CHUNKS" \
      calculator="$SURFACE" data="$PKL" > "$LOG" 2>&1 &&
      .venv/bin/python irc_check.py run "$DIR/irc" "$CHUNK" "$N_CHUNKS" \
        calculator="$SURFACE" data="$PKL" reference="$DIR/sella" paired=1 >> "$LOG" 2>&1
  else
    .venv/bin/python sella_check.py run "$DIR/sella" "$CHUNK" "$N_CHUNKS" \
      calculator="$SURFACE" reference_from="$REF/sella" "$RUN=$PKL" > "$LOG" 2>&1 &&
      .venv/bin/python irc_check.py run "$DIR/irc" "$CHUNK" "$N_CHUNKS" \
        calculator="$SURFACE" data="$PKL" "$RUN=$DIR/sella" paired=1 >> "$LOG" 2>&1
  fi
}

slot() {
  local S=$1
  while true; do
    local waiting=0 claimed=0
    while IFS=$'\t' read -r IDX SURFACE RUN DATA_RUN CHUNK N_CHUNKS; do
      [[ $IDX == idx ]] && continue
      local CLAIM=$CLAIMS/$(printf '%05d' "$IDX")
      [[ -d $CLAIM ]] && continue
      local RUNDIR=$OUT/runs/$DATA_RUN
      if [[ -e $RUNDIR/FAILED ]]; then
        mkdir "$CLAIM" 2>/dev/null && echo "sampling failed" > "$CLAIM/skipped" && touch "$CLAIM/done"
        continue
      fi
      # ready: the sampling run is done and, for a run, the reference check has started
      if [[ ! -e $RUNDIR/DONE ]] ||
        [[ $RUN != reference && ! -e $OUT/saddle/$SURFACE/reference/sella/settings.json ]]; then
        waiting=1
        continue
      fi
      mkdir "$CLAIM" 2>/dev/null || continue
      claimed=1
      echo "slot $S: $IDX $SURFACE $RUN chunk $CHUNK $(date '+%F %T')"
      if run_chunk "$SURFACE" "$RUN" "$DATA_RUN" "$CHUNK" "$N_CHUNKS" < /dev/null; then
        touch "$CLAIM/done"
      else
        echo "!!! slot $S: $SURFACE $RUN chunk $CHUNK failed" | tee "$CLAIM/failed"
        touch "$CLAIM/done"
      fi
      break  # rescan from the top: earlier chunks may have become ready
    done < "$MANIFEST"
    ((claimed)) && continue
    ((waiting)) || break  # every chunk is claimed
    if ! sampling_active; then
      echo "slot $S: chunks wait for runs that no GPU worker will finish"
      touch "$OUT/logs/saddle_task${TASK}_stalled"
      break
    fi
    sleep "$POLL"
  done
}

for ((s = 0; s < SLOTS; s++)); do
  slot "$s" &
  sleep 2  # stagger the start, so the slots do not all scan the queue at once
done
wait
n_failed=$(find "$CLAIMS" -name failed | wc -l)
echo "task $TASK: done, $n_failed chunk(s) failed overall"
# fail the job when chunks failed or were left waiting for runs that never finished
[[ $n_failed -eq 0 && ! -e $OUT/logs/saddle_task${TASK}_stalled ]]
