#!/bin/bash
#SBATCH --job-name=fkflow_paper_smoke
#SBATCH --partition=GPU-l40s,GPU-a40,GPU-a100s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Smoke test before the campaign: one run of every distinct configuration in
# runs.tsv (seed 1) on a single batch of 4 reactions, then Sella + IRC on two
# reactions per surface. Catches config and override errors in minutes.
# Outputs go to $OUT/smoke/, apart from the campaign.
#
#   sbatch scripts/paper/smoke.sh  (from chiflow/)
set -uo pipefail

OUT=${PAPER_OUT:-$PWD/paper}/smoke
MANIFEST=scripts/paper/runs.tsv
CKPT=artifacts/paper/flow3/checkpoints/flow3-paper.ckpt
mkdir -p "$OUT"

# one of each: methods, potentials, schedules, window and grid variants, the largest N
RUNS=(goflow_N30 chir_fk_N30 chir_is_N30 chir_best_N30 gx_fk_N30 gx_is_N30 gx_best_N30
  aim_fk_N30 aim_is_N30 aim_best_N30 chir_fk_N64 gx_fk_N128 aim_is_N128
  chir_fk_sum_lam1 chir_fk_harmonic_sum_lam1 chir_fk_max_lam1 chir_fk_freq5 chir_fk_sigma0.0
  chir_fk_t05 chir_fk_t00 goflow_published chir_fk_published
  gx_fk_force gx_fk_force_chir aim_fk_force aim_fk_force_chir gx_fk_t07 aim_fk_u0)

n_failed=0
for name in "${RUNS[@]}"; do
  ROW=$(awk -F'\t' -v r="${name}_s1" '$4 == r' "$MANIFEST")
  [[ -n $ROW ]] || { echo "!!! $name not in $MANIFEST"; n_failed=$((n_failed + 1)); continue; }
  IFS=$'\t' read -r _ _ _ RUN _ _ _ OVERRIDES <<< "$ROW"
  DIR=$OUT/runs/$RUN
  mkdir -p "$DIR"
  echo "=== $RUN $(date +%T)"
  # shellcheck disable=SC2086
  .venv/bin/python flow_train.py experiment=flow3 $OVERRIDES data.batch_size=4 \
    +trainer.limit_test_batches=1 \
    callbacks.test_and_save_after_training.compute_pymatgen_heuristic_rmsd=true \
    +callbacks.test_and_save_after_training.predictions_path="$DIR" \
    callbacks.test_and_save_after_training.runs_stats_path="$DIR" hydra.run.dir="$DIR/hydra" \
    data.num_workers=0 train=false logger=csv task_name="smoke_$RUN" \
    ckpt_path="$CKPT" custom_model_weight_path="$CKPT" < /dev/null > "$DIR/run.log" 2>&1
  if [[ $? -eq 0 && -f $DIR/samples_all.pkl ]]; then
    echo "--- ok $(date +%T)"
  else
    echo "!!! $RUN failed:"; tail -5 "$DIR/run.log"; n_failed=$((n_failed + 1))
  fi
done

export PYTHONPATH=. OMP_NUM_THREADS=2 SLURM_CPUS_PER_TASK=2
for surface in gxtb aimnet; do
  run=$([[ $surface == gxtb ]] && echo gx_fk_N30_s1 || echo aim_fk_N30_s1)
  PKL=$OUT/runs/$run/samples_all.pkl
  S=$OUT/saddle/$surface
  echo "=== Sella + IRC on $surface, 2 reactions $(date +%T)"
  .venv/bin/python sella_check.py run "$S/reference/sella" 0 1 calculator=$surface data="$PKL" n_reactions=2 &&
    .venv/bin/python sella_check.py run "$S/$run/sella" 0 1 calculator=$surface \
      reference_from="$S/reference/sella" n_reactions=2 "$run=$PKL" &&
    .venv/bin/python irc_check.py run "$S/$run/irc" 0 1 calculator=$surface data="$PKL" \
      n_reactions=2 "$run=$S/$run/sella" paired=1 &&
    .venv/bin/python sella_check.py summarize "$S/$run/sella" &&
    .venv/bin/python irc_check.py summarize "$S/$run/irc" scripts/paper/stereo_counts.csv ||
    { echo "!!! Sella + IRC on $surface failed"; n_failed=$((n_failed + 1)); }
done
echo "smoke: $n_failed failure(s)"
[[ $n_failed -eq 0 ]]
