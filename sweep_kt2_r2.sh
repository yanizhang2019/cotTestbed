#!/usr/bin/env bash
# Kill test 2, retry 2 of 2 (see the plan): the two long cells only, with a mastery-gated curriculum
# as the single change from retry 1.
#   p = 7, 4 layers; chain T = 8 and balanced T = 7; seeds 0-2
#   unchanged from retry 1: result-first trace, latents x 1/sqrt(d), 1 latent per reduction,
#     peak lr 3e-4, 5,000 validation examples per epoch, kill-test-2 thresholds
#   new: each intermediate stage trains 3-12 epochs and advances only after validation >= 0.98 for
#     2 consecutive epochs; an unmastered stage at 12 epochs stops that seed (recorded); the final
#     stage trains 20 epochs and keeps the best validation checkpoint
#   pause trains for the realized epochs of its paired latent run; direct runs are reused from retry 1
#
#   bash sweep_kt2_r2.sh           # -> runs/kt2_retry2, direct baselines from runs/kt2_retry1
set -u
OUT=${OUT:-runs/kt2_retry2}
R1=${R1:-runs/kt2_retry1}
mkdir -p "$OUT"
[ -f "$OUT/failed.txt" ] && mv "$OUT/failed.txt" "$OUT/failed.$(date +%Y%m%d-%H%M%S).txt"
echo "sweep started $(date)" >> "$OUT/sweep_started.txt"

COMMON=(--p 7 --layers 4 --eval_n 5000 --trace compact_rf --latent_scale invsqrt_d --lr 3e-4 --out "$OUT" --skip_done)
MASTERY=(--mastery_threshold 0.98 --mastery_patience 2 --min_epochs_per_stage 3 --max_epochs_per_stage 12
         --final_epochs 20)

latent_epochs() {  # realized epochs of a finished latent run, from its results.json
  python -c "import json, sys, train; a = train.parse_args(sys.argv[1:]); \
print(json.loads((train.run_dir(a) / 'results.json').read_text())['epochs_total'])" "$@"
}

for st in chain:8 balanced:7; do
  shape=${st%%:*}; T=${st##*:}
  for seed in 0 1 2; do
    base=(--shape "$shape" --T "$T" --seed "$seed" "${COMMON[@]}")
    echo "=== $shape T=$T seed=$seed latent, mastery curriculum ($(date +%H:%M))"
    if python train.py "${base[@]}" --mode latent --feedback base "${MASTERY[@]}" "$@"; then
      ep=$(latent_epochs "${base[@]}" --mode latent --feedback base "${MASTERY[@]}" "$@")
      echo "=== $shape T=$T seed=$seed pause, $ep epochs ($(date +%H:%M))"
      python train.py "${base[@]}" --mode pause "$@" --epochs "$ep" \
        || echo "$shape T=$T seed=$seed pause" >> "$OUT/failed.txt"
    else
      echo "$shape T=$T seed=$seed latent (pause skipped)" >> "$OUT/failed.txt"
    fi
  done
done
python stats.py kt2 "$OUT" --direct-from "$R1" | tee "$OUT/kt2_report.txt"
