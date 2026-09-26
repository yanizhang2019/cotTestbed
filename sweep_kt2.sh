#!/usr/bin/env bash
# Kill test 2, frozen 2026-09-26 before any latent run (see the plan):
#   p = 7, 4 layers; chain T = 4, chain T = 8, balanced T = 7; seeds 0-2
#   per cell and seed: base latent (fixed 3 epochs per intermediate stage), then pause for the
#   latent run's realized epochs, then direct; 5,000 validation examples per epoch.
#
#   bash sweep_kt2.sh                  # the three frozen cells -> runs/kt2
#   FALLBACK=1 bash sweep_kt2.sh       # only the predefined fallback cell, balanced T = 3
#   Frozen after the out-of-gate pilots (see the plan): result-first compact trace, latents x 1/sqrt(d).
#
# Finished runs are skipped, so re-running resumes after an interruption.
set -u
OUT=${OUT:-runs/kt2}
mkdir -p "$OUT"
[ -f "$OUT/failed.txt" ] && mv "$OUT/failed.txt" "$OUT/failed.$(date +%Y%m%d-%H%M%S).txt"
echo "sweep started $(date)" >> "$OUT/sweep_started.txt"

CELLS="chain:4 chain:8 balanced:7"
[ "${FALLBACK:-0}" = 1 ] && CELLS="balanced:3"
TRACE=${TRACE:-compact_rf}   # frozen: result-first compact trace
SCALE=${SCALE:-invsqrt_d}    # frozen: one global latent scale 1/sqrt(d_model)
COMMON=(--p 7 --layers 4 --eval_n 5000 --trace "$TRACE" --latent_scale "$SCALE" --out "$OUT" --skip_done)

latent_epochs() {  # realized epochs of a finished latent run, from its results.json
  python -c "import json, sys, train; a = train.parse_args(sys.argv[1:]); \
print(json.loads((train.run_dir(a) / 'results.json').read_text())['epochs_total'])" "$@"
}

n=0
for st in $CELLS; do
  shape=${st%%:*}; T=${st##*:}
  for seed in 0 1 2; do
    base=(--shape "$shape" --T "$T" --seed "$seed" "${COMMON[@]}")
    n=$((n + 1))
    echo "=== [$n] $shape T=$T seed=$seed latent ($(date +%H:%M))"
    if python train.py "${base[@]}" --mode latent --feedback base --fixed_stages "$@"; then
      ep=$(latent_epochs "${base[@]}" --mode latent --feedback base --fixed_stages "$@")
      echo "=== [$n] $shape T=$T seed=$seed pause, $ep epochs ($(date +%H:%M))"
      python train.py "${base[@]}" --mode pause "$@" --epochs "$ep" \
        || echo "$shape T=$T seed=$seed pause" >> "$OUT/failed.txt"
    else
      echo "$shape T=$T seed=$seed latent (pause skipped)" >> "$OUT/failed.txt"
    fi
    echo "=== [$n] $shape T=$T seed=$seed direct ($(date +%H:%M))"
    python train.py "${base[@]}" --mode direct "$@" \
      || echo "$shape T=$T seed=$seed direct" >> "$OUT/failed.txt"
  done
done
python stats.py kt2 "$OUT" | tee "$OUT/kt2_report.txt"
