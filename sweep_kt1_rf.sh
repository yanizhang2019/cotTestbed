#!/usr/bin/env bash
# Kill test 1 recheck under the result-first compact trace, for the three frozen kill-test-2 cells
# only (p = 7, 4 layers; chain T = 4, chain T = 8, balanced T = 7), one seed each. Direct runs have
# no trace, so the original kill-test-1 direct runs are reused for the readout.
#
#   bash sweep_kt1_rf.sh            # needs the original sweep's runs in runs/kt1
set -u
OUT=${OUT:-runs/kt1_rf}
KT1=${KT1:-runs/kt1}
mkdir -p "$OUT"
for st in chain:4 chain:8 balanced:7; do
  shape=${st%%:*}; T=${st##*:}
  echo "=== $shape T=$T p=7 L=4 cot, result-first ($(date +%H:%M))"
  python train.py --shape "$shape" --T "$T" --p 7 --layers 4 --mode cot --trace compact_rf \
    --seed 0 --out "$OUT" --skip_done "$@" || echo "$shape T=$T cot compact_rf" >> "$OUT/failed.txt"
done
python stats.py kt1 "$KT1" "$OUT" | tee "$OUT/kt1_rf_report.txt"
