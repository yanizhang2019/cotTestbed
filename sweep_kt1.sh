#!/usr/bin/env bash
# Kill test 1 (plan): p in {5,7}; chains T in {4,8,12}; balanced T in {3,7,15}; 2 or 4 layers;
# direct and cot in each cell; one seed. 48 runs, then the gate readout.
#
#   bash sweep_kt1.sh                      # default: runs/kt1
#   OUT=/home/yani/runs/kt1 bash sweep_kt1.sh --epochs 20
#
# Extra arguments are passed to every train.py call. Finished runs are skipped, so re-running
# the script after an interruption resumes where it stopped.
set -u
OUT=${OUT:-runs/kt1}
mkdir -p "$OUT"
# keep failures from earlier sweeps separate from this one
[ -f "$OUT/failed.txt" ] && mv "$OUT/failed.txt" "$OUT/failed.$(date +%Y%m%d-%H%M%S).txt"
echo "sweep started $(date)" > "$OUT/sweep_started.txt"
n=0
for p in 5 7; do
  for st in chain:4 chain:8 chain:12 balanced:3 balanced:7 balanced:15; do
    shape=${st%%:*}; T=${st##*:}
    for L in 2 4; do
      for mode in direct cot; do
        n=$((n + 1))
        echo "=== [$n/48] $shape T=$T p=$p L=$L $mode ($(date +%H:%M))"
        python train.py --shape "$shape" --T "$T" --p "$p" --mode "$mode" --layers "$L" \
          --seed 0 --out "$OUT" --skip_done "$@" \
          || echo "$shape T=$T p=$p L=$L $mode" >> "$OUT/failed.txt"
      done
    done
  done
done
python stats.py kt1 "$OUT" | tee "$OUT/kt1_report.txt"
