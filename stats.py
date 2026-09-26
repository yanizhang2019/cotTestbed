"""Statistics and gate readouts. So far: the kill-test-1 gate.

  python stats.py kt1 runs/kt1

Kill test 1 (plan, thresholds fixed): pass if at least one chain cell and one balanced cell have
direct <= 1/p + 10 pts and CoT >= 98%, one seed. A cell here is (p, shape, T, layers).
Gates use full-validation accuracy: test is reserved for reporting frozen configurations.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

KT1_DIRECT_MARGIN = 0.10  # direct <= 1/p + 10 pts
KT1_COT_MIN = 0.98  # CoT >= 98%


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_results(root) -> list:
    return [json.loads(f.read_text()) for f in sorted(Path(root).rglob("results.json"))]


def kt1(root) -> bool:
    runs = load_results(root)
    cells: dict = {}
    for r in runs:
        if r["mode"] in ("direct", "cot") and r["trace"] == "compact":
            cells.setdefault((r["p"], r["shape"], r["T"], r["layers"]), {})[r["mode"]] = r
    hdr = f"{'p':>2} {'shape':<9}{'T':>3} {'L':>2}  {'direct':>15}  {'cot':>15}  {'chance+10':>9}  pass"
    print(hdr)
    print("-" * len(hdr))
    passing = {"chain": [], "balanced": []}
    missing = 0
    for key in sorted(cells):
        p, shape, T, L = key
        d, c = cells[key].get("direct"), cells[key].get("cot")
        bar = 1 / p + KT1_DIRECT_MARGIN

        def fmt(r):
            if r is None:
                return "missing"
            lo, hi = wilson(round(r["val_acc"] * r["n_val"]), r["n_val"])
            flag = "*" if r.get("still_improving") else " "
            return f"{r['val_acc']:.3f} [{lo:.2f},{hi:.2f}]{flag}"

        ok = d is not None and c is not None and d["val_acc"] <= bar and c["val_acc"] >= KT1_COT_MIN
        missing += (d is None) + (c is None)
        if ok:
            passing.setdefault(shape, []).append(key)
        print(f"{p:>2} {shape:<9}{T:>3} {L:>2}  {fmt(d):>15}  {fmt(c):>15}  {bar:>9.3f}  {'yes' if ok else ''}")
    gate = bool(passing.get("chain")) and bool(passing.get("balanced"))
    print("\nvalidation accuracy with 95% Wilson intervals; test is not used for gates")
    print("* = validation accuracy still rising at the end of training (a longer run could change it)")
    print(f"missing runs: {missing}")
    print(f"passing chain cells: {len(passing.get('chain', []))}; passing balanced cells: {len(passing.get('balanced', []))}")
    print(f"KILL TEST 1: {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("plan decision rules: direct succeeds everywhere -> 1-2 layers, longer T, balanced only, rerun that "
              "slice; CoT fails -> check generator and tokenization, then a smaller p")
    return gate


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "kt1":
        raise SystemExit("usage: python stats.py kt1 <runs dir>")
    kt1(sys.argv[2])
