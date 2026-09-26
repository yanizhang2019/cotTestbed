"""Statistics and gate readouts. So far: the kill-test-1 and kill-test-2 gates.

  python stats.py kt1 runs/kt1 [runs/kt1_rf ...]   (several dirs: direct runs are shared across traces)
  python stats.py kt2 runs/kt2

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


def kt1(*roots) -> bool:
    """Rows are (cell, CoT trace). Direct runs contain no trace, so a direct run is matched to every
    CoT trace of its cell (e.g. KT1's direct runs are reused for a result-first CoT recheck)."""
    direct, cot = {}, {}
    for r in (r for root in roots for r in load_results(root)):
        k = (r["p"], r["shape"], r["T"], r["layers"])
        if r["mode"] == "direct":
            direct[k] = r
        elif r["mode"] == "cot":
            cot[k + (r["trace"],)] = r
    rows = set(cot) | {k + ("-",) for k in direct if not any(c[:4] == k for c in cot)}
    hdr = (f"{'p':>2} {'shape':<9}{'T':>3} {'L':>2} {'cot trace':<11} {'direct':>15}  {'cot':>15}  "
           f"{'chance+10':>9}  pass")
    print(hdr)
    print("-" * len(hdr))
    passing: dict = {}
    missing = 0
    for key in sorted(rows):
        p, shape, T, L, trace = key
        d, c = direct.get(key[:4]), cot.get(key)
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
            passing.setdefault(trace, {}).setdefault(shape, []).append(key)
        print(f"{p:>2} {shape:<9}{T:>3} {L:>2} {trace:<11} {fmt(d):>15}  {fmt(c):>15}  {bar:>9.3f}  "
              f"{'yes' if ok else ''}")
    traces = sorted({k[4] for k in cot}) or ["-"]
    gates = {t: bool(passing.get(t, {}).get("chain")) and bool(passing.get(t, {}).get("balanced")) for t in traces}
    print("\nvalidation accuracy with 95% Wilson intervals; test is not used for gates")
    print("* = validation accuracy still rising at the end of training (a longer run could change it)")
    print(f"missing runs: {missing}")
    for t, g in gates.items():
        n = {sh: len(v) for sh, v in passing.get(t, {}).items()}
        print(f"KILL TEST 1 [{t}]: {'PASS' if g else 'FAIL'}  (passing chain cells: {n.get('chain', 0)}; "
              f"balanced: {n.get('balanced', 0)})")
    gate = all(gates.values())
    if not gate:
        print("plan decision rules: direct succeeds everywhere -> 1-2 layers, longer T, balanced only, rerun that "
              "slice; CoT fails -> check generator and tokenization, then a smaller p")
    return gate


# --------------------------------------------------------------------------- kill test 2

KT2_LATENT_MIN = 0.90  # mean latent validation accuracy
KT2_ZERO_MARGIN = 0.10  # mean zero-latent validation accuracy <= 1/p + 10 pts
KT2_PAUSE_MARGIN = 0.10  # mean pause <= mean direct + 10 pts
KT2_SEEDS = 3
KT2_SEED_SET = {0, 1, 2}
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def mean_ci(xs: list) -> tuple:
    """Mean and 95% t-interval over seeds."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = _T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return m, max(0.0, m - h), min(1.0, m + h)


def kt2(root) -> bool:
    """Kill test 2 as frozen on 2026-09-26: a cell passes on 3-seed means (validation only) if
    latent >= 90%, zero-latent <= 1/p + 10 pts and pause <= direct + 10 pts. Gate: at least one
    chain cell and balanced T = 7 pass; balanced T = 3 replaces T = 7 only if T = 7 fails to learn."""
    cells: dict = {}
    directs: dict = {}
    for r in load_results(root):
        k = (r["p"], r["shape"], r["T"], r["layers"])
        if r["mode"] == "direct":  # no trace: shared by every trace of the cell
            directs.setdefault(k, []).append(r)
        elif r["mode"] in ("latent", "pause"):
            cells.setdefault(k + (r["trace"],), {}).setdefault(r["mode"], []).append(r)
    for key in cells:
        cells[key]["direct"] = directs.get(key[:4], [])

    def pick(runs):
        """One run per seed in {0, 1, 2}. A seed found more than once (e.g. a rerun under --tag)
        is left out rather than guessed, so it can never be counted twice."""
        seen, dup = {}, set()
        for r in runs or []:
            s = r["seed"]
            if s in KT2_SEED_SET:
                dup |= {s} if s in seen else set()
                seen[s] = r
        return [seen[s] for s in sorted(seen) if s not in dup], dup

    def have_three(runs):
        return {r["seed"] for r in runs} >= KT2_SEED_SET

    def col(runs, key):
        return [r[key] for r in runs if key in r]

    def fmt(xs):
        if not xs:
            return "missing"
        m, lo, hi = mean_ci(xs)
        return f"{m:.3f} [{lo:.2f},{hi:.2f}] ({' '.join(f'{x:.3f}' for x in xs)})"

    verdict: dict = {}
    for key in sorted(cells):
        p, shape, T, L, trace = key
        picked, dups = {}, {}
        for mode in ("latent", "pause", "direct"):
            picked[mode], dups[mode] = pick(cells[key].get(mode))
        c = picked
        lat, zero = col(c["latent"], "val_acc"), col(c["latent"], "zero_val_acc")
        pause, direct = col(c["pause"], "val_acc"), col(c["direct"], "val_acc")
        complete = all(have_three(c[mode]) for mode in ("latent", "pause", "direct")) and len(zero) >= 3
        learn = bool(lat) and mean_ci(lat)[0] >= KT2_LATENT_MIN
        need = bool(zero) and mean_ci(zero)[0] <= 1 / p + KT2_ZERO_MARGIN
        ctrl = bool(pause and direct) and mean_ci(pause)[0] <= mean_ci(direct)[0] + KT2_PAUSE_MARGIN
        ok = complete and learn and need and ctrl
        verdict[key] = dict(ok=ok, learn=learn, complete=complete)
        print(f"p={p} {shape} T={T} L={L} trace={trace}  ->  {'PASS' if ok else 'fail'}"
              + ("" if complete else "  (incomplete: needs seeds 0, 1, 2 of latent, pause and direct)"))
        for mode, d in dups.items():
            if d:
                print(f"  WARNING {mode}: seed(s) {sorted(d)} found more than once; left out until resolved")
        print(f"  latent       {fmt(lat):<50} need >= {KT2_LATENT_MIN:.3f}  {'ok' if learn else 'NO'}")
        print(f"  zero-latent  {fmt(zero):<50} need <= {1 / p + KT2_ZERO_MARGIN:.3f}  {'ok' if need else 'NO'}")
        dbar = f"{mean_ci(direct)[0] + KT2_PAUSE_MARGIN:.3f}" if direct else "?"
        print(f"  pause        {fmt(pause):<50} need <= {dbar}  {'ok' if ctrl else 'NO'}")
        print(f"  direct       {fmt(direct)}")
        if c.get("latent"):
            print(f"  latent epochs {col(c['latent'], 'epochs_total')}, pause epochs {col(c.get('pause'), 'epochs_total')}")
    print("\nall numbers are validation accuracy; mean [95% t-interval over seeds] (per-seed values)")

    chain_ok = any(v["ok"] for (p, s, T, L, tr), v in verdict.items() if s == "chain")
    t7 = [v for (p, s, T, L, tr), v in verdict.items() if s == "balanced" and T == 7]
    t3 = [v for (p, s, T, L, tr), v in verdict.items() if s == "balanced" and T == 3]
    t7_ok = any(v["ok"] for v in t7)
    t7_unlearned = bool(t7) and all(v["complete"] and not v["learn"] for v in t7)
    fallback_ok = t7_unlearned and any(v["ok"] for v in t3)
    gate = chain_ok and (t7_ok or fallback_ok)
    print(f"chain cell passing: {'yes' if chain_ok else 'no'}; balanced T=7 passing: {'yes' if t7_ok else 'no'}"
          + ("; fallback balanced T=3 passing: yes" if fallback_ok else ""))
    print(f"KILL TEST 2: {'PASS' if gate else 'FAIL'}")
    if not gate and chain_ok and t7_unlearned and not t3:
        print("balanced T=7 failed to learn: run the predefined fallback with  FALLBACK=1 bash sweep_kt2.sh")
    if not gate and not any(v["learn"] for v in verdict.values()):
        print("plan decision rule: latent training fails -> 2 latents per reduction, then a slower curriculum; "
              "after two failed tries stop, or fall back to a finite-automaton task")
    return gate


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "kt1":
        kt1(*sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == "kt2":
        kt2(sys.argv[2])
    else:
        raise SystemExit("usage: python stats.py kt1 <runs dir> [<runs dir> ...]  |  python stats.py kt2 <runs dir>")
