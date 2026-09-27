"""Causal tests on the latent codes: C2 (swaps), C3 (tape test), C4 (shared alphabet).

  python swaps.py <run dir> [<run dir> ...] [--claims c2 c3 c4] [--split test] [--seed 0]

Everything is fit on validation latents (capture_val.npz) and scored on the eval split (test by
default), on the expressions the clean model answers correctly. Later latents are always
recomputed from a swapped one, as in any hooked forward pass.

C2, swaps. For every scored expression and every step t, one alternative value v' != v_t is drawn
uniformly (seeded) and latent t is replaced by
  centroid     c[t, v'] (mean validation latent of step t over expressions with v_t = v')
  donor        a random validation latent of step t with value v'
  same_donor   a random validation latent of step t with value v_t (control: same value, other expression)
A centroid or donor swap is scored only if its counterfactual answer, gen.evaluate(ex, {t: v'}),
differs from the original; it succeeds if the model's answer equals the counterfactual answer.
Plan (C2): counterfactual answer in >= 80% of those swaps; same-value donors keep the answer >= 95%.

C3, tape test. For every step t that has later steps, latent t is replaced by c[t, v'] and every
later latent is decoded with per-slot linear probes (fit on validation). Later steps that do not
depend on t (off its consumer chain) should keep their value ("keep"); the consumer chain should
follow the counterfactual trajectory gen.evaluate(ex, {t: v'}) ("follow", scored on dependent
steps whose counterfactual value differs from the original; "follow_all" also counts the steps
where it does not). Plan (C3): keep >= 95%, follow >= 80%. In a chain every later step depends on
t, so keep is undefined there. Clean probe accuracy on the eval split is reported per slot, with a
shuffled-label probe as control (C8): the tape is only as good as the probes. The full
(swapped step t) x (decoded step s) matrix is reported for keep, follow and follow_all.

C4, shared alphabet. With one draw of v' per (expression, step), latent t is replaced by c[t2, v']
for every source step t2 (a T x T matrix; the diagonal t2 = t is the same-step swap), and by the
global centroid of v' (one value codebook shared by all steps). same_step pools the diagonal and
cross_step pools the off-diagonal, over swaps that change the answer. Separately, every latent is
snapped to its nearest per-step centroid, or to its nearest global centroid, and answer accuracy
on the whole eval split is compared as retention of clean accuracy.
Plan (C4), frozen 2026-09-27 as one-sided non-inferiority tests on the mean over seeds 0-2:
  C4a  mean_s (R_cross,s - R_same,s) >= -0.10, with R = counterfactual rate on the same paired swaps,
       scored only where the counterfactual answer differs from the original
  C4b  mean_s (Retention_global,s - Retention_step,s) >= -0.05, a retention-point difference, with
       Retention = accuracy under snapping / clean accuracy
No absolute value: a shared or borrowed code doing better is no evidence against a shared alphabet.
The 95% seed interval is reported but is not part of the criterion.

Writes c2/c3/c4_<split>.json (summaries, per-step and pairwise matrices) and c2/c3/c4_<split>.npz
(raw per-swap outcomes, for paired tests and re-analysis).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import gen
from capture import get_capture, hooked_pass, load_run, predict, step_values
from codebook import fit_probes, global_centroids, probe_accuracy, probe_predict, snap_hook, step_centroids
from gen import evaluate

C2_CF_MIN = 0.80  # counterfactual answer in >= 80% of swaps where it differs from the original
C2_SAME_MIN = 0.95  # same-value donors keep the answer >= 95%
C3_KEEP_MIN = 0.95  # later steps not using t keep their value
C3_FOLLOW_MIN = 0.80  # the consuming step and later follow the counterfactual
C4_CROSS_MAX_GAP = 0.10  # cross-step swaps at most 10 pts below same-step swaps (one-sided)
C4_SNAP_MAX_GAP = 0.05  # global snap retention at most 0.05 retention points below per-step (one-sided)
CONDITIONS = ("centroid", "donor", "same_donor")


# --------------------------------------------------------------------------- shared setup

def fit(cap_val: dict, p: int) -> dict:
    """Per-step value centroids and donor pools from validation latents only."""
    lat, vals = cap_val["latents"], cap_val["values"]
    pools = {(t, v): np.flatnonzero(vals[:, t] == v) for t in range(lat.shape[1]) for v in range(p)}
    return dict(centroids=step_centroids(cap_val, p), pools=pools, latents=lat)


def draw_donors(fitted: dict, t: int, values: np.ndarray, rng) -> np.ndarray:
    """One validation latent of step t per requested value."""
    out = np.empty((len(values), fitted["latents"].shape[2]), dtype=np.float32)
    for v in np.unique(values):
        rows = np.flatnonzero(values == v)
        pool = fitted["pools"][t, int(v)]
        if len(pool) == 0:
            raise RuntimeError(f"no validation latent with value {v} at step {t}")
        out[rows] = fitted["latents"][rng.choice(pool, size=len(rows)), t]
    return out


def slot_hook(t: int, replacement: torch.Tensor):
    """Replace the latent at slot t with replacement[batch_idx]; leave every other slot alone."""
    def hook(slot, h, batch_idx):
        return replacement[batch_idx].to(h.dtype) if slot == t else h
    return hook


def setup(run_dir, split="test", device="auto", bs=1024, only_correct=True) -> SimpleNamespace:
    """Model, validation fit and the scored subset of the eval split, shared by C2-C4."""
    run_dir = Path(run_dir)
    model, a, splits, vocab, dev, amp = load_run(run_dir, device)
    cap_val = get_capture(run_dir, "val", model, a, splits, vocab, dev, amp, bs)
    full = getattr(splits, split)
    clean = predict(model, full, vocab, a.trace, dev, amp, bs)
    keep = np.flatnonzero(clean == full.answer) if only_correct else np.arange(len(full))
    if len(keep) == 0:
        raise RuntimeError(f"{run_dir}: the clean model answers no {split} expression correctly; nothing to swap")
    exprs = full.take(keep)
    return SimpleNamespace(run_dir=run_dir, model=model, a=a, splits=splits, vocab=vocab, dev=dev, amp=amp,
                           bs=bs, split=split, cap_val=cap_val, fitted=fit(cap_val, a.p), full=full, clean=clean,
                           keep=keep, only_correct=only_correct, exprs=exprs, exs=exprs.examples(),
                           true=step_values(exprs), ans=exprs.answer, n=len(exprs), p=a.p, T=a.T)


def _predict(ctx, hook=None, exprs=None):
    return predict(ctx.model, ctx.exprs if exprs is None else exprs, ctx.vocab, ctx.a.trace, ctx.dev, ctx.amp,
                   ctx.bs, hook=hook)


def _alt(ctx, t, rng) -> np.ndarray:
    return (ctx.true[:, t] + rng.integers(1, ctx.p, size=ctx.n)) % ctx.p  # uniform over the other p-1 values


def _rate(hit, mask) -> float:
    return float(hit[mask].mean()) if mask.any() else float("nan")


def _header(ctx, claim) -> dict:
    a = ctx.a
    return dict(claim=claim, run=str(ctx.run_dir), cell=f"{a.shape}_T{a.T}_p{a.p}", seed=a.seed, split=ctx.split,
                chance=1 / ctx.p, clean_acc=float((ctx.clean == ctx.full.answer).mean()),
                only_correct=ctx.only_correct, n_scored=ctx.n)


def _ctx(run_dir, split, device, bs, only_correct, ctx):
    return ctx if ctx is not None else setup(run_dir, split, device, bs, only_correct)


# --------------------------------------------------------------------------- C2

def run_c2(run_dir=None, split="test", seed=0, device="auto", bs=1024, only_correct=True, ctx=None) -> dict:
    ctx = _ctx(run_dir, split, device, bs, only_correct, ctx)
    T, n = ctx.T, ctx.n
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    outcome = {c: np.full((n, T), -1, dtype=np.int64) for c in CONDITIONS}
    alt = np.empty((n, T), dtype=np.int64)
    cf = np.empty((n, T), dtype=np.int64)
    for t in range(T):
        alt[:, t] = _alt(ctx, t, rng)
        cf[:, t] = [evaluate(e, {t: int(v)})[1] for e, v in zip(ctx.exs, alt[:, t])]
        repl = {"centroid": ctx.fitted["centroids"][t, alt[:, t]],
                "donor": draw_donors(ctx.fitted, t, alt[:, t], rng),
                "same_donor": draw_donors(ctx.fitted, t, ctx.true[:, t], rng)}
        for c in CONDITIONS:
            outcome[c][:, t] = _predict(ctx, slot_hook(t, torch.as_tensor(repl[c], device=ctx.dev)))
    ans = ctx.ans
    differs = cf != ans[:, None]
    s = _header(ctx, "c2")
    s.update(n_swaps_differ=int(differs.sum()), per_step_differ=differs.sum(0).tolist())
    for c in ("centroid", "donor"):
        hit = outcome[c] == cf
        s[c] = dict(cf_rate=_rate(hit, differs), kept_original=_rate(outcome[c] == ans[:, None], differs),
                    per_step=[_rate(hit[:, t], differs[:, t]) for t in range(T)])
    same_keep = outcome["same_donor"] == ans[:, None]
    s["same_donor"] = dict(keep_rate=float(same_keep.mean()), per_step=[float(same_keep[:, t].mean()) for t in range(T)])
    s["pass"] = dict(centroid=s["centroid"]["cf_rate"] >= C2_CF_MIN, donor=s["donor"]["cf_rate"] >= C2_CF_MIN,
                     same_donor=s["same_donor"]["keep_rate"] >= C2_SAME_MIN)
    (ctx.run_dir / f"c2_{ctx.split}.json").write_text(json.dumps(s, indent=2))
    np.savez_compressed(ctx.run_dir / f"c2_{ctx.split}.npz", index=ctx.keep, alt=alt, cf=cf, answer=ans,
                        **{f"pred_{c}": outcome[c] for c in CONDITIONS})
    return s


# --------------------------------------------------------------------------- C3

def dependents_mask(exprs: gen.Batch) -> np.ndarray:
    """(N, T, T) bool: [i, t, s] is True if step s depends on step t (s is on t's consumer chain)."""
    c = exprs.cell
    out = np.zeros((len(exprs), c.T, c.T), dtype=bool)
    for r in np.unique(exprs.rank):
        skel = gen.skeleton(c.shape, c.T, int(r))
        rows = exprs.rank == r
        for t in range(c.T):
            for s in gen.dependents(skel, t):
                out[rows, t, s] = True
    return out


def run_c3(run_dir=None, split="test", seed=0, device="auto", bs=1024, only_correct=True, ctx=None) -> dict:
    ctx = _ctx(run_dir, split, device, bs, only_correct, ctx)
    T, n, p = ctx.T, ctx.n, ctx.p
    probes = fit_probes(ctx.cap_val, p, seed=seed)
    shuffled = fit_probes(ctx.cap_val, p, seed=seed, shuffle=True)
    cap_eval = get_capture(ctx.run_dir, ctx.split, ctx.model, ctx.a, ctx.splits, ctx.vocab, ctx.dev, ctx.amp, ctx.bs)
    dep = dependents_mask(ctx.exprs)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    cent = ctx.fitted["centroids"]
    mat = {k: np.zeros((T, T, 2), dtype=np.int64) for k in ("keep", "follow", "follow_all")}  # [t, s] = (hits, n)
    alts = np.full((n, T), -1, dtype=np.int64)
    cf_traj = np.full((T, n, T), -1, dtype=np.int64)  # [t] = counterfactual step values for a swap at t
    decoded = np.full((T, n, T), -1, dtype=np.int64)  # [t, i, s] = probe-decoded value of slot s (s > t)
    for t in range(T - 1):
        alt = _alt(ctx, t, rng)
        alts[:, t] = alt
        cfv = np.array([evaluate(e, {t: int(v)})[0] for e, v in zip(ctx.exs, alt)])
        cf_traj[t] = cfv
        _, lat = hooked_pass(ctx.model, ctx.exprs, ctx.vocab, ctx.a.trace, ctx.dev, ctx.amp, ctx.bs,
                             hook=slot_hook(t, torch.as_tensor(cent[t, alt], device=ctx.dev)))
        for s in range(t + 1, T):
            dec = probe_predict(probes, lat, s)
            decoded[t, :, s] = dec
            d = dep[:, t, s]
            changed = d & (cfv[:, s] != ctx.true[:, s])
            for k, m, target in (("keep", ~d, ctx.true[:, s]), ("follow", changed, cfv[:, s]),
                                 ("follow_all", d, cfv[:, s])):
                mat[k][t, s] += ((dec == target)[m].sum(), m.sum())

    def overall(k):
        h, m = mat[k].sum((0, 1))
        return float(h / m) if m else float("nan")

    def rates(a):  # (..., 2) -> rates with NaN where nothing was scored
        return np.where(a[..., 1] > 0, a[..., 0] / np.maximum(a[..., 1], 1), np.nan)

    s = _header(ctx, "c3")
    pa = probe_accuracy(probes, cap_eval)
    s.update(probe_acc=pa, probe_acc_mean=float(np.mean(pa)),
             probe_shuffled_acc_mean=float(np.mean(probe_accuracy(shuffled, cap_eval))),
             keep_rate=overall("keep"), follow_rate=overall("follow"), follow_all_rate=overall("follow_all"),
             n_keep=int(mat["keep"][..., 1].sum()), n_follow=int(mat["follow"][..., 1].sum()),
             per_step={k: rates(mat[k].sum(1)).tolist() for k in mat},
             matrix={k: rates(mat[k]).tolist() for k in mat},
             matrix_n={k: mat[k][..., 1].tolist() for k in mat})
    keep_ok = np.isnan(s["keep_rate"]) or s["keep_rate"] >= C3_KEEP_MIN
    s["pass"] = dict(keep=bool(keep_ok), follow=s["follow_rate"] >= C3_FOLLOW_MIN)
    (ctx.run_dir / f"c3_{ctx.split}.json").write_text(json.dumps(s, indent=2))
    np.savez_compressed(ctx.run_dir / f"c3_{ctx.split}.npz", index=ctx.keep, true=ctx.true, alt=alts,
                        cf_values=cf_traj, decoded=decoded, dependents=dep)
    return s


# --------------------------------------------------------------------------- C4

def run_c4(run_dir=None, split="test", seed=0, device="auto", bs=1024, only_correct=True, ctx=None) -> dict:
    ctx = _ctx(run_dir, split, device, bs, only_correct, ctx)
    T, n, p = ctx.T, ctx.n, ctx.p
    cent = ctx.fitted["centroids"]
    glob = global_centroids(ctx.cap_val, p)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    pred_mat = np.full((T, T, n), -1, dtype=np.int64)  # [t, t2, i]: answer when latent t gets c[t2, v']
    pred_glob = np.full((T, n), -1, dtype=np.int64)
    alts = np.empty((n, T), dtype=np.int64)
    cf = np.empty((n, T), dtype=np.int64)
    for t in range(T):
        alt = _alt(ctx, t, rng)
        alts[:, t] = alt
        cf[:, t] = [evaluate(e, {t: int(v)})[1] for e, v in zip(ctx.exs, alt)]
        for t2 in range(T):
            pred_mat[t, t2] = _predict(ctx, slot_hook(t, torch.as_tensor(cent[t2, alt], device=ctx.dev)))
        pred_glob[t] = _predict(ctx, slot_hook(t, torch.as_tensor(glob[alt], device=ctx.dev)))
    differs = (cf != ctx.ans[:, None]).T  # (T, n)
    hit = pred_mat == cf.T[:, None, :]  # (T, T, n)
    mat_h = (hit & differs[:, None, :]).sum(-1)
    mat_n = np.broadcast_to(differs.sum(-1)[:, None], (T, T))
    diag, off = np.eye(T, dtype=bool), ~np.eye(T, dtype=bool)
    s = _header(ctx, "c4")
    s["cross_matrix"] = np.where(mat_n > 0, mat_h / np.maximum(mat_n, 1), np.nan).tolist()
    s["same_step_rate"] = float(mat_h[diag].sum() / max(mat_n[diag].sum(), 1))
    s["cross_step_rate"] = float(mat_h[off].sum() / max(mat_n[off].sum(), 1)) if T > 1 else float("nan")
    s["global_rate"] = _rate((pred_glob == cf.T), differs)
    s["same_step_per_step"] = [_rate(hit[t, t], differs[t]) for t in range(T)]
    s["cross_step_per_step"] = [float(mat_h[t, off[t]].sum() / max(mat_n[t, off[t]].sum(), 1)) if T > 1
                                else float("nan") for t in range(T)]
    s["global_per_step"] = [_rate(pred_glob[t] == cf[:, t], differs[t]) for t in range(T)]
    clean_acc = s["clean_acc"]
    snap_pred = {}
    for name, book in (("step", cent), ("global", glob)):
        snap_pred[name] = _predict(ctx, snap_hook(book, ctx.dev), exprs=ctx.full)
        acc = float((snap_pred[name] == ctx.full.answer).mean())
        s[f"snap_{name}_acc"] = acc
        s[f"snap_{name}_retention"] = acc / clean_acc if clean_acc else float("nan")
    s["cross_minus_same"] = s["cross_step_rate"] - s["same_step_rate"]
    s["global_minus_step_snap"] = s["snap_global_retention"] - s["snap_step_retention"]
    s["pass"] = dict(cross_step=s["cross_minus_same"] >= -C4_CROSS_MAX_GAP,
                     global_snap=s["global_minus_step_snap"] >= -C4_SNAP_MAX_GAP)
    (ctx.run_dir / f"c4_{ctx.split}.json").write_text(json.dumps(s, indent=2))
    np.savez_compressed(ctx.run_dir / f"c4_{ctx.split}.npz", index=ctx.keep, alt=alts, cf=cf, answer=ctx.ans,
                        pred_matrix=pred_mat, pred_global=pred_glob, clean_full=ctx.clean,
                        answer_full=ctx.full.answer, pred_snap_step=snap_pred["step"],
                        pred_snap_global=snap_pred["global"])
    return s


# --------------------------------------------------------------------------- output

def _steps(xs):
    return "[" + " ".join("  -  " if x is None or x != x else f"{x:.3f}" for x in xs) + "]"


def show(s: dict):
    print(f"{s['run']}  [{s['claim']}, {s['split']}]  clean {s['clean_acc']:.4f}  scored {s['n_scored']}  "
          f"(chance {s['chance']:.3f})")
    ok = lambda k: "ok" if s["pass"][k] else "NO"  # noqa: E731
    if s["claim"] == "c2":
        for c in ("centroid", "donor"):
            print(f"  {c:<11} counterfactual {s[c]['cf_rate']:.4f}  kept original {s[c]['kept_original']:.4f}  "
                  f"per step {_steps(s[c]['per_step'])}  {ok(c)} (need >= {C2_CF_MIN})")
        print(f"  same_donor  kept original {s['same_donor']['keep_rate']:.4f}  per step "
              f"{_steps(s['same_donor']['per_step'])}  {ok('same_donor')} (need >= {C2_SAME_MIN})")
    elif s["claim"] == "c3":
        print(f"  probes      clean accuracy per slot {_steps(s['probe_acc'])}  "
              f"(shuffled-label control {s['probe_shuffled_acc_mean']:.3f})")
        print(f"  keep        {s['keep_rate']:.4f} of {s['n_keep']} independent later steps  per swapped step "
              f"{_steps(s['per_step']['keep'])}  {ok('keep')} (need >= {C3_KEEP_MIN})")
        print(f"  follow      {s['follow_rate']:.4f} of {s['n_follow']} changed dependent steps  per swapped step "
              f"{_steps(s['per_step']['follow'])}  {ok('follow')} (need >= {C3_FOLLOW_MIN}); "
              f"incl. unchanged {s['follow_all_rate']:.4f}")
        print("  tape matrix, row = swapped step t, column = decoded step s (keep if s is off t's chain, "
              "follow if on it):")
        for t in range(len(s["matrix"]["keep"])):
            row = [s["matrix"]["follow_all"][t][u] if s["matrix_n"]["follow_all"][t][u] else
                   s["matrix"]["keep"][t][u] for u in range(len(s["matrix"]["keep"]))]
            print(f"    t={t}  {_steps(row)}")
    else:
        for c in ("same_step", "cross_step", "global"):
            print(f"  {c:<11} counterfactual {s[c + '_rate']:.4f}  per step {_steps(s[c + '_per_step'])}")
        print(f"  cross - same {s['cross_minus_same']:+.4f}  {ok('cross_step')} (need >= -{C4_CROSS_MAX_GAP})")
        print(f"  snap        per-step retention {s['snap_step_retention']:.4f}  global {s['snap_global_retention']:.4f}"
              f"  diff {s['global_minus_step_snap']:+.4f}  {ok('global_snap')} (need >= -{C4_SNAP_MAX_GAP})")
        print("  cross matrix, row = swapped step t, column = source step t2 of the centroid (diagonal = same step):")
        for t, row in enumerate(s["cross_matrix"]):
            print(f"    t={t}  {_steps(row)}")


RUNNERS = {"c2": run_c2, "c3": run_c3, "c4": run_c4}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--claims", nargs="+", default=["c2", "c3", "c4"], choices=list(RUNNERS))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--all", action="store_true", help="score every expression, not only correctly answered ones")
    args = ap.parse_args()
    for d in args.run_dirs:
        ctx = setup(d, args.split, args.device, only_correct=not args.all)
        for c in args.claims:
            show(RUNNERS[c](seed=args.seed, ctx=ctx))
