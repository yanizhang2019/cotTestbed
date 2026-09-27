"""Capture latents and labels from a finished latent run (Analyses step 1 in the plan).

  python capture.py <run dir> [<run dir> ...] [--splits val test]

For every expression in a split: the latent fed into each slot (after the global latent scale,
i.e. exactly what hooks see and what is fed back), the true step values, operators, each step's
consumer and depth, the answer, and the model's clean answer. Saved as capture_<split>.npz in
the run dir. Fitted objects (centroids, probes, donors, radii) use validation; reported numbers
use test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import gen
from gen import Cell, Vocab, encode, make_splits
from model import ModelConfig, TinyGPT, answer_logits, forward_latent, greedy_answer, make_batch


def load_run(run_dir, device="auto"):
    """(model, args, splits, vocab, device, amp) for a finished run, with the split fingerprint checked."""
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "best.pt", map_location="cpu")
    a = argparse.Namespace(**ck["args"])
    if a.mode != "latent":
        raise ValueError(f"{run_dir} is a {a.mode} run; capture needs a latent run")
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    amp = dev.type == "cuda" and not a.no_bf16
    splits = make_splits(Cell(a.shape, a.T, a.p), a.split_seed)
    res = json.loads((run_dir / "results.json").read_text())
    if splits.fingerprint() != res["split_fingerprint"]:
        raise RuntimeError(f"{run_dir}: regenerated splits differ from the ones this run was trained with")
    model = TinyGPT(ModelConfig(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["model"])
    return model, a, splits, Vocab(a.p), dev, amp


def step_values(exprs: gen.Batch) -> np.ndarray:
    """(N, T) true step values, vectorised per tree shape."""
    c = exprs.cell
    out = np.empty((len(exprs), c.T), dtype=np.int64)
    for r in np.unique(exprs.rank):
        m = exprs.rank == r
        out[m] = gen.eval_batch(gen.skeleton(c.shape, c.T, int(r)), exprs.ops[m], exprs.leaves[m], c.p)
    return out


def structure(exprs: gen.Batch) -> tuple:
    """Per-expression consumer (-1 for the root) and depth of every step, as (N, T) arrays."""
    c = exprs.cell
    cons = np.empty((len(exprs), c.T), dtype=np.int64)
    depth = np.empty((len(exprs), c.T), dtype=np.int64)
    for r in np.unique(exprs.rank):
        s = gen.skeleton(c.shape, c.T, int(r))
        m = exprs.rank == r
        cons[m] = [-1 if x is None else x for x in s.consumer]
        depth[m] = s.depth
    return cons, depth


def batches(exprs: gen.Batch, vocab, trace, device, bs):
    """Yield (index array, model batch) over exprs; batch_idx is the index into exprs."""
    for i in range(0, len(exprs), bs):
        idx = np.arange(i, min(i + bs, len(exprs)))
        part = exprs.take(idx)
        encs = [encode(e, vocab, "latent", trace) for e in part.examples()]
        yield idx, make_batch(encs, vocab, idx=idx, device=device)


@torch.no_grad()
def predict(model, exprs: gen.Batch, vocab, trace, device, amp, bs=1024, hook=None) -> np.ndarray:
    """Greedy answers of a final-stage latent model, optionally under a hook."""
    model.eval()
    out = np.empty(len(exprs), dtype=np.int64)
    for idx, b in batches(exprs, vocab, trace, device, bs):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred, _ = greedy_answer(model, b, "latent", hook)
        out[idx] = pred.cpu().numpy()
    return out


@torch.no_grad()
def hooked_pass(model, exprs: gen.Batch, vocab, trace, device, amp, bs=1024, hook=None) -> tuple:
    """(answers (N,), latents (N, T, d)) under a hook. Latents are pre-hook: at a hooked slot they
    are what the network produced; every later slot is recomputed from the hooked input."""
    model.eval()
    lat = np.empty((len(exprs), exprs.cell.T, model.cfg.d_model), dtype=np.float32)
    pred = np.empty(len(exprs), dtype=np.int64)
    for idx, b in batches(exprs, vocab, trace, device, bs):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, z = forward_latent(model, b, hook=hook)
        lat[idx] = z.float().cpu().numpy()
        pred[idx] = answer_logits(logits, b).argmax(-1).cpu().numpy()
    return pred, lat


@torch.no_grad()
def capture(model, exprs: gen.Batch, vocab, trace, device, amp, bs=1024) -> dict:
    model.eval()
    lat = np.empty((len(exprs), exprs.cell.T, model.cfg.d_model), dtype=np.float32)
    pred = np.empty(len(exprs), dtype=np.int64)
    for idx, b in batches(exprs, vocab, trace, device, bs):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, z = forward_latent(model, b)
        lat[idx] = z.float().cpu().numpy()
        pred[idx] = answer_logits(logits, b).argmax(-1).cpu().numpy()
    cons, depth = structure(exprs)
    return dict(latents=lat, values=step_values(exprs), ops=exprs.ops, consumer=cons, depth=depth,
                answer=exprs.answer, pred=pred, rank=exprs.rank)


def get_capture(run_dir, split, model=None, a=None, splits=None, vocab=None, dev=None, amp=None, bs=1024) -> dict:
    """Load capture_<split>.npz from the run dir, computing and saving it first if missing."""
    f = Path(run_dir) / f"capture_{split}.npz"
    if f.exists():
        return dict(np.load(f))
    if model is None:
        model, a, splits, vocab, dev, amp = load_run(run_dir)
    cap = capture(model, getattr(splits, split), vocab, a.trace, dev, amp, bs)
    np.savez_compressed(f, **cap)
    return cap


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    for d in args.run_dirs:
        loaded = load_run(d, args.device)
        for sp in args.splits:
            cap = get_capture(d, sp, *loaded)
            acc = (cap["pred"] == cap["answer"]).mean()
            print(f"{d} {sp}: {len(cap['answer'])} expressions, latents {cap['latents'].shape}, clean acc {acc:.4f}")
