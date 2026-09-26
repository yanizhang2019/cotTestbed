"""Trainer for the four modes (direct | cot | latent | pause) and both traces.

  python train.py --shape balanced --T 3 --p 5 --mode cot
  python train.py --shape chain --T 8 --p 7 --mode latent --feedback prj --seed 1

Schedule
  * An "epoch" is --epoch_size training draws (default 100k). Most cells are sampled and have
    no natural epoch, so every cell uses the same definition.
  * direct / cot: --epochs epochs at one stage.
  * latent: curriculum stages s = 0..T (stage s replaces the first s trace steps with latents;
    stage 0 is plain CoT). Each stage s < T runs --epochs_per_stage epochs (plan: 3), or advances
    early once validation accuracy has stopped improving (--plateau_patience epochs without a
    --plateau_delta gain); --fixed_stages turns early advancing off (kill test 2 uses it). The
    final stage runs --final_epochs. The optimizer is re-created at
    every stage switch (Coconut), and each stage gets its own warmup + cosine schedule.
  * pause: the final-stage slot count with no feedback, loss on the answer, for as many epochs as
    the latent curriculum uses in total (unless --epochs is given), so the two see equal training.
  * AdamW, peak lr 1e-3, 2% warmup, cosine to 10% of peak, weight decay 0.1 on matrices, batch
    256, bf16 autocast on GPU. Greedy decoding for every accuracy.

Validation versus test
  Training, model selection and the kill-test gates use validation only; train() never scores
  test. Once a configuration is frozen, score it with
      python train.py score-test <run dir> [...]      (or train with --score_test)
  which writes test_results.json and test_correct*.npy, including zero- and mean-latent test
  accuracy for latent runs (mean fit on validation latents).

Outputs
  run dir = <out>/<cell>/<mode>-<trace>[-<feedback>]-L<layers>-D<d_model>-lr<lr>-s<seed><tag>/
    log.jsonl         train loss every --log_every steps, one eval line per epoch
    best.pt, last.pt  best = highest validation accuracy in the final stage
    results.json      full-validation accuracy of best.pt (+ zero-latent validation accuracy for latent)
    val_correct*.npy  per-problem correctness on validation (paired McNemar tests)
  A run dir that already holds a finished run with a different configuration is refused, so
  --skip_done can never return another configuration's result.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import gen
from gen import MODES, SHAPES, TRACES, Cell, Vocab, encode, make_splits
from model import FEEDBACK, ModelConfig, TinyGPT, forward_latent, greedy_answer, lm_loss, make_batch, run


# --------------------------------------------------------------------------- args and plan

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train one run of the finite-codebook testbed.")
    a = ap.add_argument
    a("--shape", choices=SHAPES, required=True)
    a("--T", type=int, required=True)
    a("--p", type=int, required=True)
    a("--mode", choices=MODES, required=True)
    a("--trace", choices=TRACES, default="compact")
    a("--feedback", choices=FEEDBACK, default="base")
    a("--latent_scale", default="1", help="global scalar on fed-back latents: a number, or 'invsqrt_d' for 1/sqrt(d_model)")
    a("--layers", type=int, default=2)
    a("--d_model", type=int, default=256)
    a("--heads", type=int, default=4)
    a("--lr", type=float, default=1e-3)
    a("--min_lr_frac", type=float, default=0.1)
    a("--warmup", type=float, default=0.02)
    a("--wd", type=float, default=0.1)
    a("--batch", type=int, default=256)
    a("--epoch_size", type=int, default=100_000)
    a("--epochs", type=int, default=None, help="direct/cot (default 30); pause (default: latent total)")
    a("--epochs_per_stage", type=int, default=3)
    a("--final_epochs", type=int, default=6)
    a("--fixed_stages", action="store_true", help="every intermediate stage runs all its epochs (no early advance)")
    a("--plateau_patience", type=int, default=1)
    a("--plateau_delta", type=float, default=0.005)
    a("--seed", type=int, default=0)
    a("--split_seed", type=int, default=0)
    a("--eval_n", type=int, default=2000, help="validation examples scored each epoch")
    a("--eval_batch", type=int, default=1024)
    a("--log_every", type=int, default=100)
    a("--workers", type=int, default=4)
    a("--device", default="auto")
    a("--no_bf16", action="store_true")
    a("--out", default="runs")
    a("--tag", default="")
    a("--skip_done", action="store_true", help="exit if this configuration already finished")
    a("--score_test", action="store_true", help="also score test (frozen configurations only)")
    return ap.parse_args(argv)


def stage_plan(args) -> list:
    """[(stage or None, epochs, may_advance_early)]"""
    latent_total = args.epochs_per_stage * args.T + args.final_epochs
    if args.mode in ("direct", "cot"):
        return [(None, args.epochs or 30, False)]
    if args.mode == "pause":
        return [(None, args.epochs or latent_total, False)]
    early = not args.fixed_stages
    return [(s, args.epochs_per_stage, early) for s in range(args.T)] + [(args.T, args.final_epochs, False)]


def latent_scale(args) -> float:
    return 1 / math.sqrt(args.d_model) if str(args.latent_scale) == "invsqrt_d" else float(args.latent_scale)


def run_dir(args) -> Path:
    cell = Cell(args.shape, args.T, args.p)
    fb = f"-{args.feedback}" if args.mode == "latent" else ""
    if args.mode == "latent" and latent_scale(args) != 1.0:
        fb += "-a" + ("isd" if str(args.latent_scale) == "invsqrt_d" else f"{latent_scale(args):g}")
    return (Path(args.out) / cell.name /
            f"{args.mode}-{args.trace}{fb}-L{args.layers}-D{args.d_model}-lr{args.lr:g}-s{args.seed}{args.tag}")


# Arguments that do not change what is trained (where it runs, how it is logged or reported).
NON_CONFIG = {"skip_done", "out", "workers", "device", "eval_batch", "log_every", "score_test"}


def config_diff(old: dict, new: dict) -> dict:
    """{arg: (stored, current)} for every training-relevant argument that differs."""
    keys = (set(old) | set(new)) - NON_CONFIG
    return {k: (old.get(k), new.get(k)) for k in sorted(keys) if old.get(k) != new.get(k)}


def lr_at(step: int, total: int, peak: float, warmup_frac: float, min_frac: float) -> float:
    warm = max(1, int(round(warmup_frac * total)))
    if step < warm:
        return peak * (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def make_optimizer(model, args):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))


# --------------------------------------------------------------------------- data

class TrainStream(IterableDataset):
    """Endless batches of answer-uniform training expressions, encoded for one mode and stage.
    Each worker has its own rng from (seed, stage, worker). Held-out expressions never appear."""

    def __init__(self, splits, vocab, mode, trace, stage, batch, seed):
        self.splits, self.vocab, self.mode, self.trace = splits, vocab, mode, trace
        self.stage, self.batch, self.seed = stage, batch, seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        stage = -1 if self.stage is None else self.stage
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, stage + 1, wid, 7919]))
        while True:
            b = self.splits.sample_train(self.batch, rng)
            yield make_batch([encode(e, self.vocab, self.mode, self.trace, self.stage) for e in b.examples()],
                             self.vocab)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


# --------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, exprs: gen.Batch, vocab, mode, trace, stage, device, bs, amp, hook=None) -> np.ndarray:
    """Greedy-decoding correctness per example. The hook's batch_idx is the example's index in exprs."""
    model.eval()
    out = []
    for i in range(0, len(exprs), bs):
        part = exprs.take(np.arange(i, min(i + bs, len(exprs))))
        encs = [encode(e, vocab, mode, trace, stage) for e in part.examples()]
        b = make_batch(encs, vocab, idx=np.arange(i, i + len(part)), device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred, _ = greedy_answer(model, b, mode, hook)
        out.append(pred.cpu().numpy() == part.answer)
    return np.concatenate(out)


@torch.no_grad()
def mean_latents(model, exprs: gen.Batch, vocab, trace, device, bs, amp) -> torch.Tensor:
    """Per-slot mean of clean validation latents (the plan's mean-replacement control)."""
    model.eval()
    total, n = None, 0
    for i in range(0, len(exprs), bs):
        part = exprs.take(np.arange(i, min(i + bs, len(exprs))))
        b = make_batch([encode(e, vocab, "latent", trace) for e in part.examples()], vocab, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            _, lat = forward_latent(model, b)
        s = lat.float().sum(0)
        total = s if total is None else total + s
        n += len(part)
    return total / n


# --------------------------------------------------------------------------- training

def train(args) -> dict:
    out = run_dir(args)
    if (out / "results.json").exists():
        old = json.loads((out / "results.json").read_text())
        diff = config_diff(old["args"], vars(args))
        if diff:  # never skip to, or overwrite, a different configuration's result
            raise SystemExit(f"{out} already holds a run with a different configuration {diff}; "
                             "use --tag or another --out")
        if args.skip_done:
            print(f"skip {out} (done)")
            return old
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    amp = device.type == "cuda" and not args.no_bf16
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cell = Cell(args.shape, args.T, args.p)
    splits = make_splits(cell, args.split_seed)
    vocab = Vocab(args.p)
    if gen.seq_len(args.T, args.mode, args.trace) > 256:
        raise SystemExit(f"{cell.name} {args.mode}/{args.trace} is longer than the context (256)")
    cfg = ModelConfig(vocab_size=len(vocab), n_layer=args.layers, n_head=args.heads, d_model=args.d_model,
                      feedback=args.feedback if args.mode == "latent" else "base",
                      latent_scale=latent_scale(args) if args.mode == "latent" else 1.0)
    model = TinyGPT(cfg).to(device)
    val_sub = splits.val.balanced_head(args.eval_n)  # exactly answer-balanced, so chance is 1/p
    steps_per_epoch = math.ceil(args.epoch_size / args.batch)
    plan = stage_plan(args)
    print(f"{out} | {model.n_params():,} params | {device} | stages {[(s, e) for s, e, _ in plan]}")

    log = open(out / "log.jsonl", "w")
    buf: list = []
    best = dict(acc=-1.0, stage=None, epoch=None)
    final_curve: list = []
    stages_run: list = []
    step, t0 = 0, time.time()

    for si, (stage, epochs, may_advance) in enumerate(plan):
        is_final = si == len(plan) - 1
        opt = make_optimizer(model, args)  # reset at every stage switch
        total = epochs * steps_per_epoch
        stream = TrainStream(splits, vocab, args.mode, args.trace, stage, args.batch, args.seed)
        loader = DataLoader(stream, batch_size=None, num_workers=args.workers,
                            pin_memory=device.type == "cuda", prefetch_factor=4 if args.workers else None)
        it = iter(loader)
        stage_best, since_best, run_loss, run_n = -1.0, 0, 0.0, 0
        for ep in range(epochs):
            model.train()
            for k in range(steps_per_epoch):
                s_local = ep * steps_per_epoch + k
                for g in opt.param_groups:
                    g["lr"] = lr_at(s_local, total, args.lr, args.warmup, args.min_lr_frac)
                b = to_device(next(it), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits, _ = run(model, b, args.mode)
                    loss = lm_loss(logits, b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                run_loss += loss.item()
                run_n += 1
                if step % args.log_every == 0:
                    buf.append(dict(type="train", step=step, stage=stage, loss=run_loss / run_n,
                                    lr=opt.param_groups[0]["lr"]))
                    run_loss, run_n = 0.0, 0
            acc = float(evaluate(model, val_sub, vocab, args.mode, args.trace, stage, device,
                                 args.eval_batch, amp).mean())
            rec = dict(type="eval", step=step, stage=stage, epoch=ep, val_acc=acc,
                       minutes=(time.time() - t0) / 60)
            buf.append(rec)
            for line in buf:
                log.write(json.dumps(line) + "\n")
            log.flush()
            buf = []
            print(f"  stage {stage} epoch {ep}  val {acc:.4f}  step {step}  {rec['minutes']:.1f} min", flush=True)
            if is_final:
                final_curve.append(acc)
                if acc > best["acc"]:
                    best = dict(acc=acc, stage=stage, epoch=ep)
                    torch.save(dict(model=model.state_dict(), cfg=vars(cfg), args=vars(args)), out / "best.pt")
            if acc > stage_best + args.plateau_delta:
                stage_best, since_best = acc, 0
            else:
                since_best += 1
            if may_advance and since_best >= args.plateau_patience and ep + 1 < epochs:
                print(f"  stage {stage} plateaued after {ep + 1} epochs; advancing")
                break
        stages_run.append(dict(stage=stage, epochs=ep + 1, last_val=acc))
        del it, loader
    torch.save(dict(model=model.state_dict(), cfg=vars(cfg), args=vars(args)), out / "last.pt")
    log.close()

    # final evaluation of the best checkpoint on the full validation set; test is never touched here
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    stage = plan[-1][0]
    val_ok = evaluate(model, splits.val, vocab, args.mode, args.trace, stage, device, args.eval_batch, amp)
    np.save(out / "val_correct.npy", val_ok)
    res = dict(cell=cell.name, shape=args.shape, T=args.T, p=args.p, mode=args.mode, trace=args.trace,
               feedback=cfg.feedback, layers=args.layers, d_model=args.d_model, lr=args.lr, seed=args.seed,
               split_seed=args.split_seed, chance=1 / args.p, n_params=model.n_params(), steps=step,
               epochs_total=sum(s["epochs"] for s in stages_run), minutes=(time.time() - t0) / 60,
               stages=stages_run, best=best, val_acc=float(val_ok.mean()), n_val=len(val_ok),
               still_improving=bool(len(final_curve) >= 5 and
                                    final_curve[-1] - max(final_curve[: int(0.8 * len(final_curve))]) > 0.01),
               split_fingerprint=splits.fingerprint(), args=dict(vars(args)))
    if args.mode == "latent":  # kill test 2's necessity check, on validation (zeroing has nothing to fit)
        zero_ok = evaluate(model, splits.val, vocab, "latent", args.trace, stage, device, args.eval_batch, amp,
                           hook=lambda t, h, i: torch.zeros_like(h))
        np.save(out / "val_correct_zero.npy", zero_ok)
        res.update(zero_val_acc=float(zero_ok.mean()))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"done {out}: val {res['val_acc']:.4f}"
          + (f" zero-latent val {res['zero_val_acc']:.4f}" if args.mode == "latent" else ""))
    if args.score_test:
        score_test(out, device=args.device)
    return res


# --------------------------------------------------------------------------- test scoring (frozen configs only)

def score_test(out, device="auto", eval_batch=1024) -> dict:
    """Score a finished run's best checkpoint on test. Run this only once a configuration is frozen:
    the kill tests and all selection use validation. Latent runs also get zero- and mean-latent test
    accuracy, with the mean fit on validation latents."""
    out = Path(out)
    res = json.loads((out / "results.json").read_text())
    ck = torch.load(out / "best.pt", map_location="cpu")
    a = argparse.Namespace(**ck["args"])
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    amp = dev.type == "cuda" and not a.no_bf16
    splits = make_splits(Cell(a.shape, a.T, a.p), a.split_seed)
    if splits.fingerprint() != res["split_fingerprint"]:
        raise RuntimeError(f"{out}: splits regenerated now differ from the ones this run was trained with")
    vocab = Vocab(a.p)
    model = TinyGPT(ModelConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["model"])
    stage = stage_plan(a)[-1][0]
    ok = evaluate(model, splits.test, vocab, a.mode, a.trace, stage, dev, eval_batch, amp)
    np.save(out / "test_correct.npy", ok)
    tr = dict(test_acc=float(ok.mean()), n_test=len(ok), scored=time.strftime("%Y-%m-%d %H:%M"))
    if a.mode == "latent":
        mu = mean_latents(model, splits.val, vocab, a.trace, dev, eval_batch, amp).to(dev)
        for name, hook in (("zero", lambda t, h, i: torch.zeros_like(h)),
                           ("mean", lambda t, h, i: mu[t].to(h.dtype).expand_as(h))):
            k = evaluate(model, splits.test, vocab, "latent", a.trace, stage, dev, eval_batch, amp, hook=hook)
            np.save(out / f"test_correct_{name}.npy", k)
            tr[f"{name}_test_acc"] = float(k.mean())
    (out / "test_results.json").write_text(json.dumps(tr, indent=2))
    print(f"test {out}: " + "  ".join(f"{k} {v:.4f}" for k, v in tr.items() if k.endswith("acc")))
    return tr


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "score-test":
        if len(sys.argv) < 3:
            raise SystemExit("usage: python train.py score-test <run dir> [<run dir> ...]")
        for d in sys.argv[2:]:
            score_test(d)
    else:
        train(parse_args())
