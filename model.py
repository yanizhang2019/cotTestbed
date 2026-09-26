"""Tiny GPT with Coconut-style latent feedback, and the single hook every intervention uses.

Plan: GPT-2-style decoder, d_model 256, 4 heads, 2-4 layers, context 256, learned positions.
Feedback configs:
  base  the fed-back vector is the final-layer hidden state after the last LayerNorm (ln_f).
  prj   the fed-back vector is Prj(h) = LayerNorm(W2 GeLU(W1 h)) applied to that hidden state.
  prj + round is the prj model with round_log applied at inference, i.e. prj plus a hook.
prj matches the theorem's feedback projection only. The backbone is a standard pre-LayerNorm
GPT-2 transformer, whereas the theorem's construction has no LayerNorm inside the blocks.

Latent recurrence (as in Coconut): slot t's input embedding is the fed-back vector computed from
the output at the position just before slot t. Learned position embeddings are added to it like
any token. One sequential forward pass per latent, recomputing the whole prefix each time, with
gradients through the whole recurrence.

A "latent" in this codebase is the fed-back vector (for prj, the Prj output), before any hook.
Every intervention (zero, mean, snap, noise, swap, round) is a hook
    hook(t, h, batch_idx) -> h'      h: (B, d) the latent for slot t; batch_idx: (B,) example ids
and forward_latent returns (logits, pre-hook latents of shape (B, n_latent, d)).

Batches: within a cell every sequence has the same length and the same slot positions (questions
are fully parenthesised, so length depends only on T), so a batch never needs padding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FEEDBACK = ("base", "prj")
Hook = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 256
    context: int = 256
    dropout: float = 0.0
    feedback: str = "base"  # "base" | "prj"
    d_prj: Optional[int] = None  # hidden width of Prj; default 4 * d_model
    tie_embeddings: bool = True  # GPT-2 ties the input embedding and the output head


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head, self.dropout = cfg.n_head, cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (z.view(B, L, self.n_head, D // self.n_head).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.mlp(self.ln2(x)))


class Prj(nn.Module):
    """Prj(h) = LayerNorm(W2 GeLU(W1 h)), the theorem's projection module."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1, self.w2, self.ln = nn.Linear(d, hidden), nn.Linear(hidden, d), nn.LayerNorm(d)

    def forward(self, h):
        return self.ln(self.w2(F.gelu(self.w1(h))))


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.feedback not in FEEDBACK:
            raise ValueError(f"feedback must be one of {FEEDBACK}; prj+round is prj plus a rounding hook")
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.context, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.prj = Prj(d, cfg.d_prj or 4 * d) if cfg.feedback == "prj" else None
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2: scale residual projections
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, ids, latents=None, latent_pos=None):
        """ids (B, L). latents (B, n, d) replace the token embeddings at latent_pos[:n].
        Returns (logits (B, L, V), hidden (B, L, d) after ln_f)."""
        B, L = ids.shape
        if L > self.cfg.context:
            raise ValueError(f"sequence length {L} exceeds context {self.cfg.context}")
        x = self.tok(ids)
        if latents is not None and latents.shape[1] > 0:
            x = x.index_copy(1, latent_pos[: latents.shape[1]], latents.to(x.dtype))
        x = self.drop(x + self.pos(torch.arange(L, device=ids.device)))
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)
        return self.head(h), h

    def feedback(self, h):
        """The vector fed back into the next latent slot."""
        return h if self.prj is None else self.prj(h)


# --------------------------------------------------------------------------- batches

def make_batch(encoded: list, vocab, idx=None, device="cpu") -> dict:
    """Stack gen.Encoded sequences of one cell and stage. They must share length and slot positions."""
    e0 = encoded[0]
    for e in encoded:
        if len(e.ids) != len(e0.ids) or e.latent_pos != e0.latent_pos or e.ans_pos != e0.ans_pos:
            raise ValueError("a batch must come from one cell, mode, trace and stage")
    lp = e0.latent_pos
    if lp and lp != list(range(lp[0], lp[0] + len(lp))):
        raise ValueError("latent slots must be contiguous")
    return dict(
        ids=torch.tensor([e.ids for e in encoded], dtype=torch.long, device=device),
        targets=torch.tensor([e.targets for e in encoded], dtype=torch.bool, device=device),
        latent_pos=torch.tensor(lp, dtype=torch.long, device=device),
        ans_pos=e0.ans_pos,
        gen_start=e0.targets.index(True),  # first token the model has to produce
        ans_id=vocab.ans,
        idx=(torch.arange(len(encoded), device=device) if idx is None
             else torch.as_tensor(idx, dtype=torch.long, device=device)),
    )


def lm_loss(logits, batch):
    """Next-token cross-entropy on target positions only (question and latent slots are masked)."""
    tgt = batch["targets"][:, 1:]
    return F.cross_entropy(logits[:, :-1][tgt].float(), batch["ids"][:, 1:][tgt])


def answer_logits(logits, batch):
    """Teacher-forced logits for the answer token (B, V)."""
    return logits[:, batch["ans_pos"] - 1]


# --------------------------------------------------------------------------- latent recurrence

def _compute_latents(model: TinyGPT, ids, latent_pos, hook: Optional[Hook], batch_idx):
    """Sequential passes. Returns (pre-hook latents, post-hook latents), each (B, n, d)."""
    n = len(latent_pos)
    B, d = ids.shape[0], model.cfg.d_model
    if n == 0:
        empty = torch.zeros(B, 0, d, device=ids.device)
        return empty, empty
    p0 = int(latent_pos[0])
    pre, fed = [], []
    for t in range(n):
        lat = torch.stack(fed, 1) if fed else None
        _, h = model(ids[:, : p0 + t], lat, latent_pos)
        v = model.feedback(h[:, -1])
        pre.append(v)
        fed.append(hook(t, v, batch_idx) if hook is not None else v)
    return torch.stack(pre, 1), torch.stack(fed, 1)


def forward_latent(model: TinyGPT, batch: dict, n_latent: Optional[int] = None, hook: Optional[Hook] = None,
                   return_fed: bool = False):
    """Latent-mode forward. Returns (logits (B, L, V), pre-hook latents (B, n_latent, d)), plus the
    post-hook latents actually fed back if return_fed. Under a perturbation, pre[:, t] is what the
    network produced at slot t and fed[:, t] is what it then read (they differ only where the hook acts)."""
    lp = batch["latent_pos"]
    if n_latent is not None and n_latent != len(lp):
        raise ValueError(f"batch has {len(lp)} latent slots, not {n_latent}")
    pre, fed = _compute_latents(model, batch["ids"], lp, hook, batch["idx"])
    logits, _ = model(batch["ids"], fed, lp)
    return (logits, pre, fed) if return_fed else (logits, pre)


def run(model: TinyGPT, batch: dict, mode: str, hook: Optional[Hook] = None):
    """Teacher-forced forward for any mode. Returns (logits, latents or None).
    pause slots are the LAT token's learned embedding with no feedback, so they take no hook."""
    if mode == "latent":
        return forward_latent(model, batch, hook=hook)
    if hook is not None:
        raise ValueError(f"mode {mode!r} has no latents to hook")
    return model(batch["ids"])[0], None


@torch.no_grad()
def greedy_answer(model: TinyGPT, batch: dict, mode: str, hook: Optional[Hook] = None):
    """Greedy decoding from the first generated token. Returns (pred (B,), generated (B, n)).

    direct, pause and full-stage latent runs generate one token, the answer (ANS is supplied).
    cot and partial-stage latent runs generate the remaining trace, then ANS, then the answer;
    pred is the token after the first generated ANS, or -1 if the model never emits ANS.
    """
    ids, lp, start, ans_pos = batch["ids"], batch["latent_pos"], batch["gen_start"], batch["ans_pos"]
    fed = None
    if mode == "latent":
        _, fed = _compute_latents(model, ids, lp, hook, batch["idx"])
    elif hook is not None:
        raise ValueError(f"mode {mode!r} has no latents to hook")
    seq = ids[:, :start]
    for _ in range(ans_pos - start + 1):
        logits, _ = model(seq, fed, lp if fed is not None else None)
        seq = torch.cat([seq, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    gen = seq[:, start:]
    if start == ans_pos:
        return gen[:, 0], gen
    is_ans = gen == batch["ans_id"]
    first = torch.where(is_ans.any(1), is_ans.float().argmax(1), torch.full_like(gen[:, 0], gen.shape[1]))
    nxt = first + 1
    pred = torch.where(nxt < gen.shape[1], gen.gather(1, nxt.clamp(max=gen.shape[1] - 1)[:, None])[:, 0],
                       torch.full_like(nxt, -1))
    return pred, gen


# --------------------------------------------------------------------------- common hooks

def zero_hook(t, h, batch_idx):
    return torch.zeros_like(h)


def identity_hook(t, h, batch_idx):
    return h


def round_log(h):
    """The theorem's rounding for the optional prj+round ablation. Definition still needed."""
    raise NotImplementedError("round_log: add the theorem's rounding rule before running prj+round")

