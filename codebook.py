"""Codebook tools (C5-C8). Built so far: the pieces the tape test (C3) and the shared-alphabet test
(C4) need. Per-slot linear probes (with a shuffled-label control for C8), per-step and global value
centroids, and nearest-centroid snapping hooks. Still to come: the C5 compression sweep (k-means
codebooks, random-codebook and matched-noise controls, T = 4 -> 8 and chain -> tree transfer),
vocabulary snapping, the C6 self-correction sweep, and C8 purity / NMI.

Every object here is fit on validation latents (capture_val.npz) only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- probes

def fit_probes(cap_val: dict, p: int, epochs: int = 300, lr: float = 0.05, wd: float = 1e-4,
               seed: int = 0, shuffle: bool = False) -> list:
    """One multinomial logistic-regression probe per slot: latent at slot t -> v_t.
    Inputs are standardized per slot with validation statistics. shuffle=True permutes the labels
    (the C8 control: a probe that can only memorize should stay near chance on test)."""
    lat = torch.as_tensor(cap_val["latents"], dtype=torch.float32)
    vals = torch.as_tensor(cap_val["values"], dtype=torch.long)
    n, T, d = lat.shape
    gen = torch.Generator().manual_seed(seed)
    probes = []
    for t in range(T):
        mu, sd = lat[:, t].mean(0), lat[:, t].std(0) + 1e-6
        x = (lat[:, t] - mu) / sd
        y = vals[:, t][torch.randperm(n, generator=gen)] if shuffle else vals[:, t]
        torch.manual_seed(seed * 1000 + t)
        lin = nn.Linear(d, p)
        opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            loss = F.cross_entropy(lin(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        probes.append(dict(W=lin.weight.detach(), b=lin.bias.detach(), mu=mu, sd=sd))
    return probes


def probe_predict(probes: list, lat, t: int) -> np.ndarray:
    """Decoded value of the latent at slot t, for every row of lat (N, T, d)."""
    x = torch.as_tensor(np.asarray(lat)[:, t], dtype=torch.float32)
    pr = probes[t]
    return (((x - pr["mu"]) / pr["sd"]) @ pr["W"].T + pr["b"]).argmax(-1).numpy()


def probe_accuracy(probes: list, cap: dict) -> list:
    """Per-slot probe accuracy on a capture (use the test capture to score)."""
    return [float((probe_predict(probes, cap["latents"], t) == cap["values"][:, t]).mean())
            for t in range(cap["latents"].shape[1])]


# --------------------------------------------------------------------------- centroids and snapping

def step_centroids(cap_val: dict, p: int) -> np.ndarray:
    """(T, p, d) mean validation latent of slot t over expressions with v_t = v (NaN if none)."""
    lat, vals = cap_val["latents"], cap_val["values"]
    T, d = lat.shape[1], lat.shape[2]
    cent = np.full((T, p, d), np.nan, dtype=np.float32)
    for t in range(T):
        for v in range(p):
            m = vals[:, t] == v
            if m.any():
                cent[t, v] = lat[m, t].mean(0)
    return cent


def global_centroids(cap_val: dict, p: int) -> np.ndarray:
    """(p, d) one value codebook for all slots: mean validation latent over every (expression, slot)
    whose value is v."""
    lat, vals = cap_val["latents"], cap_val["values"]
    flat, fv = lat.reshape(-1, lat.shape[2]), vals.reshape(-1)
    out = np.full((p, lat.shape[2]), np.nan, dtype=np.float32)
    for v in range(p):
        if (fv == v).any():
            out[v] = flat[fv == v].mean(0)
    return out


def _clean_rows(c: np.ndarray) -> torch.Tensor:
    c = np.asarray(c, dtype=np.float32)
    return torch.as_tensor(c[~np.isnan(c).any(-1)])


def snap_hook(codebooks, device="cpu"):
    """Replace every latent by its nearest codebook entry (Euclidean). codebooks is one (k, d) array
    shared by all slots, or a list / (T, k, d) array with one codebook per slot."""
    per_slot = isinstance(codebooks, (list, tuple)) or np.asarray(codebooks).ndim == 3
    books = [_clean_rows(c).to(device) for c in codebooks] if per_slot else _clean_rows(codebooks).to(device)

    def hook(t, h, batch_idx):
        c = books[t] if per_slot else books
        return c[torch.cdist(h.float(), c).argmin(1)].to(h.dtype)
    return hook
