"""Model tests: the plan's hook tests, plus recurrence, causality, gradient and decoding checks."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen  # noqa: E402
from gen import Cell, Vocab, encode  # noqa: E402
from model import (ModelConfig, TinyGPT, answer_logits, forward_latent, greedy_answer,  # noqa: E402
                   identity_hook, lm_loss, make_batch, run, zero_hook)

torch.use_deterministic_algorithms(True)


def small_model(p, feedback="base", seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=len(Vocab(p)), n_layer=2, n_head=4, d_model=64, feedback=feedback)
    cfg.update(kw)
    return TinyGPT(ModelConfig(**cfg)).eval()


def batch_for(cell, n, mode, trace="compact", stage=None, seed=0):
    exs = gen.sample_uniform(cell, n, np.random.default_rng(seed)).examples()
    v = Vocab(cell.p)
    return make_batch([encode(e, v, mode, trace, stage) for e in exs], v), exs


CASES = [(fb, trace) for fb in ("base", "prj") for trace in ("compact", "full")]


# ---- the plan's two model tests ------------------------------------------------

@pytest.mark.parametrize("feedback,trace", CASES)
def test_identity_hook_reproduces_unhooked_logits_exactly(feedback, trace):
    m = small_model(5, feedback)
    b, _ = batch_for(Cell("balanced", 3, 5), 16, "latent", trace)
    with torch.no_grad():
        l0, z0 = forward_latent(m, b)
        l1, z1 = forward_latent(m, b, hook=identity_hook)
        l2, z2 = forward_latent(m, b, hook=lambda t, h, i: h.clone())
    assert torch.equal(l0, l1) and torch.equal(z0, z1)
    assert torch.equal(l0, l2) and torch.equal(z0, z2)


@pytest.mark.parametrize("feedback,trace", CASES)
def test_zero_hook_changes_outputs(feedback, trace):
    m = small_model(5, feedback)
    b, _ = batch_for(Cell("balanced", 3, 5), 16, "latent", trace)
    with torch.no_grad():
        l0, z0 = forward_latent(m, b)
        lz, zz = forward_latent(m, b, hook=zero_hook)
    assert not torch.allclose(answer_logits(l0, b), answer_logits(lz, b))
    assert torch.equal(z0[:, 0], zz[:, 0])  # the first latent is computed before any hook acts
    assert not torch.allclose(z0[:, 1:], zz[:, 1:])  # later latents see the zeroed ones


# ---- recurrence --------------------------------------------------------------------

@pytest.mark.parametrize("feedback,trace", CASES)
def test_recurrence_is_self_consistent(feedback, trace):
    """Each slot's pre-hook latent equals feedback(output just before that slot), given the
    post-hook latents in every earlier slot."""
    m = small_model(7, feedback)
    b, _ = batch_for(Cell("chain", 4, 7), 8, "latent", trace)
    half = lambda t, h, i: 0.5 * h  # noqa: E731
    with torch.no_grad():
        logits, pre = forward_latent(m, b, hook=half)
        fed = 0.5 * pre
        logits2, h = m(b["ids"], fed, b["latent_pos"])
        assert torch.equal(logits, logits2)
        for t, pos in enumerate(b["latent_pos"].tolist()):
            torch.testing.assert_close(m.feedback(h[:, pos - 1]), pre[:, t], atol=1e-5, rtol=1e-5)


def test_hook_receives_slot_index_and_batch_idx():
    m = small_model(5)
    exs = gen.sample_uniform(Cell("balanced", 3, 5), 6, np.random.default_rng(0)).examples()
    v = Vocab(5)
    b = make_batch([encode(e, v, "latent") for e in exs], v, idx=[10, 11, 12, 13, 14, 15])
    seen = []

    def hook(t, h, i):
        seen.append((t, h.shape, i.tolist()))
        return h

    with torch.no_grad():
        forward_latent(m, b, n_latent=3, hook=hook)
    assert seen == [(t, (6, 64), list(range(10, 16))) for t in range(3)]
    with pytest.raises(ValueError):
        forward_latent(m, b, n_latent=4)


def test_causality():
    m = small_model(5, "prj")
    cell = Cell("balanced", 3, 5)
    b, _ = batch_for(cell, 8, "latent", "compact", stage=1)  # 1 latent, then 2 written steps
    p0 = int(b["latent_pos"][0])
    with torch.no_grad():
        l0, z0 = forward_latent(m, b)
        lz, _ = forward_latent(m, b, hook=zero_hook)
        torch.testing.assert_close(l0[:, :p0], lz[:, :p0], atol=0, rtol=0)  # before the slot: unchanged
        b2 = dict(b, ids=b["ids"].clone())
        b2["ids"][:, p0 + 1:] = 0  # scramble everything after the slots
        _, z2 = forward_latent(m, b2)
    assert torch.equal(z0, z2)  # latents never see later tokens


@pytest.mark.parametrize("trace", ["compact", "full"])
def test_every_latent_sees_only_preceding_context(trace):
    """Final stage, all slots latent (4 compact / 33 full for chain T=4): scrambling everything
    after the slots leaves every latent and every pre-slot logit unchanged."""
    m = small_model(7, "prj")
    b, _ = batch_for(Cell("chain", 4, 7), 8, "latent", trace)
    last = int(b["latent_pos"][-1])
    b2 = dict(b, ids=b["ids"].clone())
    b2["ids"][:, last + 1:] = torch.randint(0, len(Vocab(7)), b2["ids"][:, last + 1:].shape)
    with torch.no_grad():
        l1, z1 = forward_latent(m, b)
        l2, z2 = forward_latent(m, b2)
    assert len(b["latent_pos"]) == (4 if trace == "compact" else 33)
    assert torch.equal(z1, z2)
    torch.testing.assert_close(l1[:, : last + 1], l2[:, : last + 1], atol=0, rtol=0)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_perturbing_one_latent_changes_only_later_latents(k):
    m = small_model(7, "base")
    b, _ = batch_for(Cell("chain", 4, 7), 8, "latent")
    noise = torch.randn(8, 64, generator=torch.Generator().manual_seed(k))

    def kick(t, h, i):
        return h + noise if t == k else h

    with torch.no_grad():
        _, z0, f0 = forward_latent(m, b, return_fed=True)
        _, z1, f1 = forward_latent(m, b, hook=kick, return_fed=True)
    assert torch.equal(z0[:, : k + 1], z1[:, : k + 1])  # up to and including slot k: as produced
    assert not torch.allclose(z0[:, k + 1:], z1[:, k + 1:]) or k == 3  # later slots move
    assert torch.equal(f1[:, k], z1[:, k] + noise)  # fed = post-hook at the perturbed slot
    others = [t for t in range(4) if t != k]
    assert torch.equal(f1[:, others], z1[:, others])  # elsewhere fed == produced
    assert torch.equal(f0, z0)


# ---- gradients -----------------------------------------------------------------------

def test_gradients_flow_through_the_recurrence():
    cell = Cell("balanced", 3, 5)
    b, _ = batch_for(cell, 16, "latent", "compact")  # final stage: loss on the answer only
    for detach, expect in ((False, True), (True, False)):
        m = small_model(5, "prj").train()
        hook = (lambda t, h, i: h.detach()) if detach else None
        logits, _ = forward_latent(m, b, hook=hook)
        lm_loss(logits, b).backward()
        g = m.prj.w1.weight.grad
        # Prj is used only to make latents, so its gradient is nonzero iff it flows through them
        assert (g is not None and g.abs().sum() > 0) == expect


def test_pause_has_no_feedback():
    m = small_model(5, "prj").train()
    b, _ = batch_for(Cell("balanced", 3, 5), 16, "pause")
    logits, lat = run(m, b, "pause")
    assert lat is None
    assert torch.equal(logits, m(b["ids"])[0])
    lm_loss(logits, b).backward()
    assert m.prj.w1.weight.grad is None
    lat_row = Vocab(5).lat
    assert m.tok.weight.grad[lat_row].abs().sum() > 0  # the filler embedding is learned
    with pytest.raises(ValueError):
        run(m, b, "pause", hook=zero_hook)


def test_loss_mask_covers_only_targets():
    m = small_model(5)
    b, _ = batch_for(Cell("balanced", 3, 5), 4, "cot")
    logits, _ = run(m, b, "cot")
    bumped = logits.clone()
    tgt = b["targets"][:, 1:]
    bumped[:, :-1][~tgt] += 100 * torch.randn_like(bumped[:, :-1][~tgt])
    assert torch.equal(lm_loss(logits, b), lm_loss(bumped, b))


# ---- configuration -------------------------------------------------------------------

def test_plan_sizes():
    v = len(Vocab(7))
    n = {L: TinyGPT(ModelConfig(vocab_size=v, n_layer=L)).n_params() for L in (2, 4)}
    assert 1.0e6 < n[2] < 2.0e6 and 3.0e6 < n[4] < 3.5e6, n
    prj = TinyGPT(ModelConfig(vocab_size=v, n_layer=2, feedback="prj")).n_params() - n[2]
    assert prj == 2 * 256 * 1024 + 1024 + 256 + 2 * 256
    with pytest.raises(ValueError):
        TinyGPT(ModelConfig(vocab_size=v, feedback="prj_round"))


def test_make_batch_rejects_mixed_shapes():
    v = Vocab(5)
    rng = np.random.default_rng(0)
    a = gen.random_example(5, "balanced", 3, rng)
    c = gen.random_example(5, "chain", 4, rng)
    with pytest.raises(ValueError):
        make_batch([encode(a, v, "latent"), encode(c, v, "latent")], v)
    with pytest.raises(ValueError):
        make_batch([encode(a, v, "latent", stage=1), encode(a, v, "latent", stage=2)], v)


def test_kill_test_sequences_fit_context():
    m = TinyGPT(ModelConfig(vocab_size=len(Vocab(7))))
    for c in gen.KILL_TEST_1_CELLS:
        b, _ = batch_for(c, 2, "cot")
        with torch.no_grad():
            m(b["ids"])


# ---- decoding (overfit a few examples, then decode greedily) -----------------------------

def overfit(mode, trace="compact", stage=None, feedback="base", target=0.02, max_steps=1500, n=16):
    """Train until loss < target (capped), so the test checks learnability, not convergence speed.
    Fully latent runs can need ~500 steps here, and the count varies across CPUs."""
    cell = Cell("balanced", 3, 5)
    b, exs = batch_for(cell, n, mode, trace, stage, seed=3)
    m = small_model(5, feedback, seed=1).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(max_steps):
        logits, _ = run(m, b, mode)
        loss = lm_loss(logits, b)
        if loss.item() < target:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval(), b, exs, float(loss)


@pytest.mark.parametrize("mode,stage", [("cot", None), ("latent", None), ("latent", 1), ("direct", None), ("pause", None)])
def test_greedy_decoding_after_overfitting(mode, stage):
    m, b, exs, loss = overfit(mode, stage=stage)
    assert loss < 0.02, loss
    pred, gen_toks = greedy_answer(m, b, mode)
    assert pred.tolist() == [e.answer for e in exs]
    v = Vocab(5)
    if mode == "cot" or stage is not None:  # the written part of the trace is regenerated too
        s = 0 if mode == "cot" else stage
        for e, row in zip(exs, gen_toks.tolist()):
            assert v.decode(row) == list(e.compact_trace[6 * s:]) + ["ANS", str(e.answer)]
    with torch.no_grad():  # nothing is generated before the answer here, so teacher forcing agrees
        if b["gen_start"] == b["ans_pos"]:
            logits, _ = run(m, b, mode)
            assert torch.equal(answer_logits(logits, b).argmax(-1), pred)


def test_greedy_reports_missing_ans():
    m = small_model(5)
    b, _ = batch_for(Cell("balanced", 3, 5), 4, "cot")
    b["ans_id"] = -123  # no generated token can match, as for a model that never emits ANS
    pred, _ = greedy_answer(m, b, "cot")
    assert pred.tolist() == [-1] * 4


def test_latent_scale_is_one_global_scalar():
    b, _ = batch_for(Cell("balanced", 3, 5), 8, "latent", "compact_rf")
    m1 = small_model(5, "base")
    m2 = small_model(5, "base", latent_scale=0.125)
    m2.load_state_dict(m1.state_dict())
    with torch.no_grad():
        _, h = m1(b["ids"])
        torch.testing.assert_close(m2.feedback(h), 0.125 * m1.feedback(h), atol=0, rtol=0)
        l0, z0 = forward_latent(m2, b)
        l1, z1 = forward_latent(m2, b, hook=identity_hook)
    assert torch.equal(l0, l1) and torch.equal(z0, z1)
    norms = z0.norm(dim=-1)
    assert norms.std() > 0  # per-latent magnitudes are kept, unlike per-latent normalization
