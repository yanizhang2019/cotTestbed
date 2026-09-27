"""Capture and C2 swap tests on a tiny latent run (untrained is fine: these check the machinery)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture  # noqa: E402
import stats  # noqa: E402
import swaps  # noqa: E402
import train  # noqa: E402
from gen import evaluate  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    a = train.parse_args(["--shape", "chain", "--T", "3", "--p", "5", "--mode", "latent", "--trace", "compact_rf",
                          "--latent_scale", "invsqrt_d", "--d_model", "32", "--epoch_size", "256", "--batch", "64",
                          "--epochs_per_stage", "1", "--final_epochs", "1", "--eval_n", "100", "--workers", "0",
                          "--device", "cpu", "--out", str(out)])
    train.train(a)
    return train.run_dir(a)


def test_capture_matches_labels_and_forward(run):
    model, a, splits, vocab, dev, amp = capture.load_run(run, "cpu")
    cap = capture.get_capture(run, "val", model, a, splits, vocab, dev, amp, bs=256)
    n, T = len(splits.val), a.T
    assert cap["latents"].shape == (n, T, 32) and cap["values"].shape == (n, T)
    exs = splits.val.take(np.arange(50)).examples()
    assert [tuple(r) for r in cap["values"][:50]] == [e.step_values for e in exs]
    assert (cap["consumer"][0] == [1, 2, -1]).all() and (cap["depth"][0] == [2, 1, 0]).all()
    assert (Path(run) / "capture_val.npz").exists()
    again = capture.get_capture(run, "val")  # loaded from disk, identical
    assert np.array_equal(again["latents"], cap["latents"])
    # the captured vector is exactly what is fed: replaying it through a hook changes nothing
    for t in range(T):
        r = torch.as_tensor(cap["latents"][:, t])
        hooked = capture.predict(model, splits.val, vocab, a.trace, dev, amp, 256, hook=swaps.slot_hook(t, r))
        assert np.array_equal(hooked, capture.predict(model, splits.val, vocab, a.trace, dev, amp, 256))


def test_fit_uses_validation_means_and_pools():
    rng = np.random.default_rng(0)
    vals = rng.integers(5, size=(200, 3))
    lat = rng.normal(size=(200, 3, 8)).astype(np.float32)
    f = swaps.fit(dict(latents=lat, values=vals), 5)
    for t in range(3):
        for v in range(5):
            m = vals[:, t] == v
            np.testing.assert_allclose(f["centroids"][t, v], lat[m, t].mean(0), rtol=1e-6)
            assert set(f["pools"][t, v]) == set(np.flatnonzero(m))
    d = swaps.draw_donors(f, 1, np.array([2, 2, 4]), rng)
    for row, v in zip(d, (2, 2, 4)):
        assert any(np.array_equal(row, lat[i, 1]) for i in f["pools"][1, v])


def test_run_c2_outputs_and_counterfactuals(run):
    with pytest.raises(RuntimeError, match="nothing to swap"):  # an untrained model answers nothing
        swaps.run_c2(run, "test", seed=0, device="cpu", bs=512)
    s = swaps.run_c2(run, "test", seed=0, device="cpu", bs=512, only_correct=False)
    assert s["n_scored"] == len(capture.load_run(run, "cpu")[2].test)
    npz = np.load(Path(run) / "c2_test.npz")
    stored = json.loads((Path(run) / "c2_test.json").read_text())
    assert stored["n_swaps_differ"] == s["n_swaps_differ"] and stored["pass"] == s["pass"]
    model, a, splits, *_ = capture.load_run(run, "cpu")
    exprs = splits.test.take(npz["index"])
    true = capture.step_values(exprs)
    assert (npz["alt"] != true).all()  # every swap asks for a different value
    exs = exprs.examples()
    for i in range(0, len(exs), max(1, len(exs) // 40)):
        for t in range(a.T):
            assert npz["cf"][i, t] == evaluate(exs[i], {t: int(npz["alt"][i, t])})[1]
    differs = npz["cf"] != npz["answer"][:, None]
    assert s["n_swaps_differ"] == int(differs.sum())
    hit = npz["pred_centroid"] == npz["cf"]
    if differs.any():
        assert s["centroid"]["cf_rate"] == pytest.approx(hit[differs].mean())
    assert s["same_donor"]["keep_rate"] == pytest.approx((npz["pred_same_donor"] == npz["answer"][:, None]).mean())
    assert (Path(run) / "capture_val.npz").exists() and not (Path(run) / "capture_test.npz").exists()


def test_stats_c2_aggregates_three_seeds(tmp_path, capsys):
    def fake(seed, cf, same):
        d = tmp_path / f"s{seed}"
        d.mkdir()
        (d / "results.json").write_text(json.dumps(dict(p=7, shape="chain", T=4, layers=4, trace="compact_rf",
                                                        lr=3e-4, seed=seed)))
        (d / "c2_test.json").write_text(json.dumps(dict(centroid=dict(cf_rate=cf), donor=dict(cf_rate=cf),
                                                        same_donor=dict(keep_rate=same), clean_acc=1.0)))
    fake(0, 0.97, 0.99)
    fake(1, 0.95, 0.98)
    assert list(stats.c2(tmp_path).values()) == [False]  # two seeds only
    fake(2, 0.50, 0.99)  # one weak seed pulls the mean to 0.807: still passes 0.80
    assert list(stats.c2(tmp_path).values()) == [True]
    assert "C2 PASS" in capsys.readouterr().out


# ---- C3 / C4 machinery -----------------------------------------------------------

import codebook  # noqa: E402
import gen  # noqa: E402


def test_dependents_mask_balanced_tree():
    cell = gen.Cell("balanced", 7, 5)
    b = gen.sample_uniform(cell, 4, np.random.default_rng(0))
    dep = swaps.dependents_mask(b)
    want = {0: {2, 6}, 1: {2, 6}, 2: {6}, 3: {5, 6}, 4: {5, 6}, 5: {6}, 6: set()}
    for t, ss in want.items():
        assert set(np.flatnonzero(dep[0, t])) == ss
    assert (dep == dep[:1]).all()


def test_probes_decode_separable_codes_and_shuffled_probe_does_not():
    rng = np.random.default_rng(0)
    p, d, T = 5, 16, 2
    means = rng.normal(size=(T, p, d)) * 3

    def make(n):
        v = rng.integers(p, size=(n, T))
        lat = np.stack([means[t][v[:, t]] for t in range(T)], 1) + rng.normal(size=(n, T, d)) * 0.3
        return dict(latents=lat.astype(np.float32), values=v)
    val, test = make(2000), make(2000)
    acc = codebook.probe_accuracy(codebook.fit_probes(val, p), test)
    assert min(acc) > 0.99
    shuf = codebook.probe_accuracy(codebook.fit_probes(val, p, shuffle=True), test)
    assert max(shuf) < 1 / p + 0.08


def test_centroids_and_snapping():
    rng = np.random.default_rng(1)
    vals = rng.integers(3, size=(300, 2))
    lat = rng.normal(size=(300, 2, 4)).astype(np.float32)
    cap = dict(latents=lat, values=vals)
    g = codebook.global_centroids(cap, 3)
    for v in range(3):
        np.testing.assert_allclose(g[v], lat.reshape(-1, 4)[vals.reshape(-1) == v].mean(0), rtol=1e-5)
    per = codebook.step_centroids(cap, 3)
    h = torch.as_tensor(per[1] + 0.01)  # near slot-1 centroids
    out = codebook.snap_hook(per)(1, h, None)
    assert torch.allclose(out, torch.as_tensor(per[1]))
    out_g = codebook.snap_hook(g)(0, torch.as_tensor(g + 0.01), None)
    assert torch.allclose(out_g, torch.as_tensor(g))


def test_run_c3_c4_outputs(run):
    ctx = swaps.setup(run, "test", "cpu", 512, only_correct=False)
    s3 = swaps.run_c3(ctx=ctx)
    assert json.loads((Path(run) / "c3_test.json").read_text())["n_follow"] == s3["n_follow"]
    assert len(s3["probe_acc"]) == 3 and s3["n_keep"] == 0  # a chain has no independent later steps
    assert s3["pass"]["keep"] is True and s3["keep_rate"] != s3["keep_rate"]  # NaN, passes as n/a
    s4 = swaps.run_c4(ctx=ctx)
    for k in ("same_step_rate", "cross_step_rate", "global_rate", "snap_step_retention", "snap_global_retention"):
        assert k in s4
    assert s4["cross_minus_same"] == pytest.approx(s4["cross_step_rate"] - s4["same_step_rate"], nan_ok=True)
    assert (Path(run) / "c4_test.json").exists()
    # pairwise matrices and raw outcomes
    m = np.array(s4["cross_matrix"], dtype=float)
    assert m.shape == (3, 3)
    for t in range(3):
        a, b = m[t, t], s4["same_step_per_step"][t]
        assert (a != a and b != b) or a == pytest.approx(b)  # diagonal = same-step swap
    z = np.load(Path(run) / "c4_test.npz")
    assert z["pred_matrix"].shape == (3, 3, ctx.n) and z["pred_global"].shape == (3, ctx.n)
    differs = z["cf"] != z["answer"][:, None]
    for t in range(3):
        if differs[:, t].any():
            want = (z["pred_matrix"][t, t] == z["cf"][:, t])[differs[:, t]].mean()
            assert m[t, t] == pytest.approx(want)
    z3 = np.load(Path(run) / "c3_test.npz")
    assert z3["decoded"].shape == (3, ctx.n, 3) and (z3["decoded"][2] == -1).all()  # nothing after the root
    tape = np.array(s3["matrix"]["follow_all"], dtype=float)
    assert tape.shape == (3, 3) and np.isnan(tape[np.tril_indices(3)]).all()  # only s > t is scored


def test_stats_c3_c4_readouts(tmp_path):
    def fake(seed, claim, **vals):
        d = tmp_path / f"{claim}{seed}"
        d.mkdir(exist_ok=True)
        (d / "results.json").write_text(json.dumps(dict(p=7, shape="balanced", T=7, layers=4, trace="compact_rf",
                                                        lr=3e-4, seed=seed)))
        (d / f"{claim}_test.json").write_text(json.dumps(vals))
    for s in range(3):
        fake(s, "c3", keep_rate=0.97, follow_rate=0.9, follow_all_rate=0.95, probe_acc_mean=0.99,
             probe_shuffled_acc_mean=0.15, clean_acc=1.0)
        fake(s, "c4", same_step_rate=0.9, cross_step_rate=0.85, global_rate=0.85, cross_minus_same=-0.05,
             snap_step_retention=1.0, snap_global_retention=0.99, global_minus_step_snap=-0.01, clean_acc=1.0)
    assert list(stats.c3(tmp_path).values()) == [True]
    assert list(stats.c4(tmp_path).values()) == [True]
    fake(0, "c4", same_step_rate=0.9, cross_step_rate=0.5, global_rate=0.5, cross_minus_same=-0.4,
         snap_step_retention=1.0, snap_global_retention=0.99, global_minus_step_snap=-0.01, clean_acc=1.0)
    assert list(stats.c4(tmp_path).values()) == [False]  # mean gap -0.15 is more than 0.10 below
    fake(0, "c4", same_step_rate=0.6, cross_step_rate=0.95, global_rate=0.9, cross_minus_same=0.35,
         snap_step_retention=0.9, snap_global_retention=1.0, global_minus_step_snap=0.1, clean_acc=1.0)
    assert list(stats.c4(tmp_path).values()) == [True]  # one-sided: doing better is not a failure


def test_mean_ci_bounds_never_exclude_the_mean(tmp_path, capsys):
    gap = [-0.02, -0.05, -0.08]
    m, lo, hi = stats.mean_ci(gap, bounds=(-1.0, 1.0))
    assert m == pytest.approx(-0.05) and lo < m < hi and lo < 0
    m, lo, hi = stats.mean_ci([0.97, 1.0, 1.02], bounds=(0.0, None))  # retention may exceed 1
    assert lo < m < hi and hi > 1.0
    m, lo, hi = stats.mean_ci([0.99, 1.0, 1.0])  # rates stay clipped to [0, 1]
    assert hi == 1.0 and lo <= m
    for s in range(3):  # the readout prints a negative interval for a negative gap
        d = tmp_path / f"s{s}"
        d.mkdir()
        (d / "results.json").write_text(json.dumps(dict(p=7, shape="chain", T=4, layers=4, trace="compact_rf",
                                                        lr=3e-4, seed=s)))
        (d / "c4_test.json").write_text(json.dumps(dict(
            same_step_rate=0.9, cross_step_rate=0.9 + gap[s], global_rate=0.9, cross_minus_same=gap[s],
            snap_step_retention=1.0, snap_global_retention=1.0, global_minus_step_snap=0.0, clean_acc=1.0)))
    stats.c4(tmp_path)
    line = next(l for l in capsys.readouterr().out.splitlines() if "cross-same" in l)
    lo_printed = float(line.split("[")[1].split(",")[0])
    assert lo_printed < -0.05 < float(line.split(",")[1].split("]")[0])
