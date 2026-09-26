"""Trainer tests: schedule, stage plan, data stream, and a tiny end-to-end run per mode."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stats  # noqa: E402
import train  # noqa: E402
from gen import Cell, Vocab, make_splits  # noqa: E402


def args_for(tmp_path, **kw):
    base = ["--shape", "balanced", "--T", "3", "--p", "5", "--mode", "cot", "--d_model", "32",
            "--epoch_size", "256", "--batch", "64", "--epochs", "1", "--epochs_per_stage", "1",
            "--final_epochs", "1", "--eval_n", "100", "--eval_batch", "256", "--workers", "0",
            "--device", "cpu", "--out", str(tmp_path)]
    a = train.parse_args(base)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_lr_schedule():
    peak, total = 1e-3, 1000
    lrs = [train.lr_at(s, total, peak, 0.02, 0.1) for s in range(total)]
    assert lrs[0] == pytest.approx(peak / 20) and lrs[19] == pytest.approx(peak)  # 2% warmup = 20 steps
    assert max(lrs) == pytest.approx(peak)
    assert lrs[-1] == pytest.approx(0.1 * peak, rel=1e-3)
    assert all(a >= b for a, b in zip(lrs[20:], lrs[21:]))  # cosine never increases


def test_stage_plan(tmp_path):
    a = args_for(tmp_path, mode="latent", epochs_per_stage=3, final_epochs=6)
    assert train.stage_plan(a) == [(0, 3, True), (1, 3, True), (2, 3, True), (3, 6, False)]
    a.mode, a.epochs = "pause", None
    assert train.stage_plan(a) == [(None, 3 * 3 + 6, False)]  # same total epochs as latent
    a.mode = "cot"
    a.epochs = None
    assert train.stage_plan(a) == [(None, 30, False)]


def test_weight_decay_only_on_matrices(tmp_path):
    from model import ModelConfig, TinyGPT
    m = TinyGPT(ModelConfig(vocab_size=24, d_model=32))
    opt = train.make_optimizer(m, args_for(tmp_path))
    dec, nodec = opt.param_groups
    assert dec["weight_decay"] == 0.1 and nodec["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in dec["params"]) and all(p.dim() < 2 for p in nodec["params"])
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == m.n_params()


def test_stream_is_seeded_heldout_free_and_answer_balanced():
    cell = Cell("chain", 8, 7)
    s = make_splits(cell)
    v = Vocab(7)
    mk = lambda seed, stage: iter(train.TrainStream(s, v, "latent", "compact", stage, 128, seed))  # noqa: E731
    a, b, c, d = next(mk(0, 2)), next(mk(0, 2)), next(mk(1, 2)), next(mk(0, 3))
    assert torch.equal(a["ids"], b["ids"])  # same seed, same stage: same data
    assert not torch.equal(a["ids"], c["ids"]) and not torch.equal(a["ids"][:, :33], d["ids"][:, :33])
    assert len(a["latent_pos"]) == 2 and len(d["latent_pos"]) == 3
    it = mk(5, 8)
    answers = []
    for _ in range(40):
        bt = next(it)
        answers += bt["ids"][:, bt["ans_pos"]].tolist()
    freq = np.bincount(answers, minlength=7) / len(answers)
    assert np.abs(freq - 1 / 7).max() < 0.02


@pytest.mark.parametrize("mode", ["direct", "cot", "pause", "latent"])
def test_end_to_end_tiny_run_never_touches_test(tmp_path, mode):
    r = train.train(args_for(tmp_path, mode=mode))
    d = train.run_dir(args_for(tmp_path, mode=mode))
    for f in ("best.pt", "last.pt", "log.jsonl", "results.json", "val_correct.npy"):
        assert (d / f).exists(), f
    assert not list(d.glob("test*")), "train() must not score test"
    res = json.loads((d / "results.json").read_text())
    assert res == json.loads(json.dumps(r))
    assert not any("test" in k for k in res)
    assert res["n_val"] == len(np.load(d / "val_correct.npy")) == len(make_splits(Cell("balanced", 3, 5)).val)
    assert res["val_acc"] == pytest.approx(np.load(d / "val_correct.npy").mean())
    evals = [json.loads(line) for line in (d / "log.jsonl").read_text().splitlines() if '"eval"' in line]
    if mode == "latent":
        assert [st["stage"] for st in res["stages"]] == [0, 1, 2, 3]
        assert [e["stage"] for e in evals] == [0, 1, 2, 3]
        assert (d / "val_correct_zero.npy").exists() and "zero_val_acc" in res
        assert res["best"]["stage"] == 3  # model selection only in the final stage
    assert res["epochs_total"] == sum(s["epochs"] for s in res["stages"])
    ckpt = torch.load(d / "best.pt", map_location="cpu")
    assert ckpt["args"]["mode"] == mode


@pytest.mark.parametrize("mode", ["cot", "latent"])
def test_score_test_on_a_frozen_run(tmp_path, mode):
    a = args_for(tmp_path, mode=mode)
    train.train(a)
    d = train.run_dir(a)
    tr = train.score_test(d, device="cpu")
    ok = np.load(d / "test_correct.npy")
    assert tr["n_test"] == len(ok) == len(make_splits(Cell("balanced", 3, 5)).test)
    assert tr["test_acc"] == pytest.approx(ok.mean())
    assert json.loads((d / "test_results.json").read_text()) == tr
    if mode == "latent":
        assert {"zero_test_acc", "mean_test_acc"} <= set(tr)
        assert (d / "test_correct_zero.npy").exists() and (d / "test_correct_mean.npy").exists()


def test_skip_done(tmp_path, capsys):
    a = args_for(tmp_path)
    first = train.train(a)
    a.skip_done, a.workers, a.eval_batch = True, 3, 64  # where/how it runs does not matter
    assert train.train(a) == first
    assert "skip" in capsys.readouterr().out


def test_existing_run_with_other_config_is_refused(tmp_path):
    a = args_for(tmp_path)
    train.train(a)
    for k, v in (("heads", 2), ("batch", 32), ("split_seed", 1), ("epoch_size", 128)):
        b = args_for(tmp_path, skip_done=True, **{k: v})
        assert train.run_dir(b) == train.run_dir(a)  # same directory name ...
        with pytest.raises(SystemExit, match=k):  # ... but never mistaken for it
            train.train(b)
    c = args_for(tmp_path, d_model=16, skip_done=True)
    assert train.run_dir(c) != train.run_dir(a)  # width is part of the directory name


def test_plateau_advances_early(tmp_path):
    # a model that cannot learn in 16 steps plateaus; patience 1 cuts each 3-epoch stage to 2
    a = args_for(tmp_path, mode="latent", epochs_per_stage=3, final_epochs=1, plateau_patience=1, lr=0.0)
    res = train.train(a)
    assert [s["epochs"] for s in res["stages"]] == [2, 2, 2, 1]


def test_kt1_report(tmp_path, capsys):
    def fake(shape, T, mode, acc, L=2):
        d = tmp_path / f"{shape}{T}{mode}{L}"
        d.mkdir()
        (d / "results.json").write_text(json.dumps(dict(p=5, shape=shape, T=T, layers=L, mode=mode, trace="compact",
                                                        val_acc=acc, n_val=1000, test_acc=0.0, n_test=1000)))
    fake("chain", 8, "direct", 0.25)
    fake("chain", 8, "cot", 0.99)
    fake("balanced", 7, "direct", 0.31)  # 1/5 + 10 pts = 0.30: fails
    fake("balanced", 7, "cot", 0.99)
    assert stats.kt1(tmp_path) is False
    fake("balanced", 15, "direct", 0.29)
    fake("balanced", 15, "cot", 0.985)
    assert stats.kt1(tmp_path) is True
    assert "KILL TEST 1: PASS" in capsys.readouterr().out  # gated on val_acc; test_acc=0 is ignored
