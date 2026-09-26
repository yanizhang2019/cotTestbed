"""Generator tests. References are independent of gen.py: they work from the question tokens alone."""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen  # noqa: E402
from gen import Cell, Vocab, encode, evaluate, make_splits, random_example  # noqa: E402

PY_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}

CONFIGS = [(s, T) for s, Ts in (("chain", (1, 2, 4, 8, 12)), ("balanced", (1, 3, 7, 15)),
                                ("random", (2, 3, 5, 9, 15))) for T in Ts]


def examples(n, seed=0, ps=(5, 7, 11)):
    rng = np.random.default_rng(seed)
    for i in range(n):
        shape, T = CONFIGS[i % len(CONFIGS)]
        yield random_example(ps[i % len(ps)], shape, T, rng)


# ---- independent references -------------------------------------------------

def parse(tokens):
    """Recursive-descent parse of a fully parenthesised expression -> nested tuples."""
    pos = 0

    def expr():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok != "(":
            return int(tok)
        left = expr()
        op = tokens[pos]
        pos += 1
        right = expr()
        assert tokens[pos] == ")"
        pos += 1
        return (op, left, right)

    tree = expr()
    assert pos == len(tokens)
    return tree


def ref_eval(tree, p, override=None):
    """Post-order evaluation with its own step counter; returns (step values, answer)."""
    override = override or {}
    vals = []

    def go(node):
        if isinstance(node, int):
            return node
        a, b = go(node[1]), go(node[2])
        t = len(vals)
        vals.append(None)
        vals[t] = override[t] if t in override else PY_OPS[node[0]](a, b) % p
        return vals[t]

    ans = go(tree)
    return tuple(vals), ans


def leftmost_handle_reduce(tokens, p):
    """String rewriting: repeatedly reduce the leftmost '( v op v )'. Returns [(value, expr after)]."""
    toks, out = list(tokens), []
    is_val = lambda s: s.isdigit()  # noqa: E731
    while len(toks) > 1:
        for i in range(len(toks) - 4):
            w = toks[i:i + 5]
            if w[0] == "(" and is_val(w[1]) and w[2] in PY_OPS and is_val(w[3]) and w[4] == ")":
                v = PY_OPS[w[2]](int(w[1]), int(w[3])) % p
                toks = toks[:i] + [str(v)] + toks[i + 5:]
                out.append((v, tuple(toks)))
                break
        else:
            raise AssertionError("no handle found")
    return out


# ---- plan's unit tests ------------------------------------------------------

def test_evaluate_agrees_with_direct_evaluation_on_10k_trees():
    for ex in examples(10_000):
        direct = eval(" ".join(ex.tokens)) % ex.p  # integer arithmetic, reduced mod p once
        sv, ans = evaluate(ex)
        assert ans == direct == ex.answer
        assert (sv, ans) == ref_eval(parse(ex.tokens), ex.p)


def test_postorder_matches_leftmost_handle_reduction():
    for ex in examples(3_000, seed=1):
        red = leftmost_handle_reduce(ex.tokens, ex.p)
        assert [v for v, _ in red] == list(ex.step_values)
        assert [e for _, e in red] == [tuple(t for t in line[1:] if t != "EOT") for line in ex.full_trace_lines]
        for t, s in enumerate(ex.steps):  # compact trace operands are the handle's operands
            a, b = ex.operand_values(t)
            assert ex.compact_trace[6 * t: 6 * t + 6] == (str(a), s.op, str(b), "=", str(s.value), ";")


def theorem_suffix(tokens, p):
    """The theorem's generated suffix '= E1 = ... = ET EOT', built by string rewriting from E0."""
    out = []
    for _, expr in leftmost_handle_reduce(tokens, p):
        out += ["="] + list(expr)
    return out + ["EOT"]


def test_full_trace_is_theorem_suffix_and_ends_in_answer_eot():
    for ex in examples(3_000, seed=2):
        T = ex.T
        assert list(ex.full_trace) == theorem_suffix(ex.tokens, ex.p)
        assert ex.full_trace[-2:] == (str(ex.answer), "EOT")
        assert len(ex.full_trace) == 2 * T * T + 1 == gen.seq_len(T, "latent", "full") - gen.seq_len(T, "direct")
        assert len(ex.tokens) == 4 * T + 1
        assert len(ex.compact_trace) == 6 * T
    ex = gen.make_example(5, "balanced", 3, [2, 1, 1], [1, 1, 0, 0])  # the example from the thread
    assert " ".join(ex.tokens) == "( ( 1 * 1 ) - ( 0 - 0 ) )"
    assert " ".join(ex.full_trace) == "= ( 1 - ( 0 - 0 ) ) = ( 1 - 0 ) = 1 EOT"


def test_theorem_latent_counts():
    rng = np.random.default_rng(0)
    for shape, T, n in (("balanced", 3, 19), ("chain", 4, 33)):
        ex = random_example(5, shape, T, rng)
        assert len(theorem_suffix(ex.tokens, 5)) == n
        assert len(encode(ex, Vocab(5), "latent", "full").latent_pos) == n
        assert len(encode(ex, Vocab(5), "pause", "full").latent_pos) == n


def test_generated_full_trace_alphabet_is_p_plus_7():
    for p in (5, 7, 11):
        seen = set()
        for ex in examples(2_000, seed=p, ps=(p,)):
            seen |= set(ex.full_trace)
        assert seen == {str(v) for v in range(p)} | {"+", "-", "*", "(", ")", "=", "EOT"}
        assert len(seen) == p + 7


def test_heldout_never_in_training_batches():
    for cell in (Cell("balanced", 3, 5), Cell("chain", 4, 5), Cell("chain", 8, 5), Cell("random", 4, 7)):
        s = make_splits(cell)
        val, test = set(s.val.keys()), set(s.test.keys())
        assert len(val) == len(s.val) and len(test) == len(s.test)  # no duplicates
        assert not val & test
        assert s.heldout == val | test
        rng = np.random.default_rng(1)
        for _ in range(50):
            b = s.sample_train(256, rng)
            assert not set(b.keys()) & s.heldout, cell.name


def test_enumerated_train_pool_excludes_all_heldout_candidates():
    """Val/test candidates dropped by answer balancing must not leak into training."""
    cell = Cell("balanced", 3, 5)
    s = make_splits(cell)
    N = cell.space_size
    perm = gen._rng(s.seed, cell, 0).permutation(N)
    candidates = set(perm[: 2 * (N // 10)].tolist())
    pool = np.concatenate(s.train_by_answer)
    assert len(pool) == N - 2 * (N // 10) == s.info["train_distinct"]
    assert not candidates & set(pool.tolist())


# ---- splits and balance ------------------------------------------------------

def test_plan_cell_sizes():
    c = Cell("balanced", 3, 5)
    assert c.space_size == 16_875 and c.enumerated
    s = make_splits(c)
    assert s.info["train_distinct"] == 16_875 - 2 * 1_687 == 13_501 and s.info["keep"]  # plan: "about 13.5k"
    assert not Cell("chain", 8, 5).enumerated
    big = make_splits(Cell("chain", 8, 5))
    assert (len(big.val), len(big.test)) == (5_000, 10_000)


def test_answer_balance():
    for cell in (Cell("balanced", 3, 5), Cell("chain", 8, 7)):
        s = make_splits(cell)
        p = cell.p
        assert s.info["frac_answer0"] > 1 / p + 0.02  # the raw space is skewed toward 0
        for b in (s.val, s.test):
            counts = np.bincount(b.answer, minlength=p)
            assert counts.max() - counts.min() <= 1
            assert all(b.example(i).answer == b.answer[i] for i in range(0, len(b), 97))
        draws = np.concatenate([s.sample_train(1000, np.random.default_rng(i)).answer for i in range(40)])
        freq = np.bincount(draws, minlength=p) / len(draws)
        assert np.abs(freq - 1 / p).max() < 0.01, freq


def test_split_prefixes_are_representative():
    """Balancing fills answer 0 first; after the shuffle every prefix is near-balanced, and
    balanced_head is exactly balanced."""
    for cell in (Cell("balanced", 3, 5), Cell("chain", 8, 7)):
        s = make_splits(cell)
        p = cell.p
        for b in (s.val, s.test):
            for n in (200, 1000):
                freq = np.bincount(b.answer[:n], minlength=p) / n
                assert np.abs(freq - 1 / p).max() < 4 * np.sqrt((1 / p) * (1 - 1 / p) / n), (cell.name, n, freq)
        h = s.val.balanced_head(1000)
        counts = np.bincount(h.answer, minlength=p)
        assert counts.min() == counts.max() == 1000 // p
        assert set(h.keys()) <= set(s.val.keys())


def test_enumeration_is_a_bijection():
    for cell in (Cell("balanced", 3, 3), Cell("random", 3, 3), Cell("chain", 2, 5)):
        b = gen.decode_index(cell, np.arange(cell.space_size))
        keys = b.keys()
        assert len(set(keys)) == cell.space_size
        toks = {b.example(i).tokens for i in range(len(b))}
        assert len(toks) == cell.space_size  # distinct keys <-> distinct token strings


def test_splits_deterministic_and_seeded():
    for cell in (Cell("balanced", 3, 5), Cell("chain", 8, 5)):
        a, b, c = make_splits(cell, 0), make_splits(cell, 0), make_splits(cell, 1)
        assert a.fingerprint() == b.fingerprint() != c.fingerprint()


def test_save_and_load(tmp_path):
    cell = Cell("balanced", 3, 5)
    s = make_splits(cell)
    s.save(tmp_path)
    assert gen.Splits.load(tmp_path, cell).fingerprint() == s.fingerprint()


def test_random_shapes_uniform_and_distinct():
    T = 4
    n = gen.catalan(T)
    shapes = {gen.skeleton("random", T, r).left + gen.skeleton("random", T, r).right for r in range(n)}
    assert len(shapes) == n == 14
    rng = np.random.default_rng(0)
    counts = Counter(random_example(5, "random", T, rng).skel.rank for _ in range(14_000))
    assert len(counts) == n and max(abs(c - 1000) for c in counts.values()) < 150


def test_balanced_requires_complete_tree():
    for T in (2, 4, 5, 6):
        with pytest.raises(ValueError):
            Cell("balanced", T, 5)


def test_consumer_and_depth():
    s = gen.skeleton("chain", 4)
    assert s.consumer == (1, 2, 3, None) and s.depth == (3, 2, 1, 0)
    b = gen.skeleton("balanced", 7)
    assert b.consumer == (2, 2, 6, 5, 5, 6, None)
    assert b.height == (1, 1, 2, 1, 1, 2, 3)
    for ex in examples(500, seed=3):
        roots = [t for t, c in enumerate(ex.consumer) if c is None]
        assert roots == [ex.T - 1]
        assert all(c > t for t, c in enumerate(ex.consumer) if c is not None)
        # each non-root step is read exactly once, by its consumer
        reads = Counter(s.index for st in ex.steps for s in (st.left, st.right) if s.kind == "step")
        assert reads == Counter(t for t in range(ex.T - 1))


# ---- counterfactual evaluator ------------------------------------------------

def test_counterfactual_matches_reference():
    rng = np.random.default_rng(4)
    for ex in examples(4_000, seed=4):
        t = int(rng.integers(ex.T))
        v = int(rng.integers(ex.p))
        sv, ans = evaluate(ex, {t: v})
        assert (sv, ans) == ref_eval(parse(ex.tokens), ex.p, {t: v})
        dep = set(gen.dependents(ex, t))
        for s in range(ex.T):
            if s != t and s not in dep:
                assert sv[s] == ex.step_values[s]  # steps not downstream of t keep their value
        assert sv[t] == v
        if t == ex.T - 1:
            assert ans == v
        assert evaluate(ex, {t: ex.step_values[t]}) == (ex.step_values, ex.answer)


def test_counterfactual_multiple_overrides_and_validation():
    ex = gen.make_example(7, "chain", 4, [0, 0, 0, 0], [1, 1, 1, 1, 1])  # ((((1+1)+1)+1)+1)
    assert ex.step_values == (2, 3, 4, 5)
    assert evaluate(ex, {1: 0}) == ((2, 0, 1, 2), 2)
    assert evaluate(ex, {0: 6, 2: 3}) == ((6, 0, 3, 4), 4)
    assert gen.dependents(ex, 0) == (1, 2, 3) and gen.dependents(ex, 3) == ()
    for bad in ({4: 0}, {-1: 0}, {0: 7}, {0: -1}):
        with pytest.raises(ValueError):
            evaluate(ex, bad)


def test_eval_batch_matches_scalar():
    cell = Cell("random", 5, 11)
    b = gen.sample_uniform(cell, 2_000, np.random.default_rng(0))
    for i in range(len(b)):
        assert b.example(i).answer == b.answer[i]
    for shape, T in (("chain", 12), ("balanced", 15)):
        b = gen.sample_uniform(Cell(shape, T, 7), 500, np.random.default_rng(1))
        sv = gen.eval_batch(gen.skeleton(shape, T), b.ops, b.leaves, 7)
        for i in range(len(b)):
            assert tuple(sv[i]) == b.example(i).step_values


def test_example_key_matches_batch_key():
    b = gen.sample_uniform(Cell("random", 6, 7), 300, np.random.default_rng(0))
    assert [b.example(i).key for i in range(len(b))] == b.keys()


# ---- tokenizer and sequence layout ---------------------------------------------

def test_vocab():
    v = Vocab(11)
    assert v.encode(["10"]) == [10] and len(v) == 11 + 3 + 5 + 6  # values, ops, ( ) = ; EOT, specials
    assert all(v.encode([str(k)]) == [k] for k in range(11))
    for ex in examples(200, seed=5, ps=(11,)):
        assert v.decode(v.encode(ex.full_trace)) == list(ex.full_trace)


@pytest.mark.parametrize("trace", ["compact", "full"])
def test_encode_layouts(trace):
    for ex in examples(300, seed=6):
        v = Vocab(ex.p)
        T = ex.T
        per_step = 1 if trace == "compact" else None
        cot = encode(ex, v, "cot", trace)
        assert encode(ex, v, "latent", trace, stage=0) == cot
        for mode in gen.MODES:
            e = encode(ex, v, mode, trace)
            toks = v.decode(e.ids)
            assert len(e.ids) == len(e.targets) == gen.seq_len(T, mode, trace)
            assert toks[0] == "BOS" and toks[-1] == "EOS" and toks[e.ans_pos - 1] == "ANS"
            assert e.ids[e.ans_pos] == ex.answer and e.targets[e.ans_pos]
            assert toks[1: 1 + len(ex.tokens)] == list(ex.tokens)
            assert [toks[i] for i in e.latent_pos] == ["LAT"] * len(e.latent_pos)
            assert not any(e.targets[: e.latent_pos[-1] + 1 if e.latent_pos else len(ex.tokens) + 2])
        for s in range(T + 1):
            e = encode(ex, v, "latent", trace, stage=s)
            n = s if per_step else sum(len(line) for line in ex.full_trace_lines[:s])
            assert len(e.latent_pos) == n
            text = [tok for tok, m in zip(v.decode(e.ids), e.targets) if m]
            rest = gen.trace_step_tokens(ex, trace)[s:]
            assert text == [tok for x in rest for tok in x] + (["ANS"] if rest else []) + [str(ex.answer), "EOS"]
        assert len(encode(ex, v, "pause", trace).latent_pos) == (T if trace == "compact" else 2 * T * T + 1)


def test_kill_test_cells_fit_context():
    for c in gen.KILL_TEST_1_CELLS:
        assert gen.seq_len(c.T, "cot", "compact") <= gen.CONTEXT
