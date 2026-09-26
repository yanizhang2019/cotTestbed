"""GF(p) expression testbed: trees, both trace formats, splits, counterfactual evaluator, tokenizer.

Conventions (read these before using the step indices anywhere else)
---------------------------------------------------------------------
* Steps are 0-indexed, t = 0..T-1, in post-order over operator nodes. Post-order equals
  leftmost-innermost-handle reduction order (tested). Compact-trace latent slot t is step t.
  The plan's v_T (the answer) is ``step_values[T-1]``.
* Leaves are numbered 0..T left to right. Internally, operands live in "slots": leaf i is
  slot i, step t is slot T+1+t. ``Example.steps`` exposes them as ``Source('leaf', i)`` or
  ``Source('step', t)``.
* Expressions are fully parenthesised, root included: ``( ( 3 + 1 ) * ( 0 - 4 ) )``.
  An expression with k operators is 4k+1 tokens.
* Compact trace: per step ``a op b = v ;`` (6 tokens, 6T total).
* Full trace (the theorem's): T(E0) = E0 = E1 = ... = ET EOT. E0 is the question (prompt);
  the generated suffix, stored as ``full_trace``, is ``= E1 = E2 ... = ET EOT``. Step t's
  line is ``= E_{t+1}``; the last line also carries EOT. Generated trace symbols: p values,
  + - *, ( ), =, EOT -> p+7 (BOT is in the theorem's alphabet but never generated; BOS/SEP
  play that role here). Length 2T^2+1 for every shape (balanced T=3 -> 19, chain T=4 -> 33);
  the tests check it against an independent build of the suffix, not this formula.

Splits
------
* Space < 1M distinct expressions ("enumerated"): a seeded permutation of the whole space;
  val = first 10%, test = next 10%, train = remaining 80%. Val and test are answer-balanced by
  down-sampling each answer class to the smallest one; discarded val/test candidates are
  never returned to train. Training draws are answer-uniform: pick an answer uniformly, then
  an expression uniformly from the train pool with that answer. That is exact rejection
  sampling to uniform answers over the 80% pool.
* Otherwise ("sampled"): val 5k and test 10k, answer-stratified, deduplicated, test disjoint
  from val. Training draws are answer-uniform and reject any held-out expression by key.
* Split seed is independent of training seed, so every training seed sees the same val/test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

OPS = ("+", "-", "*")
SHAPES = ("chain", "balanced", "random")
SPECIALS = ("BOS", "SEP", "LAT", "ANS", "EOS", "PAD")
ENUM_LIMIT = 1_000_000  # plan: enumerate and split 80/10/10 below this many expressions
MIN_TRAIN = 10_000  # plan: drop a cell with fewer distinct training expressions
N_VAL, N_TEST = 5_000, 10_000  # plan: sizes for large cells
CONTEXT = 256  # plan: model context length


# --------------------------------------------------------------------------- arithmetic

def apply_op(op: int, a: int, b: int, p: int) -> int:
    if op == 0:
        return (a + b) % p
    if op == 1:
        return (a - b) % p
    if op == 2:
        return (a * b) % p
    raise ValueError(f"unknown op {op}")


def _apply_op_vec(op: np.ndarray, a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    return np.where(op == 0, (a + b) % p, np.where(op == 1, (a - b) % p, (a * b) % p))


# --------------------------------------------------------------------------- tree shapes

@lru_cache(maxsize=None)
def catalan(n: int) -> int:
    return math.comb(2 * n, n) // (n + 1)


def n_shapes(shape: str, T: int) -> int:
    return catalan(T) if shape == "random" else 1


def _unrank(n: int, r: int):
    """The r-th full binary tree with n internal nodes (None = leaf, tuple = (left, right))."""
    if n == 0:
        return None
    for k in range(n):  # k internal nodes on the left
        block = catalan(k) * catalan(n - 1 - k)
        if r < block:
            lr, rr = divmod(r, catalan(n - 1 - k))
            return (_unrank(k, lr), _unrank(n - 1 - k, rr))
        r -= block
    raise ValueError("rank out of range")


def _raw_tree(shape: str, T: int, rank: int):
    if T < 1:
        raise ValueError("T must be >= 1")
    if shape == "chain":
        node = None
        for _ in range(T):
            node = (node, None)
        return node
    if shape == "balanced":
        d = (T + 1).bit_length() - 1
        if 2 ** d - 1 != T:
            raise ValueError(f"balanced trees need T = 2^d - 1, got T={T}")

        def build(h):
            return None if h == 0 else (build(h - 1), build(h - 1))

        return build(d)
    if shape == "random":
        if not 0 <= rank < catalan(T):
            raise ValueError("rank out of range")
        return _unrank(T, rank)
    raise ValueError(f"unknown shape {shape!r}")


class Source(NamedTuple):
    kind: str  # "leaf" or "step"
    index: int


class Step(NamedTuple):
    op: str
    left: Source
    right: Source
    value: int


@dataclass(frozen=True)
class Skeleton:
    """Everything about an expression except its operator symbols and leaf values."""

    shape: str
    T: int
    rank: int
    left: tuple  # operand slots per step
    right: tuple
    consumer: tuple  # step that reads step t's value; None for the root
    depth: tuple  # root = 0
    height: tuple  # 1 for a step whose operands are both leaves
    tree: tuple = field(repr=False)  # ("leaf", i) | ("op", t, left, right)

    def source(self, slot: int) -> Source:
        return Source("leaf", slot) if slot <= self.T else Source("step", slot - self.T - 1)


@lru_cache(maxsize=8192)
def skeleton(shape: str, T: int, rank: int = 0) -> Skeleton:
    raw = _raw_tree(shape, T, rank)
    left, right, depth, height = [], [], [], []
    n_leaf = [0]

    def walk(node, d):
        if node is None:
            i = n_leaf[0]
            n_leaf[0] += 1
            return ("leaf", i), i, 0
        ln, ls, lh = walk(node[0], d + 1)
        rn, rs, rh = walk(node[1], d + 1)
        t = len(left)
        left.append(ls)
        right.append(rs)
        depth.append(d)
        height.append(max(lh, rh) + 1)
        return ("op", t, ln, rn), T + 1 + t, height[-1]

    tree, _, _ = walk(raw, 0)
    assert n_leaf[0] == T + 1 and len(left) == T
    consumer = [None] * T
    for t in range(T):
        for slot in (left[t], right[t]):
            if slot > T:
                consumer[slot - T - 1] = t
    return Skeleton(shape, T, rank, tuple(left), tuple(right), tuple(consumer),
                    tuple(depth), tuple(height), tree)


# --------------------------------------------------------------------------- examples

def _key(rank: int, ops, leaves) -> bytes:
    return int(rank).to_bytes(4, "big") + bytes(int(o) for o in ops) + bytes(int(v) for v in leaves)


@dataclass(frozen=True, eq=False)
class Example:
    p: int
    skel: Skeleton
    ops: tuple  # op index per step (0 '+', 1 '-', 2 '*')
    leaves: tuple  # T+1 values in GF(p), left to right

    @property
    def T(self) -> int:
        return self.skel.T

    @property
    def shape(self) -> str:
        return self.skel.shape

    @property
    def consumer(self) -> tuple:
        return self.skel.consumer

    @cached_property
    def step_values(self) -> tuple:
        return evaluate(self)[0]

    @property
    def answer(self) -> int:
        return self.step_values[-1]

    @cached_property
    def steps(self) -> tuple:
        s, sv = self.skel, self.step_values
        return tuple(Step(OPS[self.ops[t]], s.source(s.left[t]), s.source(s.right[t]), sv[t])
                     for t in range(self.T))

    def operand_values(self, t: int) -> tuple:
        slots = self.leaves + self.step_values
        return slots[self.skel.left[t]], slots[self.skel.right[t]]

    @cached_property
    def tokens(self) -> tuple:
        """Question tokens: the fully parenthesised expression."""
        return tuple(self.render(reduced_upto=-1))

    @cached_property
    def compact_trace(self) -> tuple:
        out = []
        for t in range(self.T):
            a, b = self.operand_values(t)
            out += [str(a), OPS[self.ops[t]], str(b), "=", str(self.step_values[t]), ";"]
        return tuple(out)

    @cached_property
    def full_trace_lines(self) -> tuple:
        lines = [["="] + self.render(reduced_upto=t) for t in range(self.T)]
        lines[-1].append("EOT")
        return tuple(tuple(line) for line in lines)

    @cached_property
    def full_trace(self) -> tuple:
        return tuple(tok for line in self.full_trace_lines for tok in line)

    @cached_property
    def key(self) -> bytes:
        return _key(self.skel.rank, self.ops, self.leaves)

    def render(self, reduced_upto: int = -1) -> list:
        """Expression tokens with steps 0..reduced_upto replaced by their values."""
        out: list = []
        sv = self.step_values

        def go(node):
            if node[0] == "leaf":
                out.append(str(self.leaves[node[1]]))
                return
            _, t, l, r = node
            if t <= reduced_upto:
                out.append(str(sv[t]))
                return
            out.append("(")
            go(l)
            out.append(OPS[self.ops[t]])
            go(r)
            out.append(")")

        go(self.skel.tree)
        return out


def make_example(p: int, shape: str, T: int, ops, leaves, rank: int = 0) -> Example:
    ops, leaves = tuple(int(o) for o in ops), tuple(int(v) for v in leaves)
    if len(ops) != T or len(leaves) != T + 1:
        raise ValueError("need T ops and T+1 leaves")
    if any(not 0 <= o < 3 for o in ops) or any(not 0 <= v < p for v in leaves):
        raise ValueError("op or leaf out of range")
    return Example(p, skeleton(shape, T, rank), ops, leaves)


def random_example(p: int, shape: str, T: int, rng: np.random.Generator) -> Example:
    """Uniform over the cell's expression space (no answer balancing)."""
    rank = int(rng.integers(n_shapes(shape, T)))
    return make_example(p, shape, T, rng.integers(3, size=T), rng.integers(p, size=T + 1), rank)


# --------------------------------------------------------------------------- counterfactual evaluator

def evaluate(tree: Example, override: Optional[dict] = None) -> tuple:
    """Return (step_values, answer).

    ``override={t: v}`` forces step t to value v whatever its operands are; every step that
    reads it (directly or through later steps) is recomputed. Steps are 0-indexed.
    """
    skel, p, T = tree.skel, tree.p, tree.skel.T
    override = {int(t): int(v) for t, v in (override or {}).items()}
    for t, v in override.items():
        if not 0 <= t < T:
            raise ValueError(f"override step {t} outside 0..{T - 1}")
        if not 0 <= v < p:
            raise ValueError(f"override value {v} outside GF({p})")
    slots = list(tree.leaves) + [0] * T
    for t in range(T):
        if t in override:
            v = override[t]
        else:
            v = apply_op(tree.ops[t], slots[skel.left[t]], slots[skel.right[t]], p)
        slots[T + 1 + t] = v
    sv = tuple(slots[T + 1:])
    return sv, sv[-1]


def dependents(tree, t: int) -> tuple:
    """Steps whose value is a function of step t: its consumer chain up to the root."""
    skel = tree.skel if isinstance(tree, Example) else tree
    out, c = [], skel.consumer[t]
    while c is not None:
        out.append(c)
        c = skel.consumer[c]
    return tuple(out)


def eval_batch(skel: Skeleton, ops: np.ndarray, leaves: np.ndarray, p: int) -> np.ndarray:
    """Vectorised evaluate for many expressions sharing one skeleton. Returns (M, T) values."""
    T = skel.T
    ops = np.asarray(ops, dtype=np.int64)
    slots = np.empty((ops.shape[0], 2 * T + 1), dtype=np.int64)
    slots[:, : T + 1] = np.asarray(leaves, dtype=np.int64)
    for t in range(T):
        slots[:, T + 1 + t] = _apply_op_vec(ops[:, t], slots[:, skel.left[t]], slots[:, skel.right[t]], p)
    return slots[:, T + 1:]


# --------------------------------------------------------------------------- tokenizer

class Vocab:
    """Values first, so token id v is the value v (used for vocabulary snapping)."""

    def __init__(self, p: int):
        self.p = p
        self.itos = [str(v) for v in range(p)] + list(OPS) + ["(", ")", "=", ";", "EOT"] + list(SPECIALS)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        for s in SPECIALS:
            setattr(self, s.lower(), self.stoi[s])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks) -> list:
        return [self.stoi[t] for t in toks]

    def decode(self, ids) -> list:
        return [self.itos[i] for i in ids]


class Encoded(NamedTuple):
    ids: list
    targets: list  # targets[i]: token i is predicted (loss on logits at i-1)
    latent_pos: list  # positions of latent / pause slots, in order
    ans_pos: int  # position of the answer token


MODES = ("direct", "cot", "latent", "pause")
TRACES = ("compact", "full")


def trace_step_tokens(ex: Example, trace: str) -> list:
    if trace == "compact":
        c = ex.compact_trace
        return [c[6 * t: 6 * t + 6] for t in range(ex.T)]
    if trace == "full":
        return list(ex.full_trace_lines)
    raise ValueError(trace)


def encode(ex: Example, vocab: Vocab, mode: str, trace: str = "compact", stage: Optional[int] = None) -> Encoded:
    """Layout: BOS question SEP [latent slots | trace text] ANS answer EOS.

    latent: stage s replaces the first s trace steps (compact: 1 slot per step; full: 1 slot per
    token of the replaced lines); stage None means all T. pause: the final-stage slot count,
    no trace. Loss (targets) on trace text, ANS when it follows trace text, answer, EOS.
    """
    steps = trace_step_tokens(ex, trace)
    if mode == "direct":
        n_slots, text = 0, []
    elif mode == "cot":
        n_slots, text = 0, [tok for s in steps for tok in s]
    elif mode in ("latent", "pause"):
        s = ex.T if (stage is None or mode == "pause") else stage
        if not 0 <= s <= ex.T:
            raise ValueError(f"stage {s} outside 0..{ex.T}")
        n_slots = s if trace == "compact" else sum(len(x) for x in steps[:s])
        text = [] if mode == "pause" else [tok for x in steps[s:] for tok in x]
    else:
        raise ValueError(mode)
    prefix = ["BOS", *ex.tokens, "SEP"]
    toks = prefix + ["LAT"] * n_slots + text + ["ANS", str(ex.answer), "EOS"]
    targets = [False] * (len(prefix) + n_slots) + [True] * len(text) + [bool(text), True, True]
    latent_pos = list(range(len(prefix), len(prefix) + n_slots))
    return Encoded(vocab.encode(toks), targets, latent_pos, len(toks) - 2)


def seq_len(T: int, mode: str, trace: str = "compact") -> int:
    base = (4 * T + 1) + 5  # question + BOS SEP ANS answer EOS
    extra = {"compact": {"direct": 0, "cot": 6 * T, "latent": T, "pause": T},
             "full": {"direct": 0, "cot": 2 * T * T + 1, "latent": 2 * T * T + 1, "pause": 2 * T * T + 1}}
    return base + extra[trace][mode]


# --------------------------------------------------------------------------- batches of expressions

@dataclass
class Batch:
    """Many expressions of one cell as arrays. rank is the tree-shape rank (0 unless random)."""

    cell: "Cell"
    rank: np.ndarray  # (M,)
    ops: np.ndarray  # (M, T)
    leaves: np.ndarray  # (M, T+1)
    answer: np.ndarray  # (M,)

    def __len__(self) -> int:
        return len(self.rank)

    def keys(self) -> list:
        m = len(self)
        packed = np.concatenate([
            self.rank.astype(">u4").view(np.uint8).reshape(m, 4),
            self.ops.astype(np.uint8), self.leaves.astype(np.uint8)], axis=1)
        return [row.tobytes() for row in packed]

    def take(self, idx) -> "Batch":
        return Batch(self.cell, self.rank[idx], self.ops[idx], self.leaves[idx], self.answer[idx])

    def balanced_head(self, n: int) -> "Batch":
        """The first n // p rows of each answer class, in their original order (exactly balanced)."""
        p = self.cell.p
        keep = np.sort(np.concatenate([np.flatnonzero(self.answer == a)[: n // p] for a in range(p)]))
        return self.take(keep)

    def example(self, i: int) -> Example:
        c = self.cell
        return Example(c.p, skeleton(c.shape, c.T, int(self.rank[i])),
                       tuple(int(o) for o in self.ops[i]), tuple(int(v) for v in self.leaves[i]))

    def examples(self) -> list:
        return [self.example(i) for i in range(len(self))]

    @staticmethod
    def concat(parts: list) -> "Batch":
        return Batch(parts[0].cell, *(np.concatenate([getattr(b, f) for b in parts])
                                      for f in ("rank", "ops", "leaves", "answer")))


def _answers(cell: "Cell", rank, ops, leaves) -> np.ndarray:
    out = np.empty(len(rank), dtype=np.int64)
    for r in np.unique(rank):
        m = rank == r
        out[m] = eval_batch(skeleton(cell.shape, cell.T, int(r)), ops[m], leaves[m], cell.p)[:, -1]
    return out


def decode_index(cell: "Cell", idx: np.ndarray) -> Batch:
    """Bijection from 0..space_size-1 to expressions (enumerated cells only)."""
    idx = np.asarray(idx, dtype=np.int64)
    p, T = cell.p, cell.T
    code = idx % p ** (T + 1)
    rest = idx // p ** (T + 1)
    leaves = np.empty((len(idx), T + 1), dtype=np.int64)
    for j in range(T + 1):
        leaves[:, j] = code % p
        code //= p
    code = rest % 3 ** T
    rank = rest // 3 ** T
    ops = np.empty((len(idx), T), dtype=np.int64)
    for j in range(T):
        ops[:, j] = code % 3
        code //= 3
    return Batch(cell, rank, ops, leaves, _answers(cell, rank, ops, leaves))


def sample_uniform(cell: "Cell", m: int, rng: np.random.Generator) -> Batch:
    rank = rng.integers(n_shapes(cell.shape, cell.T), size=m)
    ops = rng.integers(3, size=(m, cell.T))
    leaves = rng.integers(cell.p, size=(m, cell.T + 1))
    return Batch(cell, rank, ops, leaves, _answers(cell, rank, ops, leaves))


def _fill_by_answer(cell: "Cell", counts: np.ndarray, rng: np.random.Generator,
                    exclude: frozenset | set = frozenset(), dedupe: bool = True) -> Batch:
    """Rejection-sample uniform expressions until each answer class a has counts[a] rows."""
    need = np.array(counts, dtype=np.int64).copy()
    seen: set = set()
    parts = []
    while need.sum() > 0:
        cand = sample_uniform(cell, max(64, int(need.sum()) * cell.p * 2), rng)
        keys = cand.keys()
        keep = []
        for i, (a, k) in enumerate(zip(cand.answer, keys)):
            if need[a] == 0 or k in exclude or (dedupe and k in seen):
                continue
            seen.add(k)
            need[a] -= 1
            keep.append(i)
        parts.append(cand.take(np.array(keep, dtype=np.int64)))
    return Batch.concat(parts)


# --------------------------------------------------------------------------- cells and splits

@dataclass(frozen=True)
class Cell:
    shape: str
    T: int
    p: int

    def __post_init__(self):
        skeleton(self.shape, self.T, 0)  # validates shape and T

    @property
    def name(self) -> str:
        return f"{self.shape}_T{self.T}_p{self.p}"

    @property
    def space_size(self) -> int:
        return n_shapes(self.shape, self.T) * 3 ** self.T * self.p ** (self.T + 1)

    @property
    def enumerated(self) -> bool:
        return self.space_size < ENUM_LIMIT


def _rng(seed: int, cell: Cell, purpose: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, cell.p, SHAPES.index(cell.shape), cell.T, purpose]))


def _stratified_counts(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.full(p, n // p, dtype=np.int64)
    counts[rng.choice(p, size=n % p, replace=False)] += 1
    return counts


def _balance_down(idx: np.ndarray, ans: np.ndarray, p: int) -> np.ndarray:
    """Keep the first m of each answer class (m = smallest class), preserving order."""
    a = ans[idx]
    m = min(int((a == c).sum()) for c in range(p))
    keep = np.zeros(len(idx), dtype=bool)
    for c in range(p):
        keep[np.flatnonzero(a == c)[:m]] = True
    return idx[keep]


@dataclass
class Splits:
    cell: Cell
    seed: int
    val: Batch
    test: Batch
    heldout: frozenset  # keys of every val/test expression
    train_by_answer: Optional[list] = None  # enumerated: train-pool indices per answer
    info: dict = field(default_factory=dict)

    def sample_train(self, n: int, rng: np.random.Generator) -> Batch:
        """n answer-uniform training expressions (with replacement), never held out."""
        c = self.cell
        counts = np.bincount(rng.integers(c.p, size=n), minlength=c.p)
        if self.train_by_answer is not None:
            idx = np.concatenate([rng.choice(self.train_by_answer[a], size=k, replace=True)
                                  for a, k in enumerate(counts) if k])
            out = decode_index(c, idx)
        else:
            out = _fill_by_answer(c, counts, rng, exclude=self.heldout, dedupe=False)
        return out.take(rng.permutation(len(out)))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for name in ("val", "test"):
            h.update(name.encode())
            for k in getattr(self, name).keys():
                h.update(k)
        return h.hexdigest()[:16]

    def save(self, root: str | Path) -> Path:
        d = Path(root) / self.cell.name
        d.mkdir(parents=True, exist_ok=True)
        for name in ("val", "test"):
            b = getattr(self, name)
            np.savez_compressed(d / f"{name}.npz", rank=b.rank, ops=b.ops, leaves=b.leaves, answer=b.answer)
        meta = dict(self.info, cell=self.cell.name, seed=self.seed, fingerprint=self.fingerprint())
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return d

    @staticmethod
    def load(root: str | Path, cell: Cell) -> "Splits":
        """Rebuild from the seed and check against the saved fingerprint (guards against code drift)."""
        meta = json.loads((Path(root) / cell.name / "meta.json").read_text())
        s = make_splits(cell, meta["seed"])
        if s.fingerprint() != meta["fingerprint"]:
            raise RuntimeError(f"{cell.name}: regenerated splits differ from the saved ones")
        return s


def make_splits(cell: Cell, seed: int = 0) -> Splits:
    p = cell.p
    if cell.enumerated:
        N = cell.space_size
        ans = decode_index(cell, np.arange(N)).answer
        perm = _rng(seed, cell, 0).permutation(N)
        n_hold = N // 10
        val_idx = _balance_down(perm[:n_hold], ans, p)
        test_idx = _balance_down(perm[n_hold: 2 * n_hold], ans, p)
        train_idx = perm[2 * n_hold:]
        by_answer = [train_idx[ans[train_idx] == a] for a in range(p)]
        if min(len(x) for x in by_answer) == 0:
            raise RuntimeError(f"{cell.name}: an answer class has no training expressions")
        val, test = decode_index(cell, val_idx), decode_index(cell, test_idx)
        counts = np.bincount(ans[train_idx], minlength=p)
        info = dict(regime="enumerated", space=N, train_distinct=len(train_idx),
                    train_if_downsampled=int(p * counts.min()),
                    frac_answer0=float((ans == 0).mean()))
    else:
        val = _fill_by_answer(cell, _stratified_counts(N_VAL, p, _rng(seed, cell, 1)), _rng(seed, cell, 2))
        test = _fill_by_answer(cell, _stratified_counts(N_TEST, p, _rng(seed, cell, 3)), _rng(seed, cell, 4),
                               exclude=frozenset(val.keys()))
        by_answer = None
        probe = sample_uniform(cell, 100_000, _rng(seed, cell, 5)).answer
        info = dict(regime="sampled", space=cell.space_size,
                    train_distinct=cell.space_size - len(val) - len(test),
                    train_if_downsampled=None, frac_answer0=float((probe == 0).mean()))
    # Balancing fills the over-represented answer (0) first, so the head of each split would be
    # enriched for it. Shuffle so any prefix is representative.
    val = val.take(_rng(seed, cell, 6).permutation(len(val)))
    test = test.take(_rng(seed, cell, 7).permutation(len(test)))
    heldout = frozenset(val.keys()) | frozenset(test.keys())
    info.update(n_val=len(val), n_test=len(test), keep=info["train_distinct"] >= MIN_TRAIN)
    return Splits(cell, seed, val, test, heldout, by_answer, info)


KILL_TEST_1_CELLS = [Cell(s, T, p) for p in (5, 7)
                     for s, Ts in (("chain", (4, 8, 12)), ("balanced", (3, 7, 15))) for T in Ts]


# --------------------------------------------------------------------------- CLI

def _report(cells, seed):
    hdr = (f"{'cell':<18}{'space':>12}  {'regime':<10}{'train':>10}{'train*':>9}{'val':>7}{'test':>7}"
           f"{'P(ans=0)':>10}{'direct':>8}{'cot-c':>7}{'cot-f':>7}  keep")
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        s = make_splits(c, seed)
        i = s.info
        tds = "-" if i["train_if_downsampled"] is None else f"{i['train_if_downsampled']:,}"
        full = seq_len(c.T, "cot", "full")
        print(f"{c.name:<18}{c.space_size:>12.3g}  {i['regime']:<10}{i['train_distinct']:>10.3g}{tds:>9}"
              f"{i['n_val']:>7}{i['n_test']:>7}{i['frac_answer0']:>10.3f}{seq_len(c.T, 'direct'):>8}"
              f"{seq_len(c.T, 'cot'):>7}{full:>6}{'!' if full > CONTEXT else ' '}  {'yes' if i['keep'] else 'DROP'}")
    print("\ntrain = distinct training expressions; train* = size if the pool were down-sampled to "
          "balance answers\n(the sampler reweights instead); ! = longer than the context of", CONTEXT)


def _show(cell, seed):
    rng = np.random.default_rng(seed)
    ex = random_example(cell.p, cell.shape, cell.T, rng)
    v = Vocab(cell.p)
    print("question     ", " ".join(ex.tokens))
    print("steps        ", *[f"t{t}: {s.op} {s.left.kind[0]}{s.left.index} {s.right.kind[0]}{s.right.index} -> {s.value}"
                             for t, s in enumerate(ex.steps)], sep="\n  ")
    print("consumer     ", ex.consumer)
    print("compact trace", " ".join(ex.compact_trace))
    print("full trace   ", " ".join(ex.full_trace), f"({len(ex.full_trace)} tokens)")
    for mode in MODES:
        for tr in TRACES:
            e = encode(ex, v, mode, tr)
            print(f"{mode:>7}/{tr:<7}", " ".join(v.decode(e.ids)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("report", "build", "show"):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--shape", choices=SHAPES)
        sp.add_argument("--T", type=int)
        sp.add_argument("--p", type=int)
        if name == "build":
            sp.add_argument("--out", default="data")
    a = ap.parse_args(argv)
    cells = [Cell(a.shape, a.T, a.p)] if a.shape else KILL_TEST_1_CELLS
    if a.cmd == "report":
        _report(cells, a.seed)
    elif a.cmd == "build":
        for c in cells:
            s = make_splits(c, a.seed)
            print(s.save(a.out), s.info)
    else:
        _show(cells[0], a.seed)


if __name__ == "__main__":
    main()
