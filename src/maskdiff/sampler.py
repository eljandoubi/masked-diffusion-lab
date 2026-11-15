"""Exact output law and simulation of the K-step parallel sampler."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import comb

import numpy as np

from maskdiff.partial import state_digits, state_index


@dataclass(frozen=True)
class Transitions:
    """All moves of one sampler step, as parallel arrays over the moves."""

    rows: np.ndarray  # source partial sequence
    cols: np.ndarray  # target partial sequence
    revealed: np.ndarray  # number of positions the move reveals
    masked: np.ndarray  # number of masked positions in the source
    factors: np.ndarray  # table index of each drawn token, padded with a 1.0 entry


@lru_cache(maxsize=None)
def transitions(vocab: int, length: int) -> Transitions:
    """All (2V + 1)**L moves of a step from the (V + 1)**L partial sequences."""
    digits = state_digits(vocab, length)
    power = (vocab + 1) ** np.arange(length - 1, -1, -1)
    n_masked = (digits == vocab).sum(axis=1)
    pad = len(digits) * length * vocab
    rows, cols, revealed, masked, factors = [], [], [], [], []
    for subset in range(2**length):
        positions = np.array([i for i in range(length) if subset >> i & 1], dtype=int)
        source = np.flatnonzero((digits[:, positions] == vocab).all(axis=1))
        for values in product(range(vocab), repeat=len(positions)):
            index = np.full((len(source), length), pad)
            shift = 0
            for j, (i, y) in enumerate(zip(positions, values, strict=True)):
                index[:, j] = (source * length + i) * vocab + y
                shift += (y - vocab) * power[i]
            rows.append(source)
            cols.append(source + shift)
            revealed.append(np.full(len(source), len(positions)))
            masked.append(n_masked[source])
            factors.append(index)
    return Transitions(
        rows=np.concatenate(rows).astype(np.int32),
        cols=np.concatenate(cols).astype(np.int32),
        revealed=np.concatenate(revealed).astype(np.int8),
        masked=np.concatenate(masked).astype(np.int8),
        factors=np.concatenate(factors).astype(np.int32),
    )


def uniform_schedule(steps: int, length: int) -> list[np.ndarray]:
    """Uniform grid alpha_k = k/K: step k reveals each masked position w.p. 1/(K-k)."""
    m = np.arange(length + 1)[:, None]
    a = np.arange(length + 1)[None, :]
    out = []
    for k in range(steps):
        r = 1.0 / (steps - k)
        out.append(np.where(a <= m, r**a * (1.0 - r) ** np.clip(m - a, 0, None), 0.0))
    return out


def random_order_schedule(counts: Sequence[int], length: int) -> list[np.ndarray]:
    """Step k reveals counts[k] uniformly chosen masked positions (all if fewer)."""
    out = []
    for n in counts:
        step = np.zeros((length + 1, length + 1))
        for m in range(length + 1):
            a = min(n, m)
            step[m, a] = 1.0 / comb(m, a)
        out.append(step)
    return out


def sampler_law(
    table: np.ndarray, vocab: int, length: int, schedule: Sequence[np.ndarray]
) -> np.ndarray:
    """Exact output law (shape (V,) * L) of the sampler, by dynamic programming."""
    moves = transitions(vocab, length)
    weight = np.append(table.reshape(-1), 1.0)[moves.factors].prod(axis=1)
    n_states = (vocab + 1) ** length
    law = np.zeros(n_states)
    law[-1] = 1.0
    for step in schedule:
        # step[m, a]: probability of one given set of a revealed among m masked
        mass = law[moves.rows] * weight * step[moves.masked, moves.revealed]
        law = np.bincount(moves.cols, weights=mass, minlength=n_states)
    full = law.reshape((vocab + 1,) * length)[(slice(0, vocab),) * length]
    if abs(full.sum() - 1.0) > 1e-9:
        raise ValueError("the schedule leaves positions masked")
    return full


def simulate(
    table: np.ndarray,
    vocab: int,
    length: int,
    steps: int,
    size: int,
    rng: np.random.Generator,
    chunk: int = 100_000,
) -> np.ndarray:
    """Run the uniform-grid sampler `size` times and return the sequences."""
    out = np.empty((size, length), dtype=np.int64)
    for start in range(0, size, chunk):
        n = min(chunk, size - start)
        state = np.full((n, length), vocab)
        for k in range(steps):
            reveal = (state == vocab) & (rng.random((n, length)) < 1.0 / (steps - k))
            cdf = table[state_index(state, vocab)].cumsum(axis=-1)
            token = (rng.random((n, length, 1)) > cdf).sum(axis=-1)
            state = np.where(reveal, np.minimum(token, vocab - 1), state)
        out[start : start + n] = state
    return out
