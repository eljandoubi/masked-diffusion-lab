"""Negative ELBO of masked diffusion with the linear schedule, in closed form."""

from math import factorial

import numpy as np

from maskdiff.laws import sequences
from maskdiff.partial import state_index


def pattern_weights(length: int) -> np.ndarray:
    """w[m] = int t**(m-1) (1-t)**(L-m) dt = (m-1)! (L-m)! / L!, with w[0] = 0."""
    w = np.zeros(length + 1)
    for m in range(1, length + 1):
        w[m] = factorial(m - 1) * factorial(length - m) / factorial(length)
    return w


def negative_elbo(table: np.ndarray, vocab: int, length: int) -> np.ndarray:
    """-ELBO(x) for every sequence x, shape (V,) * L, given conditionals (S, L, V).

    The expectation over t of (1/t) times the masked cross-entropy is a sum over
    mask patterns M, each weighted by the integral over t of its probability / t.
    """
    x = sequences(length, vocab)
    w = pattern_weights(length)
    positions = np.arange(length)
    out = np.zeros(len(x))
    for pattern in range(1, 2**length):
        masked = (pattern >> positions) & 1 == 1
        state = state_index(np.where(masked, vocab, x), vocab)
        loss = -np.log(table[state[:, None], positions[None, :], x])
        out += w[masked.sum()] * (loss * masked).sum(axis=1)
    return out.reshape((vocab,) * length)
