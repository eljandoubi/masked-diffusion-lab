"""Negative ELBO of masked diffusion with the linear schedule, in closed form."""

from math import factorial

import numpy as np

from maskdiff.laws import sequences
from maskdiff.partial import state_index


def pattern_weights(length: int, floor: float = 0.0) -> np.ndarray:
    """w[m] = int t**m (1-t)**(L-m) / max(t, floor) dt, with w[0] = 0."""
    w = np.zeros(length + 1)
    if floor == 0.0:
        for m in range(1, length + 1):
            w[m] = factorial(m - 1) * factorial(length - m) / factorial(length)
        return w
    # Both pieces are polynomials in t, so Gauss-Legendre is exact on each.
    nodes, weights = np.polynomial.legendre.leggauss(length + 2)
    low = floor * (nodes + 1) / 2
    high = floor + (1 - floor) * (nodes + 1) / 2
    for m in range(1, length + 1):
        below = 0.5 * (weights * low**m * (1 - low) ** (length - m)).sum()
        above = (
            (1 - floor)
            / 2
            * (weights * high ** (m - 1) * (1 - high) ** (length - m)).sum()
        )
        w[m] = below + above
    return w


def negative_elbo(
    table: np.ndarray, vocab: int, length: int, floor: float = 0.0
) -> np.ndarray:
    """-ELBO(x) for all sequences (shape (V,) * L) from conditionals (S, L, V)."""
    x = sequences(length, vocab)
    w = pattern_weights(length, floor)
    positions = np.arange(length)
    out = np.zeros(len(x))
    for pattern in range(1, 2**length):
        masked = (pattern >> positions) & 1 == 1
        state = state_index(np.where(masked, vocab, x), vocab)
        loss = -np.log(table[state[:, None], positions[None, :], x])
        out += w[masked.sum()] * (loss * masked).sum(axis=1)
    return out.reshape((vocab,) * length)
