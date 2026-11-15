"""Negative ELBO of masked diffusion with the linear schedule, in closed form."""

from math import factorial

import numpy as np

from maskdiff.laws import sequences
from maskdiff.partial import state_index


def pattern_weights(length: int, floor: float = 0.0) -> np.ndarray:
    """w[m] = int t**m (1-t)**(L-m) / max(t, floor) dt, with w[0] = 0.

    Without a floor this is the Beta integral (m-1)! (L-m)! / L!. With one it is the
    weight of the clipped objective used for training, by Gauss-Legendre quadrature
    on [0, floor] and [floor, 1] (the integrands are polynomials).
    """
    w = np.zeros(length + 1)
    if floor == 0.0:
        for m in range(1, length + 1):
            w[m] = factorial(m - 1) * factorial(length - m) / factorial(length)
        return w
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
    """-ELBO(x) for every sequence x, shape (V,) * L, given conditionals (S, L, V).

    The expectation over t of (1/t) times the masked cross-entropy is a sum over
    mask patterns M, each weighted by the integral over t of its probability / t.
    With a floor, 1/t is replaced by 1/max(t, floor).
    """
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
