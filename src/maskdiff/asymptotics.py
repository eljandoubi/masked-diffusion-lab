"""Large-K behaviour of the uniform-grid sampler with exact conditionals."""

import itertools

import numpy as np

from maskdiff.laws import sequences
from maskdiff.partial import partial_marginals, state_index


def tie_probability(steps: int, length: int) -> float:
    """P(two positions are revealed in the same step) = 1 - K! / ((K-L)! K**L)."""
    if steps < length:
        return 1.0
    return 1.0 - float(np.prod(1.0 - np.arange(length) / steps))


def first_order_change(p: np.ndarray) -> np.ndarray:
    """lim K (q_K - p) / p of the exact-conditional sampler, from single tied pairs."""
    length, vocab = p.ndim, p.shape[0]
    mass = partial_marginals(p).reshape(-1)
    x = sequences(length, vocab)
    change = np.zeros(len(x))
    for i, j in itertools.combinations(range(length), 2):
        rest = [k for k in range(length) if k not in (i, j)]
        for size in range(len(rest) + 1):
            subsets = list(itertools.combinations(rest, size))
            for subset in subsets:
                known = np.full_like(x, vocab)
                known[:, list(subset)] = x[:, list(subset)]
                with_i, with_j = known.copy(), known.copy()
                with_i[:, i], with_j[:, j] = x[:, i], x[:, j]
                with_both = with_i.copy()
                with_both[:, j] = x[:, j]
                m0, mi, mj, mij = (
                    mass[state_index(s, vocab)]
                    for s in (known, with_i, with_j, with_both)
                )
                weight = 1.0 / (len(subsets) * (length - 1))
                change += weight * (mi * mj / (m0 * mij) - 1.0)
    return change.reshape(p.shape)


def leading_coefficient(p: np.ndarray) -> float:
    """lim K**2 KL(p || q_K) = (1/2) sum_x p(x) first_order_change(x)**2."""
    return 0.5 * float(np.sum(p * first_order_change(p) ** 2))
