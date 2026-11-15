"""Exact data laws on sequences of length L over an alphabet of size V."""

import numpy as np


def sequences(length: int, vocab: int) -> np.ndarray:
    """All vocab**length sequences in C order, shape (vocab**length, length)."""
    return np.indices((vocab,) * length).reshape(length, -1).T


def draw(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `size` independent sequences from the law p, shape (size, L)."""
    index = rng.choice(p.size, size=size, p=p.reshape(-1))
    return np.stack(np.unravel_index(index, p.shape), axis=1)


def markov_chain(length: int, vocab: int, persistence: float) -> np.ndarray:
    """Stationary chain with a uniform start; persistence 0 gives the uniform law.

    A token repeats its predecessor with probability persistence + (1 - persistence) / V
    and moves to each other symbol with probability (1 - persistence) / V.
    """
    if not 0.0 <= persistence < 1.0:
        raise ValueError("persistence must lie in [0, 1)")
    x = sequences(length, vocab)
    move = (1.0 - persistence) / vocab
    stay = persistence + move
    same = (x[:, :-1] == x[:, 1:]).sum(axis=1)
    p = stay**same * move ** (length - 1 - same) / vocab
    return p.reshape((vocab,) * length)


def potts_ring(length: int, vocab: int, coupling: float) -> np.ndarray:
    """Cyclic Potts chain: p(x) is proportional to exp(coupling * #{i: x_i = x_i+1}).

    For V = 2 this is the Ising ring with coupling / 2 on spins in {-1, +1}.
    """
    if length < 3:
        raise ValueError("a ring needs at least 3 sites")
    x = sequences(length, vocab)
    same = (x == np.roll(x, -1, axis=1)).sum(axis=1)
    weight = np.exp(coupling * same)
    return (weight / weight.sum()).reshape((vocab,) * length)


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def marginal(p: np.ndarray, i: int) -> np.ndarray:
    """Law of the token at position i."""
    return p.sum(axis=tuple(a for a in range(p.ndim) if a != i))


def product_of_marginals(p: np.ndarray) -> np.ndarray:
    """The independent law with the same one-position marginals as p."""
    out = np.ones(())
    for i in range(p.ndim):
        out = np.multiply.outer(out, marginal(p, i))
    return out


def total_correlation(p: np.ndarray) -> float:
    """KL(p || product of its marginals), which equals sum_i H(x_i) - H(x)."""
    return sum(entropy(marginal(p, i)) for i in range(p.ndim)) - entropy(p)
