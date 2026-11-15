"""Exact data laws on sequences of length L over an alphabet of size V."""

import numpy as np


def sequences(length: int, vocab: int) -> np.ndarray:
    """All vocab**length sequences in C order, shape (vocab**length, length)."""
    return np.indices((vocab,) * length).reshape(length, -1).T


def draw(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `size` independent sequences from the law p, shape (size, L)."""
    index = rng.choice(p.size, size=size, p=p.reshape(-1))
    return np.stack(np.unravel_index(index, p.shape), axis=1)


def training_sample(p: np.ndarray, n: int, seed: int) -> np.ndarray:
    """The n sequences of one study run: the pair (seed, n) fixes the sample."""
    return draw(p, n, np.random.default_rng([seed, n]))


def empirical_law(data: np.ndarray, vocab: int) -> np.ndarray:
    """Relative frequencies of the rows of data (n, L), shape (V,) * L."""
    shape = (vocab,) * data.shape[1]
    counts = np.bincount(
        np.ravel_multi_index(data.T, shape), minlength=vocab ** len(shape)
    )
    return counts.reshape(shape) / len(data)


def markov_chain(length: int, vocab: int, persistence: float) -> np.ndarray:
    """Stationary symmetric chain with a uniform start; persistence 0 is uniform."""
    if not 0.0 <= persistence < 1.0:
        raise ValueError("persistence must lie in [0, 1)")
    x = sequences(length, vocab)
    move = (1.0 - persistence) / vocab
    stay = persistence + move
    same = (x[:, :-1] == x[:, 1:]).sum(axis=1)
    p = stay**same * move ** (length - 1 - same) / vocab
    return p.reshape((vocab,) * length)


def potts_ring(length: int, vocab: int, coupling: float) -> np.ndarray:
    """Cyclic Potts chain, p(x) ~ exp(coupling * #{i: x_i = x_i+1}); V = 2 is Ising."""
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
