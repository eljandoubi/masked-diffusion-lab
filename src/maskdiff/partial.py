"""Partial sequences: tokens 0..V-1 or the mask V, indexed in base V + 1."""

import numpy as np


def state_digits(vocab: int, length: int) -> np.ndarray:
    """Tokens (V for the mask) of every partial sequence, shape (S, L)."""
    return np.indices((vocab + 1,) * length).reshape(length, -1).T


def state_index(tokens: np.ndarray, vocab: int) -> np.ndarray:
    """Index of partial sequences stored along the last axis of an integer array."""
    power = (vocab + 1) ** np.arange(tokens.shape[-1] - 1, -1, -1)
    return tokens @ power


def partial_marginals(p: np.ndarray) -> np.ndarray:
    """m[s] = P(x_i = s_i at every known position i of s), shape (V + 1,) * L."""
    out = p
    for axis in range(p.ndim):
        out = np.concatenate([out, out.sum(axis=axis, keepdims=True)], axis=axis)
    return out


def conditional_table(m: np.ndarray) -> np.ndarray:
    """Conditionals c[s, i, y] = P(x_i = y | known part of s), shape (S, L, V)."""
    length, vocab = m.ndim, m.shape[0] - 1
    table = np.zeros(m.shape + (length, vocab))
    for i in range(length):
        masked = [slice(None)] * length
        masked[i] = vocab
        context = m[tuple(masked)]
        for y in range(vocab):
            known = [slice(None)] * length
            known[i] = y
            table[(*masked, i, y)] = m[tuple(known)] / context
            table[(*known, i, y)] = 1.0
    return table.reshape(-1, length, vocab)


def mask_tokens(
    x: np.ndarray, t: float | np.ndarray, rng: np.random.Generator, vocab: int
) -> tuple[np.ndarray, np.ndarray]:
    """Forward process with alpha_t = 1 - t: mask each token with probability t."""
    drop = rng.random(x.shape) < np.reshape(t, (-1, 1))
    return np.where(drop, vocab, x), drop
