import itertools

import numpy as np

from maskdiff.partial import state_index


def random_table(length, vocab, seed):
    """Positive conditionals that do not come from any joint law."""
    rng = np.random.default_rng(seed)
    raw = rng.random(((vocab + 1) ** length, length, vocab)) + 0.1
    return raw / raw.sum(axis=-1, keepdims=True)


def integrand(table, seq, t, vocab):
    """(1/t) times the masked cross-entropy of seq, summed over mask patterns at t."""
    length = len(seq)
    total = 0.0
    for mask in itertools.product([0, 1], repeat=length):
        m = sum(mask)
        if m == 0:
            continue
        state = state_index(np.where(mask, vocab, seq), vocab)
        loss = sum(-np.log(table[state, i, seq[i]]) for i in range(length) if mask[i])
        total += t ** (m - 1) * (1 - t) ** (length - m) * loss
    return total
