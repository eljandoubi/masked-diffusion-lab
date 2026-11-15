import itertools

import numpy as np
import pytest

from maskdiff.laws import draw, empirical_law, markov_chain, potts_ring
from maskdiff.partial import (
    conditional_table,
    mask_tokens,
    partial_marginals,
    plug_in_table,
    state_digits,
    state_index,
)


def brute_marginal(p, state, vocab):
    """P(known part of state) by summing p over every consistent sequence."""
    total = 0.0
    for x in itertools.product(range(vocab), repeat=len(state)):
        if all(s == vocab or s == xi for s, xi in zip(state, x, strict=True)):
            total += p[x]
    return total


def test_state_index_inverts_state_digits():
    digits = state_digits(3, 4)
    assert digits.shape == (4**4, 4)
    assert state_index(digits, 3).tolist() == list(range(4**4))
    assert digits[-1].tolist() == [3, 3, 3, 3]


@pytest.mark.parametrize(
    "make", [lambda: markov_chain(4, 3, 0.7), lambda: potts_ring(4, 3, 1.0)]
)
def test_partial_marginals_match_brute_force_sums(make):
    p = make()
    m = partial_marginals(p)
    assert m.shape == (4,) * 4
    assert m[(3,) * 4] == pytest.approx(1.0)
    assert m[:3, :3, :3, :3] == pytest.approx(p)
    rng = np.random.default_rng(0)
    for state in rng.integers(0, 4, size=(40, 4)):
        assert m[tuple(state)] == pytest.approx(brute_marginal(p, state, 3))


def test_conditionals_marginalise_correctly():
    p = potts_ring(4, 3, 0.9)
    table = conditional_table(partial_marginals(p))
    digits = state_digits(3, 4)
    rng = np.random.default_rng(1)
    for s in rng.integers(0, len(digits), size=60):
        state = digits[s]
        for i in np.flatnonzero(state == 3):
            context = brute_marginal(p, state, 3)
            for y in range(3):
                joint = state.copy()
                joint[i] = y
                expected = brute_marginal(p, joint, 3) / context
                assert table[s, i, y] == pytest.approx(expected)


def test_conditionals_are_distributions_and_one_hot_at_known_positions():
    table = conditional_table(partial_marginals(markov_chain(5, 3, 0.8)))
    digits = state_digits(3, 5)
    assert table.sum(axis=-1) == pytest.approx(1.0)
    known = digits != 3
    onehot = np.eye(3)[np.where(known, digits, 0)]
    assert table[known] == pytest.approx(onehot[known])


def test_chain_rule_in_any_order_gives_back_the_law():
    p = potts_ring(5, 3, 0.8)
    table = conditional_table(partial_marginals(p))
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = rng.integers(0, 3, size=5)
        order = rng.permutation(5)
        state = np.full(5, 3)
        prob = 1.0
        for i in order:
            prob *= table[state_index(state, 3), i, x[i]]
            state[i] = x[i]
        assert prob == pytest.approx(p[tuple(x)])


def test_plug_in_conditionals_add_alpha_to_the_counts_of_the_rows():
    data = np.array([[0, 0], [0, 0], [0, 1]])
    table = plug_in_table(data, 2, 0.5)
    # base 3 states: (m, m) is 8, (0, m) is 2, (m, 1) is 7 and (m, 0) is 6
    assert table[8, 0] == pytest.approx([3.5 / 4, 0.5 / 4])
    assert table[8, 1] == pytest.approx([2.5 / 4, 1.5 / 4])
    assert table[2, 1] == pytest.approx([2.5 / 4, 1.5 / 4])
    assert table[7, 0] == pytest.approx([1.5 / 2, 0.5 / 2])
    assert table[6, 0] == pytest.approx([2.5 / 3, 0.5 / 3])


def test_plug_in_conditionals_without_smoothing_are_the_empirical_frequencies():
    data = draw(markov_chain(3, 3, 0.5), 3000, np.random.default_rng(0))
    table = plug_in_table(data, 3, 0.0)
    empirical = conditional_table(partial_marginals(empirical_law(data, 3)))
    assert table == pytest.approx(empirical, abs=1e-12)


def test_smoothed_conditionals_are_distributions_and_uniform_in_unseen_contexts():
    data = np.zeros((5, 3), dtype=int)
    table = plug_in_table(data, 2, 1.0)
    assert table.sum(axis=-1) == pytest.approx(1.0)
    assert table[state_index(np.array([1, 2, 2]), 2), 1] == pytest.approx([0.5, 0.5])
    assert table[state_index(np.array([2, 2, 2]), 2), 0] == pytest.approx(
        [6 / 7, 1 / 7]
    )
    masked = state_digits(2, 3) == 2
    assert plug_in_table(data, 2, 1e9)[masked] == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
def test_forward_masking_rate_is_one_minus_alpha(t):
    n, length, vocab = 200_000, 6, 3
    rng = np.random.default_rng(3)
    x = rng.integers(0, vocab, size=(n, length))
    out, drop = mask_tokens(x, t, rng, vocab)
    sigma = np.sqrt(t * (1 - t) / n)
    assert np.abs(drop.mean(axis=0) - t).max() < 5 * sigma
    assert (out[drop] == vocab).all() and (out[~drop] == x[~drop]).all()
    for token in range(vocab):
        assert abs(drop[x == token].mean() - t) < 5 * sigma * np.sqrt(vocab)


def test_forward_masking_is_independent_across_positions_and_takes_a_rate_per_row():
    n = 200_000
    rng = np.random.default_rng(4)
    x = rng.integers(0, 3, size=(n, 6))
    _, drop = mask_tokens(x, 0.5, rng, 3)
    corr = np.corrcoef(drop.T.astype(float))
    assert np.abs(corr - np.eye(6)).max() < 5 / np.sqrt(n)
    t = np.where(np.arange(n) < n // 2, 0.2, 0.8)
    _, drop = mask_tokens(x, t, rng, 3)
    assert drop[: n // 2].mean() == pytest.approx(0.2, abs=0.005)
    assert drop[n // 2 :].mean() == pytest.approx(0.8, abs=0.005)
