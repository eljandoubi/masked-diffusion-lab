import itertools
from math import comb

import numpy as np
import pytest
from helpers import integrand, random_table

from maskdiff.elbo import negative_elbo, pattern_weights
from maskdiff.laws import markov_chain, potts_ring, sequences
from maskdiff.partial import conditional_table, partial_marginals, state_index


def test_pattern_weights_are_beta_integrals():
    length = 6
    nodes, weights = np.polynomial.legendre.leggauss(6)
    t, weights = (nodes + 1) / 2, weights / 2
    w = pattern_weights(length)
    assert w[0] == 0.0
    for m in range(1, length + 1):
        integral = (weights * t ** (m - 1) * (1 - t) ** (length - m)).sum()
        assert w[m] == pytest.approx(integral, rel=1e-12)


def test_weights_are_the_probabilities_of_the_set_before_a_position():
    length = 6
    w = pattern_weights(length)
    total = sum(comb(length - 1, u) * w[length - u] for u in range(length))
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize("floor", [0.05, 0.2])
def test_clipped_weights_match_numerical_integration(floor):
    length, n = 6, 2_000_000
    t = (np.arange(n) + 0.5) / n
    w = pattern_weights(length, floor)
    assert w[0] == 0.0
    for m in range(1, length + 1):
        direct = (t**m * (1 - t) ** (length - m) / np.maximum(t, floor)).mean()
        assert w[m] == pytest.approx(direct, rel=1e-6)


def test_a_tiny_floor_recovers_the_unclipped_weights():
    assert pattern_weights(6, 1e-9) == pytest.approx(pattern_weights(6), rel=1e-6)


def test_closed_form_equals_the_pattern_sum_integrated_over_t():
    length, vocab = 4, 3
    table = random_table(length, vocab, 0)
    elbo = negative_elbo(table, vocab, length)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    nodes, weights = (nodes + 1) / 2, weights / 2
    x = sequences(length, vocab)
    for index in np.random.default_rng(1).integers(0, len(x), size=25):
        seq = x[index]
        value = sum(
            weight * integrand(table, seq, t, vocab)
            for t, weight in zip(nodes, weights, strict=True)
        )
        assert elbo.reshape(-1)[index] == pytest.approx(value, rel=1e-12)


def test_closed_form_is_the_average_chain_rule_over_orders():
    length, vocab = 4, 3
    table = random_table(length, vocab, 2)
    elbo = negative_elbo(table, vocab, length).reshape(-1)
    x = sequences(length, vocab)
    orders = list(itertools.permutations(range(length)))
    for index in np.random.default_rng(3).integers(0, len(x), size=25):
        seq = x[index]
        total = 0.0
        for order in orders:
            state = np.full(length, vocab)
            for i in order:
                total -= np.log(table[state_index(state, vocab), i, seq[i]])
                state[i] = seq[i]
        assert elbo[index] == pytest.approx(total / len(orders), rel=1e-12)


@pytest.mark.parametrize("law", ["markov", "potts"])
@pytest.mark.parametrize("length,vocab", [(6, 3), (8, 2)])
def test_exact_conditionals_make_the_bound_tight_for_every_sequence(law, length, vocab):
    p = (
        markov_chain(length, vocab, 0.8)
        if law == "markov"
        else potts_ring(length, vocab, 1.0)
    )
    table = conditional_table(partial_marginals(p))
    assert negative_elbo(table, vocab, length) == pytest.approx(-np.log(p), abs=1e-11)
