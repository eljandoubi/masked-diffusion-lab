import itertools

import numpy as np
import pytest

from maskdiff.laws import markov_chain, potts_ring, product_of_marginals
from maskdiff.partial import conditional_table, partial_marginals
from maskdiff.sampler import sampler_law, transitions, uniform_schedule


def exact_table(p):
    return conditional_table(partial_marginals(p))


def known_mass(p, x, known):
    """P(x_j for j in known), summing p over the other positions."""
    index = tuple(x[j] if j in known else slice(None) for j in range(p.ndim))
    return p[index].sum()


def brute_force_law(p, steps):
    """Average over the reveal steps tau of the product of exact conditionals."""
    length, vocab = p.ndim, p.shape[0]
    law = np.zeros(p.shape)
    for tau in itertools.product(range(steps), repeat=length):
        for x in itertools.product(range(vocab), repeat=length):
            prob = 1.0
            for i in range(length):
                before = [j for j in range(length) if tau[j] < tau[i]]
                prob *= known_mass(p, x, before + [i]) / known_mass(p, x, before)
            law[x] += prob / steps**length
    return law


@pytest.mark.parametrize("vocab,length", [(2, 4), (3, 3)])
def test_every_partial_sequence_has_a_complete_set_of_moves(vocab, length):
    moves = transitions(vocab, length)
    assert len(moves.rows) == (2 * vocab + 1) ** length
    table = exact_table(potts_ring(length, vocab, 0.8))
    weight = np.append(table.reshape(-1), 1.0)[moves.factors].prod(axis=1)
    for step in uniform_schedule(3, length):
        out = weight * step[moves.masked, moves.revealed]
        total = np.bincount(moves.rows, weights=out, minlength=(vocab + 1) ** length)
        assert total == pytest.approx(1.0)


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 8])
def test_the_output_law_sums_to_one(steps):
    p = markov_chain(6, 3, 0.8)
    q = sampler_law(exact_table(p), 3, 6, uniform_schedule(steps, 6))
    assert q.shape == p.shape
    assert q.sum() == pytest.approx(1.0, abs=1e-12)
    assert (q > 0).all()


def test_an_incomplete_schedule_is_refused():
    table = exact_table(markov_chain(4, 2, 0.5))
    with pytest.raises(ValueError):
        sampler_law(table, 2, 4, uniform_schedule(3, 4)[:2])


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("make", [markov_chain, potts_ring])
def test_a_product_law_is_reproduced_at_every_step_count(make, steps):
    p = make(6, 3, 0.0)
    q = sampler_law(exact_table(p), 3, 6, uniform_schedule(steps, 6))
    assert q == pytest.approx(p, abs=1e-14)


@pytest.mark.parametrize("make", [markov_chain, potts_ring])
def test_one_step_returns_the_product_of_the_marginals(make):
    p = make(6, 3, 0.9)
    q = sampler_law(exact_table(p), 3, 6, uniform_schedule(1, 6))
    assert q == pytest.approx(product_of_marginals(p), abs=1e-14)
    assert np.abs(q - p).max() > 1e-3


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_dynamic_program_equals_the_sum_over_reveal_steps(steps):
    p = markov_chain(4, 2, 0.7)
    q = sampler_law(exact_table(p), 2, 4, uniform_schedule(steps, 4))
    assert q == pytest.approx(brute_force_law(p, steps), abs=1e-13)


def test_dynamic_program_equals_the_sum_over_reveal_steps_on_a_ring():
    p = potts_ring(4, 3, 0.9)
    q = sampler_law(exact_table(p), 3, 4, uniform_schedule(3, 4))
    assert q == pytest.approx(brute_force_law(p, 3), abs=1e-13)


@pytest.mark.parametrize("steps", [1, 2, 3, 7])
def test_two_positions_mix_the_law_with_the_product_of_marginals(steps):
    p = markov_chain(2, 3, 0.8)
    q = sampler_law(exact_table(p), 3, 2, uniform_schedule(steps, 2))
    expected = (1 - 1 / steps) * p + product_of_marginals(p) / steps
    assert q == pytest.approx(expected, abs=1e-14)
