import itertools

import numpy as np
import pytest
from helpers import integrand, random_table

from maskdiff.elbo import negative_elbo
from maskdiff.laws import markov_chain, sequences
from maskdiff.partial import conditional_table, partial_marginals, state_index
from maskdiff.sampler import sampler_law, uniform_schedule


def path_negative_elbo(table, seq, steps, vocab):
    """-ELBO of the K-step sampler as a model, by summing over reveal steps tau.

    Along the forward path the ratio of model and true step probabilities is
    q_i(x_i | tokens revealed before step tau_i) for the position revealed at that
    step, and one everywhere else.
    """
    length = len(seq)
    total = 0.0
    for tau in itertools.product(range(steps), repeat=length):
        for i in range(length):
            state = np.where(np.array(tau) < tau[i], seq, vocab)
            total -= np.log(table[state_index(state, vocab), i, seq[i]])
    return total / steps**length


def riemann_sum(table, seq, steps, vocab):
    """(1/K) sum over the grid t_k = (K - k) / K of the integrand of (1)."""
    grid = [(steps - k) / steps for k in range(steps)]
    return sum(integrand(table, seq, t, vocab) for t in grid) / steps


@pytest.mark.parametrize("steps", [1, 2, 3, 5])
def test_path_elbo_is_a_riemann_sum_of_the_continuous_time_integrand(steps):
    table = random_table(3, 2, 0)
    for seq in sequences(3, 2):
        direct = path_negative_elbo(table, seq, steps, 2)
        assert direct == pytest.approx(riemann_sum(table, seq, steps, 2), rel=1e-12)


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_path_elbo_is_a_bound_on_the_sampler_log_likelihood(steps):
    vocab, length = 2, 3
    for table in (
        random_table(length, vocab, 1),
        conditional_table(partial_marginals(markov_chain(length, vocab, 0.8))),
    ):
        law = sampler_law(table, vocab, length, uniform_schedule(steps, length))
        for seq in sequences(length, vocab):
            bound = path_negative_elbo(table, seq, steps, vocab)
            assert bound >= -np.log(law[tuple(seq)]) - 1e-12
            if steps == 1:
                assert bound == pytest.approx(-np.log(law[tuple(seq)]), rel=1e-12)


@pytest.mark.parametrize("exact", [False, True])
def test_riemann_sums_approach_the_closed_form_at_first_order(exact):
    vocab, length = 2, 3
    if exact:
        table = conditional_table(partial_marginals(markov_chain(length, vocab, 0.8)))
    else:
        table = random_table(length, vocab, 2)
    closed = negative_elbo(table, vocab, length)
    errors = []
    for steps in (50, 100, 200):
        gap = [
            riemann_sum(table, seq, steps, vocab) - closed[tuple(seq)]
            for seq in sequences(length, vocab)
        ]
        errors.append(np.abs(gap).max())
    assert errors[1] == pytest.approx(errors[0] / 2, rel=0.05)
    assert errors[2] == pytest.approx(errors[1] / 2, rel=0.05)
