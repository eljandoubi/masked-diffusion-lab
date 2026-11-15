import math

import numpy as np
import pytest

from maskdiff.laws import markov_chain
from maskdiff.metrics import Reference
from maskdiff.sampler import sampler_law, uniform_schedule


def chain_error(steps, rho):
    """Closed-form KL of the sampler on the exact conditionals, two positions."""
    kept = rho * (1 - 1 / steps)
    same = (1 + rho) / 2 * math.log((1 + rho) / (1 + kept))
    apart = (1 - rho) / 2 * math.log((1 - rho) / (1 - kept))
    return same + apart


def test_conditionals_of_a_stronger_chain_beat_the_exact_ones_at_sixteen_steps():
    target = Reference.from_law(markov_chain(6, 3, 0.8))
    other = Reference.from_law(markov_chain(6, 3, 0.81)).table
    exact_error = target.sampler_kl(target.table, 16)
    other_error = target.sampler_kl(other, 16)
    assert target.conditional_kl(other)[0] > 1e-4
    assert other_error < 0.7 * exact_error


def test_the_error_is_not_monotone_in_the_distance_to_the_exact_conditionals():
    target = Reference.from_law(markov_chain(6, 3, 0.8))
    errors = {}
    for rho in (0.8, 0.84, 0.9):
        table = Reference.from_law(markov_chain(6, 3, rho)).table
        errors[rho] = (target.conditional_kl(table)[0], target.sampler_kl(table, 16))
    assert errors[0.8][0] < errors[0.84][0] < errors[0.9][0]
    assert errors[0.84][1] < errors[0.8][1] < errors[0.9][1]


@pytest.mark.parametrize(
    ("rho", "steps"), [(0.2, 2), (0.4, 4), (0.7, 4), (0.4, 16), (0.9, 16)]
)
def test_two_positions_have_conditionals_that_make_the_sampler_exact(rho, steps):
    law = markov_chain(2, 2, rho)
    target = Reference.from_law(law)
    stronger = Reference.from_law(markov_chain(2, 2, rho / (1 - 1 / steps))).table
    output = sampler_law(stronger, 2, 2, uniform_schedule(steps, 2))
    assert np.abs(output - law).max() < 1e-14
    assert abs(target.sampler_kl(stronger, steps)) < 1e-14
    assert target.conditional_kl(stronger)[0] > 1e-6
    assert target.sampler_kl(target.table, steps) == pytest.approx(
        chain_error(steps, rho), rel=1e-9
    )
