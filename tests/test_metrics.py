import numpy as np
import pytest

from maskdiff.laws import markov_chain, potts_ring, total_correlation
from maskdiff.metrics import kl, loglog_slope
from maskdiff.partial import conditional_table, partial_marginals
from maskdiff.sampler import random_order_schedule, sampler_law, uniform_schedule


def test_kl_of_a_law_with_itself_is_zero_and_matches_the_definition():
    rng = np.random.default_rng(0)
    p = rng.random(20) + 0.1
    q = rng.random(20) + 0.1
    p, q = p / p.sum(), q / q.sum()
    assert kl(p, p) == 0.0
    assert kl(p, q) == pytest.approx(float((p * np.log(p / q)).sum()), rel=1e-12)
    assert kl(p, q) > 0


def test_kl_keeps_its_accuracy_for_nearby_laws():
    p = np.full(8, 1 / 8)
    delta = 1e-7 * np.array([1, -1, 1, -1, 1, -1, 1, -1])
    q = p * (1 + delta)
    assert kl(p, q) == pytest.approx(0.5 * (delta**2).mean(), rel=1e-6)


@pytest.mark.parametrize(
    "p",
    [markov_chain(6, 3, 0.8), potts_ring(6, 3, 1.0), potts_ring(8, 2, 1.5)],
)
def test_one_step_error_is_the_total_correlation(p):
    vocab, length = p.shape[0], p.ndim
    table = conditional_table(partial_marginals(p))
    q = sampler_law(table, vocab, length, uniform_schedule(1, length))
    assert kl(p, q) == pytest.approx(total_correlation(p), rel=1e-11)


def test_error_vanishes_when_one_position_is_revealed_per_step():
    p = potts_ring(6, 3, 1.0)
    table = conditional_table(partial_marginals(p))
    q = sampler_law(table, 3, 6, random_order_schedule([1] * 6, 6))
    assert abs(kl(p, q)) < 1e-14


def test_error_shrinks_as_the_grid_is_refined():
    p = potts_ring(6, 3, 1.0)
    table = conditional_table(partial_marginals(p))
    errors = [
        kl(p, sampler_law(table, 3, 6, uniform_schedule(k, 6)))
        for k in (1, 2, 4, 8, 16, 32)
    ]
    assert all(a > b > 0 for a, b in zip(errors, errors[1:], strict=False))


def test_slope_of_a_power_law_and_the_fit_range():
    x = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
    y = 3.0 * x**-2
    assert loglog_slope(x, y, 1, 64) == pytest.approx((-2.0, 1.0))
    y_bent = np.where(x < 8, 1.0, y)
    slope, _ = loglog_slope(x, y_bent, 8, 64)
    assert slope == pytest.approx(-2.0)
    with pytest.raises(ValueError):
        loglog_slope(x, y, 60, 64)
