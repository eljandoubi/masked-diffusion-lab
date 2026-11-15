import numpy as np
import pytest

from maskdiff.asymptotics import (
    first_order_change,
    leading_coefficient,
    tie_probability,
)
from maskdiff.laws import markov_chain, potts_ring, product_of_marginals
from maskdiff.metrics import kl, loglog_slope
from maskdiff.partial import conditional_table, partial_marginals
from maskdiff.sampler import sampler_law, uniform_schedule


def exact_law(p, steps):
    table = conditional_table(partial_marginals(p))
    return sampler_law(table, p.shape[0], p.ndim, uniform_schedule(steps, p.ndim))


def test_tie_probability_counts_injective_assignments():
    assert tie_probability(5, 3) == pytest.approx(1 - 5 * 4 * 3 / 5**3)
    assert tie_probability(2, 3) == 1.0
    assert tie_probability(48, 1) == 0.0
    assert tie_probability(1000, 6) == pytest.approx(15 / 1000, rel=0.01)


@pytest.mark.parametrize("steps", [5, 8, 12, 20])
def test_sampler_law_is_the_data_law_mixed_with_a_tie_law(steps):
    p = potts_ring(4, 3, 0.9)
    q = exact_law(p, steps)
    eps = tie_probability(steps, 4)
    tie_law = (q - (1 - eps) * p) / eps
    assert (tie_law > 0).all()
    assert tie_law.sum() == pytest.approx(1.0)
    chi2 = float((tie_law**2 / p).sum() - 1.0)
    assert 0 < kl(p, q) <= np.log1p(eps**2 * chi2 / (1 - eps))


def test_two_position_coefficient_has_a_closed_form():
    p = markov_chain(2, 3, 0.8)
    expected = 0.5 * float(((product_of_marginals(p) - p) ** 2 / p).sum())
    assert leading_coefficient(p) == pytest.approx(expected, rel=1e-12)


def test_product_laws_have_no_leading_error():
    p = potts_ring(5, 3, 0.0)
    assert np.abs(first_order_change(p)).max() < 1e-14
    assert leading_coefficient(p) < 1e-28


@pytest.mark.parametrize("p", [potts_ring(4, 3, 1.0), markov_chain(4, 3, 0.8)])
def test_coefficient_predicts_the_error_at_large_k(p):
    c = leading_coefficient(p)
    steps = np.array([64, 256, 1024, 2048])
    errors = np.array([kl(p, exact_law(p, k)) for k in steps])
    ratio = steps**2 * errors / c
    assert abs(ratio[-1] - 1) < 5e-3
    assert abs(ratio[-1] - 1) < abs(ratio[1] - 1) / 3
    slope, _ = loglog_slope(steps, errors, 256, 2048)
    assert slope == pytest.approx(-2.0, abs=0.01)
