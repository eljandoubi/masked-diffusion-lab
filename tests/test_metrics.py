import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from maskdiff.laws import markov_chain, potts_ring, total_correlation
from maskdiff.metrics import T_975, kl, loglog_slope, paired_interval
from maskdiff.partial import conditional_table, partial_marginals
from maskdiff.sampler import random_order_schedule, sampler_law, uniform_schedule


def decimal_kl(p, q):
    """KL(p || q) in 60-digit arithmetic on the float values of p and q."""
    with localcontext() as context:
        context.prec = 60
        total = Decimal(0)
        for a, b in zip(p, q, strict=True):
            if a > 0:
                total += Decimal(float(a)) * (
                    Decimal(float(a)).ln() - Decimal(float(b)).ln()
                )
        return float(total)


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
    assert kl(p, q) == float(np.sum(p * np.log1p((p - q) / q)))


def test_kl_of_a_rare_bin_is_finite_and_not_negative_infinity():
    p, q = np.array([1e-20, 1.0]), np.array([0.5, 0.5])
    assert p.sum() == 1.0 and q.sum() == 1.0
    assert kl(p, q) == pytest.approx(math.log(2), rel=1e-12)
    assert kl(p, q) == pytest.approx(decimal_kl(p, q), rel=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_kl_of_imbalanced_laws_matches_high_precision_arithmetic(seed):
    rng = np.random.default_rng(seed)
    p = np.exp(rng.uniform(-80, 0, size=40))
    p /= p.sum()
    q = p * np.exp(rng.uniform(-45, 45, size=40))
    q[20:] *= p[20:].sum() / q[20:].sum()
    q[:20] = p[:20] * (1 + 1e-6 * rng.standard_normal(20))
    q /= q.sum()
    assert kl(p, q) == pytest.approx(decimal_kl(p, q), rel=1e-11)
    assert kl(q, p) == pytest.approx(decimal_kl(q, p), rel=1e-11)


def test_kl_is_infinite_when_the_model_misses_the_support_of_the_data():
    p, q = np.array([0.5, 0.5]), np.array([1.0, 0.0])
    with np.errstate(all="raise"):
        assert kl(p, q) == math.inf


def test_kl_does_not_count_bins_without_data_mass():
    q = np.array([0.5, 0.25, 0.25])
    assert kl(np.array([0.0, 0.25, 0.75]), q) == pytest.approx(0.75 * math.log(3))
    assert kl(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.5, 0.5])) == math.log(2)
    assert kl(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0


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


def test_paired_interval_takes_the_differences_of_the_pairs():
    diff, mean, half = paired_interval([1.0, 2.0, 3.0], [2.0, 4.0, 5.0])
    assert diff.tolist() == [1.0, 2.0, 2.0]
    assert mean == pytest.approx(5 / 3)
    assert half == pytest.approx(T_975[2] * math.sqrt(1 / 3) / math.sqrt(3))
    assert (mean - half, mean + half) == pytest.approx((0.2323, 3.1010), abs=1e-3)


def test_paired_interval_is_blind_to_what_the_two_runs_of_a_pair_share():
    a = np.array([1.0, 2.0, 4.0])
    b = a + np.array([0.1, 0.2, 0.15])
    shared = np.array([0.0, 50.0, -30.0])
    plain, shifted = paired_interval(a, b), paired_interval(a + shared, b + shared)
    assert shifted[0] == pytest.approx(plain[0])
    assert shifted[1:] == pytest.approx(plain[1:])


def test_paired_interval_needs_between_two_and_six_pairs():
    with pytest.raises(ValueError):
        paired_interval([1.0], [2.0])
    with pytest.raises(ValueError):
        paired_interval(np.ones(7), np.ones(7))


@pytest.mark.parametrize("pairs", [2, 3, 5])
def test_the_t_interval_covers_the_true_mean_95_percent_of_the_time(pairs):
    rng = np.random.default_rng(pairs)
    samples = rng.standard_normal((40_000, pairs))
    half = T_975[pairs - 1] * samples.std(axis=1, ddof=1) / np.sqrt(pairs)
    covered = (np.abs(samples.mean(axis=1)) <= half).mean()
    assert covered == pytest.approx(0.95, abs=0.006)
