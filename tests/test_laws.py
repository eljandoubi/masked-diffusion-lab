import numpy as np
import pytest

from maskdiff.laws import (
    draw,
    entropy,
    marginal,
    markov_chain,
    potts_ring,
    product_of_marginals,
    sequences,
    total_correlation,
)


def test_sequences_are_all_distinct_in_c_order():
    x = sequences(3, 2)
    assert x.shape == (8, 3)
    assert x[0].tolist() == [0, 0, 0] and x[1].tolist() == [0, 0, 1]
    assert len({tuple(row) for row in x}) == 8


@pytest.mark.parametrize("persistence", [0.0, 0.3, 0.9])
@pytest.mark.parametrize("length,vocab", [(6, 3), (8, 2)])
def test_markov_probabilities_sum_to_one(length, vocab, persistence):
    p = markov_chain(length, vocab, persistence)
    assert p.shape == (vocab,) * length
    assert p.sum() == pytest.approx(1.0, abs=1e-14)
    assert (p > 0).all()


@pytest.mark.parametrize("coupling", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("length,vocab", [(6, 3), (8, 2)])
def test_potts_probabilities_sum_to_one(length, vocab, coupling):
    p = potts_ring(length, vocab, coupling)
    assert p.shape == (vocab,) * length
    assert p.sum() == pytest.approx(1.0, abs=1e-14)
    assert (p > 0).all()


def test_markov_matches_the_product_formula():
    kernel = np.array([[0.8, 0.2], [0.2, 0.8]])
    p = markov_chain(3, 2, 0.6)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                assert p[a, b, c] == pytest.approx(0.5 * kernel[a, b] * kernel[b, c])


def test_markov_is_stationary_with_uniform_marginals():
    p = markov_chain(6, 3, 0.8)
    for i in range(6):
        assert marginal(p, i) == pytest.approx(np.full(3, 1 / 3), abs=1e-14)


def test_zero_coupling_gives_product_laws():
    for p in (markov_chain(6, 3, 0.0), potts_ring(6, 3, 0.0)):
        assert p == pytest.approx(np.full(p.shape, 3.0**-6), abs=1e-16)
        assert total_correlation(p) == pytest.approx(0.0, abs=1e-12)


def test_total_correlation_grows_with_coupling():
    chain = [total_correlation(markov_chain(6, 3, r)) for r in (0.2, 0.5, 0.8)]
    ring = [total_correlation(potts_ring(6, 3, b)) for b in (0.2, 0.5, 1.0)]
    assert chain == sorted(chain) and ring == sorted(ring)


def test_total_correlation_is_the_kl_to_the_product_of_marginals():
    p = potts_ring(5, 3, 0.7)
    q = product_of_marginals(p)
    assert q.sum() == pytest.approx(1.0)
    assert total_correlation(p) == pytest.approx(float((p * np.log(p / q)).sum()))


def test_two_state_potts_ring_is_the_ising_ring():
    beta, length = 0.8, 5
    p = potts_ring(length, 2, beta)
    spins = 2.0 * sequences(length, 2) - 1.0
    weight = np.exp(beta / 2 * (spins * np.roll(spins, -1, axis=1)).sum(axis=1))
    assert p.reshape(-1) == pytest.approx(weight / weight.sum())


def test_ring_is_invariant_under_a_cyclic_shift():
    p = potts_ring(6, 3, 0.9)
    shifted = p.transpose(1, 2, 3, 4, 5, 0)
    assert shifted == pytest.approx(p)


def test_entropy_of_the_uniform_law():
    assert entropy(np.full(27, 1 / 27)) == pytest.approx(np.log(27))


def test_draw_returns_sequences_with_the_law_as_frequencies():
    p = potts_ring(4, 3, 0.8)
    size = 400_000
    x = draw(p, size, np.random.default_rng(0))
    assert x.shape == (size, 4) and x.min() == 0 and x.max() == 2
    freq = np.bincount(np.ravel_multi_index(x.T, p.shape), minlength=p.size) / size
    assert np.abs(freq - p.reshape(-1)).max() < 5 * np.sqrt(p.max() / size)
