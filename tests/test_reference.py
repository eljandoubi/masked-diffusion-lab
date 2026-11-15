import itertools

import numpy as np
import pytest
from helpers import random_table

from maskdiff.elbo import negative_elbo
from maskdiff.laws import entropy, markov_chain, potts_ring, total_correlation
from maskdiff.metrics import Reference
from maskdiff.partial import state_index


def softmax(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("name", ["markov", "potts"])
def test_gap_is_the_expected_negative_elbo_minus_the_entropy(name):
    law = markov_chain(5, 3, 0.8) if name == "markov" else potts_ring(4, 3, 1.0)
    ref = Reference.from_law(law)
    model = random_table(ref.length, ref.vocab, 0)
    expected_loss = float((law * negative_elbo(model, ref.vocab, ref.length)).sum())
    _, gap = ref.conditional_kl(model)
    assert gap == pytest.approx(expected_loss - entropy(law), rel=1e-10)


def test_conditional_kl_averages_over_t_and_the_masked_positions():
    length, vocab = 4, 3
    law = markov_chain(length, vocab, 0.7)
    ref = Reference.from_law(law)
    model = random_table(length, vocab, 1)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    nodes, weights = (nodes + 1) / 2, weights / 2
    total = 0.0
    for t, weight in zip(nodes, weights, strict=True):
        for mask in itertools.product([0, 1], repeat=length):
            m = sum(mask)
            if m == 0:
                continue
            inner = 0.0
            for x in itertools.product(range(vocab), repeat=length):
                s = state_index(np.where(mask, vocab, x), vocab)
                kl_sum = sum(
                    (ref.table[s, i] * np.log(ref.table[s, i] / model[s, i])).sum()
                    for i in range(length)
                    if mask[i]
                )
                inner += law[x] * kl_sum / m
            total += weight * t**m * (1 - t) ** (length - m) * inner
    conditional, gap = ref.conditional_kl(model)
    assert conditional == pytest.approx(total, rel=1e-10)
    assert gap == pytest.approx((length + 1) * total, rel=1e-10)


def test_exact_conditionals_have_no_error():
    ref = Reference.from_law(potts_ring(5, 3, 1.0))
    conditional, gap = ref.conditional_kl(ref.table)
    assert abs(conditional) < 1e-15 and abs(gap) < 1e-15


def test_error_grows_quadratically_with_a_small_perturbation_of_the_logits():
    ref = Reference.from_law(markov_chain(5, 3, 0.8))
    noise = np.random.default_rng(2).standard_normal(ref.table.shape)
    base = np.log(np.where(ref.table > 0, ref.table, 1e-12))
    small = ref.conditional_kl(softmax(base + 1e-3 * noise))[0]
    double = ref.conditional_kl(softmax(base + 2e-3 * noise))[0]
    assert small > 0
    assert double / small == pytest.approx(4.0, rel=0.02)


def test_sampler_error_with_exact_conditionals_at_one_step_is_the_total_correlation():
    law = markov_chain(6, 3, 0.8)
    ref = Reference.from_law(law)
    assert ref.sampler_kl(ref.table, 1) == pytest.approx(total_correlation(law))
    assert ref.sampler_kl(ref.table, 8) < ref.sampler_kl(ref.table, 2)
