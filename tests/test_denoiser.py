import numpy as np
import pytest
import torch

from maskdiff.denoiser import (
    Denoiser,
    masked_loss,
    model_table,
    train_exact,
    train_sampled,
)
from maskdiff.elbo import negative_elbo
from maskdiff.laws import draw, empirical_law, markov_chain
from maskdiff.metrics import Reference
from maskdiff.partial import mask_tokens


def test_the_network_returns_one_logit_vector_per_position():
    model = Denoiser(6, 3, 16)
    logits = model(torch.randint(0, 4, (5, 6)))
    assert logits.shape == (5, 6, 3)


def test_the_conditionals_of_the_network_are_distributions_on_every_partial_sequence():
    table = model_table(Denoiser(5, 3, 16))
    assert table.shape == (4**5, 5, 3)
    assert table.min() > 0
    assert table.sum(axis=-1) == pytest.approx(1.0, abs=1e-12)


def test_masked_loss_counts_only_masked_positions_and_clips_the_weight():
    logits = torch.zeros(2, 3, 4)
    x = torch.tensor([[0, 1, 2], [3, 3, 3]])
    masked = torch.tensor([[True, False, True], [False, False, False]])
    t = torch.tensor([0.5, 0.01])
    out = masked_loss(logits, x, masked, t, floor=0.1)
    assert out.tolist() == pytest.approx([2 * np.log(4) / 0.5, 0.0], rel=1e-6)
    t = torch.tensor([0.02, 0.5])
    masked = torch.tensor([[True, False, False], [False, False, True]])
    out = masked_loss(logits, x, masked, t, floor=0.1)
    assert out.tolist() == pytest.approx([np.log(4) / 0.1, np.log(4) / 0.5], rel=1e-6)


def test_training_loss_is_an_unbiased_estimate_of_its_exact_expectation():
    length, vocab, floor, n = 4, 3, 0.1, 400_000
    law = markov_chain(length, vocab, 0.8)
    torch.manual_seed(0)
    model = Denoiser(length, vocab, 16)
    exact = float((law * negative_elbo(model_table(model), vocab, length, floor)).sum())
    rng = np.random.default_rng(0)
    x = draw(law, n, rng)
    t = rng.random(n)
    tokens, masked = mask_tokens(x, t, rng, vocab)
    with torch.no_grad():
        per_sequence = masked_loss(
            model(torch.as_tensor(tokens)),
            torch.as_tensor(x),
            torch.as_tensor(masked),
            torch.as_tensor(t),
            floor,
        ).numpy()
    standard_error = per_sequence.std() / np.sqrt(n)
    assert abs(per_sequence.mean() - exact) < 4 * standard_error


def test_a_short_training_run_lowers_the_conditional_kl():
    law = markov_chain(5, 3, 0.8)
    ref = Reference.from_law(law)
    data = draw(law, 20_000, np.random.default_rng(0))
    torch.manual_seed(0)
    before = ref.conditional_kl(model_table(Denoiser(5, 3, 32)))[0]
    model, final_loss = train_sampled(data, 3, 32, steps=400, seed=0)
    after = ref.conditional_kl(model_table(model))[0]
    assert after < before / 5
    assert np.isfinite(final_loss)


def test_sampled_training_is_reproducible_for_a_seed():
    data = draw(markov_chain(4, 3, 0.8), 2000, np.random.default_rng(1))
    first, _ = train_sampled(data, 3, 16, steps=30, seed=3)
    second, _ = train_sampled(data, 3, 16, steps=30, seed=3)
    assert model_table(first) == pytest.approx(model_table(second), abs=1e-12)


def test_exact_training_returns_the_exact_expected_loss_of_its_model():
    law = empirical_law(
        draw(markov_chain(4, 3, 0.8), 3000, np.random.default_rng(2)), 3
    )
    model, loss = train_exact(law, 16, seed=0, iterations=60, floor=0.05)
    table = model_table(model)
    expected = float((law * negative_elbo(table, 3, 4, 0.05)).sum())
    assert next(model.parameters()).dtype == torch.float64
    assert loss == pytest.approx(expected, rel=1e-9)


def test_exact_training_on_the_data_law_learns_its_conditionals():
    law = markov_chain(5, 3, 0.8)
    ref = Reference.from_law(law)
    torch.manual_seed(0)
    before = ref.conditional_kl(model_table(Denoiser(5, 3, 32)))[0]
    model, _ = train_exact(law, 32, seed=0, iterations=300)
    assert ref.conditional_kl(model_table(model))[0] < before / 20


def test_exact_training_is_deterministic():
    law = markov_chain(4, 3, 0.8)
    first, _ = train_exact(law, 16, seed=1, iterations=40)
    second, _ = train_exact(law, 16, seed=1, iterations=40)
    assert model_table(first) == pytest.approx(model_table(second), abs=1e-12)
