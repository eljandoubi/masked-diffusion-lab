"""Plug-in baseline on the training samples of the denoiser study.

usage: python scripts/baseline.py
Writes results/baseline.json. For every law, sample size and seed of the study it holds
the error of the smoothed empirical conditional frequencies (alpha added to each count,
three values fixed in advance) and the smallest value of the training loss.
"""

import json
from pathlib import Path

from denoiser_study import FLOOR, LAWS, SAMPLE_SIZES, SAMPLER_STEPS, VOCAB, reference

from maskdiff.elbo import minimum_loss
from maskdiff.laws import empirical_law, training_sample
from maskdiff.partial import plug_in_table

RESULTS = Path(__file__).resolve().parent.parent / "results"
ALPHAS = [0.01, 0.1, 1.0]
SEEDS = 3


def law_rows(law: str) -> dict:
    ref = reference(law)
    runs, minimum = [], []
    for n in SAMPLE_SIZES:
        for seed in range(SEEDS):
            data = training_sample(ref.law, n, seed)
            loss = minimum_loss(empirical_law(data, VOCAB), FLOOR)
            minimum.append({"n": n, "seed": seed, "loss": loss})
            for alpha in ALPHAS:
                table = plug_in_table(data, VOCAB, alpha)
                runs.append(
                    {
                        "n": n,
                        "seed": seed,
                        "alpha": alpha,
                        "cond_kl": ref.conditional_kl(table)[0],
                        "sampler_kl": ref.sampler_kl(table, SAMPLER_STEPS),
                    }
                )
    return {"runs": runs, "minimum_loss": minimum}


def main() -> None:
    report = {
        "config": {
            "alphas": ALPHAS,
            "sample_sizes": SAMPLE_SIZES,
            "seeds": SEEDS,
            "sampler_steps": SAMPLER_STEPS,
            "weight_floor": FLOOR,
        },
        **{law: law_rows(law) for law in LAWS},
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "baseline.json"
    path.write_text(json.dumps(report, indent=1))
    print(path)


if __name__ == "__main__":
    main()
