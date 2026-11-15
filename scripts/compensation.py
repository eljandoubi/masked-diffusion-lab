"""Conditionals that are not the exact ones but give a smaller K-step error.

usage: python scripts/compensation.py
Writes results/compensation.json. For the Markov law of the denoiser study it holds
the conditional KL and the K = 16 sampler KL of the exact conditionals of chains with
another persistence. For two positions it holds the persistence that makes the
sampler output equal to the data law.
"""

import json
from pathlib import Path

import numpy as np

from maskdiff.laws import markov_chain
from maskdiff.metrics import Reference
from maskdiff.sampler import sampler_law, uniform_schedule

RESULTS = Path(__file__).resolve().parent.parent / "results"
STEPS = 16
LENGTH, VOCAB, PERSISTENCE = 6, 3, 0.8
OTHER_PERSISTENCES = [0.8, 0.81, 0.82, 0.84, 0.86, 0.88, 0.9]
PAIR_PERSISTENCE = 0.4


def markov_rows() -> list[dict]:
    target = Reference.from_law(markov_chain(LENGTH, VOCAB, PERSISTENCE))
    rows = []
    for rho in OTHER_PERSISTENCES:
        table = Reference.from_law(markov_chain(LENGTH, VOCAB, rho)).table
        rows.append(
            {
                "persistence": rho,
                "cond_kl": target.conditional_kl(table)[0],
                "sampler_kl": target.sampler_kl(table, STEPS),
            }
        )
    return rows


def pair_row() -> dict:
    law = markov_chain(2, 2, PAIR_PERSISTENCE)
    target = Reference.from_law(law)
    rho = PAIR_PERSISTENCE / (1 - 1 / STEPS)
    table = Reference.from_law(markov_chain(2, 2, rho)).table
    output = sampler_law(table, 2, 2, uniform_schedule(STEPS, 2))
    return {
        "persistence": PAIR_PERSISTENCE,
        "other_persistence": rho,
        "exact_sampler_kl": target.sampler_kl(target.table, STEPS),
        "other_cond_kl": target.conditional_kl(table)[0],
        "other_sampler_kl": target.sampler_kl(table, STEPS),
        "max_law_difference": float(np.abs(output - law).max()),
    }


def main() -> None:
    law = f"markov (persistence {PERSISTENCE}), L = {LENGTH}, V = {VOCAB}"
    report = {
        "steps": STEPS,
        "markov": {"law": law, "rows": markov_rows()},
        "pair": pair_row(),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "compensation.json"
    path.write_text(json.dumps(report, indent=1))
    print(path)


if __name__ == "__main__":
    main()
