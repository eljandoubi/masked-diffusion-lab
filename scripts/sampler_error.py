"""Exact KL between the data law and the K-step sampler, for several couplings.

Writes results/sampler_error.json and results/sampler_error.png.
"""

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maskdiff.asymptotics import leading_coefficient, tie_probability
from maskdiff.laws import markov_chain, potts_ring, total_correlation
from maskdiff.metrics import kl, loglog_slope
from maskdiff.partial import conditional_table, partial_marginals
from maskdiff.sampler import random_order_schedule, sampler_law, uniform_schedule

plt.switch_backend("Agg")

RESULTS = Path(__file__).resolve().parent.parent / "results"
GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
EXTENDED = [64, 128, 256, 512, 1024, 2048]
FIT = (16, 48)
FIT_EXTENDED = (512, 2048)
CASES = [
    ("potts", 6, 3, [0.0, 0.5, 1.0, 2.0]),
    ("markov", 6, 3, [0.0, 0.5, 0.8, 0.95]),
    ("potts", 8, 2, [0.0, 0.5, 1.0, 2.0]),
    ("markov", 8, 2, [0.0, 0.5, 0.8, 0.95]),
]
PARAM_NAME = {"potts": "coupling", "markov": "persistence"}


def make_law(family: str, length: int, vocab: int, param: float) -> np.ndarray:
    make = potts_ring if family == "potts" else markov_chain
    return make(length, vocab, param)


def run_case(family: str, length: int, vocab: int, param: float) -> dict:
    p = make_law(family, length, vocab, param)
    table = conditional_table(partial_marginals(p))

    def error(steps):
        schedule = uniform_schedule(steps, length)
        return kl(p, sampler_law(table, vocab, length, schedule))

    out = {
        "family": family,
        "length": length,
        "vocab": vocab,
        "param": param,
        "total_correlation": total_correlation(p),
        "kl": [error(k) for k in GRID],
        "tie_probability": [tie_probability(k, length) for k in GRID],
    }
    one_at_a_time = random_order_schedule([1] * length, length)
    out["one_at_a_time"] = kl(p, sampler_law(table, vocab, length, one_at_a_time))
    out["equal_count"] = {}
    for steps in range(1, length):
        if length % steps == 0:
            schedule = random_order_schedule([length // steps] * steps, length)
            out["equal_count"][str(steps)] = kl(
                p, sampler_law(table, vocab, length, schedule)
            )
    if out["total_correlation"] > 1e-9:
        out["kl_extended"] = [error(k) for k in EXTENDED]
        out["slope"], out["slope_r2"] = loglog_slope(GRID, out["kl"], *FIT)
        out["slope_extended"], out["slope_extended_r2"] = loglog_slope(
            EXTENDED, out["kl_extended"], *FIT_EXTENDED
        )
        out["coefficient"] = leading_coefficient(p)
    return out


def plot(cases: list[dict], path: Path) -> None:
    titles = {"potts": "Potts ring", "markov": "Markov chain"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, family in zip(axes, titles, strict=True):
        for case in cases:
            keep = (case["family"], case["length"], case["vocab"]) == (family, 6, 3)
            if not keep or "kl_extended" not in case:
                continue
            ks = GRID + EXTENDED
            values = case["kl"] + case["kl_extended"]
            label = f"{PARAM_NAME[family]} {case['param']}"
            (line,) = ax.loglog(ks, values, "o-", ms=3, label=label)
            tail = np.array([16.0, 2048.0])
            ax.loglog(tail, case["coefficient"] / tail**2, "--", color=line.get_color())
        ax.set_title(f"{titles[family]}, L = 6, V = 3")
        ax.set_xlabel("sampler steps K")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="dashed: c / K^2", fontsize=8)
    axes[0].set_ylabel("KL(data || sampler)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    start = time.time()
    cases = [
        run_case(family, length, vocab, param)
        for family, length, vocab, params in CASES
        for param in params
    ]
    report = {
        "grid": GRID,
        "extended_grid": EXTENDED,
        "fit_range": list(FIT),
        "fit_range_extended": list(FIT_EXTENDED),
        "cases": cases,
        "load_average": os.getloadavg()[0],
        "wall_seconds": time.time() - start,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sampler_error.json").write_text(json.dumps(report, indent=1))
    plot(cases, RESULTS / "sampler_error.png")
    print(RESULTS / "sampler_error.json")
    print(RESULTS / "sampler_error.png")


if __name__ == "__main__":
    main()
