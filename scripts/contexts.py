"""Error of the learned denoisers on contexts that the training sequences reach or miss.

usage: python scripts/contexts.py [--workers N]
Writes results/contexts.json. For both laws, the four widths, n = 200 and 2000 and 3
seeds (the same runs as the main grid) it holds the conditional KL on the partial
sequences whose known positions match a training sequence and on the others, the same
split for the plug-in tables, and the share of the error weight on the others.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from denoiser_study import FLOOR, ITERATIONS, LAWS, VOCAB, WIDTHS, reference

from maskdiff.denoiser import model_table, train_exact
from maskdiff.laws import empirical_law, training_sample
from maskdiff.partial import plug_in_table, reached

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIZES = [200, 2000]
ALPHAS = [0.01, 0.1, 1.0]
SEEDS = 3


def split(law: str, table: np.ndarray, data: np.ndarray) -> dict:
    ref, seen = reference(law), reached(data, VOCAB)
    return {
        "cond_kl": ref.conditional_kl(table)[0],
        "cond_kl_reached": ref.conditional_kl(table, seen)[0],
        "cond_kl_unreached": ref.conditional_kl(table, ~seen)[0],
    }


def run_job(job: tuple[str, int, int, int]) -> dict:
    """Train one network of the main grid again and split its error."""
    law, width, n, seed = job
    torch.set_num_threads(1)
    data = training_sample(reference(law).law, n, seed)
    model, _, used = train_exact(
        empirical_law(data, VOCAB), width, seed, ITERATIONS, FLOOR
    )
    row = {"law": law, "width": width, "n": n, "seed": seed, "iterations_used": used}
    return row | split(law, model_table(model), data)


def plug_in_rows() -> list[dict]:
    rows = []
    for law in LAWS:
        for n in SIZES:
            for seed in range(SEEDS):
                data = training_sample(reference(law).law, n, seed)
                for alpha in ALPHAS:
                    table = plug_in_table(data, VOCAB, alpha)
                    key = {"law": law, "n": n, "seed": seed, "alpha": alpha}
                    rows.append(key | split(law, table, data))
    return rows


def exposure_rows() -> list[dict]:
    rows = []
    for law in LAWS:
        for n in SIZES:
            for seed in range(SEEDS):
                data = training_sample(reference(law).law, n, seed)
                share = reference(law).exposure(~reached(data, VOCAB))
                rows.append({"law": law, "n": n, "seed": seed, "unreached": share})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    start = time.time()
    jobs = [
        (law, width, n, seed)
        for width in sorted(WIDTHS, reverse=True)
        for law in LAWS
        for n in SIZES
        for seed in range(SEEDS)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run_job, jobs))
    report = {
        "config": {
            "widths": WIDTHS,
            "sample_sizes": SIZES,
            "seeds": SEEDS,
            "alphas": ALPHAS,
            "iterations": ITERATIONS,
            "weight_floor": FLOOR,
        },
        "runs": runs,
        "plug_in": plug_in_rows(),
        "exposure": exposure_rows(),
        "load_average": os.getloadavg()[0],
        "workers": args.workers,
        "torch": torch.__version__,
        "wall_seconds": time.time() - start,
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "contexts.json"
    path.write_text(json.dumps(report, indent=1))
    print(path)


if __name__ == "__main__":
    main()
