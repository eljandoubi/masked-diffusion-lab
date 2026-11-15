"""Learned denoiser error against sample size and width, evaluated exactly.

usage: python scripts/denoiser_study.py --law markov|potts [--workers N]
Writes <out>/denoiser_<law>.json and <out>/denoiser_<law>.png.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from maskdiff.denoiser import model_table, train
from maskdiff.laws import draw, entropy, markov_chain, potts_ring, total_correlation
from maskdiff.metrics import Reference

plt.switch_backend("Agg")

RESULTS = Path(__file__).resolve().parent.parent / "results"
LAWS = {
    "markov": ("persistence", 0.8, markov_chain),
    "potts": ("coupling", 1.0, potts_ring),
}
LENGTH, VOCAB = 6, 3
WIDTHS = [16, 32, 64, 128]
SAMPLE_SIZES = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
STEPS, BATCH, LR, FLOOR = 8000, 512, 3e-3, 0.05
SAMPLER_STEPS = 16
BUDGET_CHECK = [(16, 2000), (128, 2000), (16, 200000), (128, 200000)]


@lru_cache(maxsize=None)
def reference(law: str) -> Reference:
    _, parameter, make = LAWS[law]
    return Reference.from_law(make(LENGTH, VOCAB, parameter))


def run_job(job: tuple[str, int, int, int, int]) -> dict:
    """Train one network on n fresh sequences and measure it exactly."""
    law, width, n, seed, steps = job
    torch.set_num_threads(1)
    ref = reference(law)
    data = draw(ref.law, n, np.random.default_rng([seed, n]))
    start = time.time()
    model, train_loss = train(data, VOCAB, width, steps, seed, BATCH, LR, FLOOR)
    table = model_table(model)
    conditional, gap = ref.conditional_kl(table)
    return {
        "width": width,
        "n": n,
        "seed": seed,
        "steps": steps,
        "cond_kl": conditional,
        "elbo_gap": gap,
        "sampler_kl": ref.sampler_kl(table, SAMPLER_STEPS),
        "train_loss": train_loss,
        "seconds": time.time() - start,
    }


def plot(report: dict, path: Path) -> None:
    config = report["config"]
    cells: dict[tuple[int, int], list[dict]] = {}
    for run in report["runs"]:
        cells.setdefault((run["width"], run["n"]), []).append(run)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = (
        ("cond_kl", "conditional KL (exact)"),
        ("sampler_kl", f"KL(data || sampler), K = {SAMPLER_STEPS}"),
    )
    for ax, (key, label) in zip(axes, panels, strict=True):
        for width in config["widths"]:
            values = np.array(
                [[r[key] for r in cells[(width, n)]] for n in config["sample_sizes"]]
            )
            (line,) = ax.loglog(
                config["sample_sizes"],
                values.mean(axis=1),
                "o-",
                ms=3,
                label=f"{width}",
            )
            ax.fill_between(
                config["sample_sizes"],
                values.min(axis=1),
                values.max(axis=1),
                color=line.get_color(),
                alpha=0.15,
            )
        ax.set_xlabel("training sequences n")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[1].axhline(
        report["exact_sampler_kl"], color="k", ls="--", lw=1, label="exact conditionals"
    )
    axes[0].legend(title="hidden width", fontsize=8)
    axes[1].legend(fontsize=8)
    name = report["law"]["name"]
    fig.suptitle(f"{name}, L = {LENGTH}, V = {VOCAB}: mean and range over seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--law", choices=sorted(LAWS), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--skip-budget-check", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"denoiser_{args.law}.json"
    if args.plot_only:
        plot(json.loads(json_path.read_text()), json_path.with_suffix(".png"))
        return

    start = time.time()
    ref = reference(args.law)
    name, parameter, _ = LAWS[args.law]
    jobs = [
        (args.law, width, n, seed, args.steps)
        for n, width, seed in product(args.sizes[::-1], WIDTHS, range(args.seeds))
    ]
    budget = (
        []
        if args.skip_budget_check
        else [(args.law, width, n, 0, 3 * args.steps) for width, n in BUDGET_CHECK]
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, budget + jobs))
    report = {
        "law": {
            "name": f"{args.law} ({name} {parameter})",
            "entropy": entropy(ref.law),
            "total_correlation": total_correlation(ref.law),
        },
        "config": {
            "widths": WIDTHS,
            "sample_sizes": sorted(args.sizes),
            "seeds": args.seeds,
            "steps": args.steps,
            "batch": BATCH,
            "learning_rate": LR,
            "weight_floor": FLOOR,
            "hidden_layers": 2,
            "sampler_steps": SAMPLER_STEPS,
        },
        "exact_sampler_kl": ref.sampler_kl(ref.table, SAMPLER_STEPS),
        "runs": results[len(budget) :],
        "budget_check": results[: len(budget)],
        "load_average": os.getloadavg()[0],
        "workers": args.workers,
        "torch": torch.__version__,
        "wall_seconds": time.time() - start,
    }
    json_path.write_text(json.dumps(report, indent=1))
    plot(report, json_path.with_suffix(".png"))
    print(json_path)
    print(json_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
