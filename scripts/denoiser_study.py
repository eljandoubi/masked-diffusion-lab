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

from maskdiff.denoiser import model_table, train_exact, train_sampled
from maskdiff.laws import (
    empirical_law,
    entropy,
    markov_chain,
    potts_ring,
    total_correlation,
    training_sample,
)
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
FLOOR = 0.05
ITERATIONS = 2000
SAMPLED_WIDTHS = [16, 128]
SAMPLED_SIZES = [200, 2000, 20000, 200000]
SAMPLED_STEPS, SAMPLED_BATCH, SAMPLED_LR = 12000, 2048, 3e-3
SAMPLER_STEPS = 16
BUDGET_CHECK = [(16, 2000), (128, 2000), (16, 200000), (128, 200000)]


@lru_cache(maxsize=None)
def reference(law: str) -> Reference:
    _, parameter, make = LAWS[law]
    return Reference.from_law(make(LENGTH, VOCAB, parameter))


def run_job(job: tuple[str, str, int, int, int, int]) -> dict:
    """Train one network on n fresh sequences and measure it exactly."""
    trainer, law, width, n, seed, budget = job
    torch.set_num_threads(1)
    ref = reference(law)
    data = training_sample(ref.law, n, seed)
    start = time.time()
    if trainer == "exact":
        empirical = empirical_law(data, VOCAB)
        model, loss, used = train_exact(empirical, width, seed, budget, FLOOR)
    else:
        args = (data, VOCAB, width, budget, seed, SAMPLED_BATCH, SAMPLED_LR, FLOOR)
        model, loss = train_sampled(*args)
        used = budget
    table = model_table(model)
    conditional, gap = ref.conditional_kl(table)
    return {
        "trainer": trainer,
        "width": width,
        "n": n,
        "seed": seed,
        "budget": budget,
        "iterations_used": used,
        "cond_kl": conditional,
        "elbo_gap": gap,
        "sampler_kl": ref.sampler_kl(table, SAMPLER_STEPS),
        "train_loss": loss,
        "seconds": time.time() - start,
    }


def by_cell(runs: list[dict]) -> dict[tuple[int, int], list[dict]]:
    cells: dict[tuple[int, int], list[dict]] = {}
    for run in runs:
        cells.setdefault((run["width"], run["n"]), []).append(run)
    return cells


def plot(report: dict, path: Path) -> None:
    config = report["config"]
    exact, sampled = by_cell(report["runs"]), by_cell(report["sampled_runs"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = (
        ("cond_kl", "conditional KL (exact)"),
        ("sampler_kl", f"KL(data || sampler), K = {SAMPLER_STEPS}"),
    )
    for ax, (key, label) in zip(axes, panels, strict=True):
        for width in config["widths"]:
            sizes = config["sample_sizes"]
            values = np.array([[r[key] for r in exact[(width, n)]] for n in sizes])
            (line,) = ax.loglog(sizes, values.mean(axis=1), "o-", ms=3, label=width)
            ax.fill_between(
                sizes,
                values.min(axis=1),
                values.max(axis=1),
                color=line.get_color(),
                alpha=0.15,
            )
            sizes = [n for n in sizes if (width, n) in sampled]
            if sizes:
                mean = [np.mean([r[key] for r in sampled[(width, n)]]) for n in sizes]
                ax.loglog(sizes, mean, "x:", color=line.get_color(), ms=6)
        ax.set_xlabel("training sequences n")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[1].axhline(
        report["exact_sampler_kl"], color="k", ls="--", lw=1, label="exact conditionals"
    )
    axes[0].legend(title="hidden width", fontsize=8)
    axes[1].legend(fontsize=8)
    name = report["law"]["name"]
    title = (
        f"{name}, L = {LENGTH}, V = {VOCAB}: exact-expectation training "
        "(solid, range over seeds)"
    )
    if report["sampled_runs"]:
        title += ", sampled t (x, dotted)"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--law", choices=sorted(LAWS), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--name", help="output file stem, default denoiser_<law>")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--skip-sampled", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{args.name or 'denoiser_' + args.law}.json"
    if args.plot_only:
        plot(json.loads(json_path.read_text()), json_path.with_suffix(".png"))
        return

    start = time.time()
    ref = reference(args.law)
    name, parameter, _ = LAWS[args.law]
    law, seeds = args.law, range(args.seeds)
    sampled_sizes = (
        [] if args.skip_sampled else sorted(set(SAMPLED_SIZES) & set(args.sizes))
    )
    groups = {
        "sampled_budget_check": [
            ("sampled", law, w, n, 0, 3 * SAMPLED_STEPS)
            for w, n in BUDGET_CHECK
            if w in SAMPLED_WIDTHS and n in sampled_sizes and not args.skip_checks
        ],
        "budget_check": [
            ("exact", law, w, n, 0, 3 * args.iterations)
            for w, n in BUDGET_CHECK
            if n in args.sizes and not args.skip_checks
        ],
        "sampled_runs": [
            ("sampled", law, w, n, s, SAMPLED_STEPS)
            for n, w, s in product(sampled_sizes[::-1], SAMPLED_WIDTHS, seeds)
        ],
        "runs": [
            ("exact", law, w, n, s, args.iterations)
            for n, w, s in product(sorted(args.sizes)[::-1], WIDTHS, seeds)
        ],
    }
    jobs = [job for group in groups.values() for job in group]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, jobs))
    split, begin = {}, 0
    for key, group in groups.items():
        split[key] = results[begin : begin + len(group)]
        begin += len(group)
    report = {
        "law": {
            "name": f"{law} ({name} {parameter})",
            "entropy": entropy(ref.law),
            "total_correlation": total_correlation(ref.law),
        },
        "config": {
            "widths": WIDTHS,
            "sample_sizes": sorted(args.sizes),
            "seeds": args.seeds,
            "iterations": args.iterations,
            "weight_floor": FLOOR,
            "hidden_layers": 2,
            "sampler_steps": SAMPLER_STEPS,
            "sampled": {
                "widths": SAMPLED_WIDTHS,
                "steps": SAMPLED_STEPS,
                "batch": SAMPLED_BATCH,
                "learning_rate": SAMPLED_LR,
            },
        },
        "exact_sampler_kl": ref.sampler_kl(ref.table, SAMPLER_STEPS),
        **split,
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
