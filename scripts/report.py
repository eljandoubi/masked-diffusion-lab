"""Print the markdown tables of the README from results/*.json."""

import json
from pathlib import Path

import numpy as np

from maskdiff.metrics import loglog_slope

RESULTS = Path(__file__).resolve().parent.parent / "results"
DECADES = [(200, 2000), (2000, 20000), (20000, 200000), (200, 200000)]


def sci(x: float) -> str:
    return f"{x:.3e}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def find(report: dict, family: str, length: int, vocab: int) -> list[dict]:
    return [
        c
        for c in report["cases"]
        if (c["family"], c["length"], c["vocab"]) == (family, length, vocab)
    ]


def sampler_error_table(report: dict, family: str, length: int, vocab: int) -> str:
    cases = find(report, family, length, vocab)
    name = "coupling" if family == "potts" else "persistence"
    header = ["K"] + [f"{name} {c['param']}" for c in cases]
    rows = [
        [str(k)] + [sci(max(c["kl"][i], 0.0)) for c in cases]
        for i, k in enumerate(report["grid"])
    ]
    rows.append(
        [f"{length} (one position per step)"]
        + [sci(abs(c["one_at_a_time"])) for c in cases]
    )
    return table(header, rows)


def slope_table(report: dict, length: int, vocab: int) -> str:
    lo, hi = report["fit_range"]
    elo, ehi = report["fit_range_extended"]
    header = [
        "law",
        "total correlation",
        f"slope, K {lo} to {hi}",
        f"slope, K {elo} to {ehi}",
        "c",
        f"K^2 KL at K = {ehi}",
    ]
    rows = []
    for family in ("potts", "markov"):
        for c in find(report, family, length, vocab):
            if "slope" not in c:
                continue
            name = f"{family} {c['param']}"
            rows.append(
                [
                    name,
                    f"{c['total_correlation']:.4f}",
                    f"{c['slope']:.3f}",
                    f"{c['slope_extended']:.3f}",
                    f"{c['coefficient']:.4f}",
                    f"{c['kl_extended'][-1] * ehi**2:.4f}",
                ]
            )
    return table(header, rows)


def equal_count_table(report: dict, family: str, length: int, vocab: int) -> str:
    cases = [c for c in find(report, family, length, vocab) if "slope" in c]
    steps = [int(k) for k in cases[0]["equal_count"]]
    name = "coupling" if family == "potts" else "persistence"
    header = ["K", "schedule"] + [f"{name} {c['param']}" for c in cases]
    rows = []
    for k in steps:
        uniform = [c["kl"][report["grid"].index(k)] for c in cases]
        equal = [c["equal_count"][str(k)] for c in cases]
        rows.append([str(k), "uniform grid"] + [sci(v) for v in uniform])
        rows.append(["", "equal counts, random order"] + [sci(v) for v in equal])
    return table(header, rows)


def cell_values(
    report: dict, key: str, group: str = "runs"
) -> dict[tuple[int, int], np.ndarray]:
    cells: dict[tuple[int, int], list[float]] = {}
    for run in report[group]:
        cells.setdefault((run["width"], run["n"]), []).append(run[key])
    return {k: np.array(v) for k, v in cells.items()}


def denoiser_table(report: dict, key: str) -> str:
    cells = cell_values(report, key)
    widths, sizes = report["config"]["widths"], report["config"]["sample_sizes"]
    rows = [[str(n)] + [sci(cells[(w, n)].mean()) for w in widths] for n in sizes]
    return table(["n"] + [f"width {w}" for w in widths], rows)


def slope_by_decade(report: dict) -> str:
    cells = cell_values(report, "cond_kl")
    widths, sizes = report["config"]["widths"], report["config"]["sample_sizes"]
    header = ["width"] + [f"n {lo} to {hi}" for lo, hi in DECADES]
    rows = []
    for w in widths:
        mean = [cells[(w, n)].mean() for n in sizes]
        row = [str(w)]
        for lo, hi in DECADES:
            slope, r2 = loglog_slope(sizes, mean, lo, hi)
            row.append(f"{slope:.2f}")
        rows.append(row)
    return table(header, rows)


def plateau_widths(report: dict, tolerance: float = 0.10) -> str:
    cells = cell_values(report, "cond_kl")
    widths, sizes = report["config"]["widths"], report["config"]["sample_sizes"]
    rows = []
    for n in sizes:
        mean = [cells[(w, n)].mean() for w in widths]
        best = min(mean)
        first = next(
            w for w, m in zip(widths, mean, strict=True) if m <= (1 + tolerance) * best
        )
        rows.append([str(n), str(first), f"{mean[0] / best:.2f}"])
    return table(
        ["n", "smallest width within 10% of the best", "width 16 / best"], rows
    )


def spread(report: dict) -> str:
    cells = cell_values(report, "cond_kl")
    ratio = [v.std(ddof=1) / v.mean() for v in cells.values()]
    return (
        "seed to seed std over mean of the conditional KL: "
        f"median {np.median(ratio):.2f}, max {max(ratio):.2f}"
    )


def budget_table(report: dict, check: str, base: str) -> str:
    """Seed 0 of the main grid against the same cell with a three times longer run."""
    rows = []
    for run in report[check]:
        first = next(
            r
            for r in report[base]
            if (r["width"], r["n"], r["seed"]) == (run["width"], run["n"], 0)
        )
        rows.append(
            [
                str(run["width"]),
                str(run["n"]),
                f"{first['budget']}: {sci(first['cond_kl'])}",
                f"{run['budget']}: {sci(run['cond_kl'])}",
            ]
        )
    return table(["width", "n", "budget: cond KL", "3x budget: cond KL"], rows)


def trainer_table(report: dict) -> str:
    exact = cell_values(report, "cond_kl")
    sampled = cell_values(report, "cond_kl", "sampled_runs")
    exact_k = cell_values(report, "sampler_kl")
    sampled_k = cell_values(report, "sampler_kl", "sampled_runs")
    rows = []
    for width, n in sorted(sampled, key=lambda cell: (cell[0], cell[1])):
        rows.append(
            [
                str(width),
                str(n),
                sci(exact[(width, n)].mean()),
                sci(sampled[(width, n)].mean()),
                sci(exact_k[(width, n)].mean()),
                sci(sampled_k[(width, n)].mean()),
            ]
        )
    header = [
        "width",
        "n",
        "cond KL exact",
        "cond KL sampled",
        "K=16 exact",
        "K=16 sampled",
    ]
    return table(header, rows)


def run_note(report: dict) -> str:
    return f"wall {report['wall_seconds']:.0f} s, load {report['load_average']:.1f}"


def sampler_facts(report: dict) -> list[str]:
    """Derived numbers quoted in the README."""
    cases = report["cases"]
    dependent = [c for c in cases if c["total_correlation"] > 1e-9]
    last = report["extended_grid"][-1]
    relative = max(
        abs(c["kl"][0] - c["total_correlation"]) / c["total_correlation"]
        for c in dependent
    )
    product = max(
        abs(v) for c in cases if c["total_correlation"] <= 1e-9 for v in c["kl"]
    )
    ratio = [c["kl_extended"][-1] * last**2 / c["coefficient"] for c in dependent]
    k16, k48 = report["grid"].index(16), report["grid"].index(48)
    six = next(c for c in dependent if c["length"] == 6)
    one_step = max(abs(c["one_at_a_time"]) for c in cases)
    return [
        f"K = 1 against the total correlation, largest relative gap {relative:.1e}",
        f"one position per step, largest |KL| {one_step:.1e}",
        f"product laws, largest |KL| over the grid {product:.1e}",
        f"K^2 KL at K = {last} over c: {min(ratio):.3f} to {max(ratio):.3f}",
        f"tie probability, L = 6: {six['tie_probability'][k16]:.3f} at K = 16, "
        f"{six['tie_probability'][k48]:.3f} at K = 48",
    ]


def denoiser_facts(report: dict) -> list[str]:
    """Derived numbers quoted in the README."""
    floor = report["exact_sampler_kl"]
    runs = report["runs"]
    ratio = [r["sampler_kl"] / floor for r in runs]
    wide = [
        r["sampler_kl"] / floor for r in runs if r["n"] >= 100000 and r["width"] >= 32
    ]
    widths = report["config"]["widths"]

    def mean_ratio(n: int) -> list[str]:
        means = [
            np.mean([r["sampler_kl"] for r in runs if (r["n"], r["width"]) == (n, w)])
            for w in widths
        ]
        return [f"{m / floor:.2f}" for m in means]

    facts = [
        f"sampler KL over the exact-conditional value, {len(runs)} runs: "
        f"smallest {min(ratio):.3f}",
        f"n >= 100000, widths 32 to 128, {len(wide)} runs: "
        f"{min(wide):.3f} to {max(wide):.3f}",
        f"mean ratio by width {widths} at n = 200: {mean_ratio(200)}",
        f"mean ratio by width {widths} at n = 200000: {mean_ratio(200000)}",
    ]
    if report["sampled_runs"]:
        exact = cell_values(report, "cond_kl")
        sampled = cell_values(report, "cond_kl", "sampled_runs")
        factor = max(
            max(
                exact[c].mean() / sampled[c].mean(), sampled[c].mean() / exact[c].mean()
            )
            for c in sampled
        )
        facts.append(f"exact against sampled trainer, largest factor {factor:.1f}")
    for check, base in (
        ("budget_check", "runs"),
        ("sampled_budget_check", "sampled_runs"),
    ):
        changes = []
        for run in report[check]:
            first = next(
                r
                for r in report[base]
                if (r["width"], r["n"], r["seed"]) == (run["width"], run["n"], 0)
            )
            changes.append(
                max(
                    run["cond_kl"] / first["cond_kl"], first["cond_kl"] / run["cond_kl"]
                )
            )
        if changes:
            facts.append(f"{check}, largest change factor {max(changes):.1f}")
    return facts


def main() -> None:
    path = RESULTS / "sampler_error.json"
    if path.exists():
        report = json.loads(path.read_text())
        print(f"## sampler error ({run_note(report)})\n")
        for family in ("potts", "markov"):
            print(f"### {family}, L = 6, V = 3\n")
            print(sampler_error_table(report, family, 6, 3), "\n")
        print("### slopes, L = 6, V = 3\n")
        print(slope_table(report, 6, 3), "\n")
        print("### slopes, L = 8, V = 2\n")
        print(slope_table(report, 8, 2), "\n")
        print("### equal counts, potts L = 6\n")
        print(equal_count_table(report, "potts", 6, 3), "\n")
        print("### equal counts, markov L = 6\n")
        print(equal_count_table(report, "markov", 6, 3), "\n")
        print("### derived numbers\n")
        print("\n".join(f"- {line}" for line in sampler_facts(report)), "\n")
    for law in ("markov", "potts"):
        path = RESULTS / f"denoiser_{law}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        steps = report["config"]["sampler_steps"]
        print(f"## denoiser, {report['law']['name']} ({run_note(report)})\n")
        print(
            f"entropy {report['law']['entropy']:.4f}, "
            f"total correlation {report['law']['total_correlation']:.4f}, "
            f"exact K = {steps} sampler KL {sci(report['exact_sampler_kl'])}\n"
        )
        print("### conditional KL (mean over seeds)\n")
        print(denoiser_table(report, "cond_kl"), "\n")
        print(spread(report), "\n")
        print(f"### sampler KL at K = {steps} (mean over seeds)\n")
        print(denoiser_table(report, "sampler_kl"), "\n")
        print("### log-log slope of the conditional KL against n\n")
        print(slope_by_decade(report), "\n")
        print("### width at which the error stops improving\n")
        print(plateau_widths(report), "\n")
        print("### derived numbers\n")
        print("\n".join(f"- {line}" for line in denoiser_facts(report)), "\n")
        if report["budget_check"]:
            print("### budget check, exact trainer (L-BFGS iterations)\n")
            print(budget_table(report, "budget_check", "runs"), "\n")
        if report["sampled_runs"]:
            print("### exact against sampled-t training\n")
            print(trainer_table(report), "\n")
        if report["sampled_budget_check"]:
            print("### budget check, sampled-t trainer (Adam steps)\n")
            print(budget_table(report, "sampled_budget_check", "sampled_runs"), "\n")


if __name__ == "__main__":
    main()
