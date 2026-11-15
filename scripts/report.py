"""Print the markdown tables of the README from results/*.json."""

import json
from pathlib import Path

import numpy as np

from maskdiff.metrics import loglog_slope, paired_interval

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


def plug_in_means(base: dict, key: str) -> dict[tuple[float, int], float]:
    """Mean over seeds of a plug-in error, by (alpha, n)."""
    cells: dict[tuple[float, int], list[float]] = {}
    for run in base["runs"]:
        cells.setdefault((run["alpha"], run["n"]), []).append(run[key])
    return {k: float(np.mean(v)) for k, v in cells.items()}


def denoiser_table(report: dict, key: str, base: dict | None = None) -> str:
    cells = cell_values(report, key)
    widths, sizes = report["config"]["widths"], report["config"]["sample_sizes"]
    plug = plug_in_means(base, key) if base else {}
    alphas = sorted({alpha for alpha, _ in plug})
    header = ["n"] + [f"width {w}" for w in widths]
    header += [f"plug-in {alpha}" for alpha in alphas]
    rows = [
        [str(n)]
        + [sci(cells[(w, n)].mean()) for w in widths]
        + [sci(plug[(alpha, n)]) for alpha in alphas]
        for n in sizes
    ]
    return table(header, rows)


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


def seed_values(report: dict, width: int, n: int, key: str = "cond_kl") -> np.ndarray:
    runs = sorted(
        (r for r in report["runs"] if (r["width"], r["n"]) == (width, n)),
        key=lambda r: r["seed"],
    )
    return np.array([r[key] for r in runs])


def paired_widths(report: dict) -> str:
    """Neighbouring widths, paired by seed (the seed fixes the training sample)."""
    widths, sizes = report["config"]["widths"], report["config"]["sample_sizes"]
    header = [
        "n",
        "widths",
        "cond KL, wider minus narrower, per seed",
        "mean",
        "95% interval",
        "wider / narrower",
    ]
    rows = []
    for n in sizes:
        for narrow, wide in zip(widths, widths[1:], strict=False):
            diff, mean, half = paired_interval(
                seed_values(report, narrow, n), seed_values(report, wide, n)
            )
            ratio = seed_values(report, wide, n) / seed_values(report, narrow, n)
            rows.append(
                [
                    str(n),
                    f"{narrow}, {wide}",
                    ", ".join(f"{d:+.2e}" for d in diff),
                    f"{mean:+.2e}",
                    f"{mean - half:+.2e} to {mean + half:+.2e}",
                    f"{ratio.min():.2f} to {ratio.max():.2f}",
                ]
            )
    return table(header, rows)


def spread(report: dict) -> str:
    cells = cell_values(report, "cond_kl")
    ratio = [v.std(ddof=1) / v.mean() for v in cells.values()]
    return (
        "relative standard deviation over seeds (sd / mean, ddof 1) of the "
        f"conditional KL: median {np.median(ratio):.2f}, max {max(ratio):.2f}"
    )


def train_gap_table(report: dict, base: dict) -> str:
    """Training loss above its minimum, relative, mean over seeds."""
    best = {(r["n"], r["seed"]): r["loss"] for r in base["minimum_loss"]}
    widths, sizes = report["config"]["widths"], report["config"]["sample_sizes"]
    rows = []
    for n in sizes:
        row = [str(n)]
        for w in widths:
            gaps = [
                r["train_loss"] / best[(n, r["seed"])] - 1
                for r in report["runs"]
                if (r["width"], r["n"]) == (w, n)
            ]
            row.append(sci(float(np.mean(gaps))))
        rows.append(row)
    return table(["n"] + [f"width {w}" for w in widths], rows)


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
                f"{run['cond_kl'] / first['cond_kl']:.2f}",
            ]
        )
    header = ["width", "n", "budget: cond KL", "3x budget: cond KL", "3x / 1x"]
    return table(header, rows)


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
    note = f"wall {report['wall_seconds']:.0f} s, load {report['load_average']:.1f}"
    if "extension" in report:
        later = report["extension"]
        note += f"; extension wall {later['wall_seconds']:.0f} s"
        note += f", load {later['load_average']:.1f}"
    return note


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


def paired_facts(report: dict) -> list[str]:
    widths, sizes = report["config"]["widths"], report["config"]["sample_sizes"]
    cells = len(sizes) * report["config"]["seeds"]
    facts = []
    for narrow, wide in zip(widths, widths[1:], strict=False):
        above = sum(
            int((seed_values(report, wide, n) > seed_values(report, narrow, n)).sum())
            for n in sizes
        )
        facts.append(
            f"width {wide} above width {narrow} in {above} of {cells} seed-paired cells"
        )
    return facts


def plug_in_facts(report: dict, base: dict) -> list[str]:
    widths, sizes = report["config"]["widths"], report["config"]["sample_sizes"]
    cond, sampler = cell_values(report, "cond_kl"), cell_values(report, "sampler_kl")
    plug, plug_k = plug_in_means(base, "cond_kl"), plug_in_means(base, "sampler_kl")
    alphas = sorted({alpha for alpha, _ in plug})
    facts = []
    for alpha in alphas:
        better = [
            n
            for n in sizes
            if plug[(alpha, n)] < min(cond[(w, n)].mean() for w in widths)
        ]
        facts.append(
            f"plug-in {alpha}: conditional KL below every width at n in {better}"
        )
    ahead = [
        n
        for n in sizes
        if min(cond[(w, n)].mean() for w in widths)
        < min(plug[(alpha, n)] for alpha in alphas)
    ]
    facts.append(f"best width below all plug-in tables at n in {ahead}")
    for n in (sizes[0], sizes[-1]):
        values = ", ".join(f"{alpha}: {sci(plug_k[(alpha, n)])}" for alpha in alphas)
        best = min(sampler[(w, n)].mean() for w in widths)
        facts.append(
            f"K = {report['config']['sampler_steps']} sampler KL at n = {n}: "
            f"plug-in {values}; best width {sci(best)}; "
            f"exact conditionals {sci(report['exact_sampler_kl'])}"
        )
    best = {(r["n"], r["seed"]): r["loss"] for r in base["minimum_loss"]}
    below = [
        r
        for r in report["runs"]
        if r["train_loss"] < best[(r["n"], r["seed"])] * (1 - 1e-9)
    ]
    facts.append(
        f"training loss below its minimum: {len(below)} of {len(report['runs'])} runs"
    )
    for run in report["budget_check"]:
        first = next(
            r
            for r in report["runs"]
            if (r["width"], r["n"], r["seed"]) == (run["width"], run["n"], 0)
        )
        floor = best[(run["n"], 0)]
        facts.append(
            f"width {run['width']}, n = {run['n']}, seed 0: training loss above its "
            f"minimum {sci(first['train_loss'] / floor - 1)} after {first['budget']} "
            f"iterations, {sci(run['train_loss'] / floor - 1)} after {run['budget']}"
        )
    return facts


def rerun_facts(report: dict) -> list[str]:
    again = report["budget_rerun"]
    change = []
    for run in again:
        first = next(
            r
            for r in report["runs"]
            if (r["width"], r["n"], r["seed"]) == (run["width"], run["n"], 0)
        )
        change.append(abs(run["cond_kl"] / first["cond_kl"] - 1))
    used = [
        f"{r['iterations_used']} of {r['budget']}"
        for r in again + report["budget_check"]
    ]
    return [
        f"{len(again)} seed 0 cells of the main grid run again: largest relative "
        f"change of the conditional KL {max(change):.1e}",
        "L-BFGS iterations used (of the cap): " + ", ".join(used),
    ]


def denoiser_facts(report: dict, baseline: dict | None = None) -> list[str]:
    """Derived numbers quoted in the README."""
    reference = report["exact_sampler_kl"]
    runs = report["runs"]
    ratio = [r["sampler_kl"] / reference for r in runs]
    wide = [
        r["sampler_kl"] / reference
        for r in runs
        if r["n"] >= 100000 and r["width"] >= 32
    ]
    widths = report["config"]["widths"]

    def mean_ratio(n: int) -> list[str]:
        means = [
            np.mean([r["sampler_kl"] for r in runs if (r["n"], r["width"]) == (n, w)])
            for w in widths
        ]
        return [f"{m / reference:.2f}" for m in means]

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
    facts += paired_facts(report)
    if baseline:
        facts += plug_in_facts(report, baseline)
    if report.get("budget_rerun"):
        facts += rerun_facts(report)
    return facts


def stopped_table(report: dict, stopped: dict, base: dict) -> str:
    """Conditional KL after a short and the full L-BFGS budget, mean over seeds."""
    full, short = cell_values(report, "cond_kl"), cell_values(stopped, "cond_kl")
    widths, sizes = stopped["config"]["widths"], stopped["config"]["sample_sizes"]
    best = {(r["n"], r["seed"]): r["loss"] for r in base["minimum_loss"]}
    rows = []
    for n in sizes:
        for config, cells, runs in (
            (stopped["config"], short, stopped["runs"]),
            (report["config"], full, report["runs"]),
        ):
            gaps = [
                float(
                    np.mean(
                        [
                            r["train_loss"] / best[(n, r["seed"])] - 1
                            for r in runs
                            if (r["width"], r["n"]) == (w, n)
                        ]
                    )
                )
                for w in widths
            ]
            ratio = cells[(widths[-1], n)].mean() / cells[(widths[0], n)].mean()
            rows.append(
                [str(n), str(config["iterations"]), "error"]
                + [sci(cells[(w, n)].mean()) for w in widths]
                + [f"{ratio:.2f}"]
            )
            rows.append(
                [str(n), str(config["iterations"]), "training loss above its minimum"]
                + [sci(gap) for gap in gaps]
                + [""]
            )
    header = ["n", "iterations", "quantity"] + [f"width {w}" for w in widths]
    header.append(f"error, width {widths[-1]} / width {widths[0]}")
    return table(header, rows)


def contexts_table(contexts: dict, law: str) -> str:
    """Conditional KL on reached and unreached partial sequences, mean over seeds."""
    config = contexts["config"]
    rows = []
    for n in config["sample_sizes"]:
        weight = np.mean(
            [
                e["unreached"]
                for e in contexts["exposure"]
                if (e["law"], e["n"]) == (law, n)
            ]
        )
        models = [
            (
                f"width {w}",
                [
                    r
                    for r in contexts["runs"]
                    if (r["law"], r["width"], r["n"]) == (law, w, n)
                ],
            )
            for w in config["widths"]
        ] + [
            (
                f"plug-in {a}",
                [
                    r
                    for r in contexts["plug_in"]
                    if (r["law"], r["alpha"], r["n"]) == (law, a, n)
                ],
            )
            for a in config["alphas"]
        ]
        for name, runs in models:
            total, seen, unseen = (
                float(np.mean([r[key] for r in runs]))
                for key in ("cond_kl", "cond_kl_reached", "cond_kl_unreached")
            )
            rows.append(
                [
                    str(n),
                    name,
                    sci(total),
                    sci(seen),
                    sci(unseen),
                    f"{unseen / total:.2f}",
                    f"{weight:.2f}",
                ]
            )
    header = [
        "n",
        "model",
        "cond KL",
        "on reached",
        "on unreached",
        "share of the error on unreached",
        "share of the weight on unreached",
    ]
    return table(header, rows)


def contexts_facts(contexts: dict, report: dict, law: str) -> list[str]:
    """Largest gap between the retrained networks and the main grid."""
    gaps = []
    for run in contexts["runs"]:
        if run["law"] != law:
            continue
        first = next(
            r
            for r in report["runs"]
            if (r["width"], r["n"], r["seed"]) == (run["width"], run["n"], run["seed"])
        )
        gaps.append(abs(run["cond_kl"] / first["cond_kl"] - 1))
    used = {run["iterations_used"] for run in contexts["runs"] if run["law"] == law}
    return [
        f"{len(gaps)} networks trained again: largest relative change of the "
        f"conditional KL against the main grid {max(gaps):.1e}",
        f"L-BFGS iterations used: {sorted(used)}",
    ]


def compensation_table(report: dict) -> str:
    rows = report["markov"]["rows"]
    exact = rows[0]["sampler_kl"]
    header = [
        "persistence of the conditionals",
        "conditional KL",
        f"K = {report['steps']} sampler KL",
        "over the exact-conditional value",
    ]
    body = [
        [
            f"{r['persistence']:.2f}",
            sci(r["cond_kl"]),
            sci(r["sampler_kl"]),
            f"{r['sampler_kl'] / exact:.3f}",
        ]
        for r in rows
    ]
    return table(header, body)


def compensation_facts(report: dict) -> list[str]:
    """Derived numbers quoted in the README."""
    rows, pair = report["markov"]["rows"], report["pair"]
    exact = rows[0]["sampler_kl"]
    below = [r for r in rows if r["sampler_kl"] < exact]
    best = min(rows, key=lambda r: r["sampler_kl"])
    near = next(r for r in rows if r["persistence"] == 0.81)
    return [
        f"persistence {near['persistence']}: sampler KL {sci(near['sampler_kl'])}, "
        f"{100 * (1 - near['sampler_kl'] / exact):.1f}% below the exact-conditional "
        f"value {sci(exact)}",
        f"persistences below the exact-conditional value: {below[0]['persistence']} "
        f"to {below[-1]['persistence']}, smallest at {best['persistence']} "
        f"({sci(best['sampler_kl'])})",
        f"two positions, persistence {pair['persistence']}, K = {report['steps']}: "
        f"exact conditionals {sci(pair['exact_sampler_kl'])}",
        f"conditionals of persistence {pair['other_persistence']:.4f}: conditional "
        f"KL {sci(pair['other_cond_kl'])}, |sampler KL| "
        f"{sci(abs(pair['other_sampler_kl']))}, largest law difference "
        f"{sci(pair['max_law_difference'])}",
    ]


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
    baseline_path = RESULTS / "baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    contexts_path = RESULTS / "contexts.json"
    contexts = json.loads(contexts_path.read_text()) if contexts_path.exists() else {}
    for law in ("markov", "potts"):
        path = RESULTS / f"denoiser_{law}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        plug_in = baseline.get(law)
        steps = report["config"]["sampler_steps"]
        print(f"## denoiser, {report['law']['name']} ({run_note(report)})\n")
        print(
            f"entropy {report['law']['entropy']:.4f}, "
            f"total correlation {report['law']['total_correlation']:.4f}, "
            f"exact K = {steps} sampler KL {sci(report['exact_sampler_kl'])}\n"
        )
        print("### conditional KL (mean over seeds)\n")
        print(denoiser_table(report, "cond_kl", plug_in), "\n")
        print(spread(report), "\n")
        print(f"### sampler KL at K = {steps} (mean over seeds)\n")
        print(denoiser_table(report, "sampler_kl", plug_in), "\n")
        if plug_in:
            print("### training loss above its minimum, relative (mean over seeds)\n")
            print(train_gap_table(report, plug_in), "\n")
        print("### log-log slope of the conditional KL against n\n")
        print(slope_by_decade(report), "\n")
        print("### width at which the error stops improving\n")
        print(plateau_widths(report), "\n")
        print("### neighbouring widths, paired by seed\n")
        print(paired_widths(report), "\n")
        print("### derived numbers\n")
        print("\n".join(f"- {line}" for line in denoiser_facts(report, plug_in)), "\n")
        if contexts:
            print("### error on reached and unreached partial sequences\n")
            print(contexts_table(contexts, law), "\n")
            print(
                "\n".join(
                    f"- {line}" for line in contexts_facts(contexts, report, law)
                ),
                "\n",
            )
        stopped_path = RESULTS / f"denoiser_{law}_stop100.json"
        if stopped_path.exists() and plug_in:
            stopped = json.loads(stopped_path.read_text())
            print("### early stopping: a short L-BFGS budget\n")
            print(stopped_table(report, stopped, plug_in), "\n")
        if report["budget_check"]:
            print("### budget check, exact trainer (L-BFGS iterations)\n")
            print(budget_table(report, "budget_check", "runs"), "\n")
        if report["sampled_runs"]:
            print("### exact against sampled-t training\n")
            print(trainer_table(report), "\n")
        if report["sampled_budget_check"]:
            print("### budget check, sampled-t trainer (Adam steps)\n")
            print(budget_table(report, "sampled_budget_check", "sampled_runs"), "\n")
    path = RESULTS / "compensation.json"
    if path.exists():
        report = json.loads(path.read_text())
        print(f"## other conditionals, {report['markov']['law']}\n")
        print(compensation_table(report), "\n")
        print("### derived numbers\n")
        print("\n".join(f"- {line}" for line in compensation_facts(report)), "\n")


if __name__ == "__main__":
    main()
