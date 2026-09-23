"""Tables, statistical tests and figures for the paper.

    python -m bench.report [--sim results/sim/episodes.jsonl] [--real data/real/trials.csv] [--out-dir results]

Either input may be missing (e.g. before the real trials). Success means a
completed lap in the simulator, and a clean lap (no interventions) on the car.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from bench.conditions import CONDITIONS, MODELS, NOISE_CONDITIONS  # noqa: E402
from bench.metrics import parse_trial_log, trial_outcome, wilson_interval  # noqa: E402

SIM_METRICS = ("avg_abs_cte", "jerk", "osc_hz", "recovery_steps")
REAL_METRICS = ("interventions", "jerk", "osc_hz", "latency_ms_mean")
COMPARISONS = (
    ("RL vs BC (clean)", "sac_clean", "bc_clean"),
    ("RL vs BC (DR)", "sac_dr", "bc_dr"),
    ("DR vs clean (RL)", "sac_dr", "sac_clean"),
    ("DR vs clean (BC)", "bc_dr", "bc_clean"),
)


def load_sim(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{**r, "success": r["outcome"] == "success"} for r in rows]


def load_real(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as f:
        for trial in csv.DictReader(f):
            log = parse_trial_log(path.parent / trial["log"])
            outcome = trial_outcome(log, trial["completed"] == "1")
            rows.append({
                "model": trial["model"],
                "condition": trial["condition"],
                "success": outcome["clean_lap"],
                "interventions": outcome["interventions"],
                "lap_time": outcome["lap_time"],
                "early_intervention": outcome["early_intervention"],
                "jerk": log["jerk"],
                "osc_hz": log["osc_hz"],
                "latency_ms_mean": log["latency_ms_mean"],
            })
    return rows


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def _mean_std(values) -> tuple[float, float]:
    v = _finite(values)
    return (float(np.mean(v)), float(np.std(v))) if v else (math.nan, math.nan)


def cell_stats(rows, metrics) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["condition"]), []).append(r)
    cells = {}
    for key, group in groups.items():
        k, n = sum(bool(r["success"]) for r in group), len(group)
        cells[key] = {"n": n, "successes": k, "rate": k / n, "ci": wilson_interval(k, n),
                      **{m: _mean_std(r.get(m) for r in group) for m in metrics}}
    return cells


def compare(rows, model_a: str, model_b: str, conditions=NOISE_CONDITIONS) -> dict:
    a = [r for r in rows if r["model"] == model_a and r["condition"] in conditions]
    b = [r for r in rows if r["model"] == model_b and r["condition"] in conditions]
    ka, kb = sum(bool(r["success"]) for r in a), sum(bool(r["success"]) for r in b)
    p_success = stats.fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1] if a and b else math.nan
    ja, jb = _finite(r.get("jerk") for r in a), _finite(r.get("jerk") for r in b)
    p_jerk = stats.mannwhitneyu(ja, jb)[1] if ja and jb else math.nan
    return {
        "a": model_a, "b": model_b, "n_a": len(a), "n_b": len(b),
        "rate_a": ka / len(a) if a else math.nan, "rate_b": kb / len(b) if b else math.nan,
        "p_success": float(p_success),
        "median_jerk_a": float(np.median(ja)) if ja else math.nan,
        "median_jerk_b": float(np.median(jb)) if jb else math.nan,
        "p_jerk": float(p_jerk),
    }


def sim_real_spearman(sim_cells: dict, real_cells: dict) -> tuple[float, float, int]:
    keys = sorted(set(sim_cells) & set(real_cells))
    if len(keys) < 3:
        return math.nan, math.nan, len(keys)
    rho, p = stats.spearmanr([sim_cells[k]["rate"] for k in keys], [real_cells[k]["rate"] for k in keys])
    return float(rho), float(p), len(keys)


def _fmt(x, digits: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _ordered(cells: dict) -> list[tuple]:
    return [(m, c.name) for m in MODELS for c in CONDITIONS if (m, c.name) in cells]


def _cell_table(cells: dict, metrics) -> list[str]:
    lines = ["| model | condition | n | success | 95% CI | Δ vs nominal | " + " | ".join(metrics) + " |",
             "|" + "---|" * (6 + len(metrics))]
    for model, condition in _ordered(cells):
        cell = cells[(model, condition)]
        nominal = cells.get((model, "nominal"))
        delta = cell["rate"] - nominal["rate"] if nominal else math.nan
        metric_cols = [f"{_fmt(cell[m][0])} ± {_fmt(cell[m][1])}" for m in metrics]
        lines.append(f"| {model} | {condition} | {cell['n']} | {_fmt(cell['rate'])} | "
                     f"{_fmt(cell['ci'][0])}–{_fmt(cell['ci'][1])} | {_fmt(delta)} | " + " | ".join(metric_cols) + " |")
    return lines


def _comparison_table(rows) -> list[str]:
    lines = ["| comparison (pooled over noise conditions) | success A | success B | Fisher p | median jerk A | median jerk B | Mann–Whitney p |",
             "|---|---|---|---|---|---|---|"]
    for label, a, b in COMPARISONS:
        r = compare(rows, a, b)
        lines.append(f"| {label}: {a} (A) vs {b} (B) | {_fmt(r['rate_a'])} (n={r['n_a']}) | {_fmt(r['rate_b'])} (n={r['n_b']}) | "
                     f"{_fmt(r['p_success'], 4)} | {_fmt(r['median_jerk_a'])} | {_fmt(r['median_jerk_b'])} | {_fmt(r['p_jerk'], 4)} |")
    return lines


def _plot_real_success(cells: dict, path: Path) -> None:
    names = [c.name for c in CONDITIONS]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(MODELS):
        rates = [cells[(model, c)]["rate"] if (model, c) in cells else np.nan for c in names]
        ax.bar(np.arange(len(names)) + (i - 1.5) * width, rates, width, label=model)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("clean-lap rate (real car)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_sim_vs_real(sim_cells: dict, real_cells: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for model in MODELS:
        keys = [k for k in sorted(set(sim_cells) & set(real_cells)) if k[0] == model]
        ax.scatter([sim_cells[k]["rate"] for k in keys], [real_cells[k]["rate"] for k in keys], label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("sim success rate")
    ax.set_ylabel("real clean-lap rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(sim_rows, real_rows, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_cells = cell_stats(sim_rows, SIM_METRICS)
    real_cells = cell_stats(real_rows, REAL_METRICS)
    lines = ["# Sim-to-Real Robustness Benchmark: results", ""]
    for title, rows, cells, metrics in (("Simulator", sim_rows, sim_cells, SIM_METRICS),
                                        ("Real car", real_rows, real_cells, REAL_METRICS)):
        lines += [f"## {title}", ""]
        if not rows:
            lines += ["No data yet.", ""]
            continue
        lines += _cell_table(cells, metrics) + [""] + _comparison_table(rows) + [""]
    if real_rows:
        _plot_real_success(real_cells, out_dir / "real_success.png")
        lines += ["![real clean-lap rate](real_success.png)", ""]
    if sim_rows and real_rows:
        rho, p, n = sim_real_spearman(sim_cells, real_cells)
        _plot_sim_vs_real(sim_cells, real_cells, out_dir / "sim_vs_real.png")
        lines += ["## Sim-to-real agreement", "",
                  f"Spearman ρ = {_fmt(rho)} (p = {_fmt(p, 4)}, {n} model × condition cells)", "",
                  "![sim vs real](sim_vs_real.png)", ""]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", default="results/sim/episodes.jsonl")
    parser.add_argument("--real", default="data/real/trials.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    report = write_report(load_sim(args.sim), load_real(args.real), args.out_dir)
    print(report.read_text())
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
