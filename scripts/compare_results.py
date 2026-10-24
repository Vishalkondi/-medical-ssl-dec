"""Collect test results from all evaluated runs into tables and figures.

Reads experiments/*/test_metrics.json (written by scripts/evaluate.py), so
every number in the report traces back to a real run. Nothing is invented:
methods or fractions without runs are simply absent.

Outputs (results/):
    summary_runs.csv        one row per run
    summary_table.csv/.md   mean ± std over seeds per method and labelled fraction
    auc_vs_fraction.png     test ROC-AUC against labelled fraction
    roc_<XX>.png            ROC curves of all methods for one labelled fraction

Usage:
    python scripts/compare_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402

# Fixed categorical order: colour follows the method, never its rank.
METHODS = {
    "baseline": ("Supervised baseline", "#2a78d6"),
    "pseudo_label": ("Pseudo-labelling", "#eb6834"),
    "ssldec": ("SSLDEC", "#1baf7a"),
}
INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"
COLUMNS = ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "ece"]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def collect(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(exp_dir.glob("*/test_metrics.json")):
        run = mpath.parent
        m = json.loads(mpath.read_text(encoding="utf-8"))
        info = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["run"]
        rows.append({"run": run.name, "method": info["method"], "labeled_fraction": info["labeled_fraction"],
                     "seed": info["seed"], **{k: m[k] for k in COLUMNS},
                     "auc_ci_low": m.get("ci95", {}).get("roc_auc", [None, None])[0],
                     "auc_ci_high": m.get("ci95", {}).get("roc_auc", [None, None])[1]})
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["labeled_fraction", "method"])
    out = g[COLUMNS].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def to_markdown(table: pd.DataFrame) -> str:
    keys = ["roc_auc", "sensitivity", "specificity", "f1", "ece"]
    lines = ["| Labelled | Method | Seeds | " + " | ".join(k.replace("_", " ") for k in keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for _, r in table.iterrows():
        cells = []
        for k in keys:
            mean, std = r[f"{k}_mean"], r[f"{k}_std"]
            cells.append(f"{mean:.3f}" + (f" ± {std:.3f}" if pd.notna(std) else ""))
        name = METHODS.get(r["method"], (r["method"],))[0]
        lines.append(f"| {r['labeled_fraction']:.0%} | {name} | {int(r['n_seeds'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_auc_vs_fraction(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for method, (name, color) in METHODS.items():
        t = table[table["method"] == method].sort_values("labeled_fraction")
        if t.empty:
            continue
        x, y, e = t["labeled_fraction"] * 100, t["roc_auc_mean"], t["roc_auc_std"].fillna(0)
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=2, label=name)
        ax.annotate(name, (x.iloc[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    style(ax, "Test ROC-AUC by labelled fraction (mean ± std over seeds)", "Labelled training patients (%)", "ROC-AUC")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_roc(df: pd.DataFrame, exp_dir: Path, fraction: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=150)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for method, (name, color) in METHODS.items():
        runs = df[(df["method"] == method) & (df["labeled_fraction"] == fraction)].sort_values("seed")
        if runs.empty:
            continue
        r = runs.iloc[0]                                  # first seed; others are in the table
        pred = pd.read_csv(exp_dir / r["run"] / "test_predictions.csv")
        fpr, tpr, _ = roc_curve(pred["label"], pred["score"])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {r['roc_auc']:.3f})")
    style(ax, f"ROC curves, test set, {fraction:.0%} labelled", "False positive rate", "Sensitivity")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    exp_dir = cfg["paths"]["experiments_dir"]

    df = collect(exp_dir)
    if df.empty:
        print(f"No evaluated runs in {exp_dir}. Run scripts/train.py then scripts/evaluate.py first.")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "summary_runs.csv", index=False)
    table = summary_table(df)
    table.to_csv(args.out_dir / "summary_table.csv", index=False)
    md = to_markdown(table)
    (args.out_dir / "summary_table.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    plot_auc_vs_fraction(table, args.out_dir / "auc_vs_fraction.png")
    for frac in sorted(df["labeled_fraction"].unique()):
        plot_roc(df, exp_dir, frac, args.out_dir / f"roc_{round(frac * 100):02d}.png")
    print(f"\nWrote tables and figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
