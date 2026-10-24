"""Evaluate finished runs on the held-out TEST set (final step, once per run).

Uses the decision threshold chosen on the validation set during training.
Writes test_metrics.json (with patient-level bootstrap 95% CIs),
test_predictions.csv and roc.png / confusion.png / calibration.png into each run folder.

Usage:
    python scripts/evaluate.py experiments/baseline_lab05_seed42
    python scripts/evaluate.py experiments/*_lab05_seed42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import binary_metrics, bootstrap_ci  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device_env import configure_cuda_visibility  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"
SERIES = "#2a78d6"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
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


def plots(run_dir: Path, y, score, m: dict, name: str) -> None:
    fpr, tpr, _ = roc_curve(y, score)
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.plot(fpr, tpr, color=SERIES, linewidth=2, label=f"{name} (AUC {m['roc_auc']:.3f})")
    ax.scatter([1 - m["specificity"]], [m["sensitivity"]], s=40, color=SERIES, edgecolor="white", zorder=3,
               label=f"threshold {m['threshold']:.2f}")
    style(ax, "ROC curve, test set", "False positive rate (1 - specificity)", "Sensitivity")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(run_dir / "roc.png")
    plt.close(fig)

    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=150)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["No finding", "Nodule/mass"])
    ax.set_yticks([0, 1], ["No finding", "Nodule/mass"])
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Label", color=MUTED)
    ax.set_title("Confusion matrix, test set", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion.png")
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y, score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.plot(mean_pred, frac_pos, color=SERIES, linewidth=2, marker="o", markersize=5)
    style(ax, f"Reliability, ECE {m['ece']:.3f}", "Mean model score", "Observed positive rate")
    fig.tight_layout()
    fig.savefig(run_dir / "calibration.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config) if args.config else load_config()
    configure_cuda_visibility(cfg["training"]["device"])
    # torch is imported only after GPU visibility is settled
    from src.models.factory import build_model
    from src.training.engine import (
        amp_enabled, load_checkpoint, load_split, make_loader, patient_groups, predict_proba, write_json,
    )
    from src.utils.config import get_device
    device = get_device(cfg["training"]["device"])
    test = load_split(cfg, "test")
    logging.info("test set: %d images, %d positive, %d patients",
                 len(test), test["label"].sum(), test["Patient ID"].nunique())

    for run_dir in args.runs:
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.is_file():
            logging.error("%s has no best.pt; skipping", run_dir)
            continue
        ckpt = load_checkpoint(ckpt_path, device)
        model = build_model(ckpt["model_spec"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        score = predict_proba(model, make_loader(cfg, test, train=False), device, amp_enabled(cfg, device))[:, 1]
        y = test["label"].to_numpy()
        m = binary_metrics(y, score, ckpt["threshold"], cfg["evaluation"]["calibration_bins"])
        m["ci95"] = bootstrap_ci(y, score, patient_groups(test), ckpt["threshold"],
                                 cfg["evaluation"]["bootstrap_samples"], seed=0)
        m["method"] = ckpt.get("method", "unknown")
        write_json(run_dir / "test_metrics.json", m)
        out = test[["Image Index", "Patient ID", "label"]].copy()
        out["score"] = score
        out.to_csv(run_dir / "test_predictions.csv", index=False)
        plots(run_dir, y, score, m, m["method"])
        logging.info("%s: AUC %.4f [%.4f, %.4f] sens %.3f spec %.3f F1 %.3f", run_dir.name, m["roc_auc"],
                     *m["ci95"].get("roc_auc", [float("nan")] * 2), m["sensitivity"], m["specificity"], m["f1"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
