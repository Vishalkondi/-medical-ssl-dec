"""Binary classification metrics for imbalanced medical data.

Why not accuracy alone: with ~10% positives a model that always says
"no finding" gets ~90% accuracy and finds zero nodules. We therefore report
threshold-free ranking quality (ROC-AUC, PR-AUC), sensitivity and specificity
at a threshold chosen on the validation set, and calibration (ECE, Brier).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve,
)


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (choose on VALIDATION only)."""
    fpr, tpr, thr = roc_curve(y_true, score)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[best], 0.0, 1.0))


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    """ECE for the positive-class score: weighted mean |observed rate - mean score| per bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(score, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - score[m].mean())
    return float(ece)


def binary_metrics(y_true, score, threshold: float, n_bins: int = 10) -> dict:
    """All metrics at a fixed threshold. y_true in {0,1}; score = P(class 1)."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if y_true.shape != score.shape or y_true.ndim != 1:
        raise ValueError("y_true and score must be 1-D arrays of equal length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("need both classes present to compute ROC-AUC")
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(y_true)),
        "balanced_accuracy": float((sens + spec) / 2),
        "ece": expected_calibration_error(y_true, score, n_bins),
        "brier": float(brier_score_loss(y_true, score)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": int(len(y_true)), "n_positive": int(y_true.sum()),
    }


def bootstrap_ci(y_true, score, groups, threshold: float, n_samples: int = 1000,
                 seed: int = 0, keys=("roc_auc", "sensitivity", "specificity", "f1")) -> dict:
    """95% CIs by resampling PATIENTS (groups) with replacement, because images
    of the same patient are correlated. Returns {metric: [low, high]}."""
    y_true, score, groups = map(np.asarray, (y_true, score, groups))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    values = {k: [] for k in keys}
    for _ in range(n_samples):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = binary_metrics(y_true[idx], score[idx], threshold)
        for k in keys:
            values[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in values.items() if v}
