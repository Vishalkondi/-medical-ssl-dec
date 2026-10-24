"""Method 2: basic pseudo-labelling (Lee, 2013), offline rounds.

1. Warm up on labelled data (same loop as the baseline).
2. Each round: predict a random subset of the unlabelled pool, keep
   confident predictions (balanced to the labelled class prior), and train on
   labelled + pseudo-labelled images.
The best checkpoint across all phases is kept by validation ROC-AUC.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.factory import build_model, model_spec
from src.training.engine import (
    amp_enabled, append_history, finalize_run, load_checkpoint, load_ssl_splits, make_loader,
    predict_proba,
)
from src.training.losses import class_weights, select_pseudo_labels
from src.training.supervised import classifier_loss, fit_labeled

log = logging.getLogger(__name__)


def labeled_prior(labels: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(labels, minlength=k) / len(labels)


def train_pseudo_label(cfg: dict, fraction: float, seed: int, run_dir: Path, device) -> dict:
    splits = load_ssl_splits(cfg, fraction)
    labeled, unlabeled, val = splits["labeled"], splits["unlabeled"], splits["val"]
    k = len(cfg["task"]["classes"])
    pl, b = cfg["pseudo_label"], cfg["baseline"]
    spec = model_spec(cfg, "classifier", k)
    model = build_model(spec, pretrained=cfg["model"]["pretrained"]).to(device)
    weight = class_weights(labeled["label"].to_numpy(), k).to(device) if cfg["training"]["class_weighted_loss"] else None
    prior = labeled_prior(labeled["label"].to_numpy(), k) if pl["balance_to_prior"] else None
    rng = np.random.default_rng(seed)

    stopper = fit_labeled(model, spec, cfg, labeled, val, device, run_dir, loss_fn=classifier_loss(weight),
                          epochs=pl["warmup_epochs"], lr=b["lr"], patience=b["patience"], phase="warmup")
    epoch = pl["warmup_epochs"]

    for r in range(pl["rounds"]):
        # continue from the best weights so far, not from the last (possibly worse) epoch
        model.load_state_dict(load_checkpoint(run_dir / "best.pt", device)["state_dict"])
        n = min(pl["unlabeled_per_round"], len(unlabeled))
        pool = unlabeled.iloc[rng.choice(len(unlabeled), size=n, replace=False)].reset_index(drop=True)
        probs = predict_proba(model, make_loader(cfg, pool, train=False, with_labels=False),
                              device, amp_enabled(cfg, device))
        idx = select_pseudo_labels(probs, pl["confidence_threshold"], prior)
        if len(idx) == 0:
            log.warning("[round %d] no confident pseudo-labels for every class; stopping rounds", r)
            break
        pseudo = pool.iloc[idx].copy()
        pseudo["label"] = probs[idx].argmax(axis=1)           # predicted, NOT the hidden label
        # Diagnostic only (never used for training): how many pseudo-labels are wrong.
        acc = float((pool.iloc[idx]["label"].to_numpy() == pseudo["label"].to_numpy()).mean())
        log.info("[round %d] %d pseudo-labels (%s per class), diagnostic accuracy %.3f", r, len(idx),
                 np.bincount(pseudo["label"], minlength=k).tolist(), acc)
        append_history(run_dir, {"epoch": epoch, "phase": f"round{r}_select", "n_pseudo": len(idx),
                                 "pseudo_label_accuracy_diag": acc})
        train_df = pd.concat([labeled, pseudo], ignore_index=True)
        stopper = fit_labeled(model, spec, cfg, train_df, val, device, run_dir,
                              loss_fn=classifier_loss(weight), epochs=pl["epochs_per_round"], lr=pl["lr"],
                              patience=pl["patience"], phase=f"round{r}", stopper=stopper, epoch_offset=epoch)
        epoch += pl["epochs_per_round"]

    return finalize_run(run_dir, model, spec, cfg, val, device, "pseudo_label")
