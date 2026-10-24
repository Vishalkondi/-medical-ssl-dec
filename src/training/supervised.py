"""Method 1: supervised-only baseline (labelled images only).

Also provides `fit_labeled`, the supervised loop reused by the pseudo-label
method and by SSLDEC's warm-up, so all methods share identical training code.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.models.factory import build_model, model_spec
from src.training.engine import (
    EarlyStopping, amp_enabled, append_history, evaluate_split, finalize_run, load_ssl_splits,
    make_loader, save_checkpoint,
)
from src.training.losses import class_weights

log = logging.getLogger(__name__)

LossFn = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def classifier_loss(weight: torch.Tensor | None) -> LossFn:
    def fn(model, x, y):
        return F.cross_entropy(model(x).float(), y, weight=weight)
    return fn


def make_optimizer(model, cfg: dict, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])


def fit_labeled(model, spec: dict, cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
                device, run_dir: Path, *, loss_fn: LossFn, epochs: int, lr: float, patience: int,
                phase: str, stopper: EarlyStopping | None = None, epoch_offset: int = 0) -> EarlyStopping:
    """Train on labelled rows with `loss_fn`; validate each epoch; save best.pt
    on a new best validation ROC-AUC. Returns the early-stopping state so later
    phases continue comparing against the same best."""
    amp = amp_enabled(cfg, device)
    loader = make_loader(cfg, train_df, train=True, drop_last=True)
    opt = make_optimizer(model, cfg, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    stopper = stopper or EarlyStopping(patience)
    stopper.bad_epochs, stopper.patience = 0, patience

    for e in range(epochs):
        epoch = epoch_offset + e
        model.train()
        start, total, n = time.perf_counter(), 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model, x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total, n = total + loss.item() * len(y), n + len(y)
        sched.step()
        val, _ = evaluate_split(model, cfg, val_df, device, threshold=0.5)
        improved = stopper.step(val["roc_auc"])
        if improved:
            save_checkpoint(run_dir / "best.pt", model, spec, cfg, {"best_epoch": epoch, "threshold": 0.5})
        row = {"epoch": epoch, "phase": phase, "train_loss": total / max(n, 1),
               "val_roc_auc": val["roc_auc"], "val_pr_auc": val["pr_auc"],
               "lr": opt.param_groups[0]["lr"], "seconds": time.perf_counter() - start}
        append_history(run_dir, row)
        log.info("[%s] epoch %d loss %.4f val AUC %.4f%s", phase, epoch, row["train_loss"],
                 val["roc_auc"], " *best*" if improved else "")
        if stopper.should_stop:
            log.info("[%s] early stopping after %d epochs without improvement", phase, patience)
            break
    return stopper


def train_baseline(cfg: dict, fraction: float, seed: int, run_dir: Path, device) -> dict:
    splits = load_ssl_splits(cfg, fraction)
    labeled = splits["labeled"]
    k = len(cfg["task"]["classes"])
    spec = model_spec(cfg, "classifier", k)
    model = build_model(spec, pretrained=cfg["model"]["pretrained"]).to(device)
    weight = class_weights(labeled["label"].to_numpy(), k).to(device) if cfg["training"]["class_weighted_loss"] else None
    b = cfg["baseline"]
    fit_labeled(model, spec, cfg, labeled, splits["val"], device, run_dir,
                loss_fn=classifier_loss(weight), epochs=b["epochs"], lr=b["lr"],
                patience=b["patience"], phase="supervised")
    return finalize_run(run_dir, model, spec, cfg, splits["val"], device, "baseline")
