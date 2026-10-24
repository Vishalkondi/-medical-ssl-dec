"""Method 3: SSLDEC (semi-supervised deep embedded clustering).

Schedule (adapted from Enguehard et al., 2019; changes documented in
docs/phase1_planning.md, Section 3.3):

1. Centre initialisation: mu_j = mean embedding of labelled class-j images.
2. Warm-up: train backbone + projection + centres with L_sup = -log q_{i,y_i}
   on labelled images only (paper: "pretrain on labelled data").
3. SSL epochs, each:
     a. predict q on a random subset of the unlabelled pool (no augmentation)
     b. P = target_distribution(Q, prior)        dataset-level f_j
     c. keep confident rows (max_j p_ij >= tau), balanced to the class prior
     d. train on augmented images with
            L = L_sup(labelled batch) + lambda(t) * KL(P || Q)(unlabelled batch)
        lambda(t) ramps up from 0 so early, unreliable targets have little weight.
4. Early stopping and model selection on validation ROC-AUC (not on 100%
   labelled accuracy as in the paper, which overfits a pretrained network).
"""
from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.models.factory import build_model, model_spec
from src.training.engine import (
    EarlyStopping, amp_enabled, append_history, evaluate_split, finalize_run, load_checkpoint,
    load_ssl_splits, make_loader, predict_proba, save_checkpoint,
)
from src.training.losses import (
    class_weights, dec_kl_loss, select_pseudo_labels, sigmoid_rampup, supervised_q_loss,
)
from src.models.dec import target_distribution
from src.training.pseudo_label import labeled_prior
from src.training.supervised import fit_labeled, make_optimizer

log = logging.getLogger(__name__)


def q_loss(weight):
    def fn(model, x, y):
        return supervised_q_loss(model(x)[1], y, weight)
    return fn


@torch.no_grad()
def init_centers_from_labels(model, cfg, labeled, device) -> None:
    probs, z = predict_proba(model, make_loader(cfg, labeled, train=False), device,
                             amp_enabled(cfg, device), return_embeddings=True)
    y = torch.as_tensor(labeled["label"].to_numpy())
    model.cluster.init_centers(z, y)
    dist = torch.cdist(model.cluster.centers.detach().cpu(), model.cluster.centers.detach().cpu())
    log.info("centres initialised from %d labelled embeddings; centre distance %.3f", len(y), dist[0, 1].item())


def train_ssldec(cfg: dict, fraction: float, seed: int, run_dir: Path, device) -> dict:
    splits = load_ssl_splits(cfg, fraction)
    labeled, unlabeled, val = splits["labeled"], splits["unlabeled"], splits["val"]
    k = len(cfg["task"]["classes"])
    s = cfg["ssldec"]
    amp = amp_enabled(cfg, device)
    rng = np.random.default_rng(seed)

    spec = model_spec(cfg, "ssldec", k)
    model = build_model(spec, pretrained=cfg["model"]["pretrained"]).to(device)
    labels = labeled["label"].to_numpy()
    weight = class_weights(labels, k).to(device) if cfg["training"]["class_weighted_loss"] else None
    prior_np = labeled_prior(labels, k)
    prior = torch.tensor(prior_np, dtype=torch.float32) if s["prior_correction"] else None
    log.info("labelled class prior %s", np.round(prior_np, 4).tolist())

    # 1-2. centre initialisation + supervised warm-up on q
    init_centers_from_labels(model, cfg, labeled, device)
    stopper = fit_labeled(model, spec, cfg, labeled, val, device, run_dir, loss_fn=q_loss(weight),
                          epochs=s["warmup_epochs"], lr=cfg["baseline"]["lr"],
                          patience=cfg["baseline"]["patience"], phase="warmup")
    model.load_state_dict(load_checkpoint(run_dir / "best.pt", device)["state_dict"])
    stopper.bad_epochs, stopper.patience = 0, s["patience"]

    opt = make_optimizer(model, cfg, s["lr"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    labeled_loader = make_loader(cfg, labeled, train=True, drop_last=True)

    for e in range(s["epochs"]):
        epoch = s["warmup_epochs"] + e
        start = time.perf_counter()

        # a-c. targets on an unlabelled subset (labels hidden: with_labels=False)
        n = min(s["unlabeled_per_epoch"], len(unlabeled))
        pool = unlabeled.iloc[rng.choice(len(unlabeled), size=n, replace=False)].reset_index(drop=True)
        q_all = torch.from_numpy(predict_proba(model, make_loader(cfg, pool, train=False, with_labels=False),
                                               device, amp))
        p_all = target_distribution(q_all, prior)
        idx = select_pseudo_labels(p_all.numpy(), s["confidence_threshold"],
                                   prior_np if s["balance_to_prior"] else None)
        lam = s["lambda_max"] * sigmoid_rampup(e, s["rampup_epochs"])

        # Diagnostic only: agreement of argmax P with the hidden labels.
        diag = float((p_all[idx].argmax(1).numpy() == pool["label"].to_numpy()[idx]).mean()) if len(idx) else float("nan")

        # d. joint training
        model.train()
        tot_sup = tot_dec = 0.0
        steps = 0
        if len(idx):
            sel = pool.iloc[idx].reset_index(drop=True)
            targets = p_all[idx].to(device)                            # row i <-> dataset index i
            unl_loader = make_loader(cfg, sel, train=True, with_labels=False, drop_last=True)
            batches = zip(itertools.cycle(labeled_loader), unl_loader)
        else:
            log.warning("[ssl] epoch %d: no confident unlabelled samples; labelled-only epoch", epoch)
            batches = ((b, None) for b in labeled_loader)
        for (xl, yl, _), unl in batches:
            xl, yl = xl.to(device, non_blocking=True), yl.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                loss_sup = supervised_q_loss(model(xl)[1], yl, weight)
                loss_dec = torch.zeros((), device=device)
                if unl is not None:
                    xu, _, iu = unl
                    _, qu = model(xu.to(device, non_blocking=True))
                    loss_dec = dec_kl_loss(qu, targets[iu.to(device)])
                loss = loss_sup + lam * loss_dec
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_sup, tot_dec, steps = tot_sup + loss_sup.item(), tot_dec + loss_dec.item(), steps + 1

        val_m, _ = evaluate_split(model, cfg, val, device, threshold=0.5)
        improved = stopper.step(val_m["roc_auc"])
        if improved:
            save_checkpoint(run_dir / "best.pt", model, spec, cfg, {"best_epoch": epoch, "threshold": 0.5})
        row = {"epoch": epoch, "phase": "ssl", "train_loss": (tot_sup + lam * tot_dec) / max(steps, 1),
               "loss_sup": tot_sup / max(steps, 1), "loss_dec": tot_dec / max(steps, 1), "lambda": lam,
               "n_pseudo": len(idx), "pseudo_label_accuracy_diag": diag,
               "val_roc_auc": val_m["roc_auc"], "val_pr_auc": val_m["pr_auc"],
               "seconds": time.perf_counter() - start}
        append_history(run_dir, row)
        log.info("[ssl] epoch %d sup %.4f dec %.4f lambda %.2f selected %d (diag acc %.3f) val AUC %.4f%s",
                 epoch, row["loss_sup"], row["loss_dec"], lam, len(idx), diag, val_m["roc_auc"],
                 " *best*" if improved else "")
        if stopper.should_stop:
            log.info("[ssl] early stopping")
            break

    return finalize_run(run_dir, model, spec, cfg, val, device, "ssldec")
