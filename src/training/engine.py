"""Shared training utilities: run folders, data loaders, prediction,
validation, early stopping and checkpoints."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import CXRDataset, EvalTransform, TrainTransform
from src.data.splits import COL_PATIENT
from src.evaluation.metrics import binary_metrics, youden_threshold
from src.utils.config import get_device, split_name

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- run setup
def setup_run(cfg: dict, method: str, fraction: float, seed: int, overwrite: bool = False) -> Path:
    """experiments/<method>_lab<XX>_seed<S>/ with a config snapshot and train.log."""
    run_dir = cfg["paths"]["experiments_dir"] / f"{method}_lab{split_name(fraction)}_seed{seed}"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    snapshot = {**cfg, "paths": {k: str(v) for k, v in cfg["paths"].items()},
                "run": {"method": method, "labeled_fraction": fraction, "seed": seed}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return run_dir


def load_split(cfg: dict, name: str) -> pd.DataFrame:
    path = cfg["paths"]["splits_dir"] / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/make_splits.py first")
    return pd.read_csv(path)


def load_ssl_splits(cfg: dict, fraction: float) -> dict[str, pd.DataFrame]:
    name = split_name(fraction)
    splits = {"labeled": load_split(cfg, f"labeled_{name}"),
              "unlabeled": load_split(cfg, f"unlabeled_{name}"),
              "val": load_split(cfg, "val")}
    log.info("splits: labeled=%d (pos %d), unlabeled=%d, val=%d (pos %d)",
             len(splits["labeled"]), splits["labeled"]["label"].sum(), len(splits["unlabeled"]),
             len(splits["val"]), splits["val"]["label"].sum())
    return splits


# ---------------------------------------------------------------- loaders
def make_loader(cfg: dict, df: pd.DataFrame, train: bool, with_labels: bool = True,
                batch_size: int | None = None, drop_last: bool = False) -> DataLoader:
    d, t = cfg["data"], cfg["training"]
    tf_cls = TrainTransform if (train and t["augment"]) else EvalTransform
    ds = CXRDataset(df, cfg["paths"]["processed_dir"], tf_cls(d["image_size"], d["mean"], d["std"]),
                    with_labels=with_labels)
    workers = t["num_workers"]
    return DataLoader(ds, batch_size=batch_size or t["batch_size"], shuffle=train,
                      num_workers=workers, pin_memory=_cuda_in_use(t["device"]),
                      drop_last=drop_last and len(ds) > (batch_size or t["batch_size"]))


@lru_cache(maxsize=None)
def _cuda_in_use(preference: str) -> bool:
    """Pin memory only when training really runs on CUDA (pinning needs a working GPU)."""
    return get_device(preference).type == "cuda"


def amp_enabled(cfg: dict, device: torch.device) -> bool:
    return bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_proba(model, loader: DataLoader, device: torch.device, amp: bool,
                  return_embeddings: bool = False):
    """Class probabilities (N, K) in dataset order; optionally SSLDEC embeddings."""
    model.eval()
    probs, embs = [], []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            if return_embeddings:
                z, q = model(x)
                embs.append(z.float().cpu())
                probs.append(q.float().cpu())
            else:
                probs.append(model.predict_proba(x).float().cpu())
    p = torch.cat(probs).numpy()
    if not np.isfinite(p).all():
        raise FloatingPointError("non-finite probabilities; try mixed_precision: false or a lower lr")
    return (p, torch.cat(embs)) if return_embeddings else p


def evaluate_split(model, cfg: dict, df: pd.DataFrame, device, threshold: float | None = None) -> tuple[dict, np.ndarray]:
    """Metrics on a labelled split. threshold=None -> choose by Youden J (val only)."""
    score = predict_proba(model, make_loader(cfg, df, train=False), device, amp_enabled(cfg, device))[:, 1]
    y = df["label"].to_numpy()
    thr = youden_threshold(y, score) if threshold is None else threshold
    return binary_metrics(y, score, thr, cfg["evaluation"]["calibration_bins"]), score


# ---------------------------------------------------------------- bookkeeping
@dataclass
class EarlyStopping:
    patience: int
    best: float = -np.inf
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        """Returns True if this is a new best."""
        if value > self.best + 1e-4:
            self.best, self.bad_epochs = value, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def save_checkpoint(path: Path, model, spec: dict, cfg: dict, extra: dict) -> None:
    """Plain dict of tensors/primitives, loadable with torch.load(weights_only=True)."""
    torch.save({
        "state_dict": model.state_dict(),
        "model_spec": spec,
        "classes": list(cfg["task"]["classes"]),
        "image_size": cfg["data"]["image_size"],
        "preprocess_size": cfg["data"]["preprocess_size"],
        "mean": list(cfg["data"]["mean"]),
        "std": list(cfg["data"]["std"]),
        **extra,
    }, path)


def load_checkpoint(path: Path, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_history(run_dir: Path, row: dict) -> None:
    path = run_dir / "history.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def finalize_run(run_dir: Path, model, spec: dict, cfg: dict, val_df: pd.DataFrame, device, method: str) -> dict:
    """Reload the best weights, pick the decision threshold on validation,
    store it in the checkpoint and write val_metrics.json."""
    ckpt = load_checkpoint(run_dir / "best.pt", device)
    model.load_state_dict(ckpt["state_dict"])
    metrics, _ = evaluate_split(model, cfg, val_df, device, threshold=None)
    save_checkpoint(run_dir / "best.pt", model, spec, cfg,
                    {"threshold": metrics["threshold"], "method": method,
                     "best_epoch": ckpt.get("best_epoch", -1), "val_roc_auc": metrics["roc_auc"]})
    write_json(run_dir / "val_metrics.json", metrics)
    log.info("best val ROC-AUC %.4f, threshold %.3f", metrics["roc_auc"], metrics["threshold"])
    return metrics


def patient_groups(df: pd.DataFrame) -> np.ndarray:
    return df[COL_PATIENT].to_numpy()
