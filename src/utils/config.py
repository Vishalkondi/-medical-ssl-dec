"""Config loading and reproducibility helpers shared by all scripts."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

log = logging.getLogger(__name__)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML config and resolve every entry under `paths` to an absolute Path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: _resolve(v) for k, v in cfg["paths"].items()}
    return cfg


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto"):
    """Return a usable torch.device. 'auto' tries CUDA with a real allocation
    (torch.cuda.is_available() can be True while the device is unusable) and
    falls back to CPU with a warning."""
    import torch

    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:  # noqa: BLE001 - driver/runtime errors vary
            if preference == "cuda":
                raise RuntimeError(f"CUDA requested but unusable: {exc}") from exc
            log.warning("CUDA detected but unusable (%s); falling back to CPU", exc)
    elif preference == "cuda":
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


def split_name(fraction: float) -> str:
    """0.05 -> '05' (matches files written by scripts/make_splits.py)."""
    return f"{round(fraction * 100):02d}"
