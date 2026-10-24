"""PyTorch datasets and transforms for the preprocessed chest X-rays.

All classes are module-level (no lambdas) so DataLoader workers can pickle
them on Windows, where workers are spawned rather than forked.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.data.preprocessing import resize_square, to_model_input
from src.data.splits import COL_IMAGE

UNLABELED = -1


class EvalTransform:
    """Deterministic: resize to image_size, normalise. Mirrors inference."""

    def __init__(self, image_size: int, mean: list[float], std: list[float]):
        self.image_size, self.mean, self.std = image_size, mean, std

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(to_model_input(resize_square(img, self.image_size), self.mean, self.std))


class TrainTransform:
    """Mild augmentation suited to chest X-rays. No horizontal flip: it would
    swap anatomical sides (e.g. the heart), which never happens in real PA/AP films."""

    def __init__(self, image_size: int, mean: list[float], std: list[float]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.aug = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomAffine(degrees=7, translate=(0.03, 0.03)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img).float().div_(255.0).unsqueeze(0)   # (1, H, W)
        x = self.aug(x).clamp_(0.0, 1.0).expand(3, -1, -1)
        return (x - self.mean) / self.std


class CXRDataset(Dataset):
    """Returns (image_tensor, label, index).

    with_labels=False returns label = -1 for every item. Use it for the
    unlabelled pool so training code physically cannot see the hidden labels.
    """

    def __init__(self, df: pd.DataFrame, image_dir: str | Path, transform, with_labels: bool = True):
        if COL_IMAGE not in df.columns:
            raise KeyError(f"dataframe needs a '{COL_IMAGE}' column")
        if with_labels and "label" not in df.columns:
            raise KeyError("with_labels=True but dataframe has no 'label' column")
        self.image_dir = Path(image_dir)
        self.names = df[COL_IMAGE].tolist()
        self.labels = df["label"].to_numpy(np.int64) if with_labels else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        path = self.image_dir / self.names[i]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"cannot read {path}; did prepare_images.py finish?")
        label = int(self.labels[i]) if self.labels is not None else UNLABELED
        return self.transform(img), label, i
