"""Losses and pseudo-label selection shared by the training methods."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1: w_c = N / (K * n_c)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"every class needs labelled samples; counts={counts.tolist()}")
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def supervised_q_loss(q: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """L_sup = -log q_{i, y_i}: cross-entropy on the soft assignments, which equals
    KL(one-hot(y_i) || q_i). This is how SSLDEC ties cluster j to class j."""
    return F.nll_loss(torch.log(q.clamp_min(1e-8)), y, weight=weight)


def dec_kl_loss(q: torch.Tensor, p: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """L_DEC = mean_i KL(p_i || q_i) = mean_i sum_j p_ij log(p_ij / q_ij), over
    masked (confident) samples. p is a fixed target (no gradient)."""
    if q.shape != p.shape:
        raise ValueError(f"q {tuple(q.shape)} and p {tuple(p.shape)} must match")
    p = p.detach()
    kl = (p * (torch.log(p.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8)))).sum(dim=1)
    if mask is None:
        return kl.mean()
    mask = mask.to(kl.dtype)
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


def sigmoid_rampup(epoch: int, rampup_epochs: int) -> float:
    """exp(-5 (1 - t)^2), t = epoch / rampup (Laine & Aila, 2017). 0 -> 1."""
    if rampup_epochs <= 0:
        return 1.0
    t = min(max(epoch, 0), rampup_epochs) / rampup_epochs
    return float(math.exp(-5.0 * (1.0 - t) ** 2))


def select_pseudo_labels(probs: np.ndarray, threshold: float, prior: np.ndarray | None = None) -> np.ndarray:
    """Indices of unlabelled samples to use as pseudo-labels.

    1. Keep samples whose top class probability >= threshold.
    2. If `prior` is given, subsample each predicted class so that the kept
       class ratio matches the prior (the most confident are kept). This stops
       the majority class from flooding the pseudo-labels, one of the main ways
       wrong labels propagate.
    """
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    keep = conf >= threshold
    if prior is None:
        return np.flatnonzero(keep)
    k = probs.shape[1]
    counts = np.array([(keep & (pred == c)).sum() for c in range(k)])
    if (counts == 0).any():
        return np.array([], dtype=int)       # cannot balance; skip this round
    scale = min(counts[c] / prior[c] for c in range(k))
    chosen = []
    for c in range(k):
        idx = np.flatnonzero(keep & (pred == c))
        n = max(1, int(scale * prior[c]))
        chosen.append(idx[np.argsort(-conf[idx])][:n])
    return np.sort(np.concatenate(chosen))
