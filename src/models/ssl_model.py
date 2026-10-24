"""SSLDEC network: pretrained CNN -> projection head -> embedding z -> clustering layer q."""
from __future__ import annotations

import torch
from torch import nn

from src.models.backbone import build_backbone
from src.models.dec import ClusteringLayer


class SSLDECModel(nn.Module):
    def __init__(self, arch: str, num_classes: int, embedding_dim: int = 64,
                 alpha: float = 1.0, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(arch, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )
        self.cluster = ClusteringLayer(num_classes, embedding_dim, alpha)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected input (B, 3, H, W), got {tuple(x.shape)}")
        return self.projection(self.backbone(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z, q): embeddings (B, D) and soft assignments (B, K)."""
        z = self.embed(x)
        return z, self.cluster(z)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]
