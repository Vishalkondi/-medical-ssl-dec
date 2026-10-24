"""CNN backbones shared by the baseline, SSLDEC and the inference service."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

FEATURE_DIMS = {"densenet121": 1024, "resnet50": 2048}


def build_backbone(arch: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return (feature_extractor, feature_dim). The extractor maps
    (B, 3, H, W) -> (B, feature_dim) pooled features."""
    if arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.densenet121(weights=weights)
        net.classifier = nn.Identity()
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = nn.Identity()
    else:
        raise ValueError(f"unknown backbone '{arch}'; choose from {sorted(FEATURE_DIMS)}")
    return net, FEATURE_DIMS[arch]


class Classifier(nn.Module):
    """Backbone + linear head producing class logits (supervised baseline)."""

    def __init__(self, arch: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone, dim = build_backbone(arch, pretrained)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected input (B, 3, H, W), got {tuple(x.shape)}")
        return self.head(self.backbone(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x).float(), dim=1)
