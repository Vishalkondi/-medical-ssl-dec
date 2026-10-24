"""Build any project model from a plain dict, so checkpoints and the API
can recreate models without pickling Python objects."""
from __future__ import annotations

from torch import nn

from src.models.backbone import Classifier
from src.models.ssl_model import SSLDECModel


def model_spec(cfg: dict, model_type: str, num_classes: int) -> dict:
    """Serialisable description of a model, stored in checkpoints."""
    spec = {"type": model_type, "arch": cfg["model"]["backbone"], "num_classes": num_classes}
    if model_type == "ssldec":
        spec.update(embedding_dim=cfg["model"]["embedding_dim"], alpha=cfg["model"]["dec_alpha"])
    return spec


def build_model(spec: dict, pretrained: bool = False) -> nn.Module:
    if spec["type"] == "classifier":
        return Classifier(spec["arch"], spec["num_classes"], pretrained=pretrained)
    if spec["type"] == "ssldec":
        return SSLDECModel(spec["arch"], spec["num_classes"], spec["embedding_dim"],
                           spec["alpha"], pretrained=pretrained)
    raise ValueError(f"unknown model type '{spec['type']}'")
