"""Model loading and inference for the API.

Checkpoint contract (written by scripts/export_model.py):
    artifacts/model.pt         state_dict only, loaded with weights_only=True
    artifacts/model_meta.json  {
        "model_version": "...",
        "model_type": "classifier" | "ssldec",
        "architecture": "densenet121",
        "embedding_dim": 64, "dec_alpha": 1.0,      # ssldec only
        "classes": ["no_finding", "nodule_or_mass"],
        "image_size": 224, "preprocess_size": 256,
        "mean": [...], "std": [...],
        "threshold": 0.42,            # chosen on the validation set
        "score_type": "uncalibrated" | "calibrated",
        "metrics": {"test_roc_auc": ...}   # from a real test evaluation
    }
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data.preprocessing import inference_preprocess

log = logging.getLogger(__name__)

DISPLAY_NAMES = {
    "no_finding": "No finding labelled",
    "nodule_or_mass": "Nodule or mass (suspicious focal opacity)",
}
REQUIRED_META = ("model_version", "architecture", "classes", "image_size", "mean", "std")


@dataclass
class Prediction:
    predicted_label: str
    scores: dict[str, float]
    threshold: float
    inference_ms: float


@dataclass
class Predictor:
    model_path: Path
    meta_path: Path
    device_pref: str = "auto"
    loaded: bool = False
    message: str = "not initialised"
    meta: dict = field(default_factory=dict)
    _model: object = None
    _device: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def load(self) -> None:
        """Try to load the model. Never raises: failures are reported via
        `loaded` / `message` so the API can still start and explain the state."""
        self.loaded = False
        if not self.model_path.is_file() or not self.meta_path.is_file():
            self.message = ("No trained model found yet. Train a model (Phase 4/5) "
                            f"to create {self.model_path.name} and {self.meta_path.name}.")
            log.info(self.message)
            return
        try:
            import torch

            from src.models.factory import build_model
            from src.utils.config import get_device

            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            missing = [k for k in REQUIRED_META if k not in meta]
            if missing:
                raise KeyError(f"model_meta.json missing {missing}")
            spec = {"type": meta.get("model_type", "classifier"), "arch": meta["architecture"],
                    "num_classes": len(meta["classes"]),
                    "embedding_dim": meta.get("embedding_dim"), "alpha": meta.get("dec_alpha")}
            device = get_device(self.device_pref)
            model = build_model(spec, pretrained=False)
            model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
            model.to(device).eval()
            meta.setdefault("preprocess_size", meta["image_size"])
            self._model, self._device, self.meta = model, device, meta
            self.loaded = True
            self.message = f"Model {meta['model_version']} loaded on {device}."
            log.info(self.message)
        except Exception as exc:  # noqa: BLE001 - report any load failure
            self.message = f"Model files found but failed to load: {exc}"
            log.exception("model load failed")

    def predict(self, gray: np.ndarray) -> Prediction:
        """gray: 2-D uint8 image (H, W). Applies the training evaluation pipeline."""
        if not self.loaded:
            raise RuntimeError(self.message)
        import torch

        m = self.meta
        size = int(m["image_size"])
        x = inference_preprocess(gray, int(m["preprocess_size"]), size, m["mean"], m["std"])
        tensor = torch.from_numpy(x[None]).to(self._device)          # (1, 3, H, W)
        assert tensor.shape == (1, 3, size, size), tensor.shape

        start = time.perf_counter()
        with self._lock, torch.no_grad():
            probs = self._model.predict_proba(tensor)[0].float().cpu().numpy()
        elapsed = (time.perf_counter() - start) * 1000

        classes = m["classes"]
        threshold = float(m.get("threshold", 0.5))
        scores = {c: float(p) for c, p in zip(classes, probs)}
        # Binary task: the positive class (index 1) uses the validation-chosen threshold.
        if len(classes) == 2:
            label = classes[1] if probs[1] >= threshold else classes[0]
        else:
            label = classes[int(np.argmax(probs))]
        return Prediction(label, scores, threshold, elapsed)
