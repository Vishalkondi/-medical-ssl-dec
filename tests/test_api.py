"""API tests. They run without a trained model; prediction-path tests that
need torch are skipped when it is not installed."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main


def png_bytes(h=256, w=256):
    img = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # point the predictor at an empty folder so no real model is picked up
    monkeypatch.setattr(main.predictor, "model_path", tmp_path / "model.pt")
    monkeypatch.setattr(main.predictor, "meta_path", tmp_path / "model_meta.json")
    with TestClient(main.app) as c:
        yield c


def upload(client, data, name="x.png", ctype="image/png"):
    return client.post("/api/predict", files={"file": (name, data, ctype)})


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_status_without_model(client):
    body = client.get("/api/model/status").json()
    assert body["loaded"] is False
    assert body["classes"] == ["no_finding", "nodule_or_mass"]
    assert "No trained model" in body["message"]


def test_valid_image_without_model_returns_503(client):
    r = upload(client, png_bytes())
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_NOT_LOADED"


@pytest.mark.parametrize("data,ctype,status,code", [
    (b"hello", "text/plain", 415, "UNSUPPORTED_TYPE"),
    (b"", "image/png", 400, "EMPTY_FILE"),
    (b"not really a png", "image/png", 400, "INVALID_IMAGE"),
    (b"\x89PNG\r\n\x1a\n" + b"corrupt", "image/png", 400, "INVALID_IMAGE"),
])
def test_invalid_uploads(client, data, ctype, status, code):
    r = upload(client, data, ctype=ctype)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code


def test_blank_image_rejected(client):
    ok, buf = cv2.imencode(".png", np.zeros((256, 256), np.uint8))
    r = upload(client, buf.tobytes())
    assert r.status_code == 400 and "constant" in r.json()["error"]["message"]


def test_too_large_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    r = upload(client, png_bytes())
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_missing_file_field(client):
    r = client.post("/api/predict")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_REQUEST"


def test_prediction_with_tiny_untrained_model(tmp_path, monkeypatch):
    """End-to-end prediction path with a randomly initialised model.
    Checks the response contract only; the scores are meaningless."""
    torch = pytest.importorskip("torch")
    from src.models.backbone import Classifier

    torch.save(Classifier("resnet50", 2, pretrained=False).state_dict(), tmp_path / "model.pt")
    (tmp_path / "model_meta.json").write_text(json.dumps({
        "model_version": "test-untrained", "model_type": "classifier", "architecture": "resnet50",
        "classes": ["no_finding", "nodule_or_mass"], "image_size": 64,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "threshold": 0.5,
    }))
    monkeypatch.setattr(main.predictor, "model_path", tmp_path / "model.pt")
    monkeypatch.setattr(main.predictor, "meta_path", tmp_path / "model_meta.json")
    monkeypatch.setattr(main.predictor, "device_pref", "cpu")
    with TestClient(main.app) as c:
        assert c.get("/api/model/status").json()["loaded"] is True
        r = upload(c, png_bytes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_label"] in {"no_finding", "nodule_or_mass"}
    assert abs(sum(s["score"] for s in body["scores"]) - 1) < 1e-4
    assert body["score_type"] == "uncalibrated"
    assert "not a medical device" in body["disclaimer"]
