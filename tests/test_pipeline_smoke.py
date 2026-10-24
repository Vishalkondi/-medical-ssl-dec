"""End-to-end smoke test on tiny synthetic images (CPU, ~1-3 min).

Runs the real CLI scripts: train (all three methods) -> evaluate -> export ->
compare_results, then loads the exported model in the backend predictor.
It checks that the pipeline runs and the files/contract are right, not that
the model is good. Skip with: pytest -m "not slow"
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
SIZE = 64


def make_image(rng, positive: bool) -> np.ndarray:
    img = (np.tile(np.linspace(40, 120, SIZE), (SIZE, 1)) + rng.normal(0, 8, (SIZE, SIZE)))
    if positive:  # a bright round "lesion"
        cy, cx = rng.integers(16, SIZE - 16, 2)
        yy, xx = np.ogrid[:SIZE, :SIZE]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] += 110
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    rng = np.random.default_rng(0)
    proc, splits = tmp / "processed", tmp / "splits"
    proc.mkdir()
    splits.mkdir()
    rows = []
    for pid in range(80):
        positive_patient = pid % 3 == 0
        for k in range(2):
            name = f"{pid:08d}_{k:03d}.png"
            label = int(positive_patient and k == 0)
            cv2.imwrite(str(proc / name), make_image(rng, bool(label)))
            rows.append({"Image Index": name, "Patient ID": pid, "View Position": "PA",
                         "Finding Labels": "Nodule" if label else "No Finding", "label": label})
    df = pd.DataFrame(rows)
    pid = df["Patient ID"]
    df[pid < 16].to_csv(splits / "test.csv", index=False)
    df[(pid >= 16) & (pid < 28)].to_csv(splits / "val.csv", index=False)
    df[(pid >= 28) & (pid < 46)].to_csv(splits / "labeled_10.csv", index=False)
    df[pid >= 46].to_csv(splits / "unlabeled_10.csv", index=False)

    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {"raw_dir": str(tmp / "raw"), "processed_dir": str(proc), "splits_dir": str(splits),
                    "artifacts_dir": str(tmp / "artifacts"), "experiments_dir": str(tmp / "experiments")}
    cfg["data"].update(image_size=SIZE, preprocess_size=SIZE, default_labeled_fraction=0.10)
    cfg["model"].update(pretrained=False, embedding_dim=16)
    cfg["training"].update(device="cpu", batch_size=8, num_workers=0, mixed_precision=False)
    cfg["baseline"].update(epochs=2, patience=5)
    cfg["pseudo_label"].update(warmup_epochs=1, rounds=1, epochs_per_round=1, confidence_threshold=0.5)
    cfg["ssldec"].update(warmup_epochs=1, epochs=2, rampup_epochs=1, confidence_threshold=0.5)
    cfg["evaluation"]["bootstrap_samples"] = 20
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp, cfg_path


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"{args} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r


@pytest.mark.parametrize("method", ["baseline", "pseudo_label", "ssldec"])
def test_train_and_evaluate(workspace, method):
    tmp, cfg = workspace
    run("scripts/train.py", "--method", method, "--config", str(cfg))
    run_dir = tmp / "experiments" / f"{method}_lab10_seed42"
    for f in ("best.pt", "history.csv", "val_metrics.json", "train.log", "config.yaml"):
        assert (run_dir / f).is_file(), f
    run("scripts/evaluate.py", str(run_dir), "--config", str(cfg))
    m = json.loads((run_dir / "test_metrics.json").read_text())
    assert 0.0 <= m["roc_auc"] <= 1.0 and m["n"] == 32
    for f in ("roc.png", "confusion.png", "calibration.png", "test_predictions.csv"):
        assert (run_dir / f).is_file(), f


def test_rerun_without_overwrite_is_refused(workspace):
    tmp, cfg = workspace
    r = subprocess.run([sys.executable, "scripts/train.py", "--method", "baseline", "--config", str(cfg)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "overwrite" in r.stderr


def test_compare_export_and_serve(workspace):
    tmp, cfg = workspace
    run("scripts/compare_results.py", "--config", str(cfg), "--out-dir", str(tmp / "results"))
    assert (tmp / "results" / "summary_table.md").is_file()
    assert (tmp / "results" / "auc_vs_fraction.png").is_file()

    run("scripts/export_model.py", str(tmp / "experiments" / "ssldec_lab10_seed42"), "--config", str(cfg))
    meta = json.loads((tmp / "artifacts" / "model_meta.json").read_text())
    assert meta["model_type"] == "ssldec" and "test_roc_auc" in meta["metrics"]

    from backend.services.predictor import Predictor
    p = Predictor(tmp / "artifacts" / "model.pt", tmp / "artifacts" / "model_meta.json", device_pref="cpu")
    p.load()
    assert p.loaded, p.message
    pred = p.predict(make_image(np.random.default_rng(1), True))
    assert pred.predicted_label in {"no_finding", "nodule_or_mass"}
    assert sum(pred.scores.values()) == pytest.approx(1.0, abs=1e-4)
