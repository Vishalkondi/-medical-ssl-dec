"""Export a trained run for the web API (artifacts/model.pt + model_meta.json).

Usage:
    python scripts/export_model.py experiments/ssldec_lab05_seed42

Test metrics are included only if scripts/evaluate.py has produced
test_metrics.json for that run; otherwise the model card shows none.
Restart the backend afterwards to load the new model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.engine import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402

METRIC_KEYS = ["roc_auc", "sensitivity", "specificity", "precision", "f1", "accuracy"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: paths.artifacts_dir")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    out_dir = args.out_dir or cfg["paths"]["artifacts_dir"]
    ckpt = load_checkpoint(args.run / "best.pt")
    spec = ckpt["model_spec"]

    metrics = {}
    test_path = args.run / "test_metrics.json"
    if test_path.is_file():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        metrics = {f"test_{k}": round(float(test[k]), 4) for k in METRIC_KEYS if k in test}
    else:
        print(f"note: {test_path} not found; exporting without test metrics")

    meta = {
        "model_version": f"{ckpt.get('method', spec['type'])}-{spec['arch']}-{args.run.name}-{dt.date.today():%Y%m%d}",
        "model_type": spec["type"],
        "architecture": spec["arch"],
        "embedding_dim": spec.get("embedding_dim"),
        "dec_alpha": spec.get("alpha"),
        "classes": ckpt["classes"],
        "image_size": ckpt["image_size"],
        "preprocess_size": ckpt["preprocess_size"],
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "threshold": ckpt["threshold"],
        "score_type": "uncalibrated",
        "metrics": metrics,
        "source_run": args.run.name,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt["state_dict"], out_dir / "model.pt")
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {meta['model_version']} to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
