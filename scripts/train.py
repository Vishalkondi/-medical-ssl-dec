"""Train one method on one labelled fraction.

Usage (project root, virtual environment active):
    python scripts/train.py --method baseline
    python scripts/train.py --method pseudo_label --labeled-fraction 0.05
    python scripts/train.py --method ssldec --labeled-fraction 0.10 --seed 1

Outputs go to experiments/<method>_lab<XX>_seed<S>/:
    best.pt, history.csv, val_metrics.json, train.log, config.yaml
The test set is NOT touched here; use scripts/evaluate.py once training and
all tuning are finished.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.utils.device_env import configure_cuda_visibility  # noqa: E402

METHODS = ("baseline", "pseudo_label", "ssldec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--labeled-fraction", type=float, default=None,
                        help="default: data.default_labeled_fraction in the config")
    parser.add_argument("--seed", type=int, default=None, help="default: project.seed")
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config) if args.config else load_config()
    configure_cuda_visibility(cfg["training"]["device"])

    # torch is imported only after GPU visibility is settled
    from src.training.engine import setup_run
    from src.training.pseudo_label import train_pseudo_label
    from src.training.ssl_training import train_ssldec
    from src.training.supervised import train_baseline
    from src.utils.config import get_device, set_seed
    train_fn = {"baseline": train_baseline, "pseudo_label": train_pseudo_label, "ssldec": train_ssldec}[args.method]

    fraction = args.labeled_fraction or cfg["data"]["default_labeled_fraction"]
    seed = cfg["project"]["seed"] if args.seed is None else args.seed
    set_seed(seed)
    device = get_device(cfg["training"]["device"])
    run_dir = setup_run(cfg, args.method, fraction, seed, overwrite=args.overwrite)
    logging.info("method=%s labeled_fraction=%.2f seed=%d device=%s run=%s",
                 args.method, fraction, seed, device, run_dir)

    start = time.perf_counter()
    metrics = train_fn(cfg, fraction, seed, run_dir, device)
    logging.info("done in %.1f min. Validation ROC-AUC %.4f. Results in %s",
                 (time.perf_counter() - start) / 60, metrics["roc_auc"], run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
