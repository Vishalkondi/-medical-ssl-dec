"""Phase 3, step 2: create patient-level splits and nested labelled subsets.

Inputs : Data_Entry_2017.csv, test_list.txt (paths.raw_dir)
         processed_dir/manifest.csv from prepare_images.py (only valid images kept)
Outputs: data/splits/{test,val,train_pool,labeled_XX,unlabeled_XX}.csv
         data/splits/summary.csv

The unlabelled CSVs still contain the true `label` column so that Phase 5-6
can MEASURE pseudo-label accuracy. Training code must never read it.

Usage:
    python scripts/make_splits.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import COL_IMAGE, build_task_table, split_patients, summarize  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def read_list(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Download it from the NIH folder.")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    raw, splits_dir = cfg["paths"]["raw_dir"], cfg["paths"]["splits_dir"]
    dcfg = cfg["data"]

    task_df = build_task_table(pd.read_csv(raw / dcfg["metadata_csv"]), cfg["task"])
    test_images = read_list(raw / dcfg["test_list"])
    trainval_images = read_list(raw / dcfg["train_val_list"])
    unlisted = set(task_df[COL_IMAGE]) - test_images - trainval_images
    if unlisted:
        print(f"WARNING: {len(unlisted)} task images are in neither official list; dropping them")
        task_df = task_df[~task_df[COL_IMAGE].isin(unlisted)]

    manifest_path = cfg["paths"]["processed_dir"] / "manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        valid = set(manifest.loc[manifest["status"].str.startswith("ok"), "image"])
        before = len(task_df)
        task_df = task_df[task_df[COL_IMAGE].isin(valid)]
        print(f"Kept {len(task_df)}/{before} images that passed validation")
    else:
        print(f"WARNING: {manifest_path} not found; run prepare_images.py first. "
              "Splits will include unvalidated images.")

    splits = split_patients(task_df, test_images, dcfg["val_fraction"],
                            dcfg["labeled_fractions"], cfg["project"]["seed"])

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(splits_dir / f"{name}.csv", index=False)
    summary = summarize(splits)
    summary.to_csv(splits_dir / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    for row in summary.itertuples():
        if row.split.startswith("labeled_") and row.positive_images < 10:
            print(f"WARNING: {row.split} has only {row.positive_images} positive images; "
                  "results for this fraction will be unreliable")
    print(f"\nWrote {len(splits)} split files to {splits_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
