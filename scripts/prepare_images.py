"""Phase 3, step 1: validate and resize the task's images.

Reads the NIH metadata, keeps only images in the binary task (see config
`task`), finds each file anywhere under paths.raw_dir, checks that it decodes
and is not blank, and writes a size x size grayscale PNG to paths.processed_dir.
A manifest (processed_dir/manifest.csv) records the status of every image.
Re-running skips images that were already processed.

Usage:
    python scripts/prepare_images.py
    python scripts/prepare_images.py --limit 500      # quick trial run
    python scripts/prepare_images.py --workers 8 --overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import (  # noqa: E402
    ImageValidationError, load_grayscale, resize_square, validate_content,
)
from src.data.splits import COL_IMAGE, build_task_table  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def process_one(args: tuple[str, str | None, str, int, bool]) -> dict:
    name, src, dst, size, overwrite = args
    row = {"image": name, "status": "ok", "error": "", "orig_w": None, "orig_h": None}
    if src is None:
        row.update(status="missing", error="file not found under raw_dir")
        return row
    if not overwrite and os.path.exists(dst):
        row["status"] = "ok_cached"
        return row
    try:
        img = load_grayscale(src)
        row["orig_h"], row["orig_w"] = img.shape
        validate_content(img)
        if not cv2.imwrite(dst, resize_square(img, size)):
            raise OSError(f"failed to write {dst}")
    except (ImageValidationError, OSError, cv2.error) as exc:
        row.update(status="invalid", error=str(exc))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="process only N images")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    raw_dir, out_dir = cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"]
    size = cfg["data"]["preprocess_size"]
    meta_path = raw_dir / cfg["data"]["metadata_csv"]
    if not meta_path.is_file():
        print(f"ERROR: metadata not found at {meta_path}. Download it first (README, Phase 3).")
        return 1

    task_df = build_task_table(pd.read_csv(meta_path), cfg["task"])
    names = task_df[COL_IMAGE].tolist()[: args.limit]
    print(f"{len(task_df)} task images in metadata; processing {len(names)}")

    print(f"Indexing PNG files under {raw_dir} ...")
    index = {p.name: p for p in raw_dir.rglob("*.png")}
    print(f"  found {len(index)} PNG files")

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(n, str(index[n]) if n in index else None, str(out_dir / n), size, args.overwrite)
            for n in names]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(tqdm(pool.map(process_one, jobs, chunksize=64), total=len(jobs)))

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        # keep earlier rows for images not in this run (e.g. after a --limit trial)
        old = pd.read_csv(manifest_path)
        old = old[~old["image"].isin(manifest["image"])]
        manifest = pd.concat([old, manifest], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)

    counts = manifest["status"].value_counts()
    print("\nStatus counts:\n" + counts.to_string())
    print(f"Manifest: {manifest_path}")
    bad = manifest[manifest["status"].isin(["missing", "invalid"])]
    if len(bad):
        print(f"\n{len(bad)} images missing/invalid; they will be excluded from splits. First 5:")
        print(bad.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
