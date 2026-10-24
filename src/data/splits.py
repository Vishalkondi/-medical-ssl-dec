"""Label mapping and patient-level splitting for NIH ChestX-ray14.

Every split is made by Patient ID: a patient's images never appear in more
than one of test / val / labelled / unlabelled.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.utils.config import split_name

EXCLUDED = -1

# Column names in Data_Entry_2017.csv that we rely on.
COL_IMAGE = "Image Index"
COL_LABELS = "Finding Labels"
COL_PATIENT = "Patient ID"
COL_VIEW = "View Position"


def map_label(finding_labels: str, positive: Sequence[str], negative: Sequence[str]) -> int:
    """Map an NIH 'Finding Labels' string (e.g. 'Mass|Effusion') to 1, 0 or EXCLUDED.

    1        if any positive label is present (other findings may co-occur)
    0        if the label set is exactly the negative set (e.g. {'No Finding'})
    EXCLUDED otherwise (other findings only; not part of this binary task)
    """
    labels = {s.strip() for s in str(finding_labels).split("|") if s.strip()}
    if labels & set(positive):
        return 1
    if labels == set(negative):
        return 0
    return EXCLUDED


def build_task_table(meta: pd.DataFrame, task_cfg: dict) -> pd.DataFrame:
    """Return metadata rows that belong to the binary task, with a `label` column."""
    missing = {COL_IMAGE, COL_LABELS, COL_PATIENT, COL_VIEW} - set(meta.columns)
    if missing:
        raise KeyError(f"metadata CSV is missing columns: {sorted(missing)}")
    df = meta[meta[COL_VIEW].isin(task_cfg["views"])].copy()
    df["label"] = [map_label(s, task_cfg["positive_labels"], task_cfg["negative_labels"])
                   for s in df[COL_LABELS]]
    df = df[df["label"] != EXCLUDED]
    return df[[COL_IMAGE, COL_PATIENT, COL_VIEW, COL_LABELS, "label"]].reset_index(drop=True)


def _patient_strata(df: pd.DataFrame) -> pd.Series:
    """1 if the patient has at least one positive image, else 0 (index: patient id)."""
    return df.groupby(COL_PATIENT)["label"].max()


def stratified_patient_order(df: pd.DataFrame, seed: int) -> list:
    """Seeded random order of patients that interleaves positive/negative patients
    proportionally, so any prefix of the list has roughly the same patient-level
    positive rate. Taking prefixes gives nested, stratified subsets."""
    rng = np.random.default_rng(seed)
    strata = _patient_strata(df)
    groups = [rng.permutation(strata.index[strata == s].to_numpy()) for s in (0, 1)]
    # Assign each patient a fractional rank within its stratum, then merge by rank.
    ranked = []
    for g in groups:
        n = len(g)
        ranked += [((i + 0.5) / n, pid) for i, pid in enumerate(g)]
    ranked.sort(key=lambda t: t[0])
    return [pid for _, pid in ranked]


def split_patients(
    task_df: pd.DataFrame,
    test_images: set[str],
    val_fraction: float,
    labeled_fractions: Sequence[float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split task images into test / val / train, then nested labelled subsets of train.

    test_images: image names from the official NIH test_list.txt.
    Returns a dict with keys 'test', 'val', 'train_pool' and
    'labeled_<pct>' / 'unlabeled_<pct>' for each labelled fraction.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    if any(not 0 < f < 1 for f in labeled_fractions):
        raise ValueError("labeled fractions must be in (0, 1)")

    is_test = task_df[COL_IMAGE].isin(test_images)
    test_df, trval_df = task_df[is_test], task_df[~is_test]

    leaked = set(test_df[COL_PATIENT]) & set(trval_df[COL_PATIENT])
    if leaked:
        raise ValueError(f"{len(leaked)} patients appear in both the official test list "
                         "and train_val list; refusing to continue (data leakage)")

    order = stratified_patient_order(trval_df, seed)
    n_val = max(1, round(val_fraction * len(order)))
    val_pids, train_pids = set(order[:n_val]), order[n_val:]

    out = {
        "test": test_df,
        "val": trval_df[trval_df[COL_PATIENT].isin(val_pids)],
        "train_pool": trval_df[trval_df[COL_PATIENT].isin(set(train_pids))],
    }
    # train_pids keeps the stratified order, so prefixes are nested and stratified.
    for frac in sorted(labeled_fractions):
        k = max(1, round(frac * len(train_pids)))
        lab = set(train_pids[:k])
        name = split_name(frac)
        pool = out["train_pool"]
        out[f"labeled_{name}"] = pool[pool[COL_PATIENT].isin(lab)]
        out[f"unlabeled_{name}"] = pool[~pool[COL_PATIENT].isin(lab)]
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def summarize(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        pos = int((df["label"] == 1).sum())
        rows.append({
            "split": name,
            "patients": df[COL_PATIENT].nunique(),
            "images": len(df),
            "positive_images": pos,
            "positive_rate": round(pos / len(df), 4) if len(df) else 0.0,
        })
    return pd.DataFrame(rows)
