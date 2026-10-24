import numpy as np
import pandas as pd
import pytest

from src.data.splits import COL_IMAGE, COL_PATIENT, EXCLUDED, build_task_table, map_label, split_patients

TASK = {
    "positive_labels": ["Nodule", "Mass"],
    "negative_labels": ["No Finding"],
    "views": ["PA", "AP"],
}


@pytest.mark.parametrize("labels,expected", [
    ("No Finding", 0),
    ("Nodule", 1),
    ("Mass|Effusion", 1),
    ("Effusion|Nodule", 1),
    ("Effusion", EXCLUDED),
    ("Atelectasis|Infiltration", EXCLUDED),
])
def test_map_label(labels, expected):
    assert map_label(labels, TASK["positive_labels"], TASK["negative_labels"]) == expected


def synthetic_meta(n_patients=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        for k in range(rng.integers(1, 6)):
            labels = rng.choice(["No Finding", "Nodule", "Mass|Effusion", "Effusion"],
                                p=[0.6, 0.1, 0.05, 0.25])
            rows.append({COL_IMAGE: f"{pid:08d}_{k:03d}.png", "Finding Labels": labels,
                         COL_PATIENT: pid, "View Position": rng.choice(["PA", "AP", "LL"], p=[.6, .35, .05])})
    return pd.DataFrame(rows)


def make_splits(seed=42):
    task_df = build_task_table(synthetic_meta(), TASK)
    test_pids = set(range(0, 400, 5))  # every 5th patient is "official test"
    test_images = set(task_df.loc[task_df[COL_PATIENT].isin(test_pids), COL_IMAGE])
    return task_df, split_patients(task_df, test_images, 0.1, [0.05, 0.25], seed)


def test_task_table_excludes_other_findings_and_views():
    df = build_task_table(synthetic_meta(), TASK)
    assert set(df["label"]) <= {0, 1}
    assert set(df["View Position"]) <= {"PA", "AP"}
    assert not df["Finding Labels"].eq("Effusion").any()


def test_no_patient_overlap_between_splits():
    _, s = make_splits()
    groups = ["test", "val", "labeled_05", "unlabeled_05"]
    pids = {g: set(s[g][COL_PATIENT]) for g in groups}
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not pids[a] & pids[b], f"patient leakage between {a} and {b}"


def test_splits_cover_every_task_image_exactly_once():
    task_df, s = make_splits()
    combined = pd.concat([s["test"], s["val"], s["labeled_05"], s["unlabeled_05"]])
    assert len(combined) == len(task_df)
    assert combined[COL_IMAGE].is_unique


def test_labeled_subsets_are_nested_and_deterministic():
    _, s1 = make_splits(seed=42)
    _, s2 = make_splits(seed=42)
    small, large = set(s1["labeled_05"][COL_PATIENT]), set(s1["labeled_25"][COL_PATIENT])
    assert small < large
    assert s1["labeled_05"].equals(s2["labeled_05"])


def test_leakage_in_official_lists_is_rejected():
    task_df = build_task_table(synthetic_meta(), TASK)
    first_patient = task_df[COL_PATIENT].iloc[0]
    imgs = task_df.loc[task_df[COL_PATIENT] == first_patient, COL_IMAGE]
    if len(imgs) < 2:
        pytest.skip("need a patient with 2+ images")
    with pytest.raises(ValueError, match="leakage"):
        split_patients(task_df, {imgs.iloc[0]}, 0.1, [0.05], 42)
