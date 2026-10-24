import numpy as np
import pytest

from src.evaluation.metrics import binary_metrics, bootstrap_ci, expected_calibration_error, youden_threshold


def test_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["roc_auc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 2, 0)


def test_accuracy_misleading_under_imbalance():
    y = np.array([0] * 90 + [1] * 10)
    m = binary_metrics(y, np.zeros(100) + 0.1, 0.5)       # always "no finding"
    assert m["accuracy"] == 0.9 and m["sensitivity"] == 0.0


def test_single_class_rejected():
    with pytest.raises(ValueError):
        binary_metrics(np.zeros(5), np.zeros(5), 0.5)


def test_youden_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(y, s)
    assert 0.3 < t <= 0.7


def test_ece_zero_for_calibrated_scores():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    groups = np.repeat(np.arange(200), 2)
    m = binary_metrics(y, s, 0.5)
    ci = bootstrap_ci(y, s, groups, 0.5, n_samples=200)
    assert ci["roc_auc"][0] <= m["roc_auc"] <= ci["roc_auc"][1]
