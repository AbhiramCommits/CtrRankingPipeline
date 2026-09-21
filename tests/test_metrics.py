"""Hand-computed-value tests for the offline CTR metrics."""

import math

import numpy as np

from ctr.eval.metrics import (
    background_ctr,
    background_entropy,
    calibration_ratio,
    ece,
    gauc,
    log_loss,
    normalized_entropy,
    reliability_curve,
    roc_auc,
)


def test_log_loss_hand_computed():
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    expected = -(math.log(0.8) + math.log(0.8)) / 2
    assert log_loss(y, p) == expected


def test_normalized_entropy_matches_background_entropy_definition():
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    assert background_ctr(y) == 0.5
    assert background_entropy(y) == math.log(2)
    expected = log_loss(y, p) / math.log(2)
    assert normalized_entropy(y, p) == expected
    # NE == 1 exactly when predicting the background CTR.
    assert normalized_entropy(y, np.array([0.5, 0.5])) == 1.0


def test_perfect_ranker_has_auc_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(y, p) == 1.0
    # inverted ranker
    assert roc_auc(y, 1.0 - p) == 0.0


def test_perfectly_calibrated_ratio_one_and_ece_zero():
    y = np.array([0.0, 0.5, 1.0])
    p = np.array([0.0, 0.5, 1.0])
    assert calibration_ratio(y, p) == 1.0
    assert ece(y, p, n_buckets=20) == 0.0


def test_sampled_calibrated_predictions_are_nearly_calibrated():
    rng = np.random.default_rng(0)
    n = 1000
    p = np.repeat(np.linspace(0.05, 0.95, 20), 50)
    y = (rng.random(n) < p).astype(np.float64)
    assert abs(calibration_ratio(y, p) - 1.0) < 0.05
    assert ece(y, p, n_buckets=20) < 0.06


def test_reliability_curve_is_equal_frequency():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000).astype(np.float64)
    p = rng.random(1000)
    curve = reliability_curve(y, p, n_buckets=20)
    assert len(curve["bucket"]) == 20
    assert curve["count"].sum() == 1000
    assert np.all(curve["count"] == 50)
    assert np.all(curve["mean_predicted"] >= 0)
    assert np.all(curve["mean_actual"] <= 1)


def test_gauc_weighted_by_group_impressions():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.8, 0.2, 0.3, 0.7])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    # group a: AUC 1.0, group b: AUC 0.0, group c: AUC 1.0
    assert gauc(y, p, groups) == 2.0 / 3.0
    # weights: group a carries all the mass
    weights = np.array([5, 5, 1, 1, 1, 1])
    assert gauc(y, p, groups, weights=weights) == (1.0 * 10 + 0.0 * 2 + 1.0 * 2) / 14


def test_single_class_inputs_are_nan():
    y = np.zeros(10)
    p = np.linspace(0.1, 0.9, 10)
    assert np.isnan(roc_auc(y, p))
    assert np.isnan(normalized_entropy(y, p))
    assert np.isnan(calibration_ratio(y, p))
    assert np.isnan(gauc(y, p, np.zeros(10)))
