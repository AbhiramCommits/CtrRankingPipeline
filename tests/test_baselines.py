"""Unit tests for the LR / LightGBM baselines."""

import numpy as np

from ctr.models.baselines import (
    LogisticRegressionBaseline,
    hash_features,
    lgb_predict,
    train_lightgbm,
)
from ctr.models.common import binary_auc


def test_hash_features_shape_and_determinism():
    cats = np.random.default_rng(0).integers(0, 100, (50, 6))
    first = hash_features(cats, n_buckets=1024)
    second = hash_features(cats, n_buckets=1024)
    assert first.shape == (50, 1024)
    assert (first != second).nnz == 0
    # one entry per field (no intra-row duplicates)
    assert first.nnz == 50 * 6


def test_logistic_regression_baseline_recovers_separable_signal():
    rng = np.random.default_rng(0)
    n = 400
    cats = rng.integers(0, 10, (n, 3))
    numeric = rng.normal(size=(n, 3))
    y = (numeric[:, 0] + 0.7 * numeric[:, 1] > 0).astype(np.float32)

    baseline = LogisticRegressionBaseline(n_buckets=4096)
    baseline.fit(cats, numeric, y)
    probs = baseline.predict_proba(cats, numeric)
    assert probs.shape == (n,)
    assert binary_auc(y, probs) > 0.95


def test_lightgbm_baseline_native_categorical_signal():
    rng = np.random.default_rng(0)
    n = 300
    cats = rng.integers(0, 20, (n, 2)).astype(np.int64)
    numeric = rng.normal(size=(n, 2))
    # The label depends on a categorical value directly (native handling).
    y = ((cats[:, 0] % 3 == 0) | (numeric[:, 0] > 0.5)).astype(np.float32)

    booster = train_lightgbm(
        cats,
        numeric,
        y,
        cats[:150],
        numeric[:150],
        y[:150],
        categorical_columns=["c0", "c1"],
        params={"learning_rate": 0.1, "num_leaves": 15},
        num_boost_round=50,
        early_stopping_rounds=10,
    )
    probs = lgb_predict(booster, cats, numeric, ["c0", "c1"])
    assert probs.shape == (n,)
    assert binary_auc(y, probs) > 0.8
