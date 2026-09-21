"""Scikit-learn / LightGBM baselines on the same feature inputs as DLRM.

Both baselines consume exactly the same splits and the same feature columns
(categorical index columns + the numeric block) so their metrics are
directly comparable with the deep ranker:

* ``LogisticRegressionBaseline``: hashed one-hot of ``(field, value)``
  pairs plus MaxAbs-scaled numeric features, fit with scikit-learn's
  LogisticRegression.
* ``train_lightgbm``: GBDT with LightGBM's *native* categorical handling
  (integer codes passed via ``categorical_feature``), early stopping on the
  validation split.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

# lightgbm bundles its own libomp; when torch (or faiss) has already loaded a
# different OpenMP runtime, allow duplicates explicitly (see also
# ctr/retrieval/index.py for the faiss variant of this clash).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm as lgb

logger = logging.getLogger(__name__)

_HASH_MIX1 = np.uint64(0x9E3779B1)
_HASH_MIX2 = np.uint64(0x85EBCA6B)


def hash_features(
    categorical_indices: np.ndarray, n_buckets: int = 1 << 18
) -> sparse.csr_matrix:
    """Hashed one-hot of ``(field, value)`` pairs into ``n_buckets`` buckets.

    The hashing trick trades exact one-hot identity for a fixed-size sparse
    representation; collisions are tolerated. The mixing is a deterministic
    pure-integer function of ``(field_index, value)`` so train/val/test
    hashing always agrees.
    """
    n_rows, n_fields = categorical_indices.shape
    field_ids = np.arange(n_fields, dtype=np.uint64)[None, :]
    values = categorical_indices.astype(np.uint64)
    hashed = (field_ids * _HASH_MIX1) + values
    hashed = (hashed ^ (hashed >> 16)) * _HASH_MIX2
    hashed = hashed ^ (hashed >> 13)
    cols = (hashed % np.uint64(n_buckets)).astype(np.int64).reshape(-1)
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), n_fields)
    data = np.ones(rows.shape[0], dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, n_buckets))


class LogisticRegressionBaseline:
    """Logistic regression on hashed one-hot + scaled numeric features.

    The numeric scaler is fitted on the training split only, mirroring the
    no-leakage contract of the feature pipeline.
    """

    def __init__(self, n_buckets: int = 1 << 18, C: float = 1.0, max_iter: int = 200):
        self.n_buckets = n_buckets
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self.scaler = MaxAbsScaler()

    def _matrix(self, cat_indices: np.ndarray, numeric: np.ndarray) -> sparse.csr_matrix:
        return sparse.hstack(
            [hash_features(cat_indices, self.n_buckets), self.scaler.transform(numeric)]
        ).tocsr()

    def fit(self, cat_indices: np.ndarray, numeric: np.ndarray, y: np.ndarray) -> None:
        self.scaler.fit(numeric)
        self.model.fit(self._matrix(cat_indices, numeric), y)

    def predict_proba(self, cat_indices: np.ndarray, numeric: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._matrix(cat_indices, numeric))[:, 1]


def _to_frame(
    cat_indices: np.ndarray, numeric: np.ndarray, categorical_columns: Sequence[str]
):
    import pandas as pd

    frame = pd.DataFrame(
        numeric, columns=[f"num_{i}" for i in range(numeric.shape[1])]
    )
    for i, col in enumerate(categorical_columns):
        frame[col] = cat_indices[:, i]
    return frame


def train_lightgbm(
    train_cats: np.ndarray,
    train_numeric: np.ndarray,
    y_train: np.ndarray,
    val_cats: np.ndarray,
    val_numeric: np.ndarray,
    y_val: np.ndarray,
    categorical_columns: Sequence[str],
    params: dict | None = None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
):
    """Train a LightGBM binary classifier with native categorical handling."""
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
    }
    if params:
        lgb_params.update(params)
    train = lgb.Dataset(
        _to_frame(train_cats, train_numeric, categorical_columns),
        label=y_train,
        categorical_feature=list(categorical_columns),
    )
    valid = lgb.Dataset(
        _to_frame(val_cats, val_numeric, categorical_columns),
        label=y_val,
        categorical_feature=list(categorical_columns),
    )
    return lgb.train(
        lgb_params,
        train,
        num_boost_round=num_boost_round,
        valid_sets=[valid],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )


def lgb_predict(
    booster, cat_indices: np.ndarray, numeric: np.ndarray, categorical_columns: Sequence[str]
) -> np.ndarray:
    """Predicted positive probabilities from a trained LightGBM booster."""
    return booster.predict(
        _to_frame(cat_indices, numeric, categorical_columns),
        num_iteration=booster.best_iteration,
    )
