"""Offline metrics for CTR prediction, measured the way ads systems are.

* ROC AUC and log loss: global discrimination and probability quality.
* Normalized entropy (NE): log loss divided by the entropy of the background
  CTR. From the Facebook ads paper (He et al., 2014, "Practical Lessons from
  Predicting Clicks on Ads at Facebook"), this is the standard CTR-model
  metric there: it normalizes log loss by the intrinsic entropy of the
  label distribution so that 1.0 means "as good as always predicting the
  background CTR" and lower is strictly better. Because NE is
  scale-invariant to the background rate, it is comparable across datasets
  with different click rates -- something raw log loss is not.
* Calibration: overall predicted/actual CTR ratio, a vigintile (20-bucket)
  reliability curve and the Expected Calibration Error (ECE, Naeini et al.
  2015): the impression-weighted mean absolute difference between predicted
  and actual CTR per bucket. Ads systems bid with calibrated probabilities,
  so miscalibration costs money directly.
* GAUC: AUC computed per grouping key (here the user-segment field) and
  averaged weighted by impressions. Ranking-quality metrics matter per
  request: ad scores are only ever compared *within* a context, never
  globally, so a per-group AUC is the number that tracks ranking quality
  in serving.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def log_loss(y_true: np.ndarray, y_score: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy (natural log) between labels and probabilities."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_score, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC; NaN when a label class is absent (ranking is undefined)."""
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, y_score))


def background_ctr(y_true: np.ndarray) -> float:
    """Mean label (the click rate a constant model would predict)."""
    return float(np.mean(np.asarray(y_true, dtype=np.float64)))


def background_entropy(y_true: np.ndarray) -> float:
    """Entropy (nats) of the background CTR, the NE denominator."""
    p = background_ctr(y_true)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def normalized_entropy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """log loss / entropy of the background CTR; lower is better (see module docstring)."""
    entropy = background_entropy(y_true)
    if entropy == 0.0:
        return float("nan")
    return log_loss(y_true, y_score) / entropy


def calibration_ratio(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """mean predicted CTR / mean actual CTR; 1.0 is perfectly calibrated.

    A ratio above 1 means systematic over-prediction (overbidding),
    below 1 under-prediction.
    """
    actual = background_ctr(y_true)
    if actual == 0.0:
        return float("nan")
    return float(np.mean(np.asarray(y_score, dtype=np.float64)) / actual)


def reliability_curve(
    y_true: np.ndarray, y_score: np.ndarray, n_buckets: int = 20
) -> dict:
    """Equal-frequency (by predicted score) calibration buckets.

    Returns bucket ids plus ``mean_predicted``, ``mean_actual`` and
    ``count`` per bucket -- the standard vigintile reliability curve.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_score, dtype=np.float64)
    n = len(y)
    if n == 0:
        raise ValueError("cannot bucket empty predictions")
    ranks = np.argsort(np.argsort(p, kind="mergesort"))
    buckets = np.minimum(ranks * n_buckets // n, n_buckets - 1)
    counts = np.bincount(buckets, minlength=n_buckets)
    pred_sum = np.bincount(buckets, weights=p, minlength=n_buckets)
    actual_sum = np.bincount(buckets, weights=y, minlength=n_buckets)
    return {
        "bucket": np.arange(n_buckets),
        "mean_predicted": pred_sum / np.maximum(counts, 1),
        "mean_actual": actual_sum / np.maximum(counts, 1),
        "count": counts,
    }


def ece(y_true: np.ndarray, y_score: np.ndarray, n_buckets: int = 20) -> float:
    """Expected Calibration Error: volume-weighted |predicted - actual|."""
    curve = reliability_curve(y_true, y_score, n_buckets)
    total = curve["count"].sum()
    if total == 0:
        return float("nan")
    return float(
        np.sum(
            curve["count"]
            / total
            * np.abs(curve["mean_predicted"] - curve["mean_actual"])
        )
    )


def gauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Impression-weighted mean of per-group AUC (see module docstring).

    Groups with a single label class have no defined AUC and are skipped.
    ``weights`` optionally gives per-row impression weights (defaults to 1).
    """
    y = np.asarray(y_true)
    p = np.asarray(y_score)
    g = np.asarray(groups)
    w = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.ones(len(y), dtype=np.float64)
    )
    total_auc = 0.0
    total_weight = 0.0
    for group in np.unique(g):
        mask = g == group
        if len(np.unique(y[mask])) < 2:
            continue
        group_weight = float(w[mask].sum())
        total_auc += roc_auc(y[mask], p[mask]) * group_weight
        total_weight += group_weight
    return float(total_auc / total_weight) if total_weight > 0 else float("nan")
