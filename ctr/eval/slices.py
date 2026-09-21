"""Per-slice evaluation for targeting segments.

Breaks the test split down along the axes an ads team reviews before a
launch:

* user-segment buckets (deciles of the ``C14`` user-segment field by
  impression volume; bucket 0 is the busiest segment),
* device/placement buckets (deciles of the ``C9`` placement field by volume),
* point-in-time impression-count deciles of the user segment (bucket 0 is
  cold-start entities with no history),
* time-of-day buckets (six 4-hour windows).

Each slice reports volume, AUC, log loss, normalized entropy and the
calibration ratio. Slices are flagged when their calibration ratio falls
outside ``[0.9, 1.1]`` or their AUC drops more than 2 points below the
global value -- both are launch blockers in practice.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ctr.data.synthetic import CATEGORICAL_FEATURES, INTEGER_FEATURES
from ctr.eval import metrics as m
from ctr.features.pit import PIT_KEY_COLUMNS

logger = logging.getLogger(__name__)

CALIBRATION_FLAG_BOUNDS = (0.9, 1.1)
AUC_DROP_FLAG = 0.02

SLICE_COLUMNS = ["slice", "volume", "auc", "logloss", "ne", "calibration_ratio", "flags"]

_PIT_OFFSET = len(INTEGER_FEATURES)


def pit_block_index(key: str, kind: str) -> int:
    """Index of a PIT aggregate inside the numeric feature block.

    The block is ``INTEGER_FEATURES`` followed by
    ``{key}_impressions, {key}_ctr`` per PIT key in ``PIT_KEY_COLUMNS`` order.
    """
    position = PIT_KEY_COLUMNS.index(key)
    offset = 0 if kind == "impressions" else 1
    return _PIT_OFFSET + position * 2 + offset


def evaluate_slice(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Core metric bundle for one slice."""
    return {
        "volume": len(y_true),
        "auc": m.roc_auc(y_true, y_score),
        "logloss": m.log_loss(y_true, y_score),
        "ne": m.normalized_entropy(y_true, y_score),
        "calibration_ratio": m.calibration_ratio(y_true, y_score),
    }


def flag_slices(
    slices: Sequence[dict],
    global_auc: float,
    ratio_bounds: tuple = CALIBRATION_FLAG_BOUNDS,
    auc_drop: float = AUC_DROP_FLAG,
) -> list[dict]:
    """Annotate slice rows with launch-blocker flags.

    * ``calibration``: calibration ratio outside ``ratio_bounds``.
    * ``auc_below_global``: AUC more than ``auc_drop`` under the global AUC.
    """
    flagged = []
    for row in slices:
        flags = []
        ratio = row["calibration_ratio"]
        if ratio is not None and not np.isnan(ratio) and not (
            ratio_bounds[0] <= ratio <= ratio_bounds[1]
        ):
            flags.append("calibration")
        auc = row["auc"]
        if (
            auc is not None
            and not np.isnan(auc)
            and not np.isnan(global_auc)
            and auc < global_auc - auc_drop
        ):
            flags.append("auc_below_global")
        flagged.append({**row, "flags": flags})
    return flagged


def volume_deciles(keys: np.ndarray, n_buckets: int = 10) -> np.ndarray:
    """Bucket group ids into volume deciles (0 = busiest groups).

    The decile rank is derived from the group's impression volume, not its
    raw id, so buckets are comparable across runs.
    """
    keys = np.asarray(keys)
    counts = pd.Series(keys).value_counts()
    n_groups = len(counts)
    rank_of = {key: rank for rank, key in enumerate(counts.index)}
    ranks = np.array([rank_of[key] for key in keys])
    return np.minimum(ranks * n_buckets // max(n_groups, 1), n_buckets - 1)


def time_of_day_buckets(series: pd.Series) -> np.ndarray:
    """Six 4-hour window labels (00-03, ..., 20-23) from a datetime column."""
    hours = series.dt.hour.to_numpy()
    windows = hours // 4
    labels = np.array(
        [f"{window * 4:02d}-{window * 4 + 3:02d}" for window in windows]
    )
    return labels


def _bucket_rows(
    y_true: np.ndarray,
    y_score: np.ndarray,
    bucket_values: np.ndarray,
    prefix: str,
) -> list[dict]:
    rows = []
    for value in np.unique(bucket_values):
        mask = bucket_values == value
        rows.append(
            {
                "slice": f"{prefix}_{value}",
                **evaluate_slice(y_true[mask], y_score[mask]),
            }
        )
    return rows


def build_slices(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    global_auc: float,
) -> dict[str, list[dict]]:
    """All four slice tables (flagged) for one model's test predictions.

    ``frame`` must contain the ``C*_idx`` columns, the ``numeric_features``
    block and ``event_ts`` (i.e. the features parquet schema).
    """
    numeric = np.stack(frame["numeric_features"].to_numpy(), axis=0).astype(
        np.float32
    )
    cats = frame[[f"{f}_idx" for f in CATEGORICAL_FEATURES]].to_numpy(dtype=np.int64)

    segment_keys = cats[:, CATEGORICAL_FEATURES.index("C14")]
    placement_keys = cats[:, CATEGORICAL_FEATURES.index("C9")]
    pit_impressions = numeric[:, pit_block_index("C14", "impressions")]

    tables = {}
    for name, values in (
        ("user_segment_decile", volume_deciles(segment_keys)),
        ("placement_decile", volume_deciles(placement_keys)),
        (
            "pit_impression_decile",
            volume_deciles(pit_impressions.astype(np.int64)),
        ),
        ("time_of_day", time_of_day_buckets(frame["event_ts"])),
    ):
        rows = _bucket_rows(y_true, y_score, values, name)
        tables[name] = flag_slices(rows, global_auc)
    return tables
