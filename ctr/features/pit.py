"""Point-in-time-correct (PIT) rolling aggregates.

A naive ``groupBy(key).agg(avg(label))`` is leaky for temporal splits: for
every row it includes that row's own label and every future label of the
same key. Offline that inflates the CTR features and any model trained on
them; in production such features would not exist at scoring time.

These aggregates are computed over a trailing time window that **ends
strictly before each row's event timestamp**, so a row never observes its
own label, any row sharing its timestamp, or any future label. They are
implemented with a Spark ``Window`` ordered by the event timestamp using
``rangeBetween`` over a trailing interval (frame boundaries are relative to
the ordering value in seconds, so the ``-1`` upper bound excludes rows with
the exact same timestamp).

Keys
----
``C1``, ``C6``, ``C9`` and ``C14`` are treated as advertiser, campaign,
placement and user-segment identifiers respectively (illustrative semantics;
the mechanics only require stable high-cardinality keys). For each key we
emit:

* ``{key}_impressions``: trailing window row count,
* ``{key}_ctr``: Bayesian-smoothed CTR
  ``(clicks + alpha * prior) / (impressions + alpha)`` using the global
  **train** split CTR as the prior -- a constant prior fitted on train is
  not a leak, unlike per-row or per-split statistics.

Rows with a null key get ``impressions = 0`` and ``ctr = prior``: an
unknown entity has no reliable history and null keys must not share a
history partition with each other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from ctr.data.ingest import EVENT_TS_COLUMN
from ctr.data.synthetic import LABEL_COLUMN

logger = logging.getLogger(__name__)

PIT_KEY_COLUMNS: tuple[str, ...] = (
    "C1",  # advertiser
    "C6",  # campaign
    "C9",  # placement
    "C14",  # user segment
)

_TS_SECONDS_COLUMN = "_ts_sec"


@dataclass(frozen=True)
class PitConfig:
    """Configuration for :func:`add_pit_features`."""

    key_columns: tuple[str, ...] = PIT_KEY_COLUMNS
    window_seconds: int = 86_400
    alpha: float = 50.0


def add_pit_features(
    df: DataFrame, cfg: PitConfig, prior_ctr: float
) -> DataFrame:
    """Add ``{key}_impressions`` and ``{key}_ctr`` columns to ``df``.

    Args:
        df: raw split DataFrame with ``label`` and ``event_ts`` columns.
        cfg: keys, trailing window length and smoothing strength.
        prior_ctr: global CTR of the train split, used as the smoothing
            prior. Must be fitted on train only.

    The aggregates are computed independently within each split: a row in
    the validation split sees validation-period history only, which is the
    conservative (cold-start) reading of the production contract.
    """
    if cfg.window_seconds < 1:
        raise ValueError("window_seconds must be >= 1")
    if not 0.0 <= prior_ctr <= 1.0:
        raise ValueError(f"prior_ctr must be in [0, 1], got {prior_ctr}")

    df = df.withColumn(_TS_SECONDS_COLUMN, F.unix_seconds(F.col(EVENT_TS_COLUMN)))
    alpha = F.lit(float(cfg.alpha))
    prior = F.lit(float(prior_ctr))

    for key in cfg.key_columns:
        window = (
            Window.partitionBy(key)
            .orderBy(F.col(_TS_SECONDS_COLUMN))
            .rangeBetween(-cfg.window_seconds, -1)
        )
        clicks = F.coalesce(F.sum(F.col(LABEL_COLUMN)).over(window), F.lit(0))
        impressions = F.count(F.lit(1)).over(window)
        smoothed_ctr = (clicks + alpha * prior) / (impressions + alpha)

        df = df.withColumn(
            f"{key}_impressions",
            F.when(F.col(key).isNull(), F.lit(0)).otherwise(impressions).cast("long"),
        ).withColumn(
            f"{key}_ctr",
            F.when(F.col(key).isNull(), prior).otherwise(smoothed_ctr).cast("float"),
        )

    return df.drop(_TS_SECONDS_COLUMN)
