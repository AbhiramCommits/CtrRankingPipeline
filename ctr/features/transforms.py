"""Train-fit, leakage-free feature transforms for the CTR pipeline.

All fitted quantities (numeric clipping/medians, categorical vocabularies)
are computed on the **train split only** and persisted under
``artifacts/``, so validation/test rows are transformed with train-fitted
values and never learn from their own distribution. Fitting on
validation/test data would leak their statistics into the features and
inflate offline metrics.

Numeric features
----------------
For each integer column the transform is ``sign(x) * log1p(|x|)`` -- the
classic skew-reducing ``log1p`` transform, sign-preserving so the
occasionally negative integer features stay meaningful. Non-null values are
clipped at the train-split 99th percentile (computed on the transformed
scale), and nulls are imputed with the train-split median. All 13 columns
are assembled into a single ``array<float>`` column (``numeric_features``),
the float32 matrix consumed by downstream models.

Categorical features
--------------------
A vocabulary is built per column from the train split. Values with
frequency >= ``min_count`` are kept and assigned indices 1..V in order of
decreasing frequency (ties broken by value, so fits are deterministic);
everything else -- rare values, values unseen at fit time, and nulls -- maps
to the explicit OOV index 0. Vocabularies are persisted as Parquet under
``artifacts/vocab/{column}.parquet``; index columns are stored as int64.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ctr.data.ingest import DS_COLUMN, EVENT_TS_COLUMN, SPLIT_COLUMN
from ctr.data.synthetic import CATEGORICAL_FEATURES, INTEGER_FEATURES, LABEL_COLUMN

logger = logging.getLogger(__name__)

OOV_INDEX = 0
NUMERIC_BLOCK_COLUMN = "numeric_features"
VOCAB_SUBDIR = "vocab"
STATS_FILENAME = "feature_stats.json"

_PERCENTILE_ACCURACY = 10_000


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for fitting and applying feature transforms."""

    numeric_columns: tuple[str, ...] = tuple(INTEGER_FEATURES)
    categorical_columns: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)
    min_count: int = 10
    clip_quantile: float = 0.99
    artifacts_dir: str = "artifacts"


def _transformed(col_name: str) -> F.Column:
    """sign(x) * log1p(|x|): skew-reducing, sign-preserving transform."""
    x = F.col(col_name)
    return F.sign(x) * F.log1p(F.abs(x))


def _fit_numeric_stats(train_df: DataFrame, cfg: FeatureConfig) -> dict[str, dict[str, float]]:
    """Compute per-column median and clip percentile on the train split.

    Both statistics are computed on the transformed scale (see
    :func:`_transformed`), ignoring nulls, so they can be applied directly
    to the transformed values at inference time.
    """
    stats: dict[str, dict[str, float]] = {}
    for col in cfg.numeric_columns:
        row = train_df.agg(
            F.percentile_approx(
                _transformed(col), 0.5, _PERCENTILE_ACCURACY
            ).alias("median"),
            F.percentile_approx(
                _transformed(col), cfg.clip_quantile, _PERCENTILE_ACCURACY
            ).alias("p99"),
        ).first()
        stats[col] = {
            "median": float(row["median"] or 0.0),
            "p99": float(row["p99"] or 0.0),
        }
    return stats


def _vocab_path(artifacts_dir: str, col: str) -> str:
    return os.path.join(artifacts_dir, VOCAB_SUBDIR, col)


def _fit_vocabularies(train_df: DataFrame, cfg: FeatureConfig) -> dict[str, int]:
    """Build and persist a per-column vocabulary from the train split.

    Keeps values with frequency >= ``min_count``, indexed 1..V by descending
    frequency (ties broken by value for determinism). OOV index 0 is
    reserved for everything else. Returns the vocabulary size per column.
    """
    sizes: dict[str, int] = {}
    for col in cfg.categorical_columns:
        vocab = (
            train_df.filter(F.col(col).isNotNull())
            .groupBy(col)
            .count()
            .filter(F.col("count") >= cfg.min_count)
            .withColumn(
                "index",
                F.row_number().over(
                    Window.orderBy(F.desc("count"), F.col(col))
                ),
            )
            .select(
                F.col(col).alias("value"),
                F.col("index").cast("long"),
                F.col("count").cast("long"),
            )
        )
        vocab.write.mode("overwrite").parquet(_vocab_path(cfg.artifacts_dir, col))
        sizes[col] = vocab.count()
        logger.info(
            "vocab %s: %d entries (min_count=%d)",
            col,
            sizes[col],
            cfg.min_count,
        )
    return sizes


def save_feature_stats(summary: dict[str, Any], artifacts_dir: str) -> str:
    """Persist the fitted numeric statistics and vocabulary sizes as JSON."""
    path = os.path.join(artifacts_dir, STATS_FILENAME)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return path


def load_feature_stats(artifacts_dir: str) -> dict[str, Any]:
    """Load the persisted fitted statistics (numeric stats + vocab sizes)."""
    with open(os.path.join(artifacts_dir, STATS_FILENAME), "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_vocab(spark: SparkSession, artifacts_dir: str, col: str) -> DataFrame:
    """Load a persisted vocabulary (columns: value, index, count)."""
    return spark.read.parquet(_vocab_path(artifacts_dir, col))


def fit_transforms(train_df: DataFrame, cfg: FeatureConfig) -> dict[str, Any]:
    """Fit numeric stats and vocabularies on the train split and persist them.

    This must be called with the train split only; validation/test must be
    transformed via :func:`build_feature_frame` using these persisted
    artifacts, never re-fit.
    """
    summary = {
        "min_count": cfg.min_count,
        "clip_quantile": cfg.clip_quantile,
        "numeric": _fit_numeric_stats(train_df, cfg),
        "vocab_sizes": _fit_vocabularies(train_df, cfg),
    }
    path = save_feature_stats(summary, cfg.artifacts_dir)
    logger.info("Fitted transforms saved to %s", path)
    return summary


def _transform_numeric(
    df: DataFrame, cfg: FeatureConfig, numeric_stats: dict[str, dict[str, float]]
) -> DataFrame:
    """Apply log1p -> clip at train p99 -> impute nulls with train median."""
    for col in cfg.numeric_columns:
        stats = numeric_stats[col]
        value = F.when(
            F.col(col).isNull(), F.lit(stats["median"])
        ).otherwise(F.least(_transformed(col), F.lit(stats["p99"])))
        df = df.withColumn(f"{col}_f", value.cast("float"))
    return df


def _transform_categorical(
    df: DataFrame, cfg: FeatureConfig, vocabs: dict[str, DataFrame]
) -> DataFrame:
    """Map categorical values to vocabulary indices (int64), OOV -> 0."""
    for col in cfg.categorical_columns:
        vocab = vocabs[col].select(F.col("value").alias(col), "index")
        df = df.join(F.broadcast(vocab), on=col, how="left").withColumn(
            f"{col}_idx", F.coalesce(F.col("index"), F.lit(OOV_INDEX)).cast("long")
        )
        df = df.drop("index")
    return df


def _extra_numeric_expr(name: str) -> F.Column:
    """Numeric expression for an extra column folded into the feature block.

    Impression counts are heavy-tailed, so they enter the block log1p-scaled;
    CTR columns are already probabilities in [0, 1] and pass through.
    """
    if name.endswith("_impressions"):
        return F.log1p(F.col(name).cast("float"))
    return F.col(name).cast("float")


def build_feature_frame(
    df: DataFrame,
    cfg: FeatureConfig,
    numeric_stats: dict[str, dict[str, float]],
    vocabs: dict[str, DataFrame],
    extra_numeric: Sequence[str] = (),
) -> DataFrame:
    """Apply transforms and assemble the final feature schema.

    Output columns: ``label``, ``event_ts``, ``ds`` (plus ``split`` when
    present), a float32 ``numeric_features`` array containing the transformed
    integer block followed by the ``extra_numeric`` columns (e.g.
    point-in-time aggregates), and the int64 ``{col}_idx`` columns.
    """
    df = _transform_numeric(df, cfg, numeric_stats)
    df = _transform_categorical(df, cfg, vocabs)

    numeric_exprs = [F.col(f"{col}_f") for col in cfg.numeric_columns] + [
        _extra_numeric_expr(name) for name in extra_numeric
    ]
    meta = [LABEL_COLUMN, EVENT_TS_COLUMN, DS_COLUMN]
    if SPLIT_COLUMN in df.columns:
        meta.append(SPLIT_COLUMN)
    keep: list[str] = meta + [
        F.array(*numeric_exprs).alias(NUMERIC_BLOCK_COLUMN)
    ] + [f"{col}_idx" for col in cfg.categorical_columns]
    return df.select(*keep)
