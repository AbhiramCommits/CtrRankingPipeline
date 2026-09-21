"""Spark ingestion job for the raw Criteo-style TSV dump.

Layout produced::

    data/parquet/
    ├── train/        # days 0..16 of the event window
    ├── validation/   # days 17..18
    └── test/         # days 19..20

Each split is partitioned by ``ds`` (the calendar day of ``event_ts``),
mirroring how production ad-serving logs are laid out as daily partitions.

Why time-based splits matter for CTR models
-------------------------------------------
* **Label leakage.** Ad logs contain the same users, publishers and campaigns
  many times. A random split lets one row about user X land in train while
  another row about user X (or the same campaign) lands in test, so a model
  that memorizes per-entity CTRs scores unrealistically well offline and
  collapses in production. Time-ordered splits drastically cut that overlap
  for the entity-heavy features that dominate CTR models.
* **Temporal drift.** User behavior, creatives and auction dynamics shift
  over time. Random splits leak the drift into the training set and overstate
  offline metrics; production ranking always predicts the *future*, so the
  offline analogue must be a contiguous time cut: train on the past,
  validate and test on strictly later windows.
* **Retraining realism.** Online systems retrain on trailing logs, so a
  temporal split lets us measure how models degrade as data ages and when
  retraining is actually needed.

The event timestamp is assigned deterministically: the CRC-32 of the
tab-joined feature string decides the day and the intra-day second, so a
given raw file always produces the same splits regardless of cluster
topology or task partitioning. :mod:`ctr.data.synthetic` uses the identical
hash, keeping the drift it injects into the labels aligned with these splits.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

from ctr.data.synthetic import (
    ALL_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FIELD_SEPARATOR,
    INTEGER_FEATURES,
    LABEL_COLUMN,
)

logger = logging.getLogger(__name__)

DAY_COLUMN = "event_day"
EVENT_TS_COLUMN = "event_ts"
DS_COLUMN = "ds"
SPLIT_COLUMN = "split"
SECONDS_PER_DAY = 86_400

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_NAMES = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

# Explicit schema for the header-less TSV. Never use inferSchema here: on a
# 40-column dump with empty fields, inference is slow, fragile, and breaks
# the contract between the generator and every downstream consumer.
RAW_SCHEMA = StructType(
    [StructField(LABEL_COLUMN, IntegerType(), True)]
    + [StructField(col, LongType(), True) for col in INTEGER_FEATURES]
    + [StructField(col, StringType(), True) for col in CATEGORICAL_FEATURES]
)


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for :func:`run` (see ``configs/spark.yaml``)."""

    raw_path: str = "data/raw/train.txt"
    parquet_root: str = "data/parquet"
    base_ts: str = "2026-09-01 00:00:00"
    days: int = 21
    train_days: int = 17
    validation_days: int = 2
    test_days: int = 2


def split_name_for_day(
    day: int,
    train_days: int = 17,
    validation_days: int = 2,
    test_days: int = 2,
) -> str:
    """Map a 0-based day index within the event window to a split name.

    The window is cut contiguously: ``[0, train_days)`` -> train,
    ``[train_days, train_days + validation_days)`` -> validation,
    the remainder -> test. Splitting by time rather than at random is
    essential for CTR models (see the module docstring).
    """
    if day < 0:
        raise ValueError(f"day must be >= 0, got {day}")
    if day < train_days:
        return SPLIT_TRAIN
    if day < train_days + validation_days:
        return SPLIT_VALIDATION
    if day < train_days + validation_days + test_days:
        return SPLIT_TEST
    raise ValueError(
        f"day {day} is outside the {train_days + validation_days + test_days}-day window"
    )


def read_raw(spark: SparkSession, path: str, schema: StructType = RAW_SCHEMA) -> DataFrame:
    """Read the header-less tab-separated dump with an explicit schema."""
    return (
        spark.read.option("sep", FIELD_SEPARATOR)
        .option("header", "false")
        .schema(schema)
        .csv(path)
    )


def normalize_missing(df: DataFrame) -> DataFrame:
    """Convert empty-string categorical fields to SQL NULL.

    Missing values are written as empty fields in the raw dump (matching
    Criteo); empty strings in ``C`` columns become NULL here so downstream
    null-rate stats and feature engineering treat them consistently.
    """
    for col in CATEGORICAL_FEATURES:
        df = df.withColumn(col, F.nullif(F.col(col), F.lit("")))
    return df


def add_event_time(df: DataFrame, base_ts: str, days: int = 21) -> DataFrame:
    """Assign a deterministic event timestamp in ``[base_ts, base_ts + days)``.

    The timestamp is derived from the CRC-32 of the tab-joined feature
    string, so it is stable across runs and independent of how the file is
    partitioned across executors::

        crc    = crc32(concat_ws('\\t', coalesce(I1, ''), ..., coalesce(C26, '')))
        day    = crc % days
        second = (crc // days) % 86400

    ``ctr.data.synthetic`` computes the same hash in Python, which keeps the
    label drift it injects aligned with these timestamps.
    """
    feature_string = F.concat_ws(
        FIELD_SEPARATOR,
        *[F.coalesce(F.col(col).cast("string"), F.lit("")) for col in FEATURE_COLUMNS],
    )
    crc = F.crc32(feature_string)
    day_col = (crc % F.lit(days)).cast("int")
    second_col = ((crc / F.lit(days)).cast("long") % F.lit(SECONDS_PER_DAY)).cast("int")
    # NOTE: from_unixtime/unix_timestamp resolve the JVM default timezone
    # rather than spark.sql.session.timeZone, which silently shifts wall-clock
    # times when the two differ. make_interval keeps the math tz-consistent.
    event_ts = F.to_timestamp(F.lit(base_ts)) + F.make_interval(
        F.lit(0), F.lit(0), F.lit(0), day_col, F.lit(0), F.lit(0), second_col
    )

    return (
        df.withColumn(DAY_COLUMN, day_col)
        .withColumn(EVENT_TS_COLUMN, event_ts)
        .withColumn(DS_COLUMN, F.date_format(F.col(EVENT_TS_COLUMN), "yyyy-MM-dd"))
    )


def assign_split(
    df: DataFrame, train_days: int, validation_days: int, test_days: int
) -> DataFrame:
    """Add a ``split`` column via a contiguous time cut (see module docstring)."""
    validation_start = train_days
    test_start = train_days + validation_days
    return df.withColumn(
        SPLIT_COLUMN,
        F.when(F.col(DAY_COLUMN) < F.lit(validation_start), F.lit(SPLIT_TRAIN))
        .when(F.col(DAY_COLUMN) < F.lit(test_start), F.lit(SPLIT_VALIDATION))
        .otherwise(F.lit(SPLIT_TEST)),
    )


def write_split(df: DataFrame, split_name: str, parquet_root: str) -> str:
    """Write one split as day-partitioned Parquet under ``parquet_root/{split}``."""
    out_path = os.path.join(parquet_root, split_name)
    (
        df.repartition(DS_COLUMN)
        .write.mode("overwrite")
        .partitionBy(DS_COLUMN)
        .parquet(out_path)
    )
    logger.info("Wrote split=%s -> %s", split_name, out_path)
    return out_path


def compute_stats(df: DataFrame) -> dict[str, Any]:
    """Aggregate row counts, positive rates and per-column null rates.

    Uses two passes: one group-by over (split, day) for counts and positive
    rates, and one aggregate for the per-column null counts.
    """
    split_day_rows = (
        df.groupBy(SPLIT_COLUMN, DAY_COLUMN)
        .agg(
            F.count("*").alias("rows"),
            F.avg(F.col(LABEL_COLUMN)).alias("positive_rate"),
        )
        .collect()
    )

    split_rows: dict[str, int] = {name: 0 for name in SPLIT_NAMES}
    split_pos: dict[str, float] = {name: 0.0 for name in SPLIT_NAMES}
    per_day_pos: dict[int, float] = {}
    per_day_rows: dict[int, int] = {}
    for row in split_day_rows:
        name = row[SPLIT_COLUMN]
        day = int(row[DAY_COLUMN])
        split_rows[name] += row["rows"]
        split_pos[name] += row["positive_rate"] * row["rows"]
        per_day_pos[day] = row["positive_rate"] * row["rows"]
        per_day_rows[day] = row["rows"]

    null_agg = df.agg(
        *[
            F.sum(F.col(col).isNull().cast("long")).alias(col)
            for col in ALL_COLUMNS
        ],
        F.count("*").alias("total_rows"),
    ).collect()[0]
    total_rows = int(null_agg["total_rows"])
    null_rates = {
        col: (int(null_agg[col]) / total_rows if total_rows else 0.0)
        for col in ALL_COLUMNS
    }

    return {
        "rows": total_rows,
        "splits": {
            name: {
                "rows": split_rows[name],
                "positive_rate": split_pos[name] / split_rows[name]
                if split_rows[name]
                else 0.0,
            }
            for name in SPLIT_NAMES
        },
        "positive_rate_by_day": {
            day: (per_day_pos[day] / per_day_rows[day] if per_day_rows[day] else 0.0)
            for day in sorted(per_day_rows)
        },
        "null_rates": null_rates,
    }


def log_stats(stats: Mapping[str, Any]) -> None:
    """Print split statistics, per-day positive rates and null rates."""
    logger.info("Ingestion complete: total rows=%d", stats["rows"])
    for name in SPLIT_NAMES:
        split = stats["splits"][name]
        logger.info(
            "split=%-10s rows=%d positive_rate=%.4f",
            name,
            split["rows"],
            split["positive_rate"],
        )
    for day in sorted(stats["positive_rate_by_day"]):
        logger.info("day=%02d positive_rate=%.4f", day, stats["positive_rate_by_day"][day])
    logger.info("Null rate per column:")
    for col in ALL_COLUMNS:
        logger.info("  %-6s %.4f", col, stats["null_rates"][col])


def run(spark: SparkSession, cfg: IngestConfig) -> dict[str, Any]:
    """Run the full ingestion pipeline and return its statistics."""
    if cfg.train_days + cfg.validation_days + cfg.test_days != cfg.days:
        raise ValueError(
            "train_days + validation_days + test_days must equal days "
            f"({cfg.train_days} + {cfg.validation_days} + {cfg.test_days} != {cfg.days})"
        )

    logger.info("Reading raw TSV from %s", cfg.raw_path)
    df = read_raw(spark, cfg.raw_path)
    df = normalize_missing(df)
    df = add_event_time(df, cfg.base_ts, cfg.days)
    df = assign_split(df, cfg.train_days, cfg.validation_days, cfg.test_days)

    stats = compute_stats(df)
    for name in SPLIT_NAMES:
        split_df = df.filter(F.col(SPLIT_COLUMN) == name)
        if stats["splits"][name]["rows"] == 0:
            logger.warning("split=%s is empty", name)
        write_split(split_df, name, cfg.parquet_root)

    log_stats(stats)
    return stats


def build_spark_session(conf: Mapping[str, Any]) -> SparkSession:
    """Build a SparkSession from the ``spark`` section of ``configs/spark.yaml``.

    Driver memory and arbitrary ``spark.*`` properties are configurable
    through the YAML file without code changes. PySpark worker processes are
    pinned to the same interpreter as the driver so UDFs and Python-backed
    DataFrames never pick up an incompatible ``python3`` from PATH.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark_conf = conf.get("spark") or {}
    builder = (
        SparkSession.builder.appName(spark_conf.get("app_name", "ctr-ingest"))
        .master(spark_conf.get("master", "local[*]"))
    )
    if spark_conf.get("driver_memory"):
        builder = builder.config("spark.driver.memory", str(spark_conf["driver_memory"]))
    for key, value in (spark_conf.get("properties") or {}).items():
        builder = builder.config(str(key), str(value))
    return builder.getOrCreate()
