"""Tests for train-fit transforms: clipping, imputation, vocab/OOV mapping.

The key property under test is the no-leakage contract: everything a
validation/test row is transformed with was fitted on the train split and
persisted to ``artifacts/``. The assertions verify that a held-out split
reuses the persisted train statistics (median, p99, vocabularies) rather
than learning from its own distribution.
"""

import json
import math
import os

from pyspark.sql import functions as F

from ctr.data.ingest import DS_COLUMN, EVENT_TS_COLUMN
from ctr.data.synthetic import LABEL_COLUMN
from ctr.features.transforms import (
    NUMERIC_BLOCK_COLUMN,
    OOV_INDEX,
    FeatureConfig,
    build_feature_frame,
    fit_transforms,
    load_feature_stats,
    load_vocab,
)

BASE_TS = "2026-09-01 00:00:00"


def _with_event_ts(df):
    return (
        df.withColumn("__row", F.monotonically_increasing_id())
        .withColumn(
            EVENT_TS_COLUMN,
            F.to_timestamp(F.lit(BASE_TS))
            + F.make_interval(
                F.lit(0),
                F.lit(0),
                F.lit(0),
                F.lit(0),
                F.lit(0),
                F.lit(0),
                F.col("__row"),
            ),
        )
        .withColumn(DS_COLUMN, F.date_format(F.col(EVENT_TS_COLUMN), "yyyy-MM-dd"))
        .drop("__row")
    )


def make_train(spark):
    # C1 counts: b=3, a=2, null=1 -> with min_count=2 the vocab keeps
    # {b -> index 1, a -> index 2} (descending frequency).
    # C2 counts: x=2, y=1, z=1, w=1, null=1 -> vocab keeps {x -> index 1}.
    rows = [
        (1, 5, 10, "a", "x"),
        (0, 10, 20, "b", "x"),
        (1, 100, 30, "b", "y"),
        (0, 1000, 40, "b", "z"),
        (1, 7, 50, "a", "w"),
        (0, 3, None, None, None),
    ]
    return _with_event_ts(
        spark.createDataFrame(rows, [LABEL_COLUMN, "I1", "I2", "C1", "C2"])
    )


def make_validation(spark):
    # Row 0: I1=1e9 -> must clip at the TRAIN p99; C2="new" unseen at fit.
    # Row 1: C1="c" seen once at train (below min_count) -> OOV.
    # Row 2: all nulls -> train medians and OOV indices.
    rows = [
        (1, 10**9, None, "a", "new"),
        (0, 5, 6, "c", "x"),
        (1, None, None, None, None),
    ]
    return _with_event_ts(
        spark.createDataFrame(rows, [LABEL_COLUMN, "I1", "I2", "C1", "C2"])
    )


def test_fit_persists_and_transform_reuses_train_stats(tmp_path, spark):
    cfg = FeatureConfig(
        numeric_columns=("I1", "I2"),
        categorical_columns=("C1", "C2"),
        min_count=2,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    summary = fit_transforms(make_train(spark), cfg)

    # Fitted statistics are persisted and round-trip through JSON.
    stats_path = os.path.join(cfg.artifacts_dir, "feature_stats.json")
    assert os.path.isfile(stats_path)
    with open(stats_path, "r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert (
        persisted["numeric"]["I1"]["median"]
        == summary["numeric"]["I1"]["median"]
    )
    assert summary["vocab_sizes"] == {"C1": 2, "C2": 1}

    # Vocabularies are persisted as Parquet and readable.
    c1_values = {
        row["value"] for row in load_vocab(spark, cfg.artifacts_dir, "C1").select("value").collect()
    }
    assert c1_values == {"a", "b"}

    # Transform a held-out split using ONLY the persisted train artifacts.
    stats = load_feature_stats(cfg.artifacts_dir)
    vocabs = {
        col: load_vocab(spark, cfg.artifacts_dir, col)
        for col in cfg.categorical_columns
    }
    features = build_feature_frame(
        make_validation(spark), cfg, stats["numeric"], vocabs
    )
    rows = features.collect()
    assert len(rows) == 3

    p99_i1 = stats["numeric"]["I1"]["p99"]
    median_i1 = stats["numeric"]["I1"]["median"]
    median_i2 = stats["numeric"]["I2"]["median"]

    # Row 0: huge I1 clipped at train p99; null I2 -> train median;
    # C1 "a" -> index 2; C2 "new" -> OOV (never in train vocab).
    r0 = rows[0]
    assert r0[LABEL_COLUMN] == 1
    assert abs(r0[NUMERIC_BLOCK_COLUMN][0] - p99_i1) < 1e-4
    assert abs(r0[NUMERIC_BLOCK_COLUMN][1] - median_i2) < 1e-4
    assert r0["C1_idx"] == 2
    assert r0["C2_idx"] == OOV_INDEX

    # Row 1: I1=5 -> log1p(5), below p99 so unchanged;
    # C1 "c" -> below min_count -> OOV; C2 "x" -> index 1.
    r1 = rows[1]
    assert abs(r1[NUMERIC_BLOCK_COLUMN][0] - math.log1p(5)) < 1e-4
    assert r1["C1_idx"] == OOV_INDEX
    assert r1["C2_idx"] == 1

    # Row 2: nulls -> train medians and OOV everywhere.
    r2 = rows[2]
    assert abs(r2[NUMERIC_BLOCK_COLUMN][0] - median_i1) < 1e-4
    assert abs(r2[NUMERIC_BLOCK_COLUMN][1] - median_i2) < 1e-4
    assert r2["C1_idx"] == OOV_INDEX
    assert r2["C2_idx"] == OOV_INDEX

    # Sanity: medians are inside the transformed value range of the train
    # data (log1p of [3..1000]).
    assert 1.3 < median_i1 < 7.0
    assert r0[EVENT_TS_COLUMN] is not None
    assert r0[DS_COLUMN] is not None
