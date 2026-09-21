#!/usr/bin/env python3
"""CLI entrypoint for the feature engineering step.

Fits transforms on the train split only (persisted under ``artifacts/``),
computes point-in-time-correct rolling aggregates, and writes feature
Parquet for every split to ``data/features/{split}/``.

Usage::

    python scripts/run_features.py --config configs/features.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

from ctr.config import load_yaml
from ctr.data.ingest import SPLIT_NAMES, build_spark_session
from ctr.data.synthetic import LABEL_COLUMN
from ctr.features.pit import PIT_KEY_COLUMNS, PitConfig, add_pit_features
from ctr.features.transforms import (
    FeatureConfig,
    build_feature_frame,
    fit_transforms,
    load_feature_stats,
    load_vocab,
)

logger = logging.getLogger("ctr.features")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/features.yaml", help="YAML config path"
    )
    return parser.parse_args(argv)


def _int(value, default: int) -> int:
    return int(value) if value is not None else default


def _float(value, default: float) -> float:
    return float(value) if value is not None else default


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    config = load_yaml(args.config)
    feats = config.get("features") or {}
    pit_conf = feats.get("pit") or {}

    cfg = FeatureConfig(
        min_count=_int(feats.get("min_count"), 10),
        clip_quantile=_float(feats.get("clip_quantile"), 0.99),
        artifacts_dir=str(feats.get("artifacts_dir", "artifacts")),
    )
    pit = PitConfig(
        key_columns=tuple(pit_conf.get("key_columns", PIT_KEY_COLUMNS)),
        window_seconds=_int(pit_conf.get("window_seconds"), 86_400),
        alpha=_float(pit_conf.get("alpha"), 50.0),
    )
    input_root = str(feats.get("input_root", "data/parquet"))
    output_root = str(feats.get("output_root", "data/features"))

    spark = build_spark_session(config)
    try:
        spark.sparkContext.setLogLevel("WARN")

        train = spark.read.parquet(os.path.join(input_root, "train"))
        prior_ctr = float(train.agg(F.avg(LABEL_COLUMN)).first()[0])
        logger.info("Global train CTR (PIT smoothing prior) = %.5f", prior_ctr)

        # Fit on train only; everything below reuses the persisted artifacts
        # so validation/test never see their own statistics.
        fit_transforms(train, cfg)
        stats = load_feature_stats(cfg.artifacts_dir)
        vocabs = {
            col: load_vocab(spark, cfg.artifacts_dir, col)
            for col in cfg.categorical_columns
        }

        pit_columns = [
            name
            for key in pit.key_columns
            for name in (f"{key}_impressions", f"{key}_ctr")
        ]

        last_features = None
        for split in SPLIT_NAMES:
            split_df = spark.read.parquet(os.path.join(input_root, split))
            split_df = add_pit_features(split_df, pit, prior_ctr)
            features = build_feature_frame(
                split_df, cfg, stats["numeric"], vocabs, extra_numeric=pit_columns
            )
            out_path = os.path.join(output_root, split)
            features.write.mode("overwrite").parquet(out_path)
            rows = features.count()
            logger.info("split=%-10s rows=%d -> %s", split, rows, out_path)
            last_features = features

        numeric_count = len(cfg.numeric_columns) + len(pit_columns)
        total_vocab = sum(stats["vocab_sizes"].values()) + len(
            cfg.categorical_columns
        )
        logger.info("Feature engineering complete:")
        logger.info(
            "  numeric features: %d (%d transformed integer + %d point-in-time)",
            numeric_count,
            len(cfg.numeric_columns),
            len(pit_columns),
        )
        logger.info("  categorical index fields: %d", len(cfg.categorical_columns))
        logger.info(
            "  total embedding vocabulary size: %d (incl. %d OOV slots)",
            total_vocab,
            len(cfg.categorical_columns),
        )
        if last_features is not None:
            last_features.printSchema()
    finally:
        spark.stop()
    logger.info("Feature job finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
