#!/usr/bin/env python3
"""CLI entrypoint for the Spark ingestion job.

Reads the raw header-less TSV dump, assigns deterministic event timestamps,
splits by time into train/validation/test, writes day-partitioned Parquet
and prints split statistics to stdout.

Usage::

    python scripts/run_ingest.py --config configs/spark.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctr.config import load_yaml
from ctr.data.ingest import IngestConfig, build_spark_session, run

logger = logging.getLogger("ctr.ingest")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spark.yaml", help="YAML config path")
    parser.add_argument("--raw", default=None, help="override ingest.raw_path")
    parser.add_argument("--out", default=None, help="override ingest.parquet_root")
    parser.add_argument("--base-ts", default=None, help="override ingest.base_ts")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    config = load_yaml(args.config)
    ingest_conf = config.get("ingest") or {}
    defaults = IngestConfig()
    cfg = IngestConfig(
        raw_path=args.raw or ingest_conf.get("raw_path", defaults.raw_path),
        parquet_root=args.out or ingest_conf.get("parquet_root", defaults.parquet_root),
        base_ts=args.base_ts or ingest_conf.get("base_ts", defaults.base_ts),
        days=int(ingest_conf.get("days", defaults.days)),
        train_days=int(ingest_conf.get("train_days", defaults.train_days)),
        validation_days=int(ingest_conf.get("validation_days", defaults.validation_days)),
        test_days=int(ingest_conf.get("test_days", defaults.test_days)),
    )
    if cfg.train_days + cfg.validation_days + cfg.test_days != cfg.days:
        raise SystemExit(
            "error: train_days + validation_days + test_days must equal days "
            f"({cfg.train_days} + {cfg.validation_days} + {cfg.test_days} != {cfg.days})"
        )
    if not Path(cfg.raw_path).is_file():
        raise SystemExit(
            f"error: raw file {cfg.raw_path!r} not found; run `make data`'s "
            "download step first"
        )

    spark = build_spark_session(config)
    try:
        spark.sparkContext.setLogLevel("WARN")
        run(spark, cfg)
    finally:
        spark.stop()
    logger.info("Ingestion job finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
