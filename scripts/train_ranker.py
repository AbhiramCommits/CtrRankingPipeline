#!/usr/bin/env python3
"""Train the DLRM ranker and the LR / LightGBM baselines.

All models consume the same train/validation/test splits and the same
feature columns (categorical index columns + the numeric block) and report
AUC and logloss per split, so the deep model is directly comparable with
the baselines.

Usage::

    python scripts/train_ranker.py --config configs/ranker.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Must run before torch/lightgbm load their bundled libomp runtimes; loading
# two OpenMP runtimes into one process corrupts/aborts the runtime on macOS
# (see also ctr/models/baselines.py and ctr/retrieval/index.py). Sustained
# multithreaded torch training after lightgbm's runtime loads provably
# crashes in libomp workers, so OpenMP runs single-threaded here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from ctr.config import load_yaml
from ctr.data.ingest import build_spark_session
from ctr.data.synthetic import CATEGORICAL_FEATURES
from ctr.features.transforms import load_feature_stats
from ctr.models.baselines import (
    LogisticRegressionBaseline,
    lgb_predict,
    train_lightgbm,
)
from ctr.models.common import binary_auc, binary_logloss, resolve_device
from ctr.models.data import index_columns, load_split_arrays, to_tensor_dataset
from ctr.models.dlrm import DLRM
from ctr.models.train import TrainConfig, evaluate, train_dlrm

logger = logging.getLogger("ctr.ranker")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/ranker.yaml", help="YAML config path"
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
    section = config.get("ranker") or {}
    dlrm_conf = section.get("dlrm") or {}
    training_conf = section.get("training") or {}
    baseline_conf = section.get("baselines") or {}

    features_root = str(section.get("features_root", "data/features"))
    feature_artifacts = str(section.get("features_artifacts_dir", "artifacts"))
    artifacts_dir = str(section.get("artifacts_dir", "artifacts/ranker"))
    seed = int(section.get("seed", 42))

    spark = build_spark_session(config)
    try:
        spark.sparkContext.setLogLevel("WARN")
        stats = load_feature_stats(feature_artifacts)
        vocab_sizes = [
            int(stats["vocab_sizes"][field]) + 1 for field in CATEGORICAL_FEATURES
        ]

        splits = {
            name: load_split_arrays(
                spark,
                os.path.join(features_root, name),
                categorical_fields=CATEGORICAL_FEATURES,
            )
            for name in ("train", "validation", "test")
        }
        for name, split in splits.items():
            logger.info(
                "split=%s rows=%d positive_rate=%.4f",
                name,
                len(split["labels"]),
                float(split["labels"].mean()),
            )

        # --- DLRM ---
        numeric_dim = splits["train"]["numeric"].shape[1]
        model = DLRM(
            field_vocab_sizes=vocab_sizes,
            num_numeric=numeric_dim,
            embedding_dim=_int(dlrm_conf.get("embedding_dim"), 16),
            bottom_mlp_dims=tuple(dlrm_conf.get("bottom_mlp_dims", [128, 64])),
            top_mlp_dims=tuple(dlrm_conf.get("top_mlp_dims", [128, 64])),
        )
        train_cfg = TrainConfig(
            device=str(training_conf.get("device", "auto")),
            optimizer=str(training_conf.get("optimizer", "adagrad")),
            lr=_float(training_conf.get("lr"), 0.01),
            batch_size=_int(training_conf.get("batch_size"), 4096),
            epochs=_int(training_conf.get("epochs"), 8),
            patience=_int(training_conf.get("patience"), 3),
            grad_clip=_float(training_conf.get("grad_clip"), 1.0),
            seed=seed,
        )
        device = resolve_device(train_cfg.device)
        train_metrics = train_dlrm(
            model,
            to_tensor_dataset(splits["train"]),
            to_tensor_dataset(splits["validation"]),
            train_cfg,
            artifacts_dir,
        )
        test_metrics = evaluate(
            model,
            to_tensor_dataset(splits["test"]),
            device=str(device),
            batch_size=train_cfg.batch_size,
        )

        # --- Baselines ---
        lr_conf = baseline_conf.get("logistic") or {}
        lgb_conf = baseline_conf.get("lightgbm") or {}
        logger.info("Fitting LogisticRegression baseline (hashed one-hot + numeric)")
        logistic = LogisticRegressionBaseline(
            n_buckets=_int(lr_conf.get("n_buckets"), 1 << 18),
            C=_float(lr_conf.get("C"), 1.0),
            max_iter=_int(lr_conf.get("max_iter"), 200),
        )
        logistic.fit(
            splits["train"]["cats"],
            splits["train"]["numeric"],
            splits["train"]["labels"],
        )

        logger.info("Fitting LightGBM baseline (native categorical handling)")
        cat_columns = index_columns()
        booster = train_lightgbm(
            splits["train"]["cats"],
            splits["train"]["numeric"],
            splits["train"]["labels"],
            splits["validation"]["cats"],
            splits["validation"]["numeric"],
            splits["validation"]["labels"],
            categorical_columns=cat_columns,
            params={
                "num_leaves": _int(lgb_conf.get("num_leaves"), 31),
                "learning_rate": _float(lgb_conf.get("learning_rate"), 0.1),
            },
            num_boost_round=_int(lgb_conf.get("num_boost_round"), 200),
            early_stopping_rounds=_int(lgb_conf.get("early_stopping_rounds"), 20),
        )

        # --- Comparison report ---
        report = {
            "dlrm": {
                "validation": {
                    "auc": train_metrics["final_val"]["auc"],
                    "logloss": train_metrics["final_val"]["logloss"],
                },
                "test": test_metrics,
            },
            "logistic_regression": {},
            "lightgbm": {},
        }
        for name, split in splits.items():
            lr_probs = logistic.predict_proba(split["cats"], split["numeric"])
            lgb_probs = lgb_predict(
                booster, split["cats"], split["numeric"], cat_columns
            )
            report["logistic_regression"][name] = {
                "auc": binary_auc(split["labels"], lr_probs),
                "logloss": binary_logloss(split["labels"], lr_probs),
            }
            report["lightgbm"][name] = {
                "auc": binary_auc(split["labels"], lgb_probs),
                "logloss": binary_logloss(split["labels"], lgb_probs),
            }

        logger.info("Model comparison (validation/test):")
        for model_name, splits_metrics in report.items():
            for split_name, metrics in splits_metrics.items():
                logger.info(
                    "  %-22s %-10s auc=%.4f logloss=%.5f",
                    model_name,
                    split_name,
                    metrics["auc"],
                    metrics["logloss"],
                )

        with open(os.path.join(artifacts_dir, "comparison.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    finally:
        spark.stop()
    logger.info("Ranking stage finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
