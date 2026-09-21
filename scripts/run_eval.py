#!/usr/bin/env python3
"""Evaluate the ranking models and the two-stage pipeline on the test split.

Loads the DLRM checkpoint and the persisted LR / LightGBM baselines, scores
the test split with each, and emits under ``artifacts/eval/``:

* ``report.md`` / ``report.json`` — global metrics (AUC, logloss, NE, ECE,
  GAUC, calibration), flagged per-slice tables, and the retrieval/ranking
  two-stage analysis;
* ``roc.png`` and ``reliability.png``.

Usage::

    python scripts/run_eval.py --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Same libomp duplicate-runtime guard as the training scripts: this process
# loads torch, lightgbm and faiss together.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ctr.config import load_yaml
from ctr.data.synthetic import CATEGORICAL_FEATURES
from ctr.eval import slices as sl
from ctr.eval.report import (
    compute_global_metrics,
    evaluate_retrieval_stage,
    plot_reliability,
    plot_roc,
    predict_dlrm,
    render_markdown,
    write_report,
)
from ctr.models.baselines import lgb_predict
from ctr.models.data import index_columns
from ctr.models.dlrm import DLRM

logger = logging.getLogger("ctr.eval")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/eval.yaml", help="YAML config path"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    config = load_yaml(args.config)
    section = config.get("eval") or {}

    features_root = str(section.get("features_root", "data/features"))
    retrieval_artifacts = str(
        section.get("retrieval_artifacts", "artifacts/retrieval")
    )
    ranker_artifacts = str(section.get("ranker_artifacts", "artifacts/ranker"))
    output_dir = str(section.get("output_dir", "artifacts/eval"))
    n_buckets = int(section.get("n_buckets", 20))
    recall_ks = tuple(section.get("recall_ks", [10, 50, 100]))
    search_k = int(section.get("retrieval_search_k", 100))

    logger.info("Loading test split from %s", features_root)
    frame = pd.read_parquet(os.path.join(features_root, "test"))
    numeric = np.stack(frame["numeric_features"].to_numpy(), axis=0).astype(
        np.float32
    )
    cats = frame[[f"{f}_idx" for f in CATEGORICAL_FEATURES]].to_numpy(dtype=np.int64)
    labels = frame["label"].to_numpy(dtype=np.float32)
    groups = cats[:, CATEGORICAL_FEATURES.index("C14")]
    logger.info("test rows=%d, positives=%d", len(labels), int(labels.sum()))

    logger.info("Scoring the test split with all three rankers")
    dlrm = DLRM.from_checkpoint(os.path.join(ranker_artifacts, "model.pt"), "cpu")
    logistic = joblib.load(os.path.join(ranker_artifacts, "logistic.joblib"))
    booster = lgb.Booster(model_file=os.path.join(ranker_artifacts, "lightgbm.txt"))
    cat_columns = index_columns()

    predictions = {
        "dlrm": predict_dlrm(dlrm, cats, numeric),
        "logistic_regression": logistic.predict_proba(cats, numeric),
        "lightgbm": lgb_predict(booster, cats, numeric, cat_columns),
    }

    logger.info("Computing global metrics")
    global_table = {
        name: compute_global_metrics(labels, probs, groups, n_buckets)
        for name, probs in predictions.items()
    }
    for name, metrics in global_table.items():
        logger.info(
            "%s: auc=%.4f logloss=%.4f ne=%.4f ece=%.4f gauc=%.4f ratio=%.3f",
            name,
            metrics["auc"],
            metrics["logloss"],
            metrics["ne"],
            metrics["ece"],
            metrics["gauc"],
            metrics["calibration_ratio"],
        )

    logger.info("Computing per-slice metrics")
    slice_tables = {
        name: sl.build_slices(frame, labels, probs, global_table[name]["auc"])
        for name, probs in predictions.items()
    }
    for name, tables in slice_tables.items():
        flagged = sum(
            1
            for rows in tables.values()
            for row in rows
            if row["flags"]
        )
        logger.info("%s: %d flagged slice rows", name, flagged)

    logger.info("Evaluating the two-stage retrieval -> ranking pipeline")
    retrieval = evaluate_retrieval_stage(
        frame, retrieval_artifacts, recall_ks=recall_ks, search_k=search_k
    )
    logger.info(
        "retrieval recall=%s mrr@%d=%.4f p@1=%.4f",
        retrieval["recall_at_k"],
        search_k,
        retrieval["mrr_on_shortlist"],
        retrieval["precision_at_1"],
    )

    logger.info("Writing the report to %s", output_dir)
    markdown = render_markdown(global_table, slice_tables, retrieval)
    payload = {
        "global": global_table,
        "slices": slice_tables,
        "two_stage": retrieval,
    }
    paths = write_report(output_dir, markdown, payload)
    paths["roc"] = plot_roc(output_dir, labels, predictions)
    paths["reliability"] = plot_reliability(
        output_dir, labels, predictions, n_buckets
    )
    for kind, path in paths.items():
        logger.info("  %s -> %s", kind, path)
    logger.info("Evaluation finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
