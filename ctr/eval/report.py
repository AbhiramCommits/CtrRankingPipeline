"""Offline evaluation report: comparison tables, slices, plots, two-stage.

Produces, under ``artifacts/eval/``:

* ``report.md`` / ``report.json``: global metrics (AUC, log loss, NE, ECE,
  GAUC, calibration ratio) for every ranker model on the test split, all
  slice tables, and the two-stage retrieval/ranking analysis;
* ``roc.png``: ROC curves for all models;
* ``reliability.png``: 20-bucket reliability diagram per model.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ctr.data.synthetic import CATEGORICAL_FEATURES
from ctr.eval import metrics as m

logger = logging.getLogger(__name__)

GLOBAL_METRIC_COLUMNS = [
    "auc",
    "logloss",
    "ne",
    "ece",
    "gauc",
    "calibration_ratio",
]


def compute_global_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_buckets: int = 20,
) -> dict[str, float]:
    """The metric bundle reported per model on the test split."""
    return {
        "auc": m.roc_auc(y_true, y_score),
        "logloss": m.log_loss(y_true, y_score),
        "ne": m.normalized_entropy(y_true, y_score),
        "ece": m.ece(y_true, y_score, n_buckets),
        "gauc": m.gauc(y_true, y_score, groups),
        "calibration_ratio": m.calibration_ratio(y_true, y_score),
        "volume": len(y_true),
    }


def predict_dlrm(
    model, cats: np.ndarray, numeric: np.ndarray, batch_size: int = 8192
) -> np.ndarray:
    """Sigmoid probabilities for the test split, batched over the rows."""
    import torch

    was_training = model.training
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(cats), batch_size):
            batch_cats = torch.from_numpy(cats[start : start + batch_size])
            batch_numeric = torch.from_numpy(numeric[start : start + batch_size])
            outputs.append(
                torch.sigmoid(model(batch_cats, batch_numeric)).cpu().numpy()
            )
    model.train(was_training)
    return np.concatenate(outputs)


def evaluate_retrieval_stage(
    frame: pd.DataFrame,
    retrieval_artifacts: str,
    recall_ks: Sequence[int] = (10, 50, 100),
    search_k: int = 100,
) -> dict[str, Any]:
    """Retrieval recall@k on test positives + shortlist ranking stats.

    Retrieval caps the whole pipeline: a positive whose ad is not retrieved
    can never be ranked by any downstream ranker, so ``recall@k`` is the
    hard ceiling on achievable top-line quality. For the positives that are
    retrieved, MRR and P@1 over the shortlist measure how well the ad would
    surface within the candidate list.
    """
    import torch

    from ctr.retrieval.index import RetrievalIndex
    from ctr.retrieval.two_tower import (
        AD_IDENTITY_FIELDS,
        CONTEXT_FIELDS,
        TwoTower,
    )

    positives = frame[frame["label"] == 1]
    cats_all = positives[
        [f"{f}_idx" for f in CATEGORICAL_FEATURES]
    ].to_numpy(dtype=np.int64)
    numeric = np.stack(positives["numeric_features"].to_numpy(), axis=0).astype(
        np.float32
    )
    context_positions = [CATEGORICAL_FEATURES.index(f) for f in CONTEXT_FIELDS]
    ad_positions = [CATEGORICAL_FEATURES.index(f) for f in AD_IDENTITY_FIELDS]

    model = TwoTower.from_checkpoint(
        os.path.join(retrieval_artifacts, "model.pt"), device="cpu"
    )
    service = RetrievalIndex.from_artifacts(model, retrieval_artifacts, use_ivf=True)

    true_ids = [
        f"{row[0]}|{row[1]}|{row[2]}" for row in cats_all[:, ad_positions]
    ]
    context_cats = torch.from_numpy(cats_all[:, context_positions])
    context_numeric = torch.from_numpy(numeric)
    retrieved_ids, _ = service.retrieve((context_cats, context_numeric), k=search_k)

    ranks = []
    for true_id, candidates in zip(true_ids, retrieved_ids):
        try:
            ranks.append(candidates.index(true_id) + 1)
        except ValueError:
            ranks.append(-1)
    ranks = np.array(ranks, dtype=np.int64)

    hit = ranks > 0
    return {
        "num_queries": len(true_ids),
        "search_k": int(search_k),
        "recall_at_k": {int(k): float(np.mean(ranks <= k)) for k in recall_ks},
        "mrr_on_shortlist": float(np.mean(1.0 / ranks[hit])) if hit.any() else 0.0,
        "precision_at_1": float(np.mean(ranks == 1)),
        "cap_note": (
            "Retrieval recall@k caps the achievable top-line quality: a "
            "positive whose ad is not retrieved can never be ranked by the "
            "downstream ranker, however good it is."
        ),
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def render_markdown(
    global_table: dict[str, dict[str, float]],
    slice_tables: dict[str, dict[str, list[dict]]],
    retrieval: dict[str, Any],
) -> str:
    """Render the full evaluation report as markdown."""
    lines = ["# CTR Offline Evaluation Report", ""]
    lines.append("## Global metrics (test split)")
    lines.append("")
    lines.append(
        "| model | " + " | ".join(GLOBAL_METRIC_COLUMNS) + " | volume |"
    )
    lines.append(
        "| --- | " + " | ".join(["---"] * (len(GLOBAL_METRIC_COLUMNS) + 1)) + " |"
    )
    for model_name, metrics in global_table.items():
        lines.append(
            f"| {model_name} | "
            + " | ".join(_fmt(metrics[col]) for col in GLOBAL_METRIC_COLUMNS)
            + f" | {metrics['volume']} |"
        )
    lines.append("")

    lines.append("## Slice analysis")
    lines.append("")
    lines.append(
        "Flags: `calibration` = ratio outside [0.9, 1.1]; "
        "`auc_below_global` = AUC > 2 points below the global value."
    )
    for slice_kind in ("user_segment_decile", "placement_decile",
                       "pit_impression_decile", "time_of_day"):
        for model_name, tables in slice_tables.items():
            rows = tables[slice_kind]
            lines.append("")
            lines.append(f"### {slice_kind} — {model_name}")
            lines.append("")
            lines.append(
                "| slice | volume | auc | logloss | ne | calibration_ratio | flags |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for row in rows:
                lines.append(
                    f"| {row['slice']} | {row['volume']} | {_fmt(row['auc'])} | "
                    f"{_fmt(row['logloss'])} | {_fmt(row['ne'])} | "
                    f"{_fmt(row['calibration_ratio'])} | "
                    f"{', '.join(row['flags']) or '—'} |"
                )
        lines.append("")

    lines.append("## Two-stage pipeline (retrieval → ranking)")
    lines.append("")
    lines.append(f"- test positive queries: {retrieval['num_queries']}")
    for k, recall in retrieval["recall_at_k"].items():
        lines.append(f"- retrieval recall@{k}: {recall:.4f}")
    lines.append(f"- shortlist MRR@{retrieval['search_k']}: {retrieval['mrr_on_shortlist']:.4f}")
    lines.append(f"- shortlist precision@1: {retrieval['precision_at_1']:.4f}")
    lines.append(f"- {retrieval['cap_note']}")
    lines.append("")
    return "\n".join(lines)


def _sanitize(value: Any) -> Any:
    """Replace NaN/inf with None so report.json stays valid JSON."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _sanitize(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def write_report(
    output_dir: str,
    markdown: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    """Persist report.md and report.json; return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "report.md")
    json_path = os.path.join(output_dir, "report.json")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize(payload), handle, indent=2)
    return {"markdown": md_path, "json": json_path}


def plot_roc(
    output_dir: str, y_true: np.ndarray, predictions: dict[str, np.ndarray]
) -> str:
    """ROC curves for every model on one axes."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7, 6))
    for name, probs in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={m.roc_auc(y_true, probs):.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (test split)")
    plt.legend(loc="lower right")
    path = os.path.join(output_dir, "roc.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def plot_reliability(
    output_dir: str,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_buckets: int = 20,
) -> str:
    """20-bucket reliability diagram per model (mean predicted vs actual)."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(predictions), figsize=(6 * len(predictions), 5), squeeze=False
    )
    for ax, (name, probs) in zip(axes[0], predictions.items()):
        curve = m.reliability_curve(y_true, probs, n_buckets)
        ax.plot(
            curve["mean_predicted"],
            curve["mean_actual"],
            "o-",
            label="model",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        ax.set_xlabel("mean predicted")
        ax.set_ylabel("mean actual")
        ax.set_title(f"{name} (ECE={m.ece(y_true, probs, n_buckets):.4f})")
        ax.legend(loc="upper left")
    fig.suptitle("Reliability diagram (test split)")
    path = os.path.join(output_dir, "reliability.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
