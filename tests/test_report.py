"""Smoke tests for the report rendering and plotting."""

import json
import os

import numpy as np

from ctr.eval.report import (
    plot_reliability,
    plot_roc,
    render_markdown,
    write_report,
)


def _fixture_global_table():
    return {
        "dlrm": {
            "auc": 0.55, "logloss": 0.17, "ne": 0.98, "ece": 0.01,
            "gauc": 0.56, "calibration_ratio": 1.02, "volume": 95047,
        },
        "lightgbm": {
            "auc": 0.59, "logloss": 0.16, "ne": 0.95, "ece": 0.005,
            "gauc": 0.60, "calibration_ratio": 1.0, "volume": 95047,
        },
    }


def _fixture_slice_tables():
    row = {
        "slice": "time_of_day_00-03",
        "volume": 1000,
        "auc": 0.5,
        "logloss": 0.2,
        "ne": 1.1,
        "calibration_ratio": 1.2,
        "flags": ["calibration"],
    }
    return {
        model: {kind: [row] for kind in
                ("user_segment_decile", "placement_decile",
                 "pit_impression_decile", "time_of_day")}
        for model in ("dlrm", "lightgbm")
    }


def test_render_markdown_contains_tables_and_flags(tmp_path):
    retrieval = {
        "num_queries": 100,
        "search_k": 100,
        "recall_at_k": {10: 0.1, 50: 0.3, 100: 0.5},
        "mrr_on_shortlist": 0.2,
        "precision_at_1": 0.1,
        "cap_note": "cap",
    }
    markdown = render_markdown(
        _fixture_global_table(), _fixture_slice_tables(), retrieval
    )
    assert "# CTR Offline Evaluation Report" in markdown
    assert "dlrm" in markdown and "lightgbm" in markdown
    assert "calibration" in markdown
    assert "retrieval recall@100: 0.5000" in markdown
    assert "cap" in markdown


def test_write_report_roundtrip(tmp_path):
    payload = {"global": _fixture_global_table(), "slices": {}}
    paths = write_report(str(tmp_path), "hello\n", payload)
    assert os.path.isfile(paths["markdown"])
    assert os.path.isfile(paths["json"])
    with open(paths["json"]) as handle:
        assert json.load(handle)["global"]["dlrm"]["auc"] == 0.55


def test_plots_are_written(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(np.float64)
    predictions = {
        "dlrm": rng.random(500),
        "lightgbm": np.clip(rng.random(500) + 0.1 * y, 0, 1),
    }
    roc_path = plot_roc(str(tmp_path), y, predictions)
    rel_path = plot_reliability(str(tmp_path), y, predictions, n_buckets=10)
    assert os.path.isfile(roc_path)
    assert os.path.isfile(rel_path)
