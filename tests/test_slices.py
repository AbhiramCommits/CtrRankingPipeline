"""Tests for the per-slice evaluation helpers."""

import numpy as np
import pandas as pd

from ctr.eval.slices import (
    evaluate_slice,
    flag_slices,
    pit_block_index,
    time_of_day_buckets,
    volume_deciles,
)


def test_volume_deciles_rank_by_volume_not_id():
    keys = np.array(["a", "a", "a", "b", "b", "c"])
    deciles = volume_deciles(keys, n_buckets=10)
    # a is the busiest -> decile 0; b -> 3; c -> 6 (with 3 groups).
    assert list(deciles) == [0, 0, 0, 3, 3, 6]


def test_pit_block_index_layout():
    # block = 13 integer features, then per PIT key (C1,C6,C9,C14):
    # [impressions, ctr]. C14_impressions is at offset 13 + 3*2.
    assert pit_block_index("C14", "impressions") == 19
    assert pit_block_index("C1", "impressions") == 13
    assert pit_block_index("C1", "ctr") == 14


def test_time_of_day_buckets_cover_24_hours():
    series = pd.Series(
        pd.to_datetime(
            ["2026-09-20 00:00", "2026-09-20 05:00", "2026-09-20 09:00",
             "2026-09-20 13:00", "2026-09-20 17:00", "2026-09-20 21:00",
             "2026-09-20 23:59"]
        )
    )
    buckets = list(time_of_day_buckets(series))
    assert buckets[:6] == ["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"]
    assert buckets[6] == "20-23"


def test_evaluate_slice_reports_metric_bundle():
    y = np.array([0, 1, 1])
    p = np.array([0.1, 0.7, 0.9])
    row = evaluate_slice(y, p)
    assert row["volume"] == 3
    assert row["auc"] == 1.0
    assert 0 < row["logloss"] < 1
    assert 0.5 < row["calibration_ratio"] < 1.5


def test_flag_slices_calibration_and_auc():
    rows = [
        {"slice": "ok", "volume": 100, "auc": 0.7, "logloss": 0.1,
         "ne": 0.5, "calibration_ratio": 1.0},
        {"slice": "overbid", "volume": 100, "auc": 0.7, "logloss": 0.1,
         "ne": 0.5, "calibration_ratio": 1.2},
        {"slice": "weak", "volume": 100, "auc": 0.65, "logloss": 0.1,
         "ne": 0.5, "calibration_ratio": 1.0},
    ]
    flagged = flag_slices(rows, global_auc=0.70)
    assert flagged[0]["flags"] == []
    assert flagged[1]["flags"] == ["calibration"]
    assert flagged[2]["flags"] == ["auc_below_global"]
