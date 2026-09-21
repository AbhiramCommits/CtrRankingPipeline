"""Proof that point-in-time aggregates never leak future (or own-row) labels.

Fixture
-------
Seven rows, five for key ``C1=A``, one for ``C1=B`` and one with a null
key. Offsets are seconds from ``2026-09-01 00:00:00`` (session timezone
UTC, matching the test session config). The trailing window is 150 seconds
and smoothing uses ``alpha=2`` with ``prior=0.1``.

::

    row_id  label  C1     offset
    ------  -----  ----   ------
    1       1      A      0
    2       0      A      100
    3       1      A      200     <- same timestamp as row 4
    4       1      A      200
    5       0      A      350
    6       1      B      50
    7       1      NULL   300

Hand-computed PIT values for key A (frame = [ts-150, ts-1], boundaries in
seconds, both inclusive):

* row 1 (t=0):   no earlier rows                       -> imps=0, ctr=prior
* row 2 (t=100): sees row 1 (t=0)                      -> imps=1, clicks=1
* row 3 (t=200): sees row 2 only: row 1 is 200s old (outside the 150s
  window) and row 4 shares its timestamp (excluded by the -1 bound)
                                                       -> imps=1, clicks=0
* row 4 (t=200): identical frame to row 3              -> imps=1, clicks=0
* row 5 (t=350): sees rows 3 and 4 (t=200); row 2 is 250s old
                                                       -> imps=2, clicks=2

The naive (leaky) alternative is ``groupBy(C1).agg(avg(label))``, which for
key A is 3/5 = 0.6 -- equal to none of the PIT values above. That
difference is the whole point of the test: if the implementation mixed its
own or future labels into the history, the values would drift toward 0.6
and the assertions below would fail.
"""

import pytest
from pyspark.sql import functions as F

from ctr.data.ingest import EVENT_TS_COLUMN
from ctr.data.synthetic import LABEL_COLUMN
from ctr.features.pit import PitConfig, add_pit_features

BASE_TS = "2026-09-01 00:00:00"
ALPHA = 2.0
PRIOR = 0.1
WINDOW_SECONDS = 150


def _smoothed(clicks: float, impressions: int) -> float:
    return round((clicks + ALPHA * PRIOR) / (impressions + ALPHA), 6)


def build_fixture(spark):
    rows = [
        (1, 1, "A", 0),
        (2, 0, "A", 100),
        (3, 1, "A", 200),
        (4, 1, "A", 200),
        (5, 0, "A", 350),
        (6, 1, "B", 50),
        (7, 1, None, 300),
    ]
    df = spark.createDataFrame(rows, ["row_id", LABEL_COLUMN, "C1", "offset_sec"])
    return df.withColumn(
        EVENT_TS_COLUMN,
        F.to_timestamp(F.lit(BASE_TS))
        + F.make_interval(
            F.lit(0),
            F.lit(0),
            F.lit(0),
            F.lit(0),
            F.lit(0),
            F.lit(0),
            F.col("offset_sec"),
        ),
    )


def test_pit_matches_hand_computed_values(spark):
    cfg = PitConfig(
        key_columns=("C1",), window_seconds=WINDOW_SECONDS, alpha=ALPHA
    )
    df = add_pit_features(build_fixture(spark), cfg, PRIOR)

    got = {
        row["row_id"]: (row["C1_impressions"], round(float(row["C1_ctr"]), 6))
        for row in df.collect()
    }
    expected = {
        1: (0, _smoothed(0, 0)),  # no history -> prior
        2: (1, _smoothed(1, 1)),  # row 1 only
        3: (1, _smoothed(0, 1)),  # row 2 only: row 1 outside window, row 4 same ts
        4: (1, _smoothed(0, 1)),  # same frame as row 3
        5: (2, _smoothed(2, 2)),  # rows 3+4; row 2 outside window
        6: (0, _smoothed(0, 0)),  # different key -> independent history
        7: (0, _smoothed(0, 0)),  # null key -> no shared history, prior only
    }
    assert got.keys() == expected.keys()
    for row_id, (imps, ctr) in expected.items():
        assert got[row_id][0] == imps, f"row {row_id} impressions"
        assert got[row_id][1] == ctr, f"row {row_id} smoothed ctr"


def test_pit_differs_from_naive_leaky_groupby(spark):
    cfg = PitConfig(
        key_columns=("C1",), window_seconds=WINDOW_SECONDS, alpha=ALPHA
    )
    df = add_pit_features(build_fixture(spark), cfg, PRIOR)

    # The naive implementation a model might reach for first: it uses every
    # label of the key, including the row's own and all future ones.
    naive = {
        row["C1"]: row["avg"]
        for row in df.groupBy("C1").agg(F.avg(LABEL_COLUMN).alias("avg")).collect()
    }
    assert naive["A"] == pytest.approx(0.6)  # 3 clicks / 5 impressions
    assert naive["B"] == pytest.approx(1.0)

    pit_by_row = {row["row_id"]: row["C1_ctr"] for row in df.collect()}
    # Every PIT value for key A is provably different from the leaky 0.6.
    for row_id in (1, 2, 3, 4, 5):
        assert pit_by_row[row_id] != pytest.approx(naive["A"])
    assert pit_by_row[6] != pytest.approx(naive["B"])


def test_invalid_configs_raise(spark):
    df = build_fixture(spark)
    with pytest.raises(ValueError):
        add_pit_features(df, PitConfig(window_seconds=0), PRIOR)
    with pytest.raises(ValueError):
        add_pit_features(df, PitConfig(), prior_ctr=1.5)
