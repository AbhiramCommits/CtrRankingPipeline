"""Tests for the Spark ingestion job: schema, split logic, end-to-end run."""

import datetime

import pytest

from ctr.data.ingest import (
    RAW_SCHEMA,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    IngestConfig,
    run,
    split_name_for_day,
)
from ctr.data.synthetic import ALL_COLUMNS, SyntheticConfig, generate_synthetic

EXPECTED_FIELD_TYPES = {
    "label": "integer",
    **{f"I{i}": "long" for i in range(1, 14)},
    **{f"C{i}": "string" for i in range(1, 27)},
}


def test_raw_schema_is_explicit_and_complete():
    assert [field.name for field in RAW_SCHEMA.fields] == ALL_COLUMNS
    for field in RAW_SCHEMA.fields:
        assert field.dataType.typeName() == EXPECTED_FIELD_TYPES[field.name]


def test_split_name_for_day_boundaries():
    assert split_name_for_day(0) == SPLIT_TRAIN
    assert split_name_for_day(16) == SPLIT_TRAIN
    assert split_name_for_day(17) == SPLIT_VALIDATION
    assert split_name_for_day(18) == SPLIT_VALIDATION
    assert split_name_for_day(19) == SPLIT_TEST
    assert split_name_for_day(20) == SPLIT_TEST
    with pytest.raises(ValueError):
        split_name_for_day(21)
    with pytest.raises(ValueError):
        split_name_for_day(-1)


def test_end_to_end_ingest(tmp_path, spark):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    parquet_root = tmp_path / "parquet"
    generate_synthetic(
        SyntheticConfig(
            rows=60_000, seed=3, chunk_size=10_000, output_path=str(raw_dir / "train.txt")
        )
    )

    cfg = IngestConfig(
        raw_path=str(raw_dir / "train.txt"),
        parquet_root=str(parquet_root),
        base_ts="2026-09-01 00:00:00",
    )
    stats = run(spark, cfg)

    # Row conservation across splits.
    split_rows = {name: split["rows"] for name, split in stats["splits"].items()}
    assert sum(split_rows.values()) == 60_000
    assert all(rows > 0 for rows in split_rows.values())

    # Temporal boundaries and timestamp window, verified on the persisted data.
    # NOTE: timestamps are compared as epoch seconds (unix_seconds) because
    # PySpark converts collected timestamps to datetimes in the OS-local
    # timezone, which need not match the Spark session timezone (UTC here).
    day_ranges = {
        SPLIT_TRAIN: (0, 16),
        SPLIT_VALIDATION: (17, 18),
        SPLIT_TEST: (19, 20),
    }
    base_unix = int(
        datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc).timestamp()
    )
    for name, (day_lo, day_hi) in day_ranges.items():
        df = spark.read.parquet(str(parquet_root / name))
        days = {row["event_day"] for row in df.select("event_day").collect()}
        assert days, f"split {name} is empty"
        assert min(days) >= day_lo and max(days) <= day_hi
        ts_lo, ts_hi = df.selectExpr(
            "min(unix_seconds(event_ts))", "max(unix_seconds(event_ts))"
        ).collect()[0]
        assert ts_lo >= base_unix
        assert ts_hi < base_unix + 21 * 86_400
        # day partition column is consistent with the timestamp (ds strings
        # are formatted in the session timezone, UTC, so they are exact).
        ds = {row["ds"] for row in df.select("ds").collect()}
        assert all(
            datetime.date(2026, 9, 1) <= datetime.date.fromisoformat(str(d))
            < datetime.date(2026, 9, 22)
            for d in ds
        )

    # The injected day-over-day drift survived into the time-based splits.
    by_day = stats["positive_rate_by_day"]
    early = sum(by_day[d] for d in range(3)) / 3
    late = sum(by_day[d] for d in range(18, 21)) / 3
    assert late > early + 0.003

    # Missing values survived as SQL NULLs (I6 is ~15% empty in the generator).
    assert stats["null_rates"]["I6"] > 0.05
    assert stats["null_rates"]["C1"] > 0.01
