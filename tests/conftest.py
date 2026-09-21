"""Shared pytest fixtures."""

import shutil

import pytest


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local SparkSession for integration tests.

    Skipped when pyspark or a JVM is unavailable so the unit tests still run
    in minimal environments.
    """
    pytest.importorskip("pyspark")
    if shutil.which("java") is None:
        pytest.skip("java is required for Spark integration tests")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("ctr-ingest-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
