"""Unit tests for the deterministic synthetic dataset generator."""

import zlib
from collections import Counter
from pathlib import Path

from ctr.data.synthetic import (
    ALL_COLUMNS,
    CATEGORICAL_FEATURES,
    INTEGER_FEATURES,
    SyntheticConfig,
    day_of_features,
    generate_synthetic,
)


def make_cfg(tmp_path: Path, rows: int, seed: int, chunk: int = 5_000) -> SyntheticConfig:
    return SyntheticConfig(
        rows=rows, seed=seed, chunk_size=chunk, output_path=str(tmp_path / "train.txt")
    )


def read_lines(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def parse_line(line: str):
    return line.split("\t")


def test_schema_shape_and_basic_stats(tmp_path):
    cfg = make_cfg(tmp_path, rows=20_000, seed=123)
    stats = generate_synthetic(cfg)
    lines = read_lines(tmp_path / "train.txt")
    assert len(lines) == cfg.rows
    for line in lines[:200]:
        fields = parse_line(line)
        assert len(fields) == len(ALL_COLUMNS) == 40
        assert fields[0] in {"0", "1"}
        for value, col in zip(fields[1:14], INTEGER_FEATURES):
            assert value == "" or value.lstrip("-").isdigit(), col
        for value, col in zip(fields[14:], CATEGORICAL_FEATURES):
            assert value == "" or (
                len(value) <= 8 and all(ch in "0123456789abcdef" for ch in value)
            ), col
    assert abs(stats.positive_rate - 0.035) < 0.005


def test_deterministic_generation(tmp_path):
    first_path = tmp_path / "train.txt"
    second_path = tmp_path / "train_again.txt"
    generate_synthetic(make_cfg(tmp_path, rows=10_000, seed=7))
    first = first_path.read_bytes()
    generate_synthetic(
        SyntheticConfig(rows=10_000, seed=7, chunk_size=5_000, output_path=str(second_path))
    )
    assert first == second_path.read_bytes()


def test_positive_rate_drifts_up_over_days(tmp_path):
    cfg = make_cfg(tmp_path, rows=60_000, seed=99, chunk=10_000)
    stats = generate_synthetic(cfg)
    by_day = stats.positive_rate_by_day
    early = sum(by_day[d] for d in range(3)) / 3
    late = sum(by_day[d] for d in range(18, 21)) / 3
    assert late > early + 0.003


def test_categorical_zipf_skew(tmp_path):
    rows = 60_000
    generate_synthetic(make_cfg(tmp_path, rows=rows, seed=11))
    counter = Counter()
    for line in read_lines(tmp_path / "train.txt"):
        c1 = parse_line(line)[14]  # C1 is the first categorical column
        if c1:
            counter[c1] += 1
    top5_share = sum(n for _, n in counter.most_common(5)) / rows
    assert top5_share > 0.2
    assert len(counter) > 100  # non-degenerate cardinality


def test_missing_values_present(tmp_path):
    generate_synthetic(make_cfg(tmp_path, rows=20_000, seed=5))
    empty_i6 = empty_c1 = 0
    for line in read_lines(tmp_path / "train.txt"):
        fields = parse_line(line)
        if fields[6] == "":  # I6 is column index 6
            empty_i6 += 1
        if fields[14] == "":  # C1 is column index 14
            empty_c1 += 1
    assert empty_i6 > 0
    assert empty_c1 > 0


def test_day_hash_matches_ingest_convention():
    line = "5\t1\t\ta73ee510\t0"
    expected = (zlib.crc32(line.encode("utf-8")) & 0xFFFFFFFF) % 21
    assert day_of_features(line) == expected
    assert 0 <= day_of_features(line) < 21
