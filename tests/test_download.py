"""Tests for the download/synthesize fallback logic in scripts/download_data.py."""

from ctr.data.synthetic import SyntheticConfig
from scripts import download_data


def test_no_url_generates_synthetic(tmp_path):
    out = tmp_path / "train.txt"
    cfg = SyntheticConfig(rows=5_000, seed=11, chunk_size=1_000, output_path=str(out))
    stats = download_data.generate_or_download(cfg, None, out)
    assert stats.source == "synthetic"
    assert out.is_file()
    assert stats.rows == 5_000


def test_unreachable_url_falls_back_to_synthetic(tmp_path):
    out = tmp_path / "train.txt"
    cfg = SyntheticConfig(rows=5_000, seed=12, chunk_size=1_000, output_path=str(out))
    stats = download_data.generate_or_download(
        cfg, "http://127.0.0.1:9/unreachable.tar.gz", out
    )
    assert stats.source == "synthetic"
    assert out.is_file()
    assert stats.rows == 5_000
