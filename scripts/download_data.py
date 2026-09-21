#!/usr/bin/env python3
"""Download (or synthesize) the raw dataset in Criteo format.

Source-of-truth priority:

1. ``--url`` CLI flag
2. ``CRITEO_SAMPLE_URL`` environment variable
3. ``dataset.url`` in ``configs/data.yaml``
4. (nothing) -> deterministic synthetic generation

If a URL is configured but the download or validation fails (e.g. no network
in CI), the script logs a warning and falls back to the synthetic dataset, so
``data/raw/train.txt`` is always produced and the rest of the pipeline runs
offline.

Usage::

    python scripts/download_data.py --config configs/data.yaml [--rows N]
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctr.config import load_yaml
from ctr.data.synthetic import (
    ALL_COLUMNS,
    FIELD_SEPARATOR,
    DatasetStats,
    SyntheticConfig,
    generate_synthetic,
)

logger = logging.getLogger(__name__)

MAX_VALIDATION_LINES = 1_000
MAX_RATE_SAMPLE_LINES = 200_000


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="YAML config path")
    parser.add_argument("--url", default=None, help="override the Criteo sample archive URL")
    parser.add_argument("--rows", type=int, default=None, help="override dataset.rows")
    parser.add_argument("--seed", type=int, default=None, help="override dataset.seed")
    parser.add_argument(
        "--target-rate", type=float, default=None, help="override dataset.target_positive_rate"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="override dataset.chunk_size"
    )
    parser.add_argument("--output", default=None, help="override dataset.output_path")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> SyntheticConfig:
    config = load_yaml(args.config)
    dataset = config.get("dataset") or {}
    defaults = SyntheticConfig()
    cfg = SyntheticConfig(
        rows=int(args.rows if args.rows is not None else dataset.get("rows", defaults.rows)),
        seed=int(args.seed if args.seed is not None else dataset.get("seed", defaults.seed)),
        target_positive_rate=float(
            args.target_rate
            if args.target_rate is not None
            else dataset.get("target_positive_rate", defaults.target_positive_rate)
        ),
        chunk_size=int(
            args.chunk_size
            if args.chunk_size is not None
            else dataset.get("chunk_size", defaults.chunk_size)
        ),
        days=int(dataset.get("days", defaults.days)),
        output_path=str(args.output or dataset.get("output_path", defaults.output_path)),
    )
    return cfg


def download_and_extract(url: str, dest: Path) -> Path | None:
    """Download a ``.tar.gz`` archive and return the first ``.txt`` member."""
    archive = dest / "criteo_sample.tar.gz"
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".txt"):
                try:
                    tar.extract(member, dest, filter="data")
                except TypeError:  # filter= argument needs Python >= 3.11.4
                    tar.extract(member, dest)
                return dest / member.name
    return None


def looks_like_criteo(path: Path, max_lines: int = MAX_VALIDATION_LINES) -> bool:
    """Cheap sanity check that a file matches the Criteo TSV schema."""
    expected_fields = len(ALL_COLUMNS)
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= max_lines:
                break
            fields = line.rstrip("\n").split(FIELD_SEPARATOR)
            if len(fields) != expected_fields:
                return False
            if fields[0] not in ("0", "1"):
                return False
            seen += 1
    return seen > 0


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _sample_positive_rate(path: Path, max_lines: int = MAX_RATE_SAMPLE_LINES) -> float:
    positive = total = 0
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= max_lines:
                break
            positive += line[0] == "1"
            total += 1
    return positive / total if total else 0.0


def generate_or_download(
    cfg: SyntheticConfig,
    url: str | None,
    out_path: Path,
) -> DatasetStats:
    """Produce ``out_path`` from the URL, falling back to synthetic generation.

    Any failure to download or validate the archive triggers the fallback so
    the pipeline never blocks on the network.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if url:
        try:
            downloaded = download_and_extract(url, out_path.parent)
            if downloaded is not None and looks_like_criteo(downloaded):
                os.replace(downloaded, out_path)
                stats = DatasetStats(
                    source="downloaded",
                    rows=_count_lines(out_path),
                    positive_rate=_sample_positive_rate(out_path),
                    output_path=str(out_path),
                )
                logger.info(
                    "Downloaded %d rows from %s; sampled positive rate=%.4f",
                    stats.rows,
                    url,
                    stats.positive_rate,
                )
                return stats
            logger.warning("Downloaded archive does not match the Criteo schema")
        except Exception as exc:  # noqa: BLE001 - any network error triggers fallback
            logger.warning("Download from %s failed: %s", url, exc)
        logger.warning("Falling back to the deterministic synthetic dataset")
    return generate_synthetic(cfg, out_path)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    cfg = _config_from_args(args)
    url = args.url or os.environ.get("CRITEO_SAMPLE_URL") or _url_from_config(args.config)
    stats = generate_or_download(cfg, url, Path(cfg.output_path))
    logger.info(
        "Dataset ready: source=%s rows=%d positive_rate=%.4f path=%s",
        stats.source,
        stats.rows,
        stats.positive_rate,
        stats.output_path,
    )
    return 0


def _url_from_config(config_path: str) -> str | None:
    dataset = (load_yaml(config_path).get("dataset") or {})
    url = dataset.get("url")
    return str(url) if url else None


if __name__ == "__main__":
    sys.exit(main())
