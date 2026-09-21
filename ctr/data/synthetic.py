"""Deterministic synthetic dataset generator in the Criteo display-ad format.

Produces a header-less, tab-separated file with the exact schema of the Criteo
Display Advertising Challenge dataset::

    label \\t I1..I13 \\t C1..C26

* ``label``: binary click indicator with an average positive rate close to
  ``target_positive_rate`` (realistic for display ads: ~3.5%).
* ``I1..I13``: integer features (counts, device characteristics). A subset of
  columns contains missing values, written as empty fields -- mirroring the
  real dump where e.g. ``I6``/``I9``/``I10`` are frequently empty.
* ``C1..C26``: categorical features stored as 32-bit lowercase hex strings
  (e.g. ``a73ee510``). Value cardinalities follow a Zipf distribution, matching
  the heavy-tailed value distribution observed in real ad-serving logs.
  Missing values are empty fields.

Signal injection
----------------
The generator is not pure noise; it injects structure a model can learn:

1. Six integer features contribute to the log-odds of a click.
2. Five categorical features carry per-value effects; because Zipf-distributed
   values repeat heavily, the popular values' effects are learnable.
3. The base click rate drifts from ~2.8% on day 0 to ~4.2% on the last day of
   the event window. Each row's day is derived from the CRC-32 of its
   tab-joined feature string -- ``crc32(features) % days`` -- which is exactly
   the mapping :mod:`ctr.data.ingest` uses to assign ``event_ts``. The injected
   drift is therefore aligned with the time-based train/validation/test splits,
   and models trained on the train split genuinely under-forecast the test split.

Determinism
-----------
Given the same configuration (rows, seed, ...), the output file is
byte-identical across runs and machines, which keeps the pipeline reproducible
in CI without shipping data artifacts.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LABEL_COLUMN = "label"
INTEGER_FEATURES = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_FEATURES = [f"C{i}" for i in range(1, 27)]
FEATURE_COLUMNS = INTEGER_FEATURES + CATEGORICAL_FEATURES
ALL_COLUMNS = [LABEL_COLUMN] + FEATURE_COLUMNS

FIELD_SEPARATOR = "\t"

DAYS = 21

# Zipf cardinality per categorical feature (roughly in the same ballpark as
# the real Criteo dump, where C19/C20/C26 are the largest).
CATEGORICAL_CARDINALITIES: dict[str, int] = {
    "C1": 100_000,
    "C2": 50_000,
    "C3": 10_000,
    "C4": 500_000,
    "C5": 2_000_000,
    "C6": 1_000_000,
    "C7": 300_000,
    "C8": 100_000,
    "C9": 30_000,
    "C10": 5_000,
    "C11": 1_000_000,
    "C12": 800_000,
    "C13": 20_000,
    "C14": 200_000,
    "C15": 60_000,
    "C16": 100_000,
    "C17": 40_000,
    "C18": 10_000,
    "C19": 5_000_000,
    "C20": 2_000_000,
    "C21": 15_000,
    "C22": 250_000,
    "C23": 1_000_000,
    "C24": 90_000,
    "C25": 8_000,
    "C26": 3_000_000,
}

# Missing-value rates per column (empty field in the TSV), loosely matching
# the real Criteo dump where a few integer and categorical columns are
# frequently empty.
INTEGER_MISSING_RATES: dict[str, float] = {
    "I6": 0.15,
    "I9": 0.20,
    "I10": 0.20,
    "I11": 0.05,
    "I12": 0.10,
    "I13": 0.10,
}

CATEGORICAL_MISSING_RATES: dict[str, float] = {
    "C1": 0.05,
    "C2": 0.03,
    "C9": 0.08,
    "C15": 0.12,
    "C19": 0.05,
    "C26": 0.02,
}

# Columns whose values enter the click log-odds (injected signal). Weights
# are deliberately small: the total logit variance must stay ~0.6 or the
# realized per-day positive rates stop tracking the injected drift (the
# sigmoid mean becomes dominated by the noise tail).
INFORMATIVE_INTEGER_WEIGHTS: dict[str, float] = {
    "I1": 0.25,
    "I2": 0.18,
    "I3": 0.16,
    "I7": -0.12,
    "I8": 0.14,
    "I12": 0.12,
}

INFORMATIVE_CATEGORICAL: list[str] = ["C1", "C3", "C5", "C7", "C14"]
CATEGORICAL_EFFECT_STD = 0.14

ZIPF_EXPONENT = 1.3
NOISE_STD = 0.35
DRIFT_MIN_RATE = 0.028
DRIFT_MAX_RATE = 0.042

# Fixed standardization constants for log1p(count) features. The count-like
# features are generated as exp(N(2.0, 1.2)); log1p(x) is then approximately
# N(2.0, 1.2) as well.
INT_LOG1P_MEAN = 2.0
INT_LOG1P_STD = 1.2


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for :func:`generate_synthetic`."""

    rows: int = 1_000_000
    seed: int = 42
    target_positive_rate: float = 0.035
    chunk_size: int = 50_000
    days: int = DAYS
    zipf_exponent: float = ZIPF_EXPONENT
    drift_min_rate: float = DRIFT_MIN_RATE
    drift_max_rate: float = DRIFT_MAX_RATE
    noise_std: float = NOISE_STD
    output_path: str = "data/raw/train.txt"


@dataclass
class DatasetStats:
    """Summary statistics produced by dataset generation or download."""

    source: str = "synthetic"
    rows: int = 0
    positive_rate: float = 0.0
    positive_rate_by_day: dict[int, float] = field(default_factory=dict)
    output_path: str = ""


def crc32_of_features(feature_line: str) -> int:
    """Unsigned CRC-32 of a tab-joined feature string (Spark-compatible).

    Spark's ``crc32`` SQL function computes the same CRC-32 (IEEE 802.3)
    value; this mirrors it in Python so the generator and the ingest job
    agree on each row's day without coordinating.
    """
    return zlib.crc32(feature_line.encode("utf-8")) & 0xFFFFFFFF


def day_of_features(feature_line: str, days: int = DAYS) -> int:
    """0-based day index within the event window for a feature string."""
    return crc32_of_features(feature_line) % days


def desired_positive_rate(day: np.ndarray, cfg: SyntheticConfig) -> np.ndarray:
    """Base positive rate for a day index, interpolated across the window."""
    return cfg.drift_min_rate + (cfg.drift_max_rate - cfg.drift_min_rate) * (
        day.astype(np.float64) / max(cfg.days - 1, 1)
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def generate_synthetic(
    cfg: SyntheticConfig,
    out_path: str | Path | None = None,
) -> DatasetStats:
    """Generate the deterministic synthetic dataset and write it as TSV.

    The file is produced in chunks to keep memory usage flat regardless of
    ``cfg.rows``. For every row the day index is computed first
    (``crc32(features) % days``), then the label is sampled from a logistic
    model whose intercept is the drifting day-level base rate, so the drift
    is visible in the time-based splits produced by the ingest job.
    """
    out_path = Path(out_path) if out_path is not None else Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Generating %d synthetic rows (seed=%d) -> %s", cfg.rows, cfg.seed, out_path
    )

    rng = np.random.default_rng(cfg.seed)

    # Per-value effects for the informative categorical features, sampled once
    # so the same value carries the same effect everywhere.
    effect_tables = {
        col: rng.normal(0.0, CATEGORICAL_EFFECT_STD, CATEGORICAL_CARDINALITIES[col])
        for col in INFORMATIVE_CATEGORICAL
    }

    drift_line = np.linspace(cfg.drift_min_rate, cfg.drift_max_rate, cfg.days)
    per_day_pos = np.zeros(cfg.days, dtype=np.float64)
    per_day_cnt = np.zeros(cfg.days, dtype=np.int64)
    total_pos = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for start in range(0, cfg.rows, cfg.chunk_size):
            n = min(cfg.chunk_size, cfg.rows - start)
            feature_lines, labels, days = _build_chunk(
                rng, cfg, effect_tables, n, drift_line
            )
            pos = labels.astype(np.int64)
            per_day_pos += np.bincount(days, weights=pos, minlength=cfg.days)
            per_day_cnt += np.bincount(days, minlength=cfg.days)
            total_pos += int(pos.sum())
            handle.write(
                "\n".join(
                    f"{1 if label else 0}{FIELD_SEPARATOR}{line}"
                    for label, line in zip(labels, feature_lines)
                )
                + "\n"
            )

    positive_rate = total_pos / cfg.rows if cfg.rows else 0.0
    positive_rate_by_day = {
        day: (per_day_pos[day] / per_day_cnt[day] if per_day_cnt[day] else 0.0)
        for day in range(cfg.days)
    }
    logger.info(
        "Generated %d rows; overall positive rate=%.4f", cfg.rows, positive_rate
    )
    return DatasetStats(
        source="synthetic",
        rows=cfg.rows,
        positive_rate=positive_rate,
        positive_rate_by_day=positive_rate_by_day,
        output_path=str(out_path),
    )


def _build_chunk(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    effect_tables: dict[str, np.ndarray],
    n: int,
    drift_line: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Generate one chunk of rows.

    Returns the tab-joined feature strings, the sampled labels, and the day
    index of each row. The RNG draw order is fixed (integer features, then
    categorical features, then label noise) so a given config always produces
    the same bytes.
    """
    column_strings: list = []
    integer_values: dict[str, np.ndarray] = {}

    for col in INTEGER_FEATURES:
        if col in ("I4", "I5", "I6", "I9", "I10"):
            values = rng.normal(0.0, 10.0, n).round().astype(np.int64)
        else:
            values = np.exp(rng.normal(2.0, 1.2, n)).round().astype(np.int64)
        integer_values[col] = values
        rate = INTEGER_MISSING_RATES.get(col, 0.0)
        missing = rng.random(n) < rate if rate > 0 else np.zeros(n, dtype=bool)
        column_strings.append(np.where(missing, "", values.astype(str)))

    categorical_values: dict[str, np.ndarray] = {}
    for col in CATEGORICAL_FEATURES:
        cardinality = CATEGORICAL_CARDINALITIES[col]
        draws = rng.zipf(cfg.zipf_exponent, n)
        values = (draws % cardinality).astype(np.int64)
        categorical_values[col] = values
        rate = CATEGORICAL_MISSING_RATES.get(col, 0.0)
        missing = rng.random(n) < rate if rate > 0 else np.zeros(n, dtype=bool)
        column_strings.append(
            ["" if miss else f"{value:08x}" for miss, value in zip(missing, values)]
        )

    feature_lines = [FIELD_SEPARATOR.join(row) for row in zip(*column_strings)]
    days = np.fromiter(
        (day_of_features(line, cfg.days) for line in feature_lines),
        dtype=np.int64,
        count=n,
    )

    logit = np.zeros(n, dtype=np.float64)
    for col, weight in INFORMATIVE_INTEGER_WEIGHTS.items():
        x = np.log1p(integer_values[col].astype(np.float64))
        z = (x - INT_LOG1P_MEAN) / INT_LOG1P_STD
        logit += weight * z
    for col in INFORMATIVE_CATEGORICAL:
        values = categorical_values[col]
        logit += np.where(values >= 0, effect_tables[col][values], 0.0)
    logit += rng.normal(0.0, cfg.noise_std, n)
    logit += _logit(drift_line[days])

    # Calibrate an intercept so the chunk's mean probability equals the
    # target exactly. Because sigmoid is convex around p < 0.5, feature
    # variance alone would push the mean rate well above the base rates
    # (Jensen's inequality); the intercept cancels that shift while keeping
    # the day-over-day drift intact (it shifts every day in the chunk by the
    # same amount).
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _sigmoid(logit + mid).mean() > cfg.target_positive_rate:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2.0

    p = _sigmoid(logit + intercept)
    labels = rng.random(n) < p
    return feature_lines, labels, days
