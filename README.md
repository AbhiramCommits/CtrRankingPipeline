# CtrRankingPipeline

End-to-end ads click-through-rate (CTR) prediction and ranking system, modeled on how
production ad-serving stacks are built and measured:

```
data → features → models → retrieval → evaluation → serving
```

## Status

- [x] Step 1 — repo skeleton, dataset download/generation, Spark ingestion with time-based splits
- [x] Step 2 — feature engineering (`ctr/features`): train-fit transforms (log1p + clip + median impute, OOV vocabularies) and point-in-time rolling aggregates
- [ ] Step 3 — model training (`ctr/models`)
- [ ] Step 4 — retrieval (`ctr/retrieval`)
- [ ] Step 5 — evaluation (`ctr/eval`)
- [ ] Step 6 — serving (`ctr/serving`)

## Layout

```
ctr/            main package
  data/         dataset generation + Spark ingestion
  features/     train-fit transforms (log1p/clip/impute, OOV vocabularies) + point-in-time aggregates
  models/       CTR models (TODO)
  retrieval/    ad retrieval / candidate ranking (TODO)
  eval/         offline metrics (TODO)
  serving/      online serving API (TODO)
configs/        YAML configs (spark.yaml, data.yaml)
scripts/        CLI entrypoints
tests/          pytest suite
notebooks/      exploration notebooks
```

## Quickstart

```bash
make install   # uv (or venv + pip) editable install
make data      # produce data/parquet/{train,validation,test} from data/raw/train.txt
make features  # produce data/features/{split} + fitted artifacts (train-fit, leakage-free)
make test      # pytest with coverage
make lint      # ruff (falls back to a syntax check)
```

## Data

The pipeline consumes the Criteo Display Advertising Challenge format: a header-less,
tab-separated file with `label`, integer features `I1..I13`, and categorical hash
features `C1..C26`.

`scripts/download_data.py` attempts to download the public Criteo sample archive
(`dac_sample.tar.gz`) when a URL is configured (`--url` flag, `CRITEO_SAMPLE_URL` env
var, or `dataset.url` in `configs/data.yaml`). Otherwise — or whenever the download
fails, e.g. offline CI — it generates a deterministic synthetic dataset with the exact
same schema: a realistic ~3.5% positive rate, Zipf-distributed categorical
cardinalities, per-column missing values, and injected feature-label correlation plus
day-over-day drift so downstream models actually have signal to learn. `data/raw/train.txt`
is always produced, so the whole pipeline runs offline.

## Why time-based splits?

The ingest job assigns each row a deterministic `event_ts` spread over 21 days and
splits by time (first 17 days = train, next 2 = validation, last 2 = test) rather than
randomly, because:

1. **Label leakage** — ad logs contain repeated users, publishers and campaigns. Random
   splits let rows for the same entity land in both train and test, so models that
   memorize per-entity CTRs score unrealistically well offline and collapse online.
2. **Temporal drift** — user behavior, creatives and auction dynamics change over time.
   Training on data that overlaps the evaluation window silently leaks that drift and
   overstates metrics; production models always predict the future.
3. **Retraining realism** — online systems train on trailing logs, so the offline
   analogue must be a contiguous time cut: train on the past, validate and test on
   strictly later windows.
