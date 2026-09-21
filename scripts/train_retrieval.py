#!/usr/bin/env python3
"""Train the two-tower retriever, build the FAISS index, benchmark recall.

1. Trains the two-tower on positive pairs with in-batch negatives, logging
   in-batch recall@k on the validation positives.
2. Embeds every unique ad from the train split and builds an exact
   ``IndexFlatIP`` plus an approximate ``IndexIVFFlat``, persisted with the
   id mapping under ``artifacts/retrieval/`` (and ``artifacts/faiss/`` via
   the shared filenames).
3. Benchmarks recall@k of the IVF index against exact flat search and
   reports per-query latency for the speed/quality tradeoff.

Usage::

    python scripts/train_retrieval.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Must run before torch/faiss load their bundled libomp runtimes (macOS
# duplicate-OpenMP abort); see ctr/retrieval/index.py for details.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ctr.config import load_yaml
from ctr.data.ingest import build_spark_session
from ctr.data.synthetic import CATEGORICAL_FEATURES
from ctr.features.transforms import load_feature_stats
from ctr.models.data import load_split_arrays
from ctr.retrieval.index import (
    AD_IDS_FILENAME,
    FLAT_FILENAME,
    IVF_FILENAME,
    FaissConfig,
    RetrievalIndex,
    benchmark_recall,
    build_flat_index,
    build_ivf_index,
    save_ad_ids,
    save_index,
)
from ctr.retrieval.two_tower import (
    AD_IDENTITY_FIELDS,
    CONTEXT_FIELDS,
    TwoTower,
    TwoTowerConfig,
    train_two_tower,
)

logger = logging.getLogger("ctr.retrieval")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/retrieval.yaml", help="YAML config path"
    )
    return parser.parse_args(argv)


def _field_positions(fields):
    """Positions of `fields` within CATEGORICAL_FEATURES (column order of tensors)."""
    order = {field: i for i, field in enumerate(CATEGORICAL_FEATURES)}
    return [order[f] for f in fields]


def _to_dataloader_tensors(split, fields):
    positions = _field_positions(fields)
    cats = torch.from_numpy(split["cats"][:, positions]).long()
    numeric = torch.from_numpy(split["numeric"]).float()
    return cats, numeric


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    config = load_yaml(args.config)
    section = config.get("retrieval") or {}
    model_conf = section.get("model") or {}
    training_conf = section.get("training") or {}
    faiss_conf = section.get("faiss") or {}

    features_root = str(section.get("features_root", "data/features"))
    artifacts_dir = str(section.get("artifacts_dir", "artifacts/retrieval"))
    feature_artifacts = str(section.get("features_artifacts_dir", "artifacts"))
    seed = int(section.get("seed", 42))

    cfg = TwoTowerConfig(
        embedding_dim=int(model_conf.get("embedding_dim", 32)),
        tower_mlp_dims=tuple(model_conf.get("tower_mlp_dims", [256, 128])),
        output_dim=int(model_conf.get("output_dim", 64)),
        lr=float(training_conf.get("lr", 0.001)),
        epochs=int(training_conf.get("epochs", 10)),
        batch_size=int(training_conf.get("batch_size", 1024)),
        eval_batch_size=int(training_conf.get("eval_batch_size", 2048)),
        recall_ks=tuple(training_conf.get("recall_ks", [1, 10, 50])),
        seed=seed,
        device=str(training_conf.get("device", "auto")),
    )
    faiss_cfg = FaissConfig(
        nlist=int(faiss_conf.get("nlist", 256)),
        nprobe=int(faiss_conf.get("nprobe", 8)),
        dimension=cfg.output_dim,
    )

    spark = build_spark_session(config)
    try:
        spark.sparkContext.setLogLevel("WARN")
        stats = load_feature_stats(feature_artifacts)
        vocab_sizes = {
            field: int(stats["vocab_sizes"][field]) + 1
            for field in CATEGORICAL_FEATURES
        }

        logger.info("Loading positive pairs from train/validation")
        train_split = load_split_arrays(
            spark,
            os.path.join(features_root, "train"),
            categorical_fields=CATEGORICAL_FEATURES,
            positives_only=True,
        )
        val_split = load_split_arrays(
            spark,
            os.path.join(features_root, "validation"),
            categorical_fields=CATEGORICAL_FEATURES,
            positives_only=True,
        )
        logger.info("train positives=%d, validation positives=%d",
                    len(train_split["labels"]), len(val_split["labels"]))

        ctx_cats_train, ctx_numeric_train = _to_dataloader_tensors(
            train_split, CONTEXT_FIELDS
        )
        ad_cats_train, _ = _to_dataloader_tensors(train_split, AD_IDENTITY_FIELDS)
        ctx_cats_val, ctx_numeric_val = _to_dataloader_tensors(val_split, CONTEXT_FIELDS)
        ad_cats_val, _ = _to_dataloader_tensors(val_split, AD_IDENTITY_FIELDS)

        numeric_dim = train_split["numeric"].shape[1]
        model = TwoTower(
            context_fields=CONTEXT_FIELDS,
            context_vocab_sizes={f: vocab_sizes[f] for f in CONTEXT_FIELDS},
            numeric_dim=numeric_dim,
            ad_fields=AD_IDENTITY_FIELDS,
            ad_vocab_sizes={f: vocab_sizes[f] for f in AD_IDENTITY_FIELDS},
            embedding_dim=cfg.embedding_dim,
            tower_mlp_dims=cfg.tower_mlp_dims,
            output_dim=cfg.output_dim,
        )
        train_ds = TensorDataset(ctx_cats_train, ctx_numeric_train, ad_cats_train)
        val_ds = TensorDataset(ctx_cats_val, ctx_numeric_val, ad_cats_val)

        logger.info("Training two-tower (emb=%d, out=%d)", cfg.embedding_dim, cfg.output_dim)
        train_two_tower(model, train_ds, val_ds, cfg, artifacts_dir)

        # Embed the catalog of unique ads seen in the train split.
        logger.info("Embedding the ad catalog (C1|C6|C9) from the train split")
        catalog_df = (
            spark.read.parquet(os.path.join(features_root, "train"))
            .select(*[f"{f}_idx" for f in AD_IDENTITY_FIELDS])
            .distinct()
            .toPandas()
        )
        ad_cat_ids = torch.from_numpy(
            catalog_df[[f"{f}_idx" for f in AD_IDENTITY_FIELDS]].to_numpy(
                dtype=np.int64
            )
        )
        ad_ids = [
            f"{row[0]}|{row[1]}|{row[2]}" for row in ad_cat_ids.tolist()
        ]
        ad_vectors = model.embed_ad(ad_cat_ids).cpu().numpy()
        logger.info("catalog size=%d, embedding dim=%d", len(ad_ids), ad_vectors.shape[1])

        logger.info("Building FAISS indices (nlist=%d, nprobe=%d)",
                    faiss_cfg.nlist, faiss_cfg.nprobe)
        flat = build_flat_index(ad_vectors)
        ivf = build_ivf_index(ad_vectors, faiss_cfg.nlist, faiss_cfg.nprobe)
        save_index(flat, os.path.join(artifacts_dir, FLAT_FILENAME))
        save_index(ivf, os.path.join(artifacts_dir, IVF_FILENAME))
        save_ad_ids(ad_ids, os.path.join(artifacts_dir, AD_IDS_FILENAME))

        # Benchmark: recall@k of IVF vs exact flat + per-query latency.
        n_queries = int(faiss_conf.get("benchmark_queries", 1000))
        recall_k = int(faiss_conf.get("recall_k", 100))
        query_idx = np.random.default_rng(seed).choice(
            len(val_split["labels"]), size=min(n_queries, len(val_split["labels"])),
            replace=False,
        )
        query_vectors = model.embed_context(
            ctx_cats_val[query_idx], ctx_numeric_val[query_idx]
        ).cpu().numpy()
        report = benchmark_recall(flat, ivf, query_vectors, k=recall_k)
        logger.info(
            "FAISS benchmark: recall@%d=%.4f | flat=%.3f ms/query | ivf(nprobe=%d)=%.3f ms/query",
            recall_k,
            report["recall_at_k"],
            report["flat_ms_per_query"],
            faiss_cfg.nprobe,
            report["ivf_ms_per_query"],
        )
        with open(os.path.join(artifacts_dir, "benchmark.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        # Demonstrate end-to-end retrieval.
        service = RetrievalIndex.from_artifacts(model, artifacts_dir, use_ivf=True)
        demo_ids, demo_scores = service.retrieve(
            (ctx_cats_val[:3], ctx_numeric_val[:3]), k=3
        )
        for i, (ids, scores) in enumerate(zip(demo_ids, demo_scores)):
            logger.info("query %d -> ads=%s scores=%s", i, ids, [round(s, 4) for s in scores])
    finally:
        spark.stop()
    logger.info("Retrieval stage finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
