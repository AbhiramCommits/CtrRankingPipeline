"""FAISS index construction, persistence, search and benchmarking.

After the two-tower is trained, every unique ad (``C1|C6|C9`` combination)
is embedded with the ad tower and indexed:

* ``IndexFlatIP``: exact inner-product baseline,
* ``IndexIVFFlat``: approximate search with configurable ``nlist`` /
  ``nprobe`` -- the speed/quality tradeoff is quantified by benchmarking
  recall@k of the IVF index against exact flat search plus per-query
  latency.

The index and the id mapping are persisted under ``artifacts/faiss/`` so
serving can load them without retraining.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

# torch and faiss each bundle their own copy of libomp on macOS; loading both
# runtimes into one process aborts OpenMP unless duplicates are explicitly
# allowed. Must be set before faiss (or torch) is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch
from torch import nn

# The faiss-cpu wheel bundles its own libomp; when torch has already loaded a
# different OpenMP runtime, spawning a parallel team (e.g. kmeans training)
# can segfault on macOS. Single-threaded faiss avoids the clash entirely and
# is plenty fast at the catalog sizes this pipeline targets.
faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)

FLAT_FILENAME = "index_flat.faiss"
IVF_FILENAME = "index_ivf.faiss"
AD_IDS_FILENAME = "ad_ids.json"


@dataclass(frozen=True)
class FaissConfig:
    """Index construction/search parameters."""

    nlist: int = 256
    nprobe: int = 8
    dimension: int = 64


def build_flat_index(vectors: np.ndarray) -> faiss.Index:
    """Exact inner-product index over the given row vectors."""
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_ivf_index(
    vectors: np.ndarray, nlist: int, nprobe: int
) -> faiss.Index:
    """Approximate IVF inner-product index (trained on the vectors)."""
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    dimension = vectors.shape[1]
    n = vectors.shape[0]
    if n == 0:
        raise ValueError("cannot build an index from zero vectors")
    nlist = max(1, min(int(nlist), n))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = max(1, min(int(nprobe), nlist))
    return index


def save_index(index: faiss.Index, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(index, path)
    return path


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def save_ad_ids(ad_ids: Sequence[str], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(ad_ids), handle)
    return path


def load_ad_ids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def search(
    index: faiss.Index, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, scores) for the top-k nearest vectors per query."""
    queries = np.ascontiguousarray(queries.astype(np.float32))
    scores, ids = index.search(queries, k)
    return ids, scores


def benchmark_recall(
    flat: faiss.Index, ivf: faiss.Index, queries: np.ndarray, k: int = 100
) -> dict[str, float]:
    """Recall@k of the IVF index vs exact flat search + per-query latency."""
    queries = np.ascontiguousarray(queries.astype(np.float32))
    start = time.perf_counter()
    flat_ids, _ = search(flat, queries, k)
    flat_ms = (time.perf_counter() - start) / len(queries) * 1000.0

    start = time.perf_counter()
    ivf_ids, _ = search(ivf, queries, k)
    ivf_ms = (time.perf_counter() - start) / len(queries) * 1000.0

    recall = float(
        np.mean(
            [
                len(set(flat_ids[i].tolist()) & set(ivf_ids[i].tolist())) / k
                for i in range(len(queries))
            ]
        )
    )
    return {
        "k": float(k),
        "recall_at_k": recall,
        "flat_ms_per_query": flat_ms,
        "ivf_ms_per_query": ivf_ms,
    }


class RetrievalIndex:
    """End-to-end retrieval: two-tower + FAISS index + ad id mapping."""

    def __init__(self, model: nn.Module, index: faiss.Index, ad_ids: Sequence[str]):
        self.model = model
        self.index = index
        self.ad_ids = list(ad_ids)

    def retrieve(
        self,
        context_features: tuple[torch.Tensor, torch.Tensor],
        k: int,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Return top-k ad ids and scores for the given context features.

        Args:
            context_features: ``(context_cats, context_numeric)`` tensors,
                the same inputs the context tower was trained on.
            k: number of ads to return per query.
        """
        context_cats, context_numeric = context_features
        vectors = self.model.embed_context(
            torch.as_tensor(context_cats, dtype=torch.long),
            torch.as_tensor(context_numeric, dtype=torch.float32),
        ).cpu().numpy()
        ids, scores = search(self.index, vectors, k)
        return (
            [[self.ad_ids[i] for i in row] for row in ids],
            [row.tolist() for row in scores],
        )

    @classmethod
    def from_artifacts(
        cls, model: nn.Module, artifacts_dir: str, use_ivf: bool = True
    ) -> RetrievalIndex:
        """Load the persisted index + id mapping (IVF by default, flat on request)."""
        filename = IVF_FILENAME if use_ivf else FLAT_FILENAME
        index = load_index(os.path.join(artifacts_dir, filename))
        ad_ids = load_ad_ids(os.path.join(artifacts_dir, AD_IDS_FILENAME))
        return cls(model, index, ad_ids)
