"""Unit tests for the FAISS index: build, search, persistence, benchmark."""

import numpy as np
import torch
from torch import nn

from ctr.retrieval.index import (
    RetrievalIndex,
    benchmark_recall,
    build_flat_index,
    build_ivf_index,
    load_ad_ids,
    load_index,
    save_ad_ids,
    save_index,
    search,
)


class _IdentityModel(nn.Module):
    """Stand-in whose embed_context returns the numeric input as the vector."""

    def embed_context(self, cats, numeric):
        return numeric


def test_flat_index_returns_exact_top_k():
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    index = build_flat_index(vectors)
    ids, scores = search(index, vectors[[0]], k=2)
    assert list(ids[0]) == [0, 2]
    assert scores[0][0] > scores[0][1]


def test_ivf_recall_against_flat_and_benchmark():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(64, 16)).astype(np.float32)
    queries = rng.normal(size=(16, 16)).astype(np.float32)
    flat = build_flat_index(vectors)
    ivf = build_ivf_index(vectors, nlist=4, nprobe=4)
    report = benchmark_recall(flat, ivf, queries, k=5)
    # nprobe == nlist visits every cell, so IVF matches exact search.
    assert report["recall_at_k"] == 1.0
    assert report["flat_ms_per_query"] > 0
    assert report["ivf_ms_per_query"] > 0


def test_index_persistence_and_retrieve_roundtrip(tmp_path):
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    ivf = build_ivf_index(vectors, nlist=2, nprobe=2)
    save_index(ivf, str(tmp_path / "index_ivf.faiss"))
    loaded = load_index(str(tmp_path / "index_ivf.faiss"))

    save_ad_ids(["a0", "a1", "a2", "a3", "a4", "a5"], str(tmp_path / "ad_ids.json"))
    assert load_ad_ids(str(tmp_path / "ad_ids.json")) == [
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
    ]

    service = RetrievalIndex(_IdentityModel(), loaded, load_ad_ids(str(tmp_path / "ad_ids.json")))
    # Inner-product search over linearly growing vectors: the largest vector
    # is its own nearest neighbor.
    top_ids, scores = service.retrieve((torch.empty(0), vectors[[5]]), k=2)
    assert top_ids[0][0] == "a5"
    assert len(scores[0]) == 2
    assert scores[0][0] >= scores[0][1]
