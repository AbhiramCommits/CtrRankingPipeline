"""Tests for the serving layer: validation, metrics registry, /rank paths.

Builds a tiny self-contained model zoo (two-tower + FAISS + DLRM +
vocabularies) under ``tmp_path`` so the tests do not depend on trained
artifacts; the CI smoke job exercises the real pipeline end to end.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ctr.data.synthetic import CATEGORICAL_FEATURES
from ctr.models.dlrm import DLRM
from ctr.retrieval.index import (
    AD_IDS_FILENAME,
    IVF_FILENAME,
    build_ivf_index,
    save_ad_ids,
    save_index,
)
from ctr.retrieval.two_tower import TwoTower
from ctr.serving.app import (
    NUMERIC_DIM,
    MetricsRegistry,
    RankRequest,
    ServingConfig,
    ServingEngine,
    create_app,
)

N_ADS = 6


def _context_payload():
    context = {f: f"0000000{i}" for i, f in enumerate(
        ["C2", "C3", "C4", "C5", "C7", "C8", "C10", "C11", "C12", "C13",
         "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22",
         "C23", "C24", "C25", "C26"]
    )}
    return context


def make_tiny_artifacts(tmp_path) -> ServingConfig:
    """Minimal retrieval + ranker + vocab artifacts in tmp dirs."""
    retrieval_dir = tmp_path / "retrieval"
    ranker_dir = tmp_path / "ranker"
    features_dir = tmp_path / "features"
    retrieval_dir.mkdir()
    ranker_dir.mkdir()
    (features_dir / "vocab").mkdir(parents=True)

    torch.manual_seed(0)
    vocab_sizes = {field: 32 for field in CATEGORICAL_FEATURES}
    two_tower = TwoTower(
        context_fields=["C2", "C3", "C4", "C5", "C7", "C8", "C10", "C11",
                        "C12", "C13", "C14", "C15", "C16", "C17", "C18",
                        "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"],
        context_vocab_sizes=vocab_sizes,
        numeric_dim=NUMERIC_DIM,
        ad_fields=["C1", "C6", "C9"],
        ad_vocab_sizes=vocab_sizes,
        embedding_dim=8,
        tower_mlp_dims=(16,),
        output_dim=16,
    )
    torch.save(
        {
            "state_dict": two_tower.state_dict(),
            "model_config": two_tower.config(),
        },
        retrieval_dir / "model.pt",
    )

    rng = np.random.default_rng(0)
    ad_vectors = rng.normal(size=(N_ADS, 16)).astype(np.float32)
    save_index(
        build_ivf_index(ad_vectors, nlist=2, nprobe=2),
        str(retrieval_dir / IVF_FILENAME),
    )
    save_ad_ids(
        [f"{i}|{i + 3}|{i + 6}" for i in range(N_ADS)],
        str(retrieval_dir / AD_IDS_FILENAME),
    )

    dlrm = DLRM(
        field_vocab_sizes=[32] * 26, num_numeric=NUMERIC_DIM, embedding_dim=8,
        bottom_mlp_dims=(16,), top_mlp_dims=(16,),
    )
    torch.save(
        {
            "state_dict": dlrm.state_dict(),
            "model_config": dlrm.config(),
        },
        ranker_dir / "model.pt",
    )

    for field in CATEGORICAL_FEATURES:
        pd.DataFrame(
            {"value": [f"v{i}" for i in range(7)], "index": list(range(1, 8)), "count": [5] * 7}
        ).to_parquet(features_dir / "vocab" / field, index=False)
    with open(features_dir / "feature_stats.json", "w") as handle:
        json.dump({"vocab_sizes": {f: 7 for f in CATEGORICAL_FEATURES}}, handle)

    return ServingConfig(
        retrieval_artifacts=str(retrieval_dir),
        ranker_artifacts=str(ranker_dir),
        features_artifacts=str(features_dir),
        default_k=N_ADS,
        latency_budget_ms=100.0,
        degraded_candidate_limit=3,
        retrieval_budget_fraction=0.5,
    )


def test_metrics_registry_percentiles():
    registry = MetricsRegistry()
    for latency in [1.0, 2.0, 3.0, 4.0, 5.0]:
        registry.record(total_ms=latency, retrieval_ms=0.5, ranking_ms=0.5, degraded=False)
    snapshot = registry.snapshot()
    assert snapshot["request_count"] == 5
    assert snapshot["latency_ms"]["total"]["p50"] == 3.0
    assert snapshot["latency_ms"]["total"]["p99"] == 5.0
    assert snapshot["retrieval_fraction_of_total"] == pytest.approx(0.5 / 3.0)
    assert sum(snapshot["histogram_total_ms"].values()) == 5


def test_rank_request_validation():
    context = _context_payload()
    with pytest.raises(ValidationError):
        RankRequest(context_categorical={}, numeric_features=[1.0] * NUMERIC_DIM)
    with pytest.raises(ValidationError):
        RankRequest(context_categorical=context, numeric_features=[1.0] * 10)


def test_serving_endpoints(tmp_path):
    cfg = make_tiny_artifacts(tmp_path)
    client = TestClient(create_app(cfg))
    with client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["ads_in_index"] == N_ADS

        payload = {"context_categorical": _context_payload(),
                   "numeric_features": [1.0] * NUMERIC_DIM}
        response = client.post("/rank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["num_candidates"] == N_ADS
        assert len(data["ads"]) == N_ADS
        assert data["degraded"] is False
        ctrs = [ad["ctr"] for ad in data["ads"]]
        assert ctrs == sorted(ctrs, reverse=True)
        assert all(0.0 <= ctr <= 1.0 for ctr in ctrs)

        # explicit candidates skip retrieval
        explicit = client.post(
            "/rank",
            json={**payload, "ad_ids": ["0|3|6", "1|4|7"], "k": 2},
        ).json()
        assert {ad["ad_id"] for ad in explicit["ads"]} == {"0|3|6", "1|4|7"}
        assert explicit["latency_ms"]["retrieval_ms"] == 0.0

        batch = client.post(
            "/rank:batch", json={"requests": [payload, payload, payload]}
        ).json()
        assert len(batch["responses"]) == 3

        metrics = client.get("/metrics").json()
        assert metrics["request_count"] >= 4
        assert "p99" in metrics["latency_ms"]["total"]


def test_latency_budget_degrades_gracefully(tmp_path):
    cfg = make_tiny_artifacts(tmp_path)
    # Zero budget: retrieval alone exceeds it -> truncation + degraded flag.
    cfg = ServingConfig(
        retrieval_artifacts=cfg.retrieval_artifacts,
        ranker_artifacts=cfg.ranker_artifacts,
        features_artifacts=cfg.features_artifacts,
        default_k=N_ADS,
        latency_budget_ms=0.0,
        degraded_candidate_limit=3,
        retrieval_budget_fraction=0.5,
    )
    client = TestClient(create_app(cfg))
    with client:
        response = client.post(
            "/rank",
            json={"context_categorical": _context_payload(),
                  "numeric_features": [1.0] * NUMERIC_DIM},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["degraded"] is True
        assert data["num_candidates"] == 3
        metrics = client.get("/metrics").json()
        assert metrics["degraded_count"] == 1


def test_engine_rank_batch_handles_varying_candidate_lists(tmp_path):
    cfg = make_tiny_artifacts(tmp_path)
    engine = ServingEngine(cfg)
    try:
        payload = {"context_categorical": _context_payload(),
                   "numeric_features": [1.0] * NUMERIC_DIM}
        requests = [
            RankRequest(**payload),
            RankRequest(**payload, ad_ids=["0|3|6", "1|4|7"], k=2),
        ]
        responses = engine.rank_batch(requests)
        assert [len(response.ads) for response in responses] == [N_ADS, 2]
    finally:
        engine.close()
