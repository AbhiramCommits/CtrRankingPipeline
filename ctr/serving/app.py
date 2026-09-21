"""FastAPI serving for the two-stage CTR pipeline.

Startup (lifespan) loads once, with gradient mode disabled and all models
in eval mode:

* the two-tower retriever, the FAISS index and the ad id mapping,
* the per-field categorical vocabularies and the fitted feature stats,
* the DLRM ranking checkpoint.

Endpoints:

* ``POST /rank`` — context + optional explicit candidate ad ids; runs FAISS
  retrieval (top-k, default 200) when candidates are omitted, then scores
  the whole shortlist with the ranker in a single forward pass and returns
  ads sorted by predicted CTR.
* ``POST /rank:batch`` — coalesces many requests into one padded tensor
  batch for throughput.
* ``GET /healthz`` / ``GET /metrics`` — liveness and request-count /
  p50/p95/p99 latency histograms with the retrieval-vs-ranking split.

Graceful degradation: a configurable latency budget is enforced -- when
retrieval alone consumes too much of it the candidate set is truncated and
``degraded: true`` is returned, the standard ads-serving pattern for
staying within the deadline at lower quality instead of blowing it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

# Same libomp guard as the training scripts (torch + faiss in one process).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ctr.data.synthetic import CATEGORICAL_FEATURES, INTEGER_FEATURES
from ctr.models.dlrm import DLRM
from ctr.retrieval.index import RetrievalIndex
from ctr.retrieval.two_tower import AD_IDENTITY_FIELDS, CONTEXT_FIELDS, TwoTower

logger = logging.getLogger(__name__)

NUMERIC_DIM = len(INTEGER_FEATURES) + 8  # 13 integer transforms + 8 PIT features
_LATENCY_BUCKET_EDGES_MS = [1, 2, 5, 10, 20, 50, 100]


@dataclass(frozen=True)
class ServingConfig:
    """Serving settings (see ``configs/serving.yaml``)."""

    retrieval_artifacts: str = "artifacts/retrieval"
    ranker_artifacts: str = "artifacts/ranker"
    features_artifacts: str = "artifacts"
    default_k: int = 200
    latency_budget_ms: float = 20.0
    degraded_candidate_limit: int = 100
    retrieval_budget_fraction: float = 0.5


def load_serving_config(path: str | None = None) -> ServingConfig:
    """Build the serving config from ``configs/serving.yaml`` (or a path)."""
    path = path or os.environ.get("SERVING_CONFIG", "configs/serving.yaml")
    section: dict[str, Any] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            import yaml

            section = (yaml.safe_load(handle) or {}).get("serving") or {}
    return ServingConfig(
        retrieval_artifacts=str(
            section.get("retrieval_artifacts", "artifacts/retrieval")
        ),
        ranker_artifacts=str(section.get("ranker_artifacts", "artifacts/ranker")),
        features_artifacts=str(section.get("features_artifacts", "artifacts")),
        default_k=int(section.get("default_k", 200)),
        latency_budget_ms=float(section.get("latency_budget_ms", 20.0)),
        degraded_candidate_limit=int(section.get("degraded_candidate_limit", 100)),
        retrieval_budget_fraction=float(
            section.get("retrieval_budget_fraction", 0.5)
        ),
    )


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------


class RankRequest(BaseModel):
    context_categorical: dict[str, str | None]
    numeric_features: list[float] = Field(min_length=NUMERIC_DIM, max_length=NUMERIC_DIM)
    ad_ids: list[str] | None = None
    k: int | None = Field(default=None, ge=1, le=1000)

    @field_validator("context_categorical")
    @classmethod
    def _validate_context_fields(cls, value: dict[str, str | None]):
        missing = [f for f in CONTEXT_FIELDS if f not in value]
        extra = [f for f in value if f not in CONTEXT_FIELDS]
        if missing or extra:
            raise ValueError(
                f"context_categorical must contain exactly {CONTEXT_FIELDS}; "
                f"missing={missing}, unexpected={extra}"
            )
        return value


class AdScore(BaseModel):
    ad_id: str
    ctr: float
    score: float


class LatencyBreakdown(BaseModel):
    total_ms: float
    retrieval_ms: float
    ranking_ms: float


class RankResponse(BaseModel):
    ads: list[AdScore]
    degraded: bool
    num_candidates: int
    latency_ms: LatencyBreakdown


class BatchRankRequest(BaseModel):
    requests: list[RankRequest] = Field(min_length=1, max_length=512)


class BatchRankResponse(BaseModel):
    responses: list[RankResponse]


# --------------------------------------------------------------------------
# Metrics registry
# --------------------------------------------------------------------------


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round(p / 100.0 * (len(ordered) - 1)), len(ordered) - 1)
    return float(ordered[index])


class MetricsRegistry:
    """Request counts, latency percentiles and a fixed-bucket histogram."""

    def __init__(self, maxlen: int = 4096):
        self._lock = threading.Lock()
        self.request_count = 0
        self.degraded_count = 0
        self._total = deque(maxlen=maxlen)
        self._retrieval = deque(maxlen=maxlen)
        self._ranking = deque(maxlen=maxlen)
        self._histogram: dict[str, int] = {f"<{_LATENCY_BUCKET_EDGES_MS[0]}ms": 0}
        for lo, hi in pairwise(_LATENCY_BUCKET_EDGES_MS):
            self._histogram[f"{lo}-{hi}ms"] = 0
        self._histogram[f">={_LATENCY_BUCKET_EDGES_MS[-1]}ms"] = 0

    def record(self, total_ms: float, retrieval_ms: float, ranking_ms: float, degraded: bool) -> None:
        with self._lock:
            self.request_count += 1
            self.degraded_count += degraded
            self._total.append(total_ms)
            self._retrieval.append(retrieval_ms)
            self._ranking.append(ranking_ms)
            for lo, hi in pairwise(_LATENCY_BUCKET_EDGES_MS):
                if lo <= total_ms < hi:
                    self._histogram[f"{lo}-{hi}ms"] += 1
                    break
            else:
                if total_ms < _LATENCY_BUCKET_EDGES_MS[0]:
                    self._histogram[f"<{_LATENCY_BUCKET_EDGES_MS[0]}ms"] += 1
                else:
                    self._histogram[f">={_LATENCY_BUCKET_EDGES_MS[-1]}ms"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            series = {
                "total": list(self._total),
                "retrieval": list(self._retrieval),
                "ranking": list(self._ranking),
            }
            return {
                "request_count": self.request_count,
                "degraded_count": self.degraded_count,
                "latency_ms": {
                    name: {
                        "p50": _percentile(values, 50),
                        "p95": _percentile(values, 95),
                        "p99": _percentile(values, 99),
                        "mean": float(np.mean(values)) if values else 0.0,
                    }
                    for name, values in series.items()
                },
                "histogram_total_ms": dict(self._histogram),
                "retrieval_fraction_of_total": float(
                    np.mean(self._retrieval) / (np.mean(self._total) + 1e-9)
                    if self._total
                    else 0.0
                ),
            }


# --------------------------------------------------------------------------
# Serving engine
# --------------------------------------------------------------------------


class ServingEngine:
    """Holds the loaded models/artifacts and executes ranking requests."""

    def __init__(self, cfg: ServingConfig):
        self.cfg = cfg
        # Inference only: disable autograd globally for the lifetime of the
        # engine (restored by close()) and keep the models in eval mode.
        self._previous_grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        self.two_tower = TwoTower.from_checkpoint(
            os.path.join(cfg.retrieval_artifacts, "model.pt"), device="cpu"
        )
        self.retrieval = RetrievalIndex.from_artifacts(
            self.two_tower, cfg.retrieval_artifacts, use_ivf=True
        )
        self.dlrm = DLRM.from_checkpoint(
            os.path.join(cfg.ranker_artifacts, "model.pt"), device="cpu"
        )

        self.vocabs = {
            field: self._load_vocab(field) for field in CATEGORICAL_FEATURES
        }
        with open(
            os.path.join(cfg.features_artifacts, "feature_stats.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            self.feature_stats = json.load(handle)

        self._context_positions = [
            CATEGORICAL_FEATURES.index(f) for f in CONTEXT_FIELDS
        ]
        self._ad_positions = [CATEGORICAL_FEATURES.index(f) for f in AD_IDENTITY_FIELDS]
        self._context_columns = [f"{f}_idx" for f in CONTEXT_FIELDS]
        self.metrics = MetricsRegistry()
        logger.info(
            "Serving engine loaded: %d ads in index, %d vocabularies, DLRM ready",
            len(self.retrieval.ad_ids),
            len(self.vocabs),
        )

    def close(self) -> None:
        """Restore the global autograd mode captured at construction."""
        torch.set_grad_enabled(self._previous_grad_mode)

    def _load_vocab(self, field: str) -> dict[str, int]:
        path = os.path.join(self.cfg.features_artifacts, "vocab", field)
        frame = pd.read_parquet(path)
        return {
            str(value): int(index)
            for value, index in zip(frame["value"], frame["index"])
        }

    def encode_context(self, request: RankRequest):
        """Map a request to (context_cats [23], numeric [21]) tensors."""
        indices = [
            self.vocabs[field].get(str(request.context_categorical[field]), 0)
            if request.context_categorical[field] is not None
            else 0
            for field in CONTEXT_FIELDS
        ]
        cats = torch.tensor([indices], dtype=torch.long)
        numeric = torch.tensor(
            [request.numeric_features], dtype=torch.float32
        )
        return cats, numeric

    def _retrieve(self, context_cats: torch.Tensor, context_numeric: torch.Tensor, k: int):
        ids, scores = self.retrieval.retrieve((context_cats, context_numeric), k)
        return ids[0], scores[0]

    def _build_candidate_rows(
        self,
        context_indices: list[int],
        numeric: np.ndarray,
        ad_ids: Sequence[str],
    ):
        """Full (26 categorical + 21 numeric) feature rows per candidate ad.

        The request's numeric block (including the point-in-time
        aggregates) is reused for every candidate -- a documented demo
        simplification: a production feature store would compute
        pair-specific PIT features per candidate at serving time.
        """
        base = np.zeros(len(CATEGORICAL_FEATURES), dtype=np.int64)
        for position, index in zip(self._context_positions, context_indices):
            base[position] = index
        rows = np.tile(base, (len(ad_ids), 1))
        for row, ad_id in zip(rows, ad_ids):
            parts = ad_id.split("|")
            for position, value in zip(self._ad_positions, parts):
                row[position] = int(value)
        numeric_rows = np.tile(numeric, (len(ad_ids), 1))
        return torch.from_numpy(rows), torch.from_numpy(numeric_rows)

    def rank(self, request: RankRequest) -> RankResponse:
        start = time.perf_counter()
        context_cats, context_numeric = self.encode_context(request)
        context_indices = context_cats[0].tolist()
        numeric = context_numeric[0].numpy()

        k = request.k or self.cfg.default_k
        retrieval_ms = 0.0
        if request.ad_ids:
            candidate_ids = list(request.ad_ids)[:k]
        else:
            t0 = time.perf_counter()
            candidate_ids, _ = self._retrieve(context_cats, context_numeric, k)
            retrieval_ms = (time.perf_counter() - t0) * 1000.0

        degraded = False
        budget = self.cfg.latency_budget_ms
        if retrieval_ms > budget * self.cfg.retrieval_budget_fraction:
            candidate_ids = candidate_ids[: self.cfg.degraded_candidate_limit]
            degraded = True

        t1 = time.perf_counter()
        cat_rows, numeric_rows = self._build_candidate_rows(
            context_indices, numeric, candidate_ids
        )
        logits = self.dlrm(cat_rows, numeric_rows)
        probs = torch.sigmoid(logits).numpy()
        ranking_ms = (time.perf_counter() - t1) * 1000.0

        total_ms = (time.perf_counter() - start) * 1000.0
        if total_ms > budget:
            degraded = True

        order = np.argsort(-probs)
        ads = [
            AdScore(
                ad_id=candidate_ids[i],
                ctr=float(probs[i]),
                score=float(probs[i]),
            )
            for i in order
        ]
        self.metrics.record(total_ms, retrieval_ms, ranking_ms, degraded)
        return RankResponse(
            ads=ads,
            degraded=degraded,
            num_candidates=len(ads),
            latency_ms=LatencyBreakdown(
                total_ms=total_ms, retrieval_ms=retrieval_ms, ranking_ms=ranking_ms
            ),
        )

    def rank_batch(self, requests: Sequence[RankRequest]) -> list[RankResponse]:
        """Coalesce requests into one padded tensor batch.

        Contexts are encoded together (one retrieval embed + one FAISS
        search), and all candidate feature rows are scored in a single
        ranker forward pass; per-request candidate lists are padded to the
        longest one with a mask.
        """
        start = time.perf_counter()
        context_tensors = [self.encode_context(request) for request in requests]
        context_cats = torch.cat([cats for cats, _ in context_tensors], dim=0)
        context_numeric = torch.cat(
            [numeric for _, numeric in context_tensors], dim=0
        )

        t0 = time.perf_counter()
        retrieval_candidates = []
        for i, request in enumerate(requests):
            if request.ad_ids:
                retrieval_candidates.append(list(request.ad_ids))
            else:
                k = request.k or self.cfg.default_k
                ids, _ = self.retrieval.retrieve(
                    (context_cats[i : i + 1], context_numeric[i : i + 1]), k
                )
                retrieval_candidates.append(ids[0])
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        budgets = [request.k or self.cfg.default_k for request in requests]
        per_request_candidates = []
        for candidates, k in zip(retrieval_candidates, budgets):
            per_request_candidates.append(candidates[:k])

        # Pad candidate lists and build one big feature batch.
        max_candidates = max(len(candidates) for candidates in per_request_candidates)
        flat_rows = []
        masks = []
        for i, request in enumerate(requests):
            candidates = per_request_candidates[i]
            context_indices = context_cats[i].tolist()
            cat_rows, numeric_rows = self._build_candidate_rows(
                context_indices,
                context_numeric[i].numpy(),
                candidates,
            )
            pad = max_candidates - len(candidates)
            if pad > 0:
                cat_rows = torch.cat(
                    [cat_rows, torch.zeros(pad, cat_rows.shape[1], dtype=torch.long)],
                    dim=0,
                )
                numeric_rows = torch.cat(
                    [numeric_rows, torch.zeros(pad, numeric_rows.shape[1])], dim=0
                )
            flat_rows.append((cat_rows, numeric_rows))
            masks.append([1] * len(candidates) + [0] * pad)

        all_cats = torch.cat([cats for cats, _ in flat_rows], dim=0)
        all_numeric = torch.cat([numeric for _, numeric in flat_rows], dim=0)
        t1 = time.perf_counter()
        logits = self.dlrm(all_cats, all_numeric)
        probs = torch.sigmoid(logits).numpy()
        ranking_ms = (time.perf_counter() - t1) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0

        budget = self.cfg.latency_budget_ms
        responses = []
        offset = 0
        for i, request in enumerate(requests):
            candidates = per_request_candidates[i]
            mask = np.array(masks[i], dtype=bool)
            batch_probs = probs[offset : offset + max_candidates][mask]
            order = np.argsort(-batch_probs)
            ads = [
                AdScore(ad_id=candidates[j], ctr=float(batch_probs[j]), score=float(batch_probs[j]))
                for j in order
            ]
            degraded = bool(total_ms > budget or retrieval_ms > budget * self.cfg.retrieval_budget_fraction)
            self.metrics.record(
                total_ms, retrieval_ms, ranking_ms, degraded
            )
            responses.append(
                RankResponse(
                    ads=ads,
                    degraded=degraded,
                    num_candidates=len(ads),
                    latency_ms=LatencyBreakdown(
                        total_ms=total_ms,
                        retrieval_ms=retrieval_ms,
                        ranking_ms=ranking_ms,
                    ),
                )
            )
            offset += max_candidates
        return responses


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------


def create_app(config: ServingConfig | None = None) -> FastAPI:
    """App factory; the module-level ``app`` uses ``configs/serving.yaml``."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        cfg = config or load_serving_config()
        engine = ServingEngine(cfg)
        application.state.engine = engine
        yield
        engine.close()
        application.state.engine = None

    application = FastAPI(title="ctr-ranking serving", lifespan=lifespan)

    @application.get("/healthz")
    async def healthz() -> dict[str, Any]:
        engine = application.state.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="not loaded")
        return {
            "status": "ok",
            "ads_in_index": len(engine.retrieval.ad_ids),
            "vocabularies": len(engine.vocabs),
        }

    @application.get("/metrics")
    async def metrics() -> dict[str, Any]:
        engine = application.state.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="not loaded")
        return engine.metrics.snapshot()

    @application.post("/rank", response_model=RankResponse)
    async def rank(request: RankRequest) -> RankResponse:
        engine = application.state.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="not loaded")
        return engine.rank(request)

    @application.post("/rank:batch", response_model=BatchRankResponse)
    async def rank_batch(request: BatchRankRequest) -> BatchRankResponse:
        engine = application.state.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="not loaded")
        return BatchRankResponse(responses=engine.rank_batch(request.requests))

    return application


app = create_app()
