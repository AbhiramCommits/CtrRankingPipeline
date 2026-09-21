#!/usr/bin/env python3
"""Closed-loop load generator and ranking batch-size sweep.

Part 1 — closed-loop load test: ``--concurrency`` workers hammer ``/rank``
as fast as they can for ``--duration`` seconds; reports QPS, client-observed
p50/p95/p99 end-to-end latency, and the retrieval-vs-ranking time split
(reported by the server per response).

Part 2 — batch-size sweep: issues ``/rank:batch`` requests of size
1/8/32/128/512 and reports items-scored-per-second for each, showing how
coalescing amortizes the per-request overhead.

Requires a running server (see ``ctr/serving/app.py``)::

    uvicorn ctr.serving.app:app --host 127.0.0.1 --port 8000
    python scripts/benchmark.py --concurrency 8 --duration 10

Results are written to ``artifacts/bench/results.json`` and ``results.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import numpy as np

from ctr.retrieval.two_tower import CONTEXT_FIELDS
from ctr.serving.app import NUMERIC_DIM

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES = (1, 8, 32, 128, 512)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES))
    )
    parser.add_argument("--sweep-iterations", type=int, default=25)
    parser.add_argument("--output-dir", default="artifacts/bench")
    return parser.parse_args(argv)


def make_request() -> dict:
    """A random plausible request (values land in OOV when unseen, fine)."""
    context = {}
    for field in CONTEXT_FIELDS:
        context[field] = f"{random.getrandbits(32):08x}" if random.random() < 0.95 else None
    numeric = [round(random.uniform(0.0, 8.0), 4) for _ in range(NUMERIC_DIM)]
    return {"context_categorical": context, "numeric_features": numeric}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round(p / 100.0 * (len(ordered) - 1)), len(ordered) - 1)
    return float(ordered[index])


def run_load_test(client: httpx.Client, concurrency: int, duration: float) -> dict:
    """Closed-loop load generation until the deadline."""
    latencies: list[float] = []
    retrieval_ms: list[float] = []
    ranking_ms: list[float] = []
    counters = {"requests": 0, "degraded": 0}
    lock = threading.Lock()
    stop_at = time.perf_counter() + duration

    def worker():
        while time.perf_counter() < stop_at:
            start = time.perf_counter()
            response = client.post("/rank", json=make_request(), timeout=30.0)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            data = response.json()
            with lock:
                latencies.append(elapsed_ms)
                breakdown = data.get("latency_ms", {})
                retrieval_ms.append(float(breakdown.get("retrieval_ms", 0.0)))
                ranking_ms.append(float(breakdown.get("ranking_ms", 0.0)))
                counters["requests"] += 1
                counters["degraded"] += int(data.get("degraded", False))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for future in futures:
            future.result()
    wall = time.perf_counter() - started

    return {
        "concurrency": concurrency,
        "duration_seconds": round(wall, 3),
        "requests": counters["requests"],
        "qps": round(counters["requests"] / wall, 2),
        "degraded_requests": counters["degraded"],
        "latency_ms": {
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
            "mean": round(float(np.mean(latencies)) if latencies else 0.0, 3),
        },
        "server_side_split_ms": {
            "retrieval_mean": round(float(np.mean(retrieval_ms)) if retrieval_ms else 0.0, 3),
            "ranking_mean": round(float(np.mean(ranking_ms)) if ranking_ms else 0.0, 3),
        },
    }


def run_batch_sweep(
    client: httpx.Client, batch_sizes: list[int], iterations: int
) -> list[dict]:
    """Items-scored-per-second for each /rank:batch size."""
    rows = []
    for batch_size in batch_sizes:
        timings = []
        for _ in range(iterations):
            payload = {"requests": [make_request() for _ in range(batch_size)]}
            start = time.perf_counter()
            response = client.post("/rank:batch", json=payload, timeout=60.0)
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            assert len(response.json()["responses"]) == batch_size
        mean_s = float(np.mean(timings))
        rows.append(
            {
                "batch_size": batch_size,
                "mean_latency_ms": round(mean_s * 1000.0, 3),
                "items_scored_per_second": round(batch_size / mean_s, 1),
            }
        )
        logger.info(
            "batch_size=%4d mean_latency=%.2fms items/s=%.1f",
            batch_size,
            mean_s * 1000.0,
            batch_size / mean_s,
        )
    return rows


def render_markdown(load: dict, sweep: list[dict]) -> str:
    lines = [
        "# Serving benchmark",
        "",
        "## Closed-loop load test",
        "",
        f"- concurrency: {load['concurrency']}",
        f"- duration: {load['duration_seconds']}s",
        (
            f"- throughput: {load['qps']} QPS ({load['requests']} requests, "
            f"{load['degraded_requests']} degraded)"
        ),
        "",
        "| metric | p50 | p95 | p99 | mean |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| end-to-end latency (ms) | {load['latency_ms']['p50']} | "
            f"{load['latency_ms']['p95']} | {load['latency_ms']['p99']} | "
            f"{load['latency_ms']['mean']} |"
        ),
        "",
        (
            "Server-side split (mean ms): "
            f"retrieval={load['server_side_split_ms']['retrieval_mean']}, "
            f"ranking={load['server_side_split_ms']['ranking_mean']}"
        ),
        "",
        "## Ranking batch-size sweep",
        "",
        "| batch size | mean latency (ms) | items scored / s |",
        "| --- | --- | --- |",
    ]
    for row in sweep:
        lines.append(
            f"| {row['batch_size']} | {row['mean_latency_ms']} | "
            f"{row['items_scored_per_second']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    with httpx.Client(base_url=args.url) as client:
        health = client.get("/healthz", timeout=10.0)
        health.raise_for_status()
        logger.info("server healthy: %s", health.json())

        load = run_load_test(client, args.concurrency, args.duration)
        logger.info(
            "load test: %s QPS, p50=%.2fms p95=%.2fms p99=%.2fms, %d degraded",
            load["qps"],
            load["latency_ms"]["p50"],
            load["latency_ms"]["p95"],
            load["latency_ms"]["p99"],
            load["degraded_requests"],
        )

        sweep = run_batch_sweep(client, batch_sizes, args.sweep_iterations)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"load_test": load, "batch_sweep": sweep}
    with open(output_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(output_dir / "results.md", "w", encoding="utf-8") as handle:
        handle.write(render_markdown(load, sweep))
    logger.info("results written to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
