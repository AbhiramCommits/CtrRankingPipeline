# Serving benchmark

## Closed-loop load test

- concurrency: 8
- duration: 10.011s
- throughput: 635.41 QPS (6361 requests, 4 degraded)

| metric | p50 | p95 | p99 | mean |
| --- | --- | --- | --- | --- |
| end-to-end latency (ms) | 11.918 | 14.388 | 24.781 | 12.417 |

Server-side split (mean ms): retrieval=0.322, ranking=0.784

## Ranking batch-size sweep

| batch size | mean latency (ms) | items scored / s |
| --- | --- | --- |
| 1 | 2.087 | 479.1 |
| 8 | 10.347 | 773.1 |
| 32 | 49.607 | 645.1 |
| 128 | 180.341 | 709.8 |
| 512 | 825.306 | 620.4 |
