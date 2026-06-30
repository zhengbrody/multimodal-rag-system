# Load test report — Personal RAG API

Real, reproducible load test of the FastAPI service using the Locust driver
(`tests/load/locustfile.py`). Traffic mix mirrors the JMeter plan: **70% `POST /ask`,
20% `GET /health`, 10% `GET /metrics`**.

## Setup (honest framing)

- **Mode**: `USE_MOCK=true` — measures the **API + retrieval/serving layer** (FastAPI,
  request handling, knowledge-base retrieval, response shaping). It does **not** include an
  external LLM generation call; that latency is provider-bound and separate.
- **Server**: a single `uvicorn` worker, `api.personal_api:app`.
- **Hardware**: Apple Silicon laptop (M-series), 24 GB.
- **Driver**: Locust 2.44, headless, 30 s steady-state per run.
- Raw artifacts: `locust_stats.csv` (100 users), `locust500_stats.csv` (500 users).

## Results

| Concurrency | Requests | Failures | p50 | p95 | p99 | max | Throughput |
|---|---|---|---|---|---|---|---|
| 100 users | 8,715 | **0 (0.00%)** | 2 ms | 5 ms | 19 ms | 44 ms | **325 req/s** |
| 500 users | 33,378 | **0 (0.00%)** | 110 ms | 180 ms | **210 ms** | 689 ms | **1,151 req/s** |

`POST /ask` alone at 500 users: p50 = 120 ms, p95 = 190 ms, p99 = 220 ms, ~804 req/s, 0 failures.

## Reading

- **Zero failures at 500 concurrent users**, with p99 = 210 ms — i.e. **sub-second tail latency**
  for the serving layer under heavy concurrency on a single worker.
- 100 → 500 users: throughput scales 325 → 1,151 req/s while latency stays bounded (p99 19 → 210 ms),
  consistent with a healthy queue rather than saturation/errors.
- These numbers describe the **serving + retrieval** path. End-to-end latency in a live LLM
  deployment is dominated by the model call and should be reported separately.

## Reproduce

```bash
USE_MOCK=true uvicorn api.personal_api:app --app-dir src --port 8000   # terminal 1
# terminal 2:
locust -f tests/load/locustfile.py --host http://localhost:8000 \
       --headless -u 500 -r 100 -t 30s --csv tests/load/results/locust500
```

A JMeter equivalent (`test_plan.jmx` + `run_load_test.sh`) is also provided for those with JMeter.
