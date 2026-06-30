# Kubernetes HPA autoscaling — observed behavior

Real, reproducible run of the RAG API on a **local Kubernetes cluster (kind)** with a
Horizontal Pod Autoscaler driving CPU-based autoscaling under load. This is a local,
free demo — **not** a cloud/production deployment — but the HPA behavior (scale-up,
sustain, scale-down) is genuinely observed, not asserted.

## Setup

- **Cluster:** kind (Kubernetes in Docker), single node, on a macOS/Docker Desktop VM.
- **App:** the FastAPI service in **mock mode** (key-free; `k8s/kind/config-demo.yaml`).
- **Metrics:** metrics-server (patched with `--kubelet-insecure-tls` for kind).
- **Deployment:** `cpu: request 100m / limit 300m`, readiness + liveness probes on `/health`.
- **HPA** (`k8s/hpa.yaml`): target **CPU 50%** of request, `minReplicas 2`, `maxReplicas 6`,
  fast scale-up (0s stabilization), 60s scale-down stabilization.
- **Load:** the existing Locust test (`tests/load/locustfile.py`), 500 users, through a
  NodePort so kube-proxy load-balances across all pods.

## Observed timeline (from `hpa_samples.log`)

| Phase | CPU (util/target) | Ready pods | Note |
|---|---|---|---|
| baseline | 3% / 50% | **2** | HPA floor, no load |
| load +30s | 3% → 175% / 50% | 2 | load ramping; HPA computing |
| load +40s | **300% / 50%** | **6** | **scaled 2 → 6** (hit maxReplicas) |
| load steady | ~100% / 50% | 6 | 6 pods absorb the load |
| cooldown +30s | 96% → 9% → 3% | 6 | load removed; CPU collapses |
| cooldown +70s | 3% / 50% | 6 → **3** | scale-down begins (after 60s stabilization) |
| cooldown +115s | 3% / 50% | **2** | **scaled back to floor (6 → 3 → 2)** |

## What this demonstrates (defensible in interview)

- **Scale-up:** CPU crossed the 50% target (peaked **300%**) and HPA scaled **2 → 6 pods in ~40s**.
- **Stability under load:** 6 pods served **79,228 requests with 0 failures**; p50 600 ms /
  p95 980 ms / p99 1100 ms, ~531 req/s **inside the resource-constrained kind VM**
  (each pod CPU-capped at 300m + a kube-proxy hop — this is an autoscaling demo, not a
  throughput benchmark; the bare-process numbers in `tests/load/results/load_report.md`
  are much faster).
- **Scale-down:** once load stopped and CPU fell, HPA scaled **6 → 3 → 2** back to the floor
  after its 60s stabilization window (~115 s total) — no thrashing.

## Honest caveats

- **Local kind, not cloud.** This shows HPA/k8s mechanics on a laptop; it is not an AWS/GKE
  production deployment and isn't claimed as one.
- **HPA is demo-tuned** for a short, observable run (0s scale-up stabilization). Production
  would dampen scale-up to avoid flapping — the trade-off is responsiveness vs. churn.
- **Mock mode**, so per-request work is retrieval-only (no external LLM call).
- An earlier run was corrupted by the machine sleeping mid-test; the Makefile now wraps the
  demo in `caffeinate -i` so the timed sampling stays contiguous.

## Reproduce

```bash
make k8s-demo     # kind up -> metrics-server -> build+load image -> deploy -> Locust load -> sample HPA
make k8s-clean    # kind delete cluster --name rag-demo
```

Raw artifacts: `k8s/results/hpa_samples.log` (full timeline), `k8s/results/k8s_locust_stats.csv`.
Manifests: `k8s/api-deployment.yaml`, `k8s/hpa.yaml`, `k8s/kind/*`.
