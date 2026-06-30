# Reproducing every number

Every figure this project cites is regenerated from source by a `make` target — no
hand-entered results. The eval/retrieval work runs in a dedicated venv (`.venv-eval`,
Python 3.11/3.12) so the heavy Torch/CLIP deps stay out of the lightweight app/CI env.

## One command from a fresh machine

```bash
make venv          # .venv-eval + requirements_eval.txt (Torch, OpenCLIP, Qdrant, …)
make data          # download + prepare Flickr30k (1K gallery + eval queries)
make reproduce     # eval + ablation + faithfulness -> eval/results/*.json
make manifest      # results_manifest.json: git SHA, package versions, key metrics
```

`make help` lists every target.

## What produces each cited number

| Claim | Command | Artifact | Value (measured) |
|---|---|---|---|
| **Headline R@5 = 0.942 ±0.006** (OpenCLIP ViT-H/14, standard 5k protocol) | `make eval` | `eval/results/phase3b_headline_*` | 0.942 ±0.006 |
| Baseline R@5 = 0.844 (CLIP ViT-B/32) | `make eval` (B/32 via `--retrievers clip`) | `eval/results/phase2_headline_*` | 0.844 ±0.010 |
| Backbone curve (lift source: backbone, **not** rerank) | `make backbone-sweep` | `eval/results/backbone_sweep_*` | B/32→L/14→H/14 |
| Hybrid + rerank ablation (does **not** reach 0.92) | `make ablation` | `eval/results/phase3_ablation*` | best ~0.84 |
| Vector-DB faithfulness (Qdrant == numpy) | `make faithfulness` | `eval/results/faithfulness.json` | 1000/1000 |
| Load: 0 failures @ 500 users, p99 = 210 ms | `make load-test` (server up first) | `tests/load/results/load_report.md` | p99 210 ms, 1151 req/s |
| Metric correctness (Recall@k/MRR/nDCG) | `make test` | `tests/test_eval_metrics.py` | pinned to hand-computed values |

## Honesty notes

- The 0.94 lift comes from a **stronger dense backbone (OpenCLIP ViT-H/14)** — the
  ablation (`make ablation`) shows BM25-RRF and the text cross-encoder reranker plateau
  at ~0.84. The backbone curve is corroborated by LAION's published ViT-H/14 number (94.0).
- Recall is meaningless without its gallery size: 0.942 is on the **1K** Karpathy test
  gallery (standard protocol); the full **31K** corpus is harder (see `phase2_scale_*`).
- Load numbers are the **serving + retrieval** layer in mock mode (no external LLM call),
  on a single uvicorn worker. End-to-end LLM latency is provider-bound and reported separately.
- `results_manifest.json` stamps the git SHA and package versions used for a given run, so
  any cited number is traceable to the exact code + environment that produced it.
