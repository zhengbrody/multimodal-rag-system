# Phase 3 — Hybrid (dense+sparse RRF) + cross-encoder rerank ablation

Held-out 1k protocol · 1000-image gallery · 1000 queries · rerank top-50

| Stage | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| Dense (CLIP only) | 0.570 | 0.821 ±0.024 | 0.887 | 0.684 | 0.730 |
| + RRF (CLIP + BM25) | 0.497 | 0.745 ±0.027 | 0.796 | 0.612 | 0.650 |
| Dense + rerank (no sparse) | 0.492 | 0.824 ±0.024 | 0.906 | 0.634 | 0.696 |
| + RRF + rerank (bge-reranker-v2-m3) | 0.470 | 0.805 ±0.025 | 0.895 | 0.614 | 0.679 |
