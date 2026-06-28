# Phase 3 findings — hybrid + rerank, and the honest path to 0.92

## Ablation (held-out 1k protocol, n=1000, depth 10)

| Stage | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| Dense (CLIP ViT-B/32) | 0.570 | 0.821 ±0.024 | 0.887 | 0.684 | 0.730 |
| + RRF (CLIP + BM25), equal weight | 0.497 | 0.745 ±0.027 | 0.796 | 0.612 | 0.650 |
| Dense + rerank (bge-reranker-v2-m3) | 0.492 | 0.824 ±0.024 | 0.906 | 0.634 | 0.696 |
| + RRF + rerank | 0.470 | 0.805 ±0.025 | 0.895 | 0.614 | 0.679 |

**Headline: hybrid + text-rerank does NOT reach 0.92. Best ~0.82, statistically
unchanged from dense CLIP alone.** Verified real (not a bug): dense reproduced
bit-for-bit by an independent re-run; the reranker demonstrably reorders
(R@1 ↓ 0.57→0.49, R@10 ↑ 0.887→0.906).

## Why (the honest mechanism)

- **Equal-weight RRF hurts** because BM25 (lexical over caption[0], ~0.45 R@5) is
  much weaker than dense (0.82); equal voting dilutes the strong signal. A
  dense-heavy **weighted RRF** recovers and modestly beats dense:
  weight(dense)=3→0.810, =5→0.838, =10→**0.842** (best), =20→0.834. Lexical-fusion
  ceiling ≈ 0.84 — nowhere near 0.92. So "+RRF hurts" is an equal-weight artifact,
  not proof hybrid is worthless.
- **Text reranker is a wash.** Top-50 oracle ceiling is 0.983 (dense R@50), but the
  bge text cross-encoder realizes ~0% of the 0.82→0.98 headroom. It scores
  (query_caption, candidate caption[0]) — i.e. it re-ranks with strictly LESS image
  information than CLIP already used, so it can't fix CLIP's visually-confusable
  errors. Not a tuning problem; a structural ceiling.

## The legitimate lever: a stronger dense backbone (measured on THIS protocol)

| Dense backbone | Recall@5 | Recall@10 |
|---|---|---|
| OpenAI CLIP ViT-B/32 (current) | 0.821 | 0.887 |
| OpenAI CLIP ViT-B/16 | 0.842 | — |
| OpenAI CLIP ViT-L/14 | 0.852 | 0.921 |
| OpenAI CLIP ViT-L/14@336 | 0.879 | 0.936 |
| OpenCLIP ViT-L/14 (laion2b) | 0.909 | 0.953 |
| **OpenCLIP ViT-H/14 (laion2b)** | **0.929** | 0.963 |

(Measured by the Phase-3 verification agent in this repo's venv on the held-out 1k
protocol; to be reproduced as a first-class harness artifact in Phase 3b.)
Corroborated by published standard-protocol R@5 (LAION): OpenAI ViT-L/14 87.0,
LAION-2B ViT-L/14 92.9, ViT-H/14 94.0.

**Conclusion:** ~0.92 R@5 is honestly reachable on Flickr30k caption→image — via a
stronger CLIP backbone (OpenCLIP ViT-H/14), NOT via BM25-RRF or a text reranker.
The resume number should be set to whichever backbone we ship: ~0.82 (ViT-B/32)
or ~0.93 (ViT-H/14), each cited on the standard protocol.
