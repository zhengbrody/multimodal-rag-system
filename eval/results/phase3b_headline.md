# Phase 3b — stronger dense backbone reaches the 0.92 target (honestly)

Flickr30k caption→image retrieval. Dense retriever = **OpenCLIP ViT-H/14 (laion2b_s32b_b79k)**,
zero-shot, 1024-d embeddings. Standard 5000-caption protocol over the 1,000-image Karpathy test
gallery. Metrics are means over the query set; ± is the 95% CI half-width (Wald, 1.96·sem); MRR/nDCG
at depth 10. Numpy fast path (proven identical to the Qdrant-served system for ViT-B/32 via
`faithfulness.json`; the same exact cosine search is used here).

## Headline (cite this on the resume): standard Flickr30k protocol

| Setting | Queries | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| **OpenCLIP ViT-H/14, 1K gallery** | 5,000 | 0.777 | **0.942 ±0.006** | 0.967 | 0.847 | 0.877 |

## What changed vs the ViT-B/32 baseline

| Backbone | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| CLIP ViT-B/32 (Phase 2 baseline) | 0.593 | 0.844 ±0.010 | 0.902 | 0.699 | 0.748 |
| **OpenCLIP ViT-H/14 (shipped)** | 0.777 | **0.942 ±0.006** | 0.967 | 0.847 | 0.877 |
| Δ | +0.184 | **+0.098** | +0.065 | +0.148 | +0.129 |

## Reading

- **R@5 = 0.942 ±0.006** clears the 0.92 resume target on the standard protocol — and it is reached
  via the **legitimate lever identified in Phase 3: a stronger dense backbone, NOT BM25-RRF or a text
  cross-encoder** (those plateau at ~0.84; see `phase3_findings.md` for the structural reason).
- **Corroborated, not cherry-picked.** LAION's published standard-protocol text→image R@5 for
  ViT-H/14 (laion2b) is 94.0; our independent in-harness measurement (94.2) matches it.
- The CI is tight (±0.006) because this is the full n=5,000 protocol, not a 1,000-caption subsample.
- Cost is modest: the 1,000-image gallery encodes in ~78s on an M-series MPS device; queries are a
  single batched text-encode. The gallery is encoded once and cached
  (`data/flickr30k/openclip_gallery.pkl`), so serving cost at query time is one short text forward.

## Reproduce

```bash
source .venv-eval/bin/activate
uv pip install open_clip_torch>=2.24.0          # ViT-H/14 laion2b weights (~3.9GB, cached)
python -m eval.datasets.prepare_flickr30k        # 1K gallery + eval_queries_5k.jsonl
python -m eval.evaluate --retrievers openclip --queries 5k --tag phase3b_headline
```

Source artifacts: `phase3b_headline_flickr30k.json` (full metrics + run metadata),
`phase3b_headline_table.md` (the cited table). Honest hybrid result and the why: `phase3_findings.md`.
