# Phase 2 — Standard-protocol re-baseline (with 95% CIs)

Flickr30k caption→image retrieval. Dense retriever = CLIP ViT-B/32 (zero-shot).
Metrics are means over the query set; ± is the 95% CI half-width (Wald, 1.96·sem).
MRR/nDCG are computed at depth 10. Qdrant local mode is **exact** brute-force search
and returns rankings **identical** to the in-memory numpy path (verified: 1000/1000
top-5 agreement, see `faithfulness.json`), so the dense number below is reported via
the fast numpy path and holds for the Qdrant-served system.

## Headline (cite this on the resume): standard Flickr30k protocol

Candidate pool = the **1,000-image** Karpathy test gallery. Queries = all **5,000**
captions (5 per image) — the canonical text→image protocol, comparable to published
CLIP numbers (ViT-B/32 ≈ 0.83–0.85 R@5).

| Setting | Queries | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| **CLIP ViT-B/32, 1K gallery** | 5,000 | 0.593 | **0.844 ±0.010** | 0.902 | 0.699 | 0.748 |

## Honest scale setting: full corpus

Candidate pool = the **31,014-image** full Flickr30k corpus (≈62K vectors ingested
incl. captions). Same model; recall drops because the gold image competes with 31×
more distractors. Reported on the 1,000 held-out-caption set.

| Setting | Queries | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| **CLIP ViT-B/32, 31K gallery** | 1,000 | 0.188 | **0.382 ±0.030** | 0.473 | 0.273 | 0.321 |

## Reading

- **Headline R@5 = 0.844 ±0.010** on the standard protocol — the honest dense baseline.
  Resume target is 0.92; the ~7.6-pt gap is what Phase 3 (BM25+CLIP RRF + cross-encoder
  rerank) must legitimately close. If it doesn't, the resume gets the real number.
- Any Recall number is meaningless without its **gallery size**: 0.844 @ 1K vs 0.382 @ 31K
  is the same model on the same queries. Always quote the gallery.
- Source artifacts: `phase2_headline_flickr30k.json` (1K, 5k protocol),
  `phase2_scale_flickr30k.json` (31K), `faithfulness.json` (vector-DB faithfulness).
