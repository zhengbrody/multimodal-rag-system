# CLAUDE.md — multimodal-rag-system

Guidance for AI agents (and humans) working in this repo.

## What this project is

Two layers:

1. **The app** — a production-style RAG Q&A system for a personal website:
   Streamlit UI (`frontend/personal_app.py`) → FastAPI (`src/api/personal_api.py`)
   → pluggable retrievers (`src/rag/`) over a personal knowledge base. Mock mode
   runs with zero API cost; live demo on Streamlit Cloud. `python run.py` starts both.

2. **The evidence engagement (active work)** — an *eval-first* effort to back specific
   résumé claims with **real, reproducible numbers** on a public benchmark, and to
   correct any claim that isn't true. **Integrity rule: never fabricate to hit a
   target; report the real number.** Benchmark = **Flickr30k caption→image retrieval**.

## Locked technical decisions

- **Vector store**: self-hosted **Qdrant** is the real backing store; `src/rag/vector_store.py`
  keeps Pinecone/Chroma pluggable via `VECTOR_BACKEND` env. Résumé says Qdrant.
- **Dataset**: Flickr30k (~31K images × 5 captions). "50K+ items" = the ~62K ingested vectors.
- **Retrieval**: hybrid **dense (CLIP) + sparse (BM25)** fused with **RRF**, then a
  **cross-encoder reranker** (`bge-reranker-v2-m3`, local, text-only).
- **Eval protocol**: standard Flickr30k. Report **both** the standard 1K-gallery
  protocol (headline) **and** the full ~31K corpus, always with 95% CIs.
- **Hardware**: Mac mini M4, 16GB. Load models **on demand**; never CLIP + reranker +
  a 7B LLM at once (CLIP ViT-B/32 ~600MB, bge-reranker ~2.2GB).

## Environment & key commands

Eval/retrieval work uses a dedicated venv (system python is 3.9; repo targets 3.11-3.12):

```bash
source .venv-eval/bin/activate          # uv venv, py3.12; deps in requirements_eval.txt
                                        # NOTE: CLIP via PyPI `clip-anytorch` (not the git source);
                                        # datasets pinned to 2.21 (v4 dropped script loaders)
python -m eval.datasets.prepare_flickr30k    # build gallery + eval_queries{,_5k}.jsonl (gitignored)
python -m scripts.ingest                     # CLIP-encode full Flickr30k → Qdrant (~62K vectors)
python -m eval.evaluate --retrievers clip --queries 5k --tag headline   # standard-protocol R@5 + CI
python -m eval.ablation_rerank --rerank-n 50                            # dense vs +RRF vs +rerank
python -m eval.faithfulness_check                                       # vector-DB == numpy proof
pytest tests/test_eval_metrics.py tests/test_fusion.py -q               # metric/fusion unit tests

# Phase 3b — stronger dense backbone (OpenCLIP ViT-H/14, laion2b, 1024-d). Needs:
#   uv pip install open_clip_torch>=2.24.0
python -m eval.evaluate --retrievers openclip --queries 5k --tag phase3b_headline   # ViT-H/14 R@5 (numpy, 1K gallery — the headline)
python -m scripts.ingest --backbone open_clip                          # re-ingest 62K vectors → flickr30k_openclip collection
python -m eval.evaluate --retrievers qdrant_openclip_full --queries 1k --tag phase3b_scale  # ViT-H/14 31K-corpus number (slow)
```

Perf note: **Qdrant local mode = exact full-scan** (~0.28s/query over 62K), so 5K-query
Qdrant runs take ~20 min. For the 1K headline use the `clip` numpy adapter (fast; proven
identical to Qdrant via `faithfulness.json`). Eval adapters batch-encode queries
(`encode_queries`) to avoid per-query CLIP overhead.

## New code added by this engagement

- `eval/metrics.py` — Recall@k, MRR, nDCG@10 + 95% CIs (Wald); unit-tested.
- `eval/datasets/prepare_flickr30k.py` — 1K gallery; `eval_queries.jsonl` (held-out, proxy-safe)
  and `eval_queries_5k.jsonl` (standard 5000-caption protocol).
- `eval/adapters.py` — uniform `retrieve_ids` over clip/openclip/mock/dense/qdrant{,_openclip}{,_full}.
- `eval/evaluate.py` — runner; `--queries 1k|5k`, `--tag`, CI columns.
- `eval/ablation_rerank.py` — dense vs +RRF vs +rerank ablation.
- `eval/faithfulness_check.py` — isolated proof the vector DB preserves rankings.
- `src/rag/vector_store.py` — pluggable Qdrant/Pinecone VectorStore (embedder-agnostic).
- `scripts/ingest.py` — batched CLIP ingest of full Flickr30k.
- `src/rag/sparse_retriever.py` (BM25), `src/rag/fusion.py` (RRF), `src/rag/reranker.py` (cross-encoder).
- `src/rag/openclip_retriever.py` — **Phase 3b** OpenCLIP ViT-H/14 (laion2b, 1024-d) retriever; mirrors
  `CLIPRetriever`'s interface, embed-dim probed (never hard-coded). `eval/adapters.py` adds `openclip`
  (numpy fast path → headline) and `qdrant_openclip{,_full}` (vector-DB); `scripts/ingest.py --backbone
  open_clip` writes 1024-d vectors to a SEPARATE `flickr30k_openclip` collection.
- `src/rag/image_search_service.py` — **Phase 4** lazy + gated bridge from the measured OpenCLIP
  gallery to the served API; import-safe under `requirements_simple` (heavy deps loaded only on first
  use), returns 503 when no gallery cache (keeps the mock deployment working). Backs the new
  `POST /search/text` (caption→image) and `POST /search/image` (reverse image) FastAPI endpoints.
- `eval/qualitative_demo.py` — **Phase 2** writes `eval/results/phase2_qualitative.md`: text→image and
  image→image retrieved examples (by image_id + caption) — the qualitative companion to R@5=0.942.
- Tests: `tests/test_eval_metrics.py`, `tests/test_fusion.py` (BM25 test `importorskip`s `rank_bm25`
  so the lightweight CI env skips it cleanly, not fails).

## Evidence so far (REAL measured numbers)

Flickr30k caption→image. Artifacts under `eval/results/`.

| Claim | Status | Real number |
|---|---|---|
| "92% Recall@5, 1K eval set" | **MET (reproduced in-harness)** | Shipped backbone **OpenCLIP ViT-H/14 (laion2b): R@5 = 0.942 ±0.006** (standard 5000-caption protocol, 1K gallery; `phase3b_headline_*`). Corroborated by LAION published 94.0. Baseline CLIP ViT-B/32 was 0.844 ±0.010. |
| Hybrid RRF + rerank → 0.92 | **DOES NOT reach it** | Held-out 1k: dense 0.821 / +RRF(equal) 0.745 / dense+rerank 0.824 / +RRF+rerank 0.805. Weighted RRF best 0.842. Text rerank is a wash (structural ceiling). See `phase3_findings.md`. |
| Path to 0.92 | **ACHIEVED via backbone** | Stronger dense backbone, NOT RRF/rerank. ViT-B/32 0.844 → **ViT-H/14 0.942** (reproduced; LAION ViT-L/14 0.909, ViT-H/14 published 94.0 corroborate). Phase-3b run: 1000 imgs encoded in 78s on MPS, n=5000 queries. |
| "50K+ items" | **met** | 62,028 vectors ingested (31,014 images + 31,014 captions) in Qdrant. |
| "CLIP multimodal + Qdrant" | **real** | CLIP cross-modal + Qdrant backing store; faithfulness 1000/1000 vs numpy. |

## Stale/false strings (résumé/KB layer) — evidence-backed ones FIXED

`data/raw/knowledge_base.json`: the **evidence-backed** false strings about THIS project are
corrected — "92% accuracy/Recall@5" → **94% Recall@5** (real, 5k protocol), "50K+ products" →
**60K+ image/text vectors**, "1000 manually labeled queries" → **standard 5,000-caption protocol**
(auto-generated, seeded), and "vector search with Pinecone" → **Qdrant** (the real store). Penn State
"92% F1-score"/"92% code coverage" are a DIFFERENT past job — deliberately left untouched.

**Deliberately NOT changed** (user's call — unverifiable narrative, not in the evidence scope):
"Product Search" framing, "deployed on AWS EC2", "18% CTR via A/B testing", "2K+ QPS", "99.9% uptime".
If those don't reflect reality, they're the user's to correct. `docs/PROJECT_POSITIONING.md` is a
positioning/honesty doc (its "92% Recall@5" line is a self-aware caveat); left as-is.

## Follow-up plan (status against the original 7-phase plan)

Legend: ✅ done · ⚠️ partial · ❌ not started. (An extra "eval upgrade" mini-phase —
standard 5000-caption protocol + 95% CIs — was completed between P1 and P3.)

- **P0 eval harness** ✅ — Recall@5/10, MRR, nDCG@10 + CIs; baseline table; unit tests; faithfulness proof.
- **P1 50K ingest + real vector store** ✅ — Qdrant, 62,028 vectors; faithfulness 1000/1000 vs numpy.
- **P2 CLIP end-to-end** ✅ — text→image fully activated & measured; image→image + text→image
  qualitative demos generated (`eval/qualitative_demo.py` → `phase2_qualitative.md`, ViT-H/14;
  retrieval is visibly on-topic). Served live via the Phase-4 `/search/*` endpoints.
- **P3 RRF + cross-encoder rerank ablation** ✅ — honest result: does **not** reach 0.92; lever is the
  dense backbone, not RRF/text-rerank (see `eval/results/phase3_findings.md`).
- **P3b dense-backbone decision** ✅ DONE — shipped **OpenCLIP ViT-H/14** (laion2b); headline
  **reproduced in-harness: R@5 = 0.942 ±0.006** (standard 5k protocol, 1K gallery, n=5000;
  `phase3b_headline_*`). Code: `openclip_retriever.py`, `openclip`/`qdrant_openclip` adapters,
  `ingest.py --backbone open_clip`, `open_clip_torch` dep. The 0.92 target is MET via the backbone,
  NOT RRF/rerank. Keep weighted-RRF (dense-heavy, ~0.84) as the documented hybrid. **Still optional:**
  the 31K full-corpus ViT-H/14 number (re-ingest with `--backbone open_clip` → `qdrant_openclip_full`).
- **P4 endpoints (multimodal search)** ✅ — added `POST /search/text` (caption→image) and
  `POST /search/image` (reverse image), Pydantic-typed, auto-documented in Swagger; gated/lazy so the
  mock deployment is untouched. Verified end-to-end (text query returns on-topic images; image self-match
  score=1.0). API now exposes 11 functional routes. `/health` already existed.
- **P5 load test** ✅ — Locust driver (`tests/load/locustfile.py`, pure-Python alt to the JMeter plan);
  measured **0 failures at 500 concurrent users, p99 = 210 ms, ~1,150 req/s** (mock mode, single worker;
  100 users: p99 19 ms, 325 req/s). Report: `tests/load/results/load_report.md`.
- **P6 evidence pack** ✅ (mostly) — README "Evidence & Benchmarks" section (eval table + honest ablation
  note + load table + search endpoints) and `/search/*` curl docs; KB evidence-strings fixed (see above);
  résumé bullets drafted (in chat). Optional polish remaining: a dedicated multimodal arch diagram.
- **CI hygiene (bonus)** ✅ — fixed pre-existing CI-red from the eval merge: black-formatted
  `sparse_retriever`/`reranker`/`vector_store`/`test_eval_metrics`, removed an F541 f-string, and made
  the BM25 test skip without `rank_bm25`. Verified `pytest tests/` (29 pass/1 skip) + black + ruff green
  in a `requirements_simple`-only env. Also: `.gitignore` now covers `.venv-eval/`.

**Overall: plan complete.** P0–P6 ✅ (headline R@5=0.942 reproduced; multimodal search served; 500-user
load test 0 failures/p99 210 ms; CI green; KB evidence-strings honest). Optional follow-ups only:
the 31K full-corpus ViT-H/14 number, a multimodal arch diagram, and the user's call on the
unverifiable KB narrative (AWS/CTR/QPS/product-search).

## Working agreement

Eval-first; **one phase at a time, stop for the user's confirmation after each**; explain
design + trade-offs (interview defense) and teach the implementation after each phase;
report the real number even when it misses the target.
