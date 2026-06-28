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
```

Perf note: **Qdrant local mode = exact full-scan** (~0.28s/query over 62K), so 5K-query
Qdrant runs take ~20 min. For the 1K headline use the `clip` numpy adapter (fast; proven
identical to Qdrant via `faithfulness.json`). Eval adapters batch-encode queries
(`encode_queries`) to avoid per-query CLIP overhead.

## New code added by this engagement

- `eval/metrics.py` — Recall@k, MRR, nDCG@10 + 95% CIs (Wald); unit-tested.
- `eval/datasets/prepare_flickr30k.py` — 1K gallery; `eval_queries.jsonl` (held-out, proxy-safe)
  and `eval_queries_5k.jsonl` (standard 5000-caption protocol).
- `eval/adapters.py` — uniform `retrieve_ids` over clip/mock/dense/qdrant/qdrant_full.
- `eval/evaluate.py` — runner; `--queries 1k|5k`, `--tag`, CI columns.
- `eval/ablation_rerank.py` — dense vs +RRF vs +rerank ablation.
- `eval/faithfulness_check.py` — isolated proof the vector DB preserves rankings.
- `src/rag/vector_store.py` — pluggable Qdrant/Pinecone VectorStore (embedder-agnostic).
- `scripts/ingest.py` — batched CLIP ingest of full Flickr30k.
- `src/rag/sparse_retriever.py` (BM25), `src/rag/fusion.py` (RRF), `src/rag/reranker.py` (cross-encoder).
- Tests: `tests/test_eval_metrics.py`, `tests/test_fusion.py`.

## Evidence so far (REAL measured numbers)

Flickr30k caption→image. Artifacts under `eval/results/`.

| Claim | Status | Real number |
|---|---|---|
| "92% Recall@5, 1K eval set" | **measured** | Dense CLIP ViT-B/32 **R@5 = 0.844 ±0.010** (standard 5000-caption protocol; matches published ~0.83-0.85). Full 31K-corpus R@5 = 0.382 ±0.030. |
| Hybrid RRF + rerank → 0.92 | **DOES NOT reach it** | Held-out 1k: dense 0.821 / +RRF(equal) 0.745 / dense+rerank 0.824 / +RRF+rerank 0.805. Weighted RRF best 0.842. Text rerank is a wash (structural ceiling). See `phase3_findings.md`. |
| Path to 0.92 | **measured, honest** | Stronger dense backbone: OpenCLIP ViT-L/14(laion2b) 0.909, **ViT-H/14(laion2b) 0.929** (on our protocol; corroborated by LAION published 92.9/94.0). NOT via RRF/rerank. |
| "50K+ items" | **met** | 62,028 vectors ingested (31,014 images + 31,014 captions) in Qdrant. |
| "CLIP multimodal + Qdrant" | **real** | CLIP cross-modal + Qdrant backing store; faithfulness 1000/1000 vs numpy. |

## Stale/false strings to fix (résumé/KB layer)

`data/raw/knowledge_base.json` and `docs/PROJECT_POSITIONING.md` (line ~57) contain:
"92% Recall@5", "50K+ products", "1000 manually labeled queries", "deployed Pinecone".
Truth: photos not products; queries auto-generated (held-out captions, seeded); Qdrant not
Pinecone; 0.92 only with a ViT-H/14 backbone we haven't shipped yet. **Fix in Phase 6**
once the final backbone/number is chosen.

## Follow-up plan (status against the original 7-phase plan)

Legend: ✅ done · ⚠️ partial · ❌ not started. (An extra "eval upgrade" mini-phase —
standard 5000-caption protocol + 95% CIs — was completed between P1 and P3.)

- **P0 eval harness** ✅ — Recall@5/10, MRR, nDCG@10 + CIs; baseline table; unit tests; faithfulness proof.
- **P1 50K ingest + real vector store** ✅ — Qdrant, 62,028 vectors; faithfulness 1000/1000 vs numpy.
- **P2 CLIP end-to-end** ⚠️ PARTIAL — text→image is fully activated & measured; **TODO: image→image
  and text→image qualitative demos (retrieved examples) + add multimodal query cases to the eval set.**
- **P3 RRF + cross-encoder rerank ablation** ✅ — honest result: does **not** reach 0.92; lever is the
  dense backbone, not RRF/text-rerank (see `eval/results/phase3_findings.md`).
- **P3b dense-backbone decision** ❌ NEEDS USER DECISION — (a) ship **OpenCLIP ViT-H/14** (add `open_clip`
  adapter, re-ingest gallery, re-run harness → cite ~0.92-0.93), or (b) keep ViT-B/32 → cite **0.84**.
  Either way keep weighted-RRF (dense-heavy, ~0.84) as the documented hybrid. **Blocks the headline
  number, so P2 sign-off and P6 résumé bullets depend on this.**
- **P4 endpoints → 7** ❌ — add `/search/text`, `/search/image`, `/health` (Pydantic); update Swagger.
- **P5 load test** ❌ — run JMeter/Locust vs the containerized API; capture p50/p95/p99 @ 100/300/500.
- **P6 evidence pack** ❌ — README with eval tables + ablation + load report + arch diagram; fix the
  stale KB/positioning strings (products / manually-labeled / Pinecone / 92%); 2 rewritten résumé bullets.

**Overall: ~half done.** P0/P1/P3 ✅; P2 ⚠️; P3b decision + P4/P5/P6 ❌.

## Working agreement

Eval-first; **one phase at a time, stop for the user's confirmation after each**; explain
design + trade-offs (interview defense) and teach the implementation after each phase;
report the real number even when it misses the target.
