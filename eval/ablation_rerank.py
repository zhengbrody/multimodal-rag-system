"""
Phase-3 ablation: dense-only vs +RRF(sparse) vs +RRF+cross-encoder-rerank.

Caption->image retrieval on the HELD-OUT 1k protocol (query = a caption in
[1..4]; gallery text = the image's caption[0]) so the BM25 sparse path and the
text reranker cannot win by exact-string match against the query.

Pipeline stages (each row adds one):
  1. dense        : CLIP image-vs-caption cosine (the pixel-grounded signal).
  2. +rrf         : fuse dense with BM25 lexical over the gallery captions (RRF).
  3. +rrf+rerank  : rerank the top-N fused candidates with bge-reranker-v2-m3
                    over (query_caption, candidate_caption) pairs.

Models load ON DEMAND and are freed before the next loads (16GB ceiling): CLIP is
freed after query encoding; the cross-encoder loads only for stage 3. No 7B LLM.

Run:
    python -m eval.ablation_rerank                 # full 1000-query ablation
    python -m eval.ablation_rerank --limit 100     # quick smoke test
    python -m eval.ablation_rerank --rerank-n 50   # candidates fed to the reranker
Artifacts: eval/results/phase3_ablation.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import aggregate  # noqa: E402

DATA = REPO_ROOT / "data" / "flickr30k"
GALLERY_PKL = DATA / "clip_gallery.pkl"
GALLERY_JSONL = DATA / "gallery.jsonl"
QUERIES = DATA / "eval_queries.jsonl"
RESULTS = REPO_ROOT / "eval" / "results"

CAND_DEPTH = 100  # candidates each retriever contributes to fusion
METRIC_DEPTH = 10  # depth for recall/mrr/ndcg


def _load_gallery():
    with open(GALLERY_PKL, "rb") as f:
        state = pickle.load(f)
    emb = np.asarray(state["embeddings"], dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    image_ids = [d["metadata"]["image_id"] for d in state["documents"]]
    rep = {}
    for line in GALLERY_JSONL.open():
        r = json.loads(line)
        rep[r["image_id"]] = r["rep_caption"]
    rep_texts = [rep[i] for i in image_ids]
    return emb, image_ids, rep_texts, rep


def _dense_rankings(query_vecs, gallery_emb, image_ids, depth):
    sims = query_vecs @ gallery_emb.T  # (Q, G)
    idx = np.argsort(-sims, axis=1)[:, :depth]
    return [[image_ids[i] for i in row] for row in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dense vs +RRF vs +rerank ablation.")
    ap.add_argument("--limit", type=int, default=None, help="first N queries (smoke test)")
    ap.add_argument("--rerank-n", type=int, default=50, help="candidates fed to the reranker")
    args = ap.parse_args()

    emb, image_ids, rep_texts, rep = _load_gallery()
    queries = [json.loads(line) for line in QUERIES.open()]
    if args.limit:
        queries = queries[: args.limit]
    captions = [q["caption"] for q in queries]
    gts = [q["gt_image_id"] for q in queries]
    print(f"Gallery: {len(image_ids)} images | Queries: {len(queries)} (held-out 1k protocol)")

    # --- Stage 1: dense (CLIP). Load CLIP only to encode queries, then free. ---
    t = time.time()
    from src.rag.clip_retriever import CLIPRetriever

    clip = CLIPRetriever(model_name="ViT-B/32")
    qvecs = np.asarray(clip._get_embeddings_batch(captions), dtype=np.float32)
    qvecs = qvecs / np.linalg.norm(qvecs, axis=1, keepdims=True)
    del clip
    gc.collect()
    dense_lists = _dense_rankings(qvecs, emb, image_ids, CAND_DEPTH)
    print(f"  dense done in {time.time() - t:.1f}s")

    # --- Stage 2: sparse (BM25) + RRF fusion ---
    t = time.time()
    from src.rag.fusion import rrf_ids
    from src.rag.sparse_retriever import BM25Retriever

    bm25 = BM25Retriever()
    bm25.index(image_ids, rep_texts)
    sparse_lists = [bm25.search_ids(c, CAND_DEPTH) for c in captions]
    rrf_lists = [rrf_ids([d, s]) for d, s in zip(dense_lists, sparse_lists)]
    print(f"  sparse+rrf done in {time.time() - t:.1f}s")

    # --- Stage 3: cross-encoder rerank of the top-N fused candidates ---
    t = time.time()
    from src.rag.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()

    def _rerank_over(base_lists):
        heads = [b[: args.rerank_n] for b in base_lists]
        cand_lists = [[(iid, rep[iid]) for iid in head] for head in heads]
        reranked_heads = reranker.rerank_batch(captions, cand_lists)
        out = []
        for base, head, reranked in zip(base_lists, heads, reranked_heads):
            # final = reranked head + untouched base tail (preserves recall beyond N)
            tail = [iid for iid in base if iid not in set(head)]
            out.append(reranked + tail)
        return out

    # Diagnostic: rerank over DENSE candidates (no sparse) vs over RRF candidates,
    # to isolate whether the BM25/RRF path helps the final number or the reranker
    # carries it.
    dense_rerank_lists = _rerank_over(dense_lists)
    rerank_lists = _rerank_over(rrf_lists)
    reranker.unload()
    print(f"  rerank done in {time.time() - t:.1f}s ({len(queries) * args.rerank_n} pairs)")

    # --- Metrics for each stage ---
    stages = {
        "dense": dense_lists,
        "rrf": rrf_lists,
        "dense_rerank": dense_rerank_lists,
        "rrf_rerank": rerank_lists,
    }
    results: Dict[str, Dict] = {}
    for name, lists in stages.items():
        results[name] = aggregate(lists, gts, k_values=(1, 5, 10), ndcg_k=METRIC_DEPTH)

    # --- Report ---
    labels = {
        "dense": "Dense (CLIP only)",
        "rrf": "+ RRF (CLIP + BM25)",
        "dense_rerank": "Dense + rerank (no sparse)",
        "rrf_rerank": "+ RRF + rerank (bge-reranker-v2-m3)",
    }
    lines = [
        "| Stage | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |",
        "|---|---|---|---|---|---|",
    ]
    for name in ["dense", "rrf", "dense_rerank", "rrf_rerank"]:
        m = results[name]
        lines.append(
            f"| {labels[name]} | {m['recall@1']:.3f} | "
            f"{m['recall@5']:.3f} ±{m['recall@5_ci95']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} |"
        )
    table = "\n".join(lines)

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "held-out 1k (query=caption[1..4], gallery text=caption[0])",
        "gallery_size": len(image_ids),
        "num_queries": len(queries),
        "rerank_n": args.rerank_n,
        "reranker": "BAAI/bge-reranker-v2-m3 (text cross-encoder over captions)",
        "results": results,
    }
    (RESULTS / "phase3_ablation.json").write_text(json.dumps(payload, indent=2))
    (RESULTS / "phase3_ablation.md").write_text(
        "# Phase 3 — Hybrid (dense+sparse RRF) + cross-encoder rerank ablation\n\n"
        f"Held-out 1k protocol · {len(image_ids)}-image gallery · {len(queries)} queries · "
        f"rerank top-{args.rerank_n}\n\n{table}\n"
    )
    print("\n" + "=" * 70)
    print(table)
    print("=" * 70)
    print(f"saved: {RESULTS / 'phase3_ablation.md'}")


if __name__ == "__main__":
    main()
