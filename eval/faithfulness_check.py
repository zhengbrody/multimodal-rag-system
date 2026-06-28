"""
Vector-store faithfulness proof (isolated, no confounds).

The Phase-1 end-to-end comparison (numpy CLIP R@5=0.822 vs Qdrant CLIP R@5=0.813)
changed TWO things at once: the database AND the gallery image source (the numpy
gallery was encoded from re-saved quality-90 JPEGs, while the ingest encoded the
original PIL images). A near-match across that confound does not, by itself,
prove the database preserves rankings.

This script removes the confound: it takes the EXACT same gallery vectors and the
EXACT same query vectors, ranks them two ways —

    (a) numpy argsort over cosine similarity   (the in-memory reference)
    (b) Qdrant top-k over an injected collection of those identical vectors

— and reports top-k set agreement. If they agree, the vector DB is rank-faithful
*given fixed vectors*, so any residual end-to-end delta is upstream of the DB
(the image-encoding pipeline), not the store.

Note: qdrant-client local/embedded mode performs EXACT brute-force search (not an
approximate HNSW index), so this proves exact-search faithfulness. An approximate
index (Qdrant server >~20k pts, or Pinecone serverless) would have its own ANN
recall to measure separately.

Run:  python -m eval.faithfulness_check
Artifact: eval/results/faithfulness.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GALLERY_PKL = REPO_ROOT / "data" / "flickr30k" / "clip_gallery.pkl"
QUERIES = REPO_ROOT / "data" / "flickr30k" / "eval_queries.jsonl"
RESULTS = REPO_ROOT / "eval" / "results" / "faithfulness.json"
TMP_QDRANT = "/tmp/qdrant_faithfulness_check"
DEPTH = 10


def _load_gallery():
    with open(GALLERY_PKL, "rb") as f:
        state = pickle.load(f)
    embs = np.asarray(state["embeddings"], dtype=np.float32)
    image_ids = [d["metadata"]["image_id"] for d in state["documents"]]
    return embs, image_ids


def _query_vectors(n: int) -> np.ndarray:
    """Embed the real eval-query captions with CLIP (the representative queries)."""
    from src.rag.clip_retriever import CLIPRetriever

    clip = CLIPRetriever(model_name="ViT-B/32")
    captions = [json.loads(line)["caption"] for line in QUERIES.open()][:n]
    return np.vstack([clip._get_text_embedding(c) for c in captions])


def main() -> None:
    import shutil

    from qdrant_client import QdrantClient, models

    gallery, image_ids = _load_gallery()
    # Normalize for a clean cosine == dot product (CLIP vectors are already unit-norm).
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    print(f"Gallery: {gallery.shape[0]} vectors, dim {gallery.shape[1]}")

    queries = _query_vectors(gallery.shape[0])
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    print(f"Queries: {queries.shape[0]}")

    # (a) numpy reference ranking
    sims = queries @ gallery.T  # (Q, G)
    numpy_topk = np.argsort(-sims, axis=1)[:, :DEPTH]

    # (b) Qdrant over the IDENTICAL gallery vectors
    shutil.rmtree(TMP_QDRANT, ignore_errors=True)
    client = QdrantClient(path=TMP_QDRANT)
    client.create_collection(
        "faith",
        vectors_config=models.VectorParams(size=gallery.shape[1], distance=models.Distance.COSINE),
    )
    client.upsert(
        "faith",
        points=[
            models.PointStruct(id=i, vector=gallery[i].tolist(), payload={"row": i})
            for i in range(gallery.shape[0])
        ],
    )

    top1_match = 0
    top5_setmatch = 0
    top10_setmatch = 0
    for q in range(queries.shape[0]):
        hits = client.query_points(
            "faith", query=queries[q].tolist(), limit=DEPTH, with_payload=True
        ).points
        q_rows = [h.payload["row"] for h in hits]
        n_rows = list(numpy_topk[q])
        if q_rows[0] == n_rows[0]:
            top1_match += 1
        if set(q_rows[:5]) == set(n_rows[:5]):
            top5_setmatch += 1
        if set(q_rows[:10]) == set(n_rows[:10]):
            top10_setmatch += 1
    client.close()

    n = queries.shape[0]
    out = {
        "queries": n,
        "gallery": gallery.shape[0],
        "search": "exact brute-force (qdrant local mode)",
        "top1_agreement": top1_match / n,
        "top5_set_agreement": top5_setmatch / n,
        "top10_set_agreement": top10_setmatch / n,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print("\n=== FAITHFULNESS (identical vectors, numpy vs Qdrant) ===")
    print(f"  top-1   agreement: {top1_match}/{n}  ({out['top1_agreement']:.3f})")
    print(f"  top-5  set-agree : {top5_setmatch}/{n}  ({out['top5_set_agreement']:.3f})")
    print(f"  top-10 set-agree : {top10_setmatch}/{n}  ({out['top10_set_agreement']:.3f})")
    print(f"  saved: {RESULTS}")


if __name__ == "__main__":
    main()
