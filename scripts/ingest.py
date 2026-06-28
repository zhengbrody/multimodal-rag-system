"""
Ingest the full Flickr30k corpus into the pluggable vector store as CLIP vectors.

This backs the "50K+ items" and "CLIP multimodal + vector DB" resume claims with
real, reproducible artifacts. For every image in Flickr30k we store TWO vectors:

  * one IMAGE vector   (CLIP image embedding)   payload {type: image,   image_id, split}
  * one CAPTION vector (CLIP text embedding of the representative caption)
                                                payload {type: caption, image_id, split}

So the index holds 2x the image count. The nlphuji/flickr30k build used here has
31,014 images -> 62,028 vectors total (> 50K), a genuinely multimodal index that
also enables the text<->image search exercised in Phase 2. (Counts are derived
from the actual run, printed at the end; do not hard-code them elsewhere.)

Images are encoded in batches directly from the in-memory dataset (no 9GB of
JPEGs written to disk). The script reports ingest throughput, which is itself
evidence. Backend is chosen by VECTOR_BACKEND (default: local Qdrant, no keys).

Memory note (16GB): only CLIP ViT-B/32 (~600MB) + the batch tensors are resident;
the vector store streams to disk. No reranker/LLM is loaded here.

Run:
    python -m scripts.ingest                      # full corpus, local Qdrant
    python -m scripts.ingest --limit 500          # quick smoke test
    VECTOR_BACKEND=pinecone python -m scripts.ingest   # hosted (needs keys)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.vector_store import VectorStore  # noqa: E402

COLLECTION = "flickr30k_clip"
CLIP_DIM = 512
IMAGE_BATCH = 128
TEXT_BATCH = 256


def _load_clip():
    """Load CLIP ViT-B/32 on the best available device (MPS on M-series)."""
    import clip

    device = (
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading CLIP ViT-B/32 on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return clip, model, preprocess, device


@torch.no_grad()
def _encode_images(model, preprocess, device, pil_images) -> np.ndarray:
    """Preprocess a list of PIL images and return L2-normalized CLIP embeddings."""
    tensors = torch.stack([preprocess(im.convert("RGB")) for im in pil_images]).to(device)
    feats = model.encode_image(tensors)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


@torch.no_grad()
def _encode_texts(clip, model, device, texts: List[str]) -> np.ndarray:
    tokens = clip.tokenize(texts, truncate=True).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def ingest(limit: int | None = None) -> None:
    from datasets import load_dataset

    print("Loading nlphuji/flickr30k (cached after first run)...")
    ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    n_images = len(ds)
    print(f"  {n_images} images to ingest (-> {2 * n_images} vectors).")

    clip, model, preprocess, device = _load_clip()
    store = VectorStore(collection=COLLECTION, dim=CLIP_DIM)
    print(f"Vector backend: {store.backend} | collection: {COLLECTION}")
    store.recreate()

    next_id = 0
    total_img = 0
    t_start = time.time()

    # --- Pass 1: image vectors (the throughput-critical part) ---------------
    buf_imgs, buf_meta = [], []

    def flush_images():
        nonlocal next_id, total_img, buf_imgs, buf_meta
        if not buf_imgs:
            return
        vecs = _encode_images(model, preprocess, device, buf_imgs)
        ids = list(range(next_id, next_id + len(buf_imgs)))
        store.upsert(ids, vecs, buf_meta)
        next_id += len(buf_imgs)
        total_img += len(buf_imgs)
        buf_imgs, buf_meta = [], []

    for row in ds:
        image_id = str(row.get("img_id") or row.get("filename") or next_id)
        buf_imgs.append(row["image"])
        buf_meta.append({"type": "image", "image_id": image_id, "split": row.get("split", "?")})
        if len(buf_imgs) >= IMAGE_BATCH:
            flush_images()
            elapsed = time.time() - t_start
            print(f"  images {total_img}/{n_images}  ({total_img / elapsed:.0f} img/s)")
    flush_images()
    img_seconds = time.time() - t_start
    print(f"Image vectors: {total_img} in {img_seconds:.1f}s ({total_img / img_seconds:.0f} img/s)")

    # --- Pass 2: caption vectors (representative caption per image) ----------
    t_text = time.time()
    total_cap = 0
    buf_txt, buf_meta = [], []

    def flush_texts():
        nonlocal next_id, total_cap, buf_txt, buf_meta
        if not buf_txt:
            return
        vecs = _encode_texts(clip, model, device, [t for t, _ in buf_txt])
        ids = list(range(next_id, next_id + len(buf_txt)))
        store.upsert(ids, vecs, buf_meta)
        next_id += len(buf_txt)
        total_cap += len(buf_txt)
        buf_txt, buf_meta = [], []

    for row in ds:
        image_id = str(row.get("img_id") or row.get("filename") or "?")
        rep_caption = list(row["caption"])[0]
        buf_txt.append((rep_caption, image_id))
        buf_meta.append({"type": "caption", "image_id": image_id, "split": row.get("split", "?")})
        if len(buf_txt) >= TEXT_BATCH:
            flush_texts()
    flush_texts()
    print(f"Caption vectors: {total_cap} in {time.time() - t_text:.1f}s")

    # --- Report --------------------------------------------------------------
    total = store.count()
    n_image_vecs = store.count(where={"type": "image"})
    print("\n" + "=" * 60)
    print("INGEST COMPLETE")
    print("=" * 60)
    print(f"  Backend            : {store.backend}")
    print(f"  Total vectors      : {total}")
    print(f"  Image vectors      : {n_image_vecs}")
    print(f"  Caption vectors    : {store.count(where={'type': 'caption'})}")
    print(f"  Image throughput   : {total_img / img_seconds:.0f} img/s on {device}")
    print(f"  Wall time          : {time.time() - t_start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Flickr30k CLIP vectors into the store.")
    parser.add_argument("--limit", type=int, default=None, help="Cap images (smoke test).")
    args = parser.parse_args()
    ingest(limit=args.limit)


if __name__ == "__main__":
    main()
