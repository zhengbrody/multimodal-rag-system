"""
Prepare the Flickr30k caption->image retrieval eval set.

Uses the public ``nlphuji/flickr30k`` dataset, which ships all 31,783 images with
5 captions each and a ``split`` column following the standard Karpathy split. We
keep the 1,000 **test** images as the retrieval gallery (the standard 1K Karpathy
test gallery) and build a 1,000-query evaluation set (one held-out caption per
image). Note: the full standard protocol uses all 5,000 captions (5 per image) as
queries; this is a 1,000-query *subsample* (one seeded caption per image). Mean
per-query recall is an unbiased estimator of the 5,000-query mean, so the point
estimate is comparable to published numbers, but with a wider confidence interval
(~+/-0.025 at n=1,000). For a tighter, fully-standard run, generate all 5 captions.

Outputs (under data/flickr30k/, git-ignored):
    images/<image_id>.jpg         the 1,000 gallery images
    gallery.jsonl                 {image_id, image_path, rep_caption}
    eval_queries.jsonl            {query_id, caption, gt_image_id}

The script is idempotent: if the manifests already exist with the expected row
counts it exits early. Query selection is seeded for reproducibility.

Run:  python -m eval.datasets.prepare_flickr30k
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "flickr30k"
IMAGES_DIR = DATA_DIR / "images"
GALLERY_PATH = DATA_DIR / "gallery.jsonl"
QUERIES_PATH = DATA_DIR / "eval_queries.jsonl"
QUERIES_5K_PATH = DATA_DIR / "eval_queries_5k.jsonl"

SEED = 42


def _already_prepared(expected: int) -> bool:
    if not (GALLERY_PATH.exists() and QUERIES_PATH.exists() and QUERIES_5K_PATH.exists()):
        return False
    try:
        g = sum(1 for _ in GALLERY_PATH.open())
        q = sum(1 for _ in QUERIES_PATH.open())
        q5 = sum(1 for _ in QUERIES_5K_PATH.open())
    except OSError:
        return False
    return g == expected and q == expected and q5 == expected * 5


def prepare(max_images: int | None = None, force: bool = False) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "The 'datasets' package is required. Install eval deps with:\n"
            "  pip install -r requirements_eval.txt"
        ) from exc

    expected = max_images or 1000
    if not force and _already_prepared(expected):
        print(
            f"Eval set already prepared at {DATA_DIR} ({expected} images). "
            "Use --force to rebuild."
        )
        return

    print("Loading nlphuji/flickr30k (this downloads ~4GB on first run)...")
    # The dataset exposes a single 'test' split that contains ALL images; the
    # per-image 'split' column carries the Karpathy assignment.
    ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)

    print("Filtering to the Karpathy 'test' split (the 1K retrieval gallery)...")
    ds = ds.filter(lambda row: row["split"] == "test")
    print(f"  {len(ds)} test images found.")

    if max_images is not None:
        ds = ds.select(range(min(max_images, len(ds))))
        print(f"  Limiting to {len(ds)} images (--max-images).")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    gallery_rows = []
    query_rows = []
    query_rows_5k = []  # standard protocol: ALL 5 captions per image as queries

    for i, row in enumerate(ds):
        # nlphuji uses 'img_id' (string) and 'filename'; fall back gracefully.
        image_id = str(row.get("img_id") or row.get("filename") or f"img_{i}")
        captions = list(row["caption"])
        if len(captions) < 2:
            # Need at least one caption for the gallery and one held out as query.
            continue

        image_path = IMAGES_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            row["image"].convert("RGB").save(image_path, format="JPEG", quality=90)

        rep_caption = captions[0]  # representative caption -> text proxy corpus
        # Hold out a *different* caption as the query (no leakage into proxy corpus).
        query_caption = rng.choice(captions[1:])

        gallery_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.relative_to(REPO_ROOT)),
                "rep_caption": rep_caption,
            }
        )
        query_rows.append(
            {
                "query_id": f"q{i}",
                "caption": query_caption,
                "gt_image_id": image_id,
            }
        )
        # Standard 5,000-caption protocol: every caption is a query. caption_idx
        # is recorded so text-proxy baselines can drop idx==0 (== rep_caption) to
        # avoid exact-string leakage; the cross-modal CLIP path has no such leak.
        for cidx, cap in enumerate(captions):
            query_rows_5k.append(
                {
                    "query_id": f"q{i}_{cidx}",
                    "caption": cap,
                    "gt_image_id": image_id,
                    "caption_idx": cidx,
                }
            )

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1} images...")

    with GALLERY_PATH.open("w") as f:
        for r in gallery_rows:
            f.write(json.dumps(r) + "\n")
    with QUERIES_PATH.open("w") as f:
        for r in query_rows:
            f.write(json.dumps(r) + "\n")
    with QUERIES_5K_PATH.open("w") as f:
        for r in query_rows_5k:
            f.write(json.dumps(r) + "\n")

    print("\nDone.")
    print(f"  Gallery images   : {len(gallery_rows)}  -> {IMAGES_DIR}")
    print(f"  Gallery manifest : {GALLERY_PATH}")
    print(f"  Eval queries (1k): {len(query_rows)}  -> {QUERIES_PATH}")
    print(f"  Eval queries (5k): {len(query_rows_5k)}  -> {QUERIES_5K_PATH}  (standard protocol)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Flickr30k caption->image eval set.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the number of gallery images (for a quick smoke test).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if manifests exist.")
    args = parser.parse_args()
    prepare(max_images=args.max_images, force=args.force)


if __name__ == "__main__":
    main()
