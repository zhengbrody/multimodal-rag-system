"""
Run the Flickr30k caption->image retrieval evaluation.

For each requested retriever this script builds its index over the 1,000-image
gallery, runs the 1,000 caption queries, and reports Recall@5, Recall@10, MRR and
nDCG@10. Retrievers are evaluated one at a time and their models freed between
runs so peak memory stays well under the 16GB ceiling.

Examples:
    python -m eval.evaluate --retrievers clip,mock,dense
    python -m eval.evaluate --retrievers clip --limit 100      # quick smoke test
    python -m eval.evaluate --retrievers clip,mock,dense,openai --openai

Artifacts written to eval/results/:
    baseline_flickr30k.json   full metrics + run metadata
    baseline_table.md         the markdown table cited in the README/resume
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "flickr30k"
GALLERY_PATH = DATA_DIR / "gallery.jsonl"
QUERY_FILES = {
    "1k": DATA_DIR / "eval_queries.jsonl",      # 1 held-out caption/image (proxy-safe)
    "5k": DATA_DIR / "eval_queries_5k.jsonl",   # standard protocol: all 5 captions/image
}
RESULTS_DIR = REPO_ROOT / "eval" / "results"

K_VALUES = (1, 5, 10)
NDCG_K = 10
RETRIEVE_DEPTH = max(max(K_VALUES), NDCG_K)  # how deep each retriever must return


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run the dataset prep first:\n"
            "  python -m eval.datasets.prepare_flickr30k"
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_retriever(name: str, gallery: List[Dict], queries: List[Dict]) -> Dict:
    """Build one retriever's index, run all queries, return its metrics."""
    from eval.adapters import build_adapter
    from eval.metrics import aggregate

    adapter = build_adapter(name)
    print(f"\n=== {name} ({adapter.modality}) ===")

    t0 = time.time()
    adapter.build_index(gallery)
    build_s = time.time() - t0
    print(f"  index built in {build_s:.1f}s")

    rankings: List[List[str]] = []
    relevants = [q["gt_image_id"] for q in queries]
    captions = [q["caption"] for q in queries]
    t0 = time.time()
    if hasattr(adapter, "encode_queries"):
        # Batched fast path: encode all captions once, then vector-query each.
        vecs = adapter.encode_queries(captions)
        for i, vec in enumerate(vecs):
            rankings.append(adapter.retrieve_ids_vec(vec, depth=RETRIEVE_DEPTH))
            if (i + 1) % 1000 == 0:
                print(f"  queried {i + 1}/{len(queries)}")
    else:
        for i, cap in enumerate(captions):
            rankings.append(adapter.retrieve_ids(cap, depth=RETRIEVE_DEPTH))
            if (i + 1) % 1000 == 0:
                print(f"  queried {i + 1}/{len(queries)}")
    query_s = time.time() - t0

    metrics = aggregate(rankings, relevants, k_values=K_VALUES, ndcg_k=NDCG_K)
    metrics["modality"] = adapter.modality
    metrics["build_seconds"] = round(build_s, 2)
    metrics["query_seconds"] = round(query_s, 2)

    # Free the model + release any vector-store lock before the next retriever loads.
    adapter.close()
    del adapter
    gc.collect()

    print(
        f"  Recall@5={metrics['recall@5']:.3f}  Recall@10={metrics['recall@10']:.3f}  "
        f"MRR={metrics['mrr']:.3f}  nDCG@10={metrics[f'ndcg@{NDCG_K}']:.3f}"
    )
    return metrics


def render_table(results: Dict[str, Dict]) -> str:
    # Rankings are truncated to RETRIEVE_DEPTH (=10), so "MRR" here is MRR@10:
    # queries whose gold image falls beyond rank 10 contribute 0 (bias <= ~0.01).
    # Recall@5 carries its 95% CI half-width (the headline number's error bar).
    header = (
        "| Retriever | Modality | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    label = {
        "clip": "CLIP ViT-B/32 (numpy, 1K)",
        "mock": "Mock (keyword)",
        "dense": "Dense MiniLM",
        "qdrant": "CLIP + Qdrant (1K gallery)",
        "qdrant_full": "CLIP + Qdrant (31K gallery)",
        "openai": "OpenAI text-embed-3-small",
    }
    for name, m in results.items():
        r5 = m["recall@5"]
        ci5 = m.get("recall@5_ci95", 0.0)
        rows.append(
            f"| {label.get(name, name)} | {m['modality']} | "
            f"{m['recall@1']:.3f} | {r5:.3f} ±{ci5:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m[f'ndcg@{NDCG_K}']:.3f} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Flickr30k caption->image retrieval eval.")
    parser.add_argument(
        "--retrievers",
        default="clip,mock,dense",
        help="Comma-separated: clip,mock,dense,openai (default: clip,mock,dense).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N queries (smoke test)."
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Acknowledge that the 'openai' retriever will call the OpenAI API.",
    )
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Output filename stem under eval/results/ (default: baseline). "
        "Use e.g. --tag phase1 to avoid overwriting the Phase-0 baseline.",
    )
    parser.add_argument(
        "--queries",
        default="1k",
        choices=sorted(QUERY_FILES),
        help="'1k' = 1 held-out caption/image (proxy-safe). '5k' = standard "
        "Flickr30k protocol, all 5 captions/image (use for cross-modal headline; "
        "text-proxy retrievers would leak on the rep caption).",
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    if "openai" in names and not args.openai:
        raise SystemExit("Refusing to run 'openai' retriever without --openai (it makes API calls).")
    proxy = {"mock", "dense", "openai"} & set(names)
    if args.queries == "5k" and proxy:
        raise SystemExit(
            f"Text-proxy retrievers {sorted(proxy)} leak on the 5k protocol "
            "(query caption[0] == gallery rep caption). Run them with --queries 1k."
        )

    gallery = _load_jsonl(GALLERY_PATH)
    queries = _load_jsonl(QUERY_FILES[args.queries])
    if args.limit:
        queries = queries[: args.limit]
    print(f"Gallery: {len(gallery)} images | Queries: {len(queries)} captions "
          f"({args.queries} protocol)")

    results: Dict[str, Dict] = {}
    for name in names:
        results[name] = evaluate_retriever(name, gallery, queries)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "flickr30k",
        "task": "caption->image retrieval",
        "protocol": args.queries,
        "gallery_size": len(gallery),
        "num_queries": len(queries),
        "k_values": list(K_VALUES),
        "ndcg_k": NDCG_K,
        "results": results,
    }
    json_path = RESULTS_DIR / f"{args.tag}_flickr30k.json"
    md_path = RESULTS_DIR / f"{args.tag}_table.md"
    json_path.write_text(json.dumps(payload, indent=2))
    table = render_table(results)
    md_path.write_text(
        f"# Flickr30k caption->image retrieval ({args.tag})\n\n"
        f"Queries: {len(queries)} captions · depth={RETRIEVE_DEPTH}\n\n{table}"
    )

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    print(table)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
