"""
Backbone sweep — the reproducible source of the Recall@5 lift.

Phase 3 showed BM25-RRF + a text cross-encoder reranker plateau at ~0.84 (a structural
ceiling). The legitimate lever is a stronger DENSE backbone. This script makes that claim
first-class and reproducible: it runs the exact same caption->image harness (standard
5,000-caption protocol, 1K gallery) across a family of OpenCLIP backbones that differ
ONLY in model size (training data held fixed at LAION-2B), and emits one table.

Apples-to-apples: same protocol, same gallery, same queries, same metric code — the only
variable is the backbone. Recall@5 should climb monotonically with model size, with NO
reranking involved.

Run:  python -m eval.backbone_sweep            (or: make backbone-sweep)
Outputs: eval/results/backbone_sweep_flickr30k.json, eval/results/backbone_sweep_table.md
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.adapters import OpenCLIPAdapter  # noqa: E402
from eval.evaluate import (  # noqa: E402
    GALLERY_PATH,
    K_VALUES,
    NDCG_K,
    QUERY_FILES,
    RESULTS_DIR,
    RETRIEVE_DEPTH,
    _load_jsonl,
    metric_cells,
)
from eval.metrics import aggregate  # noqa: E402

# Same training data (LAION-2B), increasing model size — so size is the only variable.
BACKBONES: List[Tuple[str, str, str]] = [
    ("ViT-B-32", "laion2b_s34b_b79k", "OpenCLIP ViT-B/32 (laion2b)"),
    ("ViT-L-14", "laion2b_s32b_b82k", "OpenCLIP ViT-L/14 (laion2b)"),
    ("ViT-H-14", "laion2b_s32b_b79k", "OpenCLIP ViT-H/14 (laion2b)"),
]


def _free() -> None:
    """Release the model between backbones so peak memory stays bounded."""
    gc.collect()
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - cache clearing is best-effort
        pass


def run() -> Dict[str, Dict]:
    gallery = _load_jsonl(GALLERY_PATH)
    queries = _load_jsonl(QUERY_FILES["5k"])
    captions = [q["caption"] for q in queries]
    relevants = [q["gt_image_id"] for q in queries]
    print(f"Gallery: {len(gallery)} images | Queries: {len(queries)} captions (5k protocol)")

    results: Dict[str, Dict] = {}
    for model_name, pretrained, label in BACKBONES:
        print(f"\n=== {label} ===")
        # use_cache=False: each backbone has a different dim/embedding, so they must not
        # share the single on-disk gallery cache.
        adapter = OpenCLIPAdapter(model_name=model_name, pretrained=pretrained, use_cache=False)
        t0 = time.time()
        adapter.build_index(gallery)
        vecs = adapter.encode_queries(captions)
        rankings = [adapter.retrieve_ids_vec(v, depth=RETRIEVE_DEPTH) for v in vecs]
        metrics: Dict[str, Any] = aggregate(rankings, relevants, k_values=K_VALUES, ndcg_k=NDCG_K)
        metrics["backbone"] = f"{model_name} ({pretrained})"
        metrics["seconds"] = round(time.time() - t0, 1)
        results[label] = metrics
        print(
            f"  Recall@5={metrics['recall@5']:.3f} ±{metrics.get('recall@5_ci95', 0):.3f}  "
            f"Recall@10={metrics['recall@10']:.3f}  ({metrics['seconds']}s)"
        )
        adapter.close()
        del adapter
        _free()
    return results


def render_table(results: Dict[str, Dict]) -> str:
    header = (
        "| Dense backbone | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = [f"| {label} | {metric_cells(m)} |" for label, m in results.items()]
    return header + "\n".join(rows) + "\n"


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "flickr30k",
        "task": "caption->image retrieval",
        "protocol": "5k",
        "gallery_size": 1000,
        "note": "Same protocol/gallery/queries; only the dense backbone size varies "
        "(LAION-2B weights throughout). No reranking. Lift is from the backbone.",
        "results": results,
    }
    (RESULTS_DIR / "backbone_sweep_flickr30k.json").write_text(json.dumps(payload, indent=2))
    table = render_table(results)
    (RESULTS_DIR / "backbone_sweep_table.md").write_text(
        "# Backbone sweep — Recall@5 lift comes from the dense backbone\n\n"
        "Flickr30k caption→image, standard 5,000-caption protocol, 1K gallery. Only the\n"
        "backbone size varies (LAION-2B weights throughout); **no reranking**.\n\n" + table
    )
    print("\n" + "=" * 70)
    print(table)
    print(f"Saved: {RESULTS_DIR / 'backbone_sweep_table.md'}")


if __name__ == "__main__":
    main()
