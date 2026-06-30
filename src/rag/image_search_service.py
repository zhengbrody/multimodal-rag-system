"""
Lazy, gated multimodal image-search service backing the /search/* API endpoints.

This is the bridge from the *measured* cross-modal retrieval (the Phase-3b eval work)
to the *served* API: it reuses the exact OpenCLIP ViT-H/14 gallery the harness already
built and cached (``data/flickr30k/openclip_gallery.pkl``), falling back to the CLIP
ViT-B/32 gallery if that's the only one present.

Design constraints (do not break either deployment):
  * **Import-safe under requirements_simple.txt** — NO torch/open_clip at module load.
    Heavy deps are imported only inside ``ensure_loaded``. So importing this module — and
    therefore the FastAPI app and the CI tests — never fails on the lightweight mock build.
  * **Gated**: ``available()`` is True only when a gallery cache exists on disk. The
    endpoints return 503 otherwise, so the zero-cost Streamlit Cloud (mock) deployment
    keeps running without ever loading a ~4GB model.
  * **Loaded once, on first use** (not at import, not at app startup) — thread-safe.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "flickr30k"

# Preference order: the strongest backbone whose gallery is present wins.
# (label, cache file, backend key)
_BACKENDS: List[Tuple[str, Path, str]] = [
    ("openclip-vit-h14", DATA_DIR / "openclip_gallery.pkl", "openclip"),
    ("clip-vit-b32", DATA_DIR / "clip_gallery.pkl", "clip"),
]


class ImageSearchService:
    """Serves caption->image and image->image search over a cached CLIP/OpenCLIP gallery."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._retriever: Any = None  # OpenCLIPRetriever | CLIPRetriever, loaded lazily
        self.backend_label: Optional[str] = None
        self._captions: Dict[str, str] = {}

    @staticmethod
    def _which_cache() -> Optional[Tuple[str, Path, str]]:
        for label, cache, key in _BACKENDS:
            if cache.exists():
                return label, cache, key
        return None

    @classmethod
    def available(cls) -> bool:
        """True if a gallery cache exists. (Backend deps are checked lazily on load.)"""
        return cls._which_cache() is not None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            picked = self._which_cache()
            if picked is None:
                raise RuntimeError(
                    "No gallery cache found. Build it first:\n"
                    "  python -m eval.datasets.prepare_flickr30k\n"
                    "  python -m eval.evaluate --retrievers openclip --queries 5k --tag phase3b_headline"
                )
            label, cache, key = picked
            if key == "openclip":
                from .openclip_retriever import OpenCLIPRetriever  # heavy import, lazy

                retriever: Any = OpenCLIPRetriever()
            else:
                from .clip_retriever import CLIPRetriever  # heavy import, lazy

                retriever = CLIPRetriever(model_name="ViT-B/32")
            retriever.load(str(cache))
            self._retriever = retriever
            self.backend_label = label
            self._load_captions()
            self._loaded = True

    def _load_captions(self) -> None:
        """Map image_id -> representative caption for nicer results (optional)."""
        gallery_path = DATA_DIR / "gallery.jsonl"
        if not gallery_path.exists():
            return
        with gallery_path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    self._captions[str(row["image_id"])] = row.get("rep_caption", "")

    def _format(self, docs: List[Dict]) -> List[Dict]:
        results = []
        for d in docs:
            image_id = str(d["metadata"].get("image_id"))
            results.append(
                {
                    "image_id": image_id,
                    "score": round(float(d["score"]), 4),
                    "rep_caption": self._captions.get(image_id),
                }
            )
        return results

    def search_text(self, query: str, k: int = 5) -> List[Dict]:
        """Caption -> image: rank gallery images by similarity to the query text."""
        self.ensure_loaded()
        return self._format(self._retriever.retrieve(query, k=k, threshold=-1.0))

    def search_image(self, image_path: str, k: int = 5) -> List[Dict]:
        """Image -> image: rank gallery images by similarity to a query image file."""
        self.ensure_loaded()
        return self._format(self._retriever.retrieve_by_image(image_path, k=k))

    def info(self) -> Dict:
        """Status for the endpoints / health surface (no heavy load triggered)."""
        picked = self._which_cache()
        return {
            "available": picked is not None,
            "loaded": self._loaded,
            "backend": self.backend_label or (picked[0] if picked else None),
            "gallery_size": (
                len(self._retriever.documents) if self._loaded and self._retriever else None
            ),
        }
