"""
Adapters that wrap each retriever behind a uniform interface for evaluation.

Every adapter exposes:
    build_index(gallery_rows)            # ingest the eval gallery once
    retrieve_ids(text, depth) -> [id]    # ranked list of gallery image_ids

This hides the retrievers' inconsistencies (different default thresholds, an
extra ``weighted_score`` field on CLIP/PersonalRAGRetriever, image vs. text
ingestion) so eval/evaluate.py stays retriever-agnostic.

Modalities:
  * CLIPAdapter    -- TRUE cross-modal: gallery = images, query = caption text
                      (OpenAI CLIP ViT-B/32, 512-d).
  * OpenCLIPAdapter-- TRUE cross-modal with a stronger backbone (OpenCLIP ViT-H/14,
                      laion2b, 1024-d) — the Phase-3b lever to ~0.92-0.93 R@5.
  * MockAdapter    -- text->text PROXY: gallery = one representative caption per
                      image (keyword/Jaccard lexical baseline).
  * DenseAdapter   -- text->text PROXY: local sentence-transformer (MiniLM) over
                      the representative captions (dense-semantic baseline).
  * OpenAIAdapter  -- text->text PROXY via the existing OpenAI retriever.py
                      (only when OPENAI_API_KEY is set / --openai is passed).

The vector-DB variants (qdrant*, qdrant_openclip*) serve a gallery ingested once by
scripts/ingest.py. Note the 1024-d OpenCLIP vectors live in their OWN collection;
mixing them with the 512-d ViT-B/32 vectors in one collection is a dimensionality bug.

Memory note (16GB ceiling): adapters load their model in build_index and the
runner evaluates them one at a time, freeing each before the next. CLIP ViT-B/32
(~600MB) and MiniLM (~90MB) are never resident together. OpenCLIP ViT-H/14 is
~3.9GB (fp32) — run it as the only model loaded; never alongside the reranker/LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.knowledge_processor import Document  # noqa: E402

CLIP_GALLERY_CACHE = REPO_ROOT / "data" / "flickr30k" / "clip_gallery.pkl"
OPENCLIP_GALLERY_CACHE = REPO_ROOT / "data" / "flickr30k" / "openclip_gallery.pkl"


class RetrieverAdapter:
    """Base interface. Subclasses implement build_index and retrieve_ids."""

    name = "base"
    modality = "text->text"

    def build_index(self, gallery_rows: List[Dict]) -> None:
        raise NotImplementedError

    def retrieve_ids(self, text: str, depth: int) -> List[str]:
        raise NotImplementedError

    def close(self) -> None:
        """Optional teardown (e.g. release a vector-store file lock). No-op by default."""
        return None


class _InMemoryCLIPAdapter(RetrieverAdapter):
    """
    Template for numpy-fast-path cross-modal adapters that OWN an in-memory gallery
    (CLIP ViT-B/32, OpenCLIP ViT-H/14). Subclasses set ``cache_path`` and implement
    ``_make_retriever()``; the cache-or-build flow and the batched fast path live here.

    (Vector-DB adapters use a separate base — they query an external store rather than
    owning the gallery, a genuinely different storage strategy; don't merge across that axis.)
    """

    cache_path: Optional[Path] = None

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._retriever: Any = None  # CLIPRetriever | OpenCLIPRetriever, loaded lazily

    def _make_retriever(self) -> Any:
        raise NotImplementedError

    def build_index(self, gallery_rows: List[Dict]) -> None:
        self._retriever = self._make_retriever()

        if self.use_cache and self.cache_path and self.cache_path.exists():
            self._retriever.load(str(self.cache_path))
            if len(self._retriever.documents) == len(gallery_rows):
                return
            # Cache size mismatch -> rebuild in place (reuse the already-loaded model
            # instead of reconstructing the retriever, which would reload the weights).
            self._retriever.documents = []
            self._retriever.embeddings = None

        image_paths = [str(REPO_ROOT / r["image_path"]) for r in gallery_rows]
        metadata = [{"image_id": r["image_id"], "category": "image"} for r in gallery_rows]
        self._retriever.add_images(image_paths, metadata)

        if self.use_cache and self.cache_path:
            self._retriever.save(str(self.cache_path))

    def retrieve_ids(self, text: str, depth: int) -> List[str]:
        # threshold=-1.0 so the full top-`depth` is always returned (never truncated).
        results = self._retriever.retrieve(text, k=depth, threshold=-1.0)
        return [r["metadata"]["image_id"] for r in results]

    # --- batched fast path (avoids per-query encode at 5K-query scale) ---
    def encode_queries(self, texts: List[str]):
        return np.asarray(self._retriever._get_embeddings_batch(texts), dtype=np.float32)

    def retrieve_ids_vec(self, vec, depth: int) -> List[str]:
        emb = self._retriever.embeddings
        sims = emb @ (vec / np.linalg.norm(vec))
        top = np.argsort(sims)[::-1][:depth]
        return [self._retriever.documents[i]["metadata"]["image_id"] for i in top]


class CLIPAdapter(_InMemoryCLIPAdapter):
    """True cross-modal caption->image retrieval with CLIP ViT-B/32."""

    name = "clip"
    modality = "text->image (cross-modal)"
    cache_path = CLIP_GALLERY_CACHE

    def __init__(self, model_name: str = "ViT-B/32", use_cache: bool = True):
        super().__init__(use_cache=use_cache)
        self.model_name = model_name

    def _make_retriever(self) -> Any:
        from src.rag.clip_retriever import CLIPRetriever

        return CLIPRetriever(model_name=self.model_name)


class OpenCLIPAdapter(_InMemoryCLIPAdapter):
    """
    Cross-modal caption->image with a stronger OpenCLIP backbone (default ViT-H/14,
    laion2b) — the Phase-3b lever to the ~0.92-0.93 R@5 headline. 1024-d embeddings in
    a SEPARATE cache file so it never collides with the 512-d ViT-B/32 gallery.
    """

    name = "openclip"
    modality = "text->image (cross-modal, ViT-H/14)"
    cache_path = OPENCLIP_GALLERY_CACHE

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        use_cache: bool = True,
    ):
        super().__init__(use_cache=use_cache)
        self.model_name = model_name
        self.pretrained = pretrained

    def _make_retriever(self) -> Any:
        from src.rag.openclip_retriever import OpenCLIPRetriever

        return OpenCLIPRetriever(self.model_name, self.pretrained)


class MockAdapter(RetrieverAdapter):
    """Lexical (keyword/Jaccard) text->text proxy via the existing MockRetriever."""

    name = "mock"
    modality = "text->text (lexical proxy)"

    def __init__(self):
        self._retriever: Any = None  # lazily-loaded retriever (type varies by adapter)

    def build_index(self, gallery_rows: List[Dict]) -> None:
        from src.rag.mock_retriever import MockRetriever

        self._retriever = MockRetriever()
        docs = [
            Document(
                content=r["rep_caption"],
                metadata={"image_id": r["image_id"], "category": "image", "type": "image"},
            )
            for r in gallery_rows
        ]
        self._retriever.add_documents(docs)

    def retrieve_ids(self, text: str, depth: int) -> List[str]:
        results = self._retriever.retrieve(text, k=depth, threshold=0.0)
        return [r["metadata"]["image_id"] for r in results]


class DenseAdapter(RetrieverAdapter):
    """Local dense-text baseline using a sentence-transformer over rep captions."""

    name = "dense"
    modality = "text->text (dense proxy, local)"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self._emb: Optional[np.ndarray] = None
        self._ids: List[str] = []

    def build_index(self, gallery_rows: List[Dict]) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        captions = [r["rep_caption"] for r in gallery_rows]
        self._ids = [r["image_id"] for r in gallery_rows]
        # Normalized embeddings -> cosine similarity is a plain dot product.
        self._emb = self._model.encode(
            captions, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )

    def retrieve_ids(self, text: str, depth: int) -> List[str]:
        q = self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self._emb @ q
        top = np.argsort(sims)[::-1][:depth]
        return [self._ids[i] for i in top]


class QdrantCLIPAdapter(RetrieverAdapter):
    """
    CLIP caption->image retrieval served from the persisted vector store.

    The gallery is NOT rebuilt here — it was ingested once by scripts/ingest.py
    into the vector store (local Qdrant by default). This adapter only loads the
    CLIP text encoder to embed queries, then searches the store with a metadata
    filter. ``split_filter='test'`` restricts the gallery to the same 1,000 test
    images as the Phase-0 numpy baseline (faithfulness check); ``None`` searches
    the full 31,783-image gallery (the harder, large-corpus eval).
    """

    name = "qdrant"
    modality = "text->image (vector DB, 1K gallery)"
    COLLECTION = "flickr30k_clip"
    CLIP_DIM = 512

    def __init__(self, split_filter: Optional[str] = "test"):
        self.split_filter = split_filter
        self._clip: Any = None
        self._store: Any = None

    def build_index(self, gallery_rows: List[Dict]) -> None:
        from src.rag.clip_retriever import CLIPRetriever
        from src.rag.vector_store import VectorStore

        # Reuse the project's CLIP encoder purely for query-side text embeddings.
        self._clip = CLIPRetriever(model_name="ViT-B/32")
        self._store = VectorStore(collection=self.COLLECTION, dim=self.CLIP_DIM)
        gallery = self._store.count(
            where={"type": "image", **({"split": self.split_filter} if self.split_filter else {})}
        )
        print(f"  vector store ready: {gallery} image vectors in gallery scope")
        if gallery == 0:
            raise SystemExit(
                "Vector store is empty. Run the ingest first:\n  python -m scripts.ingest"
            )

    def _where(self) -> Dict:
        where = {"type": "image"}
        if self.split_filter:
            where["split"] = self.split_filter
        return where

    def retrieve_ids(self, text: str, depth: int) -> List[str]:
        vec = self._clip._get_text_embedding(text)
        hits = self._store.query(vec, k=depth, where=self._where())
        return [h.payload["image_id"] for h in hits]

    # --- batched fast path: encode all query captions in one pass ---
    def encode_queries(self, texts: List[str]):
        return np.asarray(self._clip._get_embeddings_batch(texts), dtype=np.float32)

    def retrieve_ids_vec(self, vec, depth: int) -> List[str]:
        hits = self._store.query(vec, k=depth, where=self._where())
        return [h.payload["image_id"] for h in hits]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


class QdrantCLIPFullAdapter(QdrantCLIPAdapter):
    """Same vector-DB retrieval but against the FULL 31,783-image gallery."""

    name = "qdrant_full"
    modality = "text->image (vector DB, 31K gallery)"

    def __init__(self):
        super().__init__(split_filter=None)


class QdrantOpenCLIPAdapter(QdrantCLIPAdapter):
    """
    OpenCLIP (ViT-H/14) caption->image served from the persisted vector store.

    Same vector-DB flow as QdrantCLIPAdapter, but the gallery was ingested with the
    1024-d ViT-H/14 backbone into its OWN collection (see scripts/ingest.py
    --backbone open_clip). Query-side text is embedded with the matching OpenCLIP
    encoder. ``split_filter='test'`` = 1K gallery; ``None`` = full 31K corpus.
    """

    name = "qdrant_openclip"
    modality = "text->image (vector DB, ViT-H/14, 1K gallery)"
    COLLECTION = "flickr30k_openclip"
    CLIP_DIM = 1024  # ViT-H/14 embedding dim (NOT 512)

    def build_index(self, gallery_rows: List[Dict]) -> None:
        from src.rag.openclip_retriever import OpenCLIPRetriever
        from src.rag.vector_store import VectorStore

        # OpenCLIP encoder for query-side text embeddings only (gallery already ingested).
        self._clip = OpenCLIPRetriever("ViT-H-14", "laion2b_s32b_b79k")
        self._store = VectorStore(collection=self.COLLECTION, dim=self.CLIP_DIM)
        gallery = self._store.count(
            where={"type": "image", **({"split": self.split_filter} if self.split_filter else {})}
        )
        print(f"  vector store ready: {gallery} image vectors in gallery scope")
        if gallery == 0:
            raise SystemExit(
                "OpenCLIP vector store is empty. Run the ViT-H/14 ingest first:\n"
                "  python -m scripts.ingest --backbone open_clip"
            )


class QdrantOpenCLIPFullAdapter(QdrantOpenCLIPAdapter):
    """OpenCLIP ViT-H/14 vector-DB retrieval against the FULL 31K-image gallery."""

    name = "qdrant_openclip_full"
    modality = "text->image (vector DB, ViT-H/14, 31K gallery)"

    def __init__(self):
        super().__init__(split_filter=None)


class OpenAIAdapter(RetrieverAdapter):
    """Hosted dense-text proxy via the existing OpenAI retriever (text-embedding-3-small)."""

    name = "openai"
    modality = "text->text (OpenAI embeddings proxy)"

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self._retriever: Any = None  # lazily-loaded retriever (type varies by adapter)

    def build_index(self, gallery_rows: List[Dict]) -> None:
        from src.rag.retriever import SimpleRetriever

        self._retriever = SimpleRetriever(embedding_model=self.embedding_model)
        docs = [
            Document(
                content=r["rep_caption"],
                metadata={"image_id": r["image_id"], "category": "image", "type": "image"},
            )
            for r in gallery_rows
        ]
        # SimpleRetriever builds its embedding index from added documents.
        self._retriever.add_documents(docs)

    def retrieve_ids(self, text: str, depth: int) -> List[str]:
        results = self._retriever.retrieve(text, k=depth, threshold=-1.0)
        return [r["metadata"]["image_id"] for r in results]


ADAPTERS: Dict[str, Type[RetrieverAdapter]] = {
    "clip": CLIPAdapter,
    "openclip": OpenCLIPAdapter,
    "mock": MockAdapter,
    "dense": DenseAdapter,
    "qdrant": QdrantCLIPAdapter,
    "qdrant_full": QdrantCLIPFullAdapter,
    "qdrant_openclip": QdrantOpenCLIPAdapter,
    "qdrant_openclip_full": QdrantOpenCLIPFullAdapter,
    "openai": OpenAIAdapter,
}


def build_adapter(name: str) -> RetrieverAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"Unknown retriever '{name}'. Choices: {sorted(ADAPTERS)}")
    return ADAPTERS[name]()
