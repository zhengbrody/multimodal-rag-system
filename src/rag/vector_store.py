"""
Pluggable vector-store layer for scalable similarity search.

This decouples *where vectors live* (the backend) from *how they are produced*
(the embedder). The existing ``PineconeRetriever`` is hard-wired to OpenAI text
embeddings (1536-d, text only) and therefore cannot back the multimodal
caption->image task. This layer accepts pre-computed vectors of any dimension
(e.g. 512-d CLIP) and exposes a single small interface over two backends:

    backend="local"     -> Qdrant in embedded/on-disk mode (no server, no keys)
    backend="pinecone"  -> Pinecone serverless (needs PINECONE_API_KEY)

Both backends share the ``upsert`` / ``query`` / ``count`` contract and the same
simple metadata filter dict (e.g. ``{"type": "image"}``). Honest caveats:

  * qdrant-client local/embedded mode does EXACT brute-force search (no HNSW), so
    its results are exact, not approximate. For an approximate (ANN) index, run
    Qdrant as a server or use the Pinecone backend — and measure ANN recall vs
    this exact baseline separately.
  * ``count(where=...)`` is filtered on the local backend but UNFILTERED on
    Pinecone (it returns the index-wide total); see ``_PineconeBackend.count``.
  * The Pinecone backend is implemented behind the same interface but, without a
    PINECONE_API_KEY, is not exercised in CI/eval. It needs ``pinecone>=3``.

Selected via the ``VECTOR_BACKEND`` env var (default ``local``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_PATH = REPO_ROOT / "data" / "flickr30k" / "qdrant"


@dataclass
class Hit:
    """A single search result: id, similarity score, and stored metadata."""

    id: str
    score: float
    payload: Dict


class VectorStore:
    """Backend-agnostic vector store. Caller supplies the embeddings."""

    def __init__(
        self,
        collection: str,
        dim: int,
        backend: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        self.collection = collection
        self.dim = dim
        self.backend = (backend or os.getenv("VECTOR_BACKEND") or "local").lower()
        self._impl = self._make_backend(local_path)

    # -- backend construction -------------------------------------------------

    def _make_backend(self, local_path: Optional[str]):
        if self.backend == "local":
            return _QdrantBackend(self.collection, self.dim, local_path or str(DEFAULT_LOCAL_PATH))
        if self.backend == "pinecone":
            return _PineconeBackend(self.collection, self.dim)
        raise ValueError(f"Unknown VECTOR_BACKEND '{self.backend}'. Use 'local' or 'pinecone'.")

    # -- public API -----------------------------------------------------------

    def recreate(self) -> None:
        """Drop and re-create the collection (clean slate for a fresh ingest)."""
        self._impl.recreate()

    def upsert(
        self,
        ids: Sequence[int],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Dict],
        batch_size: int = 512,
    ) -> None:
        """Insert/overwrite vectors with metadata payloads, batched."""
        self._impl.upsert(ids, vectors, payloads, batch_size)

    def query(self, vector: Sequence[float], k: int, where: Optional[Dict] = None) -> List[Hit]:
        """Top-k nearest neighbours, optionally filtered by an exact-match metadata dict."""
        return cast(List[Hit], self._impl.query(vector, k, where))

    def count(self, where: Optional[Dict] = None) -> int:
        """Number of stored vectors (optionally matching a metadata filter)."""
        return int(self._impl.count(where))

    def close(self) -> None:
        """Release the backend (frees the local Qdrant file lock)."""
        close = getattr(self._impl, "close", None)
        if close:
            close()


# ---------------------------------------------------------------------------
# Local backend: Qdrant embedded (on-disk, no server)
# ---------------------------------------------------------------------------


class _QdrantBackend:
    def __init__(self, collection: str, dim: int, path: str):
        from qdrant_client import QdrantClient, models

        self._models = models
        self.collection = collection
        self.dim = dim
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=path)
        if not self.client.collection_exists(collection):
            self._create()

    def _create(self):
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=self._models.VectorParams(
                size=self.dim, distance=self._models.Distance.COSINE
            ),
        )

    def recreate(self):
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._create()

    def _filter(self, where: Optional[Dict]):
        if not where:
            return None
        return self._models.Filter(
            must=[
                self._models.FieldCondition(key=key, match=self._models.MatchValue(value=value))
                for key, value in where.items()
            ]
        )

    def upsert(self, ids, vectors, payloads, batch_size):
        models = self._models
        n = len(ids)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = [
                models.PointStruct(
                    id=int(ids[i]),
                    vector=list(map(float, vectors[i])),
                    payload=payloads[i],
                )
                for i in range(start, end)
            ]
            self.client.upsert(collection_name=self.collection, points=points)

    def query(self, vector, k, where=None) -> List[Hit]:
        res = self.client.query_points(
            collection_name=self.collection,
            query=list(map(float, vector)),
            limit=k,
            query_filter=self._filter(where),
            with_payload=True,
        ).points
        return [Hit(id=str(p.id), score=float(p.score), payload=dict(p.payload or {})) for p in res]

    def count(self, where=None) -> int:
        return int(
            self.client.count(
                collection_name=self.collection, count_filter=self._filter(where)
            ).count
        )

    def close(self):
        # Releases the embedded-storage lock so another client can open the path.
        try:
            self.client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Hosted backend: Pinecone serverless (wired; exercised only when keys present)
# ---------------------------------------------------------------------------


class _PineconeBackend:
    """
    Experimental / optional hosted backend. Demonstrates the pluggable-backend design
    behind the same VectorStore interface, but is NOT exercised in CI/eval (the default
    'local' Qdrant backend needs no keys). Requires PINECONE_API_KEY and pinecone>=3.
    """

    def __init__(self, collection: str, dim: int):
        # Check the key first so the common "no key" case fails fast with a clear message,
        # before the optional dependency is even imported.
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "VECTOR_BACKEND=pinecone but PINECONE_API_KEY is not set. Pinecone is an "
                "experimental/optional backend; the default 'local' Qdrant backend needs no keys."
            )
        try:
            from pinecone import Pinecone, ServerlessSpec  # lazy: only when selected
        except ImportError as e:
            raise ImportError(
                "VECTOR_BACKEND=pinecone requires the optional 'pinecone-client>=3' package."
            ) from e
        self.collection = collection
        self.dim = dim
        self._pc = Pinecone(api_key=api_key)
        if collection not in [i.name for i in self._pc.list_indexes()]:
            self._pc.create_index(
                name=collection,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Serverless index creation is async — wait until it is queryable,
            # otherwise the first upsert/recreate races a not-ready index.
            self._wait_ready(collection)
        self.index = self._pc.Index(collection)

    def _wait_ready(self, collection: str, timeout_s: int = 120) -> None:
        import time

        for _ in range(timeout_s):
            try:
                if self._pc.describe_index(collection).status.get("ready"):
                    return
            except Exception:
                pass
            time.sleep(1)

    def recreate(self):
        # delete_all 404s on a brand-new/empty namespace — tolerate it.
        try:
            self.index.delete(delete_all=True)
        except Exception:
            pass

    @staticmethod
    def _filter(where: Optional[Dict]):
        if not where:
            return None
        return {key: {"$eq": value} for key, value in where.items()}

    def upsert(self, ids, vectors, payloads, batch_size):
        n = len(ids)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            vecs = [
                (str(ids[i]), list(map(float, vectors[i])), payloads[i]) for i in range(start, end)
            ]
            self.index.upsert(vectors=vecs)

    def query(self, vector, k, where=None) -> List[Hit]:
        resp = self.index.query(
            vector=list(map(float, vector)),
            top_k=k,
            include_metadata=True,
            filter=self._filter(where),
        )
        return [
            Hit(id=str(m["id"]), score=float(m["score"]), payload=dict(m.get("metadata", {})))
            for m in resp.get("matches", [])
        ]

    def count(self, where=None) -> int:
        # NOTE: Pinecone has no cheap filtered count; this returns the index-wide
        # total and IGNORES `where`. Do not rely on filtered counts on this backend.
        return int(self.index.describe_index_stats().get("total_vector_count", 0))
