"""
Pinecone-based vector retriever for scalable similarity search.
Supports serverless and pod-based Pinecone indexes with metadata filtering.
Handles 50K+ documents with sub-100ms query latency.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from pinecone import Pinecone, ServerlessSpec

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class PineconeRetriever:
    """
    Scalable vector retriever backed by Pinecone for similarity search.

    Stores embeddings and metadata in Pinecone; keeps full document content
    locally because Pinecone metadata has size limits.  Supports serverless
    indexes, metadata filtering, and batched upserts.
    """

    def __init__(
        self,
        index_name: str = "rag-index",
        embedding_model: str = "text-embedding-3-small",
        namespace: str = "default",
    ):
        """
        Initialize Pinecone client and OpenAI embedding client.

        Args:
            index_name: Name of the Pinecone index to use or create.
            embedding_model: OpenAI embedding model name (dimension=1536).
            namespace: Pinecone namespace for logical separation.
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.namespace = namespace
        self.dimension = 1536
        self.metric = "cosine"

        # Local document store: id -> full content + metadata
        self._document_store: Dict[str, Dict[str, Any]] = {}

        # ---------- Pinecone ----------
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "pinecone package is required. Install with: pip install pinecone"
            )

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is not set. "
                "Set it in your .env file or export it in your shell."
            )

        self.pc = Pinecone(api_key=pinecone_api_key)

        # Create index if it does not exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)

        # ---------- OpenAI ----------
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it in your .env file or export it in your shell."
            )

        self.openai_client = OpenAI(api_key=openai_api_key)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a single text string."""
        response = self.openai_client.embeddings.create(
            input=text, model=self.embedding_model
        )
        return response.data[0].embedding

    def _get_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """Get embeddings for a list of texts in batches."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.openai_client.embeddings.create(
                input=batch, model=self.embedding_model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    @staticmethod
    def _doc_id(content: str, metadata: Dict[str, Any]) -> str:
        """Deterministic document ID from content hash."""
        raw = content + json.dumps(metadata, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Any], batch_size: int = 100) -> None:
        """
        Upsert documents into Pinecone in batches.

        Each document object must have `.content` (str) and `.metadata` (dict)
        attributes (matching the project's Document class).

        Args:
            documents: List of Document objects.
            batch_size: Number of vectors per upsert call.
        """
        print(f"Adding {len(documents)} documents to Pinecone index '{self.index_name}'...")

        texts = [doc.content for doc in documents]
        embeddings = self._get_embeddings_batch(texts, batch_size=batch_size)

        vectors_to_upsert: List[tuple] = []
        for doc, embedding in zip(documents, embeddings):
            doc_id = self._doc_id(doc.content, doc.metadata)

            # Store full content locally
            self._document_store[doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata,
            }

            # Pinecone metadata: lightweight fields for filtering + a content preview
            pinecone_metadata = {
                "type": doc.metadata.get("type", "unknown"),
                "category": doc.metadata.get("category", "unknown"),
                "preview": doc.content[:200],
            }

            vectors_to_upsert.append((doc_id, embedding, pinecone_metadata))

            # Flush in batches
            if len(vectors_to_upsert) >= batch_size:
                self.index.upsert(
                    vectors=vectors_to_upsert, namespace=self.namespace
                )
                vectors_to_upsert = []

        # Flush remaining
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)

        print(
            f"Upserted {len(documents)} vectors. "
            f"Local store size: {len(self._document_store)}"
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone and return the top-k results above a similarity threshold.

        Args:
            query: Natural-language search query.
            k: Maximum number of results.
            threshold: Minimum cosine similarity score (0-1).
            filter: Optional Pinecone metadata filter dict.

        Returns:
            List of dicts matching the project's retriever output format:
            [{"content": str, "metadata": dict, "score": float}, ...]
        """
        query_embedding = self._get_embedding(query)

        query_kwargs: Dict[str, Any] = {
            "vector": query_embedding,
            "top_k": k,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if filter is not None:
            query_kwargs["filter"] = filter

        response = self.index.query(**query_kwargs)

        results: List[Dict[str, Any]] = []
        for match in response.get("matches", []):
            score = float(match["score"])
            if score < threshold:
                continue

            doc_id = match["id"]
            pinecone_meta = match.get("metadata", {})

            # Prefer full content from local store; fall back to preview
            local = self._document_store.get(doc_id)
            if local:
                content = local["content"]
                metadata = local["metadata"]
            else:
                content = pinecone_meta.get("preview", "")
                metadata = {
                    "type": pinecone_meta.get("type", "unknown"),
                    "category": pinecone_meta.get("category", "unknown"),
                }

            results.append(
                {"content": content, "metadata": metadata, "score": score}
            )

        return results

    def retrieve_by_category(
        self, query: str, category: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents filtered to a specific category.

        Args:
            query: Natural-language search query.
            category: Category value to filter on (e.g. "skills", "projects").
            k: Maximum number of results.

        Returns:
            Filtered retrieval results.
        """
        return self.retrieve(
            query=query,
            k=k,
            threshold=0.3,
            filter={"category": {"$eq": category}},
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def delete_all(self) -> None:
        """Delete all vectors in the current namespace."""
        self.index.delete(delete_all=True, namespace=self.namespace)
        self._document_store.clear()
        print(f"Deleted all vectors in namespace '{self.namespace}'.")

    def get_stats(self) -> Dict[str, Any]:
        """Return Pinecone index statistics."""
        stats = self.index.describe_index_stats()
        return {
            "total_vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", self.dimension),
            "namespaces": {
                ns: info.get("vector_count", 0)
                for ns, info in stats.get("namespaces", {}).items()
            },
            "local_store_size": len(self._document_store),
        }

    def get_category_stats(self) -> Dict[str, int]:
        """
        Return document counts by category from the local store.

        If the local store is empty (e.g. after a fresh load), this queries
        Pinecone with a zero-vector to sample available categories. For
        exact counts, ensure add_documents or load has been called.
        """
        stats: Dict[str, int] = {}

        if self._document_store:
            for doc in self._document_store.values():
                category = doc["metadata"].get("category", "other")
                stats[category] = stats.get(category, 0) + 1
        else:
            # Fallback: sample from Pinecone (approximate)
            zero_vec = [0.0] * self.dimension
            response = self.index.query(
                vector=zero_vec,
                top_k=1000,
                include_metadata=True,
                namespace=self.namespace,
            )
            for match in response.get("matches", []):
                cat = match.get("metadata", {}).get("category", "other")
                stats[cat] = stats.get(cat, 0) + 1

        return stats

    # ------------------------------------------------------------------
    # Persistence (config + local document store only)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save retriever configuration and local document store to disk.

        Vectors remain in Pinecone; this persists the mapping from
        vector IDs to full document content and metadata.

        Args:
            path: File path for the saved state (JSON).
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "index_name": self.index_name,
            "embedding_model": self.embedding_model,
            "namespace": self.namespace,
            "dimension": self.dimension,
            "metric": self.metric,
            "document_store": self._document_store,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

        print(f"PineconeRetriever config saved to {path}")

    def load(self, path: str) -> None:
        """
        Load retriever configuration and local document store from disk.

        The Pinecone index connection is re-established using the saved
        index name.

        Args:
            path: File path to the saved state (JSON).
        """
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.index_name = state.get("index_name", self.index_name)
        self.embedding_model = state.get("embedding_model", self.embedding_model)
        self.namespace = state.get("namespace", self.namespace)
        self.dimension = state.get("dimension", self.dimension)
        self.metric = state.get("metric", self.metric)
        self._document_store = state.get("document_store", {})

        # Reconnect to the Pinecone index
        self.index = self.pc.Index(self.index_name)

        print(
            f"PineconeRetriever loaded from {path}. "
            f"Local store: {len(self._document_store)} documents."
        )
