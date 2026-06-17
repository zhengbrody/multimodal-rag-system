"""
Semantic caching layer for RAG queries using Redis.
Caches query embeddings and results to avoid redundant LLM calls.
Supports TTL-based expiration, similarity-based cache lookup,
and automatic cache invalidation on knowledge base updates.
"""

import hashlib
import json
import time
import logging
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger("rag_cache")


class InMemoryCache:
    """
    Simple in-memory cache with TTL and max size eviction.
    Used as a fallback when Redis is unavailable.
    Provides the same public interface as SemanticCache.
    """

    def __init__(self, max_size: int = 256, ttl: int = 3600, similarity_threshold: float = 0.95):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
        self.max_size = max_size
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold

        # Stats
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-used entry when cache is full."""
        while len(self._store) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._store.pop(oldest_key, None)
            self._embeddings.pop(oldest_key, None)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        return (time.time() - entry["timestamp"]) > self.ttl

    def _touch(self, key: str) -> None:
        """Move *key* to the end of the access order list."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _hash_query(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str, query_embedding: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Look up a cached result.

        First tries an exact-match by query hash.  If *query_embedding* is
        provided and no exact match is found, falls back to a similarity
        search across cached embeddings.
        """
        key = self._hash_query(query)

        # --- exact match ---
        entry = self._store.get(key)
        if entry is not None and not self._is_expired(entry):
            self._hits += 1
            self._touch(key)
            logger.debug("In-memory cache hit (exact) for query: %s", query[:80])
            return entry["result"]

        # Remove expired entry if present
        if entry is not None:
            self._store.pop(key, None)
            self._embeddings.pop(key, None)

        # --- similarity match ---
        if query_embedding is not None:
            result = self._find_similar_cached(query_embedding)
            if result is not None:
                self._hits += 1
                logger.debug("In-memory cache hit (similar) for query: %s", query[:80])
                return result

        self._misses += 1
        return None

    def set(
        self,
        query: str,
        result: Dict,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Cache a query result with TTL."""
        key = self._hash_query(query)
        self._evict_if_needed()

        self._store[key] = {
            "result": result,
            "timestamp": time.time(),
            "query": query,
        }
        if embedding is not None:
            self._embeddings[key] = embedding
        self._touch(key)
        logger.debug("Cached result for query: %s", query[:80])

    def invalidate_all(self) -> None:
        """Clear all cached results."""
        self._store.clear()
        self._embeddings.clear()
        self._access_order.clear()
        logger.info("In-memory cache invalidated (all entries cleared)")

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": "in-memory",
            "size": len(self._store),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "miss_rate": round(self._misses / total, 4) if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def _find_similar_cached(self, query_embedding: np.ndarray) -> Optional[Dict]:
        """Return the cached result whose embedding is most similar, if above threshold."""
        best_score = -1.0
        best_key: Optional[str] = None

        for key, emb in self._embeddings.items():
            entry = self._store.get(key)
            if entry is None or self._is_expired(entry):
                continue
            score = self._cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is not None and best_score >= self.similarity_threshold:
            self._touch(best_key)
            return self._store[best_key]["result"]
        return None


class SemanticCache:
    """
    Redis-backed semantic cache for RAG query results.

    Stores query results keyed by a hash of the query text, and maintains
    a parallel set of embedding vectors so that semantically similar (but
    not identical) queries can be served from cache.

    Falls back transparently to ``InMemoryCache`` when Redis is unreachable.
    """

    CACHE_PREFIX = "rag:cache:"
    EMBEDDING_PREFIX = "rag:emb:"
    STATS_KEY = "rag:cache:stats"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 3600,
        similarity_threshold: float = 0.95,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._redis = None
        self._fallback: Optional[InMemoryCache] = None
        self._openai_client = None

        # Attempt Redis connection
        try:
            import redis as redis_lib

            self._redis = redis_lib.from_url(
                redis_url,
                decode_responses=False,
                socket_connect_timeout=3,
            )
            # Verify connectivity
            self._redis.ping()
            logger.info("Connected to Redis at %s", redis_url)
        except Exception as exc:
            logger.warning(
                "Redis unavailable (%s). Falling back to in-memory cache.", exc
            )
            self._redis = None
            self._fallback = InMemoryCache(
                ttl=ttl, similarity_threshold=similarity_threshold
            )

        # Lazy-init OpenAI client for embedding generation
        try:
            from openai import OpenAI

            self._openai_client = OpenAI()
        except Exception:
            logger.warning(
                "OpenAI client unavailable; embedding-based cache keys disabled."
            )
            self._openai_client = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(data: Dict) -> str:
        """Serialize a dict to a JSON string for Redis storage."""
        return json.dumps(data, ensure_ascii=False, default=str)

    @staticmethod
    def _deserialize(data: str) -> Dict:
        """Deserialize a JSON string back to a dict."""
        return json.loads(data)

    # ------------------------------------------------------------------
    # Cache key helpers
    # ------------------------------------------------------------------

    def _get_cache_key(self, query: str) -> str:
        """
        Generate a deterministic cache key from the query text.

        Uses a SHA-256 hash of the lowercased, stripped query so that
        minor whitespace/casing differences still match.
        """
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    def _compute_embedding(self, query: str) -> Optional[np.ndarray]:
        """Compute an embedding vector for *query* using OpenAI."""
        if self._openai_client is None:
            return None
        try:
            response = self._openai_client.embeddings.create(
                input=query, model=self.embedding_model
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as exc:
            logger.warning("Failed to compute embedding: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Similarity search (Redis path)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _find_similar_cached(self, query_embedding: np.ndarray) -> Optional[Dict]:
        """
        Scan cached embeddings in Redis for a cosine similarity hit.

        Returns the cached result dict if a match above
        ``self.similarity_threshold`` is found, otherwise ``None``.
        """
        if self._redis is None:
            return None

        try:
            cursor = "0"
            best_score = -1.0
            best_cache_key: Optional[str] = None

            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor,
                    match=f"{self.EMBEDDING_PREFIX}*".encode(),
                    count=100,
                )
                for emb_key in keys:
                    raw = self._redis.get(emb_key)
                    if raw is None:
                        continue
                    cached_emb = np.frombuffer(raw, dtype=np.float32)
                    score = self._cosine_similarity(query_embedding, cached_emb)
                    if score > best_score:
                        best_score = score
                        # Derive the cache key from the embedding key
                        hash_part = emb_key.decode().replace(self.EMBEDDING_PREFIX, "")
                        best_cache_key = hash_part

                if cursor == 0 or cursor == b"0":
                    break

            if best_cache_key is not None and best_score >= self.similarity_threshold:
                raw_result = self._redis.get(
                    f"{self.CACHE_PREFIX}{best_cache_key}".encode()
                )
                if raw_result is not None:
                    return self._deserialize(raw_result.decode())
        except Exception as exc:
            logger.warning("Similarity search failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[Dict]:
        """
        Look up a cached result for *query*.

        1. Try exact-match by query hash in Redis.
        2. If no exact match, compute an embedding and try similarity search.
        3. If Redis is down, delegate to the in-memory fallback.
        """
        # Fallback path
        if self._redis is None and self._fallback is not None:
            embedding = self._compute_embedding(query)
            return self._fallback.get(query, query_embedding=embedding)

        cache_key = self._get_cache_key(query)

        try:
            # Exact match
            raw = self._redis.get(f"{self.CACHE_PREFIX}{cache_key}".encode())
            if raw is not None:
                self._redis.hincrby(self.STATS_KEY, "hits", 1)
                logger.debug("Redis cache hit (exact) for: %s", query[:80])
                return self._deserialize(raw.decode())

            # Similarity match
            embedding = self._compute_embedding(query)
            if embedding is not None:
                similar = self._find_similar_cached(embedding)
                if similar is not None:
                    self._redis.hincrby(self.STATS_KEY, "hits", 1)
                    logger.debug("Redis cache hit (similar) for: %s", query[:80])
                    return similar

            self._redis.hincrby(self.STATS_KEY, "misses", 1)
            return None
        except Exception as exc:
            logger.warning("Redis GET failed (%s), trying fallback.", exc)
            if self._fallback is None:
                self._fallback = InMemoryCache(
                    ttl=self.ttl, similarity_threshold=self.similarity_threshold
                )
            return self._fallback.get(query)

    def set(
        self,
        query: str,
        result: Dict,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        Cache a query result with TTL.

        If *embedding* is ``None`` and an OpenAI client is available, the
        embedding will be computed automatically.
        """
        # Fallback path
        if self._redis is None and self._fallback is not None:
            if embedding is None:
                embedding = self._compute_embedding(query)
            self._fallback.set(query, result, embedding=embedding)
            return

        cache_key = self._get_cache_key(query)

        try:
            serialized = self._serialize(result)
            self._redis.setex(
                f"{self.CACHE_PREFIX}{cache_key}",
                self.ttl,
                serialized.encode(),
            )

            # Store embedding for similarity lookup
            if embedding is None:
                embedding = self._compute_embedding(query)
            if embedding is not None:
                self._redis.setex(
                    f"{self.EMBEDDING_PREFIX}{cache_key}",
                    self.ttl,
                    embedding.tobytes(),
                )

            logger.debug("Cached result in Redis for: %s", query[:80])
        except Exception as exc:
            logger.warning("Redis SET failed (%s), caching in memory.", exc)
            if self._fallback is None:
                self._fallback = InMemoryCache(
                    ttl=self.ttl, similarity_threshold=self.similarity_threshold
                )
            self._fallback.set(query, result, embedding=embedding)

    def invalidate_all(self) -> None:
        """
        Clear all cached results and embeddings.
        Should be called when the knowledge base is rebuilt (e.g. /rebuild-index).
        """
        if self._redis is None and self._fallback is not None:
            self._fallback.invalidate_all()
            return

        try:
            # Delete all cache keys
            for prefix in (self.CACHE_PREFIX, self.EMBEDDING_PREFIX):
                cursor = "0"
                while True:
                    cursor, keys = self._redis.scan(
                        cursor=cursor,
                        match=f"{prefix}*".encode(),
                        count=200,
                    )
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0 or cursor == b"0":
                        break

            # Reset stats
            self._redis.delete(self.STATS_KEY)
            logger.info("Redis cache invalidated (all entries cleared)")
        except Exception as exc:
            logger.warning("Redis INVALIDATE failed: %s", exc)
            if self._fallback is not None:
                self._fallback.invalidate_all()

    def get_stats(self) -> Dict[str, Any]:
        """Return hit rate, miss rate, and cache size."""
        if self._redis is None and self._fallback is not None:
            return self._fallback.get_stats()

        try:
            raw_stats = self._redis.hgetall(self.STATS_KEY)
            hits = int(raw_stats.get(b"hits", 0))
            misses = int(raw_stats.get(b"misses", 0))
            total = hits + misses

            # Count cache entries
            size = 0
            cursor = "0"
            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor,
                    match=f"{self.CACHE_PREFIX}*".encode(),
                    count=200,
                )
                size += len(keys)
                if cursor == 0 or cursor == b"0":
                    break

            return {
                "backend": "redis",
                "size": size,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
                "miss_rate": round(misses / total, 4) if total > 0 else 0.0,
            }
        except Exception as exc:
            logger.warning("Failed to retrieve Redis stats: %s", exc)
            return {
                "backend": "redis (error)",
                "size": -1,
                "hits": -1,
                "misses": -1,
                "hit_rate": 0.0,
                "miss_rate": 0.0,
                "error": str(exc),
            }
