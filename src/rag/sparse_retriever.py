"""
Sparse (lexical) retriever using BM25 — the sparse half of the hybrid pipeline.

In the caption->image task each gallery image carries text metadata (its
representative caption). BM25 indexes that text and matches it against the query
caption by term overlap, giving a lexical signal that complements the dense CLIP
(pixel) signal. Fusing a genuine sparse + dense pair via RRF is the point of the
hybrid design — not fusing two dense models.

Backed by rank_bm25 (BM25Okapi); pure-Python, no heavy deps.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, standard English stop list — BM25 already down-weights frequent terms,
# but dropping these trims noise on short caption text.
_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "with",
    "for",
    "by",
    "as",
    "that",
    "this",
    "it",
    "its",
    "from",
    "into",
    "over",
    "near",
    "while",
    "who",
    "his",
    "her",
    "their",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stop words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class BM25Retriever:
    """BM25 lexical retriever over a fixed set of (doc_id, text) documents."""

    def __init__(self):
        self._bm25 = None
        self.doc_ids: List[str] = []

    def index(self, doc_ids: List[str], texts: List[str]) -> None:
        """Build the BM25 index from parallel doc_id / text lists."""
        from rank_bm25 import BM25Okapi

        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length.")
        self.doc_ids = list(doc_ids)
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Return the top-k (doc_id, bm25_score) for a query, score-descending."""
        if self._bm25 is None:
            raise RuntimeError("index() must be called before search().")
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]

    def search_ids(self, query: str, k: int) -> List[str]:
        """Return just the top-k doc_ids (ranked)."""
        return [doc_id for doc_id, _ in self.search(query, k)]
