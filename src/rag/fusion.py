"""
Reciprocal Rank Fusion (RRF) — combine several ranked lists into one.

RRF is rank-based, not score-based, so it fuses retrievers whose scores live on
different scales (CLIP cosine in [-1,1] vs BM25's unbounded term-weight sums)
without any normalization. Each list contributes 1/(k + rank) to a document's
fused score; the constant k (default 60, from Cormack et al. 2009) dampens the
influence of very high ranks so no single list dominates.

    fused_score(d) = sum over lists L of  1 / (k + rank_L(d))

Used to fuse the DENSE (CLIP) and SPARSE (BM25) paths of the hybrid retriever.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = 60
) -> List[Tuple[str, float]]:
    """
    Fuse ranked id lists into one ranking.

    Args:
        ranked_lists: each is a list of doc_ids ordered best-first.
        k: RRF damping constant (default 60).

    Returns:
        List of (doc_id, fused_score) sorted by score descending.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rrf_ids(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> List[str]:
    """RRF fusion returning just the ranked doc_ids."""
    return [doc_id for doc_id, _ in reciprocal_rank_fusion(ranked_lists, k)]
