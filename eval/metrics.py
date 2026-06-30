"""
Information-retrieval metrics for ranked retrieval results.

Each query produces a *ranked* list of candidate ids (best first) and has a set
of relevant (ground-truth) ids. For the Flickr30k caption->image task there is
exactly one relevant image per query, in which case these formulas reduce to:

    recall@k = 1 if the gold image is in the top-k else 0
    mrr      = 1 / rank_of_gold        (rank is 1-based; 0 if not retrieved)
    ndcg@k   = 1 / log2(rank + 1)      if rank <= k else 0   (IDCG = 1)

The functions accept a *set* of relevant ids so the same code also handles the
multi-relevant case (used later when several captions/images share ground truth).
All functions are pure and deterministic so the unit tests can pin exact values.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Set


def _as_relevant_set(relevant) -> Set:
    """Normalize a single id or an iterable of ids into a set."""
    if relevant is None:
        return set()
    if isinstance(relevant, (set, frozenset)):
        return set(relevant)
    if isinstance(relevant, (list, tuple)):
        return set(relevant)
    # A bare scalar id (e.g. a single gold image_id).
    return {relevant}


def recall_at_k(ranked: Sequence, relevant, k: int) -> float:
    """
    Fraction of the relevant items that appear in the top-k.

    With a single relevant item this is the familiar hit-rate (1.0 or 0.0).
    """
    rel = _as_relevant_set(relevant)
    if not rel:
        return 0.0
    topk = list(ranked)[:k]
    hits = sum(1 for item in topk if item in rel)
    return hits / len(rel)


def reciprocal_rank(ranked: Sequence, relevant) -> float:
    """Reciprocal of the 1-based rank of the first relevant item (0 if none)."""
    rel = _as_relevant_set(relevant)
    for idx, item in enumerate(ranked, start=1):
        if item in rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked: Sequence, relevant, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k with binary relevance.

    DCG sums 1/log2(rank+1) over relevant items found in the top-k; IDCG is the
    DCG of the ideal ranking (all relevant items packed at the top). With one
    relevant item IDCG = 1, so nDCG@k = 1/log2(rank+1) when the item is in top-k.
    """
    rel = _as_relevant_set(relevant)
    if not rel:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in rel:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _mean_ci95(values: Sequence[float]) -> tuple:
    """
    Mean and 95% confidence-interval half-width of per-query metric values.

    Uses the normal (Wald) approximation: half-width = 1.96 * std / sqrt(n), with
    the sample standard deviation. For binary outcomes (recall@k) this is the
    standard proportion CI; for continuous values in [0,1] (MRR, nDCG) it is the
    usual CI of the mean. Deterministic (no bootstrap RNG), so runs are
    reproducible. With n=5,000 the half-widths are ~+/-0.01.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = 1.96 * (var**0.5) / (n**0.5)
    return mean, half


def aggregate(
    rankings: Iterable[Sequence],
    relevants: Iterable,
    k_values: Sequence[int] = (1, 5, 10),
    ndcg_k: int = 10,
) -> Dict[str, float]:
    """
    Compute mean metrics (with 95% CIs) across a set of queries.

    Args:
        rankings:  iterable of ranked id lists (one per query).
        relevants: iterable of relevant-id sets/scalars (one per query, aligned).
        k_values:  the k's to report Recall@k for.
        ndcg_k:    cutoff for nDCG.

    Returns:
        Dict with ``recall@{k}`` for each k, plus ``mrr`` and ``ndcg@{ndcg_k}``,
        and a ``{metric}_ci95`` half-width for each. Returns zeros if no queries.
    """
    rankings = list(rankings)
    relevants = list(relevants)
    if len(rankings) != len(relevants):
        raise ValueError(
            f"rankings ({len(rankings)}) and relevants ({len(relevants)}) "
            "must have the same length."
        )

    n = len(rankings)
    out: Dict[str, float] = {}
    if n == 0:
        for k in k_values:
            out[f"recall@{k}"] = 0.0
            out[f"recall@{k}_ci95"] = 0.0
        out["mrr"] = 0.0
        out["mrr_ci95"] = 0.0
        out[f"ndcg@{ndcg_k}"] = 0.0
        out[f"ndcg@{ndcg_k}_ci95"] = 0.0
        out["num_queries"] = 0
        return out

    pairs = list(zip(rankings, relevants))
    for k in k_values:
        mean, half = _mean_ci95([recall_at_k(r, rel, k) for r, rel in pairs])
        out[f"recall@{k}"] = mean
        out[f"recall@{k}_ci95"] = half
    mean, half = _mean_ci95([reciprocal_rank(r, rel) for r, rel in pairs])
    out["mrr"], out["mrr_ci95"] = mean, half
    mean, half = _mean_ci95([ndcg_at_k(r, rel, ndcg_k) for r, rel in pairs])
    out[f"ndcg@{ndcg_k}"], out[f"ndcg@{ndcg_k}_ci95"] = mean, half
    out["num_queries"] = n
    return out
