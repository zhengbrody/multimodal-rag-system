"""
Unit tests for eval/metrics.py.

These pin the metric math against hand-computed values so the headline numbers
(Recall@5, MRR, nDCG@10) are provably correct, not just plausible.
"""

import math

from eval.metrics import aggregate, ndcg_at_k, recall_at_k, reciprocal_rank

# ---- Recall@k (single relevant item) ---------------------------------------


def test_recall_hit_in_topk():
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, "c", 5) == 1.0
    assert recall_at_k(ranked, "c", 3) == 1.0
    assert recall_at_k(ranked, "c", 2) == 0.0  # 'c' is at rank 3, outside top-2


def test_recall_miss():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, "z", 5) == 0.0


def test_recall_multi_relevant():
    # 2 relevant items, 1 of them in the top-3 -> recall = 0.5
    ranked = ["a", "x", "b", "y"]
    assert recall_at_k(ranked, {"x", "q"}, 3) == 0.5


# ---- Reciprocal rank / MRR --------------------------------------------------


def test_reciprocal_rank():
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, "a") == 1.0
    assert reciprocal_rank(ranked, "b") == 0.5
    assert reciprocal_rank(ranked, "c") == 1.0 / 3
    assert reciprocal_rank(ranked, "z") == 0.0


# ---- nDCG@k -----------------------------------------------------------------


def test_ndcg_single_relevant_rank2():
    # gold at rank 2 -> 1/log2(3); IDCG = 1 -> nDCG = 1/log2(3)
    ranked = ["a", "g", "c", "d"]
    expected = 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(ranked, "g", 10), expected)


def test_ndcg_rank1_is_one():
    assert ndcg_at_k(["g", "a", "b"], "g", 10) == 1.0


def test_ndcg_outside_cutoff_is_zero():
    ranked = ["a", "b", "c", "g"]
    assert ndcg_at_k(ranked, "g", 3) == 0.0


# ---- Aggregation across queries --------------------------------------------


def test_aggregate_mixed():
    # q1: gold at rank 1, q2: gold at rank 3, q3: gold missing (rank > list)
    rankings = [
        ["g1", "x", "y"],  # gold g1 @1
        ["x", "y", "g2", "z"],  # gold g2 @3
        ["x", "y", "z"],  # gold g3 absent
    ]
    relevants = ["g1", "g2", "g3"]
    out = aggregate(rankings, relevants, k_values=(1, 5), ndcg_k=10)

    # Recall@1: only q1 hits -> 1/3
    assert math.isclose(out["recall@1"], 1 / 3)
    # Recall@5: q1 and q2 hit -> 2/3
    assert math.isclose(out["recall@5"], 2 / 3)
    # MRR: (1/1 + 1/3 + 0) / 3
    assert math.isclose(out["mrr"], (1.0 + 1 / 3 + 0.0) / 3)
    # nDCG@10: (1 + 1/log2(4) + 0) / 3
    expected_ndcg = (1.0 + 1.0 / math.log2(4) + 0.0) / 3
    assert math.isclose(out["ndcg@10"], expected_ndcg)
    assert out["num_queries"] == 3


def test_aggregate_empty():
    out = aggregate([], [], k_values=(5,), ndcg_k=10)
    assert out["recall@5"] == 0.0
    assert out["mrr"] == 0.0
