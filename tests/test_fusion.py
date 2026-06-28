"""
Unit tests for RRF fusion (src/rag/fusion.py) and BM25 sparse retrieval
(src/rag/sparse_retriever.py). RRF is pinned against hand-computed scores.
"""

from src.rag.fusion import reciprocal_rank_fusion, rrf_ids
from src.rag.sparse_retriever import BM25Retriever, tokenize


def test_rrf_scores_hand_computed():
    # list A ranks x>y>z, list B ranks y>w. With k=60:
    #   y = 1/(60+2) + 1/(60+1) = 0.03252  (rank2 in A, rank1 in B)
    #   x = 1/(60+1)            = 0.01639
    #   w = 1/(60+2)            = 0.01613
    #   z = 1/(60+3)            = 0.01587
    fused = dict(reciprocal_rank_fusion([["x", "y", "z"], ["y", "w"]], k=60))
    assert abs(fused["y"] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(fused["x"] - (1 / 61)) < 1e-9
    # y is fused-best because it appears high in BOTH lists
    assert rrf_ids([["x", "y", "z"], ["y", "w"]]) == ["y", "x", "w", "z"]


def test_rrf_rewards_agreement_over_single_top():
    # An item ranked 2nd in both lists beats an item ranked 1st in only one.
    order = rrf_ids([["solo", "both", "x"], ["other", "both", "y"]])
    assert order[0] == "both"


def test_rrf_empty():
    assert rrf_ids([]) == []


def test_tokenize_drops_stopwords():
    assert tokenize("A man on the BEACH") == ["man", "beach"]


def test_bm25_ranks_lexical_match_first():
    bm25 = BM25Retriever()
    bm25.index(
        ["img1", "img2", "img3"],
        [
            "a brown dog runs on the beach",
            "two men play chess indoors",
            "a child rides a red bicycle",
        ],
    )
    top = bm25.search_ids("dog running on a sandy beach", k=3)
    assert top[0] == "img1"  # the only doc sharing 'dog'/'beach'
