"""
Cross-encoder reranker (bge-reranker-v2-m3) — the final precision stage.

A bi-encoder (CLIP / sentence-transformer) embeds query and document
independently; a CROSS-encoder feeds the (query, document) pair jointly through a
transformer and emits one relevance score, so it models fine-grained interaction
the bi-encoder cannot. It is far more accurate but far more expensive, so it runs
ONLY on a short candidate list produced by the cheap dense+sparse retrieval.

IMPORTANT (honest scope): bge-reranker-v2-m3 is a TEXT cross-encoder. In the
caption->image task it cannot read pixels, so it scores (query_caption,
candidate_image_caption) pairs — i.e. it reranks on each image's caption metadata.
The DENSE CLIP stage is what grounds retrieval in the actual image; the reranker
sharpens ordering using text. This mirrors real systems where images carry
alt-text/titles.

Memory (16GB ceiling): the model (~2.2GB fp32) is loaded lazily on first use and
should never be co-resident with a 7B LLM. Use fp16 on MPS/CUDA to halve it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder that reorders a candidate list by relevance."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import CrossEncoder

        if self.device is None:
            self.device = (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        print(f"Loading cross-encoder '{self.model_name}' on {self.device}...")
        # fp16 halves the ~2.2GB footprint; safe on MPS/CUDA for inference.
        self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
        print("Cross-encoder loaded.")

    def rerank(
        self, query: str, candidates: Sequence[Tuple[str, str]], top_k: Optional[int] = None,
        batch_size: int = 64,
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates by cross-encoder relevance to the query.

        Args:
            query: the query text.
            candidates: list of (doc_id, doc_text) pairs to score.
            top_k: keep only this many after reranking (default: all).
            batch_size: cross-encoder inference batch size.

        Returns:
            (doc_id, score) sorted by score descending.
        """
        self._ensure_loaded()
        if not candidates:
            return []
        pairs = [[query, text] for _, text in candidates]
        scores = self._model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(
            ((doc_id, float(s)) for (doc_id, _), s in zip(candidates, scores)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ranked[:top_k] if top_k else ranked

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidate_lists: Sequence[Sequence[Tuple[str, str]]],
        batch_size: int = 256,
    ) -> List[List[str]]:
        """
        Rerank many queries at once by flattening ALL (query, candidate) pairs into
        a single batched forward pass — far better GPU utilization than per-query
        calls — then scattering scores back. Returns a reranked id list per query.
        """
        self._ensure_loaded()
        flat_pairs: List[List[str]] = []
        spans: List[Tuple[int, int, List[str]]] = []
        for query, cands in zip(queries, candidate_lists):
            start = len(flat_pairs)
            flat_pairs.extend([[query, text] for _, text in cands])
            spans.append((start, len(flat_pairs), [doc_id for doc_id, _ in cands]))

        if not flat_pairs:
            return [[] for _ in queries]

        scores = self._model.predict(flat_pairs, batch_size=batch_size, show_progress_bar=True)

        out: List[List[str]] = []
        for start, end, ids in spans:
            sc = scores[start:end]
            ranked = [iid for iid, _ in sorted(zip(ids, sc), key=lambda kv: kv[1], reverse=True)]
            out.append(ranked)
        return out

    def unload(self) -> None:
        """Free the model (respect the 16GB ceiling before loading another model)."""
        self._model = None
