"""
OpenCLIP-based multimodal retriever (stronger dense backbone for Phase 3b).

Mirrors the interface of ``CLIPRetriever`` (src/rag/clip_retriever.py) so the eval
adapters can swap it in unchanged, but encodes with an OpenCLIP backbone (default
**ViT-H/14, LAION-2B**) instead of OpenAI CLIP ViT-B/32.

Why this exists: Phase 3 proved that BM25-RRF and a text cross-encoder reranker do
NOT close the gap to the resume's ~0.92 Recall@5 on Flickr30k caption->image (the
ceiling is structural, see eval/results/phase3_findings.md). The one legitimate
lever is a stronger dense backbone. OpenCLIP ViT-H/14 (laion2b) reaches ~0.929 R@5
on the held-out 1k protocol, corroborated by LAION's published standard-protocol
numbers (ViT-H/14 = 94.0). This class makes that an in-harness, reproducible result.

Key difference from CLIPRetriever: the embedding dimension is NOT 512. ViT-H/14
produces **1024-d** embeddings (read off the loaded model, never hard-coded), so
the vector-store collection and any cached gallery for this backbone are separate
from the ViT-B/32 ones — mixing 512-d and 1024-d vectors in one collection is a bug.

Memory note (16GB ceiling): ViT-H/14 is ~3.9GB in fp32. Load it on demand and never
co-resident with the reranker or an LLM. The gallery is encoded once (offline); at
query time only a single short caption is encoded, so serving cost stays small.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import open_clip
    import torch
    from PIL import Image

    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


# Canonical OpenCLIP backbone for the ~0.92-0.93 R@5 headline. Model names use
# dashes (OpenCLIP convention), not slashes like the OpenAI `clip` package.
DEFAULT_MODEL = "ViT-H-14"
DEFAULT_PRETRAINED = "laion2b_s32b_b79k"


class OpenCLIPRetriever:
    """Cross-modal retriever using an OpenCLIP backbone (default ViT-H/14, laion2b)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: OpenCLIP architecture, e.g. 'ViT-H-14', 'ViT-L-14'.
            pretrained: OpenCLIP pretrained tag, e.g. 'laion2b_s32b_b79k'.
            device: 'cuda' | 'mps' | 'cpu'. Auto-detected if None.

        Raises:
            ImportError: If open_clip / torch / PIL are not installed.
        """
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "OpenCLIP dependencies are not installed. Install the Phase-3b extra:\n"
                "  uv pip install open_clip_torch>=2.24.0\n"
                "(also requires torch, torchvision, pillow — already in requirements_eval.txt)"
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained

        print(f"Loading OpenCLIP '{model_name}' ({pretrained}) on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # Embedding dim is backbone-specific (ViT-H/14 -> 1024). Probe it with a tiny
        # forward pass — architecture/version-agnostic, never hard-coded or assumed.
        with torch.no_grad():
            probe = self.model.encode_text(self.tokenizer(["probe"]).to(device))
        self.embed_dim = int(probe.shape[-1])
        print(f"OpenCLIP loaded. Embedding dim = {self.embed_dim}.")

        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    # --- encoders -----------------------------------------------------------
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Encode one text string -> L2-normalized 1-D embedding of shape (embed_dim,)."""
        return self._get_embeddings_batch([text])[0]

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
        """Batch-encode texts -> list of L2-normalized embeddings (one per input)."""
        if not texts:
            return []
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                tokens = self.tokenizer(batch).to(self.device)
                emb = self.model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy().astype(np.float32))
        combined = np.vstack(out)
        return [combined[i] for i in range(combined.shape[0])]

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """Encode one image file -> L2-normalized 1-D embedding of shape (embed_dim,)."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return np.asarray(self._encode_images_batch([Image.open(image_path)])[0], dtype=np.float32)

    def _encode_images_batch(self, pil_images) -> np.ndarray:
        """Preprocess + encode a list of PIL images -> (n, embed_dim) normalized array."""
        with torch.no_grad():
            tensors = torch.stack([self.preprocess(im.convert("RGB")) for im in pil_images])
            tensors = tensors.to(self.device)
            feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return np.asarray(feats.cpu().numpy(), dtype=np.float32)

    # --- gallery building (mirrors CLIPRetriever.add_images) ----------------
    def add_images(self, image_paths: List[str], metadata: List[Dict[str, Any]], batch: int = 64):
        """Encode images in batches and append to the in-memory index."""
        if len(image_paths) != len(metadata):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and metadata ({len(metadata)}) "
                "must have the same length."
            )
        print(f"Adding {len(image_paths)} images to OpenCLIP retriever...")
        new_embeddings: List[np.ndarray] = []
        for i in range(0, len(image_paths), batch):
            chunk_paths = image_paths[i : i + batch]
            chunk_meta = metadata[i : i + batch]
            pil = []
            kept_meta = []
            for p, m in zip(chunk_paths, chunk_meta):
                try:
                    pil.append(Image.open(p).convert("RGB"))
                    kept_meta.append((p, m))
                except Exception as e:  # noqa: BLE001 - skip unreadable images, keep ingest going
                    print(f"  Warning: skipping image '{p}': {e}")
            if not pil:
                continue
            feats = self._encode_images_batch(pil)
            for (p, m), vec in zip(kept_meta, feats):
                self.documents.append(
                    {"content": f"[Image: {Path(p).name}]", "metadata": {**m, "image_path": p}}
                )
                new_embeddings.append(vec)
        if new_embeddings:
            arr = np.vstack(new_embeddings)
            self.embeddings = arr if self.embeddings is None else np.vstack([self.embeddings, arr])
        print(f"Total documents: {len(self.documents)}")

    def retrieve(self, query: str, k: int = 5, threshold: float = -1.0) -> List[Dict[str, Any]]:
        """Top-k images for a text query (cosine over normalized embeddings)."""
        if self.embeddings is None or not self.documents:
            return []
        q = self._get_text_embedding(query)
        sims = self.embeddings @ q
        top = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top:
            score = float(sims[idx])
            if score >= threshold:
                results.append(
                    {
                        "content": self.documents[idx]["content"],
                        "metadata": self.documents[idx]["metadata"],
                        "score": score,
                    }
                )
        return results

    def retrieve_by_image(self, image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """Top-k gallery images most similar to a query IMAGE (reverse image search)."""
        if self.embeddings is None or not self.documents:
            return []
        q = self._get_image_embedding(image_path)
        sims = self.embeddings @ q
        top = np.argsort(sims)[::-1][:k]
        return [
            {
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(sims[idx]),
            }
            for idx in top
        ]

    # --- persistence (mirrors CLIPRetriever.save/load) ----------------------
    def save(self, path: str):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "embed_dim": self.embed_dim,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        print(f"OpenCLIP retriever saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.documents = state["documents"]
        self.embeddings = state["embeddings"]
        saved = state.get("model_name", DEFAULT_MODEL)
        if saved != self.model_name:
            print(
                f"Warning: cache used model '{saved}' but current model is "
                f"'{self.model_name}'. Embeddings are incompatible — rebuild the cache."
            )
        print(
            f"OpenCLIP retriever loaded from {path}: "
            f"{len(self.documents)} docs, shape {None if self.embeddings is None else self.embeddings.shape}"
        )
