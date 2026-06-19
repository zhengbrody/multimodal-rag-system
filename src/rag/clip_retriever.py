"""
CLIP-based multimodal retriever supporting text and image embeddings.
Uses OpenAI CLIP (ViT-B/32) to embed both text and images into a shared
512-dimensional vector space, enabling cross-modal semantic search.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path

try:
    import clip
    import torch
    from PIL import Image

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class CLIPRetriever:
    """
    Multimodal retriever using OpenAI CLIP embeddings.

    Encodes both text and images into a shared 512-dimensional vector space,
    enabling cross-modal retrieval (e.g., find images from text queries or
    find text documents from image queries).
    """

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize the CLIP retriever.

        Args:
            model_name: CLIP model variant to use. Options include
                        'ViT-B/32', 'ViT-B/16', 'ViT-L/14', etc.
            device: Device for inference ('cuda', 'mps', 'cpu').
                    Auto-detected if None.

        Raises:
            ImportError: If clip, torch, or PIL are not installed.
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP dependencies are not installed. Please install them with:\n"
                "  pip install git+https://github.com/openai/CLIP.git torch torchvision pillow\n"
                "or:\n"
                "  pip install clip-by-openai torch torchvision pillow"
            )

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name

        print(f"Loading CLIP model '{model_name}' on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print("CLIP model loaded successfully.")

        # Document storage
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        # Separate tracking for document types
        self.doc_types: List[str] = []  # "text" or "image" per entry

        # Category weights (consistent with PersonalRAGRetriever)
        self.category_weights: Dict[str, float] = {
            "faq": 1.2,
            "about": 1.1,
            "projects": 1.0,
            "blog": 1.0,
            "experience": 1.0,
            "education": 1.0,
            "skills": 1.1,
            "contact": 1.0,
        }

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a CLIP embedding.

        Args:
            text: The text to encode. CLIP truncates to 77 tokens.

        Returns:
            Normalized embedding as a 1-D numpy array of shape (512,).
        """
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            embedding = self.model.encode_text(tokens)
            # Normalize to unit length
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Encode an image file into a CLIP embedding.

        Args:
            image_path: Path to the image file (JPEG, PNG, etc.).

        Returns:
            Normalized embedding as a 1-D numpy array of shape (512,).

        Raises:
            FileNotFoundError: If the image file does not exist.
            Exception: If the image cannot be opened or processed.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(preprocessed)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten()

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode a batch of texts into CLIP embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of normalized embeddings, each of shape (512,).
        """
        if not texts:
            return []

        # Process in batches to avoid memory issues
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                tokens = clip.tokenize(batch, truncate=True).to(self.device)
                batch_embeddings = self.model.encode_text(tokens)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(batch_embeddings.cpu().numpy())

        combined = np.vstack(all_embeddings)
        return [combined[i] for i in range(combined.shape[0])]

    def add_documents(self, documents: List[Any]):
        """
        Add documents to the retriever.

        Each document should have a `content` attribute (str) and a `metadata`
        attribute (dict). If metadata contains an 'image_path' key, the image
        is embedded instead of the text.

        Args:
            documents: List of Document objects with content and metadata.
        """
        print(f"Adding {len(documents)} documents to CLIP retriever...")

        text_docs = []
        text_indices = []
        image_docs = []

        for idx, doc in enumerate(documents):
            image_path = doc.metadata.get("image_path") if hasattr(doc.metadata, "get") else None

            self.documents.append(
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                }
            )

            if image_path and Path(image_path).exists():
                image_docs.append((len(self.documents) - 1, image_path))
                self.doc_types.append("image")
            else:
                text_docs.append(doc.content)
                text_indices.append(len(self.documents) - 1)
                self.doc_types.append("text")

        # Batch-encode text documents
        new_embeddings = [None] * len(documents)

        if text_docs:
            print(f"  Encoding {len(text_docs)} text documents...")
            text_embs = self._get_embeddings_batch(text_docs)
            for local_idx, global_idx in enumerate(text_indices):
                doc_offset = global_idx - (len(self.documents) - len(documents))
                new_embeddings[doc_offset] = text_embs[local_idx]

        # Encode image documents one at a time
        if image_docs:
            print(f"  Encoding {len(image_docs)} image documents...")
            for global_idx, image_path in image_docs:
                doc_offset = global_idx - (len(self.documents) - len(documents))
                try:
                    new_embeddings[doc_offset] = self._get_image_embedding(image_path)
                except Exception as e:
                    print(f"  Warning: Failed to encode image '{image_path}': {e}")
                    # Fall back to text encoding of the content
                    new_embeddings[doc_offset] = self._get_text_embedding(
                        self.documents[global_idx]["content"]
                    )

        emb_array = np.array(new_embeddings)

        if self.embeddings is None:
            self.embeddings = emb_array
        else:
            self.embeddings = np.vstack([self.embeddings, emb_array])

        print(f"Total documents: {len(self.documents)}")
        print(f"Embedding shape: {self.embeddings.shape}")

    def add_images(self, image_paths: List[str], metadata: List[Dict[str, Any]]):
        """
        Add images directly to the index.

        Args:
            image_paths: List of file paths to images.
            metadata: List of metadata dicts (one per image). Each should
                      contain at least a 'category' key for weighting.

        Raises:
            ValueError: If image_paths and metadata have different lengths.
        """
        if len(image_paths) != len(metadata):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and metadata ({len(metadata)}) "
                "must have the same length."
            )

        print(f"Adding {len(image_paths)} images to CLIP retriever...")

        new_embeddings = []
        for img_path, meta in zip(image_paths, metadata):
            try:
                emb = self._get_image_embedding(img_path)
                new_embeddings.append(emb)
                self.documents.append(
                    {
                        "content": f"[Image: {Path(img_path).name}]",
                        "metadata": {**meta, "image_path": img_path},
                    }
                )
                self.doc_types.append("image")
            except Exception as e:
                print(f"  Warning: Skipping image '{img_path}': {e}")

        if new_embeddings:
            emb_array = np.array(new_embeddings)
            if self.embeddings is None:
                self.embeddings = emb_array
            else:
                self.embeddings = np.vstack([self.embeddings, emb_array])

        print(f"Total documents: {len(self.documents)}")

    def _cosine_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query and all document embeddings.

        Both inputs are expected to already be normalized, but this method
        normalizes again for safety.

        Args:
            query_embedding: 1-D array of shape (512,).
            doc_embeddings: 2-D array of shape (n_docs, 512).

        Returns:
            1-D array of similarity scores in [-1, 1].
        """
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents for a text query.

        Uses category weighting to boost results from high-priority categories
        (faq, about, skills).

        Args:
            query: Natural language search query.
            k: Maximum number of results to return.
            threshold: Minimum cosine similarity score to include a result.

        Returns:
            List of dicts with 'content', 'metadata', 'score', and
            'weighted_score' keys, sorted by weighted score descending.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query_embedding = self._get_text_embedding(query)
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Apply category weights
        weighted_similarities = similarities.copy()
        for i, doc in enumerate(self.documents):
            category = doc["metadata"].get("category", "other")
            weight = self.category_weights.get(category, 1.0)
            weighted_similarities[i] *= weight

        # Get top-k indices by weighted score
        top_indices = np.argsort(weighted_similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            weighted_score = float(weighted_similarities[idx])
            if score >= threshold:
                results.append(
                    {
                        "content": self.documents[idx]["content"],
                        "metadata": self.documents[idx]["metadata"],
                        "score": score,
                        "weighted_score": weighted_score,
                    }
                )

        return results

    def retrieve_by_image(self, image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents using an image as the query.

        This enables reverse image search and finding text documents that are
        semantically similar to an image.

        Args:
            image_path: Path to the query image file.
            k: Maximum number of results to return.

        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys,
            sorted by score descending.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query_embedding = self._get_image_embedding(image_path)
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            results.append(
                {
                    "content": self.documents[idx]["content"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": score,
                }
            )

        return results

    def save(self, path: str):
        """
        Save retriever state to disk using pickle.

        Args:
            path: File path for the saved state.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "doc_types": self.doc_types,
            "model_name": self.model_name,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        print(f"CLIP retriever saved to {path}")

    def load(self, path: str):
        """
        Load retriever state from disk.

        The CLIP model must already be initialized (the model weights are
        not stored in the pickle file).

        Args:
            path: File path to load the state from.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.documents = state["documents"]
        self.embeddings = state["embeddings"]
        self.doc_types = state.get("doc_types", ["text"] * len(self.documents))

        saved_model = state.get("model_name", "ViT-B/32")
        if saved_model != self.model_name:
            print(
                f"Warning: Saved state used model '{saved_model}' but current "
                f"model is '{self.model_name}'. Embeddings may be incompatible."
            )

        print(f"CLIP retriever loaded from {path}")
        print(f"Documents: {len(self.documents)}, " f"Embedding shape: {self.embeddings.shape}")

    def get_category_stats(self) -> Dict[str, int]:
        """
        Get document count by category.

        Returns:
            Dict mapping category names to document counts.
        """
        stats: Dict[str, int] = {}
        for doc in self.documents:
            category = doc["metadata"].get("category", "other")
            stats[category] = stats.get(category, 0) + 1
        return stats


if __name__ == "__main__":
    # Demonstrate CLIP retriever with sample text documents
    from dataclasses import dataclass, field

    @dataclass
    class Document:
        content: str
        metadata: dict = field(default_factory=dict)

    # Sample documents simulating a personal knowledge base
    sample_documents = [
        Document(
            content="I am a software engineer with expertise in Python, machine learning, and NLP.",
            metadata={"category": "about", "type": "personal_info"},
        ),
        Document(
            content="Built a multimodal RAG system using CLIP embeddings for cross-modal retrieval.",
            metadata={"category": "projects", "type": "project"},
        ),
        Document(
            content="Proficient in Python, PyTorch, TensorFlow, React, and cloud services like AWS.",
            metadata={"category": "skills", "type": "skills"},
        ),
        Document(
            content="Graduated from UC San Diego with a degree in Computer Science.",
            metadata={"category": "education", "type": "education"},
        ),
        Document(
            content="Frequently asked: What is your favorite programming language? Answer: Python.",
            metadata={"category": "faq", "type": "faq"},
        ),
        Document(
            content="Worked as a machine learning intern at Allianz, building NLP pipelines.",
            metadata={"category": "experience", "type": "experience"},
        ),
        Document(
            content="You can reach me at zheng@example.com or connect on LinkedIn.",
            metadata={"category": "contact", "type": "contact"},
        ),
    ]

    print("=" * 60)
    print("CLIP Retriever Demo")
    print("=" * 60)

    retriever = CLIPRetriever()
    retriever.add_documents(sample_documents)

    print(f"\nCategory stats: {retriever.get_category_stats()}")

    # Test text queries
    test_queries = [
        "What technologies are you proficient in?",
        "Tell me about your RAG project",
        "What is your education background?",
        "How can I contact you?",
        "What is your work experience?",
    ]

    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"Query: {query}")
        results = retriever.retrieve(query, k=3)
        for i, result in enumerate(results):
            print(
                f"  Result {i + 1} (score: {result['score']:.3f}, "
                f"weighted: {result['weighted_score']:.3f}):"
            )
            print(f"    Category: {result['metadata'].get('category')}")
            print(f"    Content:  {result['content'][:100]}...")

    # Save and reload
    save_path = Path(__file__).parent.parent.parent / "data" / "processed" / "clip_retriever.pkl"
    retriever.save(str(save_path))

    retriever2 = CLIPRetriever()
    retriever2.load(str(save_path))
    print(f"\nReloaded category stats: {retriever2.get_category_stats()}")
