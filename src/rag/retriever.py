"""
Lightweight Retriever for Personal RAG System

Uses OpenAI embeddings with simple cosine similarity search
No heavy dependencies like FAISS or Torch needed
"""

import numpy as np
from typing import List, Dict, Any, Optional
import json
import pickle
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class SimpleRetriever:
    """
    Simple but effective retriever using OpenAI embeddings
    and numpy-based cosine similarity
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize retriever

        Args:
            embedding_model: OpenAI embedding model to use
                - text-embedding-3-small: faster, cheaper, 1536 dims
                - text-embedding-3-large: better quality, 3072 dims
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_model = embedding_model

        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts in batches"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def add_documents(self, documents: List[Any]):
        """
        Add documents to the retriever

        Args:
            documents: List of Document objects with content and metadata
        """
        print(f"Adding {len(documents)} documents to retriever...")

        # Convert to internal format
        texts = []
        for doc in documents:
            self.documents.append({
                'content': doc.content,
                'metadata': doc.metadata
            })
            texts.append(doc.content)

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self._get_embeddings_batch(texts)

        # Store as numpy array
        if self.embeddings is None:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, np.array(embeddings)])

        print(f"Total documents: {len(self.documents)}")
        print(f"Embedding shape: {self.embeddings.shape}")

    def _cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all documents"""
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Compute similarities
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents

        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of documents with scores
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Get query embedding
        query_embedding = np.array(self._get_embedding(query))

        # Compute similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({
                    'content': self.documents[idx]['content'],
                    'metadata': self.documents[idx]['metadata'],
                    'score': score
                })

        return results

    def save(self, path: str):
        """Save retriever state to disk"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'embedding_model': self.embedding_model
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"Retriever saved to {path}")

    def load(self, path: str):
        """Load retriever state from disk"""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.documents = state['documents']
        self.embeddings = state['embeddings']
        self.embedding_model = state['embedding_model']

        print(f"Retriever loaded from {path}")
        print(f"Documents: {len(self.documents)}, Embedding shape: {self.embeddings.shape}")


class PersonalRAGRetriever(SimpleRetriever):
    """
    Specialized retriever for personal Q&A with additional features
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        super().__init__(embedding_model)
        self.category_weights = {
            'faq': 1.2,  # Boost FAQ answers
            'about': 1.1,  # Personal info is important
            'projects': 1.0,
            'blog': 1.0,
            'experience': 1.0,
            'education': 1.0,
            'skills': 1.1,
            'contact': 1.0
        }

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval with category weighting
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Get query embedding
        query_embedding = np.array(self._get_embedding(query))

        # Compute base similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Apply category weights
        weighted_similarities = similarities.copy()
        for i, doc in enumerate(self.documents):
            category = doc['metadata'].get('category', 'other')
            weight = self.category_weights.get(category, 1.0)
            weighted_similarities[i] *= weight

        # Get top-k indices
        top_indices = np.argsort(weighted_similarities)[::-1][:k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])  # Return original score
            weighted_score = float(weighted_similarities[idx])

            if score >= threshold:
                results.append({
                    'content': self.documents[idx]['content'],
                    'metadata': self.documents[idx]['metadata'],
                    'score': score,
                    'weighted_score': weighted_score
                })

        return results

    def get_category_stats(self) -> Dict[str, int]:
        """Get document count by category"""
        stats = {}
        for doc in self.documents:
            category = doc['metadata'].get('category', 'other')
            stats[category] = stats.get(category, 0) + 1
        return stats


if __name__ == "__main__":
    # Test the retriever
    from knowledge_processor import build_knowledge_base

    kb_path = Path(__file__).parent.parent.parent / "data" / "raw" / "knowledge_base.json"

    # Build knowledge base
    documents = build_knowledge_base(str(kb_path))

    # Initialize retriever
    retriever = PersonalRAGRetriever()

    # Add documents
    retriever.add_documents(documents)

    # Test retrieval
    test_queries = [
        "What technologies are you proficient in?",
        "Tell me about your RAG project",
        "What is your education background?",
        "How can I contact you?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, k=3)
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Score: {result['score']:.3f}):")
            print(f"Type: {result['metadata'].get('type')}")
            print(f"Content: {result['content'][:150]}...")

    # Save retriever
    save_path = Path(__file__).parent.parent.parent / "data" / "processed" / "retriever.pkl"
    retriever.save(str(save_path))

    print(f"\nCategory stats: {retriever.get_category_stats()}")
