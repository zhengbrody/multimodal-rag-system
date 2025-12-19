"""
Tests for Retriever components

Tests cover:
- Mock retriever functionality
- Document addition and retrieval
- Category statistics
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.mock_retriever import MockRetriever
from rag.knowledge_processor import Document


def test_mock_retriever_initialization():
    """Test mock retriever can be initialized"""
    retriever = MockRetriever()
    assert retriever.documents == []
    assert retriever.embedding_model == "mock-keyword-matching"


def test_mock_retriever_add_documents():
    """Test adding documents to mock retriever"""
    retriever = MockRetriever()
    
    doc1 = Document(
        content="I am proficient in Python and machine learning",
        metadata={"type": "skills", "category": "skills"}
    )
    doc2 = Document(
        content="I worked on a RAG project using FastAPI",
        metadata={"type": "project", "category": "projects"}
    )
    
    retriever.add_documents([doc1, doc2])
    
    assert len(retriever.documents) == 2
    assert len(retriever.documents[0]['keywords']) > 0


def test_mock_retriever_retrieve():
    """Test retrieval functionality"""
    retriever = MockRetriever()
    
    doc1 = Document(
        content="I am proficient in Python, machine learning, and deep learning",
        metadata={"type": "skills", "category": "skills"}
    )
    doc2 = Document(
        content="I worked on a recommendation system project",
        metadata={"type": "project", "category": "projects"}
    )
    
    retriever.add_documents([doc1, doc2])
    
    # Query about skills
    results = retriever.retrieve("What technologies do you know?", k=2)
    
    assert len(results) > 0
    assert results[0]['score'] > 0
    assert 'content' in results[0]
    assert 'metadata' in results[0]


def test_mock_retriever_category_stats():
    """Test category statistics"""
    retriever = MockRetriever()
    
    docs = [
        Document("Python skills", {"category": "skills"}),
        Document("RAG project", {"category": "projects"}),
        Document("More Python", {"category": "skills"}),
    ]
    
    retriever.add_documents(docs)
    
    stats = retriever.get_category_stats()
    
    assert stats["skills"] == 2
    assert stats["projects"] == 1


def test_mock_retriever_save_load(tmp_path):
    """Test saving and loading retriever"""
    retriever = MockRetriever()
    
    doc = Document(
        content="Test document",
        metadata={"type": "test", "category": "test"}
    )
    retriever.add_documents([doc])
    
    # Save
    save_path = tmp_path / "test_retriever.pkl"
    retriever.save(str(save_path))
    assert save_path.exists()
    
    # Load
    new_retriever = MockRetriever()
    new_retriever.load(str(save_path))
    
    assert len(new_retriever.documents) == 1
    assert new_retriever.documents[0]['content'] == "Test document"


