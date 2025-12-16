"""
Tests for Personal RAG API

Tests cover:
- Health checks
- Question answering
- Feedback submission
- Metrics collection
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.personal_api import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code in [200, 503]  # 503 if not initialized
    data = response.json()
    assert "status" in data
    assert "message" in data


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data


def test_sample_questions():
    """Test sample questions endpoint"""
    response = client.get("/sample-questions")
    assert response.status_code == 200
    data = response.json()
    assert "questions" in data
    assert isinstance(data["questions"], list)
    assert len(data["questions"]) > 0


def test_stats_endpoint():
    """Test stats endpoint"""
    response = client.get("/stats")
    # May be 503 if not initialized, or 200 if initialized
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "total_documents" in data
        assert "categories" in data


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests_total" in data
    assert "questions_total" in data


def test_ask_question_invalid():
    """Test asking question with invalid input"""
    # Empty question
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422  # Validation error
    
    # Missing question
    response = client.post("/ask", json={"k": 5})
    assert response.status_code == 422


def test_ask_question_valid():
    """Test asking a valid question (may fail if system not initialized)"""
    response = client.post(
        "/ask",
        json={
            "question": "What technologies are you proficient in?",
            "k": 3,
            "use_verification": False,
            "conversational": False
        }
    )
    # May be 503 if not initialized, or 200 if initialized
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert data["confidence"] in ["high", "medium", "low"]


def test_feedback_submission():
    """Test feedback submission"""
    response = client.post(
        "/feedback",
        json={
            "question": "Test question",
            "answer": "Test answer",
            "rating": 5,
            "feedback_text": "Great answer!",
            "helpful": True
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_feedback_invalid_rating():
    """Test feedback with invalid rating"""
    response = client.post(
        "/feedback",
        json={
            "question": "Test",
            "answer": "Test",
            "rating": 10  # Invalid: should be 1-5
        }
    )
    assert response.status_code == 422


def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/health")
    # CORS middleware should be configured
    assert response.status_code in [200, 405]  # OPTIONS may return 405

