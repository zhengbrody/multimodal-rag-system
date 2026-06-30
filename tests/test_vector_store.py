"""
Tests for VectorStore backend selection (src/rag/vector_store.py).

These exercise the pluggable-backend dispatch and the experimental Pinecone backend's
fail-fast behavior WITHOUT needing qdrant-client, the pinecone SDK, or any API key — so
they run in the lightweight requirements_simple CI env.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.vector_store import VectorStore  # noqa: E402


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown VECTOR_BACKEND"):
        VectorStore(collection="x", dim=8, backend="bogus")


def test_pinecone_backend_requires_api_key(monkeypatch):
    """The experimental Pinecone backend fails fast (clear ValueError) with no key —
    before importing the optional SDK, so this is deterministic on any machine."""
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        VectorStore(collection="x", dim=8, backend="pinecone")
