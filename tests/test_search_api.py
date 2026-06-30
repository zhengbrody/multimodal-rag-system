"""
Tests for the multimodal search endpoints (/search/text, /search/image).

These are deterministic and dependency-light: the image-search backend is monkeypatched,
so they never load a multi-GB model and pass in the lightweight requirements_simple CI env
(where no CLIP gallery / Torch is present and the endpoints must degrade to HTTP 503).
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from api import personal_api  # noqa: E402
from api.personal_api import app  # noqa: E402

client = TestClient(app)


def _force_available(monkeypatch, value: bool):
    monkeypatch.setattr(personal_api.ImageSearchService, "available", staticmethod(lambda: value))


class _FakeSearch:
    """Stand-in for the loaded ImageSearchService (no model)."""

    backend_label = "openclip-vit-h14"

    def search_text(self, query, k=5):
        return [{"image_id": "42", "score": 0.91, "rep_caption": "a dog on the beach"}][:k]

    def search_image(self, path, k=5):
        return [{"image_id": "7", "score": 1.0, "rep_caption": "self"}][:k]


# ---- gating: 503 when no gallery (the mock/CI deployment) -------------------


def test_search_text_503_when_unavailable(monkeypatch):
    _force_available(monkeypatch, False)
    r = client.post("/search/text", json={"query": "a dog", "k": 5})
    assert r.status_code == 503
    assert "not available" in r.json()["detail"].lower()


def test_search_image_503_when_unavailable(monkeypatch):
    _force_available(monkeypatch, False)
    r = client.post("/search/image", files={"file": ("q.jpg", b"\xff\xd8\xff", "image/jpeg")})
    assert r.status_code == 503


# ---- input validation (Pydantic) -------------------------------------------


def test_search_text_rejects_empty_query(monkeypatch):
    _force_available(monkeypatch, True)
    monkeypatch.setattr(personal_api, "image_search", _FakeSearch())
    assert client.post("/search/text", json={"query": "", "k": 5}).status_code == 422


def test_search_text_rejects_out_of_range_k(monkeypatch):
    _force_available(monkeypatch, True)
    monkeypatch.setattr(personal_api, "image_search", _FakeSearch())
    assert client.post("/search/text", json={"query": "x", "k": 0}).status_code == 422
    assert client.post("/search/text", json={"query": "x", "k": 999}).status_code == 422


# ---- success shape (backend faked, no model) -------------------------------


def test_search_text_success_shape(monkeypatch):
    _force_available(monkeypatch, True)
    monkeypatch.setattr(personal_api, "image_search", _FakeSearch())
    r = client.post("/search/text", json={"query": "a dog on the beach", "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "text->image"
    assert body["backend"] == "openclip-vit-h14"
    assert body["count"] == 1
    assert body["results"][0]["image_id"] == "42"
    assert "latency_ms" in body


def test_search_image_success_shape(monkeypatch):
    _force_available(monkeypatch, True)
    monkeypatch.setattr(personal_api, "image_search", _FakeSearch())
    r = client.post("/search/image?k=1", files={"file": ("q.jpg", b"\xff\xd8\xff", "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "image->image"
    assert body["results"][0]["score"] == 1.0
