"""
Unit tests for ImageSearchService gating logic (src/rag/image_search_service.py).

The service is the lazy/gated bridge from the cached CLIP gallery to the /search/* API.
These tests exercise only the gating/metadata logic (cache present? loaded? which backend?)
by monkeypatching the cache lookup — they never import Torch/OpenCLIP, so they run in the
lightweight requirements_simple CI env.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.image_search_service import ImageSearchService  # noqa: E402


def test_available_false_when_no_cache(monkeypatch):
    monkeypatch.setattr(ImageSearchService, "_which_cache", staticmethod(lambda: None))
    assert ImageSearchService.available() is False
    info = ImageSearchService().info()
    assert info["available"] is False
    assert info["loaded"] is False
    assert info["backend"] is None


def test_available_true_reflects_picked_backend(monkeypatch):
    picked = ("openclip-vit-h14", Path("/tmp/does-not-matter.pkl"), "openclip")
    monkeypatch.setattr(ImageSearchService, "_which_cache", staticmethod(lambda: picked))
    assert ImageSearchService.available() is True
    info = ImageSearchService().info()
    assert info["available"] is True
    assert info["backend"] == "openclip-vit-h14"
    assert info["loaded"] is False  # info() must not trigger a model load


def test_ensure_loaded_raises_without_cache(monkeypatch):
    monkeypatch.setattr(ImageSearchService, "_which_cache", staticmethod(lambda: None))
    svc = ImageSearchService()
    with pytest.raises(RuntimeError, match="No gallery cache"):
        svc.ensure_loaded()


def test_backend_preference_order_prefers_openclip(tmp_path, monkeypatch):
    """When both galleries exist, the stronger OpenCLIP backbone wins."""
    clip_cache = tmp_path / "clip_gallery.pkl"
    oc_cache = tmp_path / "openclip_gallery.pkl"
    clip_cache.write_bytes(b"x")
    oc_cache.write_bytes(b"x")
    monkeypatch.setattr(
        "rag.image_search_service._BACKENDS",
        [
            ("openclip-vit-h14", oc_cache, "openclip"),
            ("clip-vit-b32", clip_cache, "clip"),
        ],
    )
    label, _, key = ImageSearchService._which_cache()
    assert (label, key) == ("openclip-vit-h14", "openclip")
