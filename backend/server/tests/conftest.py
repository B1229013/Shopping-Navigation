import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("UNIGOAL_TEST_MODE", "1")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_image_quality_gate(request, monkeypatch):
    """Most API tests upload a flat grey placeholder JPEG, which the blur check
    (server/photo_quality.py) would rightly reject. Only test_photo_quality keeps it on."""
    if request.module.__name__.endswith("test_photo_quality"):
        return
    from server import photo_quality
    monkeypatch.setattr(photo_quality, "MIN_SHARPNESS", -1.0)
    monkeypatch.setattr(photo_quality, "MIN_BRIGHTNESS", -1.0)


@pytest.fixture(autouse=True)
def _fresh_route_graph_cache(monkeypatch):
    """Route graphs are cached per place name and every test calls its place "test",
    so one test's map would leak into the next."""
    import sys
    srv = sys.modules.get("server.server")
    if srv is not None:
        monkeypatch.setattr(srv, "_editor_graphs", {})


@pytest.fixture(autouse=True)
def _no_label_embeddings(monkeypatch):
    """Label embeddings would be fetched from OpenAI (the repo .env has a key) — never in
    tests. Tests of the embedding path fill visual_localization._emb_cache themselves."""
    from server import visual_localization as VL
    monkeypatch.setattr(VL, "_emb_cache", {})
    monkeypatch.setattr(VL, "_emb_persisted", set())
    monkeypatch.setattr(VL, "_emb_loaded", True)
    monkeypatch.setattr(VL, "_emb_on", False)


@pytest.fixture(autouse=True)
def _no_server_dino(monkeypatch):
    """Server-side DINOv2 would load (or download) facebook/dinov2-base for every API
    test photo; tests that cover it turn it back on and fake the embedding."""
    from server import config
    monkeypatch.setattr(config, "LOC_SERVER_DINO", False)


@pytest.fixture(autouse=True)
def _sessions_in_tmp(tmp_path, monkeypatch):
    """Session records and photos go to a temp folder, not backend/output/sessions/ —
    every full run used to leave ~33 fake sessions among the real store-test records."""
    from server import config
    import server.server as srv
    root = tmp_path / "sessions"
    monkeypatch.setattr(config, "OUTPUT_ROOT", root)     # photos (ensure_output_dir)
    monkeypatch.setattr(srv, "OUTPUT_ROOT", root)        # the records
