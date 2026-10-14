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
