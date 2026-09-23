"""Smoke test — only runs when SMOKE=1 env var is set, since it needs GPU + weights."""
import os
import pytest
from pathlib import Path
from server.perception import Perception
from server.config import SAM_WEIGHTS, GROUNDINGDINO_WEIGHTS


@pytest.mark.skipif(os.environ.get("SMOKE") != "1", reason="GPU smoke test")
def test_perception_loads_and_detects():
    if not (SAM_WEIGHTS.exists() and GROUNDINGDINO_WEIGHTS.exists()):
        pytest.skip("model weights not present")

    p = Perception()
    p.load()

    sample = Path("/home/user/UniGoal/assets/demo_real.gif")
    if not sample.exists():
        pytest.skip("no sample image")
    detections = p.detect(str(sample), ["person", "wall"])
    assert isinstance(detections, list)
