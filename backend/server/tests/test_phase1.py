"""Tests for Phase 1 (object-detection) improvements: TASK 1-5."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("UNIGOAL_TEST_MODE", "1")

from PIL import Image

from server.config import normalize_label, LABEL_SYNONYMS
from server.perception import verify_goal_detection, Detection
from server import scene
from server.models import VLMAction, VLMResponse


# ---- TASK 2: label normalization ------------------------------------------

def test_normalize_label_collapses_variants():
    assert normalize_label("milk carton") == "milk"
    assert normalize_label("Milk Bottle") == "milk"
    assert normalize_label("whole milk") == "milk"


def test_normalize_label_is_idempotent():
    for canonical in LABEL_SYNONYMS:
        assert normalize_label(normalize_label(canonical)) == normalize_label(canonical)


def test_normalize_label_passthrough_unknown():
    assert normalize_label("Banana") == "banana"


# ---- TASK 3: crop verification --------------------------------------------

def _img(w=100, h=100):
    return Image.new("RGB", (w, h), (200, 200, 200))


def test_verify_goal_detection_yes():
    ask = MagicMock(return_value="Yes, this is milk.")
    assert verify_goal_detection(_img(), [10, 10, 80, 80], "milk", ask) is True
    ask.assert_called_once()


def test_verify_goal_detection_no():
    ask = MagicMock(return_value="No, that is a cereal box.")
    assert verify_goal_detection(_img(), [10, 10, 80, 80], "milk", ask) is False


def test_verify_goal_detection_degenerate_box_is_false():
    ask = MagicMock(return_value="yes")
    assert verify_goal_detection(_img(), [50, 50, 50, 50], "milk", ask) is False
    ask.assert_not_called()


def test_verify_goal_detection_swallows_errors():
    ask = MagicMock(side_effect=RuntimeError("ollama down"))
    assert verify_goal_detection(_img(), [10, 10, 80, 80], "milk", ask) is False


# ---- TASK 3: arrival gating honours the crop verdict ----------------------

def _arrived():
    return VLMResponse(action=VLMAction.ARRIVED, guidance="here it is", question=None, vlm_summary="x")


def test_verify_arrival_kept_when_crop_confirms():
    resp = scene.verify_arrival(_arrived(), detections=[], ocr_matches=[],
                                goal_objects=["milk"], min_score=0.35, goal_verified=True)
    assert resp.action == VLMAction.ARRIVED


def test_verify_arrival_downgraded_when_crop_rejects_despite_detection():
    det = [Detection(label="milk", box=[0, 0, 10, 10], score=0.99)]
    resp = scene.verify_arrival(_arrived(), detections=det, ocr_matches=[],
                                goal_objects=["milk"], min_score=0.35, goal_verified=False)
    assert resp.action == VLMAction.ASK


def test_verify_arrival_unknown_falls_back_to_detection_evidence():
    det = [Detection(label="milk", box=[0, 0, 10, 10], score=0.99)]
    resp = scene.verify_arrival(_arrived(), detections=det, ocr_matches=[],
                                goal_objects=["milk"], min_score=0.35, goal_verified=None)
    assert resp.action == VLMAction.ARRIVED


# ---- TASK 5: OCR section confirmation -------------------------------------

def test_confirm_zone_by_ocr_matches_section():
    assert scene.confirm_zone_by_ocr(["DAIRY", "Aisle 5"], ["milk"]) is True


def test_confirm_zone_by_ocr_no_match():
    assert scene.confirm_zone_by_ocr(["Hardware", "Tools"], ["milk"]) is False


def test_confirm_zone_by_ocr_handles_empty():
    assert scene.confirm_zone_by_ocr([], ["milk"]) is False
    assert scene.confirm_zone_by_ocr(["", None], ["milk"]) is False
