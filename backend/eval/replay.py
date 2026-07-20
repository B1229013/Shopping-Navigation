"""Live replay: run photos through the real navigation pipeline to produce a
predictions file the harness can score.

Each photo is evaluated as an independent first-turn decision (no topomap
carry-over) so the score reflects per-frame perception + reasoning quality.
Requires the model stack (GroundingDINO, EasyOCR, Ollama) — run in the project
env, not the pure-Python interpreter.
"""
from __future__ import annotations

import glob
import logging
import os

from PIL import Image

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_FALLBACK_MARKER = "I had trouble understanding"


def predict_over_photos(photo_dir: str, goal: str) -> list[dict]:
    # Imported here so the pure harness modules don't pull the model stack.
    from server import scene
    from server.config import (
        ARRIVED_MIN_DETECTION_SCORE, OCR_ENABLED, OCR_LANGUAGES,
        OCR_MIN_CONFIDENCE, OCR_MAX_RESULTS,
    )
    from server.goal_decomposer import decompose_goal
    from server.models import VLMAction
    from server.ocr import OCR
    from server.perception import Perception
    from server.vlm import decide

    goal_objects = decompose_goal(goal)
    log.info("goal_objects: %s", goal_objects)

    perception = Perception()
    perception.load()
    ocr_engine = None
    if OCR_ENABLED:
        ocr_engine = OCR(languages=OCR_LANGUAGES)
        ocr_engine.load()

    photos = sorted(
        p for p in glob.glob(os.path.join(photo_dir, "*"))
        if os.path.splitext(p)[1].lower() in _IMAGE_EXTS
    )

    predictions: list[dict] = []
    for path in photos:
        img_w, img_h = Image.open(path).size
        detections = perception.detect(path, goal_objects)
        ocr_results = ocr_engine.read(path, OCR_MIN_CONFIDENCE, OCR_MAX_RESULTS) if ocr_engine else []

        ocr_summary = scene.format_ocr(ocr_results, img_w, img_h)
        ocr_matches = scene.match_ocr_to_goal(ocr_results, goal_objects)
        if ocr_matches:
            ocr_summary += "  SIGN MATCH: " + "; ".join(ocr_matches)

        resp = decide(
            image_path=path, goal=goal, goal_objects=goal_objects,
            topomap_summary="(starting location)",
            detections_summary=scene.format_detections(detections, img_w, img_h),
            prior_question=None, prior_answer=None, ocr_summary=ocr_summary,
        )
        resp = scene.verify_arrival(
            resp, detections, ocr_matches=ocr_matches, goal_objects=goal_objects,
            min_score=ARRIVED_MIN_DETECTION_SCORE,
        )

        predictions.append({
            "photo": os.path.basename(path),
            "predicted_action": resp.action.value,
            "guidance": resp.guidance,
            "detection_sides": scene.detection_sides(detections, img_w),
            "json_valid": _FALLBACK_MARKER not in resp.guidance,
            "arrival_corroborated": resp.action == VLMAction.ARRIVED,
        })
        log.info("%s -> %s", os.path.basename(path), resp.action.value)

    return predictions
