"""VLM client: send (image + prompt) to Ollama and parse structured response."""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import List, Optional

import requests
from PIL import Image

from server.config import OLLAMA_URL, OLLAMA_MODEL, VLM_TIMEOUT_S
from server.models import VLMAction, VLMResponse
from server.prompts import PER_TURN_PROMPT, PRIOR_ANSWER_BLOCK

log = logging.getLogger(__name__)

_FALLBACK = VLMResponse(
    action=VLMAction.MOVE,
    guidance="I had trouble understanding the scene. Try walking forward a few steps and uploading another photo.",
    question=None,
    vlm_summary="",
)


def _build_prompt(
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_summary: Optional[str] = None,
) -> str:
    if prior_question and prior_answer:
        block = PRIOR_ANSWER_BLOCK.format(previous_question=prior_question, user_answer=prior_answer)
    else:
        block = ""
    return PER_TURN_PROMPT.format(
        goal=goal,
        goal_objects=", ".join(goal_objects) or "(none)",
        topomap_summary=topomap_summary or "(starting location)",
        detections_summary=detections_summary or "(no detections)",
        ocr_summary=ocr_summary or "(no text detected)",
        prior_answer_block=block,
    )


def _parse(text: str) -> Optional[VLMResponse]:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return VLMResponse(
            action=VLMAction(obj["action"]),
            guidance=str(obj.get("guidance", "")),
            question=obj.get("question"),
            vlm_summary=str(obj.get("vlm_summary", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("VLM JSON parse failed: %s", e)
        return None


def _generate(prompt: str, images: Optional[List[str]] = None) -> str:
    """One Ollama generate call. ``format=json`` makes the model emit parseable JSON."""
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    if images:
        body["images"] = images
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json().get("response", "")


def ask_about_image(image_pil: Image.Image, prompt: str) -> str:
    """TASK 3 helper — ask the VLM a free-form question about a (cropped) image.

    Used to confirm a goal detection. Plain text reply (no ``format=json``) so a
    short "yes"/"no" comes back cleanly. Encodes the PIL image to base64 JPEG.
    """
    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "images": [img_b64]}
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json().get("response", "")


def warm_up() -> None:
    """Prime the model so the first real photo doesn't eat the cold-start cost.

    Best-effort: failures (Ollama down/not pulled) are logged, never raised.
    """
    try:
        _generate("Reply with an empty JSON object.")
        log.info("VLM warm-up complete")
    except Exception as e:
        log.warning("VLM warm-up failed (continuing): %s", e)


def decide(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_summary: Optional[str] = None,
) -> VLMResponse:
    prompt = _build_prompt(goal, goal_objects, topomap_summary, detections_summary, prior_question, prior_answer, ocr_summary=ocr_summary)
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        log.warning("VLM could not read image %s: %s", image_path, e)
        return _FALLBACK

    # Retry once: the first response (especially right after startup) is the most
    # likely to be unparseable, and a single retry recovers most of those.
    for attempt in (1, 2):
        try:
            text = _generate(prompt, images=[img_b64])
        except Exception as e:
            log.warning("VLM call failed (attempt %d/2): %s", attempt, e)
            continue
        parsed = _parse(text)
        if parsed is not None:
            return parsed
        log.warning("VLM unparseable response (attempt %d/2): %.200s", attempt, text)
    return _FALLBACK
