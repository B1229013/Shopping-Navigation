"""VLM client: send (image + prompt) to Gemini (default) or Ollama, parse structured response."""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import List, Optional

import requests
from PIL import Image

from server.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL,
    OLLAMA_URL, OLLAMA_MODEL,
    VLM_TIMEOUT_S, VLM_BACKEND,
)
from server.models import VLMAction, VLMResponse
from server.prompts import PER_TURN_PROMPT, PRIOR_ANSWER_BLOCK

log = logging.getLogger(__name__)

_FALLBACK = VLMResponse(
    action=VLMAction.MOVE,
    guidance="I had trouble understanding the scene. Try walking forward a few steps and uploading another photo.",
    question=None,
    vlm_summary="",
)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


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


# ── Gemini backend ──

def _gemini_generate(prompt: str, image_b64: Optional[str] = None) -> str:
    url = _GEMINI_URL.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)
    parts: list = []
    if image_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    parts.append({"text": prompt})
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }
    r = requests.post(url, json=body, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    resp = r.json()
    candidates = resp.get("candidates", [])
    if not candidates:
        return ""
    text_parts = [p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])]
    return "".join(text_parts)


# ── Ollama backend ──

def _ollama_generate(prompt: str, images: Optional[List[str]] = None) -> str:
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    if images:
        body["images"] = images
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json().get("response", "")


# ── OpenAI backend ──

def _openai_generate(prompt: str, image_b64: Optional[str] = None) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    content: list = []
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    content.append({"type": "text", "text": prompt})
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, json=body, headers=headers, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    choices = r.json().get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


# ── Unified dispatch ──

def _generate(prompt: str, image_b64: Optional[str] = None) -> str:
    if VLM_BACKEND == "gemini":
        return _gemini_generate(prompt, image_b64=image_b64)
    elif VLM_BACKEND == "openai":
        return _openai_generate(prompt, image_b64=image_b64)
    else:
        return _ollama_generate(prompt, images=[image_b64] if image_b64 else None)


def ask_about_image(image_pil: Image.Image, prompt: str) -> str:
    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return _generate(prompt, image_b64=img_b64)


def warm_up() -> None:
    backend = VLM_BACKEND
    if backend == "gemini" and not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set — VLM warm-up skipped")
        return
    if backend == "openai" and not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — VLM warm-up skipped")
        return
    try:
        _generate("Reply with an empty JSON object: {}")
        log.info("VLM warm-up complete (%s)", backend)
    except Exception as e:
        log.warning("VLM warm-up failed (continuing, backend=%s): %s", backend, e)


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
    if VLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set — cannot call VLM")
        return _FALLBACK
    if VLM_BACKEND == "openai" and not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot call VLM")
        return _FALLBACK

    prompt = _build_prompt(goal, goal_objects, topomap_summary, detections_summary, prior_question, prior_answer, ocr_summary=ocr_summary)
    try:
        img = Image.open(image_path).convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        log.info("VLM image resized to %s, %d KB", img.size, len(buf.getvalue()) // 1024)
    except Exception as e:
        log.warning("VLM could not read image %s: %s", image_path, e)
        return _FALLBACK

    for attempt in (1, 2):
        try:
            text = _generate(prompt, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM call failed (attempt %d/2): %s", attempt, e)
            continue
        parsed = _parse(text)
        if parsed is not None:
            return parsed
        log.warning("VLM unparseable response (attempt %d/2): %.200s", attempt, text)
    return _FALLBACK
