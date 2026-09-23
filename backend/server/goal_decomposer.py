"""Decompose a free-text goal into a list of object/region prompts for GroundingDINO."""
from __future__ import annotations

import logging
import re
from typing import List

import requests

from server.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL,
    OLLAMA_URL, OLLAMA_MODEL,
    GOAL_DECOMPOSE_TIMEOUT_S, VLM_BACKEND,
)
from server.prompts import GOAL_DECOMPOSE_PROMPT

log = logging.getLogger(__name__)

_MAX_ITEMS = 35

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def _fallback(goal: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    stop = {"find", "the", "a", "an", "to", "where", "is", "are", "all", "every"}
    return [w for w in words if w not in stop] or [goal.strip().lower()]


def _call_gemini(prompt: str) -> str:
    url = _GEMINI_URL.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
    }
    r = requests.post(url, json=body, timeout=GOAL_DECOMPOSE_TIMEOUT_S)
    r.raise_for_status()
    candidates = r.json().get("candidates", [])
    if not candidates:
        return ""
    return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")


def _call_ollama(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=GOAL_DECOMPOSE_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def _call_openai(prompt: str) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, json=body, headers=headers, timeout=GOAL_DECOMPOSE_TIMEOUT_S)
    r.raise_for_status()
    choices = r.json().get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def decompose_goal(goal: str) -> List[str]:
    prompt = GOAL_DECOMPOSE_PROMPT.format(goal=goal)

    try:
        if VLM_BACKEND == "gemini":
            if not GEMINI_API_KEY:
                log.warning("GEMINI_API_KEY not set — using fallback")
                return _fallback(goal)
            text = _call_gemini(prompt)
        elif VLM_BACKEND == "openai":
            if not OPENAI_API_KEY:
                log.warning("OPENAI_API_KEY not set — using fallback")
                return _fallback(goal)
            text = _call_openai(prompt)
        else:
            text = _call_ollama(prompt)
    except Exception as e:
        log.warning("goal decompose failed (%s): %s — falling back", VLM_BACKEND, e)
        return _fallback(goal)

    items = [s.strip() for s in text.split(",")]
    items = [s for s in items if 1 <= len(s) <= 40]
    if not (1 <= len(items) <= _MAX_ITEMS):
        log.warning("goal decompose produced %d items — falling back", len(items))
        return _fallback(goal)
    return items


# ── Multi-goal splitting ──

_QTY_RE = re.compile(r"\s*[xX×]\s*\d+\s*$")
_SPLIT_RE = re.compile(r"[,，、;；\n]+")


def split_multi_goals(goal: str) -> List[str]:
    """Split a combined goal string into individual item names.

    Handles formats like:
      "牛奶 x1, 麵包 x2"       → ["牛奶", "麵包"]
      "牛奶、麵包、衛生紙"      → ["牛奶", "麵包", "衛生紙"]
      "找飲水機"                → ["找飲水機"]   (single goal)
    """
    parts = _SPLIT_RE.split(goal)
    items = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = _QTY_RE.sub("", p).strip()
        if p:
            items.append(p)
    if not items:
        items = [goal.strip()]
    seen: List[str] = []
    for it in items:
        if it not in seen:
            seen.append(it)
    return seen
