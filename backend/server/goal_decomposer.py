"""Decompose a free-text goal into a list of object/region prompts for GroundingDINO."""
from __future__ import annotations

import logging
import re
from typing import List

import requests

from server.config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL,
    GOAL_DECOMPOSE_TIMEOUT_S,
)
from server.prompts import GOAL_DECOMPOSE_PROMPT

log = logging.getLogger(__name__)

_MAX_ITEMS = 10


def _fallback(goal: str) -> List[str]:
    # Extract Chinese tokens (consecutive CJK chars) and English words
    tokens = re.findall(r'[一-鿿㐀-䶿]+|[a-zA-Z]+', goal)
    stop = {"find", "the", "a", "an", "to", "where", "is", "are", "all", "every", "x"}
    result = [w for w in tokens if w.lower() not in stop and len(w) > 0]
    return result or [goal.strip().lower()]


def _call_openai(prompt: str) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_completion_tokens": 256,
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
        if not OPENAI_API_KEY:
            log.warning("OPENAI_API_KEY not set — using fallback")
            return _fallback(goal)
        text = _call_openai(prompt)
    except Exception as e:
        log.warning("goal decompose failed: %s — falling back", e)
        return _fallback(goal)

    items = [s.strip().lower() for s in text.split(",")]
    items = [s for s in items if 1 <= len(s) <= 30]
    if not (1 <= len(items) <= _MAX_ITEMS):
        log.warning("goal decompose produced %d items — falling back", len(items))
        return _fallback(goal)
    return items
