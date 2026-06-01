"""Decompose a free-text goal into a list of object/region prompts for GroundingDINO."""
from __future__ import annotations

import logging
import re
from typing import List

import requests

from server.config import OLLAMA_URL, OLLAMA_MODEL, GOAL_DECOMPOSE_TIMEOUT_S
from server.prompts import GOAL_DECOMPOSE_PROMPT

log = logging.getLogger(__name__)

_MAX_ITEMS = 10


def _fallback(goal: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    stop = {"find", "the", "a", "an", "to", "where", "is", "are", "all", "every"}
    return [w for w in words if w not in stop] or [goal.strip().lower()]


def decompose_goal(goal: str) -> List[str]:
    prompt = GOAL_DECOMPOSE_PROMPT.format(goal=goal)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=GOAL_DECOMPOSE_TIMEOUT_S,
        )
        r.raise_for_status()
        text = r.json().get("response", "")
    except Exception as e:
        log.warning("goal decompose failed: %s — falling back", e)
        return _fallback(goal)

    items = [s.strip().lower() for s in text.split(",")]
    items = [s for s in items if 1 <= len(s) <= 30 and s.isascii()]
    if not (1 <= len(items) <= _MAX_ITEMS):
        log.warning("goal decompose produced %d items — falling back", len(items))
        return _fallback(goal)
    return items
