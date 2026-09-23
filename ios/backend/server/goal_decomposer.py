"""Decompose a free-text goal into a list of object/region prompts for GroundingDINO."""
from __future__ import annotations

import logging
import re
from typing import Iterable, List, Tuple

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


_QTY_SUFFIX = re.compile(r"\s*x\s*\d+\s*$", re.IGNORECASE)


def _goal_items(goal: str) -> List[str]:
    """Product words from the raw goal string, e.g. "牛奶 x2, find the milk" ->
    ["牛奶", "milk"]. Splits on commas/、, strips the iOS " xN" quantity suffix and
    drops stop words via the same tokenizer as the LLM fallback."""
    items: List[str] = []
    for part in re.split(r"[,，、]", goal):
        part = _QTY_SUFFIX.sub("", part).strip().lower()
        if not part:
            continue
        for tok in _fallback(part):
            tok = tok.lower()
            if tok and tok not in items:
                items.append(tok)
    return items


def split_goal_objects(
    goal: str,
    goal_objects: List[str],
    extra_target_terms: Iterable[str] = (),
) -> Tuple[List[str], List[str]]:
    """Split decomposed ``goal_objects`` into (targets, context).

    ``decompose_goal`` intentionally mixes the product with its section and nearby
    landmarks ("dairy section", "cooler") so the detector has context to look for.
    Only the product and its variants may count as *the target* (for the arrival
    gate, crop verification and "goal visible in photo" hints); everything else
    is context. A decomposed object is a target when it contains, or is contained
    in, one of the goal's own product words (or ``extra_target_terms``, e.g. the
    GOAL_CLASS_MAP synonyms). The goal's own product words are always targets,
    listed first, even if the LLM omitted them.
    """
    items = _goal_items(goal)
    seeds = [t.lower().strip() for t in extra_target_terms if t and t.strip()]

    targets: List[str] = list(items)
    context: List[str] = []
    for g in goal_objects:
        gl = g.lower().strip()
        if not gl:
            continue
        if gl in targets:
            continue
        is_target = gl in seeds or any(it in gl or gl in it for it in items)
        (targets if is_target else context).append(gl)
    return targets, context
