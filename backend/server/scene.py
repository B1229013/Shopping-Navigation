"""Spatial scene formatting for the VLM prompt.

Pure (no model/torch imports) so it is fast to unit-test. Converts detection and
OCR bounding boxes into coarse, human-readable positions ("right, near") that
give the VLM the spatial grounding it needs to produce correct directions.

Inputs are duck-typed: a detection is anything with ``.label``, ``.box``
([x1,y1,x2,y2] absolute pixels) and ``.score``; an OCR result is anything with
``.text``, ``.confidence`` and ``.bbox`` (four [x,y] corner points).
"""
from __future__ import annotations

import difflib

from server.models import VLMAction, VLMResponse

# Bilingual (EN / 中文) category synonyms so an OCR'd aisle/section sign can be
# matched to a goal even when the words differ. Extend as needed per store.
SYNONYMS: dict[str, list[str]] = {
    "milk": ["dairy", "refrigerated", "牛奶", "鮮奶", "乳製品"],
    "cheese": ["dairy", "refrigerated", "起司", "乳酪", "乳製品"],
    "egg": ["eggs", "dairy", "refrigerated", "蛋", "雞蛋"],
    "eggs": ["egg", "dairy", "refrigerated", "蛋", "雞蛋"],
    "refrigerator": ["fridge", "freezer", "冰箱", "冷藏", "冷凍"],
    "fire extinguisher": ["extinguisher", "滅火器", "消防"],
    "bread": ["bakery", "baked", "麵包", "烘焙"],
    "vegetable": ["produce", "vegetables", "fresh", "蔬菜", "生鮮"],
    "fruit": ["produce", "fruits", "fresh", "水果", "生鮮"],
    "produce": ["vegetables", "fruit", "fresh", "蔬果", "生鮮"],
    "frozen": ["freezer", "frozen foods", "冷凍"],
    "meat": ["butcher", "肉", "肉品", "生鮮"],
    "drink": ["beverage", "beverages", "drinks", "soda", "juice", "water", "飲料"],
    "snack": ["snacks", "零食", "餅乾"],
    "checkout": ["cashier", "checkout", "register", "結帳", "收銀"],
    "exit": ["出口"],
    "entrance": ["入口"],
    "toilet": ["restroom", "washroom", "wc", "廁所", "洗手間"],
}

# Fuzzy ratio at/above which an OCR string is considered a match for a term.
OCR_MATCH_RATIO = 0.85

# Box area as a fraction of the image at/above which an object is called "near".
NEAR_AREA_RATIO = 0.15

_HORIZONTAL = ("left", "center", "right")
_VERTICAL = ("top", "middle", "bottom")


def _attr(item, name):
    """Read a field from either an object (attribute) or a dict (key)."""
    return item[name] if isinstance(item, dict) else getattr(item, name)


def _third(frac: float) -> int:
    if frac < 1 / 3:
        return 0
    if frac < 2 / 3:
        return 1
    return 2


def _horizontal_side(center_x: float, img_w: int) -> str:
    if not img_w:
        return "center"
    return _HORIZONTAL[_third(center_x / img_w)]


def _position(box: list[float], img_w: int, img_h: int) -> tuple[str, str, str]:
    """Return (horizontal, vertical, depth) for an [x1,y1,x2,y2] box."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    horizontal = _horizontal_side(cx, img_w)
    vertical = _VERTICAL[_third(cy / img_h)] if img_h else "middle"
    area_ratio = (abs(x2 - x1) * abs(y2 - y1)) / (img_w * img_h) if img_w and img_h else 0.0
    depth = "near" if area_ratio >= NEAR_AREA_RATIO else "far"
    return horizontal, vertical, depth


def format_detections(detections, img_w: int, img_h: int) -> str:
    """Render detections with positions, e.g. ``refrigerator (91%) - right middle, near``."""
    if not detections:
        return "(none)"
    parts = []
    for d in detections:
        h, v, depth = _position(_attr(d, "box"), img_w, img_h)
        parts.append(f"{_attr(d, 'label')} ({_attr(d, 'score'):.0%}) - {h} {v}, {depth}")
    return "; ".join(parts)


def _bbox_center(bbox: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def format_ocr(ocr_results, img_w: int, img_h: int) -> str:
    """Render OCR text with its horizontal side, e.g. ``"Dairy" (98%) - left``."""
    if not ocr_results:
        return "(no text detected)"
    parts = []
    for r in ocr_results:
        cx, _ = _bbox_center(r.bbox)
        side = _horizontal_side(cx, img_w)
        parts.append(f'"{r.text}" ({r.confidence:.0%}) - {side}')
    return "; ".join(parts)


def _goal_terms(goal_objects) -> set[str]:
    """Goal words plus their bilingual synonyms, lowercased."""
    terms: set[str] = set()
    for g in goal_objects:
        gl = g.lower().strip()
        if not gl:
            continue
        terms.add(gl)
        terms.update(s.lower() for s in SYNONYMS.get(gl, []))
        # also pull in the synonym group if the goal word IS a synonym of some key
        for key, syns in SYNONYMS.items():
            low = [s.lower() for s in syns]
            if gl == key or gl in low:
                terms.add(key)
                terms.update(low)
    return terms


def match_ocr_to_goal(ocr_results, goal_objects) -> list[str]:
    """OCR strings that likely indicate the goal's section/aisle.

    In a store the sign is the most reliable signal, so a match here is strong
    evidence. Matching is case-insensitive substring or fuzzy (>= OCR_MATCH_RATIO),
    against the goal words and their synonyms.
    """
    terms = _goal_terms(goal_objects)
    out: list[str] = []
    seen: set[str] = set()
    for r in ocr_results:
        text = r.text.lower().strip()
        if not text:
            continue
        for term in terms:
            hit = term in text or text in term or \
                difflib.SequenceMatcher(None, term, text).ratio() >= OCR_MATCH_RATIO
            if hit:
                if r.text not in seen:
                    out.append(f'"{r.text}" (relates to your goal "{term}")')
                    seen.add(r.text)
                break
    return out


def confirm_zone_by_ocr(ocr_texts, goal_objects) -> bool:
    """TASK 5 — True if any OCR sign text names the goal product's store section.

    A plain-string counterpart to :func:`match_ocr_to_goal`: it takes raw OCR
    strings (not result objects) and the goal words, expands them with the section
    synonyms above, and reports whether the current zone's signage corroborates the
    goal. A True here is a strong store signal — the aisle sign agrees with the goal
    — so the caller can treat the zone as a candidate even on a borderline detection.
    """
    terms = _goal_terms(goal_objects)
    for text in ocr_texts:
        t = (text or "").lower().strip()
        if not t:
            continue
        for term in terms:
            if term in t or t in term or \
                    difflib.SequenceMatcher(None, term, t).ratio() >= OCR_MATCH_RATIO:
                return True
    return False


def _label_matches_goal(label: str, goal_objects) -> bool:
    label = label.lower()
    for g in goal_objects:
        g = g.lower()
        if g and (g in label or label in g):
            return True
    return False


def verify_arrival(resp: VLMResponse, detections, ocr_matches, goal_objects,
                   min_score: float, prior_was_confirm: bool = False,
                   goal_verified: bool | None = None) -> VLMResponse:
    """Gate ARRIVED on real evidence to prevent false "you've arrived" claims.

    ARRIVED is kept only if a goal-related object was detected at >= ``min_score``
    or an OCR sign matched the goal (``ocr_matches`` non-empty). Otherwise it is
    downgraded to an ASK that asks the user to confirm. If the user is already
    answering a confirm question (``prior_was_confirm``), ARRIVED is trusted so we
    never loop.

    ``goal_verified`` (TASK 3) is the result of cropping the goal detection and
    asking the VLM to confirm it: ``True`` is strong evidence (ARRIVED kept even
    without other signals), ``False`` is an explicit rejection that overrides a
    borderline detection (ARRIVED downgraded), and ``None`` means no crop check ran
    (fall back to the detection/OCR evidence test).
    """
    if resp.action != VLMAction.ARRIVED or prior_was_confirm:
        return resp

    if goal_verified is True:
        return resp
    if goal_verified is False:
        return _confirm_question(resp)

    detection_ok = any(
        _attr(d, "score") >= min_score and _label_matches_goal(_attr(d, "label"), goal_objects)
        for d in detections
    )
    if detection_ok or ocr_matches:
        return resp

    return _confirm_question(resp)


def _confirm_question(resp: VLMResponse) -> VLMResponse:
    return VLMResponse(
        action=VLMAction.ASK,
        guidance="I think you may have reached it, but I'm not certain.",
        question="Can you confirm you can see what you're looking for right in front of you?",
        vlm_summary=resp.vlm_summary,
    )


def detection_sides(detections, img_w: int) -> list[str]:
    """Unique horizontal sides (in first-seen order) where objects were detected.

    Used by the evaluation harness to check that MOVE guidance points at a side
    where something actually is.
    """
    sides: list[str] = []
    for d in detections:
        x1, _, x2, _ = _attr(d, "box")
        side = _horizontal_side((x1 + x2) / 2, img_w)
        if side not in sides:
            sides.append(side)
    return sides
