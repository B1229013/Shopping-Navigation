"""Pure metric functions over navigation turn records.

No model dependencies — fully unit-testable. A TurnRecord captures one photo's
outcome plus optional ground-truth labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnRecord:
    photo: str
    goal: str
    predicted_action: str  # "MOVE" | "ASK" | "ARRIVED"
    guidance: str = ""  # the guidance text returned to the user
    detection_sides: list[str] = field(default_factory=list)  # horizontal sides with detections
    json_valid: bool = True  # did the VLM response parse without falling back?
    arrival_corroborated: bool = False  # was an ARRIVED backed by a detection/OCR match?
    expected_action: Optional[str] = None  # ground-truth label, if known
    is_goal_frame: Optional[bool] = None  # ground-truth: is the goal visible here?


_DIRECTION_WORDS = ("left", "right")


def success_rate(records: list[TurnRecord]) -> float:
    """Fraction of labeled records whose predicted action matches the label."""
    labeled = [r for r in records if r.expected_action is not None]
    if not labeled:
        return 0.0
    correct = sum(1 for r in labeled if r.predicted_action == r.expected_action)
    return correct / len(labeled)


def false_arrival_rate(records: list[TurnRecord]) -> float:
    """Fraction of non-goal frames the system wrongly declared ARRIVED on.

    The worst failure mode: telling a shopper they have arrived when they have not.
    Denominator is frames labeled ``is_goal_frame == False``.
    """
    non_goal = [r for r in records if r.is_goal_frame is False]
    if not non_goal:
        return 0.0
    false_arrivals = sum(1 for r in non_goal if r.predicted_action == "ARRIVED")
    return false_arrivals / len(non_goal)


def _mentioned_sides(guidance: str) -> list[str]:
    text = guidance.lower()
    return [w for w in _DIRECTION_WORDS if re.search(rf"\b{w}\b", text)]


def direction_sanity(records: list[TurnRecord]) -> float:
    """Label-free self-consistency: of MOVE records whose guidance names a
    horizontal direction, the fraction with a detection on that same side.

    Flags guidance that confidently says "turn left" with nothing detected there.
    Denominator is directional MOVE records; returns 0.0 if there are none.
    """
    directional = [
        r for r in records
        if r.predicted_action == "MOVE" and _mentioned_sides(r.guidance)
    ]
    if not directional:
        return 0.0
    sane = sum(
        1 for r in directional
        if any(side in r.detection_sides for side in _mentioned_sides(r.guidance))
    )
    return sane / len(directional)


def json_validity_rate(records: list[TurnRecord]) -> float:
    """Fraction of turns whose VLM response parsed (no fallback). Empty -> 1.0."""
    if not records:
        return 1.0
    return sum(1 for r in records if r.json_valid) / len(records)


def arrival_corroboration_rate(records: list[TurnRecord]) -> float:
    """Of turns predicting ARRIVED, the fraction backed by a detection/OCR match."""
    arrived = [r for r in records if r.predicted_action == "ARRIVED"]
    if not arrived:
        return 0.0
    return sum(1 for r in arrived if r.arrival_corroborated) / len(arrived)


def summarize(records: list[TurnRecord]) -> dict:
    """Aggregate every metric with the denominator counts each was computed over."""
    return {
        "n_turns": len(records),
        "n_labeled": sum(1 for r in records if r.expected_action is not None),
        "n_non_goal": sum(1 for r in records if r.is_goal_frame is False),
        "n_arrived": sum(1 for r in records if r.predicted_action == "ARRIVED"),
        "n_directional": sum(
            1 for r in records
            if r.predicted_action == "MOVE" and _mentioned_sides(r.guidance)
        ),
        "success_rate": success_rate(records),
        "false_arrival_rate": false_arrival_rate(records),
        "direction_sanity": direction_sanity(records),
        "json_validity_rate": json_validity_rate(records),
        "arrival_corroboration_rate": arrival_corroboration_rate(records),
    }
