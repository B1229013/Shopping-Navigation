"""Heading estimation and relative-direction instructions.

Each topological-map node stores up to 4 reference photos taken at
roughly 90° intervals (front / right / back / left).  When the user
takes a photo, we compare it against the directional reference set of
the matched node to determine which way they are facing.

With the user's heading known, absolute navigation actions
("go north") are converted into relative instructions
("turn right", "go straight", "turn around") that are far more
intuitive in an indoor environment where compass bearings are useless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Cardinal direction slots — one per reference photo
# ---------------------------------------------------------------------------

class DirectionSlot(str, Enum):
    """The four directional slots stored per node."""
    FRONT = "front"      # primary heading of the node (≈ heading_deg)
    RIGHT = "right"      # +90°
    BACK  = "back"       # +180°
    LEFT  = "left"       # +270°


@dataclass
class DirectionalPhoto:
    """One of the 4 reference photos at a node."""
    slot: DirectionSlot
    heading_deg: float        # absolute heading in degrees [0, 360)
    photo_file: str           # filename or path
    objects: list             # detected objects (labels/OCR) for matching


# ---------------------------------------------------------------------------
# Heading estimation
# ---------------------------------------------------------------------------

def _normalize_angle(deg: float) -> float:
    """Normalize to [0, 360)."""
    return deg % 360


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference a - b, in [-180, 180)."""
    d = (a - b) % 360
    return d if d < 180 else d - 360


def slot_headings(node_heading: float) -> dict[DirectionSlot, float]:
    """Compute the four absolute headings from a node's primary heading."""
    h = _normalize_angle(node_heading)
    return {
        DirectionSlot.FRONT: h,
        DirectionSlot.RIGHT: _normalize_angle(h + 90),
        DirectionSlot.BACK:  _normalize_angle(h + 180),
        DirectionSlot.LEFT:  _normalize_angle(h + 270),
    }


def estimate_heading(
    user_labels: set[str],
    user_ocr: set[str],
    directional_photos: Sequence[DirectionalPhoto],
) -> Optional[float]:
    """Estimate the user's heading by comparing their photo against
    directional reference photos at the matched node.

    Uses the same Jaccard + OCR scoring as visual_localization, but
    runs it per-direction instead of per-node.

    Returns the estimated absolute heading in degrees, or None if
    no directional photos are available.
    """
    if not directional_photos:
        return None

    best_heading: Optional[float] = None
    best_score = -1.0

    for dp in directional_photos:
        ref_labels = {o.get("label", "").lower().strip()
                      for o in dp.objects if o.get("label")}
        ref_ocr = {o.get("ocr_text", "").lower().strip()
                   for o in dp.objects if o.get("ocr_text")}

        # Jaccard on labels
        union = user_labels | ref_labels
        jaccard = len(user_labels & ref_labels) / len(union) if union else 0.0

        # OCR bonus
        ocr_matches = user_ocr & ref_ocr - {""}
        ocr_bonus = min(len(ocr_matches) * 0.2, 0.4)

        score = jaccard + ocr_bonus
        if score > best_score:
            best_score = score
            best_heading = dp.heading_deg

    return best_heading


def best_matching_slot(
    user_labels: set[str],
    user_ocr: set[str],
    directional_photos: Sequence[DirectionalPhoto],
) -> Optional[DirectionSlot]:
    """Return which of the 4 directional slots best matches the user's photo."""
    if not directional_photos:
        return None

    best_slot: Optional[DirectionSlot] = None
    best_score = -1.0

    for dp in directional_photos:
        ref_labels = {o.get("label", "").lower().strip()
                      for o in dp.objects if o.get("label")}
        ref_ocr = {o.get("ocr_text", "").lower().strip()
                   for o in dp.objects if o.get("ocr_text")}

        union = user_labels | ref_labels
        jaccard = len(user_labels & ref_labels) / len(union) if union else 0.0
        ocr_matches = user_ocr & ref_ocr - {""}
        ocr_bonus = min(len(ocr_matches) * 0.2, 0.4)

        score = jaccard + ocr_bonus
        if score > best_score:
            best_score = score
            best_slot = dp.slot

    return best_slot


# ---------------------------------------------------------------------------
# Relative direction instructions
# ---------------------------------------------------------------------------

class RelativeDirection(str, Enum):
    """Turn instruction relative to the user's current heading."""
    STRAIGHT     = "straight"       # |Δ| ≤ 45°
    SLIGHT_RIGHT = "slight_right"   # 45° < Δ ≤ 80°
    RIGHT        = "right"          # 80° < Δ ≤ 120°
    SHARP_RIGHT  = "sharp_right"    # 120° < Δ ≤ 170°
    BEHIND       = "behind"         # |Δ| > 170°
    SHARP_LEFT   = "sharp_left"     # -170° < Δ ≤ -120°
    LEFT         = "left"           # -120° < Δ ≤ -80°
    SLIGHT_LEFT  = "slight_left"    # -80° < Δ ≤ -45°


_DIRECTION_ZH = {
    RelativeDirection.STRAIGHT:     "直走",
    RelativeDirection.SLIGHT_RIGHT: "稍微右轉",
    RelativeDirection.RIGHT:        "右轉",
    RelativeDirection.SHARP_RIGHT:  "大幅右轉",
    RelativeDirection.BEHIND:       "迴轉",
    RelativeDirection.SHARP_LEFT:   "大幅左轉",
    RelativeDirection.LEFT:         "左轉",
    RelativeDirection.SLIGHT_LEFT:  "稍微左轉",
}

_DIRECTION_EN = {
    RelativeDirection.STRAIGHT:     "Go straight",
    RelativeDirection.SLIGHT_RIGHT: "Bear right",
    RelativeDirection.RIGHT:        "Turn right",
    RelativeDirection.SHARP_RIGHT:  "Sharp right",
    RelativeDirection.BEHIND:       "Turn around",
    RelativeDirection.SHARP_LEFT:   "Sharp left",
    RelativeDirection.LEFT:         "Turn left",
    RelativeDirection.SLIGHT_LEFT:  "Bear left",
}


def compute_relative_direction(
    user_heading: float,
    target_heading: float,
) -> RelativeDirection:
    """Given the user's current heading and the absolute heading they
    need to walk toward, return the relative turn instruction.

    Both angles are in degrees [0, 360).
    """
    delta = _angle_diff(target_heading, user_heading)

    if abs(delta) <= 45:
        return RelativeDirection.STRAIGHT
    if delta > 170 or delta < -170:
        return RelativeDirection.BEHIND
    if 45 < delta <= 80:
        return RelativeDirection.SLIGHT_RIGHT
    if 80 < delta <= 120:
        return RelativeDirection.RIGHT
    if 120 < delta <= 170:
        return RelativeDirection.SHARP_RIGHT
    if -80 <= delta < -45:
        return RelativeDirection.SLIGHT_LEFT
    if -120 <= delta < -80:
        return RelativeDirection.LEFT
    # -170 <= delta < -120
    return RelativeDirection.SHARP_LEFT


def relative_direction_text(
    direction: RelativeDirection,
    lang: str = "zh",
) -> str:
    """Human-readable text for a relative direction."""
    if lang == "zh":
        return _DIRECTION_ZH.get(direction, str(direction.value))
    return _DIRECTION_EN.get(direction, str(direction.value))


# ---------------------------------------------------------------------------
# Convert absolute leg actions into relative instructions
# ---------------------------------------------------------------------------

def heading_between_nodes(
    from_x: float, from_y: float,
    to_x: float, to_y: float,
) -> float:
    """Compute the absolute heading (degrees, 0°=North/+Y, 90°=East/+X)
    from one node's position to another.
    """
    dx = to_x - from_x
    dy = to_y - from_y
    if dx == 0 and dy == 0:
        return 0.0
    # atan2 with North-up convention: angle from +Y axis, clockwise
    angle_rad = math.atan2(dx, dy)
    return _normalize_angle(math.degrees(angle_rad))


@dataclass
class RelativeInstruction:
    """One step of relative navigation."""
    from_node: int
    to_node: int
    direction: RelativeDirection
    text_zh: str
    text_en: str
    distance_m: float
    # After executing this step, the user's new heading
    new_heading: float


def convert_leg_to_relative(
    path_nodes: List[int],
    node_positions: dict[int, tuple[float, float]],
    user_heading: float,
    distances: Optional[dict[tuple[int, int], float]] = None,
) -> List[RelativeInstruction]:
    """Convert an absolute A* path into relative turn-by-turn instructions.

    Args:
        path_nodes: ordered node IDs [from, ..., to]
        node_positions: {node_id: (x, y)} lookup
        user_heading: user's current facing direction in degrees
        distances: optional {(a,b): meters} edge distances

    Returns:
        List of RelativeInstruction, one per edge in the path.
    """
    if len(path_nodes) < 2:
        return []

    instructions: List[RelativeInstruction] = []
    current_heading = user_heading

    for i in range(len(path_nodes) - 1):
        a, b = path_nodes[i], path_nodes[i + 1]
        ax, ay = node_positions.get(a, (0.0, 0.0))
        bx, by = node_positions.get(b, (0.0, 0.0))

        # Absolute heading of this edge
        edge_heading = heading_between_nodes(ax, ay, bx, by)

        # Relative turn from current heading
        rel = compute_relative_direction(current_heading, edge_heading)

        dist = 0.0
        if distances and (a, b) in distances:
            dist = distances[(a, b)]

        instructions.append(RelativeInstruction(
            from_node=a,
            to_node=b,
            direction=rel,
            text_zh=relative_direction_text(rel, "zh"),
            text_en=relative_direction_text(rel, "en"),
            distance_m=dist,
            new_heading=edge_heading,
        ))

        # After this step, the user faces the direction they walked
        current_heading = edge_heading

    return instructions
