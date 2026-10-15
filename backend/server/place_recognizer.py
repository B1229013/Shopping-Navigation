"""Place recognizer — identify which known place the user is in, or flag unknown.

When the user uploads their first photo, this module matches the detected objects
and OCR texts against ALL places in the Neo4j reference database. Each place has
a "fingerprint" — the union of all object labels across its photo nodes. Matching
is done via weighted Jaccard similarity + OCR bonus, same as visual_localization
but at the place level instead of the node level.

Three outcomes:
  1. KNOWN    — high-confidence match to exactly one place
  2. AMBIGUOUS — multiple places score similarly (ask user to confirm)
  3. UNKNOWN  — no place matches well enough (offer to start a new map)

Usage:
    from server.place_recognizer import recognize_place, PlaceMatch

    result = recognize_place(
        detected_labels=["shelf", "refrigerator", "checkout"],
        ocr_texts=["家樂福", "特價"],
        neo4j=neo4j_client,
    )
    if result.status == "known":
        print(f"You are at {result.place}")
    elif result.status == "ambiguous":
        print(f"Could be: {[m.place for m in result.candidates]}")
    else:
        print("Unknown place — start a new map?")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from server.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)


@dataclass
class PlaceFingerprint:
    """Compact signature of a known place for fast matching."""
    place: str
    object_labels: Set[str]       # union of all normalized object labels
    ocr_texts: Set[str]           # union of all OCR texts
    photo_count: int
    object_count: int
    # Distinctive objects: labels that appear in THIS place but rarely in others
    distinctive_labels: Set[str] = field(default_factory=set)


@dataclass
class PlaceMatch:
    """One candidate result from place recognition."""
    place: str
    score: float
    reasoning: str
    photo_count: int
    object_count: int


@dataclass
class PlaceRecognitionResult:
    """Result of matching against all known places."""
    status: str                   # "known", "ambiguous", "unknown"
    place: Optional[str]          # best match (if known or ambiguous)
    confidence: float             # 0.0–1.0
    candidates: List[PlaceMatch]  # all scored candidates, best-first
    reasoning: str


# ── Thresholds ────────────────────────────────────────────────────────────

# Minimum score to consider a place match valid
PLACE_MATCH_THRESHOLD = 0.20

# If top score is above this AND gap to #2 is large enough, auto-select
PLACE_CONFIDENT_THRESHOLD = 0.30

# Minimum gap between #1 and #2 to avoid ambiguity
PLACE_AMBIGUITY_GAP = 0.08


# ── Label normalization (same as visual_localization) ─────────────────────

_SYNONYMS = {
    "shelf": {"shelf", "shelves", "shelving", "display rack", "rack",
              "store shelf", "display shelf"},
    "refrigerator": {"refrigerator", "fridge", "cooler", "refrigerated section",
                     "freezer", "frozen section"},
    "snack": {"snack bags", "snack packs", "snacks", "chips", "snack bag"},
    "aisle": {"aisle", "walkway", "corridor", "shopping aisle"},
    "checkout": {"checkout counter", "cashier counter", "cashier", "register",
                 "checkout", "cash register"},
    "bottle": {"bottles", "bottled water", "beverage bottles", "bottle",
               "water bottle"},
}


def _normalize(label: str) -> str:
    label = label.lower().strip()
    for canonical, variants in _SYNONYMS.items():
        if label in variants:
            return canonical
    return label


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Build fingerprints ───────────────────────────────────────────────────

def build_fingerprints(neo4j: Neo4jClient) -> List[PlaceFingerprint]:
    """Load all places from Neo4j and build their fingerprints.

    This queries ALL places in one go, grouped by place name.
    Results are cached by the caller (neo4j_client caches the ref map,
    and we cache fingerprints at module level).
    """
    # Get all distinct places
    places_rows = neo4j._query("""
        MATCH (t:TopoNode {ntype: 'photo'})
        RETURN DISTINCT t.place AS place, count(t) AS photo_count
        ORDER BY photo_count DESC
    """)

    fingerprints: List[PlaceFingerprint] = []

    for pr in places_rows:
        place_name = pr["place"]
        if not place_name:
            continue

        # Load reference map for this place (uses cache if available)
        ref = neo4j.load_reference_map(place_name)

        labels: Set[str] = set()
        ocr: Set[str] = set()
        obj_count = 0

        for photo in ref.photos.values():
            for obj in photo.objects:
                if obj.label:
                    labels.add(_normalize(obj.label))
                if obj.ocr_text:
                    ocr.add(obj.ocr_text.lower().strip())
                obj_count += 1

        fingerprints.append(PlaceFingerprint(
            place=place_name,
            object_labels=labels,
            ocr_texts=ocr - {""},
            photo_count=len(ref.photos),
            object_count=obj_count,
        ))

    # Compute distinctive labels: labels unique to each place
    if len(fingerprints) > 1:
        all_labels_by_place = {fp.place: fp.object_labels for fp in fingerprints}
        for fp in fingerprints:
            others = set()
            for other_place, other_labels in all_labels_by_place.items():
                if other_place != fp.place:
                    others.update(other_labels)
            fp.distinctive_labels = fp.object_labels - others

    log.info("Built %d place fingerprints: %s",
             len(fingerprints),
             [(fp.place, len(fp.object_labels)) for fp in fingerprints])

    return fingerprints


# ── Module-level cache ────────────────────────────────────────────────────

_cached_fingerprints: Optional[List[PlaceFingerprint]] = None


def get_fingerprints(neo4j: Neo4jClient, force_reload: bool = False) -> List[PlaceFingerprint]:
    """Get or build cached place fingerprints."""
    global _cached_fingerprints
    if _cached_fingerprints is None or force_reload:
        _cached_fingerprints = build_fingerprints(neo4j)
    return _cached_fingerprints


def invalidate_fingerprint_cache():
    """Call when places are added/removed in Neo4j."""
    global _cached_fingerprints
    _cached_fingerprints = None


# ── Main recognition entry point ─────────────────────────────────────────

def recognize_place(
    detected_labels: List[str],
    ocr_texts: List[str],
    neo4j: Neo4jClient,
    *,
    force_reload: bool = False,
) -> PlaceRecognitionResult:
    """Match a user's live photo against all known places.

    Args:
        detected_labels: object labels from VLM/perception on the user's photo
        ocr_texts: OCR text strings found in the user's photo
        neo4j: connected Neo4j client

    Returns:
        PlaceRecognitionResult with status, best match, and all candidates.
    """
    fingerprints = get_fingerprints(neo4j, force_reload=force_reload)

    if not fingerprints:
        return PlaceRecognitionResult(
            status="unknown",
            place=None,
            confidence=0.0,
            candidates=[],
            reasoning="No places found in database",
        )

    # If only one place exists, just return it with moderate confidence
    if len(fingerprints) == 1:
        fp = fingerprints[0]
        live = {_normalize(l) for l in detected_labels if l}
        j = _jaccard(live, fp.object_labels)
        return PlaceRecognitionResult(
            status="known",
            place=fp.place,
            confidence=max(j, 0.5),  # single place gets benefit of doubt
            candidates=[PlaceMatch(
                place=fp.place, score=j,
                reasoning=f"Only known place (jaccard={j:.2f})",
                photo_count=fp.photo_count, object_count=fp.object_count,
            )],
            reasoning=f"Only one place in database: {fp.place}",
        )

    # Score each place
    live_labels = {_normalize(l) for l in detected_labels if l}
    live_ocr = {t.lower().strip() for t in ocr_texts if t}

    candidates: List[PlaceMatch] = []

    for fp in fingerprints:
        # Base: Jaccard on object labels
        j = _jaccard(live_labels, fp.object_labels)

        # OCR bonus: matching OCR text (e.g. store name on signs)
        ocr_matches = live_ocr & fp.ocr_texts - {""}
        ocr_bonus = min(len(ocr_matches) * 0.15, 0.3)

        # Distinctive label bonus: if the photo contains labels unique to this place
        distinctive_matches = live_labels & fp.distinctive_labels
        distinctive_bonus = min(len(distinctive_matches) * 0.10, 0.2)

        total = j + ocr_bonus + distinctive_bonus

        parts = [f"jaccard={j:.2f}"]
        if ocr_matches:
            parts.append(f"ocr={ocr_matches}")
        if distinctive_matches:
            parts.append(f"distinctive={distinctive_matches}")

        candidates.append(PlaceMatch(
            place=fp.place,
            score=total,
            reasoning=" | ".join(parts),
            photo_count=fp.photo_count,
            object_count=fp.object_count,
        ))

    candidates.sort(key=lambda c: -c.score)

    best = candidates[0]
    runner = candidates[1] if len(candidates) > 1 else None
    gap = best.score - (runner.score if runner else 0)

    # Decide status
    if best.score < PLACE_MATCH_THRESHOLD:
        return PlaceRecognitionResult(
            status="unknown",
            place=None,
            confidence=best.score,
            candidates=candidates,
            reasoning=f"Best score {best.score:.2f} below threshold {PLACE_MATCH_THRESHOLD}",
        )

    if best.score >= PLACE_CONFIDENT_THRESHOLD and gap >= PLACE_AMBIGUITY_GAP:
        return PlaceRecognitionResult(
            status="known",
            place=best.place,
            confidence=min(best.score / 0.5, 1.0),
            candidates=candidates,
            reasoning=f"Confident match: {best.place} ({best.reasoning})",
        )

    # Ambiguous: multiple candidates too close
    return PlaceRecognitionResult(
        status="ambiguous",
        place=best.place,  # best guess
        confidence=min(best.score / 0.5, 1.0),
        candidates=candidates,
        reasoning=f"Ambiguous: {best.place} ({best.score:.2f}) vs {runner.place} ({runner.score:.2f}), gap={gap:.2f}",
    )
