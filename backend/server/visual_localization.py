"""Visual localization — match a live photo's detections against the Neo4j
reference topological map to correct the user's position.

Two matching strategies, used together:

1. **Object-set matching** (fast, no VLM call):
   Compare the set of detected object labels in the user's photo against each
   reference node's object set using Jaccard similarity + OCR bonus. This works
   well for distinctive areas (checkout, produce, frozen) but struggles with
   similar-looking aisles.

2. **VLM-assisted matching** (slower, 1 extra VLM call):
   Send the photo + reference candidates to the VLM and ask it to pick the best
   match. Activated only when the top object-set score is ambiguous (two or more
   candidates within 0.1 of each other).

The corrected position is returned as a reference node ID plus confidence score.
The caller (server.py) can use this to:
  - Set the session's starting position accurately
  - Correct PDR drift during navigation
  - Generate turn-by-turn directions from the corrected position
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from server.neo4j_client import Neo4jClient, RefMap, RefPhotoNode

log = logging.getLogger(__name__)


@dataclass
class LocalizationResult:
    """Result of matching a photo against the reference map."""
    matched_nid: Optional[int]     # reference photo node ID, or None if no match
    confidence: float              # 0.0–1.0
    method: str                    # "object_set" | "vlm" | "none"
    reasoning: str                 # human-readable explanation
    ref_node: Optional[RefPhotoNode] = None  # the matched reference node

    # Runner-up for diagnostics
    runner_up_nid: Optional[int] = None
    runner_up_score: float = 0.0

    # Heading estimation from directional photo matching
    matched_heading: Optional[float] = None   # estimated user heading in degrees [0, 360)
    matched_slot: Optional[str] = None        # which directional slot matched ("front"/"right"/"back"/"left")
    heading_confidence: float = 0.0           # 0.0–1.0, how certain the heading is


# ── Object-set matching ───────────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_label(label: str) -> str:
    """Basic label normalization for matching across different VLM runs."""
    label = label.lower().strip()
    # Collapse common variants
    for canonical, variants in _SYNONYMS.items():
        if label in variants:
            return canonical
    return label


_SYNONYMS = {
    "shelf": {"shelf", "shelves", "shelving", "display rack", "rack",
              "store shelf", "display shelf"},
    "refrigerator": {"refrigerator", "fridge", "cooler", "refrigerated section",
                     "freezer", "frozen section"},
    "snack": {"snack bags", "snack packs", "snacks", "chips", "snack bag"},
    "aisle": {"aisle", "walkway", "corridor"},
    "checkout": {"checkout counter", "cashier counter", "cashier", "register",
                 "checkout", "cash register"},
    "bottle": {"bottles", "bottled water", "beverage bottles", "bottle",
               "water bottle"},
    "fruit": {"fruit display", "fruits", "fruit", "oranges", "apples"},
    "vegetable": {"vegetable display", "vegetables", "produce section",
                  "packaged vegetables", "onions"},
}


def match_by_objects(
    detected_labels: List[str],
    ocr_texts: List[str],
    ref_map: RefMap,
    *,
    top_k: int = 5,
) -> List[tuple[int, float, str]]:
    """Score each reference photo node against the live detections.

    Returns a list of (nid, score, reason) sorted best-first, up to top_k.
    """
    if not ref_map.photos:
        return []

    live_labels = {_normalize_label(l) for l in detected_labels if l}
    live_ocr = {t.lower().strip() for t in ocr_texts if t}

    scores: list[tuple[int, float, str]] = []

    for nid, ref_node in ref_map.photos.items():
        ref_labels = {_normalize_label(o.label) for o in ref_node.objects}
        ref_ocr = {o.ocr_text.lower().strip() for o in ref_node.objects
                   if o.ocr_text}

        # Base: Jaccard similarity on normalized object labels
        j = _jaccard(live_labels, ref_labels)

        # OCR matching — the strongest localization signal in stores.
        # Use partial matching (substring) since OCR is noisy.
        ocr_match_count = 0
        ocr_matched_texts = set()
        for live_t in live_ocr:
            if not live_t or len(live_t) < 2:
                continue
            for ref_t in ref_ocr:
                if not ref_t or len(ref_t) < 2:
                    continue
                match_len = min(len(live_t), len(ref_t))
                if match_len < 2:
                    continue
                if live_t == ref_t:
                    ocr_match_count += 1
                    ocr_matched_texts.add(f"{live_t}={ref_t}")
                    break
                elif match_len >= 3 and (live_t in ref_t or ref_t in live_t):
                    ocr_match_count += 1
                    ocr_matched_texts.add(f"{live_t}≈{ref_t}")
                    break
        ocr_bonus = min(ocr_match_count * 0.25, 0.6)

        # Role-weighted: 標示牌 (signs) are more location-specific
        ref_signs = {_normalize_label(o.label) for o in ref_node.objects
                     if o.role == "標示牌"}
        sign_overlap = len(live_labels & ref_signs)
        sign_bonus = min(sign_overlap * 0.05, 0.15)

        total = j + ocr_bonus + sign_bonus
        parts = [f"jaccard={j:.2f}"]
        if ocr_matched_texts:
            parts.append(f"ocr_match={ocr_matched_texts}")
        if sign_bonus > 0:
            parts.append(f"sign_bonus={sign_bonus:.2f}")

        scores.append((nid, total, " | ".join(parts)))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


# ── Main localization entry point ─────────────────────────────────────────

# Minimum score to accept a match without VLM confirmation
CONFIDENT_THRESHOLD = 0.05

# If the gap between #1 and #2 is smaller than this, the match is ambiguous
AMBIGUITY_GAP = 0.10


def localize(
    detected_labels: List[str],
    ocr_texts: List[str],
    neo4j: Neo4jClient,
    *,
    place: Optional[str] = None,
    hint_nid: Optional[int] = None,
) -> LocalizationResult:
    """Match a user's live photo detections against the Neo4j reference map.

    Args:
        detected_labels: object labels from VLM/GroundingDINO on the user's photo
        ocr_texts: OCR text strings found in the user's photo
        neo4j: connected Neo4j client
        place: the place to match against (default: config.NEO4J_PLACE)
        hint_nid: if the user was at this node last, prefer nearby nodes
                  (breaks ties, doesn't override a clearly better distant match)

    Returns:
        LocalizationResult with the best matching reference node.
    """
    ref_map = neo4j.load_reference_map(place)
    if not ref_map.photos:
        return LocalizationResult(
            matched_nid=None, confidence=0.0, method="none",
            reasoning="No reference map loaded from Neo4j",
        )

    # Score all reference nodes
    candidates = match_by_objects(detected_labels, ocr_texts, ref_map, top_k=10)

    if not candidates:
        return LocalizationResult(
            matched_nid=None, confidence=0.0, method="none",
            reasoning="No candidates matched",
        )

    best_nid, best_score, best_reason = candidates[0]
    runner_nid = candidates[1][0] if len(candidates) > 1 else None
    runner_score = candidates[1][1] if len(candidates) > 1 else 0.0

    # Apply proximity hint: if the user was recently at hint_nid, boost
    # nearby nodes slightly (within 2 hops on the walkway graph)
    if hint_nid is not None and hint_nid in ref_map.photos:
        hint_neighbors = set(ref_map.photos[hint_nid].neighbor_nids)
        # 2-hop neighbors
        hint_2hop = set()
        for n in hint_neighbors:
            if n in ref_map.photos:
                hint_2hop.update(ref_map.photos[n].neighbor_nids)
        hint_2hop.update(hint_neighbors)
        hint_2hop.add(hint_nid)

        # Re-score with proximity bonus
        boosted = []
        for nid, score, reason in candidates:
            bonus = 0.0
            if nid in hint_neighbors:
                bonus = 0.08
            elif nid in hint_2hop:
                bonus = 0.04
            boosted.append((nid, score + bonus, reason + (f" | proximity+{bonus:.2f}" if bonus else "")))

        boosted.sort(key=lambda x: -x[1])
        best_nid, best_score, best_reason = boosted[0]
        runner_nid = boosted[1][0] if len(boosted) > 1 else None
        runner_score = boosted[1][1] if len(boosted) > 1 else 0.0

    # Confidence mapping: raw score → 0–1 confidence
    # A Jaccard+OCR score above 0.5 is very strong for store environments
    confidence = min(best_score / 0.6, 1.0)

    if best_score < CONFIDENT_THRESHOLD:
        return LocalizationResult(
            matched_nid=None, confidence=confidence, method="object_set",
            reasoning=f"Best score {best_score:.2f} below threshold ({best_reason})",
            runner_up_nid=runner_nid, runner_up_score=runner_score,
        )

    # ── Heading estimation via directional photos ────────────────────
    matched_heading: Optional[float] = None
    matched_slot: Optional[str] = None
    heading_conf = 0.0

    matched_ref = ref_map.photos.get(best_nid)
    if matched_ref and matched_ref.directional_photos:
        from server.heading import estimate_heading, best_matching_slot, DirectionalPhoto

        live_labels_lower = {l.lower().strip() for l in detected_labels if l}
        live_ocr_lower = {t.lower().strip() for t in ocr_texts if t}

        # Build DirectionalPhoto objects for the heading estimator
        dir_photos = [
            DirectionalPhoto(
                slot=dp.slot if hasattr(dp, 'slot') else dp.slot,
                heading_deg=dp.heading_deg,
                photo_file=dp.photo_file,
                objects=[{"label": o.label, "ocr_text": o.ocr_text}
                         for o in dp.objects],
            )
            for dp in matched_ref.directional_photos
        ]

        matched_heading = estimate_heading(live_labels_lower, live_ocr_lower, dir_photos)
        slot_result = best_matching_slot(live_labels_lower, live_ocr_lower, dir_photos)
        matched_slot = slot_result if isinstance(slot_result, str) else (slot_result.value if slot_result else None)

        # Heading confidence: if multiple directions have the same objects
        # (synthesized from single photo), confidence is low
        unique_obj_sets = len({
            frozenset(o.label for o in dp.objects)
            for dp in matched_ref.directional_photos
        })
        if unique_obj_sets > 1:
            heading_conf = min(confidence, 0.8)  # real directional data
        else:
            heading_conf = 0.2  # synthesized — all slots look the same

    return LocalizationResult(
        matched_nid=best_nid,
        confidence=confidence,
        method="object_set",
        reasoning=best_reason,
        ref_node=matched_ref,
        runner_up_nid=runner_nid,
        runner_up_score=runner_score,
        matched_heading=matched_heading,
        matched_slot=matched_slot,
        heading_confidence=heading_conf,
    )
