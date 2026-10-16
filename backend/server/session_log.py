"""Per-session records on disk, kept as a log (sessions themselves stay in memory).

Under ``output/sessions/<session_id>/``:

- ``turns.jsonl`` — one JSON line per request that moves the navigation along
  (``/photo``, ``/answer``, ``/confirm``): what the phone sent (PDR position, map position,
  distance since the last fix, heading, …), what localization did (every node's image
  evidence, σ, walkable nodes, the fused top candidates, the chosen node and its
  confidence) and what the app was told. Enough to re-run ``loc_fusion.fuse_v2`` offline
  with other parameters, without going back to the store.
- ``session.json`` — the session's latest state (goal, place, phase, targets, last fix),
  rewritten after every turn.

Logging must never break navigation: every write is wrapped and only logs a warning.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


def _clean(v: Any) -> Any:
    """JSON-safe copy: sets → sorted lists, dict keys → str, NaN/inf → None, tuples → lists."""
    if isinstance(v, float):
        return round(v, 4) if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (set, frozenset)):
        return [_clean(x) for x in sorted(v, key=str)]
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "value") and not isinstance(v, (str, int, bool)):   # enums
        return v.value
    return v


def localization_record(loc_result, *, sigma: Optional[float] = None, reachable=None,
                        server_evidence=None, aisle_evidence=None, rerank: Optional[dict] = None) -> dict:
    """What localization did for one photo (None-safe)."""
    rec: dict = {
        "sensor_sigma_m": sigma,
        "reachable_nodes": reachable,
        "image_evidence_source": "server_dino" if server_evidence else ("phone" if loc_result is not None
                                                                         and loc_result.image_evidence else None),
        "aisle_evidence": aisle_evidence,
        "rerank": rerank,
    }
    if loc_result is None:
        rec["result"] = None
        return _clean(rec)
    rec["result"] = {
        "matched_nid": loc_result.matched_nid,
        "confidence": loc_result.confidence,
        "method": loc_result.method,
        "estimated_xy": loc_result.estimated_xy,
        "top_candidates": [[c[0], c[1]] for c in (loc_result.top_candidates or [])],
        "heading_deg": loc_result.matched_heading,
        "heading_slot": loc_result.matched_slot,
        "heading_confidence": loc_result.heading_confidence,
        "reasoning": loc_result.reasoning,
    }
    rec["image_evidence"] = loc_result.image_evidence
    return _clean(rec)


def append_turn(out_dir: Path, record: dict) -> None:
    """Append one line to ``turns.jsonl`` (timestamped)."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = {"t": time.time(), "time": datetime.now().isoformat(timespec="seconds"), **record}
        with open(out_dir / "turns.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(_clean(rec), ensure_ascii=False) + "\n")
    except Exception as e:   # never let logging take a navigation turn down
        log.warning("session log: could not append turn in %s: %s", out_dir, e)


def write_session(out_dir: Path, s) -> None:
    """Rewrite ``session.json`` with the session's current state."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "session_id": s.id,
            "goal": s.goal,
            "goal_objects": s.goal_objects,
            "target_objects": getattr(s, "target_objects", None),
            "place": s.place,
            "phase": s.phase,
            "created_at": s.created_at,
            "arrived": s.arrived,
            "target_nodes": s.target_nodes,
            "target_node_to_item": getattr(s, "target_node_to_item", None),
            "visited_targets": s.visited_targets,
            "current_leg_index": getattr(s, "current_leg_index", None),
            "route_visit_order": getattr(s.route_plan, "visit_order", None) if s.route_plan else None,
            "last_corrected_nid": s.last_corrected_nid,
            "user_heading": s.user_heading,
            "heading_confidence": s.heading_confidence,
            "photos": getattr(s.topomap.graph, "number_of_nodes", lambda: None)(),
        }
        tmp = out_dir / "session.json.tmp"
        tmp.write_text(json.dumps(_clean(state), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_dir / "session.json")
    except Exception as e:
        log.warning("session log: could not write session.json in %s: %s", out_dir, e)


def response_record(resp) -> dict:
    """The parts of a TurnResponse worth keeping."""
    keys = ("action", "guidance", "question", "node_id", "retake_hint", "second_photo_hint",
            "corrected_node_id", "corrected_confidence", "corrected_location", "estimated_x", "estimated_y",
            "heading_deg", "heading_confidence", "next_instruction", "phase", "current_target_name",
            "visited_item_names")
    return {k: getattr(resp, k, None) for k in keys}
