"""Sensor-Nav router — PDR-enhanced navigation with map persistence.

Mounted at /snav/. Reuses existing modules (TopoMap, Perception, VLM, OCR)
but maintains its own session store and adds PDR metric data to map edges.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from server import scene
from server.annotator import annotate
from server.config import (
    PERCEPTION_ENABLED, OCR_ENABLED, OCR_LANGUAGES,
    OCR_MIN_CONFIDENCE, OCR_MAX_RESULTS, ARRIVED_MIN_DETECTION_SCORE,
    GOAL_CROP_VERIFY, GENERIC_INDOOR_OBJECTS, ensure_output_dir,
)
from server.goal_decomposer import decompose_goal, split_multi_goals
from server.graph_renderer import (
    render_goal_graph, augment_goal_classes, postprocess_detections,
)
from server.config import normalize_label
from server.map_store import MapStore
from server.models import VLMAction, VLMResponse, SubGoalInfo
from server.neo4j_map_store import Neo4jMapStore
from server.ocr import OCR, OCRResult
from server.perception import Detection as PerceptionDetection, Perception, verify_goal_detection
from server.session import Session, SubGoal
from server.topomap import TopoMap
from server.topomap_v2 import TopoGraphV2, NTYPE_OBJECT, ETYPE_WALKWAY
from server.vlm import (
    decide as vlm_decide,
    perceive_and_decide as vlm_perceive_and_decide,
    perceive_only as vlm_perceive_only,
    navigate_with_perception as vlm_navigate,
    format_perception_detections as vlm_format_dets,
    ask_about_image as vlm_ask_about_image,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/snav", tags=["sensor-nav"])

# ── Stores (independent from main server) ──

_sessions: Dict[str, Session] = {}
_map_store = MapStore()
_neo4j_store: Optional[Neo4jMapStore] = None

_DEBUG_DIR = Path("output/sessions")


def _persist_session_debug(s: Session):
    """Save session debug data to disk so it survives server restarts."""
    d = _DEBUG_DIR / f"snav_{s.id}"
    d.mkdir(parents=True, exist_ok=True)
    data = {
        "session_id": s.id,
        "goal": s.goal,
        "goal_objects": s.goal_objects,
        "place_name": s.place_name,
        "nav_mode": s.nav_mode,
        "arrived": s.arrived,
        "pending_arrival": s.pending_arrival,
        "goal_photo_ids": s.goal_photo_ids[:5] if s.goal_photo_ids else [],
        "sub_goals": _sub_goals_info(s),
        "current_goal_idx": s.current_goal_idx,
        "total_goals": s.shopping_goal_count or 1,
        "turns": s.history,
        "created_at": s.created_at.isoformat(),
        "turn_count": len(s.history),
        "type": "snav",
    }
    (d / "debug.json").write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")


def _load_persisted_sessions() -> List[dict]:
    """Scan disk for persisted snav session debug files (including legacy dirs)."""
    results = []
    if not _DEBUG_DIR.exists():
        return results
    for d in _DEBUG_DIR.iterdir():
        if not d.is_dir() or not d.name.startswith("snav_"):
            continue
        sid = d.name[5:]
        if sid in _sessions:
            continue
        f = d / "debug.json"
        if f.exists():
            try:
                results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        else:
            # Legacy session without debug.json — build minimal entry from dir
            photo_dir = d / "photo"
            annotated_dir = d / "annotated"
            photo_count = len(list(photo_dir.glob("*.jpg"))) if photo_dir.exists() else 0
            if photo_count == 0 and not annotated_dir.exists():
                continue
            created = None
            try:
                created = os.path.getctime(str(d))
            except Exception:
                pass
            from datetime import datetime
            results.append({
                "session_id": sid,
                "goal": "(舊 session)",
                "place_name": "",
                "nav_mode": "",
                "arrived": False,
                "turn_count": photo_count,
                "created_at": datetime.utcfromtimestamp(created).isoformat() if created else "",
                "type": "snav",
                "_legacy": True,
            })
    return results


def _get_neo4j_store() -> Optional[Neo4jMapStore]:
    """Lazy-init Neo4j connection; returns None if env vars are missing."""
    global _neo4j_store
    if _neo4j_store is not None:
        return _neo4j_store
    import os
    if not os.getenv("NEO4J_URI"):
        return None
    try:
        _neo4j_store = Neo4jMapStore()
        _neo4j_store.ping()
        log.info("Neo4j map store connected")
        return _neo4j_store
    except Exception as e:
        log.warning("Neo4j map store unavailable: %s", e)
        return None


def _load_topo_v2(place_name: str) -> "Optional[TopoGraphV2]":
    """Load TopoGraphV2: prefer local JSON (TOPO_MAP_JSON), fall back to Neo4j."""
    from server.config import TOPO_MAP_JSON
    from server.topomap_v2 import TopoGraphV2
    if TOPO_MAP_JSON and os.path.isfile(TOPO_MAP_JSON):
        try:
            topo = TopoGraphV2.load(TOPO_MAP_JSON)
            log.info("[snav] Loaded TopoGraphV2 from JSON: %s", TOPO_MAP_JSON)
            return topo
        except Exception as e:
            log.warning("[snav] Failed to load JSON map %s: %s", TOPO_MAP_JSON, e)
    neo4j = _get_neo4j_store()
    if neo4j is not None:
        try:
            return neo4j.download(place_name)
        except Exception as e:
            log.warning("[snav] Failed to load from Neo4j: %s", e)
    return None


_TOPO_SEARCH_STOP_WORDS = {
    "shelf", "shelves", "rack", "display rack", "display", "aisle",
    "section", "area", "counter", "refrigerator", "fridge", "cooler",
    "freezer", "door", "sign", "cabinet", "window", "ceiling",
    "grocery aisle", "grocery section", "noodle section",
    "架子", "貨架", "走道", "區", "區域", "冰箱", "冷藏櫃", "冷凍櫃",
    "門", "標誌", "櫃子", "雜貨區", "麵食區", "乳製品區", "冷凍食品區",
}


def _search_goal_in_topo(topo: TopoGraphV2, goal: str, goal_objects: List[str]) -> List[int]:
    """Search TopoGraphV2 for photo nodes whose objects match the goal.

    Returns photo_ids sorted by match score (best first).
    Uses keyword position weighting (earlier = more relevant) and
    IDF penalty (labels appearing on many photos score lower).
    """
    import math

    goal_lower = goal.lower()

    # Filter out generic location/container terms that match everywhere
    filtered = [kw for kw in goal_objects if kw.lower() not in _TOPO_SEARCH_STOP_WORDS]
    if not filtered:
        filtered = goal_objects

    # Build keyword list with position-based weight:
    # first 1/3 of list = core (weight 1.0), rest = context (weight 0.3)
    core_cutoff = max(len(filtered) // 3, 3)
    kw_entries = []
    for i, kw in enumerate(filtered):
        w = 1.0 if i < core_cutoff else 0.3
        kw_entries.append((kw.lower(), normalize_label(kw), w))

    # Build IDF: count how many photos each label_norm appears on
    label_photo_count: Dict[str, int] = {}
    total_photos = 0
    for nid, data in topo.graph.nodes(data=True):
        if data.get("ntype") != NTYPE_OBJECT:
            continue
        ln = data.get("label_norm", "")
        pid = data.get("photo_id")
        if ln and pid is not None:
            key = f"{ln}::{pid}"
            if key not in label_photo_count:
                label_photo_count.setdefault(ln, 0)
                label_photo_count[ln] += 1
                label_photo_count[key] = 1
    total_photos = len(topo.all_photo_nodes())
    # Clean up compound keys
    label_photo_count = {k: v for k, v in label_photo_count.items() if "::" not in k}

    def _idf(label_norm: str) -> float:
        df = label_photo_count.get(label_norm, 0)
        if df == 0 or total_photos == 0:
            return 1.0
        ratio = df / total_photos
        if ratio > 0.15:
            return 0.2
        if ratio > 0.05:
            return 0.5
        return 1.0

    photo_scores: Dict[int, float] = {}
    for nid, data in topo.graph.nodes(data=True):
        if data.get("ntype") != NTYPE_OBJECT:
            continue
        label_norm = data.get("label_norm", "")
        label_raw = (data.get("label") or "").lower()
        ocr_text = (data.get("ocr_text") or "").lower()
        photo_id = data.get("photo_id")
        if photo_id is None:
            continue

        score = 0.0
        idf = _idf(label_norm)

        # Label matching against keywords
        best_label_score = 0.0
        for kw_low, kw_norm, kw_weight in kw_entries:
            if not kw_norm and not kw_low:
                continue
            s = 0.0
            if label_norm and kw_norm and label_norm == kw_norm:
                s = 3.0 * kw_weight * idf
            elif label_norm and kw_norm and (kw_norm in label_norm or label_norm in kw_norm):
                s = 1.5 * kw_weight * idf
            elif kw_low and label_raw and (kw_low in label_raw or label_raw in kw_low):
                s = 1.5 * kw_weight * idf
            if s > best_label_score:
                best_label_score = s
        score += best_label_score

        # OCR matching — lower weight than labels (VLM OCR can hallucinate)
        if ocr_text:
            best_ocr_score = 0.0
            if goal_lower in ocr_text:
                best_ocr_score = 1.5
            for kw_low, _, kw_weight in kw_entries:
                if kw_low and len(kw_low) >= 2 and kw_low in ocr_text:
                    s = 1.0 * kw_weight
                    if s > best_ocr_score:
                        best_ocr_score = s
            score += best_ocr_score

        if score > 0:
            photo_scores[photo_id] = photo_scores.get(photo_id, 0.0) + score

    ranked = sorted(photo_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return []
    # Only return the single best-scored photo as destination
    best_pid, best_score = ranked[0]
    log.info("[snav] goal search top match: photo=%d score=%.2f (total candidates=%d)",
             best_pid, best_score, len(ranked))
    return [best_pid]


# ── Default terminal destinations: checkout → exit ──
#
# Every shopping route ends by heading to the checkout and then the exit, so we
# auto-append these two goals after the user's items. The generic goal search is
# unreliable here (its IDF penalty crushes checkout/exit signage that appears on
# many photos), so we score photos directly: a "strong" term names the fixture
# you actually stand at (a real counter/kiosk) and outweighs "weak" directional
# signage that merely points toward it.
_TERMINAL_SPECS = [
    ("結帳",
     ("收銀台", "收銀機", "結帳機", "自助結帳", "收銀螢幕", "cash register",
      "checkout counter", "self-checkout", "cashier counter", "pos screen"),
     ("結帳", "收銀", "checkout", "cashier")),
    ("出口",
     ("出口", "exit"),
     ()),
]


def _score_terminal_photos(
    topo: TopoGraphV2, strong: tuple, weak: tuple,
) -> List[tuple]:
    """Rank photo nodes by how strongly they represent a terminal destination.

    Each object contributes 2.0 when it matches a strong (fixture) term, else
    1.0 for a weak (signage) term. Returns ``[(photo_id, score), ...]`` sorted
    best-first.
    """
    scores: Dict[int, float] = {}
    for _nid, d in topo.graph.nodes(data=True):
        if d.get("ntype") != NTYPE_OBJECT:
            continue
        pid = d.get("photo_id")
        if pid is None:
            continue
        blob = " ".join([
            d.get("label") or "", d.get("label_norm") or "", d.get("ocr_text") or "",
        ]).lower()
        if any(t in blob for t in strong):
            scores[pid] = scores.get(pid, 0.0) + 2.0
        elif weak and any(t in blob for t in weak):
            scores[pid] = scores.get(pid, 0.0) + 1.0
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _build_terminal_goals(
    topo: TopoGraphV2, sub_goals: List[SubGoal],
) -> List[SubGoal]:
    """Locate checkout → exit destinations to append after the shopping goals.

    Returns SubGoal objects (checkout first, then exit) for whichever terminals
    exist in the map, each pointing at a distinct photo node (and distinct from
    the shopping goals). Returns an empty list when the map has no recognizable
    checkout/exit.
    """
    used = {pid for sg in sub_goals for pid in sg.goal_photo_ids}
    terminals: List[SubGoal] = []
    for name, strong, weak in _TERMINAL_SPECS:
        ranked = _score_terminal_photos(topo, strong, weak)
        pid = next((p for p, _ in ranked if p not in used), None)
        if pid is None:
            log.info("[snav] terminal goal '%s' not found in map", name)
            continue
        used.add(pid)
        terminals.append(SubGoal(name=name, goal_objects=list(strong),
                                 goal_photo_ids=[pid], is_terminal=True))
        log.info("[snav] terminal goal '%s' -> photo=%d", name, pid)
    return terminals


def _tsp_order_shopping_goals(
    topo: TopoGraphV2, sub_goals: List[SubGoal], matched: List[int], start_pid: int,
) -> List[int]:
    """Return an open-path TSP visit order (indices into ``sub_goals``) for the
    located shopping goals in ``matched``, starting from ``start_pid``.

    Falls back to the input order when there is nothing to reorder or TSP is
    unavailable. ``start_pid`` is treated as already-visited by ``solve_tsp``, so
    passing a goal node as the origin would drop it; a safety net re-appends any
    located goal the solver left out.
    """
    order = list(matched)
    if len(matched) <= 1:
        return order
    try:
        from server.route_planner import solve_tsp
        valid_ids = [sub_goals[i].goal_photo_ids[0] for i in matched]
        pid_to_idx = {pid: i for pid, i in zip(valid_ids, matched)}
        tsp = solve_tsp(topo, start_pid, valid_ids)
        if tsp is not None:
            order = [pid_to_idx[valid_ids[k]] for k in tsp.order]
            order += [i for i in matched if i not in order]
            log.info("[snav] TSP order (from start %s): %s (distance: %.1f)",
                     start_pid, order, tsp.total_distance)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("[snav] TSP solve failed: %s", e)
    return order


def _setup_map_goals(
    topo: TopoGraphV2, s: Session, sub_goals: List[SubGoal],
) -> bool:
    """Search shopping goals, then append checkout → exit.

    Populates ``s.tsp_order``, ``s.current_goal_idx``, ``s.goal_objects`` and
    ``s.goal_photo_ids``. The auto-appended checkout/exit terminals are kept in
    fixed order at the end and excluded from TSP reordering. Returns True when at
    least one navigable goal (shopping or terminal) was located, i.e. the session
    should enter map mode.

    When more than one shopping goal is located, the visit order is left
    provisional (input order) and ``s.tsp_pending_reorder`` is set: the real
    ordering is solved from the user's actual localized position on the first
    successful localization (see :func:`_reorder_goals_from_start`), because a
    meaningful start point only exists after the user is localized.
    """
    for sg in sub_goals:
        sg.goal_photo_ids = _search_goal_in_topo(topo, sg.name, sg.goal_objects)

    # Order only the shopping goals that were actually located.
    matched = [i for i, sg in enumerate(sub_goals) if sg.goal_photo_ids]

    # Build the checkout → exit terminals so they stay pinned at the end.
    terminals = _build_terminal_goals(topo, sub_goals)

    # Defer TSP ordering until the user is localized — the tour start point is
    # wherever the user actually is, not a pre-assumed node. Keep a provisional
    # sequential order so the session is usable until the first localization.
    order = list(matched)
    s.tsp_pending_reorder = len(matched) > 1

    # Auto-append checkout → exit as the fixed final destinations.
    for tg in terminals:
        sub_goals.append(tg)
        order.append(len(sub_goals) - 1)

    s.tsp_order = order
    if not order:
        return False
    s.current_goal_idx = 0
    cur = s.current_sub_goal
    if cur:
        s.goal_objects = cur.goal_objects
        s.goal_photo_ids = cur.goal_photo_ids
    return True


def _reorder_goals_from_start(
    topo: TopoGraphV2, s: Session, start_pid: int,
) -> None:
    """Solve the shopping-goal visit order from the user's real localized
    position ``start_pid``, keeping checkout → exit terminals pinned at the end.

    Called once, right after the first successful localization (guarded by
    ``s.tsp_pending_reorder``). Rebuilds ``s.tsp_order`` and resets the cursor to
    the first goal so route planning targets the nearest optimal goal, then
    clears the pending flag.
    """
    if not s.sub_goals:
        s.tsp_pending_reorder = False
        return

    shopping = [i for i, sg in enumerate(s.sub_goals)
                if not sg.is_terminal and sg.goal_photo_ids and not sg.arrived]
    terminals = [i for i, sg in enumerate(s.sub_goals) if sg.is_terminal]

    shopping = _tsp_order_shopping_goals(topo, s.sub_goals, shopping, start_pid)

    s.tsp_order = shopping + terminals
    s.current_goal_idx = 0
    cur = s.current_sub_goal
    if cur:
        s.goal_objects = cur.goal_objects
        s.goal_photo_ids = cur.goal_photo_ids
        s.planned_path = []
        s.last_route_guidance = None
        s.last_route_waypoints = []
    s.tsp_pending_reorder = False
    log.info("[snav %s] reordered goals from localized start %s: tsp_order=%s",
             s.id, start_pid, s.tsp_order)


# ── Lazy singletons (same pattern as server.py) ──

_perception: Optional[Perception] = None
_perception_failed = False
_ocr: Optional[OCR] = None
_ocr_failed = False


def _get_perception() -> Optional[Perception]:
    global _perception, _perception_failed
    if not PERCEPTION_ENABLED or _perception_failed:
        return None
    if _perception is None:
        try:
            p = Perception()
            p.load()
            _perception = p
        except Exception as e:
            log.warning("Perception unavailable: %s", e)
            _perception_failed = True
            return None
    return _perception


def _get_ocr() -> Optional[OCR]:
    global _ocr, _ocr_failed
    if _ocr_failed or not OCR_ENABLED:
        return None
    if _ocr is None:
        try:
            o = OCR(languages=OCR_LANGUAGES)
            o.load()
            _ocr = o
        except Exception as e:
            log.warning("OCR unavailable: %s", e)
            _ocr_failed = True
            return None
    return _ocr


# ── Pydantic models ──

class PdrData(BaseModel):
    steps: int = 0
    distance_m: float = 0.0
    heading_deg: float = 0.0
    pdr_x: float = 0.0
    pdr_y: float = 0.0
    total_steps: int = 0
    total_distance_m: float = 0.0
    raw_sensors: Optional[Dict] = None


class SNavStartRequest(BaseModel):
    goal: str
    map_id: Optional[str] = None
    place_name: Optional[str] = None


class SNavStartResponse(BaseModel):
    session_id: str
    guidance: str
    action: str = "TAKE_PHOTO"
    goal_objects: List[str]
    nav_mode: str = "explore"  # "map" or "explore"
    goal_photo_ids: List[int] = []
    place_name: str = ""
    sub_goals: List[SubGoalInfo] = []
    tsp_order: List[int] = []
    current_goal_idx: int = 0
    current_goal_name: str = ""
    total_goals: int = 1


class SNavTurnResponse(BaseModel):
    action: VLMAction
    guidance: str
    question: Optional[str] = None
    node_id: int
    annotated_photo_url: Optional[str] = None
    step_guidance: Optional[str] = None
    estimated_remaining_steps: Optional[int] = None
    estimated_remaining_distance: Optional[float] = None
    # 定位檢索結果（有 topo_v2 地圖時）
    localized_photo_id: Optional[int] = None
    localization_score: Optional[float] = None
    localization_candidates: List[dict] = []
    goal_photo_ids: List[int] = []
    # 路徑規劃結果
    planned_route: List[int] = []
    route_target_photo_id: Optional[int] = None
    route_distance: Optional[float] = None
    route_hops: Optional[int] = None
    route_guidance: Optional[str] = None
    route_waypoints: List[dict] = []
    # 多目標進度
    sub_goals: List[SubGoalInfo] = []
    current_goal_idx: int = 0
    current_goal_name: str = ""
    total_goals: int = 1


class AnswerRequest(BaseModel):
    answer: str


class ConfirmLocationRequest(BaseModel):
    photo_id: int
    matched_direction: str = "front"


class ConfirmLocationResponse(BaseModel):
    guidance: str
    localized_photo_id: int
    goal_photo_ids: List[int] = []
    route_guidance: Optional[str] = None
    route_waypoints: List[dict] = []
    route_distance: Optional[float] = None
    route_hops: Optional[int] = None
    sub_goals: List[SubGoalInfo] = []
    current_goal_idx: int = 0
    current_goal_name: str = ""
    total_goals: int = 1


class ConfirmArrivalRequest(BaseModel):
    kind: str  # "confirmed" | "false_positive" | "wrong_instance"


class SaveMapRequest(BaseModel):
    session_id: str
    name: str


class SaveMapResponse(BaseModel):
    map_id: str
    name: str


# ── Helpers ──

def _image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.size


def _top_goal_detection(detections, goal_objects):
    matches = [d for d in detections if scene._label_matches_goal(d.label, goal_objects)]
    return max(matches, key=lambda d: d.score) if matches else None


def _is_goal_det(d: dict, goal_objects: list[str]) -> bool:
    label_l = d.get("label", "").lower()
    return any(g.lower() in label_l or label_l in g.lower() for g in goal_objects)


def _estimate_remaining(topomap: TopoMap, current_id: int, goal_id: Optional[int]):
    """Estimate steps/distance from current to goal via shortest path."""
    if goal_id is None:
        return None, None
    try:
        import networkx as nx
        path = nx.shortest_path(topomap.graph, current_id, goal_id)
        total_steps = sum(
            topomap.graph.edges[path[i], path[i + 1]].get("steps", 0) or 0
            for i in range(len(path) - 1)
        )
        total_dist = sum(
            topomap.graph.edges[path[i], path[i + 1]].get("distance_m", 0.0) or 0.0
            for i in range(len(path) - 1)
        )
        return total_steps if total_steps > 0 else None, total_dist if total_dist > 0 else None
    except Exception:
        return None, None


def _make_step_guidance(pdr: Optional[PdrData], remaining_steps: Optional[int]) -> Optional[str]:
    """Generate step guidance text."""
    parts = []
    if pdr and pdr.steps > 0:
        parts.append(f"已走 {pdr.steps} 步（{pdr.distance_m:.1f}m）")
    if remaining_steps and remaining_steps > 0:
        parts.append(f"預估還需 ~{remaining_steps} 步")
    return "　".join(parts) if parts else None


def _build_progress_string(s: Session) -> str:
    """Build a progress line for VLM prompts when multi-goal is active."""
    cur = s.current_sub_goal
    if cur is None:
        return ""
    # Checkout/exit are appended after the shopping list; report them separately
    # so they never inflate the "N goals" count.
    if cur.is_terminal:
        return f"購物清單已完成，前往：{cur.name}"
    total = s.shopping_goal_count
    if total <= 1:
        return ""
    done = [sg.name for sg in s.sub_goals if sg.arrived and not sg.is_terminal]
    parts = f"進度：{s.shopping_progress_index}/{total}，正在尋找：{cur.name}"
    if done:
        parts += f"（已找到：{'、'.join(done)}）"
    return parts


def _sub_goals_info(s: Session) -> List[SubGoalInfo]:
    return [SubGoalInfo(name=sg.name, arrived=sg.arrived, is_terminal=sg.is_terminal) for sg in s.sub_goals]


def _build_route_info_string(s: Session) -> str:
    if not s.last_route_guidance:
        return ""
    parts = [f"地圖路線參考：{s.last_route_guidance}"]
    if s.last_route_waypoints and len(s.last_route_waypoints) > 1:
        next_wp = s.last_route_waypoints[1]
        next_desc = next_wp.get("description") or ""
        # Compute relative direction to the next waypoint
        if s.topo_v2 is not None:
            cur_pid = s.last_route_waypoints[0].get("photo_id")
            next_pid = next_wp.get("photo_id")
            if cur_pid is not None and next_pid is not None:
                from server.route_planner import compute_relative_direction
                rel_dir = compute_relative_direction(
                    s.topo_v2, cur_pid, next_pid, s.last_matched_direction,
                )
                if rel_dir:
                    dir_line = f"方向指引：下一個路標在你的{rel_dir}"
                    if next_desc:
                        dir_line += f"（{next_desc}）"
                    parts.append(dir_line)
                elif next_desc:
                    parts.append(f"下一步你應該會看到：{next_desc}")
            elif next_desc:
                parts.append(f"下一步你應該會看到：{next_desc}")
        elif next_desc:
            parts.append(f"下一步你應該會看到：{next_desc}")
    if s.last_route_waypoints and len(s.last_route_waypoints) > 3:
        mid = s.last_route_waypoints[1:-1]
        wp_descs = [wp.get("description", "") for wp in mid if wp.get("description")]
        if wp_descs:
            parts.append(f"沿途路標：{'→'.join(wp_descs[:4])}")
    return "\n".join(parts)


def _get_session(session_id: str) -> Session:
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    return s


# ── Endpoints ──

@router.post("/session", response_model=SNavStartResponse)
def start_session(req: SNavStartRequest) -> SNavStartResponse:
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail={"error": "bad_request", "detail": "goal is empty"})

    # Split multi-goal text into individual items
    items = split_multi_goals(req.goal)

    from server.graph_renderer import GOAL_CLASS_MAP
    _generic_lower = {g.lower() for g in GENERIC_INDOOR_OBJECTS}

    def _decompose_item(item: str) -> List[str]:
        objs = decompose_goal(item)
        objs = augment_goal_classes(item, objs)
        _goal_mapped: set[str] = set()
        for key, info in GOAL_CLASS_MAP.items():
            if key in item:
                _goal_mapped.update(c.lower() for c in info["classes"])
        return [g for g in objs if g.lower() not in _generic_lower or g.lower() in _goal_mapped]

    # Build sub-goals with per-item decomposition
    sub_goals: List[SubGoal] = []
    all_goal_objects: List[str] = []
    for item in items:
        item_objects = _decompose_item(item)
        sub_goals.append(SubGoal(name=item, goal_objects=item_objects))
        all_goal_objects.extend(item_objects)
    all_goal_objects = list(dict.fromkeys(all_goal_objects))

    import uuid
    sid = uuid.uuid4().hex[:8]
    # Use first sub-goal's objects as initial goal_objects
    first_objects = sub_goals[0].goal_objects if sub_goals else all_goal_objects
    s = Session(id=sid, goal=req.goal, goal_objects=first_objects)
    s.sub_goals = sub_goals

    # Load existing map if specified (local JSON)
    if req.map_id:
        loaded_topo, meta = _map_store.load(req.map_id)
        if loaded_topo is not None:
            s.topomap = loaded_topo
            log.info("Loaded map %s (%s) with %d nodes",
                     req.map_id, meta.get("name", ""), loaded_topo.graph.number_of_nodes())

    # Load TopoGraphV2: prefer local JSON (TOPO_MAP_JSON), fall back to Neo4j
    nav_mode = "explore"
    goal_photo_ids: List[int] = []
    place_name = req.place_name or ""

    if place_name:
        topo_v2 = _load_topo_v2(place_name)
        if topo_v2 is not None:
            s.topo_v2 = topo_v2
            s.place_name = place_name
            photo_count = len(topo_v2.all_photo_nodes())
            obj_count = len(topo_v2.all_object_nodes())
            log.info("[snav] Loaded TopoGraphV2 '%s': %d photos, %d objects",
                     place_name, photo_count, obj_count)

            # Search shopping goals, TSP-order them, and append checkout → exit.
            if _setup_map_goals(topo_v2, s, sub_goals):
                nav_mode = "map"
                goal_photo_ids = s.goal_photo_ids
                log.info("[snav] Goal found in map, sub_goals: %s",
                         [(sg.name, sg.goal_photo_ids[:3]) for sg in sub_goals])
            else:
                log.info("[snav] Goal not found in map, falling back to explore mode")
        else:
            log.info("[snav] No map found for place '%s'", place_name)

    # If no TSP order set (explore mode), default sequential
    if not s.tsp_order:
        s.tsp_order = list(range(len(sub_goals)))

    s.nav_mode = nav_mode

    _sessions[sid] = s

    cur_name = s.current_sub_goal.name if s.current_sub_goal else req.goal
    log.info("[snav] New session %s | goal: %s | items: %d | objects: %s | mode: %s",
             sid, req.goal, len(sub_goals), s.goal_objects, nav_mode)

    out_dir = ensure_output_dir(f"snav_{sid}")
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    render_goal_graph(req.goal, all_goal_objects, str(graph_dir / "goal_graph.png"))

    shopping_total = s.shopping_goal_count
    if nav_mode == "map":
        if shopping_total > 1:
            guidance = f"已載入「{place_name}」地圖，共 {shopping_total} 個目標。先在原地拍一張照片定位你的位置，系統會依你的所在位置安排最順的購物順序。"
        else:
            guidance = f"已載入「{place_name}」地圖，找到目標位置。拍一張照片開始定位導航。"
    else:
        guidance = "拍一張照片開始導航，感測器已在背景追蹤步行。"

    return SNavStartResponse(
        session_id=sid,
        guidance=guidance,
        goal_objects=s.goal_objects,
        nav_mode=nav_mode,
        goal_photo_ids=goal_photo_ids[:5],
        place_name=place_name,
        sub_goals=_sub_goals_info(s),
        tsp_order=s.tsp_order,
        current_goal_idx=s.current_goal_idx,
        current_goal_name=cur_name,
        total_goals=shopping_total or 1,
    )


@router.post("/{session_id}/photo", response_model=SNavTurnResponse)
async def upload_photo(
    session_id: str,
    photo: UploadFile = File(...),
    pdr_data: Optional[str] = Form(None),
) -> SNavTurnResponse:
    s = _get_session(session_id)
    if s.arrived:
        raise HTTPException(status_code=409, detail={"error": "already_arrived", "detail": "session is complete"})
    if s.pending_arrival:
        raise HTTPException(status_code=409, detail={"error": "confirm_pending", "detail": "POST /confirm first"})
    if s.pending_question is not None:
        raise HTTPException(status_code=409, detail={"error": "answer_pending", "detail": "POST /answer first"})

    # Parse PDR data
    pdr: Optional[PdrData] = None
    if pdr_data:
        try:
            pdr = PdrData(**json.loads(pdr_data))
        except Exception as e:
            log.warning("Invalid pdr_data: %s", e)

    t_start = time.time()
    out_dir = ensure_output_dir(f"snav_{session_id}")
    photo_bytes = await photo.read()
    nid_for_path = s.topomap.graph.number_of_nodes()
    photo_path = out_dir / "photo" / f"{nid_for_path}.jpg"
    photo_path.write_bytes(photo_bytes)

    # Fix EXIF rotation and resize
    im = Image.open(photo_path)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    img_w, img_h = im.size
    if max(img_w, img_h) > 1600:
        im.thumbnail((1600, 1600))
        img_w, img_h = im.size
    im.save(photo_path, format="JPEG", quality=85)
    im.close()

    perception_engine = _get_perception()
    ocr_engine = _get_ocr()
    has_grounding = perception_engine is not None

    # EasyOCR
    easyocr_results = []
    if ocr_engine is not None:
        easyocr_results = ocr_engine.read(
            str(photo_path),
            min_confidence=OCR_MIN_CONFIDENCE,
            max_results=OCR_MAX_RESULTS,
        )

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Detection (mode-dependent)
    # ══════════════════════════════════════════════════════════════
    if has_grounding:
        # ── Mode A: GroundingDINO + EasyOCR ──
        combined_prompts = list(dict.fromkeys(s.goal_objects + GENERIC_INDOOR_OBJECTS))
        detections = perception_engine.detect(str(photo_path), combined_prompts)
        detected_labels = [d.label for d in detections]

        ocr_results = easyocr_results
        ocr_text_summary = scene.format_ocr(ocr_results, img_w, img_h) if ocr_results else "(no text detected)"
        ocr_matches = scene.match_ocr_to_goal(ocr_results, s.goal_objects)
        if ocr_matches:
            ocr_text_summary += "  SIGN MATCH: " + "; ".join(ocr_matches)

        ocr_texts = [r.text for r in ocr_results]
        ocr_with_conf = [{"text": r.text, "confidence": r.confidence} for r in ocr_results]
        nid = s.topomap.add_node(
            photo_path=str(photo_path), detected=detected_labels,
            summary="", ocr_texts=ocr_texts, ocr_with_conf=ocr_with_conf,
        )
        if s.last_node_id is not None:
            s.topomap.add_edge(s.last_node_id, nid, action=s.last_planned_action or "(unknown)")
            if pdr:
                edata = s.topomap.graph.edges[s.last_node_id, nid]
                edata["steps"] = pdr.steps
                edata["distance_m"] = round(pdr.distance_m, 2)
                edata["heading_deg"] = round(pdr.heading_deg, 1)

        if pdr:
            s.topomap.graph.nodes[nid]["pdr_x"] = pdr.pdr_x
            s.topomap.graph.nodes[nid]["pdr_y"] = pdr.pdr_y

        detections_summary = scene.format_detections(detections, img_w, img_h)
        scene_description = None

    else:
        # ── Mode B/C: VLM Stage 1 (perception only) ──
        nid = s.topomap.add_node(
            photo_path=str(photo_path), detected=[],
            summary="", ocr_texts=[], ocr_with_conf=[],
        )
        if s.last_node_id is not None:
            s.topomap.add_edge(s.last_node_id, nid, action=s.last_planned_action or "(unknown)")
            if pdr:
                edata = s.topomap.graph.edges[s.last_node_id, nid]
                edata["steps"] = pdr.steps
                edata["distance_m"] = round(pdr.distance_m, 2)
                edata["heading_deg"] = round(pdr.heading_deg, 1)

        if pdr:
            s.topomap.graph.nodes[nid]["pdr_x"] = pdr.pdr_x
            s.topomap.graph.nodes[nid]["pdr_y"] = pdr.pdr_y

        vlm_perception = vlm_perceive_only(str(photo_path), s.goal_objects, img_w, img_h)

        detections = [
            PerceptionDetection(label=d.label, box=d.bbox, score=d.score, position=d.position)
            for d in vlm_perception.detections
        ]
        detected_labels = [d.label for d in detections]

        if easyocr_results:
            ocr_results = easyocr_results
        else:
            ocr_results = [
                OCRResult(text=t.text, confidence=t.score, bbox=t.bbox)
                for t in vlm_perception.ocr_texts
            ]

        ocr_text_summary = scene.format_ocr(ocr_results, img_w, img_h) if ocr_results else "(no text detected)"
        ocr_matches = scene.match_ocr_to_goal(ocr_results, s.goal_objects)
        if ocr_matches:
            ocr_text_summary += "  SIGN MATCH: " + "; ".join(ocr_matches)

        ocr_texts = [r.text for r in ocr_results]
        ocr_with_conf = [{"text": r.text, "confidence": r.confidence} for r in ocr_results]
        s.topomap.graph.nodes[nid]["detected"] = detected_labels
        s.topomap.graph.nodes[nid]["ocr_texts"] = ocr_texts
        s.topomap.graph.nodes[nid]["ocr_with_conf"] = ocr_with_conf

        detections_summary = vlm_format_dets(vlm_perception.detections)
        scene_description = vlm_perception.scene_description

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Postprocess + Localize + Route Plan (common)
    # ══════════════════════════════════════════════════════════════
    topomap_summary = s.topomap.summarize_for_vlm(current_id=nid)
    if pdr and pdr.steps > 0:
        topomap_summary += (
            f"\n[PDR] 自上次拍照走了 {pdr.steps} 步（{pdr.distance_m:.1f}m），"
            f"航向 {pdr.heading_deg:.0f}°，累計 {pdr.total_steps} 步（{pdr.total_distance_m:.1f}m）"
        )

    det_dicts = [{"label": d.label, "score": d.score, "box": list(d.box),
                  **({"position": d.position} if d.position else {})}
                 for d in detections]
    ocr_bbox_list = [
        {"text": r.text, "confidence": r.confidence,
         "bbox": [list(p) for p in r.bbox]}
        for r in (ocr_results or [])
    ]
    det_dicts = postprocess_detections(
        det_dicts, img_w, img_h,
        ocr_with_bbox=ocr_bbox_list or None,
        prev_dets=s.last_detections or None,
        prev_w=s.last_img_w,
        prev_h=s.last_img_h,
        skip_nms=not has_grounding,
    )
    s.topomap.graph.nodes[nid]["det_dicts"] = det_dicts

    # Build raw_vlm_items for weighted grid matching (proportional 0-1 boxes)
    raw_vlm_items = []
    if img_w > 0 and img_h > 0:
        for d in det_dicts:
            box = d.get("box", [])
            if box and len(box) == 4:
                raw_vlm_items.append({
                    "box": [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h],
                    "label": d.get("label", ""),
                })

    localized_photo_id: Optional[int] = None
    localization_score: Optional[float] = None
    localization_candidates: list[dict] = []
    if s.topo_v2 is not None:
        try:
            from server.locate_v2 import localize_photo as _localize
            loc_result = _localize(
                s.topo_v2, det_dicts, img_w, img_h,
                ocr_items=ocr_bbox_list or None,
                top_k=5,
                query_direction="front",
                raw_vlm_items=raw_vlm_items or None,
            )
            if loc_result.best:
                localized_photo_id = loc_result.best.photo_id
                localization_score = round(loc_result.best.score, 4)
                # Keep "" when undetermined — see server.py note; don't assert a
                # facing (and opening turn) we don't actually know.
                s.last_matched_direction = loc_result.best.matched_direction or ""
                log.info("[snav %s] localized: photo_id=%d rank=%d score=%.3f matched=%d/%d region=%s dir=%s",
                         session_id, loc_result.best.photo_id,
                         loc_result.best.photo_rank,
                         loc_result.best.score, loc_result.best.matched_count,
                         loc_result.best.query_object_count,
                         loc_result.best.region,
                         s.last_matched_direction)
            for i, c in enumerate(loc_result.candidates[:3]):
                log.info("[snav %s]   #%d photo_id=%d rank=%d score=%.3f matched=%d",
                         session_id, i+1, c.photo_id, c.photo_rank, c.score, c.matched_count)
            localization_candidates = [
                {
                    "nid": c.photo_id,
                    "photo_id": c.photo_id,
                    "photo_rank": c.photo_rank,
                    "score": round(c.score, 4),
                    "matched_count": c.matched_count,
                    "region": c.region,
                    "matched_direction": c.matched_direction,
                    "photo_url": f"/snav/{session_id}/topo_photo/{c.photo_id}",
                }
                for c in loc_result.candidates[:5]
            ]
            if loc_result.warning:
                log.info("[snav %s] localization warning: %s", session_id, loc_result.warning)
        except Exception as e:
            log.warning("[snav %s] localization failed: %s", session_id, e)

    # First localization: now that we know where the user actually is, solve the
    # multi-goal visit order from that real start point (once).
    if (localized_photo_id is not None and s.tsp_pending_reorder
            and s.topo_v2 is not None):
        _reorder_goals_from_start(s.topo_v2, s, localized_photo_id)

    planned_route: list[int] = []
    route_target_photo_id: Optional[int] = None
    route_distance: Optional[float] = None
    route_hops: Optional[int] = None
    route_guidance: Optional[str] = None
    route_waypoints: list[dict] = []
    if localized_photo_id is not None and s.goal_photo_ids:
        try:
            from server.route_planner import (plan_route, format_route_guidance,
                                              build_route_waypoints, user_facing_heading)
            route = plan_route(s.topo_v2, localized_photo_id, s.goal_photo_ids)
            if route is not None:
                planned_route = route.path
                route_target_photo_id = route.target_photo_id
                route_distance = route.distance
                route_hops = route.hops
                s.planned_path = route.path
                src_rank = s.topo_v2._photo_sequence_rank(localized_photo_id)
                tgt_rank = s.topo_v2._photo_sequence_rank(route.target_photo_id)
                uh = user_facing_heading(s.topo_v2, localized_photo_id, s.last_matched_direction)
                route_guidance = format_route_guidance(s.topo_v2, route, src_rank, tgt_rank,
                                                       goal_objects=s.goal_objects,
                                                       user_heading=uh)
                route_waypoints = build_route_waypoints(s.topo_v2, route,
                                                        goal_objects=s.goal_objects)
                log.info("[snav %s] route: %s → %s, %d hops, %.1fm",
                         session_id, localized_photo_id, route.target_photo_id,
                         route.hops, route.distance)
                s.last_route_guidance = route_guidance
                s.last_route_waypoints = route_waypoints
            else:
                log.info("[snav %s] route: no path found", session_id)
                s.last_route_guidance = None
                s.last_route_waypoints = []
        except Exception as e:
            log.warning("[snav %s] route planning failed: %s", session_id, e)

    route_info = _build_route_info_string(s)

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Navigation Decision (mode-dependent)
    # ══════════════════════════════════════════════════════════════
    cur_goal_name = s.current_sub_goal.name if s.current_sub_goal else s.goal
    progress = _build_progress_string(s)

    if has_grounding:
        vlm_resp = vlm_decide(
            image_path=str(photo_path),
            goal=cur_goal_name,
            goal_objects=s.goal_objects,
            topomap_summary=topomap_summary,
            detections_summary=detections_summary,
            prior_question=None,
            prior_answer=None,
            ocr_summary=ocr_text_summary,
            progress=progress,
            route_info=route_info,
        )
    else:
        vlm_resp = vlm_navigate(
            image_path=str(photo_path),
            goal=cur_goal_name,
            goal_objects=s.goal_objects,
            topomap_summary=topomap_summary,
            scene_description=scene_description or "",
            detections_summary=detections_summary,
            ocr_summary=ocr_text_summary,
            progress=progress,
            route_info=route_info,
        )

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Verify arrival + annotate
    # ══════════════════════════════════════════════════════════════
    goal_verified = None
    if GOAL_CROP_VERIFY and vlm_resp.action == VLMAction.ARRIVED:
        cand = _top_goal_detection(detections, s.goal_objects)
        if cand is not None:
            try:
                with Image.open(photo_path) as im:
                    goal_verified = verify_goal_detection(
                        im.convert("RGB"), cand.box, s.goal, vlm_ask_about_image,
                    )
            except Exception as e:
                log.warning("goal crop verification errored: %s", e)

    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, detections, ocr_matches=ocr_matches, goal_objects=s.goal_objects,
        min_score=ARRIVED_MIN_DETECTION_SCORE, goal_verified=goal_verified,
    )
    # Reverse gate: catch missed arrivals when VLM is too conservative
    vlm_resp = scene.check_missed_arrival(
        vlm_resp, det_dicts, ocr_matches=ocr_matches, goal_objects=s.goal_objects,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action == VLMAction.ASK

    s.topomap.graph.nodes[nid]["summary"] = vlm_resp.vlm_summary

    annotated_path = out_dir / "annotated" / f"{nid}.jpg"
    annotate(str(photo_path), str(annotated_path), detections,
             banner_text=f"{vlm_resp.action.value}: {vlm_resp.guidance}",
             ocr_results=ocr_results, goal_objects=s.goal_objects,
             region_mode=not has_grounding)

    # Save raw sensor data
    if pdr and pdr.raw_sensors:
        sensor_dir = out_dir / "sensors"
        sensor_dir.mkdir(parents=True, exist_ok=True)
        sensor_json = {
            "node_id": nid,
            "pdr_summary": {
                "steps": pdr.steps, "distance_m": pdr.distance_m,
                "heading_deg": pdr.heading_deg,
                "pdr_x": pdr.pdr_x, "pdr_y": pdr.pdr_y,
            },
            **pdr.raw_sensors,
        }
        (sensor_dir / f"sensors_{nid}.json").write_text(
            json.dumps(sensor_json, ensure_ascii=False), encoding="utf-8")
        counts = pdr.raw_sensors.get("sample_counts", {})
        log.info("[snav %s] raw sensors node %d: accel=%d gyro=%d mag=%d rot=%d steps=%d",
                 session_id, nid,
                 counts.get("accel", 0), counts.get("gyro", 0),
                 counts.get("mag", 0), counts.get("rot_vec", 0),
                 counts.get("step_events", 0))

    # Save map snapshot
    map_dir = out_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    map_json = s.topomap.to_dict(current_node=nid, goal_node=s.goal_node)
    (map_dir / f"map_{nid}.json").write_text(
        json.dumps(map_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update session state
    s.last_detections = det_dicts
    s.last_img_w = img_w
    s.last_img_h = img_h
    s.last_ocr_summary = ocr_text_summary
    s.last_ocr_matches = ocr_matches
    s.last_photo_path = str(photo_path)
    s.last_node_id = nid

    s.history.append({
        "kind": "photo", "node_id": nid,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
        **({"pdr_steps": pdr.steps, "pdr_distance_m": pdr.distance_m} if pdr else {}),
        "timestamp": time.time(),
        "photo_url": f"/snav/{session_id}/raw_photo/{nid}",
        "annotated_photo_url": f"/snav/{session_id}/photo/{nid}.jpg",
        "detections": det_dicts,
        "ocr_results": [
            {"text": r.text, "confidence": r.confidence,
             "bbox": [list(p) for p in r.bbox]}
            for r in (ocr_results or [])
        ],
        "localization": {
            "photo_id": localized_photo_id,
            "score": localization_score,
            "candidates": localization_candidates,
        },
        "route": {
            "path": planned_route,
            "target_photo_id": route_target_photo_id,
            "distance": route_distance,
            "hops": route_hops,
            "guidance": route_guidance,
            "waypoints": route_waypoints,
        },
        "vlm_input": {
            "topomap_summary": topomap_summary,
            "detections_summary": detections_summary,
            "ocr_summary": ocr_text_summary,
            "scene_description": scene_description if not has_grounding else None,
            "progress": progress,
            "route_info": route_info,
        },
        "vlm_output": {
            "action": vlm_resp.action.value,
            "guidance": vlm_resp.guidance,
            "question": vlm_resp.question,
            "vlm_summary": vlm_resp.vlm_summary,
        },
    })

    if vlm_resp.action == VLMAction.ARRIVED:
        s.pending_arrival = True
        s.goal_node = nid
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
        s.pending_is_confirm = arrival_downgraded
    else:
        s.last_planned_action = vlm_resp.guidance

    # Step guidance
    remaining_steps, remaining_dist = _estimate_remaining(s.topomap, nid, s.goal_node)
    step_guidance = _make_step_guidance(pdr, remaining_steps)

    _persist_session_debug(s)

    t_total = time.time() - t_start
    log.info("[snav %s] photo #%d | %s | %.1fs | pdr: %s",
             session_id, nid, vlm_resp.action.value, t_total,
             f"{pdr.steps}steps/{pdr.distance_m:.1f}m" if pdr else "none")

    return SNavTurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        node_id=nid,
        annotated_photo_url=f"/snav/{session_id}/photo/{nid}.jpg",
        step_guidance=step_guidance,
        estimated_remaining_steps=remaining_steps,
        estimated_remaining_distance=round(remaining_dist, 1) if remaining_dist else None,
        localized_photo_id=localized_photo_id,
        localization_score=localization_score,
        localization_candidates=localization_candidates,
        goal_photo_ids=s.goal_photo_ids[:5] if s.goal_photo_ids else [],
        planned_route=planned_route,
        route_target_photo_id=route_target_photo_id,
        route_distance=route_distance,
        route_hops=route_hops,
        route_guidance=route_guidance,
        route_waypoints=route_waypoints,
        sub_goals=_sub_goals_info(s),
        current_goal_idx=s.current_goal_idx,
        current_goal_name=s.current_sub_goal.name if s.current_sub_goal else s.goal,
        total_goals=s.shopping_goal_count or 1,
    )


@router.post("/{session_id}/answer", response_model=SNavTurnResponse)
def post_answer(session_id: str, req: AnswerRequest) -> SNavTurnResponse:
    s = _get_session(session_id)
    if s.pending_question is None:
        raise HTTPException(status_code=409, detail={"error": "no_question_pending"})
    if s.last_photo_path is None or s.last_node_id is None:
        raise HTTPException(status_code=409, detail={"error": "no_prior_photo"})

    img_w, img_h = _image_size(s.last_photo_path)
    detections_summary = scene.format_detections(s.last_detections, img_w, img_h)
    topomap_summary = s.topomap.summarize_for_vlm(current_id=s.last_node_id)
    prior_question = s.pending_question

    cur_goal_name = s.current_sub_goal.name if s.current_sub_goal else s.goal
    progress = _build_progress_string(s)
    route_info = _build_route_info_string(s)
    vlm_resp = vlm_decide(
        image_path=s.last_photo_path,
        goal=cur_goal_name,
        goal_objects=s.goal_objects,
        topomap_summary=topomap_summary,
        detections_summary=detections_summary,
        prior_question=prior_question,
        prior_answer=req.answer,
        ocr_summary=getattr(s, "last_ocr_summary", None),
        progress=progress,
        route_info=route_info,
    )

    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, s.last_detections, ocr_matches=s.last_ocr_matches,
        goal_objects=s.goal_objects,
        min_score=ARRIVED_MIN_DETECTION_SCORE, prior_was_confirm=s.pending_is_confirm,
    )
    # Reverse gate: catch missed arrivals when VLM is too conservative
    vlm_resp = scene.check_missed_arrival(
        vlm_resp, s.last_detections, ocr_matches=s.last_ocr_matches,
        goal_objects=s.goal_objects,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action == VLMAction.ASK

    _unsure_keywords = ("不確定", "繼續", "不是", "沒有", "沒看到", "不對", "再找", "還沒")
    if vlm_resp.action == VLMAction.ARRIVED and any(k in req.answer for k in _unsure_keywords):
        vlm_resp = VLMResponse(
            action=VLMAction.MOVE,
            guidance="請繼續前進，拍另一張照片。",
            question=None,
            vlm_summary=vlm_resp.vlm_summary,
        )

    s.history.append({
        "kind": "answer", "user_answer": req.answer,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
    })
    _persist_session_debug(s)

    s.pending_question = None
    s.pending_is_confirm = False
    if vlm_resp.action == VLMAction.ARRIVED:
        s.pending_arrival = True
        s.goal_node = s.last_node_id
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
        s.pending_is_confirm = arrival_downgraded
    else:
        s.last_planned_action = vlm_resp.guidance

    return SNavTurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        node_id=s.last_node_id,
        annotated_photo_url=f"/snav/{session_id}/photo/{s.last_node_id}.jpg",
        sub_goals=_sub_goals_info(s),
        current_goal_idx=s.current_goal_idx,
        current_goal_name=s.current_sub_goal.name if s.current_sub_goal else s.goal,
        total_goals=s.shopping_goal_count or 1,
    )


def _plan_route_for_next_goal(
    s: Session, session_id: str,
    source_photo_id: Optional[int], goal_photo_ids: List[int],
):
    """Plan a route from the just-arrived location to the next goal.

    ``source_photo_id`` is the photo node of the goal the user just reached —
    a valid on-graph node, so we can navigate onward without forcing another
    photo to re-localize. Facing is unknown after picking up the item, so no
    opening turn is asserted (no ``user_heading``), matching ``confirm_location``.

    Returns ``(route_guidance, route_waypoints, route_distance, route_hops,
    planned_route, route_target_photo_id)``; all empty/None when no route.
    """
    if source_photo_id is None or not goal_photo_ids or s.topo_v2 is None:
        return None, [], None, None, [], None
    try:
        from server.route_planner import (
            plan_route, format_route_guidance, build_route_waypoints,
        )
        route = plan_route(s.topo_v2, source_photo_id, goal_photo_ids)
        if route is None:
            log.info("[snav %s] next-goal route: no path from photo_id=%s",
                     session_id, source_photo_id)
            s.planned_path = []
            s.last_route_guidance = None
            s.last_route_waypoints = []
            return None, [], None, None, [], None
        src_rank = s.topo_v2._photo_sequence_rank(source_photo_id)
        tgt_rank = s.topo_v2._photo_sequence_rank(route.target_photo_id)
        route_guidance = format_route_guidance(
            s.topo_v2, route, src_rank, tgt_rank, goal_objects=s.goal_objects)
        route_waypoints = build_route_waypoints(
            s.topo_v2, route, goal_objects=s.goal_objects)
        s.planned_path = route.path
        s.last_route_guidance = route_guidance
        s.last_route_waypoints = route_waypoints
        log.info("[snav %s] next-goal route: %s → %s, %d hops, %.1fm",
                 session_id, source_photo_id, route.target_photo_id,
                 route.hops, route.distance)
        return (route_guidance, route_waypoints, route.distance, route.hops,
                route.path, route.target_photo_id)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("[snav %s] next-goal route planning failed: %s", session_id, e)
        return None, [], None, None, [], None


@router.post("/{session_id}/confirm", response_model=SNavTurnResponse)
def confirm_arrival(session_id: str, req: ConfirmArrivalRequest) -> SNavTurnResponse:
    s = _get_session(session_id)
    if not s.pending_arrival:
        raise HTTPException(status_code=409, detail={"error": "no_pending_arrival"})

    arrived_node = s.goal_node
    s.pending_arrival = False

    # Route info to hand straight back to the next goal (see confirmed branch).
    route_guidance: Optional[str] = None
    route_waypoints: list[dict] = []
    route_distance: Optional[float] = None
    route_hops: Optional[int] = None
    planned_route: list[int] = []
    route_target_photo_id: Optional[int] = None
    localized_photo_id: Optional[int] = None

    if req.kind == "confirmed":
        cur = s.current_sub_goal
        if cur:
            cur.arrived = True
            cur.arrived_node = arrived_node

        if s.all_sub_goals_arrived or not s.sub_goals:
            s.arrived = True
            msg = f"已確認到達目標：{s.goal}"
            action = VLMAction.ARRIVED
        else:
            # The user is standing at the goal they just reached, so we already
            # know their position in the map — plan the route to the next goal
            # right away instead of forcing another photo to re-localize.
            arrived_photo_id = (cur.goal_photo_ids[0]
                                if cur and cur.goal_photo_ids else None)
            localized_photo_id = arrived_photo_id
            next_sg = s.advance_to_next_goal()
            if next_sg is not None:
                route_guidance, route_waypoints, route_distance, route_hops, \
                    planned_route, route_target_photo_id = _plan_route_for_next_goal(
                        s, session_id, arrived_photo_id, next_sg.goal_photo_ids)
                if route_guidance:
                    msg = (f"已找到「{cur.name}」！接下來前往「{next_sg.name}」，"
                           f"路線已規劃：{route_guidance}")
                else:
                    msg = f"已找到「{cur.name}」！接下來前往「{next_sg.name}」，請繼續拍照導航。"
                action = VLMAction.MOVE
            else:
                s.arrived = True
                msg = f"已確認到達所有目標：{s.goal}"
                action = VLMAction.ARRIVED
        s.history.append({"kind": "confirm", "node_id": arrived_node, "message": msg,
                          "route": {"path": planned_route,
                                    "target_photo_id": route_target_photo_id,
                                    "distance": route_distance, "hops": route_hops,
                                    "guidance": route_guidance,
                                    "waypoints": route_waypoints}})
    else:
        s.goal_node = None
        if req.kind == "false_positive":
            s.false_positive_nodes.append(arrived_node)
            cur_name = s.current_sub_goal.name if s.current_sub_goal else s.goal
            msg = f"步驟 {arrived_node} 是誤判，繼續搜索{cur_name}"
        else:
            msg = f"步驟 {arrived_node} 不是要找的，繼續搜索"
        s.corrections.append(msg)
        s.history.append({"kind": "reject", "node_id": arrived_node, "reject_type": req.kind, "message": msg})
        action = VLMAction.MOVE

    _persist_session_debug(s)
    return SNavTurnResponse(
        action=action,
        guidance=msg,
        question=None,
        node_id=arrived_node,
        annotated_photo_url=f"/snav/{session_id}/photo/{arrived_node}.jpg",
        localized_photo_id=localized_photo_id,
        goal_photo_ids=s.goal_photo_ids[:5] if s.goal_photo_ids else [],
        planned_route=planned_route,
        route_target_photo_id=route_target_photo_id,
        route_distance=route_distance,
        route_hops=route_hops,
        route_guidance=route_guidance,
        route_waypoints=route_waypoints,
        sub_goals=_sub_goals_info(s),
        current_goal_idx=s.current_goal_idx,
        current_goal_name=s.current_sub_goal.name if s.current_sub_goal else s.goal,
        total_goals=s.shopping_goal_count or 1,
    )


# ── Map persistence endpoints ──

@router.post("/maps/save", response_model=SaveMapResponse)
def save_map(req: SaveMapRequest) -> SaveMapResponse:
    s = _sessions.get(req.session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found"})

    map_id = _map_store.save(
        topomap=s.topomap,
        name=req.name,
        metadata={"goal": s.goal},
    )
    log.info("[snav] Saved map %s (%s) from session %s", map_id, req.name, req.session_id)
    return SaveMapResponse(map_id=map_id, name=req.name)


@router.get("/maps")
def list_maps():
    return _map_store.list_maps()


@router.get("/maps/{map_id}")
def get_map(map_id: str):
    topo, meta = _map_store.load(map_id)
    if topo is None:
        raise HTTPException(status_code=404, detail={"error": "map_not_found"})
    return {
        **meta,
        "map": topo.to_dict(current_node=None, goal_node=None),
    }


@router.delete("/maps/{map_id}")
def delete_map(map_id: str):
    if not _map_store.delete(map_id):
        raise HTTPException(status_code=404, detail={"error": "map_not_found"})
    return {"status": "deleted", "map_id": map_id}


# ── Neo4j map endpoints ──

@router.get("/neo4j/places")
def list_neo4j_places():
    """List all places stored in Neo4j."""
    neo4j = _get_neo4j_store()
    if neo4j is None:
        raise HTTPException(status_code=503, detail={"error": "neo4j_unavailable"})
    return neo4j.list_places()


@router.get("/neo4j/places/{place_name}/preview")
def preview_neo4j_place(place_name: str, goal: Optional[str] = Query(default=None)):
    """Preview a Neo4j map: node counts and optional goal search."""
    neo4j = _get_neo4j_store()
    if neo4j is None:
        raise HTTPException(status_code=503, detail={"error": "neo4j_unavailable"})
    topo = neo4j.download(place_name)
    if topo is None:
        raise HTTPException(status_code=404, detail={"error": "place_not_found"})

    photo_nodes = topo.all_photo_nodes()
    object_nodes = topo.all_object_nodes()

    result = {
        "place_name": place_name,
        "photo_count": len(photo_nodes),
        "object_count": len(object_nodes),
    }

    if goal:
        goal_objects = decompose_goal(goal)
        goal_objects = augment_goal_classes(goal, goal_objects)
        goal_photo_ids = _search_goal_in_topo(topo, goal, goal_objects)
        result["goal"] = goal
        result["goal_objects"] = goal_objects
        result["goal_found"] = len(goal_photo_ids) > 0
        result["goal_photo_ids"] = goal_photo_ids[:5]

    return result


# ── Confirm location (user picks from candidates) ──

@router.post("/{session_id}/confirm_location", response_model=ConfirmLocationResponse)
async def confirm_location(session_id: str, req: ConfirmLocationRequest) -> ConfirmLocationResponse:
    s = _get_session(session_id)
    if s.topo_v2 is None:
        raise HTTPException(status_code=400, detail={"error": "no_map", "detail": "no topo_v2 map loaded"})
    if req.photo_id not in s.topo_v2.graph.nodes:
        raise HTTPException(status_code=404, detail={"error": "photo_not_found", "detail": f"photo_id {req.photo_id} not in map"})

    localized_photo_id = req.photo_id
    s.last_matched_direction = req.matched_direction or ""
    cur_goal = s.current_sub_goal
    cur_name = cur_goal.name if cur_goal else s.goal
    goal_photo_ids = s.goal_photo_ids

    route_guidance: Optional[str] = None
    route_waypoints: list[dict] = []
    route_distance: Optional[float] = None
    route_hops: Optional[int] = None

    if goal_photo_ids:
        try:
            from server.route_planner import plan_route, format_route_guidance, build_route_waypoints
            route = plan_route(s.topo_v2, localized_photo_id, goal_photo_ids)
            if route is not None:
                s.planned_path = route.path
                route_distance = route.distance
                route_hops = route.hops
                src_rank = s.topo_v2._photo_sequence_rank(localized_photo_id)
                tgt_rank = s.topo_v2._photo_sequence_rank(route.target_photo_id)
                route_guidance = format_route_guidance(s.topo_v2, route, src_rank, tgt_rank,
                                                       goal_objects=s.goal_objects)
                route_waypoints = build_route_waypoints(s.topo_v2, route,
                                                        goal_objects=s.goal_objects)
                s.last_route_guidance = route_guidance
                s.last_route_waypoints = route_waypoints
                log.info("[snav %s] confirm_location: photo_id=%d → route to %d, %d hops, %.1fm",
                         session_id, localized_photo_id, route.target_photo_id, route.hops, route.distance)
            else:
                log.info("[snav %s] confirm_location: no route found from photo_id=%d", session_id, localized_photo_id)
        except Exception as e:
            log.warning("[snav %s] confirm_location route planning failed: %s", session_id, e)

    # Run VLM with the last photo + route info to give immediate navigation guidance
    guidance = f"已定位，請拍照開始導航前往「{cur_name}」。"
    vlm_out = None
    if s.last_photo_path and s.last_node_id is not None:
        try:
            route_info = _build_route_info_string(s)
            topomap_summary = s.topomap.summarize_for_vlm(current_id=s.last_node_id)
            img_w, img_h = _image_size(s.last_photo_path)
            detections_summary = scene.format_detections(s.last_detections, img_w, img_h)
            progress = _build_progress_string(s)

            vlm_resp = vlm_navigate(
                image_path=s.last_photo_path,
                goal=cur_name,
                goal_objects=s.goal_objects,
                topomap_summary=topomap_summary,
                scene_description="",
                detections_summary=detections_summary,
                ocr_summary=s.last_ocr_summary or "",
                progress=progress,
                route_info=route_info,
            )
            if vlm_resp.guidance:
                guidance = vlm_resp.guidance
                log.info("[snav %s] confirm_location VLM guidance: %s", session_id, guidance)
            vlm_out = {
                "action": vlm_resp.action.value,
                "guidance": vlm_resp.guidance,
                "question": vlm_resp.question,
                "vlm_summary": vlm_resp.vlm_summary,
            }
        except Exception as e:
            log.warning("[snav %s] confirm_location VLM call failed: %s", session_id, e)

    # Record this confirmation as a turn so it appears in the debug dashboard
    s.history.append({
        "kind": "confirm_location",
        "node_id": localized_photo_id,
        "vlm_guidance": guidance,
        "timestamp": time.time(),
        "localization": {
            "photo_id": localized_photo_id,
            "score": None,
            "candidates": [],
        },
        "route": {
            "path": s.planned_path,
            "target_photo_id": s.planned_path[-1] if s.planned_path else None,
            "distance": route_distance,
            "hops": route_hops,
            "guidance": route_guidance,
            "waypoints": route_waypoints,
        },
        **({"vlm_output": vlm_out} if vlm_out else {}),
    })
    _persist_session_debug(s)

    return ConfirmLocationResponse(
        guidance=guidance,
        localized_photo_id=localized_photo_id,
        goal_photo_ids=goal_photo_ids[:5],
        route_guidance=route_guidance,
        route_waypoints=route_waypoints,
        route_distance=route_distance,
        route_hops=route_hops,
        sub_goals=_sub_goals_info(s),
        current_goal_idx=s.current_goal_idx,
        current_goal_name=cur_name,
        total_goals=s.shopping_goal_count or 1,
    )


# ── Debug endpoints ──

@router.get("/sessions/list")
def list_sessions():
    """List all sensor-nav sessions (active + persisted on disk)."""
    result = []
    for sid, s in _sessions.items():
        result.append({
            "session_id": sid,
            "goal": s.goal,
            "place_name": s.place_name,
            "nav_mode": s.nav_mode,
            "arrived": s.arrived,
            "turn_count": len(s.history),
            "created_at": s.created_at.isoformat(),
        })
    for d in _load_persisted_sessions():
        result.append({
            "session_id": d["session_id"],
            "goal": d.get("goal", ""),
            "place_name": d.get("place_name", ""),
            "nav_mode": d.get("nav_mode", ""),
            "arrived": d.get("arrived", False),
            "turn_count": d.get("turn_count", 0),
            "created_at": d.get("created_at", ""),
        })
    return sorted(result, key=lambda x: x["created_at"], reverse=True)

@router.get("/{session_id}/debug")
def get_debug_info(session_id: str):
    """Return full debug info for a navigation session (memory or disk)."""
    s = _sessions.get(session_id)
    if s is not None:
        return {
            "session_id": s.id,
            "goal": s.goal,
            "goal_objects": s.goal_objects,
            "place_name": s.place_name,
            "nav_mode": s.nav_mode,
            "arrived": s.arrived,
            "pending_arrival": s.pending_arrival,
            "goal_photo_ids": s.goal_photo_ids[:5] if s.goal_photo_ids else [],
            "sub_goals": _sub_goals_info(s),
            "current_goal_idx": s.current_goal_idx,
            "total_goals": s.shopping_goal_count or 1,
            "turns": s.history,
        }
    session_dir = _DEBUG_DIR / f"snav_{session_id}"
    f = session_dir / "debug.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    # Legacy session: reconstruct turns from photo/annotated/sensor directories
    if session_dir.exists():
        photo_dir = session_dir / "photo"
        annotated_dir = session_dir / "annotated"
        sensors_dir = session_dir / "sensors"
        turns = []
        if photo_dir.exists():
            for img in sorted(photo_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
                if not img.stem.isdigit():
                    continue
                nid = int(img.stem)
                turn = {
                    "kind": "photo",
                    "node_id": nid,
                    "photo_url": f"/snav/{session_id}/raw_photo/{nid}",
                }
                if (annotated_dir / f"{nid}.jpg").exists():
                    turn["annotated_photo_url"] = f"/snav/{session_id}/photo/{nid}.jpg"
                det_file = annotated_dir / f"detections_{nid}.json"
                if det_file.exists():
                    try:
                        det_data = json.loads(det_file.read_text(encoding="utf-8"))
                        turn["detections"] = det_data.get("detections", [])
                        turn["ocr_results"] = det_data.get("ocr", [])
                    except Exception:
                        pass
                sensor_file = sensors_dir / f"sensors_{nid}.json"
                if sensor_file.exists():
                    try:
                        sd = json.loads(sensor_file.read_text(encoding="utf-8"))
                        pdr = sd.get("pdr_summary") or {}
                        if "steps" in pdr:
                            turn["pdr_steps"] = pdr.get("steps")
                        if "distance_m" in pdr:
                            turn["pdr_distance_m"] = pdr.get("distance_m")
                    except Exception:
                        pass
                turns.append(turn)
        return {
            "session_id": session_id,
            "goal": "(舊 session)",
            "goal_objects": [],
            "place_name": "",
            "nav_mode": "",
            "arrived": False,
            "pending_arrival": False,
            "goal_photo_ids": [],
            "sub_goals": [],
            "current_goal_idx": 0,
            "total_goals": 1,
            "turns": turns,
        }
    raise HTTPException(status_code=404, detail={"error": "session_not_found"})


@router.get("/{session_id}/raw_photo/{node_id}")
def serve_raw_photo(session_id: str, node_id: int):
    """Serve the original (unannoted) photo for a node."""
    out_dir = Path("output/sessions") / f"snav_{session_id}" / "photo"
    fpath = out_dir / f"{node_id}.jpg"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="photo not found")
    return Response(content=fpath.read_bytes(), media_type="image/jpeg")


# ── Topo map photo serving ──

@router.get("/{session_id}/topo_photo/{photo_id}")
def serve_topo_photo(session_id: str, photo_id: int):
    """Serve a photo from the loaded topo_v2 map (or fallback to global map)."""
    from server.config import resolve_topo_photo, TOPO_MAP_JSON
    topo = None
    s = _sessions.get(session_id)
    if s is not None and s.topo_v2 is not None:
        topo = s.topo_v2
    if topo is None and TOPO_MAP_JSON and os.path.isfile(TOPO_MAP_JSON):
        topo = TopoGraphV2.load(TOPO_MAP_JSON)
    if topo is None:
        raise HTTPException(status_code=404, detail={"error": "no_map"})
    if photo_id not in topo.graph.nodes:
        raise HTTPException(status_code=404, detail={"error": "photo_not_found"})
    node = topo.graph.nodes[photo_id]
    photo_path = node.get("photo_path", "")
    resolved = resolve_topo_photo(photo_path, photo_id=photo_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail={"error": "file_not_found"})
    return Response(content=resolved.read_bytes(), media_type="image/jpeg")


# ── Static files (annotated photos) ──

@router.get("/{session_id}/photo/{filename}")
def get_annotated_photo(session_id: str, filename: str):
    out_dir = Path("output/sessions") / f"snav_{session_id}" / "annotated"
    fpath = out_dir / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="photo not found")
    return Response(content=fpath.read_bytes(), media_type="image/jpeg")


@router.get("/{session_id}/map")
def get_session_map(session_id: str, format: str = Query(default="json")):
    s = _get_session(session_id)
    if format == "png":
        png = s.topomap.render_png(current_id=s.last_node_id)
        return Response(content=png, media_type="image/png")
    return s.topomap.to_dict(current_node=s.last_node_id, goal_node=s.goal_node)


# ══════════════════════════════════════════════════════════════════════════
# Test-only endpoints (route planning without localization)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/test/route")
def test_route_plan(
    place_name: str = Query(...),
    goal: str = Query(...),
    source_rank: int = Query(..., description="起點照片 rank (1-based)"),
):
    """直接指定起點 rank 跑路徑規劃，不需要 session 或照片上傳。"""
    neo4j = _get_neo4j_store()
    if neo4j is None:
        raise HTTPException(status_code=503, detail="neo4j_unavailable")
    topo = neo4j.download(place_name)
    if topo is None:
        raise HTTPException(status_code=404, detail="place_not_found")

    goal_objects = decompose_goal(goal)
    goal_objects = augment_goal_classes(goal, goal_objects)
    goal_photo_ids = _search_goal_in_topo(topo, goal, goal_objects)

    if not goal_photo_ids:
        return {"error": "goal_not_found", "goal_objects": goal_objects}

    photo_nodes = sorted(topo.all_photo_nodes())
    if source_rank < 1 or source_rank > len(photo_nodes):
        raise HTTPException(status_code=400, detail=f"rank must be 1-{len(photo_nodes)}")
    source_photo_id = photo_nodes[source_rank - 1]

    from server.route_planner import plan_route, build_route_waypoints, format_route_guidance
    route = plan_route(topo, source_photo_id, goal_photo_ids)
    if route is None:
        return {"error": "no_path", "source": source_photo_id, "goals": goal_photo_ids}

    waypoints = build_route_waypoints(topo, route, goal_objects=goal_objects)
    guidance = format_route_guidance(topo, route, source_rank,
                                     topo._photo_sequence_rank(route.target_photo_id),
                                     goal_objects=goal_objects)

    return {
        "source_photo_id": source_photo_id,
        "source_rank": source_rank,
        "goal_photo_ids": goal_photo_ids[:5],
        "goal_objects": goal_objects,
        "planned_route": route.path,
        "target_photo_id": route.target_photo_id,
        "target_rank": topo._photo_sequence_rank(route.target_photo_id),
        "distance": route.distance,
        "hops": route.hops,
        "guidance": guidance,
        "waypoints": waypoints,
    }


@router.get("/test/photos")
def test_list_photos(place_name: str = Query(...)):
    """列出地圖裡所有照片節點的 rank、region、objects。"""
    neo4j = _get_neo4j_store()
    if neo4j is None:
        raise HTTPException(status_code=503, detail="neo4j_unavailable")
    topo = neo4j.download(place_name)
    if topo is None:
        raise HTTPException(status_code=404, detail="place_not_found")

    photo_nodes = sorted(topo.all_photo_nodes())
    result = []
    for rank, pid in enumerate(photo_nodes, 1):
        data = topo.graph.nodes[pid]
        objs = [
            topo.graph.nodes[nid].get("label", "")
            for nid in topo.graph.successors(pid)
            if topo.graph.nodes[nid].get("ntype") == NTYPE_OBJECT
        ]
        result.append({
            "photo_id": pid,
            "rank": rank,
            "region": data.get("region", ""),
            "objects": objs,
        })
    return result


@router.get("/test/topo_photo/{place_name}/{photo_id}")
def test_serve_topo_photo(place_name: str, photo_id: int):
    """不需要 session，直接用 place_name 讀照片。"""
    neo4j = _get_neo4j_store()
    if neo4j is None:
        raise HTTPException(status_code=503, detail="neo4j_unavailable")
    topo = neo4j.download(place_name)
    if topo is None:
        raise HTTPException(status_code=404, detail="place_not_found")
    if photo_id not in topo.graph.nodes:
        raise HTTPException(status_code=404, detail="photo_not_found")
    node = topo.graph.nodes[photo_id]
    photo_path = node.get("photo_path", "")
    from server.config import resolve_topo_photo
    resolved = resolve_topo_photo(photo_path, photo_id=photo_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="file_not_found")
    return Response(content=resolved.read_bytes(), media_type="image/jpeg")
