"""FastAPI app — endpoints for in-store navigation sessions."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, JSONResponse, Response
from PIL import Image, ImageOps

from server import scene
from server.annotator import annotate
from server.graph_renderer import (
    render_goal_graph, render_scene_graph, assign_detection_ids,
    postprocess_detections, augment_goal_classes,
)
from server import editor_map as editor_map_mod
from server.config import (
    EDITOR_MAP_DIR,
    PERCEPTION_ENABLED, OCR_ENABLED, OCR_LANGUAGES,
    OCR_MIN_CONFIDENCE, OCR_MAX_RESULTS, ARRIVED_MIN_DETECTION_SCORE,
    GOAL_CROP_VERIFY, GENERIC_INDOOR_OBJECTS, OCCLUDER_PROMPTS, ensure_output_dir, OUTPUT_ROOT,
    VLM_ROUNDS, VLM_PASS3, GRID_N,
)
import server.config as _config
from server.sensor_test_render import render_sensor_test_png
from server.goal_decomposer import decompose_goal, split_goal_objects
from server.ocr import OCR
from server.models import (
    AnswerRequest,
    ErrorResponse,
    ConfirmArrivalRequest,
    PathResponse,
    StartSessionRequest,
    StartSessionResponse,
    TurnResponse,
    VLMAction,
    VLMResponse,
)
from server.perception import Detection as PerceptionDetection, Perception, verify_goal_detection
from server.ocr import OCRResult
from server.session import SessionStore
import re as _re


# Debug images (the map PNG and the scene graph) are drawn after the reply instead of
# before it — the scene graph alone grew to ~9 s per photo during a walk. One at a time:
# matplotlib is not thread-safe.
_RENDER_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")


def _render_later(fn, *args) -> None:
    def run():
        try:
            fn(*args)
        except Exception as e:
            log.warning("background render failed: %s", e)
    _RENDER_POOL.submit(run)


def _has_cjk(text: str) -> bool:
    return bool(_re.search(r'[一-鿿]', text))


def _bbox_key(bbox: list) -> tuple:
    """Hashable key from a 4-point bbox for grouping same-region OCR entries."""
    return tuple(round(c, 0) for p in bbox for c in p)


def _clean_vlm_ocr(ocr_results: list[OCRResult], img_h: int = 0) -> list[OCRResult]:
    """Clean VLM OCR: drop English text when CJK exists in same bbox region."""
    if not ocr_results:
        return ocr_results

    from collections import defaultdict
    groups: dict[tuple, list[OCRResult]] = defaultdict(list)
    for r in ocr_results:
        groups[_bbox_key(r.bbox)].append(r)

    result: list[OCRResult] = []
    for group in groups.values():
        has_cjk = any(_has_cjk(r.text) for r in group)
        for r in group:
            if has_cjk and not _has_cjk(r.text) and _re.search(r'[a-zA-Z]', r.text):
                continue
            result.append(r)

    return result


from server.vlm import (
    decide as _vlm_decide_impl,
    perceive as _vlm_perceive,
    perceive_two_pass as _vlm_perceive_two_pass,
    perceive_three_pass as _vlm_perceive_three_pass,
    perceive_multi_round as _vlm_perceive_multi_round,
    perceive_and_decide as _vlm_perceive_and_decide,
    warm_up as _vlm_warm_up,
    ask_about_image as _vlm_ask_about_image,
    localization_label as _vlm_localization_label,
)
from server.neo4j_client import get_neo4j
from server import aisle_signs, photo_quality, session_log, start_points
from server.loc_fusion import (adaptive_weights, dino_gap, parse_dino_topk, parse_image_k, parse_weights,
                               sensor_sigma)
from server.visual_localization import (
    localize as _localize_photo, vlm_rerank as _vlm_rerank,
    rerank_candidates as _rerank_candidates,
    estimate_node_heading as _estimate_node_heading,
)
from server.heading import (
    convert_leg_to_relative,
    relative_direction_text,
    heading_between_nodes,
    merge_instructions, next_instruction_text,
    RelativeDirection,
)
from server.prompts import ROUTE_CONTEXT_BLOCK

log = logging.getLogger(__name__)

# How many route steps the spoken 🗺 line chains together at most — long
# enough to cover the manoeuvre ahead, short enough to stay speakable.
_MAX_CHAINED_STEPS = 3

app = FastAPI(title="UniGoal Indoor Navigation Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = SessionStore()
_perception: Optional[Perception] = None
_ocr: Optional[OCR] = None


def get_store() -> SessionStore:
    return _store


_perception_failed = False

def get_perception() -> Optional[Perception]:
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


_ocr_failed = False

def get_ocr() -> Optional[OCR]:
    global _ocr, _ocr_failed
    if _ocr_failed:
        return None
    if not OCR_ENABLED:
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


def vlm_decide(*args, **kwargs):
    return _vlm_decide_impl(*args, **kwargs)


_USELESS_LANDMARK = frozenset({
    "促銷立牌", "紅色促銷立牌", "寵物食品促銷立牌",
    "立牌", "促銷", "紙箱", "塑膠袋", "購物車", "推車",
    "柱子", "天花板", "地板", "白色地磚走道",
})

def _route_nx_graph(place: Optional[str]):
    """The one route graph — the mini-map's (_editor_graph_for: Neo4j walkways +
    maps/neo4j_links) — as a weighted networkx graph for planning the visit order and
    the legs, or None. It used to be a second graph that also joined every pair of nodes
    within 6 m, which in A7 put 18 links straight through the shelves between aisles."""
    import networkx as _nx
    g = _editor_graph_for(place)
    if g is None:
        return None
    out = _nx.Graph()
    out.add_nodes_from(g.graph.nodes)
    for a, b, d in g.graph.edges(data=True):
        out.add_edge(a, b, weight=d.get("length", 1.0))
    return out


# "Arrived" also needs the map to agree: the fused position or the photo's own best
# match within this many route metres of the goal node. In the 2026-10-17 test the
# noodle walk was "arrived" at node 52, ~20 m from the noodle aisle (node 40), after
# the user answered an unsure-arrival question; the milk walk's fused position was off
# (58) while the photo itself matched the milk fridge (16) — hence either one counts.
ARRIVAL_NEAR_M = 6.0
ARRIVAL_ASK_MAX_M = 15.0


def _far_from_goal_question(s, matched_nid: Optional[int], where: Optional[str] = None) -> tuple[Optional[str], Optional[float]]:
    """(None, None) when an arrival is plausible on the map; otherwise
    (question, distance_m).  When distance_m > ARRIVAL_ASK_MAX_M the caller
    should silently downgrade to MOVE instead of asking the user."""
    import networkx as _nx
    leg = _phase_leg(s)
    g = _route_nx_graph(s.place) if leg is not None else None
    if g is None or leg.to_node not in g:
        return None, None
    image_ev = (s.last_loc_log or {}).get("image_evidence") or {}
    image_top = int(max(image_ev, key=image_ev.get)) if image_ev else None
    dists = {}
    for c in (matched_nid, image_top):
        if c is not None and c in g and c not in dists:
            try:
                dists[c] = _nx.shortest_path_length(g, c, leg.to_node, weight="weight")
            except _nx.NetworkXNoPath:
                pass
    if not dists or min(dists.values()) <= ARRIVAL_NEAR_M:
        return None, None
    if s.phase == "checkout":
        item = "收銀台"
    elif s.phase == "exit":
        item = "出口"
    else:
        item = s.target_node_to_item.get(leg.to_node) or (s.target_objects or s.goal_objects or ["目標"])[0]
    far = dists.get(matched_nid, min(dists.values()))
    where = where or "目前的位置"
    q = f"App 認為你在{where}，離「{item}」還有約 {far:.0f} 公尺。你已經看到「{item}」了嗎？"
    return q, far


_CONFIRM_NO = ("不是", "沒有", "沒看到", "不對", "不確定", "還沒", "繼續", "再找", "否", "no", "nope")
_CONFIRM_YES = ("是", "對", "到了", "有", "看到", "找到", "yes", "ok", "好")


def _confirm_answer(text: str) -> Optional[bool]:
    """Yes / no to an arrival question; None when the reply is neither (it used to count
    as yes — "在哪裡？" confirmed an arrival). "不是"/"沒有" are checked first since they
    contain "是"/"有"."""
    t = (text or "").strip().lower()
    if any(k in t for k in _CONFIRM_NO):
        return False
    if any(k in t for k in _CONFIRM_YES):
        return True
    return None


def _replan_from(s, start_nid: int) -> bool:
    """Re-plan the rest of the walk from ``start_nid`` on the route graph: the targets not
    yet visited, then checkout (unless already past it) and exit. False when it cannot."""
    try:
        from server.path_planner import plan_route as _plan_route
        g = _route_nx_graph(s.place)
        if g is None or start_nid not in g:
            return False
        remaining = [t for t in (s.target_nodes or []) if t not in (s.visited_targets or set())]
        if s.phase == "shopping" and not remaining:
            return False
        route = _plan_route(
            g, start_nid, remaining if s.phase == "shopping" else [],
            checkout=getattr(s, "checkout_node", None) if s.phase in ("shopping", "checkout") else None,
            exit_node=getattr(s, "exit_node", None),
        )
        if not route.legs:
            return False
        s.route_plan = route
        s.current_leg_index = 0
        return True
    except Exception as e:
        log.warning("[session %s] route re-plan failed: %s", s.id, e)
        return False


def _default_start_wp(place: Optional[str]) -> Optional[int]:
    """Where the visit order is planned from before any photo or chosen start point: the
    store's entrance, else its first start point, else the node nearest the origin —
    never an arbitrary node (it used to be the lowest node id, mid-store in A7)."""
    g = _editor_graph_for(place)
    if g is None or not g.graph.number_of_nodes():
        return None
    if g.entrance == editor_map_mod.ENTRANCE_ID:
        return g.entrance
    starts = start_points.load_starts(place, EDITOR_MAP_DIR) if place else []
    if starts:
        return editor_map_mod.nearest_waypoint(g, starts[0]["x"], starts[0]["y"])
    return g.entrance


def _is_useless_name(name: str) -> bool:
    n = name.strip().lower()
    return (n in _USELESS_LANDMARK
            or "促銷" in n or "promotional" in n
            or "海報" in n or "poster" in n)


def _node_area_name(nid: int, ref_map, goal_keywords=None, goal_item=None) -> str:
    """Generate a human-friendly area name for a map node from its OCR/objects.

    When *goal_item* is provided (the user's own search term, e.g. "泡麵"), it is
    used directly so the area name matches what the user typed.  *goal_keywords*
    is still used to verify the node actually carries the goal.
    """
    if not ref_map or nid not in ref_map.photos:
        return "未知區域"
    node = ref_map.photos[nid]
    if not node.objects:
        return "未知區域"

    import re as _re
    kws = [k.lower() for k in (goal_keywords or []) if k]

    # Priority 1: aisle/section signs with actual aisle numbers — most useful
    # for navigation.  When goal_keywords is set, prefer a sign that mentions
    # the goal (「泡麵 12」) over an unrelated aisle sign at the same node.
    best_aisle_name = None
    for obj in node.objects:
        lbl = obj.label
        raw = obj.ocr_text.strip() if obj.ocr_text else lbl
        _m = _re.search(r'走道\s*\d+[／/][^A-Za-z]+', raw) or _re.search(r'走道\s*\d+[／/][^A-Za-z]+', lbl)
        if _m:
            aisle_name = _m.group(0).rstrip("標示牌 ／/")
            if aisle_name:
                if kws and any(k in raw.lower() or k in lbl.lower() for k in kws):
                    return f"「{aisle_name}」"
                if best_aisle_name is None:
                    best_aisle_name = aisle_name
                continue
        _m2 = _re.search(r'走道\s*\d+', raw) or _re.search(r'走道\s*\d+', lbl)
        if _m2:
            if kws and any(k in raw.lower() or k in lbl.lower() for k in kws):
                return f"「{_m2.group(0)}」"
            if best_aisle_name is None:
                best_aisle_name = _m2.group(0)

    # Priority 2: goal-keyword matched area name
    if kws:
        # 2a: a sign/吊牌 whose OCR mentions the goal — e.g. "泡麵 12 Instant Noodles"
        _sign_hints = ("區", "吊牌", "標示", "招牌", "sign", "category", "section", "aisle")
        for obj in node.objects:
            if not any(h in obj.label.lower() for h in _sign_hints):
                continue
            ocr = (obj.ocr_text or "").strip()
            if not ocr or len(ocr) < 2:
                continue
            if any(k in ocr.lower() for k in kws):
                # Strip English/number suffixes, trailing slashes and whitespace
                clean = _re.sub(r'[\s/]+[A-Za-z][\w\s/]*$', '', ocr).strip()
                clean = _re.sub(r'\s+\d+$', '', clean).strip().rstrip("/ ")
                # Keep the cleaned name only when it still has CJK content
                # AND one of the user's keywords appears in the cleaned text
                # (avoids showing "米果" when the user searched for "餅乾")
                if clean and _re.search(r'[一-鿿]', clean) \
                        and any(k in clean.lower() for k in kws):
                    return f"「{clean}」"

        # 2b: node carries the goal — use the user's own search term for clarity
        has_match = any(
            k in obj.label.lower() or k in (obj.ocr_text or "").lower()
            for obj in node.objects for k in kws
        )
        if has_match:
            if goal_item:
                return f"「{goal_item}」區"
            orig = next((g for g in (goal_keywords or []) if g.lower() in kws), None)
            if orig and not _is_useless_name(orig):
                return f"「{orig}」區"

    # If Priority 1 found an aisle sign but it didn't mention the goal, still use it
    if best_aisle_name:
        return f"「{best_aisle_name}」"

    # Priority 3: OCR texts
    ocr_texts = []
    seen = set()
    for obj in node.objects:
        if obj.ocr_text and obj.ocr_text.strip():
            t = obj.ocr_text.strip()
            if len(t) < 2 or t.isdigit() or t.lower() in seen or len(t) > 20:
                continue
            if _is_useless_name(t):
                continue
            ocr_texts.append(t)
            seen.add(t.lower())
        if len(ocr_texts) >= 2:
            break

    if ocr_texts:
        return "「" + "」「".join(ocr_texts) + "」附近"

    # Priority 4: object labels
    labels = []
    for obj in node.objects:
        l = obj.label.strip()
        if l.lower() not in seen and l.lower() not in ("shelf", "shelves", "wall", "floor", "ceiling"):
            if _is_useless_name(l):
                continue
            labels.append(l)
            seen.add(l.lower())
        if len(labels) >= 2:
            break
    if labels:
        return "、".join(labels) + " 區域"
    return "未知區域"


_BASE_SLOT_ORDER = ["front", "right", "back", "left"]
_SLOT_ZH = {"front": "前方", "right": "右手邊", "back": "後方", "left": "左手邊"}

_LANDMARK_KEYWORDS = frozenset({
    "區", "櫃", "架", "標示", "指示牌", "指示", "專區", "走道", "通道",
    "section", "aisle", "shelf", "display", "sign", "refrigerator", "cooler",
})


def _classify_arrival_objects(goal: str, goal_objects: list[str]) -> list[str]:
    """Return only goal_objects that represent the actual product (not landmarks).

    Landmark terms (冷藏展示櫃, 貨架, 乳製品區 …) are useful for navigation
    but must NOT trigger ARRIVED — only the product itself should.
    """
    base_items = [_re.sub(r'\s*x\d+\s*$', '', g.strip()).strip().lower()
                  for g in goal.split(",")]
    base_items = [g for g in base_items if g]

    arrival = []
    for obj in goal_objects:
        obj_clean = obj.lower().strip("「」")
        if any(base in obj_clean or obj_clean in base for base in base_items):
            arrival.append(obj)
            continue
        if not any(kw in obj for kw in _LANDMARK_KEYWORDS):
            arrival.append(obj)
    return arrival if arrival else goal_objects


_CHECKOUT_KW = ['收銀台', '收銀機', '結帳機', 'cashier counter', 'cash register',
                '自助結帳機', 'self-checkout kiosk', 'pos machine']
_CHECKOUT_SIGN_KW = ['結帳', '收銀', 'cashier', 'checkout']
_EXIT_DOOR_KW = ['出口門', '玻璃門', 'entrance door', 'exit door']
_EXIT_SIGN_KW = ['出口', 'exit']


def _find_checkout_exit_nodes(ref_map) -> tuple[int | None, int | None]:
    """Auto-detect best checkout and exit nodes from reference map objects.

    Exit is picked first (fewer candidates, higher confidence), then checkout
    is picked from the remaining nodes so the two are always distinct.
    """
    if not ref_map or not ref_map.photos:
        return None, None

    checkout_scores: dict[int, float] = {}
    exit_scores: dict[int, float] = {}

    for nid, node in ref_map.photos.items():
        cs = 0.0
        es = 0.0
        for o in node.objects:
            txt = (o.label + ' ' + (o.ocr_text or '')).lower()
            if any(kw in txt for kw in _CHECKOUT_KW):
                cs += o.score * 2.0
            elif any(kw in txt for kw in _CHECKOUT_SIGN_KW):
                cs += o.score * 0.5
            if any(kw in txt for kw in _EXIT_DOOR_KW):
                es += o.score * 3.0
            elif any(kw in txt for kw in _EXIT_SIGN_KW):
                es += o.score * 0.3
        if cs > 0:
            checkout_scores[nid] = cs
        if es > 0:
            exit_scores[nid] = es

    # Pick exit first, then exclude it from checkout candidates.
    exit_nid = max(exit_scores, key=exit_scores.get) if exit_scores else None
    checkout_candidates = {n: s for n, s in checkout_scores.items() if n != exit_nid}
    checkout_nid = max(checkout_candidates, key=checkout_candidates.get) if checkout_candidates else None

    if checkout_nid is not None:
        log.info("Auto-detected checkout node: WP%d (score=%.1f)", checkout_nid, checkout_scores[checkout_nid])
    if exit_nid is not None:
        log.info("Auto-detected exit node: WP%d (score=%.1f)", exit_nid, exit_scores[exit_nid])

    return checkout_nid, exit_nid


def _detect_goal_in_photo(detections, goal_keywords, img_w=1, img_h=1, ocr_results=None):
    """Check if goal-related objects are visible in the user's photo.

    Uses bounding box position to determine left/center/right. Text read off
    hanging aisle signs ("泡麵 12") counts too — in a store that is the most
    reliable cue of all, and it is often visible before the product itself.
    Returns (direction_zh, object_names) or (None, []).
    """
    if not goal_keywords or not (detections or ocr_results):
        return None, []

    keywords = [kw.lower().strip() for kw in goal_keywords if kw]
    matches = []

    for det in (detections or []):
        label = (det.label if isinstance(det.label, str) else str(det.label)).lower()
        box = det.box if hasattr(det, 'box') else det.get('box')
        if not box or len(box) < 4:
            continue
        # boxes are absolute pixels; the 0.35/0.65 thresholds below are fractions
        cx = (box[0] + box[2]) / 2 / (img_w or 1)
        for kw in keywords:
            if kw in label or label in kw:
                matches.append((cx, det.label if isinstance(det.label, str) else str(det.label)))
                break
            elif len(kw) >= 2 and label[:2] == kw[:2] and _re.search(r'[一-鿿]', kw[:2]):
                matches.append((cx, det.label if isinstance(det.label, str) else str(det.label)))
                break

    for ocr in (ocr_results or []):
        text = getattr(ocr, 'text', '') or ''
        text_lower = text.lower().strip()
        bbox = getattr(ocr, 'bbox', None)
        if not text_lower or not bbox:
            continue
        if isinstance(bbox[0], (list, tuple)):
            cx = sum(p[0] for p in bbox) / len(bbox) / (img_w or 1)
        elif len(bbox) >= 4:
            cx = (bbox[0] + bbox[2]) / 2 / (img_w or 1)
        else:
            continue
        for kw in keywords:
            # A one-character OCR fragment inside a keyword is noise, not a match.
            if kw in text_lower or (len(text_lower) >= 2 and text_lower in kw):
                matches.append((cx, text))
                break

    if not matches:
        return None, []

    avg_cx = sum(cx for cx, _ in matches) / len(matches)
    names = list(dict.fromkeys(name for _, name in matches))

    if avg_cx < 0.35:
        return "左手邊", names
    elif avg_cx > 0.65:
        return "右手邊", names
    else:
        return "正前方", names


def _infer_target_direction(matched_node, target_nid, ref_map,
                            user_slot: str | None,
                            goal_keywords: list[str] | None = None):
    """Infer which direction the target is from the current node.

    Strategy:
    1. Check current node's directional photos for goal keyword matches.
    2. If no match, check 1-hop neighbor nodes' directional photos.
    Uses substring + CJK prefix matching for robustness.

    Returns (direction_zh, slot_name, landmarks_in_direction) or (None, None, []).
    """
    if not ref_map:
        return None, None, []

    target_node = ref_map.photos.get(target_nid)

    keywords = set()
    if goal_keywords:
        for kw in goal_keywords:
            keywords.add(kw.lower().strip())
    if target_node:
        for obj in target_node.objects:
            if obj.role in ("landmark", "sign", "equipment"):
                keywords.add(obj.label.lower().strip())
                if obj.ocr_text:
                    keywords.add(obj.ocr_text.lower().strip())

    if not keywords:
        return None, None, []

    def _keyword_score(text: str) -> float:
        text_lower = text.lower()
        score = 0.0
        for kw in keywords:
            if kw in text_lower or text_lower in kw:
                score += 1.0
            elif len(kw) >= 2 and text_lower[:2] == kw[:2] and _re.search(r'[一-鿿]', kw[:2]):
                score += 0.6
        return score

    def _score_node_slots(node):
        slot_scores = {}
        slot_landmarks = {}
        if not node or not node.directional_photos:
            return slot_scores, slot_landmarks
        for dp in node.directional_photos:
            base_slot = dp.slot
            if "_" in base_slot:
                base_slot = base_slot.split("_", 1)[1]
            if base_slot not in _BASE_SLOT_ORDER:
                continue
            score = 0.0
            landmarks = []
            for obj in dp.objects:
                obj_text = obj.label + " " + (obj.ocr_text or "")
                s = _keyword_score(obj_text)
                if s > 0:
                    score += s
                    name = obj.ocr_text or obj.label
                    if name and name not in landmarks:
                        landmarks.append(name)
            slot_scores[base_slot] = slot_scores.get(base_slot, 0) + score
            if base_slot not in slot_landmarks:
                slot_landmarks[base_slot] = []
            slot_landmarks[base_slot].extend(landmarks)
        return slot_scores, slot_landmarks

    # 1. Check current node
    slot_scores, slot_landmarks = _score_node_slots(matched_node)

    # 2. If no match on current node, check 1-hop neighbors
    if (not slot_scores or max(slot_scores.values(), default=0) == 0) and matched_node:
        current_nid = None
        for nid, node in ref_map.photos.items():
            if node is matched_node:
                current_nid = nid
                break
        if current_nid is not None and hasattr(ref_map, 'walkway_edges'):
            # each neighbour once, also when WALKWAY is stored both ways (a→b and b→a)
            neighbor_nids = dict.fromkeys(
                edge['to'] if edge.get('from') == current_nid else edge['from']
                for edge in ref_map.walkway_edges
                if current_nid in (edge.get('from'), edge.get('to')))
            for neighbor_nid in neighbor_nids:
                if neighbor_nid and neighbor_nid in ref_map.photos:
                    nb_scores, nb_landmarks = _score_node_slots(
                        ref_map.photos[neighbor_nid])
                    for sl, sc in nb_scores.items():
                        slot_scores[sl] = slot_scores.get(sl, 0) + sc * 0.7
                    for sl, lms in nb_landmarks.items():
                        if sl not in slot_landmarks:
                            slot_landmarks[sl] = []
                        slot_landmarks[sl].extend(lms)

    if not slot_scores or max(slot_scores.values(), default=0) == 0:
        return None, None, []

    target_slot = max(slot_scores, key=slot_scores.get)
    landmarks = list(dict.fromkeys(slot_landmarks.get(target_slot, [])))

    if not user_slot:
        return _SLOT_ZH.get(target_slot), target_slot, landmarks

    user_base = user_slot
    if "_" in user_base:
        user_base = user_base.split("_", 1)[1]

    if user_base not in _BASE_SLOT_ORDER or target_slot not in _BASE_SLOT_ORDER:
        return _SLOT_ZH.get(target_slot), target_slot, landmarks

    user_idx = _BASE_SLOT_ORDER.index(user_base)
    target_idx = _BASE_SLOT_ORDER.index(target_slot)
    diff = (target_idx - user_idx) % 4

    rel_zh = {0: "正前方", 1: "右手邊", 2: "後方", 3: "左手邊"}
    return rel_zh.get(diff), target_slot, landmarks


PHONE_HEADING_CONFIDENCE = 0.85
TURN_DIRECTION_MIN_HEADING_CONFIDENCE = 0.6
LOCALIZE_FAILED_HINT = "這張照片對不到地圖上的位置，請換個角度、拍到走道標示牌或招牌再試一次"


def _goal_node_score(objects, search_terms: list[str]) -> tuple[float, float]:
    """How well a reference node's objects match a goal item, as a sortable (text, match) pair.

    text  — summed over objects: text read off the photo (signs, shelf labels, packaging)
            equal to a search term counts 1.0, containing one 0.8. The milk fridge with
            「Milk」「Milk」「鮮乳」 beats a shelf where one drink reads 「milk」.
    match — the old score, deciding ties on text: the best single object (text as above,
            a detector label naming the item 0.7) plus 0.02 per matching object, up to 0.1.

    Single-character text ("E", "D") is ignored: as a substring it "matches" almost any term.
    """
    terms = [t.lower() for t in search_terms]
    text_sum = 0.0
    best = 0.0
    matches = 0
    for obj in objects:
        text_score = 0.0
        label_score = 0.0
        ocr = (obj.ocr_text or "").strip().lower()
        label = obj.label.lower()
        for term in terms:
            if len(ocr) >= 2:
                if term == ocr:
                    text_score = max(text_score, 1.0)
                elif term in ocr or ocr in term:
                    text_score = max(text_score, 0.8)
            if label and (term in label or label in term):
                label_score = 0.7
        obj_score = max(text_score, label_score)
        if obj_score > 0:
            matches += 1
        text_sum += text_score
        best = max(best, obj_score)
    return round(text_sum, 4), best + min(matches * 0.02, 0.1)


def _useful_area_name(area: str) -> Optional[str]:
    if not area or not any(k in area for k in ("走道", "結帳", "出口", "收銀")):
        return None
    clean = area.replace("「", "").replace("」", "").split("附近")[0].strip()
    return clean or None


def _apply_phone_heading(s, loc_result, heading: Optional[float]) -> None:
    if heading is None:
        return
    h = float(heading) % 360.0
    s.user_heading = h
    s.heading_confidence = max(s.heading_confidence or 0.0, PHONE_HEADING_CONFIDENCE)
    if loc_result is not None and loc_result.matched_nid is not None:
        loc_result.matched_heading = h
        loc_result.heading_confidence = max(loc_result.heading_confidence, PHONE_HEADING_CONFIDENCE)


def _walkway_distances(ref_map) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    if not ref_map:
        return out
    for edge in getattr(ref_map, "walkway_edges", None) or []:
        a, b = edge.get("from"), edge.get("to")
        if a is None or b is None:
            continue
        d = float(edge.get("distance_m") or 0.0)
        if d <= 0:
            na, nb = ref_map.photos.get(a), ref_map.photos.get(b)
            if na and nb:
                d = math.hypot(nb.pdr_x - na.pdr_x, nb.pdr_y - na.pdr_y)
        out[(a, b)] = d
        out[(b, a)] = d
    return out


def _path_from_current(ref_map, current_nid, leg_path: list[int], target_nid: int) -> list[int]:
    if current_nid is None:
        return leg_path
    if current_nid in leg_path:
        return leg_path[leg_path.index(current_nid):]
    distances = _walkway_distances(ref_map)
    if distances and current_nid in ref_map.photos and target_nid in ref_map.photos:
        import networkx as _nx
        from server.path_planner import astar as _astar
        g = _nx.Graph()
        g.add_nodes_from(ref_map.photos)
        for (a, b), d in distances.items():
            g.add_edge(a, b, weight=d or 1.0)
        replanned = _astar(g, current_nid, target_nid)
        if replanned and len(replanned) > 1:
            return replanned
    return leg_path


def _build_route_context(s, loc_result, detections=None,
                         img_w: int = 0, img_h: int = 0,
                         ocr_results=None,
                         is_first_localization: bool = False) -> tuple[str | None, str | None]:
    """Build a route context string for the VLM prompt + a one-line next instruction.

    Returns (route_context_block, next_instruction).
    route_context_block is the full text block for the VLM prompt (or None).
    next_instruction is a short Chinese instruction for the iOS app (or None).
    """
    localized = (loc_result
                 and loc_result.matched_nid is not None
                 and loc_result.ref_node is not None)
    has_route = (s.route_plan
                 and hasattr(s.route_plan, 'legs')
                 and s.route_plan.legs)

    if not localized and not has_route:
        log.warning("🔴 [session %s] _build_route_context: no localization AND no route plan → skipping", s.id)
        # Silence looks like "the map was ignored" when a map is in use — say so.
        if s.place:
            return None, LOCALIZE_FAILED_HINT
        return None, None

    if localized:
        log.info("🟡 [session %s] _build_route_context: localized to node #%s (conf=%.2f, heading=%s, slot=%s)",
                 s.id, loc_result.matched_nid, loc_result.confidence,
                 loc_result.matched_heading, loc_result.matched_slot)
    else:
        log.warning("🟡 [session %s] _build_route_context: localization failed but route plan exists — providing degraded context", s.id)

    # Load reference map once for area name lookups
    neo4j = get_neo4j()
    ref_map = neo4j.load_reference_map(s.place) if neo4j and s.place else None
    ref_node = loc_result.ref_node if localized else None

    # Position description
    if localized:
        friendly_loc = _ref_location_name(loc_result) or "未知區域"
        position_desc = f"{friendly_loc}（信心 {loc_result.confidence:.0%}）"
    else:
        position_desc = "定位中（尚未確認確切位置）"

    # Heading description
    heading_ok = (localized
                  and loc_result.matched_heading is not None
                  and loc_result.heading_confidence
                  >= TURN_DIRECTION_MIN_HEADING_CONFIDENCE
                  and loc_result.confidence >= 0.55)
    if heading_ok:
        slot_zh = {"front": "前方（行走方向）", "right": "右方", "back": "後方", "left": "左方"}
        slot_label = slot_zh.get(loc_result.matched_slot or "", "未知方向")
        heading_desc = f"面朝 {slot_label}"
    else:
        heading_desc = "朝向不確定"

    # Route description: only the CURRENT leg — guide one step at a time
    next_instruction = None
    log.info("🟡 [session %s] route_plan exists=%s, legs=%d, current_leg_index=%d, phase=%s",
             s.id, bool(s.route_plan), len(s.route_plan.legs) if has_route else 0,
             s.current_leg_index, s.phase)
    if has_route:
        remaining = s.remaining_targets
        leg_idx = min(s.current_leg_index, len(s.route_plan.legs) - 1)
        current_leg = s.route_plan.legs[leg_idx]

        # Re-plan route when user is localized with sufficient confidence
        # Lower threshold on first localization (no prior position known)
        _replan_threshold = 0.25 if is_first_localization else 0.45
        if (localized and ref_map
                and loc_result.confidence >= _replan_threshold
                and loc_result.matched_nid not in current_leg.path):
            if _replan_from(s, loc_result.matched_nid):
                leg_idx = 0
                current_leg = s.route_plan.legs[0]
                log.info("🔄 [session %s] re-planned route from WP%d (was not on leg path %s)",
                         s.id, loc_result.matched_nid, current_leg.path[:5])

        # The node the user walks to now: the leg for the current phase, or none (a store
        # without a detected checkout has no checkout leg — the last shopping leg must
        # not stand in for it, or "地圖顯示您已在收銀台附近" shows up at the last item).
        phase_leg = _phase_leg(s)
        target_nid = phase_leg.to_node if phase_leg is not None else None

        # Friendly name for the target area — depends on current phase
        if s.phase == "checkout":
            target_area = "收銀台"
        elif s.phase == "exit":
            target_area = "出口"
        elif target_nid is not None:
            target_area = _node_area_name(target_nid, ref_map, goal_keywords=s.goal_objects,
                                          goal_item=s.target_node_to_item.get(target_nid))
        else:
            target_area = "目標區域"
        at_target = localized and target_nid is not None and loc_result.matched_nid == target_nid

        # Which goal item is at this target? Phase first, then the product word
        # that the target node actually carries. ``target_objects`` is the split
        # product list — its first entry is the user's own wording ("泡麵"), not
        # an English label the decomposer invented, so it is the safe fallback.
        _targets = s.target_objects or s.goal_objects
        if s.phase == "checkout":
            goal_item = "收銀台"
        elif s.phase == "exit":
            goal_item = "出口"
        elif target_nid is not None and s.target_node_to_item.get(target_nid):
            goal_item = s.target_node_to_item[target_nid]     # the item chosen for this node at session start
        else:
            goal_item = "target"
            if _targets and target_nid is not None:
                # Try to find which goal object maps to this target node
                for g in _targets:
                    if ref_map and target_nid in ref_map.photos:
                        for obj in ref_map.photos[target_nid].objects:
                            if (obj.ocr_text and g.lower() in obj.ocr_text.lower()) or \
                               g.lower() in obj.label.lower():
                                goal_item = g
                                break
                        if goal_item != "target":
                            break
            if goal_item in ("target", "checkout", "exit") and _targets:
                goal_item = _targets[0]

        if s.phase == "checkout":
            phase_status = "所有商品已找到，正前往收銀台"
        elif s.phase == "exit":
            phase_status = "已結帳完成，正前往出口"
        else:
            phase_status = f"還有 {len(remaining)} 項商品待尋找"

        # --- Priority 1: check if goal is visible in the user's photo ---
        goal_kws = [goal_item] + (_targets or [])
        photo_dir_zh, photo_obj_names = _detect_goal_in_photo(
            detections, goal_kws, img_w=img_w or 1, img_h=img_h or 1, ocr_results=ocr_results)

        direction_hint = ""
        if photo_dir_zh:
            obj_text = "、".join(photo_obj_names[:3])
            direction_hint = (
                f"重要：使用者的照片中已經可以看到目標相關物件（{obj_text}），"
                f"位於畫面的「{photo_dir_zh}」。請直接引導使用者往那個方向走。\n"
            )
        elif localized and loc_result.confidence >= 0.45 and target_nid is not None:
            # --- Priority 2: infer from directional reference photos (needs reliable localization) ---
            matched_node = ref_map.photos.get(loc_result.matched_nid) if ref_map else None
            dir_zh, dir_slot, dir_landmarks = _infer_target_direction(
                matched_node, target_nid, ref_map, loc_result.matched_slot,
                goal_keywords=goal_kws)

            if dir_zh:
                lm_text = ""
                if dir_landmarks:
                    lm_text = f"附近會看到：{'、'.join(dir_landmarks[:4])}\n"
                if dir_zh == "後方":
                    direction_hint = (
                        f"根據地圖資料，「{goal_item}」在使用者來的方向。"
                        f"請引導使用者掉頭，沿著走道往回走。\n{lm_text}"
                    )
                else:
                    direction_hint = (
                        f"根據地圖資料，「{goal_item}」在使用者的「{dir_zh}」方向。\n"
                        f"{lm_text}"
                    )
            else:
                direction_hint = (
                    f"目前無法判斷「{goal_item}」的確切方向。"
                    f"請用照片中偵測到的具體物件當地標來描述目前位置，"
                    f"引導使用者繼續沿走道前進。不要猜測目標在左或右，"
                    f"不要提供具體距離，也不要叫使用者自己去找標示。\n"
                )
        else:
            # Not localized (or no map node for this phase's goal) — can't infer direction
            # from the map, but still guide toward the target
            _clean_target = target_area.replace("「", "").replace("」", "")
            direction_hint = (
                (f"目前尚未精確定位，無法從地圖推斷方向。" if not localized
                 else f"地圖上沒有「{goal_item}」的確切位置，無法從地圖推斷方向。")
                + f"請根據照片中看得到的物件來引導使用者繼續前進。"
                f"目標「{goal_item}」在「{_clean_target}」區域。\n"
            )

        # user_heading=None makes the route read as if the user faces along the
        # first edge: the distances stay, the left/right claims drop out.
        heading_reliable = (localized
                            and ref_map
                            and loc_result.matched_heading is not None
                            and loc_result.heading_confidence
                            >= TURN_DIRECTION_MIN_HEADING_CONFIDENCE
                            and loc_result.confidence >= 0.55)
        user_heading = loc_result.matched_heading if heading_reliable else None

        # Steps are computed on the route graph (_editor_graph_for: the Neo4j map +
        # maps/neo4j_links, or the hand-corrected editor map) — the same route the
        # phone's PathFollower walks — so the 🗺 line and the VLM prompt agree with
        # the mini-map. The raw walkway-leg conversion below is only the fallback.
        steps_zh: list[str] = []      # for the VLM prompt: step + the area it leads to
        step_texts: list[str] = []    # for the app/TTS: the steps themselves
        next_zh: Optional[str] = None
        # How many entries of step_texts the one-line head (next_instruction_text)
        # already covers: 2 when the first step is a straight run that ends in a
        # turn ("直走約 12 公尺，然後右轉"), otherwise only the first step. The
        # multi-step chain below must resume after exactly these, or a leg is lost.
        head_steps = 1
        editor = _editor_graph_for(s.place) if localized else None
        editor_route_found = False
        if editor is not None:
            start_wp = editor.waypoint_for_neo(loc_result.matched_nid)
            if start_wp is None and ref_node is not None:
                start_wp = editor_map_mod.nearest_waypoint(editor, ref_node.pdr_x, ref_node.pdr_y)
            # The session's leg first, as the mini-map's /path does, so the steps lead
            # to the node the area name and the arrival check use (泡麵: 40, not the
            # label search's 7); the label search only without a leg for this phase.
            goal_wp = editor.waypoint_for_neo(target_nid) if target_nid is not None else None
            if goal_wp is None:
                goal_wp = editor_map_mod.find_goal_waypoint(editor, goal_item)
            path = (editor_map_mod.plan(editor, start_wp, goal_wp)
                    if start_wp is not None and goal_wp is not None else [])
            if path:   # unreachable on the route graph → the walkway-leg fallback below
                editor_route_found = True
                at_target = at_target or start_wp == goal_wp
                route = editor_map_mod.route_payload(editor, path, user_heading=user_heading)
                turns = route["turns"]
                steps_zh = [t["text_zh"] for t in turns[:3]]
                step_texts = [t["text_zh"] for t in turns]
                next_zh = route.get("next_instruction_zh") or None
                if len(turns) > 1 and turns[0]["direction"] == RelativeDirection.STRAIGHT:
                    head_steps = 2
        if not editor_route_found and heading_reliable and phase_leg is not None:
            node_positions = {nid: (node.pdr_x, node.pdr_y) for nid, node in ref_map.photos.items()}
            route_graph = _editor_graph_for(s.place)
            if route_graph is not None:       # the virtual entrance is a waypoint too
                node_positions.update({n: (d["x"], d["y"]) for n, d in route_graph.graph.nodes(data=True)})
            rel_instructions = convert_leg_to_relative(
                _path_from_current(ref_map, loc_result.matched_nid,
                                   phase_leg.path, phase_leg.to_node),
                node_positions,
                loc_result.matched_heading,
                distances=_walkway_distances(ref_map),
            )
            # Fold straight runs into one step with the summed distance and the
            # number of intersections passed, so "右轉" becomes "直走約 12 公尺
            # （經過 1 個路口），然後右轉" when two aisles both open to the right.
            steps = merge_instructions(rel_instructions)
            steps_zh = [f"{st.text_zh}，往{_node_area_name(st.to_node, ref_map)}方向" for st in steps[:3]]
            for st in steps:
                # Name where a turn leads, but only when the area name says
                # something ("右轉往走道14") — never "右轉往「貨架」附近".
                area = _useful_area_name(_node_area_name(st.to_node, ref_map))
                step_texts.append(f"{st.text_zh}往{area}" if area else st.text_zh)
            next_zh = next_instruction_text(steps) or None
            if len(steps) > 1 and steps[0].direction == RelativeDirection.STRAIGHT:
                head_steps = 2

        # Multi-step guidance: let the user hear the whole manoeuvre, not just the
        # edge in front of them. next_instruction_text() may return the combined
        # "直走約 12 公尺（經過 1 個路口），然後右轉" head, which names step #1's
        # direction but never its distance — so once there is anything to chain past
        # that head, chain the bare steps instead, each of which carries its own
        # distance and junction count. Otherwise step #1's distance would never be
        # spoken and the user would turn early. A route short enough to need no
        # chaining keeps the combined head.
        if next_zh and len(step_texts) > head_steps:
            next_zh = "→".join(step_texts[:_MAX_CHAINED_STEPS])

        if at_target:
            if s.phase in ("checkout", "exit"):
                next_instruction = f"地圖顯示您已在{target_area}附近，請環顧四周"
            else:
                next_instruction = f"地圖顯示您就在「{goal_item}」附近（{target_area}），請環顧四周找找看"
            route_desc = (
                f"目前目標：「{goal_item}」\n"
                f"位於：{target_area}\n"
                f"地圖定位顯示使用者已在目標節點（信心 {loc_result.confidence:.0%}），"
                f"請依照片判斷目標是否就在附近；若照片中沒有，請引導使用者環顧或往前幾步再拍。\n"
                f"{direction_hint}"
                f"{phase_status}"
            )
        elif next_zh:
            if s.phase in ("checkout", "exit"):
                next_instruction = f"{next_zh}，到{target_area}"
            else:
                next_instruction = f"{next_zh}，到{target_area}找「{goal_item}」"
            route_desc = (
                f"目前目標：「{goal_item}」\n"
                f"位於：{target_area}\n"
                f"{direction_hint}"
                f"建議走法：{'；'.join(steps_zh)}\n"
                f"{phase_status}"
            )
        else:

            if s.phase == "checkout":
                phase_status = "所有商品已找到，正前往收銀台"
            elif s.phase == "exit":
                phase_status = "已結帳完成，正前往出口"
            else:
                phase_status = f"還有 {len(remaining)} 項商品待尋找"

            area_steps = ""
            if localized and ref_map and phase_leg is not None and len(phase_leg.path) > 1:
                areas_on_route = []
                for step_nid in phase_leg.path[1:6]:
                    area = _node_area_name(step_nid, ref_map)
                    if area and area not in areas_on_route[-1:]:
                        areas_on_route.append(area)
                if areas_on_route:
                    area_steps = f"路線經過的區域：{'→'.join(areas_on_route)}\n"

            route_desc = (
                f"目前目標：「{goal_item}」\n"
                f"位於：{target_area}\n"
                f"{direction_hint}"
                f"{area_steps}"
                f"{phase_status}"
            )

        # Fallback next_instruction
        if not next_instruction and localized:
            if s.phase in ("checkout", "exit"):
                next_instruction = f"前往{target_area}"
            else:
                next_instruction = f"前往{target_area}找「{goal_item}」"
        elif not next_instruction and s.place:
            next_instruction = LOCALIZE_FAILED_HINT
    else:
        route_desc = "尚未規劃路線，請根據照片中的標示和商品引導使用者。"
        log.warning("🔴 [session %s] _build_route_context: NO route plan → VLM will navigate by photo only", s.id)

    log.info("🟢 [session %s] _build_route_context: route_desc=%s", s.id, route_desc[:120])
    context_block = ROUTE_CONTEXT_BLOCK.format(
        position_description=position_desc,
        heading_description=heading_desc,
        route_description=route_desc,
    )
    return context_block, next_instruction


def _phase_leg(s):
    """The route leg the user walks now — chosen when the session started — or None.

    Only a leg serving the current phase counts: without a checkout node on the map
    the plan has no checkout leg, and the checkout phase must not keep pointing at
    the last shopping target.
    """
    if not (s.route_plan and getattr(s.route_plan, "legs", None)):
        return None
    leg = s.route_plan.legs[min(s.current_leg_index, len(s.route_plan.legs) - 1)]
    if leg.purpose == {"checkout": "checkout", "exit": "exit"}.get(s.phase, "target"):
        return leg
    # Checkout and exit at the same node (A7: node 25) plan as one leg labelled "exit"
    if s.phase == "checkout" and leg.purpose == "exit" and leg.to_node == getattr(s, "checkout_node", None):
        return leg
    return None


def _current_goal_node_id(s) -> int | None:
    """Return the node the user should walk toward right now (for AR arrow): the leg for
    the current phase; None when the plan has none (e.g. no checkout node on the map)."""
    if s.route_plan and hasattr(s.route_plan, 'legs') and s.route_plan.legs:
        leg = _phase_leg(s)
        return leg.to_node if leg is not None else None
    return s.goal_node


def _shopping_progress(s) -> tuple[list[str], list[str], str | None]:
    """Return (goal_item_names, visited_item_names, current_target_name) for shopping checklist."""
    if not s.target_node_to_item:
        return None, None, None
    visit_order = s.route_plan.visit_order if s.route_plan else s.target_nodes
    goal_items = [s.target_node_to_item.get(n, f"WP{n}") for n in visit_order
                  if n in s.target_node_to_item]
    visited = [s.target_node_to_item[n] for n in s.visited_targets
               if n in s.target_node_to_item]
    current = None
    if s.phase == "shopping" and s.route_plan and s.route_plan.legs:
        leg_idx = min(s.current_leg_index, len(s.route_plan.legs) - 1)
        leg = s.route_plan.legs[leg_idx]
        if leg.purpose == "target":
            current = s.target_node_to_item.get(leg.to_node)
    return goal_items, visited, current


def _current_goal_objects(s) -> list[str]:
    """Return ALL remaining goal objects so VLM can recognize any target in the photo.

    When a route plan exists, the current leg's target is listed first (primary focus),
    but all unvisited targets are included so VLM can trigger ARRIVED opportunistically
    if the user happens to pass by a non-current target.
    """
    if s.phase == "checkout":
        return ["收銀台", "結帳", "cashier", "checkout"]
    if s.phase == "exit":
        return ["出口", "exit"]

    goal_objects = s.target_objects or s.goal_objects
    if not s.route_plan or not s.route_plan.legs or not goal_objects:
        return goal_objects

    leg_idx = min(s.current_leg_index, len(s.route_plan.legs) - 1)
    current_leg = s.route_plan.legs[leg_idx]
    target_nid = current_leg.to_node

    neo4j = get_neo4j()
    ref_map = neo4j.load_reference_map(s.place) if neo4j and s.place else None

    primary = None
    if ref_map and target_nid in ref_map.photos:
        node = ref_map.photos[target_nid]
        for g in goal_objects:
            for obj in node.objects:
                if (obj.ocr_text and g.lower() in obj.ocr_text.lower()) or \
                   g.lower() in obj.label.lower():
                    primary = g
                    break
            if primary:
                break

    result = []
    if primary:
        result.append(primary)
    for g in goal_objects:
        if g not in result:
            result.append(g)
    return result


def _build_localization_items(detections, ocr_results, img_w, img_h):
    """Build detected_items (with normalized [0,1] bboxes) for grid matching.
    OCR items are now handled separately via _build_ocr_items with dedicated weight.
    """
    items = []
    for d in detections:
        if photo_quality.is_occluder(d.label):
            continue   # people / carts move around — never part of the reference map
        box = getattr(d, 'box', None) or getattr(d, 'bbox', None)
        if box and len(box) == 4 and not isinstance(box[0], (list, tuple)):
            norm_box = [box[0] / img_w, box[1] / img_h,
                        box[2] / img_w, box[3] / img_h]
            items.append({"label": d.label, "box": norm_box})
        else:
            items.append({"label": d.label, "box": None})
    return items


def _build_ocr_items(ocr_results, img_w, img_h):
    """Build normalized OCR items with polygon bboxes for grid-level OCR matching."""
    items = []
    for r in ocr_results:
        text = getattr(r, 'text', "")
        if not text or len(text.strip()) < 2:
            continue
        bbox = getattr(r, 'bbox', None)
        if bbox and isinstance(bbox[0], (list, tuple)):
            norm_bbox = [[p[0] / img_w, p[1] / img_h] for p in bbox]
        elif bbox and len(bbox) == 4:
            norm_bbox = [[bbox[0] / img_w, bbox[1] / img_h],
                         [bbox[2] / img_w, bbox[1] / img_h],
                         [bbox[2] / img_w, bbox[3] / img_h],
                         [bbox[0] / img_w, bbox[3] / img_h]]
        else:
            norm_bbox = None
        items.append({"text": text.strip(), "bbox": norm_bbox})
    return items


def _finite(v) -> bool:
    return v is not None and math.isfinite(v)


def _phone_localization_inputs(dino_topk, dino_node_hint, map_x, map_y,
                               dist_since_fix=None, fix_node=None) -> dict:
    """The phone's own localization signals from the /photo form, for loc_fusion.
    A position or distance that is not a finite number is dropped (NaN would poison
    every node's score)."""
    pos = (map_x, map_y) if _finite(map_x) and _finite(map_y) else None
    if not (_finite(dist_since_fix) and dist_since_fix >= 0):
        dist_since_fix = None
    return {
        "dino_sims": parse_dino_topk(dino_topk, dino_node_hint),
        "sensor_pos": pos,
        "dist_since_fix": dist_since_fix if pos is not None else None,
        "fix_node": fix_node if pos is not None else None,
    }


def _reachable_nodes(ref_map, fix_node, dist_since_fix):
    """Nodes walkable from the phone's last accepted fix within 1.5 × the distance
    walked since + 8 m (walkway graph, so a node behind a shelf is far)."""
    import heapq
    if fix_node is None or dist_since_fix is None or fix_node not in ref_map.photos:
        return None
    limit = 1.5 * dist_since_fix + 8.0
    # Walkway edges are stored one-way and sparsely (89 of 91 have no reverse link),
    # so walk them both ways — as neighbourhood_nids and _build_nav_graph do.
    adj: dict = {nid: set() for nid in ref_map.photos}
    for a_nid, a_node in ref_map.photos.items():
        for b_nid in a_node.neighbor_nids:
            if b_nid in ref_map.photos:
                adj[a_nid].add(b_nid)
                adj[b_nid].add(a_nid)
    best = {fix_node: 0.0}
    heap = [(0.0, fix_node)]
    while heap:
        d, n = heapq.heappop(heap)
        if d > best.get(n, float("inf")) or d > limit:
            continue
        a = ref_map.photos[n]
        for m in adj[n]:
            b = ref_map.photos[m]
            nd = d + math.hypot(a.pdr_x - b.pdr_x, a.pdr_y - b.pdr_y)
            if nd <= limit and nd < best.get(m, float("inf")):
                best[m] = nd
                heapq.heappush(heap, (nd, m))
    return set(best)


def _fusion_weights_for(dist_since_fix, n_objects, dino_sims=None):
    """LOC_FUSION_WEIGHTS, adapted per photo (LOC_FUSION_ADAPTIVE): the sensor weight
    follows the expected PDR error for the distance walked since the last photo fix,
    and the DINOv2 weight follows DINOv2's own top-1 vs top-2 margin."""
    base = parse_weights(_config.LOC_FUSION_WEIGHTS)
    if not _config.LOC_FUSION_ADAPTIVE:
        return base
    sigma = sensor_sigma(dist_since_fix, _config.LOC_SENSOR_SIGMA0, _config.LOC_DRIFT_RATE)
    return adaptive_weights(base, sigma, n_objects, c=_config.LOC_SENSOR_C,
                            cap=_config.LOC_SENSOR_CAP, n_full=_config.LOC_GRID_NFULL,
                            gap=dino_gap(dino_sims or {}), gap_full=_config.LOC_DINO_GAP_FULL)


def _server_dino_evidence(session_id, ref_map, image_path, detected_labels, ocr_texts,
                          detected_items=None):
    """Server-side DINOv2 + object + OCR evidence (ServerImageEvidence) for a phone that
    sent no dino_topk, or None (model / sidecar missing, or embedding failed) → grid
    matching as before. detected_items (labels + 0..1 boxes) feed the optional
    上下左右 relation term (config LOC_RELATION_WEIGHT)."""
    from server.locate_fusion import server_image_evidence
    t0 = time.time()
    try:
        ev = server_image_evidence(ref_map, image_path, detected_labels, ocr_texts,
                                   _config.DINO_SIDECAR_PATH, q_items=detected_items,
                                   relation_weight=_config.LOC_RELATION_WEIGHT)
    except Exception:
        log.exception("[session %s] server DINOv2 evidence failed — using grid matching", session_id)
        return None
    if ev:
        top = sorted(ev.scores.items(), key=lambda kv: -kv[1])[:3]
        log.info("⏱ server DINOv2: %.1fs, top3 %s", time.time() - t0,
                 [(n, round(v, 2), getattr(ev.best_photo.get(n), "slot", None)) for n, v in top])
    else:
        log.info("[session %s] server DINOv2 unavailable — using grid matching", session_id)
    return ev


def _run_early_localization(s, session_id, detected_labels, ocr_texts,
                            detected_items=None, image_path=None,
                            store_zone=None, ocr_items=None,
                            dino_sims=None, sensor_pos=None, dist_since_fix=None,
                            fix_node=None, extra_evidence=None, sign_numbers=None):
    """Run Neo4j localization early (before VLM decide) so route context is available.

    Returns the LocalizationResult, or None if localization is skipped/failed.
    Also stores heading on the session.
    """
    from server.config import (
        RERANK_ENABLED, REF_PHOTO_ROOT,
        OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_BACKUP_KEY,
    )
    if not s.place:
        log.warning("🔴 [session %s] _run_early_localization: s.place is empty — skipped", session_id)
        return None
    neo4j = get_neo4j()
    if not neo4j:
        log.warning("🔴 [session %s] _run_early_localization: Neo4j not connected — skipped", session_id)
        return None
    s.last_loc_log = None
    rerank_log: Optional[dict] = None
    jump_log: Optional[float] = None
    try:
        v2_kwargs = {}
        if _config.LOC_FUSION_V2:
            ref_for_v2 = neo4j.load_reference_map(s.place)
            numbers = dict(aisle_signs.numbers_in_items(detected_items or []))
            for n, a in (sign_numbers or {}).items():
                numbers[n] = max(numbers.get(n, 0.0), a)
            aisle_ev = (aisle_signs.sign_evidence(numbers, ref_for_v2, s.place)
                        if numbers and s.place in _config.AISLE_SIGN_PLACES else None)
            if numbers:
                log.info("[session %s] aisle signs seen: %s", session_id,
                         {n: round(a, 4) for n, a in sorted(numbers.items())})
            v2_kwargs = dict(
                v2=True,
                sensor_sigma=sensor_sigma(dist_since_fix, _config.LOC_SENSOR_SIGMA0, _config.LOC_DRIFT_RATE),
                reachable=_reachable_nodes(ref_for_v2, fix_node, dist_since_fix),
                extra_evidence=extra_evidence,
                aisle_evidence=aisle_ev,
                phone_image_k=parse_image_k(_config.LOC_PHONE_IMAGE_K),
            )
            if not dino_sims and image_path and _config.LOC_SERVER_DINO:
                server_ev = _server_dino_evidence(session_id, ref_for_v2, image_path,
                                                  detected_labels, ocr_texts, detected_items)
                if server_ev:
                    v2_kwargs.update(server_evidence=server_ev.scores,
                                     server_best_photo=server_ev.best_photo,
                                     server_image_k=parse_image_k(_config.LOC_SERVER_IMAGE_K))
        loc_result = _localize_photo(
            detected_labels=detected_labels,
            ocr_texts=ocr_texts,
            neo4j=neo4j,
            hint_nid=getattr(s, "last_corrected_nid", None),
            place=s.place,
            detected_items=detected_items,
            store_zone=store_zone,
            ocr_items=ocr_items,
            grid_n=_config.GRID_N,
            dino_sims=dino_sims,
            sensor_pos=sensor_pos,
            fusion_weights=None if v2_kwargs else _fusion_weights_for(
                dist_since_fix, len(detected_items) if detected_items else len(detected_labels),
                dino_sims),
            **v2_kwargs,
        )

        ref_map = neo4j.load_reference_map(s.place)

        if (RERANK_ENABLED and image_path and REF_PHOTO_ROOT
                and loc_result.matched_nid is not None
                and loc_result.top_candidates):
            # Word-matcher top-N plus the nodes around the previous fix: the
            # user cannot have walked far, and the word matcher often leaves
            # the true node out of its top-N altogether.
            top = _rerank_candidates(loc_result.top_candidates, ref_map,
                                     getattr(s, "last_corrected_nid", None))
            log.info("[session %s] rerank triggered (always-on): WP%d %.3f | candidates %s",
                     session_id, top[0][0], top[0][1], [n for n, _, _ in top])
            t_rerank = time.time()
            reranked = _vlm_rerank(
                query_image_path=image_path,
                candidates=top,
                ref_map=ref_map,
                ref_photo_root=REF_PHOTO_ROOT,
                api_key=OPENAI_API_KEY,
                api_base_url=OPENAI_BASE_URL,
                api_model=OPENAI_MODEL,
                backup_key=OPENAI_BACKUP_KEY,
            )
            log.info("⏱ rerank: %.1fs (%d candidate nodes)", time.time() - t_rerank, len(top))
            rerank_log = {"candidates": [n for n, _, _ in top],
                          "picked": reranked[0][0] if reranked else None, "accepted": False}
            if reranked and reranked[0][0] != loc_result.matched_nid:
                new_nid = reranked[0][0]
                grid_top_score = top[0][1] if top else 0
                grid_runner_score = top[1][1] if len(top) > 1 else 0
                grid_gap = grid_top_score - grid_runner_score
                rerank_nid_score = next((sc for nid, sc, _ in top if nid == new_nid), 0)
                score_drop = grid_top_score - rerank_nid_score
                new_ref = None  # stays None when rejected: keep the grid node and its heading
                if grid_gap > 0.08 and score_drop > 0.1:
                    log.info("[session %s] rerank REJECTED: grid gap=%.3f, score drop=%.3f (WP%d→WP%d)",
                             session_id, grid_gap, score_drop, top[0][0], new_nid)
                else:
                    new_ref = ref_map.photos.get(new_nid)
                    log.info("[session %s] rerank accepted: WP%d → WP%d (gap=%.3f, drop=%.3f)",
                             session_id, loc_result.matched_nid, new_nid, grid_gap, score_drop)
                    loc_result.matched_nid = new_nid
                    loc_result.ref_node = new_ref
                    loc_result.method = "grid+rerank"
                    loc_result.reasoning = reranked[0][2]
                    loc_result.matched_photo = None   # it was the old node's photo
                    rerank_log["accepted"] = True

                if new_ref and new_ref.directional_photos:
                    best_photo = (v2_kwargs.get("server_best_photo") or {}).get(new_nid)
                    loc_result.matched_photo = best_photo.photo_file if best_photo is not None else None
                    new_heading, new_slot_str, new_conf = _estimate_node_heading(
                        new_ref, detected_labels, ocr_texts, loc_result.confidence, best_photo=best_photo)
                    if new_heading is not None:
                        loc_result.matched_heading = new_heading
                        loc_result.matched_slot = new_slot_str
                        loc_result.heading_confidence = new_conf
                        log.info("[session %s] rerank: recomputed heading=%.0f° slot=%s for WP%d",
                                 session_id, new_heading, new_slot_str, new_nid)
                    else:
                        loc_result.matched_heading = None
                        loc_result.matched_slot = None
                        loc_result.heading_confidence = 0.0
                        log.info("[session %s] rerank: WP%d has no reliable heading data", session_id, new_nid)

        # Spatial continuity: penalise jumps that are too large for walking
        prev_nid = getattr(s, "last_corrected_nid", None)
        if (loc_result.matched_nid is not None
                and prev_nid is not None
                and prev_nid != loc_result.matched_nid
                and ref_map):
            prev_ref = ref_map.photos.get(prev_nid)
            cur_ref = loc_result.ref_node or ref_map.photos.get(loc_result.matched_nid)
            if prev_ref and cur_ref:
                import math as _m
                dx = cur_ref.pdr_x - prev_ref.pdr_x
                dy = cur_ref.pdr_y - prev_ref.pdr_y
                jump_m = _m.sqrt(dx * dx + dy * dy)
                if jump_m > 15 and loc_result.confidence < 0.65:
                    log.warning("[session %s] spatial jump WP%d→WP%d = %.1fm with conf=%.2f — "
                                "clearing heading, keeping position for fallback",
                                session_id, prev_nid, loc_result.matched_nid, jump_m, loc_result.confidence)
                    loc_result.matched_heading = None
                    loc_result.matched_slot = None
                    loc_result.heading_confidence = 0.0
                    loc_result.confidence *= 0.6
                    jump_log = jump_m

        # what localization did, for output/sessions/<id>/turns.jsonl
        s.last_loc_log = session_log.localization_record(
            loc_result, sigma=v2_kwargs.get("sensor_sigma"), reachable=v2_kwargs.get("reachable"),
            server_evidence=v2_kwargs.get("server_evidence"), aisle_evidence=v2_kwargs.get("aisle_evidence"),
            rerank=rerank_log)
        if jump_log is not None:
            s.last_loc_log["spatial_jump_penalty_m"] = round(jump_log, 2)

        if loc_result.matched_nid is not None:
            s.last_corrected_nid = loc_result.matched_nid
            if loc_result.matched_heading is not None:
                s.user_heading = loc_result.matched_heading
                s.heading_slot = loc_result.matched_slot
                s.heading_confidence = loc_result.heading_confidence
            log.info("[session %s] localization: nid=%d conf=%.2f heading=%.0f° slot=%s (%s)",
                     session_id, loc_result.matched_nid,
                     loc_result.confidence,
                     loc_result.matched_heading or 0,
                     loc_result.matched_slot or "?",
                     loc_result.reasoning)
        else:
            log.info("[session %s] localization: no confident match (%.2f)",
                     session_id, loc_result.confidence)
        return loc_result
    except Exception:
        log.exception("[session %s] localization error", session_id)
        return None


def _image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (width, height)


def _top_goal_detections(detections, goal_objects):
    """Best-scoring detection *per distinct goal object it matches*, not just one
    overall pick — a multi-item goal (e.g. "milk x1, bread x2, eggs x1") needs each
    matched item verified against its own specific label, not the whole combined
    goal string, or verification becomes meaningless once more than one product is
    being searched for at once. Returns a list of (goal_label, detection) pairs."""
    best_by_goal: dict[str, object] = {}
    for d in detections:
        for g in goal_objects:
            if not g:
                continue
            label_l = d.label.lower()
            g_l = g.lower()
            if g_l in label_l or label_l in g_l:
                current = best_by_goal.get(g)
                if current is None or d.score > current.score:
                    best_by_goal[g] = d
                break
    return list(best_by_goal.items())


@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        body = ErrorResponse(
            error=exc.detail.get("error", "error"),
            detail=exc.detail.get("detail", ""),
        ).model_dump()
    else:
        body = ErrorResponse(error="error", detail=str(exc.detail)).model_dump()
    return JSONResponse(status_code=exc.status_code, content=body)


@app.get("/places")
def list_places():
    """Return all available map places from Neo4j."""
    neo4j = get_neo4j()
    if neo4j is None:
        # Fall back: return the default place from config
        from server.config import NEO4J_PLACE
        if NEO4J_PLACE:
            return {"places": [{"name": NEO4J_PLACE, "photo_count": 0}]}
        return {"places": []}
    places = neo4j.list_places()
    return {"places": places}


@app.post("/session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    resp = _handle_start_session(req)
    s = _store.get(resp.session_id)
    if s is not None:
        out_dir = OUTPUT_ROOT / s.id
        session_log.append_turn(out_dir, {"kind": "start", "request": {"goal": req.goal, "place": req.place},
                                          "response": {"guidance": resp.guidance, "goal_objects": resp.goal_objects,
                                                       "place": resp.place, "goal_node_id": resp.goal_node_id}})
        session_log.write_session(out_dir, s)
    return resp


def _handle_start_session(req: StartSessionRequest) -> StartSessionResponse:
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail={"error": "bad_request", "detail": "goal is empty"})
    goal_objects = decompose_goal(req.goal)
    goal_objects = augment_goal_classes(req.goal, goal_objects)
    # 過濾掉和通用地標重複的詞，避免 shelf/aisle 等每張照片都標紅
    # 但如果該詞是由 GOAL_CLASS_MAP 映射加入的（與目標直接相關），則保留
    from server.graph_renderer import GOAL_CLASS_MAP
    _goal_mapped: set[str] = set()
    for key, info in GOAL_CLASS_MAP.items():
        if key in req.goal:
            _goal_mapped.update(c.lower() for c in info["classes"])
    _generic_lower = {g.lower() for g in GENERIC_INDOOR_OBJECTS}
    goal_objects = [
        g for g in goal_objects
        if g.lower() not in _generic_lower or g.lower() in _goal_mapped
    ]
    # Product words vs. section/landmark context: only the former may ever
    # count as "the target is in view" (arrival gate, crop-verify, photo hint).
    target_objects, context_objects = split_goal_objects(
        req.goal, goal_objects, extra_target_terms=_goal_mapped)
    log.info("goal split | targets: %s | context: %s", target_objects, context_objects)
    # Resolve place: empty string "" = explore mode (no reference map);
    # None/missing = fall back to .env default; otherwise use client's selection
    from server.config import NEO4J_PLACE
    if req.place == "":
        # Explicit "new map / explore" — no reference map
        selected_place = None
        log.info("New session | goal: %s | place: (explore mode) | goal_objects: %s", req.goal, goal_objects)
    else:
        selected_place = req.place or NEO4J_PLACE or None
        log.info("New session | goal: %s | place: %s | goal_objects: %s", req.goal, selected_place, goal_objects)
    s = _store.create(goal=req.goal, goal_objects=goal_objects, place=selected_place,
                      target_objects=target_objects, context_objects=context_objects)
    s.arrival_goal_objects = _classify_arrival_objects(req.goal, goal_objects)
    log.info("arrival_goal_objects: %s", s.arrival_goal_objects)

    # Generate goal graph
    out_dir = ensure_output_dir(s.id)
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    render_goal_graph(req.goal, goal_objects, str(graph_dir / "goal_graph.png"))
    log.info("Goal graph saved to %s", graph_dir / "goal_graph.png")

    # ── Auto route planning ───────────────────────────────────────
    # Search the Neo4j reference map for goal items by matching OCR
    # text and object labels, then plan an optimal route.
    auto_route = None
    if selected_place:
        try:
            neo4j = get_neo4j()
            if neo4j is None:
                log.warning("🔴 [session %s] NEO4J NOT CONNECTED — all map/route features disabled", s.id)
            else:
                log.info("🟢 [session %s] Neo4j connected, loading ref_map for '%s'", s.id, selected_place)
            ref_map = neo4j.load_reference_map(selected_place) if neo4j else None
            if ref_map and ref_map.photos:
                log.info("🟢 [session %s] ref_map loaded: %d nodes, %d edges",
                         s.id, len(ref_map.photos), len(ref_map.walkway_edges))
            elif ref_map:
                log.warning("🔴 [session %s] ref_map loaded but EMPTY (0 photo nodes) for '%s'", s.id, selected_place)
            else:
                log.warning("🔴 [session %s] ref_map is None for '%s'", s.id, selected_place)
            if ref_map and ref_map.photos:
                # Extract individual goal items from the goal string
                import re as _re_auto
                goal_items = [g.strip() for g in req.goal.split(",")]
                # Strip quantity suffixes like " x1", " x2"
                goal_items = [_re_auto.sub(r'\s*x\d+\s*$', '', g).strip() for g in goal_items]
                goal_items = [g for g in goal_items if g]

                # Synonym groups for fuzzy goal matching
                _GOAL_SYNONYMS = {
                    "文旦": ["柚子", "pomelo", "grapefruit", "文旦柚"],
                    "柚子": ["文旦", "pomelo", "grapefruit"],
                    "優格": ["yogurt", "yoghurt", "優酪乳", "酸奶"],
                    "可樂": ["cola", "coke", "coca-cola", "百事可樂", "pepsi"],
                    "牛奶": ["鮮奶", "milk", "鮮乳"],
                    "鮮奶": ["牛奶", "milk", "鮮乳"],
                    "泡麵": ["方便麵", "instant noodle", "杯麵", "碗麵"],
                    "餅乾": ["cookie", "biscuit", "cracker"],
                    "洋芋片": ["薯片", "potato chips", "chips"],
                    "啤酒": ["beer", "台啤", "海尼根", "heineken"],
                    "醬油": ["soy sauce", "醬料"],
                    "麵包": ["bread", "吐司", "toast"],
                    "衛生紙": ["toilet paper", "面紙", "tissue"],
                }

                def _expand_search_terms(item: str) -> list[str]:
                    terms = [item]
                    for key, synonyms in _GOAL_SYNONYMS.items():
                        if item == key or item.lower() == key.lower():
                            terms.extend(synonyms)
                        elif key in item or item in key:
                            terms.extend(synonyms)
                    return terms

                # Search Neo4j map: match each goal item against node OCR/objects
                target_nodes: list[int] = []
                for item in goal_items:
                    search_terms = _expand_search_terms(item)
                    best_nid = None
                    best_key = (0.0, 0.0)
                    for nid, node in ref_map.photos.items():
                        key = _goal_node_score(node.objects, search_terms)
                        if key > best_key:
                            best_key = key
                            best_nid = nid
                    best_score = best_key[1]
                    if best_nid is not None and best_score >= 0.5:
                        if best_nid not in target_nodes:
                            target_nodes.append(best_nid)
                        s.target_node_to_item[best_nid] = item
                        log.info("[session %s] Goal '%s' → node #%d (score=%.2f)",
                                 s.id, item, best_nid, best_score)
                    else:
                        log.info("[session %s] Goal '%s' → no match in Neo4j map", s.id, item)

                # Also try matching decomposed goal_objects if direct items found nothing
                if not target_nodes and goal_objects:
                    for item in goal_objects:
                        item_lower = item.lower()
                        best_nid = None
                        best_score = 0.0
                        for nid, node in ref_map.photos.items():
                            for obj in node.objects:
                                score = 0.0
                                if obj.ocr_text:
                                    ocr_lower = obj.ocr_text.lower()
                                    if item_lower == ocr_lower:
                                        score = 1.0
                                    elif item_lower in ocr_lower or ocr_lower in item_lower:
                                        score = 0.8
                                label_lower = obj.label.lower()
                                if item_lower in label_lower or label_lower in item_lower:
                                    score = max(score, 0.7)
                                if score > best_score:
                                    best_score = score
                                    best_nid = nid
                        if best_nid is not None and best_score >= 0.5:
                            if best_nid not in target_nodes:
                                target_nodes.append(best_nid)
                            log.info("[session %s] Goal object '%s' → node #%d (score=%.2f)",
                                     s.id, item, best_nid, best_score)

                if not target_nodes:
                    log.warning("🔴 [session %s] NO goal items matched any Neo4j node — route planning SKIPPED", s.id)
                if target_nodes:
                    log.info("🟢 [session %s] %d goal items matched nodes: %s", s.id, len(target_nodes), target_nodes)
                    # Plan on the route graph the mini-map and the text steps use
                    from server.path_planner import plan_route as _plan_route
                    neo4j_graph = _route_nx_graph(selected_place)
                    if neo4j_graph is not None and neo4j_graph.number_of_nodes() > 0:
                        # Auto-detect checkout and exit nodes (need exit_nid for default start)
                        checkout_nid, exit_nid = _find_checkout_exit_nodes(ref_map)
                        if checkout_nid and checkout_nid not in neo4j_graph:
                            checkout_nid = None
                        if exit_nid and exit_nid not in neo4j_graph:
                            exit_nid = None

                        # From the entrance (or the first start point) until a chosen start
                        # point or the first photo fix re-plans from where the user is
                        start_nid = _default_start_wp(selected_place)
                        # Filter targets to only nodes that exist in graph
                        valid_targets = [t for t in target_nodes if t in neo4j_graph]

                        s.checkout_node = checkout_nid
                        s.exit_node = exit_nid

                        if valid_targets and start_nid is not None:
                            route = _plan_route(
                                neo4j_graph, start_nid, valid_targets,
                                checkout=checkout_nid, exit_node=exit_nid,
                            )
                            s.target_nodes = valid_targets
                            s.visited_targets = set()
                            s.route_plan = route
                            s.current_leg_index = 0
                            # Set goal_node to the first target so the AR arrow works immediately
                            if route.visit_order:
                                s.goal_node = route.visit_order[0]
                            auto_route = {
                                "visit_order": route.visit_order,
                                "total_cost": route.total_cost,
                                "legs": len(route.legs),
                                "checkout_node": checkout_nid,
                                "exit_node": exit_nid,
                            }
                            log.info("[session %s] Auto route: %d targets, cost=%.1f, order=%s, checkout=WP%s, exit=WP%s",
                                     s.id, len(valid_targets), route.total_cost, route.visit_order,
                                     checkout_nid, exit_nid)
        except Exception as e:
            log.warning("[session %s] Auto route planning failed: %s", s.id, e)

    north_offset = start_points.load_north_offset(selected_place, EDITOR_MAP_DIR) if selected_place else None
    return StartSessionResponse(
        session_id=s.id,
        guidance="Upload a starting photo so I can see where you are.",
        goal_objects=goal_objects,
        place=selected_place,
        goal_node_id=_current_goal_node_id(s),
        map_north_offset_deg=north_offset,
    )


@app.post("/session/{session_id}/photo", response_model=TurnResponse)
async def upload_photo(
    session_id: str,
    photo: UploadFile = File(...),
    dino_node_hint: Optional[int] = Form(None),
    dino_topk: Optional[str] = Form(None),
    pdr_x: Optional[float] = Form(None),
    pdr_y: Optional[float] = Form(None),
    map_x: Optional[float] = Form(None),
    map_y: Optional[float] = Form(None),
    dist_since_fix: Optional[float] = Form(None),
    fix_node: Optional[int] = Form(None),
    heading: Optional[float] = Form(None),
    source: Optional[str] = Form(None),
) -> TurnResponse:
    """Handle one navigation photo, then record the turn in output/sessions/<id>/."""
    s = _store.get(session_id)
    if s is not None:
        s.last_loc_log = None
    resp = await _handle_photo(session_id, photo, dino_node_hint, dino_topk, pdr_x, pdr_y, map_x, map_y,
                               dist_since_fix, fix_node, heading, source)
    if s is not None:
        out_dir = OUTPUT_ROOT / session_id
        session_log.append_turn(out_dir, {
            "kind": "photo",
            "request": {"source": source or "camera", "pdr_x": pdr_x, "pdr_y": pdr_y, "map_x": map_x, "map_y": map_y,
                        "dist_since_fix": dist_since_fix, "fix_node": fix_node, "heading": heading,
                        "dino_node_hint": dino_node_hint, "dino_topk": dino_topk},
            "localization": s.last_loc_log,
            "response": session_log.response_record(resp),
        })
        session_log.write_session(out_dir, s)
    return resp


async def _handle_photo(
    session_id: str,
    photo: UploadFile = File(...),
    dino_node_hint: Optional[int] = Form(None),
    dino_topk: Optional[str] = Form(None),
    pdr_x: Optional[float] = Form(None),
    pdr_y: Optional[float] = Form(None),
    map_x: Optional[float] = Form(None),
    map_y: Optional[float] = Form(None),
    dist_since_fix: Optional[float] = Form(None),
    fix_node: Optional[int] = Form(None),
    heading: Optional[float] = Form(None),
    source: Optional[str] = Form(None),
) -> TurnResponse:
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if s.arrived:
        raise HTTPException(status_code=409, detail={"error": "already_arrived", "detail": "session is complete"})
    if s.pending_arrival:
        raise HTTPException(status_code=409, detail={"error": "confirm_pending", "detail": "POST /confirm first"})
    if s.pending_question is not None:
        raise HTTPException(status_code=409, detail={"error": "answer_pending", "detail": "POST /answer first"})

    if dino_node_hint is not None or pdr_x is not None:
        log.info("[session %s] iOS sensor data: dino_hint=%s pdr=(%.1f, %.1f) heading=%s",
                 session_id, dino_node_hint,
                 pdr_x or 0, pdr_y or 0,
                 f"{heading:.1f}" if heading is not None else "N/A")
    if source == "gallery":
        # A gallery photo was taken some other time: the live heading / position
        # describe where the phone is now, not where the photo was taken.
        heading = map_x = map_y = dist_since_fix = fix_node = pdr_x = pdr_y = None
    if not _finite(heading):
        heading = None
    _phone_loc = _phone_localization_inputs(dino_topk, dino_node_hint, map_x, map_y,
                                            dist_since_fix, fix_node)
    _prev_corrected_nid = s.last_corrected_nid
    # A second photo asked for by the previous turn: combined only when we KNOW the user
    # stayed put (both photos carry the phone's step-count position); unknown movement
    # counts as "moved". The request is kept until a photo actually gets localized, so a
    # blurred / blocked retake doesn't lose it.
    _second_view = None
    if s.pending_view is not None:
        pv = s.pending_view
        known = _finite(pdr_x) and _finite(pdr_y) and pv.get("raw") is not None
        moved = math.hypot(pdr_x - pv["raw"][0], pdr_y - pv["raw"][1]) if known else float("inf")
        if time.time() - pv["t"] > _config.SECOND_PHOTO_WINDOW_S or moved > _config.SECOND_PHOTO_MAX_MOVE_M:
            s.pending_view = None
        else:
            _second_view = pv["evidence"]
            log.info("[session %s] combining with the previous photo (moved %.1f m)", session_id, moved)
    _phone_loc["extra_evidence"] = _second_view
    if _phone_loc["dino_sims"] or _phone_loc["sensor_pos"]:
        log.info("[session %s] phone localization: dino_topk=%s map_pos=%s dist_since_fix=%s",
                 session_id,
                 sorted(_phone_loc["dino_sims"].items(), key=lambda kv: -kv[1])[:5],
                 _phone_loc["sensor_pos"], _phone_loc["dist_since_fix"])

    t_start = time.time()
    out_dir = ensure_output_dir(session_id)
    photo_bytes = await photo.read()
    nid_for_path = s.topomap.graph.number_of_nodes()
    photo_path = out_dir / "photo" / f"{nid_for_path}.jpg"
    photo_path.write_bytes(photo_bytes)

    # Fix rotation from EXIF and resize if needed
    im = Image.open(photo_path)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    img_w, img_h = im.size
    resized = max(img_w, img_h) > 1600
    if resized:
        im.thumbnail((1600, 1600))
        img_w, img_h = im.size
    im.save(photo_path, format="JPEG", quality=85)
    log.info("Photo %s %dx%d", "resized" if resized else "saved", img_w, img_h)
    im.close()

    # Blurred / dark photos can't be localized — ask for a retake before spending
    # any detector or VLM time on them.
    _quality = photo_quality.assess_image(str(photo_path))
    log.info("[session %s] photo quality: sharpness=%.0f brightness=%.0f",
             session_id, _quality.sharpness, _quality.brightness)
    if _quality.hint:
        log.info("[session %s] photo rejected: %s", session_id, _quality.hint)
        _g_items, _v_items, _c_target = _shopping_progress(s)
        return TurnResponse(
            action=VLMAction.MOVE, guidance=_quality.hint, retake_hint=_quality.hint,
            node_id=-1, phase=s.phase, goal_item_names=_g_items,
            visited_item_names=_v_items, current_target_name=_c_target,
            goal_node_id=_current_goal_node_id(s),
        )

    t_resize = time.time()

    perception_engine = get_perception()
    ocr_engine = get_ocr()
    has_grounding = perception_engine is not None

    # ── 先跑 EasyOCR（如果啟用）──────────────────────────────────
    easyocr_results = []
    if ocr_engine is not None:
        easyocr_results = ocr_engine.read(
            str(photo_path),
            min_confidence=OCR_MIN_CONFIDENCE,
            max_results=OCR_MAX_RESULTS,
        )
        if easyocr_results:
            log.info("EasyOCR found %d text regions", len(easyocr_results))
    t_ocr = time.time()
    log.info("⏱ EasyOCR: %.1fs", t_ocr - t_resize)

    if has_grounding:
        # ══ 模式 A：GroundingDINO + (EasyOCR) + 單階段 VLM ══
        combined_prompts = list(dict.fromkeys(s.goal_objects + GENERIC_INDOOR_OBJECTS + OCCLUDER_PROMPTS))
        log.info("GroundingDINO prompt: %s", combined_prompts)
        detections = perception_engine.detect(str(photo_path), combined_prompts)
        detected_labels = [d.label for d in detections]
        t_detect = time.time()
        log.info("⏱ GroundingDINO: %.1fs", t_detect - t_ocr)

        ocr_results = easyocr_results
        ocr_text_summary = scene.format_ocr(ocr_results, img_w, img_h) if ocr_results else "(no text detected)"

        ocr_matches = scene.match_ocr_to_goal(ocr_results, s.target_objects)
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

        detections_summary = scene.format_detections(detections, img_w, img_h)
        topomap_summary = s.topomap.summarize_for_vlm(current_id=nid)

        log.info("[session %s] photo #%d | detections: %s | ocr: %s",
                 session_id, nid, detected_labels or "(none)", ocr_text_summary[:80])

        # ── Early localization for route-aware VLM prompt ──────────────
        _loc_items = _build_localization_items(detections, ocr_results or [], img_w, img_h)
        _ocr_items = _build_ocr_items(ocr_results or [], img_w, img_h)
        # A blocked view must not move the user, re-plan the route or set the heading.
        _occlusion = photo_quality.assess_occlusion(detections, img_w, img_h, goals=s.goal_objects)
        _early_loc = None if _occlusion.hint else _run_early_localization(
            s, session_id, detected_labels, ocr_texts,
            detected_items=_loc_items,
            image_path=str(photo_path),
            ocr_items=_ocr_items,
            sign_numbers=aisle_signs.numbers_from_ocr(ocr_results, img_w, img_h),
            **_phone_loc)
        _apply_phone_heading(s, _early_loc, heading)
        _route_ctx, _next_instr = _build_route_context(s, _early_loc, detections=detections,
                                                       img_w=img_w, img_h=img_h, ocr_results=ocr_results,
                                                       is_first_localization=_prev_corrected_nid is None)

        if _route_ctx:
            log.info("🟢 [session %s] VLM will receive route context (%d chars), next_instr=%s",
                     session_id, len(_route_ctx), _next_instr[:80] if _next_instr else "(none)")
        else:
            log.warning("🔴 [session %s] VLM will receive NO route context — navigating blind", session_id)

        t_vlm_start = time.time()
        _active_goals = _current_goal_objects(s)
        _active_goal_str = "找到：" + "、".join(_active_goals)
        vlm_resp = vlm_decide(
            image_path=str(photo_path),
            goal=_active_goal_str,
            goal_objects=_active_goals,
            topomap_summary=topomap_summary,
            detections_summary=detections_summary,
            prior_question=None,
            prior_answer=None,
            ocr_summary=ocr_text_summary,
            route_context=_route_ctx,
            context_objects=s.context_objects,
        )
        t_vlm = time.time()
        log.info("[session %s] VLM → %s | %s", session_id, vlm_resp.action.value, vlm_resp.guidance[:120])
        log.info("⏱ VLM: %.1fs | Total: %.1fs", t_vlm - t_vlm_start, t_vlm - t_start)

    else:
        # ══ 模式 B/C：兩階段 VLM（可搭配 EasyOCR）══
        mode_label = "two-stage VLM + EasyOCR" if easyocr_results else "two-stage VLM only"
        log.info("[session %s] Using %s", session_id, mode_label)

        nid = s.topomap.add_node(
            photo_path=str(photo_path), detected=[],
            summary="", ocr_texts=[], ocr_with_conf=[],
        )
        if s.last_node_id is not None:
            s.topomap.add_edge(s.last_node_id, nid, action=s.last_planned_action or "(unknown)")

        topomap_summary = s.topomap.summarize_for_vlm(current_id=nid)

        def _ocr_formatter(vlm_ocr_items):
            # EasyOCR 可用時，完全用 EasyOCR 結果取代 VLM OCR（避免幻覺）
            if easyocr_results:
                if not easyocr_results:
                    return "(no text detected)"
                summary = scene.format_ocr(easyocr_results, img_w, img_h)
                matches = scene.match_ocr_to_goal(easyocr_results, s.target_objects)
                if matches:
                    summary += "  SIGN MATCH: " + "; ".join(matches)
                return summary
            raw = [OCRResult(text=t.text, confidence=t.score, bbox=t.bbox)
                   for t in vlm_ocr_items]
            cleaned = _clean_vlm_ocr(raw, img_h)
            if not cleaned:
                return "(no text detected)"
            summary = scene.format_ocr(cleaned, img_w, img_h)
            matches = scene.match_ocr_to_goal(cleaned, s.target_objects)
            if matches:
                summary += "  SIGN MATCH: " + "; ".join(matches)
            return summary

        # ── Stage 1: VLM perceive (multi-round if VLM_ROUNDS > 1) ────────
        _active_goals = _current_goal_objects(s)
        _active_goal_str = "找到：" + "、".join(_active_goals)
        t_vlm_start = time.time()
        if _config.VLM_ROUNDS > 1:
            vlm_perception, _perception_debug = _vlm_perceive_multi_round(
                image_path=str(photo_path),
                goal=_active_goal_str,
                goal_objects=_active_goals,
                img_w=img_w,
                img_h=img_h,
                num_rounds=_config.VLM_ROUNDS,
            )
        elif _config.VLM_PASS3:
            vlm_perception, _perception_debug = _vlm_perceive_three_pass(
                image_path=str(photo_path),
                goal=_active_goal_str,
                goal_objects=_active_goals,
                img_w=img_w,
                img_h=img_h,
            )
        else:
            vlm_perception, _perception_debug = _vlm_perceive_two_pass(
                image_path=str(photo_path),
                goal=_active_goal_str,
                goal_objects=_active_goals,
                img_w=img_w,
                img_h=img_h,
            )

        detections = [
            PerceptionDetection(label=d.label, box=d.bbox, score=d.score,
                                position=d.position)
            for d in vlm_perception.detections
        ]
        detected_labels = [d.label for d in detections]

        # ── Generate pass1 / pass2 / merged annotated images ──
        _pass1_dets = [
            PerceptionDetection(label=d["label"], box=d["box"], score=d["score"],
                                position=d.get("position"))
            for d in _perception_debug.get("pass1", {}).get("detections", [])
            if d.get("box") and len(d["box"]) == 4
        ]
        _pass2_dets = [
            PerceptionDetection(label=d["label"], box=d["box"], score=d["score"],
                                position=d.get("position"))
            for d in _perception_debug.get("pass2", {}).get("detections", [])
            if d.get("box") and len(d["box"]) == 4
        ]
        _pass1_ocr = [
            OCRResult(text=t["text"], confidence=t.get("score", 0.5), bbox=t["bbox"])
            for t in _perception_debug.get("pass1", {}).get("ocr", [])
            if t.get("bbox")
        ]
        _pass2_ocr = [
            OCRResult(text=t["text"], confidence=t.get("score", 0.5), bbox=t["bbox"])
            for t in _perception_debug.get("pass2", {}).get("ocr", [])
            if t.get("bbox")
        ]
        _ann_dir = ensure_output_dir(session_id) / "annotated"
        if _pass1_dets or _pass1_ocr:
            annotate(str(photo_path), str(_ann_dir / f"{nid}_pass1.jpg"),
                     _pass1_dets,
                     banner_text=f"Pass 1: {len(_pass1_dets)} detections, {len(_pass1_ocr)} OCR",
                     ocr_results=_pass1_ocr if _pass1_ocr else None,
                     goal_objects=s.goal_objects, region_mode=True)
        if _pass2_dets or _pass2_ocr:
            annotate(str(photo_path), str(_ann_dir / f"{nid}_pass2.jpg"),
                     _pass2_dets,
                     banner_text=f"Pass 2: {len(_pass2_dets)} detections, {len(_pass2_ocr)} OCR",
                     ocr_results=_pass2_ocr if _pass2_ocr else None,
                     goal_objects=s.goal_objects, region_mode=True)

        vlm_ocr_results = [
            OCRResult(text=t.text, confidence=t.score, bbox=t.bbox)
            for t in vlm_perception.ocr_texts
        ]

        if easyocr_results:
            ocr_results = easyocr_results
        else:
            ocr_results = vlm_ocr_results
        ocr_results = _clean_vlm_ocr(ocr_results, img_h)
        ocr_texts = [r.text for r in ocr_results]

        # ── Localize with detection labels + OCR + zone BEFORE decide ──
        _is_first_localization = s.last_corrected_nid is None
        # Localization matches 「中文 English」 names (the map's label format); the rest of
        # the turn keeps the VLM's 中文 label.
        _loc_dets = [PerceptionDetection(label=_vlm_localization_label(v), box=d.box, score=d.score,
                                         position=d.position)
                     for v, d in zip(vlm_perception.detections, detections)]
        _loc_items = _build_localization_items(_loc_dets, ocr_results, img_w, img_h)
        _ocr_items = _build_ocr_items(ocr_results, img_w, img_h)
        _store_zone = vlm_perception.store_zone or ""
        log.info("[session %s] VLM store_zone: %s | scene: %s",
                 session_id, _store_zone or "(none)", vlm_perception.scene_description[:80])
        # A blocked view must not move the user, re-plan the route or set the heading.
        _occlusion = photo_quality.assess_occlusion(detections, img_w, img_h, goals=s.goal_objects)
        _early_loc = None if _occlusion.hint else _run_early_localization(
            s, session_id, [d.label for d in _loc_dets], ocr_texts,
            detected_items=_loc_items,
            image_path=str(photo_path),
            store_zone=_store_zone,
            ocr_items=_ocr_items,
            sign_numbers=aisle_signs.numbers_from_ocr(ocr_results, img_w, img_h),
            **_phone_loc)
        _apply_phone_heading(s, _early_loc, heading)
        _route_ctx, _next_instr = _build_route_context(s, _early_loc, detections=detections,
                                                       img_w=img_w, img_h=img_h, ocr_results=ocr_results,
                                                       is_first_localization=_prev_corrected_nid is None)

        # ── Store debug info ──
        _loc_debug = {}
        if _early_loc:
            _loc_debug = {
                "matched_nid": _early_loc.matched_nid,
                "confidence": round(_early_loc.confidence, 3),
                "method": _early_loc.method,
                "reasoning": _early_loc.reasoning,
                "runner_up_nid": _early_loc.runner_up_nid,
                "runner_up_score": round(_early_loc.runner_up_score, 3),
                "top_candidates": [(c[0], round(c[1], 3), c[2]) for c in (_early_loc.top_candidates or [])[:10]],
            }
        s.debug_perception = {
            **_perception_debug,
            "localization": _loc_debug,
            "store_zone": _store_zone,
            "photo_turn": len(s.history),
            "images": {
                "pass1": f"/session/{session_id}/photo/{nid}/pass1.jpg",
                "pass2": f"/session/{session_id}/photo/{nid}/pass2.jpg",
                "merged": f"/session/{session_id}/photo/{nid}.jpg",
                "original": f"/session/{session_id}/original/{nid}.jpg",
            },
        }

        if _route_ctx:
            log.info("🟢 [session %s] Mode B: VLM will receive route context (%d chars), next_instr=%s",
                     session_id, len(_route_ctx), _next_instr[:80] if _next_instr else "(none)")
        else:
            log.warning("🔴 [session %s] Mode B: VLM will receive NO route context — navigating blind", session_id)

        # ── Stage 2: VLM decide (with map-aware route context) ────────
        ocr_text_summary = scene.format_ocr(ocr_results, img_w, img_h) if ocr_results else "(no text detected)"
        ocr_matches = scene.match_ocr_to_goal(ocr_results, s.target_objects)
        if ocr_matches:
            ocr_text_summary += "  SIGN MATCH: " + "; ".join(ocr_matches)

        def _ocr_formatter_final(vlm_ocr_items):
            return ocr_text_summary

        detections_summary = scene.format_detections(
            [{"label": d.label, "box": d.box, "score": d.score} for d in detections],
            img_w, img_h,
        )
        vlm_resp = vlm_decide(
            image_path=str(photo_path),
            goal=_active_goal_str,
            goal_objects=_active_goals,
            topomap_summary=topomap_summary,
            detections_summary=detections_summary,
            prior_question=None,
            prior_answer=None,
            ocr_summary=ocr_text_summary,
            route_context=_route_ctx,
            context_objects=s.context_objects,
        )
        t_vlm = time.time()

        ocr_with_conf = [{"text": r.text, "confidence": r.confidence} for r in ocr_results]
        s.topomap.graph.nodes[nid]["detected"] = detected_labels
        s.topomap.graph.nodes[nid]["ocr_texts"] = ocr_texts
        s.topomap.graph.nodes[nid]["ocr_with_conf"] = ocr_with_conf

        det_detail = "; ".join(f"{d.label}@{d.position}" for d in detections if d.position)
        ocr_detail = "; ".join(f'"{r.text}"' for r in ocr_results if r.text)
        log.info("[session %s] VLM 2-stage → %s | scene: %s",
                 session_id, vlm_resp.action.value,
                 vlm_perception.scene_description[:80])
        log.info("  detections(%d): %s", len(detections), det_detail or "(none)")
        log.info("  OCR(%d): %s", len(ocr_results), ocr_detail or "(none)")
        log.info("  localization: nid=%s conf=%.2f | route_ctx=%s",
                 _early_loc.matched_nid if _early_loc else None,
                 _early_loc.confidence if _early_loc else 0,
                 "YES" if _route_ctx else "NO")
        log.info("⏱ VLM 2-stage: %.1fs | Total: %.1fs", t_vlm - t_vlm_start, t_vlm - t_start)

    # TASK 3 — when the VLM claims ARRIVED, crop each detection that matches a
    # distinct goal object (not just one overall top pick — see
    # _top_goal_detections) and ask the VLM to confirm each region individually.
    # This catches GroundingDINO false positives on dense shelves before we
    # declare success, and lets a multi-item goal ("milk x1, bread x2, eggs x1")
    # get more than one item confirmed from the same photo.
    goal_verified = None
    verified_labels: list[str] = []
    if s.phase in ("checkout", "exit"):
        _arrival_goals = _active_goals
    else:
        _arrival_goals = s.arrival_goal_objects or s.goal_objects
    if GOAL_CROP_VERIFY and vlm_resp.action == VLMAction.ARRIVED:
        candidates = _top_goal_detections(detections, _arrival_goals)
        if not has_grounding:
            candidates = [(g, d) for g, d in candidates if d.score >= ARRIVED_MIN_DETECTION_SCORE]
        any_checked = False
        any_true = False
        for goal_label, cand in candidates:
            try:
                with Image.open(photo_path) as im:
                    result = verify_goal_detection(
                        im.convert("RGB"), cand.box, goal_label, _vlm_ask_about_image,
                    )
                any_checked = True
                log.info("goal crop verification: %s (goal=%s, label=%s)", result, goal_label, cand.label)
                if result:
                    any_true = True
                    verified_labels.append(goal_label)
            except Exception as e:
                log.warning("goal crop verification errored (goal=%s): %s", goal_label, e)
        if any_checked:
            goal_verified = any_true

    # Gate ARRIVED on real evidence (a fresh photo can't be answering a confirm).
    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, detections, ocr_matches=ocr_matches, goal_objects=_arrival_goals,
        min_score=ARRIVED_MIN_DETECTION_SCORE, img_w=img_w, img_h=img_h,
        goal_verified=goal_verified,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action != VLMAction.ARRIVED
    if _raw_action == VLMAction.ARRIVED:
        log.info("[session %s] verify_arrival: %s → %s | goal_verified=%s ocr_matches=%s",
                 session_id, _raw_action.value, vlm_resp.action.value,
                 goal_verified, bool(ocr_matches))
    # ...and on the map: far from the goal node, ask or silently reject
    if _raw_action == VLMAction.ARRIVED:
        _far_q, _far_m = _far_from_goal_question(
            s, _early_loc.matched_nid if _early_loc else None,
            _ref_location_name(_early_loc) if _early_loc else None)
        if _far_q:
            if _far_m is not None and _far_m > ARRIVAL_ASK_MAX_M:
                log.info("[session %s] arrival too far (%.0f m) → silent MOVE", session_id, _far_m)
                vlm_resp = VLMResponse(action=VLMAction.MOVE, guidance=vlm_resp.guidance,
                                       question=None, vlm_summary=vlm_resp.vlm_summary)
            else:
                log.info("[session %s] arrival not near the goal node on the map → asking: %s", session_id, _far_q)
                vlm_resp = VLMResponse(action=VLMAction.ASK, guidance=_far_q, question=_far_q,
                                       vlm_summary=vlm_resp.vlm_summary)
            arrival_downgraded = True
    if verified_labels:
        log.info("[session %s] confirmed goal items in this photo: %s", session_id, verified_labels)

    s.topomap.graph.nodes[nid]["summary"] = vlm_resp.vlm_summary
    det_dicts = [{"label": d.label, "score": d.score, "box": list(d.box),
                  **({"position": d.position} if d.position else {})}
                 for d in detections]

    # Build OCR bbox list for door-nameplate association
    ocr_bbox_list = [
        {"text": r.text, "confidence": r.confidence,
         "bbox": [list(p) for p in r.bbox]}
        for r in (ocr_results or [])
    ]

    # Full post-processing: NMS → merge → OCR-door → spatial → entity tracking → IDs
    # VLM 模式的 bbox 是粗略區域，跳過 NMS/合併避免誤刪不同物件
    det_dicts = postprocess_detections(
        det_dicts, img_w, img_h,
        ocr_with_bbox=ocr_bbox_list or None,
        prev_dets=s.last_detections or None,
        prev_w=s.last_img_w,
        prev_h=s.last_img_h,
        skip_nms=not has_grounding,
    )

    # Persist full detection dicts in topomap node for scene graph history
    s.topomap.graph.nodes[nid]["det_dicts"] = det_dicts

    s.last_detections = det_dicts
    s.last_img_w = img_w
    s.last_img_h = img_h
    s.last_ocr_summary = ocr_text_summary
    s.last_ocr_matches = ocr_matches
    s.last_photo_path = str(photo_path)
    s.last_node_id = nid

    annotated_path = out_dir / "annotated" / f"{nid}.jpg"
    annotate(str(photo_path), str(annotated_path), detections,
             banner_text=f"{vlm_resp.action.value}: {vlm_resp.guidance}",
             ocr_results=ocr_results, goal_objects=s.goal_objects,
             region_mode=not has_grounding)

    # Save map JSON and PNG after each photo
    map_dir = out_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    map_json = s.topomap.to_dict(current_node=nid, goal_node=s.goal_node)
    (map_dir / f"map_{nid}.json").write_text(json.dumps(map_json, ensure_ascii=False, indent=2), encoding="utf-8")
    topo_now = copy.deepcopy(s.topomap)      # the next photo changes s.topomap
    _render_later(lambda: (map_dir / f"map_{nid}.png").write_bytes(topo_now.render_png(current_id=nid)))

    _loc_for_hist = _early_loc
    s.history.append({
        "kind": "photo",
        "node_id": nid,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
        "localization": {
            "matched_nid": _loc_for_hist.matched_nid if _loc_for_hist else None,
            "confidence": round(_loc_for_hist.confidence, 3) if _loc_for_hist else 0,
            "reasoning": _loc_for_hist.reasoning if _loc_for_hist else "",
            "heading": round(_loc_for_hist.matched_heading, 1) if _loc_for_hist and _loc_for_hist.matched_heading is not None else None,
            "slot": _loc_for_hist.matched_slot if _loc_for_hist else None,
        } if _loc_for_hist else None,
        "detections": [d.get("label", "") if isinstance(d, dict) else getattr(d, "label", "") for d in (s.last_detections or [])],
        "ocr_texts": [t for t in (ocr_texts if ocr_texts else [])],
    })

    # Build observations list and render cumulative scene graph
    observations = _build_observations(s)
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    _render_later(render_scene_graph, observations, s.goal, list(s.goal_objects),
                  str(graph_dir / f"scene_graph_{nid}.png"))

    # Save detection + OCR results as JSON
    det_json = {
        "node_id": nid,
        "goal": s.goal,
        "detections": [
            {
                "id": d.get("id", d["label"]),
                "label": d["label"],
                "score": round(d["score"], 4),
                "box": d["box"],
                **({"context": d["context"]} if d.get("context") else {}),
                **({"nameplate_text": d["nameplate_text"]} if d.get("nameplate_text") else {}),
                **({"same_entity": True} if d.get("same_entity") else {}),
                **({"position": d["position"]} if d.get("position") else {}),
            }
            for d in det_dicts
        ],
        "ocr": [
            {"text": r.text, "confidence": round(r.confidence, 4),
             "bbox": [list(p) for p in r.bbox]}
            for r in (ocr_results or [])
        ],
    }
    (out_dir / "annotated" / f"detections_{nid}.json").write_text(
        json.dumps(det_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Final localization result ──────────────────────────────────────
    # Mode B already ran localization with detection labels + OCR before
    # the decide step. Mode A ran it with GroundingDINO labels earlier.
    loc_result = _early_loc

    # Occluded view (people / carts in front of the camera) or almost nothing
    # recognisable: localization was skipped above, so this photo moved nothing.
    _retake_hint = _occlusion.hint
    if _retake_hint:
        log.info("[session %s] retake: %s (occluded %.0f%%, %d useful objects) — "
                 "localization not applied", session_id, _retake_hint,
                 100 * _occlusion.fraction, _occlusion.useful_objects)
        loc_result = None
        # say why there is no map step this time, not "this photo doesn't match the map"
        _next_instr = _retake_hint
    elif _second_view is not None:
        s.pending_view = None            # the requested second photo has been used

    # Several places look alike: ask for one more photo from the same spot and combine.
    # Only while walking (an ASK / ARRIVED turn blocks the next /photo) and only when
    # the phone's position is known, so the next photo can be checked for "same spot".
    _second_photo_hint = None
    if (loc_result is not None and loc_result.image_evidence is not None and _second_view is None
            and vlm_resp.action == VLMAction.MOVE and source != "gallery"
            and _finite(pdr_x) and _finite(pdr_y)
            and loc_result.confidence < _config.SECOND_PHOTO_BELOW):
        s.pending_view = {"evidence": loc_result.image_evidence, "t": time.time(), "raw": (pdr_x, pdr_y)}
        _second_photo_hint = "這附近有幾個地方很像，請原地向左轉 90° 再拍一張，兩張一起判斷"
        log.info("[session %s] unsure (p=%.2f) — asking for a second photo", session_id, loc_result.confidence)

    # VLM already receives route_context (map + path plan) in its prompt,
    # so its guidance is photo-aware AND route-aware.  No override needed.
    _final_guidance = vlm_resp.guidance
    _route_step = _next_instr if _next_instr != LOCALIZE_FAILED_HINT else None
    if _retake_hint and vlm_resp.action == VLMAction.MOVE:
        _final_guidance = _retake_hint
    elif vlm_resp.action == VLMAction.MOVE and _route_step:
        _final_guidance = _route_step
        log.info("[session %s] route-first: overriding VLM guidance with next_instruction", session_id)
    elif vlm_resp.action == VLMAction.MOVE and not _route_step:
        log.info("[session %s] route-first: no next_instruction, using VLM guidance as fallback", session_id)

    if vlm_resp.action == VLMAction.ARRIVED:
        s.pending_arrival = True
        s.goal_node = nid
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
        s.pending_is_confirm = arrival_downgraded
        s.pending_next_instruction = _next_instr
    else:  # MOVE
        s.last_planned_action = _final_guidance

    _g_items, _v_items, _c_target = _shopping_progress(s)
    return TurnResponse(
        action=vlm_resp.action,
        guidance=_final_guidance,
        question=vlm_resp.question,
        confirm_question=vlm_resp.action == VLMAction.ASK and arrival_downgraded,
        node_id=nid,
        annotated_photo_url=f"/session/{session_id}/photo/{nid}.jpg",
        retake_hint=_retake_hint,
        second_photo_hint=_second_photo_hint,
        corrected_node_id=loc_result.matched_nid if loc_result else None,
        corrected_confidence=round(loc_result.confidence, 3) if loc_result else None,
        corrected_location=_ref_location_name(loc_result) if loc_result else None,
        estimated_x=round(loc_result.estimated_xy[0], 2) if loc_result and loc_result.estimated_xy else None,
        estimated_y=round(loc_result.estimated_xy[1], 2) if loc_result and loc_result.estimated_xy else None,
        heading_deg=round(loc_result.matched_heading, 1) if loc_result and loc_result.matched_heading is not None else None,
        heading_slot=loc_result.matched_slot if loc_result else None,
        heading_confidence=round(loc_result.heading_confidence, 3) if loc_result else None,
        next_instruction=_next_instr,
        remaining_targets=len(s.remaining_targets) if s.target_nodes else None,
        phase=s.phase,
        goal_item_names=_g_items,
        visited_item_names=_v_items,
        current_target_name=_c_target,
        goal_node_id=_current_goal_node_id(s),
    )


@app.post("/session/{session_id}/answer", response_model=TurnResponse)
def post_answer(session_id: str, req: AnswerRequest) -> TurnResponse:
    resp = _handle_answer(session_id, req)
    _log_turn(session_id, "answer", {"answer": req.answer}, resp)
    return resp


def _log_turn(session_id: str, kind: str, request: dict, resp) -> None:
    """Record a non-photo turn (/answer, /confirm) in output/sessions/<id>/."""
    s = _store.get(session_id)
    if s is None:
        return
    out_dir = OUTPUT_ROOT / session_id
    session_log.append_turn(out_dir, {"kind": kind, "request": request,
                                      "response": session_log.response_record(resp)})
    session_log.write_session(out_dir, s)


def _handle_answer(session_id: str, req: AnswerRequest) -> TurnResponse:
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if s.pending_question is None:
        raise HTTPException(status_code=409, detail={"error": "no_question_pending", "detail": "no question is open"})
    if s.last_photo_path is None or s.last_node_id is None:
        raise HTTPException(status_code=409, detail={"error": "no_prior_photo", "detail": "answer requires a prior photo"})

    img_w, img_h = _image_size(s.last_photo_path)
    detections_summary = scene.format_detections(s.last_detections, img_w, img_h)
    topomap_summary = s.topomap.summarize_for_vlm(current_id=s.last_node_id)
    prior_question = s.pending_question
    log.info("[session %s] answer: Q=%s A=%s", session_id, prior_question[:60], req.answer[:60])

    # If this is answering a confirm-arrival question, handle directly
    # without re-asking VLM — user says yes/no, that's the answer.
    if s.pending_is_confirm:
        nid = s.last_node_id
        _yes = _confirm_answer(req.answer)
        if _yes is None:
            log.info("[session %s] Unclear reply to an arrival question → asking again", session_id)
            vlm_resp = VLMResponse(
                action=VLMAction.ASK,
                guidance=f"請回答「有」或「沒有」：{prior_question}",
                question=prior_question,
                vlm_summary="unclear reply to arrival question",
            )
        elif not _yes:
            log.info("[session %s] User denied confirm → continue MOVE", session_id)
            _saved_instr = s.pending_next_instruction
            s.pending_question = None
            s.pending_is_confirm = False
            s.pending_next_instruction = None
            if _saved_instr:
                _deny_guidance = f"好的，{_saved_instr}，請往前走拍另一張照片。"
            else:
                _active_goals = _current_goal_objects(s)
                _deny_guidance = f"好的，繼續找「{'、'.join(_active_goals)}」，請往前走拍另一張照片。"
            vlm_resp = VLMResponse(
                action=VLMAction.MOVE,
                guidance=_deny_guidance,
                question=None,
                vlm_summary="user denied arrival",
            )
        else:
            # 「有」 already is the arrival confirmation: finish it exactly like
            # POST /confirm "confirmed" (one route advance, then checkout / exit /
            # done), so the app does not ask "確認到達" a second time. This used to
            # advance the route here and still leave pending_arrival set, so the
            # next /photo got 409 confirm_pending and /confirm advanced it again.
            log.info("[session %s] User confirmed arrival via answer", session_id)
            s.pending_question = None
            s.pending_is_confirm = False
            s.pending_next_instruction = None
            s.pending_arrival = True
            s.goal_node = nid
            s.history.append({
                "kind": "answer", "user_answer": req.answer,
                "vlm_action": "CONFIRM", "vlm_guidance": None,
            })
            return _handle_confirm(session_id, ConfirmArrivalRequest(kind="confirmed"))

        s.history.append({
            "kind": "answer", "user_answer": req.answer,
            "vlm_action": vlm_resp.action.value, "vlm_guidance": vlm_resp.guidance,
        })
        _g_items, _v_items, _c_target = _shopping_progress(s)
        return TurnResponse(
            action=vlm_resp.action,
            guidance=vlm_resp.guidance,
            question=vlm_resp.question,
            confirm_question=vlm_resp.action == VLMAction.ASK,
            node_id=nid,
            remaining_targets=len(s.remaining_targets) if s.target_nodes else None,
            phase=s.phase,
            goal_item_names=_g_items,
            visited_item_names=_v_items,
            current_target_name=_c_target,
            goal_node_id=_current_goal_node_id(s),
        )

    # Normal answer flow (not a confirm question)
    _active_goals = _current_goal_objects(s)
    _active_goal_str = "找到：" + "、".join(_active_goals)

    # Build route context from session's stored localization state
    _answer_route_ctx = None
    if s.user_heading is not None and s.last_corrected_nid is not None and s.place:
        from server.visual_localization import LocalizationResult
        from server.neo4j_client import get_neo4j as _get_neo4j
        _neo4j = _get_neo4j()
        _ref_node = _neo4j.get_photo_node(s.last_corrected_nid, s.place) if _neo4j else None
        _pseudo_loc = LocalizationResult(
            matched_nid=s.last_corrected_nid,
            confidence=s.heading_confidence,
            method="session_cache",
            reasoning="from previous photo localization",
            ref_node=_ref_node,
            matched_heading=s.user_heading,
            matched_slot=s.heading_slot,
            heading_confidence=s.heading_confidence,
        )
        _answer_route_ctx, _ = _build_route_context(s, _pseudo_loc)

    vlm_resp = vlm_decide(
        image_path=s.last_photo_path,
        goal=_active_goal_str,
        goal_objects=_active_goals,
        topomap_summary=topomap_summary,
        detections_summary=detections_summary,
        prior_question=prior_question,
        prior_answer=req.answer,
        ocr_summary=getattr(s, "last_ocr_summary", None),
        route_context=_answer_route_ctx,
        context_objects=s.context_objects,
    )

    log.info("[session %s] VLM → %s | %s", session_id, vlm_resp.action.value, vlm_resp.guidance[:120])

    # Gate ARRIVED on real evidence — use arrival-specific goal objects
    _answer_arrival_goals = s.arrival_goal_objects or _active_goals
    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, s.last_detections, ocr_matches=s.last_ocr_matches, goal_objects=_answer_arrival_goals,
        min_score=ARRIVED_MIN_DETECTION_SCORE, img_w=img_w, img_h=img_h,
        prior_was_confirm=False,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action != VLMAction.ARRIVED
    if _raw_action == VLMAction.ARRIVED:
        _far_q, _far_m = _far_from_goal_question(s, s.last_corrected_nid)
        if _far_q:
            if _far_m is not None and _far_m > ARRIVAL_ASK_MAX_M:
                log.info("[session %s] answer: arrival too far (%.0f m) → silent MOVE", session_id, _far_m)
                vlm_resp = VLMResponse(action=VLMAction.MOVE, guidance=vlm_resp.guidance,
                                       question=None, vlm_summary=vlm_resp.vlm_summary)
            else:
                log.info("[session %s] arrival not near the goal node on the map → asking: %s", session_id, _far_q)
                vlm_resp = VLMResponse(action=VLMAction.ASK, guidance=_far_q, question=_far_q,
                                       vlm_summary=vlm_resp.vlm_summary)
            arrival_downgraded = True

    s.history.append({
        "kind": "answer",
        "user_answer": req.answer,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
    })

    s.pending_question = None
    s.pending_is_confirm = False
    s.pending_next_instruction = None
    if vlm_resp.action == VLMAction.ARRIVED:
        s.pending_arrival = True
        s.goal_node = s.last_node_id
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
        s.pending_is_confirm = arrival_downgraded
    else:
        s.last_planned_action = vlm_resp.guidance

    _g_items, _v_items, _c_target = _shopping_progress(s)
    return TurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        confirm_question=vlm_resp.action == VLMAction.ASK and arrival_downgraded,
        node_id=s.last_node_id,
        annotated_photo_url=f"/session/{session_id}/photo/{s.last_node_id}.jpg",
        phase=s.phase,
        goal_item_names=_g_items,
        visited_item_names=_v_items,
        current_target_name=_c_target,
        goal_node_id=_current_goal_node_id(s),
    )


@app.post("/session/{session_id}/confirm", response_model=TurnResponse)
def confirm_arrival(session_id: str, req: ConfirmArrivalRequest) -> TurnResponse:
    """User confirms or rejects the ARRIVED decision."""
    resp = _handle_confirm(session_id, req)
    _log_turn(session_id, "confirm", {"kind": req.kind}, resp)
    return resp


def _handle_confirm(session_id: str, req: ConfirmArrivalRequest) -> TurnResponse:
    """User confirms or rejects the ARRIVED decision."""
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if req.kind == "confirmed_pdr":
        # The phone's step tracking put the user at the current stop; the server never
        # decided ARRIVED. Confirm the stop of the leg being walked.
        if s.arrived:
            raise HTTPException(status_code=409, detail={"error": "already_arrived", "detail": "session is complete"})
        leg = _phase_leg(s)
        s.goal_node = leg.to_node if leg else s.last_node_id
        if s.goal_node is None:
            raise HTTPException(status_code=409, detail={"error": "no_current_stop", "detail": "no stop to confirm"})
        s.pending_arrival = True
        req = ConfirmArrivalRequest(kind="confirmed")
    if not s.pending_arrival:
        raise HTTPException(status_code=409, detail={"error": "no_pending_arrival", "detail": "no ARRIVED to confirm"})

    arrived_node = s.goal_node
    s.pending_arrival = False

    if req.kind == "confirmed":
        # Mark goal detections as arrived in scene graph
        if arrived_node == s.last_node_id and s.last_detections:
            for d in s.last_detections:
                if _is_goal_det(d, s.goal_objects):
                    d["status"] = "arrived"
        # Mark goal-matching OCR texts as arrived evidence
        if arrived_node is not None:
            node_data = s.topomap.graph.nodes.get(arrived_node, {})
            ocr_texts = node_data.get("ocr_texts", [])
            arrived_ocr = [t for t in ocr_texts
                           if any(g.lower() in t.lower() or t.lower() in g.lower()
                                  for g in s.goal_objects)]
            if arrived_ocr:
                node_data["arrived_ocr"] = arrived_ocr

        if s.phase == "shopping":
            # Advance route: mark current target as visited and move to next leg
            found_item_name = None
            if s.route_plan and s.target_nodes:
                legs = s.route_plan.legs
                last = len(legs) - 1
                planned = legs[min(s.current_leg_index, last)].to_node

                # Credit the stop the user ACTUALLY reached.
                # _current_goal_objects deliberately offers every remaining item
                # to the VLM so it can call ARRIVED opportunistically when the
                # user passes a non-current target. Crediting the planned leg
                # then marks the WRONG item found: the planned item is never
                # visited, and the item being stood at stays outstanding.
                reached = planned
                if (arrived_node is not None
                        and arrived_node in s.target_nodes
                        and arrived_node not in s.visited_targets
                        and arrived_node != planned):
                    reached = arrived_node
                    log.info("[session %s] arrived at WP%d but the plan expected WP%d — "
                             "crediting the stop actually reached",
                             session_id, arrived_node, planned)

                found_item_name = s.target_node_to_item.get(reached)
                s.mark_target_visited(reached)

                replanned = False
                if reached != planned:
                    # The visit order was built around reaching `planned` next, so
                    # the stops behind the user are no longer on any leg. Re-plan
                    # from where they actually are. Without this the skipped stop
                    # is unreachable from the stale leg list, and the off-path
                    # re-plan never fires because the user IS on the path of the
                    # leg we would otherwise advance to.
                    replanned = _replan_from(s, reached)
                    if replanned:
                        # _replan_from already points current_leg_index at the
                        # next unvisited target, so do NOT advance past it.
                        log.info("[session %s] re-planned from WP%d after out-of-order arrival; "
                                 "order now %s", session_id, reached,
                                 getattr(s.route_plan, "visit_order", None))

                if not replanned:
                    # Skip any later leg whose target is already credited, so
                    # finding an item early does not walk the user back to a
                    # stop they have already stood at.
                    nxt = s.current_leg_index + 1
                    while (nxt < last
                           and legs[nxt].purpose == "target"
                           and legs[nxt].to_node in s.visited_targets):
                        nxt += 1
                    s.current_leg_index = min(nxt, last)
                legs = s.route_plan.legs
                log.info("[session %s] route advanced: visited node #%d, leg %d/%d, remaining=%s",
                         session_id, reached, s.current_leg_index,
                         len(legs), s.remaining_targets)

            remaining = len(s.remaining_targets) if s.target_nodes else 0
            found_label = found_item_name or "商品"
            if remaining > 0:
                _, _, next_target = _shopping_progress(s)
                next_label = next_target or "下一項商品"
                msg = f"已找到「{found_label}」！接下來找「{next_label}」，請拍一張照片讓我定位。"
                s.arrived = False
            else:
                if s.checkout_node is not None:
                    s.phase = "checkout"
                    msg = f"已找到「{found_label}」！所有商品都找到了，現在帶您前往收銀台結帳，請拍一張照片讓我定位。"
                    s.arrived = False
                    log.info("[session %s] phase → checkout (WP%d)", session_id, s.checkout_node)
                elif s.exit_node is not None:
                    s.phase = "exit"
                    msg = f"已找到「{found_label}」！所有商品都找到了，現在帶您前往出口，請拍一張照片讓我定位。"
                    s.arrived = False
                    log.info("[session %s] phase → exit (WP%d)", session_id, s.exit_node)
                else:
                    msg = f"已找到「{found_label}」！所有商品都找到了！"
                    s.arrived = True
                    s.phase = "done"
        elif s.phase == "checkout":
            # Confirmed at checkout — transition to exit phase
            if s.exit_node is not None:
                s.phase = "exit"
                s.current_leg_index = min(s.current_leg_index + 1, len(s.route_plan.legs) - 1) if s.route_plan else 0
                msg = "結帳完成！現在帶您前往出口，請拍一張照片讓我定位。"
                s.arrived = False
                log.info("[session %s] phase → exit (WP%d)", session_id, s.exit_node)
            else:
                msg = "結帳完成！導航結束。"
                s.arrived = True
                s.phase = "done"
        elif s.phase == "exit":
            # Confirmed at exit — done
            msg = "已到達出口，導航完成！感謝使用。"
            s.arrived = True
            s.phase = "done"
            log.info("[session %s] phase → done", session_id)
        else:
            msg = "導航已完成。"
            s.arrived = True

        log.info("[session %s] confirmed arrival at node %s, phase=%s", session_id, arrived_node, s.phase)
        s.history.append({"kind": "confirm", "node_id": arrived_node, "phase": s.phase, "message": msg})
        action = VLMAction.ARRIVED if s.arrived else VLMAction.MOVE

    elif req.kind == "repeated_misidentification":
        # The same goal has now been wrongly declared ARRIVED multiple times in a
        # row — retrying the same MOVE search again is unlikely to help. Change
        # strategy: ask the user directly to describe the item instead.
        s.goal_node = None
        for h in s.history:
            if h.get("kind") == "photo" and h.get("node_id") == arrived_node:
                h["rejected"] = req.kind
                break
        if arrived_node == s.last_node_id and s.last_detections:
            for d in s.last_detections:
                if _is_goal_det(d, s.goal_objects):
                    d["status"] = req.kind
        msg = f"連續判斷錯誤，請描述「{s.goal}」的外觀、包裝顏色或位置，幫助我重新確認"
        s.pending_question = msg
        s.pending_is_confirm = False
        s.repeated_misidentification_count += 1
        log.info("[session %s] repeated misidentification (#%d) — asking user to describe item",
                 session_id, s.repeated_misidentification_count)
        s.corrections.append(msg)
        s.history.append({
            "kind": "reject", "node_id": arrived_node,
            "reject_type": req.kind, "message": msg,
        })
        action = VLMAction.ASK

    else:
        s.goal_node = None
        status_label = req.kind

        for h in s.history:
            if h.get("kind") == "photo" and h.get("node_id") == arrived_node:
                h["rejected"] = status_label
                break

        if arrived_node == s.last_node_id and s.last_detections:
            for d in s.last_detections:
                if _is_goal_det(d, s.goal_objects):
                    d["status"] = status_label

        if req.kind == "false_positive":
            s.false_positive_nodes.append(arrived_node)
            msg = f"步驟 {arrived_node} 的目標宣告是誤判，繼續搜索{s.goal}"
        else:
            msg = f"步驟 {arrived_node} 是同種目標（{s.goal}）但非要找的，繼續搜索"

        s.corrections.append(msg)
        log.info("[session %s] rejected arrival: %s | %s", session_id, req.kind, msg)
        s.history.append({
            "kind": "reject", "node_id": arrived_node,
            "reject_type": req.kind, "message": msg,
        })
        action = VLMAction.MOVE

    # Re-render scene graph
    out_dir = ensure_output_dir(session_id)
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    observations = _build_observations(s)
    render_scene_graph(observations, s.goal, s.goal_objects,
                       str(graph_dir / f"scene_graph_{arrived_node}_confirm.png"))

    _g_items, _v_items, _c_target = _shopping_progress(s)
    return TurnResponse(
        action=action,
        guidance=msg,
        question=s.pending_question,
        node_id=arrived_node,
        annotated_photo_url=f"/session/{session_id}/photo/{arrived_node}.jpg",
        phase=s.phase,
        goal_item_names=_g_items,
        visited_item_names=_v_items,
        current_target_name=_c_target,
        goal_node_id=_current_goal_node_id(s),
    )


def _is_goal_det(d: dict, goal_objects: list[str]) -> bool:
    label_l = d.get("label", "").lower()
    return any(g.lower() in label_l or label_l in g.lower() for g in goal_objects)


def _build_observations(s) -> list[dict]:
    """Build observations list from session history for scene graph rendering."""
    observations = []
    for h in s.history:
        if h["kind"] == "photo":
            hid = h["node_id"]
            node_data = s.topomap.graph.nodes.get(hid, {})
            stored_dets = node_data.get("det_dicts")
            if stored_dets is not None:
                dets = [dict(d) for d in stored_dets]
            else:
                dets = [{"label": d, "score": 0.5}
                        for d in node_data.get("detected", [])]
            obs = {
                "step": hid,
                "photo": Path(node_data.get("photo_path", "")).name,
                "detections": dets,
                "vlm_summary": node_data.get("summary", ""),
                "ocr_texts": node_data.get("ocr_texts", []),
                "ocr_with_conf": node_data.get("ocr_with_conf", []),
                "arrived_ocr": node_data.get("arrived_ocr", []),
                "false_positive": hid in s.false_positive_nodes,
            }
            rejected = h.get("rejected")
            if rejected:
                for d in obs["detections"]:
                    if _is_goal_det(d, s.goal_objects):
                        d["status"] = rejected
            observations.append(obs)
    return observations


@app.get("/session/{session_id}/map")
def get_map(session_id: str, format: str = Query(default="json")):
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if format == "png":
        png = s.topomap.render_png(current_id=s.last_node_id)
        return Response(content=png, media_type="image/png")
    return s.topomap.to_dict(current_node=s.last_node_id, goal_node=s.goal_node)


@app.post("/session/{session_id}/sensor-map")
def upload_sensor_map(session_id: str, payload: dict = Body(...)):
    """Stores the phone's on-device PDR sensor map (nodes/edges from CoreMotion
    dead-reckoning) next to this session's VLM output, purely for inspection —
    it is not merged into or used by the VLM navigation logic."""
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    out_dir = ensure_output_dir(session_id)
    (out_dir / "sensor_map.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"status": "ok"}


@app.post("/sensor-test")
def upload_sensor_test(payload: dict = Body(...)):
    """Standalone PDR sensor accuracy test (SensorTestView.swift) — no navigation
    session involved. Stores the raw path + lap markers and renders a PNG plot
    so drift/error can be inspected visually after a multi-lap walk."""
    test_id = uuid.uuid4().hex[:8]
    out_dir = OUTPUT_ROOT.parent / "sensor_tests" / test_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "record.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    png = render_sensor_test_png(payload)
    (out_dir / "plot.png").write_bytes(png)
    return {"test_id": test_id, "status": "ok"}


@app.get("/sensor-test/{test_id}/plot.png")
def get_sensor_test_plot(test_id: str):
    path = OUTPUT_ROOT.parent / "sensor_tests" / test_id / "plot.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": test_id})
    return FileResponse(str(path), media_type="image/png")


def _log_rerank_status(ref) -> None:
    """One startup line saying whether photo-based re-ranking will actually run."""
    from server.config import RERANK_ENABLED, REF_PHOTO_ROOT
    from server.visual_localization import ref_photo_coverage
    if not RERANK_ENABLED:
        log.info("Visual re-ranker: OFF (set RERANK_ENABLED=1 and REF_PHOTO_ROOT to enable)")
        return
    if not REF_PHOTO_ROOT:
        log.warning("Visual re-ranker: OFF — RERANK_ENABLED=1 but REF_PHOTO_ROOT is empty")
        return
    found, total, nodes_ok = ref_photo_coverage(ref, REF_PHOTO_ROOT)
    if found == 0:
        log.warning("Visual re-ranker: OFF — none of the %d reference photos found under %s",
                    total, REF_PHOTO_ROOT)
    else:
        log.info("Visual re-ranker: ON — %d/%d reference photos found under %s (%d/%d nodes)",
                 found, total, REF_PHOTO_ROOT, nodes_ok, len(ref.photos))


@app.get("/", response_class=PlainTextResponse)
def root_status() -> str:
    """Opening the server address in a browser shows what it is and whether it is
    ready — a bare 404 there reads as "the server is broken" during field tests."""
    from server.config import OPENAI_MODEL, RERANK_ENABLED
    neo4j = get_neo4j()
    lines = [
        "APPNAV navigation server",
        f"VLM model      : {OPENAI_MODEL}",
        f"Neo4j          : {'connected' if neo4j else 'not configured'}",
        f"Visual re-rank : {'on' if RERANK_ENABLED else 'off'}",
        "",
        "Checks : /health (JSON)   API docs : /docs",
        "The iOS app talks to /session, /session/{id}/photo and /session/{id}/path.",
    ]
    return "\n".join(lines)


@app.on_event("startup")
def _check_vlm_and_warm():
    from server.config import OPENAI_API_KEY, OPENAI_MODEL
    log.info("VLM backend: OpenAI %s", OPENAI_MODEL)
    log.info("Perception (GroundingDINO): %s | OCR (EasyOCR): %s",
             "ON" if PERCEPTION_ENABLED else "OFF (VLM-only)",
             "ON" if OCR_ENABLED else "OFF")
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set. Add it to .env or environment variables.")

    if not os.environ.get("UNIGOAL_TEST_MODE"):
        _vlm_warm_up()
        # DINOv2 for photo localization: load it now, in the background, instead of on
        # the first photo (~9 s on this CPU-only torch).
        if _config.LOC_SERVER_DINO:
            import threading
            from server import dino_embed
            threading.Thread(target=dino_embed.available, name="dino-preload", daemon=True).start()

    # Neo4j: connect and pre-load reference map
    neo4j = get_neo4j()
    if neo4j:
        try:
            ref = neo4j.load_reference_map()
            log.info("Neo4j reference map: %d photos, %d edges",
                     len(ref.photos), len(ref.walkway_edges))
            _log_rerank_status(ref)
        except Exception as e:
            log.warning("Neo4j reference map load failed: %s", e)
    else:
        log.info("Neo4j not configured — visual localization disabled")


@app.post("/ocr")
async def ocr_extract(image: UploadFile = File(...)):
    """Run the existing EasyOCR engine on one uploaded image and return its text."""
    engine = get_ocr()
    if engine is None:
        raise HTTPException(status_code=503, detail={"error": "ocr_unavailable",
                            "detail": "EasyOCR disabled or failed to load"})
    import tempfile
    data = await image.read()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(data)
            tmp = tf.name
        results = engine.read(tmp, min_confidence=OCR_MIN_CONFIDENCE, max_results=OCR_MAX_RESULTS)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return {
        "engine": "easyocr",
        "languages": OCR_LANGUAGES,
        "text": " ".join(r.text for r in results),
        "regions": [
            {"text": r.text, "confidence": round(r.confidence, 4),
             "bbox": [list(p) for p in r.bbox]}
            for r in results
        ],
    }


def _ref_location_name(loc_result) -> Optional[str]:
    """Build a human-readable location name from the localization result.

    Uses OCR texts (signs, labels) and prominent object labels from the
    matched reference node to describe the area in a way that makes sense
    to the user, e.g. "「衛生紙」「洗衣精」附近" or "冷凍食品區".
    """
    if not loc_result or not loc_result.ref_node:
        return None
    node = loc_result.ref_node
    if not node.objects:
        return f"節點 #{node.nid}"

    # Collect OCR texts (signs are the most informative)
    ocr_signs = []
    ocr_products = []
    labels = []
    for obj in node.objects:
        if obj.ocr_text and obj.ocr_text.strip():
            text = obj.ocr_text.strip()
            if obj.role == "標示牌":
                ocr_signs.append(text)
            else:
                ocr_products.append(text)
        if obj.label and obj.label.strip():
            labels.append(obj.label.strip())

    # Priority: signs > product OCR > object labels
    parts = []
    seen = set()
    for text in ocr_signs + ocr_products:
        t_lower = text.lower()
        if len(text) < 2 or text.isdigit() or t_lower in seen or len(text) > 20:
            continue
        parts.append(f"「{text}」")
        seen.add(t_lower)
        if len(parts) >= 3:
            break

    if parts:
        return "".join(parts) + " 附近"

    # Fallback: use distinct object labels
    unique_labels = []
    for l in labels:
        l_lower = l.lower()
        if l_lower not in seen and l_lower not in ("shelf", "shelves", "wall", "floor", "ceiling"):
            unique_labels.append(l)
            seen.add(l_lower)
        if len(unique_labels) >= 3:
            break
    if unique_labels:
        return "、".join(unique_labels) + " 區域"

    return f"節點 #{node.nid}"


@app.post("/localize")
async def localize_photo(photo: UploadFile = File(...), place: str = Query(default="")):
    """Standalone endpoint: upload a photo, get the matching reference node.

    This is for ad-hoc position checks outside a navigation session — the app
    can call it when the user wants to know "where am I?" without starting a
    full navigation session. It runs the same VLM perception + Neo4j matching
    that the session photo upload does, but returns only the localization result.
    """
    neo4j = get_neo4j()
    if neo4j is None:
        raise HTTPException(status_code=503, detail={
            "error": "neo4j_unavailable",
            "detail": "Neo4j not configured or not connected",
        })

    import tempfile
    photo_bytes = await photo.read()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(photo_bytes)
            tmp = tf.name

        # Run perception (VLM-only mode since we don't need GroundingDINO for localization)
        im = Image.open(tmp)
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        img_w, img_h = im.size
        if max(img_w, img_h) > 1600:
            im.thumbnail((1600, 1600))
            img_w, img_h = im.size
        im.save(tmp, format="JPEG", quality=85)
        im.close()

        if _config.VLM_ROUNDS > 1:
            perception, _ = _vlm_perceive_multi_round(
                image_path=tmp, goal="定位", goal_objects=[],
                img_w=img_w, img_h=img_h, num_rounds=_config.VLM_ROUNDS,
            )
        elif _config.VLM_PASS3:
            perception, _ = _vlm_perceive_three_pass(
                image_path=tmp, goal="定位", goal_objects=[],
                img_w=img_w, img_h=img_h,
            )
        else:
            perception, _ = _vlm_perceive_two_pass(
                image_path=tmp, goal="定位", goal_objects=[],
                img_w=img_w, img_h=img_h,
            )

        detected_labels = [_vlm_localization_label(d) for d in perception.detections]
        ocr_texts = [t.text for t in perception.ocr_texts]

        # Also try EasyOCR if available
        ocr_engine = get_ocr()
        if ocr_engine:
            easyocr_results = ocr_engine.read(
                tmp, min_confidence=OCR_MIN_CONFIDENCE, max_results=OCR_MAX_RESULTS)
            ocr_texts = [r.text for r in easyocr_results] if easyocr_results else ocr_texts

    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    result = _localize_photo(
        detected_labels=detected_labels,
        ocr_texts=ocr_texts,
        neo4j=neo4j,
        place=place or None,
        grid_n=_config.GRID_N,
    )

    resp = {
        "matched_nid": result.matched_nid,
        "confidence": round(result.confidence, 3),
        "method": result.method,
        "reasoning": result.reasoning,
        "detected_objects": detected_labels,
        "ocr_texts": ocr_texts,
    }
    if result.ref_node:
        resp["ref_location"] = {
            "nid": result.ref_node.nid,
            "photo_file": result.ref_node.photo_file,
            "pdr_x": result.ref_node.pdr_x,
            "pdr_y": result.ref_node.pdr_y,
            "heading_deg": result.ref_node.heading_deg,
            "session": result.ref_node.session,
        }
    if result.runner_up_nid is not None:
        resp["runner_up"] = {
            "nid": result.runner_up_nid,
            "score": round(result.runner_up_score, 3),
        }
    return resp


@app.post("/localize-test")
async def localize_test(
    photo: UploadFile = File(...),
    place: str = Query(default=""),
    ground_truth_nid: Optional[int] = Query(default=None),
):
    """Full-diagnostic localization endpoint for testing.

    Returns Top-5 candidates with positions, matched directional photos,
    object/OCR breakdown with scores, and ground truth comparison.
    """
    neo4j = get_neo4j()
    if neo4j is None:
        raise HTTPException(status_code=503, detail="neo4j_unavailable")

    import tempfile
    photo_bytes = await photo.read()
    tmp = None
    t_start = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(photo_bytes)
            tmp = tf.name

        im = Image.open(tmp)
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        img_w, img_h = im.size
        if max(img_w, img_h) > 1600:
            im.thumbnail((1600, 1600))
            img_w, img_h = im.size
        im.save(tmp, format="JPEG", quality=85)
        im.close()

        # VLM perception
        if _config.VLM_ROUNDS > 1:
            perception, perc_debug = _vlm_perceive_multi_round(
                image_path=tmp, goal="定位", goal_objects=[],
                img_w=img_w, img_h=img_h, num_rounds=_config.VLM_ROUNDS,
            )
        elif _config.VLM_PASS3:
            perception, perc_debug = _vlm_perceive_three_pass(
                image_path=tmp, goal="定位", goal_objects=[],
                img_w=img_w, img_h=img_h,
            )
        else:
            perception, perc_debug = _vlm_perceive_two_pass(
                image_path=tmp, goal="定位", goal_objects=[],
                img_w=img_w, img_h=img_h,
            )

        detected_labels = [_vlm_localization_label(d) for d in perception.detections]
        ocr_texts = [t.text for t in perception.ocr_texts]

        ocr_engine = get_ocr()
        if ocr_engine:
            easyocr_results = ocr_engine.read(
                tmp, min_confidence=OCR_MIN_CONFIDENCE, max_results=OCR_MAX_RESULTS)
            ocr_texts = [r.text for r in easyocr_results] if easyocr_results else ocr_texts

        # Build detected_items with bboxes for object relations
        detected_items = [
            {"label": _vlm_localization_label(d), "box": list(d.bbox) if d.bbox else None}
            for d in perception.detections
        ]

        the_place = place or None
        ref_map = neo4j.load_reference_map(the_place)

        # Grid-based localization
        result = _localize_photo(
            detected_labels=detected_labels,
            ocr_texts=ocr_texts,
            neo4j=neo4j,
            place=the_place,
            grid_n=_config.GRID_N,
            detected_items=detected_items,
        )

        # Server-side DINOv2 fusion
        server_ev = None
        if _config.LOC_SERVER_DINO:
            from server.locate_fusion import server_image_evidence
            try:
                server_ev = server_image_evidence(
                    ref_map, tmp, detected_labels, ocr_texts,
                    _config.DINO_SIDECAR_PATH, q_items=detected_items,
                    relation_weight=_config.LOC_RELATION_WEIGHT)
            except Exception:
                log.exception("localize-test: server DINOv2 failed")

        # fuse_v2 if we have server evidence
        fusion_candidates = None
        if server_ev and _config.LOC_FUSION_V2:
            from server.loc_fusion import fuse_v2, parse_image_k
            node_positions = {nid: (n.pdr_x, n.pdr_y) for nid, n in ref_map.photos.items()}
            fusion_candidates = fuse_v2(
                server_ev.scores, node_positions,
                image_k=parse_image_k(_config.LOC_SERVER_IMAGE_K),
            )

        t_elapsed = time.time() - t_start

    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    # Build Top-5 candidates with position and direction info
    top5 = []
    raw_candidates = fusion_candidates or (result.top_candidates if result.top_candidates else [])
    for i, c in enumerate(raw_candidates[:5]):
        nid = c[0]
        score = round(c[1], 4)
        reason = c[2] if len(c) > 2 else ""
        node = ref_map.photos.get(nid)
        entry = {
            "rank": i + 1,
            "nid": nid,
            "score": score,
            "reason": reason,
        }
        if node:
            entry["pdr_x"] = node.pdr_x
            entry["pdr_y"] = node.pdr_y
            entry["heading_deg"] = node.heading_deg
            entry["photo_file"] = node.photo_file
            entry["photo_url"] = f"/reference-photo/{nid}.jpg"
            # Find matched directional photo
            matched_dp = (server_ev.best_photo.get(nid) if server_ev else None)
            if matched_dp:
                entry["matched_slot"] = matched_dp.slot
                entry["matched_heading"] = matched_dp.heading_deg
                entry["matched_photo_file"] = matched_dp.photo_file
                entry["matched_photo_url"] = f"/reference-dir-photo/{nid}/{matched_dp.slot}.jpg"
            elif node.directional_photos:
                dp = node.directional_photos[0]
                entry["matched_slot"] = dp.slot
                entry["matched_heading"] = dp.heading_deg
        top5.append(entry)

    # Build detections with scores
    det_list = []
    for d in perception.detections:
        det_list.append({
            "label": d.label,
            "score": round(d.score, 3) if d.score else 0,
            "position": d.position or "",
            "bbox": list(d.bbox) if d.bbox else None,
        })

    ocr_list = []
    for t in perception.ocr_texts:
        ocr_list.append({
            "text": t.text,
            "score": round(t.score, 3) if t.score else 0,
            "bbox": list(t.bbox) if t.bbox else None,
        })

    # Matched result
    matched = None
    if result.matched_nid is not None:
        node = ref_map.photos.get(result.matched_nid)
        matched = {
            "nid": result.matched_nid,
            "confidence": round(result.confidence, 3),
            "method": result.method,
            "reasoning": result.reasoning,
            "heading_deg": result.matched_heading,
            "heading_slot": result.matched_slot,
            "matched_photo": result.matched_photo,
        }
        if node:
            matched["pdr_x"] = node.pdr_x
            matched["pdr_y"] = node.pdr_y
            matched["photo_url"] = f"/reference-photo/{result.matched_nid}.jpg"

    # Ground truth comparison
    gt = None
    if ground_truth_nid is not None:
        gt_node = ref_map.photos.get(ground_truth_nid)
        gt = {"nid": ground_truth_nid, "found_in_top5": False, "rank": None, "distance_m": None}
        if gt_node:
            gt["pdr_x"] = gt_node.pdr_x
            gt["pdr_y"] = gt_node.pdr_y
            gt["photo_url"] = f"/reference-photo/{ground_truth_nid}.jpg"
        for i, c in enumerate(top5):
            if c["nid"] == ground_truth_nid:
                gt["found_in_top5"] = True
                gt["rank"] = i + 1
                break
        if matched and gt_node:
            m_node = ref_map.photos.get(matched["nid"])
            if m_node:
                import math as _m
                gt["distance_m"] = round(_m.sqrt(
                    (m_node.pdr_x - gt_node.pdr_x)**2 + (m_node.pdr_y - gt_node.pdr_y)**2), 1)

    # Reference map nodes for minimap
    map_nodes = []
    for nid, node in ref_map.photos.items():
        map_nodes.append({"nid": nid, "x": node.pdr_x, "y": node.pdr_y, "heading": node.heading_deg})

    map_edges = []
    for e in ref_map.walkway_edges:
        map_edges.append({"from": e.get("from") or e.get("a"), "to": e.get("to") or e.get("b")})

    return {
        "place": ref_map.place,
        "elapsed_s": round(t_elapsed, 1),
        "matched": matched,
        "top5": top5,
        "detections": det_list,
        "ocr_texts": ocr_list,
        "ground_truth": gt,
        "map_nodes": map_nodes,
        "map_edges": map_edges,
        "perception_debug": {
            "scene": getattr(perception, "scene_description", ""),
            "store_zone": getattr(perception, "store_zone", ""),
        },
    }


@app.get("/reference-dir-photo/{nid}/{slot}.jpg")
def serve_reference_dir_photo(nid: int, slot: str):
    """Serve a directional reference photo (front/right/back/left)."""
    if slot not in ("front", "right", "back", "left"):
        raise HTTPException(status_code=400, detail=f"bad slot: {slot}")
    neo4j = get_neo4j()
    if not neo4j:
        raise HTTPException(status_code=503, detail="neo4j unavailable")
    ref_map = neo4j.load_reference_map()
    if nid not in ref_map.photos:
        raise HTTPException(status_code=404, detail=f"node {nid} not found")
    node = ref_map.photos[nid]
    dp = next((d for d in node.directional_photos if d.slot == slot), None)
    if dp is None:
        return FileResponse(str(OUTPUT_ROOT / node.session / "photo" / node.photo_file), media_type="image/jpeg") \
            if (OUTPUT_ROOT / node.session / "photo" / node.photo_file).exists() \
            else HTTPException(status_code=404, detail=f"no {slot} photo for node {nid}")
    for p in [OUTPUT_ROOT / node.session / "photo" / dp.photo_file,
              OUTPUT_ROOT / node.session / "annotated" / dp.photo_file]:
        if p.exists():
            return FileResponse(str(p), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail=f"photo file not found: {dp.photo_file}")


@app.get("/localize-test-page")
def localize_test_page():
    """Localization test page with full diagnostics."""
    html_path = Path(__file__).parent / "localize_test.html"
    if not html_path.exists():
        return HTMLResponse("<h1>localize_test.html not found</h1>", status_code=500)
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/reference-map")
def get_reference_map(place: str = Query(default="")):
    """Return the reference topological map from Neo4j (for debugging/display)."""
    neo4j = get_neo4j()
    if neo4j is None:
        raise HTTPException(status_code=503, detail={
            "error": "neo4j_unavailable",
            "detail": "Neo4j not configured or not connected",
        })
    ref = neo4j.load_reference_map(place or None)
    return {
        "place": ref.place,
        "photo_count": len(ref.photos),
        "edge_count": len(ref.walkway_edges),
        "photos": [
            {
                "nid": p.nid,
                "photo_file": p.photo_file,
                "pdr_x": p.pdr_x,
                "pdr_y": p.pdr_y,
                "heading_deg": p.heading_deg,
                "session": p.session,
                "object_count": len(p.objects),
                "objects": [o.label for o in p.objects],
                "ocr_texts": [o.ocr_text for o in p.objects if o.ocr_text],
            }
            for p in ref.photos.values()
        ],
    }


@app.post("/test/config")
def set_test_config(grid_n: int = Query(default=None), vlm_rounds: int = Query(default=None), vlm_pass3: bool = Query(default=None),
                    fusion_weights: str = Query(default=None)):
    """Runtime config override for testing different grid sizes, VLM rounds and
    localization fusion weights ("dino,sensor,grid", e.g. "5,1,1")."""
    changed = {}
    if fusion_weights is not None:
        try:
            parse_weights(fusion_weights)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": "bad_fusion_weights", "detail": str(e)})
        _config.LOC_FUSION_WEIGHTS = fusion_weights
        changed["fusion_weights"] = fusion_weights
    if grid_n is not None and grid_n in (3, 4, 5):
        _config.GRID_N = grid_n
        changed["grid_n"] = grid_n
    if vlm_rounds is not None and vlm_rounds in (1, 2, 3):
        _config.VLM_ROUNDS = vlm_rounds
        changed["vlm_rounds"] = vlm_rounds
    if vlm_pass3 is not None:
        _config.VLM_PASS3 = vlm_pass3
        changed["vlm_pass3"] = vlm_pass3
    return {"status": "ok", "changed": changed, "current": {"grid_n": _config.GRID_N, "vlm_rounds": _config.VLM_ROUNDS, "vlm_pass3": _config.VLM_PASS3,
                                                    "fusion_weights": _config.LOC_FUSION_WEIGHTS}}


# ── Editor-map path following ──────────────────────────────────────────
_editor_graphs: dict = {}


def _neo4j_route_graph(place: str):
    """Route graph built from the Neo4j reference map (+ maps/neo4j_links/*.json for
    that place), or None when Neo4j has no map for it."""
    neo4j = get_neo4j()
    ref_map = neo4j.load_reference_map(place) if neo4j else None
    if not ref_map or not ref_map.photos:
        return None
    extra = None
    for path in sorted((EDITOR_MAP_DIR / "neo4j_links").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # a broken links file must not take the server down
            log.warning("neo4j links %s unreadable: %s", path.name, e)
            continue
        if data.get("place") == place:
            extra = data
            break
    graph = editor_map_mod.build_from_ref_map(ref_map, extra)
    if graph.graph.number_of_edges() == 0:
        log.warning("route map for %s: Neo4j has no WALKWAY edges — no route graph", place)
        return None
    log.info("route map for %s from Neo4j: %d nodes, %d edges (extra links: %s, entrance: %s)",
             place, graph.graph.number_of_nodes(), graph.graph.number_of_edges(),
             len((extra or {}).get("links", [])), graph.entrance)
    return graph


def _editor_graph_for(place: Optional[str]):
    """The walkable route graph for ``place`` (cached). ROUTE_MAP_SOURCE=neo4j (default)
    builds it from the Neo4j reference map, so routes and localization share one set of
    coordinates; "editor" uses the hand-corrected maps/*.json with Neo4j labels merged
    in. None when neither has the place."""
    if not place:
        return None
    if place in _editor_graphs:
        return _editor_graphs[place]
    if _config.ROUTE_MAP_SOURCE == "neo4j":
        graph = _neo4j_route_graph(place)
        if graph is not None:      # not cached when Neo4j was unreachable: retry next time
            _editor_graphs[place] = graph
        return graph
    graph = None
    for path in sorted(EDITOR_MAP_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # a broken map file must not take the server down
            log.warning("editor map %s unreadable: %s", path.name, e)
            continue
        if data.get("place") != place:
            continue
        graph = editor_map_mod.build_graph(data)
        neo4j = get_neo4j()
        ref_map = neo4j.load_reference_map(place) if neo4j else None
        if ref_map and ref_map.photos:
            unmapped = editor_map_mod.merge_from_ref_map(graph, ref_map)
            log.info("editor map %s: %d waypoints, %d edges; %d/%d Neo4j nodes attached (unmapped: %s)",
                     path.name, graph.graph.number_of_nodes(), graph.graph.number_of_edges(),
                     len(ref_map.photos) - len(unmapped), len(ref_map.photos), unmapped)
        else:
            log.warning("editor map %s loaded without Neo4j labels — goals cannot be resolved", path.name)
        break
    _editor_graphs[place] = graph
    return graph


@app.get("/session/{session_id}/path", response_model=PathResponse,
         responses={404: {"model": ErrorResponse}})
def get_session_path(session_id: str,
                     x: Optional[float] = Query(default=None),
                     y: Optional[float] = Query(default=None),
                     heading: Optional[float] = Query(default=None)):
    """Route on the hand-corrected editor map for the phone to follow.

    Start = the phone's dead-reckoned position (``x``/``y``) when given, else the
    waypoint of the last photo localization, else the entrance. Target = the
    current route leg's Neo4j node (chosen when the session started — the same
    node the text guidance and the arrival check use) mapped onto the editor map,
    else the waypoint whose product/sign labels name the current goal item.
    """
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    g = _editor_graph_for(s.place)
    if g is None:
        raise HTTPException(status_code=404, detail={
            "error": "no_editor_map",
            "detail": f"no route map for place {s.place!r} (ROUTE_MAP_SOURCE={_config.ROUTE_MAP_SOURCE})"})

    goals = _current_goal_objects(s)
    goal_item = goals[0] if goals else (s.target_objects[0] if s.target_objects else s.goal)

    # The session's route leg first, so the mini-map heads where the text guidance
    # does (a separate label search picked other nodes: 「牛奶」 misses a 「牛乳」 sign).
    leg = _phase_leg(s)
    target_wp = g.waypoint_for_neo(leg.to_node) if leg is not None else None
    if target_wp is None:
        # Sign-weighted label search: a "牛奶區" category sign beats scattered product mentions
        target_wp = editor_map_mod.find_goal_waypoint(g, goal_item)
    if target_wp is None:
        raise HTTPException(status_code=404, detail={
            "error": "goal_not_on_map",
            "detail": f"no waypoint on the editor map mentions {goal_item!r}"})

    if x is not None and y is not None:
        start_wp, source = editor_map_mod.nearest_waypoint(g, x, y), "phone"
    elif s.last_corrected_nid is not None and g.waypoint_for_neo(s.last_corrected_nid) is not None:
        start_wp, source = g.waypoint_for_neo(s.last_corrected_nid), "photo"
    else:
        start_wp, source = g.entrance, "entrance"

    path = editor_map_mod.plan(g, start_wp, target_wp)
    route = editor_map_mod.route_payload(g, path, user_heading=heading)
    sx, sy = g.position(start_wp)
    tx, ty = g.position(target_wp)
    log.info("[session %s] editor path: %s wp%d → wp%d (%s), %.1f m, %d turns",
             session_id, source, start_wp, target_wp, goal_item, route["distance_m"], len(route["turns"]))
    return PathResponse(
        place=g.place, goal_item=goal_item,
        start={"wp": start_wp, "x": sx, "y": sy, "source": source},
        target={"wp": target_wp, "x": tx, "y": ty,
                "neo_nid": (g.graph.nodes[target_wp]["neo_nids"] or [None])[0],
                "products": g.graph.nodes[target_wp]["products"][:8]},
        path=route["path"], polyline=route["polyline"], turns=route["turns"],
        distance_m=route["distance_m"],
        nodes=[{"id": n["id"], "x": n["x"], "y": n["y"]} for n in g.nodes_payload()],
        edges=g.edges_payload(),
    )


@app.post("/session/{session_id}/start")
def set_session_start(session_id: str, payload: dict = Body(...)):
    """The user stands at a start point ({"x", "y"} in the map frame, from 「我在起點」):
    re-plan the visit order from the waypoint there instead of the default start."""
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    try:
        x, y = float(payload["x"]), float(payload["y"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=422, detail="body needs numeric 'x' and 'y'")
    g = _editor_graph_for(s.place)
    if g is None:
        raise HTTPException(status_code=404, detail={"error": "no_editor_map", "detail": f"no route map for {s.place!r}"})
    wp = editor_map_mod.nearest_waypoint(g, x, y)
    replanned = _replan_from(s, wp)
    log.info("[session %s] start point (%.1f, %.1f) → WP%s, re-planned=%s, order=%s",
             session_id, x, y, wp, replanned, s.route_plan.visit_order if s.route_plan else None)
    return {"start_node": wp, "replanned": replanned,
            "visit_order": list(s.route_plan.visit_order) if s.route_plan else [],
            "goal_node_id": _current_goal_node_id(s)}


@app.get("/session/{session_id}/route")
def get_current_route(session_id: str):
    """Get the current planned route for a session."""
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})

    if s.route_plan is None:
        return {"status": "no_route", "message": "No route planned yet"}

    return {
        "status": "active",
        "target_nodes": s.target_nodes,
        "visited": list(s.visited_targets),
        "remaining": s.remaining_targets,
        "route": s.route_plan.to_dict(),
    }


@app.get("/health")
def health():
    neo4j = get_neo4j()
    return {
        "status": "ok",
        "neo4j": "connected" if neo4j and neo4j.is_connected else "disconnected",
    }


@app.get("/dashboard")
def dashboard():
    """Live monitoring dashboard."""
    html_path = Path(__file__).parent / "dashboard.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/tracking")
def tracking_page():
    """Sensor path vs image localization monitoring page."""
    html_path = Path(__file__).parent / "tracking.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/session/{session_id}/tracking")
def session_tracking(session_id: str):
    """Return sensor PDR path, photo localization fixes, and reference map nodes
    for the tracking visualizer."""
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found"})

    # Collect PDR positions and localization results from session history
    sensor_points = []
    photo_fixes = []
    for i, h in enumerate(s.history):
        if h.get("kind") != "photo":
            continue
        loc = h.get("localization")
        nid = h.get("node_id")
        sensor_points.append({"step": i, "node_id": nid})
        if loc and loc.get("matched_nid") is not None:
            matched_nid = loc["matched_nid"]
            conf = loc.get("confidence", 0)
            photo_fixes.append({
                "step": i,
                "node_id": nid,
                "matched_nid": matched_nid,
                "confidence": round(conf, 3),
                "reasoning": loc.get("reasoning", ""),
            })

    # Collect PDR + map positions from the turn log files
    pdr_path = []
    out_dir = OUTPUT_ROOT / session_id
    turn_log_path = out_dir / "turns.jsonl"
    if turn_log_path.exists():
        for line in turn_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            req = entry.get("request", {})
            if entry.get("kind") == "photo" and req:
                pdr_x = req.get("pdr_x")
                pdr_y = req.get("pdr_y")
                map_x = req.get("map_x")
                map_y = req.get("map_y")
                heading = req.get("heading")
                loc_data = entry.get("localization", {}) or {}
                loc_result = loc_data.get("result", {}) or {}
                resp_data = entry.get("response", {}) or {}
                pdr_path.append({
                    "pdr_x": pdr_x, "pdr_y": pdr_y,
                    "map_x": map_x, "map_y": map_y,
                    "heading": heading,
                    "matched_nid": loc_result.get("matched_nid"),
                    "confidence": round(loc_result.get("confidence", 0), 3),
                    "corrected_nid": resp_data.get("corrected_node_id"),
                    "estimated_xy": loc_result.get("estimated_xy"),
                })

    # Reference map nodes (positions on the map)
    ref_nodes = []
    ref_edges = []
    neo4j = get_neo4j()
    ref_map = neo4j.load_reference_map(s.place) if neo4j and s.place else None
    if ref_map:
        for nid_key, node in ref_map.photos.items():
            ref_nodes.append({
                "nid": nid_key,
                "x": round(node.pdr_x, 2),
                "y": round(node.pdr_y, 2),
            })
        for edge in ref_map.walkway_edges:
            ref_edges.append({"from": edge.get("from", edge.get("from_nid")),
                              "to": edge.get("to", edge.get("to_nid"))})

    # Sensor map (phone's PDR topo nodes) if uploaded
    sensor_map = None
    sensor_map_path = out_dir / "sensor_map.json"
    if sensor_map_path.exists():
        try:
            sensor_map = json.loads(sensor_map_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    return {
        "session_id": session_id,
        "goal": s.goal,
        "place": s.place,
        "arrived": s.arrived,
        "photo_count": sum(1 for h in s.history if h.get("kind") == "photo"),
        "pdr_path": pdr_path,
        "photo_fixes": photo_fixes,
        "ref_nodes": ref_nodes,
        "ref_edges": ref_edges,
        "sensor_map": sensor_map,
    }


@app.get("/debug")
def debug_latest():
    """Redirect to the latest session's debug page."""
    from fastapi.responses import RedirectResponse
    latest = None
    for s in _store._sessions.values():
        if latest is None or s.created_at > latest.created_at:
            latest = s
    if not latest:
        return HTMLResponse("<h1>尚無 session</h1>", status_code=404)
    return RedirectResponse(f"/session/{latest.id}/debug-page")


@app.get("/sessions")
def list_sessions():
    """List all active sessions (newest first)."""
    sessions = []
    for sid, s in _store._sessions.items():
        sessions.append({
            "id": s.id,
            "goal": s.goal,
            "goal_objects": s.goal_objects,
            "photo_count": sum(1 for h in s.history if h.get("kind") == "photo"),
            "arrived": s.arrived,
            "created_at": s.created_at.isoformat(),
            "target_nodes": s.target_nodes,
            "visited_targets": list(s.visited_targets),
            "current_leg_index": s.current_leg_index,
        })
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return {"sessions": sessions}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    return {
        "id": s.id,
        "goal": s.goal,
        "goal_objects": s.goal_objects,
        "history": s.history,
        "pending_question": s.pending_question,
        "arrived": s.arrived,
        "last_node_id": s.last_node_id,
        "goal_node": s.goal_node,
        "created_at": s.created_at.isoformat(),
        "target_nodes": s.target_nodes,
        "visited_targets": list(s.visited_targets),
        "current_leg_index": s.current_leg_index,
        "route_plan": s.route_plan.to_dict() if s.route_plan and hasattr(s.route_plan, 'to_dict') else None,
    }


@app.get("/session/{session_id}/debug")
def session_debug(session_id: str):
    """Perception & localization debug report for the last photo."""
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    route_info = None
    if s.route_plan:
        visit_order = s.route_plan.visit_order if hasattr(s.route_plan, 'visit_order') else []
        route_info = {
            "visit_order": visit_order,
            "visit_order_items": [s.target_node_to_item.get(n, f"WP{n}") for n in visit_order],
            "total_cost": round(s.route_plan.total_cost, 1) if hasattr(s.route_plan, 'total_cost') else None,
            "current_leg_index": s.current_leg_index,
            "legs": [{"from": lg.from_node, "to": lg.to_node, "purpose": lg.purpose,
                       "cost": round(lg.cost, 1)}
                      for lg in s.route_plan.legs] if hasattr(s.route_plan, 'legs') else [],
        }
    return {
        "session_id": s.id,
        "goal": s.goal,
        "phase": s.phase,
        "target_nodes": s.target_nodes,
        "target_node_to_item": s.target_node_to_item,
        "visited_targets": list(s.visited_targets),
        "last_corrected_nid": s.last_corrected_nid,
        "route": route_info,
        "perception": s.debug_perception,
    }


@app.get("/session/{session_id}/debug-page")
def session_debug_page(session_id: str):
    """HTML debug page for perception & localization diagnostics."""
    html_path = Path(__file__).parent / "debug_page.html"
    if not html_path.exists():
        return HTMLResponse("<h1>debug_page.html not found</h1>", status_code=500)
    template = html_path.read_text(encoding="utf-8")
    return HTMLResponse(template.replace("{{SESSION_ID}}", session_id))


@app.get("/session/{session_id}/photo/{node_id}.jpg")
def serve_annotated_photo(session_id: str, node_id: int):
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    out_dir = ensure_output_dir(session_id)
    p = out_dir / "annotated" / f"{node_id}.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "photo_not_found", "detail": str(p)})
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/session/{session_id}/photo/{node_id}/{pass_name}.jpg")
def serve_pass_photo(session_id: str, node_id: int, pass_name: str):
    """Serve pass1 / pass2 annotated photos."""
    if pass_name not in ("pass1", "pass2"):
        raise HTTPException(status_code=400, detail={"error": "bad_pass", "detail": pass_name})
    out_dir = ensure_output_dir(session_id)
    p = out_dir / "annotated" / f"{node_id}_{pass_name}.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "photo_not_found", "detail": str(p)})
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/session/{session_id}/original/{node_id}.jpg")
def serve_original_photo(session_id: str, node_id: int):
    """Serve the original uploaded photo (before annotation)."""
    out_dir = ensure_output_dir(session_id)
    p = out_dir / "photo" / f"{node_id}.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "photo_not_found"})
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/reference-photo/{nid}.jpg")
def serve_reference_photo(nid: int):
    """Serve a reference node's photo from the explore session output."""
    neo4j = get_neo4j()
    if not neo4j:
        raise HTTPException(status_code=503, detail="neo4j unavailable")
    ref_map = neo4j.load_reference_map()
    if nid not in ref_map.photos:
        raise HTTPException(status_code=404, detail=f"node {nid} not found")
    photo_file = ref_map.photos[nid].photo_file
    session_name = ref_map.photos[nid].session
    candidates = [
        OUTPUT_ROOT / session_name / "photo" / photo_file,
        OUTPUT_ROOT / session_name / "annotated" / photo_file,
    ]
    for p in candidates:
        if p.exists():
            return FileResponse(str(p), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail=f"photo file not found: {photo_file}")


def _ws(place: str) -> Path:
    """The store's map-building workspace, MAP_BUILD_ROOT/<店名>/ (see server/map_build.py)."""
    from server.map_build import workspace_for
    try:
        return workspace_for(place, _config.MAP_BUILD_ROOT)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _read_topomap(ws: Path) -> Optional[dict]:
    p = ws / "topomap.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ── Start points (server/start_points.py): where a walk begins and which way to face ──

def _start_payload(place: str, s: dict) -> dict:
    from urllib.parse import quote
    out = {k: s[k] for k in ("id", "name", "x", "y", "heading_deg")}
    out["photo_url"] = (f"/starts/photo?place={quote(place)}&id={quote(s['id'])}"
                        if start_points.photo_path(place, s["id"], EDITOR_MAP_DIR) else None)
    return out


@app.get("/starts")
def get_starts(place: str = Query(...)):
    """The store's start points for the app's 「我在起點」 (empty list when none are set)."""
    return {"place": place,
            "starts": [_start_payload(place, s) for s in start_points.load_starts(place, EDITOR_MAP_DIR)]}


@app.get("/starts/photo")
def get_start_photo(place: str = Query(...), id: str = Query(...)):
    p = start_points.photo_path(place, id, EDITOR_MAP_DIR)
    if p is None:
        raise HTTPException(status_code=404, detail=f"no photo for start {id!r}")
    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/starts")
def save_starts(payload: dict = Body(...), place: str = Query(...)):
    """Replace the store's start points (the map builder's 「起點」 panel)."""
    starts = payload.get("starts")
    if not isinstance(starts, list):
        raise HTTPException(status_code=422, detail="body needs a 'starts' list")
    try:
        ws = _ws(place)
    except HTTPException:
        ws = None
    try:
        saved = start_points.save_starts(place, starts, EDITOR_MAP_DIR,
                                         workspace=ws if ws is not None and ws.is_dir() else None)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"place": place, "starts": [_start_payload(place, s) for s in saved]}


@app.get("/test-photo/{set_id}/{filename}")
def serve_test_photo(set_id: str, filename: str, place: str = Query(...)):
    """A walk's photo from the store's workspace (<workspace>/<set_id>/<filename>)."""
    base = _ws(place).resolve()
    p = (base / set_id / filename).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(status_code=404, detail=f"not found: {set_id}/{filename}")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/topomap-builder")
def topomap_builder():
    """Interactive topological map builder page (?place=<店名> picks the store)."""
    html_path = Path(__file__).parent / "topomap_builder.html"
    return FileResponse(str(html_path), media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/topomap-places")
def topomap_places():
    """Stores that have a map-building workspace, newest first."""
    from server.map_build import list_places
    return {"root": str(_config.MAP_BUILD_ROOT), "places": list_places(_config.MAP_BUILD_ROOT)}


@app.get("/existing-topomap")
def existing_topomap():
    """Reference map drawn under the builder: the Neo4j place MAP_BUILD_REF_PLACE (if set)."""
    place = _config.MAP_BUILD_REF_PLACE
    if not place:
        raise HTTPException(status_code=404, detail="MAP_BUILD_REF_PLACE not set")
    neo4j = get_neo4j()
    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not configured or not connected")
    ref = neo4j.load_reference_map(place)
    nodes = [{"id": p.nid, "x": p.pdr_x, "y": p.pdr_y, "sn": p.session,
              "labels": [o.ocr_text for o in p.objects if o.ocr_text][:3]}
             for p in ref.photos.values()]
    edges = [{"a": e["from"], "b": e["to"]} for e in ref.walkway_edges]
    return JSONResponse({"nodes": nodes, "edges": edges})


@app.get("/topomap-data")
def topomap_data(place: str = Query(...)):
    """The builder's input: <workspace>/topomap.json (from 「路線」 or build_map_v2_topomap.py)."""
    topo = _read_topomap(_ws(place))
    if topo is None:
        raise HTTPException(status_code=404, detail=f"「{place}」還沒有路線：按「路線…」選要匯入的 Sensor Lab 資料")
    return JSONResponse(topo)


@app.get("/topomap-sources")
def topomap_sources(place: str = Query("")):
    """Sensor Lab exports the builder can import, and the walks the store's workspace uses now
    (no place yet = a new store: nothing in use)."""
    from server.map_build import list_sources
    ws = _ws(place) if place.strip() else None
    topo = (_read_topomap(ws) if ws else None) or {}
    walks = topo.get("walks") or [{"set": s, "source": "", "nodes": n}
                                  for s, n in (topo.get("stats", {}).get("sets") or {}).items()]
    return {"place": topo.get("place", place), "walks": walks,
            "roots": [str(r) for r in _config.MAP_BUILD_SOURCES],
            "sources": list_sources(_config.MAP_BUILD_SOURCES, ws)}


@app.post("/topomap-sources/upload")
async def topomap_sources_upload(file: UploadFile = File(...), place: str = Query(...)):
    """A Sensor Lab .zip uploaded from the builder → <workspace>/imports/<name>.zip."""
    import io
    import re as _re
    import zipfile
    from server.map_build import CSV_NAME, IMPORTS_DIR
    data = await file.read()
    if len(data) > 1024 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="zip 超過 1 GB")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if not any(n.endswith(CSV_NAME) for n in zf.namelist()):
                raise HTTPException(status_code=422, detail=f"zip 裡沒有 {CSV_NAME}（不是 Sensor Lab 匯出）")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=422, detail="不是 zip 檔")
    folder = _ws(place) / IMPORTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stem = _re.sub(r'[\\/:*?"<>|]+', "_", Path(file.filename or "walk").stem).strip() or "walk"
    out, k = folder / f"{stem}.zip", 1
    while out.exists():
        k += 1
        out = folder / f"{stem}_{k}.zip"
    out.write_bytes(data)
    return {"path": str(out.resolve()), "name": out.stem}


@app.post("/topomap-walks")
def topomap_walks(payload: dict = Body(...)):
    """(Re)build the store's topomap.json from the chosen Sensor Lab exports (paths from
    /topomap-sources). A store name with no workspace yet starts a new one. Walks already in
    the workspace keep their set id; new ones get the next unused id."""
    from server.map_build import assign_sets, build_workspace, list_sources
    place = str(payload.get("place") or "").strip()
    chosen = [str(p) for p in payload.get("sources") or []]
    if not place:
        raise HTTPException(status_code=422, detail="請填店名（Neo4j place）")
    if not chosen:
        raise HTTPException(status_code=422, detail="至少選一條路線")
    ws = _ws(place)
    # the store's own uploads, plus zips uploaded under the store the dialog was opened from
    dirs = {ws} | ({_ws(str(payload["from_place"]))} if str(payload.get("from_place") or "").strip() else set())
    allowed = {e["path"] for d in dirs for e in list_sources(_config.MAP_BUILD_SOURCES, d)}
    bad = [p for p in chosen if str(Path(p).resolve()) not in allowed]
    if bad:
        raise HTTPException(status_code=422, detail=f"不在可選的來源裡：{bad[0]}")
    walks, next_set = assign_sets(_read_topomap(ws), [Path(p) for p in chosen])
    try:
        topo, report = build_workspace(place, walks, ws, next_set)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    log.info("topomap rebuilt for %s: %s", place, " | ".join(report))
    return {"report": report, "nodes": len(topo["nodes"]), "walks": topo.get("walks", []),
            "place": place, "folder": ws.name}


@app.get("/topomap-suggest")
def topomap_suggest(place: str = Query(...)):
    """DINOv2 merge suggestions for the builder (loop closures, same spot on two walks, a
    waypoint on another walk's edge). Photo embeddings are cached in
    <workspace>/dino_embeddings.pkl ("<set>/<file>" → vector); the first call embeds every
    photo (about 1 s each on CPU)."""
    import pickle
    from server import dino_embed
    from server.map_v2 import photo_key, suggest_merges

    ws = _ws(place)
    topo = _read_topomap(ws)
    if topo is None:
        raise HTTPException(status_code=404, detail=f"「{place}」還沒有路線")
    nodes = topo["nodes"]
    cache_path = ws / "dino_embeddings.pkl"
    emb = {}
    if cache_path.exists():
        try:
            emb = pickle.loads(cache_path.read_bytes())
        except Exception:
            log.warning("unreadable %s — re-embedding", cache_path)
    todo = [(photo_key(n, s), ws / n["set"] / n["photos"][s]) for n in nodes for s in n["photos"]]
    todo = [(k, p) for k, p in todo if k not in emb and p.is_file()]
    if todo:
        if not dino_embed.available():
            raise HTTPException(status_code=503, detail="DINOv2 無法載入（需要 torch + transformers）")
        t0 = time.time()
        for k, p in todo:
            v = dino_embed.embed_image(str(p))
            if v is not None:
                emb[k] = v
        cache_path.write_bytes(pickle.dumps(emb))
        log.info("topomap-suggest: embedded %d photos in %.0fs", len(todo), time.time() - t0)
    out = suggest_merges(nodes, emb)
    out["embedded"] = len(todo)
    return out


_FLOORPLAN_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


def _floorplan_file(ws: Path) -> Optional[Path]:
    return next((p for p in ws.glob("floorplan.*") if p.suffix.lower() in _FLOORPLAN_TYPES.values()), None)


@app.post("/topomap-floorplan")
async def topomap_floorplan_upload(file: UploadFile = File(...), place: str = Query(...)):
    """Floor-plan image drawn under the builder; kept as <workspace>/floorplan.<ext>."""
    ext = _FLOORPLAN_TYPES.get((file.content_type or "").lower())
    if not ext:
        raise HTTPException(status_code=415, detail="請用 PNG / JPG / WebP / GIF 圖檔")
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="圖檔超過 30 MB")
    base = _ws(place)
    base.mkdir(parents=True, exist_ok=True)
    old = _floorplan_file(base)
    if old:
        old.unlink()
    (base / f"floorplan{ext}").write_bytes(data)
    return {"ok": True}


@app.get("/topomap-floorplan")
def topomap_floorplan(place: str = Query(...)):
    p = _floorplan_file(_ws(place))
    if not p:
        raise HTTPException(status_code=404, detail="no floor plan")
    return FileResponse(str(p), headers={"Cache-Control": "no-store"})


@app.delete("/topomap-floorplan")
def topomap_floorplan_delete(place: str = Query(...)):
    p = _floorplan_file(_ws(place))
    if p:
        p.unlink()
    return {"ok": True}


@app.post("/topomap-confirm")
async def topomap_confirm(payload: dict = Body(...)):
    """Save the builder's result to <workspace>/topomap_confirmed.json for upload_map_v2.py."""
    place = str(payload.get("place") or "").strip()
    out = _ws(place) / "topomap_confirmed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    n_nodes = len(payload.get("nodes", []))
    n_edges = len(payload.get("edges", []))
    log.info("Topomap confirmed: %s nodes=%d edges=%d → %s", place, n_nodes, n_edges, out)
    return {"message": f"已儲存到 {out.parent.name}/{out.name}（{n_nodes} 個節點、{n_edges} 條邊）。"
                       f"下一步：python upload_map_v2.py --place \"{place}\" --dry-run",
            "path": str(out)}
