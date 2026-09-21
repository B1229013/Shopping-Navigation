"""FastAPI app — endpoints for in-store navigation sessions."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps

from server import scene
from server.annotator import annotate
from server.graph_renderer import (
    render_goal_graph, render_scene_graph, assign_detection_ids,
    postprocess_detections, augment_goal_classes,
)
from server.config import (
    PERCEPTION_ENABLED, OCR_ENABLED, OCR_LANGUAGES,
    OCR_MIN_CONFIDENCE, OCR_MAX_RESULTS, ARRIVED_MIN_DETECTION_SCORE,
    GOAL_CROP_VERIFY, GENERIC_INDOOR_OBJECTS, ensure_output_dir, OUTPUT_ROOT,
)
from server.sensor_test_render import render_sensor_test_png
from server.goal_decomposer import decompose_goal
from server.ocr import OCR
from server.models import (
    AnswerRequest,
    ErrorResponse,
    ConfirmArrivalRequest,
    ModifyRouteRequest,
    PlanRouteRequest,
    PlanRouteResponse,
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
    perceive_and_decide as _vlm_perceive_and_decide,
    warm_up as _vlm_warm_up,
    ask_about_image as _vlm_ask_about_image,
)
from server.neo4j_client import get_neo4j
from server.visual_localization import localize as _localize_photo, vlm_rerank as _vlm_rerank
from server.heading import (
    convert_leg_to_relative,
    relative_direction_text,
    heading_between_nodes,
)
from server.prompts import ROUTE_CONTEXT_BLOCK

log = logging.getLogger(__name__)

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

def _build_nav_graph(ref_map) -> "nx.Graph":
    """Build a networkx graph from the reference map, auto-adding proximity edges."""
    import networkx as _nx
    import math as _m
    g = _nx.Graph()
    for nid in ref_map.photos:
        g.add_node(nid)
    for edge in ref_map.walkway_edges:
        w = edge.get("distance_m", 1.0) or 1.0
        g.add_edge(edge["from"], edge["to"], weight=w)
    # Auto-connect nearby nodes that aren't already linked (< 6m)
    nodes = list(ref_map.photos.items())
    added = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a_id, a = nodes[i]
            b_id, b = nodes[j]
            if g.has_edge(a_id, b_id):
                continue
            dx = a.pdr_x - b.pdr_x
            dy = a.pdr_y - b.pdr_y
            dist = _m.sqrt(dx * dx + dy * dy)
            if dist < 6.0:
                g.add_edge(a_id, b_id, weight=max(dist, 0.5))
                added += 1
    if added:
        log.info("Auto-added %d proximity edges (< 6m)", added)
    return g


def _is_useless_name(name: str) -> bool:
    n = name.strip().lower()
    return (n in _USELESS_LANDMARK
            or "促銷" in n or "promotional" in n
            or "海報" in n or "poster" in n)


def _node_area_name(nid: int, ref_map, goal_keywords=None) -> str:
    """Generate a human-friendly area name for a map node from its OCR/objects.

    When *goal_keywords* is provided (for the target node), prioritize objects
    whose label or OCR text matches the goal so the area name is relevant.
    """
    if not ref_map or nid not in ref_map.photos:
        return "未知區域"
    node = ref_map.photos[nid]
    if not node.objects:
        return "未知區域"

    # Priority 1: aisle/section signs with actual aisle numbers — most useful for navigation
    import re as _re
    for obj in node.objects:
        lbl = obj.label
        raw = obj.ocr_text.strip() if obj.ocr_text else lbl
        # Match "走道N" with category info (e.g. "走道14／調味／沖泡食品標示牌")
        _m = _re.search(r'走道\s*\d+[／/][^A-Za-z]+', raw) or _re.search(r'走道\s*\d+[／/][^A-Za-z]+', lbl)
        if _m:
            aisle_name = _m.group(0).rstrip("標示牌 ／/")
            if aisle_name:
                return f"「{aisle_name}」"
        # Match "Aisle N" style with category
        _m2 = _re.search(r'走道\s*\d+', raw) or _re.search(r'走道\s*\d+', lbl)
        if _m2:
            return f"「{_m2.group(0)}」"

    # Priority 2: goal-keyword matched objects
    kws = [k.lower() for k in (goal_keywords or []) if k]
    if kws:
        for obj in node.objects:
            lbl = obj.label.lower()
            ocr = (obj.ocr_text or "").lower()
            for k in kws:
                if k in lbl or k in ocr or lbl in k:
                    name = obj.ocr_text.strip() if obj.ocr_text else obj.label.split("/")[0].strip()
                    if name and not _is_useless_name(name):
                        return f"「{name}」區"

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

_CHECKOUT_KW = ['收銀台', '收銀機', '結帳機', 'pos', 'cashier counter', 'cash register',
                '自助結帳機', 'self-checkout kiosk']
_CHECKOUT_SIGN_KW = ['結帳', '收銀', 'cashier', 'checkout']
_EXIT_DOOR_KW = ['出口門', '玻璃門', 'entrance door', 'exit door']
_EXIT_SIGN_KW = ['出口', 'exit']


def _find_checkout_exit_nodes(ref_map) -> tuple[int | None, int | None]:
    """Auto-detect best checkout and exit nodes from reference map objects."""
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

    checkout_nid = max(checkout_scores, key=checkout_scores.get) if checkout_scores else None
    exit_nid = max(exit_scores, key=exit_scores.get) if exit_scores else None

    if checkout_nid is not None:
        log.info("Auto-detected checkout node: WP%d (score=%.1f)", checkout_nid, checkout_scores[checkout_nid])
    if exit_nid is not None:
        log.info("Auto-detected exit node: WP%d (score=%.1f)", exit_nid, exit_scores[exit_nid])

    return checkout_nid, exit_nid


def _detect_goal_in_photo(detections, goal_keywords, img_w=1, img_h=1, ocr_results=None):
    """Check if goal-related objects are visible in the user's photo.

    Uses bounding box position to determine left/center/right.
    Also checks OCR text for goal keyword matches.
    Returns (direction_zh, object_names) or (None, []).
    """
    if not goal_keywords:
        return None, []

    keywords = [kw.lower().strip() for kw in goal_keywords if kw]
    matches = []

    for det in (detections or []):
        label = (det.label if isinstance(det.label, str) else str(det.label)).lower()
        box = det.box if hasattr(det, 'box') else det.get('box')
        if not box or len(box) < 4:
            continue
        for kw in keywords:
            if kw in label or label in kw:
                cx = (box[0] + box[2]) / 2
                matches.append((cx, det.label if isinstance(det.label, str) else str(det.label)))
                break
            elif len(kw) >= 2 and label[:2] == kw[:2] and _re.search(r'[一-鿿]', kw[:2]):
                cx = (box[0] + box[2]) / 2
                matches.append((cx, det.label if isinstance(det.label, str) else str(det.label)))
                break

    for ocr in (ocr_results or []):
        text = getattr(ocr, 'text', '') or ''
        text_lower = text.lower().strip()
        bbox = getattr(ocr, 'bbox', None)
        if not text_lower or not bbox:
            continue
        if isinstance(bbox[0], (list, tuple)):
            cx = sum(p[0] for p in bbox) / len(bbox)
        elif len(bbox) >= 4:
            cx = (bbox[0] + bbox[2]) / 2
        else:
            continue
        for kw in keywords:
            if kw in text_lower or text_lower in kw:
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
            for edge in ref_map.walkway_edges:
                neighbor_nid = None
                if edge.get('from') == current_nid:
                    neighbor_nid = edge['to']
                elif edge.get('to') == current_nid:
                    neighbor_nid = edge['from']
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


def _build_route_context(s, loc_result, detections=None, ocr_results=None) -> tuple[str | None, str | None]:
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

    # Position description
    if localized:
        friendly_loc = _ref_location_name(loc_result) or "未知區域"
        position_desc = f"{friendly_loc}（信心 {loc_result.confidence:.0%}）"
    else:
        position_desc = "定位中（尚未確認確切位置）"

    # Heading description
    heading_ok = (localized
                  and loc_result.matched_heading is not None
                  and loc_result.heading_confidence > 0.5
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

        # Re-plan route when user is localized but NOT on the current leg's path
        if (localized and ref_map
                and loc_result.matched_nid not in current_leg.path):
            try:
                from server.path_planner import plan_route as _plan_route
                _g = _build_nav_graph(ref_map)
                if loc_result.matched_nid in _g:
                    _remaining_targets = [t for t in (s.target_nodes or [])
                                          if t not in (s.visited_targets or set())]
                    if not _remaining_targets and current_leg.purpose == "target":
                        _remaining_targets = [current_leg.to_node]
                    new_route = _plan_route(
                        _g, loc_result.matched_nid, _remaining_targets,
                        checkout=getattr(s, "checkout_node", None),
                        exit_node=getattr(s, "exit_node", None),
                    )
                    s.route_plan = new_route
                    s.current_leg_index = 0
                    leg_idx = 0
                    current_leg = new_route.legs[0]
                    log.info("🔄 [session %s] re-planned route from WP%d (was not on leg path %s)",
                             s.id, loc_result.matched_nid, current_leg.path[:5])
            except Exception as e:
                log.warning("[session %s] route re-plan failed: %s", s.id, e)

        # Friendly name for the target area — depends on current phase
        if s.phase == "checkout":
            target_area = "收銀台"
        elif s.phase == "exit":
            target_area = "出口"
        else:
            target_area = _node_area_name(current_leg.to_node, ref_map,
                                           goal_keywords=s.goal_objects)

        # Which goal item is at this target? Use user's original input language
        if s.phase == "checkout":
            goal_item = "收銀台"
        elif s.phase == "exit":
            goal_item = "出口"
        elif current_leg.purpose in ("checkout", "exit"):
            goal_item = "收銀台" if current_leg.purpose == "checkout" else "出口"
        else:
            # Use the user's original goal text (e.g., "泡麵") not English decomposed labels
            goal_item = s.goal.split(",")[0].strip() if s.goal else current_leg.purpose
            import re as _re_goal
            goal_item = _re_goal.sub(r'\s*x\d+\s*$', '', goal_item).strip() or goal_item

        # --- Priority 1: check if goal is visible in the user's photo ---
        goal_kws = [goal_item] + (s.goal_objects or [])
        photo_dir_zh, photo_obj_names = _detect_goal_in_photo(
            detections, goal_kws, ocr_results=ocr_results)

        direction_hint = ""
        if photo_dir_zh:
            obj_text = "、".join(photo_obj_names[:3])
            direction_hint = (
                f"重要：使用者的照片中已經可以看到目標相關物件（{obj_text}），"
                f"位於畫面的「{photo_dir_zh}」。請直接引導使用者往那個方向走。\n"
            )
        elif localized:
            # --- Priority 2: infer from directional reference photos (needs localization) ---
            matched_node = ref_map.photos.get(loc_result.matched_nid) if ref_map else None
            dir_zh, dir_slot, dir_landmarks = _infer_target_direction(
                matched_node, current_leg.to_node, ref_map, loc_result.matched_slot,
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
            # Not localized — can't infer direction from map, but still guide toward target
            direction_hint = (
                f"目前尚未精確定位，無法從地圖推斷方向。"
                f"請根據照片中的標示和地標引導使用者前往「{goal_item}」所在的「{target_area}」。\n"
            )

        heading_reliable = (localized
                            and ref_map
                            and loc_result.matched_heading is not None
                            and loc_result.heading_confidence > 0.5
                            and loc_result.confidence >= 0.55)

        if heading_reliable:
            node_positions = {
                nid: (node.pdr_x, node.pdr_y)
                for nid, node in ref_map.photos.items()
            }

            rel_instructions = convert_leg_to_relative(
                current_leg.path,
                node_positions,
                loc_result.matched_heading,
            )

            if rel_instructions:
                # Build condensed multi-step instruction:
                # merge consecutive 直走, mention only useful landmarks at turns
                _steps = []
                _straight_count = 0
                _straight_areas = []
                for ri in rel_instructions:
                    area = _node_area_name(ri.to_node, ref_map)
                    is_useful_area = ("走道" in area or "結帳" in area or "出口" in area
                                      or "收銀" in area)
                    if ri.text_zh == "直走":
                        _straight_count += 1
                        if is_useful_area:
                            _straight_areas.append(area.strip("「」"))
                    else:
                        if _straight_count > 0:
                            if _straight_areas:
                                _steps.append(f"直走經過{'、'.join(_straight_areas[-2:])}")
                            else:
                                _steps.append("直走")
                            _straight_count = 0
                            _straight_areas = []
                        if is_useful_area:
                            # Simplify area for turn instructions: strip brackets, keep first term
                            _clean = area.replace("「", "").replace("」", "").split("附近")[0].strip()
                            _steps.append(f"{ri.text_zh}往{_clean}")
                        else:
                            _steps.append(ri.text_zh)
                if _straight_count > 0:
                    if _straight_areas:
                        _steps.append(f"直走經過{'、'.join(_straight_areas[-2:])}")
                    else:
                        _steps.append("直走")

                steps_text = "→".join(_steps[:6])
                if s.phase in ("checkout", "exit"):
                    next_instruction = f"{steps_text}，到{target_area}"
                else:
                    next_instruction = f"{steps_text}，到{target_area}找「{goal_item}」"

                # Also build path_steps for VLM context
                path_steps = []
                for ri in rel_instructions[:5]:
                    step_area = _node_area_name(ri.to_node, ref_map)
                    path_steps.append(f"{ri.text_zh}往{step_area}方向走")

                if s.phase == "checkout":
                    phase_status = "所有商品已找到，正前往收銀台"
                elif s.phase == "exit":
                    phase_status = "已結帳完成，正前往出口"
                else:
                    phase_status = f"還有 {len(remaining)} 項商品待尋找"

                route_desc = (
                    f"目前目標：「{goal_item}」\n"
                    f"位於：{target_area}\n"
                    f"{direction_hint}"
                    f"建議走法：{'；'.join(path_steps)}\n"
                    f"{phase_status}"
                )
            else:
                if s.phase == "checkout":
                    phase_status = "所有商品已找到，正前往收銀台"
                elif s.phase == "exit":
                    phase_status = "已結帳完成，正前往出口"
                else:
                    phase_status = f"還有 {len(remaining)} 項商品待尋找"
                route_desc = (
                    f"目前目標：「{goal_item}」\n"
                    f"位於：{target_area}\n"
                    f"{direction_hint}"
                    f"{phase_status}"
                )
        else:
            if s.phase == "checkout":
                phase_status = "所有商品已找到，正前往收銀台"
            elif s.phase == "exit":
                phase_status = "已結帳完成，正前往出口"
            else:
                phase_status = f"還有 {len(remaining)} 項商品待尋找"
            route_desc = (
                f"目前目標：「{goal_item}」\n"
                f"位於：{target_area}\n"
                f"{direction_hint}"
                f"{phase_status}"
            )

        # Fallback next_instruction: only when localized (we know where the user is)
        # Without localization, the fallback is meaningless — let VLM guide by photo
        if not next_instruction and localized:
            if s.phase in ("checkout", "exit"):
                next_instruction = f"前往{target_area}"
            else:
                next_instruction = f"前往{target_area}找「{goal_item}」"
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


def _current_goal_objects(s) -> list[str]:
    """Return only the goal object(s) for the current route leg.

    When a route plan exists, we guide the user one item at a time.
    In checkout/exit phases, return checkout/exit related objects.
    When there's no route plan, fall back to all goal objects.
    """
    if s.phase == "checkout":
        return ["收銀台", "結帳", "cashier", "checkout"]
    if s.phase == "exit":
        return ["出口", "exit"]

    if not s.route_plan or not s.route_plan.legs or not s.goal_objects:
        return s.goal_objects

    leg_idx = min(s.current_leg_index, len(s.route_plan.legs) - 1)
    current_leg = s.route_plan.legs[leg_idx]
    target_nid = current_leg.to_node

    neo4j = get_neo4j()
    ref_map = neo4j.load_reference_map(s.place) if neo4j and s.place else None
    if not ref_map or target_nid not in ref_map.photos:
        return s.goal_objects

    node = ref_map.photos[target_nid]
    for g in s.goal_objects:
        for obj in node.objects:
            if (obj.ocr_text and g.lower() in obj.ocr_text.lower()) or \
               g.lower() in obj.label.lower():
                return [g]
    return s.goal_objects[:1]


def _build_localization_items(detections, ocr_results, img_w, img_h):
    """Build detected_items (with normalized [0,1] bboxes) for grid matching."""
    items = []
    for d in detections:
        box = getattr(d, 'box', None) or getattr(d, 'bbox', None)
        if box and len(box) == 4 and not isinstance(box[0], (list, tuple)):
            norm_box = [box[0] / img_w, box[1] / img_h,
                        box[2] / img_w, box[3] / img_h]
            items.append({"label": d.label, "box": norm_box})
        else:
            items.append({"label": d.label, "box": None})
    for r in ocr_results:
        bbox = getattr(r, 'bbox', None)
        text = getattr(r, 'text', "")
        if not text:
            continue
        if bbox:
            if isinstance(bbox[0], (list, tuple)):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                norm_box = [min(xs) / img_w, min(ys) / img_h,
                            max(xs) / img_w, max(ys) / img_h]
            elif len(bbox) == 4:
                norm_box = [bbox[0] / img_w, bbox[1] / img_h,
                            bbox[2] / img_w, bbox[3] / img_h]
            else:
                norm_box = None
            items.append({"label": text, "box": norm_box})
        else:
            items.append({"label": text, "box": None})
    return items


def _run_early_localization(s, session_id, detected_labels, ocr_texts,
                            detected_items=None, image_path=None):
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
    try:
        loc_result = _localize_photo(
            detected_labels=detected_labels,
            ocr_texts=ocr_texts,
            neo4j=neo4j,
            hint_nid=getattr(s, "last_corrected_nid", None),
            place=s.place,
            detected_items=detected_items,
        )

        ref_map = neo4j.load_reference_map(s.place)

        if (RERANK_ENABLED and image_path and REF_PHOTO_ROOT
                and loc_result.matched_nid is not None
                and loc_result.top_candidates):
            top = loc_result.top_candidates
            log.info("[session %s] rerank triggered (always-on): WP%d %.3f",
                     session_id, top[0][0], top[0][1])
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
            if reranked and reranked[0][0] != loc_result.matched_nid:
                new_nid = reranked[0][0]
                new_ref = ref_map.photos.get(new_nid)
                log.info("[session %s] rerank changed: WP%d → WP%d",
                         session_id, loc_result.matched_nid, new_nid)
                loc_result.matched_nid = new_nid
                loc_result.ref_node = new_ref
                loc_result.method = "grid+rerank"
                loc_result.reasoning = reranked[0][2]

                # Recompute heading for the NEW node's directional photos
                if new_ref and new_ref.directional_photos:
                    from server.heading import estimate_heading, best_matching_slot, DirectionalPhoto
                    _live_labels = {l.lower().strip() for l in detected_labels if l}
                    _live_ocr = {t.lower().strip() for t in ocr_texts if t}
                    _dir_photos = [
                        DirectionalPhoto(
                            slot=dp.slot, heading_deg=dp.heading_deg,
                            photo_file=dp.photo_file,
                            objects=[{"label": o.label, "ocr_text": o.ocr_text}
                                     for o in dp.objects],
                        )
                        for dp in new_ref.directional_photos
                    ]
                    new_heading = estimate_heading(_live_labels, _live_ocr, _dir_photos)
                    new_slot = best_matching_slot(_live_labels, _live_ocr, _dir_photos)
                    new_slot_str = new_slot if isinstance(new_slot, str) else (
                        new_slot.value if new_slot else None)
                    all_h = [dp.heading_deg for dp in new_ref.directional_photos]
                    has_real = len(set(all_h)) > 1
                    if has_real and new_heading is not None:
                        loc_result.matched_heading = new_heading
                        loc_result.matched_slot = new_slot_str
                        unique_sets = len({
                            frozenset(o.label for o in dp.objects)
                            for dp in new_ref.directional_photos
                        })
                        loc_result.heading_confidence = min(loc_result.confidence, 0.8) if unique_sets > 1 else 0.2
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
    except Exception as e:
        log.warning("[session %s] localization error: %s", session_id, e)
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
    s = _store.create(goal=req.goal, goal_objects=goal_objects, place=selected_place)

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

                # Search Neo4j map: match each goal item against node OCR/objects
                target_nodes: list[int] = []
                for item in goal_items:
                    item_lower = item.lower()
                    best_nid = None
                    best_score = 0.0
                    for nid, node in ref_map.photos.items():
                        for obj in node.objects:
                            score = 0.0
                            # OCR exact/partial match
                            if obj.ocr_text:
                                ocr_lower = obj.ocr_text.lower()
                                if item_lower == ocr_lower:
                                    score = 1.0
                                elif item_lower in ocr_lower or ocr_lower in item_lower:
                                    score = 0.8
                            # Object label match
                            label_lower = obj.label.lower()
                            if item_lower in label_lower or label_lower in item_lower:
                                score = max(score, 0.7)
                            if score > best_score:
                                best_score = score
                                best_nid = nid
                    if best_nid is not None and best_score >= 0.5:
                        if best_nid not in target_nodes:
                            target_nodes.append(best_nid)
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
                    # Build networkx graph from Neo4j walkway edges + proximity
                    from server.path_planner import plan_route as _plan_route
                    neo4j_graph = _build_nav_graph(ref_map)
                    if neo4j_graph.number_of_nodes() > 0:
                        # Use node 0 as default start; if missing, use smallest nid
                        start_nid = 0 if 0 in neo4j_graph else min(neo4j_graph.nodes())
                        # Filter targets to only nodes that exist in graph
                        valid_targets = [t for t in target_nodes if t in neo4j_graph]

                        # Auto-detect checkout and exit nodes
                        checkout_nid, exit_nid = _find_checkout_exit_nodes(ref_map)
                        if checkout_nid and checkout_nid not in neo4j_graph:
                            checkout_nid = None
                        if exit_nid and exit_nid not in neo4j_graph:
                            exit_nid = None
                        s.checkout_node = checkout_nid
                        s.exit_node = exit_nid

                        if valid_targets:
                            route = _plan_route(
                                neo4j_graph, start_nid, valid_targets,
                                checkout=checkout_nid, exit_node=exit_nid,
                            )
                            s.target_nodes = valid_targets
                            s.visited_targets = set()
                            s.route_plan = route
                            s.current_leg_index = 0
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

    return StartSessionResponse(
        session_id=s.id,
        guidance="Upload a starting photo so I can see where you are.",
        goal_objects=goal_objects,
        place=selected_place,
    )


@app.post("/session/{session_id}/photo", response_model=TurnResponse)
async def upload_photo(session_id: str, photo: UploadFile = File(...)) -> TurnResponse:
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if s.arrived:
        raise HTTPException(status_code=409, detail={"error": "already_arrived", "detail": "session is complete"})
    if s.pending_arrival:
        raise HTTPException(status_code=409, detail={"error": "confirm_pending", "detail": "POST /confirm first"})
    if s.pending_question is not None:
        raise HTTPException(status_code=409, detail={"error": "answer_pending", "detail": "POST /answer first"})

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
        combined_prompts = list(dict.fromkeys(s.goal_objects + GENERIC_INDOOR_OBJECTS))
        log.info("GroundingDINO prompt: %s", combined_prompts)
        detections = perception_engine.detect(str(photo_path), combined_prompts)
        detected_labels = [d.label for d in detections]
        t_detect = time.time()
        log.info("⏱ GroundingDINO: %.1fs", t_detect - t_ocr)

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

        detections_summary = scene.format_detections(detections, img_w, img_h)
        topomap_summary = s.topomap.summarize_for_vlm(current_id=nid)

        log.info("[session %s] photo #%d | detections: %s | ocr: %s",
                 session_id, nid, detected_labels or "(none)", ocr_text_summary[:80])

        # ── Early localization for route-aware VLM prompt ──────────────
        _loc_items = _build_localization_items(detections, ocr_results or [], img_w, img_h)
        _early_loc = _run_early_localization(s, session_id, detected_labels, ocr_texts,
                                             detected_items=_loc_items,
                                             image_path=str(photo_path))
        _route_ctx, _next_instr = _build_route_context(s, _early_loc, detections=detections, ocr_results=ocr_results)

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
                matches = scene.match_ocr_to_goal(easyocr_results, s.goal_objects)
                if matches:
                    summary += "  SIGN MATCH: " + "; ".join(matches)
                return summary
            raw = [OCRResult(text=t.text, confidence=t.score, bbox=t.bbox)
                   for t in vlm_ocr_items]
            cleaned = _clean_vlm_ocr(raw, img_h)
            if not cleaned:
                return "(no text detected)"
            summary = scene.format_ocr(cleaned, img_w, img_h)
            matches = scene.match_ocr_to_goal(cleaned, s.goal_objects)
            if matches:
                summary += "  SIGN MATCH: " + "; ".join(matches)
            return summary

        # ── Stage 1: VLM perceive ──────────────────────────────────────
        _active_goals = _current_goal_objects(s)
        _active_goal_str = "找到：" + "、".join(_active_goals)
        t_vlm_start = time.time()
        vlm_perception = _vlm_perceive(
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

        # ── Localize with detection labels + OCR BEFORE decide ────────
        _loc_items = _build_localization_items(detections, ocr_results, img_w, img_h)
        _early_loc = _run_early_localization(s, session_id, detected_labels, ocr_texts,
                                             detected_items=_loc_items,
                                             image_path=str(photo_path))
        _route_ctx, _next_instr = _build_route_context(s, _early_loc, detections=detections, ocr_results=ocr_results)

        if _route_ctx:
            log.info("🟢 [session %s] Mode B: VLM will receive route context (%d chars), next_instr=%s",
                     session_id, len(_route_ctx), _next_instr[:80] if _next_instr else "(none)")
        else:
            log.warning("🔴 [session %s] Mode B: VLM will receive NO route context — navigating blind", session_id)

        # ── Stage 2: VLM decide (with map-aware route context) ────────
        ocr_text_summary = scene.format_ocr(ocr_results, img_w, img_h) if ocr_results else "(no text detected)"
        ocr_matches = scene.match_ocr_to_goal(ocr_results, s.goal_objects)
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
    if GOAL_CROP_VERIFY and vlm_resp.action == VLMAction.ARRIVED:
        candidates = _top_goal_detections(detections, s.goal_objects)
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
        vlm_resp, detections, ocr_matches=ocr_matches, goal_objects=s.goal_objects,
        min_score=ARRIVED_MIN_DETECTION_SCORE, img_w=img_w, img_h=img_h,
        goal_verified=goal_verified,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action != VLMAction.ARRIVED
    if _raw_action == VLMAction.ARRIVED:
        log.info("[session %s] verify_arrival: %s → %s | goal_verified=%s ocr_matches=%s",
                 session_id, _raw_action.value, vlm_resp.action.value,
                 goal_verified, bool(ocr_matches))
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
    map_png = s.topomap.render_png(current_id=nid)
    (map_dir / f"map_{nid}.png").write_bytes(map_png)

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
    render_scene_graph(observations, s.goal, s.goal_objects,
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

    # ── Route-first: override VLM MOVE guidance with path planner ─────
    # The path planner's next_instruction is the primary navigation text.
    # VLM guidance is only used when: (1) VLM says ARRIVED, or (2) no
    # route instruction available.
    _final_guidance = vlm_resp.guidance
    if vlm_resp.action == VLMAction.MOVE and _next_instr:
        _final_guidance = _next_instr
        log.info("[session %s] route-first: overriding VLM guidance with next_instruction", session_id)
    elif vlm_resp.action == VLMAction.MOVE and not _next_instr:
        log.info("[session %s] route-first: no next_instruction, using VLM guidance as fallback", session_id)

    if vlm_resp.action == VLMAction.ARRIVED:
        s.pending_arrival = True
        s.goal_node = nid
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
        s.pending_is_confirm = arrival_downgraded
    else:  # MOVE
        s.last_planned_action = _final_guidance

    return TurnResponse(
        action=vlm_resp.action,
        guidance=_final_guidance,
        question=vlm_resp.question,
        node_id=nid,
        annotated_photo_url=f"/session/{session_id}/photo/{nid}.jpg",
        corrected_node_id=loc_result.matched_nid if loc_result else None,
        corrected_confidence=round(loc_result.confidence, 3) if loc_result else None,
        corrected_location=_ref_location_name(loc_result) if loc_result else None,
        heading_deg=round(loc_result.matched_heading, 1) if loc_result and loc_result.matched_heading is not None else None,
        heading_slot=loc_result.matched_slot if loc_result else None,
        heading_confidence=round(loc_result.heading_confidence, 3) if loc_result else None,
        next_instruction=_next_instr,
        remaining_targets=len(s.remaining_targets) if s.target_nodes else None,
        phase=s.phase,
    )


@app.post("/session/{session_id}/answer", response_model=TurnResponse)
def post_answer(session_id: str, req: AnswerRequest) -> TurnResponse:
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
    _deny_keywords = ("不確定", "繼續", "不是", "沒有", "沒看到", "不對", "再找", "還沒", "no")
    if s.pending_is_confirm:
        nid = s.last_node_id
        if any(k in req.answer for k in _deny_keywords):
            log.info("[session %s] User denied confirm → continue MOVE", session_id)
            s.pending_question = None
            s.pending_is_confirm = False
            _active_goals = _current_goal_objects(s)
            vlm_resp = VLMResponse(
                action=VLMAction.MOVE,
                guidance=f"好的，繼續找「{'、'.join(_active_goals)}」，請往前走拍另一張照片。",
                question=None,
                vlm_summary="user denied arrival",
            )
        else:
            log.info("[session %s] User confirmed arrival via answer", session_id)
            s.pending_question = None
            s.pending_is_confirm = False
            s.pending_arrival = True
            s.goal_node = nid
            _active_goals = _current_goal_objects(s)
            # Auto-advance the route
            if s.route_plan and s.target_nodes:
                current_leg = s.route_plan.legs[min(s.current_leg_index, len(s.route_plan.legs) - 1)]
                s.mark_target_visited(current_leg.to_node)
                s.current_leg_index = min(s.current_leg_index + 1, len(s.route_plan.legs) - 1)
                log.info("[session %s] route advanced: visited node #%d, leg %d/%d",
                         session_id, current_leg.to_node, s.current_leg_index, len(s.route_plan.legs))

            next_goals = _current_goal_objects(s)
            remaining = len(s.remaining_targets) if s.target_nodes else 0
            if remaining > 0:
                vlm_resp = VLMResponse(
                    action=VLMAction.MOVE,
                    guidance=f"已找到「{'、'.join(_active_goals)}」！接下來找「{'、'.join(next_goals)}」，請拍一張照片讓我定位。",
                    question=None,
                    vlm_summary=f"confirmed {_active_goals}, moving to next",
                )
            else:
                vlm_resp = VLMResponse(
                    action=VLMAction.ARRIVED,
                    guidance=f"已找到「{'、'.join(_active_goals)}」！所有商品都找到了！",
                    question=None,
                    vlm_summary="all items found",
                )
                s.arrived = True

        s.history.append({
            "kind": "answer", "user_answer": req.answer,
            "vlm_action": vlm_resp.action.value, "vlm_guidance": vlm_resp.guidance,
        })
        return TurnResponse(
            action=vlm_resp.action,
            guidance=vlm_resp.guidance,
            question=vlm_resp.question,
            node_id=nid,
            remaining_targets=len(s.remaining_targets) if s.target_nodes else None,
            phase=s.phase,
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
    )

    log.info("[session %s] VLM → %s | %s", session_id, vlm_resp.action.value, vlm_resp.guidance[:120])

    # Gate ARRIVED on real evidence
    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, s.last_detections, ocr_matches=s.last_ocr_matches, goal_objects=_active_goals,
        min_score=ARRIVED_MIN_DETECTION_SCORE, img_w=img_w, img_h=img_h,
        prior_was_confirm=False,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action != VLMAction.ARRIVED

    s.history.append({
        "kind": "answer",
        "user_answer": req.answer,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
    })

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

    return TurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        node_id=s.last_node_id,
        annotated_photo_url=f"/session/{session_id}/photo/{s.last_node_id}.jpg",
        phase=s.phase,
    )


@app.post("/session/{session_id}/confirm", response_model=TurnResponse)
def confirm_arrival(session_id: str, req: ConfirmArrivalRequest) -> TurnResponse:
    """User confirms or rejects the ARRIVED decision."""
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
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

        _active_goals = _current_goal_objects(s)

        if s.phase == "shopping":
            # Advance route: mark current target as visited and move to next leg
            if s.route_plan and s.target_nodes:
                current_leg = s.route_plan.legs[min(s.current_leg_index, len(s.route_plan.legs) - 1)]
                s.mark_target_visited(current_leg.to_node)
                s.current_leg_index = min(s.current_leg_index + 1, len(s.route_plan.legs) - 1)
                log.info("[session %s] route advanced: visited node #%d, leg %d/%d, remaining=%s",
                         session_id, current_leg.to_node, s.current_leg_index,
                         len(s.route_plan.legs), s.remaining_targets)

            remaining = len(s.remaining_targets) if s.target_nodes else 0
            if remaining > 0:
                next_goals = _current_goal_objects(s)
                msg = f"已找到「{'、'.join(_active_goals)}」！接下來找「{'、'.join(next_goals)}」，請拍一張照片讓我定位。"
                s.arrived = False
            else:
                # All items found — transition to checkout or exit phase
                if s.checkout_node is not None:
                    s.phase = "checkout"
                    msg = f"已找到所有商品！現在帶您前往收銀台結帳，請拍一張照片讓我定位。"
                    s.arrived = False
                    log.info("[session %s] phase → checkout (WP%d)", session_id, s.checkout_node)
                elif s.exit_node is not None:
                    s.phase = "exit"
                    msg = f"已找到所有商品！現在帶您前往出口，請拍一張照片讓我定位。"
                    s.arrived = False
                    log.info("[session %s] phase → exit (WP%d)", session_id, s.exit_node)
                else:
                    msg = f"已找到「{'、'.join(_active_goals)}」！所有商品都找到了！"
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

    return TurnResponse(
        action=action,
        guidance=msg,
        question=s.pending_question,
        node_id=arrived_node,
        annotated_photo_url=f"/session/{session_id}/photo/{arrived_node}.jpg",
        phase=s.phase,
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

    # Neo4j: connect and pre-load reference map
    neo4j = get_neo4j()
    if neo4j:
        try:
            ref = neo4j.load_reference_map()
            log.info("Neo4j reference map: %d photos, %d edges",
                     len(ref.photos), len(ref.walkway_edges))
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

        # Use VLM to perceive the scene
        perception, _ = _vlm_perceive_and_decide(
            image_path=tmp,
            goal="定位",
            goal_objects=[],
            topomap_summary="(localization only)",
            img_w=img_w,
            img_h=img_h,
            prior_question=None,
            prior_answer=None,
            ocr_formatter=lambda vlm_ocr: "(localization)",
        )

        detected_labels = [d.label for d in perception.detections]
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


# ══════════════════════════════════════════════════════════════════════════
# Multi-target route planning endpoints
# ══════════════════════════════════════════════════════════════════════════

from server.navigator import resolve_targets, plan_multi_target_route, replan_route
from server.store_map import get_store_topomap


@app.post("/session/{session_id}/plan-route", response_model=PlanRouteResponse)
def plan_session_route(session_id: str, req: PlanRouteRequest):
    """Plan an optimised route for a navigation session.

    Resolves target queries to map nodes, then runs A* + TSP to find the
    shortest total path:  current_position → [targets in best order] → checkout → exit.
    """
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})

    # Resolve target queries to node IDs
    resolved = resolve_targets(req.targets)
    target_nodes = [r["node_id"] for r in resolved if r["node_id"] is not None]

    if not target_nodes:
        raise HTTPException(status_code=400, detail={
            "error": "no_targets_resolved",
            "detail": "None of the target queries could be matched to map locations",
        })

    # Determine start node: explicit > visual localization > lobby
    start = req.start_node
    if start is None:
        start = getattr(s, "last_corrected_nid", None)
    if start is None:
        from server.store_map import NODE_LOBBY
        start = NODE_LOBBY

    topo = get_store_topomap()
    route = plan_multi_target_route(
        topo.graph, start, target_nodes,
        checkout_node=req.checkout_node,
        exit_node=req.exit_node,
    )

    # Save to session
    s.target_nodes = target_nodes
    s.visited_targets = set()
    s.route_plan = route
    s.current_leg_index = 0
    s.checkout_node = req.checkout_node
    s.exit_node = req.exit_node

    log.info("[session %s] Route planned: %d targets, cost=%.1f, order=%s",
             session_id, len(target_nodes), route.total_cost, route.visit_order)

    return PlanRouteResponse(
        visit_order=route.visit_order,
        total_cost=route.total_cost,
        full_path=route.full_path,
        legs=[
            {"from": leg.from_node, "to": leg.to_node, "path": leg.path,
             "cost": leg.cost, "actions": leg.actions, "purpose": leg.purpose}
            for leg in route.legs
        ],
        resolved_targets=resolved,
        checkout_node=req.checkout_node,
        exit_node=req.exit_node,
    )


@app.post("/session/{session_id}/arrive-target")
def arrive_at_target(session_id: str, node_id: int = Body(..., embed=True)):
    """Mark a target as reached and re-plan the remaining route.

    Call this when the user arrives at a target node. The route is re-planned
    from the current position through the remaining targets → checkout → exit.
    """
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})

    if node_id not in s.target_nodes:
        raise HTTPException(status_code=400, detail={
            "error": "not_a_target",
            "detail": f"Node {node_id} is not in the target list",
        })

    s.mark_target_visited(node_id)

    remaining = s.remaining_targets
    log.info("[session %s] Arrived at target node %d. Remaining: %s",
             session_id, node_id, remaining)

    if not remaining and s.checkout_node is None and s.exit_node is None:
        # All done
        return {"status": "complete", "message": "所有目標都已到達！"}

    topo = get_store_topomap()
    route = replan_route(
        topo.graph, node_id, remaining,
        checkout_node=s.checkout_node,
        exit_node=s.exit_node,
    )
    s.route_plan = route
    s.current_leg_index = 0

    return {
        "status": "replanned",
        "visited": list(s.visited_targets),
        "remaining": remaining,
        "route": route.to_dict(),
    }


@app.post("/session/{session_id}/modify-route")
def modify_route(session_id: str, req: ModifyRouteRequest):
    """Add or remove targets mid-route, then re-plan.

    The user can change their shopping list while walking — this re-runs
    the full TSP from their current position.
    """
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})

    added_nodes = []
    removed_nodes = []

    # Add new targets
    if req.add:
        resolved = resolve_targets(req.add)
        for r in resolved:
            if r["node_id"] is not None:
                s.add_target(r["node_id"])
                added_nodes.append(r)

    # Remove targets
    if req.remove:
        resolved = resolve_targets(req.remove)
        for r in resolved:
            if r["node_id"] is not None:
                s.remove_target(r["node_id"])
                removed_nodes.append(r)

    remaining = s.remaining_targets

    # Determine current position
    current = getattr(s, "last_corrected_nid", None) or s.last_node_id
    if current is None:
        from server.store_map import NODE_LOBBY
        current = NODE_LOBBY

    topo = get_store_topomap()
    route = replan_route(
        topo.graph, current, remaining,
        checkout_node=s.checkout_node,
        exit_node=s.exit_node,
    )
    s.route_plan = route
    s.current_leg_index = 0

    log.info("[session %s] Route modified: added=%s removed=%s remaining=%s",
             session_id,
             [n["query"] for n in added_nodes],
             [n["query"] for n in removed_nodes],
             remaining)

    return {
        "status": "replanned",
        "added": added_nodes,
        "removed": removed_nodes,
        "remaining": remaining,
        "visited": list(s.visited_targets),
        "route": route.to_dict(),
    }


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


@app.get("/test-photo/{set_id}/{filename}")
def serve_test_photo(set_id: str, filename: str):
    """Serve test photos from 0916 test data."""
    base = Path("/Users/shingchou/Downloads/學校家樂福/0916測試")
    p = base / set_id / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"not found: {p}")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/topomap-builder")
def topomap_builder():
    """Interactive topological map builder page."""
    html_path = Path(__file__).parent / "topomap_builder.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/existing-topomap")
def existing_topomap():
    """Serve existing A7 topo map data for the builder's reference background."""
    import re as _re
    artifact_path = Path(
        "/Users/shingchou/.claude/projects/-Users-shingchou-Desktop-APPNAV-ios/"
        "a6b51ce6-65fa-4d5b-ad57-358da0cc2412/tool-results/"
        "artifact-fcf22acb-1789444754-adea.html"
    )
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="existing topo map artifact not found")
    html = artifact_path.read_text(encoding="utf-8")
    m = _re.search(
        r'<script id="topodata" type="application/json">(.*?)</script>', html, _re.DOTALL
    )
    if not m:
        raise HTTPException(status_code=500, detail="could not parse artifact topodata")
    data = json.loads(m.group(1))
    nodes = []
    for ph in data["ph"]:
        ocr_texts = [o["t"] for o in ph.get("o", []) if o.get("t")]
        nodes.append({"id": ph["id"], "x": ph["x"], "y": ph["y"],
                       "sn": ph.get("sn", ""), "labels": ocr_texts[:3]})
    edges = [{"a": w["a"], "b": w["b"]} for w in data.get("w", [])]
    return JSONResponse({"nodes": nodes, "edges": edges})


@app.get("/topomap-data")
def topomap_data():
    """Serve the generated topomap JSON."""
    p = Path("/private/tmp/claude-501/-Users-shingchou-Desktop-APPNAV-ios/a6b51ce6-65fa-4d5b-ad57-358da0cc2412/scratchpad/topomap.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail="topomap not generated yet")
    return JSONResponse(json.loads(p.read_text()))


@app.post("/topomap-confirm")
async def topomap_confirm(payload: dict = Body(...)):
    """Save the confirmed topomap selection for Neo4j upload."""
    out = Path("/private/tmp/claude-501/-Users-shingchou-Desktop-APPNAV-ios/a6b51ce6-65fa-4d5b-ad57-358da0cc2412/scratchpad/topomap_confirmed.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    n_nodes = len(payload.get("nodes", []))
    n_edges = len(payload.get("edges", []))
    primary = payload.get("primary_set", "?")
    log.info("Topomap confirmed: primary=%s nodes=%d edges=%d", primary, n_nodes, n_edges)
    return {"message": f"已確認！路線 {primary}，{n_nodes} 節點、{n_edges} 條邊。準備上傳到 Neo4j…"}
