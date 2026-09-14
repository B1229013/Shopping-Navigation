"""Sensor-Nav router — PDR-enhanced navigation with map persistence.

Mounted at /snav/. Reuses existing modules (TopoMap, Perception, VLM, OCR)
but maintains its own session store and adds PDR metric data to map edges.
"""
from __future__ import annotations

import json
import logging
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
from server.goal_decomposer import decompose_goal
from server.graph_renderer import (
    render_goal_graph, augment_goal_classes, postprocess_detections,
)
from server.config import normalize_label
from server.map_store import MapStore
from server.models import VLMAction, VLMResponse
from server.neo4j_map_store import Neo4jMapStore
from server.ocr import OCR, OCRResult
from server.perception import Detection as PerceptionDetection, Perception, verify_goal_detection
from server.session import Session
from server.topomap import TopoMap
from server.topomap_v2 import TopoGraphV2, NTYPE_OBJECT, ETYPE_WALKWAY
from server.vlm import (
    decide as vlm_decide,
    perceive_and_decide as vlm_perceive_and_decide,
    ask_about_image as vlm_ask_about_image,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/snav", tags=["sensor-nav"])

# ── Stores (independent from main server) ──

_sessions: Dict[str, Session] = {}
_map_store = MapStore()
_neo4j_store: Optional[Neo4jMapStore] = None


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


def _search_goal_in_topo(topo: TopoGraphV2, goal: str, goal_objects: List[str]) -> List[int]:
    """Search TopoGraphV2 for photo nodes whose objects match the goal.

    Returns photo_ids sorted by match score (best first).
    Uses keyword position weighting (earlier = more relevant) and
    IDF penalty (labels appearing on many photos score lower).
    """
    import math

    goal_lower = goal.lower()

    # Build keyword list with position-based weight:
    # first 1/3 of list = core (weight 1.0), rest = context (weight 0.3)
    core_cutoff = max(len(goal_objects) // 3, 3)
    kw_entries = []
    for i, kw in enumerate(goal_objects):
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

        # OCR matching (no IDF — OCR text is specific)
        if ocr_text:
            best_ocr_score = 0.0
            if goal_lower in ocr_text:
                best_ocr_score = 3.0
            for kw_low, _, kw_weight in kw_entries:
                if kw_low and len(kw_low) >= 2 and kw_low in ocr_text:
                    s = 2.0 * kw_weight
                    if s > best_ocr_score:
                        best_ocr_score = s
            score += best_ocr_score

        if score > 0:
            photo_scores[photo_id] = photo_scores.get(photo_id, 0.0) + score

    ranked = sorted(photo_scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked]

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


class AnswerRequest(BaseModel):
    answer: str


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

    goal_objects = decompose_goal(req.goal)
    goal_objects = augment_goal_classes(req.goal, goal_objects)
    from server.graph_renderer import GOAL_CLASS_MAP
    _generic_lower = {g.lower() for g in GENERIC_INDOOR_OBJECTS}
    _goal_mapped: set[str] = set()
    for key, info in GOAL_CLASS_MAP.items():
        if key in req.goal:
            _goal_mapped.update(c.lower() for c in info["classes"])
    goal_objects = [
        g for g in goal_objects
        if g.lower() not in _generic_lower or g.lower() in _goal_mapped
    ]

    import uuid
    sid = uuid.uuid4().hex[:8]
    s = Session(id=sid, goal=req.goal, goal_objects=goal_objects)

    # Load existing map if specified (local JSON)
    if req.map_id:
        loaded_topo, meta = _map_store.load(req.map_id)
        if loaded_topo is not None:
            s.topomap = loaded_topo
            log.info("Loaded map %s (%s) with %d nodes",
                     req.map_id, meta.get("name", ""), loaded_topo.graph.number_of_nodes())

    # Load TopoGraphV2 from Neo4j if place_name is given
    nav_mode = "explore"
    goal_photo_ids: List[int] = []
    place_name = req.place_name or ""

    if place_name:
        neo4j = _get_neo4j_store()
        if neo4j is not None:
            try:
                topo_v2 = neo4j.download(place_name)
                if topo_v2 is not None:
                    s.topo_v2 = topo_v2
                    s.place_name = place_name
                    photo_count = len(topo_v2.all_photo_nodes())
                    obj_count = len(topo_v2.all_object_nodes())
                    log.info("[snav] Loaded TopoGraphV2 '%s' from Neo4j: %d photos, %d objects",
                             place_name, photo_count, obj_count)

                    goal_photo_ids = _search_goal_in_topo(topo_v2, req.goal, goal_objects)
                    if goal_photo_ids:
                        nav_mode = "map"
                        s.goal_photo_ids = goal_photo_ids
                        log.info("[snav] Goal found in map at photo nodes: %s", goal_photo_ids[:5])
                    else:
                        log.info("[snav] Goal not found in map, falling back to explore mode")
                else:
                    log.info("[snav] No map found in Neo4j for place '%s'", place_name)
            except Exception as e:
                log.warning("[snav] Failed to load from Neo4j: %s", e)

    s.nav_mode = nav_mode

    _sessions[sid] = s
    log.info("[snav] New session %s | goal: %s | objects: %s | mode: %s",
             sid, req.goal, goal_objects, nav_mode)

    out_dir = ensure_output_dir(f"snav_{sid}")
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    render_goal_graph(req.goal, goal_objects, str(graph_dir / "goal_graph.png"))

    if nav_mode == "map":
        guidance = f"已載入「{place_name}」地圖，找到目標位置。拍一張照片開始定位導航。"
    else:
        guidance = "拍一張照片開始導航，感測器已在背景追蹤步行。"

    return SNavStartResponse(
        session_id=sid,
        guidance=guidance,
        goal_objects=goal_objects,
        nav_mode=nav_mode,
        goal_photo_ids=goal_photo_ids[:5],
        place_name=place_name,
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

    if has_grounding:
        # Mode A: GroundingDINO + EasyOCR + single-stage VLM
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
            # Attach PDR metrics to the edge
            if pdr:
                edata = s.topomap.graph.edges[s.last_node_id, nid]
                edata["steps"] = pdr.steps
                edata["distance_m"] = round(pdr.distance_m, 2)
                edata["heading_deg"] = round(pdr.heading_deg, 1)

        # Store PDR position on node
        if pdr:
            s.topomap.graph.nodes[nid]["pdr_x"] = pdr.pdr_x
            s.topomap.graph.nodes[nid]["pdr_y"] = pdr.pdr_y

        detections_summary = scene.format_detections(detections, img_w, img_h)
        topomap_summary = s.topomap.summarize_for_vlm(current_id=nid)

        # Inject PDR context into VLM prompt
        if pdr and pdr.steps > 0:
            pdr_context = (
                f"\n[PDR] 自上次拍照走了 {pdr.steps} 步（{pdr.distance_m:.1f}m），"
                f"航向 {pdr.heading_deg:.0f}°，累計 {pdr.total_steps} 步（{pdr.total_distance_m:.1f}m）"
            )
            topomap_summary += pdr_context

        vlm_resp = vlm_decide(
            image_path=str(photo_path),
            goal=s.goal,
            goal_objects=s.goal_objects,
            topomap_summary=topomap_summary,
            detections_summary=detections_summary,
            prior_question=None,
            prior_answer=None,
            ocr_summary=ocr_text_summary,
        )

    else:
        # Mode B/C: Two-stage VLM (± EasyOCR)
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

        topomap_summary = s.topomap.summarize_for_vlm(current_id=nid)
        if pdr and pdr.steps > 0:
            topomap_summary += (
                f"\n[PDR] 自上次拍照走了 {pdr.steps} 步（{pdr.distance_m:.1f}m），"
                f"航向 {pdr.heading_deg:.0f}°，累計 {pdr.total_steps} 步（{pdr.total_distance_m:.1f}m）"
            )

        def _ocr_formatter(vlm_ocr_items):
            if easyocr_results:
                summary = scene.format_ocr(easyocr_results, img_w, img_h)
                matches = scene.match_ocr_to_goal(easyocr_results, s.goal_objects)
                if matches:
                    summary += "  SIGN MATCH: " + "; ".join(matches)
                return summary
            raw = [OCRResult(text=t.text, confidence=t.score, bbox=t.bbox)
                   for t in vlm_ocr_items]
            if not raw:
                return "(no text detected)"
            summary = scene.format_ocr(raw, img_w, img_h)
            matches = scene.match_ocr_to_goal(raw, s.goal_objects)
            if matches:
                summary += "  SIGN MATCH: " + "; ".join(matches)
            return summary

        vlm_perception, vlm_resp = vlm_perceive_and_decide(
            image_path=str(photo_path),
            goal=s.goal,
            goal_objects=s.goal_objects,
            topomap_summary=topomap_summary,
            img_w=img_w,
            img_h=img_h,
            prior_question=None,
            prior_answer=None,
            ocr_formatter=_ocr_formatter,
        )

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

    # Verify arrival
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
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action == VLMAction.ASK

    s.topomap.graph.nodes[nid]["summary"] = vlm_resp.vlm_summary

    # Post-process detections
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

    # Annotate photo
    annotated_path = out_dir / "annotated" / f"{nid}.jpg"
    annotate(str(photo_path), str(annotated_path), detections,
             banner_text=f"{vlm_resp.action.value}: {vlm_resp.guidance}",
             ocr_results=ocr_results, goal_objects=s.goal_objects,
             region_mode=not has_grounding)

    # ── Localization against topo_v2 map ──
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
            )
            if loc_result.best:
                localized_photo_id = loc_result.best.photo_id
                localization_score = round(loc_result.best.score, 4)
                log.info("[snav %s] localized to P%d (score=%.3f, matched=%d/%d)",
                         session_id, loc_result.best.photo_rank,
                         loc_result.best.score, loc_result.best.matched_count,
                         loc_result.best.query_object_count)
            localization_candidates = [
                {
                    "photo_id": c.photo_id,
                    "photo_rank": c.photo_rank,
                    "score": round(c.score, 4),
                    "matched_count": c.matched_count,
                    "region": c.region,
                }
                for c in loc_result.candidates[:5]
            ]
            if loc_result.warning:
                log.info("[snav %s] localization warning: %s", session_id, loc_result.warning)
        except Exception as e:
            log.warning("[snav %s] localization failed: %s", session_id, e)

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

    vlm_resp = vlm_decide(
        image_path=s.last_photo_path,
        goal=s.goal,
        goal_objects=s.goal_objects,
        topomap_summary=topomap_summary,
        detections_summary=detections_summary,
        prior_question=prior_question,
        prior_answer=req.answer,
        ocr_summary=getattr(s, "last_ocr_summary", None),
    )

    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, s.last_detections, ocr_matches=s.last_ocr_matches,
        goal_objects=s.goal_objects,
        min_score=ARRIVED_MIN_DETECTION_SCORE, prior_was_confirm=s.pending_is_confirm,
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
    )


@router.post("/{session_id}/confirm", response_model=SNavTurnResponse)
def confirm_arrival(session_id: str, req: ConfirmArrivalRequest) -> SNavTurnResponse:
    s = _get_session(session_id)
    if not s.pending_arrival:
        raise HTTPException(status_code=409, detail={"error": "no_pending_arrival"})

    arrived_node = s.goal_node
    s.pending_arrival = False

    if req.kind == "confirmed":
        s.arrived = True
        msg = f"已確認到達目標：{s.goal}"
        s.history.append({"kind": "confirm", "node_id": arrived_node, "message": msg})
    else:
        s.goal_node = None
        if req.kind == "false_positive":
            s.false_positive_nodes.append(arrived_node)
            msg = f"步驟 {arrived_node} 是誤判，繼續搜索{s.goal}"
        else:
            msg = f"步驟 {arrived_node} 不是要找的，繼續搜索"
        s.corrections.append(msg)
        s.history.append({"kind": "reject", "node_id": arrived_node, "reject_type": req.kind, "message": msg})

    action = VLMAction.ARRIVED if req.kind == "confirmed" else VLMAction.MOVE
    return SNavTurnResponse(
        action=action,
        guidance=msg,
        question=None,
        node_id=arrived_node,
        annotated_photo_url=f"/snav/{session_id}/photo/{arrived_node}.jpg",
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
