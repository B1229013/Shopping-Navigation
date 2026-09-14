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

import requests as _requests
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
    OLLAMA_URL, OLLAMA_MODEL,
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
    perceive_and_decide as _vlm_perceive_and_decide,
    warm_up as _vlm_warm_up,
    ask_about_image as _vlm_ask_about_image,
)
from server.neo4j_client import get_neo4j
from server.visual_localization import localize as _localize_photo

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

        t_vlm_start = time.time()
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

        t_vlm_start = time.time()
        vlm_perception, vlm_resp = _vlm_perceive_and_decide(
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
        t_vlm = time.time()

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

        # EasyOCR 可用時完全取代 VLM OCR（VLM OCR 容易幻覺）
        if easyocr_results:
            ocr_results = easyocr_results
        else:
            ocr_results = vlm_ocr_results

        # 清理 VLM OCR：合併走道編號、去除中英文重複
        ocr_results = _clean_vlm_ocr(ocr_results, img_h)

        ocr_text_summary = scene.format_ocr(ocr_results, img_w, img_h) if ocr_results else "(no text detected)"
        ocr_matches = scene.match_ocr_to_goal(ocr_results, s.goal_objects)
        if ocr_matches:
            ocr_text_summary += "  SIGN MATCH: " + "; ".join(ocr_matches)

        ocr_texts = [r.text for r in ocr_results]
        ocr_with_conf = [{"text": r.text, "confidence": r.confidence} for r in ocr_results]
        s.topomap.graph.nodes[nid]["detected"] = detected_labels
        s.topomap.graph.nodes[nid]["ocr_texts"] = ocr_texts
        s.topomap.graph.nodes[nid]["ocr_with_conf"] = ocr_with_conf

        det_detail = "; ".join(f"{d.label}@{d.position}" for d in detections if d.position)
        ocr_detail = "; ".join(f'"{r.text}"@{r.position}' for r in ocr_results
                               if getattr(r, "position", ""))
        log.info("[session %s] VLM 2-stage → %s | scene: %s",
                 session_id, vlm_resp.action.value,
                 vlm_perception.scene_description[:80])
        log.info("  detections(%d): %s", len(detections), det_detail or "(none)")
        log.info("  OCR(%d): %s", len(ocr_results), ocr_detail or "(none)")
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
        min_score=ARRIVED_MIN_DETECTION_SCORE, goal_verified=goal_verified,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action == VLMAction.ASK
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

    s.history.append({
        "kind": "photo",
        "node_id": nid,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
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

    # ── Neo4j visual localization (position correction) ──────────────
    # Skip when in explore mode (s.place is None — no reference map)
    loc_result = None
    neo4j = get_neo4j() if s.place else None
    if neo4j:
        try:
            loc_result = _localize_photo(
                detected_labels=detected_labels,
                ocr_texts=ocr_texts,
                neo4j=neo4j,
                hint_nid=getattr(s, "last_corrected_nid", None),
                place=s.place,
            )
            if loc_result.matched_nid is not None:
                s.last_corrected_nid = loc_result.matched_nid
                log.info("[session %s] Neo4j localization: nid=%d conf=%.2f (%s)",
                         session_id, loc_result.matched_nid,
                         loc_result.confidence, loc_result.reasoning)
            else:
                log.info("[session %s] Neo4j localization: no confident match (%.2f)",
                         session_id, loc_result.confidence)
        except Exception as e:
            log.warning("[session %s] Neo4j localization error: %s", session_id, e)

    if vlm_resp.action == VLMAction.ARRIVED:
        s.pending_arrival = True
        s.goal_node = nid
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
        s.pending_is_confirm = arrival_downgraded
    else:  # MOVE
        s.last_planned_action = vlm_resp.guidance

    return TurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        node_id=nid,
        annotated_photo_url=f"/session/{session_id}/photo/{nid}.jpg",
        corrected_node_id=loc_result.matched_nid if loc_result else None,
        corrected_confidence=round(loc_result.confidence, 3) if loc_result else None,
        corrected_location=_ref_location_name(loc_result) if loc_result else None,
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

    log.info("[session %s] VLM → %s | %s", session_id, vlm_resp.action.value, vlm_resp.guidance[:120])

    # If the user is answering our own confirm question, trust ARRIVED; otherwise gate it.
    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, s.last_detections, ocr_matches=s.last_ocr_matches, goal_objects=s.goal_objects,
        min_score=ARRIVED_MIN_DETECTION_SCORE, prior_was_confirm=s.pending_is_confirm,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action == VLMAction.ASK

    # 用戶回答含否定/不確定語氣時，ARRIVED 降級為 MOVE
    _unsure_keywords = ("不確定", "繼續", "不是", "沒有", "沒看到", "不對", "再找", "還沒")
    if vlm_resp.action == VLMAction.ARRIVED and any(k in req.answer for k in _unsure_keywords):
        log.info("[session %s] User answer unsure (%s) — downgrade ARRIVED → MOVE", session_id, req.answer[:30])
        vlm_resp = VLMResponse(
            action=VLMAction.MOVE,
            guidance="請繼續前進，拍另一張照片。",
            question=None,
            vlm_summary=vlm_resp.vlm_summary,
        )

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
        s.arrived = True
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
        msg = f"已確認到達目標：{s.goal}"
        log.info("[session %s] confirmed arrival at node %s", session_id, arrived_node)
        s.history.append({"kind": "confirm", "node_id": arrived_node, "message": msg})
        action = VLMAction.ARRIVED

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
    from server.config import VLM_BACKEND, GEMINI_API_KEY, GEMINI_MODEL, OPENAI_API_KEY, OPENAI_MODEL
    log.info("VLM backend: %s", VLM_BACKEND)
    log.info("Perception (GroundingDINO): %s | OCR (EasyOCR): %s",
             "ON" if PERCEPTION_ENABLED else "OFF (VLM-only)",
             "ON" if OCR_ENABLED else "OFF")
    if VLM_BACKEND == "gemini":
        if not GEMINI_API_KEY:
            log.error("GEMINI_API_KEY not set. Add it to .env or environment variables.")
        else:
            log.info("Gemini model: %s", GEMINI_MODEL)
    elif VLM_BACKEND == "openai":
        if not OPENAI_API_KEY:
            log.error("OPENAI_API_KEY not set. Add it to .env or environment variables.")
        else:
            log.info("OpenAI model: %s", OPENAI_MODEL)
    else:
        try:
            r = _requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(m.startswith(OLLAMA_MODEL) for m in models):
                log.warning("Ollama model %s not pulled. Run: ollama pull %s", OLLAMA_MODEL, OLLAMA_MODEL)
        except Exception as e:
            log.error("Ollama unreachable: %s", e)

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
    """Build a human-readable location name from the localization result."""
    if not loc_result or not loc_result.ref_node:
        return None
    node = loc_result.ref_node
    # Use session + photo file as a rough location label
    return f"{node.session} #{node.nid} ({node.photo_file})"


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


@app.get("/health")
def health():
    neo4j = get_neo4j()
    return {
        "status": "ok",
        "neo4j": "connected" if neo4j and neo4j.is_connected else "disconnected",
    }


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
