"""FastAPI app — endpoints for in-store navigation sessions."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

import requests as _requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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
    PERCEPTION_ENABLED, OCR_ENABLED, OCR_BACKEND, OCR_LANGUAGES,
    OCR_MIN_CONFIDENCE, OCR_MAX_RESULTS, ARRIVED_MIN_DETECTION_SCORE,
    GOAL_CROP_VERIFY, GENERIC_INDOOR_OBJECTS, ensure_output_dir,
)
from server.goal_decomposer import decompose_goal, split_multi_goals
from server.ocr import OCR, OpenAIOCR
from server.models import (
    AnswerRequest,
    ErrorResponse,
    ConfirmArrivalRequest,
    ConfirmLocationRequest,
    ConfirmLocationResponse,
    StartSessionRequest,
    StartSessionResponse,
    SubGoalInfo,
    TurnResponse,
    VLMAction,
    VLMResponse,
)
from server.perception import Detection as PerceptionDetection, Perception, verify_goal_detection
from server.ocr import OCRResult
from server.session import Session, SessionStore, SubGoal

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
    perceive_only as _vlm_perceive_only,
    navigate_with_perception as _vlm_navigate,
    format_perception_detections as _vlm_format_dets,
    warm_up as _vlm_warm_up,
    ask_about_image as _vlm_ask_about_image,
)

log = logging.getLogger(__name__)

app = FastAPI(title="UniGoal Indoor Navigation Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from server.sensor_nav import router as _snav_router
app.include_router(_snav_router)

# Serve test UI
from fastapi.responses import FileResponse as _FileResp
from pathlib import Path as _P

@app.get("/test")
def _serve_test_ui():
    html = _P(__file__).resolve().parent.parent / "test_route_ui.html"
    return _FileResp(html, media_type="text/html")

@app.get("/debug")
def _serve_debug_dashboard():
    html = _P(__file__).resolve().parent.parent / "debug_dashboard.html"
    return _FileResp(html, media_type="text/html")


_AR_DEBUG_DIR = _P("output/sessions")


def _persist_ar_debug(s):
    """Save AR session debug data to disk."""
    from server.models import SubGoalInfo
    d = _AR_DEBUG_DIR / s.id
    d.mkdir(parents=True, exist_ok=True)
    sub_goals = [SubGoalInfo(name=sg.name, arrived=sg.arrived, is_terminal=sg.is_terminal) for sg in s.sub_goals] if s.sub_goals else []
    data = {
        "session_id": s.id,
        "goal": s.goal,
        "goal_objects": s.goal_objects,
        "place_name": s.place_name,
        "nav_mode": s.nav_mode,
        "arrived": s.arrived,
        "pending_arrival": s.pending_arrival,
        "goal_photo_ids": s.goal_photo_ids[:5] if s.goal_photo_ids else [],
        "sub_goals": [sg.dict() for sg in sub_goals],
        "current_goal_idx": s.current_goal_idx,
        "total_goals": s.shopping_goal_count or 1,
        "turns": s.history,
        "created_at": s.created_at.isoformat(),
        "turn_count": len(s.history),
        "type": "ar",
    }
    (d / "debug.json").write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")


def _load_persisted_ar_sessions():
    """Scan disk for persisted AR session debug files (including legacy dirs)."""
    results = []
    if not _AR_DEBUG_DIR.exists():
        return results
    active_ids = set(get_store()._sessions.keys())
    for d in _AR_DEBUG_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("snav_"):
            continue
        sid = d.name
        if sid in active_ids:
            continue
        f = d / "debug.json"
        if f.exists():
            try:
                results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        else:
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
                "type": "ar",
                "_legacy": True,
            })
    return results


@app.get("/sessions/list")
def _list_sessions():
    """List all AR-navigation sessions (active + persisted on disk)."""
    result = []
    for sid, s in get_store()._sessions.items():
        result.append({
            "session_id": sid,
            "goal": s.goal,
            "place_name": s.place_name,
            "nav_mode": s.nav_mode,
            "arrived": s.arrived,
            "turn_count": len(s.history),
            "created_at": s.created_at.isoformat(),
            "type": "ar",
        })
    for d in _load_persisted_ar_sessions():
        result.append({
            "session_id": d["session_id"],
            "goal": d.get("goal", ""),
            "place_name": d.get("place_name", ""),
            "nav_mode": d.get("nav_mode", ""),
            "arrived": d.get("arrived", False),
            "turn_count": d.get("turn_count", 0),
            "created_at": d.get("created_at", ""),
            "type": "ar",
        })
    return sorted(result, key=lambda x: x["created_at"], reverse=True)


@app.get("/sessions/{session_id}/debug")
def _get_session_debug(session_id: str):
    """Return full debug info for an AR-navigation session (memory or disk)."""
    s = get_store().get(session_id)
    if s is not None:
        from server.models import SubGoalInfo
        sub_goals = [SubGoalInfo(name=sg.name, arrived=sg.arrived, is_terminal=sg.is_terminal) for sg in s.sub_goals] if s.sub_goals else []
        return {
            "session_id": s.id,
            "goal": s.goal,
            "goal_objects": s.goal_objects,
            "place_name": s.place_name,
            "nav_mode": s.nav_mode,
            "arrived": s.arrived,
            "pending_arrival": s.pending_arrival,
            "goal_photo_ids": s.goal_photo_ids[:5] if s.goal_photo_ids else [],
            "sub_goals": [sg.dict() for sg in sub_goals],
            "current_goal_idx": s.current_goal_idx,
            "total_goals": s.shopping_goal_count or 1,
            "turns": s.history,
        }
    session_dir = _AR_DEBUG_DIR / session_id
    f = session_dir / "debug.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    # Legacy session: reconstruct turns from photo/annotated directories
    if session_dir.exists():
        photo_dir = session_dir / "photo"
        annotated_dir = session_dir / "annotated"
        turns = []
        if photo_dir.exists():
            for img in sorted(photo_dir.glob("*.jpg")):
                try:
                    nid = int(img.stem)
                except ValueError:
                    continue
                turn = {
                    "kind": "photo",
                    "node_id": nid,
                    "photo_url": f"/sessions/{session_id}/raw_photo/{nid}",
                }
                if (annotated_dir / f"{nid}.jpg").exists():
                    turn["annotated_photo_url"] = f"/session/{session_id}/photo/{nid}.jpg"
                det_file = annotated_dir / f"detections_{nid}.json"
                if det_file.exists():
                    try:
                        det_data = json.loads(det_file.read_text(encoding="utf-8"))
                        turn["detections"] = det_data.get("detections", [])
                        turn["ocr_results"] = det_data.get("ocr", [])
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


@app.get("/sessions/{session_id}/raw_photo/{node_id}")
def _serve_raw_photo(session_id: str, node_id: int):
    """Serve the original photo for a node."""
    from server.config import ensure_output_dir
    out_dir = _P("output/sessions") / session_id / "photo"
    fpath = out_dir / f"{node_id}.jpg"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="photo not found")
    return Response(content=fpath.read_bytes(), media_type="image/jpeg")


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
            if OCR_BACKEND == "openai":
                o = OpenAIOCR(languages=OCR_LANGUAGES)
            else:
                o = OCR(languages=OCR_LANGUAGES)
            o.load()
            _ocr = o
            log.info("OCR backend: %s", OCR_BACKEND)
        except Exception as e:
            log.warning("OCR unavailable: %s", e)
            _ocr_failed = True
            return None
    return _ocr


def vlm_decide(*args, **kwargs):
    return _vlm_decide_impl(*args, **kwargs)


def _image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.size  # (width, height)


def _top_goal_detection(detections, goal_objects):
    """Highest-scoring detection whose label matches a goal object, or None."""
    matches = [d for d in detections if scene._label_matches_goal(d.label, goal_objects)]
    return max(matches, key=lambda d: d.score) if matches else None


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


def _build_progress_string(s: Session) -> str:
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


def _sub_goals_info(s: Session) -> list[SubGoalInfo]:
    return [SubGoalInfo(name=sg.name, arrived=sg.arrived, is_terminal=sg.is_terminal) for sg in s.sub_goals]


def _build_route_info_string(s: Session) -> str:
    if not s.last_route_guidance:
        return ""
    parts = [f"地圖路線參考：{s.last_route_guidance}"]
    if s.last_route_waypoints:
        if len(s.last_route_waypoints) > 1:
            next_wp = s.last_route_waypoints[1]
            next_desc = next_wp.get("description") or ""
            # Egocentric direction to the next waypoint, relative to the way the
            # user is actually facing (front heading + matched view direction).
            rel_dir = None
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
        if len(s.last_route_waypoints) > 3:
            mid = s.last_route_waypoints[1:-1]
            wp_descs = [wp.get("description", "") for wp in mid if wp.get("description")]
            if wp_descs:
                parts.append(f"沿途路標：{'→'.join(wp_descs[:4])}")
    return "\n".join(parts)


@app.post("/session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail={"error": "bad_request", "detail": "goal is empty"})

    items = split_multi_goals(req.goal)

    from server.graph_renderer import GOAL_CLASS_MAP
    _generic_lower = {g.lower() for g in GENERIC_INDOOR_OBJECTS}

    def _decompose_item(item: str) -> list[str]:
        objs = decompose_goal(item)
        objs = augment_goal_classes(item, objs)
        _goal_mapped: set[str] = set()
        for key, info in GOAL_CLASS_MAP.items():
            if key in item:
                _goal_mapped.update(c.lower() for c in info["classes"])
        return [g for g in objs if g.lower() not in _generic_lower or g.lower() in _goal_mapped]

    sub_goals: list[SubGoal] = []
    all_goal_objects: list[str] = []
    for item in items:
        item_objects = _decompose_item(item)
        sub_goals.append(SubGoal(name=item, goal_objects=item_objects))
        all_goal_objects.extend(item_objects)
    all_goal_objects = list(dict.fromkeys(all_goal_objects))

    first_objects = sub_goals[0].goal_objects if sub_goals else all_goal_objects
    log.info("New session | goal: %s | items: %d | place_name: '%s' | goal_objects: %s",
             req.goal, len(sub_goals), req.place_name, first_objects)
    s = _store.create(goal=req.goal, goal_objects=first_objects)
    s.sub_goals = sub_goals

    # Generate goal graph
    out_dir = ensure_output_dir(s.id)
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    render_goal_graph(req.goal, all_goal_objects, str(graph_dir / "goal_graph.png"))
    log.info("Goal graph saved to %s", graph_dir / "goal_graph.png")

    # Load TopoGraphV2: prefer local JSON (TOPO_MAP_JSON), fall back to Neo4j
    nav_mode = "explore"
    goal_photo_ids: list[int] = []
    place_name = (req.place_name or "").strip()

    if place_name:
        from server.sensor_nav import _load_topo_v2, _setup_map_goals
        topo_v2 = _load_topo_v2(place_name)
        if topo_v2 is not None:
            s.topo_v2 = topo_v2
            s.place_name = place_name
            log.info("Loaded TopoGraphV2 '%s': %d photos, %d objects",
                     place_name,
                     len(topo_v2.all_photo_nodes()),
                     len(topo_v2.all_object_nodes()))

            # Search shopping goals, TSP-order them, and append checkout → exit.
            if _setup_map_goals(topo_v2, s, sub_goals):
                nav_mode = "map"
                goal_photo_ids = s.goal_photo_ids
                log.info("Goal found in map, sub_goals: %s",
                         [(sg.name, sg.goal_photo_ids[:3]) for sg in sub_goals])
            else:
                log.info("Goal not found in map, falling back to explore mode")
        else:
            log.info("No map found for place '%s'", place_name)

    if not s.tsp_order:
        s.tsp_order = list(range(len(sub_goals)))

    s.nav_mode = nav_mode
    cur_name = s.current_sub_goal.name if s.current_sub_goal else req.goal

    shopping_total = s.shopping_goal_count
    if nav_mode == "map":
        if shopping_total > 1:
            guidance = f"已載入「{place_name}」地圖，共 {shopping_total} 個目標。先在原地拍一張照片定位你的位置，系統會依你的所在位置安排最順的購物順序。"
        else:
            guidance = f"已載入「{place_name}」地圖，找到目標位置。拍一張照片開始定位導航。"
    else:
        guidance = "Upload a starting photo so I can see where you are."

    return StartSessionResponse(
        session_id=s.id,
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
            log.info("OCR (%s) found %d text regions", OCR_BACKEND, len(easyocr_results))
    t_ocr = time.time()
    log.info("⏱ OCR (%s): %.1fs", OCR_BACKEND, t_ocr - t_resize)

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Detection (mode-dependent)
    # ══════════════════════════════════════════════════════════════
    if has_grounding:
        # ── Mode A: GroundingDINO + EasyOCR ──
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
        scene_description = None

        log.info("[session %s] photo #%d | detections: %s | ocr: %s",
                 session_id, nid, detected_labels or "(none)", ocr_text_summary[:80])

    else:
        # ── Mode B/C: VLM Stage 1 (perception only) ──
        mode_label = f"two-stage VLM + OCR({OCR_BACKEND})" if easyocr_results else "two-stage VLM only"
        log.info("[session %s] Using %s", session_id, mode_label)

        nid = s.topomap.add_node(
            photo_path=str(photo_path), detected=[],
            summary="", ocr_texts=[], ocr_with_conf=[],
        )
        if s.last_node_id is not None:
            s.topomap.add_edge(s.last_node_id, nid, action=s.last_planned_action or "(unknown)")

        t_vlm_start = time.time()
        vlm_perception = _vlm_perceive_only(str(photo_path), s.goal_objects, img_w, img_h)
        t_percept = time.time()
        log.info("⏱ VLM perception: %.1fs", t_percept - t_vlm_start)

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

        ocr_text_summary = scene.format_ocr(ocr_results, img_w, img_h) if ocr_results else "(no text detected)"
        ocr_matches = scene.match_ocr_to_goal(ocr_results, s.goal_objects)
        if ocr_matches:
            ocr_text_summary += "  SIGN MATCH: " + "; ".join(ocr_matches)

        ocr_texts = [r.text for r in ocr_results]
        ocr_with_conf = [{"text": r.text, "confidence": r.confidence} for r in ocr_results]
        s.topomap.graph.nodes[nid]["detected"] = detected_labels
        s.topomap.graph.nodes[nid]["ocr_texts"] = ocr_texts
        s.topomap.graph.nodes[nid]["ocr_with_conf"] = ocr_with_conf

        detections_summary = _vlm_format_dets(vlm_perception.detections)
        scene_description = vlm_perception.scene_description

        det_detail = "; ".join(f"{d.label}@{d.position}" for d in detections if d.position)
        ocr_detail = "; ".join(f'"{r.text}"@{r.position}' for r in ocr_results
                               if getattr(r, "position", ""))
        log.info("[session %s] VLM perception | scene: %s",
                 session_id, vlm_perception.scene_description[:80])
        log.info("  detections(%d): %s", len(detections), det_detail or "(none)")
        log.info("  OCR(%d): %s", len(ocr_results), ocr_detail or "(none)")

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Postprocess + Localize + Route Plan (common)
    # ══════════════════════════════════════════════════════════════
    topomap_summary = s.topomap.summarize_for_vlm(current_id=nid)

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
                # Keep "" when the view direction is undetermined — don't coerce
                # to "front", or we'd assert a facing (and turn) we don't know.
                s.last_matched_direction = loc_result.best.matched_direction or ""
                log.info("[session %s] localized: photo_id=%d rank=%d score=%.3f matched=%d/%d region=%s",
                         session_id, loc_result.best.photo_id,
                         loc_result.best.photo_rank,
                         loc_result.best.score, loc_result.best.matched_count,
                         loc_result.best.query_object_count,
                         loc_result.best.region)
            for i, c in enumerate(loc_result.candidates[:3]):
                log.info("[session %s]   #%d photo_id=%d rank=%d score=%.3f matched=%d",
                         session_id, i+1, c.photo_id, c.photo_rank, c.score, c.matched_count)
            localization_candidates = [
                {
                    "nid": c.photo_id,
                    "photo_id": c.photo_id,
                    "photo_rank": c.photo_rank,
                    "score": round(c.score, 4),
                    "matched_count": c.matched_count,
                    "region": c.region,
                    "photo_url": f"/session/{session_id}/topo_photo/{c.photo_id}",
                }
                for c in loc_result.candidates[:5]
            ]
            if loc_result.warning:
                log.info("[session %s] localization warning: %s", session_id, loc_result.warning)
        except Exception as e:
            log.warning("[session %s] localization failed: %s", session_id, e)

    # First localization: solve the multi-goal visit order from the user's real
    # localized position (once), now that a meaningful start point exists.
    if (localized_photo_id is not None and s.tsp_pending_reorder
            and s.topo_v2 is not None):
        from server.sensor_nav import _reorder_goals_from_start
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
                log.info("[session %s] route: %s → %s, %d hops, %.1fm",
                         session_id, localized_photo_id, route.target_photo_id,
                         route.hops, route.distance)
                s.last_route_guidance = route_guidance
                s.last_route_waypoints = route_waypoints
            else:
                log.info("[session %s] route: no path found", session_id)
                s.last_route_guidance = None
                s.last_route_waypoints = []
        except Exception as e:
            log.warning("[session %s] route planning failed: %s", session_id, e)

    route_info = _build_route_info_string(s)

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Navigation Decision (mode-dependent)
    # ══════════════════════════════════════════════════════════════
    cur_goal_name = s.current_sub_goal.name if s.current_sub_goal else s.goal
    progress = _build_progress_string(s)

    if has_grounding:
        t_vlm_start = time.time()
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
        t_vlm = time.time()
        log.info("[session %s] VLM → %s | %s", session_id, vlm_resp.action.value, vlm_resp.guidance[:120])
        log.info("⏱ VLM: %.1fs | Total: %.1fs", t_vlm - t_vlm_start, t_vlm - t_start)
    else:
        vlm_resp = _vlm_navigate(
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
        t_vlm = time.time()
        log.info("[session %s] VLM 2-stage → %s | %s | Total: %.1fs",
                 session_id, vlm_resp.action.value, vlm_resp.guidance[:120], t_vlm - t_start)

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Verify arrival + annotate + state update
    # ══════════════════════════════════════════════════════════════
    goal_verified = None
    if GOAL_CROP_VERIFY and vlm_resp.action == VLMAction.ARRIVED:
        cand = _top_goal_detection(detections, s.goal_objects)
        if cand is not None:
            try:
                with Image.open(photo_path) as im:
                    goal_verified = verify_goal_detection(
                        im.convert("RGB"), cand.box, s.goal, _vlm_ask_about_image,
                    )
                log.info("goal crop verification: %s (label=%s)", goal_verified, cand.label)
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
        "timestamp": time.time(),
        "photo_url": f"/sessions/{session_id}/raw_photo/{nid}",
        "annotated_photo_url": f"/session/{session_id}/photo/{nid}.jpg",
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

    if vlm_resp.action == VLMAction.ARRIVED:
        s.pending_arrival = True
        s.goal_node = nid
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
        s.pending_is_confirm = arrival_downgraded
    else:  # MOVE
        s.last_planned_action = vlm_resp.guidance

    _persist_ar_debug(s)

    return TurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        node_id=nid,
        annotated_photo_url=f"/session/{session_id}/photo/{nid}.jpg",
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

    log.info("[session %s] VLM → %s | %s", session_id, vlm_resp.action.value, vlm_resp.guidance[:120])

    # If the user is answering our own confirm question, trust ARRIVED; otherwise gate it.
    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, s.last_detections, ocr_matches=s.last_ocr_matches, goal_objects=s.goal_objects,
        min_score=ARRIVED_MIN_DETECTION_SCORE, prior_was_confirm=s.pending_is_confirm,
    )
    # Reverse gate: catch missed arrivals when VLM is too conservative
    vlm_resp = scene.check_missed_arrival(
        vlm_resp, s.last_detections, ocr_matches=s.last_ocr_matches, goal_objects=s.goal_objects,
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

    _persist_ar_debug(s)

    return TurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        node_id=s.last_node_id,
        annotated_photo_url=f"/session/{session_id}/photo/{s.last_node_id}.jpg",
        sub_goals=_sub_goals_info(s),
        current_goal_idx=s.current_goal_idx,
        current_goal_name=s.current_sub_goal.name if s.current_sub_goal else s.goal,
        total_goals=s.shopping_goal_count or 1,
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
        cur = s.current_sub_goal
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

        if cur:
            cur.arrived = True
            cur.arrived_node = arrived_node

        if s.all_sub_goals_arrived or not s.sub_goals:
            s.arrived = True
            msg = f"已確認到達目標：{s.goal}"
            action = VLMAction.ARRIVED
        else:
            next_sg = s.advance_to_next_goal()
            if next_sg is not None:
                msg = f"已找到「{cur.name}」！接下來前往「{next_sg.name}」，請繼續拍照導航。"
                action = VLMAction.MOVE
            else:
                s.arrived = True
                msg = f"已確認到達所有目標：{s.goal}"
                action = VLMAction.ARRIVED

        log.info("[session %s] confirmed arrival at node %s", session_id, arrived_node)
        s.history.append({"kind": "confirm", "node_id": arrived_node, "message": msg})
        _persist_ar_debug(s)
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
            cur_name = s.current_sub_goal.name if s.current_sub_goal else s.goal
            msg = f"步驟 {arrived_node} 的目標宣告是誤判，繼續搜索{cur_name}"
        else:
            msg = f"步驟 {arrived_node} 是同種目標但非要找的，繼續搜索"

        s.corrections.append(msg)
        log.info("[session %s] rejected arrival: %s | %s", session_id, req.kind, msg)
        s.history.append({
            "kind": "reject", "node_id": arrived_node,
            "reject_type": req.kind, "message": msg,
        })
        _persist_ar_debug(s)
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
        question=None,
        node_id=arrived_node,
        annotated_photo_url=f"/session/{session_id}/photo/{arrived_node}.jpg",
        sub_goals=_sub_goals_info(s),
        current_goal_idx=s.current_goal_idx,
        current_goal_name=s.current_sub_goal.name if s.current_sub_goal else s.goal,
        total_goals=s.shopping_goal_count or 1,
    )


@app.post("/session/{session_id}/confirm_location", response_model=ConfirmLocationResponse)
def confirm_location(session_id: str, req: ConfirmLocationRequest) -> ConfirmLocationResponse:
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if s.topo_v2 is None:
        raise HTTPException(status_code=400, detail={"error": "no_map", "detail": "no topo_v2 map loaded"})
    if req.photo_id not in s.topo_v2.graph.nodes:
        raise HTTPException(status_code=404, detail={"error": "photo_not_found", "detail": f"photo_id {req.photo_id} not in map"})

    localized_photo_id = req.photo_id
    cur = s.current_sub_goal
    cur_name = cur.name if cur else s.goal
    goal_photo_ids = s.goal_photo_ids

    route_guidance: Optional[str] = None
    route_waypoints: list[dict] = []
    route_distance: Optional[float] = None
    route_hops: Optional[int] = None

    if goal_photo_ids:
        try:
            from server.route_planner import (plan_route, format_route_guidance,
                                              build_route_waypoints, user_facing_heading)
            route = plan_route(s.topo_v2, localized_photo_id, goal_photo_ids)
            if route is not None:
                s.planned_path = route.path
                route_distance = route.distance
                route_hops = route.hops
                src_rank = s.topo_v2._photo_sequence_rank(localized_photo_id)
                tgt_rank = s.topo_v2._photo_sequence_rank(route.target_photo_id)
                uh = user_facing_heading(s.topo_v2, localized_photo_id, s.last_matched_direction)
                route_guidance = format_route_guidance(s.topo_v2, route, src_rank, tgt_rank,
                                                       goal_objects=s.goal_objects,
                                                       user_heading=uh)
                route_waypoints = build_route_waypoints(s.topo_v2, route,
                                                        goal_objects=s.goal_objects)
                s.last_route_guidance = route_guidance
                s.last_route_waypoints = route_waypoints
        except Exception as e:
            log.warning("[session %s] confirm_location route planning failed: %s", session_id, e)

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

            vlm_resp = _vlm_navigate(
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
                log.info("[session %s] confirm_location VLM guidance: %s", session_id, guidance)
            vlm_out = {
                "action": vlm_resp.action.value,
                "guidance": vlm_resp.guidance,
                "question": vlm_resp.question,
                "vlm_summary": vlm_resp.vlm_summary,
            }
        except Exception as e:
            log.warning("[session %s] confirm_location VLM call failed: %s", session_id, e)

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
    _persist_ar_debug(s)

    return ConfirmLocationResponse(
        guidance=guidance,
        localized_photo_id=localized_photo_id,
        goal_photo_ids=goal_photo_ids[:5] if goal_photo_ids else [],
        route_guidance=route_guidance,
        route_waypoints=route_waypoints,
        route_distance=route_distance,
        route_hops=route_hops,
        sub_goals=_sub_goals_info(s),
        current_goal_idx=s.current_goal_idx,
        current_goal_name=cur_name,
        total_goals=s.shopping_goal_count or 1,
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


@app.on_event("startup")
def _check_vlm_and_warm():
    from server.config import VLM_BACKEND, GEMINI_API_KEY, GEMINI_MODEL, OPENAI_API_KEY, OPENAI_MODEL
    log.info("VLM backend: %s", VLM_BACKEND)
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


@app.post("/ocr")
async def ocr_extract(image: UploadFile = File(...)):
    """Run the configured OCR engine (OCR_BACKEND) on one image and return its text."""
    engine = get_ocr()
    if engine is None:
        raise HTTPException(status_code=503, detail={"error": "ocr_unavailable",
                            "detail": "OCR disabled or failed to load"})
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
        "engine": OCR_BACKEND,
        "languages": OCR_LANGUAGES,
        "text": " ".join(r.text for r in results),
        "regions": [
            {"text": r.text, "confidence": round(r.confidence, 4),
             "bbox": [list(p) for p in r.bbox]}
            for r in results
        ],
    }


@app.get("/session/{session_id}/topo_photo/{photo_id}")
def serve_topo_photo(session_id: str, photo_id: int):
    """Serve a photo from the loaded topo_v2 map (or fallback to global map)."""
    from server.config import resolve_topo_photo, TOPO_MAP_JSON
    from server.topomap_v2 import TopoGraphV2
    topo = None
    s = _store.get(session_id)
    if s is not None and s.topo_v2 is not None:
        topo = s.topo_v2
    if topo is None and TOPO_MAP_JSON and os.path.isfile(TOPO_MAP_JSON):
        topo = TopoGraphV2.load(TOPO_MAP_JSON)
    if topo is None:
        raise HTTPException(status_code=404, detail={"error": "no_map"})
    if photo_id not in topo.graph.nodes:
        raise HTTPException(status_code=404, detail={"error": "photo_not_found", "detail": str(photo_id)})
    node = topo.graph.nodes[photo_id]
    photo_path = node.get("photo_path", "")
    resolved = resolve_topo_photo(photo_path, photo_id=photo_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail={"error": "file_not_found", "detail": photo_path})
    return FileResponse(str(resolved), media_type="image/jpeg")


@app.get("/health")
def health():
    return {"status": "ok"}


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
    # Read directly from disk so restarted/legacy sessions still serve photos.
    p = _P("output/sessions") / session_id / "annotated" / f"{node_id}.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "photo_not_found", "detail": str(p)})
    return FileResponse(str(p), media_type="image/jpeg")


