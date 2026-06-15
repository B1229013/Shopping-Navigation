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
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from server import scene
from server.annotator import annotate
from server.config import (
    OLLAMA_URL, OLLAMA_MODEL,
    PERCEPTION_ENABLED, OCR_ENABLED, OCR_LANGUAGES,
    OCR_MIN_CONFIDENCE, OCR_MAX_RESULTS, ARRIVED_MIN_DETECTION_SCORE,
    GOAL_CROP_VERIFY, ensure_output_dir,
)
from server.goal_decomposer import decompose_goal
from server.ocr import OCR
from server.models import (
    AnswerRequest,
    ErrorResponse,
    StartSessionRequest,
    StartSessionResponse,
    TurnResponse,
    VLMAction,
)
from server.perception import Perception, verify_goal_detection
from server.session import SessionStore
from server.vlm import (
    decide as _vlm_decide_impl,
    warm_up as _vlm_warm_up,
    ask_about_image as _vlm_ask_about_image,
)

log = logging.getLogger(__name__)

app = FastAPI(title="UniGoal Indoor Navigation Server")

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


@app.post("/session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail={"error": "bad_request", "detail": "goal is empty"})
    goal_objects = decompose_goal(req.goal)
    log.info("New session | goal: %s | goal_objects: %s", req.goal, goal_objects)
    s = _store.create(goal=req.goal, goal_objects=goal_objects)
    return StartSessionResponse(
        session_id=s.id,
        guidance="Upload a starting photo so I can see where you are.",
        goal_objects=goal_objects,
    )


@app.post("/session/{session_id}/photo", response_model=TurnResponse)
async def upload_photo(session_id: str, photo: UploadFile = File(...)) -> TurnResponse:
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if s.arrived:
        raise HTTPException(status_code=409, detail={"error": "already_arrived", "detail": "session is complete"})
    if s.pending_question is not None:
        raise HTTPException(status_code=409, detail={"error": "answer_pending", "detail": "POST /answer first"})

    t_start = time.time()
    out_dir = ensure_output_dir(session_id)
    photo_bytes = await photo.read()
    nid_for_path = s.topomap.graph.number_of_nodes()
    photo_path = out_dir / "photo" / f"{nid_for_path}.jpg"
    photo_path.write_bytes(photo_bytes)

    # Read image dimensions; only re-save if resize is needed
    im = Image.open(photo_path)
    im = im.convert("RGB")
    img_w, img_h = im.size
    if max(img_w, img_h) > 1600:
        im.thumbnail((1600, 1600))
        img_w, img_h = im.size
        im.save(photo_path, format="JPEG", quality=85)
        log.info("Photo resized to %dx%d for faster processing", img_w, img_h)
    im.close()

    t_resize = time.time()

    perception = get_perception()
    log.info("GroundingDINO prompt: %s", s.goal_objects)
    detections = perception.detect(str(photo_path), s.goal_objects) if perception else []
    detected_labels = [d.label for d in detections]
    t_detect = time.time()
    log.info("⏱ GroundingDINO: %.1fs", t_detect - t_resize)

    # Run OCR to extract text from the photo
    ocr_engine = get_ocr()
    ocr_results = []
    ocr_text_summary = "(no text detected)"
    if ocr_engine is not None:
        ocr_results = ocr_engine.read(
            str(photo_path),
            min_confidence=OCR_MIN_CONFIDENCE,
            max_results=OCR_MAX_RESULTS,
        )
        ocr_text_summary = scene.format_ocr(ocr_results, img_w, img_h)
        if ocr_results:
            log.info("OCR found %d text regions: %s", len(ocr_results), ocr_text_summary[:100])
    t_ocr = time.time()
    log.info("⏱ EasyOCR: %.1fs", t_ocr - t_detect)

    # Match OCR signs to the goal (store aisle/section signs are the strongest signal).
    ocr_matches = scene.match_ocr_to_goal(ocr_results, s.goal_objects)
    if ocr_matches:
        ocr_text_summary += "  SIGN MATCH: " + "; ".join(ocr_matches)

    ocr_texts = [r.text for r in ocr_results]
    nid = s.topomap.add_node(
        photo_path=str(photo_path), detected=detected_labels,
        summary="", ocr_texts=ocr_texts,
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

    # TASK 3 — when the VLM claims ARRIVED, crop the best goal-matching detection
    # and ask the VLM to confirm that region really shows the goal. This catches
    # GroundingDINO false positives on dense shelves before we declare success.
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

    # Gate ARRIVED on real evidence (a fresh photo can't be answering a confirm).
    _raw_action = vlm_resp.action
    vlm_resp = scene.verify_arrival(
        vlm_resp, detections, ocr_matches=ocr_matches, goal_objects=s.goal_objects,
        min_score=ARRIVED_MIN_DETECTION_SCORE, goal_verified=goal_verified,
    )
    arrival_downgraded = _raw_action == VLMAction.ARRIVED and vlm_resp.action == VLMAction.ASK

    s.topomap.graph.nodes[nid]["summary"] = vlm_resp.vlm_summary
    s.last_detections = [d.__dict__ for d in detections]
    s.last_ocr_summary = ocr_text_summary
    s.last_ocr_matches = ocr_matches
    s.last_photo_path = str(photo_path)
    s.last_node_id = nid

    annotated_path = out_dir / "annotated" / f"{nid}.jpg"
    annotate(str(photo_path), str(annotated_path), detections, banner_text=f"{vlm_resp.action.value}: {vlm_resp.guidance}", ocr_results=ocr_results)

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

    if vlm_resp.action == VLMAction.ARRIVED:
        s.arrived = True
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

    s.history.append({
        "kind": "answer",
        "user_answer": req.answer,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
    })

    s.pending_question = None
    s.pending_is_confirm = False
    if vlm_resp.action == VLMAction.ARRIVED:
        s.arrived = True
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
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    out_dir = ensure_output_dir(session_id)
    p = out_dir / "annotated" / f"{node_id}.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "photo_not_found", "detail": str(p)})
    return FileResponse(str(p), media_type="image/jpeg")
