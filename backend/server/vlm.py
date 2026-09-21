"""VLM 客戶端：把「照片 + 提示詞」送給 OpenAI GPT-4o，並解析出結構化的導航決策。

採用「結構化決策」模式 —— 要求模型回傳單行 JSON
（action / guidance / question / vlm_summary），server.py 的導航狀態機
（ARRIVED/MOVE/ASK、pending_question、pending_arrival …）直接讀 action
欄位來驅動流程。
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import requests
from PIL import Image

from server import scene
from server.config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL,
    VLM_TIMEOUT_S,
)
from server.models import VLMAction, VLMResponse
from server.prompts import PER_TURN_PROMPT, PRIOR_ANSWER_BLOCK, PERCEIVE_PROMPT, ROUTE_CONTEXT_BLOCK, CONTEXT_OBJECTS_BLOCK

log = logging.getLogger(__name__)

# 當 VLM 完全無法回應（呼叫失敗、金鑰未設定、影像讀取失敗…）時的保底回覆。
# 用 MOVE 而不是 ASK/ARRIVED，是為了避免在模型掛掉時卡住使用者或誤判抵達。
_FALLBACK = VLMResponse(
    action=VLMAction.MOVE,
    guidance="無法辨識目前場景，請往前走幾步再拍一張照片上傳。",
    question=None,
    vlm_summary="",
)

# ── VLM-only perception (used when GroundingDINO/EasyOCR are disabled) ─────

@dataclass
class VLMDetectedObject:
    label: str
    bbox: List[float]  # [x1,y1,x2,y2] absolute pixels
    score: float
    position: str = ""


@dataclass
class VLMDetectedText:
    text: str
    score: float
    bbox: List[List[float]]  # four [x,y] corner points, matches OCRResult.bbox
    position: str = ""


@dataclass
class VLMPerception:
    scene_description: str = ""
    detections: List[VLMDetectedObject] = field(default_factory=list)
    ocr_texts: List[VLMDetectedText] = field(default_factory=list)


_EMPTY_PERCEPTION = VLMPerception()


def _build_prompt(
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_summary: Optional[str] = None,
    route_context: Optional[str] = None,
    context_objects: Optional[List[str]] = None,
) -> str:
    """組合本輪要送給 VLM 的完整提示詞（目標、地圖摘要、偵測結果、OCR 文字、路徑上下文）。"""
    if prior_question and prior_answer:
        block = PRIOR_ANSWER_BLOCK.format(previous_question=prior_question, user_answer=prior_answer)
    else:
        block = ""
    # Section/landmark words are listed apart from the product so the VLM does
    # not treat "cooler" or "dairy sign" as the thing being looked for.
    context_block = (CONTEXT_OBJECTS_BLOCK.format(context_objects=", ".join(context_objects))
                     if context_objects else "")
    return PER_TURN_PROMPT.format(
        goal=goal,
        goal_objects=", ".join(goal_objects) or "(none)",
        context_block=context_block,
        topomap_summary=topomap_summary or "(starting location)",
        detections_summary=detections_summary or "(no detections)",
        ocr_summary=ocr_summary or "(no text detected)",
        route_context_block=route_context or "",
        prior_answer_block=block,
    )


def _parse(text: str) -> Optional[VLMResponse]:
    """從模型回覆中擷取第一個 JSON 物件並轉成 VLMResponse。

    模型偶爾會在 JSON 前後加上多餘的說明文字，所以用正則抓出第一段
    大括號內容，而不是直接對整段文字做 json.loads。解析失敗（格式錯誤、
    缺欄位、action 不是合法值）一律回傳 None，交給呼叫端重試或使用保底回覆。
    """
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return VLMResponse(
            action=VLMAction(obj["action"]),
            guidance=str(obj.get("guidance", "")),
            question=obj.get("question"),
            vlm_summary=str(obj.get("vlm_summary", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("VLM JSON parse failed: %s", e)
        return None


_HALLUCINATION_LANDMARKS = [
    "收銀區", "收銀台", "結帳區", "結帳台", "自助結帳",
    "玻璃隔板", "玻璃門", "玻璃櫃", "電扶梯", "手扶梯", "停車場",
    "洗手間", "廁所", "服務台", "客服", "美食街", "餐飲區",
    "藥妝區", "化妝品區", "服飾區", "家電區", "文具區",
]


def _validate_guidance(resp: VLMResponse, detections_summary: str,
                       ocr_summary: str | None) -> VLMResponse:
    """Check guidance/vlm_summary for landmarks not in the detection list.

    If a known landmark term appears in guidance but NOT in the detections or
    OCR, remove that phrase from guidance to prevent hallucination.
    """
    if not resp or not detections_summary:
        return resp

    ref_text = (detections_summary + " " + (ocr_summary or "")).lower()
    guidance = resp.guidance
    summary = resp.vlm_summary
    flagged = []

    for term in _HALLUCINATION_LANDMARKS:
        if term in guidance.lower() or term in summary.lower():
            if term.lower() not in ref_text:
                flagged.append(term)

    if not flagged:
        return resp

    log.warning("VLM hallucination detected — terms not in detections: %s", flagged)

    for term in flagged:
        for pattern in [
            f"先經過[右左前後]手?邊?的?[^，。、]*{term}[^，。、]*[，、]?",
            f"前方[是的有][^，。、]*{term}[^，。、]*[，、]?",
            f"[經過往向朝]?[^，。、]*{term}[^，。、]*[，、]?",
        ]:
            guidance = re.sub(pattern, "", guidance)
            summary = re.sub(pattern, "", summary)
        guidance = guidance.replace(term, "")
        summary = summary.replace(term, "")

    guidance = re.sub(r"[，、]{2,}", "，", guidance).strip("，、 ")
    summary = re.sub(r"[，、]{2,}", "，", summary).strip("，、 ")

    return VLMResponse(
        action=resp.action,
        guidance=guidance,
        question=resp.question,
        vlm_summary=summary,
    )


# ── OpenAI GPT-4o ──────────────────────────────────────────────────────

def _generate(prompt: str, image_b64: Optional[str] = None) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    content: list = []
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    content.append({"type": "text", "text": prompt})
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_completion_tokens": 512,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, json=body, headers=headers, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    choices = r.json().get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def ask_about_image(image_pil: Image.Image, prompt: str) -> str:
    """對單張圖片問一個自由文字問題，回傳模型的原始文字回覆（不解析 JSON）。

    目前用於 server.py 的「目標裁切驗證」流程：把 GroundingDINO 判斷為目標的
    區域裁切出來，單獨問模型「這是不是要找的東西」，避免密集貨架上的誤判。
    """
    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return _generate(prompt, image_b64=img_b64)


def warm_up() -> None:
    """伺服器啟動時預先呼叫一次 VLM，讓後端模型/連線提前就緒，減少第一輪的延遲。

    若金鑰未設定則直接跳過（不視為錯誤），呼叫失敗也只記警告，不阻擋啟動。
    """
    if not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — VLM warm-up skipped")
        return
    try:
        _generate("Reply with an empty JSON object: {}")
        log.info("VLM warm-up complete (OpenAI %s)", OPENAI_MODEL)
    except Exception as e:
        log.warning("VLM warm-up failed (continuing): %s", e)


def decide(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_summary: Optional[str] = None,
    route_context: Optional[str] = None,
    context_objects: Optional[List[str]] = None,
) -> VLMResponse:
    """本輪導航的主要入口：讀圖 → 建提示詞 → 呼叫 VLM → 解析 JSON → 驗證幻覺，
    並在失敗時重試一次。

    任何一步失敗（金鑰缺失、圖片讀取失敗、呼叫例外、回覆無法解析）都回傳
    _FALLBACK，讓 server.py 的狀態機可以安全地繼續（視為 MOVE，請使用者
    再拍一張照片），不會讓整個 API 請求噴例外。
    """
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot call VLM")
        return _FALLBACK

    prompt = _build_prompt(goal, goal_objects, topomap_summary, detections_summary, prior_question, prior_answer,
                           ocr_summary=ocr_summary, route_context=route_context, context_objects=context_objects)
    try:
        img = Image.open(image_path).convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        log.info("VLM image resized to %s, %d KB", img.size, len(buf.getvalue()) // 1024)
    except Exception as e:
        log.warning("VLM could not read image %s: %s", image_path, e)
        return _FALLBACK

    for attempt in (1, 2):
        try:
            text = _generate(prompt, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM call failed (attempt %d/2): %s", attempt, e)
            continue
        parsed = _parse(text)
        if parsed is not None:
            return _validate_guidance(parsed, detections_summary, ocr_summary)
        log.warning("VLM unparseable response (attempt %d/2): %.200s", attempt, text)
    return _FALLBACK


def _parse_perception(text: str, img_w: int, img_h: int) -> Optional[VLMPerception]:
    """Parse the PERCEIVE_PROMPT JSON reply, converting its 0-1 fractional boxes
    to absolute pixel coordinates (matching what GroundingDINO/EasyOCR produce,
    so downstream code — scene.py formatting, annotator.py drawing — doesn't
    need to know whether detections came from a real detector or the VLM)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))

        detections: List[VLMDetectedObject] = []
        for d in obj.get("detections", []):
            box = d.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(c) for c in box)
            pixel_box = [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]
            h, v, depth = scene._position(pixel_box, img_w, img_h)
            detections.append(VLMDetectedObject(
                label=str(d.get("label", "")),
                bbox=pixel_box,
                score=float(d.get("score", 0.5)),
                position=f"{h} {v}, {depth}",
            ))

        ocr_texts: List[VLMDetectedText] = []
        for t in obj.get("ocr_texts", []):
            box = t.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(c) for c in box)
            px1, py1, px2, py2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
            corners = [[px1, py1], [px2, py1], [px2, py2], [px1, py2]]
            side = scene._horizontal_side((px1 + px2) / 2, img_w)
            ocr_texts.append(VLMDetectedText(
                text=str(t.get("text", "")),
                score=float(t.get("score", 0.5)),
                bbox=corners,
                position=side,
            ))

        return VLMPerception(
            scene_description=str(obj.get("scene_description", "")),
            detections=detections,
            ocr_texts=ocr_texts,
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        log.warning("VLM perception JSON parse failed: %s", e)
        return None


def perceive(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    img_w: int,
    img_h: int,
) -> VLMPerception:
    """Stage 1: ask VLM to perceive the photo (detections + OCR + scene).

    Returns perception only — caller runs localization with the labels,
    then calls decide() with the route context.
    """
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot call VLM")
        return _EMPTY_PERCEPTION

    try:
        img = Image.open(image_path).convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        log.warning("VLM could not read image %s: %s", image_path, e)
        return _EMPTY_PERCEPTION

    prompt = PERCEIVE_PROMPT.format(goal=goal, goal_objects=", ".join(goal_objects) or "(none)")
    perception = _EMPTY_PERCEPTION
    for attempt in (1, 2):
        try:
            text = _generate(prompt, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM perception call failed (attempt %d/2): %s", attempt, e)
            continue
        parsed = _parse_perception(text, img_w, img_h)
        if parsed is not None:
            perception = parsed
            break
        log.warning("VLM perception unparseable response (attempt %d/2): %.200s", attempt, text)
    return perception


def perceive_and_decide(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    img_w: int,
    img_h: int,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_formatter: Callable[[List[VLMDetectedText]], str],
    route_context: Optional[str] = None,
    context_objects: Optional[List[str]] = None,
) -> Tuple[VLMPerception, VLMResponse]:
    """VLM-only mode: perceive then decide (legacy combined call)."""
    perception = perceive(image_path, goal, goal_objects, img_w, img_h)

    detections_summary = scene.format_detections(
        [{"label": d.label, "box": d.bbox, "score": d.score} for d in perception.detections],
        img_w, img_h,
    )
    ocr_summary = ocr_formatter(perception.ocr_texts)

    decision = decide(
        image_path=image_path,
        goal=goal,
        goal_objects=goal_objects,
        topomap_summary=topomap_summary,
        detections_summary=detections_summary,
        prior_question=prior_question,
        prior_answer=prior_answer,
        ocr_summary=ocr_summary,
        route_context=route_context,
        context_objects=context_objects,
    )
    return perception, decision
