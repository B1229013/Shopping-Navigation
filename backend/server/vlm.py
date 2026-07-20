"""VLM 客戶端：把「照片 + 提示詞」送給視覺語言模型，並解析出結構化的導航決策。

支援三種後端（由 config.VLM_BACKEND 決定）：Gemini（預設）、OpenAI、Ollama。

跟 Navigation-main-4/navigate_one_by_one.py 的 VLM 呼叫方式不同：
    navigate_one_by_one.py 是「自由對話」模式 —— 直接問模型「你看到什麼、
    該往哪走」，模型回傳一段自然語言中文文字，ARRIVED（是否抵達）則是由
    OCR 文字比對 + GroundingDINO 信心分數在腳本主流程另外判斷，VLM 完全
    不參與這個決策。

    APPNAV 這裡刻意維持「結構化決策」模式 —— 要求模型回傳單行 JSON
    （action / guidance / question / vlm_summary），因為 server.py 整個
    導航狀態機（ARRIVED/MOVE/ASK、pending_question、pending_arrival …）
    都是直接讀這個 action 欄位來驅動流程；如果改成自由文字，server.py
    與手機 App 前端的 API 合約都要跟著大改。因此本檔案只從參考腳本移植
    「呼叫方式與容錯設計」的精神（例如重試、失敗時給使用者看得懂的中文
    回覆），並沒有更動對外的 JSON 決策介面。
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
    GEMINI_API_KEY, GEMINI_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL,
    OLLAMA_URL, OLLAMA_MODEL,
    VLM_TIMEOUT_S, VLM_BACKEND,
)
from server.models import VLMAction, VLMResponse
from server.prompts import PER_TURN_PROMPT, PRIOR_ANSWER_BLOCK, PERCEIVE_PROMPT

log = logging.getLogger(__name__)

# 當 VLM 完全無法回應（呼叫失敗、金鑰未設定、影像讀取失敗…）時的保底回覆。
# 用 MOVE 而不是 ASK/ARRIVED，是為了避免在模型掛掉時卡住使用者或誤判抵達。
_FALLBACK = VLMResponse(
    action=VLMAction.MOVE,
    guidance="無法辨識目前場景，請往前走幾步再拍一張照片上傳。",
    question=None,
    vlm_summary="",
)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


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
) -> str:
    """組合本輪要送給 VLM 的完整提示詞（目標、地圖摘要、偵測結果、OCR 文字）。"""
    if prior_question and prior_answer:
        # 使用者剛回答了上一輪的確認問題，把問答內容補進提示詞讓模型有上下文
        block = PRIOR_ANSWER_BLOCK.format(previous_question=prior_question, user_answer=prior_answer)
    else:
        block = ""
    return PER_TURN_PROMPT.format(
        goal=goal,
        goal_objects=", ".join(goal_objects) or "(none)",
        topomap_summary=topomap_summary or "(starting location)",
        detections_summary=detections_summary or "(no detections)",
        ocr_summary=ocr_summary or "(no text detected)",
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


# ── Gemini 後端（預設）──────────────────────────────────────────────────

def _gemini_generate(prompt: str, image_b64: Optional[str] = None) -> str:
    url = _GEMINI_URL.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)
    parts: list = []
    if image_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    parts.append({"text": prompt})
    body = {
        "contents": [{"parts": parts}],
        # temperature 調低讓 JSON 格式更穩定，maxOutputTokens 限制避免回覆過長
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }
    r = requests.post(url, json=body, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    resp = r.json()
    candidates = resp.get("candidates", [])
    if not candidates:
        return ""
    text_parts = [p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])]
    return "".join(text_parts)


# ── Ollama 後端（本機模型，對應 navigate_one_by_one.py 唯一使用的後端）───

def _ollama_generate(prompt: str, images: Optional[List[str]] = None) -> str:
    # format="json" 要求 Ollama 端強制輸出合法 JSON，
    # 這點比 navigate_one_by_one.py 單純用自然語言提示詞更穩健。
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    if images:
        body["images"] = images
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json().get("response", "")


# ── OpenAI 後端 ──────────────────────────────────────────────────────────

def _openai_generate(prompt: str, image_b64: Optional[str] = None) -> str:
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


# ── 統一派發：依 config.VLM_BACKEND 選擇實際呼叫哪個後端 ─────────────────

def _generate(prompt: str, image_b64: Optional[str] = None) -> str:
    if VLM_BACKEND == "gemini":
        return _gemini_generate(prompt, image_b64=image_b64)
    elif VLM_BACKEND == "openai":
        return _openai_generate(prompt, image_b64=image_b64)
    else:
        return _ollama_generate(prompt, images=[image_b64] if image_b64 else None)


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
    backend = VLM_BACKEND
    if backend == "gemini" and not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set — VLM warm-up skipped")
        return
    if backend == "openai" and not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — VLM warm-up skipped")
        return
    try:
        _generate("Reply with an empty JSON object: {}")
        log.info("VLM warm-up complete (%s)", backend)
    except Exception as e:
        log.warning("VLM warm-up failed (continuing, backend=%s): %s", backend, e)


def decide(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_summary: Optional[str] = None,
) -> VLMResponse:
    """本輪導航的主要入口：讀圖 → 建提示詞 → 呼叫 VLM → 解析 JSON，並在失敗時重試一次。

    任何一步失敗（金鑰缺失、圖片讀取失敗、呼叫例外、回覆無法解析）都回傳
    _FALLBACK，讓 server.py 的狀態機可以安全地繼續（視為 MOVE，請使用者
    再拍一張照片），不會讓整個 API 請求噴例外。
    """
    if VLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set — cannot call VLM")
        return _FALLBACK
    if VLM_BACKEND == "openai" and not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot call VLM")
        return _FALLBACK

    prompt = _build_prompt(goal, goal_objects, topomap_summary, detections_summary, prior_question, prior_answer, ocr_summary=ocr_summary)
    try:
        img = Image.open(image_path).convert("RGB")
        # 縮圖降低上傳流量與模型延遲，1024px 已足夠模型辨識場景與物件
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

    # 最多重試一次：模型偶爾會回傳無法解析的格式，重試一次通常就能拿到合法 JSON，
    # 兩次都失敗才真的放棄、回傳保底回覆。
    for attempt in (1, 2):
        try:
            text = _generate(prompt, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM call failed (attempt %d/2): %s", attempt, e)
            continue
        parsed = _parse(text)
        if parsed is not None:
            return parsed
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
) -> Tuple[VLMPerception, VLMResponse]:
    """VLM-only mode (PERCEPTION_ENABLED=0): instead of GroundingDINO + EasyOCR
    feeding the per-turn decision prompt, ask the VLM to look at the photo itself
    first (objects + text + scene) and use THAT as the detections/OCR summaries
    for the exact same `decide()` call the real-detector path uses. Two separate
    VLM calls (perceive, then decide) rather than one combined call, so the
    decision prompt/schema stays identical either way — server.py and the phone
    app don't need to know which mode produced the detections.
    """
    if (VLM_BACKEND == "gemini" and not GEMINI_API_KEY) or (VLM_BACKEND == "openai" and not OPENAI_API_KEY):
        log.error("VLM API key not set — cannot call VLM")
        return _EMPTY_PERCEPTION, _FALLBACK

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
        return _EMPTY_PERCEPTION, _FALLBACK

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
    )
    return perception, decision
