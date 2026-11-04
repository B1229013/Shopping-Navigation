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
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import requests
from PIL import Image

from server import scene
from server.config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL,
    VLM_TIMEOUT_S, VLM_ROUNDS,
)
from server.models import VLMAction, VLMResponse
from server.prompts import (
    PER_TURN_PROMPT, PRIOR_ANSWER_BLOCK, PERCEIVE_PROMPT,
    PERCEIVE_REFINE_PROMPT, PERCEIVE_DISTINCTIVE_PROMPT,
    ROUTE_CONTEXT_BLOCK, CONTEXT_OBJECTS_BLOCK,
)

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
    # The English name the VLM gave next to `label` (its 繁體中文 name), when it gave both.
    # Only localization uses it — see localization_label.
    label_en: str = ""


def localization_label(det: VLMDetectedObject) -> str:
    """The name to match against the reference map's object labels, which are stored as
    「中文 English」 ("區域招牌 area sign"). `label` alone is the 中文 half, and the label
    similarity scores "區域招牌" vs "區域招牌 area sign" at 0.33 — below the 0.55 match
    threshold — so localization uses both halves. Everything else (prompts, goal and
    arrival checks, annotations) keeps `label`."""
    en = (det.label_en or "").strip()
    if en and en.lower() not in det.label.lower():
        return f"{det.label} {en}"
    return det.label


@dataclass
class VLMDetectedText:
    text: str
    score: float
    bbox: List[List[float]]  # four [x,y] corner points, matches OCRResult.bbox
    position: str = ""


@dataclass
class VLMPerception:
    scene_description: str = ""
    store_zone: str = ""
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

# Token budgets. The decide reply is one short JSON object; the perceive reply
# lists every object and sign in the photo and easily runs past 512 tokens on a
# busy aisle — a truncated reply used to be thrown away wholesale.
DECIDE_MAX_TOKENS = 512
PERCEIVE_MAX_TOKENS = 2000


def _generate(prompt: str, image_b64: Optional[str] = None,
              max_tokens: int = DECIDE_MAX_TOKENS) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    content: list = []
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    content.append({"type": "text", "text": prompt})
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": 4096 if "luna" in OPENAI_MODEL else 2048,
    }
    if "luna" not in OPENAI_MODEL:
        body["temperature"] = 0.2
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
            if "429" in str(e):
                time.sleep(30 * attempt)
            continue
        parsed = _parse(text)
        if parsed is not None:
            return _validate_guidance(parsed, detections_summary, ocr_summary)
        log.warning("VLM unparseable response (attempt %d/2): %.200s", attempt, text)
    return _FALLBACK


def _load_json_or_salvage(text: str) -> Optional[dict]:
    """json.loads, or — when the reply was cut off by the token cap — the longest
    prefix that ends on a complete list item, with the open list/object closed.
    Losing the tail of the object list is far better than losing the whole photo."""
    end = text.rfind("}")
    if end >= 0:
        try:
            return json.loads(text[:end + 1])
        except json.JSONDecodeError:
            pass
    cut = text.rfind("},")
    while cut > 0:
        for closer in ("]}", "]}]}"):
            try:
                return json.loads(text[:cut + 1] + closer)
            except json.JSONDecodeError:
                continue
        cut = text.rfind("},", 0, cut)
    for closer in ("]}", "]}]}", "}"):   # truncated inside the very first item
        head = text.rfind("[")
        if head > 0:
            try:
                return json.loads(text[:head + 1] + closer)
            except json.JSONDecodeError:
                continue
    log.warning("VLM perception JSON parse failed even after salvage: %.120s", text)
    return None


def _parse_perception(text: str, img_w: int, img_h: int) -> Optional[VLMPerception]:
    """Parse a perception JSON reply, converting its 0-1 fractional boxes
    to absolute pixel coordinates (matching what GroundingDINO/EasyOCR produce,
    so downstream code — scene.py formatting, annotator.py drawing — doesn't
    need to know whether detections came from a real detector or the VLM).

    Two reply formats: PERCEIVE_PROMPT / PERCEIVE_REFINE_PROMPT answer with "items"
    (zh / en / conf / box / text); PERCEIVE_DISTINCTIVE_PROMPT (pass 3) answers with
    "detections" (label / box / score) and "ocr_texts" (text / score / box). Until
    2026-10 only "items" was read, so every pass-3 reply was parsed as empty."""
    start = text.find("{")
    if start < 0:
        return None
    obj = _load_json_or_salvage(text[start:])
    if obj is None:
        return None
    try:
        detections: List[VLMDetectedObject] = []
        ocr_texts: List[VLMDetectedText] = []

        for item in obj.get("items", []):
            box = item.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(c) for c in box)
            pixel_box = [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]
            h, v, depth = scene._position(pixel_box, img_w, img_h)

            zh = str(item.get("zh", "") or "").strip()
            en = str(item.get("en", "") or "").strip()
            detections.append(VLMDetectedObject(
                label=zh or en,
                bbox=pixel_box,
                score=float(item.get("conf", 0.5)),
                position=f"{h} {v}, {depth}",
                label_en=en if zh else "",
            ))

            item_text = str(item.get("text", "")).strip()
            if item_text:
                px1, py1, px2, py2 = pixel_box
                corners = [[px1, py1], [px2, py1], [px2, py2], [px1, py2]]
                side = scene._horizontal_side((px1 + px2) / 2, img_w)
                ocr_texts.append(VLMDetectedText(
                    text=item_text,
                    score=float(item.get("conf", 0.5)),
                    bbox=corners,
                    position=side,
                ))

        # pass-3 format
        def _pixel_box(box):
            if not box or len(box) != 4:
                return None
            x1, y1, x2, y2 = (float(c) for c in box)
            if x2 <= x1 or y2 <= y1:      # the model fills [0,0,0,0] for "not really seen"
                return None
            return [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]

        for item in obj.get("detections", []) or []:
            label = str(item.get("label", "") or "").strip()
            pixel_box = _pixel_box(item.get("box"))
            if not label or pixel_box is None:
                continue
            h, v, depth = scene._position(pixel_box, img_w, img_h)
            detections.append(VLMDetectedObject(
                label=label, bbox=pixel_box, score=float(item.get("score", 0.5)),
                position=f"{h} {v}, {depth}"))
        for item in obj.get("ocr_texts", []) or []:
            item_text = str(item.get("text", "") or "").strip()
            pixel_box = _pixel_box(item.get("box"))
            if not item_text or pixel_box is None:
                continue
            px1, py1, px2, py2 = pixel_box
            ocr_texts.append(VLMDetectedText(
                text=item_text, score=float(item.get("score", 0.5)),
                bbox=[[px1, py1], [px2, py1], [px2, py2], [px1, py2]],
                position=scene._horizontal_side((px1 + px2) / 2, img_w)))

        return VLMPerception(
            scene_description=str(obj.get("scene", "") or obj.get("scene_description", "")),
            store_zone="",
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

    prompt = PERCEIVE_PROMPT.format(goal=goal, goal_objects=", ".join(goal_objects),
                                     goal_label_example=(goal_objects[0] if goal_objects else "milk") or "(none)")
    perception = _EMPTY_PERCEPTION
    for attempt in (1, 2):
        try:
            text = _generate(prompt, image_b64=img_b64, max_tokens=PERCEIVE_MAX_TOKENS)
        except Exception as e:
            log.warning("VLM perception call failed (attempt %d/2): %s", attempt, e)
            if "429" in str(e):
                time.sleep(30 * attempt)
            continue
        parsed = _parse_perception(text, img_w, img_h)
        if parsed is not None:
            perception = parsed
            break
        log.warning("VLM perception unparseable response (attempt %d/2): %.200s", attempt, text)
    return perception


def _merge_perceptions(pass1: VLMPerception, pass2: VLMPerception) -> VLMPerception:
    """Merge two perception passes — keep ALL detections and OCR from both."""
    all_detections = list(pass1.detections) + list(pass2.detections)
    all_ocr = list(pass1.ocr_texts) + list(pass2.ocr_texts)

    return VLMPerception(
        scene_description=pass2.scene_description or pass1.scene_description,
        store_zone=pass2.store_zone or pass1.store_zone,
        detections=all_detections,
        ocr_texts=all_ocr,
    )


def perceive_two_pass(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    img_w: int,
    img_h: int,
) -> Tuple[VLMPerception, dict]:
    """Two-pass VLM perception: first pass detects, second pass refines with context.
    Returns (merged_perception, debug_dict)."""
    _empty_debug = {"pass1": {}, "pass2": {}, "merged": {}}
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot call VLM")
        return _EMPTY_PERCEPTION, _empty_debug

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
        return _EMPTY_PERCEPTION, _empty_debug

    # ── Pass 1: initial perception ──
    prompt1 = PERCEIVE_PROMPT
    pass1 = _EMPTY_PERCEPTION
    for attempt in (1, 2):
        try:
            text = _generate(prompt1, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM pass1 failed (attempt %d/2): %s", attempt, e)
            if "429" in str(e):
                time.sleep(30 * attempt)
            continue
        parsed = _parse_perception(text, img_w, img_h)
        if parsed is not None:
            pass1 = parsed
            break
        log.warning("VLM pass1 unparseable (attempt %d/2): %.200s", attempt, text)

    if not pass1.detections and not pass1.ocr_texts:
        log.warning("Pass 1 returned nothing — skipping pass 2")
        return pass1, _empty_debug

    # ── Pass 2: refine with pass 1 context ──
    if pass1.detections:
        prev_strs = ", ".join(d.label for d in pass1.detections)
    else:
        prev_strs = "(無)"
    prompt2 = PERCEIVE_REFINE_PROMPT.format(prev_items=prev_strs)

    pass2 = _EMPTY_PERCEPTION
    for attempt in (1, 2):
        try:
            text = _generate(prompt2, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM pass2 failed (attempt %d/2): %s", attempt, e)
            if "429" in str(e):
                time.sleep(30 * attempt)
            continue
        parsed = _parse_perception(text, img_w, img_h)
        if parsed is not None:
            pass2 = parsed
            break
        log.warning("VLM pass2 unparseable (attempt %d/2): %.200s", attempt, text)

    merged = _merge_perceptions(pass1, pass2)
    log.info("Two-pass perceive: pass1=%d det + %d ocr → pass2=%d det + %d ocr → merged=%d det + %d ocr",
             len(pass1.detections), len(pass1.ocr_texts),
             len(pass2.detections), len(pass2.ocr_texts),
             len(merged.detections), len(merged.ocr_texts))

    def _perc_to_dict(p: VLMPerception) -> dict:
        return {
            "scene": p.scene_description,
            "zone": p.store_zone,
            "detections": [{"label": d.label, "box": [round(b, 3) for b in d.bbox], "score": round(d.score, 2), "position": d.position} for d in p.detections],
            "ocr": [{"text": t.text, "score": round(t.score, 2), "position": t.position,
                     "bbox": [[round(c, 1) for c in pt] for pt in t.bbox] if t.bbox else None} for t in p.ocr_texts],
        }

    debug = {
        "pass1": _perc_to_dict(pass1),
        "pass2": _perc_to_dict(pass2),
        "merged": _perc_to_dict(merged),
    }
    return merged, debug


def _is_dup_detection(new_det: VLMDetectedObject, existing: List[VLMDetectedObject],
                      iou_thresh: float = 0.5) -> bool:
    """Check if new_det duplicates any existing detection (label overlap or bbox IoU)."""
    nl = new_det.label.lower()
    nb = new_det.bbox
    for ed in existing:
        el = ed.label.lower()
        if nl and el and (nl in el or el in nl):
            return True
        if nb and ed.bbox and len(nb) == 4 and len(ed.bbox) == 4:
            x1 = max(nb[0], ed.bbox[0]); y1 = max(nb[1], ed.bbox[1])
            x2 = min(nb[2], ed.bbox[2]); y2 = min(nb[3], ed.bbox[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (nb[2] - nb[0]) * (nb[3] - nb[1])
            a2 = (ed.bbox[2] - ed.bbox[0]) * (ed.bbox[3] - ed.bbox[1])
            union = a1 + a2 - inter
            if union > 0 and inter / union > iou_thresh:
                return True
    return False


def perceive_three_pass(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    img_w: int,
    img_h: int,
) -> Tuple[VLMPerception, dict]:
    """Three-pass VLM perception: two-pass + distinctive features pass.
    Returns (merged_perception, debug_dict)."""
    merged_2p, debug_2p = perceive_two_pass(image_path, goal, goal_objects, img_w, img_h)

    if not merged_2p.detections and not merged_2p.ocr_texts:
        log.warning("Pass 1+2 returned nothing — skipping pass 3")
        return merged_2p, debug_2p

    try:
        img = Image.open(image_path).convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        log.warning("Pass 3: could not read image %s: %s", image_path, e)
        return merged_2p, debug_2p

    prev_det_strs = [f"{d.label} (score={d.score:.1f})" for d in merged_2p.detections]
    prev_ocr_strs = [f'"{t.text}" (score={t.score:.1f})' for t in merged_2p.ocr_texts]
    prompt3 = PERCEIVE_DISTINCTIVE_PROMPT.format(
        goal=goal,
        goal_objects=", ".join(goal_objects) or "(none)",
        goal_label_example=(goal_objects[0] if goal_objects else "milk") or "(none)",
        prev_detections=", ".join(prev_det_strs) or "(none)",
        prev_ocr=", ".join(prev_ocr_strs) or "(none)",
    )

    pass3 = _EMPTY_PERCEPTION
    for attempt in (1, 2):
        try:
            text = _generate(prompt3, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM pass3 failed (attempt %d/2): %s", attempt, e)
            if "429" in str(e):
                time.sleep(30 * attempt)
            continue
        parsed = _parse_perception(text, img_w, img_h)
        if parsed is not None:
            pass3 = parsed
            break
        log.warning("VLM pass3 unparseable (attempt %d/2): %.200s", attempt, text)

    new_dets = [d for d in pass3.detections
                if not _is_dup_detection(d, merged_2p.detections)]
    new_ocr = []
    existing_texts = {t.text.lower().strip() for t in merged_2p.ocr_texts}
    for t in pass3.ocr_texts:
        if t.text.lower().strip() not in existing_texts:
            new_ocr.append(t)

    all_dets = list(merged_2p.detections) + new_dets
    all_ocr = list(merged_2p.ocr_texts) + new_ocr

    merged = VLMPerception(
        scene_description=merged_2p.scene_description,
        store_zone=merged_2p.store_zone,
        detections=all_dets,
        ocr_texts=all_ocr,
    )

    log.info("Three-pass perceive: 2-pass=%d det + %d ocr → pass3=%d raw, %d new det, %d new ocr → total=%d det + %d ocr",
             len(merged_2p.detections), len(merged_2p.ocr_texts),
             len(pass3.detections), len(new_dets), len(new_ocr),
             len(all_dets), len(all_ocr))

    def _perc_to_dict(p: VLMPerception) -> dict:
        return {
            "scene": p.scene_description,
            "zone": p.store_zone,
            "detections": [{"label": d.label, "box": [round(b, 3) for b in d.bbox],
                            "score": round(d.score, 2), "position": d.position}
                           for d in p.detections],
            "ocr": [{"text": t.text, "score": round(t.score, 2), "position": t.position,
                     "bbox": [[round(c, 1) for c in pt] for pt in t.bbox] if t.bbox else None}
                    for t in p.ocr_texts],
        }

    debug = {**debug_2p}
    debug["pass3"] = _perc_to_dict(pass3)
    debug["pass3_new_det"] = len(new_dets)
    debug["pass3_new_ocr"] = len(new_ocr)
    debug["merged"] = _perc_to_dict(merged)
    return merged, debug


def _grid_cell_for_bbox(bbox: List[float], img_w: int, img_h: int, grid_n: int = 3) -> int:
    """Map an absolute-pixel bbox [x1,y1,x2,y2] to an n×n grid cell (0 to n²-1)."""
    cx = (bbox[0] + bbox[2]) / 2 / max(img_w, 1)
    cy = (bbox[1] + bbox[3]) / 2 / max(img_h, 1)
    col = min(int(cx * grid_n), grid_n - 1)
    row = min(int(cy * grid_n), grid_n - 1)
    return row * grid_n + col


def _label_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b)) > 0.5
    a_chars = set(a) - {' '}
    b_chars = set(b) - {' '}
    if a_chars and b_chars:
        return len(a_chars & b_chars) / len(a_chars | b_chars) > 0.6
    return False


def _stabilize_detections(
    rounds: List[List[VLMDetectedObject]],
    img_w: int, img_h: int,
    min_appearances: int = 2,
) -> List[VLMDetectedObject]:
    """Keep only detections that appear in min_appearances+ rounds."""
    if len(rounds) == 1:
        return rounds[0]

    tagged = []
    for ri, dets in enumerate(rounds):
        for d in dets:
            cell = _grid_cell_for_bbox(d.bbox, img_w, img_h)
            tagged.append((ri, d, cell))

    clusters: List[List[tuple]] = []
    used = [False] * len(tagged)

    for i, (ri, di, ci) in enumerate(tagged):
        if used[i]:
            continue
        cluster = [(ri, di, ci)]
        used[i] = True
        rounds_in = {ri}
        for j in range(i + 1, len(tagged)):
            if used[j]:
                continue
            rj, dj, cj = tagged[j]
            if rj in rounds_in:
                continue
            if ci == cj and _label_match(di.label.lower(), dj.label.lower()):
                cluster.append((rj, dj, cj))
                used[j] = True
                rounds_in.add(rj)
        clusters.append(cluster)

    stable = []
    for cluster in clusters:
        if len(cluster) < min_appearances:
            continue
        items = [d for _, d, _ in cluster]
        best = max(items, key=lambda x: x.score)
        avg_bbox = [
            sum(d.bbox[k] for d in items) / len(items)
            for k in range(4)
        ]
        stable.append(VLMDetectedObject(
            label=best.label,
            bbox=[round(b, 1) for b in avg_bbox],
            score=round(max(d.score for d in items), 2),
            position=best.position,
        ))
    return stable


def _stabilize_ocr(
    rounds: List[List[VLMDetectedText]],
    min_appearances: int = 2,
) -> List[VLMDetectedText]:
    """Keep only OCR texts that appear in min_appearances+ rounds."""
    if len(rounds) == 1:
        return rounds[0]

    tagged = []
    for ri, texts in enumerate(rounds):
        for t in texts:
            tagged.append((ri, t))

    clusters: List[List[tuple]] = []
    used = [False] * len(tagged)

    for i, (ri, ti) in enumerate(tagged):
        if used[i]:
            continue
        cluster = [(ri, ti)]
        used[i] = True
        rounds_in = {ri}
        for j in range(i + 1, len(tagged)):
            if used[j]:
                continue
            rj, tj = tagged[j]
            if rj in rounds_in:
                continue
            if _label_match(ti.text.lower(), tj.text.lower()):
                cluster.append((rj, tj))
                used[j] = True
                rounds_in.add(rj)
        clusters.append(cluster)

    stable = []
    for cluster in clusters:
        if len(cluster) < min_appearances:
            continue
        items = [t for _, t in cluster]
        best = max(items, key=lambda x: x.score)
        stable.append(VLMDetectedText(
            text=best.text,
            score=round(max(t.score for t in items), 2),
            bbox=best.bbox,
            position=best.position,
        ))
    return stable


def perceive_multi_round(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    img_w: int,
    img_h: int,
    num_rounds: int = 3,
) -> Tuple[VLMPerception, dict]:
    """Run perceive_two_pass N times, keep only stable detections (2+/N).

    Returns (stable_perception, debug_dict) with combined debug info.
    """
    if num_rounds <= 1:
        return perceive_two_pass(image_path, goal, goal_objects, img_w, img_h)

    all_det_rounds: List[List[VLMDetectedObject]] = []
    all_ocr_rounds: List[List[VLMDetectedText]] = []
    round_debugs = []
    best_scene = ""
    best_zone = ""

    for r in range(num_rounds):
        log.info("Multi-round perception: round %d/%d", r + 1, num_rounds)
        perc, dbg = perceive_two_pass(image_path, goal, goal_objects, img_w, img_h)
        all_det_rounds.append(list(perc.detections))
        all_ocr_rounds.append(list(perc.ocr_texts))
        round_debugs.append(dbg)
        if perc.scene_description:
            best_scene = perc.scene_description
        if perc.store_zone:
            best_zone = perc.store_zone

    min_app = 2
    stable_dets = _stabilize_detections(all_det_rounds, img_w, img_h, min_app)
    stable_ocr = _stabilize_ocr(all_ocr_rounds, min_app)

    total_raw = sum(len(r) for r in all_det_rounds)
    log.info("Multi-round stabilize: %d rounds × 2-pass, %d raw det → %d stable det, "
             "%d raw ocr → %d stable ocr",
             num_rounds, total_raw, len(stable_dets),
             sum(len(r) for r in all_ocr_rounds), len(stable_ocr))

    stable_perc = VLMPerception(
        scene_description=best_scene,
        store_zone=best_zone,
        detections=stable_dets,
        ocr_texts=stable_ocr,
    )

    def _perc_to_dict(p: VLMPerception) -> dict:
        return {
            "scene": p.scene_description,
            "zone": p.store_zone,
            "detections": [{"label": d.label, "box": [round(b, 3) for b in d.bbox],
                            "score": round(d.score, 2), "position": d.position}
                           for d in p.detections],
            "ocr": [{"text": t.text, "score": round(t.score, 2), "position": t.position,
                     "bbox": [[round(c, 1) for c in pt] for pt in t.bbox] if t.bbox else None}
                    for t in p.ocr_texts],
        }

    debug = {
        "pass1": round_debugs[0].get("pass1", {}),
        "pass2": round_debugs[0].get("pass2", {}),
        "merged": _perc_to_dict(stable_perc),
        "multi_round": {
            "num_rounds": num_rounds,
            "raw_det_per_round": [len(r) for r in all_det_rounds],
            "raw_ocr_per_round": [len(r) for r in all_ocr_rounds],
            "stable_det": len(stable_dets),
            "stable_ocr": len(stable_ocr),
        },
    }
    return stable_perc, debug


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
