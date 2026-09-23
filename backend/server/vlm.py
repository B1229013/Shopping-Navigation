"""VLM client: send (image + prompt) to OpenAI (default, gpt-4o-mini), Gemini, or Ollama; parse structured response."""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import List, Optional

import requests
from PIL import Image, ImageOps

from server.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL,
    OLLAMA_URL, OLLAMA_MODEL,
    VLM_TIMEOUT_S, VLM_BACKEND,
)
from server.models import (
    VLMAction, VLMResponse,
    VLMDetectionItem, VLMOCRItem, AisleSignItem, VLMPerceptionResponse,
)
from server.prompts import (
    PER_TURN_PROMPT, PRIOR_ANSWER_BLOCK,
    VLM_PERCEPTION_PROMPT, VLM_NAVIGATION_PROMPT,
    PERCEIVE_PROMPT,
)

log = logging.getLogger(__name__)

_FALLBACK = VLMResponse(
    action=VLMAction.MOVE,
    guidance="無法辨識目前場景，請往前走幾步再拍一張照片上傳。",
    question=None,
    vlm_summary="",
)

_PERCEPTION_FALLBACK = VLMPerceptionResponse(
    detections=[], ocr_texts=[], scene_description="",
)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def _build_prompt(
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_summary: Optional[str] = None,
    progress: str = "",
    route_info: str = "",
) -> str:
    if prior_question and prior_answer:
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
        progress=progress,
        route_info=route_info,
    )


def _parse(text: str) -> Optional[VLMResponse]:
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


# ── Gemini backend ──

def _gemini_generate(prompt: str, image_b64: Optional[str] = None, max_tokens: int = 512) -> str:
    url = _GEMINI_URL.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)
    parts: list = []
    if image_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    parts.append({"text": prompt})
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens},
    }
    r = requests.post(url, json=body, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    resp = r.json()
    candidates = resp.get("candidates", [])
    if not candidates:
        return ""
    text_parts = [p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])]
    return "".join(text_parts)


# ── Ollama backend ──

def _ollama_generate(prompt: str, images: Optional[List[str]] = None) -> str:
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    if images:
        body["images"] = images
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json().get("response", "")


# ── OpenAI backend ──

def _openai_generate(prompt: str, image_b64: Optional[str] = None, max_tokens: int = 512) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    content: list = []
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    content.append({"type": "text", "text": prompt})
    model = OPENAI_MODEL
    use_new_api = any(model.startswith(p) for p in ("gpt-5", "gpt-4.1", "o3", "o4"))
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_completion_tokens" if use_new_api else "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, json=body, headers=headers, timeout=VLM_TIMEOUT_S)
    r.raise_for_status()
    choices = r.json().get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


# ── Unified dispatch ──

def _generate(prompt: str, image_b64: Optional[str] = None, max_tokens: int = 512) -> str:
    if VLM_BACKEND == "gemini":
        return _gemini_generate(prompt, image_b64=image_b64, max_tokens=max_tokens)
    elif VLM_BACKEND == "openai":
        return _openai_generate(prompt, image_b64=image_b64, max_tokens=max_tokens)
    else:
        return _ollama_generate(prompt, images=[image_b64] if image_b64 else None)


def ask_about_image(image_pil: Image.Image, prompt: str) -> str:
    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return _generate(prompt, image_b64=img_b64)


def warm_up() -> None:
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
    progress: str = "",
    route_info: str = "",
) -> VLMResponse:
    if VLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set — cannot call VLM")
        return _FALLBACK
    if VLM_BACKEND == "openai" and not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot call VLM")
        return _FALLBACK

    prompt = _build_prompt(goal, goal_objects, topomap_summary, detections_summary, prior_question, prior_answer, ocr_summary=ocr_summary, progress=progress, route_info=route_info)
    if route_info:
        log.info("VLM decide with route_info: %s", route_info[:150])
    try:
        img = ImageOps.exif_transpose(Image.open(image_path))
        img = img.convert("RGB")
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
            return parsed
        log.warning("VLM unparseable response (attempt %d/2): %.200s", attempt, text)
    return _FALLBACK


# ── Two-stage VLM perception + navigation ──
# Stage 1: VLM analyzes the image → detections + OCR + scene_description (with bbox AND position)
# Stage 2: VLM decides navigation action using Stage 1's spatial descriptions


def _parse_perception(text: str) -> Optional[VLMPerceptionResponse]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        dets = []
        for d in obj.get("detections", []):
            label = str(d.get("label", ""))
            position = str(d.get("position", ""))
            if not label:
                continue
            box = d.get("bbox", d.get("box", []))
            bbox = [float(v) for v in box] if isinstance(box, list) and len(box) == 4 else [0, 0, 0, 0]
            dets.append(VLMDetectionItem(
                label=label, bbox=bbox,
                score=float(d.get("score", d.get("confidence", 0.5))),
                position=position,
            ))
        ocr_items = []
        for t in obj.get("ocr_texts", []):
            text = str(t.get("text", ""))
            position = str(t.get("position", ""))
            if not text:
                continue
            raw_bbox = t.get("bbox", [])
            if isinstance(raw_bbox, list) and len(raw_bbox) == 4 and all(
                isinstance(p, list) and len(p) == 2 for p in raw_bbox
            ):
                bbox = [[float(c) for c in p] for p in raw_bbox]
            else:
                bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]
            ocr_items.append(VLMOCRItem(
                text=text, bbox=bbox,
                score=float(t.get("score", t.get("confidence", 0.5))),
                position=position,
            ))
        aisle_signs = []
        for a in obj.get("aisle_signs", []):
            num = a.get("aisle_number")
            cats = a.get("categories", [])
            pos = str(a.get("position", ""))
            if num is not None or cats:
                aisle_signs.append(AisleSignItem(
                    aisle_number=int(num) if num is not None else None,
                    categories=[str(c) for c in cats] if isinstance(cats, list) else [],
                    position=pos,
                ))
        return VLMPerceptionResponse(
            detections=dets,
            ocr_texts=ocr_items,
            aisle_signs=aisle_signs,
            scene_description=str(obj.get("scene_description", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("VLM perception JSON parse failed: %s", e)
        return None


# ── Sign / category keyword sets for OCR extraction from zh labels ──
_SIGN_KEYWORDS = {
    "標示", "指示", "告示", "看板", "招牌", "吊牌", "吊旗",
    "走道", "區域", "出口", "入口", "結帳", "收銀",
    "sign", "banner", "aisle", "exit", "zone",
    "手扶梯", "電扶梯", "樓梯", "廁所", "服務台",
}
_CATEGORY_KEYWORDS = {
    "堅果", "海苔", "飲料", "零食", "冷凍", "冷藏", "生鮮", "日用",
    "烘焙", "麵包", "肉品", "海鮮", "水果", "蔬菜", "花卉",
    "咖啡", "茶", "乳品", "奶粉", "保健", "清潔", "衛生",
    "寵物", "酒", "啤酒", "調味", "罐頭", "餅乾", "早餐",
    "麵條", "即食", "米", "南北貨", "護理", "洗髮", "染髮",
    # ── 乳製品 / 常見商品名（防止包裝文字被過濾） ──
    "牛奶", "鮮奶", "鮮乳", "豆漿", "優酪乳", "優格",
    "起司", "乳酪", "雞蛋", "牙膏", "洗碗", "衛生紙",
    "milk", "dairy", "yogurt", "cheese",
}

# Bilingual synonym map for goal-term expansion in OCR extraction.
# When goal_objects is Chinese-only (e.g. decompose fallback), this
# lets English VLM detection labels still match.
_GOAL_SYNONYMS: dict[str, list[str]] = {
    "牛奶": ["milk", "dairy", "鮮奶", "鮮乳", "乳品"],
    "鮮奶": ["milk", "dairy", "牛奶", "鮮乳"],
    "鮮乳": ["milk", "dairy", "牛奶", "鮮奶"],
    "雞蛋": ["egg", "eggs", "蛋"],
    "麵包": ["bread", "bakery", "烘焙"],
    "衛生紙": ["toilet paper", "tissue", "紙巾"],
    "飲料": ["drink", "beverage", "drinks"],
    "零食": ["snack", "snacks", "chips"],
    "水果": ["fruit", "fruits", "produce"],
    "蔬菜": ["vegetable", "vegetables", "produce"],
    "起司": ["cheese", "乳酪"],
    "優格": ["yogurt", "優酪乳"],
    "咖啡": ["coffee"],
    "豆漿": ["soy milk", "soymilk"],
    "洗衣精": ["laundry detergent", "detergent"],
    "牙膏": ["toothpaste"],
}


def _expand_goal_terms(goal_objects: List[str]) -> set[str]:
    """Expand goal_objects with bilingual synonyms for substring matching."""
    terms: set[str] = set()
    for g in goal_objects:
        gl = g.lower().strip()
        if not gl:
            continue
        terms.add(gl)
        if gl in _GOAL_SYNONYMS:
            terms.update(s.lower() for s in _GOAL_SYNONYMS[gl])
        for key, syns in _GOAL_SYNONYMS.items():
            if gl == key.lower() or gl in [s.lower() for s in syns]:
                terms.add(key.lower())
                terms.update(s.lower() for s in syns)
    return terms


def _parse_perceive_prompt(
    text: str, img_w: int, img_h: int,
    goal_objects: Optional[List[str]] = None,
) -> Optional[VLMPerceptionResponse]:
    """Parse PERCEIVE_PROMPT output → VLMPerceptionResponse.

    PERCEIVE_PROMPT returns: {scene, items[{zh, en, conf, box}]}
      - box values are 0~1 ratios → convert to absolute pixel coords
      - big boxes (>50% width or height) are filtered out
      - zh labels containing sign/category keywords are also emitted as ocr_texts
      - zh/en labels matching goal_objects are also emitted as ocr_texts
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    detections: List[VLMDetectionItem] = []
    ocr_texts: List[VLMOCRItem] = []

    for item in data.get("items", []):
        box = item.get("box", [])
        if not isinstance(box, list) or len(box) != 4:
            continue

        # Big-box filter (>50%)
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        if bw >= 0.55 or bh >= 0.55:
            continue

        # Ratio → absolute pixel bbox
        px_box = [
            box[0] * img_w,
            box[1] * img_h,
            box[2] * img_w,
            box[3] * img_h,
        ]

        zh = item.get("zh", "")
        en = item.get("en", "")
        label = en if en else zh   # APP internals use English labels
        conf = float(item.get("conf", 0.5))

        # Compute coarse position description from ratio box
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        h_pos = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
        v_pos = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
        area = bw * bh
        depth = "near" if area > 0.05 else "far"
        position = f"{h_pos} {v_pos}, {depth}"

        detections.append(VLMDetectionItem(
            label=label,
            bbox=px_box,
            score=conf,
            position=position,
        ))

        # If zh/en contains sign/category keywords or matches goal, also add as OCR text
        emit_as_ocr = False
        combined_lower = (zh + " " + en).lower()
        is_sign = any(kw in combined_lower for kw in _SIGN_KEYWORDS)
        is_category = any(kw in combined_lower for kw in _CATEGORY_KEYWORDS)
        emit_as_ocr = is_sign or is_category or bool(re.search(r"走道\s*\d+", zh))
        # Goal-match: if zh or en matches any goal_object (with synonym expansion)
        # so downstream match_ocr_to_goal() can produce SIGN MATCH evidence
        if not emit_as_ocr and goal_objects:
            expanded = _expand_goal_terms(goal_objects)
            for g in expanded:
                if g and (g in combined_lower or combined_lower in g):
                    emit_as_ocr = True
                    break
        if emit_as_ocr and (zh or en):
            ocr_text = zh if zh else en
            ocr_bbox = [
                [px_box[0], px_box[1]],
                [px_box[2], px_box[1]],
                [px_box[2], px_box[3]],
                [px_box[0], px_box[3]],
            ]
            ocr_texts.append(VLMOCRItem(
                text=ocr_text,
                bbox=ocr_bbox,
                score=conf,
                position=position,
            ))

    return VLMPerceptionResponse(
        detections=detections,
        ocr_texts=ocr_texts,
        aisle_signs=[],
        scene_description=data.get("scene", ""),
    )


def format_perception_detections(dets: List[VLMDetectionItem]) -> str:
    if not dets:
        return "(no detections)"
    lines = []
    for d in dets:
        pos = f" - {d.position}" if d.position else ""
        lines.append(f"{d.label} ({d.score:.0%}){pos}")
    return "\n".join(lines)


def _format_perception_ocr(ocr_items: List[VLMOCRItem]) -> str:
    if not ocr_items:
        return "(no text detected)"
    lines = []
    for t in ocr_items:
        pos = f" - {t.position}" if t.position else ""
        lines.append(f'"{t.text}" ({t.score:.0%}){pos}')
    return "\n".join(lines)


def _prepare_image_b64(image_path: str) -> Optional[str]:
    try:
        img = ImageOps.exif_transpose(Image.open(image_path))
        img = img.convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        log.info("VLM image prepared: %s, %d KB", img.size, len(buf.getvalue()) // 1024)
        return img_b64
    except Exception as e:
        log.warning("VLM could not read image %s: %s", image_path, e)
        return None


def _pos_to_bbox(position: str, img_w: int, img_h: int) -> List[float]:
    """Convert a position description like 'left top, near' into a pixel bbox.

    VLMs are good at relative position descriptions but bad at exact
    coordinates, so we derive the bbox from the text position instead.
    The image is divided into a 3x3 grid; each cell gets slight inward
    padding so overlapping labels don't stack exactly.
    """
    pos = position.lower()
    # horizontal
    if "left" in pos:
        x1, x2 = 0.0, 0.38
    elif "right" in pos:
        x1, x2 = 0.62, 1.0
    else:
        x1, x2 = 0.28, 0.72
    # vertical
    if "top" in pos:
        y1, y2 = 0.0, 0.38
    elif "bottom" in pos:
        y1, y2 = 0.62, 1.0
    else:
        y1, y2 = 0.28, 0.72
    # depth: near = larger box (closer), far = smaller box
    if "near" in pos:
        margin = 0.02
    elif "far" in pos:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * 0.55, (y2 - y1) * 0.55
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        margin = 0.0
    else:
        margin = 0.01
    return [
        max(0, (x1 + margin)) * img_w,
        max(0, (y1 + margin)) * img_h,
        min(1, (x2 - margin)) * img_w,
        min(1, (y2 - margin)) * img_h,
    ]


def _is_valid_bbox(bbox, img_w: int, img_h: int) -> bool:
    """Check if a bbox looks like valid absolute pixel coordinates."""
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0 or x2 > img_w * 1.05 or y2 > img_h * 1.05:
        return False
    area = (x2 - x1) * (y2 - y1)
    if area < 4:
        return False
    return True


def _denormalize_perception(
    p: VLMPerceptionResponse, img_w: int, img_h: int,
) -> VLMPerceptionResponse:
    """Prefer VLM-returned bbox; fall back to position-based estimate."""
    new_dets = []
    for d in p.detections:
        if _is_valid_bbox(d.bbox, img_w, img_h):
            bbox = [float(v) for v in d.bbox[:4]]
        elif d.position:
            bbox = _pos_to_bbox(d.position, img_w, img_h)
        else:
            bbox = [0, 0, img_w, img_h]
        new_dets.append(VLMDetectionItem(
            label=d.label, bbox=bbox, score=d.score, position=d.position))

    new_ocr = []
    for t in p.ocr_texts:
        if t.bbox and len(t.bbox) == 4 and all(
            isinstance(v, (int, float)) for v in t.bbox
        ):
            x1, y1, x2, y2 = [float(v) for v in t.bbox]
            if x2 > x1 and y2 > y1:
                bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                b = _pos_to_bbox(t.position, img_w, img_h) if t.position else [0, 0, img_w, img_h]
                bbox = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        elif t.position:
            b = _pos_to_bbox(t.position, img_w, img_h)
            bbox = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        else:
            bbox = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
        new_ocr.append(VLMOCRItem(
            text=t.text, bbox=bbox, score=t.score, position=t.position))

    return VLMPerceptionResponse(
        detections=new_dets, ocr_texts=new_ocr,
        aisle_signs=p.aisle_signs,
        scene_description=p.scene_description)


def perceive_and_decide(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    img_w: int,
    img_h: int,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_formatter: Optional[callable] = None,
    progress: str = "",
    route_info: str = "",
) -> tuple[VLMPerceptionResponse, VLMResponse]:
    """Two-stage VLM call. Returns (perception, navigation_decision)."""
    if VLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set — cannot call VLM")
        return _PERCEPTION_FALLBACK, _FALLBACK
    if VLM_BACKEND == "openai" and not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot call VLM")
        return _PERCEPTION_FALLBACK, _FALLBACK

    img_b64 = _prepare_image_b64(image_path)
    if img_b64 is None:
        return _PERCEPTION_FALLBACK, _FALLBACK

    # ── Stage 1: Perception (PERCEIVE_PROMPT — 繁中、比例座標) ──
    perception_prompt = PERCEIVE_PROMPT.format(
        goal_objects=", ".join(goal_objects) or "(none)",
    )
    perception = None
    for attempt in (1, 2):
        try:
            text = _generate(perception_prompt, image_b64=img_b64, max_tokens=4096)
        except Exception as e:
            log.warning("VLM Stage 1 failed (attempt %d/2): %s", attempt, e)
            continue
        perception = _parse_perceive_prompt(text, img_w, img_h, goal_objects=goal_objects)
        if perception is not None:
            log.info("VLM Stage 1: %d detections, %d OCR texts, scene: %s",
                     len(perception.detections), len(perception.ocr_texts),
                     perception.scene_description[:80])
            break
        log.warning("VLM Stage 1 unparseable (attempt %d/2): %.300s", attempt, text)

    if perception is None:
        perception = _PERCEPTION_FALLBACK

    # PERCEIVE_PROMPT 的 _parse_perceive_prompt 已經輸出像素座標，不需 _denormalize

    # ── Stage 2: Navigation decision ──
    detections_summary = format_perception_detections(perception.detections)
    if ocr_formatter:
        ocr_summary = ocr_formatter(perception.ocr_texts)
        log.info("VLM Stage 2 OCR (cleaned): %s", ocr_summary[:200])
    else:
        ocr_summary = _format_perception_ocr(perception.ocr_texts)

    if prior_question and prior_answer:
        answer_block = PRIOR_ANSWER_BLOCK.format(
            previous_question=prior_question, user_answer=prior_answer,
        )
    else:
        answer_block = ""

    nav_prompt = VLM_NAVIGATION_PROMPT.format(
        goal=goal,
        goal_objects=", ".join(goal_objects) or "(none)",
        topomap_summary=topomap_summary or "(starting location)",
        scene_description=perception.scene_description or "(no description)",
        detections_summary=detections_summary,
        ocr_summary=ocr_summary,
        prior_answer_block=answer_block,
        progress=progress,
        route_info=route_info,
    )

    nav_resp = None
    for attempt in (1, 2):
        try:
            text = _generate(nav_prompt, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM Stage 2 failed (attempt %d/2): %s", attempt, e)
            continue
        nav_resp = _parse(text)
        if nav_resp is not None:
            log.info("VLM Stage 2: %s | %s", nav_resp.action.value, nav_resp.guidance[:120])
            break
        log.warning("VLM Stage 2 unparseable (attempt %d/2): %.200s", attempt, text)

    if nav_resp is None:
        nav_resp = _FALLBACK

    return perception, nav_resp


def perceive_only(
    image_path: str,
    goal_objects: List[str],
    img_w: int,
    img_h: int,
) -> VLMPerceptionResponse:
    """Returns perception with pixel-space bboxes.
    Call navigate_with_perception() separately for Stage 2 after
    localization and route planning.
    """
    if VLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set")
        return _PERCEPTION_FALLBACK
    if VLM_BACKEND == "openai" and not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set")
        return _PERCEPTION_FALLBACK

    img_b64 = _prepare_image_b64(image_path)
    if img_b64 is None:
        return _PERCEPTION_FALLBACK

    perception_prompt = PERCEIVE_PROMPT.format(
        goal_objects=", ".join(goal_objects) or "(none)",
    )

    perception = None
    for attempt in (1, 2):
        try:
            text = _generate(perception_prompt, image_b64=img_b64, max_tokens=4096)
        except Exception as e:
            log.warning("VLM perception failed (attempt %d/2): %s", attempt, e)
            continue
        perception = _parse_perceive_prompt(text, img_w, img_h, goal_objects=goal_objects)
        if perception is not None:
            log.info("VLM perception: %d dets, %d OCR, scene: %s",
                     len(perception.detections), len(perception.ocr_texts),
                     perception.scene_description[:80])
            break
        log.warning("VLM perception unparseable (attempt %d/2): %.300s", attempt, text)

    if perception is None:
        perception = _PERCEPTION_FALLBACK

    return perception


def navigate_with_perception(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    scene_description: str,
    detections_summary: str,
    ocr_summary: str,
    prior_question: Optional[str] = None,
    prior_answer: Optional[str] = None,
    progress: str = "",
    route_info: str = "",
) -> VLMResponse:
    """Stage 2 only: navigation decision using pre-computed perception context.

    Designed to be called after perceive_only() + localization + route planning,
    so route_info reflects the CURRENT turn's route data.
    """
    if VLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        return _FALLBACK
    if VLM_BACKEND == "openai" and not OPENAI_API_KEY:
        return _FALLBACK

    img_b64 = _prepare_image_b64(image_path)
    if img_b64 is None:
        return _FALLBACK

    if prior_question and prior_answer:
        answer_block = PRIOR_ANSWER_BLOCK.format(
            previous_question=prior_question, user_answer=prior_answer,
        )
    else:
        answer_block = ""

    nav_prompt = VLM_NAVIGATION_PROMPT.format(
        goal=goal,
        goal_objects=", ".join(goal_objects) or "(none)",
        topomap_summary=topomap_summary or "(starting location)",
        scene_description=scene_description or "(no description)",
        detections_summary=detections_summary,
        ocr_summary=ocr_summary,
        prior_answer_block=answer_block,
        progress=progress,
        route_info=route_info,
    )

    if route_info:
        log.info("VLM navigate with route_info: %s", route_info[:150])

    nav_resp = None
    for attempt in (1, 2):
        try:
            text = _generate(nav_prompt, image_b64=img_b64)
        except Exception as e:
            log.warning("VLM navigate failed (attempt %d/2): %s", attempt, e)
            continue
        nav_resp = _parse(text)
        if nav_resp is not None:
            log.info("VLM navigate: %s | %s", nav_resp.action.value, nav_resp.guidance[:120])
            break
        log.warning("VLM navigate unparseable (attempt %d/2): %.200s", attempt, text)

    return nav_resp if nav_resp is not None else _FALLBACK
