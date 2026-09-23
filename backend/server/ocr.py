"""
ocr.py — 文字辨識模組 / OCR (Optical Character Recognition) Module

功能說明 (Purpose):
    使用 EasyOCR 從照片中辨識文字，支援英文 + 繁體中文。
    辨識結果會提供給 VLM 作為導航判斷的輔助資訊。
    Uses EasyOCR to extract text from photos (English + Traditional Chinese).
    Results are fed into the VLM prompt alongside object detections.

應用場景 (Use cases):
    - 門牌號碼（如「E301」）→ 精確定位地點
      Door nameplates (e.g., "E301") → precise location identification
    - 走廊標示（如「往實驗室→」）→ 方向指引
      Corridor signs (e.g., "→ Lab") → directional guidance
    - 商店貨架標籤（如「乳製品區」）→ 區域辨識
      Store shelf labels (e.g., "Dairy Section") → zone identification

被誰呼叫 (Called by):
    - server.py: upload_photo() 呼叫 ocr.read() 辨識照片文字

依賴 (Dependencies):
    - easyocr: 多語言文字辨識引擎
    - torch: GPU/CPU 裝置選擇
    - PIL: 影像縮放
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from server import config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 資料結構 / Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OCRResult:
    """
    單一 OCR 辨識結果 / A single OCR recognition result

    Attributes:
        text:       辨識出的文字內容 / Recognized text content
        confidence: 辨識信心分數 0~1，越高越確定 / Confidence score 0~1
        bbox:       文字區域的四個角點座標 / Four corner points of text region
                    格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]（順時針）
                    Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (clockwise)

    說明 (Note):
        bbox 是四邊形而非矩形，因為文字可能有傾斜角度
        bbox is a quadrilateral (not rectangle) because text can be tilted
    """
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


# ══════════════════════════════════════════════════════════════════════════════
# OCR 引擎 / OCR Engine
# ══════════════════════════════════════════════════════════════════════════════

class OCR:
    """
    EasyOCR 封裝，支援延遲載入 / EasyOCR wrapper with lazy loading

    設計決策 (Design decisions):
        - 延遲載入：第一次呼叫 read() 時才載入模型，避免拖慢伺服器啟動
          Lazy loading: model loads on first read() call, not at server startup
        - GPU 自動選擇：和 perception.py 相同的 VRAM 判斷邏輯
          Auto GPU selection: same VRAM check logic as perception.py
        - 預設語言：英文 + 繁體中文（台灣室內環境常見雙語標示）
          Default languages: English + Traditional Chinese (common in Taiwan indoor)
    """

    def __init__(self, languages: Optional[List[str]] = None) -> None:
        """
        初始化 OCR 引擎（不立即載入模型）
        Initialize OCR engine (does NOT load model yet)

        參數 (Parameters):
            languages: EasyOCR 語言代碼列表 / EasyOCR language code list
                      預設 ["en", "ch_tra"]（英文 + 繁體中文）
                      Default: ["en", "ch_tra"] (English + Traditional Chinese)
        """
        self._reader = None  # EasyOCR Reader 實例，load() 後才有值 / set after load()
        self._languages = languages or ["en", "ch_tra"]

    def load(self) -> None:
        """
        載入 EasyOCR 模型 / Load EasyOCR model

        GPU 選擇邏輯 (GPU selection logic):
            與 Perception 相同：VRAM >= 3GB 才用 GPU
            Same as Perception: use GPU only if VRAM >= 3GB
        """
        import easyocr

        log.info("Loading EasyOCR with languages: %s", self._languages)

        # 判斷是否使用 GPU / Determine whether to use GPU
        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                use_gpu = vram_mb >= 3072
            else:
                use_gpu = False
        except ImportError:
            use_gpu = False

        # 建立 EasyOCR Reader / Create EasyOCR Reader
        self._reader = easyocr.Reader(
            self._languages,
            gpu=use_gpu,
            verbose=False,   # 關閉冗長輸出 / Suppress verbose output
        )
        log.info("EasyOCR loaded successfully")

    def read(
        self,
        image_path: str,
        min_confidence: float = 0.3,
        max_results: int = 15,
    ) -> List[OCRResult]:
        """
        從照片中辨識文字 / Extract text from a photo

        參數 (Parameters):
            image_path:     照片檔案路徑 / Path to image file
            min_confidence: 最低信心閾值，低於此值的結果會被丟棄
                           Minimum confidence threshold, results below are discarded
                           預設 0.3（30%）/ Default: 0.3 (30%)
            max_results:    最多回傳幾個結果 / Maximum results to return
                           預設 15 / Default: 15

        回傳 (Returns):
            OCRResult 列表，按信心分數降序排列
            List of OCRResult, sorted by confidence descending

        處理流程 (Processing flow):
            1. 若模型未載入 → 自動載入（延遲載入）
               If model not loaded → auto load (lazy loading)
            2. 若影像太大（>2048px）→ 先縮小（避免 OOM）
               If image too large (>2048px) → resize first (prevent OOM)
            3. 呼叫 EasyOCR → 取得原始結果
               Call EasyOCR → get raw results
            4. 過濾：移除空文字、低信心結果
               Filter: remove empty text, low confidence results
            5. 按信心排序，取前 max_results 個
               Sort by confidence, take top max_results
        """
        # 延遲載入：第一次呼叫時才載入模型
        # Lazy loading: load model on first call
        if self._reader is None:
            self.load()

        try:
            import numpy as np
            from PIL import Image as _PILImage
            img = _PILImage.open(image_path)

            max_dim = 2048
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim))

            img_array = np.array(img.convert("RGB"))
            raw = self._reader.readtext(img_array)

        except Exception as e:
            log.warning("OCR failed on %s: %s", image_path, e)
            return []

        # 過濾與轉換原始結果 / Filter and convert raw results
        # EasyOCR 原始格式: [(bbox, text, confidence), ...]
        # EasyOCR raw format: [(bbox, text, confidence), ...]
        results: List[OCRResult] = []
        for bbox, text, conf in raw:
            text = text.strip()
            # 跳過空文字和低信心結果 / Skip empty text and low confidence
            if not text or conf < min_confidence:
                continue
            results.append(OCRResult(
                text=text,
                confidence=float(conf),
                # 將 bbox 轉為 float 列表 / Convert bbox to float list
                bbox=[[float(c) for c in pt] for pt in bbox],
            ))

        # 按信心降序排列，取前 max_results 個
        # Sort by confidence descending, take top max_results
        results.sort(key=lambda r: -r.confidence)
        return results[:max_results]

    @property
    def is_loaded(self) -> bool:
        """模型是否已載入 / Whether the model is loaded"""
        return self._reader is not None


# ══════════════════════════════════════════════════════════════════════════════
# OCR 摘要工具 / OCR Summary Utility
# ══════════════════════════════════════════════════════════════════════════════

def summarize_ocr(results: List[OCRResult]) -> str:
    """
    將 OCR 結果整理成一行文字摘要，供 VLM prompt 使用
    Build a concise one-line text summary for the VLM prompt

    範例輸出 (Example output):
        '"E301" (95%), "實驗室" (87%), "往電梯→" (72%)'

    去重邏輯 (Deduplication):
        同一張照片可能在不同位置辨識到相同文字（如重複的標示），
        用 normalize + set 去除重複
        Same text may be recognized at multiple positions in one photo,
        deduplicated using normalize + set

    參數 (Parameters):
        results: OCRResult 列表 / List of OCRResult

    回傳 (Returns):
        格式化的文字摘要，最多 10 個結果
        Formatted text summary, at most 10 results
    """
    if not results:
        return "(no text detected)"

    # 去重：正規化後比對 / Deduplicate: compare after normalization
    seen: set[str] = set()
    unique: List[OCRResult] = []
    for r in results:
        normalized = r.text.lower().strip()
        if normalized not in seen and len(normalized) > 0:
            seen.add(normalized)
            unique.append(r)

    # 格式化輸出 / Format output
    parts: List[str] = []
    for r in unique[:10]:  # 最多 10 個 / At most 10
        parts.append(f'"{r.text}" ({r.confidence:.0%})')

    return ", ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI 視覺 OCR / OpenAI vision OCR backend (gpt-4o-mini)
# ══════════════════════════════════════════════════════════════════════════════

_OPENAI_OCR_PROMPT = (
    "You are an OCR engine. Extract every piece of text visible in this image, "
    "exactly as written, preserving Traditional Chinese characters. For each "
    "distinct text region, give its text, a coarse position, and a confidence "
    "between 0 and 1. Respond with ONLY a JSON object of the form:\n"
    '{"texts": [{"text": "...", "position": "left|center|right top|middle|bottom", '
    '"confidence": 0.0}]}\n'
    'If no text is visible, respond with {"texts": []}.'
)


def _position_to_bbox(position: str, img_w: int, img_h: int) -> List[List[float]]:
    """Map a coarse position word ('left top', 'center', …) to a 4-point pixel bbox.

    gpt-4o-mini reports rough position reliably but not exact coordinates, so we
    derive a quadrilateral from the position text. A valid (non-empty) 4-point
    bbox is required downstream — scene._bbox_center averages the corners and
    server._bbox_key hashes them, so an empty bbox would crash the pipeline.
    """
    pos = (position or "").lower()
    if "left" in pos:
        x1, x2 = 0.0, 0.34
    elif "right" in pos:
        x1, x2 = 0.66, 1.0
    else:
        x1, x2 = 0.33, 0.67
    if "top" in pos:
        y1, y2 = 0.0, 0.34
    elif "bottom" in pos:
        y1, y2 = 0.66, 1.0
    else:
        y1, y2 = 0.33, 0.67
    ax1, ay1, ax2, ay2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
    return [[ax1, ay1], [ax2, ay1], [ax2, ay2], [ax1, ay2]]


class OpenAIOCR:
    """gpt-4o-mini vision OCR — a drop-in replacement for the EasyOCR ``OCR`` class.

    Exposes the same load()/read()/is_loaded interface and returns List[OCRResult],
    so server.get_ocr() swaps engines via the OCR_BACKEND setting with no other
    pipeline changes. Text is extracted by a single OpenAI /chat/completions vision
    call (temperature 0) using the standard payload — same endpoint, headers, and
    model family as the VLM, so OCR and VLM share one key and both run on gpt-4o-mini.
    """

    def __init__(self, languages: Optional[List[str]] = None) -> None:
        # languages kept for interface parity with OCR(); the prompt asks for
        # Traditional Chinese + English regardless of this value.
        self._languages = languages or ["en", "ch_tra"]
        self._ready = False

    def load(self) -> None:
        """Validate that a key is configured. No model weights to download."""
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY (or CGU_API_KEY) is empty — the OpenAI OCR backend "
                "cannot run. Set it in .env, or set OCR_BACKEND=easyocr."
            )
        self._ready = True

    @property
    def is_loaded(self) -> bool:
        return self._ready

    # ── internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_image(image_path: str) -> tuple[str, int, int]:
        import base64
        import io
        from PIL import Image as _PILImage

        img = _PILImage.open(image_path).convert("RGB")
        w, h = img.size
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode(), w, h

    @staticmethod
    def _call(img_b64: str) -> str:
        import requests

        url = f"{config.OPENAI_BASE_URL}/chat/completions"
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": _OPENAI_OCR_PROMPT},
        ]
        body = {
            "model": config.OPENAI_OCR_MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        r = requests.post(url, json=body, headers=headers, timeout=config.VLM_TIMEOUT_S)
        r.raise_for_status()
        choices = r.json().get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    @staticmethod
    def _parse(text: str) -> list:
        import json
        import re

        if not text:
            return []
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            obj = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return []
        items = obj.get("texts") or obj.get("ocr_texts") or obj.get("results") or []
        return items if isinstance(items, list) else []

    def read(
        self,
        image_path: str,
        min_confidence: float = 0.3,
        max_results: int = 15,
    ) -> List[OCRResult]:
        """Extract text via gpt-4o-mini. Same signature/return type as OCR.read()."""
        if not self._ready:
            self.load()
        try:
            img_b64, img_w, img_h = self._prepare_image(image_path)
        except Exception as e:
            log.warning("OpenAI OCR could not read image %s: %s", image_path, e)
            return []
        try:
            raw = self._call(img_b64)
        except Exception as e:
            log.warning("OpenAI OCR request failed: %s", e)
            return []

        results: List[OCRResult] = []
        for item in self._parse(raw):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                conf = float(item.get("confidence", item.get("score", 0.9)))
            except (TypeError, ValueError):
                conf = 0.9
            if conf < min_confidence:
                continue
            bbox = _position_to_bbox(str(item.get("position", "")), img_w, img_h)
            results.append(OCRResult(text=text, confidence=conf, bbox=bbox))

        results.sort(key=lambda r: -r.confidence)
        return results[:max_results]
