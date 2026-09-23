"""OCR 模組：從照片中擷取文字，協助辨識地點與標示。

使用 EasyOCR 做多語言文字辨識（英文 + 繁體中文）。辨識出的文字會跟
GroundingDINO 的偵測結果一起放進 VLM 提示詞，讓系統不需要使用者輸入，
就能自己讀懂走道標示、商品標籤、分區名稱、門牌等環境文字。

跟 Navigation-main-4/navigate_one_by_one.py 的 run_ocr_with_bbox() 相比，
這裡的實作其實更完整：
    - navigate_one_by_one.py 每次都用原圖尺寸辨識，大圖（例如手機直出的
      4032×3024 相片）會拖慢 EasyOCR；這裡在圖片超過 2048px 時會先縮圖
      再辨識（見 read() 內的縮放邏輯）。
    - navigate_one_by_one.py 沒有做 GPU/CPU 自動判斷；這裡會偵測顯卡
      VRAM 是否足夠再決定是否用 GPU 跑 EasyOCR（見 load()）。
    - navigate_one_by_one.py 用 tuple 回傳 (box, text, conf)；這裡改用
      OCRResult dataclass，型別更清楚也方便加欄位。
因此這次只補上中文註解說明邏輯，不搬動參考腳本的寫法。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]，四個角點座標


class OCR:
    """延遲載入（lazy-load）的 EasyOCR 包裝：模型直到第一次呼叫 read() 才會真正載入。"""

    def __init__(self, languages: Optional[List[str]] = None) -> None:
        self._reader = None
        self._languages = languages or ["en", "ch_tra"]

    def load(self) -> None:
        """實際建立 EasyOCR Reader（會下載/載入模型權重，第一次呼叫較慢）。"""
        import easyocr

        log.info("Loading EasyOCR with languages: %s", self._languages)
        try:
            import torch
            if torch.cuda.is_available():
                # EasyOCR 的辨識模型約需 3GB VRAM，VRAM 不夠時寧可退回 CPU，
                # 避免跟 GroundingDINO 搶顯卡記憶體導致兩邊都爆記憶體。
                vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                use_gpu = vram_mb >= 3072
            else:
                use_gpu = False
        except ImportError:
            use_gpu = False
        self._reader = easyocr.Reader(
            self._languages,
            gpu=use_gpu,
            verbose=False,
        )
        log.info("EasyOCR loaded successfully")

    def read(
        self,
        image_path: str,
        min_confidence: float = 0.3,
        max_results: int = 15,
    ) -> List[OCRResult]:
        """從圖片中擷取文字。

        Args:
            image_path: 圖片檔案路徑。
            min_confidence: 信心分數低於此值的文字會被濾掉。
            max_results: 最多回傳幾筆結果（依信心分數排序後取前 N 筆）。

        Returns:
            依信心分數由高到低排序的 OCRResult 清單。
        """
        if self._reader is None:
            self.load()

        try:
            from PIL import Image as _PILImage
            img = _PILImage.open(image_path)
            # 圖片過大時先縮小再辨識：EasyOCR 的耗時大致與像素數成正比，
            # 手機直出的相片動輒 4000px 以上，縮到 2048px 對辨識準確度
            # 影響不大，卻能大幅縮短這一步的時間。
            max_dim = 2048
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim))
                resized_path = image_path + ".ocr_tmp.jpg"
                img.save(resized_path, format="JPEG", quality=90)
                raw = self._reader.readtext(resized_path)
                import os; os.remove(resized_path)
            else:
                raw = self._reader.readtext(image_path)
        except Exception as e:
            log.warning("OCR failed on %s: %s", image_path, e)
            return []

        results: List[OCRResult] = []
        for bbox, text, conf in raw:
            text = text.strip()
            if not text or conf < min_confidence:
                continue
            results.append(OCRResult(
                text=text,
                confidence=float(conf),
                bbox=[[float(c) for c in pt] for pt in bbox],
            ))

        results.sort(key=lambda r: -r.confidence)
        return results[:max_results]

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None


def summarize_ocr(results: List[OCRResult]) -> str:
    """把 OCR 結果整理成一段精簡文字，放進 VLM 提示詞。

    依信心分數保留前幾筆，並去除大小寫正規化後重複的文字
    （同一段文字常被 EasyOCR 用不同框重複偵測到）。
    """
    if not results:
        return "(no text detected)"

    seen: set[str] = set()
    unique: List[OCRResult] = []
    for r in results:
        normalized = r.text.lower().strip()
        if normalized not in seen and len(normalized) > 0:
            seen.add(normalized)
            unique.append(r)

    parts: List[str] = []
    for r in unique[:10]:
        parts.append(f'"{r.text}" ({r.confidence:.0%})')

    return ", ".join(parts)
