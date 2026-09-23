"""
perception.py — 物件偵測模組 / Object Detection Module

功能說明 (Purpose):
    本模組封裝 GroundingDINO 零樣本物件偵測模型，提供：
    1. 物件偵測 (Object Detection) — 給定文字提示詞，在照片中找出對應物件
    2. 目標裁切驗證 (Goal Crop Verification) — 裁切偵測框，用 VLM 二次確認是否為目標

運作流程 (Workflow):
    照片輸入 → GroundingDINO 偵測 → 回傳 Detection 列表 (label, box, score)
                                          ↓ (若 VLM 判定 ARRIVED)
                                     裁切目標框 → VLM 確認 → 是/否

被誰呼叫 (Called by):
    - server.py: upload_photo() 呼叫 detect() 做物件偵測
    - server.py: upload_photo() 呼叫 verify_goal_detection() 做目標二次確認

依賴 (Dependencies):
    - groundingdino: 零樣本物件偵測模型
    - torch: GPU/CPU 裝置選擇
    - PIL: 影像裁切
    - config.py: 模型路徑、偵測閾值等設定
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

try:
    import torch
except ImportError:
    # torch 未安裝時不阻擋模組載入，但偵測功能會降級為 CPU
    # If torch is not installed, module still loads but detection falls back to CPU
    torch = None  # type: ignore[assignment]
from PIL import Image, ImageOps

from server.config import (
    GROUNDINGDINO_CONFIG, GROUNDINGDINO_WEIGHTS, SAM_WEIGHTS,
    GROUNDINGDINO_BOX_THRESHOLD, GROUNDINGDINO_TEXT_THRESHOLD,
    GROUNDINGDINO_BOX_THRESHOLD_FALLBACK, SAM_TOP_K_BOXES,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 資料結構 / Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """
    單一偵測結果 / A single detection result

    Attributes:
        label: 偵測到的物件名稱，由 GroundingDINO 從提示詞中匹配
               The detected object name, matched from prompt by GroundingDINO
        box:   邊界框座標 [x1, y1, x2, y2]，絕對像素值
               Bounding box coordinates in absolute pixels [left, top, right, bottom]
        score: 偵測信心分數 0~1，越高越確定
               Detection confidence score 0~1, higher means more certain
    """
    label: str
    box: List[float]  # [x1, y1, x2, y2] absolute pixels / 絕對像素座標
    score: float
    position: str = ""  # VLM 位置描述，如 "left top, near"（GroundingDINO 無此欄位）


# ══════════════════════════════════════════════════════════════════════════════
# 目標裁切驗證 / Goal Crop Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_goal_detection(
    image_pil: Image.Image,
    box_xyxy: List[float],
    goal_label: str,
    ask_fn: Callable[[Image.Image, str], str],
) -> bool:
    """
    裁切目標偵測框，用 VLM 二次確認是否真的是目標物件
    Crop the goal detection box and ask VLM to confirm it's really the target

    為什麼需要這個 (Why needed):
        GroundingDINO 在視覺密集場景（如商店貨架）容易產生誤判，
        例如把「牛奶」的偵測框框到「優格」上。裁切該區域單獨問 VLM
        可以大幅降低 false positive。
        GroundingDINO produces false positives on dense shelves (e.g., detecting
        "milk" on a yogurt container). Cropping and asking VLM reduces this.

    參數 (Parameters):
        image_pil:  完整照片的 PIL Image / Full photo as PIL Image
        box_xyxy:   偵測框座標 [x1, y1, x2, y2] / Detection box coordinates
        goal_label: 目標名稱，如 "milk" / Target name like "milk"
        ask_fn:     VLM 問答函式，輸入 (PIL Image, prompt) → 回傳文字
                    VLM function: takes (PIL Image, prompt) → returns text

    回傳 (Returns):
        True  = VLM 確認這是目標 / VLM confirms this is the target
        False = VLM 否認 或 驗證過程出錯（安全起見視為未確認）
                VLM denies OR verification errored (fail-safe: treat as unconfirmed)

    安全設計 (Safety design):
        任何例外都回傳 False，確保驗證故障不會產生假的「已到達」
        Any exception returns False, so verification failure never fakes arrival
    """
    try:
        # 將浮點座標轉為整數像素 / Convert float coordinates to integer pixels
        x1, y1, x2, y2 = [int(c) for c in box_xyxy]

        # 退化框（寬或高為0）直接回傳 False / Degenerate box → False
        if x2 <= x1 or y2 <= y1:
            return False

        # 裁切偵測區域 / Crop the detection region
        cropped = image_pil.crop((x1, y1, x2, y2))

        # 用英文問 VLM 是/否問題 / Ask VLM a yes/no question in English
        prompt = f"Does this image clearly show '{goal_label}'? Answer only: yes or no."
        response = ask_fn(cropped, prompt)

        # 回應中包含 "yes" 即視為確認 / "yes" in response = confirmed
        return "yes" in (response or "").lower()

    except Exception as e:
        # 最佳努力原則：驗證失敗不應影響整個流程
        # Best-effort: never let verification crash a turn
        log.warning("goal verification failed (%s) — treating as unconfirmed", e)
        return False


def ocr_on_detection(
    image_pil: Image.Image,
    box_xyxy: List[float],
    models: List[str],
    prompt: str = "Extract all text visible in this image exactly as written, "
                  "preserving Traditional Chinese characters. Return only the text.",
) -> List[dict]:
    """Crop a detection box and run it through gateway OCR/chat models for comparison.

    Bridges the object-detection pipeline to the CGU gateway: feed a detected object's
    image crop into the OCR models (deepseek-ocr / glm-ocr) to compare their text
    extraction, or into chat models for scene/navigation reasoning. Returns one entry
    per model ({model, kind, text, error}); a dead model (e.g. an OCR 502) is captured
    per-entry rather than raised, so one outage never sinks the rest.
    """
    from server.llm_gateway import run_on_image  # lazy: keeps perception import light

    x1, y1, x2, y2 = [int(c) for c in box_xyxy]
    if x2 <= x1 or y2 <= y1:
        return [{"model": m, "kind": None, "text": "", "error": "invalid box"} for m in models]
    crop = image_pil.crop((x1, y1, x2, y2))
    return run_on_image(models, prompt, image_pil=crop, temperature=0.0)


class Perception:
    """
    GroundingDINO 物件偵測引擎，伺服器啟動時載入一次
    GroundingDINO object detection engine, loaded once at server startup

    運作方式 (How it works):
        1. __init__: 根據 GPU VRAM 決定用 cuda 或 cpu
        2. load():   載入 GroundingDINO 模型權重
        3. detect(): 給定照片和提示詞列表，回傳偵測結果

    自適應閾值策略 (Adaptive threshold strategy):
        先用嚴格閾值偵測 (BOX_THRESHOLD) 以減少誤判
        → 如果什麼都沒偵測到 → 用寬鬆閾值重試 (BOX_THRESHOLD_FALLBACK)
        這確保在物件明顯時精確，在物件不明顯時仍有最佳嘗試
        First try strict threshold to reduce false positives
        → If nothing found → retry with looser fallback threshold
    """

    def __init__(self) -> None:
        """
        初始化：選擇運算裝置 / Initialize: select computation device

        GPU 選擇邏輯 (GPU selection logic):
            - 有 CUDA 且 VRAM >= 3GB → 使用 GPU（GroundingDINO + BERT 需要 ~1.8GB）
            - 否則 → 使用 CPU（較慢但一定能跑）
        """
        if torch and torch.cuda.is_available():
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            # GroundingDINO + BERT 需要約 1.8GB VRAM
            # GroundingDINO + BERT needs ~1.8GB VRAM
            self.device = "cuda" if vram_mb >= 3072 else "cpu"
            if self.device == "cpu":
                log.info("GPU has %dMB VRAM (< 3GB), GroundingDINO will use CPU", vram_mb)
        else:
            self.device = "cpu"

        self._gd_model = None       # GroundingDINO 模型實例 / model instance
        self._sam_predictor = None   # SAM 預測器（目前未使用）/ SAM predictor (unused)

    def load(self) -> None:
        """
        載入 GroundingDINO 模型權重 / Load GroundingDINO model weights

        只在第一次需要時呼叫（延遲載入），避免伺服器啟動時佔用太多時間
        Called only when first needed (lazy loading) to avoid slow server startup
        """
        from groundingdino.util.inference import load_model

        log.info("loading GroundingDINO from %s (device=%s)", GROUNDINGDINO_WEIGHTS, self.device)
        self._gd_model = load_model(
            str(GROUNDINGDINO_CONFIG),
            str(GROUNDINGDINO_WEIGHTS),
            device=self.device,
        )
        self._sam_predictor = None
        log.info("perception loaded on %s (GroundingDINO only)", self.device)

    def detect(self, image_path: str, prompt_classes: List[str]) -> List[Detection]:
        """
        偵測照片中的指定物件 / Detect specified objects in a photo

        參數 (Parameters):
            image_path:     照片檔案路徑 / Path to photo file
            prompt_classes: 要偵測的物件名稱列表，如 ["door", "fire extinguisher", "飲水機"]
                           List of object names to detect

        回傳 (Returns):
            Detection 列表，按信心分數降序排列，最多 SAM_TOP_K_BOXES 個
            List of Detection, sorted by score descending, at most SAM_TOP_K_BOXES

        偵測流程 (Detection flow):
            1. 將物件名稱用 " . " 連接成 GroundingDINO 格式的提示詞
               Join class names with " . " into GroundingDINO prompt format
               例: "door . fire extinguisher . 飲水機 ."

            2. 用嚴格閾值偵測 → 找到結果就回傳
               Detect with strict threshold → return if found

            3. 沒找到 → 用寬鬆閾值重試
               Nothing found → retry with fallback threshold

            4. 按分數排序，取前 K 個
               Sort by score, take top K
        """
        if not prompt_classes:
            return []

        import numpy as np
        import groundingdino.datasets.transforms as T

        # 組合提示詞：GroundingDINO 格式為 "class1 . class2 . class3 ."
        # Build prompt: GroundingDINO format is "class1 . class2 . class3 ."
        text_prompt = " . ".join(prompt_classes) + " ."

        # EXIF 轉正後再送進 GroundingDINO（與 load_image 相同的 transform）
        pil_img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        image_source = np.array(pil_img)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(pil_img, None)

        # 第一次：嚴格閾值（減少誤判）/ First pass: strict threshold (fewer false positives)
        results = self._predict(image, image_source, text_prompt, GROUNDINGDINO_BOX_THRESHOLD)

        # 第二次：如果沒結果，用寬鬆閾值重試 / Second pass: fallback if nothing found
        if not results:
            results = self._predict(image, image_source, text_prompt, GROUNDINGDINO_BOX_THRESHOLD_FALLBACK)

        # 過濾空 label（GroundingDINO 有時偵測到框但無法匹配類別名稱）
        results = [d for d in results if d.label.strip()]
        # 按信心分數降序排列，只取前 K 個 / Sort by score descending, take top K
        results.sort(key=lambda d: -d.score)
        return results[:SAM_TOP_K_BOXES]

    def _predict(self, image, image_source, text_prompt: str, box_threshold: float) -> List[Detection]:
        """
        內部方法：呼叫 GroundingDINO 模型進行一次偵測
        Internal: call GroundingDINO model for one detection pass

        參數 (Parameters):
            image:          預處理後的 tensor 影像 / Preprocessed tensor image
            image_source:   原始影像 numpy 陣列（用於取得寬高）/ Raw image array (for width/height)
            text_prompt:    GroundingDINO 格式的提示詞 / GroundingDINO format prompt
            box_threshold:  邊界框信心閾值，低於此值的偵測會被丟棄
                           Box confidence threshold, detections below this are discarded

        座標轉換 (Coordinate conversion):
            GroundingDINO 回傳的是正規化的中心座標格式 (cx, cy, w, h)，值域 0~1
            需要轉換成絕對像素的 (x1, y1, x2, y2) 格式
            GroundingDINO returns normalized center format (cx, cy, w, h) in 0~1
            Must convert to absolute pixel (x1, y1, x2, y2) format
        """
        from groundingdino.util.inference import predict

        # 呼叫 GroundingDINO 推論 / Call GroundingDINO inference
        boxes, logits, phrases = predict(
            model=self._gd_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=GROUNDINGDINO_TEXT_THRESHOLD,
            device=self.device,
        )

        # 取得原始影像尺寸 / Get original image dimensions
        h, w = image_source.shape[:2]

        results: List[Detection] = []
        for box_cxcywh, score, phrase in zip(boxes, logits, phrases):
            # 正規化中心座標 → 絕對像素左上右下座標
            # Normalized center coords → absolute pixel top-left/bottom-right
            cx, cy, bw, bh = box_cxcywh.tolist()
            x1 = (cx - bw / 2) * w   # 左邊界 / left edge
            y1 = (cy - bh / 2) * h   # 上邊界 / top edge
            x2 = (cx + bw / 2) * w   # 右邊界 / right edge
            y2 = (cy + bh / 2) * h   # 下邊界 / bottom edge
            results.append(Detection(label=phrase, box=[x1, y1, x2, y2], score=float(score)))

        return results
