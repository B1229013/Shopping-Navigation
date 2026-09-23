"""
annotator.py — 照片標註模組 / Photo Annotation Module

功能說明 (Purpose):
    在原始照片上繪製偵測框、OCR 文字區域和導航指引橫幅，
    產出一張視覺化的標註照片，方便使用者和開發者理解偵測結果。
    Draws detection boxes, OCR text regions, and guidance banner onto the photo,
    producing an annotated image for user/developer understanding.

標註色彩系統 (Color coding system):
    - 紅色 (Red):   目標物件 [TARGET] — 與導航目標直接匹配
                    Goal object — directly matches navigation target
    - 橘色 (Orange): 相似物件 — 名稱與目標有一定相似度（可能是同類別）
                    Similar object — name somewhat similar to goal
    - 綠色 (Green):  一般環境物件 — 地標、背景物
                    Generic environment object — landmarks, background
    - 青色 (Cyan):   OCR 文字區域 — 辨識到的文字
                    OCR text region — recognized text

被誰呼叫 (Called by):
    - server.py: upload_photo() → 每張照片處理後都會呼叫 annotate() 產出標註圖

依賴 (Dependencies):
    - PIL (Pillow): 影像繪製
    - perception.py: Detection 資料結構
    - difflib: 模糊字串比對（判斷「相似物件」）
"""
from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

from server.perception import Detection

if TYPE_CHECKING:
    from server.ocr import OCRResult

# 底部橫幅高度（像素）/ Bottom banner height in pixels
_BANNER_HEIGHT = 80

# ══════════════════════════════════════════════════════════════════════════════
# 顏色定義 / Color Definitions
# ══════════════════════════════════════════════════════════════════════════════

# 偵測框顏色（RGB）/ Detection box colors (RGB)
_COLOR_GOAL = (220, 50, 50)       # 紅色 — 目標物件 / Red — goal object
_COLOR_SIMILAR = (255, 140, 0)    # 橘色 — 相似物件 / Orange — similar to goal
_COLOR_GENERIC = (50, 200, 80)    # 綠色 — 一般物件 / Green — generic object

# 標籤背景色（較深，讓白字清晰）/ Label background (darker, for white text readability)
_BG_GOAL = (180, 30, 30)         # 深紅 / Dark red
_BG_SIMILAR = (200, 100, 0)      # 深橘 / Dark orange
_BG_GENERIC = (0, 160, 0)        # 深綠 / Dark green


# ══════════════════════════════════════════════════════════════════════════════
# 字型載入 / Font Loading
# ══════════════════════════════════════════════════════════════════════════════

def _get_font(size: int):
    """
    嘗試載入中文字型，失敗則用預設字型
    Try loading Chinese font, fallback to default if unavailable

    嘗試順序 (Try order):
        1. 微軟正黑體 (msjh.ttc) — Windows 繁體中文
        2. 微軟雅黑 (msyh.ttc) — Windows 簡體中文
        3. Arial — Windows 英文
        4. DejaVuSans — Linux
        5. PIL 預設字型（最後手段）/ PIL default font (last resort)
    """
    for path in [
        "C:/Windows/Fonts/msjh.ttc",                           # 微軟正黑體 / Microsoft JhengHei
        "C:/Windows/Fonts/msyh.ttc",                           # 微軟雅黑 / Microsoft YaHei
        "C:/Windows/Fonts/arial.ttf",                          # Arial
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # Linux DejaVu
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════════════════════
# 繪製輔助函式 / Drawing Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def _draw_label(draw: ImageDraw.Draw, xy: tuple, text: str,
                fill: tuple, bg: tuple, font) -> None:
    """
    在指定位置繪製帶背景色的文字標籤
    Draw a text label with colored background at specified position

    流程 (Process):
        1. 先用 textbbox 計算文字邊界框大小
           Calculate text bounding box size
        2. 畫背景矩形（稍微擴大 2px padding）
           Draw background rectangle (with 2px padding)
        3. 在背景上畫白色文字
           Draw white text on background
    """
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    # 背景矩形：左右各擴 2px，上下各擴 1px / Background rect: ±2px horizontal, ±1px vertical
    draw.rectangle((bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1), fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def _is_goal(label: str, goal_objects: List[str]) -> bool:
    """
    判斷偵測標籤是否為目標物件 / Check if detection label is a goal object

    比對方式：雙向子字串包含（不區分大小寫）
    Matching: bidirectional substring containment (case-insensitive)

    例 (Examples):
        label="milk bottle", goal=["milk"] → True（"milk" in "milk bottle"）
        label="door", goal=["milk"] → False
    """
    label_l = label.lower()
    for g in goal_objects:
        if g.lower() in label_l or label_l in g.lower():
            return True
    return False


def _is_similar(label: str, goal_objects: List[str]) -> bool:
    """
    判斷偵測標籤是否與目標「相似」/ Check if detection label is "similar" to goal

    使用 difflib.SequenceMatcher 計算字串相似度
    Uses difflib.SequenceMatcher for string similarity

    閾值：相似度 > 0.5（50%）即判定為相似
    Threshold: similarity > 0.5 (50%) = similar

    例 (Examples):
        label="refrigerator", goal=["fridge"] → 相似度 0.56 → True
        label="chair", goal=["milk"] → 相似度 0.11 → False

    用途 (Purpose):
        即使不是精確匹配，也用橘色框標出，讓使用者注意可能相關的物件
        Even without exact match, highlight in orange so user notices potentially related objects
    """
    label_l = label.lower()
    for g in goal_objects:
        ratio = difflib.SequenceMatcher(None, label_l, g.lower()).ratio()
        if ratio > 0.5:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 主要標註函式 / Main Annotation Function
# ══════════════════════════════════════════════════════════════════════════════

def annotate(
    src_path: str,
    dst_path: str,
    detections: List[Detection],
    banner_text: str,
    ocr_results: Optional[List["OCRResult"]] = None,
    goal_objects: Optional[List[str]] = None,
    region_mode: bool = False,
) -> None:
    """
    在照片上繪製所有標註，儲存為新檔案
    Draw all annotations on photo and save as new file

    參數 (Parameters):
        src_path:     原始照片路徑 / Source photo path
        dst_path:     標註後照片的儲存路徑 / Annotated photo save path
        detections:   GroundingDINO 偵測結果列表 / Detection results list
        banner_text:  底部橫幅文字（通常是 VLM 的判定和指引）
                     Bottom banner text (usually VLM's decision + guidance)
        ocr_results:  OCR 辨識結果列表（可選）/ OCR results (optional)
        goal_objects: 目標物件詞列表（用於色彩分類）/ Goal words (for color coding)

    輸出圖片結構 (Output image structure):
        ┌─────────────────────────────────┐
        │                                 │
        │  [TARGET] milk 91%   (紅框)     │ ← 原始照片 + 偵測框
        │                                 │   Original photo + detection boxes
        │     door 85%         (綠框)     │
        │                                 │
        │  "乳製品" 95%        (青框)     │ ← OCR 文字區域
        │                                 │   OCR text regions
        ├─────────────────────────────────┤
        │ ARRIVED: 找到了！    ■■■■ 圖例  │ ← 底部橫幅
        │                                 │   Bottom banner
        └─────────────────────────────────┘

    自適應縮放 (Adaptive scaling):
        字型大小、框線粗細根據影像尺寸自動調整
        Font size and line width auto-scale based on image dimensions
    """
    # 開啟原始照片 / Open source photo
    img = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")
    w, h = img.size

    # 根據影像大小計算縮放比例 / Calculate scale based on image size
    scale = max(w, h) / 1000
    font_size = max(14, int(16 * scale))     # 偵測標籤字型 / Detection label font
    box_width = max(2, int(3 * scale))       # 框線粗細 / Box line width

    font = _get_font(font_size)
    banner_font = _get_font(max(14, int(14 * scale)))  # 橫幅字型 / Banner font

    # 建立畫布：原圖高度 + 底部橫幅 / Create canvas: original height + banner
    canvas = Image.new("RGB", (w, h + _BANNER_HEIGHT), color=(20, 20, 20))
    canvas.paste(img, (0, 0))  # 貼上原圖 / Paste original
    draw = ImageDraw.Draw(canvas)

    goal_objs = goal_objects or []

    # ── 繪製偵測框 / Draw detection boxes ──
    if region_mode:
        # VLM 模式：半透明區域覆蓋（bbox 由位置描述產生，是近似區域）
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

    label_offset: dict[tuple, int] = {}
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.box]

        if _is_goal(d.label, goal_objs):
            color, bg = _COLOR_GOAL, _BG_GOAL
            prefix = "[TARGET] "
        elif _is_similar(d.label, goal_objs):
            color, bg = _COLOR_SIMILAR, _BG_SIMILAR
            prefix = ""
        else:
            color, bg = _COLOR_GENERIC, _BG_GENERIC
            prefix = ""

        if region_mode:
            # VLM 區域模式：半透明填充 + 邊框
            alpha = 35 if prefix else 25
            overlay_draw.rectangle((x1, y1, x2, y2),
                                   fill=color + (alpha,))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=box_width)
            region_key = (x1 // (w // 3 + 1), y1 // (h // 3 + 1))
            n = label_offset.get(region_key, 0)
            label_offset[region_key] = n + 1
            label_y = y1 + 4 + n * (font_size + 6)
        else:
            # GroundingDINO 模式：精確框，標籤在框上方
            draw.rectangle((x1, y1, x2, y2), outline=color, width=box_width)
            label_y = max(0, y1 - font_size - 4)

        pos_tag = f" [{d.position}]" if region_mode and getattr(d, "position", "") else ""
        label = f"{prefix}{d.label} {d.score:.0%}{pos_tag}"
        _draw_label(draw, (x1 + 4, label_y), label,
                    fill=(255, 255, 255), bg=bg, font=font)

    if region_mode:
        img_rgba = canvas.convert("RGBA")
        full_overlay = Image.new("RGBA", (w, h + _BANNER_HEIGHT), (0, 0, 0, 0))
        full_overlay.paste(overlay, (0, 0))
        canvas = Image.alpha_composite(img_rgba, full_overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)

    # ── 繪製 OCR 文字區域 / Draw OCR text regions ──
    if ocr_results:
        for r in ocr_results:
            if len(r.bbox) >= 4:
                # 畫四邊形（OCR 區域可能有傾斜）/ Draw quadrilateral (OCR region may be tilted)
                pts = [(int(p[0]), int(p[1])) for p in r.bbox]
                draw.polygon(pts, outline=(0, 255, 255), width=box_width)  # 青色邊框 / Cyan outline

                # 畫文字標籤 / Draw text label
                label = f'"{r.text}" {r.confidence:.0%}'
                _draw_label(draw, (pts[0][0], max(0, pts[0][1] - font_size - 4)),
                            label, fill=(255, 255, 255), bg=(0, 140, 140), font=font)

    # ── 繪製底部橫幅 / Draw bottom banner ──
    legend_y = h + 8
    legend_font = _get_font(max(12, int(12 * scale)))

    # 左側：VLM 判定 + 指引文字（最多 200 字）/ Left: VLM decision + guidance (max 200 chars)
    draw.text((10, legend_y), banner_text[:200], fill=(255, 255, 255), font=banner_font)

    # 右側：色彩圖例 / Right: color legend
    lx = w - int(280 * scale)
    for color, text in [
        (_COLOR_GOAL, "■ Target"),       # 紅 = 目標 / Red = target
        (_COLOR_SIMILAR, "■ Similar"),   # 橘 = 相似 / Orange = similar
        (_COLOR_GENERIC, "■ Generic"),   # 綠 = 一般 / Green = generic
        ((0, 255, 255), "■ OCR"),        # 青 = 文字 / Cyan = OCR text
    ]:
        draw.text((lx, legend_y), text, fill=color, font=legend_font)
        lx += int(70 * scale)

    # 儲存標註照片 / Save annotated photo
    canvas.save(dst_path, "JPEG", quality=85)
