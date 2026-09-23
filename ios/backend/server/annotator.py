"""Draw detection boxes, OCR text regions, and a guidance banner onto a copy of the photo."""
from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, List, Optional

from PIL import Image, ImageDraw, ImageFont

from server.perception import Detection

if TYPE_CHECKING:
    from server.ocr import OCRResult

_BANNER_HEIGHT = 80

# Colors for detection categories
_COLOR_GOAL = (220, 50, 50)       # red — goal object
_COLOR_SIMILAR = (255, 140, 0)    # orange — similar to goal
_COLOR_GENERIC = (50, 200, 80)    # green — generic environment
_BG_GOAL = (180, 30, 30)
_BG_SIMILAR = (200, 100, 0)
_BG_GENERIC = (0, 160, 0)


def _get_font(size: int):
    for path in [
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_label(draw: ImageDraw.Draw, xy: tuple, text: str,
                fill: tuple, bg: tuple, font) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1), fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def _is_goal(label: str, goal_objects: List[str]) -> bool:
    label_l = label.lower()
    for g in goal_objects:
        if g.lower() in label_l or label_l in g.lower():
            return True
    return False


def _is_similar(label: str, goal_objects: List[str]) -> bool:
    label_l = label.lower()
    for g in goal_objects:
        ratio = difflib.SequenceMatcher(None, label_l, g.lower()).ratio()
        if ratio > 0.5:
            return True
    return False


def annotate(
    src_path: str,
    dst_path: str,
    detections: List[Detection],
    banner_text: str,
    ocr_results: Optional[List["OCRResult"]] = None,
    goal_objects: Optional[List[str]] = None,
    region_mode: bool = False,
) -> None:
    """``region_mode``: VLM-only perception (PERCEPTION_ENABLED=0) gives rough
    estimated regions, not GroundingDINO's precise boxes — labels get a "~" prefix
    so it's visually obvious these boxes are approximate, not exact detections."""
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    scale = max(w, h) / 1000
    font_size = max(14, int(16 * scale))
    box_width = max(2, int(3 * scale))

    font = _get_font(font_size)
    banner_font = _get_font(max(14, int(14 * scale)))

    canvas = Image.new("RGB", (w, h + _BANNER_HEIGHT), color=(20, 20, 20))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    goal_objs = goal_objects or []

    for d in detections:
        x1, y1, x2, y2 = d.box
        if _is_goal(d.label, goal_objs):
            color, bg = _COLOR_GOAL, _BG_GOAL
            prefix = "[TARGET] "
        elif _is_similar(d.label, goal_objs):
            color, bg = _COLOR_SIMILAR, _BG_SIMILAR
            prefix = ""
        else:
            color, bg = _COLOR_GENERIC, _BG_GENERIC
            prefix = ""

        draw.rectangle((x1, y1, x2, y2), outline=color, width=box_width)
        approx = "~" if region_mode else ""
        label = f"{approx}{prefix}{d.label} {d.score:.0%}"
        _draw_label(draw, (x1 + 4, max(0, y1 - font_size - 4)), label,
                    fill=(255, 255, 255), bg=bg, font=font)

    if ocr_results:
        for r in ocr_results:
            if len(r.bbox) >= 4:
                pts = [(int(p[0]), int(p[1])) for p in r.bbox]
                draw.polygon(pts, outline=(0, 255, 255), width=box_width)
                label = f'"{r.text}" {r.confidence:.0%}'
                _draw_label(draw, (pts[0][0], max(0, pts[0][1] - font_size - 4)),
                            label, fill=(255, 255, 255), bg=(0, 140, 140), font=font)

    # Legend
    legend_y = h + 8
    legend_font = _get_font(max(12, int(12 * scale)))
    draw.text((10, legend_y), banner_text[:200], fill=(255, 255, 255), font=banner_font)
    lx = w - int(280 * scale)
    for color, text in [(_COLOR_GOAL, "■ Target"), (_COLOR_SIMILAR, "■ Similar"), (_COLOR_GENERIC, "■ Generic"), ((0, 255, 255), "■ OCR")]:
        draw.text((lx, legend_y), text, fill=color, font=legend_font)
        lx += int(70 * scale)

    canvas.save(dst_path, "JPEG", quality=85)
