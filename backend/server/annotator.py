"""Draw detection boxes, OCR text regions, and a guidance banner onto a copy of the photo."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from PIL import Image, ImageDraw, ImageFont

from server.perception import Detection

if TYPE_CHECKING:
    from server.ocr import OCRResult

_BANNER_HEIGHT = 60


def annotate(
    src_path: str,
    dst_path: str,
    detections: List[Detection],
    banner_text: str,
    ocr_results: Optional[List["OCRResult"]] = None,
) -> None:
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    canvas = Image.new("RGB", (w, h + _BANNER_HEIGHT), color=(20, 20, 20))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Draw GroundingDINO detection boxes (green)
    for d in detections:
        x1, y1, x2, y2 = d.box
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{d.label} {d.score:.2f}", fill=(0, 255, 0))

    # Draw OCR text regions (cyan)
    if ocr_results:
        for r in ocr_results:
            if len(r.bbox) >= 4:
                pts = [(int(p[0]), int(p[1])) for p in r.bbox]
                draw.polygon(pts, outline=(0, 255, 255))
                draw.text(
                    (pts[0][0], max(0, pts[0][1] - 14)),
                    f'OCR: "{r.text}" {r.confidence:.0%}',
                    fill=(0, 255, 255),
                )

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, h + 8), banner_text[:200], fill=(255, 255, 255), font=font)

    canvas.save(dst_path, "JPEG", quality=85)
