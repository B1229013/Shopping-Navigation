"""Is this photo usable for localization? Blur / darkness / occlusion checks.

Two stages, both cheap:

1. `assess_image` — before any model call. Sharpness is the variance of a 4-neighbour
   Laplacian on a 512 px grayscale copy; brightness is its mean (0–255). Thresholds
   were set on the 23 photos of 測試路線1 (sharpness 1662–2597, brightness 106–138)
   against Gaussian-blurred (σ=12 px: ≤ 45) and darkened (25 %: ≤ 34) copies, with a
   wide margin so normal photos are never rejected.
2. `assess_occlusion` — after detection. People / carts / hands covering ≥ 30 % of the
   frame, or covering ≥ 10 % while fewer than 3 other objects remain to match.

Either returns a Traditional Chinese retake hint, or None when the photo is fine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageOps

MIN_SHARPNESS = 150.0
MIN_BRIGHTNESS = 45.0
MAX_OCCLUDED_FRACTION = 0.30
MIN_USEFUL_OBJECTS = 3
PARTIAL_OCCLUDED_FRACTION = 0.10

# English as whole words ("person" must not match "personal care"); Chinese as substrings,
# except 人/手 which only count on their own (人參, 手套 are products).
_OCCLUDER_EN = re.compile(r"\b(person|people|man|woman|child|customer|shopper|hand|hands|"
                          r"shopping cart|trolley)\b")
_OCCLUDER_ZH = ("遮擋", "顧客", "店員", "推車", "購物車", "行人")
_OCCLUDER_EXACT = {"人", "手"}

HINT_BLUR = "照片太模糊，請拿穩手機、停一下再拍"
HINT_DARK = "照片太暗，請開啟閃光燈（右上角燈泡）再拍"
HINT_OCCLUDED = "畫面被人或推車擋住了，請避開或換個角度再拍"
HINT_FEW_OBJECTS = "畫面裡可辨識的東西太少，請對準走道、拍到招牌或貨架再試一次"


@dataclass
class ImageQuality:
    sharpness: float
    brightness: float
    hint: Optional[str]


def assess_image(path: str) -> ImageQuality:
    with Image.open(path) as im:
        g = ImageOps.exif_transpose(im).convert("L")
        g.thumbnail((512, 512))
        a = np.asarray(g, dtype=np.float32)
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    sharpness, brightness = float(lap.var()), float(a.mean())
    hint = None
    if brightness < MIN_BRIGHTNESS:
        hint = HINT_DARK          # a dark photo is also low-contrast; report the cause
    elif sharpness < MIN_SHARPNESS:
        hint = HINT_BLUR
    return ImageQuality(sharpness=sharpness, brightness=brightness, hint=hint)


def is_occluder(label: str) -> bool:
    l = (label or "").strip().lower()
    if not l:
        return False
    if l in _OCCLUDER_EXACT or any(w in l for w in _OCCLUDER_ZH):
        return True
    return bool(_OCCLUDER_EN.search(l))


@dataclass
class Occlusion:
    fraction: float           # share of the frame covered by occluders (union)
    useful_objects: int       # detections that are not occluders
    hint: Optional[str]


def assess_occlusion(detections: Iterable, img_w: int, img_h: int) -> Occlusion:
    """detections: objects with .label and a pixel .box/.bbox [x1, y1, x2, y2]."""
    mask = np.zeros((64, 64), bool)
    useful = 0
    for d in detections:
        box = getattr(d, "box", None) or getattr(d, "bbox", None)
        if not is_occluder(getattr(d, "label", "")):
            useful += 1
            continue
        if not box or len(box) != 4 or isinstance(box[0], (list, tuple)) or img_w <= 0 or img_h <= 0:
            continue
        x1, y1, x2, y2 = (int(np.clip(v / s * 64, 0, 64)) for v, s in
                          zip(box, (img_w, img_h, img_w, img_h)))
        mask[y1:y2, x1:x2] = True
    fraction = float(mask.mean())
    hint = None
    if fraction >= MAX_OCCLUDED_FRACTION:
        hint = HINT_OCCLUDED
    elif useful < MIN_USEFUL_OBJECTS and fraction >= PARTIAL_OCCLUDED_FRACTION:
        # few objects on their own is normal for a narrow detector prompt; it only
        # means "blocked" when an occluder is also taking up part of the view
        hint = HINT_FEW_OBJECTS
    return Occlusion(fraction=fraction, useful_objects=useful, hint=hint)
