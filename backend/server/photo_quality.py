"""Is this photo usable for localization? Blur / darkness / occlusion checks.

Two stages, both cheap:

1. `assess_image` — before any model call. Sharpness is the variance of a 4-neighbour
   Laplacian on a 512 px grayscale copy; brightness is its mean (0–255). Thresholds
   were set against Gaussian-blurred (σ=12 px: ≤ 45) and darkened (25 %: ≤ 34)
   copies. Supermarket photos score 800–2600, office hallway photos 100–400; the
   threshold must clear both environments.
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

MIN_SHARPNESS = 60.0
MIN_BRIGHTNESS = 45.0
MAX_OCCLUDED_FRACTION = 0.30
MIN_USEFUL_OBJECTS = 3
PARTIAL_OCCLUDED_FRACTION = 0.10

# 人/手 only count on their own (人參, 手套 are products).
# A label is an occluder when its head noun is one — English and Chinese noun phrases
# end in the head ("shopping cart", "手推車"), so "hand soap", "customer service" and
# "顧客服務台" are products / signs, not people. English labels are cut at the first
# preposition or participle first ("person with cart", "hands holding phone").
_OCCLUDER_EN_HEADS = {"person", "persons", "people", "man", "men", "woman", "women", "child",
                      "children", "kid", "kids", "customer", "customers", "shopper", "shoppers",
                      "pedestrian", "pedestrians", "hand", "hands", "cart", "carts",
                      "trolley", "trolleys"}
_EN_TAIL = re.compile(r"\s+(?:with|in|on|at|of|near|holding|pushing|walking|standing|blocking)\b.*$")
_OCCLUDER_ZH_ENDINGS = ("顧客", "客人", "店員", "人員", "行人", "人群", "男人", "女人", "小孩",
                        "推車", "購物車")
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
    # the perception prompt reports blockers as "遮擋物:人" / "遮擋物:推車"
    zh = l[:-1] if l.endswith("們") and len(l) > 2 else l      # 顧客們 → 顧客
    if l in _OCCLUDER_EXACT or l.startswith("遮擋") or zh.endswith(_OCCLUDER_ZH_ENDINGS):
        return True
    words = re.findall(r"[a-z]+", _EN_TAIL.sub("", l))
    return bool(words) and words[-1] in _OCCLUDER_EN_HEADS


@dataclass
class Occlusion:
    fraction: float           # share of the frame covered by occluders (union)
    useful_objects: int       # detections that are not occluders
    hint: Optional[str]


def assess_occlusion(detections: Iterable, img_w: int, img_h: int,
                     goals: Iterable[str] = ()) -> Occlusion:
    """detections: objects with .label and a pixel .box/.bbox [x1, y1, x2, y2].
    `goals`: what the user is looking for — a shopping cart they are walking up to
    is the target, not something in the way."""
    goal_labels = {g.strip().lower() for g in goals}
    mask = np.zeros((64, 64), bool)
    useful = 0
    for d in detections:
        box = getattr(d, "box", None) or getattr(d, "bbox", None)
        label = getattr(d, "label", "") or ""
        if label.strip().lower() in goal_labels or not is_occluder(label):
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
