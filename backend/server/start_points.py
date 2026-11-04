"""Start points: where a walk may begin on a store's map, and which way the user faces there.

The app's 「我在起點」 puts the user on the map at a start point and takes the way they
face as that start's map heading, so position and heading are right from the first step
(the compass's own north was off by 90° for a whole walk in the 2026-10-17 store test).
Each start carries a photo — "stand here and face this view" — so "facing into the
store" never has to be guessed.

Stored per store beside its extra route links, in ``maps/neo4j_links/<store>.json``::

    "starts": [{"id": "entrance", "name": "入口", "x": 0.0, "y": 0.0,
                "heading_deg": 0.0, "photo": "entrance.jpg"}]

``x``/``y`` are metres in the store map's frame, ``heading_deg`` is clockwise from +y.
Photos live in ``maps/starts/<store folder>/``. The map builder writes the list; the app
reads it with ``GET /starts?place=``.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageOps

from server.map_build import place_folder

log = logging.getLogger(__name__)

PHOTO_MAX_SIDE = 900
_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


def links_file(place: str, maps_dir: Path) -> Path:
    """The store's links file: the existing one naming ``place``, else a new path for it."""
    links_dir = Path(maps_dir) / "neo4j_links"
    for p in sorted(links_dir.glob("*.json")) if links_dir.is_dir() else []:
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("place") == place:
                return p
        except Exception:
            continue
    return links_dir / f"{place_folder(place)}.json"


def photo_dir(place: str, maps_dir: Path) -> Path:
    return Path(maps_dir) / "starts" / place_folder(place)


def load_starts(place: str, maps_dir: Path) -> List[dict]:
    p = links_file(place, maps_dir)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [s for s in data.get("starts") or [] if isinstance(s, dict) and s.get("id")]


def load_north_offset(place: str, maps_dir: Path) -> Optional[float]:
    """The angle (degrees) from geographic north to the map's +Y axis, or None."""
    p = links_file(place, maps_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = data.get("map_north_offset_deg")
    if v is None:
        return None
    try:
        deg = float(v)
    except (TypeError, ValueError):
        log.warning("%s: map_north_offset_deg %r is not a number — ignored", p.name, v)
        return None
    return deg if math.isfinite(deg) else None


def photo_path(place: str, start_id: str, maps_dir: Path) -> Optional[Path]:
    for s in load_starts(place, maps_dir):
        if s["id"] == start_id and s.get("photo"):
            p = photo_dir(place, maps_dir) / s["photo"]
            return p if p.is_file() else None
    return None


def _clean(raw: dict, used: set) -> dict:
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError("起點要有名稱")
    try:
        x, y, h = float(raw["x"]), float(raw["y"]), float(raw["heading_deg"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"起點「{name}」缺少座標或方向")
    if not all(math.isfinite(v) for v in (x, y, h)):
        raise ValueError(f"起點「{name}」的座標或方向不是數字")
    sid = _ID_RE.sub("", str(raw.get("id") or "")) or None
    if sid is None or sid in used:
        n = 1
        while f"s{n}" in used:
            n += 1
        sid = f"s{n}"
    used.add(sid)
    return {"id": sid, "name": name, "x": round(x, 2), "y": round(y, 2), "heading_deg": round(h % 360.0, 1)}


def save_starts(place: str, starts: List[dict], maps_dir: Path, workspace: Optional[Path] = None) -> List[dict]:
    """Replace the store's start list (other keys of its links file are kept).

    Each start may name a photo to copy in: ``{"photo": {"set": "01", "file": "wp_1_front.jpg"}}``
    from the map-build ``workspace``; a start sent back with its existing ``photo`` file
    name keeps it. Photos of starts no longer listed are deleted.
    """
    path = links_file(place, maps_dir)
    data = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    old = {s["id"]: s for s in data.get("starts") or [] if isinstance(s, dict) and s.get("id")}
    pdir = photo_dir(place, maps_dir)
    used: set = set()
    out = []
    for raw in starts:
        s = _clean(raw, used)
        photo = raw.get("photo")
        if isinstance(photo, dict) and photo.get("set") and photo.get("file"):
            if workspace is None:
                raise ValueError("沒有建圖工作資料夾，無法複製起點照片")
            base = Path(workspace).resolve()
            src = (base / str(photo["set"]) / str(photo["file"])).resolve()
            if base not in src.parents or not src.is_file():
                raise ValueError(f"找不到起點「{s['name']}」的照片 {photo['set']}/{photo['file']}")
            pdir.mkdir(parents=True, exist_ok=True)
            im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            im.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE))
            im.save(pdir / f"{s['id']}.jpg", format="JPEG", quality=82)
            s["photo"] = f"{s['id']}.jpg"
        elif isinstance(photo, str) and photo and (pdir / Path(photo).name).is_file():
            s["photo"] = Path(photo).name
        elif s["id"] in old and old[s["id"]].get("photo") and (pdir / old[s["id"]]["photo"]).is_file():
            s["photo"] = old[s["id"]]["photo"]
        out.append(s)
    keep = {s.get("photo") for s in out}
    for sid, s in old.items():
        f = s.get("photo")
        if f and f not in keep and (pdir / f).is_file():
            (pdir / f).unlink()
    data.setdefault("place", place)
    data["starts"] = out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
