"""Aisle-number signs: which map rows a photo can be in, from the hanging aisle signs it shows.

In this store the shelf aisles run across the map (along x), so each aisle number
belongs to one row of nodes (one y value). The reference photos already recorded the
signs: the right-hand photo at every node on the main walkway (x = -6.5) shows that
row's aisle sign — 2 at y 4.6, 4 at 8.3, 6 at 11.1, 8 at 14.4, 10 at 19, 12 at 22,
14 at 25.5, 16 at 27.5 — and the far wall (x = -21.3) repeats them. Odd aisles hang
beside the even one over the same entrance ("5 | 6" in the route-1 photos), so 2k-1
shares 2k's row.

The signs hang in numbered pairs over the central walkway (x = -11.3), so a photo
taken along that walkway shows a whole sequence of signs (13 | 14, then 11 | 12, 9 | 10
smaller behind). Two cues come out of a photo:

- the nearest (largest) sign pair → the row the user is at or just passed;
- three or more different aisle numbers → the user is on the central walkway (the
  reference photos that show ≥ 3 numbers are all taken from that column).

On the 16 route-1 walk photos taken along the walkway, DINOv2 alone put the right spot
first 6% of the time (9.2 m mean error); DINOv2 + these two cues 62.5% (3.4 m).

The row table is derived from the reference map; a hand-checked table can override it:
backend/maps/aisles.json = {"<place>": {"<aisle>": [nid, ...]}}. Both cues assume this
store's layout, so the server only uses them for the places in config AISLE_SIGN_PLACES.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

AISLES_FILE = Path(__file__).resolve().parents[1] / "maps" / "aisles.json"
ROW_TOLERANCE_M = 1.0
MAX_AISLE = 30

_SIGN = re.compile(r"走道|aisle", re.I)
_NUMBER = re.compile(r"^\s*(?:走道|aisle)?\s*(\d{1,2})\s*(?:號)?\s*$", re.I)


def _numbers(label: str, text: str) -> List[int]:
    """Aisle numbers named by one detected object (only aisle signs count)."""
    if not _SIGN.search(label or ""):
        return []
    out = []
    for t in (text or "", re.sub(r"[^\d ]", " ", label or "")):
        m = _NUMBER.match(t) if t.strip() else None
        if m and 1 <= int(m.group(1)) <= MAX_AISLE:
            out.append(int(m.group(1)))
    return sorted(set(out))


def _area(box) -> float:
    if not box or len(box) != 4:
        return 0.0
    return max(0.0, (box[2] - box[0]) * (box[3] - box[1]))


def numbers_in_items(items: Iterable[dict]) -> Dict[int, float]:
    """Query detections ({"label", "box" (0-1), "text"?}) → {aisle number: sign area}.
    A bigger sign is a nearer sign, so the area is the weight."""
    found: Dict[int, float] = {}
    for it in items:
        for n in _numbers(it.get("label", ""), it.get("text", "")):
            found[n] = max(found.get(n, 0.0), _area(it.get("box")) or 1e-4)
    return found


def build_table(ref_map, exclude_photo: Optional[str] = None) -> Dict[int, List[int]]:
    """{aisle number: [nid, ...]} from the signs in the reference photos.

    Only photos showing at most two aisle numbers vote (a photo looking down the
    cross walkway sees a whole row of signs and says nothing about its own row).
    """
    votes: Dict[int, List[float]] = defaultdict(list)
    for node in ref_map.photos.values():
        for dp in node.directional_photos:
            if dp.photo_file == exclude_photo:
                continue
            nums = set()
            for o in dp.objects:
                nums.update(_numbers(o.label or "", o.ocr_text or ""))
            if 1 <= len(nums) <= 2:
                for n in nums:
                    votes[n].append(node.pdr_y)
    # pair 2k-1 with 2k: same entrance, same row
    rows: Dict[int, float] = {}
    for n in range(2, MAX_AISLE + 1, 2):
        ys = votes.get(n, []) + votes.get(n - 1, [])
        if ys:
            rows[n] = rows[n - 1] = median(ys)
    table = {}
    for n, y in rows.items():
        nids = [nid for nid, node in ref_map.photos.items() if abs(node.pdr_y - y) <= ROW_TOLERANCE_M]
        if nids:
            table[n] = sorted(nids)
    return table


_cache: Dict[str, Dict[int, List[int]]] = {}


def table_for(place: str, ref_map, derive: bool = True) -> Optional[Dict[int, List[int]]]:
    """Hand-checked table from maps/aisles.json if present, else derived from the map."""
    if place in _cache:
        return _cache[place]
    table = None
    if AISLES_FILE.exists():
        try:
            data = json.loads(AISLES_FILE.read_text(encoding="utf-8")).get(place)
            if data:
                table = {int(k): [int(x) for x in v] for k, v in data.items()}
                log.info("aisle table for %s from %s (%d aisles)", place, AISLES_FILE.name, len(table))
        except (ValueError, TypeError) as e:
            log.warning("ignoring malformed %s: %s", AISLES_FILE, e)
    if table is None:
        if not derive:
            return None
        table = build_table(ref_map)
        log.info("aisle table for %s derived from reference photos: %s", place,
                 {k: len(v) for k, v in sorted(table.items())})
    _cache[place] = table
    return table


def evidence(numbers: Dict[int, float], table: Dict[int, List[int]]) -> Dict[int, float]:
    """{nid: 0..1} — share of the seen signs' (area-weighted) votes that fall on each node."""
    known = {n: a for n, a in numbers.items() if n in table}
    total = sum(known.values())
    if not total:
        return {}
    out: Dict[int, float] = defaultdict(float)
    for n, a in known.items():
        for nid in table[n]:
            out[nid] += a / total
    return dict(out)


ROW_NEAR_M = 4.0          # rows either side of the nearest sign pair get half credit
MIN_SIGN_CONFIDENCE = 0.9


def numbers_from_ocr(results, img_w: int, img_h: int, min_conf: float = MIN_SIGN_CONFIDENCE,
                     max_center_y: float = 0.6) -> Dict[int, float]:
    """Hanging aisle numbers read by OCR (EasyOCR results or VLM ocr_texts, pixel corner
    boxes): a bare 1–2 digit number in the upper part of the frame → {number: area}."""
    found: Dict[int, float] = {}
    if not img_w or not img_h:
        return found
    for r in results or []:
        text = (getattr(r, "text", "") or "").strip()
        conf = getattr(r, "confidence", None)
        if conf is None:
            conf = getattr(r, "score", 0.0)
        if not re.fullmatch(r"\d{1,2}", text) or conf < min_conf:
            continue
        n = int(text)
        bbox = getattr(r, "bbox", None) or []
        if not (1 <= n <= MAX_AISLE) or len(bbox) < 2 or not isinstance(bbox[0], (list, tuple)):
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        if (min(ys) + max(ys)) / 2 / img_h > max_center_y:
            continue
        area = (max(xs) - min(xs)) * (max(ys) - min(ys)) / (img_w * img_h)
        found[n] = max(found.get(n, 0.0), area)
    return found


_walkway_cache: Dict[str, set] = {}


def walkway_nodes(ref_map, place: str = "") -> set:
    """Nodes on the signed walkway: the column of the nodes whose reference photos show
    three or more aisle numbers at once."""
    if place and place in _walkway_cache:
        return _walkway_cache[place]
    xs = []
    for node in ref_map.photos.values():
        for dp in node.directional_photos:
            nums = set()
            for o in dp.objects:
                nums.update(_numbers(o.label or "", o.ocr_text or ""))
            if len(nums) >= 3:
                xs.append(node.pdr_x)
                break
    out = set()
    if xs:
        cx = median(xs)
        out = {nid for nid, n in ref_map.photos.items() if abs(n.pdr_x - cx) <= ROW_TOLERANCE_M}
    if place:
        _walkway_cache[place] = out
    return out


def sign_evidence(numbers: Dict[int, float], ref_map, place: str = "",
                  table: Optional[Dict[int, List[int]]] = None,
                  walkway: Optional[set] = None) -> Dict[int, float]:
    """{nid: evidence} from the aisle numbers seen in one photo (see the module docstring)."""
    if not numbers:
        return {}
    table = table if table is not None else table_for(place, ref_map)
    walkway = walkway if walkway is not None else walkway_nodes(ref_map, place)
    ev: Dict[int, float] = defaultdict(float)
    pairs = range(1, MAX_AISLE // 2 + 1)
    k = max(pairs, key=lambda k: numbers.get(2 * k - 1, 0.0) + numbers.get(2 * k, 0.0))
    row_ys = [ref_map.photos[n].pdr_y for n in (table or {}).get(2 * k, []) if n in ref_map.photos]
    if numbers.get(2 * k - 1, 0.0) + numbers.get(2 * k, 0.0) > 0 and row_ys:
        y0 = median(row_ys)
        for nid, node in ref_map.photos.items():
            dy = abs(node.pdr_y - y0)
            if dy <= ROW_TOLERANCE_M:
                ev[nid] += 1.0
            elif dy <= ROW_NEAR_M:
                ev[nid] += 0.5
    if len(numbers) >= 3:
        for nid in walkway:
            ev[nid] += 1.0
    return dict(ev)
