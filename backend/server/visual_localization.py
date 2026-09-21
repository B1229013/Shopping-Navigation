"""Visual localization — match a live photo's detections against the Neo4j
reference topological map to correct the user's position.

Three matching strategies:

1. **Grid matching** (primary, requires bounding boxes):
   Split each photo into 3×3 grid cells, classify features by weight
   (landmark 5×, sign 3×, equipment 2×, normal 1×, generic 0.3×),
   then compare per-cell with semantic similarity. Achieves 91.5% accuracy.

2. **Object-set matching** (fallback, no bbox needed):
   Compare detected object labels using Jaccard similarity + OCR bonus.
   Works when bounding boxes aren't available.

3. **VLM Re-ranking** (optional, adds ~10s):
   After initial top-5, send query photo + 5 best-matching reference photos
   to VLM for visual comparison. Improves to 94.6% accuracy.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PIL import Image, ImageOps

from server.neo4j_client import Neo4jClient, RefMap, RefPhotoNode

log = logging.getLogger(__name__)


@dataclass
class LocalizationResult:
    """Result of matching a photo against the reference map."""
    matched_nid: Optional[int]
    confidence: float
    method: str                    # "grid" | "grid+rerank" | "object_set" | "none"
    reasoning: str
    ref_node: Optional[RefPhotoNode] = None

    runner_up_nid: Optional[int] = None
    runner_up_score: float = 0.0

    matched_heading: Optional[float] = None
    matched_slot: Optional[str] = None
    heading_confidence: float = 0.0

    top_candidates: list = None  # [(nid, score, reason), ...] for optional re-ranking


# ── Grid-based weighted matching (primary) ────────────────────────────────

GRID_ORDER = [
    "top-left", "top-center", "top-right",
    "mid-left", "mid-center", "mid-right",
    "bottom-left", "bottom-center", "bottom-right",
]

ADJACENT = {
    "top-left": ["top-center", "mid-left", "mid-center"],
    "top-center": ["top-left", "top-right", "mid-center", "mid-left", "mid-right"],
    "top-right": ["top-center", "mid-right", "mid-center"],
    "mid-left": ["top-left", "top-center", "mid-center", "bottom-left", "bottom-center"],
    "mid-center": GRID_ORDER,
    "mid-right": ["top-right", "top-center", "mid-center", "bottom-right", "bottom-center"],
    "bottom-left": ["mid-left", "mid-center", "bottom-center"],
    "bottom-center": ["bottom-left", "bottom-right", "mid-center", "mid-left", "mid-right"],
    "bottom-right": ["mid-right", "mid-center", "bottom-center"],
}

WEIGHT_LANDMARK = 5.0
WEIGHT_SIGN = 3.0
WEIGHT_EQUIPMENT = 2.0
WEIGHT_NORMAL = 1.0
WEIGHT_GENERIC = 0.3

_SIGN_WORDS = {'標示', '指示', '告示', '看板', '招牌', '吊牌', '吊旗', 'sign', 'banner'}
_ZONE_MARKERS = {'走道', '區域', '出口', 'exit', 'aisle', 'zone', 'decathlon',
                 '結帳', '收銀', '手扶梯', '電扶梯'}
_CATEGORY_SIGNS = {'堅果', '海苔', '飲料', '零食', '冷凍', '冷藏', '生鮮', '日用',
                   '烘焙', '麵包', '肉品', '海鮮', '水果', '蔬菜', '花卉',
                   '咖啡', '茶', '乳品', '奶粉', '保健', '清潔', '衛生',
                   '寵物', '酒', '啤酒', '調味', '罐頭', '餅乾', '早餐',
                   '麵條', '即食', '米', '南北貨', '護理', '洗髮', '染髮'}
_EQUIP_WORDS = {'冷藏展示櫃', '冷凍展示櫃', '冷凍櫃', '冰櫃', '冰箱',
                '展示櫃', '貨架', '櫃台', '磅秤', '烤箱',
                'refrigerat', 'freezer', 'display case', 'counter'}
_GENERIC_WORDS = {'促銷立牌', '紅色促銷', '促銷', 'price tag', 'promotional',
                  'price sign', '價牌', '價格'}


def classify_feature(label: str, label2: str = "") -> tuple[float, str]:
    combined = (label + ' ' + label2).lower()
    for kw in _ZONE_MARKERS:
        if kw in combined:
            return WEIGHT_LANDMARK, "landmark"
    has_sign = any(sw in combined for sw in _SIGN_WORDS)
    has_category = any(cw in combined for cw in _CATEGORY_SIGNS)
    if has_sign and has_category:
        return WEIGHT_LANDMARK, "landmark"
    if has_sign:
        return WEIGHT_SIGN, "sign"
    for kw in _EQUIP_WORDS:
        if kw in combined:
            return WEIGHT_EQUIPMENT, "equipment"
    for kw in _GENERIC_WORDS:
        if kw in combined:
            return WEIGHT_GENERIC, "generic"
    return WEIGHT_NORMAL, "normal"


def box_to_grid(box: Optional[List[float]]) -> str:
    if not box or len(box) != 4:
        return "mid-center"
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "mid")
    return f"{row}-{col}"


def _extract_numbers(text: str) -> set:
    return set(re.findall(r'\d+', text))


def _semantic_similarity(ql: str, rl: str) -> float:
    ql_low, rl_low = ql.lower(), rl.lower()
    scores: list[float] = []
    ql_words = set(ql_low.split())
    rl_words = set(rl_low.split())
    if ql_words and rl_words:
        common = ql_words & rl_words
        if common:
            scores.append(len(common) / max(len(ql_words), len(rl_words)))
    ql_chars = set(ql_low)
    rl_chars = set(rl_low)
    char_common = ql_chars & rl_chars - {' ', '/', '／', '（', '）', '「', '」'}
    if char_common:
        scores.append(len(char_common) / max(len(ql_chars), len(rl_chars)) * 0.8)
    q_nums = _extract_numbers(ql)
    r_nums = _extract_numbers(rl)
    if q_nums and r_nums:
        scores.append(0.9 if q_nums & r_nums else -0.3)
    q_cats = {cw for cw in _CATEGORY_SIGNS if cw in ql_low}
    r_cats = {cw for cw in _CATEGORY_SIGNS if cw in rl_low}
    if q_cats and r_cats and q_cats & r_cats:
        scores.append(0.85)
    if ql_low in rl_low or rl_low in ql_low:
        scores.append(min(len(ql), len(rl)) / max(len(ql), len(rl)))
    return max(scores) if scores else 0.0


def weighted_grid_match(
    query_grid: Dict[str, list],
    ref_grid: Dict[str, list],
) -> tuple[float, int, dict]:
    total_weighted = 0.0
    total_possible = 0.0
    landmark_matches: list[tuple] = []
    cell_matches: dict = {}

    for cell in GRID_ORDER:
        q_items = query_grid.get(cell, [])
        r_same = ref_grid.get(cell, [])
        r_adj: list[dict] = []
        for ac in ADJACENT.get(cell, []):
            for ri in ref_grid.get(ac, []):
                r_adj.append({**ri, "_p": 0.7})
        all_r = [{**ri, "_p": 1.0} for ri in r_same] + r_adj

        if not q_items or not all_r:
            total_possible += sum(it["weight"] for it in q_items)
            continue

        used: set = set()
        matches_in_cell: list[dict] = []
        for qi in q_items:
            ql, qw = qi["label"], qi["weight"]
            best_m, best_s, best_i = None, 0.0, -1
            for ri_i, ri in enumerate(all_r):
                k = (id(ri), ri["label"])
                if k in used:
                    continue
                s = _semantic_similarity(ql, ri["label"]) * ri.get("_p", 1.0)
                if s > best_s:
                    best_s, best_m, best_i = s, ri["label"], ri_i
            if best_m and best_s > 0.25:
                used.add((id(all_r[best_i]), best_m))
                total_weighted += qw * best_s
                matches_in_cell.append({
                    "q": ql, "r": best_m,
                    "sim": round(best_s, 3), "cat": qi["cat"],
                })
                if qi["cat"] == "landmark":
                    landmark_matches.append((cell, ql, best_m, best_s))
            total_possible += qw

        if matches_in_cell:
            cell_matches[cell] = matches_in_cell

    return total_weighted / max(total_possible, 1.0), len(landmark_matches), cell_matches


def _build_ref_grid(objects) -> Dict[str, list]:
    grid: Dict[str, list] = defaultdict(list)
    for obj in objects:
        cell = obj.grid_cell or "mid-center"
        label = obj.label_norm or obj.label or ""
        weight, cat = classify_feature(label)
        grid[cell].append({"label": label, "weight": weight, "cat": cat})
    return dict(grid)


def match_by_grid(
    detected_items: List[dict],
    ref_map: RefMap,
    *,
    top_k: int = 10,
) -> List[tuple[int, float, str]]:
    """Score reference nodes using 3×3 grid weighted matching.

    detected_items: list of {"label": str, "box": [x1,y1,x2,y2] normalized [0,1]}
    """
    if not ref_map.photos or not detected_items:
        return []

    query_grid: Dict[str, list] = defaultdict(list)
    for item in detected_items:
        label = item.get("label", "")
        if not label:
            continue
        cell = box_to_grid(item.get("box"))
        weight, cat = classify_feature(label)
        query_grid[cell].append({"label": label, "weight": weight, "cat": cat})

    if not query_grid:
        return []

    scores: list[tuple[int, float, str]] = []

    for nid, ref_node in ref_map.photos.items():
        best_score = 0.0
        best_slot = None
        best_lm = 0

        for dir_photo in ref_node.directional_photos:
            ref_grid = _build_ref_grid(dir_photo.objects)
            score, lm_count, _ = weighted_grid_match(dict(query_grid), ref_grid)
            if score > best_score:
                best_score = score
                best_slot = dir_photo.slot
                best_lm = lm_count

        if not ref_node.directional_photos and ref_node.objects:
            ref_grid = _build_ref_grid(ref_node.objects)
            score, lm_count, _ = weighted_grid_match(dict(query_grid), ref_grid)
            if score > best_score:
                best_score = score
                best_slot = "aggregate"
                best_lm = lm_count

        reason = f"grid={best_score:.3f}"
        if best_slot:
            reason += f" slot={best_slot}"
        if best_lm:
            reason += f" landmarks={best_lm}"

        scores.append((nid, best_score, reason))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


# ── Object-set matching (fallback) ───────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_label(label: str) -> str:
    label = label.lower().strip()
    for canonical, variants in _SYNONYMS.items():
        if label in variants:
            return canonical
    return label


_SYNONYMS = {
    "shelf": {"shelf", "shelves", "shelving", "display rack", "rack",
              "store shelf", "display shelf"},
    "refrigerator": {"refrigerator", "fridge", "cooler", "refrigerated section",
                     "freezer", "frozen section"},
    "snack": {"snack bags", "snack packs", "snacks", "chips", "snack bag"},
    "aisle": {"aisle", "walkway", "corridor"},
    "checkout": {"checkout counter", "cashier counter", "cashier", "register",
                 "checkout", "cash register"},
    "bottle": {"bottles", "bottled water", "beverage bottles", "bottle",
               "water bottle"},
    "fruit": {"fruit display", "fruits", "fruit", "oranges", "apples"},
    "vegetable": {"vegetable display", "vegetables", "produce section",
                  "packaged vegetables", "onions"},
}


def match_by_objects(
    detected_labels: List[str],
    ocr_texts: List[str],
    ref_map: RefMap,
    *,
    top_k: int = 10,
) -> List[tuple[int, float, str]]:
    if not ref_map.photos:
        return []

    live_labels = {_normalize_label(l) for l in detected_labels if l}
    live_ocr = {t.lower().strip() for t in ocr_texts if t}

    scores: list[tuple[int, float, str]] = []

    for nid, ref_node in ref_map.photos.items():
        ref_labels = {_normalize_label(o.label) for o in ref_node.objects}
        ref_ocr = {o.ocr_text.lower().strip() for o in ref_node.objects
                   if o.ocr_text}

        j = _jaccard(live_labels, ref_labels)

        ocr_match_count = 0
        ocr_matched_texts: set = set()
        for live_t in live_ocr:
            if not live_t or len(live_t) < 2:
                continue
            for ref_t in ref_ocr:
                if not ref_t or len(ref_t) < 2:
                    continue
                match_len = min(len(live_t), len(ref_t))
                if match_len < 2:
                    continue
                if live_t == ref_t:
                    ocr_match_count += 1
                    ocr_matched_texts.add(f"{live_t}={ref_t}")
                    break
                elif match_len >= 3 and (live_t in ref_t or ref_t in live_t):
                    ocr_match_count += 1
                    ocr_matched_texts.add(f"{live_t}≈{ref_t}")
                    break
        ocr_bonus = min(ocr_match_count * 0.25, 0.6)

        ref_signs = {_normalize_label(o.label) for o in ref_node.objects
                     if o.role == "標示牌"}
        sign_overlap = len(live_labels & ref_signs)
        sign_bonus = min(sign_overlap * 0.05, 0.15)

        total = j + ocr_bonus + sign_bonus
        parts = [f"jaccard={j:.2f}"]
        if ocr_matched_texts:
            parts.append(f"ocr_match={ocr_matched_texts}")
        if sign_bonus > 0:
            parts.append(f"sign_bonus={sign_bonus:.2f}")

        scores.append((nid, total, " | ".join(parts)))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


# ── Main localization entry point ─────────────────────────────────────────

CONFIDENT_THRESHOLD = 0.05
AMBIGUITY_GAP = 0.10


def estimate_node_heading(matched_ref, detected_labels, ocr_texts, confidence: float):
    """(heading_deg, slot, heading_confidence) of the user's view at ``matched_ref``.

    Matches the live labels/OCR against the node's four directional photos.
    Kept separate from :func:`localize` so the caller can re-run it whenever the
    matched node changes afterwards (e.g. VLM re-ranking) — a heading carried
    over from a different node would make every relative turn wrong.
    """
    matched_heading: Optional[float] = None
    matched_slot: Optional[str] = None
    heading_conf = 0.0

    if matched_ref and matched_ref.directional_photos:
        from server.heading import estimate_heading, best_matching_slot, DirectionalPhoto

        live_labels_lower = {l.lower().strip() for l in detected_labels if l}
        live_ocr_lower = {t.lower().strip() for t in ocr_texts if t}

        dir_photos = [
            DirectionalPhoto(
                slot=dp.slot,
                heading_deg=dp.heading_deg,
                photo_file=dp.photo_file,
                objects=[{"label": o.label, "ocr_text": o.ocr_text}
                         for o in dp.objects],
            )
            for dp in matched_ref.directional_photos
        ]

        matched_heading = estimate_heading(live_labels_lower, live_ocr_lower, dir_photos)
        slot_result = best_matching_slot(live_labels_lower, live_ocr_lower, dir_photos)
        matched_slot = slot_result if isinstance(slot_result, str) else (
            slot_result.value if slot_result else None)

        # Detect if heading data is actually calibrated (not all zeros)
        all_headings = [dp.heading_deg for dp in matched_ref.directional_photos]
        has_real_headings = len(set(all_headings)) > 1

        unique_obj_sets = len({
            frozenset(o.label for o in dp.objects)
            for dp in matched_ref.directional_photos
        })
        if not has_real_headings:
            heading_conf = 0.0
            matched_heading = None
        elif unique_obj_sets > 1:
            heading_conf = min(confidence, 0.8)
        else:
            heading_conf = 0.2

    return matched_heading, matched_slot, heading_conf


def localize(
    detected_labels: List[str],
    ocr_texts: List[str],
    neo4j: Neo4jClient,
    *,
    place: Optional[str] = None,
    hint_nid: Optional[int] = None,
    detected_items: Optional[List[dict]] = None,
) -> LocalizationResult:
    """Match a user's live photo detections against the Neo4j reference map.

    Args:
        detected_labels: object labels from VLM on the user's photo
        ocr_texts: OCR text strings found in the user's photo
        neo4j: connected Neo4j client
        place: the place to match against (default: config.NEO4J_PLACE)
        hint_nid: prefer nearby nodes when breaking ties
        detected_items: items with bounding boxes for grid matching.
            Each item: {"label": str, "box": [x1,y1,x2,y2] in [0,1]}.
            When provided with boxes, uses grid matching (91.5% accuracy)
            instead of Jaccard fallback.
    """
    ref_map = neo4j.load_reference_map(place)
    if not ref_map.photos:
        return LocalizationResult(
            matched_nid=None, confidence=0.0, method="none",
            reasoning="No reference map loaded from Neo4j",
        )

    has_boxes = detected_items and any(it.get("box") for it in detected_items)
    if has_boxes:
        candidates = match_by_grid(detected_items, ref_map, top_k=10)
        match_method = "grid"
    else:
        candidates = match_by_objects(detected_labels, ocr_texts, ref_map, top_k=10)
        match_method = "object_set"

    if not candidates:
        return LocalizationResult(
            matched_nid=None, confidence=0.0, method=match_method,
            reasoning="No candidates matched",
        )

    best_nid, best_score, best_reason = candidates[0]
    runner_nid = candidates[1][0] if len(candidates) > 1 else None
    runner_score = candidates[1][1] if len(candidates) > 1 else 0.0

    if hint_nid is not None and hint_nid in ref_map.photos:
        hint_neighbors = set(ref_map.photos[hint_nid].neighbor_nids)
        hint_2hop: set[int] = set()
        for n in hint_neighbors:
            if n in ref_map.photos:
                hint_2hop.update(ref_map.photos[n].neighbor_nids)
        hint_2hop.update(hint_neighbors)
        hint_2hop.add(hint_nid)

        boosted = []
        for nid, score, reason in candidates:
            bonus = 0.0
            if nid in hint_neighbors:
                bonus = 0.08
            elif nid in hint_2hop:
                bonus = 0.04
            boosted.append((nid, score + bonus,
                            reason + (f" | proximity+{bonus:.2f}" if bonus else "")))

        boosted.sort(key=lambda x: -x[1])
        best_nid, best_score, best_reason = boosted[0]
        runner_nid = boosted[1][0] if len(boosted) > 1 else None
        runner_score = boosted[1][1] if len(boosted) > 1 else 0.0

    confidence = min(best_score / 0.6, 1.0)

    if best_score < CONFIDENT_THRESHOLD:
        return LocalizationResult(
            matched_nid=None, confidence=confidence, method=match_method,
            reasoning=f"Best score {best_score:.2f} below threshold ({best_reason})",
            runner_up_nid=runner_nid, runner_up_score=runner_score,
        )

    # ── Heading estimation via directional photos ────────────────────
    matched_ref = ref_map.photos.get(best_nid)
    matched_heading, matched_slot, heading_conf = estimate_node_heading(
        matched_ref, detected_labels, ocr_texts, confidence)

    return LocalizationResult(
        matched_nid=best_nid,
        confidence=confidence,
        method=match_method,
        reasoning=best_reason,
        ref_node=matched_ref,
        runner_up_nid=runner_nid,
        runner_up_score=runner_score,
        matched_heading=matched_heading,
        matched_slot=matched_slot,
        heading_confidence=heading_conf,
        top_candidates=candidates[:5],
    )


# ── VLM Re-ranking (optional) ────────────────────────────────────────────

RERANK_PROMPT = (
    "你正在進行超市室內定位。上面是一張使用者拍攝的「查詢照片」，"
    "後面是 {n} 張編號 1~{n} 的「候選參考照片」。\n\n"
    "請比較查詢照片與每張候選照片的視覺相似度，考慮：\n"
    "- 相同的標示牌、吊牌、區域標示文字\n"
    "- 相同的商品種類和擺設方式\n"
    "- 相同的設備（冷藏櫃、貨架類型等）\n"
    "- 整體場景和空間佈局的相似程度\n\n"
    '回覆 JSON（不要加其他文字）：\n'
    '{{"ranking": [最像的編號, 第二像, ..., 最不像的編號], '
    '"reason": "簡短說明為何第一名最像"}}'
)


def _encode_photo_b64(path: str, max_size: int = 512) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    img = Image.open(p)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _resolve_ref_photo(neo4j_path: str, ref_photo_root: str) -> Optional[str]:
    if not neo4j_path or not ref_photo_root:
        return None
    parts = neo4j_path.split("/", 1)
    if len(parts) == 2:
        folder = parts[0].replace("set", "")
        resolved = Path(ref_photo_root) / folder / parts[1]
        if resolved.exists():
            return str(resolved)
    full = Path(ref_photo_root) / neo4j_path
    if full.exists():
        return str(full)
    return None


def vlm_rerank(
    query_image_path: str,
    candidates: List[tuple[int, float, str]],
    ref_map: RefMap,
    *,
    ref_photo_root: str,
    api_key: str,
    api_base_url: str,
    api_model: str,
    backup_key: str = "",
    top_n: int = 5,
    timeout: int = 60,
) -> List[tuple[int, float, str]]:
    """Re-rank top candidates using VLM visual comparison.

    Sends the query image + top-N reference photos to VLM in a single call.
    Returns re-ordered candidates list.
    """
    if not candidates or not query_image_path:
        return candidates

    top = candidates[:top_n]

    query_b64 = _encode_photo_b64(query_image_path)
    if not query_b64:
        log.warning("rerank: cannot encode query image %s", query_image_path)
        return candidates

    ref_images: list[tuple[int, str]] = []
    for nid, score, reason in top:
        ref_node = ref_map.photos.get(nid)
        if not ref_node:
            continue
        photo_path = _resolve_ref_photo(ref_node.photo_file, ref_photo_root)
        if not photo_path:
            for dp in ref_node.directional_photos:
                photo_path = _resolve_ref_photo(dp.photo_file, ref_photo_root)
                if photo_path:
                    break
        if photo_path:
            b64 = _encode_photo_b64(photo_path)
            if b64:
                ref_images.append((nid, b64))

    if len(ref_images) < 2:
        log.warning("rerank: only %d ref images found, skipping", len(ref_images))
        return candidates

    content: list[dict] = [
        {"type": "text", "text": "查詢照片："},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
    ]
    for i, (nid, b64) in enumerate(ref_images, 1):
        content.append({"type": "text", "text": f"候選 {i}（WP{nid}）："})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    prompt = RERANK_PROMPT.format(n=len(ref_images))
    content.append({"type": "text", "text": prompt})

    body = {
        "model": api_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_completion_tokens": 512,
    }

    keys = [api_key]
    if backup_key:
        keys.append(backup_key)

    for key in keys:
        if not key:
            continue
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            r = requests.post(
                f"{api_base_url}/chat/completions",
                json=body, headers=headers, timeout=timeout,
            )
            if r.status_code in (429, 402, 401, 403):
                log.warning("rerank: key rejected (HTTP %d), trying next", r.status_code)
                continue
            r.raise_for_status()
            text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                log.warning("rerank: no JSON in response")
                return candidates
            result = json.loads(match.group(0))
            ranking = result.get("ranking", [])
            reason = result.get("reason", "")

            nid_list = [ref_images[idx - 1][0] for idx in ranking
                        if 1 <= idx <= len(ref_images)]

            if not nid_list:
                return candidates

            nid_to_candidate = {nid: (nid, sc, rs) for nid, sc, rs in candidates}
            reranked = []
            seen = set()
            for nid in nid_list:
                if nid in nid_to_candidate and nid not in seen:
                    orig_nid, orig_score, orig_reason = nid_to_candidate[nid]
                    reranked.append((orig_nid, orig_score,
                                     orig_reason + f" | rerank_reason={reason}"))
                    seen.add(nid)
            for nid, sc, rs in candidates:
                if nid not in seen:
                    reranked.append((nid, sc, rs))
                    seen.add(nid)

            log.info("rerank: %s → %s (%s)",
                     [c[0] for c in top], nid_list, reason[:60])
            return reranked

        except Exception as e:
            log.warning("rerank: VLM call failed: %s", e)
            continue

    return candidates
