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
import functools
import io
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from PIL import Image, ImageOps

from server.loc_fusion import (AISLE_WEIGHT, FusionWeights, ImageK, estimate_position, fuse_candidates,
                               fuse_v2, image_evidence)
from server.neo4j_client import Neo4jClient, RefMap, RefPhotoNode

log = logging.getLogger(__name__)

# ── Embedding cache for hybrid_max similarity ───────────────────────────
_emb_cache: Dict[str, np.ndarray] = {}
_emb_loaded = False
_emb_on = False                 # embeddings usable: cache file loaded and/or an API key to fetch
_emb_persisted: set = set()     # labels stored in the cache file (the reference map's)
_EMB_BATCH = 256
_EMB_RETRY_S = 300              # after a failed request, don't try again for this long
_emb_retry_at = 0.0


def _load_embedding_cache() -> bool:
    """Load the label-embedding cache file once. Returns whether hybrid similarity is on:
    EMBEDDING_ENABLED and (vectors on disk or an API key to fetch them)."""
    global _emb_loaded, _emb_on
    if _emb_loaded:
        return _emb_on
    _emb_loaded = True

    from server.config import EMBEDDING_API_KEY, EMBEDDING_CACHE_PATH, EMBEDDING_ENABLED, EMBEDDING_MODEL
    if not EMBEDDING_ENABLED or not EMBEDDING_CACHE_PATH:
        return False
    cache_path = Path(EMBEDDING_CACHE_PATH)
    if cache_path.exists():
        try:
            if cache_path.suffix == ".npz":
                z = np.load(cache_path, allow_pickle=False)
                if str(z["model"]) != EMBEDDING_MODEL:
                    raise ValueError(f"cache is {z['model']}, config wants {EMBEDDING_MODEL}")
                vecs = z["vectors"].astype(np.float32)
                data = dict(zip(z["labels"].tolist(), vecs))
            else:
                data = {label: np.array(vec, dtype=np.float32)
                        for label, vec in json.loads(cache_path.read_text(encoding="utf-8")).items()}
            _emb_cache.update(data)
            _emb_persisted.update(data)
            log.info("Loaded %d label embeddings from %s", len(data), cache_path)
        except Exception as e:
            log.warning("Failed to load embedding cache %s: %s", cache_path, e)
    _emb_on = bool(_emb_cache) or bool(EMBEDDING_API_KEY)
    if not _emb_on:
        log.warning("Label embeddings off: no cache at %s and no EMBEDDING_API_KEY / OPENAI_API_KEY", cache_path)
    return _emb_on


def ensure_label_embeddings(labels: List[str], persist: bool = False) -> int:
    """Fetch the embeddings of labels not cached yet (EMBEDDING_BASE_URL, batched); persist=True also
    writes them to EMBEDDING_CACHE_PATH (.npz) — for the reference map's labels, which
    are the same on every photo. Returns how many were fetched; failures leave the
    labels without a vector (character similarity only)."""
    if not _load_embedding_cache():
        return 0
    global _emb_retry_at
    import time as _time
    missing = list(dict.fromkeys(l for l in labels if l and l not in _emb_cache))
    from server.config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_CACHE_PATH, EMBEDDING_MODEL
    fetched = 0
    if missing and EMBEDDING_API_KEY and _time.time() >= _emb_retry_at:
        for i in range(0, len(missing), _EMB_BATCH):
            batch = missing[i:i + _EMB_BATCH]
            try:
                resp = requests.post(
                    f"{EMBEDDING_BASE_URL}/embeddings",
                    json={"input": batch, "model": EMBEDDING_MODEL},
                    headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"},
                    timeout=30,
                )
                resp.raise_for_status()
            except Exception as e:
                log.warning("Label embedding request failed (%d labels): %s — character similarity "
                            "only for them; retry in %d s", len(batch), e, _EMB_RETRY_S)
                _emb_retry_at = _time.time() + _EMB_RETRY_S
                break
            for item in resp.json()["data"]:
                _emb_cache[batch[item["index"]]] = np.array(item["embedding"], dtype=np.float32)
                fetched += 1
    if persist:
        new = [l for l in labels if l in _emb_cache and l not in _emb_persisted]
        if new and str(EMBEDDING_CACHE_PATH).endswith(".npz"):
            _emb_persisted.update(new)
            keep = sorted(_emb_persisted)
            try:
                path = Path(EMBEDDING_CACHE_PATH)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, model=np.array(EMBEDDING_MODEL), labels=np.array(keep),
                                    vectors=np.stack([_emb_cache[l] for l in keep]).astype(np.float16))
                tmp.replace(path)
                log.info("Saved %d label embeddings to %s", len(keep), path)
            except Exception as e:
                log.warning("Could not save label embeddings to %s: %s", EMBEDDING_CACHE_PATH, e)
    if fetched:
        log.info("Fetched %d label embeddings", fetched)
    return fetched


def _compute_query_embeddings(labels: List[str]) -> int:
    """Embed a photo's labels that are not cached yet (kept in memory only)."""
    return ensure_label_embeddings(labels)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(dot / (na * nb))


def _hybrid_max_sim(ql: str, rl: str) -> float:
    """max(character similarity, embedding cosine similarity)."""
    s_char = _semantic_similarity(ql, rl)
    q_emb = _emb_cache.get(ql)
    r_emb = _emb_cache.get(rl)
    if q_emb is not None and r_emb is not None:
        return max(s_char, _cosine_sim(q_emb, r_emb))
    return s_char


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
    # the matched node's reference photo that the image matcher found most similar
    matched_photo: Optional[str] = None

    top_candidates: list = None  # [(nid, score, reason), ...] for optional re-ranking

    # Map position between nodes (score-weighted mean of the top candidates), metres
    estimated_xy: Optional[tuple] = None
    # v2 fusion: this photo's own image evidence per node (kept for a second photo)
    image_evidence: Optional[dict] = None


# ── Grid-based weighted matching (primary) ────────────────────────────────

def _make_grid(n: int):
    """Generate GRID_ORDER and ADJACENT for an n×n grid using (row, col) cell names."""
    order = [f"{r}-{c}" for r in range(n) for c in range(n)]
    adj: Dict[str, list] = {}
    for r in range(n):
        for c in range(n):
            cell = f"{r}-{c}"
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (dr == 0 and dc == 0) or nr < 0 or nc < 0 or nr >= n or nc >= n:
                        continue
                    neighbors.append(f"{nr}-{nc}")
            adj[cell] = neighbors
    return order, adj

_grid_cache: Dict[int, tuple] = {}

def _get_grid(n: int):
    if n not in _grid_cache:
        _grid_cache[n] = _make_grid(n)
    return _grid_cache[n]

def _box_to_grid_cell(box, n: int) -> str:
    """Map a normalized [0,1] bbox to an n×n grid cell name."""
    if not box or len(box) != 4:
        mid = n // 2
        return f"{mid}-{mid}"
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    col = min(int(cx * n), n - 1)
    row = min(int(cy * n), n - 1)
    return f"{row}-{col}"

# Legacy 3×3 string-based cell names for reference data compatibility
_LEGACY_GRID_MAP_3 = {
    "top-left": "0-0", "top-center": "0-1", "top-right": "0-2",
    "mid-left": "1-0", "mid-center": "1-1", "mid-right": "1-2",
    "bottom-left": "2-0", "bottom-center": "2-1", "bottom-right": "2-2",
}

def _legacy_cell_to_new(cell_str: str, from_n: int, to_n: int) -> str:
    """Remap a cell name from from_n grid to the closest cell in to_n grid."""
    if from_n == to_n and "-" in cell_str and cell_str[0].isdigit():
        return cell_str
    mapped = _LEGACY_GRID_MAP_3.get(cell_str, cell_str)
    if "-" not in mapped or not mapped[0].isdigit():
        mid = to_n // 2
        return f"{mid}-{mid}"
    parts = mapped.split("-")
    old_r, old_c = int(parts[0]), int(parts[1])
    new_r = min(int(old_r / from_n * to_n), to_n - 1)
    new_c = min(int(old_c / from_n * to_n), to_n - 1)
    return f"{new_r}-{new_c}"

WEIGHT_LANDMARK = 5.0
WEIGHT_SIGN = 3.0
WEIGHT_EQUIPMENT = 2.0
WEIGHT_NORMAL = 1.0
WEIGHT_GENERIC = 0.3
RARITY_ALPHA = 1.0
SPATIAL_BONUS_COEFF = 0.10
IDF_AGGRESSIVE_THRESHOLD = 0.35
# "cap"  — legacy: sim·(1+rarity·α) capped at 1.0, so an exact match (sim=1) gets no
#          rarity credit at all and rarity only nudges partial matches.
# "idf"  — every query object's weight is scaled by (1+rarity·α) in both numerator
#          and denominator: rare objects count more whether matched or missed, and
#          the score stays in [0, 1].
RARITY_MODE = "cap"

_OVERRIDABLE = {
    "landmark": "WEIGHT_LANDMARK", "sign": "WEIGHT_SIGN", "equipment": "WEIGHT_EQUIPMENT",
    "normal": "WEIGHT_NORMAL", "generic": "WEIGHT_GENERIC", "ocr": "WEIGHT_OCR",
    "rarity_alpha": "RARITY_ALPHA", "spatial": "SPATIAL_BONUS_COEFF", "rarity_mode": "RARITY_MODE",
}


def apply_weight_overrides(spec) -> Dict[str, object]:
    """Override grid-matching weights, e.g. "landmark=6,ocr=3,rarity_mode=idf" or a dict.

    Returns the values now in effect. Unknown keys raise ValueError.
    """
    if isinstance(spec, str):
        spec = dict(kv.split("=", 1) for kv in spec.split(",") if kv.strip())
    g = globals()
    for key, val in (spec or {}).items():
        key = key.strip()
        if key not in _OVERRIDABLE:
            raise ValueError(f"unknown grid weight {key!r}; choose from {sorted(_OVERRIDABLE)}")
        if key == "rarity_mode":
            val = str(val).strip()
            if val not in ("cap", "idf"):
                raise ValueError("rarity_mode must be 'cap' or 'idf'")
        else:
            val = float(val)
        g[_OVERRIDABLE[key]] = val
    return {k: g[v] for k, v in _OVERRIDABLE.items()}


def _size_importance(bbox_area: float) -> float:
    """Larger objects (closer) get higher matching weight."""
    if bbox_area < 0.005:
        return 0.8
    return 1.0 + 0.5 * math.log2(max(bbox_area, 0.01) / 0.01)


def _compute_aggressive_rarity(ref_map: RefMap) -> Dict[str, float]:
    """IDF with heavy penalty for labels appearing in >35% of photos."""
    label_doc_freq: Dict[str, int] = defaultdict(int)
    total_dp = 0
    for _nid, node in ref_map.photos.items():
        for dp in node.directional_photos:
            total_dp += 1
            seen: set[str] = set()
            for obj in dp.objects:
                ln = (obj.label_norm or obj.label or "").strip()
                if ln and ln not in seen:
                    seen.add(ln)
                    label_doc_freq[ln] += 1
                ocr = (getattr(obj, "ocr_text", "") or "").strip()
                if ocr and len(ocr) >= 2 and ocr not in seen:
                    seen.add(ocr)
                    label_doc_freq[ocr] += 1
        if not node.directional_photos and node.objects:
            total_dp += 1
            seen = set()
            for obj in node.objects:
                ln = (obj.label_norm or obj.label or "").strip()
                if ln and ln not in seen:
                    seen.add(ln)
                    label_doc_freq[ln] += 1
    if total_dp <= 1:
        return {}
    rarity: Dict[str, float] = {}
    for label, df in label_doc_freq.items():
        freq = df / total_dp
        if freq > IDF_AGGRESSIVE_THRESHOLD:
            rarity[label] = max(0.03, 0.15 * (1.0 - freq))
        else:
            rarity[label] = math.log(total_dp / df) / math.log(total_dp)
    return rarity

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


def box_to_grid(box: Optional[List[float]], n: int = 3) -> str:
    return _box_to_grid_cell(box, n)


def compute_query_spatial(items: List[dict]) -> Dict[tuple, dict]:
    """Compute pairwise spatial relationships for query objects from bboxes.

    Returns {(label_a, label_b): {'v': str, 'h': str, 'overlap': bool}}
    where 'v' is 'above'/'below'/'same_row' and 'h' is 'left'/'right'/'same_col'.
    """
    rels: Dict[tuple, dict] = {}
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i >= j:
                continue
            ba, bb = a.get("box"), b.get("box")
            if not ba or not bb or len(ba) != 4 or len(bb) != 4:
                continue
            a_cx = (ba[0] + ba[2]) / 2
            a_cy = (ba[1] + ba[3]) / 2
            b_cx = (bb[0] + bb[2]) / 2
            b_cy = (bb[1] + bb[3]) / 2
            dy = b_cy - a_cy
            dx = b_cx - a_cx
            rel = {
                "v": "above" if dy > 0.08 else ("below" if dy < -0.08 else "same_row"),
                "h": "left" if dx > 0.08 else ("right" if dx < -0.08 else "same_col"),
            }
            a_area = (ba[2] - ba[0]) * (ba[3] - ba[1])
            b_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            ix1, iy1 = max(ba[0], bb[0]), max(ba[1], bb[1])
            ix2, iy2 = min(ba[2], bb[2]), min(ba[3], bb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = a_area + b_area - inter
            rel["overlap"] = union > 0 and inter / union > 0.3

            la = a.get("label", "")
            lb = b.get("label", "")
            rels[(la, lb)] = rel
            rels[(lb, la)] = {
                "v": {"above": "below", "below": "above", "same_row": "same_row"}[rel["v"]],
                "h": {"left": "right", "right": "left", "same_col": "same_col"}[rel["h"]],
                "overlap": rel["overlap"],
            }
    return rels


def _spatial_direction_match(q_rel: dict, ref_rel: dict) -> float:
    """How well a photo pair's relation matches the map's ADJACENT one. Both say where
    object a is relative to b: q_rel from compute_query_spatial, ref_rel from
    neo4j_client.adjacent_relation ({"v_rel", "h_rel", "overlap"})."""
    if not ref_rel:
        return 0.0
    matches = 0.0
    checks = 3
    qv = q_rel["v"]
    if qv in ("above", "below") and qv == ref_rel.get("v_rel"):
        matches += 1
    qh = q_rel["h"]
    if qh in ("left", "right") and qh == ref_rel.get("h_rel"):
        matches += 1
    if q_rel["overlap"] and ref_rel.get("overlap", False):
        matches += 1
    elif not q_rel["overlap"] and not ref_rel.get("overlap", False):
        matches += 0.5
    return matches / checks


def _extract_numbers(text: str) -> set:
    return set(re.findall(r'\d+', text))


_DIR_RE = re.compile(
    r'^(?:右前方|左前方|右後方|左後方|'
    r'左上方|右上方|左下方|右下方|'
    r'走道盡頭(?:左側|右側)?|'
    r'前方|後方|左方|右方|左側|右側|上方|下方|中央)')
_POS_RE = re.compile(r'上層|中層|下層|頂部|底部|上部')
_ENG_POS_RE = re.compile(
    r'\b(?:top|middle|bottom|upper|lower|left|right|front|back|rear)\b',
    re.IGNORECASE)


def _strip_direction(s: str) -> str:
    """Strip directional/positional qualifiers for cross-angle matching."""
    orig = s
    s = _DIR_RE.sub('', s)
    s = _POS_RE.sub('', s)
    s = _ENG_POS_RE.sub('', s)
    s = re.sub(r'  +', ' ', s).strip()
    return s if s else orig


_CHAR_IGNORE = frozenset({' ', '/', '／', '（', '）', '「', '」'})


@functools.lru_cache(maxsize=1 << 16)
def _label_features(label: str):
    """Everything _semantic_similarity derives from one label, computed once per label.

    The reference map's labels are fixed and every photo compares against all of them
    (~600k pairs per photo), so redoing the regexes per pair cost ~15 s per photo.
    """
    low = label.lower()
    n = _strip_direction(low)
    return (low, n, frozenset(n.split()),
            frozenset(cw for cw in _CATEGORY_SIGNS if cw in low),
            frozenset(_extract_numbers(label)),
            frozenset(set(n) - _CHAR_IGNORE))


@functools.lru_cache(maxsize=1 << 20)
def _semantic_similarity(ql: str, rl: str) -> float:
    ql_low, ql_n, ql_words, q_cats, q_nums, ql_chars = _label_features(ql)
    rl_low, rl_n, rl_words, r_cats, r_nums, rl_chars = _label_features(rl)
    if ql_low == rl_low:
        return 1.0

    if ql_n == rl_n:
        return 0.95

    scores: list[float] = []

    if ql_words and rl_words:
        common = ql_words & rl_words
        if common:
            scores.append(len(common) / max(len(ql_words), len(rl_words)))

    if q_cats or r_cats:
        if q_cats and r_cats and q_cats == r_cats:
            scores.append(0.4)
        elif q_cats and r_cats and q_cats & r_cats:
            scores.append(0.3)
        elif q_cats != r_cats and q_cats and r_cats:
            return max(scores) if scores else 0.0

    if q_nums and r_nums:
        if q_nums & r_nums:
            scores.append(0.9)
        else:
            return max(max(scores) - 0.3, 0.0) if scores else 0.0

    if ql_n in rl_n or rl_n in ql_n:
        ratio = min(len(ql_n), len(rl_n)) / max(len(ql_n), len(rl_n))
        scores.append(ratio if ratio > 0.4 else ratio * 0.5)

    if ql_chars and rl_chars:
        char_common = ql_chars & rl_chars
        jaccard = len(char_common) / len(ql_chars | rl_chars)
        scores.append(jaccard * 0.7)

    return max(scores) if scores else 0.0


def _compute_label_rarity(ref_map: RefMap) -> Dict[str, float]:
    """IDF-based rarity: label appearing in fewer DirPhotos → higher score [0, 1]."""
    label_doc_freq: Dict[str, int] = defaultdict(int)
    total_dp = 0

    for _nid, node in ref_map.photos.items():
        for dp in node.directional_photos:
            total_dp += 1
            seen: set[str] = set()
            for obj in dp.objects:
                ln = (obj.label_norm or obj.label or "").strip()
                if ln and ln not in seen:
                    seen.add(ln)
                    label_doc_freq[ln] += 1
                ocr = (getattr(obj, "ocr_text", "") or "").strip()
                if ocr and len(ocr) >= 2 and ocr not in seen:
                    seen.add(ocr)
                    label_doc_freq[ocr] += 1
        if not node.directional_photos and node.objects:
            total_dp += 1
            seen = set()
            for obj in node.objects:
                ln = (obj.label_norm or obj.label or "").strip()
                if ln and ln not in seen:
                    seen.add(ln)
                    label_doc_freq[ln] += 1
                ocr = (getattr(obj, "ocr_text", "") or "").strip()
                if ocr and len(ocr) >= 2 and ocr not in seen:
                    seen.add(ocr)
                    label_doc_freq[ocr] += 1

    if total_dp <= 1:
        return {}

    log_n = math.log(total_dp)
    return {
        label: math.log(total_dp / df) / log_n
        for label, df in label_doc_freq.items()
    }


def weighted_grid_match(
    query_grid: Dict[str, list],
    ref_grid: Dict[str, list],
    rarity_map: Optional[Dict[str, float]] = None,
    grid_n: int = 3,
    query_spatial: Optional[Dict[tuple, dict]] = None,
    ref_adjacent: Optional[Dict[tuple, dict]] = None,
) -> tuple[float, int, dict]:
    """Set-based matching: flatten grids, match all query vs all ref objects."""
    sim_func = _hybrid_max_sim if _emb_cache else _semantic_similarity

    query_items = [qi for items in query_grid.values() for qi in items]
    ref_items = [ri for items in ref_grid.values() for ri in items]

    if not query_items or not ref_items:
        total_possible = sum(
            qi["weight"] * _size_importance(qi.get("bbox_area", 0.0))
            for qi in query_items
        )
        return 0.0, 0, {}

    total_weighted = 0.0
    total_possible = 0.0
    landmark_count = 0
    obj_matches: list[tuple] = []

    used: set[int] = set()
    idf_mode = RARITY_MODE == "idf" and bool(rarity_map)
    for qi in query_items:
        ql, qw = qi["label"], qi["weight"]
        si = _size_importance(qi.get("bbox_area", 0.0))
        if idf_mode:
            # rarity of the query object itself, so the weight is the same for every
            # candidate node; labels never seen in the map count as fully rare
            qw *= 1.0 + rarity_map.get(ql, 1.0) * RARITY_ALPHA
        total_possible += qw * si

        best_s, best_m, best_i = 0.0, None, -1
        for ri_i, ri in enumerate(ref_items):
            if ri_i in used:
                continue
            s = sim_func(ql, ri["label"])
            if s > best_s:
                best_s, best_m, best_i = s, ri["label"], ri_i

        if best_m and best_s > 0.25:
            used.add(best_i)
            combined = best_s
            if rarity_map and not idf_mode:
                rarity = rarity_map.get(best_m, 0.0)
                combined = min(combined * (1.0 + rarity * RARITY_ALPHA), 1.0)
            ew = qw * si
            total_weighted += ew * combined
            obj_matches.append((ql, best_m, combined, ew))
            if qi["cat"] == "landmark":
                landmark_count += 1

    base_score = total_weighted / max(total_possible, 1.0)

    if len(obj_matches) >= 2 and query_spatial and ref_adjacent:
        spatial_hits = 0
        spatial_checks = 0
        spatial_weight = 0.0
        for i, (ql_a, rl_a, _, w_a) in enumerate(obj_matches):
            for j, (ql_b, rl_b, _, w_b) in enumerate(obj_matches):
                if i >= j:
                    continue
                q_rel = query_spatial.get((ql_a, ql_b))
                if not q_rel:
                    continue
                ref_cp = ref_adjacent.get((rl_a, rl_b))      # the loader also stores b → a
                if not ref_cp:
                    continue
                spatial_checks += 1
                score = _spatial_direction_match(q_rel, ref_cp)
                if score > 0.3:
                    spatial_hits += 1
                    spatial_weight += min(w_a, w_b) * score
        if spatial_checks > 0:
            ratio = spatial_hits / spatial_checks
            total_match_weight = sum(m[3] for m in obj_matches)
            bonus = ratio * SPATIAL_BONUS_COEFF * (
                spatial_weight / max(total_match_weight, 1.0))
            base_score += bonus

    return base_score, landmark_count, {}


def _build_ref_grid(objects, grid_n: int = 3) -> Dict[str, list]:
    grid: Dict[str, list] = defaultdict(list)
    for obj in objects:
        bbox = getattr(obj, "bbox", None) or []
        if bbox and len(bbox) == 4:
            cell = _box_to_grid_cell(bbox, grid_n)
        else:
            stored = obj.grid_cell or ""
            cell = _legacy_cell_to_new(stored, 3, grid_n) if stored else _box_to_grid_cell(None, grid_n)
        label = obj.label_norm or obj.label or ""
        weight, cat = classify_feature(label)
        pc = getattr(obj, "piece_count", 0) or 0
        bbox_area = 0.0
        if bbox and len(bbox) == 4:
            bbox_area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        grid[cell].append({"label": label, "weight": weight, "cat": cat, "piece_count": pc, "bbox_area": bbox_area})
        ocr = getattr(obj, "ocr_text", "") or ""
        if ocr.strip() and len(ocr.strip()) >= 2:
            grid[cell].append({"label": ocr.strip(), "weight": WEIGHT_OCR, "cat": "ocr", "piece_count": 0})
    return dict(grid)


WEIGHT_OCR = 4.0


def match_by_grid(
    detected_items: List[dict],
    ref_map: RefMap,
    *,
    top_k: int = 10,
    store_zone: Optional[str] = None,
    ocr_items: Optional[List[dict]] = None,
    grid_n: int = 3,
) -> List[tuple[int, float, str]]:
    """Score reference nodes using n×n grid weighted matching.

    detected_items: list of {"label": str, "box": [x1,y1,x2,y2] normalized [0,1]}
    ocr_items: list of {"text": str, "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] normalized}
    store_zone: VLM-identified store section — candidates with matching objects get a boost.
    grid_n: grid dimension (3=3×3, 4=4×4, 5=5×5)
    """
    if not ref_map.photos or not detected_items:
        return []

    # Load embedding cache and compute query embeddings for hybrid_max
    use_hybrid = _load_embedding_cache()
    if use_hybrid:
        query_labels = [item.get("label", "") for item in detected_items if item.get("label")]
        _compute_query_embeddings(query_labels)

    query_grid: Dict[str, list] = defaultdict(list)
    for item in detected_items:
        label = item.get("label", "")
        if not label:
            continue
        box = item.get("box")
        cell = box_to_grid(box, grid_n)
        weight, cat = classify_feature(label)
        bbox_area = 0.0
        if box and len(box) == 4:
            bbox_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
        query_grid[cell].append({"label": label, "weight": weight, "cat": cat, "bbox_area": bbox_area})

    if ocr_items:
        for oi in ocr_items:
            text = oi.get("text", "").strip()
            if not text or len(text) < 2:
                continue
            bbox = oi.get("bbox")
            if bbox and len(bbox) >= 2:
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                box = [min(xs), min(ys), max(xs), max(ys)]
                cell = box_to_grid(box, grid_n)
            else:
                cell = _box_to_grid_cell(None, grid_n)
            query_grid[cell].append({
                "label": text, "weight": WEIGHT_OCR, "cat": "ocr", "piece_count": 0,
            })

    if not query_grid:
        return []

    query_spatial = compute_query_spatial(detected_items)
    rarity_map = _compute_aggressive_rarity(ref_map)

    scores: list[tuple[int, float, str]] = []

    for nid, ref_node in ref_map.photos.items():
        best_score = 0.0
        best_slot = None
        best_lm = 0

        for dir_photo in ref_node.directional_photos:
            ref_grid = _build_ref_grid(dir_photo.objects, grid_n=grid_n)
            ref_adj = ref_map.adjacent_map.get((nid, dir_photo.slot), None)
            score, lm_count, _ = weighted_grid_match(
                dict(query_grid), ref_grid, rarity_map=rarity_map, grid_n=grid_n,
                query_spatial=query_spatial, ref_adjacent=ref_adj)
            if score > best_score:
                best_score = score
                best_slot = dir_photo.slot
                best_lm = lm_count

        if not ref_node.directional_photos and ref_node.objects:
            ref_grid = _build_ref_grid(ref_node.objects, grid_n=grid_n)
            score, lm_count, _ = weighted_grid_match(
                dict(query_grid), ref_grid, rarity_map=rarity_map, grid_n=grid_n,
                query_spatial=query_spatial)
            if score > best_score:
                best_score = score
                best_slot = "aggregate"
                best_lm = lm_count

        zone_bonus = 0.0
        if store_zone:
            all_objs = list(ref_node.objects or [])
            for dp in (ref_node.directional_photos or []):
                all_objs.extend(dp.objects or [])
            zone_bonus = _zone_boost_for_candidate(store_zone, all_objs)
            best_score += zone_bonus

        reason = f"grid={best_score - zone_bonus:.3f}"
        if zone_bonus:
            reason += f" zone({store_zone})+{zone_bonus:.2f}"
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

CONFIDENT_THRESHOLD = 0.15
AMBIGUITY_GAP = 0.05


_ZONE_KEYWORDS: Dict[str, set[str]] = {
    "寵物食品": {"寵物", "飼料", "貓糧", "貓食", "狗糧", "狗食", "pet food", "pet-life"},
    "冷藏食品": {"冷藏食品", "chilled food"},
    "冷凍食品": {"冷凍食品", "冷凍櫃", "freezer", "冰櫃"},
    "生鮮蔬果": {"蔬菜", "水果", "生鮮", "洋蔥", "橘子", "香蕉", "蘋果"},
    "飲料":     {"飲料", "瓶裝水", "罐裝飲", "椰子水"},
    "零食糖果": {"零食", "糖果", "餅乾", "巧克力", "堅果"},
    "日用品":   {"日用", "清潔劑", "洗衣", "衛生紙", "紙盤", "免洗"},
    "烘焙麵包": {"烘焙", "麵包", "蛋糕"},
    "調味料":   {"調味", "醬油", "沖泡"},
    "肉品海鮮": {"肉品", "肉片", "海鮮", "seafood"},
    "乳製品":   {"乳品", "牛奶", "奶粉", "優酪"},
    "結帳區":   {"結帳", "收銀", "收銀台"},
    "出入口":   {"出口", "入口"},
}


def _zone_boost_for_candidate(zone: str, ref_objects: list) -> float:
    """Return a score boost (0.0 to 0.12) if the waypoint's objects match the VLM zone."""
    keywords = _ZONE_KEYWORDS.get(zone)
    if not keywords:
        return 0.0
    hits = 0
    for obj in ref_objects:
        label = (getattr(obj, 'label_norm', '') or getattr(obj, 'label', '') or '').lower()
        if any(kw in label for kw in keywords):
            hits += 1
    if hits >= 3:
        return 0.12
    elif hits >= 1:
        return 0.06
    return 0.0


def estimate_node_heading(matched_ref, detected_labels, ocr_texts, confidence: float,
                          best_photo=None):
    """(heading_deg, slot, heading_confidence) of the user's view at ``matched_ref``.

    ``best_photo``: the node's directional photo the image matcher (server DINOv2
    fusion) found most similar — its heading is the user's. Without it, matches the
    live labels/OCR against the node's four directional photos.
    Kept separate from :func:`localize` so the caller can re-run it whenever the
    matched node changes afterwards (e.g. VLM re-ranking) — a heading carried
    over from a different node would make every relative turn wrong.
    """
    matched_heading: Optional[float] = None
    matched_slot: Optional[str] = None
    heading_conf = 0.0

    if (matched_ref and best_photo is not None
            and any(dp.photo_file == best_photo.photo_file for dp in matched_ref.directional_photos)):
        # Uncalibrated maps store the same heading for all four photos.
        if len({dp.heading_deg for dp in matched_ref.directional_photos}) > 1:
            return best_photo.heading_deg, best_photo.slot, min(confidence, 0.8)
        return None, best_photo.slot, 0.0

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
    store_zone: Optional[str] = None,
    ocr_items: Optional[List[dict]] = None,
    grid_n: int = 3,
    dino_sims: Optional[Dict[int, float]] = None,
    sensor_pos: Optional[tuple] = None,
    fusion_weights: Optional[FusionWeights] = None,
    v2: bool = False,
    sensor_sigma: Optional[float] = None,
    reachable: Optional[set] = None,
    extra_evidence: Optional[Dict[int, float]] = None,
    aisle_evidence: Optional[Dict[int, float]] = None,
    server_evidence: Optional[Dict[int, float]] = None,
    server_best_photo: Optional[Dict[int, object]] = None,
    server_image_k: Optional[ImageK] = None,
    phone_image_k: Optional[ImageK] = None,
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
        server_evidence: server-side DINOv2 fusion evidence {nid: 0..2}
            (locate_fusion.server_image_evidence), used as the v2 image evidence when
            the phone sent no DINOv2 top-K; fused with server_image_k.
        server_best_photo: {nid: DirectionalRef} — the photo behind each node's
            server_evidence; the matched node's one gives the heading.
        phone_image_k: k(σ) for the phone-evidence path (default loc_fusion constants).
    """
    ref_map = neo4j.load_reference_map(place)
    if not ref_map.photos:
        return LocalizationResult(
            matched_nid=None, confidence=0.0, method="none",
            reasoning="No reference map loaded from Neo4j",
        )

    has_boxes = detected_items and any(it.get("box") for it in detected_items)
    if has_boxes:
        candidates = match_by_grid(detected_items, ref_map, top_k=10,
                                   store_zone=store_zone, ocr_items=ocr_items,
                                   grid_n=grid_n)
        match_method = "grid"
    else:
        candidates = match_by_objects(detected_labels, ocr_texts, ref_map, top_k=10)
        match_method = "object_set"

    node_xy = {nid: (n.pdr_x, n.pdr_y) for nid, n in ref_map.photos.items()}
    if v2 and (dino_sims or sensor_pos is not None or extra_evidence or server_evidence):
        dsims = {n: s for n, s in (dino_sims or {}).items() if n in ref_map.photos}
        image_k = phone_image_k
        best_photos = {}
        if server_evidence and not dsims:
            # Server DINOv2 already fused objects + OCR; add only the aisle signs.
            ev = {n: v for n, v in server_evidence.items() if n in ref_map.photos}
            for nid, v in (aisle_evidence or {}).items():
                ev[nid] = ev.get(nid, 0.0) + AISLE_WEIGHT * v
            image_k = server_image_k
            best_photos = server_best_photo or {}
            match_method = "server_dino"
        else:
            ev = image_evidence(candidates, dsims, aisle_evidence)
        combined = dict(ev)
        for nid, v in (extra_evidence or {}).items():
            combined[nid] = combined.get(nid, 0.0) + v
        post = fuse_v2(combined, node_xy, sensor_pos=sensor_pos,
                       sigma=sensor_sigma if sensor_pos is not None else None, reachable=reachable,
                       image_k=image_k)
        log.info("v2 localization top3: %s%s", [(c[0], round(c[1], 3)) for c in post[:3]],
                 " (with previous photo)" if extra_evidence else "")
        best_nid, best_p, best_reason = post[0]
        common = dict(
            confidence=best_p, method=match_method + "+v2",
            runner_up_nid=post[1][0] if len(post) > 1 else None,
            runner_up_score=post[1][1] if len(post) > 1 else 0.0,
            top_candidates=post[:5], estimated_xy=estimate_position(post, node_xy), image_evidence=ev,
        )
        if best_p < CONFIDENT_THRESHOLD:
            # Spread over many nodes: report it (and keep the evidence for a second
            # photo) but don't name a node, so nothing downstream moves the user.
            return LocalizationResult(
                matched_nid=None,
                reasoning=f"Best probability {best_p:.2f} below threshold ({best_reason})", **common)
        matched_ref = ref_map.photos.get(best_nid)
        best_photo = best_photos.get(best_nid)
        matched_heading, matched_slot, heading_conf = estimate_node_heading(
            matched_ref, detected_labels, ocr_texts, best_p, best_photo=best_photo)
        return LocalizationResult(
            matched_nid=best_nid, reasoning=best_reason, ref_node=matched_ref,
            matched_heading=matched_heading, matched_slot=matched_slot, heading_confidence=heading_conf,
            matched_photo=best_photo.photo_file if best_photo is not None else None,
            **common,
        )

    if fusion_weights is not None and (dino_sims or sensor_pos is not None):
        fused = fuse_candidates(candidates, {n: s for n, s in (dino_sims or {}).items()
                                             if n in ref_map.photos},
                                sensor_pos, node_xy, fusion_weights)
        if fused is not candidates:
            candidates = fused
            match_method += "+fused"
            log.info("Fused localization top3: %s",
                     [(c[0], round(c[1], 3), c[2].split("]")[0] + "]") for c in candidates[:3]])

    if not candidates:
        return LocalizationResult(
            matched_nid=None, confidence=0.0, method=match_method,
            reasoning="No candidates matched",
        )

    if store_zone:
        log.info("Zone-aware matching: zone=%s, top3: %s",
                 store_zone, [(c[0], round(c[1], 3)) for c in candidates[:3]])

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
        # The re-ranker must see the node we actually picked, so hand it the
        # proximity-boosted ordering rather than the raw one.
        candidates = boosted

    confidence = min(best_score, 1.0)

    if best_score < CONFIDENT_THRESHOLD:
        return LocalizationResult(
            matched_nid=None, confidence=confidence, method=match_method,
            reasoning=f"Best score {best_score:.2f} below threshold ({best_reason})",
            runner_up_nid=runner_nid, runner_up_score=runner_score,
        )

    gap = best_score - runner_score
    if gap < AMBIGUITY_GAP and best_score < 0.35:
        log.info("Ambiguous match: WP%s=%.3f vs WP%s=%.3f (gap=%.3f < %.2f)",
                 best_nid, best_score, runner_nid, runner_score, gap, AMBIGUITY_GAP)
        confidence *= 0.5
        best_reason += f" | ambiguous(gap={gap:.3f})"

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
        estimated_xy=estimate_position(
            candidates, {n: (p.pdr_x, p.pdr_y) for n, p in ref_map.photos.items()}),
    )


# ── VLM Re-ranking (optional) ────────────────────────────────────────────

RERANK_PROMPT = (
    "你正在進行超市室內定位。第一張是使用者拍攝的「查詢照片」，"
    "後面是 {n} 個編號 1~{n} 的「候選地點」，每個候選地點有多張在同一位置朝不同方向"
    "（front/right/back/left）拍的參考照片。使用者可能面向任何方向，"
    "所以只要候選地點的「任一方向」照片和查詢照片是同一個位置、同一段走道，就算符合。\n\n"
    "判斷是否為同一位置，請看：\n"
    "- 相同的標示牌、吊牌文字、走道編號\n"
    "- 相同的貨架排列順序、端架商品、牆面顏色與裝飾\n"
    "- 整體走道幾何與空間佈局\n"
    "只是同類型的設備（都是冷藏櫃、都是貨架）不算證據，賣場裡到處都有。\n\n"
    '回覆 JSON（不要加其他文字）：\n'
    '{{"ranking": [最像的候選編號, 第二像, ..., 最不像的候選編號], '
    '"reason": "簡短說明決定性的線索"}}'
)

# How many nodes the re-ranker may compare in one call (each contributes up to
# 4 directional photos). Field runs tether the Mac to the phone's hotspot, so this
# upload competes with the phone's own photo upload: 8 nodes × 4 views × 512 px was
# ~1.9 MB and the proxy connection timed out. 6 × 4 × 384 px is ~0.9 MB.
RERANK_MAX_NODES = 6
REF_PHOTO_MAX_PX = 384
RERANK_HINT_SLOTS = 3   # of those nodes, how many are held for the previous fix and its ring


def neighbourhood_nids(ref_map: RefMap, nid: int, hops: int = 2) -> List[List[int]]:
    """Rings of nodes around ``nid``: [[nid], [1-hop...], [2-hop...]]. Walkway
    edges in the map are stored one-way and sparsely, so both directions count."""
    rings: List[List[int]] = [[nid]]
    seen = {nid}
    frontier = {nid}
    for _ in range(hops):
        nxt: set[int] = set()
        for n in frontier:
            node = ref_map.photos.get(n)
            if node:
                nxt.update(node.neighbor_nids)
            nxt.update(k for k, v in ref_map.photos.items() if n in v.neighbor_nids)
        frontier = {n for n in nxt if n not in seen and n in ref_map.photos}
        seen |= frontier
        rings.append(sorted(frontier))
    return rings


def rerank_candidates(
    top: List[tuple[int, float, str]],
    ref_map: RefMap,
    hint_nid: Optional[int],
    cap: int = RERANK_MAX_NODES,
) -> List[tuple[int, float, str]]:
    """The nodes the re-ranker should compare: the word-matcher's top-N first,
    then the previous fix and its neighbours (nearest rings first) — the user
    cannot have walked far since the last photo, and the word matcher alone
    often leaves the true node out entirely."""
    if hint_nid is None or hint_nid not in ref_map.photos:
        return list(top)

    # Reserve slots for the neighbourhood: where the user stood one photo ago is
    # better evidence than the tail of a word-similarity ranking, and with a small
    # cap the word candidates would otherwise crowd it out entirely.
    keep = max(1, cap - RERANK_HINT_SLOTS)
    merged = list(top[:keep])
    have = {c[0] for c in merged}
    for ring_i, ring in enumerate(neighbourhood_nids(ref_map, hint_nid)):
        for nid in ring:
            if len(merged) >= cap:
                return merged
            if nid not in have:
                merged.append((nid, 0.0, f"near last fix WP{hint_nid} ({ring_i} hop)"))
                have.add(nid)
    # Few or no neighbours: give the spare slots back to the word matcher.
    for cand in top[keep:]:
        if len(merged) >= cap:
            break
        if cand[0] not in have:
            merged.append(cand)
            have.add(cand[0])
    return merged


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


def ref_photo_coverage(ref_map: RefMap, ref_photo_root: str) -> tuple[int, int, int]:
    """(photo files found on disk, photo files referenced by the map, nodes with
    at least one file). Lets startup report whether REF_PHOTO_ROOT really holds
    this map's photos instead of the re-ranker silently skipping every call."""
    found = total = nodes_ok = 0
    for node in ref_map.photos.values():
        files = {node.photo_file, *(dp.photo_file for dp in node.directional_photos)}
        files.discard("")
        node_hits = sum(1 for f in files if _resolve_ref_photo(f, ref_photo_root))
        total += len(files)
        found += node_hits
        nodes_ok += 1 if node_hits else 0
    return found, total, nodes_ok


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
    top_n: int = RERANK_MAX_NODES,
    timeout: int = 60,
) -> List[tuple[int, float, str]]:
    """Re-rank top candidates using VLM visual comparison.

    Sends the query image + every directional reference photo of the top-N
    candidate nodes to the VLM in a single call. Returns the candidates
    re-ordered; ones the VLM did not rank keep their original order after.
    """
    if not candidates or not query_image_path:
        return candidates

    top = candidates[:top_n]

    query_b64 = _encode_photo_b64(query_image_path)
    if not query_b64:
        log.warning("rerank: cannot encode query image %s", query_image_path)
        return candidates

    # One entry per node, with every directional photo we can find for it —
    # the user may be facing any way, so the "front" photo alone misses matches.
    ref_nodes: list[tuple[int, list[tuple[str, str]]]] = []
    for nid, score, reason in top:
        ref_node = ref_map.photos.get(nid)
        if not ref_node:
            continue
        views: list[tuple[str, str]] = []
        for dp in ref_node.directional_photos:
            photo_path = _resolve_ref_photo(dp.photo_file, ref_photo_root)
            b64 = _encode_photo_b64(photo_path, max_size=REF_PHOTO_MAX_PX) if photo_path else None
            if b64:
                views.append((dp.slot, b64))
        if not views:
            photo_path = _resolve_ref_photo(ref_node.photo_file, ref_photo_root)
            b64 = _encode_photo_b64(photo_path, max_size=REF_PHOTO_MAX_PX) if photo_path else None
            if b64:
                views.append(("front", b64))
        if views:
            ref_nodes.append((nid, views))

    if len(ref_nodes) < 2:
        log.warning("rerank: only %d candidate nodes have reference photos, skipping", len(ref_nodes))
        return candidates

    content: list[dict] = [
        {"type": "text", "text": "查詢照片："},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
    ]
    for i, (nid, views) in enumerate(ref_nodes, 1):
        for slot, b64 in views:
            content.append({"type": "text", "text": f"候選 {i}（WP{nid}）{slot}："})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    prompt = RERANK_PROMPT.format(n=len(ref_nodes))
    content.append({"type": "text", "text": prompt})

    body = {
        "model": api_model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": 4096 if "luna" in api_model else 512,
    }
    if "luna" not in api_model:
        body["temperature"] = 0.1

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

            nid_list = [ref_nodes[idx - 1][0] for idx in ranking
                        if 1 <= idx <= len(ref_nodes)]

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


# Apply LOC_GRID_WEIGHTS last, after every weight constant above is defined.
from server.config import LOC_GRID_WEIGHTS as _LOC_GRID_WEIGHTS  # noqa: E402
if _LOC_GRID_WEIGHTS:
    log.info("grid weights override: %s", apply_weight_overrides(_LOC_GRID_WEIGHTS))
