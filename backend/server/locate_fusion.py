"""locate_fusion.py — DINOv2 visual + object embedding + OCR 融合定位。

Based on classmate's approach: DINOv2 image embeddings as primary signal,
supplemented by object label embedding similarity and OCR text matching.
Weights: visual 5.0 : object 0.5 : OCR 0.5 (from 5-fold CV).

Usage:
  from server.locate_fusion import fusion_localize
  results = fusion_localize(ref_dino_embs, query_dino_emb, ref_map, query_items, query_ocr)
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Optional, Tuple

import numpy as np

from server.neo4j_client import RefMap

log_ = logging.getLogger(__name__)

W_V, W_N, W_O = 5.0, 0.5, 0.5
NODE_SIM_THRESHOLD = 0.55


def _lookup_emb(dino_embs, photo_file):
    """Look up DINOv2 embedding, trying both full path and basename."""
    if not photo_file:
        return None
    emb = dino_embs.get(photo_file)
    if emb is not None:
        return emb
    import os
    return dino_embs.get(os.path.basename(photo_file))


def _norm_ocr(s: str) -> str:
    return (s or "").strip().lower()


def _embedding_sim(a: str, b: str) -> float:
    from server.visual_localization import _hybrid_max_sim
    return _hybrid_max_sim(a, b)


def build_fusion_index(
    ref_map: RefMap,
    dino_embs: Dict[str, np.ndarray],
) -> dict:
    """Build fusion index from RefMap + DINOv2 embeddings keyed by photo_file."""
    entries = []

    for nid, node in ref_map.photos.items():
        for dp in node.directional_photos:
            labels = [o.label_norm or o.label for o in dp.objects if o.label]
            ocrset = set()
            for o in dp.objects:
                t = _norm_ocr(o.ocr_text)
                if t and len(t) >= 2:
                    ocrset.add(t)

            emb = _lookup_emb(dino_embs, dp.photo_file)
            entries.append({
                "nid": nid,
                "slot": dp.slot,
                "labels": labels,
                "ocrset": ocrset,
                "emb": emb,
            })

        if not node.directional_photos and node.objects:
            labels = [o.label_norm or o.label for o in node.objects if o.label]
            ocrset = set()
            for o in node.objects:
                t = _norm_ocr(o.ocr_text)
                if t and len(t) >= 2:
                    ocrset.add(t)
            emb = _lookup_emb(dino_embs, node.photo_file)
            entries.append({
                "nid": nid, "slot": "all",
                "labels": labels, "ocrset": ocrset, "emb": emb,
            })

    N = max(len(entries), 1)
    dfc = Counter()
    for e in entries:
        for t in e["ocrset"]:
            dfc[t] += 1
    idf = {t: log((N + 1) / (d + 1)) + 1 for t, d in dfc.items()}
    def_idf = log(N + 1) + 1

    return {"entries": entries, "idf": idf, "def_idf": def_idf}


def _node_score(q_labels: List[str], r_labels: List[str]) -> float:
    if not q_labels or not r_labels:
        return 0.0
    pairs = []
    for i, ql in enumerate(q_labels):
        for j, rl in enumerate(r_labels):
            s = _embedding_sim(ql, rl)
            if s >= NODE_SIM_THRESHOLD:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    uq, ur, tot = set(), set(), 0.0
    for s, i, j in pairs:
        if i in uq or j in ur:
            continue
        uq.add(i)
        ur.add(j)
        tot += s
    return tot / len(q_labels) if q_labels else 0.0


def fusion_localize(
    index: dict,
    q_emb: Optional[np.ndarray],
    q_labels: List[str],
    q_ocr_tokens: List[str],
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """Return [(node_id, fused_score)] ranked high→low."""
    entries = index["entries"]
    idf = index["idf"]
    DEF = index["def_idf"]
    q_ocr = {_norm_ocr(t) for t in (q_ocr_tokens or []) if _norm_ocr(t)}

    vis = np.zeros(len(entries))
    nod = np.zeros(len(entries))
    ocr = np.zeros(len(entries))

    for k, e in enumerate(entries):
        if q_emb is not None and e["emb"] is not None:
            vis[k] = float(np.dot(q_emb, e["emb"]))
        nod[k] = _node_score(q_labels, e["labels"])
        if q_ocr and e["ocrset"]:
            shared = q_ocr & e["ocrset"]
            if shared:
                ocr[k] = sum(idf.get(t, DEF) for t in shared) / sum(idf.get(t, DEF) for t in q_ocr)

    if len(entries) == 0:
        return []

    def mm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if hi > lo else a * 0

    fused = W_V * mm(vis) + W_N * mm(nod) + W_O * ocr

    best = {}
    for k, e in enumerate(entries):
        if e["nid"] not in best or fused[k] > best[e["nid"]]:
            best[e["nid"]] = fused[k]

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k] if top_k else ranked


# ── Server-side image evidence for loc_fusion.fuse_v2 ─────────────────────
#
# For phones that do not run DINOv2 (Android): embed the uploaded photo here and use
# the fused DINOv2 + object + OCR score as fuse_v2's image evidence, rescaled to 0..2
# per photo (best node 2, worst 0) — the scale config LOC_SERVER_IMAGE_K was tuned on.

_sidecar: dict = {"path": None, "embs": None}
_index_cache: dict = {}


def _load_sidecar(path: str) -> Optional[Dict[str, np.ndarray]]:
    if _sidecar["path"] != path:
        import pickle
        from pathlib import Path
        embs = None
        if path and Path(path).is_file():
            try:
                with open(path, "rb") as f:
                    embs = pickle.load(f)
                log_.info("DINOv2 sidecar: %d reference embeddings from %s", len(embs), path)
            except Exception as e:  # corrupt / incompatible pickle → no server DINOv2
                log_.warning("DINOv2 sidecar %s unreadable: %s", path, e)
        else:
            log_.warning("DINOv2 sidecar not found: %s", path)
        _sidecar.update(path=path, embs=embs)
    return _sidecar["embs"]


_combined_cache: dict = {"key": None, "embs": None}


def _reference_embeddings(ref_map: RefMap, sidecar: Optional[Dict[str, np.ndarray]]):
    """Sidecar vectors plus DirPhoto.dino vectors from Neo4j (Neo4j wins), keyed by
    photo_file. The same dict object is returned while the inputs stay the same, so the
    fusion index cache below keeps working."""
    from_db = {dp.photo_file: dp.dino for node in ref_map.photos.values()
               for dp in node.directional_photos if getattr(dp, "dino", None)}
    if not from_db:
        return sidecar
    key = (id(ref_map), id(sidecar))
    if _combined_cache["key"] != key:
        embs = dict(sidecar or {})
        embs.update({k: np.asarray(v, dtype=np.float32) for k, v in from_db.items()})
        _combined_cache.update(key=key, embs=embs)
        log_.info("DINOv2 reference vectors: %d from Neo4j, %d in total", len(from_db), len(embs))
    return _combined_cache["embs"]


def _index_for(ref_map: RefMap, embs: Dict[str, np.ndarray]) -> dict:
    # RefMap is cached per place by the Neo4j client, so its id is a stable key.
    key = (id(ref_map), id(embs))
    if key not in _index_cache:
        _index_cache.clear()
        _index_cache[key] = build_fusion_index(ref_map, embs)
    return _index_cache[key]


def server_image_evidence(
    ref_map: RefMap,
    image_path: str,
    q_labels: List[str],
    q_ocr_tokens: List[str],
    sidecar_path: str,
) -> Optional[Dict[int, float]]:
    """{nid: 0..2} image evidence for every reference node, or None when the DINOv2
    model, the sidecar or the photo's embedding is unavailable (caller falls back)."""
    from pathlib import Path
    embs = _reference_embeddings(ref_map, _load_sidecar(sidecar_path))
    if not embs or not ref_map.photos or not Path(image_path).is_file():
        return None
    from server import dino_embed
    q_emb = dino_embed.embed_image(image_path)
    if q_emb is None:
        return None
    index = _index_for(ref_map, embs)
    if not any(e["emb"] is not None for e in index["entries"]):
        log_.warning("DINOv2 sidecar has no embedding for any reference photo of this place")
        return None
    ranked = fusion_localize(index, q_emb, q_labels, q_ocr_tokens, top_k=0)
    if not ranked:
        return None
    scores = dict(ranked)
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {nid: 0.0 for nid in ref_map.photos}
    return {nid: 2.0 * (scores.get(nid, lo) - lo) / (hi - lo) for nid in ref_map.photos}
