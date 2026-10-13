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
