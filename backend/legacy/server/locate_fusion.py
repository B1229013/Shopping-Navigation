"""locate_fusion.py — 融合定位器（視覺 DINOv2 + 物件 embedding + OCR）。

正式系統可呼叫的定位方法（取代/補充 locate_v2 的 WGM 路徑）。
實驗結論：單張定位 Top-1 ≈ 66%、Top-5 ≈ 90%（5-fold CV），權重 5:0.5:0.5。
見 backend/定位方法彙整與測試規劃.md。

介面：
  index = build_fusion_index(topo, emb_by_nd)   # emb_by_nd: {(node_id, base_dir): np.ndarray}
  result = localize_fusion(index, query_dino_emb, query_labels, query_ocr_tokens, top_k=5)
     → [(node_id, score), ...]  由高到低

節點相似度用 locate_v2 的 text-embedding（_embedding_sim，需先 ensure_embeddings）。
"""
from __future__ import annotations
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Tuple

import numpy as np

from server.locate_v2 import _embedding_sim

NODE_THR = 0.55
W_V, W_N, W_O = 5.0, 0.5, 0.5   # 視覺 : 物件 : OCR（CV 選出的穩定權重）


def base_dir(slot: str) -> str:
    for w in ("front", "back", "left", "right"):
        if w in (slot or ""):
            return w
    return slot or ""


def norm_ocr(s: str) -> str:
    return (s or "").strip().lower()


def build_fusion_index(topo, emb_by_nd: Dict[Tuple[int, str], np.ndarray]) -> dict:
    """把地圖整理成融合定位用的索引：每個 (節點,方向) 一筆，帶視覺向量/物件/OCR。"""
    nodes_objs = defaultdict(list)
    for nid in topo.graph.nodes:
        dd = topo.graph.nodes[nid]
        if dd.get("ntype") == "object":
            nodes_objs[dd.get("photo_id")].append(
                {"label": dd.get("label", ""), "ocr": dd.get("ocr_text", "") or "",
                 "dir": base_dir(dd.get("direction", ""))})
    entries = []
    for pid, objs in nodes_objs.items():
        bydir = defaultdict(list)
        for o in objs:
            bydir[o["dir"]].append(o)
        for bdir, os_ in bydir.items():
            raw = emb_by_nd.get((pid, bdir))
            if raw is None:
                embs = []
            elif isinstance(raw, list):
                embs = [e for e in raw if e is not None]
            else:
                embs = [raw]   # 向下相容：單一 ndarray
            entries.append({
                "pid": pid, "dir": bdir,
                "labels": [o["label"] for o in os_ if o["label"]],
                "ocrset": {norm_ocr(o["ocr"]) for o in os_ if o["ocr"]},
                "embs": embs,
            })
    N = max(len(entries), 1)
    dfc = Counter()
    for e in entries:
        for t in e["ocrset"]:
            dfc[t] += 1
    idf = {t: log((N + 1) / (d + 1)) + 1 for t, d in dfc.items()}
    return {"entries": entries, "idf": idf, "def_idf": log(N + 1) + 1}


def _node_score(q_labels: List[str], r_labels: List[str]) -> float:
    if not q_labels or not r_labels:
        return 0.0
    pairs = []
    for i, ql in enumerate(q_labels):
        for j, rl in enumerate(r_labels):
            s = _embedding_sim(ql, rl)
            if s >= NODE_THR:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    uq, ur, tot = set(), set(), 0.0
    for s, i, j in pairs:
        if i in uq or j in ur:
            continue
        uq.add(i); ur.add(j); tot += s
    return tot / len(q_labels)


def localize_fusion(index: dict, q_emb, q_labels: List[str], q_ocr_tokens,
                    top_k: int = 5) -> List[Tuple[int, float]]:
    """回傳 [(node_id, 融合分數)]，由高到低（已彙整到節點層級，取各方向最佳）。"""
    entries = index["entries"]; idf = index["idf"]; DEF = index["def_idf"]
    q_ocr = {norm_ocr(t) for t in (q_ocr_tokens or []) if norm_ocr(t)}

    vis = np.zeros(len(entries)); nod = np.zeros(len(entries)); ocr = np.zeros(len(entries))
    for k, e in enumerate(entries):
        if q_emb is not None and e["embs"]:
            vis[k] = max(float(np.dot(q_emb, em)) for em in e["embs"])
        nod[k] = _node_score(q_labels, e["labels"])
        if q_ocr and e["ocrset"]:
            sh = q_ocr & e["ocrset"]
            if sh:
                ocr[k] = sum(idf.get(t, DEF) for t in sh) / sum(idf.get(t, DEF) for t in q_ocr)

    def mm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if hi > lo else a * 0
    fused = W_V * mm(vis) + W_N * mm(nod) + W_O * ocr

    # 彙整到節點：同一節點取各方向最高分
    best = {}
    for k, e in enumerate(entries):
        if e["pid"] not in best or fused[k] > best[e["pid"]]:
            best[e["pid"]] = fused[k]
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k] if top_k else ranked


# ──────────────────────────────────────────────────────────────────
# 正式 server 用的高階入口（lazy 建索引 + 即時嵌入查詢 + 安全退回）
# ──────────────────────────────────────────────────────────────────
import pickle as _pickle
from pathlib import Path as _Path

_SIDECAR = _Path(__file__).resolve().parent.parent / "carrefour_a7_dino.pkl"
_index_cache: dict = {}


def emb_by_nd_from_sidecar(topo, sidecar: Dict[str, np.ndarray]) -> Dict[Tuple[int, str], np.ndarray]:
    """用 topo 各節點 photos dict 的 photo_file 去 sidecar 查 DINO 向量。"""
    out = {}
    for nid in topo.graph.nodes:
        dd = topo.graph.nodes[nid]
        if dd.get("ntype") != "photo":
            continue
        for slot, pf in (dd.get("photos") or {}).items():
            emb = sidecar.get(pf)
            if emb is not None:
                # 合併節點同一方向可能有多組拍攝（set01/set03…），全部保留、比對時取最大，
                # 否則後者會蓋掉前者、丟失最佳視角。
                out.setdefault((nid, base_dir(slot)), []).append(emb)
    return out


def _get_index(topo):
    """lazy 建並快取融合索引（含確保地圖 label 的 text-embedding）。缺 sidecar 回 None。"""
    key = id(topo)
    if key in _index_cache:
        return _index_cache[key]
    if not _SIDECAR.exists():
        _index_cache[key] = None
        return None
    try:
        sidecar = _pickle.load(open(_SIDECAR, "rb"))
        emb_by_nd = emb_by_nd_from_sidecar(topo, sidecar)
        idx = build_fusion_index(topo, emb_by_nd)
        from server.locate_v2 import ensure_embeddings
        ensure_embeddings([l for e in idx["entries"] for l in e["labels"] if l])
        _index_cache[key] = idx
    except Exception:
        _index_cache[key] = None
    return _index_cache[key]


def fusion_localize_for_server(topo, query_image_path: str, det_dicts, ocr_items,
                               top_k: int = 5):
    """正式 server 呼叫點：回傳 [(node_id, score)] 或 None（None → 呼叫端退回 WGM）。

    會：lazy 建索引 → 對現場照片即時算 DINO 向量 → 物件/OCR 比對 → 融合排名。
    DINO 不可用、sidecar 缺、查詢向量算不出來 → 回 None。
    """
    idx = _get_index(topo)
    if idx is None:
        return None
    from server import dino_embed
    q_emb = dino_embed.embed_image(query_image_path)
    if q_emb is None:
        return None
    q_labels = [d.get("label", "") for d in (det_dicts or []) if d.get("label")]
    q_ocr = []
    for it in (ocr_items or []):
        t = it.get("text", "") if isinstance(it, dict) else str(it)
        if t:
            q_ocr.append(t)
    if q_labels:
        from server.locate_v2 import ensure_embeddings
        ensure_embeddings(q_labels)
    return localize_fusion(idx, q_emb, q_labels, q_ocr, top_k=top_k)
