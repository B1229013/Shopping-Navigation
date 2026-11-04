"""eval_localization_gm_ocr.py — 忠實 GraphMatch + OCR（單張）

在忠實版 GraphMatch（embedding 節點相似 + spectral 邊約束）之上，加入 OCR 文字比對：
  - 查詢端 OCR：讀 testPDR/<session>/vlm_detections/wp<N>_<dir>.json 的 ocr_texts
  - 地圖端 OCR：物件的 ocr_text，彙整成每張地圖照片的文字集
  - 加成：TF-IDF 加權的文字重疊（獨特詞加重、常見價標降權）
  - 最終分 = GM 分數 + ALPHA × OCR 加成

一次跑完比較 WGM / GM / GM+OCR(多個 ALPHA)。embedding 存硬碟快取避免重載。

Usage: python eval_localization_gm_ocr.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from math import log
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from server import locate_v2
from server.topomap_v2 import TopoGraphV2
from server.locate_v2 import _build_query_grid, _build_ref_grids, ensure_embeddings
from eval_localization_real import build_node_adjacency, TOPOMAP, GT_FILE
from eval_localization_single import load_query_photos, build_map_photos, score_wgm_single, _base_dir
from eval_localization_gm_faithful import spectral_match

EMB_CACHE = HERE / "benchmark_results" / "realworld_0916" / "emb_cache.pkl"
ALPHAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]   # 0.0 等於純 GM


def _norm(s):
    return (s or "").strip().lower()


def load_query_ocr(session, wp, direction):
    p = HERE / "testPDR" / session / "vlm_detections" / f"wp{wp}_{direction}.json"
    if not p.exists():
        return set()
    try:
        data = json.load(open(p, encoding="utf-8"))
    except Exception:
        return set()
    return {_norm(t.get("text")) for t in data.get("ocr_texts", []) if _norm(t.get("text"))}


def main():
    # embedding 硬碟快取
    if EMB_CACHE.exists():
        try:
            locate_v2._embedding_cache.update(pickle.load(open(EMB_CACHE, "rb")))
            print(f"載入 embedding 快取 {len(locate_v2._embedding_cache)} 筆")
        except Exception:
            pass

    topo = TopoGraphV2.load(str(TOPOMAP))
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    node_adj = build_node_adjacency(topo)
    cand_grids = {p: _build_ref_grids(topo, p, False) for p in topo.all_photo_nodes()}
    map_photos = build_map_photos(topo, node_adj, cand_grids)

    # 地圖端 OCR 集 + IDF
    dfc = Counter()
    for mp in map_photos:
        toks = {_norm(o["ocr"]) for o in mp["objs"] if o.get("ocr")}
        mp["ocrset"] = toks
        for t in toks:
            dfc[t] += 1
    N = len(map_photos)
    idf = {t: log((N + 1) / (df + 1)) + 1 for t, df in dfc.items()}
    DEF_IDF = log(N + 1) + 1  # 查詢端獨有詞視為最稀有
    print(f"地圖單張 {N}, OCR 詞彙 {len(idf)}")

    def ocr_bonus(q_ocr, mp_ocr):
        if not q_ocr:
            return 0.0
        shared = q_ocr & mp_ocr
        if not shared:
            return 0.0
        num = sum(idf.get(t, DEF_IDF) for t in shared)
        den = sum(idf.get(t, DEF_IDF) for t in q_ocr)
        return num / den if den else 0.0

    # 查詢 + embedding 預載
    labels = set(o["label"] for mp in map_photos for o in mp["objs"])
    queries = []
    for key, v in gt.items():
        if v.get("ambiguous"):
            continue
        session, wp = key.rsplit("_wp", 1)
        photos = load_query_photos(session, wp)
        items = []
        for qdir, objs, adj in photos:
            for o in objs:
                labels.add(o["label"])
            items.append((qdir, objs, adj, load_query_ocr(session, wp, qdir)))
        if items:
            queries.append((key, v["node_id"], items))
    labels = [l for l in labels if l]
    print(f"載入 {len(labels)} label embedding...", flush=True)
    t0 = time.time()
    ensure_embeddings(labels)
    try:
        EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(locate_v2._embedding_cache, open(EMB_CACHE, "wb"))
    except Exception:
        pass
    print(f"  embedding 完成 ({time.time()-t0:.1f}s)", flush=True)

    methods = ["WGM", "GM"] + [f"GM+OCR×{a}" for a in ALPHAS if a > 0]
    stat = {m: {"top1": 0, "top3": 0, "top5": 0, "n": 0} for m in methods}
    # 另記：有 OCR 命中的查詢，GM vs 最佳 GM+OCR
    q_with_ocr = 0

    t0 = time.time()
    nph = 0
    for key, gt_node, items in queries:
        for qdir, q_objs, q_adj, q_ocr in items:
            nph += 1
            if q_ocr:
                q_with_ocr += 1
            qgrid = _build_query_grid([{"box": o["box"], "label": o["label"]} for o in q_objs])
            rows = []
            for mp in map_photos:
                gm, _ = spectral_match(q_objs, q_adj, mp["objs"], mp["adj"])
                ob = ocr_bonus(q_ocr, mp["ocrset"])
                wg = score_wgm_single(qgrid, mp["grid"])
                rows.append((mp["pid"], wg, gm, ob))

            def evalm(mname, keyfn):
                order = [pid for pid, *_ in sorted(rows, key=keyfn, reverse=True)]
                s = stat[mname]; s["n"] += 1
                if order[0] == gt_node: s["top1"] += 1
                if gt_node in order[:3]: s["top3"] += 1
                if gt_node in order[:5]: s["top5"] += 1

            evalm("WGM", lambda r: r[1])
            evalm("GM", lambda r: r[2])
            for a in ALPHAS:
                if a > 0:
                    evalm(f"GM+OCR×{a}", lambda r, a=a: r[2] + a * r[3])
        if nph % 40 == 0:
            print(f"  ...{nph} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n查詢照片 {nph}（其中有 OCR 的 {q_with_ocr}）  定位耗時 {time.time()-t0:.1f}s\n")
    print(f"{'方法':<14}{'Top-1':>9}{'Top-3':>9}{'Top-5':>9}")
    print("-" * 44)
    for m in methods:
        s = stat[m]; n = max(s["n"], 1)
        print(f"{m:<14}{s['top1']/n*100:>8.1f}%{s['top3']/n*100:>8.1f}%{s['top5']/n*100:>8.1f}%")


if __name__ == "__main__":
    main()
