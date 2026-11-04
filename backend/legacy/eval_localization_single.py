"""eval_localization_single.py — 單張照片定位 A/B（貼近真實：使用時只拍一張）

查詢單位＝單張照片；地圖單位也拆成單張（每個節點 × 每個方向一張）。
每張查詢照片對地圖所有照片打分，最像的那張 → 其節點＝定位地點、其方向＝對到的視角。
判對：預測照片的「節點」== GT 節點（位置對就算對，方向為附帶資訊）。

WGM        : 查詢單張的九宮格 vs 地圖單張的九宮格（_weighted_grid_match）
GraphMatch : 查詢單張的物件圖 vs 地圖單張的物件圖（S_N+S_E+S_T）

Usage:
    python eval_localization_single.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from server.topomap_v2 import TopoGraphV2
from server import scene_graph
from server.locate_v2 import _build_query_grid, _weighted_grid_match
from eval_localization_real import (
    match_nodes, score_graphmatch, build_node_adjacency, TOPOMAP, GT_FILE,
)

VLM_CACHE = HERE / "benchmark_results" / "realworld_0916" / "vlm_cache"
DIRECTIONS = ("front", "right", "back", "left")


def _base_dir(d):
    for w in ("front", "back", "left", "right"):
        if w in (d or ""):
            return w
    return d or ""


# ── 查詢：每張照片各自一個 (direction, objs, adj) ──

def load_query_photos(session, wp):
    out = []
    for direction in DIRECTIONS:
        hits = sorted(glob.glob(str(VLM_CACHE / f"{session}_wp_{wp}_{direction}_*.json")))
        if not hits:
            continue
        data = json.load(open(hits[0], encoding="utf-8"))
        objs, nid = [], 0
        for it in data.get("items", []):
            label = f"{it.get('zh','')} {it.get('en','')}".strip()
            box = it.get("box") or []
            if not label or len(box) != 4:
                continue
            objs.append({"id": nid, "label": label, "box": box,
                         "grid_cell": "", "depth_class": ""})
            nid += 1
        if not objs:
            continue
        adj = {}
        for e in scene_graph.build_adjacency(objs):
            adj[(e["a"], e["b"])] = e["rel_type"]
            adj[(e["b"], e["a"])] = e["rel_type_inv"]
        out.append((direction, objs, adj))
    return out


# ── 地圖：拆成單張 (pid, base_dir, objs, adj, ref_grid) ──

def build_map_photos(topo, node_adj, cand_grids):
    map_photos = []
    for pid in topo.all_photo_nodes():
        objs_by_dir = defaultdict(list)
        for o in topo.photo_objects(pid):
            objs_by_dir[_base_dir(o.get("direction", ""))].append(o)
        grids = cand_grids.get(pid, {})
        for bdir, objs in objs_by_dir.items():
            oid_set = {o["id"] for o in objs}
            adj = {(a, b): r for (a, b), r in node_adj.get(pid, {}).items()
                   if a in oid_set and b in oid_set}
            ref_grid = grids.get(bdir, {})
            map_photos.append({
                "pid": pid, "dir": bdir,
                "objs": [{"id": o["id"], "label": o.get("label", ""), "box": o.get("box") or [],
                          "ocr": o.get("ocr_text", "") or ""}
                         for o in objs],
                "adj": adj, "grid": ref_grid,
            })
    return map_photos


def score_wgm_single(query_grid, ref_grid):
    if not query_grid or not ref_grid:
        return 0.0
    return _weighted_grid_match(query_grid, ref_grid)


def main():
    topo = TopoGraphV2.load(str(TOPOMAP))
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    node_adj = build_node_adjacency(topo)
    from server.locate_v2 import _build_ref_grids
    cand_grids = {p: _build_ref_grids(topo, p, False) for p in topo.all_photo_nodes()}
    map_photos = build_map_photos(topo, node_adj, cand_grids)
    print(f"地圖單張數: {len(map_photos)}（節點×方向）")

    stat = {m: {"top1": 0, "top3": 0, "top5": 0, "dir_ok": 0, "n": 0}
            for m in ("WGM", "GraphMatch")}
    t0 = time.time()
    n_photos = 0

    for key, v in gt.items():
        if v.get("ambiguous"):
            continue
        gt_node = v["node_id"]
        session, wp = key.rsplit("_wp", 1)
        for qdir, q_objs, q_adj in load_query_photos(session, wp):
            n_photos += 1
            qgrid = _build_query_grid([{"box": o["box"], "label": o["label"]} for o in q_objs])

            wgm = [(score_wgm_single(qgrid, mp["grid"]), mp) for mp in map_photos]
            gm = [(score_graphmatch(q_objs, q_adj, mp["objs"], mp["adj"]), mp) for mp in map_photos]

            for m, scored in (("WGM", wgm), ("GraphMatch", gm)):
                scored = sorted(scored, key=lambda x: x[0], reverse=True)
                top_nodes = [mp["pid"] for _, mp in scored]
                s = stat[m]
                s["n"] += 1
                if top_nodes[0] == gt_node:
                    s["top1"] += 1
                    if scored[0][1]["dir"] == _base_dir(qdir):
                        s["dir_ok"] += 1
                if gt_node in top_nodes[:3]:
                    s["top3"] += 1
                if gt_node in top_nodes[:5]:
                    s["top5"] += 1

    print(f"查詢照片數: {n_photos}  (耗時 {time.time()-t0:.1f}s)\n")
    print(f"{'方法':<12}{'Top-1':>10}{'Top-3':>10}{'Top-5':>10}{'方向也對':>10}")
    print("-" * 56)
    for m in ("WGM", "GraphMatch"):
        s = stat[m]
        n = max(s["n"], 1)
        print(f"{m:<12}{s['top1']/n*100:>9.1f}%{s['top3']/n*100:>9.1f}%"
              f"{s['top5']/n*100:>9.1f}%{s['dir_ok']/max(s['top1'],1)*100:>9.1f}%")
    print("\n（Top-k 以『節點』判定；方向也對＝Top-1 命中中，連拍攝方向都對的比例）")


if __name__ == "__main__":
    main()
