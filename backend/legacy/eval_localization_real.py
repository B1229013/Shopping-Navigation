"""eval_localization_real.py — 真實查詢 A/B：WGM vs UniGoal-style graph matching

查詢來源：0916家樂福1/2 的實拍照片 VLM 偵測（未參與建圖），每個 test waypoint 有
4 方向。Ground truth 用你手動標註翻譯到現在節點編號的 0916_ground_truth_current.json。

對每個 test waypoint：組出查詢物件圖 → 對地圖全部 59 節點用兩種方法打分排名 →
看正確節點(GT)排第幾。分「GT 節點有物件(可定位)」與「GT 節點無物件(不可能定位)」
兩組分別統計，避免用無物件節點灌壞分數。

Usage:
    python eval_localization_real.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from server.topomap_v2 import TopoGraphV2
from server import scene_graph
from server.locate_v2 import (
    _wgm_similarity, _build_query_grid, _build_ref_grids, _best_wgm_score,
)

TOPOMAP = HERE / "carrefour_a7_imported.json"
GT_FILE = HERE / "testPDR" / "0916_ground_truth_current.json"
VLM_CACHE = HERE / "benchmark_results" / "realworld_0916" / "vlm_cache"
DIRECTIONS = ["front", "right", "back", "left"]
_NODE_SIM_THR = 0.25
_HORIZONTAL = {"left", "right"}
_VERTICAL = {"above", "below"}


def _same_axis(r1: str, r2: str) -> bool:
    return (r1 in _HORIZONTAL and r2 in _HORIZONTAL) or (r1 in _VERTICAL and r2 in _VERTICAL)


# ── 查詢：把一個 test waypoint 的 4 方向 VLM 偵測組成物件圖 ──

def load_query(session: str, wp_num: str) -> Tuple[List[dict], Dict[Tuple[int, int], str]]:
    objs: List[dict] = []
    q_adj: Dict[Tuple[int, int], str] = {}
    nid = 0
    for direction in DIRECTIONS:
        hits = sorted(glob.glob(str(VLM_CACHE / f"{session}_wp_{wp_num}_{direction}_*.json")))
        if not hits:
            continue
        data = json.load(open(hits[0], encoding="utf-8"))
        dir_objs = []
        for it in data.get("items", []):
            zh, en = it.get("zh", ""), it.get("en", "")
            box = it.get("box") or []
            label = f"{zh} {en}".strip()
            if not label or len(box) != 4:
                continue
            o = {"id": nid, "label": label, "box": box,
                 "direction": direction, "grid_cell": "", "depth_class": ""}
            objs.append(o)
            dir_objs.append(o)
            nid += 1
        # 同方向鄰接邊：用與地圖端相同的 scene_graph 邏輯
        for e in scene_graph.build_adjacency(dir_objs, k=scene_graph.DEFAULT_K):
            q_adj[(e["a"], e["b"])] = e["rel_type"]
            q_adj[(e["b"], e["a"])] = e["rel_type_inv"]
    return objs, q_adj


# ── 節點匹配 + 兩種計分（與骨架一致）──

def match_nodes(q_objs, r_objs):
    pairs = []
    for q in q_objs:
        for r in r_objs:
            s = _wgm_similarity(q["label"], r["label"])
            if s > _NODE_SIM_THR:
                pairs.append((s, q["id"], r["id"]))
    pairs.sort(reverse=True)
    uq, ur, mapping, sims = set(), set(), {}, []
    for s, qi, ri in pairs:
        if qi in uq or ri in ur:
            continue
        uq.add(qi); ur.add(ri); mapping[qi] = ri; sims.append(s)
    return mapping, sims


def score_graphmatch(q_objs, q_adj, r_objs, r_adj):
    n_q = len(q_objs)
    if n_q == 0 or not r_objs:
        return 0.0
    mapping, sims = match_nodes(q_objs, r_objs)
    s_n = sum(sims) / n_q
    if not q_adj:
        return s_n
    e_sims, preserved = [], 0
    for (qa, qb), q_rel in q_adj.items():
        if qa in mapping and qb in mapping:
            r_rel = r_adj.get((mapping[qa], mapping[qb]))
            if r_rel is not None:
                preserved += 1
                e_sims.append(1.0 if r_rel == q_rel else (0.5 if _same_axis(r_rel, q_rel) else 0.0))
                continue
        e_sims.append(0.0)
    s_e = sum(e_sims) / len(q_adj)
    s_t = preserved / len(q_adj)
    return (s_n + s_e + s_t) / 3.0


def score_wgm(q_objs, ref_grids):
    """逐方向 WGM（貼近正式 pipeline 的逐照片做法）：每個查詢方向各自建
    query_grid、對候選的參考網格比對，取跨方向的最佳分數（任一張照片強
    匹配即算）。先前把 4 方向塞進同一個 grid 會互相干擾、壓低 WGM。"""
    if not ref_grids:
        return 0.0
    by_dir: Dict[str, list] = defaultdict(list)
    for q in q_objs:
        if q["box"]:
            by_dir[q["direction"]].append(q)
    best = 0.0
    for objs in by_dir.values():
        dets = [{"box": q["box"], "label": q["label"]} for q in objs]
        qg = _build_query_grid(dets)
        if not qg:
            continue
        s = _best_wgm_score(qg, ref_grids)[0]
        if s > best:
            best = s
    return best


def rank_of(scores, target):
    ranked = [pid for _, pid in sorted(scores, key=lambda x: x[0], reverse=True)]
    return ranked.index(target) + 1 if target in ranked else -1


def build_node_adjacency(topo):
    na = defaultdict(dict)
    for u, v, dd in topo.graph.edges(data=True):
        if dd.get("etype") == "adjacent":
            pid = topo.graph.nodes[u].get("photo_id")
            if pid is not None:
                na[pid][(u, v)] = dd.get("rel_type", "")
    return na


def main():
    topo = TopoGraphV2.load(str(TOPOMAP))
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    photo_ids = topo.all_photo_nodes()
    node_adj = build_node_adjacency(topo)
    cand_objs = {p: [{"id": o["id"], "label": o.get("label", ""), "box": o.get("box") or []}
                     for o in topo.photo_objects(p)] for p in photo_ids}
    cand_adj = {p: node_adj.get(p, {}) for p in photo_ids}
    cand_grids = {p: _build_ref_grids(topo, p, False) for p in photo_ids}
    has_obj = {p for p in photo_ids if cand_objs[p]}

    # 分組：GT 節點是否有物件（無物件的節點兩種方法都不可能定位到）
    groups = {"winnable": [], "objectless_gt": []}
    for key, v in gt.items():
        if v.get("ambiguous"):
            continue
        gt_node = v["node_id"]
        (groups["winnable"] if gt_node in has_obj else groups["objectless_gt"]).append((key, gt_node))

    print(f"地圖: {len(photo_ids)} 節點, 有物件 {len(has_obj)}")
    print(f"GT(非模糊): {sum(len(v) for v in groups.values())}  "
          f"→ 可定位(GT有物件): {len(groups['winnable'])}, "
          f"GT為無物件節點: {len(groups['objectless_gt'])}\n")

    for gname, items in groups.items():
        if not items:
            continue
        stat = {m: {"top1": 0, "top3": 0, "top5": 0, "ranksum": 0, "found": 0}
                for m in ("WGM", "GraphMatch")}
        t0 = time.time()
        n_used = 0
        for key, gt_node in items:
            session, wp = key.rsplit("_wp", 1)
            q_objs, q_adj = load_query(session, wp)
            if not q_objs:
                continue
            n_used += 1
            wgm = [(score_wgm(q_objs, cand_grids[c]), c) for c in photo_ids]
            gm = [(score_graphmatch(q_objs, q_adj, cand_objs[c], cand_adj[c]), c) for c in photo_ids]
            for m, sc in (("WGM", wgm), ("GraphMatch", gm)):
                r = rank_of(sc, gt_node)
                if r > 0:
                    s = stat[m]
                    s["found"] += 1; s["ranksum"] += r
                    if r == 1: s["top1"] += 1
                    if r <= 3: s["top3"] += 1
                    if r <= 5: s["top5"] += 1
        print(f"{'='*64}")
        print(f"[{gname}]  查詢數={n_used}  (耗時 {time.time()-t0:.1f}s)")
        print(f"{'方法':<12}{'Top-1':>10}{'Top-3':>10}{'Top-5':>10}{'平均排名':>10}")
        print("-" * 64)
        for m in ("WGM", "GraphMatch"):
            s = stat[m]
            mr = s["ranksum"] / max(s["found"], 1)
            print(f"{m:<12}{s['top1']/max(n_used,1)*100:>9.1f}%"
                  f"{s['top3']/max(n_used,1)*100:>9.1f}%"
                  f"{s['top5']/max(n_used,1)*100:>9.1f}%{mr:>10.2f}")
        print()


if __name__ == "__main__":
    main()
