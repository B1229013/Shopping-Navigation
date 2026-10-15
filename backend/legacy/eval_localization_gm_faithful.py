"""eval_localization_gm_faithful.py — 忠實版 UniGoal-style graph matching（單張）

相對於陽春版的三個修正：
  1. 節點相似度：embedding cosine（text-embedding-3-small），非字串比對
  2. 匹配＝spectral graph matching：用「邊關係一致」當約束找對應（quadratic assignment
     的近似），讓關係真的去限制匹配，而非事後加分
  3. 分數 S=(S_N+S_E+S_T)/3 在「找到的對應」下計算
     （degree 串接、真 GED 可再加；此版先驗證 embedding+邊約束的效果）

與 WGM baseline 並列比較。單張查詢對單張地圖照片。

Usage: python eval_localization_gm_faithful.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from server.topomap_v2 import TopoGraphV2
from server.locate_v2 import (
    _build_query_grid, _build_ref_grids, ensure_embeddings, _embedding_sim,
)
from eval_localization_real import build_node_adjacency, TOPOMAP, GT_FILE
from eval_localization_single import load_query_photos, build_map_photos, score_wgm_single, _base_dir

_HORIZONTAL = {"left", "right"}
_VERTICAL = {"above", "below"}
NODE_THR = 0.55     # embedding cosine 門檻
TOPK = 3            # 每個查詢節點保留前 K 個候選地圖節點（限制 affinity 矩陣大小）
_SIM_MEMO = {}


def _same_axis(a, b):
    return (a in _HORIZONTAL and b in _HORIZONTAL) or (a in _VERTICAL and b in _VERTICAL)


def _edge_cons(qrel, rrel):
    if qrel == rrel:
        return 1.0
    return 0.5 if _same_axis(qrel, rrel) else 0.0


def _nsim(la, lb):
    k = (la, lb)
    v = _SIM_MEMO.get(k)
    if v is None:
        v = _embedding_sim(la, lb)
        _SIM_MEMO[k] = v
    return v


def spectral_match(q_objs, q_adj, r_objs, r_adj):
    """spectral graph matching：回傳 (score, 對應 dict)。"""
    n_q = len(q_objs)
    if n_q == 0 or not r_objs:
        return 0.0, {}
    # 候選指派：每個查詢節點取 embedding 最像的前 TOPK 個地圖節點
    assigns = []  # (qi, rj, nodesim)
    for q in q_objs:
        cand = sorted(((_nsim(q["label"], r["label"]), r["id"]) for r in r_objs), reverse=True)[:TOPK]
        for s, rid in cand:
            if s >= NODE_THR:
                assigns.append((q["id"], rid, s))
    m = len(assigns)
    if m == 0:
        return 0.0, {}
    # affinity：對角＝節點相似；非對角＝兩個指派的邊關係是否一致
    W = np.zeros((m, m), dtype=np.float32)
    for a in range(m):
        W[a, a] = assigns[a][2]
    for a in range(m):
        qi, rj, _ = assigns[a]
        for b in range(a + 1, m):
            qk, rl, _ = assigns[b]
            if qi == qk or rj == rl:
                continue  # 衝突指派不給 pairwise
            qrel = q_adj.get((qi, qk))
            rrel = r_adj.get((rj, rl))
            if qrel is not None and rrel is not None:
                c = _edge_cons(qrel, rrel)
                W[a, b] = W[b, a] = c
    # 主特徵向量 → 貪婪離散化（一對一）
    if m == 1:
        v = np.array([1.0])
    else:
        vals, vecs = np.linalg.eigh(W)
        v = np.abs(vecs[:, -1])
    order = np.argsort(-v)
    used_q, used_r, mp = set(), set(), {}
    for a in order:
        qi, rj, _ = assigns[a]
        if qi in used_q or rj in used_r:
            continue
        used_q.add(qi); used_r.add(rj); mp[qi] = rj
    sim_of = {(qi, rj): s for qi, rj, s in assigns}
    # 分數（在找到的對應下）
    s_n = sum(sim_of[(qi, rj)] for qi, rj in mp.items()) / n_q
    if q_adj:
        tot = len(q_adj); preserved = 0; consist = 0.0
        for (a, b), qrel in q_adj.items():
            if a in mp and b in mp:
                rrel = r_adj.get((mp[a], mp[b]))
                if rrel is not None:
                    preserved += 1
                    consist += _edge_cons(qrel, rrel)
        s_e = consist / tot
        s_t = preserved / tot
        score = (s_n + s_e + s_t) / 3.0
    else:
        score = s_n
    return score, mp


def main():
    topo = TopoGraphV2.load(str(TOPOMAP))
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    node_adj = build_node_adjacency(topo)
    cand_grids = {p: _build_ref_grids(topo, p, False) for p in topo.all_photo_nodes()}
    map_photos = build_map_photos(topo, node_adj, cand_grids)
    print(f"地圖單張數: {len(map_photos)}")

    # 預先載入所有 label 的 embedding（快取；走 API，一次性）
    labels = set()
    for mp in map_photos:
        for o in mp["objs"]:
            labels.add(o["label"])
    queries = []
    for key, v in gt.items():
        if v.get("ambiguous"):
            continue
        session, wp = key.rsplit("_wp", 1)
        photos = load_query_photos(session, wp)
        for _, objs, _ in photos:
            for o in objs:
                labels.add(o["label"])
        if photos:
            queries.append((key, v["node_id"], photos))
    labels = [l for l in labels if l]
    print(f"載入 {len(labels)} 個 label 的 embedding...", flush=True)
    t0 = time.time()
    ensure_embeddings(labels)
    print(f"  embedding 完成 ({time.time()-t0:.1f}s)", flush=True)

    stat = {m: {"top1": 0, "top3": 0, "top5": 0, "dir_ok": 0, "n": 0}
            for m in ("WGM", "GM-faithful")}
    t0 = time.time()
    nph = 0
    for key, gt_node, photos in queries:
        for qdir, q_objs, q_adj in photos:
            nph += 1
            qgrid = _build_query_grid([{"box": o["box"], "label": o["label"]} for o in q_objs])
            wgm = [(score_wgm_single(qgrid, mp["grid"]), mp) for mp in map_photos]
            gm = [(spectral_match(q_objs, q_adj, mp["objs"], mp["adj"])[0], mp) for mp in map_photos]
            for mname, scored in (("WGM", wgm), ("GM-faithful", gm)):
                scored = sorted(scored, key=lambda x: x[0], reverse=True)
                top = [mp["pid"] for _, mp in scored]
                s = stat[mname]; s["n"] += 1
                if top[0] == gt_node:
                    s["top1"] += 1
                    if scored[0][1]["dir"] == _base_dir(qdir):
                        s["dir_ok"] += 1
                if gt_node in top[:3]: s["top3"] += 1
                if gt_node in top[:5]: s["top5"] += 1
        if nph % 40 == 0:
            print(f"  ...{nph} 張 ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n查詢照片數: {nph}  (定位耗時 {time.time()-t0:.1f}s)\n")
    print(f"{'方法':<14}{'Top-1':>10}{'Top-3':>10}{'Top-5':>10}{'方向也對':>10}")
    print("-" * 56)
    for mname in ("WGM", "GM-faithful"):
        s = stat[mname]; n = max(s["n"], 1)
        print(f"{mname:<14}{s['top1']/n*100:>9.1f}%{s['top3']/n*100:>9.1f}%"
              f"{s['top5']/n*100:>9.1f}%{s['dir_ok']/max(s['top1'],1)*100:>9.1f}%")


if __name__ == "__main__":
    main()
