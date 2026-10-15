#!/usr/bin/env python3
"""
merge_4dir_topomaps.py — 合併多個已建好的四方向 topomap 成一張完整地圖

載入各 session 已建好的 topomap（含四方向物件子圖），重新映射 node ID 後
合併成一張圖，真正合併座標接近的同位置節點（物件子圖合併、邊重導向）。

Usage:
    python merge_4dir_topomaps.py \
        --sessions "0916家樂福1" "0916家樂福2" \
        --place "0916家樂福B1F" \
        --merge-radius 0.5 \
        --intra-merge "0916家樂福2:4=8,3=9"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server.topomap_v2 import TopoGraphV2


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _get_object_children(G, photo_nid: int) -> List[int]:
    """取得 photo node 下所有 object children（透過 contains 邊）。"""
    children = []
    for _, v, k, _ in G.out_edges(photo_nid, keys=True, data=True):
        if "contains" in str(k):
            children.append(v)
    return children


def _merge_node_pair(
    topo: TopoGraphV2,
    survivor_nid: int,
    absorbed_nid: int,
) -> int:
    """將 absorbed node 合併進 survivor node，回傳新建的 same_object 連結數。"""
    G = topo.graph

    survivor_objects = _get_object_children(G, survivor_nid)
    absorbed_objects = _get_object_children(G, absorbed_nid)

    # --- 重導向所有 out-edges ---
    for _, v, k, data in list(G.out_edges(absorbed_nid, keys=True, data=True)):
        if v == survivor_nid:
            continue
        new_key = k
        if G.has_edge(survivor_nid, v, key=new_key):
            new_key = f"{k}_m{absorbed_nid}"
        G.add_edge(survivor_nid, v, key=new_key, **dict(data))

    # --- 重導向所有 in-edges ---
    for u, _, k, data in list(G.in_edges(absorbed_nid, keys=True, data=True)):
        if u == absorbed_nid or u == survivor_nid:
            continue
        new_key = k
        if G.has_edge(u, survivor_nid, key=new_key):
            new_key = f"{k}_m{absorbed_nid}"
        G.add_edge(u, survivor_nid, key=new_key, **dict(data))

    # --- 合併 photos dict ---
    s_photos = G.nodes[survivor_nid].get("photos", {})
    a_photos = G.nodes[absorbed_nid].get("photos", {})
    for d, path in a_photos.items():
        if d not in s_photos:
            s_photos[d] = path
    G.nodes[survivor_nid]["photos"] = s_photos

    # --- 平均座標 ---
    s_sd = G.nodes[survivor_nid].get("sensor_data", {})
    a_sd = G.nodes[absorbed_nid].get("sensor_data", {})
    s_sd["pdr_x"] = (s_sd.get("pdr_x", 0) + a_sd.get("pdr_x", 0)) / 2
    s_sd["pdr_y"] = (s_sd.get("pdr_y", 0) + a_sd.get("pdr_y", 0)) / 2

    # --- 標記合併來源 ---
    sources = list(G.nodes[survivor_nid].get("merged_from", []))
    sources.append(absorbed_nid)
    G.nodes[survivor_nid]["merged_from"] = sources
    a_session = G.nodes[absorbed_nid].get("session", "")
    s_session = G.nodes[survivor_nid].get("session", "")
    if a_session and a_session != s_session:
        sessions = list(G.nodes[survivor_nid].get("merged_sessions", [s_session]))
        if a_session not in sessions:
            sessions.append(a_session)
        G.nodes[survivor_nid]["merged_sessions"] = sessions

    # --- 刪除 absorbed node ---
    G.remove_node(absorbed_nid)

    # --- 跨來源物件關聯 ---
    n_linked = 0
    if survivor_objects and absorbed_objects:
        linked = topo.link_same_objects_between_photos(
            survivor_objects, absorbed_objects,
        )
        n_linked = len(linked)

    return n_linked


def merge_topomaps(
    session_dirs: List[Path],
    place_name: str,
    merge_radius: float = 0.5,
    intra_merges: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> TopoGraphV2:
    merged = TopoGraphV2(place_name)

    # session_name -> {wp_index(1-based) -> new_node_id}
    session_wp_map: Dict[str, Dict[int, int]] = {}
    # session_name -> [(new_nid, (x, y))]
    session_photo_coords: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {}

    for sess_dir in session_dirs:
        sess_name = sess_dir.name
        topo_path = sess_dir / "topomap_v2" / "topomap.json"
        if not topo_path.exists():
            print(f"[跳過] {topo_path} 不存在")
            continue

        src = TopoGraphV2.load(topo_path)
        print(f"\n[載入] {sess_name}: "
              f"{len(src.all_photo_nodes())} photo nodes, "
              f"{len(src.all_object_nodes())} object nodes, "
              f"{src.graph.number_of_edges()} edges")

        id_map: Dict[int, int] = {}
        for old_id, data in src.graph.nodes(data=True):
            new_id = merged._new_id()
            id_map[old_id] = new_id
            node_data = dict(data)
            node_data["session"] = sess_name
            merged.graph.add_node(new_id, **node_data)

        for u, v, k, data in src.graph.edges(keys=True, data=True):
            merged.graph.add_edge(id_map[u], id_map[v], key=k, **dict(data))

        # 建立 wp_index -> node_id 映射（photo nodes 按原始 ID 排序 = wp 順序）
        src_photo_ids = sorted(src.all_photo_nodes())
        wp_map: Dict[int, int] = {}
        coords_list = []
        for i, old_id in enumerate(src_photo_ids):
            wp_idx = i + 1  # 1-based
            new_id = id_map[old_id]
            wp_map[wp_idx] = new_id
            ndata = merged.graph.nodes[new_id]
            sd = ndata.get("sensor_data", {})
            x = sd.get("pdr_x", 0.0)
            y = sd.get("pdr_y", 0.0)
            coords_list.append((new_id, (x, y)))

        session_wp_map[sess_name] = wp_map
        session_photo_coords[sess_name] = coords_list
        print(f"  -> 映射完成: {len(id_map)} nodes, wp1~wp{len(src_photo_ids)}")

    # ================================================================
    # 第一階段：Intra-session 合併（使用者指定的同 session 同位置點）
    # ================================================================
    merged_away: Set[int] = set()  # 被吸收掉的 node IDs
    # survivor_map: absorbed_nid -> survivor_nid（處理鏈式合併）
    survivor_map: Dict[int, int] = {}

    def resolve(nid: int) -> int:
        while nid in survivor_map:
            nid = survivor_map[nid]
        return nid

    if intra_merges:
        print(f"\n[合併] Intra-session 合併...")
        for sess_name, pairs in intra_merges.items():
            wp_map = session_wp_map.get(sess_name, {})
            for wp_a, wp_b in pairs:
                nid_a = resolve(wp_map.get(wp_a, -1))
                nid_b = resolve(wp_map.get(wp_b, -1))
                if nid_a < 0 or nid_b < 0:
                    print(f"  [跳過] {sess_name} wp{wp_a}↔wp{wp_b}: 找不到 node")
                    continue
                if nid_a == nid_b:
                    print(f"  [跳過] {sess_name} wp{wp_a}↔wp{wp_b}: 已是同一 node")
                    continue

                # 物件多的當 survivor
                obj_a = len(_get_object_children(merged.graph, nid_a))
                obj_b = len(_get_object_children(merged.graph, nid_b))
                if obj_b > obj_a:
                    nid_a, nid_b = nid_b, nid_a

                n_linked = _merge_node_pair(merged, nid_a, nid_b)
                survivor_map[nid_b] = nid_a
                merged_away.add(nid_b)
                print(f"  {sess_name} wp{wp_a}↔wp{wp_b}: "
                      f"node {nid_b} -> {nid_a} "
                      f"({obj_a}+{obj_b} objects, {n_linked} same-object links)")

    # ================================================================
    # 第二階段：Cross-session 自動配對合併（座標距離 < merge_radius）
    # ================================================================
    print(f"\n[合併] Cross-session 自動配對 (半徑 {merge_radius}m)...")

    # 建立候選配對
    all_sessions = list(session_photo_coords.keys())
    candidates: List[Tuple[float, int, int, str, str]] = []
    for i in range(len(all_sessions)):
        for j in range(i + 1, len(all_sessions)):
            sa, sb = all_sessions[i], all_sessions[j]
            for nid_a, xy_a in session_photo_coords[sa]:
                if nid_a in merged_away:
                    continue
                for nid_b, xy_b in session_photo_coords[sb]:
                    if nid_b in merged_away:
                        continue
                    d = _dist(xy_a, xy_b)
                    if d <= merge_radius:
                        candidates.append((d, nid_a, nid_b, sa, sb))

    candidates.sort(key=lambda x: x[0])

    cross_merge_count = 0
    for d, nid_a, nid_b, sa, sb in candidates:
        nid_a = resolve(nid_a)
        nid_b = resolve(nid_b)
        if nid_a == nid_b:
            continue
        if nid_a in merged_away or nid_b in merged_away:
            continue
        if not merged.graph.has_node(nid_a) or not merged.graph.has_node(nid_b):
            continue

        obj_a = len(_get_object_children(merged.graph, nid_a))
        obj_b = len(_get_object_children(merged.graph, nid_b))
        if obj_b > obj_a:
            nid_a, nid_b = nid_b, nid_a

        n_linked = _merge_node_pair(merged, nid_a, nid_b)
        survivor_map[nid_b] = nid_a
        merged_away.add(nid_b)
        cross_merge_count += 1

        xy_a = merged.graph.nodes[nid_a].get("sensor_data", {})
        print(f"  {sa}↔{sb}: node {nid_b} -> {nid_a} "
              f"({xy_a.get('pdr_x',0):.1f}, {xy_a.get('pdr_y',0):.1f}) "
              f"dist={d:.2f}m, {obj_a}+{obj_b} obj, {n_linked} same-obj links")

    # Set start position
    first_coords = list(session_photo_coords.values())[0]
    if first_coords:
        start = resolve(first_coords[0][0])
        if merged.graph.has_node(start):
            merged.set_start_position(start)

    # Summary
    n_photo = len(merged.all_photo_nodes())
    n_object = len(merged.all_object_nodes())
    n_edges = merged.graph.number_of_edges()
    print(f"\n{'='*60}")
    print(f"合併完成: {place_name}")
    print(f"  Photo nodes: {n_photo}")
    print(f"  Object nodes: {n_object}")
    print(f"  Total edges: {n_edges}")
    print(f"  Intra-session 合併: {len(merged_away) - cross_merge_count} 組")
    print(f"  Cross-session 合併: {cross_merge_count} 組")
    print(f"{'='*60}")

    return merged


def _parse_intra_merges(specs: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """解析 --intra-merge 參數，格式: 'session_name:wpA=wpB,wpC=wpD'"""
    result: Dict[str, List[Tuple[int, int]]] = {}
    for spec in specs:
        if ":" not in spec:
            continue
        sess_name, pairs_str = spec.split(":", 1)
        pairs = []
        for pair in pairs_str.split(","):
            if "=" not in pair:
                continue
            a, b = pair.split("=", 1)
            pairs.append((int(a.strip()), int(b.strip())))
        if pairs:
            result[sess_name] = pairs
    return result


def main():
    parser = argparse.ArgumentParser(description="合併多個四方向 topomap（含節點合併）")
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--base-dir", default="testPDR")
    parser.add_argument("--place", default="0916家樂福B1F")
    parser.add_argument("--merge-radius", type=float, default=0.5,
                        help="跨 session 自動合併距離閾值（公尺）")
    parser.add_argument("--intra-merge", nargs="*", default=[],
                        help="Intra-session 合併，格式: 'session:wpA=wpB,wpC=wpD'")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = Path(args.base_dir)
    session_dirs = []
    for s in args.sessions:
        d = base / s
        if d.exists():
            session_dirs.append(d)
        else:
            print(f"[警告] {d} 不存在，跳過")

    if not session_dirs:
        print("需要至少 1 個 session")
        sys.exit(1)

    intra_merges = _parse_intra_merges(args.intra_merge) if args.intra_merge else None

    merged = merge_topomaps(
        session_dirs, args.place, args.merge_radius,
        intra_merges=intra_merges,
    )

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = base / args.place / "topomap_v2" / "topomap.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path)
    print(f"\n儲存: {out_path}")


if __name__ == "__main__":
    main()
