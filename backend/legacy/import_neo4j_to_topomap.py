"""
import_neo4j_to_topomap.py — 從同學的 Neo4j V2 schema 匯入到 TopoGraphV2 JSON

讀取 Neo4j Aura 中以 Waypoint → DirPhoto → Object 結構儲存的地圖資料，
轉換成本專案使用的 TopoGraphV2 格式（NetworkX + JSON）。

Usage:
    # 用環境變數設定連線（同 .env）
    python import_neo4j_to_topomap.py --place "A7家樂福（9/16)" -o carrefour_a7.json

    # 直接指定連線參數
    python import_neo4j_to_topomap.py --place "A7家樂福（9/16)" \\
        --neo4j-host xxxx.databases.neo4j.io \\
        --neo4j-user neo4j \\
        --neo4j-password SECRET \\
        -o carrefour_a7.json

    # 列出所有 place
    python import_neo4j_to_topomap.py --list-places

Neo4j V2 Schema (同學的):
    (:Waypoint {place, nid, pdr_x, pdr_y, heading_deg, session, timestamp})
    (:Waypoint)-[:HAS_PHOTO {slot}]->(:DirPhoto {heading_deg, photo_file})
    (:DirPhoto)-[:DETECTED]->(:Object {label, label_norm, score, ocr_text, role, grid_cell})
    (:Waypoint)-[:WALKWAY {direction, distance_m, steps}]->(:Waypoint)

TopoGraphV2 Schema (本專案):
    Photo node: {ntype=photo, photo_path, photos={front:..., right:...}, sensor_data={pdr_x, pdr_y, heading_deg, ...}}
    Object node: {ntype=object, label, label_norm, score, ocr_text, role, grid_cell, direction}
    Edges: contains_* (photo→object), adjacent (object↔object), walkway (photo→photo)
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # backend/legacy/ → Shopping-Navigation/.env

from server.topomap_v2 import TopoGraphV2

log = logging.getLogger(__name__)


# ── Neo4j HTTP query (no pip neo4j needed) ──

def _build_endpoint(host: str, database: str) -> str:
    host = host.strip().rstrip("/")
    if host.startswith("http"):
        if "/query/v2" not in host:
            return f"{host}/db/{database}/query/v2"
        return host
    return f"https://{host}/db/{database}/query/v2"


def _neo4j_query(
    endpoint: str, auth_header: str,
    cypher: str, params: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    resp = requests.post(
        endpoint,
        json={"statement": cypher, "parameters": params or {}},
        headers={
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    errors = payload.get("errors", [])
    if errors:
        msg = "; ".join(e.get("message", str(e)) for e in errors)
        raise RuntimeError(f"Neo4j error: {msg}")
    data = payload.get("data", {})
    fields = data.get("fields", [])
    values = data.get("values", [])
    return [dict(zip(fields, row)) for row in values]


# ── Main conversion ──

def list_places(endpoint: str, auth: str) -> List[dict]:
    v2 = _neo4j_query(endpoint, auth, """
        MATCH (w:Waypoint)
        WITH w.place AS name, count(w) AS cnt
        RETURN name, cnt ORDER BY name
    """)
    v1 = _neo4j_query(endpoint, auth, """
        MATCH (t:TopoNode {ntype: 'photo'})
        WITH t.place AS name, count(t) AS cnt
        RETURN name, cnt ORDER BY name
    """)
    seen = set()
    result = []
    for r in v2 + v1:
        name = r.get("name")
        if name and name not in seen:
            seen.add(name)
            schema = "V2" if r in v2 else "V1"
            result.append({"name": name, "count": r["cnt"], "schema": schema})
    return result


def import_from_neo4j(
    endpoint: str, auth: str, place: str,
) -> TopoGraphV2:
    """Read classmate's Neo4j V2 data and build a TopoGraphV2."""

    topo = TopoGraphV2(place_name=place)

    # We need to manage our own ID mapping:
    # Neo4j nid (int) → TopoGraphV2 photo node id (int)
    nid_to_photo: Dict[int, int] = {}

    # ── Step 1: Load Waypoints ──
    print("Loading waypoints...", flush=True)
    waypoints = _neo4j_query(endpoint, auth, """
        MATCH (w:Waypoint {place: $place})
        RETURN w.nid AS nid,
               w.pdr_x AS pdr_x, w.pdr_y AS pdr_y,
               w.heading_deg AS heading,
               w.session AS session,
               w.timestamp AS ts,
               w.total_steps AS steps,
               w.total_distance_m AS dist
        ORDER BY w.nid
    """, {"place": place})
    print(f"  {len(waypoints)} waypoints found", flush=True)

    # ── Step 2: Load DirPhotos per Waypoint ──
    print("Loading directional photos...", flush=True)
    dir_photos = _neo4j_query(endpoint, auth, """
        MATCH (w:Waypoint {place: $place})-[hp:HAS_PHOTO]->(p:DirPhoto)
        RETURN w.nid AS nid, hp.slot AS slot,
               p.heading_deg AS heading, p.photo_file AS photo_file,
               elementId(p) AS pid
        ORDER BY w.nid, hp.slot
    """, {"place": place})

    # Group by waypoint nid
    wp_photos: Dict[int, List[dict]] = defaultdict(list)
    for r in dir_photos:
        wp_photos[r["nid"]].append(r)
    print(f"  {len(dir_photos)} directional photos", flush=True)

    # ── Step 3: Create Photo nodes ──
    print("Creating photo nodes...", flush=True)
    pid_to_dir: Dict[str, tuple] = {}  # Neo4j elementId → (topo_photo_id, slot)

    for wp in waypoints:
        nid = wp["nid"]
        photos_for_wp = wp_photos.get(nid, [])

        # Build photos dict {slot: photo_file} and capture per-slot headings.
        # DirPhoto.heading_deg is the REAL compass heading each directional photo
        # was captured at (the Waypoint.heading_deg field is unpopulated / 0), so
        # we must carry it through — it is what downstream direction guidance uses
        # to know which way "front" points. Slots may be plain (front/left/right/
        # back) or session-prefixed on merged nodes (e.g. set01_front, set03_front).
        photos_dict = {}
        dir_headings: Dict[str, float] = {}
        primary_photo = ""
        for dp in photos_for_wp:
            slot = dp["slot"] or "front"
            pfile = dp["photo_file"] or ""
            photos_dict[slot] = pfile
            if dp.get("heading") is not None:
                dir_headings[slot] = float(dp["heading"])
            if slot == "front" and pfile:
                primary_photo = pfile
            pid_to_dir[dp["pid"]] = (nid, slot)

        if not primary_photo and photos_dict:
            primary_photo = next(iter(photos_dict.values()))

        # Resolve the node's front-facing heading. Prefer a 'front' slot; else
        # derive it from any other slot (slots are 90° apart: right=front+90,
        # back=front+180, left=front+270); else fall back to the (usually empty)
        # Waypoint heading. Matches slot names that merely contain the direction
        # word so session-prefixed slots (set01_front, …) still work.
        _SLOT_OFFSET = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}

        def _slot_offset(slot_name: str):
            s = slot_name.lower()
            for word, off in _SLOT_OFFSET.items():
                if word in s:
                    return off
            return None

        front_heading = None
        for slot, h in dir_headings.items():
            off = _slot_offset(slot)
            if off is None:
                continue
            cand = (h - off) % 360.0
            if _slot_offset(slot) == 0.0:  # an actual front slot — most trustworthy
                front_heading = cand
                break
            if front_heading is None:
                front_heading = cand
        if front_heading is None:
            front_heading = wp["heading"] or 0.0

        sensor_data = {
            "pdr_x": wp["pdr_x"] or 0.0,
            "pdr_y": wp["pdr_y"] or 0.0,
            "heading_deg": front_heading,
            "session": wp["session"] or "",
            "total_steps": wp["steps"] or 0,
            "total_distance_m": wp["dist"] or 0.0,
        }
        if dir_headings:
            sensor_data["dir_headings"] = dir_headings

        photo_id = topo.add_photo_node(
            primary_photo,
            timestamp=wp["ts"] or "",
            sensor_data=sensor_data,
            photos=photos_dict,
        )
        nid_to_photo[nid] = photo_id

    print(f"  Created {len(nid_to_photo)} photo nodes", flush=True)

    # ── Step 4: Load Objects and create Object nodes ──
    print("Loading objects...", flush=True)
    objects = _neo4j_query(endpoint, auth, """
        MATCH (w:Waypoint {place: $place})-[hp:HAS_PHOTO]->(p:DirPhoto)-[:DETECTED]->(o:Object)
        RETURN w.nid AS nid, hp.slot AS slot,
               o.label AS label, o.label_norm AS label_norm,
               o.score AS score, o.ocr_text AS ocr_text,
               o.role AS role, o.grid_cell AS grid_cell,
               o.bbox AS bbox, o.source AS source,
               o.grid_cells AS grid_cells, o.depth_class AS depth_class,
               o.piece_count AS piece_count, elementId(o) AS eid
    """, {"place": place})
    print(f"  {len(objects)} objects found", flush=True)

    # Group objects by (nid, slot) for proper ordering
    slot_objects: Dict[tuple, List[dict]] = defaultdict(list)
    for o in objects:
        key = (o["nid"], o["slot"] or "front")
        slot_objects[key].append(o)

    # Neo4j 的 Object.bbox 存成字串 "[x1, y1, x2, y2]"（正規化 0~1，左上原點），
    # 不是陣列，需自行解析。有了 bbox 就能用中心 x 精準排左右，
    # 與本地 build_subgraph_from_detections() 一致；bbox 缺失時退回九宮格欄位。
    _COL_CENTER = {"left": 0.17, "center": 0.5, "right": 0.83}
    _ROW_CENTER = {"top": 0.17, "mid": 0.5, "bottom": 0.83}

    def _parse_bbox(raw) -> Optional[List[float]]:
        if raw is None:
            return None
        vals = raw
        if isinstance(raw, str):
            try:
                vals = json.loads(raw)
            except Exception:
                return None
        if not isinstance(vals, (list, tuple)) or len(vals) != 4:
            return None
        try:
            return [float(v) for v in vals]
        except Exception:
            return None

    def _center(box: Optional[List[float]], cell: str):
        # 物件中心 (x, y)。有 bbox 用 bbox 中心；否則退回九宮格欄/列近似中心。
        if box:
            return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        parts = (cell or "mid-center").split("-")
        row = parts[0] if len(parts) == 2 else "mid"
        col = parts[1] if len(parts) == 2 else "center"
        return (_COL_CENTER.get(col, 0.5), _ROW_CENTER.get(row, 0.5))

    # ── 邊豐富化用的關係計算（對齊 UniGoal：邊帶語意，供 S_E / S_T 計分）──
    # 影像座標 y 向下，故較大的 y = 畫面較低 = below。
    _REL_INV = {"right": "left", "left": "right", "above": "below", "below": "above"}
    _DEPTH_RANK = {"near": 0, "mid": 1, "far": 2}
    _DEPTH_INV = {"nearer": "farther", "farther": "nearer", "same": "same", "": ""}

    def _rel_type(ca, cb) -> str:
        """b 相對 a 的主方位（取水平/垂直位移較大的那軸）。"""
        dx, dy = cb[0] - ca[0], cb[1] - ca[1]
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "below" if dy > 0 else "above"

    def _depth_rel(da: str, db: str) -> str:
        """b 相對 a 的深度關係。"""
        if da not in _DEPTH_RANK or db not in _DEPTH_RANK:
            return ""
        d = _DEPTH_RANK[db] - _DEPTH_RANK[da]
        return "farther" if d > 0 else ("nearer" if d < 0 else "same")

    _ADJ_K = 3  # 每個物件連最近的 K 個鄰居（scene graph 稀疏鄰接）

    obj_count = 0
    adj_count = 0
    bbox_ok = 0
    rel_dist: Dict[str, int] = defaultdict(int)
    eid2oid: Dict[str, int] = {}  # Neo4j Object elementId → 本地節點 id（供匯入上游 ADJACENT）
    for (nid, slot), objs in slot_objects.items():
        if nid not in nid_to_photo:
            continue
        photo_id = nid_to_photo[nid]

        # 建物件節點，記下 (oid, center, label, depth) 供鄰接圖建構用
        created: List[tuple] = []  # (oid, center(x,y), label, depth_class)
        for o in objs:
            label = o["label"] or ""
            if not label:
                continue
            grid_cell = o["grid_cell"] or "center"
            box = _parse_bbox(o.get("bbox"))
            if box:
                bbox_ok += 1
            depth = o.get("depth_class") or ""
            oid = topo.add_object_node(
                photo_id, label,
                box=box,  # 正規化 [x1,y1,x2,y2]，先前寫死 None 而丟失
                score=float(o["score"] or 0.0),
                ocr_text=o["ocr_text"] or "",
                role=o["role"] or None,
                grid_cell=grid_cell,
                direction=slot,
            )
            # add_object_node 沒有這些參數，建點後直接補到節點屬性上：
            #   depth_class: near/mid/far 深度分級
            #   piece_count: 同一偵測框內的同類件數
            #   grid_cells:  跨越的九宮格清單（存字串，解析成 list）
            node = topo.graph.nodes[oid]
            if depth:
                node["depth_class"] = depth
            if o.get("eid"):
                node["neo4j_eid"] = o["eid"]
                eid2oid[o["eid"]] = oid
            if o.get("piece_count") is not None:
                node["piece_count"] = o["piece_count"]
            gcs = o.get("grid_cells")
            if gcs:
                if isinstance(gcs, str):
                    try:
                        gcs = json.loads(gcs)
                    except Exception:
                        gcs = [gcs]
                node["grid_cells"] = gcs
            created.append((oid, _center(box, grid_cell), label, depth))
            obj_count += 1

        # ── 同方向鄰接邊（adjacent，UniGoal 式豐富化）──
        # 只在「同一張照片、同一方向(slot)」內連，不跨方向：跨方向照片視角
        # 是否接壤無法從資料判斷，硬連容易連錯（跨站點的空間銜接由 walkway
        # 邊負責）。改用 kNN 稀疏鄰接（每物件連最近 K 個），每條邊帶：
        #   rel_type  : left/right/above/below（由 bbox 中心相對位置決定）
        #   depth_rel : nearer/farther/same（由 depth_class 決定）
        #   relation  : 含兩端 label 的描述字串，供 S_E 的 embedding 比對
        # 這些是 graph matching（S_E 邊語意、S_T 拓撲）所需的邊屬性。
        n = len(created)
        pairs = set()
        for i in range(n):
            ci = created[i][1]
            order = sorted(
                (j for j in range(n) if j != i),
                key=lambda j: (created[j][1][0] - ci[0]) ** 2 + (created[j][1][1] - ci[1]) ** 2,
            )
            for j in order[:_ADJ_K]:
                pairs.add((min(i, j), max(i, j)))

        for i, j in pairs:
            a_oid, ca, a_label, a_depth = created[i]
            b_oid, cb, b_label, b_depth = created[j]
            rel_ab = _rel_type(ca, cb)
            dep_ab = _depth_rel(a_depth, b_depth)
            rel_str = f"{a_label} [{rel_ab}] {b_label}"
            topo.add_adjacent_edge(a_oid, b_oid, relation=rel_str)
            # add_adjacent_edge 建了雙向兩條邊，逐向補上 rel_type / depth_rel。
            ea = topo.graph.edges[a_oid, b_oid, f"adjacent_{a_oid}_{b_oid}"]
            ea["rel_type"], ea["depth_rel"] = rel_ab, dep_ab
            eb = topo.graph.edges[b_oid, a_oid, f"adjacent_{b_oid}_{a_oid}"]
            eb["rel_type"] = _REL_INV.get(rel_ab, rel_ab)
            eb["depth_rel"] = _DEPTH_INV.get(dep_ab, dep_ab)
            eb["relation"] = f"{b_label} [{eb['rel_type']}] {a_label}"
            rel_dist[rel_ab] += 1
            adj_count += 1

    print(f"  Created {obj_count} object nodes ({bbox_ok} 個帶 bbox)", flush=True)
    print(f"  Created {adj_count} adjacent edges (同方向, kNN-{_ADJ_K}, 帶 rel_type)", flush=True)
    print(f"    rel_type 分佈: {dict(rel_dist)}", flush=True)

    # ── Step 4c: Load upstream ADJACENT (V2, 豐富關係) → etype=adjacent_v2 ──
    # 上游已在 Neo4j 替 V2 物件建好物件↔物件關係，含 depth_diff / overlap /
    # 方向計數 / cell_pairs（細到格對格方向）。原樣帶進來（另存一種邊型，
    # 保留我的 kNN adjacent 以便對照）。用 elementId 對回本地節點 id。
    print("Loading upstream ADJACENT (adjacent_v2)...", flush=True)
    adj_v2 = _neo4j_query(endpoint, auth, """
        MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(:DirPhoto)-[:DETECTED]->(a:Object)
        MATCH (a)-[e:ADJACENT]->(b:Object)
        MATCH (:DirPhoto)-[:DETECTED]->(b)
        RETURN elementId(a) AS ea, elementId(b) AS eb,
               e.depth_diff AS depth_diff, e.overlap AS overlap,
               e.horizontal_count AS h, e.vertical_count AS v,
               e.diagonal_count AS d, e.cell_pairs AS cell_pairs
    """, {"place": place})

    def _dominant_dir(cp_raw):
        try:
            cps = json.loads(cp_raw or "[]")
        except Exception:
            return ""
        c = defaultdict(int)
        for cp in cps:
            c[cp.get("dir", "")] += 1
        return max(c, key=c.get) if c else ""

    v2_count = 0
    for r in adj_v2:
        ao = eid2oid.get(r["ea"]); bo = eid2oid.get(r["eb"])
        if ao is None or bo is None:
            continue
        topo.graph.add_edge(
            ao, bo, key=f"adjv2_{ao}_{bo}", etype="adjacent_v2",
            depth_diff=r.get("depth_diff") or "", overlap=bool(r.get("overlap")),
            horizontal_count=r.get("h") or 0, vertical_count=r.get("v") or 0,
            diagonal_count=r.get("d") or 0, cell_pairs=r.get("cell_pairs") or "",
            dominant_dir=_dominant_dir(r.get("cell_pairs")),
        )
        v2_count += 1
    print(f"  Created {v2_count} adjacent_v2 edges (上游豐富關係)", flush=True)

    # ── Step 5: Load Walkway edges ──
    print("Loading walkway edges...", flush=True)
    walkways = _neo4j_query(endpoint, auth, """
        MATCH (a:Waypoint {place: $place})-[r:WALKWAY]->(b:Waypoint)
        RETURN a.nid AS from_nid, b.nid AS to_nid,
               r.direction AS direction,
               r.distance_m AS distance_m,
               r.steps AS steps
    """, {"place": place})

    edge_count = 0
    for w in walkways:
        from_nid = w["from_nid"]
        to_nid = w["to_nid"]
        if from_nid not in nid_to_photo or to_nid not in nid_to_photo:
            continue
        from_id = nid_to_photo[from_nid]
        to_id = nid_to_photo[to_nid]

        if from_nid == to_nid:
            topo.set_terminal_facing(from_id, direction=w["direction"] or "")
        else:
            topo.add_walkway_edge(
                from_id, to_id,
                direction=w["direction"] or "",
                distance_m=w["distance_m"],
                steps=w["steps"],
            )
        edge_count += 1

    print(f"  Created {edge_count} walkway edges", flush=True)

    # ── Summary ──
    print(f"\n{'='*50}", flush=True)
    print(f"Import complete: {place}", flush=True)
    print(f"  Photo nodes: {len(topo.all_photo_nodes())}", flush=True)
    print(f"  Object nodes: {len(topo.all_object_nodes())}", flush=True)
    print(f"  Adjacent edges (同方向): {adj_count}", flush=True)
    print(f"  Walkway edges: {edge_count}", flush=True)

    # Print nid mapping for reference
    print(f"\n  NID mapping (Neo4j nid → TopoGraphV2 node id):", flush=True)
    for nid in sorted(nid_to_photo.keys()):
        print(f"    WP{nid} → node {nid_to_photo[nid]}", flush=True)

    return topo


def main():
    parser = argparse.ArgumentParser(
        description="Import classmate's Neo4j V2 topomap into TopoGraphV2 JSON",
    )
    parser.add_argument("--place", help="Place name in Neo4j (e.g. 'A7家樂福（9/16)')")
    parser.add_argument("--list-places", action="store_true", help="List all places in Neo4j")
    parser.add_argument("-o", "--output", help="Output JSON path")

    parser.add_argument("--neo4j-host", default=os.getenv("NEO4J_HOST", ""),
                        help="Neo4j Aura host (or set NEO4J_HOST env)")
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"),
                        help="Neo4j username")
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""),
                        help="Neo4j password (or set NEO4J_PASSWORD env)")
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"),
                        help="Neo4j database name")

    # Also support NEO4J_URI format from .env
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", ""),
                        help="Neo4j URI (bolt+s:// or https://) — alternative to --neo4j-host")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Resolve connection
    host = args.neo4j_host
    if not host and args.neo4j_uri:
        uri = args.neo4j_uri
        if "://" in uri:
            host = uri.split("://", 1)[1].split(":")[0].split("/")[0]
        else:
            host = uri
    user = args.neo4j_user
    password = args.neo4j_password
    database = args.neo4j_database

    if not host or not password:
        print("Error: Neo4j connection not configured.", flush=True)
        print("  Set NEO4J_HOST + NEO4J_PASSWORD env vars, or use --neo4j-host/--neo4j-password", flush=True)
        print("  Or set NEO4J_URI in .env (bolt+s://... format)", flush=True)
        sys.exit(1)

    if not user:
        user = host.split(".")[0]

    endpoint = _build_endpoint(host, database)
    creds = base64.b64encode(f"{user}:{password}".encode()).decode()
    auth = f"Basic {creds}"

    # Test connection
    try:
        rows = _neo4j_query(endpoint, auth, "RETURN 1 AS ok")
        if not rows or rows[0].get("ok") != 1:
            print(f"Error: Neo4j ping failed", flush=True)
            sys.exit(1)
        print(f"Connected to Neo4j: {endpoint}", flush=True)
    except Exception as e:
        print(f"Error: Cannot connect to Neo4j: {e}", flush=True)
        sys.exit(1)

    if args.list_places:
        places = list_places(endpoint, auth)
        if not places:
            print("No places found in Neo4j.", flush=True)
        else:
            print(f"\n{'Name':<40} {'Nodes':>6} {'Schema':>8}")
            print("-" * 56)
            for p in places:
                print(f"{p['name']:<40} {p['count']:>6} {p['schema']:>8}")
        return

    if not args.place:
        print("Error: --place is required (use --list-places to see available places)", flush=True)
        sys.exit(1)

    topo = import_from_neo4j(endpoint, auth, args.place)

    out_path = args.output or f"imported_{args.place.replace(' ', '_').replace('/', '_')}.json"
    topo.save(out_path)
    print(f"\nSaved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
