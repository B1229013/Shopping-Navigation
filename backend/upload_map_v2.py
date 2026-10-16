"""Upload the confirmed builder map to Neo4j in the shared V2 schema (server/map_v2.py).

Step 3 of building a new store map: reads <workspace>/topomap_confirmed.json (saved by
/topomap-builder) and <workspace>/topomap.json, then creates the place's
Waypoint / HAS_PHOTO / DirPhoto / WALKWAY. Objects come afterwards from rebuild_objects_v2.py.

Usage (from backend/):
    python upload_map_v2.py --dry-run           # show what would be written, write nothing
    python upload_map_v2.py                     # create the place (refuses if it exists)
    python upload_map_v2.py --replace           # delete this place first, then create it

--replace only deletes a place this script created (its DirPhotos carry ``place``); places
built another way — e.g. the classmate's "A7家樂福（9/16)" — are never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.config import MAP_BUILD_ROOT  # noqa: E402
from server.map_build import list_places, workspace_for  # noqa: E402
from server.map_v2 import photo_local_path, upload_rows  # noqa: E402
from server.neo4j_client import Neo4jClient  # noqa: E402

BATCH = 500


def batches(rows, size=BATCH):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def existing_state(q, place: str) -> dict:
    r = q("""
        MATCH (w:Waypoint {place: $place})
        OPTIONAL MATCH (w)-[:HAS_PHOTO]->(p:DirPhoto)
        RETURN count(DISTINCT w) AS waypoints, count(DISTINCT p) AS photos,
               count(DISTINCT CASE WHEN p.place IS NULL OR p.place <> $place THEN p END) AS foreign_photos
    """, {"place": place})[0]
    return r


def delete_place(q, place: str) -> None:
    """Objects (with their ADJACENT/DETECTED), then DirPhotos, then Waypoints — in batches."""
    for label, match in (
        ("Object", "MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(:DirPhoto)-[:DETECTED]->(n:Object)"),
        ("DirPhoto", "MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(n:DirPhoto)"),
        ("Waypoint", "MATCH (n:Waypoint {place: $place})"),
    ):
        total = 0
        while True:
            n = q(match + " WITH DISTINCT n LIMIT 2000 DETACH DELETE n RETURN count(*) AS n",
                  {"place": place})[0]["n"]
            total += n
            if n == 0:
                break
        print(f"  刪除 {label} {total}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", default="", help="店名（Neo4j place）；也決定工作資料夾 <MAP_BUILD_ROOT>/<店名>")
    ap.add_argument("--workspace", type=Path, default=None,
                    help="建圖工作資料夾（預設依 --place）；給了就可以不給 --place")
    ap.add_argument("--dry-run", action="store_true", help="只顯示會寫入什麼，不寫 Neo4j")
    ap.add_argument("--replace", action="store_true", help="place 已存在時先刪掉再建（只限本工具建的 place）")
    args = ap.parse_args()

    if not args.place and not args.workspace:
        places = list_places(MAP_BUILD_ROOT)
        print("請用 --place 指定店名。目前有建圖資料的店：")
        for p in places:
            print(f"  {p['place']}（{p['nodes']} 個節點{'，已確認' if p['confirmed'] else '，還沒按確認並儲存'}）")
        return 1
    ws: Path = args.workspace or workspace_for(args.place, MAP_BUILD_ROOT)
    confirmed_path = ws / "topomap_confirmed.json"
    if not confirmed_path.exists():
        print(f"找不到 {confirmed_path}：先在 /topomap-builder 按「確認並儲存」")
        return 1
    confirmed = json.loads(confirmed_path.read_text(encoding="utf-8"))
    topo_path = ws / "topomap.json"
    topomap = json.loads(topo_path.read_text(encoding="utf-8")) if topo_path.exists() else None
    place = args.place or confirmed.get("place") or (topomap or {}).get("place") or ""
    if not place:
        print("沒有 place 名稱：請加 --place")
        return 1

    rows = upload_rows(confirmed, topomap, place)
    missing = [p["photo_file"] for p in rows["photos"]
               if not (ws / "/".join(photo_local_path(p["photo_file"]))).is_file()]
    print(f"place「{place}」：Waypoint {len(rows['waypoints'])}、DirPhoto {len(rows['photos'])}、"
          f"WALKWAY {len(rows['walkways'])}")
    if missing:
        print(f"  ⚠ 工作資料夾裡缺 {len(missing)} 張照片（之後 rebuild_objects_v2.py 會跳過）：{missing[:3]}")
    isolated = {w["nid"] for w in rows["waypoints"]} - {e[k] for e in rows["walkways"] for k in ("from", "to")}
    if isolated:
        print(f"  ⚠ 沒有任何連線的節點：{sorted(isolated)}")

    client = Neo4jClient()
    if not client.connect():
        print("連不上 Neo4j（檢查 .env 的 NEO4J_URI / NEO4J_PASSWORD）")
        return 1
    q = client._query
    state = existing_state(q, place)
    if state["waypoints"]:
        print(f"Neo4j 已有這個 place：Waypoint {state['waypoints']}、DirPhoto {state['photos']}")
        if state["foreign_photos"]:
            print("  這個 place 不是本工具建的（DirPhoto 沒有 place 欄位），拒絕覆蓋。請換一個 place 名稱。")
            return 1
        if not args.replace:
            print("  要覆蓋請加 --replace")
            return 1

    if args.dry_run:
        print("\n--dry-run：沒有寫入。範例資料：")
        for key in ("waypoints", "photos", "walkways"):
            if rows[key]:
                print(f"  {key}: {json.dumps(rows[key][0], ensure_ascii=False)}")
        return 0

    if state["waypoints"]:
        print("刪除舊資料…")
        delete_place(q, place)

    for chunk in batches(rows["waypoints"]):
        q("UNWIND $rows AS r CREATE (w:Waypoint) SET w = r", {"rows": chunk})
    for chunk in batches(rows["photos"]):
        q("""
            UNWIND $rows AS r
            MATCH (w:Waypoint {place: $place, nid: r.nid})
            CREATE (w)-[:HAS_PHOTO {slot: r.slot}]->
                   (:DirPhoto {photo_file: r.photo_file, heading_deg: r.heading_deg, place: $place})
        """, {"rows": chunk, "place": place})
    for chunk in batches(rows["walkways"]):
        q("""
            UNWIND $rows AS r
            MATCH (a:Waypoint {place: $place, nid: r.from}), (b:Waypoint {place: $place, nid: r.to})
            CREATE (a)-[:WALKWAY {distance_m: r.distance_m, steps: 0, direction: ''}]->(b)
        """, {"rows": chunk, "place": place})

    check = q("""
        MATCH (w:Waypoint {place: $place})
        OPTIONAL MATCH (w)-[:HAS_PHOTO]->(p:DirPhoto)
        WITH count(DISTINCT w) AS w, count(DISTINCT p) AS p
        OPTIONAL MATCH (:Waypoint {place: $place})-[e:WALKWAY]->(:Waypoint)
        RETURN w, p, count(e) AS e
    """, {"place": place})[0]
    ok = (check["w"], check["p"], check["e"]) == (
        len(rows["waypoints"]), len(rows["photos"]), len(rows["walkways"]))
    print(f"已寫入：Waypoint {check['w']}、DirPhoto {check['p']}、WALKWAY {check['e']}"
          + ("" if ok else "  ⚠ 和預期筆數不同"))
    print("下一步：python rebuild_objects_v2.py --place \"%s\"" % place)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
