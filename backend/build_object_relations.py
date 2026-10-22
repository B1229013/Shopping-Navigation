"""Write the 上下左右 relations between the objects of each reference photo into Neo4j.

    (a:Object)-[:ABOVE]->(b:Object)     a 在 b 上面   （同時寫 (b)-[:BELOW]->(a)）
    (a:Object)-[:LEFT_OF]->(b:Object)   a 在 b 左邊   （同時寫 (b)-[:RIGHT_OF]->(a)）

Only between objects of the same DirPhoto, each object with its NEIGHBOURS nearest ones;
the relation itself comes from server/object_relations.pair_relation — the same function
localization uses (LOC_RELATION_WEIGHT), so the graph and the localizer agree. Each edge
carries dx / dy (box-centre difference, 0..1 of the photo) and source="object_relations".
The map's older ADJACENT edges (map_v2, 3×3 grid cells) are left as they are.

Usage (from backend/):
    python build_object_relations.py --place 系辦 --dry-run     # counts only, writes nothing
    python build_object_relations.py --place 系辦               # (re)write the edges
    python build_object_relations.py --place 系辦 --delete      # remove them again

Like store_dino_v2.py, only places this project uploaded (DirPhoto.place set) are written;
the classmate's A7 map is read-only unless --allow-foreign is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.neo4j_client import Neo4jClient  # noqa: E402
from server.object_relations import NEIGHBOURS, neighbour_pairs, pair_relation  # noqa: E402

TYPES = ("ABOVE", "BELOW", "LEFT_OF", "RIGHT_OF")
BATCH = 500


def relation_rows(objects):
    """objects: [{"oid", "bbox"}] of one photo → [{"a", "b", "type", "dx", "dy"}]."""
    boxes = [o["bbox"] for o in objects]
    rows = []
    for i, j in neighbour_pairs(boxes):
        rel = pair_relation(boxes[i], boxes[j])
        a, b = objects[i], objects[j]
        dx = round(((b["bbox"][0] + b["bbox"][2]) - (a["bbox"][0] + a["bbox"][2])) / 2, 3)
        dy = round(((b["bbox"][1] + b["bbox"][3]) - (a["bbox"][1] + a["bbox"][3])) / 2, 3)
        if rel.v:
            up, down = (a, b) if rel.v < 0 else (b, a)
            d = abs(dy)
            rows += [{"a": up["oid"], "b": down["oid"], "type": "ABOVE", "dx": abs(dx), "dy": d},
                     {"a": down["oid"], "b": up["oid"], "type": "BELOW", "dx": abs(dx), "dy": d}]
        if rel.h:
            left, right = (a, b) if rel.h < 0 else (b, a)
            d = abs(dx)
            rows += [{"a": left["oid"], "b": right["oid"], "type": "LEFT_OF", "dx": d, "dy": abs(dy)},
                     {"a": right["oid"], "b": left["oid"], "type": "RIGHT_OF", "dx": d, "dy": abs(dy)}]
    return rows


def _bbox(raw):
    try:
        b = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    return b if isinstance(b, list) and len(b) == 4 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", required=True)
    ap.add_argument("--dry-run", action="store_true", help="只算數量，不寫任何東西")
    ap.add_argument("--delete", action="store_true", help="刪掉這個 place 的 ABOVE/BELOW/LEFT_OF/RIGHT_OF")
    ap.add_argument("--allow-foreign", action="store_true",
                    help="也寫不是本專案上傳的 place（例如同學的 A7）")
    args = ap.parse_args()

    client = Neo4jClient()
    if not client.connect():
        print("連不上 Neo4j（檢查 .env 的 NEO4J_URI / NEO4J_PASSWORD）")
        return 1
    q = client._query
    rows = q("""
        MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(p:DirPhoto)-[:DETECTED]->(o:Object)
        RETURN elementId(p) AS pid, p.place AS owner, elementId(o) AS oid, o.bbox AS bbox
    """, {"place": args.place})
    if not rows:
        print(f"Neo4j 沒有 place「{args.place}」的物件")
        return 1
    foreign = any(r["owner"] != args.place for r in rows)
    existing = q("""
        MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(:DirPhoto)-[:DETECTED]->(:Object)-[r]->(:Object)
        WHERE type(r) IN $types RETURN type(r) AS t, count(r) AS n
    """, {"place": args.place, "types": list(TYPES)})
    print(f"place「{args.place}」：{len({r['pid'] for r in rows})} 張照片、{len(rows)} 個物件；"
          f"已有關係邊 {({r['t']: r['n'] for r in existing}) or 0}")

    if args.delete:
        if args.dry_run:
            print("--dry-run：沒有刪除")
            return 0
        if foreign and not args.allow_foreign:
            print("這個 place 不是本專案上傳的（例如同學的 A7），不動；要動請加 --allow-foreign")
            return 1
        n = q("""
            MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(:DirPhoto)-[:DETECTED]->(:Object)-[r]->(:Object)
            WHERE type(r) IN $types AND r.source = 'object_relations'
            DELETE r RETURN count(r) AS n
        """, {"place": args.place, "types": list(TYPES)})[0]["n"]
        print(f"刪掉 {n} 條")
        return 0

    by_photo = defaultdict(list)
    for r in rows:
        b = _bbox(r["bbox"])
        if b:
            by_photo[r["pid"]].append({"oid": r["oid"], "bbox": b})
    edges = [e for objs in by_photo.values() for e in relation_rows(objs)]
    counts = defaultdict(int)
    for e in edges:
        counts[e["type"]] += 1
    print(f"要寫 {len(edges)} 條（每個物件連最近 {NEIGHBOURS} 個）：{dict(counts)}")
    if args.dry_run:
        print("--dry-run：沒有寫入")
        return 0
    if foreign and not args.allow_foreign:
        print("這個 place 不是本專案上傳的（例如同學的 A7），不寫；要寫請加 --allow-foreign")
        return 1

    # rewrite: drop this script's old edges first so a changed rule leaves no stale ones
    q("""
        MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(:DirPhoto)-[:DETECTED]->(:Object)-[r]->(:Object)
        WHERE type(r) IN $types AND r.source = 'object_relations'
        DELETE r
    """, {"place": args.place, "types": list(TYPES)})
    for t in TYPES:
        part = [e for e in edges if e["type"] == t]
        for i in range(0, len(part), BATCH):
            q(f"""
                UNWIND $rows AS e
                MATCH (a:Object) WHERE elementId(a) = e.a
                MATCH (b:Object) WHERE elementId(b) = e.b
                CREATE (a)-[:{t} {{dx: e.dx, dy: e.dy, source: 'object_relations'}}]->(b)
            """, {"rows": part[i:i + BATCH]})
    n = q("""
        MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(:DirPhoto)-[:DETECTED]->(:Object)-[r]->(:Object)
        WHERE type(r) IN $types RETURN type(r) AS t, count(r) AS n
    """, {"place": args.place, "types": list(TYPES)})
    print(f"寫入完成：{ {r['t']: r['n'] for r in n} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
