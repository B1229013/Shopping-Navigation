"""Store the DINOv2 vector of every DirPhoto of a place — in the sidecar and in Neo4j.

Server-side localization compares the uploaded photo's DINOv2 vector with the reference
photos' (server/locate_fusion.server_image_evidence). The reference vectors are read from
  • the sidecar file DINO_SIDECAR_PATH (backend/dino_sidecar.pkl, photo file name → vector)
  • DirPhoto.dino in Neo4j (list of 768 floats, plus dino_model) — shared by every server
    that connects to the database, e.g. a cloud deployment that has no sidecar file.
This script writes both. Vectors come from the store's workspace cache
(<workspace>/dino_embeddings.pkl, made by the builder's 「分析照片」) or are computed now
(facebook/dinov2-base, the same model the server embeds the uploaded photo with).

Usage (from backend/):
    python store_dino_v2.py --place 系辦 --dry-run
    python store_dino_v2.py --place 系辦
    python store_dino_v2.py --place 系辦 --no-neo4j        # sidecar only

Only places this project uploaded (DirPhoto.place set) are written to Neo4j; the classmate's
A7 nodes are never modified.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import config  # noqa: E402
from server.map_build import workspace_for  # noqa: E402
from server.map_v2 import photo_local_path  # noqa: E402
from server.neo4j_client import Neo4jClient  # noqa: E402

DINO_MODEL = "facebook/dinov2-base"
BATCH = 40


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", required=True)
    ap.add_argument("--workspace", type=Path, default=None,
                    help=f"建圖工作資料夾（預設 {config.MAP_BUILD_ROOT}/<店名>）")
    ap.add_argument("--sidecar", type=Path, default=Path(config.DINO_SIDECAR_PATH))
    ap.add_argument("--no-sidecar", action="store_true", help="不寫 sidecar")
    ap.add_argument("--no-neo4j", action="store_true", help="不寫 Neo4j")
    ap.add_argument("--dry-run", action="store_true", help="只檢查，不寫任何東西")
    args = ap.parse_args()
    ws = args.workspace or workspace_for(args.place, config.MAP_BUILD_ROOT)

    client = Neo4jClient()
    if not client.connect():
        print("連不上 Neo4j（檢查 .env 的 NEO4J_URI / NEO4J_PASSWORD）")
        return 1
    q = client._query
    photos = q("""
        MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(p:DirPhoto)
        RETURN DISTINCT elementId(p) AS pid, p.photo_file AS photo_file, p.place AS owner
    """, {"place": args.place})
    if not photos:
        print(f"Neo4j 沒有 place「{args.place}」的 DirPhoto：先跑 upload_map_v2.py")
        return 1
    if any(p["owner"] != args.place for p in photos) and not args.no_neo4j:
        print("這個 place 不是 upload_map_v2.py 建的（例如同學的 A7），不寫 Neo4j；可加 --no-neo4j 只寫 sidecar")
        return 1

    # vectors: workspace cache ("01/x.jpg") first, else embed the photo now
    cache_path = ws / "dino_embeddings.pkl"
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    vectors, todo, missing = {}, [], []
    for p in photos:
        folder, name = photo_local_path(p["photo_file"])
        key = f"{folder}/{name}"
        if key in cache:
            vectors[p["photo_file"]] = cache[key]
        elif (ws / folder / name).is_file():
            todo.append((p["photo_file"], key, ws / folder / name))
        else:
            missing.append(p["photo_file"])
    print(f"place「{args.place}」：{len(photos)} 張照片；快取已有 {len(vectors)}、要計算 {len(todo)}、"
          f"找不到檔案 {len(missing)}")
    if missing:
        print(f"  ⚠ 找不到照片（略過）：{missing[:3]}")
    if args.dry_run:
        print("--dry-run：沒有寫入")
        return 0
    if todo:
        from server import dino_embed
        if not dino_embed.available():
            print("DINOv2 無法載入（需要 torch + transformers）")
            return 1
        t0 = time.time()
        for photo_file, key, path in todo:
            v = dino_embed.embed_image(str(path))
            if v is not None:
                vectors[photo_file] = cache[key] = v
        cache_path.write_bytes(pickle.dumps(cache))
        print(f"  計算 {len(todo)} 張，{time.time() - t0:.0f} 秒")

    if not args.no_sidecar:
        sidecar = {}
        if args.sidecar.is_file():
            sidecar = pickle.loads(args.sidecar.read_bytes())
            backup = args.sidecar.with_name(f"{args.sidecar.stem}.bak-{time.strftime('%Y%m%d-%H%M%S')}{args.sidecar.suffix}")
            shutil.copy2(args.sidecar, backup)
            print(f"  sidecar 備份：{backup.name}")
        before = len(sidecar)
        for photo_file, v in vectors.items():
            sidecar[Path(photo_file).name] = v           # keyed by file name, like the A7 entries
        args.sidecar.write_bytes(pickle.dumps(sidecar))
        print(f"  sidecar：{before} → {len(sidecar)} 筆（{args.sidecar}）")

    if not args.no_neo4j:
        rows = [{"pid": p["pid"], "v": [float(x) for x in vectors[p["photo_file"]]]}
                for p in photos if p["photo_file"] in vectors]
        for i in range(0, len(rows), BATCH):
            q("""
                UNWIND $rows AS r
                MATCH (p:DirPhoto) WHERE elementId(p) = r.pid AND p.place = $place
                SET p.dino = r.v, p.dino_model = $model
            """, {"rows": rows[i:i + BATCH], "place": args.place, "model": DINO_MODEL})
        n = q("""
            MATCH (:Waypoint {place: $place})-[:HAS_PHOTO]->(p:DirPhoto)
            WHERE p.dino IS NOT NULL RETURN count(DISTINCT p) AS n, min(size(p.dino)) AS dim
        """, {"place": args.place})[0]
        print(f"  Neo4j：{n['n']}/{len(photos)} 個 DirPhoto 有 dino（{n['dim']} 維）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
