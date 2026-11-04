"""Detect objects on every DirPhoto of a place and write them in the shared V2 schema.

Step 4 of building a new store map (server/map_v2.py): for each DirPhoto the VLM is asked
--calls times with the classmate's prompt (results merged, near-duplicates dropped), EasyOCR
text is attached to the object it sits on, and Object / DETECTED / ADJACENT are created —
the same fields as the A7 map (bbox, grid_cell(s), piece_count, depth_class, source "rebuild").

Every VLM / OCR result is cached in <workspace>/vlm_cache/, so an interrupted run resumes
without paying again; photos that already have objects are skipped unless --redo.

Usage (from backend/):
    python rebuild_objects_v2.py --place "全聯 XX店" --limit 2 --dry-run   # try 2 photos, no writes
    python rebuild_objects_v2.py --place "全聯 XX店"
    python rebuild_objects_v2.py --place "全聯 XX店" --redo --calls 3

VLM endpoint / key / model: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL in .env
(--model overrides; the A7 map used gpt-5.4-mini through the CGU gateway).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import config  # noqa: E402
from server.map_v2 import (  # noqa: E402
    PERCEIVE_PROMPT, adjacent_pairs, attach_ocr, merge_vlm_calls, object_props,
    parse_vlm_items, photo_local_path,
)
from server.neo4j_client import Neo4jClient  # noqa: E402

VLM_MAX_SIDE = 1024
OCR_MAX_SIDE = 2048


def load_image(path: Path, max_side: int) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def vlm_call(img: Image.Image, model: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": PERCEIVE_PROMPT},
        ]}],
        "temperature": 0.2,
        "max_completion_tokens": 4096,
    }
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}
    last = None
    for attempt in range(3):
        try:
            r = requests.post(f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                              json=body, headers=headers, timeout=120)
            if r.status_code == 400 and "temperature" in r.text and "temperature" in body:
                body.pop("temperature")      # some models only accept the default temperature
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") or ""
        except Exception as e:  # network / 5xx / malformed reply → retry
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"VLM 呼叫失敗：{last}")


class OcrReader:
    """EasyOCR on the exif-corrected image, boxes normalized to [x1,y1,x2,y2] in 0..1."""

    def __init__(self, enabled: bool):
        self._ocr = None
        if enabled:
            try:
                from server.ocr import OCR
                self._ocr = OCR(languages=config.OCR_LANGUAGES)
                self._ocr.load()
            except Exception as e:
                print(f"⚠ EasyOCR 無法載入（{e}），ocr_text 全部留空；可加 --no-ocr 略過")
                self._ocr = None

    @property
    def enabled(self) -> bool:
        return self._ocr is not None

    def read(self, path: Path, tmp: Path) -> list:
        img = load_image(path, OCR_MAX_SIDE)
        img.save(tmp, format="JPEG", quality=92)
        w, h = img.size
        out = []
        for r in self._ocr.read(str(tmp)):
            xs = [p[0] for p in r.bbox]
            ys = [p[1] for p in r.bbox]
            out.append([r.text, [min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h]])
        tmp.unlink(missing_ok=True)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", required=True)
    ap.add_argument("--workspace", type=Path, default=None,
                    help=f"建圖工作資料夾（預設 {config.MAP_BUILD_ROOT}/<店名>）")
    ap.add_argument("--calls", type=int, default=3, help="每張照片呼叫 VLM 幾次再合併（A7 是 1+2 次）")
    ap.add_argument("--model", default=config.OPENAI_MODEL)
    ap.add_argument("--redo", action="store_true", help="已有物件的照片也重做（先刪掉該照片的舊物件）")
    ap.add_argument("--no-ocr", action="store_true", help="不跑 EasyOCR，ocr_text 留空")
    ap.add_argument("--no-adjacent", action="store_true", help="不建 ADJACENT（後端目前沒用到它）")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 張（試跑用）")
    ap.add_argument("--dry-run", action="store_true", help="照樣跑 VLM / OCR 並寫快取，但不寫 Neo4j")
    args = ap.parse_args()
    if args.workspace is None:
        from server.map_build import workspace_for
        args.workspace = workspace_for(args.place, config.MAP_BUILD_ROOT)

    if not config.OPENAI_API_KEY:
        print("沒有 OPENAI_API_KEY（.env）")
        return 1
    client = Neo4jClient()
    if not client.connect():
        print("連不上 Neo4j（檢查 .env 的 NEO4J_URI / NEO4J_PASSWORD）")
        return 1
    q = client._query

    photos = q("""
        MATCH (w:Waypoint {place: $place})-[h:HAS_PHOTO]->(p:DirPhoto)
        OPTIONAL MATCH (p)-[:DETECTED]->(o:Object)
        RETURN elementId(p) AS pid, p.photo_file AS photo_file, p.place AS owner,
               w.nid AS nid, h.slot AS slot, count(o) AS n_obj
        ORDER BY w.nid, h.slot
    """, {"place": args.place})
    if not photos:
        print(f"place「{args.place}」沒有任何 DirPhoto：先跑 upload_map_v2.py")
        return 1
    if any(p["owner"] != args.place for p in photos) and not args.dry_run:
        print("這個 place 不是 upload_map_v2.py 建的（例如同學的 A7），拒絕寫入。")
        return 1

    todo = [p for p in photos if args.redo or p["n_obj"] == 0]
    if args.limit:
        todo = todo[:args.limit]
    print(f"place「{args.place}」：{len(photos)} 張照片，要處理 {len(todo)} 張"
          f"（每張 VLM {args.calls} 次，模型 {args.model}）")
    ocr = OcrReader(enabled=not args.no_ocr)
    cache_root = args.workspace / "vlm_cache"

    t0, n_obj, n_adj, skipped = time.time(), 0, 0, []
    for k, p in enumerate(todo, start=1):
        folder, name = photo_local_path(p["photo_file"])
        path = args.workspace / folder / name
        if not path.is_file():
            skipped.append(p["photo_file"])
            continue
        cache_path = cache_root / folder / (name + ".json")
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        calls = cache.setdefault("vlm", [])
        if len(calls) < args.calls:
            img = load_image(path, VLM_MAX_SIDE)
            while len(calls) < args.calls:
                calls.append(vlm_call(img, args.model))
        if ocr.enabled and "ocr" not in cache:
            cache["ocr"] = ocr.read(path, cache_root / "_ocr_tmp.jpg")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

        items = merge_vlm_calls([parse_vlm_items(t) for t in calls[:args.calls]])
        texts = attach_ocr(items, cache.get("ocr") or [])
        objs = [object_props(it, t) for it, t in zip(items, texts)]
        pairs = [] if args.no_adjacent else adjacent_pairs(objs)
        n_obj += len(objs)
        n_adj += len(pairs)

        if not args.dry_run:
            if p["n_obj"]:
                q("MATCH (p:DirPhoto)-[:DETECTED]->(o:Object) WHERE elementId(p) = $pid DETACH DELETE o",
                  {"pid": p["pid"]})
            created = q("""
                MATCH (p:DirPhoto) WHERE elementId(p) = $pid
                UNWIND range(0, size($objs) - 1) AS i
                CREATE (p)-[:DETECTED]->(o:Object)
                SET o = $objs[i], o.place = $place
                RETURN i, elementId(o) AS oid
            """, {"pid": p["pid"], "objs": objs, "place": args.place}) if objs else []
            oid = {r["i"]: r["oid"] for r in created}
            if pairs:
                q("""
                    UNWIND $pairs AS pr
                    MATCH (a:Object), (b:Object) WHERE elementId(a) = pr.a AND elementId(b) = pr.b
                    CREATE (a)-[r:ADJACENT]->(b) SET r = pr.props
                """, {"pairs": [{"a": oid[x["i"]], "b": oid[x["j"]], "props": x["props"]} for x in pairs]})

        eta = (time.time() - t0) / k * (len(todo) - k)
        print(f"  [{k}/{len(todo)}] WP{p['nid']}/{p['slot']}: {len(objs)} 物件"
              f"（OCR {sum(bool(t) for t in texts)}）、{len(pairs)} ADJACENT · 剩 {eta / 60:.0f} 分")
        if args.dry_run and k <= 2:
            for o in objs[:5]:
                print(f"      {o['label']} | {o['grid_cell']} {o['depth_class']} | ocr={o['ocr_text']!r}")

    print(f"\n完成：{n_obj} 個物件、{n_adj} 條 ADJACENT" + ("（--dry-run，未寫入 Neo4j）" if args.dry_run else ""))
    if skipped:
        print(f"⚠ 工作資料夾裡找不到 {len(skipped)} 張照片，已略過：{skipped[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
