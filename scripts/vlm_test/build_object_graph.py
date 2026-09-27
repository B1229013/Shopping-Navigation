"""Rebuild Neo4j Object nodes with grid_cells + ADJACENT relationships.

Deletes all existing Object nodes and DETECTED/ADJACENT relationships,
re-runs VLM perception on every reference photo, and writes new Object
nodes with grid_cells/piece_count/depth_class + ADJACENT relationships
between objects in the same DirPhoto — all in one pass.

Waypoint / DirPhoto / HAS_PHOTO / WALKWAY are NOT touched.

Usage:
    export NEO4J_HOST=... NEO4J_PASSWORD=... OPENAI_API_KEY=...
    python build_object_graph.py [--dry-run] [--vlm-calls 2] [--start-from 0]
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import os
import requests
from PIL import Image, ImageOps

# ── Load .env from project root ─────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_dotenv = _PROJECT_ROOT / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.split("#")[0].strip()
            os.environ.setdefault(k.strip(), v)

# ── Config ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://air.cgu.edu.tw/cgullmapi/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

_host = os.environ.get("NEO4J_HOST", "")
if not _host:
    _uri = os.environ.get("NEO4J_URI", "")
    if "://" in _uri:
        _host = _uri.split("://", 1)[1].rstrip("/")
        if ":" in _host:
            _host = _host.split(":")[0]

_neo4j_user = os.environ.get("NEO4J_USER", "") or (_host.split('.')[0] if _host else "")
_neo4j_pass = os.environ.get("NEO4J_PASSWORD", "")
_neo4j_db = os.environ.get("NEO4J_DATABASE", _neo4j_user)
NEO4J_ENDPOINT = f"https://{_host}/db/{_neo4j_db}/query/v2"
NEO4J_AUTH = "Basic " + base64.b64encode(f"{_neo4j_user}:{_neo4j_pass}".encode()).decode()
PLACE = os.environ.get("NEO4J_PLACE", os.environ.get("PLACE", "A7家樂福（9/16)"))

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/Users/shingchou/Downloads/學校家樂福/0916測試"))

# ── Grid computation (inlined from server/grid_utils.py) ────────────────
ROWS = ["top", "mid", "bottom"]
COLS = ["left", "center", "right"]
GRID_SIZE = 3

CELL_COORDS = {
    f"{r}-{c}": (ri, ci)
    for ri, r in enumerate(ROWS)
    for ci, c in enumerate(COLS)
}

_DIR_NAMES = {
    (-1, -1): "above-left",
    (-1,  0): "above",
    (-1,  1): "above-right",
    ( 0, -1): "left",
    ( 0,  0): "overlap",
    ( 0,  1): "right",
    ( 1, -1): "below-left",
    ( 1,  0): "below",
    ( 1,  1): "below-right",
}


def compute_grid_cells(bbox):
    x1, y1, x2, y2 = bbox
    col_start = min(int(x1 * GRID_SIZE), GRID_SIZE - 1)
    col_end = min(int(x2 * GRID_SIZE), GRID_SIZE - 1)
    row_start = min(int(y1 * GRID_SIZE), GRID_SIZE - 1)
    row_end = min(int(y2 * GRID_SIZE), GRID_SIZE - 1)

    cells = []
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            cells.append(f"{ROWS[r]}-{COLS[c]}")

    cx_col = min(int(((x1 + x2) / 2) * GRID_SIZE), GRID_SIZE - 1)
    cx_row = min(int(((y1 + y2) / 2) * GRID_SIZE), GRID_SIZE - 1)
    grid_cell = f"{ROWS[cx_row]}-{COLS[cx_col]}"

    piece_count = len(cells)
    if piece_count == 1:
        depth_class = "far"
    elif piece_count <= 3:
        depth_class = "mid"
    else:
        depth_class = "near"

    return grid_cell, cells, piece_count, depth_class


def compute_cell_pairs(cells_a, cells_b):
    pairs = []
    for ca in cells_a:
        ra, col_a = CELL_COORDS.get(ca, (None, None))
        if ra is None:
            continue
        for cb in cells_b:
            rb, col_b = CELL_COORDS.get(cb, (None, None))
            if rb is None:
                continue
            dr = rb - ra
            dc = col_b - col_a
            if abs(dr) <= 1 and abs(dc) <= 1:
                pairs.append({"a": ca, "b": cb, "dir": _DIR_NAMES.get((dr, dc), "unknown")})
    return pairs


def depth_diff(da, db):
    order = {"far": 0, "mid": 1, "near": 2}
    va, vb = order.get(da, 1), order.get(db, 1)
    if va == vb:
        return "same_level"
    return "closer" if vb > va else "farther"


# ── Neo4j helpers ───────────────────────────────────────────────────────

def neo4j_query(cypher, params=None):
    resp = requests.post(
        NEO4J_ENDPOINT,
        json={"statement": cypher, "parameters": params or {}},
        headers={
            "Authorization": NEO4J_AUTH,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    errors = payload.get("errors", [])
    if errors:
        raise RuntimeError(f"Neo4j error: {errors}")
    data = payload.get("data", {})
    fields = data.get("fields", [])
    values = data.get("values", [])
    return [dict(zip(fields, row)) for row in values]


# ── VLM perception ──────────────────────────────────────────────────────

PERCEIVE_PROMPT = """\
你正在分析一張超市內部的照片，用來建立室內定位用的物件地圖。目標是盡可能完整地記錄照片中**所有可辨識的物件**，數量越多越好。

仔細觀察照片的每一個角落，回報你**確實看到**的**所有**物品。回覆格式為一個 JSON 物件，不要加其他文字：

{{"scene": "<用繁體中文描述這是什麼區域>", "items": [{{"zh": "<繁體中文名稱>", "en": "<English name>", "conf": <0.0-1.0 信心值>, "box": [x1,y1,x2,y2]}}]}}

嚴格規則：
- box 座標是圖片的比例值（0.0 到 1.0），格式 [左, 上, 右, 下]。框要緊密包住物件。
- 禁止大框：單個框的寬度和高度都不應超過圖片的50%。
- 只回報你**確實看到**的具體物件，不要猜測或編造。

必須回報的物件種類（全部都要找）：
1. 設備：冷藏櫃、冷凍櫃、開放式冰櫃、貨架、促銷堆頭、購物車、磅秤、收銀機、推車
2. 標示牌/吊牌：走道編號吊牌（如「走道5」）、區域分類標示（如「堅果海苔」「飲料」）、品牌招牌
3. 商品類別：每個貨架上可見的商品類別要分開報（零食貨架、飲料貨架、日用品貨架等）
4. 獨立商品/品牌：可辨識的具體品牌或商品（如「可口可樂」「舒潔衛生紙」）
5. 環境地標：柱子、走道交叉口、轉角、出口標示、消防設備

回報數量：**15-25 個**偵測。寧可多報也不要少報。遠處的、角落的、部分遮擋的物件也必須報出。同一個大型設備（如一整排冷藏櫃）如果有明顯不同的分區，要分開報。
"""

REFINE_PROMPT = """\
你正在第二次、更仔細地分析同一張超市照片。目標是把第一次漏掉的物件全部補上。

第一次分析已經找到了這些物件：
{prev_items}

請再看一次照片，**逐區掃描**（左上→右上→左中→右中→左下→右下），重點補充：
1. **漏掉的設備**：有沒有冷藏櫃、推車、磅秤、貨架沒被報出？
2. **漏掉的標示牌**：天花板上的吊牌、走道編號、分類標示、促銷海報
3. **漏掉的商品分區**：同一個貨架上如果有不同商品類別（上層 vs 下層、左側 vs 右側），要分開報
4. **遠處/背景的物件**：照片深處可見的設備、標示、商品
5. **部分遮擋的物件**：只露出一部分但仍可辨識的物件

回覆格式同上，只回報**新發現的**物件（不要重複第一次已報過的）：

{{"scene": "<更新後的區域描述>", "items": [{{"zh": "<繁體中文名稱>", "en": "<English name>", "conf": <0.0-1.0 信心值>, "box": [x1,y1,x2,y2]}}]}}

規則同上：box 是 0-1 比例值，禁止大框，只報確實看到的。目標補上 **10-20 個**新偵測。
"""


def resolve_photo_path(neo4j_path):
    if not neo4j_path:
        return None
    parts = neo4j_path.split("/", 1)
    if len(parts) == 2:
        folder = parts[0].replace("set", "")
        return DATA_ROOT / folder / parts[1]
    return None


def _encode_image(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _vlm_call(img_b64, prompt):
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0.2,
        "max_completion_tokens": 4096,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    for attempt in (1, 2, 3):
        try:
            r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", json=body, headers=headers, timeout=120)
            r.raise_for_status()
            text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"    VLM attempt {attempt} failed: {e}", flush=True)
            if attempt < 3:
                time.sleep(2)
    return {"items": [], "scene": ""}


def vlm_perceive_two_pass(image_path):
    """Two-pass VLM perception: first pass detects, second pass refines."""
    img_b64 = _encode_image(image_path)

    # Pass 1: initial detection
    pass1 = _vlm_call(img_b64, PERCEIVE_PROMPT)
    items1 = pass1.get("items", [])

    # Pass 2: refine — ask VLM to find what it missed
    if items1:
        prev_strs = ", ".join(
            f"{it.get('zh', '')}({it.get('en', '')})" for it in items1
        )
    else:
        prev_strs = "(無)"
    refine_prompt = REFINE_PROMPT.format(prev_items=prev_strs)
    pass2 = _vlm_call(img_b64, refine_prompt)
    items2 = pass2.get("items", [])

    merged_scene = pass2.get("scene") or pass1.get("scene", "")
    return {"scene": merged_scene, "items": items1 + items2}


def normalize_label(zh, en=""):
    return (zh + " " + en).strip().lower()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rebuild Neo4j Object nodes with grid_cells + ADJACENT")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing to Neo4j")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N photos (for resume)")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not _host or not _neo4j_pass:
        print("ERROR: NEO4J_HOST/NEO4J_URI or NEO4J_PASSWORD not set", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: Load all DirPhotos ──
    print("Loading DirPhoto nodes from Neo4j...", flush=True)
    rows = neo4j_query("""
        MATCH (w:Waypoint {place: $place})-[hp:HAS_PHOTO]->(p:DirPhoto)
        RETURN w.nid AS nid, hp.slot AS slot, p.photo_file AS photo_file,
               elementId(p) AS photo_id
        ORDER BY w.nid, hp.slot
    """, {"place": PLACE})

    photos = []
    for r in rows:
        fpath = resolve_photo_path(r["photo_file"])
        if fpath and fpath.exists():
            photos.append({
                "nid": r["nid"],
                "slot": r["slot"],
                "photo_file": r["photo_file"],
                "photo_id": r["photo_id"],
                "fpath": fpath,
            })
        else:
            print(f"  WARN: photo not found: {r['photo_file']}", flush=True)

    vlm_calls_per_photo = 2  # two-pass (perceive + refine) = 2 calls
    print(f"  {len(photos)} photos found (×{vlm_calls_per_photo} VLM calls each, two-pass)", flush=True)
    print(f"  Estimated time: ~{len(photos) * vlm_calls_per_photo * 5 / 60:.0f} minutes", flush=True)

    if args.dry_run:
        print("\n[DRY RUN] Would delete all Object/DETECTED/ADJACENT and rebuild.", flush=True)
        print(f"  Photos to process: {len(photos)}", flush=True)
        return

    # ── Step 2: Delete existing Objects + relationships ──
    if args.start_from == 0:
        print("\nDeleting existing Object nodes + DETECTED + ADJACENT...", flush=True)
        result = neo4j_query("""
            MATCH (w:Waypoint {place: $place})-[:HAS_PHOTO]->(p:DirPhoto)-[:DETECTED]->(o:Object)
            OPTIONAL MATCH (o)-[adj:ADJACENT]-()
            WITH o, collect(adj) AS adjs
            FOREACH (a IN adjs | DELETE a)
            DETACH DELETE o
            RETURN count(o) AS deleted
        """, {"place": PLACE})
        deleted = result[0]["deleted"] if result else 0
        print(f"  Deleted {deleted} Object nodes", flush=True)
    else:
        print(f"\nResuming from photo index {args.start_from}, skipping delete.", flush=True)

    # ── Step 3: Process each photo ──
    total_objects = 0
    total_adjacent = 0
    total_calls = 0
    t_start = time.time()

    for i, p in enumerate(photos):
        if i < args.start_from:
            continue

        # ── 3a: VLM two-pass perception (perceive + refine) ──
        all_items = []
        seen_norms = set()

        total_calls += 2
        perc = vlm_perceive_two_pass(p["fpath"])

        for it in perc.get("items", []):
            box = it.get("box")
            if not box or len(box) != 4:
                continue
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            if bw > 0.55 or bh > 0.55 or bw < 0.01 or bh < 0.01:
                continue

            zh = it.get("zh", "")
            en = it.get("en", "")
            label_norm = normalize_label(zh, en)
            if not label_norm or len(label_norm) < 2:
                continue

            grid_cell, grid_cells, piece_count, depth_class = compute_grid_cells(box)

            dedup_key = (label_norm, grid_cell)
            if dedup_key in seen_norms:
                continue
            seen_norms.add(dedup_key)

            all_items.append({
                "zh": zh,
                "en": en,
                "label": f"{zh} {en}".strip(),
                "label_norm": label_norm,
                "conf": it.get("conf", 0.5),
                "box": [round(b, 4) for b in box],
                "grid_cell": grid_cell,
                "grid_cells": grid_cells,
                "piece_count": piece_count,
                "depth_class": depth_class,
            })

        # ── 3b: Batch write Object nodes (one UNWIND per photo) ──
        obj_id_map = {}
        if all_items:
            batch_items = []
            for idx, item in enumerate(all_items):
                batch_items.append({
                    "idx": idx,
                    "label": item["label"],
                    "label_norm": item["label_norm"],
                    "score": item["conf"],
                    "grid_cell": item["grid_cell"],
                    "grid_cells": json.dumps(item["grid_cells"], ensure_ascii=False),
                    "piece_count": item["piece_count"],
                    "depth_class": item["depth_class"],
                    "bbox": json.dumps(item["box"]),
                })
            try:
                result = neo4j_query("""
                    MATCH (p:DirPhoto) WHERE elementId(p) = $pid
                    UNWIND $items AS item
                    CREATE (o:Object {
                        label: item.label,
                        label_norm: item.label_norm,
                        score: item.score,
                        grid_cell: item.grid_cell,
                        grid_cells: item.grid_cells,
                        piece_count: item.piece_count,
                        depth_class: item.depth_class,
                        bbox: item.bbox,
                        source: 'rebuild'
                    })
                    CREATE (p)-[:DETECTED]->(o)
                    RETURN elementId(o) AS obj_id, item.idx AS idx
                """, {"pid": p["photo_id"], "items": batch_items})
                for row in result:
                    obj_id_map[row["idx"]] = row["obj_id"]
                total_objects += len(obj_id_map)
            except Exception as e:
                print(f"    Error batch writing objects: {e}", flush=True)

        # ── 3c: Batch build ADJACENT relationships (one UNWIND per photo) ──
        photo_adj = 0
        if len(obj_id_map) >= 2:
            adj_list = []
            for idx_a, idx_b in combinations(range(len(all_items)), 2):
                if idx_a not in obj_id_map or idx_b not in obj_id_map:
                    continue
                pairs = compute_cell_pairs(all_items[idx_a]["grid_cells"], all_items[idx_b]["grid_cells"])
                if not pairs:
                    continue
                adj_list.append({
                    "id_a": obj_id_map[idx_a],
                    "id_b": obj_id_map[idx_b],
                    "cell_pairs": json.dumps(pairs, ensure_ascii=False),
                    "overlap": any(pr["dir"] == "overlap" for pr in pairs),
                    "vert": sum(1 for pr in pairs if pr["dir"] in ("above", "below")),
                    "horiz": sum(1 for pr in pairs if pr["dir"] in ("left", "right")),
                    "diag": sum(1 for pr in pairs if pr["dir"] in ("above-left", "above-right", "below-left", "below-right")),
                    "depth_diff": depth_diff(all_items[idx_a]["depth_class"], all_items[idx_b]["depth_class"]),
                })

            ADJ_CHUNK = 200
            for chunk_start in range(0, len(adj_list), ADJ_CHUNK):
                chunk = adj_list[chunk_start:chunk_start + ADJ_CHUNK]
                try:
                    neo4j_query("""
                        UNWIND $rels AS rel
                        MATCH (a:Object) WHERE elementId(a) = rel.id_a
                        MATCH (b:Object) WHERE elementId(b) = rel.id_b
                        CREATE (a)-[:ADJACENT {
                            cell_pairs: rel.cell_pairs,
                            overlap: rel.overlap,
                            vertical_count: rel.vert,
                            horizontal_count: rel.horiz,
                            diagonal_count: rel.diag,
                            depth_diff: rel.depth_diff
                        }]->(b)
                    """, {"rels": chunk})
                    photo_adj += len(chunk)
                    total_adjacent += len(chunk)
                except Exception as e:
                    print(f"    Error batch writing ADJACENT (chunk {chunk_start}): {e}", flush=True)

        elapsed = time.time() - t_start
        processed = i - args.start_from + 1
        avg_per = elapsed / processed
        remaining = avg_per * (len(photos) - i - 1)

        print(
            f"  [{i+1:3d}/{len(photos)}] WP{p['nid']}/{p['slot']} | "
            f"+{len(all_items)} obj, +{photo_adj} adj | "
            f"total: {total_objects} obj, {total_adjacent} adj | "
            f"ETA {remaining/60:.0f}m",
            flush=True,
        )

    # ── Step 4: Summary ──
    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"Done in {elapsed_total:.0f}s ({total_calls} VLM calls)", flush=True)
    print(f"  Objects created: {total_objects}", flush=True)
    print(f"  ADJACENT relationships: {total_adjacent}", flush=True)

    verify = neo4j_query("""
        MATCH (w:Waypoint {place: $place})-[:HAS_PHOTO]->(p:DirPhoto)-[:DETECTED]->(o:Object)
        RETURN count(o) AS total_objects
    """, {"place": PLACE})
    print(f"  Objects in DB: {verify[0]['total_objects']}", flush=True)

    verify_adj = neo4j_query("""
        MATCH (w:Waypoint {place: $place})-[:HAS_PHOTO]->(p:DirPhoto)-[:DETECTED]->(o:Object)
        MATCH (o)-[r:ADJACENT]->()
        RETURN count(r) AS total_adj
    """, {"place": PLACE})
    print(f"  ADJACENT in DB: {verify_adj[0]['total_adj']}", flush=True)

    verify_vert = neo4j_query("""
        MATCH (w:Waypoint {place: $place})-[:HAS_PHOTO]->(p:DirPhoto)-[:DETECTED]->(o:Object)
        MATCH (o)-[r:ADJACENT]->() WHERE r.vertical_count > 0
        RETURN count(r) AS vert_adj
    """, {"place": PLACE})
    print(f"  Vertical ADJACENT: {verify_vert[0]['vert_adj']}", flush=True)


if __name__ == "__main__":
    main()
