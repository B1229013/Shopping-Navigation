"""Enrich reference DB: call VLM 2 more times per photo, merge new objects into Neo4j."""
import base64, io, json, re, time
from pathlib import Path
from collections import defaultdict

import os, requests
from PIL import Image, ImageOps

# ── Config ──
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://air.cgu.edu.tw/cgullmapi/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

_host = os.environ.get("NEO4J_HOST", "")
_neo4j_user = _host.split('.')[0] if _host else ""
_neo4j_pass = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_ENDPOINT = f"https://{_host}/db/{_neo4j_user}/query/v2"
NEO4J_AUTH = "Basic " + base64.b64encode(f"{_neo4j_user}:{_neo4j_pass}".encode()).decode()
PLACE = os.environ.get("PLACE", "A7家樂福（9/16)")

DATA_ROOT = Path("/Users/shingchou/Downloads/學校家樂福/0916測試")
EXTRA_CALLS = 2  # how many additional VLM calls per photo

PERCEIVE_PROMPT = """\
你正在分析一張超市內部的照片，用來定位使用者的位置。

仔細觀察照片，回報你**確實看到**的**具體**物品。回覆格式為一個 JSON 物件，不要加其他文字：

{{"scene": "<用繁體中文描述這是什麼區域>", "items": [{{"zh": "<繁體中文名稱>", "en": "<English name>", "conf": <0.0-1.0 信心值>, "box": [x1,y1,x2,y2]}}]}}

嚴格規則：
- box 座標是圖片的比例值（0.0 到 1.0），格式 [左, 上, 右, 下]。框要緊密包住物件。
- 禁止大框：寬度和高度都不應超過圖片的50%。
- 只回報你**確實看到**的具體物件，不要猜測。
- 回報具體商品、品牌、設備、標示牌文字、地標。
- 特別注意走道上方的吊牌、區域標示牌（如「走道5」「堅果海苔」等分類標示），這些是重要的定位特徵。
- 標示牌上的文字內容要完整辨識，寫在 zh 欄位。
- 不要回報背景（地板、天花板、牆壁、燈光）。
- 寧可少報也不要報錯。5-12個精確的偵測最好。
"""


def neo4j_query(cypher, params=None):
    resp = requests.post(NEO4J_ENDPOINT, json={"statement": cypher, "parameters": params or {}},
                         headers={"Authorization": NEO4J_AUTH, "Content-Type": "application/json",
                                  "Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", {})
    fields = data.get("fields", [])
    values = data.get("values", [])
    return [dict(zip(fields, row)) for row in values]


def resolve_photo_path(neo4j_path):
    if not neo4j_path:
        return None
    parts = neo4j_path.split("/", 1)
    if len(parts) == 2:
        folder = parts[0].replace("set", "")
        return DATA_ROOT / folder / parts[1]
    return None


def vlm_perceive(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": PERCEIVE_PROMPT},
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
            if attempt < 3:
                time.sleep(2)
    return {"items": [], "scene": ""}


def box_to_grid(box):
    if not box or len(box) != 4:
        return "mid-center"
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    col = "left" if cx < 1/3 else ("right" if cx > 2/3 else "center")
    row = "top" if cy < 1/3 else ("bottom" if cy > 2/3 else "mid")
    return f"{row}-{col}"


def normalize_label(zh, en=""):
    """Simple normalization for dedup."""
    return (zh + " " + en).strip().lower()


def is_duplicate(new_label_norm, new_cell, existing_objects):
    """Check if a label is too similar to an existing one in same/adjacent cell."""
    ADJACENT = {
        "top-left": {"top-center", "mid-left"},
        "top-center": {"top-left", "top-right", "mid-center"},
        "top-right": {"top-center", "mid-right"},
        "mid-left": {"top-left", "mid-center", "bottom-left"},
        "mid-center": {"top-center", "mid-left", "mid-right", "bottom-center"},
        "mid-right": {"top-center", "mid-center", "bottom-right"},
        "bottom-left": {"mid-left", "bottom-center"},
        "bottom-center": {"bottom-left", "bottom-right", "mid-center"},
        "bottom-right": {"mid-right", "bottom-center"},
    }
    nearby_cells = {new_cell} | ADJACENT.get(new_cell, set())

    for obj in existing_objects:
        if obj["cell"] not in nearby_cells:
            continue
        existing_norm = obj["label_norm"]
        # Exact match
        if new_label_norm == existing_norm:
            return True
        # High character overlap
        new_chars = set(new_label_norm) - {' ', '/', '（', '）'}
        ext_chars = set(existing_norm) - {' ', '/', '（', '）'}
        if new_chars and ext_chars:
            overlap = len(new_chars & ext_chars) / max(len(new_chars), len(ext_chars))
            if overlap > 0.7:
                return True
        # Substring
        if len(new_label_norm) > 3 and len(existing_norm) > 3:
            if new_label_norm in existing_norm or existing_norm in new_label_norm:
                return True
    return False


def main():
    # Step 1: Load all DirPhotos with their existing objects
    print("Loading existing reference data...", flush=True)
    rows = neo4j_query("""
        MATCH (w:Waypoint {place: $place})-[hp:HAS_PHOTO]->(p:DirPhoto)
        OPTIONAL MATCH (p)-[:DETECTED]->(o:Object)
        RETURN w.nid AS nid, hp.slot AS slot, p.photo_file AS photo_file,
               elementId(p) AS photo_id,
               collect({label: o.label, label_norm: o.label_norm, grid_cell: o.grid_cell}) AS objects
        ORDER BY w.nid, hp.slot
    """, {"place": PLACE})

    photos = []
    for r in rows:
        existing = [{"label_norm": normalize_label(o.get("label_norm") or o.get("label") or ""),
                      "cell": o.get("grid_cell") or "mid-center"}
                     for o in r["objects"] if o.get("label")]
        fpath = resolve_photo_path(r["photo_file"])
        if fpath and fpath.exists():
            photos.append({
                "nid": r["nid"], "slot": r["slot"], "photo_file": r["photo_file"],
                "photo_id": r["photo_id"], "fpath": fpath, "existing": existing,
            })
    print(f"  {len(photos)} photos to enrich (×{EXTRA_CALLS} calls each)", flush=True)
    print(f"  Estimated time: ~{len(photos) * EXTRA_CALLS * 4 / 60:.0f} minutes", flush=True)

    total_new = 0
    total_dup = 0
    total_calls = 0
    t_start = time.time()

    for i, p in enumerate(photos):
        all_new_items = []

        for call_i in range(EXTRA_CALLS):
            total_calls += 1
            perc = vlm_perceive(p["fpath"])
            items = [it for it in perc.get("items", []) if it.get("box") and len(it["box"]) == 4
                     and (it["box"][2]-it["box"][0]) < 0.55 and (it["box"][3]-it["box"][1]) < 0.55]

            for it in items:
                zh = it.get("zh", "")
                en = it.get("en", "")
                label_norm = normalize_label(zh, en)
                cell = box_to_grid(it.get("box"))
                conf = it.get("conf", 0.5)

                if not label_norm or len(label_norm) < 2:
                    continue

                combined_existing = p["existing"] + [{"label_norm": normalize_label(x["zh"], x.get("en", "")),
                                                       "cell": box_to_grid(x.get("box"))}
                                                      for x in all_new_items]
                if is_duplicate(label_norm, cell, combined_existing):
                    total_dup += 1
                    continue

                all_new_items.append({
                    "zh": zh, "en": en, "label_norm": label_norm,
                    "cell": cell, "conf": conf, "box": it.get("box"),
                })

        # Write new items to Neo4j
        for item in all_new_items:
            label = f"{item['zh']} {item['en']}".strip()
            try:
                neo4j_query("""
                    MATCH (p:DirPhoto) WHERE elementId(p) = $pid
                    CREATE (o:Object {
                        label: $label,
                        label_norm: $label_norm,
                        score: $score,
                        grid_cell: $cell,
                        source: 'enrich'
                    })
                    CREATE (p)-[:DETECTED]->(o)
                """, {
                    "pid": p["photo_id"],
                    "label": label,
                    "label_norm": item["label_norm"],
                    "score": item["conf"],
                    "cell": item["cell"],
                })
                total_new += 1
            except Exception as e:
                print(f"  Error writing: {e}", flush=True)

        # Update existing list for future dedup
        p["existing"].extend([{"label_norm": x["label_norm"], "cell": x["cell"]} for x in all_new_items])

        elapsed = time.time() - t_start
        avg_per = elapsed / (i + 1)
        remaining = avg_per * (len(photos) - i - 1)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(photos)}] WP{p['nid']}/{p['slot']} | "
                  f"+{len(all_new_items)} new, {total_dup} dup | "
                  f"total new: {total_new} | ETA {remaining/60:.0f}m", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"Done in {elapsed_total:.0f}s ({total_calls} VLM calls)", flush=True)
    print(f"  New objects added: {total_new}", flush=True)
    print(f"  Duplicates skipped: {total_dup}", flush=True)

    # Verify new count
    verify = neo4j_query("""
        MATCH (w:Waypoint {place: $place})-[:HAS_PHOTO]->(p:DirPhoto)-[:DETECTED]->(o:Object)
        RETURN count(o) AS total_objects
    """, {"place": PLACE})
    print(f"  Total objects now: {verify[0]['total_objects']} (was 2190)", flush=True)


if __name__ == "__main__":
    main()
