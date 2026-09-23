"""Full test v2: localize ALL 260 photos, save detailed JSON for artifact report."""
import base64, io, json, re, time
from pathlib import Path
from collections import defaultdict

import os, requests
from PIL import Image, ImageOps

# ── Config ──
OPENAI_API_KEYS = [
    os.environ.get("OPENAI_API_KEY", ""),       # primary
    os.environ.get("OPENAI_API_KEY_BACKUP", ""), # backup
]
OPENAI_API_KEYS = [k for k in OPENAI_API_KEYS if k]
_current_key_idx = 0
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://air.cgu.edu.tw/cgullmapi/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

_host = os.environ.get("NEO4J_HOST", "")
NEO4J_ENDPOINT = f"https://{_host}/db/{_host.split('.')[0]}/query/v2"
_neo4j_user = _host.split('.')[0] if _host else ""
_neo4j_pass = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_AUTH = "Basic " + base64.b64encode(f"{_neo4j_user}:{_neo4j_pass}".encode()).decode()
PLACE = os.environ.get("PLACE", "A7家樂福（9/16)")

DATA_ROOT = Path("/Users/shingchou/Downloads/學校家樂福/0916測試")
OUTPUT_JSON = DATA_ROOT / "檢索圖片" / "full_test_v4_merged_results.json"

GRID_ORDER = ["top-left", "top-center", "top-right",
              "mid-left", "mid-center", "mid-right",
              "bottom-left", "bottom-center", "bottom-right"]

WEIGHT_LANDMARK = 5.0
WEIGHT_EQUIPMENT = 2.0
WEIGHT_NORMAL = 1.0
WEIGHT_GENERIC = 0.3

_SIGN_WORDS = {'標示', '指示', '告示', '看板', '招牌', '吊牌', '吊旗', 'sign', 'banner'}
_ZONE_MARKERS = {'走道', '區域', '出口', 'exit', 'aisle', 'zone', 'decathlon',
                 '結帳', '收銀', '手扶梯', '電扶梯'}
_CATEGORY_SIGNS = {'堅果', '海苔', '飲料', '零食', '冷凍', '冷藏', '生鮮', '日用',
                   '烘焙', '麵包', '肉品', '海鮮', '水果', '蔬菜', '花卉',
                   '咖啡', '茶', '乳品', '奶粉', '保健', '清潔', '衛生',
                   '寵物', '酒', '啤酒', '調味', '罐頭', '餅乾', '早餐',
                   '麵條', '即食', '米', '南北貨', '護理', '洗髮', '染髮'}
_EQUIP_WORDS = {'冷藏展示櫃', '冷凍展示櫃', '冷凍櫃', '冰櫃', '冰箱',
                '展示櫃', '貨架', '櫃台', '磅秤', '烤箱',
                'refrigerat', 'freezer', 'display case', 'counter'}
_GENERIC_WORDS = {'促銷立牌', '紅色促銷', '促銷', 'price tag', 'promotional',
                  'price sign', '價牌', '價格'}


def classify_feature(zh, en=""):
    combined = (zh + ' ' + en).lower()
    for kw in _ZONE_MARKERS:
        if kw in combined:
            return WEIGHT_LANDMARK, "landmark"
    has_sign = any(sw in combined for sw in _SIGN_WORDS)
    has_category = any(cw in combined for cw in _CATEGORY_SIGNS)
    if has_sign and has_category:
        return WEIGHT_LANDMARK, "landmark"
    if has_sign:
        return 3.0, "sign"
    for kw in _EQUIP_WORDS:
        if kw in combined:
            return WEIGHT_EQUIPMENT, "equipment"
    for kw in _GENERIC_WORDS:
        if kw in combined:
            return WEIGHT_GENERIC, "generic"
    return WEIGHT_NORMAL, "normal"


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


def box_to_grid(box):
    if not box or len(box) != 4:
        return "mid-center"
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    col = "left" if cx < 1/3 else ("right" if cx > 2/3 else "center")
    row = "top" if cy < 1/3 else ("bottom" if cy > 2/3 else "mid")
    return f"{row}-{col}"


def _extract_numbers(text):
    return set(re.findall(r'\d+', text))


def _semantic_similarity(ql, rl):
    ql_low, rl_low = ql.lower(), rl.lower()
    scores = []
    ql_words = set(ql_low.split()); rl_words = set(rl_low.split())
    if ql_words and rl_words:
        common = ql_words & rl_words
        if common:
            scores.append(len(common) / max(len(ql_words), len(rl_words)))
    ql_chars = set(ql_low); rl_chars = set(rl_low)
    char_common = ql_chars & rl_chars - {' ', '/', '／', '（', '）', '「', '」'}
    if char_common:
        scores.append(len(char_common) / max(len(ql_chars), len(rl_chars)) * 0.8)
    q_nums = _extract_numbers(ql); r_nums = _extract_numbers(rl)
    if q_nums and r_nums:
        scores.append(0.9 if q_nums & r_nums else -0.3)
    q_cats = {cw for cw in _CATEGORY_SIGNS if cw in ql_low}
    r_cats = {cw for cw in _CATEGORY_SIGNS if cw in rl_low}
    if q_cats and r_cats and q_cats & r_cats:
        scores.append(0.85)
    if ql_low in rl_low or rl_low in ql_low:
        scores.append(min(len(ql), len(rl)) / max(len(ql), len(rl)))
    return max(scores) if scores else 0.0


ADJACENT = {
    "top-left": ["top-center", "mid-left", "mid-center"],
    "top-center": ["top-left", "top-right", "mid-center", "mid-left", "mid-right"],
    "top-right": ["top-center", "mid-right", "mid-center"],
    "mid-left": ["top-left", "top-center", "mid-center", "bottom-left", "bottom-center"],
    "mid-center": GRID_ORDER,
    "mid-right": ["top-right", "top-center", "mid-center", "bottom-right", "bottom-center"],
    "bottom-left": ["mid-left", "mid-center", "bottom-center"],
    "bottom-center": ["bottom-left", "bottom-right", "mid-center", "mid-left", "mid-right"],
    "bottom-right": ["mid-right", "mid-center", "bottom-center"],
}


def weighted_grid_match(query_grid, ref_grid):
    total_weighted = 0.0
    total_possible = 0.0
    landmark_matches = []
    cell_matches = {}

    for cell in GRID_ORDER:
        q_items = query_grid.get(cell, [])
        r_same = ref_grid.get(cell, [])
        r_adj = []
        for ac in ADJACENT.get(cell, []):
            for ri in ref_grid.get(ac, []):
                r_adj.append({**ri, "_p": 0.7})
        all_r = [{**ri, "_p": 1.0} for ri in r_same] + r_adj

        if not q_items or not all_r:
            total_possible += sum(it["weight"] for it in q_items)
            continue

        used = set()
        matches_in_cell = []
        for qi in q_items:
            ql, qw = qi["label"], qi["weight"]
            best_m, best_s, best_i = None, 0, -1
            for ri_i, ri in enumerate(all_r):
                k = (id(ri), ri["label"])
                if k in used: continue
                s = _semantic_similarity(ql, ri["label"]) * ri.get("_p", 1.0)
                if s > best_s:
                    best_s, best_m, best_i = s, ri["label"], ri_i
            if best_m and best_s > 0.25:
                used.add((id(all_r[best_i]), best_m))
                total_weighted += qw * best_s
                matches_in_cell.append({"q": ql, "r": best_m, "sim": round(best_s, 3), "cat": qi["cat"]})
                if qi["cat"] == "landmark":
                    landmark_matches.append((cell, ql, best_m, best_s))
            total_possible += qw

        if matches_in_cell:
            cell_matches[cell] = matches_in_cell

    return total_weighted / max(total_possible, 1.0), len(landmark_matches), cell_matches


def load_reference_map():
    print("Loading reference map from Neo4j...", flush=True)
    rows = neo4j_query("""
        MATCH (w:Waypoint {place: $place})-[hp:HAS_PHOTO]->(p:DirPhoto)-[:DETECTED]->(o:Object)
        RETURN w.nid AS nid, w.pdr_x AS x, w.pdr_y AS y, w.session AS session,
               hp.slot AS slot, p.photo_file AS photo_file,
               o.label AS label, o.label_norm AS label_norm, o.score AS score,
               o.grid_cell AS grid_cell
        ORDER BY w.nid, hp.slot
    """, {"place": PLACE})
    waypoints = {}
    for r in rows:
        nid = r["nid"]
        if nid not in waypoints:
            waypoints[nid] = {
                "nid": nid, "x": r["x"], "y": r["y"], "session": r["session"],
                "slots": defaultdict(lambda: defaultdict(list)),
                "photos": {},
            }
        wp = waypoints[nid]
        slot = r["slot"]
        cell = r["grid_cell"] or "mid-center"
        label = r["label_norm"] or r["label"] or ""
        weight, cat = classify_feature(label)
        wp["slots"][slot][cell].append({"label": label, "weight": weight, "cat": cat})
        if slot not in wp["photos"]:
            wp["photos"][slot] = r["photo_file"]
    print(f"  {len(waypoints)} waypoints loaded", flush=True)
    return waypoints


def build_photo_nid_map():
    rows = neo4j_query("""
        MATCH (w:Waypoint {place: $place})-[hp:HAS_PHOTO]->(p:DirPhoto)
        RETURN w.nid AS nid, w.session AS session, hp.slot AS slot, p.photo_file AS photo_file
    """, {"place": PLACE})
    result = {}
    for r in rows:
        fname = Path(r["photo_file"]).name
        result[fname] = {"nid": r["nid"], "session": r["session"], "slot": r["slot"]}
    return result


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


def vlm_perceive(image_path):
    global _current_key_idx
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
        "temperature": 0.1,
        "max_completion_tokens": 4096,
    }
    for key_idx in range(_current_key_idx, len(OPENAI_API_KEYS)):
        api_key = OPENAI_API_KEYS[key_idx]
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        for attempt in (1, 2, 3):
            try:
                r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", json=body, headers=headers, timeout=120)
                if r.status_code in (429, 402, 401):
                    if key_idx < len(OPENAI_API_KEYS) - 1:
                        print(f"  ⚠ Key #{key_idx+1} quota exhausted (HTTP {r.status_code}), switching to backup key", flush=True)
                        _current_key_idx = key_idx + 1
                        break
                    r.raise_for_status()
                r.raise_for_status()
                text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                if not text.strip() and key_idx < len(OPENAI_API_KEYS) - 1:
                    print(f"  ⚠ Key #{key_idx+1} returned empty, switching to backup key", flush=True)
                    _current_key_idx = key_idx + 1
                    break
            except requests.exceptions.HTTPError:
                if key_idx < len(OPENAI_API_KEYS) - 1:
                    print(f"  ⚠ Key #{key_idx+1} error, switching to backup key", flush=True)
                    _current_key_idx = key_idx + 1
                    break
                if attempt < 3:
                    time.sleep(2)
            except Exception:
                if attempt < 3:
                    time.sleep(2)
        else:
            continue
        continue
    return {"items": [], "scene": ""}


def localize(query_items, ref_map):
    qg = defaultdict(list)
    for it in query_items:
        cell = box_to_grid(it.get("box"))
        zh, en = it.get("zh", ""), it.get("en", "")
        w, c = classify_feature(zh, en)
        qg[cell].append({"label": f"{zh} {en}".strip(), "weight": w, "cat": c})

    results = []
    for nid, wp in ref_map.items():
        best_s, best_lm, best_cells = 0, 0, {}
        for slot, slot_grid in wp["slots"].items():
            s, lm, cells = weighted_grid_match(dict(qg), dict(slot_grid))
            if s > best_s:
                best_s, best_lm, best_cells = s, lm, cells
        results.append({"nid": nid, "score": best_s, "lm": best_lm, "cell_matches": best_cells})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def main():
    ref_map = load_reference_map()
    gt = build_photo_nid_map()
    print(f"  Ground truth: {len(gt)} photos", flush=True)

    photos = []
    for sd in ["01", "02", "03"]:
        d = DATA_ROOT / sd
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix == ".jpg":
                    photos.append(str(f))
    print(f"\nTesting {len(photos)} photos with FRESH VLM calls...", flush=True)
    print(f"Estimated time: ~{len(photos) * 4 / 60:.0f} minutes\n", flush=True)

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    all_results = []
    t_start = time.time()

    for i, photo in enumerate(photos):
        fname = Path(photo).name
        gt_info = gt.get(fname)
        if not gt_info:
            continue

        true_nid = gt_info["nid"]
        total += 1
        t_photo = time.time()

        perc = vlm_perceive(photo)
        vlm_time = time.time() - t_photo
        items = [it for it in perc.get("items", []) if it.get("box") and len(it["box"]) == 4
                 and (it["box"][2]-it["box"][0]) < 0.55 and (it["box"][3]-it["box"][1]) < 0.55]

        classified_items = []
        for it in items:
            zh, en = it.get("zh", ""), it.get("en", "")
            w, c = classify_feature(zh, en)
            cell = box_to_grid(it.get("box"))
            classified_items.append({
                "zh": zh, "en": en, "weight": w, "cat": c, "cell": cell,
                "conf": it.get("conf", 0), "box": it.get("box"),
            })

        loc = localize(items, ref_map)
        top5 = loc[:5]
        top5_nids = [r["nid"] for r in top5]
        top1_nid = top5_nids[0] if top5_nids else None
        top1_score = loc[0]["score"] if loc else 0

        is_top1 = top1_nid == true_nid
        is_top3 = true_nid in top5_nids[:3]
        is_top5 = true_nid in top5_nids[:5]

        if is_top1: correct_top1 += 1
        if is_top3: correct_top3 += 1
        if is_top5: correct_top5 += 1

        parent_dir = Path(photo).parent.name
        direction = fname.rsplit("_", 1)[-1].replace(".jpg", "") if "_" in fname else "?"

        true_rank = None
        true_score = 0
        for j, r in enumerate(loc):
            if r["nid"] == true_nid:
                true_rank = j + 1
                true_score = r["score"]
                break

        entry = {
            "fname": fname,
            "set": parent_dir,
            "direction": direction,
            "true_nid": true_nid,
            "pred_nid": top1_nid,
            "top1_score": round(top1_score, 4),
            "true_rank": true_rank,
            "true_score": round(true_score, 4),
            "gap": round(top1_score - true_score, 4) if true_score else round(top1_score, 4),
            "is_top1": is_top1,
            "is_top3": is_top3,
            "is_top5": is_top5,
            "scene": perc.get("scene", ""),
            "num_items": len(items),
            "vlm_time": round(vlm_time, 2),
            "items": classified_items,
            "top5": [{"nid": r["nid"], "score": round(r["score"], 4), "lm": r["lm"],
                       "cell_matches": {k: v for k, v in r.get("cell_matches", {}).items()}}
                      for r in top5],
        }
        all_results.append(entry)

        elapsed = time.time() - t_start
        avg_per = elapsed / total
        remaining = avg_per * (len(photos) - i - 1)
        acc = correct_top1 / total * 100
        mark = "✓" if is_top1 else ("△" if is_top3 else "✗")

        if total % 10 == 0 or not is_top1:
            print(f"  [{total:3d}/{len(photos)}] {mark} WP{true_nid:2d}→WP{top1_nid:2d} "
                  f"{top1_score*100:5.1f}% | acc={acc:.0f}% | ETA {remaining/60:.0f}m", flush=True)

    elapsed_total = time.time() - t_start

    set_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "top5": 0})
    for r in all_results:
        s = r["set"]
        set_stats[s]["total"] += 1
        if r["is_top1"]: set_stats[s]["top1"] += 1
        if r["is_top3"]: set_stats[s]["top3"] += 1
        if r["is_top5"]: set_stats[s]["top5"] += 1

    dir_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "top5": 0})
    for r in all_results:
        d = r["direction"]
        dir_stats[d]["total"] += 1
        if r["is_top1"]: dir_stats[d]["top1"] += 1
        if r["is_top3"]: dir_stats[d]["top3"] += 1
        if r["is_top5"]: dir_stats[d]["top5"] += 1

    wp_stats = defaultdict(lambda: {"total": 0, "top1": 0, "scores": []})
    for r in all_results:
        nid = r["true_nid"]
        wp_stats[nid]["total"] += 1
        wp_stats[nid]["scores"].append(r["top1_score"])
        if r["is_top1"]: wp_stats[nid]["top1"] += 1

    output = {
        "meta": {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total": total,
            "top1": correct_top1,
            "top3": correct_top3,
            "top5": correct_top5,
            "elapsed_seconds": round(elapsed_total, 1),
            "avg_time_per_photo": round(elapsed_total / max(total, 1), 2),
            "resolution": "1024px",
            "model": OPENAI_MODEL,
            "num_waypoints": len(ref_map),
        },
        "set_stats": {k: dict(v) for k, v in set_stats.items()},
        "dir_stats": {k: dict(v) for k, v in dir_stats.items()},
        "wp_stats": {str(k): {"total": v["total"], "top1": v["top1"],
                               "avg_score": round(sum(v["scores"])/len(v["scores"]), 4)}
                      for k, v in wp_stats.items()},
        "results": all_results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)

    print(f"\n{'='*60}", flush=True)
    print(f"TOTAL: {total} photos ({elapsed_total:.0f}s)", flush=True)
    print(f"  Top-1: {correct_top1}/{total} = {correct_top1/total*100:.1f}%", flush=True)
    print(f"  Top-3: {correct_top3}/{total} = {correct_top3/total*100:.1f}%", flush=True)
    print(f"  Top-5: {correct_top5}/{total} = {correct_top5/total*100:.1f}%", flush=True)
    for s in sorted(set_stats):
        ss = set_stats[s]
        print(f"  Set {s}: {ss['top1']}/{ss['total']} = {ss['top1']/ss['total']*100:.1f}%", flush=True)
    for d in ["front", "back", "left", "right"]:
        ds = dir_stats.get(d, {"total": 0, "top1": 0})
        if ds["total"]:
            print(f"  Dir {d}: {ds['top1']}/{ds['total']} = {ds['top1']/ds['total']*100:.1f}%", flush=True)
    print(f"\nJSON saved: {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
