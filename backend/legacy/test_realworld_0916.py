#!/usr/bin/env python3
"""
test_realworld_0916.py — Real-world 定位測試
用 0916家樂福 測試照片（未參與地圖建立）對 A7 拓樸地圖做定位檢索。
每張照片都會畫 bounding box 並存到 annotated/ 資料夾。

Usage:
    python test_realworld_0916.py
    python test_realworld_0916.py --resume       # 續跑（跳過已有 VLM cache）
    python test_realworld_0916.py --no-vlm       # 只用已快取結果
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw, ImageFont, ImageOps
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # backend/legacy/ → Shopping-Navigation/.env

from server.topomap_v2 import TopoGraphV2
from server.locate_v2 import localize_photo

import requests

# ── Config ──
TEST_SESSIONS = [
    ("0916家樂福1", Path(__file__).parent / "testPDR" / "0916家樂福1" / "photos"),
    ("0916家樂福2", Path(__file__).parent / "testPDR" / "0916家樂福2" / "photos"),
]
TOPOMAP_JSON = Path(__file__).parent / "carrefour_a7_imported.json"
GROUND_TRUTH_JSON = Path(__file__).parent / "testPDR" / "0916_ground_truth.json"
OUTPUT_DIR = Path(__file__).parent / "benchmark_results" / "realworld_0916"
VLM_CACHE_DIR = OUTPUT_DIR / "vlm_cache"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://air.cgu.edu.tw/cgullmapi/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

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

_SIGN_KEYWORDS = {
    "標示", "指示", "告示", "看板", "招牌", "吊牌", "吊旗",
    "走道", "區域", "出口", "入口", "結帳", "收銀",
    "sign", "banner", "aisle", "exit", "zone",
    "手扶梯", "電扶梯", "樓梯", "廁所", "服務台",
}
_CATEGORY_KEYWORDS = {
    "堅果", "海苔", "飲料", "零食", "冷凍", "冷藏", "生鮮", "日用",
    "烘焙", "麵包", "肉品", "海鮮", "水果", "蔬菜", "花卉",
    "咖啡", "茶", "乳品", "奶粉", "保健", "清潔", "衛生",
    "寵物", "酒", "啤酒", "調味", "罐頭", "餅乾", "早餐",
    "麵條", "即食", "米", "南北貨", "護理", "洗髮", "染髮",
}

# ── bbox color by importance ──
_COLORS = {
    "landmark": (255, 0, 0),      # red
    "sign": (255, 128, 0),        # orange
    "equipment": (0, 128, 255),   # blue
    "normal": (0, 200, 0),        # green
    "generic": (160, 160, 160),   # gray
}

_SIGN_WORDS = {'標示', '指示', '告示', '看板', '招牌', '吊牌', '吊旗', 'sign', 'banner'}
_ZONE_MARKERS = {'走道', '區域', '出口', 'exit', 'aisle', 'zone',
                 '結帳', '收銀', '手扶梯', '電扶梯'}
_EQUIP_WORDS = {'冷藏展示櫃', '冷凍展示櫃', '冷凍櫃', '冰櫃', '冰箱',
                '展示櫃', '貨架', '櫃台', '磅秤', '烤箱',
                'refrigerat', 'freezer', 'display case', 'counter'}
_GENERIC_WORDS = {'促銷立牌', '紅色促銷', '促銷', 'price tag', 'promotional',
                  'price sign', '價牌', '價格'}
_CAT_SIGNS = {'堅果', '海苔', '飲料', '零食', '冷凍', '冷藏', '生鮮', '日用',
              '烘焙', '麵包', '肉品', '海鮮', '水果', '蔬菜', '花卉',
              '咖啡', '茶', '乳品', '奶粉', '保健', '清潔', '衛生',
              '寵物', '酒', '啤酒', '調味', '罐頭', '餅乾', '早餐',
              '麵條', '即食', '米', '南北貨', '護理', '洗髮', '染髮'}


def _classify(zh: str, en: str = "") -> str:
    combined = (zh + ' ' + en).lower()
    for kw in _ZONE_MARKERS:
        if kw in combined:
            return "landmark"
    has_sign = any(sw in combined for sw in _SIGN_WORDS)
    has_cat = any(cw in combined for cw in _CAT_SIGNS)
    if has_sign and has_cat:
        return "landmark"
    if has_sign:
        return "sign"
    for kw in _EQUIP_WORDS:
        if kw in combined:
            return "equipment"
    for kw in _GENERIC_WORDS:
        if kw in combined:
            return "generic"
    return "normal"


def vlm_perceive(image_path: str) -> dict:
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
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    for attempt in range(3):
        try:
            r = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json=body, headers=headers, timeout=120,
            )
            r.raise_for_status()
            text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"    VLM error: {e}", flush=True)
    return {"items": [], "scene": ""}


def draw_bboxes(image_path: str, vlm_result: dict, save_path: str):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", size=max(16, h // 60))
    except Exception:
        font = ImageFont.load_default()

    for item in vlm_result.get("items", []):
        box = item.get("box", [])
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        zh = item.get("zh", "")
        en = item.get("en", "")
        cat = _classify(zh, en)
        color = _COLORS.get(cat, (0, 200, 0))

        lw = max(2, h // 400)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        label = zh or en or "?"
        conf = item.get("conf", 0)
        text = f"{label} ({conf:.0%})"
        tx, ty = x1, max(0, y1 - font.size - 4)
        bbox_bg = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(bbox_bg, fill=color)
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    img.save(save_path, quality=90)


def _extract_ocr_text(zh: str) -> str:
    if not zh:
        return ""
    zh_lower = zh.lower()
    if any(kw in zh_lower for kw in _SIGN_KEYWORDS):
        return zh
    if any(kw in zh_lower for kw in _CATEGORY_KEYWORDS):
        return zh
    if re.search(r'走道\s*\d+', zh):
        return zh
    if re.search(r'[A-Z][A-Za-z]{2,}', zh):
        return zh
    return ""


def vlm_to_locate_v2_format(
    vlm_result: dict, img_w: int, img_h: int,
) -> Tuple[List[dict], List[dict]]:
    detections = []
    for item in vlm_result.get("items", []):
        box = item.get("box", [])
        if not box or len(box) != 4:
            continue
        if (box[2] - box[0]) >= 0.55 or (box[3] - box[1]) >= 0.55:
            continue
        px_box = [box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h]
        zh = item.get("zh", "")
        en = item.get("en", "")
        label = f"{zh} {en}".strip() if en else zh
        detections.append({
            "label": label, "score": item.get("conf", 0.5),
            "box": px_box, "ocr_text": _extract_ocr_text(zh),
        })
    return detections, []


def collect_test_photos() -> List[Tuple[str, int, str, str, Path]]:
    """Returns list of (session_name, wp_index, direction, filename, path)."""
    photos = []
    for session_name, photo_dir in TEST_SESSIONS:
        if not photo_dir.is_dir():
            print(f"  WARNING: {photo_dir} not found", flush=True)
            continue
        for f in sorted(photo_dir.iterdir()):
            if f.suffix != ".jpg":
                continue
            m = re.match(r'wp_(\d+)_(\w+)_\d+\.jpg', f.name)
            if m:
                wp_idx = int(m.group(1))
                direction = m.group(2)
                photos.append((session_name, wp_idx, direction, f.name, f))
    return photos


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--embedding", action="store_true", help="Use embedding-based similarity")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load topomap
    print("Loading topomap...", flush=True)
    topo_data = json.loads(TOPOMAP_JSON.read_text(encoding="utf-8"))
    topo = TopoGraphV2.from_dict(topo_data)
    print(f"  {len(topo.all_photo_nodes())} photo nodes, "
          f"{len(topo.all_object_nodes())} objects", flush=True)

    # Pre-compute map label embeddings
    if args.embedding:
        from server.locate_v2 import ensure_embeddings
        map_labels = list({
            n.get("label", "") for n in topo_data["nodes"]
            if n.get("ntype") == "object" and n.get("label")
        })
        print(f"  Pre-computing embeddings for {len(map_labels)} map labels...", flush=True)
        ensure_embeddings(map_labels)
        print(f"  Embeddings cached.", flush=True)

    # Load ground truth
    gt = json.loads(GROUND_TRUTH_JSON.read_text(encoding="utf-8"))
    print(f"  Ground truth: {len(gt)} waypoint mappings", flush=True)

    # Collect test photos
    test_photos = collect_test_photos()
    print(f"  Test photos: {len(test_photos)}", flush=True)

    # Stats
    total = 0
    top1 = top3 = top5 = 0
    skipped = 0
    no_gt = 0
    all_results = []
    session_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "top5": 0})
    dir_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "top5": 0})
    t_start = time.time()

    for i, (session_name, wp_idx, direction, fname, fpath) in enumerate(test_photos):
        gt_key = f"{session_name}_wp{wp_idx}"
        gt_entry = gt.get(gt_key)
        if gt_entry is None:
            no_gt += 1
            continue
        true_nid = gt_entry["nid"]

        # VLM cache key
        cache_key = f"{session_name}_{fname.replace('.jpg', '.json')}"
        cache_path = VLM_CACHE_DIR / cache_key

        # VLM perception
        if cache_path.exists() and (args.resume or args.no_vlm):
            vlm_result = json.loads(cache_path.read_text(encoding="utf-8"))
            vlm_time = 0.0
        elif args.no_vlm:
            skipped += 1
            continue
        else:
            t_vlm = time.time()
            vlm_result = vlm_perceive(str(fpath))
            vlm_time = time.time() - t_vlm
            cache_path.write_text(
                json.dumps(vlm_result, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )

        # Draw bounding boxes on photo
        ann_dir = OUTPUT_DIR / "annotated" / session_name
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_path = ann_dir / fname
        if not ann_path.exists():
            try:
                draw_bboxes(str(fpath), vlm_result, str(ann_path))
            except Exception as e:
                print(f"    bbox draw error: {e}", flush=True)

        # Image dimensions
        try:
            with Image.open(fpath) as im:
                im = ImageOps.exif_transpose(im)
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 3024, 4032

        # Convert to locate_v2 format
        detections, ocr_items = vlm_to_locate_v2_format(vlm_result, img_w, img_h)

        # Build raw VLM items for wgm
        raw_vlm_items = []
        for item in vlm_result.get("items", []):
            box = item.get("box", [])
            if box and len(box) == 4:
                zh = item.get("zh", "")
                en = item.get("en", "")
                raw_vlm_items.append({
                    "box": box,
                    "label": f"{zh} {en}".strip() if en else zh,
                })

        if not detections:
            total += 1
            session_stats[session_name]["total"] += 1
            dir_stats[direction]["total"] += 1
            all_results.append({
                "fname": fname, "session": session_name,
                "wp": wp_idx, "direction": direction,
                "true_nid": true_nid, "pred_nid": -1,
                "score": 0, "is_top1": False, "is_top3": False, "is_top5": False,
                "num_items": 0, "vlm_time": round(vlm_time, 2),
                "gt_dist": gt_entry["dist"],
            })
            continue

        # Pre-compute query embeddings
        if args.embedding:
            from server.locate_v2 import ensure_embeddings
            q_labels = [it.get("label", "") for it in raw_vlm_items if it.get("label")]
            ensure_embeddings(q_labels)

        # Run localization
        result = localize_photo(
            topo, detections, img_w, img_h,
            ocr_items=ocr_items, top_k=5,
            query_direction=direction,
            raw_vlm_items=raw_vlm_items,
            use_embedding=args.embedding,
        )

        total += 1
        candidates = result.candidates
        pred_ids = [c.photo_id for c in candidates]

        is_top1 = len(pred_ids) > 0 and pred_ids[0] == true_nid
        is_top3 = true_nid in pred_ids[:3]
        is_top5 = true_nid in pred_ids[:5]

        if is_top1: top1 += 1
        if is_top3: top3 += 1
        if is_top5: top5 += 1

        score = candidates[0].score if candidates else 0
        pred_nid = pred_ids[0] if pred_ids else -1
        mark = "O" if is_top1 else ("~" if is_top3 else "X")

        session_stats[session_name]["total"] += 1
        dir_stats[direction]["total"] += 1
        if is_top1:
            session_stats[session_name]["top1"] += 1
            dir_stats[direction]["top1"] += 1
        if is_top3:
            session_stats[session_name]["top3"] += 1
            dir_stats[direction]["top3"] += 1
        if is_top5:
            session_stats[session_name]["top5"] += 1
            dir_stats[direction]["top5"] += 1

        all_results.append({
            "fname": fname, "session": session_name,
            "wp": wp_idx, "direction": direction,
            "true_nid": true_nid, "pred_nid": pred_nid,
            "score": round(score, 4),
            "is_top1": is_top1, "is_top3": is_top3, "is_top5": is_top5,
            "num_items": len(vlm_result.get("items", [])),
            "vlm_time": round(vlm_time, 2),
            "gt_dist": gt_entry["dist"],
            "top5": [{"nid": c.photo_id, "score": round(c.score, 4)}
                     for c in candidates[:5]],
        })

        acc = top1 / total * 100 if total else 0
        if total % 10 == 0 or not is_top1:
            print(f"  [{total:3d}/{len(test_photos)}] {mark} "
                  f"wp{wp_idx}_{direction} nid{true_nid}->nid{pred_nid} "
                  f"score={score:.3f} | acc={acc:.1f}%", flush=True)

    elapsed = time.time() - t_start

    print(f"\n{'='*65}")
    print(f"REAL-WORLD TEST — 0916家樂福 → A7 topomap")
    print(f"{'='*65}")
    print(f"  Total: {total} photos ({elapsed:.0f}s)")
    print(f"  Skipped: {skipped} (no VLM cache), No GT: {no_gt}")
    if total > 0:
        print(f"  Top-1: {top1}/{total} = {top1/total*100:.1f}%")
        print(f"  Top-3: {top3}/{total} = {top3/total*100:.1f}%")
        print(f"  Top-5: {top5}/{total} = {top5/total*100:.1f}%")
    print(f"\n  By session:")
    for sn in sorted(session_stats):
        st = session_stats[sn]
        t = max(st["total"], 1)
        print(f"    {sn}: Top-1 {st['top1']}/{st['total']}={st['top1']/t*100:.1f}%  "
              f"Top-3 {st['top3']}/{st['total']}={st['top3']/t*100:.1f}%  "
              f"Top-5 {st['top5']}/{st['total']}={st['top5']/t*100:.1f}%")
    print(f"\n  By direction:")
    for d in ["front", "back", "left", "right"]:
        if d in dir_stats:
            st = dir_stats[d]
            t = max(st["total"], 1)
            print(f"    {d:>5}: Top-1 {st['top1']}/{st['total']}={st['top1']/t*100:.1f}%  "
                  f"Top-3 {st['top3']}/{st['total']}={st['top3']/t*100:.1f}%  "
                  f"Top-5 {st['top5']}/{st['total']}={st['top5']/t*100:.1f}%")

    # Error analysis: worst cases
    errors = [r for r in all_results if not r.get("is_top1")]
    errors.sort(key=lambda r: r.get("score", 0), reverse=True)
    if errors:
        print(f"\n  Error cases ({len(errors)} total):")
        for r in errors[:15]:
            print(f"    {r['session']} wp{r['wp']}_{r['direction']} "
                  f"true=nid{r['true_nid']} pred=nid{r['pred_nid']} "
                  f"score={r.get('score',0):.3f} items={r.get('num_items',0)} "
                  f"gt_dist={r.get('gt_dist',0):.1f}m")

    # ── Waypoint-level aggregation (score fusion across 4 photos) ──
    from collections import defaultdict as _dd
    wp_scores: Dict[str, Dict[int, float]] = _dd(lambda: _dd(float))
    wp_true: Dict[str, int] = {}
    for r in all_results:
        key = f"{r['session']}_wp{r['wp']}"
        wp_true[key] = r['true_nid']
        for c in r.get('top5', []):
            wp_scores[key][c['nid']] = max(wp_scores[key][c['nid']], c['score'])

    wp_total = len(wp_true)
    wp_top1 = wp_top3 = wp_top5 = 0
    for key, true_nid in wp_true.items():
        ranked = sorted(wp_scores[key].items(), key=lambda x: -x[1])
        ranked_nids = [nid for nid, _ in ranked]
        if ranked_nids and ranked_nids[0] == true_nid:
            wp_top1 += 1
        if true_nid in ranked_nids[:3]:
            wp_top3 += 1
        if true_nid in ranked_nids[:5]:
            wp_top5 += 1

    print(f"\n  === Waypoint-level aggregation (best-of-4-photos score fusion) ===")
    if wp_total > 0:
        print(f"  Waypoints: {wp_total}")
        print(f"  Top-1: {wp_top1}/{wp_total} = {wp_top1/wp_total*100:.1f}%")
        print(f"  Top-3: {wp_top3}/{wp_total} = {wp_top3/wp_total*100:.1f}%")
        print(f"  Top-5: {wp_top5}/{wp_total} = {wp_top5/wp_total*100:.1f}%")

    # ── Waypoint-level voting (majority of 4 photos) ──
    wp_votes: Dict[str, Dict[int, int]] = _dd(lambda: _dd(int))
    for r in all_results:
        key = f"{r['session']}_wp{r['wp']}"
        if r.get('pred_nid', -1) >= 0:
            wp_votes[key][r['pred_nid']] += 1

    wp_vote_top1 = 0
    for key, true_nid in wp_true.items():
        if key not in wp_votes:
            continue
        best_nid = max(wp_votes[key].items(), key=lambda x: x[1])[0]
        if best_nid == true_nid:
            wp_vote_top1 += 1

    if wp_total > 0:
        print(f"\n  Voting (plurality of Top-1 across 4 photos):")
        print(f"  Top-1: {wp_vote_top1}/{wp_total} = {wp_vote_top1/wp_total*100:.1f}%")

    # Save results
    out = {
        "meta": {
            "system": "locate_v2 + semantic scoring",
            "test_data": "0916家樂福 (non-map photos)",
            "topomap": "carrefour_a7_imported.json",
            "total": total, "top1": top1, "top3": top3, "top5": top5,
            "elapsed": round(elapsed, 1),
            "wp_total": wp_total,
            "wp_fused_top1": wp_top1, "wp_fused_top3": wp_top3, "wp_fused_top5": wp_top5,
            "wp_vote_top1": wp_vote_top1,
        },
        "session_stats": {k: dict(v) for k, v in session_stats.items()},
        "dir_stats": {k: dict(v) for k, v in dir_stats.items()},
        "results": all_results,
    }
    out_path = OUTPUT_DIR / "realworld_0916_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  Results: {out_path}")
    print(f"  Annotated photos: {OUTPUT_DIR / 'annotated'}")


if __name__ == "__main__":
    main()
