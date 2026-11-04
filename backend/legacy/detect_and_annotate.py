#!/usr/bin/env python3
"""
detect_and_annotate.py — 用 VLM 批次物件辨識 + 標註照片 + HTML 報告

對 testPDR session 資料夾中所有照片，呼叫 VLM (gpt-4o) 做物件偵測和 OCR，
在照片上畫出偵測區域框和標籤，輸出到 annotated/ 子目錄，
並產生 HTML 報告方便快速瀏覽辨識結果。

Usage:
    python detect_and_annotate.py --sessions "0916家樂福1" "0916家樂福2"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw, ImageFont, ImageOps


DET_COLORS = [
    (255, 60, 60), (60, 200, 60), (60, 120, 255),
    (255, 200, 0), (220, 60, 220), (0, 220, 220),
    (255, 140, 0), (140, 60, 255), (60, 180, 140),
]

OCR_COLOR = (255, 255, 100)


def get_det_color(label: str) -> tuple:
    return DET_COLORS[hash(label) % len(DET_COLORS)]


def _get_fonts(img_w: int):
    scale = max(img_w / 1000, 1.0)
    size_big = max(int(28 * scale), 28)
    size_sm = max(int(22 * scale), 22)
    try:
        return ImageFont.truetype("arial.ttf", size_big), ImageFont.truetype("arial.ttf", size_sm)
    except (OSError, IOError):
        f = ImageFont.load_default()
        return f, f


def draw_results(img: Image.Image, detections: list, ocr_texts: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    font_big, font_sm = _get_fonts(img_w)
    line_w = max(int(img_w / 600), 3)
    pad = max(int(img_w / 500), 4)

    used_label_slots = {}

    for i, det in enumerate(detections):
        label = det.get("label", "")
        score = det.get("score", 0)
        bbox = det.get("bbox", [0, 0, 0, 0])
        pos = det.get("position", "")
        color = get_det_color(label)

        x1, y1, x2, y2 = [int(c) for c in bbox]
        if x2 <= x1 or y2 <= y1:
            continue

        inset = i * line_w * 2
        bx1 = min(x1 + inset, (x1 + x2) // 2)
        by1 = min(y1 + inset, (y1 + y2) // 2)
        bx2 = max(x2 - inset, (x1 + x2) // 2)
        by2 = max(y2 - inset, (y1 + y2) // 2)
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=line_w)

        text = f"{label} {score:.0%}"
        tb = draw.textbbox((0, 0), text, font=font_big)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        slot_key = f"{x1//100}_{y1//100}"
        slot_offset = used_label_slots.get(slot_key, 0)
        used_label_slots[slot_key] = slot_offset + th + pad + 2

        ty = max(by1 - th - pad + slot_offset, slot_offset)
        tx = bx1
        draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad], fill=color)
        draw.text((tx + pad, ty + pad // 2), text, fill=(255, 255, 255), font=font_big)

        if pos:
            ptb = draw.textbbox((0, 0), pos, font=font_sm)
            ptw, pth = ptb[2] - ptb[0], ptb[3] - ptb[1]
            draw.rectangle([tx, ty + th + pad, tx + ptw + pad * 2, ty + th + pad + pth + 2],
                           fill=(30, 30, 30))
            draw.text((tx + pad, ty + th + pad), pos, fill=color, font=font_sm)

    ocr_slot_offset = 0
    for ocr in ocr_texts:
        text = ocr.get("text", "")
        bbox = ocr.get("bbox", [])
        if not text or not bbox:
            continue

        if isinstance(bbox[0], list) and len(bbox) == 4:
            pts = [(int(p[0]), int(p[1])) for p in bbox]
            draw.polygon(pts, outline=OCR_COLOR, width=line_w)
            x, y = pts[0]
        elif len(bbox) == 4:
            x1o, y1o, x2o, y2o = [int(c) for c in bbox]
            draw.rectangle([x1o, y1o, x2o, y2o], outline=OCR_COLOR, width=line_w)
            x, y = x1o, y1o
        else:
            continue

        score = ocr.get("score", 0)
        ocr_label = f'OCR: "{text}" {score:.0%}'
        tb = draw.textbbox((0, 0), ocr_label, font=font_sm)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        label_y = img_h - th - pad - ocr_slot_offset
        label_x = max(x, pad)
        draw.rectangle([label_x, label_y, label_x + tw + pad * 2, label_y + th + pad],
                       fill=(60, 60, 0))
        draw.text((label_x + pad, label_y + pad // 2), ocr_label, fill=OCR_COLOR, font=font_sm)
        ocr_slot_offset += th + pad + 4

    return img


def run_vlm_perceive(photo_path: str, img_w: int, img_h: int) -> dict:
    from server.vlm import perceive_only
    result = perceive_only(
        image_path=photo_path,
        goal_objects=[],
        img_w=img_w,
        img_h=img_h,
    )
    detections = [
        {"label": d.label, "score": d.score, "bbox": d.bbox, "position": d.position}
        for d in result.detections
    ]
    ocr_texts = [
        {"text": o.text, "score": o.score, "bbox": o.bbox, "position": o.position}
        for o in result.ocr_texts
    ]
    aisle_signs = [
        {"aisle_number": a.aisle_number, "categories": a.categories, "position": a.position}
        for a in result.aisle_signs
    ]
    return {
        "detections": detections,
        "ocr_texts": ocr_texts,
        "aisle_signs": aisle_signs,
        "scene_description": result.scene_description,
    }


def generate_html_report(session_name: str, results: list, output_dir: Path) -> Path:
    html_path = output_dir / "detection_report.html"
    rows = []
    for r in results:
        wp = r["waypoint"]
        direction = r["direction"]
        dets = r["detections"]
        ocrs = r["ocr_texts"]
        scene = r.get("scene_description", "")
        n_det = len(dets)
        n_ocr = len(ocrs)

        det_labels = ", ".join(
            f'<span style="color:{_css_color(get_det_color(d["label"]))}">{d["label"]}({d["score"]:.0%})</span>'
            for d in dets
        ) or '<span style="color:#888">(none)</span>'

        ocr_labels = ", ".join(
            f'<span style="color:#ffff64">"{o["text"]}"({o["score"]:.0%})</span>'
            for o in ocrs
        ) or '<span style="color:#888">(none)</span>'

        annotated_rel = r["annotated_rel"]

        rows.append(f"""
        <tr id="wp{wp}_{direction}">
          <td class="wp-cell">wp{wp}<br><b>{direction}</b></td>
          <td>{n_det}</td>
          <td>{n_ocr}</td>
          <td class="labels-cell">{det_labels}</td>
          <td class="labels-cell">{ocr_labels}</td>
          <td class="scene-cell">{scene}</td>
          <td>
            <a href="{annotated_rel}" target="_blank">
              <img src="{annotated_rel}" width="400" loading="lazy">
            </a>
          </td>
        </tr>""")

    total_photos = len(results)
    total_dets = sum(len(r["detections"]) for r in results)
    total_ocr = sum(len(r["ocr_texts"]) for r in results)
    photos_with_det = sum(1 for r in results if r["detections"])

    all_labels = {}
    for r in results:
        for d in r["detections"]:
            lbl = d["label"]
            all_labels[lbl] = all_labels.get(lbl, 0) + 1
    label_summary = ", ".join(f"{k}({v})" for k, v in sorted(all_labels.items(), key=lambda x: -x[1])[:30])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{session_name} VLM 物件辨識報告</title>
<style>
body {{ font-family: "Segoe UI", sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }}
.stats {{ background: #161b22; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #30363d; }}
.stats b {{ color: #58a6ff; }}
.label-summary {{ background: #161b22; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #30363d; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #21262d; padding: 10px; text-align: left; position: sticky; top: 0; z-index: 10; border-bottom: 2px solid #30363d; }}
td {{ padding: 8px; border-bottom: 1px solid #21262d; vertical-align: top; }}
tr:hover {{ background: #161b22; }}
.wp-cell {{ font-weight: bold; white-space: nowrap; text-align: center; }}
.labels-cell {{ font-size: 12px; max-width: 250px; }}
.scene-cell {{ font-size: 12px; max-width: 200px; color: #8b949e; }}
img {{ border-radius: 4px; cursor: pointer; }}
a {{ color: #58a6ff; }}
</style></head><body>
<h1>{session_name} — VLM 物件辨識報告</h1>
<div class="stats">
  <b>總照片數:</b> {total_photos} |
  <b>偵測物件總數:</b> {total_dets} |
  <b>OCR 文字總數:</b> {total_ocr} |
  <b>有偵測到物件的照片:</b> {photos_with_det} ({photos_with_det/max(total_photos,1)*100:.1f}%)
</div>
<div class="label-summary">
  <b>物件標籤統計 (Top 30):</b> {label_summary}
</div>
<table>
<tr><th>照片</th><th>物件</th><th>OCR</th><th>偵測結果</th><th>OCR 文字</th><th>場景描述</th><th>標註圖</th></tr>
{"".join(rows)}
</table>
</body></html>"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


def _css_color(rgb: tuple) -> str:
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def process_session(session_dir: Path) -> list:
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[錯誤] 找不到 {manifest_path}")
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    entries.sort(key=lambda e: e["index"])

    annotated_dir = session_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)

    detections_dir = session_dir / "vlm_detections"
    detections_dir.mkdir(exist_ok=True)

    results = []
    total = sum(len(e.get("photos", {})) or 1 for e in entries)
    done = 0

    for entry in entries:
        wp_idx = entry["index"]
        photos = entry.get("photos", {})
        if not photos:
            photos = {"front": entry.get("photo", "")}

        for direction in ["front", "right", "back", "left"]:
            photo_rel = photos.get(direction, "")
            if not photo_rel:
                continue
            photo_path = session_dir / photo_rel
            if not photo_path.exists():
                done += 1
                continue

            done += 1
            cache_key = f"wp{wp_idx}_{direction}"
            cache_path = detections_dir / f"{cache_key}.json"

            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                detections = cached.get("detections", [])
                ocr_texts = cached.get("ocr_texts", [])
                scene_desc = cached.get("scene_description", "")
                source = "cache"
            else:
                img = ImageOps.exif_transpose(Image.open(photo_path))
                img_w, img_h = img.size

                t0 = time.time()
                try:
                    result = run_vlm_perceive(str(photo_path), img_w, img_h)
                    detections = result["detections"]
                    ocr_texts = result["ocr_texts"]
                    aisle_signs = result.get("aisle_signs", [])
                    scene_desc = result["scene_description"]
                except Exception as e:
                    print(f"  [{done}/{total}] {cache_key}: VLM 失敗 — {e}")
                    detections, ocr_texts, aisle_signs, scene_desc = [], [], [], ""

                elapsed = time.time() - t0
                source = f"VLM {elapsed:.1f}s"

                cache_path.write_text(
                    json.dumps({
                        "detections": detections,
                        "ocr_texts": ocr_texts,
                        "aisle_signs": aisle_signs,
                        "scene_description": scene_desc,
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            n_det = len(detections)
            n_ocr = len(ocr_texts)
            labels_str = ", ".join(d["label"] for d in detections[:5]) or "(none)"
            ocr_str = ", ".join(o["text"] for o in ocr_texts[:3]) or "(none)"
            print(f"  [{done}/{total}] {cache_key}: {n_det} dets, {n_ocr} ocr [{source}] — {labels_str} | {ocr_str}")

            img = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            annotated = draw_results(img, detections, ocr_texts)
            ann_filename = f"{cache_key}.jpg"
            annotated.save(annotated_dir / ann_filename, quality=85)

            results.append({
                "waypoint": wp_idx,
                "direction": direction,
                "detections": detections,
                "ocr_texts": ocr_texts,
                "scene_description": scene_desc,
                "annotated_rel": f"annotated/{ann_filename}",
                "original_rel": photo_rel,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="VLM 批次物件辨識 + 標註照片")
    parser.add_argument("--sessions", nargs="+", required=True, help="session 資料夾名稱")
    parser.add_argument("--base-dir", default="testPDR", help="testPDR 基底路徑")
    args = parser.parse_args()

    print(f"VLM 物件辨識模式 (使用 perceive_only)")
    print(f"Sessions: {args.sessions}\n")

    base = Path(args.base_dir)
    for session_name in args.sessions:
        session_dir = base / session_name
        if not session_dir.exists():
            print(f"[跳過] {session_dir} 不存在")
            continue

        print(f"{'='*60}")
        print(f"處理 {session_name}...")
        print(f"{'='*60}")

        t0 = time.time()
        results = process_session(session_dir)
        elapsed = time.time() - t0

        if results:
            report = generate_html_report(session_name, results, session_dir)
            total_dets = sum(len(r["detections"]) for r in results)
            total_ocr = sum(len(r["ocr_texts"]) for r in results)
            print(f"\n完成! {len(results)} 張照片, {total_dets} 偵測, {total_ocr} OCR, 耗時 {elapsed:.0f}s")
            print(f"HTML 報告: {report}")
        else:
            print(f"無結果。")
        print()


if __name__ == "__main__":
    main()
