#!/usr/bin/env python3
"""
test_localization.py — 定位檢索測試

用合併地圖做定位檢索：對每個 waypoint 的 VLM 偵測結果當 query，
呼叫 localize_photo() 看能否找到正確（或鄰近）的 photo node。

Usage:
    python test_localization.py
    python test_localization.py --direction front right back left
    python test_localization.py --session "0916家樂福1" -v
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageOps
from server.topomap_v2 import TopoGraphV2
from server.locate_v2 import localize_photo


DIRECTIONS = ["front", "right", "back", "left"]


def _load_vlm_cache(det_path: Path) -> dict:
    if not det_path.exists():
        return {}
    return json.loads(det_path.read_text(encoding="utf-8"))


def _vlm_to_query_format(vlm_data: dict) -> Tuple[List[dict], List[dict]]:
    detections = []
    for d in vlm_data.get("detections", []):
        detections.append({
            "label": d.get("label", ""),
            "score": d.get("score", 0.0),
            "box": d.get("bbox", []),
            "position": d.get("position", ""),
            "ocr_text": d.get("ocr_text", ""),
        })
    ocr_items = []
    for o in vlm_data.get("ocr_texts", []):
        ocr_items.append({
            "text": o.get("text", ""),
            "bbox": o.get("bbox", []),
            "score": o.get("score", 0.0),
            "position": o.get("position", ""),
        })
    return detections, ocr_items


def _guess_image_size(photo_path: Path) -> Tuple[int, int]:
    try:
        with Image.open(photo_path) as im:
            im = ImageOps.exif_transpose(im)
            return im.size
    except Exception:
        return (3024, 4032)


def _get_walkway_neighbors(topo: TopoGraphV2, nid: int) -> Set[int]:
    neighbors = set()
    for _, v, k, _ in topo.graph.out_edges(nid, keys=True, data=True):
        if "walkway" in str(k):
            neighbors.add(v)
    for u, _, k, _ in topo.graph.in_edges(nid, keys=True, data=True):
        if "walkway" in str(k):
            neighbors.add(u)
    neighbors.discard(nid)
    return neighbors


def _coord_dist(topo: TopoGraphV2, nid_a: int, nid_b: int) -> float:
    sa = topo.graph.nodes[nid_a].get("sensor_data", {})
    sb = topo.graph.nodes[nid_b].get("sensor_data", {})
    dx = sa.get("pdr_x", 0) - sb.get("pdr_x", 0)
    dy = sa.get("pdr_y", 0) - sb.get("pdr_y", 0)
    return math.sqrt(dx * dx + dy * dy)


def build_ground_truth(
    topo: TopoGraphV2,
    sessions: List[str],
    base_dir: Path,
) -> Dict[str, Dict[int, int]]:
    """(session, wp_index) -> merged node ID，透過座標最近匹配。"""
    photo_nodes = []
    for nid in topo.all_photo_nodes():
        sd = topo.graph.nodes[nid].get("sensor_data", {})
        photo_nodes.append((nid, sd.get("pdr_x", 0), sd.get("pdr_y", 0)))

    aligned_path = base_dir / "aligned_coords.json"
    aligned = json.loads(aligned_path.read_text(encoding="utf-8")) if aligned_path.exists() else {}

    gt: Dict[str, Dict[int, int]] = {}
    for sess in sessions:
        coords = aligned.get(sess, [])
        sess_gt: Dict[int, int] = {}
        for i, (cx, cy) in enumerate(coords):
            wp_idx = i + 1
            best_nid, best_dist = -1, float("inf")
            for nid, nx, ny in photo_nodes:
                d = math.sqrt((cx - nx) ** 2 + (cy - ny) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_nid = nid
            if best_nid >= 0 and best_dist < 2.0:
                sess_gt[wp_idx] = best_nid
        gt[sess] = sess_gt
    return gt


def run_test(
    topo: TopoGraphV2,
    sessions: List[str],
    base_dir: Path,
    directions: List[str],
    top_k: int = 5,
    verbose: bool = False,
) -> None:
    gt = build_ground_truth(topo, sessions, base_dir)

    total = 0
    top1_exact = 0
    top1_near = 0
    top3_exact = 0
    top5_exact = 0
    no_detection = 0
    dist_errors: List[float] = []

    print(f"\n{'Session':<16} {'WP':>3} {'Dir':<6} {'True':>5} {'Pred':>5} "
          f"{'Score':>6} {'T1':>3} {'T3':>3} {'Dist':>6} {'Det':>4} {'OCR':>4}")
    print("-" * 80)

    for sess in sessions:
        sess_dir = base_dir / sess
        vlm_dir = sess_dir / "vlm_detections"
        manifest_path = sess_dir / "manifest.json"

        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = sorted(manifest.get("entries", []), key=lambda e: e["index"])
        sess_gt = gt.get(sess, {})

        for entry in entries:
            wp_idx = entry["index"]
            true_nid = sess_gt.get(wp_idx, -1)
            if true_nid < 0:
                continue

            photos = entry.get("photos", {})
            if not photos:
                photos = {"front": entry.get("photo", "")}

            for direction in directions:
                photo_rel = photos.get(direction, "")
                if not photo_rel:
                    continue

                cache_key = f"wp{wp_idx}_{direction}"
                det_path = vlm_dir / f"{cache_key}.json"
                vlm_data = _load_vlm_cache(det_path)

                if not vlm_data or not vlm_data.get("detections"):
                    no_detection += 1
                    continue

                detections, ocr_items = _vlm_to_query_format(vlm_data)
                photo_path = sess_dir / photo_rel
                img_w, img_h = _guess_image_size(photo_path) if photo_path.exists() else (3024, 4032)

                result = localize_photo(
                    topo, detections, img_w, img_h,
                    ocr_items=ocr_items, top_k=top_k,
                )

                total += 1
                candidates = result.candidates
                pred_nids = [c.photo_id for c in candidates]
                pred_scores = [c.score for c in candidates]

                neighbors = _get_walkway_neighbors(topo, true_nid)

                is_top1 = len(pred_nids) > 0 and pred_nids[0] == true_nid
                is_top1_near = len(pred_nids) > 0 and pred_nids[0] in neighbors
                is_top3 = true_nid in pred_nids[:3]
                is_top5 = true_nid in pred_nids[:5]

                if is_top1:
                    top1_exact += 1
                elif is_top1_near:
                    top1_near += 1
                if is_top3:
                    top3_exact += 1
                if is_top5:
                    top5_exact += 1

                pred_dist = _coord_dist(topo, pred_nids[0], true_nid) if pred_nids else -1
                if pred_dist >= 0:
                    dist_errors.append(pred_dist)

                t1 = "O" if is_top1 else ("~" if is_top1_near else "X")
                t3 = "O" if is_top3 else "X"
                score_str = f"{pred_scores[0]:.3f}" if pred_scores else "  N/A"
                dist_str = f"{pred_dist:.1f}m" if pred_dist >= 0 else "  N/A"

                print(f"{sess:<16} {wp_idx:>3} {direction:<6} {true_nid:>5} "
                      f"{pred_nids[0] if pred_nids else -1:>5} {score_str:>6} "
                      f"{t1:>3} {t3:>3} {dist_str:>6} {len(detections):>4} {len(ocr_items):>4}")

                if verbose and not is_top1:
                    true_sd = topo.graph.nodes[true_nid].get("sensor_data", {})
                    if pred_nids:
                        pred_sd = topo.graph.nodes[pred_nids[0]].get("sensor_data", {})
                        top_labels = []
                        for c in candidates[:3]:
                            top_labels.append(f"  #{c.photo_rank}(node{c.photo_id}) "
                                              f"score={c.score:.3f} matched={c.matched_count}")
                        print(f"    真={true_nid}({true_sd.get('pdr_x',0):.1f},{true_sd.get('pdr_y',0):.1f}) "
                              f"預測={pred_nids[0]}({pred_sd.get('pdr_x',0):.1f},{pred_sd.get('pdr_y',0):.1f})")
                        for tl in top_labels:
                            print(f"    {tl}")

    if total == 0:
        print("\n沒有可測試的查詢")
        return

    avg_dist = sum(dist_errors) / len(dist_errors) if dist_errors else 0
    median_dist = sorted(dist_errors)[len(dist_errors) // 2] if dist_errors else 0

    print(f"\n{'='*80}")
    print(f"定位檢索結果摘要")
    print(f"{'='*80}")
    print(f"  查詢總數:          {total} (跳過 {no_detection} 個無偵測)")
    print(f"  Top-1 精確:        {top1_exact}/{total} ({top1_exact/total*100:.1f}%)")
    print(f"  Top-1 鄰近(±1步):  {top1_near}/{total} ({top1_near/total*100:.1f}%)")
    print(f"  Top-1 合計:        {top1_exact+top1_near}/{total} ({(top1_exact+top1_near)/total*100:.1f}%)")
    print(f"  Top-3 精確:        {top3_exact}/{total} ({top3_exact/total*100:.1f}%)")
    print(f"  Top-5 精確:        {top5_exact}/{total} ({top5_exact/total*100:.1f}%)")
    print(f"  預測距離 平均:     {avg_dist:.1f}m")
    print(f"  預測距離 中位數:   {median_dist:.1f}m")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="定位檢索測試")
    parser.add_argument("--map", default="testPDR/0916家樂福B1F/topomap_v2/topomap.json")
    parser.add_argument("--sessions", nargs="+", default=["0916家樂福1", "0916家樂福2"])
    parser.add_argument("--base-dir", default="testPDR")
    parser.add_argument("--direction", nargs="+", default=["front"],
                        help="要測試的方向 (front/right/back/left)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"[載入地圖] {args.map}")
    topo = TopoGraphV2.load(args.map)
    print(f"  Photo nodes: {len(topo.all_photo_nodes())}")
    print(f"  Object nodes: {len(topo.all_object_nodes())}")

    run_test(topo, args.sessions, Path(args.base_dir), args.direction,
             top_k=args.top_k, verbose=args.verbose)


if __name__ == "__main__":
    main()
