#!/usr/bin/env python3
"""
augment_topomap_4dir.py — 擴充既有 topomap，加入四方向照片與 VLM 物件子圖

讀取已建好的 topomap（只有 PDR 骨架），從 manifest.json 取得四方向照片路徑，
從 vlm_detections/ 讀取各方向的 VLM 偵測結果，建立物件子圖（帶 direction 標記）。

Usage:
    python augment_topomap_4dir.py --session "0916家樂福1"
    python augment_topomap_4dir.py --session "0916家樂福1" "0916家樂福2"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageOps
from server.topomap_v2 import TopoGraphV2


DIRECTIONS = ["front", "right", "back", "left"]


def _guess_image_size(photo_path: Path) -> tuple[int, int]:
    try:
        with Image.open(photo_path) as im:
            im = ImageOps.exif_transpose(im)
            return im.size
    except Exception:
        return (3024, 4032)


def _load_vlm_detections(det_path: Path) -> dict:
    if not det_path.exists():
        return {}
    return json.loads(det_path.read_text(encoding="utf-8"))


def _vlm_to_build_format(vlm_data: dict) -> tuple[list, list]:
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


def augment_session(
    session_dir: Path, fresh: bool = False,
    aligned_coords: List[List[float]] | None = None,
) -> TopoGraphV2:
    session_name = session_dir.name
    topo_path = session_dir / "topomap_v2" / "topomap.json"
    manifest_path = session_dir / "manifest.json"
    vlm_dir = session_dir / "vlm_detections"

    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = sorted(manifest.get("entries", []), key=lambda e: e["index"])

    if fresh or not topo_path.exists():
        print(f"[建圖] 從頭建立 topomap（fresh={fresh}）")
        topo = _build_fresh(session_dir, entries, aligned_coords=aligned_coords)
    else:
        print(f"[載入] 讀取既有 topomap: {topo_path}")
        topo = TopoGraphV2.load(topo_path)

    photo_nodes = sorted(topo.all_photo_nodes())
    if len(photo_nodes) != len(entries):
        print(f"[警告] topomap 有 {len(photo_nodes)} 個 photo node，"
              f"但 manifest 有 {len(entries)} 個 entries")

    total_objects = 0
    total_ocr = 0
    all_object_ids: Dict[int, List[int]] = {}

    for i, (nid, entry) in enumerate(zip(photo_nodes, entries)):
        wp_idx = entry["index"]
        photos = entry.get("photos", {})
        if not photos:
            photos = {"front": entry.get("photo", "")}

        photo_paths = {}
        for d in DIRECTIONS:
            rel = photos.get(d, "")
            if rel:
                photo_paths[d] = str(session_dir / rel)

        topo.graph.nodes[nid]["photos"] = {
            d: photos.get(d, "") for d in DIRECTIONS if photos.get(d)
        }

        node_object_ids = []
        dir_summary = []

        for direction in DIRECTIONS:
            photo_rel = photos.get(direction, "")
            if not photo_rel:
                continue

            cache_key = f"wp{wp_idx}_{direction}"
            det_path = vlm_dir / f"{cache_key}.json"
            vlm_data = _load_vlm_detections(det_path)

            if not vlm_data:
                continue

            detections, ocr_items = _vlm_to_build_format(vlm_data)
            if not detections:
                continue

            photo_path = session_dir / photo_rel
            if photo_path.exists():
                img_w, img_h = _guess_image_size(photo_path)
            else:
                img_w, img_h = 3024, 4032

            object_ids = topo.build_subgraph_from_detections(
                nid, detections, img_w, img_h,
                ocr_items=ocr_items,
                direction=direction,
            )
            node_object_ids.extend(object_ids)
            n_det = len(detections)
            n_ocr = len(ocr_items)
            total_objects += n_det
            total_ocr += n_ocr
            dir_summary.append(f"{direction}:{n_det}d/{n_ocr}o")

        all_object_ids[nid] = node_object_ids
        summary = ", ".join(dir_summary) if dir_summary else "(no detections)"
        print(f"  wp{wp_idx} (node {nid}): {summary}")

    # 跨照片同物件關聯
    ordered_nids = sorted(all_object_ids.keys())
    linked_total = 0
    for a, b in zip(ordered_nids, ordered_nids[1:]):
        if all_object_ids.get(a) and all_object_ids.get(b):
            linked = topo.link_same_objects_between_photos(
                all_object_ids[a], all_object_ids[b],
            )
            linked_total += len(linked)

    print(f"\n  完成: {len(photo_nodes)} nodes, "
          f"{total_objects} objects, {total_ocr} OCR, "
          f"{linked_total} same-object links")

    output_path = session_dir / "topomap_v2" / "topomap.json"
    topo.save(output_path)
    print(f"  儲存: {output_path}")

    return topo


def _build_fresh(
    session_dir: Path, entries: list,
    aligned_coords: List[List[float]] | None = None,
) -> TopoGraphV2:
    """從 manifest 建立只有 PDR 骨架的 topomap。"""

    topo = TopoGraphV2(session_dir.name)

    entry_to_node: Dict[int, int] = {}
    ordered = sorted(entries, key=lambda e: e["index"])

    for i, entry in enumerate(ordered):
        idx = entry["index"]
        photo_rel = entry.get("photo", "")
        photo_path = session_dir / photo_rel if photo_rel else Path("")
        photos = entry.get("photos", {})

        if aligned_coords and i < len(aligned_coords):
            pdr_x, pdr_y = aligned_coords[i]
        else:
            pdr_x = entry.get("pdr_x", 0.0)
            pdr_y = entry.get("pdr_y", 0.0)

        sensor_data = {
            "pdr_x": pdr_x,
            "pdr_y": pdr_y,
            "heading_deg": entry.get("heading_deg", 0.0),
            "total_steps": entry.get("steps", 0),
            "total_distance_m": entry.get("distance_m", 0.0),
            "segment_steps": entry.get("segment_steps", 0),
            "segment_distance_m": entry.get("segment_distance_m", 0.0),
        }
        for extra in ("calibrated_yaw_deg", "game_rot_vec_yaw_deg", "rot_vec_yaw_deg"):
            if extra in entry:
                sensor_data[extra] = entry[extra]

        nid = topo.add_photo_node(
            str(photo_path) if photo_path.exists() else "",
            timestamp=entry.get("timestamp"),
            sensor_data=sensor_data,
            photos={d: photos[d] for d in DIRECTIONS if d in photos},
        )
        entry_to_node[idx] = nid

    ordered_indices = [e["index"] for e in ordered]
    for i in range(len(ordered_indices) - 1):
        nid_a = entry_to_node[ordered_indices[i]]
        nid_b = entry_to_node[ordered_indices[i + 1]]
        entry_b = ordered[i + 1]
        seg_dist = entry_b.get("segment_distance_m", 0.0)
        seg_steps = entry_b.get("segment_steps", 0)
        heading = ordered[i].get("heading_deg", 0.0)

        h = heading % 360
        dirs = ["北", "東北", "東", "東南", "南", "西南", "西", "西北"]
        dir_text = f"朝{dirs[round(h / 45) % 8]}方向前進 (sensor: {heading:.1f}°)"

        topo.add_walkway_edge(
            nid_a, nid_b,
            direction=dir_text,
            distance_m=seg_dist if seg_dist > 0 else None,
            steps=seg_steps if seg_steps > 0 else None,
        )

    first_nid = entry_to_node[ordered_indices[0]]
    topo.set_start_position(first_nid)
    for idx in ordered_indices[1:]:
        topo.move_to(entry_to_node[idx])

    return topo


def main():
    parser = argparse.ArgumentParser(description="擴充 topomap 加入四方向照片與 VLM 物件子圖")
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--base-dir", default="testPDR")
    parser.add_argument("--fresh", action="store_true",
                        help="不讀既有 topomap，從頭建立")
    parser.add_argument("--coords", default="testPDR/aligned_coords.json",
                        help="對齊座標 JSON 檔案路徑")
    args = parser.parse_args()

    all_coords = {}
    coords_path = Path(args.coords)
    if coords_path.exists():
        all_coords = json.loads(coords_path.read_text(encoding="utf-8"))
        print(f"[座標] 載入對齊座標: {coords_path}")
    else:
        print(f"[座標] 無對齊座標檔案，使用 manifest 內的 PDR 座標")

    base = Path(args.base_dir)
    for session_name in args.sessions:
        session_dir = base / session_name
        if not session_dir.exists():
            print(f"[跳過] {session_dir} 不存在")
            continue

        coords = all_coords.get(session_name)
        if coords:
            print(f"[座標] {session_name}: 使用 {len(coords)} 個對齊座標點")

        print(f"\n{'='*60}")
        print(f"擴充 {session_name}")
        print(f"{'='*60}")
        augment_session(session_dir, fresh=args.fresh, aligned_coords=coords)


if __name__ == "__main__":
    main()
