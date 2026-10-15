#!/usr/bin/env python3
"""
merge_aligned_topomap.py — 將手動對齊後的多 session PDR 座標合併成一張拓樸地圖

用法:
    python merge_aligned_topomap.py \
        --aligned testPDR/carrefour_a7_aligned_waypoints.json \
        --place "A7家樂福" \
        --merge-radius 3.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server.topomap_v2 import TopoGraphV2

SESSION_DIRS = {
    "家樂福1": "0908家樂福1",
    "家樂福2": "0908家樂福2",
    "家樂福3": "0908家樂福3",
    "家樂福4": "0908家樂福4",
    "家樂福5": "0908家樂福5",
}


def _load_manifest(session_dir: Path) -> List[dict]:
    m = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    return sorted(m.get("entries", []), key=lambda e: e["index"])


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _load_detections(session_dir: Path, index: int) -> Tuple[List[dict], List[dict]]:
    """讀取 VLM 偵測快取，回傳 (detections, ocr_items)。"""
    p = session_dir / "detections" / f"detections_{index}.json"
    if not p.exists():
        return [], []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("detections", []), data.get("ocr", [])


def _guess_image_size(photo_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(photo_path) as im:
            return im.size
    except Exception:
        return (1600, 1200)


def merge(
    aligned_path: Path,
    place_name: str,
    output_root: Path,
    merge_radius: float = 3.0,
) -> TopoGraphV2:
    base = aligned_path.parent
    aligned: Dict[str, List[List[float]]] = json.loads(
        aligned_path.read_text(encoding="utf-8")
    )

    topo = TopoGraphV2(place_name)

    # session_name -> list of (node_id, aligned_xy)
    session_nodes: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {}

    for sess_name, pts in aligned.items():
        dir_key = sess_name
        sess_dir = base / SESSION_DIRS.get(dir_key, dir_key)
        entries = _load_manifest(sess_dir) if sess_dir.exists() else []

        node_list: List[Tuple[int, Tuple[float, float]]] = []
        photo_object_ids: Dict[int, List[int]] = {}

        for i, (ax, ay) in enumerate(pts):
            entry = entries[i] if i < len(entries) else {}
            idx = entry.get("index", i + 1)
            photo_rel = entry.get("photo", "")
            photo_path = str(sess_dir / photo_rel) if photo_rel else ""

            sensor_data = {
                "pdr_x": ax,
                "pdr_y": ay,
                "heading_deg": entry.get("heading_deg", 0.0),
                "total_steps": entry.get("steps", 0),
                "total_distance_m": entry.get("distance_m", 0.0),
                "segment_steps": entry.get("segment_steps", 0),
                "segment_distance_m": entry.get("segment_distance_m", 0.0),
                "session": sess_name,
            }

            nid = topo.add_photo_node(
                photo_path,
                timestamp=entry.get("timestamp"),
                sensor_data=sensor_data,
            )
            node_list.append((nid, (ax, ay)))

            # VLM 偵測結果
            detections, ocr_items = _load_detections(sess_dir, idx)
            if detections:
                photo_p = Path(photo_path) if photo_path else None
                img_w, img_h = _guess_image_size(photo_p) if photo_p and photo_p.exists() else (1600, 1200)
                obj_ids = topo.build_subgraph_from_detections(
                    nid, detections, img_w, img_h, ocr_items=ocr_items,
                )
                photo_object_ids[nid] = obj_ids
                print(f"  [{sess_name}] wp{idx}: {len(detections)} 物件, {len(ocr_items)} OCR")
            else:
                photo_object_ids[nid] = []

        session_nodes[sess_name] = node_list

        # 跨照片同物件關聯
        ordered_nids = [nid for nid, _ in node_list]
        for a, b in zip(ordered_nids, ordered_nids[1:]):
            if photo_object_ids.get(a) and photo_object_ids.get(b):
                linked = topo.link_same_objects_between_photos(
                    photo_object_ids[a], photo_object_ids[b],
                )
                if linked:
                    print(f"  [{sess_name}] 同物件關聯 {a}<->{b}: {len(linked)} 組")

        # intra-session walkway edges
        for j in range(len(node_list) - 1):
            nid_a, _ = node_list[j]
            nid_b, _ = node_list[j + 1]
            entry_b = entries[j + 1] if j + 1 < len(entries) else {}
            seg_dist = entry_b.get("segment_distance_m", 0.0)
            seg_steps = entry_b.get("segment_steps", 0)
            heading = entries[j].get("heading_deg", 0.0) if j < len(entries) else 0.0

            h = heading % 360
            dirs = ["北", "東北", "東", "東南", "南", "西南", "西", "西北"]
            idx = round(h / 45) % 8
            direction_text = f"朝{dirs[idx]}方向前進 ({heading:.1f}°)"

            topo.add_walkway_edge(
                nid_a, nid_b,
                direction=direction_text,
                distance_m=seg_dist if seg_dist > 0 else None,
                steps=seg_steps if seg_steps > 0 else None,
            )

        print(f"[{sess_name}] {len(node_list)} 節點, {len(node_list)-1} 邊")

    # cross-session merging: find nearby nodes and add bidirectional edges
    all_sessions = list(session_nodes.keys())
    merge_count = 0
    for i in range(len(all_sessions)):
        for j in range(i + 1, len(all_sessions)):
            sa, sb = all_sessions[i], all_sessions[j]
            for nid_a, xy_a in session_nodes[sa]:
                for nid_b, xy_b in session_nodes[sb]:
                    d = _dist(xy_a, xy_b)
                    if d <= merge_radius:
                        topo.add_walkway_edge(
                            nid_a, nid_b,
                            direction=f"同位置連接 ({sa}↔{sb}, {d:.1f}m)",
                            distance_m=d,
                        )
                        topo.add_walkway_edge(
                            nid_b, nid_a,
                            direction=f"同位置連接 ({sb}↔{sa}, {d:.1f}m)",
                            distance_m=d,
                        )
                        merge_count += 1

    print(f"\n[合併] 跨 session 連接: {merge_count} 組 (半徑 {merge_radius}m)")

    # set position
    first_session = list(session_nodes.values())[0]
    if first_session:
        topo.set_start_position(first_session[0][0])
        for nid, _ in first_session[1:]:
            topo.move_to(nid)

    return topo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", required=True,
                        help="對齊後的 waypoints JSON")
    parser.add_argument("--place", default="A7家樂福",
                        help="場所名稱")
    parser.add_argument("--output-root", default=None,
                        help="輸出目錄 (預設: output/topomaps)")
    parser.add_argument("--merge-radius", type=float, default=3.0,
                        help="跨 session 合併半徑 (公尺)")
    args = parser.parse_args()

    aligned_path = Path(args.aligned).resolve()
    output_root = Path(args.output_root) if args.output_root else (
        Path(__file__).resolve().parent / "output" / "topomaps"
    )

    topo = merge(aligned_path, args.place, output_root, args.merge_radius)

    saved_path = topo.save_for_place(output_root)

    try:
        png_path = saved_path.with_suffix(".png")
        png_path.write_bytes(topo.render_png())
    except Exception as e:
        print(f"⚠ PNG 渲染失敗: {e}")
        png_path = None

    n_photo = len(topo.all_photo_nodes())
    n_object = len(topo.all_object_nodes())
    print(f"\n{'='*50}")
    print(f"完成！場所「{args.place}」合併拓樸地圖：")
    print(f"  JSON: {saved_path}")
    if png_path:
        print(f"  PNG: {png_path}")
    print(f"  節點: {topo.graph.number_of_nodes()} (照片 {n_photo} / 物件 {n_object})")
    print(f"  邊:   {topo.graph.number_of_edges()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
