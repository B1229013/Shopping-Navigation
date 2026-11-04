#!/usr/bin/env python3
"""
rerun_vlm_detections.py — 對已有路線骨架的地圖，重跑 VLM 物件偵測

保留現有的照片節點和走道邊（路線骨架），只清除物件節點和相關邊，
用 EXIF 修正後的影像重新呼叫 VLM 偵測，重建物件子圖後上傳回 Neo4j。

用法：
    python rerun_vlm_detections.py --place "樂家康超市 龜山文青店"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageOps


def _exif_image_size(photo_path: str) -> Tuple[int, int]:
    """取得 EXIF 轉正後的影像尺寸。"""
    try:
        with Image.open(photo_path) as im:
            im = ImageOps.exif_transpose(im)
            return im.size
    except Exception:
        return (1600, 1200)


def main():
    parser = argparse.ArgumentParser(description="重跑 VLM 物件偵測（保留路線骨架）")
    parser.add_argument("--place", required=True, help="Neo4j 中的場所名稱")
    parser.add_argument("--dry-run", action="store_true", help="只顯示會做什麼，不實際執行")
    args = parser.parse_args()

    from server.neo4j_map_store import Neo4jMapStore
    from server.topomap_v2 import TopoGraphV2

    # ── 1. 從 Neo4j 下載現有地圖 ──
    print(f"[1/5] 從 Neo4j 下載地圖「{args.place}」...")
    store = Neo4jMapStore()
    topo = store.download(args.place)
    if topo is None:
        print(f"錯誤：在 Neo4j 中找不到場所「{args.place}」")
        store.close()
        return

    photo_nodes = sorted(topo.all_photo_nodes())
    object_nodes = sorted(topo.all_object_nodes())
    print(f"  下載完成：{len(photo_nodes)} 張照片, {len(object_nodes)} 個物件節點")

    # ── 2. 檢查照片檔案是否存在 ──
    print(f"\n[2/5] 檢查照片檔案...")
    missing = []
    photo_paths = {}
    for pid in photo_nodes:
        pp = topo.graph.nodes[pid].get("photo_path", "")
        if pp and Path(pp).is_file():
            photo_paths[pid] = pp
        else:
            missing.append(pid)
    print(f"  可用：{len(photo_paths)} 張, 缺少：{len(missing)} 張")
    if missing:
        print(f"  缺少的照片 ID: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    # ── 3. 清除所有物件節點（保留照片節點和走道邊）──
    print(f"\n[3/5] 清除 {len(object_nodes)} 個舊物件節點...")
    if not args.dry_run:
        for oid in object_nodes:
            topo.graph.remove_node(oid)
        print(f"  清除完成，剩餘 {topo.graph.number_of_nodes()} 個節點（純照片骨架）")
    else:
        print(f"  [DRY RUN] 跳過")

    # ── 4. 重跑 VLM 偵測 ──
    print(f"\n[4/5] 重跑 VLM 物件偵測（含 EXIF 轉正）...")
    try:
        from server.vlm import perceive_only
    except ImportError as e:
        print(f"錯誤：無法載入 VLM 模組：{e}")
        store.close()
        return

    total_objects = 0
    photo_object_map = {}

    for i, pid in enumerate(photo_nodes):
        pp = photo_paths.get(pid)
        if not pp:
            print(f"  [{i+1}/{len(photo_nodes)}] photo {pid}: 照片不存在，跳過")
            continue

        img_w, img_h = _exif_image_size(pp)
        print(f"  [{i+1}/{len(photo_nodes)}] photo {pid}: {Path(pp).name} ({img_w}x{img_h})", end="", flush=True)

        if args.dry_run:
            print(" [DRY RUN]")
            continue

        try:
            perception = perceive_only(
                image_path=pp,
                goal_objects=[],
                img_w=img_w,
                img_h=img_h,
            )
        except Exception as e:
            print(f" VLM 失敗: {e}")
            continue

        detections = [
            {
                "label": d.label,
                "score": d.score,
                "box": d.bbox,
                "position": d.position,
            }
            for d in perception.detections
        ]
        ocr_items = [
            {
                "text": o.text,
                "bbox": o.bbox,
                "score": o.score,
                "position": o.position,
            }
            for o in perception.ocr_texts
        ]

        if detections:
            obj_ids = topo.build_subgraph_from_detections(
                pid, detections, img_w, img_h, ocr_items=ocr_items,
            )
            photo_object_map[pid] = obj_ids
            total_objects += len(obj_ids)

        print(f" → {len(detections)} 物件, {len(ocr_items)} OCR")

    # ── 4.5 跨照片同物件關聯 ──
    if not args.dry_run and photo_object_map:
        print(f"\n  建立跨照片同物件關聯...")
        walkway_pairs = [
            (u, v) for u, v, d in topo.graph.edges(data=True)
            if d.get("etype") == "walkway"
            and u in photo_object_map and v in photo_object_map
        ]
        same_obj_count = 0
        for u, v in walkway_pairs:
            links = topo.link_same_objects_between_photos(
                photo_object_map[u], photo_object_map[v],
            )
            same_obj_count += len(links)
        print(f"  建立了 {same_obj_count} 條 SAME_OBJECT 邊")

    if args.dry_run:
        print(f"\n[DRY RUN] 完成，不會上傳。")
        store.close()
        return

    print(f"\n  總計：{total_objects} 個物件節點")

    # ── 5. 上傳回 Neo4j ──
    print(f"\n[5/5] 上傳回 Neo4j（place={args.place}）...")
    n_uploaded = store.upload(args.place, topo)
    print(f"  上傳完成：{n_uploaded} 個節點")

    # 也存一份本地備份
    backup_path = Path(__file__).resolve().parent / "output" / "topomaps" / "樂家康_rerun"
    backup_path.mkdir(parents=True, exist_ok=True)
    topo.save(backup_path / "topomap.json")
    try:
        (backup_path / "topomap.png").write_bytes(topo.render_png())
        print(f"  本地備份：{backup_path}")
    except Exception as e:
        print(f"  ⚠ PNG 備份失敗：{e}")

    store.close()
    print("\n完成！")


if __name__ == "__main__":
    main()
