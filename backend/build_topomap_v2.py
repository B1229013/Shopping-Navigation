#!/usr/bin/env python3
"""
build_topomap_v2.py — 從既有的導航輸出資料夾離線生成拓樸地圖 v2
Build a place-level TopoGraphV2 offline from an existing navigation
output folder, without touching the running server.

════════════════════════════════════════════════════════════════════════════
用途 (Purpose)
════════════════════════════════════════════════════════════════════════════
目前系統每次導航會把資料存在：

    output/sessions/{session_id}/
        photo/{node_id}.jpg                       ← 原始照片
        annotated/{node_id}.jpg                    ← 標註後照片
        annotated/detections_{node_id}.json         ← 該張照片的物件偵測 + OCR
        map/map_{node_id}.json                      ← 累積到該節點為止的舊版拓樸圖
        map/map_{node_id}.png
        graph/scene_graph_{node_id}.png

這支腳本直接讀這個資料夾（不需要啟動 FastAPI server、不需要呼叫任何模型），
把它轉成筆記中設計的「拓樸地圖 v2」（兩種節點、四種邊、子圖結構），
並依「場所」名稱存到 output/topomaps/{場所}/topomap.json，
下次同地點導航時可以繼續讀取、更新。

════════════════════════════════════════════════════════════════════════════
使用方式 (Usage)
════════════════════════════════════════════════════════════════════════════
    python build_topomap_v2.py \\
        --input output/sessions/<session_id> \\
        --place "資工系系辦" \\
        [--output-root output/topomaps] \\
        [--fresh]                # 不讀取場所既有地圖，從空地圖開始建

沒有真實資料也可以先用 tests_fixtures/make_sample_session.py 產生一份
範例輸出資料夾來測試整條流程（見該檔案說明）。

════════════════════════════════════════════════════════════════════════════
建圖邏輯 (Build logic)
════════════════════════════════════════════════════════════════════════════
1. 讀取資料夾中最新（node_id 最大）的 map/map_*.json，取得：
   - 這次導航依序拍了哪些照片節點（nodes，含時間戳）
   - 每個節點之間的移動方向文字（edges 的 "action" 欄位）
     → 直接對應拓樸地圖 v2 的第四種邊（走道）
2. 對每個照片節點，讀取對應的 annotated/detections_{id}.json，
   取得該張照片的物件偵測與 OCR 結果 → 呼叫
   `TopoGraphV2.build_subgraph_from_detections()` 建出子圖
   （物件節點 + 第一種邊 contains + 第二種邊 adjacent）
3. 對每一對「有走道邊相連」的相鄰照片節點，呼叫
   `link_same_objects_between_photos()` 嘗試建立第三種邊（同物件關聯）
4. 若場所已有舊地圖（--fresh 未指定），把這次導航的節點「接在」舊地圖
   最後一個節點之後，並嘗試對新舊交界處也做一次同物件關聯
   （因為使用者這次導航的起點很可能就是上次逛過的地方）
5. 設定使用者最終位置為最後一個節點，儲存 JSON + PNG
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓 `from server...` 可用

from server.topomap_v2 import TopoGraphV2  # noqa: E402


def _load_latest_map(input_dir: Path) -> dict:
    """讀取 map/ 資料夾裡 node_id 最大的 map_*.json（也就是最完整的那份）。"""
    map_dir = input_dir / "map"
    candidates = sorted(
        map_dir.glob("map_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"在 {map_dir} 找不到任何 map_*.json，無法建圖")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _load_detections(input_dir: Path, node_id: int) -> dict:
    """讀取單一節點的偵測/OCR 結果；找不到時回傳空結構（容錯，不中斷整體建圖）。"""
    det_path = input_dir / "annotated" / f"detections_{node_id}.json"
    if not det_path.exists():
        print(f"[警告] 找不到 {det_path}，此節點將不建立任何物件節點", file=sys.stderr)
        return {"detections": [], "ocr": []}
    return json.loads(det_path.read_text(encoding="utf-8"))


def _guess_image_size(input_dir: Path, node_id: int) -> tuple[int, int]:
    """讀原始照片取得寬高（用來估計 contains 邊的距離）。讀不到就給常見手機解析度。"""
    photo_path = input_dir / "photo" / f"{node_id}.jpg"
    try:
        from PIL import Image
        with Image.open(photo_path) as im:
            return im.size
    except Exception:
        return (1600, 1200)


def _side_paths_manifest(input_dir: Path) -> List[dict]:
    """選用：如果資料夾裡有 side_paths.json，用來標記「看到但沒走的路」。

    格式範例：
        [
          {"from_node": 2, "direction": "往右側走道", "note": "貨架另一側",
           "flank_labels": ["洗衣精"]}
        ]

    `flank_labels` 是希望連到的物件標籤名稱（在該 from_node 的物件節點中比對），
    找不到就略過該筆（不中斷整體建圖）。
    """
    p = input_dir / "side_paths.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def build_from_session_folder(
    input_dir: Path,
    place_name: str,
    output_root: Path,
    fresh: bool = False,
) -> TopoGraphV2:
    old_map = _load_latest_map(input_dir)
    old_nodes: List[dict] = old_map.get("nodes", [])
    old_edges: List[dict] = old_map.get("edges", [])
    if not old_nodes:
        raise ValueError("這份 map json 沒有任何節點，無法建圖")

    topo = TopoGraphV2(place_name) if fresh else TopoGraphV2.load_for_place(place_name, output_root)
    topo.place_name = place_name

    # ── 1. 建立照片節點（依原本的 node id 順序）─────────────────────
    old_id_to_new_id: Dict[int, int] = {}
    photo_object_ids: Dict[int, List[int]] = {}

    for node in sorted(old_nodes, key=lambda n: n["id"]):
        old_photo_path = node.get("photo", "")
        # map json 裡存的可能是絕對路徑；資料夾若被搬動過，改用資料夾內的相對路徑更可靠
        candidate = input_dir / "photo" / f"{node['id']}.jpg"
        photo_path = str(candidate) if candidate.exists() else old_photo_path

        new_id = topo.add_photo_node(
            photo_path,
            timestamp=node.get("timestamp"),
        )
        old_id_to_new_id[node["id"]] = new_id

        det_data = _load_detections(input_dir, node["id"])
        img_w, img_h = _guess_image_size(input_dir, node["id"])
        object_ids = topo.build_subgraph_from_detections(
            new_id, det_data.get("detections", []), img_w, img_h,
            ocr_items=det_data.get("ocr", []),
        )
        photo_object_ids[new_id] = object_ids
        print(f"[節點] 舊 id={node['id']} → 新 id={new_id} | 物件數={len(object_ids)}")

    # ── 2. 建立走道邊（第四種邊）：沿用舊 map 的 action 文字當作方向描述 ──
    ordered_new_ids = [old_id_to_new_id[n["id"]] for n in sorted(old_nodes, key=lambda n: n["id"])]
    edge_by_pair = {(e["from"], e["to"]): e.get("action", "") for e in old_edges}
    for old_from, old_to in zip(
        [n["id"] for n in sorted(old_nodes, key=lambda n: n["id"])],
        [n["id"] for n in sorted(old_nodes, key=lambda n: n["id"])][1:],
    ):
        direction = edge_by_pair.get((old_from, old_to), "")
        topo.add_walkway_edge(
            old_id_to_new_id[old_from], old_id_to_new_id[old_to],
            direction=direction or "(繼續前進)",
        )

    # ── 3. 跨照片同物件關聯（第三種邊）：只在有走道相連的相鄰節點間嘗試 ──
    for a, b in zip(ordered_new_ids, ordered_new_ids[1:]):
        linked = topo.link_same_objects_between_photos(photo_object_ids[a], photo_object_ids[b])
        print(f"[同物件關聯] 節點 {a} ↔ {b}：配對 {len(linked)} 組")

    # ── 4. 銜接場所舊地圖：把這次的起點接到舊地圖最後的使用者位置 ──
    if not fresh and topo.position_history:
        prev_last = topo.current_position
        first_new = ordered_new_ids[0]
        if prev_last is not None and prev_last != first_new:
            topo.add_walkway_edge(prev_last, first_new, direction="（銜接上次導航的終點）")
            prev_last_object_ids = [o["id"] for o in topo.photo_objects(prev_last)]
            linked = topo.link_same_objects_between_photos(
                prev_last_object_ids, photo_object_ids[first_new],
            )
            print(f"[銜接舊地圖] 節點 {prev_last} ↔ {first_new}：配對 {len(linked)} 組")

    # ── 5. 選用：處理沒走過的分支（虛假照片節點）──────────────────
    for branch in _side_paths_manifest(input_dir):
        old_from = branch.get("from_node")
        if old_from not in old_id_to_new_id:
            continue
        from_new = old_id_to_new_id[old_from]
        flank_ids = [
            o["id"] for o in topo.photo_objects(from_new)
            if o.get("label") in (branch.get("flank_labels") or [])
        ]
        vid = topo.add_virtual_branch(
            from_new, direction=branch.get("direction", ""),
            note=branch.get("note", "偵測到分支但尚未走訪"),
            flank_object_ids=flank_ids,
        )
        print(f"[虛假節點] 由節點 {from_new} 建立虛假分支節點 {vid}")

    # ── 6. 設定使用者最終位置（沿走道邊依序走過這次導航的所有節點）──
    remaining = list(ordered_new_ids)
    if topo.current_position is None:
        topo.set_start_position(remaining.pop(0))
    elif topo.current_position == remaining[0]:
        remaining.pop(0)  # 起點跟舊地圖的目前位置是同一個節點，不用重複移動
    for nid in remaining:
        topo.move_to(nid)

    # ── 7. 標記整張地圖目前的終點，用自我迴圈邊記錄它的拍照方向 ──
    # 第四種邊（走道）是單向的，方向代表「起點那張照片的拍照方向」，
    # 路徑真正的終點沒有下一步可以承載這個方向，所以額外呼叫
    # set_terminal_facing() 幫它補一條自我迴圈邊。留空 direction 代表
    #「跟走到這裡的方向一樣，沒有再轉」。每次呼叫都會先清掉上一次建圖
    # 留下的舊終點標記，所以不用自己處理「上次的終點現在已經不是終點了」
    # 這件事。
    if topo.current_position is not None:
        topo.set_terminal_facing(topo.current_position)

    return topo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="導航輸出資料夾，例如 output/sessions/<id>")
    parser.add_argument("--place", required=True, help="場所名稱或地址，例如「資工系系辦」")
    parser.add_argument("--output-root", default=None,
                         help="拓樸地圖存放的根目錄（預設：<專案根目錄>/output/topomaps）")
    parser.add_argument("--fresh", action="store_true", help="不讀取場所既有地圖，從空地圖開始建")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        raise SystemExit(f"輸入資料夾不存在：{input_dir}")

    output_root = Path(args.output_root) if args.output_root else (
        Path(__file__).resolve().parent / "output" / "topomaps"
    )

    topo = build_from_session_folder(input_dir, args.place, output_root, fresh=args.fresh)

    saved_path = topo.save_for_place(output_root)
    png_path = saved_path.with_suffix(".png")
    png_path.write_bytes(topo.render_png())

    # 同時在輸入資料夾底下留一份，方便直接對照這次導航的輸出
    debug_dir = input_dir / "topomap_v2"
    debug_dir.mkdir(exist_ok=True)
    topo.save(debug_dir / "topomap.json")
    (debug_dir / "topomap.png").write_bytes(topo.render_png())

    print("\n════════════════════════════════════════")
    print(f"完成！場所「{args.place}」的拓樸地圖已更新：")
    print(f"  - {saved_path}")
    print(f"  - {png_path}")
    print(f"  - 節點數：{topo.graph.number_of_nodes()}"
          f"（照片 {len(topo.all_photo_nodes())} / 物件 {len(topo.all_object_nodes())}）")
    print(f"  - 邊數：{topo.graph.number_of_edges()}")
    print("════════════════════════════════════════")


if __name__ == "__main__":
    main()
