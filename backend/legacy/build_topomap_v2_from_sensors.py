#!/usr/bin/env python3
"""
build_topomap_v2_from_sensors.py — 從離線感測器資料（testPDR 格式）建拓樸地圖 v2

════════════════════════════════════════════════════════════════════════════
用途 (Purpose)
════════════════════════════════════════════════════════════════════════════
讀取 Android SensorNav 離線模式匯出的資料夾（manifest.json + photos/ + sensors/），
使用手機感測器的 PDR（步行航位推算）資料建立走道邊的方向與距離，
取代原本依賴 VLM 方向文字的死算推位。

這支腳本適合搭配 testPDR 資料夾做離線驗證：

    backend/testPDR/snav_商店/local_1785735643308/
        manifest.json           ← 每張照片的 PDR 摘要（pdr_x, pdr_y, heading_deg, ...）
        photos/photo_N.jpg      ← 原始照片
        sensors/sensors_N.json  ← 詳細感測器原始資料
        pdr_path.json           ← 每一步的 x,y 軌跡

════════════════════════════════════════════════════════════════════════════
使用方式 (Usage)
════════════════════════════════════════════════════════════════════════════
    python build_topomap_v2_from_sensors.py \\
        --input testPDR/snav_商店/local_1785735643308 \\
        --place "商店" \\
        [--output-root output/topomaps] \\
        [--fresh]             # 不讀取場所既有地圖，從空地圖開始建
        [--with-vlm]          # 同時用 VLM 做物件偵測與 OCR（需要 API key）

════════════════════════════════════════════════════════════════════════════
建圖邏輯 (Build logic)
════════════════════════════════════════════════════════════════════════════
1. 讀取 manifest.json，取得照片順序與每張照片的 PDR 摘要
2. 依 manifest 順序建立照片節點，把 PDR 位置（pdr_x, pdr_y）與方向
   （heading_deg）存入 sensor_data
3. 相鄰照片之間建立走道邊，距離使用 segment_distance_m 感測器數值
4. 如果啟用 --with-vlm，對每張照片呼叫 VLM 做物件偵測與 OCR，
   建立物件子圖（CONTAINS + ADJACENT 邊）與跨照片同物件關聯（SAME_OBJECT 邊）
5. render_png 時會自動使用 sensor_data 裡的 pdr_x/pdr_y 座標
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.topomap_v2 import TopoGraphV2  # noqa: E402


def _load_manifest(input_dir: Path) -> dict:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_sensor_json(input_dir: Path, sensor_rel: str) -> dict:
    sensor_path = input_dir / sensor_rel
    if not sensor_path.exists():
        return {}
    return json.loads(sensor_path.read_text(encoding="utf-8"))


def _guess_image_size(photo_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image, ImageOps
        with Image.open(photo_path) as im:
            im = ImageOps.exif_transpose(im)
            return im.size
    except Exception:
        return (1600, 1200)


def _heading_text(heading_deg: float) -> str:
    """把感測器方向角轉成人可讀的方向描述。"""
    h = heading_deg % 360
    if h < 0:
        h += 360
    dirs = ["北", "東北", "東", "東南", "南", "西南", "西", "西北"]
    idx = round(h / 45) % 8
    return f"朝{dirs[idx]}方向前進 (sensor: {heading_deg:.1f}°)"


def _detections_cache_path(input_dir: Path, index: int) -> Path:
    return input_dir / "detections" / f"detections_{index}.json"


def _load_cached_detections(input_dir: Path, index: int) -> Optional[dict]:
    """讀取已快取的偵測結果，沒有就回傳 None。"""
    p = _detections_cache_path(input_dir, index)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save_detections(input_dir: Path, index: int, detections: list, ocr_items: list) -> None:
    cache_dir = input_dir / "detections"
    cache_dir.mkdir(exist_ok=True)
    data = {"detections": detections, "ocr": ocr_items}
    _detections_cache_path(input_dir, index).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _run_vlm_perception(photo_path: str, img_w: int, img_h: int) -> tuple[list, list]:
    """呼叫 VLM 做 Stage 1 感知（物件偵測 + OCR），回傳 (detections, ocr_items)。"""
    try:
        from server.vlm import perceive_and_decide
    except ImportError as e:
        print(f"[警告] 無法載入 VLM 模組：{e}", file=sys.stderr)
        return [], []

    try:
        perception, _nav = perceive_and_decide(
            image_path=photo_path,
            goal="(mapping)",
            goal_objects=[],
            topomap_summary="(offline mapping)",
            img_w=img_w, img_h=img_h,
            prior_question=None, prior_answer=None,
        )
    except Exception as e:
        print(f"[警告] VLM 感知失敗 ({photo_path}): {e}", file=sys.stderr)
        return [], []

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
    return detections, ocr_items


_gdino_engine = None  # lazy singleton
_ocr_engine = None    # lazy singleton
_ocr_enabled = False  # controlled by --with-ocr flag

def _get_gdino_engine():
    global _gdino_engine
    if _gdino_engine is None:
        from server.perception import Perception
        _gdino_engine = Perception()
        _gdino_engine.load()
    return _gdino_engine


_gdino_scene = "indoor"

def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from server.ocr import OCR
        _ocr_engine = OCR()
        _ocr_engine.load()
    return _ocr_engine


def _run_gdino_perception(photo_path: str) -> tuple[list, list]:
    """用本地 GroundingDINO 做物件偵測 + EasyOCR 文字辨識。"""
    from server.config import GENERIC_INDOOR_OBJECTS, SUPERMARKET_OBJECTS
    prompt_classes = SUPERMARKET_OBJECTS if _gdino_scene == "supermarket" else GENERIC_INDOOR_OBJECTS

    detections = []
    try:
        engine = _get_gdino_engine()
        results = engine.detect(photo_path, prompt_classes)
        detections = [
            {
                "label": d.label,
                "score": d.score,
                "box": d.box,
                "position": d.position,
            }
            for d in results
        ]
    except Exception as e:
        print(f"[警告] GroundingDINO 偵測失敗 ({photo_path}): {e}", file=sys.stderr)

    ocr_items = []
    if _ocr_enabled:
        try:
            ocr_engine = _get_ocr_engine()
            ocr_results = ocr_engine.read(photo_path)
            ocr_items = [
                {"text": r.text, "confidence": r.confidence, "bbox": r.bbox}
                for r in ocr_results
            ]
        except Exception as e:
            print(f"[警告] EasyOCR 辨識失敗 ({photo_path}): {e}", file=sys.stderr)

    return detections, ocr_items


def _get_detections(
    input_dir: Path, index: int, photo_path: Path,
    img_w: int, img_h: int, force_vlm: bool,
    mode: str = "vlm",
) -> tuple[list, list, str]:
    """取得偵測結果：優先讀快取，沒有才呼叫 VLM 或 GroundingDINO。
    mode: "vlm" 或 "gdino"
    回傳 (detections, ocr_items, source_label)。"""
    if not force_vlm:
        cached = _load_cached_detections(input_dir, index)
        if cached is not None:
            return cached.get("detections", []), cached.get("ocr", []), "快取"

    if not photo_path.exists():
        return [], [], "無照片"

    if mode == "gdino":
        detections, ocr_items = _run_gdino_perception(str(photo_path))
        source = "GDINO"
    else:
        detections, ocr_items = _run_vlm_perception(str(photo_path), img_w, img_h)
        source = "VLM"

    if detections or ocr_items:
        _save_detections(input_dir, index, detections, ocr_items)
    return detections, ocr_items, source


def build_from_sensor_session(
    input_dir: Path,
    place_name: str,
    output_root: Path,
    fresh: bool = False,
    with_vlm: bool = False,
    with_gdino: bool = False,
    force_vlm: bool = False,
) -> TopoGraphV2:
    manifest = _load_manifest(input_dir)
    entries = manifest.get("entries", [])
    if not entries:
        raise ValueError("manifest.json 沒有任何 entries")

    topo = TopoGraphV2(place_name) if fresh else TopoGraphV2.load_for_place(place_name, output_root)
    topo.place_name = place_name

    # ── 0. 計算航向旋轉補償（多 session 對齊）────────────────────
    #   每次 session 起始面朝方向可能不同，PDR 座標系因此旋轉。
    #   以第一次 session 的起始 heading 為基準，把後續 session
    #   的所有 (pdr_x, pdr_y) 旋轉 Δ = ref_heading − session_heading
    rotation_rad = 0.0
    ordered_entries = sorted(entries, key=lambda e: e["index"])
    session_heading = ordered_entries[0].get("heading_deg", 0.0)

    if not fresh:
        ref_heading: Optional[float] = None
        for nid_existing in sorted(topo.graph.nodes):
            nd = topo.graph.nodes[nid_existing]
            if nd.get("ntype") == "photo":
                sd = nd.get("sensor_data") or {}
                if "heading_deg" in sd:
                    ref_heading = sd["heading_deg"]
                    break
        if ref_heading is not None:
            delta_deg = ref_heading - session_heading
            rotation_rad = math.radians(delta_deg)
            if abs(delta_deg) > 1.0:
                print(f"[對齊] 基準 heading={ref_heading:.1f}°, "
                      f"本次={session_heading:.1f}°, "
                      f"旋轉 Δ={delta_deg:.1f}°")

    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)

    def _rotate_pdr(x: float, y: float) -> tuple:
        return (x * cos_r - y * sin_r, x * sin_r + y * cos_r)

    # ── 1. 建立照片節點，帶入 PDR 感測器資料 ─────────────────────
    entry_to_node: Dict[int, int] = {}
    photo_object_ids: Dict[int, List[int]] = {}

    for entry in ordered_entries:
        idx = entry["index"]
        photo_rel = entry.get("photo", f"photos/photo_{idx}.jpg")
        photo_path = input_dir / photo_rel

        rx, ry = _rotate_pdr(entry.get("pdr_x", 0.0), entry.get("pdr_y", 0.0))
        sensor_data = {
            "pdr_x": rx,
            "pdr_y": ry,
            "heading_deg": entry.get("heading_deg", 0.0),
            "total_steps": entry.get("steps", 0),
            "total_distance_m": entry.get("distance_m", 0.0),
            "segment_steps": entry.get("segment_steps", 0),
            "segment_distance_m": entry.get("segment_distance_m", 0.0),
        }
        for extra_key in ("calibrated_yaw_deg", "game_rot_vec_yaw_deg",
                          "rot_vec_yaw_deg", "azimuth_deg",
                          "gps_lat", "gps_lng", "gps_accuracy"):
            if extra_key in entry:
                sensor_data[extra_key] = entry[extra_key]

        nid = topo.add_photo_node(
            str(photo_path) if photo_path.exists() else "",
            timestamp=entry.get("timestamp"),
            sensor_data=sensor_data,
        )
        entry_to_node[idx] = nid

        # ── 物件偵測：優先讀快取，沒有才呼叫 VLM / GroundingDINO ──
        has_cache = _load_cached_detections(input_dir, idx) is not None
        detect_mode = "gdino" if with_gdino else "vlm"
        if with_vlm or with_gdino or has_cache:
            img_w, img_h = _guess_image_size(photo_path) if photo_path.exists() else (1600, 1200)
            detections, ocr_items, src = _get_detections(
                input_dir, idx, photo_path, img_w, img_h,
                force_vlm=force_vlm, mode=detect_mode,
            )
            if detections:
                object_ids = topo.build_subgraph_from_detections(
                    nid, detections, img_w, img_h, ocr_items=ocr_items,
                )
                photo_object_ids[nid] = object_ids
                print(f"[節點] index={idx} → id={nid} | [{src}] {len(detections)} 物件, {len(ocr_items)} OCR")
            else:
                photo_object_ids[nid] = []
                print(f"[節點] index={idx} → id={nid} | [{src}] 未偵測到物件")
        else:
            photo_object_ids[nid] = []
            print(f"[節點] index={idx} → id={nid} | "
                  f"PDR ({sensor_data['pdr_x']:.1f}, {sensor_data['pdr_y']:.1f}) "
                  f"heading={sensor_data['heading_deg']:.1f}°")

    # ── 2. 建立走道邊（用感測器距離與方向）──────────────────────
    ordered_indices = [e["index"] for e in ordered_entries]
    for i in range(len(ordered_indices) - 1):
        idx_a = ordered_indices[i]
        idx_b = ordered_indices[i + 1]
        nid_a = entry_to_node[idx_a]
        nid_b = entry_to_node[idx_b]

        entry_b = ordered_entries[i + 1]
        seg_dist = entry_b.get("segment_distance_m", 0.0)
        seg_steps = entry_b.get("segment_steps", 0)
        heading_a = ordered_entries[i].get("heading_deg", 0.0)
        direction_text = _heading_text(heading_a)

        topo.add_walkway_edge(
            nid_a, nid_b,
            direction=direction_text,
            distance_m=seg_dist if seg_dist > 0 else None,
            steps=seg_steps if seg_steps > 0 else None,
        )

    # ── 3. 跨照片同物件關聯（有偵測結果時才做）──────────────────
    ordered_nids = [entry_to_node[idx] for idx in ordered_indices]
    for a, b in zip(ordered_nids, ordered_nids[1:]):
        if photo_object_ids.get(a) and photo_object_ids.get(b):
            linked = topo.link_same_objects_between_photos(
                photo_object_ids[a], photo_object_ids[b],
            )
            print(f"[同物件關聯] 節點 {a} <-> {b}：配對 {len(linked)} 組")

    # ── 4. 銜接場所舊地圖 ──
    ordered_nids = [entry_to_node[idx] for idx in ordered_indices]
    if not fresh and topo.position_history:
        prev_last = topo.current_position
        first_new = ordered_nids[0]
        if prev_last is not None and prev_last != first_new:
            topo.add_walkway_edge(prev_last, first_new, direction="（銜接上次導航的終點）")
            prev_obj_ids = [o["id"] for o in topo.photo_objects(prev_last)]
            if prev_obj_ids and photo_object_ids.get(first_new):
                linked = topo.link_same_objects_between_photos(
                    prev_obj_ids, photo_object_ids[first_new],
                )
                print(f"[銜接舊地圖] 節點 {prev_last} <-> {first_new}：配對 {len(linked)} 組")

    # ── 5. 設定使用者位置 ──
    remaining = list(ordered_nids)
    if topo.current_position is None:
        topo.set_start_position(remaining.pop(0))
    elif topo.current_position == remaining[0]:
        remaining.pop(0)
    for nid in remaining:
        topo.move_to(nid)

    # ── 6. 標記終點拍照方向 ──
    if topo.current_position is not None:
        topo.set_terminal_facing(topo.current_position)

    return topo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="離線感測器資料夾，例如 testPDR/snav_商店/local_1785735643308")
    parser.add_argument("--place", required=True,
                        help="場所名稱，例如「商店」")
    parser.add_argument("--output-root", default=None,
                        help="拓樸地圖存放的根目錄（預設：<專案根目錄>/output/topomaps）")
    parser.add_argument("--fresh", action="store_true",
                        help="不讀取場所既有地圖，從空地圖開始建")
    parser.add_argument("--with-vlm", action="store_true",
                        help="用 VLM 做物件偵測與 OCR（結果會快取到 detections/ 下次自動讀取）")
    parser.add_argument("--with-gdino", action="store_true",
                        help="用本地 GroundingDINO 做物件偵測（不需 API key，需要模型權重）")
    parser.add_argument("--scene", default="indoor",
                        choices=["indoor", "supermarket"],
                        help="GroundingDINO 偵測場景：indoor（預設）或 supermarket（超市專用提示詞）")
    parser.add_argument("--with-ocr", action="store_true",
                        help="同時用 EasyOCR 做文字辨識（需要較多記憶體，可分開跑）")
    parser.add_argument("--force-vlm", action="store_true",
                        help="強制重跑 VLM/GDINO（忽略快取）")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        raise SystemExit(f"輸入資料夾不存在：{input_dir}")

    output_root = Path(args.output_root) if args.output_root else (
        Path(__file__).resolve().parent / "output" / "topomaps"
    )

    global _gdino_scene, _ocr_enabled
    _gdino_scene = args.scene
    _ocr_enabled = args.with_ocr

    topo = build_from_sensor_session(
        input_dir, args.place, output_root,
        fresh=args.fresh, with_vlm=args.with_vlm,
        with_gdino=args.with_gdino, force_vlm=args.force_vlm,
    )

    saved_path = topo.save_for_place(output_root)
    png_path = saved_path.with_suffix(".png")
    try:
        png_path.write_bytes(topo.render_png())
    except Exception as e:
        print(f"⚠ 無法寫入 PNG（{e}）")

    debug_dir = input_dir / "topomap_v2"
    debug_dir.mkdir(exist_ok=True)
    topo.save(debug_dir / "topomap.json")
    try:
        (debug_dir / "topomap.png").write_bytes(topo.render_png())
    except OSError as e:
        print(f"⚠ 無法寫入 debug PNG（{e}），跳過")

    n_photo = len(topo.all_photo_nodes())
    n_object = len(topo.all_object_nodes())
    print("\n════════════════════════════════════════")
    print(f"完成！場所「{args.place}」的感測器拓樸地圖已建立：")
    print(f"  - {saved_path}")
    print(f"  - {png_path}")
    print(f"  - 節點數：{topo.graph.number_of_nodes()}（照片 {n_photo} / 物件 {n_object}）")
    print(f"  - 邊數：{topo.graph.number_of_edges()}")
    detect_label = "+ GroundingDINO 物件偵測" if args.with_gdino else ("+ VLM 物件偵測" if args.with_vlm else "純路線骨架")
    print(f"  - 資料來源：感測器 PDR（{detect_label}）")
    print("════════════════════════════════════════")


if __name__ == "__main__":
    main()
