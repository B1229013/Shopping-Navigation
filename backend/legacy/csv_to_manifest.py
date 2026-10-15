#!/usr/bin/env python3
"""
csv_to_manifest.py — 將 SensorLab sensor_log.csv 格式轉成建圖工具需要的 manifest.json

用法:
    python csv_to_manifest.py backend/testPDR/0819全聯1/

    # 過濾零距離 waypoint（原地拍照不走路的點）
    python csv_to_manifest.py --skip-zero-dist backend/testPDR/0829全聯2/

    # 選擇 heading 來源：cal_yaw_deg（預設）、grv_yaw_deg、rv_yaw_deg
    python csv_to_manifest.py --heading grv_yaw_deg backend/testPDR/0829全聯2/

會在資料夾內產生 manifest.json，照片路徑指向 photos/ 裡的 wp_N_*.jpg
"""
import argparse
import json
import math
import os
import re
import sys
from glob import glob


def _apply_anchor_piecewise(entries, anchor_pairs):
    """Piecewise-linear drift correction using anchor point constraints.

    anchor_pairs: list of (idx_a, idx_b) where idx_a < idx_b and both are
    the same physical location. Groups like 1=29=85 are split into pairs
    (1,29) and (29,85) by the caller.
    """
    idx_set = set()
    target_of = {}
    for a, b in anchor_pairs:
        idx_set.update([a, b])
        target_of[b] = a

    all_indices = sorted(idx_set)
    idx_map = {e["index"]: i for i, e in enumerate(entries)}
    raw = {idx: (entries[idx_map[idx]]["pdr_x"], entries[idx_map[idx]]["pdr_y"])
           for idx in all_indices if idx in idx_map}

    c = {all_indices[0]: (0.0, 0.0)}

    def _interp_c(idx):
        known = sorted(c.keys())
        prev_a = max([a for a in known if a <= idx], default=None)
        next_a = min([a for a in known if a > idx], default=None)
        if prev_a is not None and next_a is not None:
            t = (idx - prev_a) / (next_a - prev_a)
            return (c[prev_a][0] + (c[next_a][0] - c[prev_a][0]) * t,
                    c[prev_a][1] + (c[next_a][1] - c[prev_a][1]) * t)
        if prev_a is not None:
            return c[prev_a]
        return (0.0, 0.0)

    for idx in sorted(target_of.keys()):
        ref = target_of[idx]
        if ref not in c:
            c[ref] = _interp_c(ref)
        c[idx] = (raw[idx][0] - raw[ref][0] + c[ref][0],
                  raw[idx][1] - raw[ref][1] + c[ref][1])

    for idx in all_indices:
        if idx not in c:
            c[idx] = _interp_c(idx)

    sorted_c = sorted(c.items())
    for a_idx, (cx, cy) in sorted_c:
        d = math.sqrt(cx * cx + cy * cy)
        print(f"  Anchor c({a_idx})=({cx:.2f}, {cy:.2f}) {d:.2f}m")

    for entry in entries:
        eidx = entry["index"]
        prev_a = next_a = None
        for a_idx, a_c in sorted_c:
            if a_idx <= eidx:
                prev_a = (a_idx, a_c)
            if a_idx >= eidx and next_a is None:
                next_a = (a_idx, a_c)
        if prev_a and next_a and prev_a[0] != next_a[0]:
            t = (eidx - prev_a[0]) / (next_a[0] - prev_a[0])
            cx = prev_a[1][0] + (next_a[1][0] - prev_a[1][0]) * t
            cy = prev_a[1][1] + (next_a[1][1] - prev_a[1][1]) * t
        elif prev_a:
            cx, cy = prev_a[1]
        elif next_a:
            cx, cy = next_a[1]
        else:
            cx, cy = 0.0, 0.0
        entry["pdr_x"] = round(entry["pdr_x"] - cx, 4)
        entry["pdr_y"] = round(entry["pdr_y"] - cy, 4)
        entry["distance_m"] = round(math.sqrt(entry["pdr_x"]**2 + entry["pdr_y"]**2), 4)

    for i in range(1, len(entries)):
        dx = entries[i]["pdr_x"] - entries[i-1]["pdr_x"]
        dy = entries[i]["pdr_y"] - entries[i-1]["pdr_y"]
        entries[i]["segment_distance_m"] = round(math.sqrt(dx * dx + dy * dy), 4)
    print(f"  Anchor correction applied: {len(anchor_pairs)} constraint(s)")


def csv_to_manifest(data_dir, skip_zero_dist=False, heading_col="cal_yaw_deg",
                    loop_closure=False, anchors=None):
    csv_path = os.path.join(data_dir, "sensor_log.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata = {}
    header_line = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.search(r"(\d{8}_\d{6})", line)
            if m:
                metadata["export_time"] = m.group(1)
        else:
            header_line = i
            break

    headers = lines[header_line].strip().split(",")

    waypoint_rows = []
    prev_steps = 0
    prev_pdr_x = 0.0
    prev_pdr_y = 0.0

    for line in lines[header_line + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        row = {}
        for j, h in enumerate(headers):
            if j >= len(parts):
                row[h] = ""
                continue
            val = parts[j].strip().strip('"')
            row[h] = val
        if row.get("waypoint"):
            waypoint_rows.append(row)

    photo_files = sorted(glob(os.path.join(data_dir, "photos", "wp_*_*.jpg")))
    photo_by_index = {}       # idx → path (single-photo format) or front photo
    photo_dirs_by_index = {}  # idx → {"front":path, "right":path, ...}
    for pf in photo_files:
        basename = os.path.basename(pf)
        # 四方向格式: wp_N_DIRECTION_TIMESTAMP.jpg
        m_dir = re.match(r"wp_(\d+)_(front|right|back|left)_(\d+)\.jpg", basename)
        if m_dir:
            idx = int(m_dir.group(1))
            direction = m_dir.group(2)
            if idx not in photo_dirs_by_index:
                photo_dirs_by_index[idx] = {}
            photo_dirs_by_index[idx][direction] = pf
            if direction == "front":
                photo_by_index[idx] = pf
            continue
        # 單張格式: wp_N_TIMESTAMP.jpg
        m = re.match(r"wp_(\d+)_(\d+)\.jpg", basename)
        if m:
            idx = int(m.group(1))
            ts = int(m.group(2))
            if idx not in photo_by_index:
                photo_by_index[idx] = pf
            else:
                existing_ts = int(re.match(r"wp_\d+_(\d+)\.jpg", os.path.basename(photo_by_index[idx])).group(1))
                wp_ts = int(waypoint_rows[idx - 1]["timestamp_ms"]) if idx - 1 < len(waypoint_rows) else 0
                if abs(ts - wp_ts) < abs(existing_ts - wp_ts):
                    photo_by_index[idx] = pf

    entries = []
    skipped = 0
    new_idx = 0
    for i, row in enumerate(waypoint_rows):
        orig_idx = i + 1
        try:
            steps = int(row.get("total_steps", 0))
            pdr_x = float(row.get("pdr_x", 0))
            pdr_y = float(row.get("pdr_y", 0))
            heading = float(row.get(heading_col, 0))
            timestamp_ms = int(row.get("timestamp_ms", 0))
            elapsed = float(row.get("elapsed_sec", 0))
        except (ValueError, TypeError):
            steps, pdr_x, pdr_y, heading = 0, 0.0, 0.0, 0.0
            timestamp_ms, elapsed = 0, 0.0

        seg_steps = steps - prev_steps
        seg_dx = pdr_x - prev_pdr_x
        seg_dy = pdr_y - prev_pdr_y
        seg_dist = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

        if skip_zero_dist and seg_steps == 0 and entries:
            skipped += 1
            continue

        new_idx += 1

        photo_rel = ""
        if orig_idx in photo_by_index:
            photo_rel = os.path.relpath(photo_by_index[orig_idx], data_dir).replace("\\", "/")

        photos_map = {}
        if orig_idx in photo_dirs_by_index:
            for direction, ppath in photo_dirs_by_index[orig_idx].items():
                photos_map[direction] = os.path.relpath(ppath, data_dir).replace("\\", "/")

        entry = {
            "index": new_idx,
            "photo": photo_rel,
            "sensor": "",
            "timestamp": f"{metadata.get('export_time', '')}+{elapsed:.1f}s",
            "steps": steps,
            "distance_m": round(math.sqrt(pdr_x * pdr_x + pdr_y * pdr_y), 4),
            "pdr_x": round(pdr_x, 4),
            "pdr_y": round(pdr_y, 4),
            "heading_deg": round(heading, 4),
            "segment_steps": seg_steps,
            "segment_distance_m": round(seg_dist, 4),
            "calibrated_yaw_deg": round(float(row.get("cal_yaw_deg", 0)), 4),
        }

        if photos_map:
            entry["photos"] = photos_map

        try:
            entry["rot_vec_yaw_deg"] = round(float(row.get("rv_yaw_deg", 0)), 4)
            entry["game_rot_vec_yaw_deg"] = round(float(row.get("grv_yaw_deg", 0)), 4)
        except (ValueError, TypeError):
            pass

        entries.append(entry)
        prev_steps = steps
        prev_pdr_x = pdr_x
        prev_pdr_y = pdr_y

    if loop_closure and len(entries) >= 2:
        last = entries[-1]
        drift_x = last["pdr_x"]
        drift_y = last["pdr_y"]
        total_elapsed = float(waypoint_rows[-1].get("elapsed_sec", 1)) or 1.0
        print(f"  Loop closure: drift=({drift_x:.2f}, {drift_y:.2f}), distributing over {len(entries)} entries")
        for e in entries:
            t = float(e["timestamp"].split("+")[-1].rstrip("s")) if "+" in e["timestamp"] else 0
            ratio = t / total_elapsed
            e["pdr_x"] = round(e["pdr_x"] - drift_x * ratio, 4)
            e["pdr_y"] = round(e["pdr_y"] - drift_y * ratio, 4)
            e["distance_m"] = round(math.sqrt(e["pdr_x"]**2 + e["pdr_y"]**2), 4)
        for i in range(1, len(entries)):
            dx = entries[i]["pdr_x"] - entries[i-1]["pdr_x"]
            dy = entries[i]["pdr_y"] - entries[i-1]["pdr_y"]
            entries[i]["segment_distance_m"] = round(math.sqrt(dx*dx + dy*dy), 4)

    if anchors and len(anchors) >= 1:
        _apply_anchor_piecewise(entries, anchors)

    dir_name = os.path.basename(data_dir.rstrip("/\\"))
    manifest = {
        "id": f"sensorlab_{dir_name}",
        "goal": dir_name,
        "created_at": metadata.get("export_time", ""),
        "heading_source": heading_col,
        "loop_closure": loop_closure,
        "anchors": [[a, b] for a, b in (anchors or [])],
        "entries": entries,
    }

    out_path = os.path.join(data_dir, "manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Generated {out_path} with {len(entries)} entries"
          f" (skipped {skipped} zero-distance)" if skipped else
          f"Generated {out_path} with {len(entries)} entries")
    for e in entries:
        print(f"  #{e['index']:2d}  steps={e['steps']:3d}  "
              f"({e['pdr_x']:7.2f}, {e['pdr_y']:7.2f})  "
              f"heading={e['heading_deg']:.1f}°  "
              f"seg={e['segment_distance_m']:.1f}m  "
              f"photo={'OK' if e['photo'] else 'MISSING'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="資料夾路徑")
    parser.add_argument("--skip-zero-dist", action="store_true",
                        help="跳過 segment_distance=0 的 waypoint（原地拍照不走路）")
    parser.add_argument("--heading", default="cal_yaw_deg",
                        choices=["cal_yaw_deg", "grv_yaw_deg", "rv_yaw_deg"],
                        help="使用的 heading 來源（預設 cal_yaw_deg）")
    parser.add_argument("--loop-closure", action="store_true",
                        help="起終點相同時，線性分配漂移誤差（loop closure correction）")
    parser.add_argument("--anchor", action="append", nargs=2, type=int, metavar=("A", "B"),
                        help="錨點對：entry A 和 entry B 是同一位置（可多次指定）")
    args = parser.parse_args()
    anchor_list = [tuple(a) for a in args.anchor] if args.anchor else None
    for d in args.dirs:
        csv_to_manifest(d.rstrip("/\\"), skip_zero_dist=args.skip_zero_dist,
                        heading_col=args.heading, loop_closure=args.loop_closure,
                        anchors=anchor_list)
