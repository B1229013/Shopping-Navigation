#!/usr/bin/env python3
"""
PDR Sensor Data Analyzer
========================
Reads SensorLab sensor_log.csv exports and generates an interactive HTML report
comparing PDR paths from different heading sources and step detection methods.

Usage:
    python analyze_pdr.py <data_dir> [<data_dir2> ...]
    python analyze_pdr.py backend/testPDR/0819全聯1/

Output:
    <data_dir>/report.html
"""

import csv
import math
import json
import os
import re
import sys
from pathlib import Path


# ── CSV Parsing ──

def parse_sensor_csv(csv_path):
    """Parse sensor_log.csv → (metadata, rows)"""
    metadata = {}
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_line = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "Export" in line:
                m = re.search(r"(\d{8}_\d{6})", line)
                if m:
                    metadata["export_time"] = m.group(1)
            elif "posture=" in line:
                m = re.search(r"posture=(\w+)", line)
                if m:
                    metadata["posture"] = m.group(1)
            elif "weinberg_k=" in line:
                m = re.search(r"weinberg_k=([\d.]+)", line)
                if m:
                    metadata["weinberg_k"] = float(m.group(1))
                for key in ("ground_truth_steps", "ground_truth_distance_m", "ground_truth_turns"):
                    m2 = re.search(rf"{key}=([\d.]+)", line)
                    if m2:
                        metadata[key] = float(m2.group(1))
        else:
            header_line = i
            break

    headers = lines[header_line].strip().split(",")
    for line in lines[header_line + 1 :]:
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
            if h in ("waypoint",):
                row[h] = val
            elif h in ("gps_lat", "gps_lng", "gps_acc_m"):
                row[h] = float(val) if val else None
            else:
                try:
                    row[h] = float(val) if "." in val else int(val)
                except (ValueError, TypeError):
                    row[h] = val
        rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    last = rows[-1]
    first = rows[0]
    duration = last.get("elapsed_sec", 0) - first.get("elapsed_sec", 0)
    metadata["duration_sec"] = duration
    metadata["n_samples"] = len(rows)
    metadata["total_steps"] = last.get("total_steps", 0)
    metadata["step_detected"] = last.get("step_detected", 0)
    metadata["sample_rate_hz"] = round(len(rows) / duration, 1) if duration > 0 else 0

    return metadata, rows


# ── Step Detection ──

def find_step_events(rows, field="total_steps"):
    """Return list of row indices where the step field increments."""
    events = []
    prev = rows[0].get(field, 0) if rows else 0
    for i, row in enumerate(rows):
        cur = row.get(field, 0)
        if cur > prev:
            n_new = cur - prev
            for _ in range(n_new):
                events.append(i)
            prev = cur
    return events


# ── Path Recomputation ──

def compute_path(rows, step_indices, heading_field, weinberg_k):
    """Recompute PDR path from step events + heading field.
    Returns list of (x, y, elapsed_sec, step_num)."""
    path = [(0.0, 0.0, rows[0].get("elapsed_sec", 0), 0)]
    x, y = 0.0, 0.0
    accel_mag_max = 0.0
    accel_mag_min = 1e9
    prev_idx = 0

    for step_num, row_idx in enumerate(step_indices, 1):
        for j in range(prev_idx, row_idx + 1):
            r = rows[j]
            ax = r.get("accel_x", 0)
            ay = r.get("accel_y", 0)
            az = r.get("accel_z", 0)
            mag = math.sqrt(ax * ax + ay * ay + az * az)
            if mag > accel_mag_max:
                accel_mag_max = mag
            if mag < accel_mag_min:
                accel_mag_min = mag

        accel_diff = max(accel_mag_max - accel_mag_min, 0.1)
        stride = weinberg_k * math.sqrt(math.sqrt(accel_diff))

        heading_deg = rows[row_idx].get(heading_field, 0)
        heading_rad = math.radians(heading_deg)

        x += stride * math.sin(heading_rad)
        y += stride * math.cos(heading_rad)
        elapsed = rows[row_idx].get("elapsed_sec", 0)
        path.append((x, y, elapsed, step_num))

        accel_mag_max = 0.0
        accel_mag_min = 1e9
        prev_idx = row_idx

    return path


def extract_original_path(rows, step_events_total):
    """Extract the original app PDR path at step event times."""
    path = [(0.0, 0.0, rows[0].get("elapsed_sec", 0), 0)]
    seen = set()
    for step_num, idx in enumerate(step_events_total, 1):
        r = rows[idx]
        px = r.get("pdr_x", 0)
        py = r.get("pdr_y", 0)
        key = (px, py)
        if key not in seen or True:
            seen.add(key)
            path.append((px, py, r.get("elapsed_sec", 0), step_num))
    return path


def extract_gps_path(rows):
    """Extract GPS coordinates → local XY meters."""
    gps_points = []
    for r in rows:
        lat = r.get("gps_lat")
        lng = r.get("gps_lng")
        if lat is not None and lng is not None:
            gps_points.append((lat, lng, r.get("elapsed_sec", 0)))

    if len(gps_points) < 2:
        return []

    lat0, lng0 = gps_points[0][0], gps_points[0][1]
    lat_rad = math.radians(lat0)
    m_per_deg_lat = 111132.92
    m_per_deg_lng = 111132.92 * math.cos(lat_rad)

    result = []
    for lat, lng, t in gps_points:
        x = (lng - lng0) * m_per_deg_lng
        y = (lat - lat0) * m_per_deg_lat
        result.append((x, y, t))
    return result


def extract_waypoints(rows):
    """Extract waypoint entries."""
    wps = []
    for r in rows:
        wp = r.get("waypoint", "")
        if wp:
            wps.append({
                "name": wp,
                "elapsed": r.get("elapsed_sec", 0),
                "total_steps": r.get("total_steps", 0),
                "step_detected": r.get("step_detected", 0),
                "pdr_x": r.get("pdr_x", 0),
                "pdr_y": r.get("pdr_y", 0),
                "rv_yaw": r.get("rv_yaw_deg", 0),
                "cal_yaw": r.get("cal_yaw_deg", 0),
                "gps_lat": r.get("gps_lat"),
                "gps_lng": r.get("gps_lng"),
                "gps_acc": r.get("gps_acc_m"),
            })
    return wps


def subsample_time_series(rows, fields, max_points=500):
    """Subsample rows for time-series charts."""
    step = max(1, len(rows) // max_points)
    result = []
    for i in range(0, len(rows), step):
        r = rows[i]
        point = {"t": r.get("elapsed_sec", 0)}
        for f in fields:
            point[f] = r.get(f, 0)
        result.append(point)
    return result


def path_total_distance(path):
    """Compute total path length."""
    d = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        d += math.sqrt(dx * dx + dy * dy)
    return d


def path_end_displacement(path):
    """Distance from start to end."""
    if len(path) < 2:
        return 0.0
    return math.sqrt(path[-1][0] ** 2 + path[-1][1] ** 2)


# ── Main Analysis ──

def analyze(data_dir):
    csv_path = os.path.join(data_dir, "sensor_log.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None

    print(f"Parsing {csv_path}...")
    metadata, rows = parse_sensor_csv(csv_path)
    weinberg_k = metadata.get("weinberg_k", 0.4667)

    print(f"  Duration: {metadata['duration_sec']:.1f}s, Samples: {metadata['n_samples']}, "
          f"Steps: {metadata['total_steps']}/{metadata['step_detected']}, "
          f"Weinberg k: {weinberg_k}")

    step_events_total = find_step_events(rows, "total_steps")
    step_events_hw = find_step_events(rows, "step_detected")
    waypoints = extract_waypoints(rows)
    gps_path = extract_gps_path(rows)

    print(f"  Step events: total={len(step_events_total)}, hw={len(step_events_hw)}, "
          f"Waypoints: {len(waypoints)}, GPS: {len(gps_path)} pts")

    heading_methods = [
        ("heading_deg", "Mag Azimuth", "加速度計＋磁力計方位角（絕對方向，易受室內干擾）"),
        ("grv_yaw_deg", "GRV Yaw", "Game Rotation Vector（無磁力計）"),
        ("rv_yaw_deg", "RV Yaw", "Rotation Vector（含磁力計）"),
        ("comp_yaw_deg", "Compass", "互補濾波（98% 陀螺 + 2% 磁力）"),
        ("gyro_yaw_deg", "Gyro Only", "純陀螺儀積分（會漂移）"),
        ("cal_yaw_deg", "Cal Yaw", "GRV + RV 校正（App 使用）"),
    ]

    paths = {}

    print("  Computing paths...")
    original = extract_original_path(rows, step_events_total)
    paths["original"] = {
        "label": "App 原始",
        "desc": "App 即時計算的 PDR 路徑",
        "data": [(p[0], p[1]) for p in original],
        "color": "#3B7DD8",
        "dash": False,
        "default_on": True,
    }

    for field, label, desc in heading_methods:
        p = compute_path(rows, step_events_total, field, weinberg_k)
        paths[field] = {
            "label": label,
            "desc": desc,
            "data": [(pt[0], pt[1]) for pt in p],
            "color": {
                "heading_deg": "#D97706",
                "grv_yaw_deg": "#2B8A4E",
                "rv_yaw_deg": "#8B5CF6",
                "comp_yaw_deg": "#F59E0B",
                "gyro_yaw_deg": "#EC4899",
                "cal_yaw_deg": "#0EA5E9",
            }[field],
            "dash": field in ("heading_deg", "comp_yaw_deg", "gyro_yaw_deg"),
            "default_on": field in ("grv_yaw_deg", "cal_yaw_deg"),
        }

    if step_events_hw:
        p_hw = compute_path(rows, step_events_hw, "cal_yaw_deg", weinberg_k)
        paths["hw_steps"] = {
            "label": "HW 計步",
            "desc": f"僅硬體 STEP_DETECTOR（{len(step_events_hw)} 步）+ Cal Yaw",
            "data": [(pt[0], pt[1]) for pt in p_hw],
            "color": "#EF4444",
            "dash": True,
            "default_on": False,
        }

    if gps_path:
        paths["gps"] = {
            "label": "GPS",
            "desc": f"GPS 軌跡（{len(gps_path)} 點）",
            "data": [(p[0], p[1]) for p in gps_path],
            "color": "#111827",
            "dark_color": "#F9FAFB",
            "dash": False,
            "default_on": True,
        }

    # Per-waypoint GPS error for each heading method
    error_methods = [
        ("original", "App"),
        ("heading_deg", "Mag"),
        ("grv_yaw_deg", "GRV"),
        ("rv_yaw_deg", "RV"),
        ("cal_yaw_deg", "Cal"),
        ("comp_yaw_deg", "Comp"),
        ("gyro_yaw_deg", "Gyro"),
    ]

    if gps_path:
        gps_lat0, gps_lng0 = None, None
        for r in rows:
            if r.get("gps_lat") is not None and r.get("gps_lng") is not None:
                gps_lat0, gps_lng0 = r["gps_lat"], r["gps_lng"]
                break

        if gps_lat0 is not None:
            lat_rad = math.radians(gps_lat0)
            m_per_deg_lat = 111132.92
            m_per_deg_lng = 111132.92 * math.cos(lat_rad)
            initial_steps = rows[0].get("total_steps", 0)

            for wp in waypoints:
                if wp["gps_lat"] is not None and wp["gps_lng"] is not None:
                    wp["gps_x"] = round((wp["gps_lng"] - gps_lng0) * m_per_deg_lng, 2)
                    wp["gps_y"] = round((wp["gps_lat"] - gps_lat0) * m_per_deg_lat, 2)
                    step_idx = wp["total_steps"] - initial_steps
                    errors = {}
                    for method_key, _ in error_methods:
                        if method_key not in paths or method_key == "gps":
                            continue
                        path_data = paths[method_key]["data"]
                        if 0 <= step_idx < len(path_data):
                            px, py = path_data[step_idx]
                            dx = px - wp["gps_x"]
                            dy = py - wp["gps_y"]
                            errors[method_key] = round(math.sqrt(dx * dx + dy * dy), 2)
                    wp["errors"] = errors
                else:
                    wp["gps_x"] = None
                    wp["gps_y"] = None
                    wp["errors"] = {}

    method_stats = []
    for key, info in paths.items():
        d = info["data"]
        total_dist = path_total_distance(d)
        end_disp = math.sqrt(d[-1][0] ** 2 + d[-1][1] ** 2) if d else 0
        method_stats.append({
            "key": key,
            "label": info["label"],
            "total_dist": round(total_dist, 2),
            "end_x": round(d[-1][0], 2) if d else 0,
            "end_y": round(d[-1][1], 2) if d else 0,
            "end_disp": round(end_disp, 2),
            "n_points": len(d),
        })

    heading_ts = subsample_time_series(
        rows,
        ["heading_deg", "grv_yaw_deg", "rv_yaw_deg", "comp_yaw_deg", "gyro_yaw_deg", "cal_yaw_deg"],
        max_points=400,
    )

    env_ts = subsample_time_series(rows, ["light_lux", "pressure_hpa"], max_points=400)
    mag_ts = subsample_time_series(rows, ["mag_x", "mag_y", "mag_z"], max_points=400)
    for p in mag_ts:
        mx, my, mz = p.get("mag_x", 0), p.get("mag_y", 0), p.get("mag_z", 0)
        p["mag_norm"] = round(math.sqrt(mx * mx + my * my + mz * mz), 1)

    step_ts = subsample_time_series(rows, ["total_steps", "step_detected"], max_points=400)

    report_data = {
        "metadata": metadata,
        "paths": {k: {"label": v["label"], "desc": v["desc"], "data": v["data"],
                       "color": v["color"], "dark_color": v.get("dark_color", ""),
                       "dash": v["dash"], "default_on": v["default_on"]}
                  for k, v in paths.items()},
        "waypoints": waypoints,
        "gps_available": len(gps_path) > 0,
        "error_methods": [{"key": k, "label": l} for k, l in error_methods],
        "method_stats": method_stats,
        "heading_ts": heading_ts,
        "env_ts": env_ts,
        "mag_ts": mag_ts,
        "step_ts": step_ts,
        "dir_name": os.path.basename(data_dir.rstrip("/\\")),
    }

    return report_data


def generate_report(report_data, output_path):
    data_json = json.dumps(report_data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("/*__DATA__*/null", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved: {output_path}")


# ── HTML Template ──

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDR 路徑分析</title>
<style>
:root {
  --bg: #F3F1EC;
  --surface: #FFFFFF;
  --surface-alt: #EAE7E0;
  --text: #1A1D1C;
  --text-m: #5D6662;
  --text-f: #8A918D;
  --border: #D4D0C8;
  --accent: #2B6E44;
  --accent-s: #D6EADD;
  --shadow: 0 1px 3px rgba(26,29,28,0.06);
  --shadow-l: 0 4px 12px rgba(26,29,28,0.08);
  --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
  --mono: 'Cascadia Code', 'SF Mono', Consolas, monospace;
}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#141716;--surface:#1E2220;--surface-alt:#272B29;--text:#E4E6E5;
  --text-m:#9CA3A0;--text-f:#6B726E;--border:#333834;--accent:#5DBF7A;
  --accent-s:rgba(93,191,122,0.12);--shadow:0 1px 3px rgba(0,0,0,0.2);
  --shadow-l:0 4px 12px rgba(0,0,0,0.3);
}}
:root[data-theme="dark"]{
  --bg:#141716;--surface:#1E2220;--surface-alt:#272B29;--text:#E4E6E5;
  --text-m:#9CA3A0;--text-f:#6B726E;--border:#333834;--accent:#5DBF7A;
  --accent-s:rgba(93,191,122,0.12);--shadow:0 1px 3px rgba(0,0,0,0.2);
  --shadow-l:0 4px 12px rgba(0,0,0,0.3);
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:var(--font);line-height:1.5;-webkit-font-smoothing:antialiased}
.page{max-width:1100px;margin:0 auto;padding:2rem 1.5rem 4rem}
header{margin-bottom:1.5rem}
.eyebrow{font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:.3rem}
h1{font-size:1.6rem;font-weight:700;line-height:1.2;text-wrap:balance}
.subtitle{color:var(--text-m);font-size:.9rem;margin-top:.25rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.65rem;margin-bottom:1.25rem}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.7rem .85rem;box-shadow:var(--shadow)}
.stat-l{font-size:.65rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--text-f)}
.stat-v{font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.15}
.stat-u{font-size:.75rem;font-weight:400;color:var(--text-m)}
.stat-n{font-size:.68rem;color:var(--text-f);margin-top:.1rem}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;box-shadow:var(--shadow);overflow:hidden;margin-bottom:1rem}
.panel-h{padding:.75rem 1rem .6rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:.5rem;flex-wrap:wrap}
.panel-t{font-size:.8rem;font-weight:700}
.panel-badge{font-size:.62rem;font-weight:600;padding:.12em .45em;border-radius:4px;background:var(--accent-s);color:var(--accent)}
.panel-b{padding:1rem}
.panel-b-flush{padding:0}
.controls{display:flex;flex-wrap:wrap;gap:.5rem;padding:.65rem 1rem;border-bottom:1px solid var(--border)}
.ctrl{display:flex;align-items:center;gap:.3rem;font-size:.72rem;cursor:pointer;user-select:none;padding:.2rem .5rem;border-radius:5px;border:1px solid var(--border);background:var(--surface);transition:background .15s}
.ctrl:hover{background:var(--surface-alt)}
.ctrl input{accent-color:var(--accent)}
.ctrl-swatch{width:10px;height:3px;border-radius:1px}
.map-wrap{position:relative;width:100%;aspect-ratio:1.4;min-height:350px}
.map-wrap canvas{width:100%;height:100%;display:block}
.tooltip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:.35rem .55rem;font-size:.7rem;box-shadow:var(--shadow-l);z-index:10;white-space:nowrap;display:none}
.chart-wrap{width:100%;overflow-x:auto}
.chart-wrap canvas{display:block;width:100%;min-width:600px}
.tbl-wrap{overflow-x:auto;max-height:380px;overflow-y:auto}
table{width:100%;border-collapse:collapse;font-size:.76rem;font-variant-numeric:tabular-nums}
th{position:sticky;top:0;background:var(--surface-alt);color:var(--text-m);font-weight:600;font-size:.66rem;letter-spacing:.03em;text-transform:uppercase;text-align:left;padding:.45rem .55rem;border-bottom:1px solid var(--border);white-space:nowrap;z-index:1}
td{padding:.35rem .55rem;border-bottom:1px solid var(--border);white-space:nowrap}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--surface-alt)}
.legend{display:flex;flex-wrap:wrap;gap:.8rem;padding:.5rem 1rem;border-top:1px solid var(--border);font-size:.7rem;color:var(--text-m)}
.legend-i{display:flex;align-items:center;gap:.25rem}
.legend-s{width:10px;height:3px;border-radius:1px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:780px){.grid2{grid-template-columns:1fr}}
h2{font-size:1rem;font-weight:700;margin:1.5rem 0 .75rem}
</style>
</head>
<body>
<div class="page" id="app"></div>
<script>
const D = /*__DATA__*/null;

function $(s,p){return(p||document).querySelector(s)}
function $$(s,p){return[...(p||document).querySelectorAll(s)]}
function el(tag,attrs,children){
  const e=document.createElement(tag);
  if(attrs)for(const[k,v]of Object.entries(attrs)){
    if(k==='style'&&typeof v==='object')Object.assign(e.style,v);
    else if(k.startsWith('on'))e.addEventListener(k.slice(2),v);
    else e.setAttribute(k,v);
  }
  if(children){
    if(typeof children==='string')e.textContent=children;
    else if(Array.isArray(children))children.forEach(c=>{if(c)e.appendChild(typeof c==='string'?document.createTextNode(c):c)});
  }
  return e;
}

const app=$('#app');
const meta=D.metadata;
const dur=meta.duration_sec;
const durMin=Math.floor(dur/60);
const durSec=Math.round(dur%60);

// Header
app.appendChild(el('header',null,[
  el('div',{class:'eyebrow'},'SensorNav PDR 路徑分析'),
  el('h1',null,D.dir_name),
  el('div',{class:'subtitle'},
    `${meta.export_time||''} · ${meta.posture||'unknown'} · Weinberg k=${meta.weinberg_k||0.4667}`)
]));

// Stats
const statsDiv=el('div',{class:'stats'});
[[`${durMin}m ${durSec}s`,'錄製時長',`${meta.n_samples} 筆`],
 [meta.total_steps,'Total Steps',`HW: ${meta.step_detected}`],
 [D.waypoints.length,'航點照片',`平均 ${dur>0&&D.waypoints.length>0?(dur/D.waypoints.length).toFixed(1):'—'}s/張`],
 [`${meta.sample_rate_hz} Hz`,'取樣率','全感測器同步'],
].forEach(([v,l,n])=>{
  const c=el('div',{class:'stat'},[
    el('div',{class:'stat-l'},l),
    el('div',{class:'stat-v'},String(v)),
    el('div',{class:'stat-n'},n),
  ]);
  statsDiv.appendChild(c);
});
app.appendChild(statsDiv);

// ── Path Map ──
const mapPanel=el('div',{class:'panel'});
const mapHead=el('div',{class:'panel-h'},[
  el('span',{class:'panel-t'},'路徑比較圖'),
  el('span',{class:'panel-badge'},Object.keys(D.paths).length+' 種方法'),
]);
mapPanel.appendChild(mapHead);

const ctrlDiv=el('div',{class:'controls'});
const pathKeys=Object.keys(D.paths);
const visibility={};
pathKeys.forEach(k=>{
  const info=D.paths[k];
  visibility[k]=info.default_on;
  const cb=el('input',{type:'checkbox'});
  cb.checked=info.default_on;
  cb.addEventListener('change',()=>{visibility[k]=cb.checked;drawMap()});
  const swatch=el('span',{class:'ctrl-swatch',style:{background:info.color}});
  if(info.dash)swatch.style.background=`repeating-linear-gradient(90deg,${info.color} 0 3px,transparent 3px 5px)`;
  const label=el('label',{class:'ctrl'},[cb,swatch,document.createTextNode(' '+info.label)]);
  ctrlDiv.appendChild(label);
});
mapPanel.appendChild(ctrlDiv);

const mapBody=el('div',{class:'panel-b panel-b-flush'});
const mapWrap=el('div',{class:'map-wrap'});
const mapCanvas=el('canvas');
const mapTooltip=el('div',{class:'tooltip'});
mapWrap.appendChild(mapCanvas);
mapWrap.appendChild(mapTooltip);
mapBody.appendChild(mapWrap);
mapPanel.appendChild(mapBody);
app.appendChild(mapPanel);

function getCSS(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function isDark(){
  const dt=document.documentElement.getAttribute('data-theme');
  if(dt==='dark')return true;
  if(dt==='light')return false;
  return window.matchMedia('(prefers-color-scheme:dark)').matches;
}

let mapWpCoords=[];

function drawMap(){
  const rect=mapWrap.getBoundingClientRect();
  const dpr=window.devicePixelRatio||1;
  const W=rect.width,H=rect.height;
  mapCanvas.width=W*dpr;mapCanvas.height=H*dpr;
  mapCanvas.style.width=W+'px';mapCanvas.style.height=H+'px';
  const ctx=mapCanvas.getContext('2d');
  ctx.scale(dpr,dpr);

  const pad=50;
  let allX=[],allY=[];
  pathKeys.forEach(k=>{
    if(!visibility[k])return;
    D.paths[k].data.forEach(p=>{allX.push(p[0]);allY.push(p[1])});
  });
  if(!allX.length){allX=[0];allY=[0]}
  const minX=Math.min(...allX)-2,maxX=Math.max(...allX)+2;
  const minY=Math.min(...allY)-2,maxY=Math.max(...allY)+2;
  const rangeX=maxX-minX,rangeY=maxY-minY;
  const scale=Math.min((W-pad*2)/rangeX,(H-pad*2)/rangeY);
  const ox=pad+((W-pad*2)-rangeX*scale)/2;
  const oy=pad+((H-pad*2)-rangeY*scale)/2;
  const tx=x=>ox+(x-minX)*scale;
  const ty=y=>oy+(maxY-y)*scale;

  // Grid
  ctx.strokeStyle=getCSS('--border');ctx.lineWidth=.5;ctx.setLineDash([3,3]);
  ctx.fillStyle=getCSS('--text-f');ctx.font='10px '+getCSS('--mono');
  for(let gx=Math.ceil(minX/5)*5;gx<=maxX;gx+=5){
    ctx.beginPath();ctx.moveTo(tx(gx),pad);ctx.lineTo(tx(gx),H-pad);ctx.stroke();
    ctx.textAlign='center';ctx.fillText(gx+'m',tx(gx),H-pad+14);
  }
  for(let gy=Math.ceil(minY/5)*5;gy<=maxY;gy+=5){
    ctx.beginPath();ctx.moveTo(pad,ty(gy));ctx.lineTo(W-pad,ty(gy));ctx.stroke();
    ctx.textAlign='right';ctx.fillText(gy+'m',pad-6,ty(gy)+3);
  }
  ctx.setLineDash([]);

  // Draw paths
  pathKeys.forEach(k=>{
    if(!visibility[k])return;
    const info=D.paths[k];
    const pts=info.data;
    if(pts.length<2)return;
    const dark=isDark();
    const color=(dark&&info.dark_color)?info.dark_color:info.color;
    ctx.strokeStyle=color;
    ctx.lineWidth=k==='original'?2.5:1.8;
    ctx.lineJoin='round';ctx.lineCap='round';
    if(info.dash)ctx.setLineDash([6,3]);
    else ctx.setLineDash([]);
    ctx.globalAlpha=k==='original'?1:0.8;
    ctx.beginPath();
    pts.forEach((p,i)=>i===0?ctx.moveTo(tx(p[0]),ty(p[1])):ctx.lineTo(tx(p[0]),ty(p[1])));
    ctx.stroke();
    ctx.globalAlpha=1;
    ctx.setLineDash([]);
  });

  // Waypoints on original path
  mapWpCoords=[];
  if(visibility.original){
    const wpColor='#D44B2F';
    D.waypoints.forEach((wp,i)=>{
      const cx=tx(wp.pdr_x),cy=ty(wp.pdr_y);
      mapWpCoords.push({cx,cy,wp,i});
      ctx.fillStyle='rgba(212,75,47,0.12)';
      ctx.beginPath();ctx.arc(cx,cy,11,0,Math.PI*2);ctx.fill();
      ctx.fillStyle=wpColor;
      ctx.beginPath();ctx.arc(cx,cy,5,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#fff';ctx.font='bold 7px '+getCSS('--font');
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(i+1,cx,cy+.5);
    });
  }

  // Start/End markers
  const orig=D.paths.original.data;
  if(orig.length>=2){
    [[orig[0],'S','var(--accent)'],[orig[orig.length-1],'E','var(--accent)']].forEach(([p,label,c])=>{
      const px=tx(p[0]),py=ty(p[1]);
      ctx.fillStyle=getCSS('--accent');
      ctx.beginPath();ctx.arc(px,py,7,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#fff';ctx.font='bold 9px '+getCSS('--font');
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(label,px,py+.5);
    });
  }
}

mapCanvas.addEventListener('mousemove',e=>{
  const rect=mapCanvas.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  let hit=null;
  for(const w of mapWpCoords){
    if(Math.hypot(mx-w.cx,my-w.cy)<14){hit=w;break}
  }
  if(hit){
    mapTooltip.style.display='block';
    mapTooltip.style.left=(hit.cx+16)+'px';
    mapTooltip.style.top=(hit.cy-10)+'px';
    let tip=`<strong>${hit.wp.name}</strong><br>t=${hit.wp.elapsed.toFixed(1)}s · ${hit.wp.total_steps} 步`;
    if(hit.wp.errors&&Object.keys(hit.wp.errors).length){
      tip+=`<br>GPS (${hit.wp.gps_x.toFixed(1)}, ${hit.wp.gps_y.toFixed(1)})`;
      (D.error_methods||[]).forEach(m=>{if(hit.wp.errors[m.key]!=null)tip+=`<br>${m.label}: ${hit.wp.errors[m.key].toFixed(1)}m`});
    }else{tip+=`<br>PDR (${hit.wp.pdr_x.toFixed(1)}, ${hit.wp.pdr_y.toFixed(1)}) m`}
    mapTooltip.innerHTML=tip;
  } else {
    mapTooltip.style.display='none';
  }
});
mapCanvas.addEventListener('mouseleave',()=>{mapTooltip.style.display='none'});

// ── Method Stats Table ──
app.appendChild(el('h2',null,'各方法比較'));
const statsPanel=el('div',{class:'panel'});
const stWrap=el('div',{class:'panel-b panel-b-flush'});
const stTblWrap=el('div',{class:'tbl-wrap'});
const stTbl=el('table');
const stHead=el('thead');
stHead.innerHTML='<tr><th>方法</th><th>路徑總長 (m)</th><th>終點 X (m)</th><th>終點 Y (m)</th><th>終點位移 (m)</th><th>步數</th></tr>';
stTbl.appendChild(stHead);
const stBody=el('tbody');
D.method_stats.forEach(s=>{
  const tr=el('tr');
  const info=D.paths[s.key];
  tr.innerHTML=`<td><span style="display:inline-block;width:10px;height:3px;border-radius:1px;background:${info.color};vertical-align:middle;margin-right:6px"></span>${s.label}</td><td>${s.total_dist.toFixed(1)}</td><td>${s.end_x.toFixed(2)}</td><td>${s.end_y.toFixed(2)}</td><td>${s.end_disp.toFixed(1)}</td><td>${s.n_points}</td>`;
  stBody.appendChild(tr);
});
stTbl.appendChild(stBody);
stTblWrap.appendChild(stTbl);
stWrap.appendChild(stTblWrap);
statsPanel.appendChild(stWrap);
app.appendChild(statsPanel);

// ── Generic Chart Drawing ──
function drawLineChart(canvas,datasets,opts){
  const wrap=canvas.parentElement;
  const rect=wrap.getBoundingClientRect();
  const dpr=window.devicePixelRatio||1;
  const W=Math.max(rect.width,600),H=opts.height||180;
  canvas.width=W*dpr;canvas.height=H*dpr;
  canvas.style.width=W+'px';canvas.style.height=H+'px';
  const ctx=canvas.getContext('2d');
  ctx.scale(dpr,dpr);

  const pad={l:52,r:16,t:14,b:28};
  const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  const maxT=opts.maxT||dur;
  const yMin=opts.yMin??0,yMax=opts.yMax??100;
  const txC=t=>pad.l+(t/maxT)*cw;
  const tyC=v=>pad.t+ch-((v-yMin)/(yMax-yMin))*ch;

  // Grid
  ctx.strokeStyle=getCSS('--border');ctx.lineWidth=.5;ctx.setLineDash([3,3]);
  ctx.fillStyle=getCSS('--text-f');ctx.font='10px '+getCSS('--mono');
  for(let i=0;i<=5;i++){
    const v=yMin+(yMax-yMin)*i/5;
    const y=tyC(v);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    ctx.textAlign='right';ctx.fillText(Math.round(v),pad.l-6,y+3);
  }
  ctx.setLineDash([]);
  ctx.textAlign='center';
  for(let t=0;t<=maxT;t+=Math.max(30,Math.ceil(maxT/8/30)*30)){
    ctx.fillText(t+'s',txC(t),H-6);
  }

  // Waypoint lines
  ctx.strokeStyle='rgba(212,75,47,0.1)';ctx.lineWidth=1;
  D.waypoints.forEach(wp=>{
    ctx.beginPath();ctx.moveTo(txC(wp.elapsed),pad.t);ctx.lineTo(txC(wp.elapsed),pad.t+ch);ctx.stroke();
  });

  // Data lines
  datasets.forEach(ds=>{
    if(!ds.data||!ds.data.length)return;
    ctx.strokeStyle=ds.color;ctx.lineWidth=ds.width||1.5;
    ctx.lineJoin='round';ctx.setLineDash(ds.dash?[5,3]:[]);
    ctx.beginPath();
    ds.data.forEach((p,i)=>{
      const x=txC(p.t),y=tyC(p.v);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    if(ds.fill){
      ctx.globalAlpha=0.06;ctx.fillStyle=ds.color;
      ctx.lineTo(txC(ds.data[ds.data.length-1].t),tyC(yMin));
      ctx.lineTo(txC(ds.data[0].t),tyC(yMin));
      ctx.closePath();ctx.fill();ctx.globalAlpha=1;
    }
  });
}

// ── Heading Chart ──
app.appendChild(el('h2',null,'航向時間序列'));
const hPanel=el('div',{class:'panel'});
const hBody=el('div',{class:'panel-b'});
const hWrap=el('div',{class:'chart-wrap'});
const hCanvas=el('canvas');
hWrap.appendChild(hCanvas);hBody.appendChild(hWrap);hPanel.appendChild(hBody);
const hLeg=el('div',{class:'legend'});
[['Mag Azimuth','#D97706'],['GRV','#2B8A4E'],['RV','#8B5CF6'],['Compass','#F59E0B'],['Gyro','#EC4899'],['Cal','#0EA5E9']].forEach(([l,c])=>{
  hLeg.appendChild(el('div',{class:'legend-i'},[el('span',{class:'legend-s',style:{background:c}}),document.createTextNode(l)]));
});
hPanel.appendChild(hLeg);
app.appendChild(hPanel);

function drawHeading(){
  const fields=['heading_deg','grv_yaw_deg','rv_yaw_deg','comp_yaw_deg','gyro_yaw_deg','cal_yaw_deg'];
  const colors=['#D97706','#2B8A4E','#8B5CF6','#F59E0B','#EC4899','#0EA5E9'];
  let yMin=Infinity,yMax=-Infinity;
  D.heading_ts.forEach(p=>{fields.forEach(f=>{const v=p[f]||0;if(v<yMin)yMin=v;if(v>yMax)yMax=v})});
  yMin=Math.floor(yMin/10)*10-10;yMax=Math.ceil(yMax/10)*10+10;
  drawLineChart(hCanvas,fields.map((f,i)=>({
    data:D.heading_ts.map(p=>({t:p.t,v:p[f]||0})),
    color:colors[i],width:i===5?2:1.2,dash:i===0||i===3||i===4
  })),{maxT:dur,yMin,yMax,height:220});
}

// ── Step Chart ──
app.appendChild(el('h2',null,'步數累計'));
const sPanel=el('div',{class:'panel'});
const sBody=el('div',{class:'panel-b'});
const sWrap=el('div',{class:'chart-wrap'});
const sCanvas=el('canvas');
sWrap.appendChild(sCanvas);sBody.appendChild(sWrap);sPanel.appendChild(sBody);
const sLeg=el('div',{class:'legend'});
[['Total Steps（HW+SW）','#2B8A4E'],['Step Detected（HW）','#EF4444']].forEach(([l,c])=>{
  sLeg.appendChild(el('div',{class:'legend-i'},[el('span',{class:'legend-s',style:{background:c}}),document.createTextNode(l)]));
});
sPanel.appendChild(sLeg);
app.appendChild(sPanel);

function drawSteps(){
  const yMax=Math.ceil(meta.total_steps*1.1/10)*10;
  drawLineChart(sCanvas,[
    {data:D.step_ts.map(p=>({t:p.t,v:p.total_steps||0})),color:'#2B8A4E',width:2,fill:true},
    {data:D.step_ts.map(p=>({t:p.t,v:p.step_detected||0})),color:'#EF4444',width:1.5},
  ],{maxT:dur,yMin:0,yMax,height:170});
}

// ── Environment Chart ──
app.appendChild(el('h2',null,'環境感測'));
const ePanel=el('div',{class:'panel'});
const eBody=el('div',{class:'panel-b'});
const eWrap=el('div',{class:'chart-wrap'});
const eCanvas=el('canvas');
eWrap.appendChild(eCanvas);eBody.appendChild(eWrap);ePanel.appendChild(eBody);
const eLeg=el('div',{class:'legend'});
[['光照度 (lux)','#C9A830'],['磁場強度 (μT)','#8B6E50']].forEach(([l,c])=>{
  eLeg.appendChild(el('div',{class:'legend-i'},[el('span',{class:'legend-s',style:{background:c}}),document.createTextNode(l)]));
});
ePanel.appendChild(eLeg);
app.appendChild(ePanel);

function drawEnv(){
  let yMax=0;
  D.env_ts.forEach(p=>{if((p.light_lux||0)>yMax)yMax=p.light_lux});
  D.mag_ts.forEach(p=>{if((p.mag_norm||0)>yMax)yMax=p.mag_norm});
  yMax=Math.ceil(yMax*1.1/100)*100;
  drawLineChart(eCanvas,[
    {data:D.env_ts.map(p=>({t:p.t,v:p.light_lux||0})),color:'#C9A830',width:1.8,fill:true},
    {data:D.mag_ts.map(p=>({t:p.t,v:p.mag_norm||0})),color:'#8B6E50',width:1.3},
  ],{maxT:dur,yMin:0,yMax,height:170});
}

// ── Waypoint GPS Error Table ──
app.appendChild(el('h2',null,'航點 GPS 誤差比較'));
const wPanel=el('div',{class:'panel'});
const wHead=el('div',{class:'panel-h'},[
  el('span',{class:'panel-t'},'各方法 vs GPS'),
  D.gps_available?el('span',{class:'panel-badge'},'★ = 該點最佳'):null,
]);
wPanel.appendChild(wHead);
const wWrap=el('div',{class:'panel-b panel-b-flush'});
const wTblWrap=el('div',{class:'tbl-wrap'});
const wTbl=el('table');
const em=D.error_methods||[];
let thHtml='<thead><tr><th>#</th><th>名稱</th><th>時間</th><th>步數</th>';
if(D.gps_available){
  thHtml+='<th>GPS (x,y)</th>';
  em.forEach(m=>{
    const info=D.paths[m.key];
    const c=info?info.color:'#888';
    thHtml+=`<th style="white-space:nowrap"><span style="display:inline-block;width:8px;height:3px;border-radius:1px;background:${c};vertical-align:middle;margin-right:4px"></span>${m.label}</th>`;
  });
}else{
  thHtml+='<th>PDR X</th><th>PDR Y</th><th>Cal Yaw°</th>';
}
thHtml+='</tr></thead>';
wTbl.innerHTML=thHtml;
const wBody=el('tbody');
const avgErr={};let nGps=0;
D.waypoints.forEach((wp,i)=>{
  const tr=el('tr');
  let h=`<td>${i+1}</td><td>${wp.name}</td><td>${wp.elapsed.toFixed(1)}</td><td>${wp.total_steps}</td>`;
  if(D.gps_available){
    h+=wp.gps_x!=null?`<td>(${wp.gps_x.toFixed(1)}, ${wp.gps_y.toFixed(1)})</td>`:'<td>—</td>';
    if(wp.errors&&Object.keys(wp.errors).length){
      nGps++;
      const vals=em.map(m=>wp.errors[m.key]).filter(v=>v!=null);
      const best=vals.length?Math.min(...vals):Infinity;
      em.forEach(m=>{
        const e=wp.errors?wp.errors[m.key]:null;
        if(e!=null){
          avgErr[m.key]=(avgErr[m.key]||0)+e;
          const ib=e===best;
          const sc=e>10?'#DC2626':e>5?'#D97706':'#16A34A';
          h+=`<td style="color:${sc}${ib?';font-weight:700':''}">${e.toFixed(1)}${ib?' ★':''}</td>`;
        }else h+='<td>—</td>';
      });
    }else{em.forEach(()=>{h+='<td>—</td>'})}
  }else{
    h+=`<td>${wp.pdr_x.toFixed(2)}</td><td>${wp.pdr_y.toFixed(2)}</td><td>${wp.cal_yaw.toFixed(1)}</td>`;
  }
  tr.innerHTML=h;wBody.appendChild(tr);
});
if(D.gps_available&&nGps>0){
  const tr=el('tr');tr.style.fontWeight='700';tr.style.borderTop='2px solid var(--border)';
  let h='<td></td><td>平均誤差</td><td></td><td></td><td></td>';
  const avgs=em.map(m=>((avgErr[m.key]||0)/nGps));
  const bestAvg=Math.min(...avgs);
  em.forEach((m,j)=>{
    const a=avgs[j];const ib=Math.abs(a-bestAvg)<0.01;
    h+=`<td style="color:${ib?'#16A34A':'var(--text)'}">${a.toFixed(1)} m${ib?' ★':''}</td>`;
  });
  tr.innerHTML=h;wBody.appendChild(tr);
}
wTbl.appendChild(wBody);
wTblWrap.appendChild(wTbl);
wWrap.appendChild(wTblWrap);
wPanel.appendChild(wWrap);
app.appendChild(wPanel);

// ── Draw All ──
function drawAll(){drawMap();drawHeading();drawSteps();drawEnv()}
drawAll();
window.addEventListener('resize',drawAll);
new MutationObserver(drawAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',drawAll);
</script>
</body>
</html>
"""


# ── CLI ──

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_pdr.py <data_dir> [<data_dir2> ...]")
        print("  Reads sensor_log.csv from each directory and generates report.html")
        sys.exit(1)

    for data_dir in sys.argv[1:]:
        data_dir = data_dir.rstrip("/\\")
        if not os.path.isdir(data_dir):
            print(f"Warning: {data_dir} is not a directory, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"Analyzing: {data_dir}")
        print(f"{'='*60}")
        report_data = analyze(data_dir)
        if report_data:
            output = os.path.join(data_dir, "report.html")
            generate_report(report_data, output)

            ms = report_data["method_stats"]
            print(f"\n  Path distances:")
            for m in ms:
                flag = " !!!" if m["total_dist"] > ms[0]["total_dist"] * 2 else ""
                print(f"    {m['label']:16s}  {m['total_dist']:8.1f} m  →  ({m['end_x']:.1f}, {m['end_y']:.1f}){flag}")
            print()


if __name__ == "__main__":
    main()
