"""Build HTML report with topo map visualization + top-3 candidate photos."""
import base64, io, json, html as html_mod, math, os
from pathlib import Path
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/Users/shingchou/Downloads/學校家樂福/0916測試"))
INPUT_JSON = DATA_ROOT / "檢索圖片" / "full_test_v4_merged_results.json"
WP_PHOTOS_JSON = SCRIPT_DIR / "wp_photos.json"
TOPO_JSON = SCRIPT_DIR / "topo_map_data.json"
OUTPUT_HTML = SCRIPT_DIR / "vlm_report_v6_merged.html"

with open(INPUT_JSON) as f:
    data = json.load(f)
with open(WP_PHOTOS_JSON) as f:
    wp_photos = json.load(f)
with open(TOPO_JSON) as f:
    topo = json.load(f)

meta = data["meta"]
results = data["results"]

# Build waypoint coordinate lookup
wp_coords = {}
for w in topo["waypoints"]:
    wp_coords[w["nid"]] = (w["x"], w["y"])

edges = [(e["from_nid"], e["to_nid"]) for e in topo["edges"]]


def resolve_photo_path(neo4j_path):
    if not neo4j_path:
        return None
    parts = neo4j_path.split("/", 1)
    if len(parts) == 2:
        folder = parts[0].replace("set", "")
        return DATA_ROOT / folder / parts[1]
    return None


def photo_to_b64(path_or_fname, max_size=280):
    if isinstance(path_or_fname, Path):
        p = path_or_fname
    else:
        for sd in ["01", "02", "03"]:
            p = DATA_ROOT / sd / path_or_fname
            if p.exists():
                break
        else:
            return None
    if not p.exists():
        return None
    try:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=55)
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None


def get_ref_photos_b64(nid, max_size=160):
    nid_str = str(nid)
    if nid_str not in wp_photos:
        return {}
    result = {}
    for slot, neo_path in wp_photos[nid_str].items():
        fpath = resolve_photo_path(neo_path)
        if fpath:
            b64 = photo_to_b64(fpath, max_size)
            if b64:
                result[slot] = b64
    return result


def esc(s):
    return html_mod.escape(str(s))


# Separate errors and corrects first to only cache what's needed
errors_list = [r for r in results if not r["is_top1"]]
corrects_list = [r for r in results if r["is_top1"]]
errors_list.sort(key=lambda x: x.get("gap", 0), reverse=True)

# Pre-cache query thumbnails (errors only to save space)
print("Generating query thumbnails (errors only)...", flush=True)
query_cache = {}
for i, r in enumerate(errors_list):
    b64 = photo_to_b64(r["fname"], 220)
    if b64:
        query_cache[r["fname"]] = b64
print(f"  {len(query_cache)} query thumbnails", flush=True)

# Pre-cache reference photos for top-3 candidates (errors + correct top-3)
print("Generating reference thumbnails for top-3...", flush=True)
ref_cache = {}
ref_nids = set()
for r in errors_list:
    ref_nids.add(r["true_nid"])
    for c in r.get("top5", [])[:3]:
        ref_nids.add(c["nid"])
for i, nid in enumerate(sorted(ref_nids)):
    ref_cache[nid] = get_ref_photos_b64(nid, 120)
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(ref_nids)} waypoints", flush=True)
print(f"  {len(ref_cache)} waypoint ref sets cached", flush=True)

# Export topo data as JS
wp_js = json.dumps({str(w["nid"]): {"x": w["x"], "y": w["y"]} for w in topo["waypoints"]})
edges_js = json.dumps(edges)

# Build test cards with inline SVG mini-maps + top-3 photos
print("Building test cards...", flush=True)

# Compute map bounds
all_x = [w["x"] for w in topo["waypoints"] if w["x"] != 0 or w["y"] != 0]
all_y = [w["y"] for w in topo["waypoints"] if w["x"] != 0 or w["y"] != 0]
min_x, max_x = min(all_x) - 2, max(all_x) + 2
min_y, max_y = min(all_y) - 2, max(all_y) + 2
map_w, map_h = 320, 320
scale_x = map_w / (max_x - min_x)
scale_y = map_h / (max_y - min_y)
scale = min(scale_x, scale_y)
off_x = (map_w - (max_x - min_x) * scale) / 2
off_y = (map_h - (max_y - min_y) * scale) / 2

def to_svg(x, y):
    if x == 0 and y == 0:
        return None, None
    sx = off_x + (x - min_x) * scale
    sy = map_h - (off_y + (y - min_y) * scale)  # flip Y
    return round(sx, 1), round(sy, 1)

def build_mini_map_svg(true_nid, top3_nids):
    """Build SVG mini-map highlighting true position and top-3 predictions."""
    svg_parts = []
    svg_parts.append(f'<svg viewBox="0 0 {map_w} {map_h}" class="mini-map">')

    # Draw edges
    for a_nid, b_nid in edges:
        ax, ay = wp_coords.get(a_nid, (0, 0))
        bx, by = wp_coords.get(b_nid, (0, 0))
        sa = to_svg(ax, ay)
        sb = to_svg(bx, by)
        if sa[0] is not None and sb[0] is not None:
            svg_parts.append(f'<line x1="{sa[0]}" y1="{sa[1]}" x2="{sb[0]}" y2="{sb[1]}" stroke="var(--edge-color)" stroke-width="1" opacity="0.3"/>')

    # Draw all waypoints as small dots
    for w in topo["waypoints"]:
        sx, sy = to_svg(w["x"], w["y"])
        if sx is None:
            continue
        svg_parts.append(f'<circle cx="{sx}" cy="{sy}" r="3" fill="var(--wp-dot)" opacity="0.3"/>')
        svg_parts.append(f'<text x="{sx}" y="{sy-5}" text-anchor="middle" font-size="7" fill="var(--wp-label)" opacity="0.4">{w["nid"]}</text>')

    # Draw top-3 predictions (largest first so #1 is on top)
    colors = ["#f87171", "#fbbf24", "#06b6d4"]  # #1 red, #2 yellow, #3 cyan
    sizes = [8, 7, 6]
    for idx in reversed(range(min(3, len(top3_nids)))):
        nid = top3_nids[idx]
        cx, cy = wp_coords.get(nid, (0, 0))
        sx, sy = to_svg(cx, cy)
        if sx is None:
            continue
        svg_parts.append(f'<circle cx="{sx}" cy="{sy}" r="{sizes[idx]}" fill="{colors[idx]}" opacity="0.8" stroke="white" stroke-width="1.5"/>')
        svg_parts.append(f'<text x="{sx}" y="{sy-sizes[idx]-3}" text-anchor="middle" font-size="9" font-weight="bold" fill="{colors[idx]}">#{idx+1}</text>')

    # Draw true position (green star, always on top)
    tx, ty = wp_coords.get(true_nid, (0, 0))
    tsx, tsy = to_svg(tx, ty)
    if tsx is not None:
        svg_parts.append(f'<circle cx="{tsx}" cy="{tsy}" r="9" fill="none" stroke="#34d399" stroke-width="2.5"/>')
        svg_parts.append(f'<circle cx="{tsx}" cy="{tsy}" r="4" fill="#34d399"/>')
        svg_parts.append(f'<text x="{tsx}" y="{tsy-12}" text-anchor="middle" font-size="10" font-weight="bold" fill="#34d399">WP{true_nid}</text>')

    # Distance lines from true to predictions
    if tsx is not None:
        for idx in range(min(3, len(top3_nids))):
            nid = top3_nids[idx]
            if nid == true_nid:
                continue
            cx, cy = wp_coords.get(nid, (0, 0))
            sx, sy = to_svg(cx, cy)
            if sx is None:
                continue
            svg_parts.append(f'<line x1="{tsx}" y1="{tsy}" x2="{sx}" y2="{sy}" stroke="{colors[idx]}" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>')
            # Distance label
            mx, my = (tsx + sx) / 2, (tsy + sy) / 2
            dist = math.sqrt((tx - cx)**2 + (ty - cy)**2)
            svg_parts.append(f'<text x="{mx}" y="{my-4}" text-anchor="middle" font-size="8" fill="{colors[idx]}">{dist:.1f}m</text>')

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def build_ref_strip(nid, rank, score, color):
    """Build reference photo strip for a candidate."""
    refs = ref_cache.get(nid, {})
    html = f'<div class="cand-strip">'
    html += f'<div class="cand-header"><span class="cand-rank" style="background:{color}">#{rank}</span> WP{nid} <span class="cand-score">{score*100:.1f}%</span></div>'
    html += '<div class="cand-photos">'
    for slot in ["front", "back", "left", "right"]:
        b64 = refs.get(slot)
        if b64:
            html += f'<div class="cand-photo"><img src="data:image/jpeg;base64,{b64}"><div class="cand-slot">{slot}</div></div>'
    html += '</div></div>'
    return html


# Build all cards
all_cards_html = ""

# Build error cards
for idx, r in enumerate(errors_list):
    true_nid = r["true_nid"]
    top3 = r.get("top5", [])[:3]
    top3_nids = [c["nid"] for c in top3]

    # Mini map
    mini_map = build_mini_map_svg(true_nid, top3_nids)

    # Query photo
    qimg = query_cache.get(r["fname"], "")
    qimg_html = f'<img class="query-img" src="data:image/jpeg;base64,{qimg}">' if qimg else '<div class="no-img">無圖片</div>'

    # Top-3 reference strips
    colors = ["#f87171", "#fbbf24", "#06b6d4"]
    ref_strips = ""
    for j, c in enumerate(top3):
        ref_strips += build_ref_strip(c["nid"], j+1, c["score"], colors[j])

    # True answer reference
    if true_nid not in top3_nids:
        true_score = r.get("true_score", 0)
        true_rank = r.get("true_rank", "?")
        ref_strips += f'<div class="cand-strip true-strip">'
        ref_strips += f'<div class="cand-header"><span class="cand-rank" style="background:#34d399">正確 #{true_rank}</span> WP{true_nid} <span class="cand-score">{true_score*100:.1f}%</span></div>'
        refs = ref_cache.get(true_nid, {})
        ref_strips += '<div class="cand-photos">'
        for slot in ["front", "back", "left", "right"]:
            b64 = refs.get(slot)
            if b64:
                ref_strips += f'<div class="cand-photo"><img src="data:image/jpeg;base64,{b64}"><div class="cand-slot">{slot}</div></div>'
        ref_strips += '</div></div>'

    mark_color = "#fbbf24" if r["is_top3"] else ("#06b6d4" if r["is_top5"] else "#f87171")
    mark = "△" if r["is_top3"] else ("⬤" if r["is_top5"] else "✗")

    # Compute distances
    dist_info = ""
    tx, ty = wp_coords.get(true_nid, (0, 0))
    for j, c in enumerate(top3):
        cx, cy = wp_coords.get(c["nid"], (0, 0))
        if (tx != 0 or ty != 0) and (cx != 0 or cy != 0):
            d = math.sqrt((tx - cx)**2 + (ty - cy)**2)
            dist_info += f'<span class="dist-tag" style="color:{colors[j]}">#{j+1} WP{c["nid"]}: {d:.1f}m</span>'

    all_cards_html += f'''
<div class="card error-card">
  <div class="card-head">
    <span class="card-mark" style="color:{mark_color}">{mark} #{idx+1}</span>
    <span class="card-title">WP{true_nid} → WP{r["pred_nid"]}</span>
    <span class="card-meta">{r["set"]}/{r["direction"]} | {esc(r.get("scene","")[:40])}</span>
  </div>
  <div class="card-body">
    <div class="card-left">
      <div class="ql">查詢照片</div>
      {qimg_html}
    </div>
    <div class="card-map">
      <div class="ql">拓撲地圖位置</div>
      {mini_map}
      <div class="dist-row">{dist_info}</div>
    </div>
  </div>
  <div class="card-refs">
    <div class="ql">前三名候選 + 正確答案參考照片</div>
    {ref_strips}
  </div>
</div>'''

# Build correct cards (compact list — no photos, just map + score)
print("Building correct cards...", flush=True)
correct_cards_html = ""
for idx, r in enumerate(corrects_list):
    true_nid = r["true_nid"]
    top3 = r.get("top5", [])[:3]
    top3_nids = [c["nid"] for c in top3]
    mini_map = build_mini_map_svg(true_nid, top3_nids)

    top3_text = " | ".join([f'#{j+1} WP{c["nid"]} {c["score"]*100:.0f}%' for j, c in enumerate(top3)])

    correct_cards_html += f'''
<div class="ok-item" onclick="this.querySelector('.ok-map').style.display=this.querySelector('.ok-map').style.display==='none'?'block':'none'">
  <span class="ok-check">✓</span>
  <span class="ok-wp">WP{true_nid}</span>
  <span class="ok-score">{r["top1_score"]*100:.1f}%</span>
  <span class="ok-detail">{r["set"]}/{r["direction"]} | {top3_text}</span>
  <div class="ok-map" style="display:none">{mini_map}</div>
</div>'''


# Assemble HTML
err_count = len(errors_list)
ok_count = len(corrects_list)

t1p = meta["top1"]/meta["total"]*100
t3p = meta["top3"]/meta["total"]*100
t5p = meta["top5"]/meta["total"]*100

report_html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>VLM 定位地圖報告</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Noto+Sans+TC:wght@400;600;700&display=swap">
<style>
:root {{
  --bg: #0f172a; --card: #1e293b; --card2: #334155; --accent: #38bdf8;
  --text: #f1f5f9; --muted: #94a3b8; --green: #34d399; --yellow: #fbbf24;
  --red: #f87171; --cyan: #06b6d4; --border: rgba(148,163,184,0.12);
  --edge-color: #94a3b8; --wp-dot: #64748b; --wp-label: #94a3b8;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    --bg: #f8fafc; --card: #ffffff; --card2: #f1f5f9; --accent: #0284c7;
    --text: #0f172a; --muted: #64748b; --green: #059669; --yellow: #d97706;
    --red: #dc2626; --cyan: #0891b2; --border: rgba(15,23,42,0.08);
    --edge-color: #94a3b8; --wp-dot: #94a3b8; --wp-label: #64748b;
  }}
}}
:root[data-theme="light"] {{
  --bg: #f8fafc; --card: #ffffff; --card2: #f1f5f9; --accent: #0284c7;
  --text: #0f172a; --muted: #64748b; --green: #059669; --yellow: #d97706;
  --red: #dc2626; --cyan: #0891b2; --border: rgba(15,23,42,0.08);
  --edge-color: #94a3b8; --wp-dot: #94a3b8; --wp-label: #64748b;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter','Noto Sans TC',-apple-system,sans-serif; background:var(--bg); color:var(--text); padding:16px; line-height:1.5; }}
.container {{ max-width:1100px; margin:0 auto; }}
h1 {{ text-align:center; font-size:22px; font-weight:700; color:var(--accent); }}
.sub {{ text-align:center; color:var(--muted); font-size:12px; margin:4px 0 16px; }}
.hero {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin-bottom:20px; }}
.stat {{ background:var(--card); border-radius:10px; padding:12px; text-align:center; border:1px solid var(--border); }}
.stat-val {{ font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }}
.stat-label {{ font-size:10px; color:var(--muted); margin-top:2px; }}
.section {{ margin-bottom:20px; }}
.section-title {{ font-size:14px; font-weight:700; color:var(--accent); margin-bottom:8px; display:flex; align-items:center; gap:6px; }}
.section-title::before {{ content:''; width:3px; height:14px; background:var(--accent); border-radius:2px; }}
.legend {{ background:var(--card); border-radius:8px; padding:8px 14px; font-size:11px; display:flex; gap:12px; flex-wrap:wrap; border:1px solid var(--border); margin-bottom:16px; }}

.card {{ background:var(--card); border-radius:12px; margin-bottom:12px; overflow:hidden; border:1px solid var(--border); }}
.card-head {{ background:var(--card2); padding:8px 14px; display:flex; align-items:center; gap:10px; font-size:12px; flex-wrap:wrap; }}
.card-mark {{ font-weight:bold; font-size:14px; }}
.card-title {{ font-weight:600; }}
.card-meta {{ color:var(--muted); font-size:11px; flex:1; text-align:right; }}
.card-body {{ display:flex; gap:12px; padding:12px; flex-wrap:wrap; align-items:flex-start; }}
.card-left {{ display:flex; flex-direction:column; align-items:center; gap:4px; }}
.ql {{ font-size:10px; color:var(--muted); margin-bottom:4px; }}
.query-img {{ width:240px; border-radius:8px; }}
.query-img-sm {{ width:200px; border-radius:8px; }}
.no-img {{ width:240px; height:160px; background:var(--card2); border-radius:8px; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:11px; }}
.card-map {{ display:flex; flex-direction:column; align-items:center; }}
.mini-map {{ width:320px; height:320px; background:var(--card2); border-radius:8px; }}
.dist-row {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:4px; }}
.dist-tag {{ font-size:11px; font-weight:600; }}

.card-refs {{ padding:10px 14px; border-top:1px solid var(--border); }}
.cand-strip {{ margin-bottom:8px; }}
.cand-header {{ font-size:11px; margin-bottom:4px; display:flex; align-items:center; gap:6px; }}
.cand-rank {{ color:#fff; font-size:10px; font-weight:700; padding:1px 6px; border-radius:4px; }}
.cand-score {{ color:var(--muted); font-size:10px; margin-left:auto; }}
.cand-photos {{ display:flex; gap:5px; flex-wrap:wrap; }}
.cand-photo {{ text-align:center; }}
.cand-photo img {{ width:130px; border-radius:5px; }}
.cand-slot {{ font-size:8px; color:var(--muted); }}
.true-strip {{ border-top:1px dashed var(--green); padding-top:8px; }}

.ok-item {{ background:var(--card); border-radius:8px; padding:8px 12px; margin-bottom:4px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; cursor:pointer; border:1px solid var(--border); font-size:12px; }}
.ok-item:hover {{ border-color:var(--green); }}
.ok-check {{ color:var(--green); font-weight:bold; }}
.ok-wp {{ font-weight:600; min-width:50px; }}
.ok-score {{ color:var(--green); font-weight:600; min-width:45px; }}
.ok-detail {{ color:var(--muted); font-size:11px; flex:1; }}
.ok-map {{ width:100%; margin-top:8px; }}

.toggle-btn {{ background:var(--card2); color:var(--accent); border:1px solid var(--border); border-radius:8px; padding:5px 12px; font-size:11px; cursor:pointer; font-family:inherit; font-weight:600; }}

@media (max-width:700px) {{
  .card-body {{ flex-direction:column; align-items:center; }}
  .mini-map {{ width:280px; height:280px; }}
  .query-img {{ width:200px; }}
}}
</style>
</head>
<body>
<div class="container">
<h1>VLM 定位測試 — 拓撲地圖 + 候選照片</h1>
<p class="sub">加厚 + 起始點合併 | {meta["total"]} 張照片 × {meta["num_waypoints"]} 航點 | {meta["timestamp"]}</p>

<div class="hero">
  <div class="stat"><div class="stat-val" style="color:var(--green)">{t1p:.0f}%</div><div class="stat-label">Top-1 ({meta["top1"]}/{meta["total"]})</div></div>
  <div class="stat"><div class="stat-val" style="color:var(--yellow)">{t3p:.0f}%</div><div class="stat-label">Top-3 ({meta["top3"]}/{meta["total"]})</div></div>
  <div class="stat"><div class="stat-val" style="color:var(--cyan)">{t5p:.0f}%</div><div class="stat-label">Top-5 ({meta["top5"]}/{meta["total"]})</div></div>
  <div class="stat"><div class="stat-val" style="color:var(--red)">{err_count}</div><div class="stat-label">Top-1 錯誤</div></div>
</div>

<div class="legend">
  <span style="color:var(--green)">◉ 正確位置</span>
  <span style="color:var(--red)">● #1 預測</span>
  <span style="color:var(--yellow)">● #2 預測</span>
  <span style="color:var(--cyan)">● #3 預測</span>
  <span style="color:var(--muted)">--- 虛線=距離(m)</span>
</div>

<div class="section">
  <div class="section-title">Top-1 錯誤（{err_count} 筆）— 查詢照片 + 地圖位置 + 前三名候選照片</div>
  {all_cards_html}
</div>

<div class="section">
  <div class="section-title">Top-1 正確（{ok_count} 筆）— 點擊展開地圖</div>
  <button class="toggle-btn" id="toggle-ok" onclick="toggleOk()">展開全部 ▸</button>
  <div id="ok-section" style="display:none;margin-top:10px">
    {correct_cards_html}
  </div>
</div>

</div>
<script>
function toggleOk() {{
  const el = document.getElementById('ok-section');
  const btn = document.getElementById('toggle-ok');
  if (el.style.display === 'none') {{
    el.style.display = 'block';
    btn.textContent = '收起 ▾';
  }} else {{
    el.style.display = 'none';
    btn.textContent = '展開全部 ▸';
  }}
}}
document.querySelectorAll('.ok-head').forEach(h => {{
  h.addEventListener('click', () => {{
    const refs = h.closest('.ok-card').querySelector('.ok-refs');
    if (refs) refs.style.display = refs.style.display === 'none' ? 'block' : 'none';
  }});
}});
</script>
</body>
</html>'''

with open(OUTPUT_HTML, "w") as f:
    f.write(report_html)
sz = OUTPUT_HTML.stat().st_size
print(f"\nReport: {OUTPUT_HTML}")
print(f"Size: {sz/1024:.0f} KB ({sz/1024/1024:.1f} MB)")
