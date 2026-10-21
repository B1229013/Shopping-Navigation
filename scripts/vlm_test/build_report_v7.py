"""Build comprehensive HTML report: errors + sample corrects, top-5 with grid matching, tech explanation."""
import base64, io, json, html as html_mod, math, os, random
from pathlib import Path
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/Users/shingchou/Downloads/學校家樂福/0916測試"))
INPUT_JSON = DATA_ROOT / "檢索圖片" / "full_test_v4_merged_results.json"
WP_PHOTOS_JSON = SCRIPT_DIR / "wp_photos.json"
TOPO_JSON = SCRIPT_DIR / "topo_map_data.json"
OUTPUT_HTML = SCRIPT_DIR / "vlm_report_v7_full.html"

with open(INPUT_JSON) as f:
    data = json.load(f)
with open(WP_PHOTOS_JSON) as f:
    wp_photos = json.load(f)
with open(TOPO_JSON) as f:
    topo = json.load(f)

meta = data["meta"]
results = data["results"]

wp_coords = {}
for w in topo["waypoints"]:
    wp_coords[w["nid"]] = (w["x"], w["y"])
edges = [(e["from_nid"], e["to_nid"]) for e in topo["edges"]]

GRID_ORDER = ["top-left", "top-center", "top-right",
              "mid-left", "mid-center", "mid-right",
              "bottom-left", "bottom-center", "bottom-right"]
GRID_LABELS = {"top-left": "左上", "top-center": "上中", "top-right": "右上",
               "mid-left": "左中", "mid-center": "中心", "mid-right": "右中",
               "bottom-left": "左下", "bottom-center": "下中", "bottom-right": "右下"}


def resolve_photo_path(neo4j_path):
    if not neo4j_path:
        return None
    parts = neo4j_path.split("/", 1)
    if len(parts) == 2:
        folder = parts[0].replace("set", "")
        return DATA_ROOT / folder / parts[1]
    return None


def photo_to_b64(path_or_fname, max_size=400):
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
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None


def get_ref_photos_b64(nid, max_size=220):
    nid_str = str(nid)
    if nid_str not in wp_photos:
        return {}
    result = {}
    for slot, neo_path in wp_photos[nid_str].items():
        base_dir = slot.split("_")[0] if "_" in slot else slot
        if base_dir not in ["front", "back", "left", "right"]:
            continue
        if base_dir in result:
            continue
        fpath = resolve_photo_path(neo_path)
        if fpath:
            b64 = photo_to_b64(fpath, max_size)
            if b64:
                result[base_dir] = b64
    return result


def esc(s):
    return html_mod.escape(str(s))


# Separate errors and corrects
errors_list = [r for r in results if not r["is_top1"]]
corrects_list = [r for r in results if r["is_top1"]]
errors_list.sort(key=lambda x: x.get("gap", 0), reverse=True)

# Pick sample of correct results: 7 highest, 7 lowest, 6 random middle
corrects_sorted = sorted(corrects_list, key=lambda x: x["top1_score"])
n_correct = len(corrects_sorted)
sample_correct = []
if n_correct > 20:
    sample_correct += corrects_sorted[-7:]  # highest
    sample_correct += corrects_sorted[:7]   # lowest
    middle = corrects_sorted[7:-7]
    random.seed(42)
    sample_correct += random.sample(middle, min(6, len(middle)))
else:
    sample_correct = corrects_sorted
sample_correct.sort(key=lambda x: x["top1_score"])

all_display = errors_list + sample_correct
print(f"Cards: {len(errors_list)} errors + {len(sample_correct)} correct samples = {len(all_display)}", flush=True)

# Pre-cache thumbnails
print("Generating query thumbnails...", flush=True)
query_cache = {}
for r in all_display:
    b64 = photo_to_b64(r["fname"], 400)
    if b64:
        query_cache[r["fname"]] = b64
print(f"  {len(query_cache)} query thumbnails", flush=True)

print("Generating reference thumbnails...", flush=True)
ref_cache = {}
ref_nids = set()
for r in all_display:
    ref_nids.add(r["true_nid"])
    for c in r.get("top5", [])[:3]:
        ref_nids.add(c["nid"])
for i, nid in enumerate(sorted(ref_nids)):
    ref_cache[nid] = get_ref_photos_b64(nid, 220)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(ref_nids)} waypoints", flush=True)
print(f"  {len(ref_cache)} waypoint ref sets cached", flush=True)

# SVG mini-map setup
all_x = [w["x"] for w in topo["waypoints"] if w["x"] != 0 or w["y"] != 0]
all_y = [w["y"] for w in topo["waypoints"] if w["x"] != 0 or w["y"] != 0]
min_x, max_x = min(all_x) - 2, max(all_x) + 2
min_y, max_y = min(all_y) - 2, max(all_y) + 2
map_w, map_h = 340, 340
scale_x = map_w / (max_x - min_x)
scale_y = map_h / (max_y - min_y)
scale = min(scale_x, scale_y)
off_x = (map_w - (max_x - min_x) * scale) / 2
off_y = (map_h - (max_y - min_y) * scale) / 2


def to_svg(x, y):
    if x == 0 and y == 0:
        return None, None
    sx = off_x + (x - min_x) * scale
    sy = map_h - (off_y + (y - min_y) * scale)
    return round(sx, 1), round(sy, 1)


def build_mini_map_svg(true_nid, top3_nids):
    svg = [f'<svg viewBox="0 0 {map_w} {map_h}" class="mini-map">']
    for a_nid, b_nid in edges:
        ax, ay = wp_coords.get(a_nid, (0, 0))
        bx, by = wp_coords.get(b_nid, (0, 0))
        sa = to_svg(ax, ay)
        sb = to_svg(bx, by)
        if sa[0] is not None and sb[0] is not None:
            svg.append(f'<line x1="{sa[0]}" y1="{sa[1]}" x2="{sb[0]}" y2="{sb[1]}" stroke="var(--edge-c)" stroke-width="1" opacity="0.3"/>')
    for w in topo["waypoints"]:
        sx, sy = to_svg(w["x"], w["y"])
        if sx is None:
            continue
        svg.append(f'<circle cx="{sx}" cy="{sy}" r="3" fill="var(--wp-dot)" opacity="0.3"/>')
        svg.append(f'<text x="{sx}" y="{sy-5}" text-anchor="middle" font-size="7" fill="var(--wp-lbl)" opacity="0.4">{w["nid"]}</text>')
    colors = ["#ef4444", "#f59e0b", "#06b6d4"]
    sizes = [8, 7, 6]
    for idx in reversed(range(min(3, len(top3_nids)))):
        nid = top3_nids[idx]
        cx, cy = wp_coords.get(nid, (0, 0))
        sx, sy = to_svg(cx, cy)
        if sx is None:
            continue
        svg.append(f'<circle cx="{sx}" cy="{sy}" r="{sizes[idx]}" fill="{colors[idx]}" opacity="0.85" stroke="white" stroke-width="1.5"/>')
        svg.append(f'<text x="{sx}" y="{sy-sizes[idx]-3}" text-anchor="middle" font-size="9" font-weight="bold" fill="{colors[idx]}">#{idx+1}</text>')
    tx, ty = wp_coords.get(true_nid, (0, 0))
    tsx, tsy = to_svg(tx, ty)
    if tsx is not None:
        svg.append(f'<circle cx="{tsx}" cy="{tsy}" r="10" fill="none" stroke="#10b981" stroke-width="2.5"/>')
        svg.append(f'<circle cx="{tsx}" cy="{tsy}" r="4" fill="#10b981"/>')
        svg.append(f'<text x="{tsx}" y="{tsy-13}" text-anchor="middle" font-size="10" font-weight="bold" fill="#10b981">WP{true_nid}</text>')
    if tsx is not None:
        for idx in range(min(3, len(top3_nids))):
            nid = top3_nids[idx]
            if nid == true_nid:
                continue
            cx, cy = wp_coords.get(nid, (0, 0))
            sx, sy = to_svg(cx, cy)
            if sx is None:
                continue
            svg.append(f'<line x1="{tsx}" y1="{tsy}" x2="{sx}" y2="{sy}" stroke="{colors[idx]}" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>')
            mx, my = (tsx + sx) / 2, (tsy + sy) / 2
            dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
            svg.append(f'<text x="{mx}" y="{my-4}" text-anchor="middle" font-size="8" fill="{colors[idx]}">{dist:.1f}m</text>')
    svg.append('</svg>')
    return '\n'.join(svg)


def build_grid_viz(cell_matches, items):
    """Build 3x3 grid visualization of cell matching."""
    # Build query items per cell
    q_by_cell = {}
    for it in (items or []):
        cell = it.get("cell", "mid-center")
        if cell not in q_by_cell:
            q_by_cell[cell] = []
        q_by_cell[cell].append(it)

    html = '<div class="grid-viz">'
    for cell in GRID_ORDER:
        matches = cell_matches.get(cell, [])
        q_items = q_by_cell.get(cell, [])
        has_match = len(matches) > 0
        cls = "grid-cell match" if has_match else "grid-cell"
        html += f'<div class="{cls}">'
        html += f'<div class="gc-label">{GRID_LABELS[cell]}</div>'
        if matches:
            for m in matches[:2]:  # max 2 per cell to save space
                cat_cls = m.get("cat", "normal")
                sim_pct = m.get("sim", 0) * 100
                html += f'<div class="gc-match {cat_cls}" title="{esc(m.get("q",""))} ↔ {esc(m.get("r",""))}">'
                html += f'<span class="gc-q">{esc(m.get("q","")[:8])}</span>'
                html += f'<span class="gc-arrow">↔</span>'
                html += f'<span class="gc-r">{esc(m.get("r","")[:8])}</span>'
                html += f'<span class="gc-sim">{sim_pct:.0f}%</span>'
                html += '</div>'
            if len(matches) > 2:
                html += f'<div class="gc-more">+{len(matches)-2} more</div>'
        elif q_items:
            for qi in q_items[:1]:
                html += f'<div class="gc-unmatched">{esc(qi.get("zh","")[:10])}</div>'
        html += '</div>'
    html += '</div>'
    return html


def build_ref_strip(nid, rank, score, color):
    refs = ref_cache.get(nid, {})
    html = f'<div class="cand-strip">'
    html += f'<div class="cand-hdr"><span class="cand-rank" style="background:{color}">#{rank}</span> WP{nid} <span class="cand-score">{score*100:.1f}%</span></div>'
    html += '<div class="cand-photos">'
    for slot in ["front", "back", "left", "right"]:
        b64 = refs.get(slot)
        if b64:
            html += f'<div class="cand-photo"><img src="data:image/jpeg;base64,{b64}"><div class="cand-slot">{slot}</div></div>'
    html += '</div></div>'
    return html


# ── Build cards ──
print("Building cards...", flush=True)

error_cards = ""
for idx, r in enumerate(errors_list):
    true_nid = r["true_nid"]
    top5 = r.get("top5", [])
    top3 = top5[:3]
    top3_nids = [c["nid"] for c in top3]

    mini_map = build_mini_map_svg(true_nid, top3_nids)
    qimg = query_cache.get(r["fname"], "")
    qimg_html = f'<img class="query-img" src="data:image/jpeg;base64,{qimg}">' if qimg else '<div class="no-img">無圖片</div>'

    # Top-5 ranking table
    rank_rows = ""
    for j, c in enumerate(top5[:5]):
        is_true = c["nid"] == true_nid
        row_cls = "rank-true" if is_true else ""
        lm_stars = "★" * c.get("lm", 0) if c.get("lm", 0) > 0 else "-"
        n_matches = sum(len(v) for v in c.get("cell_matches", {}).values())
        rank_rows += f'<tr class="{row_cls}"><td>#{j+1}</td><td>WP{c["nid"]}</td><td>{c["score"]*100:.1f}%</td><td>{lm_stars}</td><td>{n_matches}</td></tr>'
    if true_nid not in [c["nid"] for c in top5[:5]]:
        true_rank = r.get("true_rank", "?")
        true_score = r.get("true_score", 0)
        rank_rows += f'<tr class="rank-true"><td>#{true_rank}</td><td>WP{true_nid}</td><td>{true_score*100:.1f}%</td><td>-</td><td>-</td></tr>'

    # Grid matching for #1 prediction
    top1_cells = top5[0].get("cell_matches", {}) if top5 else {}
    grid_html = build_grid_viz(top1_cells, r.get("items", []))

    # Ref photo strips (top-3 + correct)
    colors = ["#ef4444", "#f59e0b", "#06b6d4"]
    ref_strips = ""
    for j, c in enumerate(top3):
        ref_strips += build_ref_strip(c["nid"], j + 1, c["score"], colors[j])
    if true_nid not in top3_nids:
        true_score = r.get("true_score", 0)
        true_rank = r.get("true_rank", "?")
        ref_strips += f'<div class="cand-strip true-strip">'
        ref_strips += f'<div class="cand-hdr"><span class="cand-rank" style="background:#10b981">正確 #{true_rank}</span> WP{true_nid} <span class="cand-score">{true_score*100:.1f}%</span></div>'
        refs = ref_cache.get(true_nid, {})
        ref_strips += '<div class="cand-photos">'
        for slot in ["front", "back", "left", "right"]:
            b64 = refs.get(slot)
            if b64:
                ref_strips += f'<div class="cand-photo"><img src="data:image/jpeg;base64,{b64}"><div class="cand-slot">{slot}</div></div>'
        ref_strips += '</div></div>'

    mark_color = "#f59e0b" if r["is_top3"] else ("#06b6d4" if r["is_top5"] else "#ef4444")
    mark = "△ Top-3" if r["is_top3"] else ("⬤ Top-5" if r["is_top5"] else "✗ 錯誤")

    error_cards += f'''
<div class="card error-card">
  <div class="card-head">
    <span class="card-mark" style="color:{mark_color}">{mark}</span>
    <span class="card-title">WP{true_nid} → WP{r["pred_nid"]}</span>
    <span class="card-idx">#{idx+1}/{len(errors_list)}</span>
    <span class="card-meta">{r["set"]}/{r["direction"]} | {esc(r.get("scene","")[:50])}</span>
  </div>
  <div class="card-upper">
    <div class="card-col">
      <div class="lbl">查詢照片</div>
      {qimg_html}
    </div>
    <div class="card-col">
      <div class="lbl">拓撲地圖</div>
      {mini_map}
    </div>
  </div>
  <div class="card-analysis">
    <div class="card-col">
      <div class="lbl">Top-5 排名</div>
      <table class="rank-tbl">
        <thead><tr><th>名次</th><th>航點</th><th>信心值</th><th>地標</th><th>匹配數</th></tr></thead>
        <tbody>{rank_rows}</tbody>
      </table>
    </div>
    <div class="card-col">
      <div class="lbl">九宮格匹配（#1 WP{top5[0]["nid"] if top5 else "?"}）</div>
      {grid_html}
    </div>
  </div>
  <div class="card-refs">
    <div class="lbl">前三名 + 正確答案 參考照片</div>
    {ref_strips}
  </div>
</div>'''

# Correct sample cards
correct_cards = ""
for idx, r in enumerate(sample_correct):
    true_nid = r["true_nid"]
    top5 = r.get("top5", [])
    top3 = top5[:3]
    top3_nids = [c["nid"] for c in top3]

    mini_map = build_mini_map_svg(true_nid, top3_nids)
    qimg = query_cache.get(r["fname"], "")
    qimg_html = f'<img class="query-img" src="data:image/jpeg;base64,{qimg}">' if qimg else '<div class="no-img">無圖片</div>'

    rank_rows = ""
    for j, c in enumerate(top5[:5]):
        is_true = c["nid"] == true_nid
        row_cls = "rank-true" if is_true else ""
        lm_stars = "★" * c.get("lm", 0) if c.get("lm", 0) > 0 else "-"
        n_matches = sum(len(v) for v in c.get("cell_matches", {}).values())
        rank_rows += f'<tr class="{row_cls}"><td>#{j+1}</td><td>WP{c["nid"]}</td><td>{c["score"]*100:.1f}%</td><td>{lm_stars}</td><td>{n_matches}</td></tr>'

    top1_cells = top5[0].get("cell_matches", {}) if top5 else {}
    grid_html = build_grid_viz(top1_cells, r.get("items", []))

    colors_c = ["#10b981", "#f59e0b", "#06b6d4"]
    ref_strips = ""
    for j, c in enumerate(top3):
        ref_strips += build_ref_strip(c["nid"], j + 1, c["score"], colors_c[j])

    correct_cards += f'''
<div class="card ok-card">
  <div class="card-head ok-head">
    <span class="card-mark" style="color:#10b981">✓ 正確</span>
    <span class="card-title">WP{true_nid}</span>
    <span class="card-idx">#{idx+1}/{len(sample_correct)}</span>
    <span class="card-meta">{r["set"]}/{r["direction"]} | {r["top1_score"]*100:.1f}% | {esc(r.get("scene","")[:50])}</span>
  </div>
  <div class="card-upper">
    <div class="card-col">
      <div class="lbl">查詢照片</div>
      {qimg_html}
    </div>
    <div class="card-col">
      <div class="lbl">拓撲地圖</div>
      {mini_map}
    </div>
  </div>
  <div class="card-analysis">
    <div class="card-col">
      <div class="lbl">Top-5 排名</div>
      <table class="rank-tbl">
        <thead><tr><th>名次</th><th>航點</th><th>信心值</th><th>地標</th><th>匹配數</th></tr></thead>
        <tbody>{rank_rows}</tbody>
      </table>
    </div>
    <div class="card-col">
      <div class="lbl">九宮格匹配（#1 WP{top5[0]["nid"] if top5 else "?"}）</div>
      {grid_html}
    </div>
  </div>
  <div class="card-refs">
    <div class="lbl">前三名參考照片</div>
    {ref_strips}
  </div>
</div>'''

# ── Technical explanation ──
tech_html = '''
<div class="tech-section">
<h2>定位系統技術流程</h2>
<p class="tech-intro">本系統透過視覺語言模型（VLM）分析超市內部照片，與預建的參考資料庫進行比對，以確定使用者在拓撲地圖上的位置。以下說明各階段的詳細實作。</p>

<div class="tech-step">
<div class="step-num">1</div>
<div class="step-body">
<h3>VLM 感知（Perception）</h3>
<p>使用 <strong>GPT-5.4-mini</strong> 視覺語言模型，對查詢照片進行物件偵測與 OCR 文字辨識。</p>
<ul>
<li>輸入：使用者拍攝的照片（縮放至 1024px，JPEG 壓縮）</li>
<li>輸出：JSON 格式的偵測清單，每項包含繁體中文名稱、英文名稱、信心值、bounding box（歸一化 [x1, y1, x2, y2]）</li>
<li>指導 prompt 要求模型只回報確實看到的物件，不猜測，並特別關注走道吊牌和區域標示</li>
<li>每張照片約回報 5-12 個精確偵測</li>
<li><strong>非確定性</strong>：相同照片多次呼叫 VLM 會產生不同結果，此特性被利用於資料庫加厚</li>
</ul>
</div>
</div>

<div class="tech-step">
<div class="step-num">2</div>
<div class="step-body">
<h3>九宮格編碼（9-Cell Grid Encoding）</h3>
<p>將照片分為 3×3 的九宮格，根據 bounding box 中心座標將物件分配到對應格子。</p>
<table class="tech-tbl">
<tr><td>左上<br><code>x&lt;⅓, y&lt;⅓</code></td><td>上中<br><code>⅓≤x≤⅔, y&lt;⅓</code></td><td>右上<br><code>x&gt;⅔, y&lt;⅓</code></td></tr>
<tr><td>左中<br><code>x&lt;⅓, ⅓≤y≤⅔</code></td><td>中心<br><code>⅓≤x≤⅔, ⅓≤y≤⅔</code></td><td>右中<br><code>x&gt;⅔, ⅓≤y≤⅔</code></td></tr>
<tr><td>左下<br><code>x&lt;⅓, y&gt;⅔</code></td><td>下中<br><code>⅓≤x≤⅔, y&gt;⅔</code></td><td>右下<br><code>x&gt;⅔, y&gt;⅔</code></td></tr>
</table>
<p>此編碼保留了空間位置資訊，使比對不只看「有什麼」，也看「在哪裡」。</p>
</div>
</div>

<div class="tech-step">
<div class="step-num">3</div>
<div class="step-body">
<h3>特徵分類與加權（Feature Classification & Weighting）</h3>
<p>根據偵測物件的語義類型，賦予不同的比對權重：</p>
<table class="weight-tbl">
<thead><tr><th>類型</th><th>權重</th><th>判定規則</th><th>範例</th></tr></thead>
<tbody>
<tr class="w-landmark"><td>地標 Landmark</td><td><strong>5.0×</strong></td><td>含走道/區域/出口等關鍵字，或同時含標示詞+分類詞</td><td>走道5、堅果海苔吊牌、出口</td></tr>
<tr class="w-sign"><td>標示 Sign</td><td><strong>3.0×</strong></td><td>含標示/看板/招牌等詞但無分類詞</td><td>促銷看板、品牌招牌</td></tr>
<tr class="w-equip"><td>設備 Equipment</td><td><strong>2.0×</strong></td><td>含冷藏櫃/冰箱/展示櫃/貨架等設備名</td><td>冷藏展示櫃、貨架</td></tr>
<tr class="w-normal"><td>一般 Normal</td><td><strong>1.0×</strong></td><td>不符合以上分類的物件</td><td>牛奶、洋芋片、購物車</td></tr>
<tr class="w-generic"><td>通用 Generic</td><td><strong>0.3×</strong></td><td>含促銷/價格等普遍存在的標記</td><td>促銷立牌、價格標示</td></tr>
</tbody>
</table>
<p>設計理念：地標和區域標示是最強的定位特徵（每個位置幾乎唯一），而促銷價格牌幾乎到處都有，給予低權重避免誤導。</p>
</div>
</div>

<div class="tech-step">
<div class="step-num">4</div>
<div class="step-body">
<h3>加權九宮格比對（Weighted Grid Matching）</h3>
<p>對每個參考航點的每張方向照片，進行逐格比對：</p>
<ol>
<li><strong>同格匹配</strong>：查詢物件與同一格子中的參考物件計算語義相似度（100% 位置獎勵）</li>
<li><strong>鄰格匹配</strong>：若同格無匹配，擴展到相鄰格子搜尋（70% 位置獎勵）</li>
<li><strong>語義相似度</strong>：綜合考慮詞彙重疊、字元重疊、數字匹配、商品分類匹配、子字串包含等多種相似度度量，取最高分</li>
<li><strong>匹配門檻</strong>：相似度 &gt; 0.25 才算有效匹配</li>
<li><strong>最終分數</strong>：Σ(物件權重 × 相似度 × 位置獎勵) / Σ(所有物件權重)</li>
</ol>
<p>每個航點取其所有方向照片（front/back/left/right）中最高的比對分數作為該航點的分數。</p>
</div>
</div>

<div class="tech-step">
<div class="step-num">5</div>
<div class="step-body">
<h3>參考資料庫加厚（Reference DB Enrichment）</h3>
<p>利用 VLM 的非確定性特性，對每張參考照片額外呼叫 VLM 2 次，將新偵測的物件合併回資料庫。</p>
<ul>
<li>去重策略：同一格或相鄰格中，若已存在高度相似的標籤（相似度 &gt; 0.6），則跳過</li>
<li>結果：260 張參考照片 × 2 次 = 520 次 VLM 呼叫</li>
<li>新增 1,879 個物件（從 2,190 增加到 4,069），<strong>增長 86%</strong></li>
<li>每張照片平均物件數 8.4 → 15.6</li>
<li>Top-1 準確率從 62.3% 提升至 91.5%</li>
</ul>
</div>
</div>

<div class="tech-step">
<div class="step-num">6</div>
<div class="step-body">
<h3>起始點合併（Start Point Merging）</h3>
<p>三組資料的起始點（WP25、WP28、WP34）為同一物理位置（超市入口），將其合併為 WP25。</p>
<ul>
<li>合併策略：保留所有方向的照片（以後綴區分來源），轉移拓撲連接邊</li>
<li>合併後：61 → 59 個航點，WP25 擁有 12 張方向照片</li>
<li>效果：消除了起始點間的歧義匹配</li>
</ul>
</div>
</div>

<div class="tech-step">
<div class="step-num">7</div>
<div class="step-body">
<h3>系統架構與效能</h3>
<ul>
<li><strong>資料庫</strong>：Neo4j 圖資料庫，Schema：<code>(:Waypoint)-[:HAS_PHOTO]->(:DirPhoto)-[:DETECTED]->(:Object)</code></li>
<li><strong>拓撲結構</strong>：航點間以 <code>WALKWAY</code> 邊相連，記錄 PDR 步行距離</li>
<li><strong>API 延遲</strong>：VLM 呼叫約 5 秒/張，佔總時間 97%；比對計算僅 0.16 秒</li>
<li><strong>模型</strong>：GPT-5.4-mini via CGU proxy，temperature=0.1</li>
</ul>
</div>
</div>
</div>
'''

# ── Assemble HTML ──
err_count = len(errors_list)
ok_count = len(corrects_list)
sample_count = len(sample_correct)
t1p = meta["top1"] / meta["total"] * 100
t3p = meta["top3"] / meta["total"] * 100
t5p = meta["top5"] / meta["total"] * 100

# Set stats
set_stats = data.get("set_stats", {})
dir_stats = data.get("dir_stats", {})
set_rows = ""
for s in sorted(set_stats):
    ss = set_stats[s]
    set_rows += f'<tr><td>Set {s}</td><td>{ss["top1"]}/{ss["total"]}</td><td>{ss["top1"]/ss["total"]*100:.1f}%</td></tr>'
dir_rows = ""
for d in ["front", "back", "left", "right"]:
    ds = dir_stats.get(d, {"total": 0, "top1": 0})
    if ds["total"]:
        dir_rows += f'<tr><td>{d}</td><td>{ds["top1"]}/{ds["total"]}</td><td>{ds["top1"]/ds["total"]*100:.1f}%</td></tr>'

report_html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>VLM 定位完整分析</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg: #0c0f1a; --bg2: #141828; --card: #1a1f36; --card2: #232847;
  --accent: #60a5fa; --accent2: #818cf8;
  --text: #e8ecf4; --text2: #c4cbda; --muted: #7c85a0;
  --green: #34d399; --yellow: #fbbf24; --red: #f87171; --cyan: #22d3ee;
  --border: rgba(130,140,180,0.12); --edge-c: #5a6380; --wp-dot: #4a5270; --wp-lbl: #6b7394;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    --bg: #f5f7fb; --bg2: #edf0f7; --card: #ffffff; --card2: #f0f2f8;
    --accent: #3b82f6; --accent2: #6366f1;
    --text: #111827; --text2: #374151; --muted: #6b7280;
    --green: #059669; --yellow: #d97706; --red: #dc2626; --cyan: #0891b2;
    --border: rgba(15,23,42,0.08); --edge-c: #94a3b8; --wp-dot: #94a3b8; --wp-lbl: #64748b;
  }}
}}
:root[data-theme="light"] {{
  --bg: #f5f7fb; --bg2: #edf0f7; --card: #ffffff; --card2: #f0f2f8;
  --accent: #3b82f6; --accent2: #6366f1;
  --text: #111827; --text2: #374151; --muted: #6b7280;
  --green: #059669; --yellow: #d97706; --red: #dc2626; --cyan: #0891b2;
  --border: rgba(15,23,42,0.08); --edge-c: #94a3b8; --wp-dot: #94a3b8; --wp-lbl: #64748b;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter','Noto Sans TC',sans-serif; background:var(--bg); color:var(--text); padding:16px; line-height:1.6; }}
.wrap {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:24px; font-weight:700; color:var(--accent); text-align:center; }}
h2 {{ font-size:18px; font-weight:700; color:var(--accent2); margin:28px 0 12px; }}
.sub {{ text-align:center; color:var(--muted); font-size:12px; margin:4px 0 20px; }}

.hero {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin-bottom:20px; }}
.stat {{ background:var(--card); border-radius:10px; padding:14px 10px; text-align:center; border:1px solid var(--border); }}
.stat-val {{ font-size:28px; font-weight:700; font-variant-numeric:tabular-nums; }}
.stat-lbl {{ font-size:10px; color:var(--muted); margin-top:2px; }}

.breakdown {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
.breakdown table {{ background:var(--card); border-radius:8px; border:1px solid var(--border); border-collapse:collapse; font-size:12px; }}
.breakdown th {{ background:var(--card2); padding:6px 12px; font-weight:600; text-align:left; }}
.breakdown td {{ padding:5px 12px; border-top:1px solid var(--border); }}

.legend {{ background:var(--card); border-radius:8px; padding:10px 16px; font-size:11px; display:flex; gap:14px; flex-wrap:wrap; border:1px solid var(--border); margin-bottom:20px; }}

.section-title {{ font-size:15px; font-weight:700; color:var(--accent); margin:24px 0 10px; display:flex; align-items:center; gap:8px; }}
.section-title::before {{ content:''; width:4px; height:16px; background:var(--accent); border-radius:2px; }}

.card {{ background:var(--card); border-radius:12px; margin-bottom:14px; overflow:hidden; border:1px solid var(--border); }}
.card-head {{ background:var(--card2); padding:10px 16px; display:flex; align-items:center; gap:10px; font-size:12px; flex-wrap:wrap; }}
.card-mark {{ font-weight:700; font-size:13px; white-space:nowrap; }}
.card-title {{ font-weight:600; font-size:14px; }}
.card-idx {{ color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
.card-meta {{ color:var(--muted); font-size:11px; flex:1; text-align:right; }}

.card-upper {{ display:flex; gap:16px; padding:14px 16px; flex-wrap:wrap; }}
.card-col {{ display:flex; flex-direction:column; align-items:center; gap:4px; }}
.lbl {{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }}
.query-img {{ width:340px; border-radius:8px; }}
.no-img {{ width:340px; height:220px; background:var(--card2); border-radius:8px; display:flex; align-items:center; justify-content:center; color:var(--muted); }}
.mini-map {{ width:340px; height:340px; background:var(--card2); border-radius:8px; }}

.card-analysis {{ display:flex; gap:16px; padding:0 16px 14px; flex-wrap:wrap; align-items:flex-start; }}
.rank-tbl {{ border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; min-width:260px; }}
.rank-tbl th {{ background:var(--card2); padding:5px 10px; font-weight:600; text-align:left; font-size:11px; }}
.rank-tbl td {{ padding:4px 10px; border-top:1px solid var(--border); }}
.rank-true {{ background:rgba(16,185,129,0.1); }}
.rank-true td {{ color:var(--green); font-weight:600; }}

.grid-viz {{ display:grid; grid-template-columns:repeat(3,1fr); gap:3px; width:340px; }}
.grid-cell {{ background:var(--card2); border-radius:4px; padding:4px; min-height:40px; font-size:9px; overflow:hidden; }}
.grid-cell.match {{ background:rgba(96,165,250,0.12); border:1px solid rgba(96,165,250,0.25); }}
.gc-label {{ font-size:8px; color:var(--muted); margin-bottom:2px; font-weight:600; }}
.gc-match {{ display:flex; align-items:center; gap:2px; margin-bottom:1px; white-space:nowrap; overflow:hidden; }}
.gc-q {{ color:var(--yellow); font-family:'JetBrains Mono',monospace; font-size:8px; }}
.gc-arrow {{ color:var(--muted); font-size:7px; }}
.gc-r {{ color:var(--cyan); font-family:'JetBrains Mono',monospace; font-size:8px; }}
.gc-sim {{ color:var(--green); font-size:8px; margin-left:auto; font-weight:600; }}
.gc-match.landmark {{ border-left:2px solid var(--red); padding-left:2px; }}
.gc-match.sign {{ border-left:2px solid var(--yellow); padding-left:2px; }}
.gc-match.equipment {{ border-left:2px solid var(--accent); padding-left:2px; }}
.gc-unmatched {{ color:var(--muted); font-size:8px; font-style:italic; }}
.gc-more {{ color:var(--muted); font-size:8px; }}

.card-refs {{ padding:12px 16px; border-top:1px solid var(--border); }}
.cand-strip {{ margin-bottom:10px; }}
.cand-hdr {{ font-size:12px; margin-bottom:4px; display:flex; align-items:center; gap:6px; }}
.cand-rank {{ color:#fff; font-size:10px; font-weight:700; padding:2px 8px; border-radius:4px; }}
.cand-score {{ color:var(--muted); font-size:11px; margin-left:auto; font-variant-numeric:tabular-nums; }}
.cand-photos {{ display:flex; gap:6px; flex-wrap:wrap; }}
.cand-photo {{ text-align:center; }}
.cand-photo img {{ width:200px; border-radius:6px; }}
.cand-slot {{ font-size:9px; color:var(--muted); margin-top:2px; }}
.true-strip {{ border-top:1px dashed var(--green); padding-top:10px; }}

.ok-card .card-head {{ border-left:3px solid var(--green); }}
.error-card .card-head {{ border-left:3px solid var(--red); }}

.toggle-btn {{ background:var(--card2); color:var(--accent); border:1px solid var(--border); border-radius:8px; padding:6px 14px; font-size:12px; cursor:pointer; font-family:inherit; font-weight:600; }}
.toggle-btn:hover {{ border-color:var(--accent); }}

/* Tech section */
.tech-section {{ background:var(--card); border-radius:12px; padding:24px; margin-bottom:24px; border:1px solid var(--border); }}
.tech-section h2 {{ text-align:left; margin:0 0 12px; }}
.tech-intro {{ color:var(--text2); font-size:13px; margin-bottom:20px; }}
.tech-step {{ display:flex; gap:14px; margin-bottom:20px; }}
.step-num {{ width:32px; height:32px; background:var(--accent); color:var(--bg); border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px; flex-shrink:0; }}
.step-body {{ flex:1; }}
.step-body h3 {{ font-size:14px; font-weight:700; color:var(--text); margin-bottom:6px; }}
.step-body p {{ font-size:12px; color:var(--text2); margin-bottom:6px; }}
.step-body ul, .step-body ol {{ font-size:12px; color:var(--text2); padding-left:18px; margin-bottom:6px; }}
.step-body li {{ margin-bottom:3px; }}
.step-body code {{ background:var(--card2); padding:1px 5px; border-radius:3px; font-family:'JetBrains Mono',monospace; font-size:11px; }}
.step-body strong {{ color:var(--text); }}
.tech-tbl {{ border-collapse:collapse; margin:8px 0; font-size:11px; width:100%; max-width:400px; }}
.tech-tbl td {{ border:1px solid var(--border); padding:6px 10px; text-align:center; background:var(--card2); }}
.tech-tbl code {{ font-size:9px; }}
.weight-tbl {{ border-collapse:collapse; font-size:11px; width:100%; margin:6px 0; }}
.weight-tbl th {{ background:var(--card2); padding:5px 8px; text-align:left; font-weight:600; }}
.weight-tbl td {{ padding:4px 8px; border-top:1px solid var(--border); }}
.w-landmark td:first-child {{ color:var(--red); font-weight:600; }}
.w-sign td:first-child {{ color:var(--yellow); font-weight:600; }}
.w-equip td:first-child {{ color:var(--accent); font-weight:600; }}
.w-normal td:first-child {{ color:var(--text2); }}
.w-generic td:first-child {{ color:var(--muted); }}

.nav-tabs {{ display:flex; gap:6px; margin-bottom:16px; position:sticky; top:env(safe-area-inset-top, 0px); background:var(--bg); padding:8px 0; z-index:10; }}
.nav-tab {{ padding:8px 16px; border-radius:8px; font-size:12px; font-weight:600; cursor:pointer; border:1px solid var(--border); background:var(--card); color:var(--muted); transition:all 0.2s; }}
.nav-tab.active {{ background:var(--accent); color:var(--bg); border-color:var(--accent); }}
.nav-tab:hover:not(.active) {{ border-color:var(--accent); color:var(--accent); }}
.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}

@media (max-width:750px) {{
  .card-upper, .card-analysis {{ flex-direction:column; align-items:center; }}
  .query-img {{ width:100%; max-width:340px; }}
  .mini-map {{ width:100%; max-width:340px; height:auto; aspect-ratio:1; }}
  .grid-viz {{ width:100%; max-width:340px; }}
  .rank-tbl {{ width:100%; }}
  .cand-photo img {{ width:160px; }}
  .tech-step {{ flex-direction:column; }}
}}
</style>
</head>
<body>
<div class="wrap">

<h1>VLM 視覺定位系統 — 完整分析報告</h1>
<p class="sub">加厚 + 起始點合併 | {meta["total"]} 張測試照片 × {meta["num_waypoints"]} 航點 | {meta["timestamp"]}</p>

<div class="hero">
  <div class="stat"><div class="stat-val" style="color:var(--green)">{t1p:.1f}%</div><div class="stat-lbl">Top-1（{meta["top1"]}/{meta["total"]}）</div></div>
  <div class="stat"><div class="stat-val" style="color:var(--yellow)">{t3p:.1f}%</div><div class="stat-lbl">Top-3（{meta["top3"]}/{meta["total"]}）</div></div>
  <div class="stat"><div class="stat-val" style="color:var(--cyan)">{t5p:.1f}%</div><div class="stat-lbl">Top-5（{meta["top5"]}/{meta["total"]}）</div></div>
  <div class="stat"><div class="stat-val" style="color:var(--red)">{err_count}</div><div class="stat-lbl">Top-1 錯誤</div></div>
  <div class="stat"><div class="stat-val" style="color:var(--muted)">{meta.get("avg_time_per_photo",0):.1f}s</div><div class="stat-lbl">平均耗時/張</div></div>
</div>

<div class="breakdown">
  <table><thead><tr><th colspan="3">資料集準確率</th></tr><tr><th>Set</th><th>Top-1</th><th>率</th></tr></thead><tbody>{set_rows}</tbody></table>
  <table><thead><tr><th colspan="3">方向準確率</th></tr><tr><th>方向</th><th>Top-1</th><th>率</th></tr></thead><tbody>{dir_rows}</tbody></table>
</div>

<div class="legend">
  <span style="color:var(--green)">◉ 正確位置</span>
  <span style="color:var(--red)">● #1 預測</span>
  <span style="color:var(--yellow)">● #2 預測</span>
  <span style="color:var(--cyan)">● #3 預測</span>
  <span style="color:var(--muted)">--- 虛線=距離(m)</span>
  <span style="color:var(--muted)">| 九宮格：<span style="color:var(--yellow)">查詢</span> ↔ <span style="color:var(--cyan)">參考</span> 匹配</span>
</div>

<div class="nav-tabs">
  <div class="nav-tab active" onclick="showTab('errors')">錯誤分析（{err_count}）</div>
  <div class="nav-tab" onclick="showTab('corrects')">正確樣本（{sample_count}）</div>
  <div class="nav-tab" onclick="showTab('tech')">技術流程</div>
</div>

<div id="tab-errors" class="tab-content active">
  <div class="section-title">Top-1 錯誤詳細分析（{err_count} 筆）</div>
  {error_cards}
</div>

<div id="tab-corrects" class="tab-content">
  <div class="section-title">Top-1 正確樣本（精選 {sample_count} 筆，含高/低信心值）</div>
  {correct_cards}
</div>

<div id="tab-tech" class="tab-content">
  {tech_html}
</div>

</div>

<script>
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>'''

with open(OUTPUT_HTML, "w") as f:
    f.write(report_html)
sz = OUTPUT_HTML.stat().st_size
print(f"\nReport: {OUTPUT_HTML}")
print(f"Size: {sz/1024:.0f} KB ({sz/1024/1024:.1f} MB)")
