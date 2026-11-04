"""Build a store map in the shared V2 Neo4j schema (the format of place "A7家樂福（9/16)").

Pipeline, one run per new store (see backend/README.md「建立新地圖」):
  1. build_map_v2_topomap.py   Sensor Lab zips → <workspace>/<set>/*.jpg + topomap.json
  2. /topomap-builder           drag / merge nodes → <workspace>/topomap_confirmed.json
  3. upload_map_v2.py           confirmed map → Waypoint / DirPhoto / HAS_PHOTO / WALKWAY
  4. rebuild_objects_v2.py      VLM (+ EasyOCR) per DirPhoto → Object / DETECTED / ADJACENT

Schema written (identical to the classmate's A7 map, plus ``place`` on DirPhoto/Object so a
place can be found and replaced without walking from its Waypoints):
  (:Waypoint {place, nid, pdr_x, pdr_y, session, heading_deg, total_steps, total_distance_m})
  (:Waypoint)-[:HAS_PHOTO {slot}]->(:DirPhoto {photo_file, heading_deg, place})
  (:DirPhoto)-[:DETECTED]->(:Object {label, label_norm, score, ocr_text, bbox, grid_cell,
                                     grid_cells, piece_count, depth_class, source, place})
  (:Object)-[:ADJACENT {cell_pairs, horizontal_count, vertical_count, diagonal_count,
                        overlap, depth_diff}]->(:Object)          (same photo only)
  (:Waypoint)-[:WALKWAY {distance_m, steps, direction}]->(:Waypoint)

Conventions copied from the existing map: slot is "front" for a single-walk node and
"set01_front" for a node merged from several walks ("set01wp26_front" when a loop closure
merged two waypoints of the same walk); photo_file is "set01/<file>"; session is
"set01_wp3" (merged: "set01_wp3+set03_wp3"); bbox / grid_cells / cell_pairs are JSON strings;
label is "<中文> <english>" and label_norm its lower-case.

Coordinate frame: metres, origin at the entrance, heading 0° = +y, 90° = +x (same as the
Android PdrTracker and the iOS MotionPathTracker).
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from server.grid_utils import are_adjacent, compute_cell_pairs, compute_grid_cells, depth_diff

SLOTS = ("front", "right", "back", "left")
SLOT_OFFSET_DEG = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}
OBJECT_SOURCE = "rebuild"

# Sensor Lab (SensorLabScreen.kt) names each directional photo wp_<n>_<slot>_<millis>.jpg and
# stamps the next sensor_log.csv row with waypoint "照片<n>" once the photo set is finished.
PHOTO_RE = re.compile(r"^wp_(\d+)_(front|right|back|left)_\d+\.jpe?g$", re.IGNORECASE)
HEADING_BASELINE_M = 1.0   # walking heading = direction over the last metre before the node


# ── 1. Sensor Lab export → topomap.json (the builder's input) ──────────────────────────

def parse_sensor_log(text: str) -> List[dict]:
    """sensor_log.csv rows (comment lines starting with '#' skipped) with the fields we use."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    rows = []
    for r in csv.DictReader(io.StringIO("\n".join(lines))):
        try:
            rows.append({
                "elapsed_sec": float(r["elapsed_sec"]),
                "total_steps": int(float(r["total_steps"])),
                "pdr_x": float(r["pdr_x"]),
                "pdr_y": float(r["pdr_y"]),
                "waypoint": (r.get("waypoint") or "").strip(),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def heading_between(x0: float, y0: float, x1: float, y1: float) -> float:
    """Heading of the move (x0,y0) → (x1,y1): 0° = +y, clockwise, in [0, 360)."""
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


def walking_heading(rows: Sequence[dict], idx: int, baseline: float = HEADING_BASELINE_M) -> float:
    """Direction the walker was going when they stopped at rows[idx] (where "front" was shot):
    from the last point ≥ baseline metres back; for the first node, toward the next one."""
    x, y = rows[idx]["pdr_x"], rows[idx]["pdr_y"]
    for j in range(idx - 1, -1, -1):
        px, py = rows[j]["pdr_x"], rows[j]["pdr_y"]
        if math.hypot(x - px, y - py) >= baseline:
            return round(heading_between(px, py, x, y), 1)
    for j in range(idx + 1, len(rows)):
        nx_, ny_ = rows[j]["pdr_x"], rows[j]["pdr_y"]
        if math.hypot(nx_ - x, ny_ - y) >= baseline:
            return round(heading_between(x, y, nx_, ny_), 1)
    return 0.0


def extract_waypoints(rows: Sequence[dict], photo_names: Iterable[str]) -> Tuple[List[dict], List[str]]:
    """Photo waypoints of one Sensor Lab walk → ([{wp_num, x, y, heading_deg, photos, headings}],
    warnings). Position is the row stamped "照片<n>"; the four photos face heading + 0/90/180/270."""
    by_wp: Dict[int, Dict[str, str]] = {}
    for name in photo_names:
        m = PHOTO_RE.match(name)
        if m:
            by_wp.setdefault(int(m.group(1)), {})[m.group(2).lower()] = name
    stamp = {r["waypoint"]: i for i, r in enumerate(rows) if r["waypoint"]}
    out, warnings = [], []
    for n in sorted(by_wp):
        idx = stamp.get(f"照片{n}")
        if idx is None:
            warnings.append(f"wp {n}: sensor_log.csv 沒有「照片{n}」的標記列，略過")
            continue
        photos = {s: by_wp[n][s] for s in SLOTS if s in by_wp[n]}
        if len(photos) < 4:
            warnings.append(f"wp {n}: 只有 {len(photos)}/4 張照片（{', '.join(photos)}）")
        h = walking_heading(rows, idx)
        out.append({
            "wp_num": n,
            "row": idx,
            "x": round(rows[idx]["pdr_x"], 2),
            "y": round(rows[idx]["pdr_y"], 2),
            "total_steps": rows[idx]["total_steps"],
            "heading_deg": h,
            "photos": photos,
            "headings": {s: round((h + SLOT_OFFSET_DEG[s]) % 360.0, 1) for s in photos},
        })
    return out, warnings


TRACK_MIN_STEP_M = 0.1    # CSV trajectory kept every this many metres (plus every waypoint row)


def walk_track(rows: Sequence[dict], waypoints: Sequence[dict],
               min_step: float = TRACK_MIN_STEP_M) -> dict:
    """The walk's full PDR trajectory from sensor_log.csv, thinned, for drawing in the builder.

    {"pts": [[x, y]…], "seg": [k…], "t": [f…]}: point i lies on the stretch from waypoint k to
    waypoint k+1 (k indexes `waypoints` in order; -1 = before the first one, the last index =
    after the last one) at fraction t of that stretch's walked length, so the builder can bend
    the line along with the waypoints when they are moved, rotated or loop-closed."""
    stamps = [wp["row"] for wp in waypoints]
    keep: List[int] = []
    last = None
    marks = set(stamps)
    for i, r in enumerate(rows):
        p = (r["pdr_x"], r["pdr_y"])
        if last is None or i in marks or math.hypot(p[0] - last[0], p[1] - last[1]) >= min_step:
            keep.append(i)
            last = p
    if keep and keep[-1] != len(rows) - 1:
        keep.append(len(rows) - 1)

    pts = [[round(rows[i]["pdr_x"], 2), round(rows[i]["pdr_y"], 2)] for i in keep]
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    # where each waypoint sits along the kept points
    at = {row: keep.index(row) for row in stamps}
    seg, frac = [], []
    k = -1
    for j, row in enumerate(keep):
        while k + 1 < len(stamps) and row >= stamps[k + 1]:
            k += 1
        if k < 0 or k >= len(stamps) - 1:
            seg.append(k if k >= 0 else -1)
            frac.append(0.0)
            continue
        a, b = at[stamps[k]], at[stamps[k + 1]]
        length = cum[b] - cum[a]
        seg.append(k)
        frac.append(round((cum[j] - cum[a]) / length, 3) if length > 0 else 0.0)
    return {"pts": pts, "seg": seg, "t": frac}


def walk_stats(rows: Sequence[dict]) -> dict:
    """Steps, walked distance (m) and duration (s) of one sensor_log.csv."""
    dist = sum(math.hypot(b["pdr_x"] - a["pdr_x"], b["pdr_y"] - a["pdr_y"]) for a, b in zip(rows, rows[1:]))
    return {"steps": rows[-1]["total_steps"] if rows else 0, "distance_m": round(dist, 1),
            "duration_s": round(rows[-1]["elapsed_sec"] - rows[0]["elapsed_sec"], 1) if rows else 0.0}


def build_topomap(place: str, walks: Sequence[Tuple[str, Sequence[dict]]],
                  walk_info: Optional[Dict[str, dict]] = None) -> dict:
    """topomap.json for topomap_builder.html: one node per photo waypoint (ids 1..N across all
    walks), edges between consecutive waypoints of the same walk. walks = [(set_id, waypoints)].
    walk_info {set_id: {"source", "stats", "track"}} adds "walks" (where each walk came from)
    and "tracks" (its CSV trajectory, see walk_track)."""
    nodes, edges, per_set = [], [], {}
    next_id = 1
    for set_id, wps in walks:
        prev = None
        for wp in wps:
            nodes.append({
                "id": next_id, "set": set_id, "wp_num": wp["wp_num"],
                "uuid": f"set{set_id}_wp{wp['wp_num']}",
                "x": wp["x"], "y": wp["y"],
                "photos": dict(wp["photos"]), "headings": dict(wp["headings"]),
            })
            if prev is not None:
                edges.append({"from": prev, "to": next_id, "set": set_id})
            prev = next_id
            next_id += 1
        per_set[set_id] = len(wps)
    topo = {"place": place, "nodes": nodes, "edges": edges,
            "stats": {"total_nodes": len(nodes), "sets": per_set}}
    if walk_info:
        topo["walks"] = [{"set": s, "source": walk_info[s].get("source", ""), "nodes": per_set[s],
                          **walk_info[s].get("stats", {})} for s, _ in walks if s in walk_info]
        topo["tracks"] = {s: walk_info[s]["track"] for s, _ in walks if "track" in walk_info.get(s, {})}
    return topo


# ── 2. confirmed builder output → Neo4j rows ───────────────────────────────────────────

def session_name(sources: Sequence[dict]) -> str:
    return "+".join(f"set{s['set']}_wp{s['wp_num']}" for s in sources)


def upload_rows(confirmed: dict, topomap: Optional[dict], place: str) -> dict:
    """Rows for upload_map_v2.py: {"waypoints", "photos", "walkways"} (see module docstring).
    Photo headings come from topomap.json (the builder does not pass them through)."""
    headings: Dict[str, Dict[str, float]] = {}
    for n in (topomap or {}).get("nodes", []):
        headings[n.get("uuid") or f"set{n['set']}_wp{n['wp_num']}"] = n.get("headings") or {}

    waypoints, photos = [], []
    ids = set()
    for node in confirmed.get("nodes", []):
        nid = int(node["id"])
        if nid in ids:
            raise ValueError(f"重複的節點 id {nid}")
        ids.add(nid)
        sources = node.get("sources") or []
        if not sources:
            raise ValueError(f"節點 {nid} 沒有 sources")
        waypoints.append({
            "place": place, "nid": nid,
            "pdr_x": round(float(node["x"]), 2), "pdr_y": round(float(node["y"]), 2),
            "session": session_name(sources),
            "heading_deg": 0.0, "total_steps": 0, "total_distance_m": 0.0,
        })
        merged = len(sources) > 1
        sets_seen = [s["set"] for s in sources]
        for src in sources:
            hd = src.get("headings") or headings.get(src.get("uuid") or session_name([src]), {})
            # A loop closure merges two waypoints of the same walk: tell them apart by number,
            # still as "<prefix>_<slot>" (the server strips everything up to the first "_").
            prefix = f"set{src['set']}" + (f"wp{src['wp_num']}" if sets_seen.count(src["set"]) > 1 else "")
            for slot in SLOTS:
                fname = (src.get("photos") or {}).get(slot)
                if not fname:
                    continue
                photos.append({
                    "nid": nid,
                    "slot": f"{prefix}_{slot}" if merged else slot,
                    "photo_file": f"set{src['set']}/{fname}",
                    "heading_deg": float(hd.get(slot, 0.0)),
                })

    walkways, seen = [], set()
    for e in confirmed.get("edges", []):
        a, b = int(e["from"]), int(e["to"])
        if a == b or (min(a, b), max(a, b)) in seen:
            continue
        if a not in ids or b not in ids:
            raise ValueError(f"邊 {a}→{b} 指向不存在的節點")
        seen.add((min(a, b), max(a, b)))
        walkways.append({"from": a, "to": b,
                         "distance_m": round(float(e.get("distance", 0.0)), 2)})
    return {"waypoints": waypoints, "photos": photos, "walkways": walkways}


def photo_local_path(photo_file: str) -> Tuple[str, str]:
    """"set01/x.jpg" → ("01", "x.jpg"): the workspace folder and file name of a DirPhoto."""
    folder, _, name = photo_file.partition("/")
    if not name:
        raise ValueError(f"photo_file 格式不對：{photo_file!r}")
    return (folder[3:] if folder.startswith("set") else folder), name


# ── 2b. DINOv2 photo matching → loop closures / same spot on two walks ─────────────────

LOOP_MIN_SIM = 0.78       # node similarity to call two waypoints the same spot
MATCH_MARGIN = 0.03       # …and it must beat the runner-up by this much (else "uncertain")
UNCERTAIN_MIN_SIM = 0.72
LOOP_MIN_GAP = 3          # same walk: waypoints closer than this along the walk are neighbours


def photo_key(node: dict, slot: str) -> str:
    """Embedding key of a node's photo: "<set>/<file>" (its path inside the workspace)."""
    return f"{node['set']}/{node['photos'][slot]}"


def node_similarity(a: dict, b: dict, emb: Dict[str, "object"]) -> float:
    """How alike two waypoints look: for each photo of one, its best cosine match among the
    other's photos (any direction — a walk may come back facing the other way), averaged,
    and made symmetric. emb maps photo_key → L2-normalized vector; missing photos are skipped."""
    def one_way(x, y):
        ys = [emb[photo_key(y, s)] for s in y["photos"] if photo_key(y, s) in emb]
        best = [max(float(emb[photo_key(x, s)] @ v) for v in ys)
                for s in x["photos"] if photo_key(x, s) in emb and ys]
        return sum(best) / len(best) if best else 0.0
    return (one_way(a, b) + one_way(b, a)) / 2


def suggest_merges(nodes: Sequence[dict], emb: Dict[str, "object"]) -> dict:
    """Merge suggestions for the builder from photo similarity.

    loops:   same walk, ≥ LOOP_MIN_GAP apart, similar enough and clearly the best match of both
    matches: different walks, mutual best between the two walks' loop-merged groups, clearly so
    between: a waypoint standing on another walk's edge — its two best look-alikes are that
             walk's consecutive waypoints a, b: {"node", "a", "b", "score", "runner_up"}
    uncertain: looked alike but not clearly (score below the bar or a close runner-up)
    Each loop / match / uncertain item: {"a": id, "b": id, "score", "runner_up"} with a the
    earlier / first-walk node.
    """
    by_id = {n["id"]: n for n in nodes}
    walks: Dict[str, List[dict]] = {}
    for n in nodes:
        walks.setdefault(n["set"], []).append(n)
    for w in walks.values():
        w.sort(key=lambda n: n["wp_num"])
    pos = {n["id"]: i for w in walks.values() for i, n in enumerate(w)}
    sim_cache: Dict[Tuple[int, int], float] = {}

    def sim(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        if key not in sim_cache:
            sim_cache[key] = node_similarity(by_id[a], by_id[b], emb)
        return sim_cache[key]

    def item(a, b, score, runner):
        return {"a": a, "b": b, "score": round(score, 3), "runner_up": round(runner, 3)}

    loops, uncertain = [], []
    # 1. loop closures inside each walk (greedy by score, each waypoint at most once)
    for w in walks.values():
        ids = [n["id"] for n in w]
        far = lambda x, y: abs(pos[x] - pos[y]) >= LOOP_MIN_GAP
        cands = sorted(((sim(a, b), a, b) for i, a in enumerate(ids) for b in ids[i + LOOP_MIN_GAP:]),
                       reverse=True)
        used = set()
        for s, a, b in cands:
            if s < LOOP_MIN_SIM:
                break
            if a in used or b in used:
                continue
            runner = max([sim(a, o) for o in ids if o != b and far(a, o)] +
                         [sim(b, o) for o in ids if o != a and far(b, o)], default=0.0)
            if s - runner >= MATCH_MARGIN:
                loops.append(item(a, b, s, runner))
                used |= {a, b}
            else:
                uncertain.append(item(a, b, s, runner))

    # 2. groups = waypoints joined by a loop closure; their best member pair counts
    group = {n["id"]: n["id"] for n in nodes}
    for lp in loops:
        group[lp["b"]] = lp["a"]
    members: Dict[int, List[int]] = {}
    for nid, g in group.items():
        members.setdefault(g, []).append(nid)

    def gsim(g1: int, g2: int) -> float:
        return max(sim(x, y) for x in members[g1] for y in members[g2])

    set_of = {g: by_id[g]["set"] for g in members}

    def neighbours(g1: int, g2: int) -> bool:
        """Two groups of the same walk joined by a walk edge (consecutive waypoints)."""
        return set_of[g1] == set_of[g2] and any(
            abs(pos[x] - pos[y]) == 1 for x in members[g1] for y in members[g2])

    order = sorted(walks)
    matches, between = [], []
    for i, wa in enumerate(order):
        for wb in order[i + 1:]:
            ga = [g for g in members if set_of[g] == wa]
            gb = [g for g in members if set_of[g] == wb]
            if not ga or not gb:
                continue
            table = {(x, y): gsim(x, y) for x in ga for y in gb}

            # A waypoint whose two best look-alikes are consecutive waypoints of the other walk,
            # about equally alike and clearly ahead of the rest, stands on that edge between them.
            on_edge = set()
            for node, others, look in ((y, ga, lambda z, y=y: table[(z, y)]) for y in gb):
                ranked = sorted(((look(z), z) for z in others), reverse=True)
                if len(ranked) < 2:
                    continue
                (s1, z1), (s2, z2) = ranked[:2]
                s3 = ranked[2][0] if len(ranked) > 2 else 0.0
                if s2 >= LOOP_MIN_SIM and s1 - s2 < MATCH_MARGIN and s2 - s3 >= MATCH_MARGIN \
                        and neighbours(z1, z2):
                    a, b = sorted((z1, z2), key=lambda g: pos[g])
                    between.append({"node": node, "a": a, "b": b,
                                    "score": round(s1, 3), "runner_up": round(s2, 3)})
                    on_edge.add(node)
            for node, others, look in ((x, gb, lambda z, x=x: table[(x, z)]) for x in ga):
                ranked = sorted(((look(z), z) for z in others if z not in on_edge), reverse=True)
                if len(ranked) < 2:
                    continue
                (s1, z1), (s2, z2) = ranked[:2]
                s3 = ranked[2][0] if len(ranked) > 2 else 0.0
                if s2 >= LOOP_MIN_SIM and s1 - s2 < MATCH_MARGIN and s2 - s3 >= MATCH_MARGIN \
                        and neighbours(z1, z2):
                    a, b = sorted((z1, z2), key=lambda g: pos[g])
                    between.append({"node": node, "a": a, "b": b,
                                    "score": round(s1, 3), "runner_up": round(s2, 3)})
                    on_edge.add(node)
            ga = [g for g in ga if g not in on_edge]
            gb = [g for g in gb if g not in on_edge]
            if not ga or not gb:
                continue

            for x in ga:
                ranked_x = sorted(((table[(x, y)], y) for y in gb), reverse=True)
                s, y = ranked_x[0]
                ranked_y = sorted(((table[(z, y)], z) for z in ga), reverse=True)
                if ranked_y[0][1] != x or s < UNCERTAIN_MIN_SIM:
                    continue
                runner = max([r for r, _ in ranked_x[1:2]] + [r for r, _ in ranked_y[1:2]], default=0.0)
                if s >= LOOP_MIN_SIM and s - runner >= MATCH_MARGIN:
                    matches.append(item(x, y, s, runner))
                else:
                    uncertain.append(item(x, y, s, runner))
    key = lambda it: -it["score"]
    return {"loops": sorted(loops, key=key), "matches": sorted(matches, key=key),
            "between": sorted(between, key=key), "uncertain": sorted(uncertain, key=key)}


# ── 3. VLM perception → Object / ADJACENT rows ─────────────────────────────────────────

# Same instructions as the classmate's A7 rebuild (B1222017/APPNAV_ios: scripts/vlm_test/enrich_refdb.py).
PERCEIVE_PROMPT = """\
你正在分析一張超市內部的照片，用來定位使用者的位置。

仔細觀察照片，回報你**確實看到**的**具體**物品。回覆格式為一個 JSON 物件，不要加其他文字：

{"scene": "<用繁體中文描述這是什麼區域>", "items": [{"zh": "<繁體中文名稱>", "en": "<English name>", "conf": <0.0-1.0 信心值>, "box": [x1,y1,x2,y2]}]}

嚴格規則：
- box 座標是圖片的比例值（0.0 到 1.0），格式 [左, 上, 右, 下]。框要緊密包住物件。
- 禁止大框：寬度和高度都不應超過圖片的50%。
- 只回報你**確實看到**的具體物件，不要猜測。
- 回報具體商品、品牌、設備、標示牌文字、地標。
- 特別注意走道上方的吊牌、區域標示牌（如「走道5」「堅果海苔」等分類標示），這些是重要的定位特徵。
- 標示牌上的文字內容要完整辨識，寫在 zh 欄位。
- 不要回報背景（地板、天花板、牆壁、燈光）。
- 寧可少報也不要報錯。5-12個精確的偵測最好。
"""
MAX_BOX_SIDE = 0.55          # the classmate's filter: drop boxes wider/taller than this
_DEDUP_NEAR = {
    "top-left": {"top-center", "mid-left"},
    "top-center": {"top-left", "top-right", "mid-center"},
    "top-right": {"top-center", "mid-right"},
    "mid-left": {"top-left", "mid-center", "bottom-left"},
    "mid-center": {"top-center", "mid-left", "mid-right", "bottom-center"},
    "mid-right": {"top-center", "mid-center", "bottom-right"},
    "bottom-left": {"mid-left", "bottom-center"},
    "bottom-center": {"bottom-left", "bottom-right", "mid-center"},
    "bottom-right": {"mid-right", "bottom-center"},
}


def parse_vlm_items(text: str) -> List[dict]:
    """VLM reply → [{zh, en, conf, box}] with a valid, not-too-large normalized box."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0)).get("items") or []
    except (json.JSONDecodeError, AttributeError):
        return []
    out = []
    for it in items:
        box = it.get("box") if isinstance(it, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 >= MAX_BOX_SIDE or y2 - y1 >= MAX_BOX_SIDE or x2 <= x1 or y2 <= y1:
            continue
        zh, en = str(it.get("zh") or "").strip(), str(it.get("en") or "").strip()
        if len(f"{zh} {en}".strip()) < 2:
            continue
        try:
            conf = float(it.get("conf", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        out.append({"zh": zh, "en": en, "conf": conf, "box": [x1, y1, x2, y2]})
    return out


def _center_cell(box: Sequence[float]) -> str:
    return compute_grid_cells(list(box))[0]


def is_duplicate(label_norm: str, cell: str, existing: Sequence[Tuple[str, str]]) -> bool:
    """Same / near-same label already reported in this or a neighbouring grid cell
    (the classmate's enrich_refdb.is_duplicate)."""
    near = {cell} | _DEDUP_NEAR.get(cell, set())
    strip = {" ", "/", "（", "）"}
    for other, other_cell in existing:
        if other_cell not in near:
            continue
        if label_norm == other:
            return True
        a, b = set(label_norm) - strip, set(other) - strip
        if a and b and len(a & b) / max(len(a), len(b)) > 0.7:
            return True
        if len(label_norm) > 3 and len(other) > 3 and (label_norm in other or other in label_norm):
            return True
    return False


def merge_vlm_calls(calls: Sequence[Sequence[dict]]) -> List[dict]:
    """Items from several VLM calls on one photo, near-duplicates dropped (first one wins)."""
    kept, seen = [], []
    for items in calls:
        for it in items:
            norm = f"{it['zh']} {it['en']}".strip().lower()
            cell = _center_cell(it["box"])
            if is_duplicate(norm, cell, seen):
                continue
            kept.append(it)
            seen.append((norm, cell))
    return kept


def attach_ocr(items: Sequence[dict], ocr: Sequence[Tuple[str, Sequence[float]]]) -> List[str]:
    """ocr_text per item: the OCR lines (text, normalized [x1,y1,x2,y2]) whose centre falls inside
    the item's box, joined with " / " — e.g. an aisle sign gets its "2", a section sign "冷藏食品"."""
    out = []
    for it in items:
        x1, y1, x2, y2 = it["box"]
        texts: List[str] = []
        for text, (ox1, oy1, ox2, oy2) in ocr:
            cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2 and text not in texts:
                texts.append(text)
        out.append(" / ".join(texts))
    return out


def object_props(item: dict, ocr_text: str = "") -> dict:
    box = [round(v, 2) for v in item["box"]]
    cell, cells, pieces, depth = compute_grid_cells(box)
    label = f"{item['zh']} {item['en']}".strip()
    return {
        "label": label, "label_norm": label.lower(),
        "score": round(float(item.get("conf", 0.5)), 2),
        "ocr_text": ocr_text,
        "bbox": json.dumps(box),
        "grid_cell": cell, "grid_cells": json.dumps(cells),
        "piece_count": pieces, "depth_class": depth,
        "source": OBJECT_SOURCE,
    }


_HORIZONTAL = {"left", "right"}
_VERTICAL = {"above", "below"}
_DIAGONAL = {"above-left", "above-right", "below-left", "below-right"}


def adjacent_pairs(objects: Sequence[dict]) -> List[dict]:
    """ADJACENT rows (i < j, i → j) between objects of one photo whose grid cells touch."""
    cells = [json.loads(o["grid_cells"]) for o in objects]
    out = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if not are_adjacent(cells[i], cells[j]):
                continue
            pairs = compute_cell_pairs(cells[i], cells[j])
            dirs = [p["dir"] for p in pairs]
            out.append({"i": i, "j": j, "props": {
                "cell_pairs": json.dumps(pairs),
                "horizontal_count": sum(d in _HORIZONTAL for d in dirs),
                "vertical_count": sum(d in _VERTICAL for d in dirs),
                "diagonal_count": sum(d in _DIAGONAL for d in dirs),
                "overlap": "overlap" in dirs,
                "depth_diff": depth_diff(objects[i]["depth_class"], objects[j]["depth_class"]),
            }})
    return out
