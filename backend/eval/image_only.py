"""Single-photo localization with NO phone sensors, through the production server path.

  python -m eval.image_only --set 0916|field1017|route1 [--photos DIR] [--data-root DIR]
                            [--out DIR] [--cache DIR] [--map neo4j|snapshot] [--limit N]
                            [--redetect] [--reembed]

Each photo goes through what POST /photo does for localization, minus everything the
phone adds (no map_x/y, dist_since_fix, fix_node, dino_topk, previous fix, goal):

  resize (exif, ≤1600 px, JPEG 85) → photo_quality.assess_image → EasyOCR →
  GroundingDINO (GENERIC_INDOOR_OBJECTS + OCCLUDER_PROMPTS) → assess_occlusion →
  server._run_early_localization → locate_fusion.server_image_evidence (DINOv2 5 :
  objects 0.5 : OCR 0.5) → visual_localization.localize(v2=True, server_evidence=…,
  server_best_photo=…) [+ aisle-sign evidence, as on the server]

The VLM re-ranker is forced off. A photo the server would reject (blurred / dark, or
blocked by people / carts) is still localized so it can be scored, and the "production
gate" block of the report counts it as no answer.

Variants (same detections, same DINOv2 embedding):
  prod       — the server as configured
  dino_only  — locate_fusion W_N = W_O = 0 and no aisle signs: DINOv2 alone
  no_objects — W_N = 0 (OCR and aisle signs kept)
  no_text    — W_O = 0 and no aisle signs (objects kept)

Caches (under --cache, default backend/output/eval/image_only/<set>/): the resized
photos, perception.json (quality, OCR, GroundingDINO, occlusion, stage times) and
dino.json (query embeddings). Changing only DINOv2 → --reembed; only fusion → nothing.
--redetect re-runs OCR + GroundingDINO. The reference map is snapshotted to
refmap.pkl next to them; --map snapshot reuses it without Neo4j.

Data sets (labels in eval/data/, photos under --data-root, default this backend/):
  0916       268 photos of legacy/testPDR/0916家樂福1,2 (not in the map). Truth node is
             0916_ground_truth_current.json translated to Neo4j node ids through the
             reference photo files of legacy/carrefour_a7_imported.json (1:1, same
             coordinates); truth_x/y is the test waypoint in map coordinates.
  field1017  24 photos of the 2026-10-17 A7 test, part A (sessions c0d808c8, e602d576).
             Truth = the planned point whose arrival time (handbook record) precedes
             the photo; phone_heading_deg (compass) is the facing label.
  route1     eval/data/route1_labels.csv (between-node photos; estimated labels);
             needs --photos <測試路線1 folder>.

Output: <out>/image_only_<set>.json and .md (default eval/results/).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import pickle
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
SETS = ("0916", "field1017", "route1")
VARIANTS = ("prod", "dino_only")
ABLATIONS = ("no_objects", "no_text")    # one of the two extras switched off
CONF_LEVELS = (0.3, 0.6)
STAGES = ("quality", "ocr", "gdino", "dino", "fusion")
AT_NODE_M = 1.0          # 0916: test waypoint within this of its node counts as "at node"
HEADING_OK_DEG = 45.0


# ── labels ─────────────────────────────────────────────────────────────────

def _read_csv(name):
    with open(DATA / name, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def load_items(name, ref_map, data_root, photos_dir=None):
    """[{id, path, ok, truth, truth_xy, condition, heading, stored, meta}]"""
    xy = {n: (p.pdr_x, p.pdr_y) for n, p in ref_map.photos.items()}
    items = []
    if name == "0916":
        for r in _read_csv("0916_labels.csv"):
            node = int(r["node"])
            d = float(r["dist_to_node_m"])
            items.append(dict(id=Path(r["photo"]).parent.parent.name + "/" + Path(r["photo"]).name,
                              path=data_root / r["photo"], ok=[node], truth=node,
                              truth_xy=[float(r["truth_x"]), float(r["truth_y"])],
                              condition="at_node" if d <= AT_NODE_M else "between", heading=None,
                              stored=False, meta={"dist_to_node_m": d, "verified": r["verified"] == "1",
                                                  "direction": r["direction"], "gt_key": r["gt_key"]}))
    elif name == "field1017":
        for r in _read_csv("field1017_labels.csv"):
            node = int(r["node"])
            h = r.get("phone_heading_deg")
            items.append(dict(id=f"r{r['round']}/{r['session']}/{Path(r['photo']).name}",
                              path=data_root / r["photo"], ok=[node], truth=node, truth_xy=list(xy[node]),
                              condition="at_node", heading=float(h) if h else None,
                              stored=True,   # already the server's own resized copy
                              meta={"round": int(r["round"]), "logged_fix": r.get("logged_fix")}))
    elif name == "route1":
        if not photos_dir:
            sys.exit("route1 needs --photos <測試路線1 photo folder> (not found on this machine)")
        for r in _read_csv("route1_labels.csv"):
            if not r.get("node_a"):
                continue
            a = int(r["node_a"])
            b = int(r["node_b"]) if r.get("node_b") else None
            f = float(r.get("fraction") or 0) if b else 0.0
            t = xy[a] if b is None else (xy[a][0] + f * (xy[b][0] - xy[a][0]), xy[a][1] + f * (xy[b][1] - xy[a][1]))
            items.append(dict(id=r["photo"], path=Path(photos_dir) / r["photo"], ok=[a] + ([b] if b else []),
                              truth=b if (b and f > 0.5) else a, truth_xy=list(t),
                              condition=r.get("condition") or ("between" if b else "at_node"),
                              heading=None, stored=False, meta={"fraction": f}))
    else:
        raise ValueError(name)
    missing = [it["path"] for it in items if not Path(it["path"]).is_file()]
    if missing:
        sys.exit(f"{len(missing)} photos missing, e.g. {missing[0]} — pass --data-root / --photos")
    return items


# ── reference map ──────────────────────────────────────────────────────────

class _MapSource:
    """Stands in for the Neo4j client: the same RefMap object for every call."""

    def __init__(self, ref_map):
        self.ref_map = ref_map

    def load_reference_map(self, place=None):
        return self.ref_map


def load_map(mode, cache_root):
    from server.config import NEO4J_PLACE
    snap = cache_root / "refmap.pkl"
    if mode == "snapshot":
        # written by this script from our own Neo4j map — a trusted local file
        with open(snap, "rb") as f:
            return pickle.load(f), "snapshot"
    from server.neo4j_client import get_neo4j
    neo4j = get_neo4j()
    if neo4j is None:
        sys.exit("Neo4j not configured — fill NEO4J_* in the repo-root .env, or use --map snapshot")
    ref_map = neo4j.load_reference_map(NEO4J_PLACE)
    snap.parent.mkdir(parents=True, exist_ok=True)
    with open(snap, "wb") as f:
        pickle.dump(ref_map, f)
    return ref_map, "neo4j"


def map_fingerprint(ref_map):
    h = hashlib.sha1()
    for nid in sorted(ref_map.photos):
        n = ref_map.photos[nid]
        h.update(f"{nid}:{n.pdr_x:.2f},{n.pdr_y:.2f}:{sorted(n.neighbor_nids)}".encode())
        for dp in n.directional_photos:
            h.update(f"{dp.slot}:{dp.photo_file}:{dp.heading_deg}:{len(dp.objects)}:{dp.dino is not None}".encode())
    return h.hexdigest()[:12]


def _file_sha(path):
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def hop_table(ref_map):
    """{a: {b: walkway hops}} — edges walked both ways, as the server does."""
    adj = {n: set() for n in ref_map.photos}
    for a, node in ref_map.photos.items():
        for b in node.neighbor_nids:
            if b in adj:
                adj[a].add(b)
                adj[b].add(a)
    hops = {}
    for s in adj:
        seen = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    q.append(v)
        hops[s] = seen
    return hops


# ── caches ─────────────────────────────────────────────────────────────────

def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def preprocess(item, img_dir):
    """The photo exactly as POST /photo stores it before any model sees it."""
    if item["stored"]:
        return Path(item["path"])
    from PIL import Image, ImageOps
    out = img_dir / (item["id"].replace("/", "__").rsplit(".", 1)[0] + ".jpg")
    if not out.is_file():
        im = ImageOps.exif_transpose(Image.open(item["path"])).convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600))
        im.save(out, format="JPEG", quality=85)
    return out


def perception_signatures():
    from server import config as C
    prompts = list(dict.fromkeys(C.GENERIC_INDOOR_OBJECTS + C.OCCLUDER_PROMPTS))
    return {"ocr": {"min_conf": C.OCR_MIN_CONFIDENCE, "max": C.OCR_MAX_RESULTS, "langs": list(C.OCR_LANGUAGES)},
            "gdino": {"backbone": C.GDINO_BACKBONE, "prompts": prompts}}


def perceive_all(jobs, cache_path, redetect=False):
    """Quality + EasyOCR, then GroundingDINO + occlusion, over every photo — what
    _handle_photo runs before localizing, with no goal objects in the prompt.

    One model at a time, freed before the next: with both resident an 8 GB laptop pages
    and each step runs ~2× slower, which would distort the stage times. Results land in
    cache_path per photo and stage, so an interrupted run resumes where it stopped."""
    import gc
    from PIL import Image
    from server import config as C
    from server import photo_quality
    from server.ocr import OCR
    from server.perception import Perception
    sig = perception_signatures()
    cache = {} if redetect else _load_json(cache_path)

    todo = [(k, img) for k, img in jobs if (cache.get(k) or {}).get("ocr_sig") != sig["ocr"]]
    if todo:
        ocr = OCR(languages=C.OCR_LANGUAGES)
        ocr.load()
        for i, (k, img) in enumerate(todo, 1):
            with Image.open(img) as im:
                w, h = im.size
            t0 = time.perf_counter()
            q = photo_quality.assess_image(str(img))
            t1 = time.perf_counter()
            res = ocr.read(str(img), min_confidence=C.OCR_MIN_CONFIDENCE, max_results=C.OCR_MAX_RESULTS)
            t2 = time.perf_counter()
            p = cache.setdefault(k, {"t": {}})
            p.update(size=[w, h], ocr_sig=sig["ocr"],
                     quality={"sharpness": q.sharpness, "brightness": q.brightness, "hint": q.hint},
                     ocr=[{"text": r.text, "confidence": r.confidence, "bbox": r.bbox} for r in res])
            p["t"].update(quality=t1 - t0, ocr=t2 - t1)
            _save_json(cache_path, cache)
            print(f"  OCR {i}/{len(todo)} {k}: {len(res)} texts, {t2 - t1:.1f}s", file=sys.stderr, flush=True)
        del ocr
        gc.collect()

    todo = [(k, img) for k, img in jobs if cache[k].get("gdino_sig") != sig["gdino"]]
    if todo:
        gd = Perception()
        gd.load()
        for i, (k, img) in enumerate(todo, 1):
            p = cache[k]
            t0 = time.perf_counter()
            dets = gd.detect(str(img), sig["gdino"]["prompts"])
            t1 = time.perf_counter()
            occ = photo_quality.assess_occlusion(dets, *p["size"], goals=[])
            p.update(gdino_sig=sig["gdino"],
                     detections=[{"label": d.label, "box": [float(v) for v in d.box], "score": float(d.score)}
                                 for d in dets],
                     occlusion={"fraction": occ.fraction, "useful_objects": occ.useful_objects, "hint": occ.hint})
            p["t"]["gdino"] = t1 - t0
            _save_json(cache_path, cache)
            print(f"  GroundingDINO {i}/{len(todo)} {k}: {len(dets)} boxes, {t1 - t0:.1f}s",
                  file=sys.stderr, flush=True)
        del gd
        gc.collect()
    return cache


class EmbedCache:
    """Wraps server.dino_embed.embed_image: the query photo's vector comes from dino.json
    when present, so a fusion-only change doesn't re-run DINOv2."""

    def __init__(self, path, real, model_name, reembed=False):
        self.path, self.real, self.model = path, real, model_name
        data = {} if reembed else _load_json(path)
        self.data = data if data.get("_model") == model_name else {"_model": model_name}
        self.key = None
        self.spent = 0.0

    def __call__(self, image_path):
        hit = self.data.get(self.key)
        if hit is not None:
            return np.asarray(hit["emb"], dtype=np.float32)
        t0 = time.perf_counter()
        v = self.real(image_path)
        dt = time.perf_counter() - t0
        self.spent += dt
        if v is not None and self.key:
            self.data[self.key] = {"emb": [float(x) for x in v], "t": dt}
            _save_json(self.path, self.data)
        return v


@contextlib.contextmanager
def variant(name):
    from server import config as C
    from server import locate_fusion as LF
    saved = (LF.W_N, LF.W_O, C.AISLE_SIGN_PLACES)
    try:
        if name in ("dino_only", "no_objects"):
            LF.W_N = 0.0
        if name in ("dino_only", "no_text"):
            LF.W_O = 0.0
            C.AISLE_SIGN_PLACES = set()
        if name not in VARIANTS + ABLATIONS:
            raise ValueError(name)
        yield
    finally:
        LF.W_N, LF.W_O, C.AISLE_SIGN_PLACES = saved


def text_coverage(perc_all, ref_map, place):
    """How many photos the text signals can act on at all: an OCR string equal to one in
    the map (locate_fusion matches exact normalized strings) / aisle-sign evidence."""
    from server import aisle_signs
    from server import locate_fusion as LF
    from server.ocr import OCRResult
    ref = set().union(*[e["ocrset"] for e in LF.build_fusion_index(ref_map, {})["entries"]])
    tok = sign = 0
    for p in perc_all.values():
        tok += bool({LF._norm_ocr(o["text"]) for o in p["ocr"]} & ref)
        nums = aisle_signs.numbers_from_ocr([OCRResult(o["text"], o["confidence"], o["bbox"]) for o in p["ocr"]],
                                            *p["size"])
        sign += bool(nums and aisle_signs.sign_evidence(nums, ref_map, place))
    return {"photos": len(perc_all), "with_ocr_text_in_map": tok, "with_aisle_sign_evidence": sign}


def localize(srv, place, perc, img_path):
    """server._run_early_localization with no phone inputs and a fresh session."""
    from server import aisle_signs
    from server.ocr import OCRResult
    from server.perception import Detection
    dets = [Detection(label=d["label"], box=d["box"], score=d["score"]) for d in perc["detections"]]
    ocr = [OCRResult(text=o["text"], confidence=o["confidence"], bbox=o["bbox"]) for o in perc["ocr"]]
    w, h = perc["size"]
    s = SimpleNamespace(place=place, last_corrected_nid=None, last_loc_log=None,
                        user_heading=None, heading_slot=None, heading_confidence=0.0)
    return srv._run_early_localization(
        s, "image_only", [d.label for d in dets], [o.text for o in ocr],
        detected_items=srv._build_localization_items(dets, ocr, w, h),
        image_path=str(img_path),
        ocr_items=srv._build_ocr_items(ocr, w, h),
        sign_numbers=aisle_signs.numbers_from_ocr(ocr, w, h),
        dino_sims=None, sensor_pos=None, dist_since_fix=None, fix_node=None, extra_evidence=None)


# ── scoring ────────────────────────────────────────────────────────────────

def _angle(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def score_record(item, loc, ref_map, hops):
    xy = {n: (p.pdr_x, p.pdr_y) for n, p in ref_map.photos.items()}
    top = [(int(c[0]), float(c[1])) for c in (loc.top_candidates or [])] if loc else []
    pred = top[0][0] if top else None
    ok = item["ok"]
    rec = {
        "id": item["id"], "truth": item["truth"], "ok": ok, "truth_xy": item["truth_xy"],
        "condition": item["condition"], "meta": item["meta"],
        "pred": pred, "top5": top[:5], "confidence": float(loc.confidence) if loc else 0.0,
        "answered": bool(loc and loc.matched_nid is not None),
        "hit1": pred in ok, "hit5": any(n in ok for n, _ in top[:5]),
        "hop": min((hops[o].get(pred, 99) for o in ok), default=99) if pred is not None else None,
        "err_m": math.dist(xy[pred], item["truth_xy"]) if pred is not None else None,
        "est_err_m": (math.dist(loc.estimated_xy, item["truth_xy"])
                      if loc and loc.estimated_xy else None),
        "slot": loc.matched_slot if loc else None,
        "heading_pred": loc.matched_heading if loc else None,
        "heading_true": item["heading"],
    }
    rec["within1"] = rec["hop"] is not None and rec["hop"] <= 1
    if rec["heading_true"] is not None and rec["heading_pred"] is not None:
        rec["heading_err"] = _angle(rec["heading_pred"], rec["heading_true"])
    return rec


def _pct(xs):
    return round(100.0 * sum(xs) / len(xs), 1) if xs else None


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "median": None}
    return {"mean": round(statistics.fmean(xs), 2), "median": round(statistics.median(xs), 2)}


def summarize(recs):
    n = len(recs)
    out = {
        "n": n,
        "top1": _pct([r["hit1"] for r in recs]),
        "top5": _pct([r["hit5"] for r in recs]),
        "within_1_hop": _pct([r["within1"] for r in recs]),
        "error_m": _stats([r["err_m"] for r in recs]),
        "estimated_xy_error_m": _stats([r["est_err_m"] for r in recs]),
    }
    cal = {}
    for t in CONF_LEVELS:
        sel = [r for r in recs if r["confidence"] >= t]
        cal[f">={t}"] = {"coverage": _pct([r["confidence"] >= t for r in recs]), "n": len(sel),
                         "top1_when_confident": _pct([r["hit1"] for r in sel]),
                         "within_1_hop_when_confident": _pct([r["within1"] for r in sel])}
    out["confidence"] = cal
    # what the server would actually do: quality / occlusion rejection, then the
    # localize() probability threshold
    gate = [r for r in recs if not r.get("rejected") and r["answered"]]
    out["production_gate"] = {
        "rejected_quality": sum(r.get("rejected") == "quality" for r in recs),
        "rejected_occlusion": sum(r.get("rejected") == "occlusion" for r in recs),
        "below_threshold": sum(not r.get("rejected") and not r["answered"] for r in recs),
        "answered": len(gate), "coverage": _pct([r in gate for r in recs]),
        "top1_of_answered": _pct([r["hit1"] for r in gate]),
    }
    hd = [r for r in recs if r["heading_true"] is not None and r["heading_pred"] is not None]
    hd_ok = [r for r in hd if r["hit1"]]
    out["heading"] = {
        "labelled": sum(r["heading_true"] is not None for r in recs),
        "with_prediction": len(hd),
        f"within_{HEADING_OK_DEG:g}deg": _pct([r["heading_err"] <= HEADING_OK_DEG for r in hd]),
        "n_correct_node": len(hd_ok),
        f"within_{HEADING_OK_DEG:g}deg_on_correct_node": _pct([r["heading_err"] <= HEADING_OK_DEG for r in hd_ok]),
    }
    conds = sorted({r["condition"] for r in recs})
    out["by_condition"] = {c: {"n": sum(r["condition"] == c for r in recs),
                               "top1": _pct([r["hit1"] for r in recs if r["condition"] == c]),
                               "within_1_hop": _pct([r["within1"] for r in recs if r["condition"] == c])}
                           for c in conds}
    return out


def failures(recs):
    bad = [r for r in recs if not r["hit1"]]

    def kind(r):
        if r["hop"] == 1:
            return "neighbour (1 hop)"
        if r["hop"] == 2:
            return "2 hops"
        return "far (≥3 hops)"
    by_kind = Counter(kind(r) for r in bad)
    pairs = Counter(tuple(sorted((r["truth"], r["pred"]))) for r in bad if r["pred"] is not None)
    directed = Counter((r["truth"], r["pred"]) for r in bad if r["pred"] is not None)
    return {
        "n_wrong": len(bad),
        "kind": dict(by_kind),
        "truth_in_top5": sum(r["hit5"] for r in bad),
        "between_nodes": sum(r["condition"] != "at_node" for r in bad),
        "occluded": sum(bool(r.get("occlusion_hint")) or (r.get("occlusion_fraction") or 0) >= 0.3 for r in bad),
        "low_quality": sum(bool(r.get("quality_hint")) for r in bad),
        "confused_pairs": [{"nodes": list(p), "count": c,
                            "directions": {f"{a}->{b}": directed[(a, b)] for a, b in (p, p[::-1])
                                           if directed[(a, b)]}}
                           for p, c in pairs.most_common(10)],
        "far_errors": [{"id": r["id"], "truth": r["truth"], "pred": r["pred"], "hop": r["hop"],
                        "err_m": round(r["err_m"], 1), "conf": round(r["confidence"], 2),
                        "condition": r["condition"]}
                       for r in sorted(bad, key=lambda r: -(r["err_m"] or 0)) if (r["hop"] or 99) >= 3][:15],
    }


# ── report ─────────────────────────────────────────────────────────────────

def _fmt(v, unit=""):
    return "–" if v is None else f"{v}{unit}"


def markdown(rep):
    L = [f"# 單張影像定位基準：{rep['set']}", "",
         f"- {rep['n_photos']} 張照片，不帶任何感測器（無 map_x/y、dist_since_fix、fix_node、手機 DINOv2、前一次定位）",
         f"- 地圖 `{rep['map']['place']}`：{rep['map']['nodes']} 節點，來源 {rep['map']['source']}，指紋 {rep['map']['fingerprint']}",
         f"- 產生時間 {rep['generated']}；commit {rep['commit'] or '?'}", ""]
    L += ["## 主要指標", "",
          "| 版本 | Top-1 | Top-5 | 1 hop 內 | 誤差 平均/中位 (m) | 內插位置誤差 平均/中位 (m) |",
          "|---|---|---|---|---|---|"]
    for v, s in rep["variants"].items():
        m = s["metrics"]
        L.append(f"| {v} | {_fmt(m['top1'], '%')} | {_fmt(m['top5'], '%')} | {_fmt(m['within_1_hop'], '%')} | "
                 f"{_fmt(m['error_m']['mean'])} / {_fmt(m['error_m']['median'])} | "
                 f"{_fmt(m['estimated_xy_error_m']['mean'])} / {_fmt(m['estimated_xy_error_m']['median'])} |")
    L += ["", "## 信心校準（信心 = localize 回傳的最佳節點機率）", "",
          "| 版本 | 門檻 | 涵蓋率 | 命中率 (Top-1) | 1 hop 內 |", "|---|---|---|---|---|"]
    for v, s in rep["variants"].items():
        for t, c in s["metrics"]["confidence"].items():
            L.append(f"| {v} | {t} | {_fmt(c['coverage'], '%')} ({c['n']}) | {_fmt(c['top1_when_confident'], '%')} | "
                     f"{_fmt(c['within_1_hop_when_confident'], '%')} |")
    L += ["", "正式伺服器的閘門（品質不佳 / 被遮擋 → 不定位；機率 < CONFIDENT_THRESHOLD → 不給節點）：", ""]
    for v, s in rep["variants"].items():
        g = s["metrics"]["production_gate"]
        L.append(f"- {v}：品質退件 {g['rejected_quality']}、遮擋退件 {g['rejected_occlusion']}、低於門檻 "
                 f"{g['below_threshold']}；有回答 {g['answered']} 張（{_fmt(g['coverage'], '%')}），其中正確 "
                 f"{_fmt(g['top1_of_answered'], '%')}")
    L += ["", "## 朝向", ""]
    for v, s in rep["variants"].items():
        h = s["metrics"]["heading"]
        if not h["labelled"]:
            L.append(f"- {v}：此資料集沒有朝向標註")
            continue
        L.append(f"- {v}：{h['labelled']} 張有朝向標註；預測朝向與標註差 ≤{HEADING_OK_DEG:g}° 的比例 "
                 f"{_fmt(h[f'within_{HEADING_OK_DEG:g}deg'], '%')}（{h['with_prediction']} 張），"
                 f"只看節點正確的 {_fmt(h[f'within_{HEADING_OK_DEG:g}deg_on_correct_node'], '%')}（{h['n_correct_node']} 張）")
    L += ["", "## 分情境（prod）", "", "| 情境 | 張數 | Top-1 | 1 hop 內 |", "|---|---|---|---|"]
    for c, b in rep["variants"]["prod"]["metrics"]["by_condition"].items():
        L.append(f"| {c} | {b['n']} | {_fmt(b['top1'], '%')} | {_fmt(b['within_1_hop'], '%')} |")
    t = rep["timing_s"]
    L += ["", "## 各階段耗時（秒／張，本機首次計算）", "",
          "| 階段 | 平均 | 中位數 |", "|---|---|---|"]
    for k in STAGES + ("total",):
        L.append(f"| {k} | {_fmt(t[k]['mean'])} | {_fmt(t[k]['median'])} |")
    L.append(f"\n機器：{rep['machine']}")
    tc = rep["text_coverage"]
    L += ["", f"文字證據能作用的照片：OCR 字串和地圖完全相同 {tc['with_ocr_text_in_map']}/{tc['photos']}、"
              f"有走道號碼證據 {tc['with_aisle_sign_evidence']}/{tc['photos']}"]
    for v, s in rep["variants"].items():
        f = s["failures"]
        L += ["", f"## 失敗案例（{v}）", "",
              f"- 錯 {f['n_wrong']} 張：" + "、".join(f"{k} {c}" for k, c in sorted(f["kind"].items())),
              f"- 其中正解仍在 Top-5：{f['truth_in_top5']}；在節點之間：{f['between_nodes']}；"
              f"被遮擋（≥30% 或伺服器會退件）：{f['occluded']}；品質提示：{f['low_quality']}",
              "- 最常混淆的節點對：" + ("、".join(
                  f"{p['nodes'][0]}↔{p['nodes'][1]} ×{p['count']}" for p in f["confused_pairs"][:8]) or "無")]
    if rep.get("notes"):
        L += ["", "## 備註", ""] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(L) + "\n"


# ── main ───────────────────────────────────────────────────────────────────

def _commit():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _machine():
    import platform
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    except Exception:
        gpu = "?"
    return f"{platform.node()} / {platform.processor() or platform.machine()} / GPU {gpu}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--photos", help="route1: the 測試路線1 photo folder")
    ap.add_argument("--data-root", default=os.environ.get("IMAGE_ONLY_DATA_ROOT", str(BACKEND)),
                    help="folder that holds legacy/testPDR and output/sessions_archive (default backend/)")
    ap.add_argument("--out", default=str(BACKEND / "eval" / "results"))
    ap.add_argument("--cache", help="default backend/output/eval/image_only/<set>")
    ap.add_argument("--map", choices=("neo4j", "snapshot"), default="neo4j")
    ap.add_argument("--limit", type=int, help="first N photos only (smoke test)")
    ap.add_argument("--redetect", action="store_true", help="re-run quality / OCR / GroundingDINO")
    ap.add_argument("--reembed", action="store_true", help="re-run DINOv2 on the query photos")
    args = ap.parse_args(argv)

    import logging
    from server import config as C
    from server import dino_embed
    from server import server as srv
    from server.visual_localization import CONFIDENT_THRESHOLD
    logging.getLogger().setLevel(logging.WARNING)

    C.RERANK_ENABLED = False      # no VLM calls; read inside _run_early_localization
    if not (C.LOC_FUSION_V2 and C.LOC_SERVER_DINO):
        sys.exit("LOC_FUSION_V2 and LOC_SERVER_DINO must be on (that is the production path)")

    cache = Path(args.cache) if args.cache else BACKEND / "output" / "eval" / "image_only" / args.set
    (cache / "img").mkdir(parents=True, exist_ok=True)
    ref_map, map_source = load_map(args.map, cache)
    srv.get_neo4j = lambda: _MapSource(ref_map)
    hops = hop_table(ref_map)
    items = load_items(args.set, ref_map, Path(args.data_root), args.photos)
    if args.limit:
        items = items[:args.limit]

    imgs = {it["id"]: preprocess(it, cache / "img") for it in items}
    perc_all = perceive_all(list(imgs.items()), cache / "perception.json", args.redetect)
    if not dino_embed.available():      # load the model now so its load time isn't billed to photo 1
        sys.exit("DINOv2 not available — pip install torch transformers")
    embed = EmbedCache(cache / "dino.json", dino_embed.embed_image, dino_embed._MODEL_NAME, args.reembed)
    dino_embed.embed_image = embed

    variants = VARIANTS + ABLATIONS
    per_variant = {v: [] for v in variants}
    times = {k: [] for k in STAGES + ("total",)}
    t_run = time.perf_counter()
    for i, it in enumerate(items, 1):
        img, p = imgs[it["id"]], perc_all[it["id"]]
        embed.key = it["id"]
        for v in variants:
            embed.spent = 0.0
            t0 = time.perf_counter()
            with variant(v):
                loc = localize(srv, C.NEO4J_PLACE, p, img)
            if loc is None or loc.method != "server_dino+v2":
                # _run_early_localization logs and swallows its own errors
                sys.exit(f"{it['id']}: server DINOv2 path not taken "
                         f"({'failed' if loc is None else loc.method}) — see the log above")
            t_fusion = time.perf_counter() - t0 - embed.spent
            rec = score_record(it, loc, ref_map, hops)
            rec.update(quality_hint=p["quality"]["hint"], occlusion_hint=p["occlusion"]["hint"],
                       occlusion_fraction=p["occlusion"]["fraction"], n_objects=p["occlusion"]["useful_objects"],
                       n_ocr=len(p["ocr"]),
                       rejected="quality" if p["quality"]["hint"] else ("occlusion" if p["occlusion"]["hint"] else None))
            per_variant[v].append(rec)
            if v == "prod":
                tt = dict(p["t"], dino=(embed.data.get(it["id"]) or {}).get("t"), fusion=t_fusion)
                for k in STAGES:
                    times[k].append(tt[k])
                times["total"].append(sum(x for x in tt.values() if x is not None))
        r = per_variant["prod"][-1]
        d = per_variant["dino_only"][-1]
        print(f"[{i:3d}/{len(items)}] {it['id']:<48s} truth {it['truth']:>2} | prod {r['pred']} "
              f"{'✓' if r['hit1'] else ('~' if r['within1'] else '✗')} {r['confidence']:.2f} | "
              f"dino {d['pred']} {'✓' if d['hit1'] else ('~' if d['within1'] else '✗')} "
              f"| {time.perf_counter() - t_run:6.0f}s", file=sys.stderr, flush=True)

    notes = []
    if args.set == "field1017":
        notes.append("朝向標註＝拍照當下手機羅盤（兩輪 D 測試羅盤 12/12 正常）；地圖方向照片的 heading_deg 與羅盤同一座標才有意義")
        notes.append("GroundingDINO 用通用物件＋遮擋物提示詞（不含當時購物目標），所以偵測結果和現場紀錄可能略有不同")
    if args.set == "0916":
        notes.append(f"0916 測試點離答案節點 ≤{AT_NODE_M:g} m 算 at_node，否則 between；方向（front/right/…）是測試者自己的相對方向，不能當朝向答案")
    if args.set == "route1":
        notes.append("route1 的位置是從走道號碼牌與地標估計的（未人工確認）；在兩節點之間時 node_a、node_b 都算對")

    rep = {
        "set": args.set, "n_photos": len(items),
        "generated": time.strftime("%Y-%m-%d %H:%M"), "commit": _commit(), "machine": _machine(),
        "map": {"place": ref_map.place, "nodes": len(ref_map.photos), "source": map_source,
                "fingerprint": map_fingerprint(ref_map), "sidecar_sha1": _file_sha(C.DINO_SIDECAR_PATH)},
        "config": {"locate_fusion_weights": [5.0, 0.5, 0.5], "server_image_k": C.LOC_SERVER_IMAGE_K,
                   "confident_threshold": CONFIDENT_THRESHOLD,
                   "dino_model": dino_embed._MODEL_NAME, "perception": perception_signatures()},
        "timing_s": {k: _stats(v) for k, v in times.items()},
        "text_coverage": text_coverage({it["id"]: perc_all[it["id"]] for it in items}, ref_map, C.NEO4J_PLACE),
        "variants": {v: {"metrics": summarize(rs), "failures": failures(rs)} for v, rs in per_variant.items()},
        "notes": notes,
        "per_photo": per_variant,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"image_only_{args.set}{'_limit' + str(args.limit) if args.limit else ''}"
    Path(f"{stem}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(f"{stem}.md").write_text(markdown(rep), encoding="utf-8")
    print(markdown(rep))
    print(f"→ {stem}.json / .md", file=sys.stderr)


if __name__ == "__main__":
    main()
