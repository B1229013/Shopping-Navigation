"""Single-photo localization through the production server path, without and with the
phone's sensors.

  python -m eval.image_only --set 0916|field1017|route1|lab1019 [--photos DIR] [--data-root DIR]
                            [--out DIR] [--cache DIR] [--map neo4j|snapshot] [--limit N]
                            [--redetect] [--reembed] [--sim]

Each photo goes through what POST /photo does for localization. Without sensors that is
everything minus what the phone adds (no map_x/y, dist_since_fix, fix_node, dino_topk,
previous fix, goal):

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
  objects_only — W_V = W_O = 0, no aisle signs: the object score alone
  text_only  — W_V = W_N = 0: OCR and aisle signs alone
  (with no signal at all every node scores 0 and the pick is just the first node)

With sensors (prod and dino_only again, same photos):
  +replay    — Sensor Lab sets: the phone inputs eval/sensorlab.replay derives from the
               walk's own PDR (the app's MapPositionEstimator, oracle fix after each point)
  +simσ      — --sim: phone position = truth + N(0, σ²), σ = 1, 2, 4, 8 m (any set)

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

  lab1019    376 photos of three Sensor Lab test walks on 2026-10-19 (output/eval_photos/
             1019_A7/<1,2,3>): 65 node points + 29 midpoints, four directions each;
             answers in eval/data/lab1019_labels.csv (the marker set before each group).

Output (default eval/results/): image_only_<set>.json and .md (summary), and
image_only_<set>_photos.jsonl — one line per photo with everything behind its result:
quality, occlusion, every detection and OCR text, DINOv2 cosine / object / OCR / fused
score per reference node, aisle-sign numbers, the server's image evidence, the Sensor Lab
replay inputs, and every variant's posterior top-10, pick, heading and errors.
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
LAB_SETS = {   # Sensor Lab test walks: labels CSV, photo folders under --data-root
    "lab1019": ("lab1019_labels.csv", "output/eval_photos/1019_A7"),
}
SETS = ("0916", "field1017", "route1") + tuple(LAB_SETS)
VARIANTS = ("prod", "dino_only")
# one of the two extras switched off, or DINOv2 switched off and one extra left on
ABLATIONS = ("no_objects", "no_text", "objects_only", "text_only")
SIM_SIGMAS = (1.0, 2.0, 4.0, 8.0)        # simulated phone position error (m)
SIM_DRAWS = 3
CONF_LEVELS = (0.3, 0.6)
STAGES_BY_PERCEPTION = {"gdino": ("quality", "ocr", "gdino", "dino", "fusion"),
                        "vlm": ("quality", "vlm", "dino", "fusion")}
VLM_GOAL, VLM_GOAL_OBJECTS = "定位", []   # as eval/real_photos.py: no shopping goal
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
    elif name in LAB_SETS:
        items = _lab_items(name, ref_map, data_root, xy)
    else:
        raise ValueError(name)
    missing = [it["path"] for it in items if not Path(it["path"]).is_file()]
    if missing:
        sys.exit(f"{len(missing)} photos missing, e.g. {missing[0]} — pass --data-root / --photos")
    # nodes stored twice at the same spot (9 / 24): either one is the right place
    for it in items:
        it["ok"] = sorted({n for n in xy for o in it["ok"] if math.dist(xy[n], xy[o]) < 0.05})
    return items


def _lab_items(name, ref_map, data_root, xy):
    """Sensor Lab test walks: one item per photo, with what the phone would have sent
    (eval/sensorlab.replay) under item["sensor"]."""
    from eval import sensorlab
    labels_csv, photo_root = LAB_SETS[name]
    rows = _read_csv(labels_csv)
    start_node = min(xy, key=lambda n: math.dist(xy[n], sensorlab.START_XY))
    items = []
    for folder in sorted({r["folder"] for r in rows}, key=lambda f: (len(f), f)):
        sess = sensorlab.read_session(data_root / photo_root / folder)
        pts = []
        for r in (r for r in rows if r["folder"] == folder):
            a = int(r["node_a"])
            b = int(r["node_b"]) if r["node_b"] else None
            f = float(r["fraction"] or 0) if b else 0.0
            t = xy[a] if b is None else (xy[a][0] + f * (xy[b][0] - xy[a][0]), xy[a][1] + f * (xy[b][1] - xy[a][1]))
            truth = b if (b and f > 0.5) else a
            pts.append((int(r["group"]), list(t), truth, a, b, f, r))
        phone, info = sensorlab.replay(sess, [(g, t, n) for g, t, n, *_ in pts], start_node)
        if not info["start_found"]:
            sys.exit(f"{folder}: no 「入口」 marker — the replay needs the start")
        for g, t, truth, a, b, f, r in pts:
            for d, (ms, path) in sorted(sess.photos[g].items(), key=lambda kv: kv[1][0]):
                ph = phone[(g, d)]
                items.append(dict(id=f"{folder}/wp{g}_{d}", path=path, ok=[a] + ([b] if b else []), truth=truth,
                                  truth_xy=t, condition="between" if b else "at_node",
                                  heading=ph["map_heading"], stored=False, sensor=ph,
                                  meta={"point": f"{folder}/{g}", "label": r["label"], "direction": d,
                                        "fraction": f, "note": r.get("note") or "",
                                        "north_jumps_in_walk": info["north_jumps"]}))
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
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows: the target is briefly open in another process (a reader, antivirus)
            if attempt == 19:
                raise
            time.sleep(0.25)


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


def vlm_signature():
    from server import config as C, prompts as P
    text = P.PERCEIVE_PROMPT + P.PERCEIVE_REFINE_PROMPT + P.PERCEIVE_DISTINCTIVE_PROMPT
    return {"model": C.OPENAI_MODEL, "base_url": C.OPENAI_BASE_URL, "rounds": C.VLM_ROUNDS, "pass3": C.VLM_PASS3,
            "goal": VLM_GOAL, "goal_objects": VLM_GOAL_OBJECTS,
            "prompts_sha1": hashlib.sha1(text.encode()).hexdigest()[:12]}


def _call_kind(prompt):
    from server import prompts as P
    for kind, p in (("pass1", P.PERCEIVE_PROMPT), ("pass2", P.PERCEIVE_REFINE_PROMPT),
                    ("pass3", P.PERCEIVE_DISTINCTIVE_PROMPT)):
        if prompt[:40] == p[:40]:
            return kind
    return "other"


def _server_perceive(srv, img):
    """The perception call _handle_photo makes in mode B, as config says."""
    from PIL import Image
    from server import config as C
    with Image.open(img) as im:
        w, h = im.size
    kw = dict(image_path=str(img), goal=VLM_GOAL, goal_objects=VLM_GOAL_OBJECTS, img_w=w, img_h=h)
    if C.VLM_ROUNDS > 1:
        perc, debug = srv._vlm_perceive_multi_round(num_rounds=C.VLM_ROUNDS, **kw)
    elif C.VLM_PASS3:
        perc, debug = srv._vlm_perceive_three_pass(**kw)
    else:
        perc, debug = srv._vlm_perceive_two_pass(**kw)
    return perc, debug, w, h


def _vlm_record(srv, perc, debug, w, h, q, calls, sig, t):
    """One photo's perception as _handle_photo turns it into detections / OCR results:
    `label` is what the turn uses, `label_loc` what localization matches (the server's
    vlm.localization_label — 「中文 English」 since 2026-10)."""
    from server import photo_quality, vlm
    from server.ocr import OCRResult
    from server.perception import Detection
    dets = [Detection(label=d.label, box=d.bbox, score=d.score, position=d.position) for d in perc.detections]
    ocr = srv._clean_vlm_ocr([OCRResult(text=t_.text, confidence=t_.score, bbox=t_.bbox) for t_ in perc.ocr_texts], h)
    occ = photo_quality.assess_occlusion(dets, w, h, goals=VLM_GOAL_OBJECTS)
    loc = getattr(vlm, "localization_label", lambda d: d.label)
    return {
        "size": [w, h], "vlm_sig": sig,
        "quality": {"sharpness": q.sharpness, "brightness": q.brightness, "hint": q.hint},
        "detections": [{"label": d.label, "label_loc": loc(v), "box": [float(x) for x in d.box],
                        "score": float(d.score), "position": d.position} for d, v in zip(dets, perc.detections)],
        "ocr": [{"text": r.text, "confidence": r.confidence, "bbox": r.bbox} for r in ocr],
        "ocr_before_clean": len(perc.ocr_texts),
        "store_zone": perc.store_zone or "", "scene": perc.scene_description,
        "occlusion": {"fraction": occ.fraction, "useful_objects": occ.useful_objects, "hint": occ.hint},
        "vlm_passes": debug, "vlm_calls": calls, "t": t,
    }


def replay_vlm(cache, jobs, srv):
    """Re-run the server's current perception code on the cached raw VLM replies — no API
    call. vlm._generate answers each pass from the cache, in the order the photo was
    originally asked; a pass the current code asks for more often than it was asked then
    (e.g. a retry after a reply the old code could not parse) gets an error, as a failed
    call would. Returns {id: perception} and how many such misses there were."""
    from server import vlm
    out, misses = {}, 0
    real = vlm._generate
    try:
        for k, img in jobs:
            cached = cache[k]
            queue = {}
            for c in cached.get("vlm_calls") or []:
                queue.setdefault(c["kind"], []).append(c)

            def fake(prompt, image_b64=None, max_tokens=vlm.DECIDE_MAX_TOKENS, queue=queue):
                nonlocal misses
                q = queue.get(_call_kind(prompt))
                if not q:
                    misses += 1
                    raise RuntimeError("no cached reply for this call")
                c = q.pop(0)
                if "error" in c:
                    raise RuntimeError(c["error"])
                return c["text"]
            vlm._generate = fake
            perc, debug, w, h = _server_perceive(srv, img)
            q = SimpleNamespace(**cached["quality"])
            out[k] = _vlm_record(srv, perc, debug, w, h, q, cached["vlm_calls"], cached.get("vlm_sig"),
                                 dict(cached["t"]))
            out[k]["replayed"] = True
    finally:
        vlm._generate = real
    return out, misses


def preload_label_embeddings(cache_dir, ref_map, perc_all):
    """The server now fetches label embeddings from OpenAI (map labels once, photo labels
    per photo). For a benchmark run, fetch every label once into eval's cache
    (label_embeddings.npz — paid only for labels never seen) and hand them to the server's
    cache, so localization makes no API call."""
    from eval.label_embeddings import LabelEmbeddings
    from server import config as C, locate_fusion as LF
    from server import visual_localization as VL
    if not VL._load_embedding_cache():
        return 0, None
    embs = LF._reference_embeddings(ref_map, LF._load_sidecar(C.DINO_SIDECAR_PATH))
    map_labels = {l for e in LF._index_for(ref_map, embs)["entries"] for l in e["labels"]}
    photo_labels = {d.get("label_loc") or d["label"] for p in perc_all.values() for d in p["detections"]}
    emb = LabelEmbeddings(cache_dir / "label_embeddings.npz")
    fetched = emb.ensure(sorted(map_labels | photo_labels))
    VL._emb_cache.update({l: emb.matrix([l])[0] for l in map_labels | photo_labels})
    VL._emb_persisted.update(map_labels)      # nothing for the server to save
    from eval.weight_sweep import SemanticSim
    return fetched, SemanticSim("hybrid", emb, sorted(map_labels))


class FastLabelSim:
    """Speed only: visual_localization._hybrid_max_sim answered from one (photo labels ×
    map labels) matrix per photo instead of ~10⁵ single calls per variant — the same
    max(character, cosine) values (eval/weight_sweep checks them against _node_score)."""

    def __init__(self, semsim):
        from server import visual_localization as VL
        self.VL, self.sim, self.real = VL, semsim, VL._hybrid_max_sim
        self.ridx = {l: j for j, l in enumerate(semsim.ref_labels)}

    def photo(self, labels):
        uniq = list(dict.fromkeys(labels))
        S, qidx, ridx, real = self.sim.matrix(uniq), {l: i for i, l in enumerate(uniq)}, self.ridx, self.real

        def lookup(a, b):
            i, j = qidx.get(a), ridx.get(b)
            return float(S[i, j]) if i is not None and j is not None else real(a, b)
        self.VL._hybrid_max_sim = lookup


def perceive_all_vlm(jobs, cache_path, srv, redetect=False):
    """The server's mode B (PERCEPTION_ENABLED=0, OCR_ENABLED=0): quality check, then
    vlm.perceive_three_pass (or two-pass / multi-round, as config says) on the stored
    photo, the VLM text turned into detections / OCRResults and _clean_vlm_ocr'd exactly
    as _handle_photo does. Every VLM call's raw reply, kind (pass1/2/3), time and error is
    kept, and each photo is cached as soon as it is done — the calls are paid."""
    from server import config as C, photo_quality, vlm
    from server.ocr import OCRResult
    from server.perception import Detection
    sig = vlm_signature()
    cache = {} if redetect else _load_json(cache_path)
    todo = [(k, img) for k, img in jobs if (cache.get(k) or {}).get("vlm_sig") != sig]
    if not todo:
        return cache
    if not C.OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is not set — the VLM path needs it")
    real = vlm._generate
    calls = []

    def recorder(prompt, image_b64=None, max_tokens=vlm.DECIDE_MAX_TOKENS):
        t0 = time.perf_counter()
        try:
            text = real(prompt, image_b64=image_b64, max_tokens=max_tokens)
        except Exception as e:
            calls.append({"kind": _call_kind(prompt), "s": round(time.perf_counter() - t0, 2), "error": repr(e)[:300]})
            raise
        calls.append({"kind": _call_kind(prompt), "s": round(time.perf_counter() - t0, 2), "text": text})
        return text

    vlm._generate = recorder
    try:
        for i, (k, img) in enumerate(todo, 1):
            calls.clear()
            t0 = time.perf_counter()
            q = photo_quality.assess_image(str(img))
            t1 = time.perf_counter()
            perc, debug, w, h = _server_perceive(srv, img)
            t2 = time.perf_counter()
            if calls and all("error" in c for c in calls):
                err = calls[-1].get("error", "")
                if "429" in str(err):
                    print(f"  ⚠ {k}: rate limited — waiting 60s then retrying…", file=sys.stderr, flush=True)
                    time.sleep(60)
                    calls.clear()
                    t0b = time.perf_counter()
                    perc, debug, w, h = _server_perceive(srv, img)
                    t2 = time.perf_counter()
                    if calls and all("error" in c for c in calls):
                        print(f"  ⚠ {k}: still failing after retry — skipping", file=sys.stderr, flush=True)
                        continue
                else:
                    sys.exit(f"{k}: every VLM call failed ({err}) — stopped, nothing cached for it")
            cache[k] = _vlm_record(srv, perc, debug, w, h, q, list(calls), sig, {"quality": t1 - t0, "vlm": t2 - t1})
            _save_json(cache_path, cache)
            dets, ocr = cache[k]["detections"], cache[k]["ocr"]
            kinds = Counter(c["kind"] for c in calls)
            print(f"  VLM {i}/{len(todo)} {k}: {len(dets)} objects, {len(ocr)} texts, "
                  f"{len(calls)} calls {dict(kinds)}, {t2 - t1:.1f}s", file=sys.stderr, flush=True)
    finally:
        vlm._generate = real
    return cache


def _vlm_call_summary(percs):
    calls = [c for p in percs for c in p.get("vlm_calls") or []]
    p3 = sum(bool(((p.get("vlm_passes") or {}).get("pass3") or {}).get("detections")
                  or ((p.get("vlm_passes") or {}).get("pass3") or {}).get("ocr")) for p in percs)
    return {"total": len(calls), "by_kind": dict(Counter(c["kind"] for c in calls)),
            "errors": sum("error" in c for c in calls), "pass3_useful_photos": p3}


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
    saved = (LF.W_V, LF.W_N, LF.W_O, C.AISLE_SIGN_PLACES)
    try:
        if name in ("objects_only", "text_only"):
            LF.W_V = 0.0
        if name in ("dino_only", "no_objects", "text_only"):
            LF.W_N = 0.0
        if name in ("dino_only", "no_text", "objects_only"):
            LF.W_O = 0.0
            C.AISLE_SIGN_PLACES = set()
        if name not in VARIANTS + ABLATIONS:
            raise ValueError(name)
        yield
    finally:
        LF.W_V, LF.W_N, LF.W_O, C.AISLE_SIGN_PLACES = saved


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


def live_comparison(items, image_ev):
    """field1017: the live server logged each photo's image evidence (DINOv2 + objects + OCR
    + aisle signs, 0..2 per node) in turns.jsonl. Compare it with this run's prod evidence
    for the same photo: the DINOv2 part is identical, objects / OCR come from a new VLM
    call (temperature 0.2) and a different goal, so they can differ somewhat."""
    rows = []
    turns_cache = {}
    for it in items:
        sess = Path(it["path"]).parent.parent
        if sess not in turns_cache:
            f = sess / "turns.jsonl"
            turns_cache[sess] = ([json.loads(l) for l in f.open(encoding="utf-8")] if f.is_file() else [])
        n = int(Path(it["path"]).stem)
        turn = next((t for t in turns_cache[sess] if t.get("kind") == "photo"
                     and (t.get("response") or {}).get("node_id") == n), None)
        live = ((turn or {}).get("localization") or {}).get("image_evidence")
        ours = image_ev.get(it["id"])
        if not live or not ours:
            continue
        keys = sorted(set(live) & set(ours), key=int)
        a = np.array([live[k] for k in keys])
        b = np.array([ours[k] for k in keys])
        top_live, top_ours = keys[int(np.argmax(a))], keys[int(np.argmax(b))]
        rows.append({"id": it["id"], "truth": it["truth"], "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 3),
                     "max_abs_diff": round(float(np.abs(a - b).max()), 3), "top_live": int(top_live),
                     "top_ours": int(top_ours), "same_top": top_live == top_ours,
                     "live_top5": [int(keys[j]) for j in np.argsort(-a)[:5]],
                     "ours_top5": [int(keys[j]) for j in np.argsort(-b)[:5]]})
    if not rows:
        return None
    return {"photos": len(rows), "same_top1": sum(r["same_top"] for r in rows),
            "median_pearson_r": round(statistics.median(r["pearson_r"] for r in rows), 3),
            "live_top1_correct": sum(r["top_live"] == r["truth"] for r in rows),
            "ours_top1_correct": sum(r["top_ours"] == r["truth"] for r in rows), "per_photo": rows}


def draw_annotations(perception, img, perc, out_dir, item_id, banner):
    """Annotated photos drawn with the server's own annotator, as _handle_photo saves
    them: VLM pass1 / pass2 (/ pass3) and the merged result; GroundingDINO: merged only.
    The merged image's banner carries the localization results. Returns {tag: path}."""
    from server.annotator import annotate
    from server.ocr import OCRResult
    from server.perception import Detection
    stem = item_id.replace("/", "__").rsplit(".", 1)[0]
    vlm = perception == "vlm"
    out = {}

    def draw(tag, dets, ocr, text):
        path = out_dir / f"{stem}_{tag}.jpg"
        annotate(str(img), str(path),
                 [Detection(label=d["label"], box=d["box"], score=d["score"], position=d.get("position"))
                  for d in dets if d.get("box") and len(d["box"]) == 4],
                 banner_text=text,
                 ocr_results=[OCRResult(text=o["text"], confidence=o.get("confidence", o.get("score", 0.5)),
                                        bbox=o["bbox"]) for o in ocr if o.get("bbox")] or None,
                 goal_objects=[], region_mode=vlm)
        out[tag] = str(path)

    if vlm:
        for tag in ("pass1", "pass2", "pass3"):
            p = (perc.get("vlm_passes") or {}).get(tag)
            if p and (p.get("detections") or p.get("ocr")) and not (out_dir / f"{stem}_{tag}.jpg").is_file():
                draw(tag, p["detections"], p["ocr"],
                     f"{tag}: {len(p['detections'])} detections, {len(p['ocr'])} OCR")
            elif (out_dir / f"{stem}_{tag}.jpg").is_file():
                out[tag] = str(out_dir / f"{stem}_{tag}.jpg")
    draw("merged", perc["detections"], perc["ocr"], banner)
    return out


def evidence_detail(ref_map, perc, q_emb, place):
    """Every signal behind one photo's image evidence, per reference node: DINOv2 cosine
    (best directional photo), object score, OCR score, and the prod fused score — the
    same terms locate_fusion.best_photo_per_node adds up — plus aisle-sign numbers."""
    from server import aisle_signs, config as C
    from server import locate_fusion as LF
    from server.ocr import OCRResult
    from server.perception import Detection
    embs = LF._reference_embeddings(ref_map, LF._load_sidecar(C.DINO_SIDECAR_PATH))
    index = LF._index_for(ref_map, embs)
    E = index["entries"]
    labels = [d.get("label_loc") or d["label"] for d in perc["detections"]]
    q_ocr = {LF._norm_ocr(o["text"]) for o in perc["ocr"]} - {""}
    vis = np.array([float(np.dot(q_emb, e["emb"])) if (q_emb is not None and e["emb"] is not None) else 0.0
                    for e in E])
    nod = np.array([LF._node_score(labels, e["labels"]) for e in E])
    ocr = np.zeros(len(E))
    for k, e in enumerate(E):
        shared = q_ocr & e["ocrset"]
        if shared:
            ocr[k] = (sum(index["idf"].get(t, index["def_idf"]) for t in shared)
                      / sum(index["idf"].get(t, index["def_idf"]) for t in q_ocr))
    mm = lambda a: (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else a * 0
    fused = LF.W_V * mm(vis) + LF.W_N * mm(nod) + LF.W_O * ocr
    per_node = {}
    for k, e in enumerate(E):
        n = per_node.setdefault(e["nid"], {"dino": -9.0, "dino_slot": None, "obj": 0.0, "ocr": 0.0,
                                           "fused": -9.0, "fused_slot": None, "ocr_shared": []})
        if vis[k] > n["dino"]:
            n["dino"], n["dino_slot"] = float(vis[k]), e["slot"]
        if fused[k] > n["fused"]:
            n["fused"], n["fused_slot"] = float(fused[k]), e["slot"]
        n["obj"] = max(n["obj"], float(nod[k]))
        if ocr[k] > n["ocr"]:
            n["ocr"], n["ocr_shared"] = float(ocr[k]), sorted(q_ocr & e["ocrset"])
    w, h = perc["size"]
    from server import photo_quality
    dets = [Detection(label=d.get("label_loc") or d["label"], box=d["box"], score=d["score"])
            for d in perc["detections"]]
    # as server._build_localization_items: occluders are not part of the map
    items = [{"label": d.label, "box": [d.box[0] / w, d.box[1] / h, d.box[2] / w, d.box[3] / h]} for d in dets
             if not photo_quality.is_occluder(d.label)]
    numbers = dict(aisle_signs.numbers_in_items(items))
    for n_, a in aisle_signs.numbers_from_ocr([OCRResult(o["text"], o["confidence"], o["bbox"]) for o in perc["ocr"]],
                                              w, h).items():
        numbers[n_] = max(numbers.get(n_, 0.0), a)
    aisle = aisle_signs.sign_evidence(numbers, ref_map, place) if numbers else {}
    r4 = lambda v: round(v, 4) if isinstance(v, float) else v
    return {
        "per_node": {str(nid): {k: r4(v) for k, v in d.items()} for nid, d in sorted(per_node.items())},
        "dino_top10": [[nid, round(d["dino"], 4), d["dino_slot"]]
                       for nid, d in sorted(per_node.items(), key=lambda kv: -kv[1]["dino"])[:10]],
        "aisle_numbers": {str(k): round(v, 4) for k, v in numbers.items()},
        "aisle_evidence": {str(k): round(v, 4) for k, v in (aisle or {}).items() if v},
    }


def localize(srv, place, perc, img_path, phone=None):
    """server._run_early_localization with a fresh session. phone = {map_xy, dist_since_fix,
    fix_node} as the app would send them (None → no sensor input at all); they go through
    the server's own _phone_localization_inputs, like the /photo form fields."""
    from server import aisle_signs
    from server.ocr import OCRResult
    from server.perception import Detection
    # mode B localizes with vlm.localization_label (label_loc); older caches have only label
    dets = [Detection(label=d.get("label_loc") or d["label"], box=d["box"], score=d["score"])
            for d in perc["detections"]]
    ocr = [OCRResult(text=o["text"], confidence=o["confidence"], bbox=o["bbox"]) for o in perc["ocr"]]
    w, h = perc["size"]
    s = SimpleNamespace(place=place, last_corrected_nid=None, last_loc_log=None,
                        user_heading=None, heading_slot=None, heading_confidence=0.0)
    xy = (phone or {}).get("map_xy") or (None, None)
    inputs = srv._phone_localization_inputs(None, None, xy[0], xy[1], (phone or {}).get("dist_since_fix"),
                                            (phone or {}).get("fix_node"))
    loc = srv._run_early_localization(
        s, "image_only", [d.label for d in dets], [o.text for o in ocr],
        detected_items=srv._build_localization_items(dets, ocr, w, h),
        image_path=str(img_path),
        ocr_items=srv._build_ocr_items(ocr, w, h),
        sign_numbers=aisle_signs.numbers_from_ocr(ocr, w, h),
        extra_evidence=None,
        # mode B (VLM) passes the VLM's store_zone (always "" today); mode A passes none
        **({"store_zone": perc["store_zone"]} if "store_zone" in perc else {}),
        **inputs)
    return loc, s.last_loc_log


def _rounded(x, nd=3):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _rounded(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v, nd) for v in x]
    return x


def sim_phone(item, sigma, draw):
    """Phone position = truth + N(0, σ²) per axis, one draw per test point (all photos of a
    point share it); dist_since_fix chosen so the server's sensor_sigma() equals σ; no
    fix_node (no reachability limit) so only the position error is tested."""
    from server import config as C
    key = f"{item['meta'].get('point', item['id'])}|{sigma}|{draw}"
    rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))
    dx, dy = rng.normal(0.0, sigma, 2)
    return {"map_xy": [item["truth_xy"][0] + dx, item["truth_xy"][1] + dy],
            "dist_since_fix": max(0.0, (sigma - C.LOC_SENSOR_SIGMA0) / C.LOC_DRIFT_RATE), "fix_node": None}


class EvidenceMemo:
    """The server's image evidence depends only on the photo and the fusion weights, not on
    the phone inputs: compute it once per (photo, weights) when the same photo is localized
    under several sensor modes. Wraps locate_fusion.server_image_evidence (looked up at
    call time by server._server_dino_evidence)."""

    def __init__(self, real):
        self.real, self.memo = real, {}

    def __call__(self, ref_map, image_path, q_labels, q_ocr_tokens, sidecar_path):
        from server import locate_fusion as LF
        key = (image_path, LF.W_V, LF.W_N, LF.W_O, tuple(q_labels), tuple(q_ocr_tokens))
        if key not in self.memo:
            if len(self.memo) > 64:
                self.memo.clear()
            self.memo[key] = self.real(ref_map, image_path, q_labels, q_ocr_tokens, sidecar_path)
        return self.memo[key]


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
        # photos where the image evidence separates nodes at all (all-zero = no signal)
        "photos_with_signal": _pct([(r.get("evidence_max") or 0.0) > 1e-9 for r in recs]),
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
    out["per_point"] = point_summary(recs)
    return out


def run_key(variant_name, mode):
    return variant_name if mode == "none" else f"{variant_name}+{mode}"


def point_summary(recs):
    """Several photos per test point (Sensor Lab: four directions) are not independent:
    score each point once — mean of its photos, any photo right, and the photo the server
    was most confident about (what a user taking all four would get)."""
    groups = {}
    for r in recs:
        groups.setdefault((r["meta"].get("point", r["id"]), r.get("draw")), []).append(r)
    if all(len(g) == 1 for g in groups.values()):
        return None
    out = {}
    for cond in [None] + sorted({r["condition"] for r in recs}):
        gs = [g for g in groups.values() if cond is None or g[0]["condition"] == cond]
        out[cond or "all"] = {
            "points": len({k[0] for k, g in groups.items() if cond is None or g[0]["condition"] == cond}),
            "mean_top1": round(100 * statistics.fmean(sum(r["hit1"] for r in g) / len(g) for g in gs), 1) if gs else None,
            "any_top1": _pct([any(r["hit1"] for r in g) for g in gs]),
            "most_confident_top1": _pct([max(g, key=lambda r: r["confidence"])["hit1"] for g in gs]),
            "most_confident_within_1_hop": _pct([max(g, key=lambda r: r["confidence"])["within1"] for g in gs]),
        }
    return out


def grid(per_variant, modes):
    """Photo condition × sensor mode, per image variant (the 2×2 table and its sim columns)."""
    out = {}
    for v in VARIANTS:
        for m in modes:
            recs = per_variant.get(run_key(v, m))
            if not recs:
                continue
            for cond in sorted({r["condition"] for r in recs}):
                rs = [r for r in recs if r["condition"] == cond]
                out.setdefault(v, {}).setdefault(cond, {})[m] = {
                    "n": len(rs), "top1": _pct([r["hit1"] for r in rs]),
                    "within_1_hop": _pct([r["within1"] for r in rs]),
                    "error_m_median": _stats([r["err_m"] for r in rs])["median"],
                    "estimated_xy_error_m_median": _stats([r["est_err_m"] for r in rs])["median"]}
    return out


def paired(a, b):
    """Same photos, two settings: how many each gets right that the other does not."""
    out = {}
    for cond in [None] + sorted({r["condition"] for r in a}):
        pairs = [(x, y) for x, y in zip(a, b) if cond is None or x["condition"] == cond]
        assert all(x["id"] == y["id"] for x, y in pairs)
        out[cond or "all"] = {"both": sum(x["hit1"] and y["hit1"] for x, y in pairs),
                              "only_first": sum(x["hit1"] and not y["hit1"] for x, y in pairs),
                              "only_second": sum(y["hit1"] and not x["hit1"] for x, y in pairs),
                              "neither": sum(not x["hit1"] and not y["hit1"] for x, y in pairs)}
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
    modes = rep.get("sensor_modes", ["none"])
    L = [f"# 單張照片定位基準：{rep['set']}", "",
         f"- {rep['n_photos']} 張照片；版本名稱沒有「+」的不帶任何感測器（無 map_x/y、dist_since_fix、fix_node、"
         f"手機 DINOv2、前一次定位），「+replay／+simσ」是帶感測器",
         f"- 地圖 `{rep['map']['place']}`：{rep['map']['nodes']} 節點，來源 {rep['map']['source']}，指紋 {rep['map']['fingerprint']}",
         f"- 產生時間 {rep['generated']}；commit {rep['commit'] or '?'}", ""]
    L += ["## 主要指標", "",
          "| 版本 | Top-1 | Top-5 | 1 hop 內 | 誤差 平均/中位 (m) | 內插位置誤差 平均/中位 (m) | 影像證據有訊號的照片 |",
          "|---|---|---|---|---|---|---|"]
    for v, s in rep["variants"].items():
        m = s["metrics"]
        L.append(f"| {v} | {_fmt(m['top1'], '%')} | {_fmt(m['top5'], '%')} | {_fmt(m['within_1_hop'], '%')} | "
                 f"{_fmt(m['error_m']['mean'])} / {_fmt(m['error_m']['median'])} | "
                 f"{_fmt(m['estimated_xy_error_m']['mean'])} / {_fmt(m['estimated_xy_error_m']['median'])} | "
                 f"{_fmt(m.get('photos_with_signal'), '%')} |")
    if len(modes) > 1:
        L += ["", "## 照片類型 × 感測器（Top-1 ／ 1 hop 內 ／ 內插位置誤差中位數）", ""]
        for v, by_cond in rep["grid"].items():
            L += [f"**{v}**", "", "| 照片 | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
            for cond, cells in by_cond.items():
                L.append(f"| {cond}（{next(iter(cells.values()))['n']} 張） | " + " | ".join(
                    (f"{_fmt(c['top1'], '%')} / {_fmt(c['within_1_hop'], '%')} / {_fmt(c['estimated_xy_error_m_median'], ' m')}"
                     if (c := cells.get(m)) else "–") for m in modes) + " |")
            L.append("")
    if rep.get("paired"):
        L += ["## 同一張照片：不帶 vs 帶感測器（Top-1）", "",
              "| 比較 | 照片 | 兩者都對 | 只有不帶感測器對 | 只有帶感測器對 | 都錯 |", "|---|---|---|---|---|---|"]
        for name, by_cond in rep["paired"].items():
            for cond, c in by_cond.items():
                L.append(f"| {name} | {cond} | {c['both']} | {c['only_first']} | {c['only_second']} | {c['neither']} |")
        L.append("")
    pp = {v: s["metrics"]["per_point"] for v, s in rep["variants"].items() if s["metrics"].get("per_point")}
    if pp:
        L += ["## 逐點計分（同一點的照片只算一個樣本）", "",
              "| 版本 | 照片 | 點數 | 每點平均 Top-1 | 任一張對 | 最有信心那張對 | 最有信心那張 1 hop 內 |", "|---|---|---|---|---|---|---|"]
        for v, by_cond in pp.items():
            for cond, c in by_cond.items():
                L.append(f"| {v} | {cond} | {c['points']} | {_fmt(c['mean_top1'], '%')} | {_fmt(c['any_top1'], '%')} | "
                         f"{_fmt(c['most_confident_top1'], '%')} | {_fmt(c['most_confident_within_1_hop'], '%')} |")
        L.append("")
    L += ["", "## 信心校準（信心 = localize 回傳的最佳節點機率）", "",
          "| 版本 | 門檻 | 涵蓋率 | 命中率 (Top-1) | 1 hop 內 |", "|---|---|---|---|---|"]
    for v, s in rep["variants"].items():
        for t, c in s["metrics"]["confidence"].items():
            L.append(f"| {v} | {t} | {_fmt(c['coverage'], '%')} ({c['n']}) | {_fmt(c['top1_when_confident'], '%')} | "
                     f"{_fmt(c['within_1_hop_when_confident'], '%')} |")
    L += ["", "正式伺服器的閘門（品質不佳 / 被遮擋 → 不定位；機率 < CONFIDENT_THRESHOLD → 不給節點）：", ""]
    for v, s in rep["variants"].items():
        if "+" in v and "replay" not in v:
            continue
        g = s["metrics"]["production_gate"]
        L.append(f"- {v}：品質退件 {g['rejected_quality']}、遮擋退件 {g['rejected_occlusion']}、低於門檻 "
                 f"{g['below_threshold']}；有回答 {g['answered']} 張（{_fmt(g['coverage'], '%')}），其中正確 "
                 f"{_fmt(g['top1_of_answered'], '%')}")
    L += ["", "## 朝向", ""]
    for v, s in rep["variants"].items():
        if v not in ("prod", "dino_only", "prod+replay", "dino_only+replay"):
            continue
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
    for k, s in t.items():
        L.append(f"| {k} | {_fmt(s['mean'])} | {_fmt(s['median'])} |")
    L.append(f"\n機器：{rep['machine']}")
    if rep.get("vlm_calls"):
        c = rep["vlm_calls"]
        L.append(f"\nVLM 呼叫：共 {c['total']} 次（{c['by_kind']}），失敗 {c['errors']} 次；"
                 f"第 3 階段有新物件或文字的照片 {c['pass3_useful_photos']}/{rep['n_photos']}")
    lc = rep.get("live_comparison")
    if lc:
        L += ["", "## 和現場伺服器紀錄比對（同一張照片的影像證據）", "",
              f"- {lc['photos']} 張：影像證據第一名相同 {lc['same_top1']} 張；Pearson r 中位數 {lc['median_pearson_r']}",
              f"- 影像證據第一名答對：現場 {lc['live_top1_correct']} 張、本次 {lc['ours_top1_correct']} 張"]
    tc = rep["text_coverage"]
    L += ["", f"文字證據能作用的照片：OCR 字串和地圖完全相同 {tc['with_ocr_text_in_map']}/{tc['photos']}、"
              f"有走道號碼證據 {tc['with_aisle_sign_evidence']}/{tc['photos']}"]
    for v, s in rep["variants"].items():
        if v not in ("prod", "dino_only", "prod+replay"):
            continue
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
    ap.add_argument("--sim", action="store_true",
                    help=f"also localize with a simulated phone position (σ = {', '.join(f'{s:g}' for s in SIM_SIGMAS)} m)")
    ap.add_argument("--perception", choices=("gdino", "vlm"), default="gdino",
                    help="gdino = server mode A (GroundingDINO + EasyOCR); vlm = mode B, what the deployed "
                         ".env runs (PERCEPTION_ENABLED=0): paid OpenAI calls, cached per photo")
    ap.add_argument("--no-annotate", action="store_true", help="skip the annotated photos")
    ap.add_argument("--vlm-replay", action="store_true",
                    help="with --perception vlm: re-run the server's current perception code on the cached "
                         "raw VLM replies (no VLM call) — to test parser / label changes. VLM_PASS3=0 in the "
                         "environment replays without pass 3")
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

    vlm_mode = args.perception == "vlm"
    if args.vlm_replay and not vlm_mode:
        sys.exit("--vlm-replay needs --perception vlm")
    suffix = (("_vlm" if vlm_mode else "") + ("_replay" if args.vlm_replay else "")
              + ("_nopass3" if args.vlm_replay and not C.VLM_PASS3 else "") + (f"_limit{args.limit}" if args.limit else ""))
    imgs = {it["id"]: preprocess(it, cache / "img") for it in items}
    replay_misses = None
    if vlm_mode:
        if args.vlm_replay:
            # never through perceive_all_vlm: its cache key includes VLM_PASS3, so a pass-3-off
            # replay would look stale and be re-asked (paid)
            cached = _load_json(cache / "perception_vlm.json")
            missing = [k for k in imgs if not (cached.get(k) or {}).get("vlm_calls")]
            if missing:
                sys.exit(f"{len(missing)} photos have no cached VLM replies (e.g. {missing[0]}) — "
                         f"run without --vlm-replay first")
            perc_all, replay_misses = replay_vlm(cached, list(imgs.items()), srv)
        else:
            perc_all = perceive_all_vlm(list(imgs.items()), cache / "perception_vlm.json", srv, args.redetect)
        n_emb, semsim = preload_label_embeddings(cache, ref_map, {k: perc_all[k] for k in imgs})
        if n_emb:
            print(f"fetched {n_emb} new label embeddings", file=sys.stderr)
        fast_sim = FastLabelSim(semsim) if semsim is not None else None
    else:
        fast_sim = None
    if not vlm_mode:
        perc_all = perceive_all(list(imgs.items()), cache / "perception.json", args.redetect)
    ann_dir = cache / f"annotated_{args.perception}"
    if not args.no_annotate:
        ann_dir.mkdir(parents=True, exist_ok=True)
    if not dino_embed.available():      # load the model now so its load time isn't billed to photo 1
        sys.exit("DINOv2 not available — pip install torch transformers")
    embed = EmbedCache(cache / "dino.json", dino_embed.embed_image, dino_embed._MODEL_NAME, args.reembed)
    dino_embed.embed_image = embed

    from server import locate_fusion as LF
    LF.server_image_evidence = EvidenceMemo(LF.server_image_evidence)

    has_pdr = any(it.get("sensor") for it in items)
    sensor_modes = (["replay"] if has_pdr else []) + ([f"sim{s:g}" for s in SIM_SIGMAS] if args.sim else [])
    runs = [(v, "none") for v in VARIANTS + ABLATIONS] + [(v, m) for m in sensor_modes for v in VARIANTS]
    per_variant = {run_key(v, m): [] for v, m in runs}
    STAGES = STAGES_BY_PERCEPTION[args.perception]
    times = {k: [] for k in STAGES + ("total",)}
    all_image_ev = {}
    detail_path = Path(args.out) / f"image_only_{args.set}{suffix}_photos.jsonl"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_f = open(detail_path, "w", encoding="utf-8")
    t_run = time.perf_counter()
    for i, it in enumerate(items, 1):
        img, p = imgs[it["id"]], perc_all[it["id"]]
        embed.key = it["id"]
        if fast_sim is not None:
            fast_sim.photo([d.get("label_loc") or d["label"] for d in p["detections"]])
        base = dict(quality_hint=p["quality"]["hint"], occlusion_hint=p["occlusion"]["hint"],
                    occlusion_fraction=p["occlusion"]["fraction"], n_objects=p["occlusion"]["useful_objects"],
                    n_ocr=len(p["ocr"]),
                    rejected="quality" if p["quality"]["hint"] else ("occlusion" if p["occlusion"]["hint"] else None))
        results = {}
        for v, m in runs:
            phones = ([(None, None)] if m == "none" else [(None, it["sensor"])] if m == "replay"
                      else [(d, sim_phone(it, float(m[3:]), d)) for d in range(SIM_DRAWS)])
            for draw, phone in phones:
                embed.spent = 0.0
                t0 = time.perf_counter()
                with variant(v):
                    loc, loc_log = localize(srv, C.NEO4J_PLACE, p, img, phone)
                if loc is None or loc.method != "server_dino+v2":
                    # _run_early_localization logs and swallows its own errors
                    sys.exit(f"{it['id']}: server DINOv2 path not taken "
                             f"({'failed' if loc is None else loc.method}) — see the log above")
                t_fusion = time.perf_counter() - t0 - embed.spent
                rec = score_record(it, loc, ref_map, hops)
                rec.update(base, mode=m, draw=draw, phone=phone,
                           evidence_max=max((loc.image_evidence or {}).values(), default=0.0),
                           sensor_sigma=(loc_log or {}).get("sensor_sigma_m"),
                           reachable_n=len((loc_log or {}).get("reachable_nodes") or []) or None)
                per_variant[run_key(v, m)].append(rec)
                results.setdefault(run_key(v, m), []).append(_rounded(
                    {k: rec[k] for k in ("pred", "confidence", "answered", "top5", "hit1", "hit5", "hop", "err_m",
                                         "est_err_m", "slot", "heading_pred", "sensor_sigma", "reachable_n", "draw")}
                    | {"phone": {k: v_ for k, v_ in (phone or {}).items()
                                 if k in ("map_xy", "dist_since_fix", "fix_node")} or None,
                       "estimated_xy": list(loc.estimated_xy) if loc.estimated_xy else None,
                       "matched_photo": loc.matched_photo}))
                if (v, m) == ("prod", "none"):
                    tt = dict(p["t"], dino=(embed.data.get(it["id"]) or {}).get("t"), fusion=t_fusion)
                    for k in STAGES:
                        times[k].append(tt[k])
                    times["total"].append(sum(x for x in tt.values() if x is not None))
                    image_ev = {str(k): round(float(val), 4) for k, val in (loc.image_evidence or {}).items()}
        all_image_ev[it["id"]] = image_ev
        q = embed.data.get(it["id"])
        detail = evidence_detail(ref_map, p, np.asarray(q["emb"], np.float32) if q else None, C.NEO4J_PLACE)
        r, d = per_variant["prod"][-1], per_variant["dino_only"][-1]
        mark = lambda x: "✓" if x["hit1"] else ("~" if x["within1"] else "✗")
        annotated = {}
        if not args.no_annotate:
            word = lambda x: "對" if x["hit1"] else ("鄰" if x["within1"] else "錯")   # the annotator font has no ✓/✗
            banner = (f"答案 {it['meta'].get('label', it['truth'])} | prod {r['pred']} {word(r)} {r['confidence']:.2f}"
                      f" | dino_only {d['pred']} {word(d)} {d['confidence']:.2f}")
            if "replay" in sensor_modes:
                rr = per_variant["prod+replay"][-1]
                banner += f" | +PDR {rr['pred']} {word(rr)} {rr['confidence']:.2f}"
            annotated = draw_annotations(args.perception, img, p, ann_dir, it["id"], banner)
        detail_f.write(json.dumps({
            "id": it["id"], "photo": str(it["path"]), "perception": args.perception, "truth": it["truth"],
            "ok": it["ok"], "truth_xy": it["truth_xy"], "condition": it["condition"], "heading_label": it["heading"],
            "meta": it["meta"], "size": p["size"], "quality": p["quality"], "occlusion": p["occlusion"],
            "detections": p["detections"], "ocr": p["ocr"], "times_s": p["t"] | {"dino": (q or {}).get("t")},
            **({k: p.get(k) for k in ("scene", "store_zone", "ocr_before_clean", "vlm_passes", "vlm_calls")}
               if vlm_mode else {}),
            "sensor_lab": it.get("sensor"), "server_image_evidence": image_ev, **detail, "results": results,
            "annotated": annotated,
        }, ensure_ascii=False) + "\n")
        extra = ""
        if "replay" in sensor_modes:
            rr = per_variant["prod+replay"][-1]
            extra = f" | +PDR {rr['pred']} {mark(rr)} {rr['confidence']:.2f}"
        print(f"[{i:3d}/{len(items)}] {it['id']:<40s} truth {it['truth']:>2} | prod {r['pred']} {mark(r)} "
              f"{r['confidence']:.2f} | dino {d['pred']} {mark(d)}{extra} | {time.perf_counter() - t_run:6.0f}s",
              file=sys.stderr, flush=True)
    detail_f.close()

    notes = []
    if vlm_mode:
        sig = vlm_signature()
        from server import locate_fusion as LF, visual_localization as VL
        notes.append(f"感知＝伺服器模式 B（部署中的 .env：PERCEPTION_ENABLED=0、OCR_ENABLED=0）："
                     f"{'三階段' if sig['pass3'] else '兩階段'} VLM（{sig['model']}），目標用中性的「{VLM_GOAL}」；每張照片只問一次並快取")
        notes.append(f"融合權重 DINOv2 : 物件 : 文字 = {LF.W_V:g} : {LF.W_N:g} : {LF.W_O:g}；物件相似度"
                     + ("＝字元與語意向量取較大值" if VL._emb_on else "＝只有字元"))
        if args.vlm_replay:
            notes.append(f"重播：用快取的 VLM 原始回覆跑目前的伺服器解析程式（未呼叫 VLM）；"
                         f"第 3 階段{'開' if C.VLM_PASS3 else '關'}；需要重問但沒有快取回覆的呼叫 {replay_misses} 次（當作失敗）")
    else:
        notes.append("感知＝伺服器模式 A（GroundingDINO＋EasyOCR）；部署中的 .env 其實是模式 B（VLM），見 _vlm 報表")
    if args.set == "field1017":
        notes.append("朝向標註＝拍照當下手機羅盤（兩輪 D 測試羅盤 12/12 正常）；地圖方向照片的 heading_deg 與羅盤同一座標才有意義")
        notes.append("感知不帶當時的購物目標（啤酒），所以物件和文字可能和現場紀錄略有不同")
    if args.set == "0916":
        notes.append(f"0916 測試點離答案節點 ≤{AT_NODE_M:g} m 算 at_node，否則 between；方向（front/right/…）是測試者自己的相對方向，不能當朝向答案")
    if args.set == "route1":
        notes.append("route1 的位置是從走道號碼牌與地標估計的（未人工確認）；在兩節點之間時 node_a、node_b 都算對")
    if args.set in LAB_SETS:
        notes.append("答案＝拍照前設的點位標記（eval/data/" + LAB_SETS[args.set][0] + "）；中點在兩端任一節點都算對，誤差量到中點本身")
        notes.append("+replay：手機 MapPositionEstimator 的 Python 版重播 Sensor Lab 的原始 PDR（入口對齊、每點拍完後用"
                     "「正確位置」修正一次＝oracle），送出的 map_x/y、dist_since_fix、fix_node 和導航模式相同")
        notes.append("朝向標註＝拍照當下 cal_yaw 經同一個估計器轉成地圖方向（入口面向賣場＝0°），是推算值")
    if args.sim:
        notes.append(f"+simσ：手機位置＝真實位置＋σ 公尺的高斯誤差（每點 {SIM_DRAWS} 次平均），dist_since_fix 設成讓伺服器的 σ 等於該值，不給 fix_node")
    notes.append("位置相同的兩個節點（9 和 24）互相算對")

    rep = {
        "set": args.set, "n_photos": len(items),
        "generated": time.strftime("%Y-%m-%d %H:%M"), "commit": _commit(), "machine": _machine(),
        "map": {"place": ref_map.place, "nodes": len(ref_map.photos), "source": map_source,
                "fingerprint": map_fingerprint(ref_map), "sidecar_sha1": _file_sha(C.DINO_SIDECAR_PATH)},
        "config": {"locate_fusion_weights": [LF.W_V, LF.W_N, LF.W_O],
                   "server_image_k": C.LOC_SERVER_IMAGE_K,
                   "confident_threshold": CONFIDENT_THRESHOLD,
                   "dino_model": dino_embed._MODEL_NAME,
                   "perception": vlm_signature() if vlm_mode else perception_signatures()},
        "perception": args.perception,
        "vlm_calls": _vlm_call_summary([perc_all[it["id"]] for it in items]) if vlm_mode else None,
        "live_comparison": live_comparison(items, all_image_ev) if args.set == "field1017" else None,
        "timing_s": {k: _stats(v) for k, v in times.items()},
        "text_coverage": text_coverage({it["id"]: perc_all[it["id"]] for it in items}, ref_map, C.NEO4J_PLACE),
        "sensor_modes": ["none"] + sensor_modes,
        "variants": {v: {"metrics": summarize(rs), "failures": failures(rs)} for v, rs in per_variant.items()},
        "grid": grid(per_variant, ["none"] + sensor_modes),
        "paired": {f"{v} vs {v}+replay": paired(per_variant[v], per_variant[run_key(v, "replay")])
                   for v in VARIANTS if "replay" in sensor_modes},
        "notes": notes,
        "detail_file": detail_path.name,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"image_only_{args.set}{suffix}"
    Path(f"{stem}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(f"{stem}.md").write_text(markdown(rep), encoding="utf-8")
    print(markdown(rep))
    print(f"→ {stem}.json / .md", file=sys.stderr)


if __name__ == "__main__":
    main()
