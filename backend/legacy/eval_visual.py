"""eval_visual.py — 加入影像 embedding(CLIP) 的定位評測。

全新訊號：直接比「照片像素」的視覺相似度，不靠物件偵測/文字。
比較：Node+OCR(現最佳) vs Visual-only vs 各種組合。
影像 embedding 存硬碟快取，重跑快。
Usage: python eval_visual.py
"""
from __future__ import annotations
import glob, json, pickle, sys, time
from collections import Counter, defaultdict
from math import log
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server import locate_v2
from server.topomap_v2 import TopoGraphV2
from server.locate_v2 import ensure_embeddings
from eval_localization_real import TOPOMAP, GT_FILE
from eval_localization_single import load_query_photos, _base_dir
from eval_localization_gm_faithful import _nsim, NODE_THR
from eval_localization_gm_ocr import load_query_ocr, _norm, EMB_CACHE

ARCHIVE = Path(r"C:\Users\yixia\Downloads\封存")
VIS_CACHE = HERE / "benchmark_results" / "realworld_0916" / "clip_cache.pkl"
A_OCR = 0.4

# ── CLIP 影像 embedding ──
_clip = {}
def _load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    print("載入 CLIP (openai/clip-vit-base-patch32)...", flush=True)
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32"); m.eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip["m"], _clip["p"], _clip["torch"] = m, p, torch

def _embed(path):
    from PIL import Image, ImageOps
    torch = _clip["torch"]
    im = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    inp = _clip["p"](images=im, return_tensors="pt")
    with torch.no_grad():
        f = _clip["m"].get_image_features(**inp)
    f = f / f.norm(dim=-1, keepdim=True)
    return f[0].numpy().astype(np.float32)

def _map_img(photos, bdir):
    for slot, pf in (photos or {}).items():
        if bdir and bdir in slot:
            if "/" in pf:
                fo, nm = pf.split("/", 1)
                return ARCHIVE / fo.replace("set", "") / nm
    return None


def main():
    if EMB_CACHE.exists():
        locate_v2._embedding_cache.update(pickle.load(open(EMB_CACHE, "rb")))
    topo = TopoGraphV2.load(str(TOPOMAP))
    gt = json.load(open(GT_FILE, encoding="utf-8"))

    nodes_objs = defaultdict(list); photos_of = {}
    for nid in topo.graph.nodes:
        dd = topo.graph.nodes[nid]
        if dd.get("ntype") == "photo":
            photos_of[nid] = dd.get("photos") or {}
        elif dd.get("ntype") == "object":
            nodes_objs[dd.get("photo_id")].append({
                "id": nid, "label": dd.get("label", ""), "ocr": dd.get("ocr_text", "") or "",
                "dir": _base_dir(dd.get("direction", ""))})
    map_photos = []
    for pid, objs in nodes_objs.items():
        bydir = defaultdict(list)
        for o in objs: bydir[o["dir"]].append(o)
        for bdir, os_ in bydir.items():
            map_photos.append({"pid": pid, "dir": bdir, "objs": os_,
                               "ocrset": {_norm(o["ocr"]) for o in os_ if o["ocr"]},
                               "img": _map_img(photos_of.get(pid, {}), bdir)})

    # OCR IDF
    N = len(map_photos); df = Counter()
    for mp in map_photos:
        for t in mp["ocrset"]: df[t] += 1
    idf = {t: log((N+1)/(d+1))+1 for t, d in df.items()}; DEF = log(N+1)+1
    def ocr_exact(q, m):
        sh = q & m
        return sum(idf.get(t, DEF) for t in sh)/sum(idf.get(t, DEF) for t in q) if (q and sh) else 0.0
    def node_plain(q_objs, r_objs):
        nq = len(q_objs)
        if not nq or not r_objs: return 0.0
        pairs = []
        for q in q_objs:
            for r in r_objs:
                s = _nsim(q["label"], r["label"])
                if s >= NODE_THR: pairs.append((s, q["id"], r["id"]))
        pairs.sort(reverse=True); uq, ur, tot = set(), set(), 0.0
        for s, qi, ri in pairs:
            if qi in uq or ri in ur: continue
            uq.add(qi); ur.add(ri); tot += s
        return tot/nq

    labels = set(o["label"] for mp in map_photos for o in mp["objs"])
    queries = []
    for key, v in gt.items():
        if v.get("ambiguous"): continue
        session, wp = key.rsplit("_wp", 1)
        for qdir, objs, _ in load_query_photos(session, wp):
            for o in objs: labels.add(o["label"])
            qimg = glob.glob(str(HERE/"testPDR"/session/"photos"/f"wp_{wp}_{qdir}_*.jpg"))
            queries.append((v["node_id"], objs, load_query_ocr(session, wp, qdir), qimg[0] if qimg else None))
    ensure_embeddings([l for l in labels if l])

    # 影像 embedding（含硬碟快取）
    cache = {}
    if VIS_CACHE.exists():
        try: cache = pickle.load(open(VIS_CACHE, "rb"))
        except Exception: cache = {}
    need = set()
    for mp in map_photos:
        if mp["img"] and str(mp["img"]) not in cache: need.add(str(mp["img"]))
    for (_, _, _, qi) in queries:
        if qi and qi not in cache: need.add(qi)
    if need:
        _load_clip()
        print(f"計算 {len(need)} 張影像 embedding...", flush=True)
        t0 = time.time()
        for i, p in enumerate(sorted(need)):
            try: cache[p] = _embed(p)
            except Exception as e: cache[p] = None
            if (i+1) % 50 == 0: print(f"  ...{i+1}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
        VIS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(cache, open(VIS_CACHE, "wb"))
    for mp in map_photos:
        mp["emb"] = cache.get(str(mp["img"])) if mp["img"] else None

    def mm(vals):  # per-query min-max 正規化到 [0,1]
        lo, hi = min(vals), max(vals)
        return [(x-lo)/(hi-lo) if hi > lo else 0.0 for x in vals]

    METHODS = ["Node+OCR(base)", "Visual-only", "Visual+OCR", "Visual+Node", "Visual+Node+OCR"]
    stat = {m: {"t1": 0, "t3": 0, "t5": 0, "n": 0} for m in METHODS}
    t0 = time.time(); used = 0
    for qi, (gt_node, q_objs, q_ocr, qimg) in enumerate(queries):
        qemb = cache.get(qimg) if qimg else None
        nodev, ocrv, visv = [], [], []
        for mp in map_photos:
            nodev.append(node_plain(q_objs, mp["objs"]))
            ocrv.append(ocr_exact(q_ocr, mp["ocrset"]))
            visv.append(float(np.dot(qemb, mp["emb"])) if (qemb is not None and mp["emb"] is not None) else 0.0)
        used += 1
        nn, no, nv = mm(nodev), mm(ocrv), mm(visv)
        pids = [mp["pid"] for mp in map_photos]
        combos = {
            "Node+OCR(base)": [nodev[i] + A_OCR*ocrv[i] for i in range(len(pids))],
            "Visual-only":    visv,
            "Visual+OCR":     [nv[i] + no[i] for i in range(len(pids))],
            "Visual+Node":    [nv[i] + nn[i] for i in range(len(pids))],
            "Visual+Node+OCR":[nv[i] + nn[i] + no[i] for i in range(len(pids))],
        }
        for m, sc in combos.items():
            order = [p for _, p in sorted(zip(sc, pids), key=lambda x: x[0], reverse=True)]
            s = stat[m]; s["n"] += 1
            if order[0] == gt_node: s["t1"] += 1
            if gt_node in order[:3]: s["t3"] += 1
            if gt_node in order[:5]: s["t5"] += 1
        if (qi+1) % 40 == 0: print(f"  eval ...{qi+1}/{len(queries)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n查詢 {used}\n")
    print(f"{'方法':<18}{'Top-1':>9}{'Top-3':>9}{'Top-5':>9}")
    print("-" * 48)
    for m in METHODS:
        s = stat[m]; n = max(s["n"], 1)
        print(f"{m:<18}{s['t1']/n*100:>8.1f}%{s['t3']/n*100:>8.1f}%{s['t5']/n*100:>8.1f}%")


if __name__ == "__main__":
    main()
