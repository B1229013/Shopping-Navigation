"""Sweep the DINOv2 : objects : text weights of locate_fusion on the image_only caches.

  python -m eval.weight_sweep --set lab1019 [--labels zh|bilingual] --data-root <backend>

The server scores every reference photo as
    W_V · minmax(DINOv2 cosine) + W_N · minmax(object score) + W_O · OCR score,
keeps each node's best photo, rescales the nodes to 0..2, adds the aisle-sign evidence
and (no sensors) ranks the nodes by that. Only the ratios W_N/W_V and W_O/W_V matter, so
W_V is fixed at 1 and both ratios run over RATIOS independently.

Step 1 computes the three terms per (photo, reference photo) once with locate_fusion's own
functions and caches them (components_<labels>.npz); step 2 reweights and ranks — every
combination in well under a second. The 5 : 0.5 : 0.5 row must reproduce the production
pick of every photo (checked against the image_only _photos.jsonl when labels=zh).

Over-fitting check: leave-one-walk-out — pick the best ratios on two walks, score the
third — reported next to the full-data table.

--labels bilingual uses the 「中文 English」 object names (eval/whatif.bilingual_labels).
Output: eval/results/weight_sweep_<set>_<labels>.md / .json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from eval import image_only as io

BACKEND = Path(__file__).resolve().parents[1]
RATIOS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
PROD = (5.0, 0.5, 0.5)


def greedy_score(sub, thr):
    """locate_fusion._node_score on a (query labels × reference labels) similarity matrix:
    pairs ≥ thr, highest first (ties: larger i, then larger j — Python's reverse tuple
    sort), one-to-one, summed and divided by the number of query labels."""
    nq = sub.shape[0]
    if nq == 0 or sub.shape[1] == 0:
        return 0.0
    qi, rj = np.nonzero(sub >= thr)
    if not len(qi):
        return 0.0
    vals = sub[qi, rj]
    order = np.lexsort((-rj, -qi, -vals))
    uq, ur, tot = set(), set(), 0.0
    for o in order:
        i, j = int(qi[o]), int(rj[o])
        if i in uq or j in ur:
            continue
        uq.add(i); ur.add(j)
        tot += float(vals[o])
    return tot / nq


class SemanticSim:
    """Label similarity from OpenAI text embeddings: 'embed' = cosine only (legacy
    locate_v2._embedding_sim), 'hybrid' = max(character similarity, cosine) (the server's
    _hybrid_max_sim with its embedding cache on)."""

    def __init__(self, mode, emb, ref_labels):
        self.mode, self.emb = mode, emb
        self.ref_labels = ref_labels
        self.R = emb.matrix(ref_labels)

    def matrix(self, q_labels):
        S = emb_S = self.emb.matrix(q_labels) @ self.R.T if q_labels else np.zeros((0, len(self.ref_labels)))
        if self.mode == "hybrid" and q_labels:
            from server.visual_localization import _semantic_similarity
            char = np.array([[_semantic_similarity(q, r) for r in self.ref_labels] for q in q_labels])
            S = np.maximum(emb_S, char)
        return S


def components(items, perc_all, dino, ref_map, labels_fn, path, sim=None):
    """{vis, nod, ocr}: (photos × reference photos); owner: node of each reference photo;
    aisle: (photos × nodes). sim: None = the server's own _node_score (character
    similarity, no embedding cache), else a SemanticSim."""
    if path.is_file():
        z = np.load(path, allow_pickle=False)
        if list(z["ids"]) == [it["id"] for it in items]:
            return {k: z[k] for k in z.files}
    from server import aisle_signs, config as C, locate_fusion as LF, photo_quality
    from server.ocr import OCRResult
    embs = LF._reference_embeddings(ref_map, LF._load_sidecar(C.DINO_SIDECAR_PATH))
    idx = LF._index_for(ref_map, embs)
    E = idx["entries"]
    if sim is not None:
        col = {l: j for j, l in enumerate(sim.ref_labels)}
        entry_cols = [np.array([col[l] for l in e["labels"]], dtype=int) for e in E]
    nodes = list(ref_map.photos)          # map order: fuse_v2 breaks ties in this order
    assert {e["nid"] for e in E} == set(nodes), "every node needs a reference photo"
    R = np.stack([e["emb"] if e["emb"] is not None else np.zeros(768, np.float32) for e in E])
    has = np.array([e["emb"] is not None for e in E])
    vis = np.zeros((len(items), len(E)), np.float32)
    nod = np.zeros_like(vis)
    ocr = np.zeros_like(vis)
    aisle = np.zeros((len(items), len(nodes)), np.float32)
    t0 = time.perf_counter()
    for i, it in enumerate(items):
        p = labels_fn(perc_all[it["id"]])
        q = np.asarray(dino[it["id"]]["emb"], np.float32)
        vis[i] = np.where(has, R @ q, 0.0)
        labels = [d["label"] for d in p["detections"]]
        q_ocr = {LF._norm_ocr(o["text"]) for o in p["ocr"]} - {""}
        denom = sum(idx["idf"].get(t, idx["def_idf"]) for t in q_ocr)
        S = sim.matrix(labels) if sim is not None else None
        for k, e in enumerate(E):
            nod[i, k] = (LF._node_score(labels, e["labels"]) if S is None
                         else greedy_score(S[:, entry_cols[k]], LF.NODE_SIM_THRESHOLD))
            sh = q_ocr & e["ocrset"]
            if sh:
                ocr[i, k] = sum(idx["idf"].get(t, idx["def_idf"]) for t in sh) / denom
        w, h = p["size"]
        # as server._build_localization_items: occluders are not part of the map
        items_n = [{"label": d["label"], "box": [d["box"][0] / w, d["box"][1] / h, d["box"][2] / w, d["box"][3] / h]}
                   for d in p["detections"] if not photo_quality.is_occluder(d["label"])]
        nums = dict(aisle_signs.numbers_in_items(items_n))
        for n_, a in aisle_signs.numbers_from_ocr([OCRResult(o["text"], o["confidence"], o["bbox"]) for o in p["ocr"]],
                                                  w, h).items():
            nums[n_] = max(nums.get(n_, 0.0), a)
        if nums and C.NEO4J_PLACE in C.AISLE_SIGN_PLACES:
            for n_, v in (aisle_signs.sign_evidence(nums, ref_map, C.NEO4J_PLACE) or {}).items():
                aisle[i, nodes.index(n_)] = v
        print(f"\r  components {i + 1}/{len(items)} {time.perf_counter() - t0:5.0f}s", end="", file=sys.stderr)
    print(file=sys.stderr)
    out = {"vis": vis, "nod": nod, "ocr": ocr, "aisle": aisle,
           "owner": np.array([nodes.index(e["nid"]) for e in E]), "nodes": np.array(nodes),
           "ids": np.array([it["id"] for it in items])}
    np.savez_compressed(path, **out)
    return out


def _mm(a):
    lo, hi = a.min(1, keepdims=True), a.max(1, keepdims=True)
    return np.where(hi > lo, (a - lo) / np.where(hi > lo, hi - lo, 1), 0.0)


def rank(comp, wv, wn, wo):
    """(photos × nodes) evidence and the node order per photo, as the server computes them."""
    fused = wv * _mm(comp["vis"]) + wn * _mm(comp["nod"]) + wo * comp["ocr"]
    n_nodes = len(comp["nodes"])
    node = np.full((fused.shape[0], n_nodes), -np.inf)
    for j in range(n_nodes):
        cols = comp["owner"] == j
        node[:, j] = fused[:, cols].max(1)
    lo, hi = node.min(1, keepdims=True), node.max(1, keepdims=True)
    ev = np.where(hi - lo > 1e-9, 2.0 * (node - lo) / np.where(hi - lo > 1e-9, hi - lo, 1), 0.0) + comp["aisle"]
    # fuse_v2 sorts the posterior (∝ exp(k·ev)) with a stable sort over nodes in map order
    order = np.argsort(-ev, axis=1, kind="stable")
    return ev, order


def score(order, comp, items, hops, sel=None):
    nodes = comp["nodes"]
    rows = [i for i in range(len(items)) if sel is None or sel(items[i])]
    if not rows:
        return None
    t1 = t5 = h1 = 0
    for i in rows:
        ok = items[i]["ok"]
        picks = [int(nodes[j]) for j in order[i, :5]]
        t1 += picks[0] in ok
        t5 += any(n in ok for n in picks)
        h1 += min(hops[o].get(picks[0], 99) for o in ok) <= 1
    n = len(rows)
    return {"n": n, "top1": round(100 * t1 / n, 1), "top5": round(100 * t5 / n, 1), "within_1_hop": round(100 * h1 / n, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=io.SETS)
    ap.add_argument("--labels", choices=("zh", "bilingual"), default="zh")
    ap.add_argument("--sim", choices=("char", "embed", "hybrid"), default="char",
                    help="object-label similarity: char = the server as deployed (no embedding cache); "
                         "embed = OpenAI text-embedding cosine (legacy); hybrid = max of both (server with its "
                         "embedding cache on). embed/hybrid fetch missing label vectors once (paid, cached)")
    ap.add_argument("--data-root", default=str(BACKEND))
    ap.add_argument("--cache")
    args = ap.parse_args(argv)

    import logging
    logging.getLogger().setLevel(logging.WARNING)
    cache = Path(args.cache) if args.cache else BACKEND / "output" / "eval" / "image_only" / args.set
    ref_map, _ = io.load_map("snapshot", cache)
    items = io.load_items(args.set, ref_map, Path(args.data_root))
    perc_all = io._load_json(cache / "perception_vlm.json")
    dino = io._load_json(cache / "dino.json")
    hops = io.hop_table(ref_map)
    if args.labels == "bilingual":
        from eval.whatif import bilingual_labels as labels_fn
    else:
        labels_fn = lambda p: p
    sim = None
    tag = args.labels if args.sim == "char" else f"{args.labels}_{args.sim}"
    if args.sim != "char":
        from eval.label_embeddings import LabelEmbeddings
        from server import config as C, locate_fusion as LF
        from server import visual_localization as VL
        embs = LF._reference_embeddings(ref_map, LF._load_sidecar(C.DINO_SIDECAR_PATH))
        ref_labels = sorted({l for e in LF._index_for(ref_map, embs)["entries"] for l in e["labels"]})
        q_labels = {d["label"] for it in items for d in labels_fn(perc_all[it["id"]])["detections"]}
        emb = LabelEmbeddings(cache / "label_embeddings.npz")
        n_new = emb.ensure(ref_labels + sorted(q_labels))
        print(f"label vectors: {len(ref_labels)} map + {len(q_labels)} photo labels; fetched {n_new} new "
              f"({emb.tokens} tokens)", file=sys.stderr)
        sim = SemanticSim(args.sim, emb, ref_labels)
        # the vectorized object score must equal the server's _node_score on the same vectors
        VL._emb_cache.clear()
        VL._emb_cache.update({l: emb.matrix([l])[0] for l in set(ref_labels) | q_labels})
        real = LF._embedding_sim
        if args.sim == "embed":
            LF._embedding_sim = lambda a, b: VL._cosine_sim(VL._emb_cache[a], VL._emb_cache[b])
        E = LF._index_for(ref_map, embs)["entries"]
        col = {l: j for j, l in enumerate(ref_labels)}
        worst = 0.0
        for it in items[:3]:
            labs = [d["label"] for d in labels_fn(perc_all[it["id"]])["detections"]]
            S = sim.matrix(labs)
            for e in E[::7]:
                worst = max(worst, abs(LF._node_score(labs, e["labels"]) -
                                       greedy_score(S[:, [col[l] for l in e["labels"]]], LF.NODE_SIM_THRESHOLD)))
        LF._embedding_sim = real
        VL._emb_cache.clear()
        if worst > 1e-4:
            sys.exit(f"vectorized object score differs from _node_score by {worst:.5f}")
        print(f"check: object score = server _node_score on the same vectors (max diff {worst:.1e})", file=sys.stderr)
    comp = components(items, perc_all, dino, ref_map, labels_fn, cache / f"components_{tag}.npz", sim)

    # the production weights must give the production picks
    _, order = rank(comp, *PROD)
    if args.labels == "zh" and args.sim == "char":
        base = {d["id"]: d["results"]["prod"][0]["pred"] for d in map(
            json.loads, (BACKEND / "eval" / "results" / f"image_only_{args.set}_vlm_photos.jsonl").read_text(encoding="utf-8").splitlines())}
        diff = [it["id"] for i, it in enumerate(items) if int(comp["nodes"][order[i, 0]]) != base[it["id"]]]
        if diff:
            sys.exit(f"5:0.5:0.5 differs from production on {len(diff)} photos, e.g. {diff[:3]} — not sweeping")
        print(f"check: 5:0.5:0.5 reproduces the production pick of all {len(items)} photos", file=sys.stderr)

    walks = sorted({it["id"].split("/")[0] for it in items})
    conds = (("all", None), ("at_node", lambda it: it["condition"] == "at_node"),
             ("between", lambda it: it["condition"] == "between"))
    rows = []
    for rn in RATIOS:
        for ro in RATIOS:
            _, order = rank(comp, 1.0, rn, ro)
            r = {"w_objects": rn, "w_text": ro}
            for name, sel in conds:
                r[name] = score(order, comp, items, hops, sel)
            for w in walks:
                r[f"walk{w}"] = score(order, comp, items, hops, lambda it, w=w: it["id"].split("/")[0] == w)
            rows.append(r)
    _, order = rank(comp, *PROD)
    prod = {name: score(order, comp, items, hops, sel) for name, sel in conds}

    # leave-one-walk-out: choose on two walks (photo-weighted Top-1), report the third
    cv = []
    for w in walks:
        train = [x for x in walks if x != w]
        def tr(r):
            n = sum(r[f"walk{x}"]["n"] for x in train)
            return sum(r[f"walk{x}"]["top1"] * r[f"walk{x}"]["n"] for x in train) / n
        best = max(rows, key=lambda r: (tr(r), -abs(r["w_objects"] - 0.1) - abs(r["w_text"] - 0.1)))
        cv.append({"held_out_walk": w, "picked": [1, best["w_objects"], best["w_text"]],
                   "train_top1": round(tr(best), 1), "held_out": best[f"walk{w}"]})
    n_all = sum(c["held_out"]["n"] for c in cv)
    cv_top1 = round(sum(c["held_out"]["top1"] * c["held_out"]["n"] for c in cv) / n_all, 1)
    best_all = max(rows, key=lambda r: r["all"]["top1"])

    def cell(r, k="top1"):
        return r["all"][k]
    sim_name = {"char": "字元相似度（現行伺服器）", "embed": "語意向量 cosine（legacy 做法）",
                "hybrid": "字元與語意向量取較大值（伺服器開啟向量快取時）"}[args.sim]
    L = [f"# DINOv2 : 物件 : 文字 權重掃描（{args.set}，物件名稱 {('中文' if args.labels == 'zh' else '中文 English')}，"
         f"相似度 {sim_name}，不帶感測器）", "",
         f"- {len(items)} 張；DINOv2 固定 1，物件和文字比例各取 {', '.join(f'{x:g}' for x in RATIOS)}（共 {len(rows)} 組）；走道號碼維持 ×1",
         f"- 正式的 5 : 0.5 : 0.5（＝1 : 0.1 : 0.1）：Top-1 {prod['all']['top1']}%（節點 {prod['at_node']['top1']}%、"
         f"節點之間 {prod['between']['top1']}%），Top-5 {prod['all']['top5']}%"
         + ("；已確認和伺服器每張照片的結果相同" if (args.labels, args.sim) == ("zh", "char") else ""),
         f"- 全部資料最好的一組：1 : {best_all['w_objects']:g} : {best_all['w_text']:g} → Top-1 {best_all['all']['top1']}%"
         f"（節點 {best_all['at_node']['top1']}%、節點之間 {best_all['between']['top1']}%），Top-5 {best_all['all']['top5']}%",
         f"- **交叉驗證**（兩趟挑比例、第三趟驗證）：Top-1 {cv_top1}%（正式比例 {prod['all']['top1']}%）；"
         + "；".join(f"驗證第 {c['held_out_walk']} 趟挑到 1 : {c['picked'][1]:g} : {c['picked'][2]:g} → {c['held_out']['top1']}%" for c in cv),
         "", "## Top-1（%）：列＝物件比例，欄＝文字比例", "",
         "| 物件 \\ 文字 | " + " | ".join(f"{x:g}" for x in RATIOS) + " |", "|---|" + "---|" * len(RATIOS)]
    for rn in RATIOS:
        L.append(f"| **{rn:g}** | " + " | ".join(
            f"{cell(r)}" for r in rows if r["w_objects"] == rn) + " |")
    for name, label in (("at_node", "節點"), ("between", "節點之間")):
        L += ["", f"## {label}照片 Top-1（%）", "", "| 物件 \\ 文字 | " + " | ".join(f"{x:g}" for x in RATIOS) + " |",
              "|---|" + "---|" * len(RATIOS)]
        for rn in RATIOS:
            L.append(f"| **{rn:g}** | " + " | ".join(f"{r[name]['top1']}" for r in rows if r["w_objects"] == rn) + " |")
    L += ["", "## 之前測過的 (k:1:1)，k = 1～8", "", "| DINOv2 : 物件 : 文字 | Top-1 | 節點 | 節點之間 | Top-5 |", "|---|---|---|---|---|"]
    for k in range(1, 9):
        _, order = rank(comp, float(k), 1.0, 1.0)
        s = {name: score(order, comp, items, hops, sel) for name, sel in conds}
        L.append(f"| {k} : 1 : 1 | {s['all']['top1']}% | {s['at_node']['top1']}% | {s['between']['top1']}% | {s['all']['top5']}% |")
    out = BACKEND / "eval" / "results" / f"weight_sweep_{args.set}_{tag}"
    Path(f"{out}.json").write_text(json.dumps({"labels": args.labels, "sim": args.sim, "ratios": RATIOS, "prod": prod, "rows": rows,
                                               "cv": cv, "cv_top1": cv_top1}, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(f"{out}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
