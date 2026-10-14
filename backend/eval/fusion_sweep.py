"""Sweep DINOv2 : sensor : grid fusion weights on the reference map (leave-one-photo-out).

Each of the reference DirPhotos is held out in turn and localized against the
remaining ones, so the query view itself is never in the database — the closest
remaining views are the same node's other directions or a neighbour's. That
makes it a stand-in for "photo taken between nodes".

  dino   — cosine vs every other DirPhoto using the phone's own reference
           embeddings (APPNAV_ios/Models/dino_ref_embeddings.bin + dino_ref_meta.json),
           max per node, top-10 nodes kept (same as DINOLocalizationService)
  grid   — server's match_by_grid with the held-out photo's own DB objects as
           the "detections" (optionally with object dropout to mimic a single
           live VLM pass instead of the enriched reference set)
  sensor — true node position + 2-D Gaussian error of σ metres (simulated drift)

Fusion is the same as server/loc_fusion.py (verified against fuse_candidates).

Usage (from backend/):
  python -m eval.fusion_sweep collect --out /tmp/fusion_cache.json [--dropout 0.5] [--workers 8]
  python -m eval.fusion_sweep sweep   --cache /tmp/fusion_cache.json [--csv out.csv]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DINO_TOPK = 10
GRID_TOPK = 10
WEIGHT_STEPS = [0, 1, 2, 3, 4, 5, 6, 8, 10]
SIGMAS = [None, 1.0, 2.0, 4.0, 8.0]   # None = phone not calibrated yet → no sensor signal
DRAWS = 5
SENSOR_DECAY_M = 8.0

_REF_MAP = None


def _load_ref_map():
    from server.config import NEO4J_PLACE
    from server.neo4j_client import get_neo4j
    neo4j = get_neo4j()
    if neo4j is None:
        sys.exit("Neo4j not configured — fill NEO4J_* in the repo-root .env")
    return neo4j.load_reference_map(NEO4J_PLACE)


def _without_photo(ref_map, nid, photo_file):
    """Shallow copy of ref_map with one DirPhoto removed."""
    from dataclasses import replace
    from server.neo4j_client import RefMap
    photos = dict(ref_map.photos)
    node = photos[nid]
    photos[nid] = replace(node, directional_photos=[d for d in node.directional_photos
                                                     if d.photo_file != photo_file])
    return RefMap(place=ref_map.place, schema_version=ref_map.schema_version, photos=photos,
                  walkway_edges=ref_map.walkway_edges, adjacent_map=ref_map.adjacent_map)


def _grid_scores(job):
    nid, photo_file, dropout, seed, grid_n, grid_weights = job
    from server.visual_localization import apply_weight_overrides, match_by_grid
    if grid_weights:
        apply_weight_overrides(grid_weights)
    ref_map = _without_photo(_REF_MAP, nid, photo_file)
    q = next(d for d in _REF_MAP.photos[nid].directional_photos if d.photo_file == photo_file)
    rng = random.Random(seed)
    objs = [o for o in q.objects if o.bbox and rng.random() >= dropout]
    items = [{"label": o.label, "box": o.bbox} for o in objs]
    ocr = [{"text": o.ocr_text, "bbox": [[o.bbox[0], o.bbox[1]], [o.bbox[2], o.bbox[3]]]}
           for o in objs if o.ocr_text]
    res = match_by_grid(items, ref_map, top_k=GRID_TOPK, ocr_items=ocr, grid_n=grid_n)
    return photo_file, {str(n): s for n, s, _ in res}, len(items)


def _init_worker(ref_map):
    global _REF_MAP
    _REF_MAP = ref_map


def collect(args):
    from server.config import GRID_N
    ref_map = _load_ref_map()
    models = Path(__file__).resolve().parents[2] / "APPNAV_ios" / "Models"
    meta = json.loads((models / "dino_ref_meta.json").read_text())
    raw = np.fromfile(models / "dino_ref_embeddings.bin", dtype=np.float32).reshape(len(meta), -1)
    side = {e["file"]: raw[i] for i, e in enumerate(meta)}

    dps = [(nid, dp.photo_file) for nid, n in sorted(ref_map.photos.items())
           for dp in n.directional_photos]
    names = [os.path.basename(f) for _, f in dps]
    emb = np.stack([np.asarray(side[n], dtype=np.float32) for n in names])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sims = emb @ emb.T
    owner = np.array([nid for nid, _ in dps])

    jobs = [(nid, f, args.dropout, i, GRID_N, args.grid_weights) for i, (nid, f) in enumerate(dps)]
    grid = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(ref_map,)) as ex:
        for k, (f, scores, n_items) in enumerate(ex.map(_grid_scores, jobs), 1):
            grid[f] = (scores, n_items)
            print(f"\r grid {k}/{len(jobs)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    queries = []
    for i, (nid, f) in enumerate(dps):
        per_node = {}
        for j in np.argsort(-sims[i]):
            if j == i:
                continue
            per_node.setdefault(int(owner[j]), float(sims[i, j]))
            if len(per_node) >= DINO_TOPK:
                break
        queries.append({"nid": nid, "photo": f, "set": f.split("/")[0],
                        "dino": {str(k): v for k, v in per_node.items()},
                        "grid": grid[f][0], "n_items": grid[f][1]})

    out = {
        "place": ref_map.place, "grid_n": GRID_N, "dropout": args.dropout,
        "grid_weights": args.grid_weights,
        "nodes": {str(n): {"x": p.pdr_x, "y": p.pdr_y, "nbrs": list(p.neighbor_nids)}
                  for n, p in ref_map.photos.items()},
        "queries": queries,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {len(queries)} queries → {args.out}")


# ── sweep ──────────────────────────────────────────────────────────────────

def _prepare(cache):
    node_ids = sorted(int(n) for n in cache["nodes"])
    idx = {n: i for i, n in enumerate(node_ids)}
    xy = np.array([[cache["nodes"][str(n)]["x"], cache["nodes"][str(n)]["y"]] for n in node_ids])
    Q, N = len(cache["queries"]), len(node_ids)
    G = np.zeros((Q, N)); D = np.zeros((Q, N))
    gmask = np.zeros((Q, N), bool); dmask = np.zeros((Q, N), bool)
    truth = np.zeros(Q, int)
    txy = np.zeros((Q, 2))
    for q, e in enumerate(cache["queries"]):
        truth[q] = idx[e["nid"]]
        # real photos can be between nodes: errors are measured to where it was taken
        txy[q] = e.get("truth_xy") or xy[truth[q]]
        for n, s in e["grid"].items():
            G[q, idx[int(n)]] = min(max(s, 0.0), 1.0); gmask[q, idx[int(n)]] = True
        vals = {idx[int(n)]: s for n, s in e["dino"].items()}
        lo, hi = min(vals.values()), max(vals.values())
        for i, s in vals.items():
            D[q, i] = 1.0 if hi - lo < 1e-9 else (s - lo) / (hi - lo)
            dmask[q, i] = True
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    nbr = np.zeros((N, N), bool)
    for n in node_ids:
        for m in cache["nodes"][str(n)]["nbrs"]:
            if m in idx:
                nbr[idx[n], idx[m]] = True
    return node_ids, xy, G, D, gmask, dmask, truth, dist, nbr, txy


def _sensor_scores(xy, txy, sigma, draw):
    rng = np.random.default_rng(1000 * draw + int(sigma * 10))
    pos = txy + rng.normal(0, sigma, size=(len(txy), 2))
    d = np.linalg.norm(pos[:, None, :] - xy[None, :, :], axis=2)
    return np.exp(-d / SENSOR_DECAY_M)


def _evaluate(G, D, S, gmask, dmask, truth, dist, nbr, w, xy=None, txy=None):
    """w = (dino, sensor, grid); each a scalar or a per-query array (adaptive policies)."""
    wd, ws, wg = (np.broadcast_to(np.asarray(x, float), (len(truth),))[:, None] for x in w)
    use_d, use_s = bool(np.any(wd > 0)), S is not None and bool(np.any(ws > 0))
    if not (use_d or use_s):
        score, cand = G.copy(), gmask
    else:
        total = wg + (wd if use_d else 0) + (ws if use_s else 0)
        score = (wg * G + (wd * D if use_d else 0) + (ws * S if use_s else 0)) / total
        cand = gmask | dmask if use_d else gmask
    score = np.where(cand, score, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")
    top1 = order[:, 0]
    rows = np.arange(len(truth))
    valid = np.isfinite(score[rows, top1])
    hit1 = valid & (top1 == truth)
    near = valid & ((top1 == truth) | nbr[truth, top1])
    top5 = np.array([truth[q] in order[q, :5][np.isfinite(score[q, order[q, :5]])]
                     for q in rows])
    err = (np.where(valid, dist[truth, top1], np.nan) if txy is None or xy is None else
           np.where(valid, np.linalg.norm(xy[top1] - txy, axis=1), np.nan))
    if xy is None:
        return hit1, near, top5, err
    # position between nodes: score-weighted mean of the top-3, weights relative to the 4th
    top = order[:, :4]
    sc = np.take_along_axis(score, top, axis=1)
    sc = np.where(np.isfinite(sc), sc, np.nan)
    floor = np.nan_to_num(sc[:, 3], nan=0.0)
    wts = np.clip(np.nan_to_num(sc[:, :3] - floor[:, None], nan=0.0), 0, None)
    wts[wts.sum(1) == 0, 0] = 1.0
    est = (wts[:, :, None] * xy[top[:, :3]]).sum(1) / wts.sum(1, keepdims=True)
    err_interp = np.where(valid, np.linalg.norm(est - (xy[truth] if txy is None else txy), axis=1), np.nan)
    return hit1, near, top5, err, err_interp


def sweep(args):
    cache = json.loads(Path(args.cache).read_text())
    node_ids, xy, G, D, gmask, dmask, truth, dist, nbr, txy = _prepare(cache)
    sets = np.array([e["set"] for e in cache["queries"]])
    combos = sorted({tuple(c) for c in itertools.product(WEIGHT_STEPS, repeat=3)
                     if c[2] + c[0] + c[1] > 0})

    rows = []
    for sigma in SIGMAS:
        draws = [None] if sigma is None else [_sensor_scores(xy, txy, sigma, d) for d in range(DRAWS)]
        for w in combos:
            if sigma is None and w[1] > 0:
                continue      # no sensor signal: sensor weight is irrelevant
            per = [_evaluate(G, D, S, gmask, dmask, truth, dist, nbr, w, xy, txy) for S in draws]
            hit1 = np.mean([p[0] for p in per], axis=0)
            near = np.mean([p[1] for p in per], axis=0)
            top5 = np.mean([p[2] for p in per], axis=0)
            err = np.nanmean([p[3] for p in per], axis=0)
            rows.append({"sigma": sigma, "w": w, "hit1": hit1, "near": near, "top5": top5, "err": err})

    def fmt(r, mask=slice(None)):
        return (f"top1 {100*r['hit1'][mask].mean():5.1f}%  node/nbr {100*r['near'][mask].mean():5.1f}%  "
                f"top5 {100*r['top5'][mask].mean():5.1f}%  err {np.nanmean(r['err'][mask]):4.1f} m")

    print(f"{len(cache['queries'])} held-out photos, {len(node_ids)} nodes, "
          f"grid_n={cache['grid_n']}, object dropout={cache['dropout']}\n")
    for sigma in SIGMAS:
        rs = [r for r in rows if r["sigma"] == sigma]
        best = max(rs, key=lambda r: (r["hit1"].mean(), r["near"].mean()))
        label = "no sensor" if sigma is None else f"sensor σ={sigma:g} m"
        print(f"── {label} ──")
        for name, w in [("grid only", (0, 0, 1)), ("dino only", (1, 0, 0)), ("5:1:1", (5, 1, 1))]:
            r = next((r for r in rs if r["w"] == w), None)
            if r:
                print(f"  {name:10s} {w!s:12s} {fmt(r)}")
        print(f"  {'best':10s} {best['w']!s:12s} {fmt(best)}")
        # leave-one-set-out: pick weights on the other sets, score on the held-out set
        if len(set(sets)) > 1:
            hits = []
            for s in sorted(set(sets)):
                train, test = sets != s, sets == s
                pick = max(rs, key=lambda r: (r["hit1"][train].mean(), r["near"][train].mean()))
                hits.append((s, pick["w"], 100 * pick["hit1"][test].mean(), test.sum()))
            cv = sum(h * n for _, _, h, n in hits) / sum(n for *_, n in hits)
            print(f"  {'CV':10s} {'':12s} top1 {cv:5.1f}%  picks: " +
                  ", ".join(f"{s}→{w}" for s, w, _, _ in hits))
        print()

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("sigma,w_dino,w_sensor,w_grid,top1,node_or_nbr,top5,err_m\n")
            for r in rows:
                f.write(f"{'' if r['sigma'] is None else r['sigma']},{r['w'][0]},{r['w'][1]},{r['w'][2]},"
                        f"{r['hit1'].mean():.4f},{r['near'].mean():.4f},{r['top5'].mean():.4f},"
                        f"{np.nanmean(r['err']):.3f}\n")
        print(f"all {len(rows)} rows → {args.csv}")


def policy(args):
    """Adaptive weights: w_sensor = min(cap, c/σ), w_grid = grid·min(1, n_objects/n_full)."""
    from server.loc_fusion import adaptive_weights, FusionWeights
    cache = json.loads(Path(args.cache).read_text())
    node_ids, xy, G, D, gmask, dmask, truth, dist, nbr, txy = _prepare(cache)
    n_items = np.array([e["n_items"] for e in cache["queries"]])
    sigmas = [1.0, 2.0, 4.0, 8.0]
    S_by = {sg: [_sensor_scores(xy, txy, sg, d) for d in range(DRAWS)] for sg in sigmas}
    base = FusionWeights(dino=4.0, sensor=0.0, grid=1.0)

    def run(c, cap, n_full, fixed=None):
        res = {}
        for sg in sigmas:
            if fixed:
                w = fixed
            else:
                ws = [adaptive_weights(base, sg, int(n), c=c, cap=cap, n_full=n_full) for n in n_items]
                w = (np.array([x.dino for x in ws]), np.array([x.sensor for x in ws]),
                     np.array([x.grid for x in ws]))
            per = [_evaluate(G, D, S, gmask, dmask, truth, dist, nbr, w, xy, txy) for S in S_by[sg]]
            res[sg] = tuple(np.mean([np.nanmean(p[i]) for p in per]) for i in range(5))
        return res

    def line(name, res):
        cells = "  ".join(f"σ{sg:g}: {100*r[0]:4.1f}/{100*r[1]:4.1f}/{r[3]:3.1f}m/{r[4]:3.1f}m"
                          for sg, r in res.items())
        avg = np.mean([r[0] for r in res.values()])
        print(f"  {name:28s} {cells}   mean top1 {100*avg:4.1f}%")
        return avg

    print(f"cache {args.cache}: dropout={cache['dropout']} grid_weights={cache.get('grid_weights')!r}")
    print("  cells = top1% / node-or-nbr% / node error / interpolated-position error")
    line("fixed 5:1:1", run(0, 0, 0, fixed=(5, 1, 1)))
    line("fixed 4:0:1", run(0, 0, 0, fixed=(4, 0, 1)))
    results = []
    for c in [5, 10, 20, 40, 80]:
        for cap in [10, 20, 40]:
            for n_full in [0, 5, 10, 15, 20]:
                res = run(c, cap, n_full)
                results.append((np.mean([r[0] for r in res.values()]), c, cap, n_full, res))
    results.sort(key=lambda r: -r[0])
    print("  best adaptive settings (c, cap, n_full):")
    for avg, c, cap, n_full, res in results[:8]:
        line(f"c={c} cap={cap} n_full={n_full}", res)


def compare(args):
    """Compare caches built with different --grid-weights: grid alone, DINOv2:grid 4:1,
    and the full adaptive policy (sensor c/σ, DINOv2 gap scaling) averaged over σ."""
    from server.loc_fusion import dino_gap
    print(f"{'cache':12s} {'grid only':>26s} {'4:0:1 (no sensor)':>26s} {'adaptive, mean σ1–8':>22s}")
    print(f"{'':12s} {'top1 / nbr / top5 / err':>26s} {'top1 / nbr / err':>26s} {'top1 / nbr':>22s}")
    for path in args.caches:
        cache = json.loads(Path(path).read_text())
        node_ids, xy, G, D, gmask, dmask, truth, dist, nbr, txy = _prepare(cache)
        gaps = np.array([dino_gap({int(k): v for k, v in e["dino"].items()}) or 0.0
                         for e in cache["queries"]])
        wd = 4 * np.clip(gaps / 0.05, 0, 1)
        g = _evaluate(G, D, None, gmask, dmask, truth, dist, nbr, (0, 0, 1), xy, txy)
        f = _evaluate(G, D, None, gmask, dmask, truth, dist, nbr, (4, 0, 1), xy, txy)
        ad = []
        for sg in [1.0, 2.0, 4.0, 8.0]:
            for d in range(DRAWS):
                ad.append(_evaluate(G, D, _sensor_scores(xy, txy, sg, d), gmask, dmask, truth, dist, nbr,
                                    (wd, min(40, 40 / sg), 1.0), xy, txy))
        name = Path(path).stem
        print(f"{name:12s} {100*g[0].mean():5.1f} / {100*g[1].mean():4.1f} / {100*g[2].mean():4.1f} / {np.nanmean(g[3]):4.1f}m"
              f"     {100*f[0].mean():5.1f} / {100*f[1].mean():4.1f} / {np.nanmean(f[3]):4.1f}m"
              f"      {100*np.mean([a[0].mean() for a in ad]):5.1f} / {100*np.mean([a[1].mean() for a in ad]):4.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--dropout", type=float, default=0.0)
    c.add_argument("--grid-weights", default="",
                   help='grid weight overrides, e.g. "landmark=6,rarity_mode=idf"')
    c.add_argument("--workers", type=int, default=os.cpu_count())
    s = sub.add_parser("sweep")
    s.add_argument("--cache", required=True)
    s.add_argument("--csv")
    p = sub.add_parser("policy")
    p.add_argument("--cache", required=True)
    m = sub.add_parser("compare")
    m.add_argument("caches", nargs="+")
    args = ap.parse_args()
    {"collect": collect, "sweep": sweep, "policy": policy, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
