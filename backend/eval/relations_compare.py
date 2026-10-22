"""Two versions of object-to-object relations in the grid matcher, on the same photos.

Replays the detections of a saved image_only run (default lab1019, 376 photos) through the
real visual_localization.match_by_grid, with the map's ADJACENT pairs (map_v2 /
rebuild_objects_v2, 3×3 grid cells) feeding the grid matcher's spatial bonus:

  沒有關係        adjacent_map empty (the server before ADJACENT was loaded)
  周 原樣         APPNAV_ios fe4c851 as committed: cell_pairs "dir" read as object a's own
                  direction and the b → a lookup not inverted
  方向修正        neo4j_client.adjacent_relation / add_adjacent (this branch): "dir" is where
                  b is from a, so it is inverted to a-relative-to-b like the photo's pairs,
                  and b → a is stored inverted

each at a few SPATIAL_BONUS_COEFF values. Two paths use the grid matcher:

  格子比對        the grid ranking alone (a map without DINOv2 vectors)
  手機 DINOv2     loc_fusion.image_evidence(grid top-10, DINOv2 top-10, aisle) → fuse_v2 with
                  the phone k — what the server does when the phone (iOS) sends dino_topk.
                  DINOv2 here is the saved server DINOv2 cosine per node (best direction).

The server-side DINOv2 path (Android) never calls the grid matcher, so ADJACENT does not
change it; the bbox relation term for that path is eval/object_relations_eval.py.

Label similarity is the server's _hybrid_max_sim; without OPENAI_API_KEY the label vectors
come from a local multilingual sentence model (--local-model), as an approximation.

  python -m eval.relations_compare [--photos …_photos.jsonl] [--map neo4j|refmap.pkl]
                                   [--adjacent neo4j|adjacent.json] [--local-model NAME] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

VARIANTS = [("沒有關係", None, 0.0), ("周 原樣", "asis", 0.10), ("周 原樣", "asis", 0.30),
            ("方向修正", "fixed", 0.10), ("方向修正", "fixed", 0.30), ("方向修正", "fixed", 1.00)]

_state: dict = {}


def asis_map(rows):
    """fe4c851's loader + its `get((a, b)) or get((b, a))` lookup, reproduced."""
    out: dict = {}
    for r in rows:
        try:
            pairs = json.loads(r["cell_pairs"]) if isinstance(r["cell_pairs"], str) else (r["cell_pairs"] or [])
        except (json.JSONDecodeError, TypeError):
            pairs = []
        dirs = set()
        for p in pairs or []:
            d = p.get("dir", "")
            for w in ("above", "below", "left", "right"):
                if w in d:
                    dirs.add(w)
        v = "above" if "above" in dirs else ("below" if "below" in dirs else "same_row")
        h = "left" if "left" in dirs else ("right" if "right" in dirs else "same_col")
        out.setdefault((r["nid"], r["slot"]), {})[(r["a_label"], r["b_label"])] = {
            "v_rel": v, "h_rel": h, "overlap": r["overlap"] or False,
            "depth_diff": r["depth_diff"] or "same_level"}
    for rels in out.values():                     # the uninverted b → a fallback
        for (a, b), rel in list(rels.items()):
            rels.setdefault((b, a), rel)
    return out


def fixed_map(rows):
    from server.neo4j_client import add_adjacent, adjacent_relation
    out: dict = {}
    for r in rows:
        rel = {**adjacent_relation(r["cell_pairs"]), "overlap": bool(r["overlap"]),
               "depth_diff": r["depth_diff"] or "same_level"}
        add_adjacent(out.setdefault((r["nid"], r["slot"]), {}), r["a_label"] or "", r["b_label"] or "", rel)
    return out


def _init(map_arg, rows, label_vecs):
    from server import visual_localization as VL
    ref_map = load_map(map_arg)
    if label_vecs:
        import numpy as np
        VL._emb_cache.update({l: np.asarray(v, dtype=np.float32) for l, v in label_vecs.items()})
        VL._emb_loaded = VL._emb_on = True
    _state.update(ref_map=ref_map, maps={None: {}, "asis": asis_map(rows), "fixed": fixed_map(rows)})


def _items(rec):
    from server import server as srv
    from server.ocr import OCRResult
    from server.perception import Detection
    w, h = rec["size"]
    dets = [Detection(label=d.get("label_loc") or d["label"], box=d["box"], score=d["score"])
            for d in rec["detections"]]
    ocr = [OCRResult(text=o["text"], confidence=o["confidence"], bbox=o["bbox"]) for o in rec["ocr"]]
    return srv._build_localization_items(dets, ocr, w, h), srv._build_ocr_items(ocr, w, h)


def _grid_for(rec):
    """[(variant index, grid candidates top-59)] for one photo."""
    from server import visual_localization as VL
    items, ocr_items = _items(rec)
    out = []
    for k, (_, which, coeff) in enumerate(VARIANTS):
        _state["ref_map"].adjacent_map = _state["maps"][which]
        VL.SPATIAL_BONUS_COEFF = coeff
        out.append(VL.match_by_grid(items, _state["ref_map"], top_k=len(_state["ref_map"].photos),
                                    ocr_items=ocr_items, store_zone=rec.get("store_zone") or None))
    return out


def load_map(arg):
    if arg == "neo4j":
        from server.config import NEO4J_PLACE
        from server.neo4j_client import get_neo4j
        return get_neo4j().load_reference_map(NEO4J_PLACE)
    # a RefMap snapshot this eval or image_only.py wrote locally (not for untrusted files)
    return pickle.load(open(arg, "rb"))


def load_adjacent(arg):
    if arg != "neo4j":
        return json.loads(Path(arg).read_text(encoding="utf-8"))
    from server.config import NEO4J_PLACE
    from server.neo4j_client import get_neo4j
    return [dict(r) for r in get_neo4j()._query("""
        MATCH (w:Waypoint {place: $place})-[hp:HAS_PHOTO]->(d:DirPhoto)
              -[:DETECTED]->(a:Object)-[r:ADJACENT]->(b:Object)<-[:DETECTED]-(d)
        RETURN w.nid AS nid, hp.slot AS slot,
               coalesce(a.label_norm, a.label) AS a_label, coalesce(b.label_norm, b.label) AS b_label,
               r.cell_pairs AS cell_pairs, r.overlap AS overlap, r.depth_diff AS depth_diff
    """, {"place": NEO4J_PLACE})]


def local_label_vectors(labels, model_name):
    from sentence_transformers import SentenceTransformer
    labels = sorted(labels)
    vecs = SentenceTransformer(model_name).encode(labels, normalize_embeddings=True, batch_size=256)
    return {l: v.tolist() for l, v in zip(labels, vecs)}


def hop_table(ref_map):
    hops = {}
    for s in ref_map.photos:
        d = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in ref_map.photos[u].neighbor_nids:
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        hops[s] = d
    return hops


def metrics(recs, picks, top5s, ref_map, hops):
    n = len(recs)
    ok = [set(r["ok"]) for r in recs]
    err = [math.hypot(ref_map.photos[p].pdr_x - r["truth_xy"][0], ref_map.photos[p].pdr_y - r["truth_xy"][1])
           for r, p in zip(recs, picks)]
    return {"top1": 100 * sum(p in o for p, o in zip(picks, ok)) / n,
            "top5": 100 * sum(bool(o & set(t)) for o, t in zip(ok, top5s)) / n,
            "hop1": 100 * sum(min(hops[p].get(t, 99) for t in o) <= 1 for p, o in zip(picks, ok)) / n,
            "err_m": statistics.mean(err)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", default="eval/results/image_only_lab1019_vlm_photos.jsonl")
    ap.add_argument("--map", default="neo4j")
    ap.add_argument("--adjacent", default="neo4j")
    ap.add_argument("--local-model", help="sentence-transformers model for label vectors when there is no "
                                          "OPENAI_API_KEY")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="eval/results/relations_compare_lab1019")
    args = ap.parse_args(argv)

    from server import config as C
    from server.loc_fusion import fuse_v2, image_evidence, parse_image_k

    recs = [json.loads(l) for l in open(args.photos, encoding="utf-8")]
    ref_map = load_map(args.map)
    rows = load_adjacent(args.adjacent)
    vecs = None
    if args.local_model:
        labels = {o.label_norm or o.label for n in ref_map.photos.values() for dp in n.directional_photos
                  for o in dp.objects}
        labels |= {d.get("label_loc") or d["label"] for r in recs for d in r["detections"]}
        vecs = local_label_vectors(labels, args.local_model)
    print(f"{len(recs)} photos, {len(rows)} ADJACENT edges, label vectors: "
          f"{'local ' + args.local_model if vecs else 'server'}")

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.map, rows, vecs)) as pool:
        grids = list(pool.map(_grid_for, recs, chunksize=4))

    xy = {n: (p.pdr_x, p.pdr_y) for n, p in ref_map.photos.items()}
    hops = hop_table(ref_map)
    phone_k = parse_image_k(C.LOC_PHONE_IMAGE_K)
    res = {}
    for k, (name, which, coeff) in enumerate(VARIANTS):
        g_pick, g_top5, p_pick, p_top5 = [], [], [], []
        for rec, per in zip(recs, grids):
            cand = per[k]
            g_pick.append(cand[0][0] if cand else next(iter(ref_map.photos)))
            g_top5.append([c[0] for c in cand[:5]])
            dino = sorted(((int(n), v["dino"]) for n, v in rec["per_node"].items()), key=lambda x: -x[1])[:10]
            ev = image_evidence(cand[:10], dict(dino), {int(n): v for n, v in rec["aisle_evidence"].items()})
            post = fuse_v2(ev, xy, image_k=phone_k)
            p_pick.append(post[0][0])
            p_top5.append([c[0] for c in post[:5]])
        res[f"{name} ×{coeff:g}" if which else name] = {
            "格子比對": metrics(recs, g_pick, g_top5, ref_map, hops),
            "手機 DINOv2": metrics(recs, p_pick, p_top5, ref_map, hops),
            "差異照片數（格子第一名和沒有關係不同）": None}
    base = [per[0][0][0] if per[0] else None for per in grids]
    for k, key in enumerate(res):
        res[key]["差異照片數（格子第一名和沒有關係不同）"] = sum(
            (per[k][0][0] if per[k] else None) != b for per, b in zip(grids, base))

    md = [f"# 物件關係兩個版本比較（{Path(args.photos).name}，{len(recs)} 張）", "",
          "同一批照片、同一個格子比對（match_by_grid），只換 ADJACENT 怎麼讀、加分多少。"
          "標籤相似度：" + (f"本機多語模型 `{args.local_model}`（沒有 OPENAI_API_KEY 時的近似）" if vecs
                         else "伺服器的 _hybrid_max_sim") + "。", "",
          "| 版本 | 格子比對 Top-1 | Top-5 | 1 hop 內 | 誤差 m | 手機 DINOv2 Top-1 | Top-5 | 1 hop 內 | 誤差 m | 格子第一名改變的照片 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, r in res.items():
        g, p = r["格子比對"], r["手機 DINOv2"]
        md.append(f"| {key} | {g['top1']:.1f} | {g['top5']:.1f} | {g['hop1']:.1f} | {g['err_m']:.2f} | "
                  f"{p['top1']:.1f} | {p['top5']:.1f} | {p['hop1']:.1f} | {p['err_m']:.2f} | "
                  f"{r['差異照片數（格子第一名和沒有關係不同）']} |")
    Path(args.out + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
