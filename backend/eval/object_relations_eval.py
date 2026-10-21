"""Does adding object-to-object 上下左右 relations (server/object_relations.py) help localization?

Replays a saved image_only run (eval/image_only.py --set lab1019 → *_photos.jsonl), which
holds, per photo, every detection with its box and every reference node's prod fused score
(DINOv2 + objects + OCR, best directional photo). The server's image evidence and posterior
are rebuilt from those numbers exactly as server_image_evidence → localize(v2) → fuse_v2 do
(checked: the prod pick and confidence of every photo are reproduced), then

    fused_n += W · relation_score(photo objects, objects of node n's best directional photo)

for several W and relation variants. No sensor input (single photo), like image_only's
"prod" variant. W is also chosen by leave-one-walk-out (pick W on two walks, score the
third) so the reported gain is not just tuned on the test photos.

  python -m eval.object_relations_eval [--photos eval/results/image_only_lab1019_vlm_photos.jsonl]
                                       [--map neo4j|<refmap.pkl>] [--char-only]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
from collections import defaultdict, deque
from pathlib import Path

from server import config as C
from server import object_relations as R
from server import photo_quality
from server.loc_fusion import fuse_v2, parse_image_k
from server.visual_localization import AISLE_WEIGHT

WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
VARIANTS = {
    # name: (w_vertical, w_horizontal, coarse 3×3 cells, all four directional photos, top-5 only)
    "上下＋左右(0.5)": (1.0, 0.5, False, False, False),
    "只有上下": (1.0, 0.0, False, False, False),
    "只有左右": (0.0, 1.0, False, False, False),
    "上下＝左右": (1.0, 1.0, False, False, False),
    "九宮格粗略版(ADJACENT)": (1.0, 1.0, True, False, False),
    "上下＋左右，四個方向取最高": (1.0, 0.5, False, True, False),
    "上下＋左右，只重排前 5 名": (1.0, 0.5, False, False, True),
}


def _cell_box(box):
    """Snap a box to its 3×3 grid cell (where its centre falls) — the resolution of the
    map's existing ADJACENT edges (map_v2.compute_cell_pairs)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    c, r = min(int(cx * 3), 2), min(int(cy * 3), 2)
    return [c / 3, r / 3, (c + 1) / 3, (r + 1) / 3]


def load_map(arg):
    if arg == "neo4j":
        from server.neo4j_client import get_neo4j
        return get_neo4j().load_reference_map(C.NEO4J_PLACE)
    # a RefMap snapshot this eval or image_only.py wrote locally (not for untrusted files)
    return pickle.load(open(arg, "rb"))


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


def _local_embedding_similarity(labels, model_name):
    """Without an OpenAI key: max(character, cosine of a local multilingual sentence model)
    on the labels with position words stripped (左側／上層…) — an approximation of the
    server's _hybrid_max_sim (text-embedding-3-small), which needs the key."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from server import visual_localization as VL
    core = {l: VL._strip_direction(l) for l in labels}
    uniq = sorted(set(core.values()))
    vecs = SentenceTransformer(model_name).encode(uniq, normalize_embeddings=True, batch_size=256)
    vec = {c: v for c, v in zip(uniq, vecs)}

    def sim(a, b):
        return max(VL._semantic_similarity(a, b), float(np.dot(vec[core[a]], vec[core[b]])))
    return sim


def make_similarity(char_only, q_labels, r_labels, local_model=None):
    from server import visual_localization as VL
    if local_model:
        base = _local_embedding_similarity(set(q_labels) | set(r_labels), local_model)
    elif not char_only and VL._load_embedding_cache():
        VL.ensure_label_embeddings(sorted(r_labels), persist=True)
        VL.ensure_label_embeddings(sorted(q_labels))
        base = VL._hybrid_max_sim
    else:
        base = VL._semantic_similarity
    cache = {}

    def sim(a, b):
        k = (a, b)
        if k not in cache:
            cache[k] = base(a, b)
        return cache[k]
    return sim


def query_items(rec):
    w, h = rec["size"]
    out = []
    for d in rec["detections"]:
        label = d.get("label_loc") or d["label"]
        if photo_quality.is_occluder(label):     # as server._build_localization_items
            continue
        b = d["box"]
        out.append((label, [b[0] / w, b[1] / h, b[2] / w, b[3] / h]))
    return out


def ref_items(ref_map, nid, slot):
    node = ref_map.photos[nid]
    dp = next((p for p in node.directional_photos if p.slot == slot), None)
    objs = dp.objects if dp else node.objects
    return [(o.label_norm or o.label, o.bbox) for o in objs]


def relation_table(recs, ref_map, sim, rarity):
    """{variant: [ {nid: score} per photo ]}"""
    out = {}
    for name, (wv, wh, coarse, all_slots, _) in VARIANTS.items():
        rows = []
        for rec in recs:
            q = query_items(rec)
            if coarse:
                q = [(l, _cell_box(b)) for l, b in q]
            row = {}
            for k, v in rec["per_node"].items():
                slots = ([p.slot for p in ref_map.photos[int(k)].directional_photos] or [v["fused_slot"]]
                         if all_slots else [v["fused_slot"]])
                best = 0.0
                for slot in slots:
                    ref = ref_items(ref_map, int(k), slot)
                    if coarse:
                        ref = [(l, _cell_box(b)) for l, b in ref if R._valid(b)]
                    best = max(best, R.relation_score(q, ref, sim, rarity=rarity,
                                                      w_vertical=wv, w_horizontal=wh))
                row[int(k)] = best
            rows.append(row)
        out[name] = rows
        print(f"  relations computed: {name}")
    return out


def posterior(rec, ref_map, xy, K, rel=None, w=0.0, only=None):
    f = {int(k): v["fused"] for k, v in rec["per_node"].items()}
    if rel and w:
        f = {n: s + w * rel.get(n, 0.0) if only is None or n in only else s for n, s in f.items()}
    lo, hi = min(f.values()), max(f.values())
    ev = ({n: 2 * (f.get(n, lo) - lo) / (hi - lo) for n in ref_map.photos} if hi > lo
          else {n: 0.0 for n in ref_map.photos})
    for n, v in rec["aisle_evidence"].items():
        ev[int(n)] = ev.get(int(n), 0.0) + AISLE_WEIGHT * v
    return fuse_v2(ev, xy, image_k=K)


def score(recs, posts, ref_map, hops):
    top1 = top5 = hop1 = 0
    err = []
    for rec, post in zip(recs, posts):
        pick = post[0][0]
        ok = set(rec["ok"])
        top1 += pick in ok
        top5 += bool(ok & {c[0] for c in post[:5]})
        hop1 += min(hops[pick].get(t, 99) for t in ok) <= 1
        p = ref_map.photos[pick]
        err.append(math.hypot(p.pdr_x - rec["truth_xy"][0], p.pdr_y - rec["truth_xy"][1]))
    n = len(recs) or 1
    return {"n": len(recs), "top1": 100 * top1 / n, "top5": 100 * top5 / n, "hop1": 100 * hop1 / n,
            "err_m": statistics.mean(err) if err else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", default="eval/results/image_only_lab1019_vlm_photos.jsonl")
    ap.add_argument("--map", default="neo4j")
    ap.add_argument("--char-only", action="store_true", help="label similarity without embeddings")
    ap.add_argument("--local-model", help="sentence-transformers model for label similarity when there is "
                                          "no OPENAI_API_KEY, e.g. "
                                          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--out", default="eval/results/object_relations_lab1019")
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in open(args.photos, encoding="utf-8")]
    ref_map = load_map(args.map)
    xy = {n: (p.pdr_x, p.pdr_y) for n, p in ref_map.photos.items()}
    K = parse_image_k(C.LOC_SERVER_IMAGE_K)
    hops = hop_table(ref_map)

    base = [posterior(r, ref_map, xy, K) for r in recs]
    same = sum(p[0][0] == r["results"]["prod"][0]["pred"] for p, r in zip(base, recs))
    print(f"{len(recs)} photos; prod pick reproduced for {same}")
    if same != len(recs):
        raise SystemExit("replay does not reproduce the saved prod result — map or config changed")

    r_labels = {o.label_norm or o.label for n in ref_map.photos.values() for dp in n.directional_photos
                for o in dp.objects}
    q_labels = {l for r in recs for l, _ in query_items(r)}
    sim = make_similarity(args.char_only, q_labels, r_labels, args.local_model)
    rarity = R.label_rarity([[o.label_norm or o.label for o in dp.objects]
                             for n in ref_map.photos.values() for dp in n.directional_photos])
    rel = relation_table(recs, ref_map, sim, rarity)

    walks = sorted({r["id"].split("/")[0] for r in recs})
    groups = {"全部": list(range(len(recs))),
              "在節點上": [i for i, r in enumerate(recs) if r["condition"] == "at_node"],
              "兩節點之間": [i for i, r in enumerate(recs) if r["condition"] != "at_node"]}
    results = {"baseline": {g: score([recs[i] for i in ix], [base[i] for i in ix], ref_map, hops)
                            for g, ix in groups.items()}, "variants": {}, "cv": {}}
    posts_all = {}
    for name in VARIANTS:
        results["variants"][name] = {}
        top5_only = VARIANTS[name][4]
        for w in WEIGHTS:
            posts = [posterior(r, ref_map, xy, K, rel[name][i], w,
                               only={c[0] for c in base[i][:5]} if top5_only else None)
                     for i, r in enumerate(recs)]
            posts_all[(name, w)] = posts
            results["variants"][name][str(w)] = {
                g: score([recs[i] for i in ix], [posts[i] for i in ix], ref_map, hops)
                for g, ix in groups.items()}
        # leave-one-walk-out: W picked on the other walks' Top-1, scored on this walk
        picks, held = [], []
        for wk in walks:
            train = [i for i, r in enumerate(recs) if not r["id"].startswith(wk + "/")]
            test = [i for i, r in enumerate(recs) if r["id"].startswith(wk + "/")]
            best_w = max(WEIGHTS, key=lambda w: (score([recs[i] for i in train],
                                                       [posts_all[(name, w)][i] for i in train],
                                                       ref_map, hops)["top1"], -w))
            picks.append(best_w)
            held += [(i, best_w) for i in test]
        results["cv"][name] = {
            "picked_w": dict(zip(walks, picks)),
            **score([recs[i] for i, _ in held], [posts_all[(name, w)][i] for i, w in held], ref_map, hops)}
        print(f"  {name}: cv top1 {results['cv'][name]['top1']:.1f}% (W per walk {picks})")

    # how often relations alone pick the right node (is the signal discriminative at all?)
    results["alone"] = {}
    for name in VARIANTS:
        hit = sum(max(rel[name][i], key=rel[name][i].get) in set(r["ok"]) for i, r in enumerate(recs)
                  if any(rel[name][i].values()))
        has = sum(1 for i in range(len(recs)) if any(rel[name][i].values()))
        results["alone"][name] = {"top1": 100 * hit / len(recs), "photos_with_any_relation": has}

    Path(args.out + ".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.out + ".md").write_text(markdown(results, len(recs), args), encoding="utf-8")
    print(Path(args.out + ".md").read_text(encoding="utf-8"))


def markdown(res, n, args):
    b = res["baseline"]["全部"]
    L = [f"# 物件上下左右關係對定位的影響（lab1019，{n} 張，單張照片、不帶感測器）", "",
         f"重播 `{args.photos}`；基準＝現在的伺服器（DINOv2＋物件＋OCR＋走道編號）。",
         "物件名稱配對：" + (f"本機多語模型 `{args.local_model}`（沒有 OPENAI_API_KEY 時的近似）"
                       if args.local_model else "字元相似度" if args.char_only else "伺服器的 _hybrid_max_sim") + "。",
         "Top-1＝第一名就是對的點；1 hop 內＝第一名是對的點或它的鄰點；誤差＝第一名到實際位置的距離。", "",
         f"**基準：Top-1 {b['top1']:.1f}%，Top-5 {b['top5']:.1f}%，1 hop 內 {b['hop1']:.1f}%，誤差 {b['err_m']:.2f} m**", "",
         "## 換一個權重 W 的 Top-1（全部照片）", "",
         "| 版本 | " + " | ".join(f"W={w}" for w in WEIGHTS) + " |",
         "|---|" + "---:|" * len(WEIGHTS)]
    for name, by_w in res["variants"].items():
        L.append(f"| {name} | " + " | ".join(f"{by_w[str(w)]['全部']['top1']:.1f}" for w in WEIGHTS) + " |")
    L += ["", "## 不偷看答案：兩條路線選 W、第三條路線測（三輪輪流）", "",
          "| 版本 | Top-1 | Top-5 | 1 hop 內 | 誤差 m | 各輪選到的 W |", "|---|---:|---:|---:|---:|---|",
          f"| 基準（不用關係） | {b['top1']:.1f} | {b['top5']:.1f} | {b['hop1']:.1f} | {b['err_m']:.2f} | — |"]
    for name, c in res["cv"].items():
        L.append(f"| {name} | {c['top1']:.1f} | {c['top5']:.1f} | {c['hop1']:.1f} | {c['err_m']:.2f} | "
                 f"{', '.join(f'{k}:{v}' for k, v in c['picked_w'].items())} |")
    L += ["", "## 在節點上 vs 兩節點之間（W 取全部照片上最好的）", "",
          "| 版本 | W | 在節點上 Top-1 | 兩節點之間 Top-1 |", "|---|---:|---:|---:|",
          f"| 基準 | 0 | {res['baseline']['在節點上']['top1']:.1f} | {res['baseline']['兩節點之間']['top1']:.1f} |"]
    for name, by_w in res["variants"].items():
        w = max(WEIGHTS, key=lambda w: (by_w[str(w)]["全部"]["top1"], -w))
        L.append(f"| {name} | {w} | {by_w[str(w)]['在節點上']['top1']:.1f} | {by_w[str(w)]['兩節點之間']['top1']:.1f} |")
    L += ["", "## 只用關係分數自己選（看這個訊號本身分不分得出地點）", "",
          "| 版本 | Top-1 | 有算出關係的照片 |", "|---|---:|---:|"]
    for name, a in res["alone"].items():
        L.append(f"| {name} | {a['top1']:.1f} | {a['photos_with_any_relation']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
