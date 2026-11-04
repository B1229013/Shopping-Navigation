"""Evaluate the v2 localization fusion (server/loc_fusion.py `fuse_v2`) on walk scenarios.

Every held-out reference photo (as in fusion_sweep) is reached by a simulated walk:
a random walk of L metres back along the map graph gives the previous photo fix; the
phone's sensor position is the true position plus Gaussian error σ_true, where the
server assumes σ = 1 m + 5% × L. σ_true = σ × drift (drift 2 = the phone drifts twice
as fast as assumed). Variants:

  v1         the previous weighted average (sensor exp(-d/8), image-only candidates)
  sensor     nearest node to the sensor position
  v2         -d²/2σ² + k(σ)·image            (all nodes are candidates)
  +reach     nodes not walkable from the last fix within 1.5·L + 8 m get -3
  +aisle     aisle signs seen in the photo (aisle_signs.py; table rebuilt without it)

  python -m eval.fusion_v2 --cache <fusion_sweep cache.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.fusion_sweep import _prepare


def _setup(cache):
    import networkx as nx
    from server.aisle_signs import build_table, numbers_in_items, sign_evidence, walkway_nodes
    from server.config import NEO4J_PLACE
    from server.loc_fusion import dino_gap
    from server.neo4j_client import get_neo4j

    node_ids, xy, G, D, gmask, dmask, truth, dist, nbr, txy = _prepare(cache)
    Q, N = G.shape
    idx = {n: i for i, n in enumerate(node_ids)}
    gap = np.array([dino_gap({int(k): v for k, v in q["dino"].items()}) or 0 for q in cache["queries"]])
    img = 1.6 * np.clip(gap / 0.05, 0, 1)[:, None] * D + 0.4 * G     # = loc_fusion.image_evidence

    ref_map = get_neo4j().load_reference_map(NEO4J_PLACE)
    by_file = {dp.photo_file: dp for n in ref_map.photos.values() for dp in n.directional_photos}
    walkway = walkway_nodes(ref_map)
    A = np.zeros((Q, N))
    has_sign = np.zeros(Q, bool)
    for q, e in enumerate(cache["queries"]):
        dp = by_file[e["photo"]]
        nums = numbers_in_items([{"label": o.label, "text": o.ocr_text, "box": o.bbox} for o in dp.objects])
        ev = sign_evidence(nums, ref_map, table=build_table(ref_map, exclude_photo=e["photo"]),
                           walkway=walkway)
        for nid, v in ev.items():
            A[q, idx[nid]] = v
        has_sign[q] = bool(ev)

    g = nx.Graph()
    for i in range(N):
        for j in np.where(nbr[i])[0]:
            g.add_edge(i, int(j), w=float(dist[i, j]))
    W = np.full((N, N), np.inf)
    for i, dd in nx.all_pairs_dijkstra_path_length(g, weight="w"):
        for j, v in dd.items():
            W[i, j] = v
    return dict(node_ids=node_ids, xy=xy, G=G, D=D, gmask=gmask, dmask=dmask, truth=truth, dist=dist,
                nbr=nbr, txy=txy, gap=gap, img=img, A=A, has_sign=has_sign, g=g, W=W)


def _walk_back(g, dist, rng, t, L):
    cur, prev, acc = t, None, 0.0
    while acc < L:
        nb = [j for j in g.neighbors(cur) if j != prev] or list(g.neighbors(cur))
        nxt = nb[rng.integers(len(nb))]
        acc += dist[cur, nxt]
        prev, cur = cur, nxt
    return cur, acc


def _metrics(st, score):
    top = score.argmax(1)
    ok = top == st["truth"]
    return np.array([ok.mean(), (ok | st["nbr"][st["truth"], top]).mean(),
                     st["dist"][st["truth"], top].mean()])


def scenario_scores(st, L, drift, rng, params):
    from server.loc_fusion import image_weight
    truth, xy, txy, dist = st["truth"], st["xy"], st["txy"], st["dist"]
    prev = [_walk_back(st["g"], dist, rng, t, L) if L > 0 else (t, 0.0) for t in truth]
    f = np.array([p[0] for p in prev])
    Lw = np.array([p[1] for p in prev])
    sig = 1 + 0.05 * Lw
    pos = txy + rng.normal(0, 1, txy.shape) * (sig * drift)[:, None]
    d = np.linalg.norm(pos[:, None, :] - xy[None, :, :], axis=2)
    reach = np.where(st["W"][f] <= 1.5 * Lw[:, None] + 8, 0.0, -3.0)
    k = np.array([image_weight(s) for s in sig])[:, None]
    ws = np.minimum(40, 40 / sig)[:, None]
    wd = 4 * np.clip(st["gap"] / 0.05, 0, 1)[:, None]
    v1 = np.where(st["gmask"] | st["dmask"], (wd * st["D"] + ws * np.exp(-d / 8) + st["G"]) / (wd + ws + 1), -np.inf)
    base = -d ** 2 / (2 * sig[:, None] ** 2) + k * st["img"]
    out = {"v1 (before)": v1, "sensor only": -d, "v2": base, "v2 +reach": base + reach}
    for ka in params.get("aisle", [1.0]):
        out[f"v2 +reach +aisle{ka:g}"] = base + reach + k * ka * st["A"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--aisle", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    ap.add_argument("--json")
    args = ap.parse_args()
    cache = json.loads(Path(args.cache).read_text())
    st = _setup(cache)
    rng = np.random.default_rng(0)
    print(f"{len(st['truth'])} held-out photos; {st['has_sign'].sum()} show an aisle sign "
          f"(top1 % / node-or-neighbour % / error m)")
    results = {}
    for drift in [1, 2]:
        for L in [0, 10, 20, 40, 80]:
            acc = {}
            for _ in range(args.reps):
                for name, sc in scenario_scores(st, L, drift, rng, {"aisle": args.aisle}).items():
                    acc.setdefault(name, []).append(_metrics(st, sc))
            res = {n: np.mean(v, 0) for n, v in acc.items()}
            results[f"{L}m x{drift}"] = {n: [round(100 * r[0], 1), round(100 * r[1], 1), round(r[2], 2)]
                                        for n, r in res.items()}
            print(f"walked {L:2d} m, drift x{drift}: " + " | ".join(
                f"{n} {100 * r[0]:.0f}/{100 * r[1]:.0f}/{r[2]:.1f}" for n, r in res.items()))
    # no sensor at all (before the first fix)
    img = st["img"]
    nos = {"image only": img}
    for ka in args.aisle:
        nos[f"image +aisle{ka:g}"] = img + ka * st["A"]
    results["no sensor"] = {}
    line = []
    for n, sc in nos.items():
        r = _metrics(st, sc)
        sub = _metrics({**st, "truth": st["truth"][st["has_sign"]], "nbr": st["nbr"]}, sc[st["has_sign"]])
        results["no sensor"][n] = [round(100 * r[0], 1), round(100 * r[1], 1), round(r[2], 2),
                                   round(100 * sub[0], 1)]
        line.append(f"{n} {100 * r[0]:.1f}/{100 * r[1]:.1f}/{r[2]:.1f} (photos with a sign: {100 * sub[0]:.1f})")
    print("no sensor: " + " | ".join(line))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
