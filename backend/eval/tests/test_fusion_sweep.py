"""The sweep's vectorised scoring must pick the same node as server/loc_fusion.py."""
import random

import numpy as np

from eval.fusion_sweep import _evaluate, _prepare
from server.loc_fusion import FusionWeights, fuse_candidates


def _cache(rng):
    nodes = {str(n): {"x": rng.uniform(0, 30), "y": rng.uniform(0, 30), "nbrs": []} for n in range(1, 13)}
    queries = []
    for q in range(40):
        ids = list(range(1, 13))
        grid = {str(n): rng.random() for n in rng.sample(ids, 6)}
        dino = {str(n): rng.uniform(0.5, 0.9) for n in rng.sample(ids, 5)}
        queries.append({"nid": rng.choice(ids), "photo": f"set01/{q}.jpg", "set": "set01",
                        "dino": dino, "grid": grid, "n_items": 5})
    return {"nodes": nodes, "queries": queries, "grid_n": 4, "dropout": 0.0}


def test_vectorised_matches_fuse_candidates():
    rng = random.Random(0)
    cache = _cache(rng)
    node_ids, xy, G, D, gmask, dmask, truth, dist, nbr, txy = _prepare(cache)
    S = np.exp(-np.linalg.norm(xy[truth][:, None, :] - xy[None, :, :], axis=2) / 8.0)
    positions = {n: tuple(xy[i]) for i, n in enumerate(node_ids)}
    for w in [(5, 1, 1), (0, 0, 1), (1, 0, 0), (2, 3, 1), (0, 1, 1)]:
        hit1, *_ = _evaluate(G, D, S, gmask, dmask, truth, dist, nbr, w)
        for q, e in enumerate(cache["queries"]):
            grid = sorted(((int(n), s, "") for n, s in e["grid"].items()), key=lambda c: -c[1])
            dino = {int(n): s for n, s in e["dino"].items()}
            fused = fuse_candidates(grid, dino, tuple(xy[truth[q]]), positions, FusionWeights(*w))
            assert bool(hit1[q]) == (fused[0][0] == e["nid"]), (w, q)
