"""Scoring helpers of eval/image_only.py (no models, no Neo4j)."""
from types import SimpleNamespace

from eval import image_only as io


def _map():
    # 1 — 2 — 3 — 4, edges stored one-way like the real map
    xy = {1: (0, 0), 2: (5, 0), 3: (10, 0), 4: (15, 0)}
    nbrs = {1: [2], 2: [], 3: [2], 4: [3]}
    return SimpleNamespace(photos={n: SimpleNamespace(pdr_x=x, pdr_y=y, neighbor_nids=nbrs[n])
                                   for n, (x, y) in xy.items()})


def _loc(top, matched=True, est=None, heading=None):
    return SimpleNamespace(top_candidates=[(n, p, "") for n, p in top], confidence=top[0][1],
                           matched_nid=top[0][0] if matched else None, estimated_xy=est,
                           matched_slot="front", matched_heading=heading)


def _item(truth, ok=None, heading=None, cond="at_node"):
    m = _map()
    return {"id": f"p{truth}", "ok": ok or [truth], "truth": truth,
            "truth_xy": [m.photos[truth].pdr_x, m.photos[truth].pdr_y],
            "condition": cond, "heading": heading, "meta": {}}


def test_hops_walk_edges_both_ways():
    h = io.hop_table(_map())
    assert h[1][4] == 3 and h[4][1] == 3 and h[2][3] == 1


def test_score_and_summary():
    m, h = _map(), io.hop_table(_map())
    recs = [
        io.score_record(_item(1, heading=10), _loc([(1, 0.9), (2, 0.1)], heading=350), m, h),   # hit, heading ok
        io.score_record(_item(2), _loc([(3, 0.7), (2, 0.2)]), m, h),                            # neighbour, top-5
        io.score_record(_item(4), _loc([(1, 0.2), (2, 0.1)], matched=False), m, h),             # far
        io.score_record(_item(2, ok=[2, 3], cond="between"), _loc([(3, 0.4)]), m, h),           # between → ok
    ]
    assert [r["hit1"] for r in recs] == [True, False, False, True]
    assert recs[1]["hop"] == 1 and recs[2]["hop"] == 3 and recs[1]["err_m"] == 5.0
    assert recs[0]["heading_err"] == 20.0
    recs[2]["rejected"] = "occlusion"
    s = io.summarize(recs)
    assert s["top1"] == 50.0 and s["top5"] == 75.0 and s["within_1_hop"] == 75.0
    assert s["confidence"][">=0.6"] == {"coverage": 50.0, "n": 2, "top1_when_confident": 50.0,
                                         "within_1_hop_when_confident": 100.0}
    g = s["production_gate"]
    assert (g["rejected_occlusion"], g["answered"], g["top1_of_answered"]) == (1, 3, 66.7)
    assert s["heading"]["within_45deg"] == 100.0
    f = io.failures(recs)
    assert f["kind"] == {"neighbour (1 hop)": 1, "far (≥3 hops)": 1} and f["truth_in_top5"] == 1
    assert f["confused_pairs"][0]["nodes"] in ([2, 3], [1, 4])


def test_variant_restores_weights():
    from server import config as C
    from server import locate_fusion as LF
    before = (LF.W_N, LF.W_O, C.AISLE_SIGN_PLACES)
    with io.variant("dino_only"):
        assert LF.W_N == LF.W_O == 0 and not C.AISLE_SIGN_PLACES
    assert (LF.W_N, LF.W_O, C.AISLE_SIGN_PLACES) == before
