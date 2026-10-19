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


def _rec(id_, point, hit, conf, cond="at_node", draw=None):
    return {"id": id_, "meta": {"point": point}, "hit1": hit, "within1": hit, "confidence": conf,
            "condition": cond, "draw": draw, "err_m": 0.0 if hit else 5.0, "est_err_m": 1.0}


def test_point_summary_scores_each_point_once():
    recs = [_rec("a1", "A", True, 0.2), _rec("a2", "A", False, 0.9),   # most confident photo is wrong
            _rec("b1", "B", True, 0.5), _rec("b2", "B", True, 0.6, ), _rec("c1", "C", False, 0.4, "between")]
    s = io.point_summary(recs)["all"]
    assert s["points"] == 3 and s["mean_top1"] == 50.0           # (0.5 + 1 + 0) / 3
    assert s["any_top1"] == 66.7 and s["most_confident_top1"] == 33.3
    assert io.point_summary([_rec("x", "X", True, 1.0)]) is None   # one photo per point: nothing to add


def test_grid_and_paired():
    none = [_rec("p1", "A", True, 0.5), _rec("p2", "B", False, 0.5, "between")]
    rep = [_rec("p1", "A", False, 0.9), _rec("p2", "B", True, 0.9, "between")]
    g = io.grid({"prod": none, "prod+replay": rep}, ["none", "replay"])
    assert g["prod"]["at_node"]["none"]["top1"] == 100.0 and g["prod"]["between"]["replay"]["top1"] == 100.0
    p = io.paired(none, rep)["all"]
    assert (p["both"], p["only_first"], p["only_second"], p["neither"]) == (0, 1, 1, 0)


def test_sim_phone_is_reproducible_and_shared_by_a_point():
    item = {"id": "1/wp3_front", "meta": {"point": "1/3"}, "truth_xy": [1.0, 2.0]}
    other = dict(item, id="1/wp3_back")
    a, b = io.sim_phone(item, 2.0, 0), io.sim_phone(other, 2.0, 0)
    assert a == b and a != io.sim_phone(item, 2.0, 1)
    assert a["dist_since_fix"] == 20.0 and a["fix_node"] is None   # σ = 1 + 0.05·d → d = 20 m


def test_variant_restores_weights():
    from server import config as C
    from server import locate_fusion as LF
    before = (LF.W_V, LF.W_N, LF.W_O, C.AISLE_SIGN_PLACES)
    with io.variant("dino_only"):
        assert LF.W_N == LF.W_O == 0 and LF.W_V == before[0] and not C.AISLE_SIGN_PLACES
    with io.variant("objects_only"):
        assert LF.W_V == LF.W_O == 0 and LF.W_N == before[1] and not C.AISLE_SIGN_PLACES
    with io.variant("text_only"):
        assert LF.W_V == LF.W_N == 0 and LF.W_O == before[2] and C.AISLE_SIGN_PLACES == before[3]
    assert (LF.W_V, LF.W_N, LF.W_O, C.AISLE_SIGN_PLACES) == before
