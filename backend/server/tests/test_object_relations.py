"""上下左右 relations between objects (server/object_relations.py), their use in the server
image evidence (locate_fusion.add_relation_evidence) and the Neo4j edge rows
(build_object_relations.relation_rows)."""
from unittest.mock import patch

import pytest

from server import object_relations as R
from server.neo4j_client import DirectionalRef, RefObject

SIGN = [0.40, 0.05, 0.60, 0.15]      # hanging sign, top middle
SHELF = [0.35, 0.40, 0.65, 0.90]     # shelf right under it
LEFT = [0.00, 0.40, 0.25, 0.90]      # fridge to the left of the shelf
RIGHT = [0.75, 0.40, 1.00, 0.90]     # poster to the right of the shelf


def same(a, b):
    return 1.0 if a == b else 0.0


def test_pair_relation_axes():
    assert R.pair_relation(SIGN, SHELF) == R.PairRelation(v=-1, h=0)     # above, same column
    assert R.pair_relation(SHELF, SIGN) == R.PairRelation(v=1, h=0)
    assert R.pair_relation(LEFT, SHELF) == R.PairRelation(v=0, h=-1)     # left, same row
    assert R.pair_relation(SIGN, LEFT) == R.PairRelation(v=-1, h=1)      # diagonal: both axes
    assert R.pair_relation(SHELF, [0.4, 0.5, 0.6, 0.8]) == R.PairRelation(0, 0)   # overlapping
    assert R.pair_relation(SHELF, []) == R.PairRelation(0, 0)


def test_neighbour_pairs_keeps_the_nearest():
    boxes = [[0.0, 0.0, 0.1, 0.1], [0.11, 0.0, 0.2, 0.1], [0.9, 0.9, 1.0, 1.0], None]
    assert R.neighbour_pairs(boxes, k=1) == [(0, 1), (1, 2)]


def test_label_rarity_common_labels_count_less():
    r = R.label_rarity([["價格標籤", "冷藏櫃"], ["價格標籤"], ["價格標籤", "出口指示牌"]])
    assert r["價格標籤"] < r["冷藏櫃"] and r["冷藏櫃"] == r["出口指示牌"]


def test_relation_score_same_layout_scores_and_flipped_does_not():
    ref = [("sign", SIGN), ("shelf", SHELF), ("fridge", LEFT), ("poster", RIGHT)]
    shifted = [(l, [b[0] * 0.9 + 0.05, b[1], b[2] * 0.9 + 0.05, b[3]]) for l, b in ref]
    upside_down = [(l, [b[0], 1 - b[3], b[2], 1 - b[1]]) for l, b in ref]
    mirrored = [(l, [1 - b[2], b[1], 1 - b[0], b[3]]) for l, b in ref]
    good = R.relation_score(shifted, ref, same)
    assert good > 0.5
    assert R.relation_score(upside_down, ref, same) == 0.0
    # left/right counts half: a mirrored view keeps the 上下 agreement but loses on 左右
    assert 0.0 <= R.relation_score(mirrored, ref, same) < good


def test_relation_score_needs_two_matched_objects():
    ref = [("sign", SIGN), ("shelf", SHELF)]
    assert R.relation_score([("sign", SIGN)], ref, same) == 0.0
    assert R.relation_score([("x", SIGN), ("y", SHELF)], ref, same) == 0.0      # nothing matches
    # one agreeing pair is weak evidence (smoothing), more pairs are stronger
    one = R.relation_score([("sign", SIGN), ("shelf", SHELF)], ref, same)
    full = [("sign", SIGN), ("shelf", SHELF), ("fridge", LEFT), ("poster", RIGHT)]
    assert 0 < one < R.relation_score(full, full, same)


def _best(nid, objects):
    dp = DirectionalRef(slot="front", heading_deg=0.0, photo_file=f"{nid}.jpg", objects=[
        RefObject(label=l, label_norm=l, score=0.9, ocr_text="", role="", grid_cell="", bbox=b)
        for l, b in objects])
    return {"photo": dp}


def test_add_relation_evidence_only_when_weighted():
    from server import locate_fusion as LF
    layout = [("sign", SIGN), ("shelf", SHELF), ("fridge", LEFT), ("poster", RIGHT)]
    flipped = [(l, [b[0], 1 - b[3], b[2], 1 - b[1]]) for l, b in layout]
    best = {1: (0.5, _best(1, layout)), 2: (0.5, _best(2, flipped)), 3: (0.4, {"photo": None})}
    items = [{"label": l, "box": b} for l, b in layout]
    with patch.object(LF, "_embedding_sim", same):
        assert LF.add_relation_evidence(best, items, {}, 0.0) == {1: 0.5, 2: 0.5, 3: 0.4}
        out = LF.add_relation_evidence(best, items, {}, 0.3)
    assert out[1] > 0.5 and out[2] == 0.5 and out[3] == 0.4


def test_server_passes_boxes_and_weight():
    from server import server as srv
    with patch("server.locate_fusion.server_image_evidence", return_value=None) as ev, \
         patch.object(srv._config, "LOC_RELATION_WEIGHT", 0.2):
        srv._server_dino_evidence("sid", object(), "p.jpg", ["sign"], [],
                                  detected_items=[{"label": "sign", "box": SIGN}])
    assert ev.call_args.kwargs == {"q_items": [{"label": "sign", "box": SIGN}], "relation_weight": 0.2}


def test_relation_on_by_default():
    from server import config
    assert config.LOC_RELATION_WEIGHT == 0.05


def test_neo4j_rows_come_in_inverse_pairs():
    import build_object_relations as B
    rows = B.relation_rows([{"oid": "s", "bbox": SIGN}, {"oid": "f", "bbox": LEFT}])
    got = {(r["a"], r["type"], r["b"]) for r in rows}
    assert got == {("s", "ABOVE", "f"), ("f", "BELOW", "s"), ("f", "LEFT_OF", "s"), ("s", "RIGHT_OF", "f")}
    assert all(r["dx"] >= 0 and r["dy"] >= 0 for r in rows)


# ── ADJACENT (map_v2 grid cells) for the grid matcher's spatial bonus ─────────────────

def test_adjacent_dir_is_where_b_is_so_a_is_the_opposite():
    from server.neo4j_client import adjacent_relation
    # grid_utils._DIR_NAMES: (1, 0) "below" = b one row lower → a is ABOVE b
    assert adjacent_relation('[{"a": "top-center", "b": "mid-center", "dir": "below"}]') == \
        {"v_rel": "above", "h_rel": "same_col"}
    assert adjacent_relation('[{"a": "mid-center", "b": "mid-right", "dir": "right"},'
                             ' {"a": "mid-center", "b": "bottom-right", "dir": "below-right"}]') == \
        {"v_rel": "above", "h_rel": "left"}
    # spans several cells both ways → that axis says nothing
    assert adjacent_relation('[{"dir": "above"}, {"dir": "below"}]')["v_rel"] == "same_row"
    assert adjacent_relation("not json") == {"v_rel": "same_row", "h_rel": "same_col"}


def test_add_adjacent_stores_the_inverse_and_drops_conflicts():
    from server.neo4j_client import add_adjacent
    rels = {}
    add_adjacent(rels, "sign", "shelf", {"v_rel": "above", "h_rel": "left", "overlap": False})
    assert rels[("shelf", "sign")]["v_rel"] == "below" and rels[("shelf", "sign")]["h_rel"] == "right"
    # a second sign/shelf pair in the same photo, on the other side → 左右 unknown, 上下 kept
    add_adjacent(rels, "sign", "shelf", {"v_rel": "above", "h_rel": "right", "overlap": False})
    assert rels[("sign", "shelf")] == {"v_rel": "above", "h_rel": "same_col", "overlap": False}


def test_same_layout_matches_the_map_and_flipped_does_not():
    from server import visual_localization as VL
    from server.neo4j_client import add_adjacent, adjacent_relation
    rels = {}
    add_adjacent(rels, "sign", "shelf", {**adjacent_relation('[{"dir": "below"}]'), "overlap": False})
    photo = VL.compute_query_spatial([{"label": "sign", "box": SIGN}, {"label": "shelf", "box": SHELF}])
    upside = VL.compute_query_spatial([{"label": "sign", "box": SHELF}, {"label": "shelf", "box": SIGN}])
    assert VL._spatial_direction_match(photo[("sign", "shelf")], rels[("sign", "shelf")]) > \
        VL._spatial_direction_match(upside[("sign", "shelf")], rels[("sign", "shelf")])
    # the b → a lookup is inverted too: it scores like a → a
    assert VL._spatial_direction_match(photo[("shelf", "sign")], rels[("shelf", "sign")]) == \
        VL._spatial_direction_match(photo[("sign", "shelf")], rels[("sign", "shelf")])


def test_adjacent_on_by_default():
    from server import config
    assert config.LOC_ADJACENT is True
