"""When the VLM re-ranker moves the match to a different reference node, the
heading (which directional photo the user's view matches) must be recomputed
for that node — otherwise relative turns are computed against the wrong node's
orientation."""
from unittest.mock import patch

from server.neo4j_client import RefMap, RefPhotoNode, RefObject, DirectionalRef
from server.session import Session
from server.visual_localization import LocalizationResult
from server import server as srv


def _obj(label):
    return RefObject(label=label, label_norm=label, score=0.9, ocr_text="", role="landmark", grid_cell="c")


def _node(nid, slots):
    """slots: {slot_name: [labels]}; headings 0/90/180/270 so they count as calibrated."""
    hdg = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}
    return RefPhotoNode(
        nid=nid, photo_file=f"{nid}.jpg", pdr_x=0, pdr_y=0, heading_deg=0.0, session="s",
        total_steps=0, total_distance_m=0.0, objects=[_obj("shelf")],
        directional_photos=[DirectionalRef(slot=sl, heading_deg=hdg[sl], photo_file=f"{nid}_{sl}.jpg",
                                           objects=[_obj(l) for l in labels])
                            for sl, labels in slots.items()],
    )


def _ref_map():
    m = RefMap(place="test")
    # At node 1 the sign is on the RIGHT; at node 2 the same sign is straight AHEAD.
    m.photos[1] = _node(1, {"front": ["cooler"], "right": ["sign"], "back": ["door"], "left": ["shelf"]})
    m.photos[2] = _node(2, {"front": ["sign"], "right": ["door"], "back": ["shelf"], "left": ["cooler"]})
    return m


class _FakeNeo4j:
    def __init__(self, m): self._m = m
    def load_reference_map(self, place): return self._m


def test_rerank_switching_node_recomputes_heading_for_new_node():
    ref_map = _ref_map()
    s = Session(id="t", goal="milk", goal_objects=["milk"], place="test")
    grid_result = LocalizationResult(
        matched_nid=1, confidence=0.9, method="grid", reasoning="", ref_node=ref_map.photos[1],
        matched_heading=90.0, matched_slot="right", heading_confidence=0.8,
        top_candidates=[(1, 0.9, "grid"), (2, 0.85, "grid")],
    )
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)), \
         patch("server.server._localize_photo", return_value=grid_result), \
         patch("server.server._vlm_rerank", return_value=[(2, 0.95, "looks like node 2")]), \
         patch("server.config.RERANK_ENABLED", True), \
         patch("server.config.REF_PHOTO_ROOT", "/refs"):
        out = srv._run_early_localization(s, "sid", ["sign"], [], image_path="p.jpg")

    assert out.matched_nid == 2
    # the user sees the sign, which at node 2 is the FRONT directional photo
    assert out.matched_slot == "front"
    assert out.matched_heading == 0.0
    assert s.heading_slot == "front"


def test_top_candidates_include_hint_boosted_winner():
    """The proximity bonus can lift a node from outside the raw top-5 into first
    place. The re-ranker must then be shown that node — otherwise the system's
    own pick is the one candidate the visual comparison never sees."""
    from server import visual_localization as vl

    m = RefMap(place="test")
    for nid in range(1, 8):
        m.photos[nid] = _node(nid, {"front": ["shelf"]})
    m.photos[7].neighbor_nids = [6]          # node 6 is next to the previous fix (7)

    raw = [(1, 0.30, "a"), (2, 0.29, "b"), (3, 0.28, "c"), (4, 0.27, "d"),
           (5, 0.26, "e"), (6, 0.25, "f")]  # node 6 is 6th → outside the raw top-5
    with patch.object(vl, "match_by_objects", return_value=raw):
        out = vl.localize(["shelf"], [], _FakeNeo4j(m), place="test", hint_nid=7)

    assert out.matched_nid == 6                       # 0.25 + 0.08 proximity = 0.33 wins
    cand_nids = [c[0] for c in out.top_candidates]
    assert cand_nids[0] == out.matched_nid            # the pick leads the re-rank list
    assert len(cand_nids) == 5


def test_ref_photo_coverage_counts_resolvable_files(tmp_path):
    """Startup check: how many of the map's reference photos exist on disk under
    REF_PHOTO_ROOT — so a wrong folder is visible in the log, not silently a no-op."""
    from server.visual_localization import ref_photo_coverage

    (tmp_path / "01").mkdir()
    (tmp_path / "01" / "a_front.jpg").write_bytes(b"x")
    m = RefMap(place="test")
    m.photos[1] = _node(1, {"front": ["shelf"]})
    m.photos[1].photo_file = "set01/a_front.jpg"
    m.photos[1].directional_photos[0].photo_file = "set01/a_front.jpg"
    m.photos[2] = _node(2, {"front": ["shelf"]})
    m.photos[2].photo_file = "set01/missing.jpg"
    m.photos[2].directional_photos[0].photo_file = "set01/missing.jpg"

    found, total, nodes_ok = ref_photo_coverage(m, str(tmp_path))

    assert (found, total) == (1, 2)
    assert nodes_ok == 1
    assert ref_photo_coverage(m, "") == (0, 2, 0)
