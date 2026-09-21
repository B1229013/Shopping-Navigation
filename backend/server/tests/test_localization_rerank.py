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
        top_candidates=[(1, 0.9), (2, 0.85)],
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
