"""_build_route_context must turn the map's edge distances into guidance.

The reference map stores distance_m on every WALKWAY edge; a bare "右轉" is
ambiguous when two aisles both open to the right, so the instruction has to say
how far to walk (and how many intersections to pass) before the turn.
"""
from unittest.mock import patch

from server.neo4j_client import RefMap, RefPhotoNode, RefObject
from server.path_planner import RouteLeg, RoutePlan
from server.session import Session
from server.visual_localization import LocalizationResult
from server import server as srv


def _node(nid, x, y, label):
    return RefPhotoNode(
        nid=nid, photo_file=f"{nid}.jpg", pdr_x=x, pdr_y=y, heading_deg=0.0,
        session="s", total_steps=0, total_distance_m=0.0,
        objects=[RefObject(label=label, label_norm=label, score=0.9, ocr_text="", role="landmark", grid_cell="c")],
    )


def _two_aisle_map() -> RefMap:
    # Corridor heading north; aisle entrances at nodes 11 and 12 open to the
    # east; the milk (node 13) is down the SECOND aisle.
    m = RefMap(place="test")
    for nid, x, y, label in [(10, 0, 0, "entrance"), (11, 0, 6, "aisle 1"),
                             (12, 0, 12, "aisle 2"), (13, 5, 12, "milk")]:
        m.photos[nid] = _node(nid, x, y, label)
    m.walkway_edges = [
        {"from": 10, "to": 11, "distance_m": 6.0},
        {"from": 11, "to": 12, "distance_m": 6.0},
        {"from": 12, "to": 13, "distance_m": 5.0},
    ]
    return m


def _session_with_route(ref_map, path):
    s = Session(id="t", goal="milk", goal_objects=["milk"], target_objects=["milk"], place="test")
    s.route_plan = RoutePlan(legs=[RouteLeg(from_node=path[0], to_node=path[-1], path=path,
                                            cost=0.0, actions=[], purpose="target")],
                             visit_order=[path[-1]])
    s.target_nodes = [path[-1]]
    return s


def _localized_at(ref_map, nid, heading=0.0):
    return LocalizationResult(
        matched_nid=nid, confidence=0.9, method="test", reasoning="",
        ref_node=ref_map.photos[nid], matched_heading=heading, matched_slot="front",
        heading_confidence=0.8,
    )


class _FakeNeo4j:
    def __init__(self, ref_map):
        self._m = ref_map

    def load_reference_map(self, place):
        return self._m


def test_next_instruction_uses_map_distances_and_counts_passed_aisles():
    ref_map = _two_aisle_map()
    s = _session_with_route(ref_map, [10, 11, 12, 13])
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)):
        ctx, next_instr = srv._build_route_context(s, _localized_at(ref_map, 10), detections=[])
    assert next_instr.startswith("直走約 12 公尺（經過 1 個路口），然後右轉")
    # the VLM prompt gets the same distance-aware steps
    assert "直走約 12 公尺（經過 1 個路口）" in ctx
    assert "右轉後直走約 5 公尺" in ctx


def test_next_instruction_when_standing_at_the_turn():
    ref_map = _two_aisle_map()
    s = _session_with_route(ref_map, [12, 13])
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)):
        _, next_instr = srv._build_route_context(s, _localized_at(ref_map, 12), detections=[])
    assert next_instr.startswith("右轉後直走約 5 公尺")


def test_missing_edge_distance_falls_back_to_pdr_geometry():
    ref_map = _two_aisle_map()
    ref_map.walkway_edges = [{"from": 10, "to": 11, "distance_m": 0.0}]  # unmeasured edge
    s = _session_with_route(ref_map, [10, 11])
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)):
        _, next_instr = srv._build_route_context(s, _localized_at(ref_map, 10), detections=[])
    assert next_instr.startswith("直走約 6 公尺")   # from pdr_x/pdr_y (0,0)->(0,6)


# ---- goal-in-photo direction must use image width, boxes are in pixels ------

from types import SimpleNamespace


def _px_det(label, x1, x2):
    return SimpleNamespace(label=label, box=[x1, 0, x2, 100], score=0.9)


def test_detect_goal_in_photo_left_side_in_pixels():
    d, names = srv._detect_goal_in_photo([_px_det("milk", 50, 150)], ["milk"], img_w=1000, img_h=800)
    assert d == "左手邊"
    assert names == ["milk"]


def test_detect_goal_in_photo_center_and_right_in_pixels():
    d, _ = srv._detect_goal_in_photo([_px_det("milk", 450, 550)], ["milk"], img_w=1000, img_h=800)
    assert d == "正前方"
    d, _ = srv._detect_goal_in_photo([_px_det("milk", 800, 950)], ["milk"], img_w=1000, img_h=800)
    assert d == "右手邊"


def test_route_context_passes_image_size_to_photo_hint():
    ref_map = _two_aisle_map()
    s = _session_with_route(ref_map, [10, 11, 12, 13])
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)):
        ctx, _ = srv._build_route_context(
            s, _localized_at(ref_map, 10), detections=[_px_det("milk", 50, 150)],
            img_w=1000, img_h=800)
    assert "「左手邊」" in ctx


# ---- instructions start from where the user IS, not where the leg began -----

def test_instruction_advances_when_localized_mid_leg():
    # Leg was planned from the entrance (10) but this photo localizes at aisle 2 (12).
    ref_map = _two_aisle_map()
    s = _session_with_route(ref_map, [10, 11, 12, 13])
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)):
        _, next_instr = srv._build_route_context(s, _localized_at(ref_map, 12), detections=[])
    assert next_instr.startswith("右轉後直走約 5 公尺")


def test_instruction_replans_when_localized_off_the_leg():
    # A node north of aisle 2, not on the planned leg; the user faces south.
    ref_map = _two_aisle_map()
    ref_map.photos[14] = _node(14, 0, 18, "north end")
    ref_map.walkway_edges.append({"from": 12, "to": 14, "distance_m": 6.0})
    s = _session_with_route(ref_map, [10, 11, 12, 13])
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)):
        _, next_instr = srv._build_route_context(s, _localized_at(ref_map, 14, heading=180.0), detections=[])
    # 14 -> 12 is straight ahead (south) for 6 m, then 12 -> 13 (east) is a left turn
    assert next_instr.startswith("直走約 6 公尺，然後左轉")


# ---- the map's verdict must always be reported, not silently dropped -------

def test_at_target_node_says_so_instead_of_nothing():
    ref_map = _two_aisle_map()
    s = _session_with_route(ref_map, [10, 11, 12, 13])
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)):
        ctx, next_instr = srv._build_route_context(s, _localized_at(ref_map, 13), detections=[])
    assert next_instr is not None
    assert "milk" in next_instr and "附近" in next_instr
    assert "已在目標節點" in ctx


def test_failed_localization_is_reported_when_a_map_is_in_use():
    ref_map = _two_aisle_map()
    s = _session_with_route(ref_map, [10, 11, 12, 13])
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)):
        ctx, next_instr = srv._build_route_context(s, None, detections=[])
    assert ctx is None
    assert next_instr and "對不到地圖" in next_instr


def test_no_map_session_stays_silent():
    s = Session(id="t", goal="milk", goal_objects=["milk"], target_objects=["milk"], place=None)
    assert srv._build_route_context(s, None, detections=[]) == (None, None)


# ---- when an editor map exists, the 🗺 line follows it (same route as the phone) ----

from server import editor_map as em


def _editor_graph():
    g = em.build_graph({"place": "test", "walks": [
        {"name": "A", "points": [[0, 0], [0, 6], [0, 12]]},
        {"name": "B", "points": [[-6, 6.3], [0.2, 6.1], [6, 6]]},
    ]})
    # Neo4j node 13 (the session's target) sits by B2 (6,6); node 10 (where the
    # user is) sits by A2 (0,12)
    em.merge_labels(g, [{"nid": 13, "x": 5.5, "y": 6.4, "labels": ["牛奶區吊牌 milk sign"]},
                        {"nid": 10, "x": 0.3, "y": 11.8, "labels": ["door"]}])
    return g


def test_next_instruction_comes_from_editor_map_when_available():
    ref_map = _two_aisle_map()
    s = _session_with_route(ref_map, [10, 11, 12, 13])
    s.goal = "牛奶"; s.target_objects = ["牛奶"]
    # facing south (180°) at A2: south 6 m to A1, then east to B2 = a left turn
    loc = _localized_at(ref_map, 10, heading=180.0)
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)), \
         patch("server.server._editor_graph_for", return_value=_editor_graph()):
        ctx, next_instr = srv._build_route_context(s, loc, detections=[])
    assert next_instr.startswith("直走約 6 公尺，然後左轉")
    assert "左轉後直走約 6 公尺" in ctx


def test_editor_map_at_target_is_reported():
    ref_map = _two_aisle_map()
    s = _session_with_route(ref_map, [10, 11, 12, 13])
    s.goal = "牛奶"; s.target_objects = ["牛奶"]
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(ref_map)), \
         patch("server.server._editor_graph_for", return_value=_editor_graph()):
        _, next_instr = srv._build_route_context(s, _localized_at(ref_map, 13), detections=[])
    assert "附近" in next_instr and "牛奶" in next_instr
