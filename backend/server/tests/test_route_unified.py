"""One route graph, a real default start, phase-aware goals and map-checked arrivals."""
from unittest.mock import patch

import networkx as nx
from fastapi.testclient import TestClient

from server import server as srv
from server.neo4j_client import RefMap, RefPhotoNode, RefObject
from server.path_planner import RouteLeg, RoutePlan
from server.session import Session
from server.visual_localization import LocalizationResult


def _node(nid, x, y, label):
    return RefPhotoNode(
        nid=nid, photo_file=f"{nid}.jpg", pdr_x=x, pdr_y=y, heading_deg=0.0,
        session="s", total_steps=0, total_distance_m=0.0,
        objects=[RefObject(label=label, label_norm=label, score=0.9, ocr_text="", role="landmark", grid_cell="c")],
    )


def _map() -> RefMap:
    # A corridor 10→11→12 heading north, milk at 13 east of 12, and 14 down that side
    # aisle — 4.5 m from 11 through the shelf, but only walkable via 13.
    m = RefMap(place="test")
    for nid, x, y, label in [(10, 0, 0, "entrance"), (11, 0, 6, "aisle 1"), (12, 0, 12, "aisle 2"),
                             (13, 5, 12, "milk"), (14, 4.5, 6, "shelf end")]:
        m.photos[nid] = _node(nid, x, y, label)
    m.walkway_edges = [{"from": 10, "to": 11, "distance_m": 6.0}, {"from": 11, "to": 12, "distance_m": 6.0},
                       {"from": 12, "to": 13, "distance_m": 5.0}, {"from": 13, "to": 14, "distance_m": 6.0}]
    return m


class _FakeNeo4j:
    def __init__(self, m):
        self._m = m

    def load_reference_map(self, place=None):
        return self._m

    def get_photo_node(self, nid, place=None):
        return self._m.photos.get(nid)


def _session(targets=(13,), phase="shopping", start=10, legs=None):
    s = Session(id="t", goal="牛奶", goal_objects=["牛奶"], target_objects=["牛奶"], place="test")
    s.target_nodes = list(targets)
    s.target_node_to_item = {t: "牛奶" for t in targets}
    s.phase = phase
    s.route_plan = RoutePlan(legs=legs or [RouteLeg(from_node=start, to_node=targets[0], path=[start, targets[0]],
                                                    cost=0.0, actions=[], purpose="target")],
                             visit_order=list(targets))
    return s


def _neo():
    return patch("server.server.get_neo4j", return_value=_FakeNeo4j(_map()))


def test_route_graph_is_the_walkway_graph_without_shortcuts():
    with _neo():
        g = srv._route_nx_graph("test")
    assert not g.has_edge(11, 14)                    # 4.5 m apart, but a shelf in between
    assert nx.shortest_path(g, 11, 14, weight="weight") == [11, 12, 13, 14]


def test_default_start_is_the_entrance_or_the_first_start_point():
    with _neo():
        assert srv._default_start_wp("test") == 10   # node nearest the origin (no entrance configured)
        with patch("server.server.start_points.load_starts",
                   return_value=[{"id": "s1", "name": "側門", "x": 0.3, "y": 11.7, "heading_deg": 90.0}]):
            assert srv._default_start_wp("test") == 12


def test_arrival_far_from_the_goal_asks_with_the_distance():
    with _neo():
        s = _session()
        q, far_m = srv._far_from_goal_question(s, 10, "「entrance」附近")
        assert q and "17 公尺" in q and "牛奶" in q
        assert far_m is not None and far_m > 15
        assert srv._far_from_goal_question(s, 12) == (None, None)  # 5 m away: plausible
        s.last_loc_log = {"image_evidence": {"10": 0.4, "13": 1.9}}
        assert srv._far_from_goal_question(s, 10) == (None, None)  # the photo itself matched the goal


def test_arrival_very_far_should_silent_move():
    """Distance > ARRIVAL_ASK_MAX_M → caller should MOVE, not ASK."""
    with _neo():
        s = _session()
        q, far_m = srv._far_from_goal_question(s, 10, "「entrance」附近")
        assert q is not None and far_m is not None
        assert far_m > srv.ARRIVAL_ASK_MAX_M


def test_arrival_is_not_gated_without_a_goal_node_for_the_phase():
    with _neo():
        s = _session(phase="checkout")                             # no checkout leg on this map
        assert srv._far_from_goal_question(s, 10) == (None, None)


def test_confirm_answers():
    for text, want in [("是", True), ("是的，到了", True), ("有", True), ("yes", True),
                       ("不是", False), ("沒有", False), ("還沒", False), ("no", False),
                       ("在哪裡？", None), ("", None)]:
        assert srv._confirm_answer(text) is want, text


def test_goal_node_follows_the_phase():
    s = _session(phase="checkout")
    assert srv._current_goal_node_id(s) is None                    # not the last shopping target
    s.phase = "shopping"
    assert srv._current_goal_node_id(s) == 13


def test_checkout_at_the_exit_node_uses_the_exit_leg():
    # A7: checkout and exit are both node 25, so the plan has one leg labelled "exit"
    s = _session(phase="checkout", legs=[
        RouteLeg(from_node=10, to_node=13, path=[10, 11, 12, 13], cost=17.0, actions=[], purpose="target"),
        RouteLeg(from_node=13, to_node=10, path=[13, 12, 11, 10], cost=17.0, actions=[], purpose="exit")])
    s.checkout_node = s.exit_node = 10
    s.current_leg_index = 1
    assert srv._current_goal_node_id(s) == 10
    s.phase = "exit"
    assert srv._current_goal_node_id(s) == 10


def test_checkout_without_a_checkout_leg_does_not_claim_the_checkout():
    with _neo():
        s = _session(phase="checkout")
        s.visited_targets = {13}
        loc = LocalizationResult(matched_nid=13, confidence=0.9, method="test", reasoning="",
                                 ref_node=_map().photos[13], matched_heading=0.0, matched_slot="front",
                                 heading_confidence=0.8)
        _, next_instr = srv._build_route_context(s, loc, detections=[])
    assert not next_instr or "已在收銀台附近" not in next_instr


def test_start_endpoint_replans_from_the_chosen_start():
    c = TestClient(srv.app)
    with patch("server.server.decompose_goal", return_value=["牛奶"]):
        sid = c.post("/session", json={"goal": "牛奶 x1", "place": ""}).json()["session_id"]
    s = srv._store.get(sid)
    s.place = "test"
    s.target_nodes = [13]
    s.route_plan = RoutePlan(legs=[RouteLeg(from_node=10, to_node=13, path=[10, 11, 12, 13], cost=17.0,
                                            actions=[], purpose="target")], visit_order=[13])
    with _neo():
        r = c.post(f"/session/{sid}/start", json={"x": 0.2, "y": 11.9})
    body = r.json()
    assert r.status_code == 200 and body["start_node"] == 12 and body["replanned"]
    assert s.route_plan.legs[0].from_node == 12 and s.route_plan.legs[0].path == [12, 13]
