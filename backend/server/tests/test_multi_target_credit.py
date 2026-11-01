"""Multi-item arrival must credit the stop the user actually reached.

_current_goal_objects deliberately hands the VLM every remaining item so it can
call ARRIVED when the user passes a non-current target. The arrival handler then
has to credit THAT stop — crediting the planned leg marks the wrong item found,
leaves the item being stood at outstanding, and (before the re-plan) strands the
skipped stop off the leg list entirely. None of this can happen with one item,
which is why single-target navigation was accurate while multi-item was not.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from server.server import app
from server import server as srv
from server.neo4j_client import RefMap, RefPhotoNode, RefObject

PLACE = "test"


def _node(nid, x, y, *labels):
    return RefPhotoNode(
        nid=nid, photo_file=f"{nid}.jpg", pdr_x=x, pdr_y=y, heading_deg=0.0,
        session="s", total_steps=0, total_distance_m=0.0,
        objects=[RefObject(label=l, label_norm=l, score=0.9, ocr_text=l,
                           role="product", grid_cell="c") for l in labels],
    )


@pytest.fixture
def store():
    # One aisle: entrance 1, 餅乾 at 2, 泡麵 at 3, 啤酒 at 4.
    m = RefMap(place=PLACE)
    m.photos[1] = _node(1, 0, 0, "entrance")
    m.photos[2] = _node(2, 0, 4, "餅乾")
    m.photos[3] = _node(3, 0, 10, "泡麵")
    m.photos[4] = _node(4, 0, 16, "啤酒")
    m.walkway_edges = [{"from": 1, "to": 2, "distance_m": 4.0},
                       {"from": 2, "to": 3, "distance_m": 6.0},
                       {"from": 3, "to": 4, "distance_m": 6.0}]
    return m


class _FakeNeo4j:
    def __init__(self, m): self._m = m
    def load_reference_map(self, place=None): return self._m


def _session(client, store, goal, items):
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(store)), \
         patch("server.server.decompose_goal", return_value=items), \
         patch("server.server._default_start_wp", return_value=1):
        r = client.post("/session", json={"goal": goal, "place": PLACE})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _confirm_at(client, store, sid, nid):
    s = srv._store.get(sid)
    s.pending_arrival = True
    s.goal_node = nid
    s.last_node_id = nid
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(store)):
        r = client.post(f"/session/{sid}/confirm", json={"kind": "confirmed"})
    assert r.status_code == 200, r.text
    return r.json()


def test_credits_the_stop_actually_reached(store):
    client = TestClient(app)
    sid = _session(client, store, "餅乾 x1, 泡麵 x1, 啤酒 x1", ["餅乾", "泡麵", "啤酒"])
    s = srv._store.get(sid)
    planned = s.route_plan.legs[s.current_leg_index].to_node
    assert planned == 2, "nearest stop should be planned first"

    body = _confirm_at(client, store, sid, 3)   # passed 泡麵 on the way to 餅乾
    s = srv._store.get(sid)
    assert 3 in s.visited_targets, "the stop actually reached was not credited"
    assert 2 not in s.visited_targets, "credited a stop the user never reached"
    assert "泡麵" in (body.get("guidance") or body.get("message") or "")


def test_the_skipped_stop_is_not_orphaned(store):
    client = TestClient(app)
    sid = _session(client, store, "餅乾 x1, 泡麵 x1, 啤酒 x1", ["餅乾", "泡麵", "啤酒"])
    _confirm_at(client, store, sid, 3)
    s = srv._store.get(sid)
    routed = {l.to_node for l in s.route_plan.legs[s.current_leg_index:]}
    assert set(s.remaining_targets) <= routed, "a remaining item is off the plan"


def test_next_leg_is_an_unvisited_target(store):
    # _replan_from resets current_leg_index to 0; the advance must not then skip it.
    client = TestClient(app)
    sid = _session(client, store, "餅乾 x1, 泡麵 x1, 啤酒 x1", ["餅乾", "泡麵", "啤酒"])
    _confirm_at(client, store, sid, 3)
    s = srv._store.get(sid)
    nxt = s.route_plan.legs[s.current_leg_index].to_node
    assert nxt not in s.visited_targets, "advanced onto an already-credited stop"


def test_in_order_arrival_still_credits_the_planned_stop(store):
    client = TestClient(app)
    sid = _session(client, store, "餅乾 x1, 泡麵 x1", ["餅乾", "泡麵"])
    s = srv._store.get(sid)
    planned = s.route_plan.legs[s.current_leg_index].to_node
    _confirm_at(client, store, sid, planned)
    s = srv._store.get(sid)
    assert s.visited_targets == {planned}


def test_single_item_unaffected(store):
    client = TestClient(app)
    sid = _session(client, store, "泡麵 x1", ["泡麵"])
    s = srv._store.get(sid)
    planned = s.route_plan.legs[s.current_leg_index].to_node
    _confirm_at(client, store, sid, planned)
    s = srv._store.get(sid)
    assert s.visited_targets == {planned}
    assert s.remaining_targets == []


def test_arrival_at_a_non_target_node_credits_the_plan(store):
    client = TestClient(app)
    sid = _session(client, store, "餅乾 x1, 泡麵 x1", ["餅乾", "泡麵"])
    s = srv._store.get(sid)
    planned = s.route_plan.legs[s.current_leg_index].to_node
    _confirm_at(client, store, sid, 99)   # not a target at all
    s = srv._store.get(sid)
    assert s.visited_targets == {planned}
