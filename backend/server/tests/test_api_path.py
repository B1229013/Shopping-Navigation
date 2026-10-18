"""GET /session/{id}/path — the route on the editor map the phone follows."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import editor_map as em
from server.server import app


def _graph():
    g = em.build_graph({"place": "test", "walks": [
        {"name": "A", "points": [[0, 0], [0, 6], [0, 12]]},
        {"name": "B", "points": [[-6, 6.3], [0.2, 6.1], [6, 6]]},
    ]})
    em.merge_labels(g, [{"nid": 3, "x": 5.5, "y": 6.4, "labels": ["牛奶區吊牌 milk sign", "牛奶紙盒"]},
                        {"nid": 7, "x": 0.3, "y": 11.8, "labels": ["麵包 bread"]}])
    return g


def _session(client, goal="牛奶 x1"):
    with patch("server.server.decompose_goal", return_value=["牛奶", "乳製品區"]):
        return client.post("/session", json={"goal": goal, "place": ""}).json()["session_id"]


def test_path_from_entrance_to_goal_waypoint():
    client = TestClient(app)
    sid = _session(client)
    with patch("server.server._editor_graph_for", return_value=_graph()):
        r = client.get(f"/session/{sid}/path")
    assert r.status_code == 200
    body = r.json()
    assert body["goal_item"] == "牛奶"
    assert body["start"]["source"] == "entrance"
    assert body["polyline"][0] == [0.0, 0.0] and body["polyline"][-1] == [6.0, 6.0]
    assert body["turns"][0]["direction"] == "straight"
    assert body["distance_m"] > 11
    assert len(body["nodes"]) == 6 and body["edges"]        # whole map for the mini-map
    assert body["target"]["products"][0].startswith("牛奶區")


def test_path_replans_from_phone_position():
    client = TestClient(app)
    sid = _session(client)
    with patch("server.server._editor_graph_for", return_value=_graph()):
        r = client.get(f"/session/{sid}/path", params={"x": 0.3, "y": 5.5, "heading": 0})
    body = r.json()
    assert body["start"]["source"] == "phone"
    assert body["polyline"][0] == [0.0, 6.0]                # snapped to the nearest waypoint
    assert body["turns"][0]["direction"] in ("right", "slight_right")


def test_path_starts_from_last_photo_localization():
    client = TestClient(app)
    sid = _session(client)
    from server.server import _store
    _store.get(sid).last_corrected_nid = 7                  # bread node (0.3, 11.8) → waypoint (0, 12)
    with patch("server.server._editor_graph_for", return_value=_graph()):
        r = client.get(f"/session/{sid}/path")
    body = r.json()
    assert body["start"]["source"] == "photo"
    assert body["polyline"][0] == [0.0, 12.0]


def _set_leg(sid, to_node, purpose="target", phase="shopping"):
    """Give the session a one-leg route plan, as session start does when the goal matches a node."""
    from types import SimpleNamespace
    from server.server import _store
    s = _store.get(sid)
    s.route_plan = SimpleNamespace(legs=[SimpleNamespace(to_node=to_node, purpose=purpose, path=[to_node])],
                                   visit_order=[to_node])
    s.current_leg_index = 0
    s.phase = phase


def test_path_heads_for_the_node_chosen_at_session_start():
    # The text guidance targets the leg's node; the mini-map must not pick another
    # node just because its labels mention the item more often.
    client = TestClient(app)
    sid = _session(client)
    _set_leg(sid, 7)                                        # node 7 → waypoint (0, 12), no 「牛奶」 label
    with patch("server.server._editor_graph_for", return_value=_graph()):
        r = client.get(f"/session/{sid}/path")
    body = r.json()
    assert body["target"]["neo_nid"] == 7
    assert body["polyline"][-1] == [0.0, 12.0]


def test_path_ignores_a_shopping_leg_once_in_checkout():
    # No checkout node on the map: the plan's last leg is still a shopping target, so
    # the label search for 收銀台 decides (and here finds nothing).
    client = TestClient(app)
    sid = _session(client)
    _set_leg(sid, 7, purpose="target", phase="checkout")
    with patch("server.server._editor_graph_for", return_value=_graph()):
        r = client.get(f"/session/{sid}/path")
    assert r.status_code == 404
    assert r.json()["error"] == "goal_not_on_map"


def test_path_404_when_no_editor_map_for_place():
    client = TestClient(app)
    sid = _session(client)
    with patch("server.server._editor_graph_for", return_value=None):
        r = client.get(f"/session/{sid}/path")
    assert r.status_code == 404
    assert r.json()["error"] == "no_editor_map"


def test_path_404_when_goal_not_on_map():
    client = TestClient(app)
    sid = _session(client, goal="泡麵 x1")
    with patch("server.server._editor_graph_for", return_value=_graph()), \
         patch("server.server.decompose_goal", return_value=["泡麵"]):
        r = client.get(f"/session/{sid}/path")
    assert r.status_code == 404
    assert r.json()["error"] == "goal_not_on_map"
