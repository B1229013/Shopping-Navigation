"""「確認到達」 after the phone's step tracking reached the stop (PathFollower within 3 m): the
server never decided ARRIVED, so there is no pending arrival — kind "confirmed_pdr" confirms
the current leg's stop. A plain "confirmed" without a pending ARRIVED stays refused, so a
double tap cannot credit the next item."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import server as srv
from server.server import app
from server.tests.test_multi_target_credit import PLACE, _FakeNeo4j, _session, store  # noqa: F401


def _confirm(client, store, sid, kind):
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(store)):
        return client.post(f"/session/{sid}/confirm", json={"kind": kind})


def test_step_tracking_arrival_credits_the_planned_stop(store):
    client = TestClient(app)
    sid = _session(client, store, "餅乾 x1, 泡麵 x1", ["餅乾", "泡麵"])
    s = srv._store.get(sid)
    planned = s.route_plan.legs[s.current_leg_index].to_node
    r = _confirm(client, store, sid, "confirmed_pdr")
    assert r.status_code == 200, r.text
    s = srv._store.get(sid)
    assert set(s.visited_targets) == {planned}
    assert "已找到「餅乾」" in r.json()["guidance"]


def test_confirmed_without_a_pending_arrival_is_still_refused(store):
    client = TestClient(app)
    sid = _session(client, store, "餅乾 x1", ["餅乾"])
    assert _confirm(client, store, sid, "confirmed").status_code == 409
    assert not srv._store.get(sid).visited_targets


def test_step_tracking_arrival_after_the_walk_is_done_is_refused(store):
    client = TestClient(app)
    sid = _session(client, store, "餅乾 x1", ["餅乾"])
    srv._store.get(sid).arrived = True
    assert _confirm(client, store, sid, "confirmed_pdr").status_code == 409
