"""/session/{id}/tracking: the tracking page's data (photo fixes, PDR path, map nodes)."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import server as srv
from server.session import Session
from server.tests.test_loc_fusion import _FakeNeo4j, _ref_map


def _get(session, neo4j):
    with patch.object(srv._store, "get", return_value=session), \
         patch("server.server.get_neo4j", return_value=neo4j):
        return TestClient(srv.app).get("/session/trk/tracking")


def test_tracking_returns_the_map_nodes():
    m = _ref_map()
    r = _get(Session(id="trk", goal="milk", goal_objects=["milk"], place="test"), _FakeNeo4j(m))
    assert r.status_code == 200
    assert {n["nid"] for n in r.json()["ref_nodes"]} == set(m.photos)


def test_tracking_without_neo4j_still_answers():
    r = _get(Session(id="trk", goal="milk", goal_objects=["milk"], place="test"), None)
    assert r.status_code == 200 and r.json()["ref_nodes"] == []


def test_tracking_unknown_session():
    assert _get(None, None).status_code == 404


def test_tracking_page_escapes_what_the_phone_sent():
    # goal / place / session ids come from whoever created the session: into innerHTML only via esc()
    from pathlib import Path
    html = (Path(srv.__file__).parent / "tracking.html").read_text(encoding="utf-8")
    for field in ("data.goal", "data.place", "s.goal_objects", "s.id"):
        assert "${" + field not in html
    assert "onclick=" not in html
