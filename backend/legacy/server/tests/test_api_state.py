from unittest.mock import patch
from fastapi.testclient import TestClient
from server.server import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_session_state():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find the milk"}).json()["session_id"]
    r = client.get(f"/session/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == sid
    assert body["goal"] == "find the milk"
    assert body["arrived"] is False
    assert body["history"] == []


def test_get_session_unknown_404():
    client = TestClient(app)
    r = client.get("/session/nope")
    assert r.status_code == 404
