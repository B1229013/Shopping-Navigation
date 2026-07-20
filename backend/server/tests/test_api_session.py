from unittest.mock import patch
from fastapi.testclient import TestClient
from server.server import app, get_store


def test_post_session_returns_id_and_decomposed_goals():
    with patch("server.server.decompose_goal", return_value=["milk", "dairy"]):
        client = TestClient(app)
        r = client.post("/session", json={"goal": "find the milk"})
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert body["goal_objects"] == ["milk", "dairy"]
    assert body["action"] == "TAKE_PHOTO"


def test_post_session_rejects_empty_goal():
    client = TestClient(app)
    r = client.post("/session", json={"goal": ""})
    assert r.status_code == 400
