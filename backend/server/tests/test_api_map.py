import io
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient
from server.server import app
from server.models import VLMAction, VLMResponse
from server.perception import Detection


def _make_jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(127, 127, 127)).save(buf, "JPEG")
    return buf.getvalue()


def _seed_session_with_one_photo(client) -> str:
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find the milk"}).json()["session_id"]
    fake_p = MagicMock()
    fake_p.detect.return_value = [Detection(label="shelf", box=[0, 0, 10, 10], score=0.5)]
    move = VLMResponse(action=VLMAction.MOVE, guidance="g", question=None, vlm_summary="aisle")
    with patch("server.server.get_perception", return_value=fake_p), \
         patch("server.server.vlm_decide", return_value=move):
        client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    return sid


def test_get_map_json():
    client = TestClient(app)
    sid = _seed_session_with_one_photo(client)
    r = client.get(f"/session/{sid}/map")
    assert r.status_code == 200
    body = r.json()
    assert len(body["nodes"]) == 1
    assert body["edges"] == []
    assert body["current_node"] == 0


def test_get_map_png():
    client = TestClient(app)
    sid = _seed_session_with_one_photo(client)
    r = client.get(f"/session/{sid}/map?format=png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


def test_get_map_unknown_session_404():
    client = TestClient(app)
    r = client.get("/session/nope/map")
    assert r.status_code == 404
