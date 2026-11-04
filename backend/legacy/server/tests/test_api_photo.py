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


def _start_session(client) -> str:
    with patch("server.server.decompose_goal", return_value=["milk"]):
        r = client.post("/session", json={"goal": "find the milk"})
    return r.json()["session_id"]


def test_post_photo_returns_move(tmp_path):
    client = TestClient(app)
    sid = _start_session(client)
    fake_perception = MagicMock()
    fake_perception.detect.return_value = [
        Detection(label="shelf", box=[0, 0, 50, 50], score=0.7)
    ]
    fake_vlm_resp = VLMResponse(action=VLMAction.MOVE, guidance="walk forward",
                                question=None, vlm_summary="aisle")
    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=fake_vlm_resp):
        r = client.post(
            f"/session/{sid}/photo",
            files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "MOVE"
    assert body["guidance"] == "walk forward"
    assert body["node_id"] == 0


def test_post_photo_unknown_session():
    client = TestClient(app)
    r = client.post("/session/nope/photo",
                    files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    assert r.status_code == 404


def test_post_photo_after_arrived_409(tmp_path):
    client = TestClient(app)
    sid = _start_session(client)
    fake_perception = MagicMock()
    # goal is "milk"; a confident milk detection corroborates the ARRIVED so the
    # arrival gate (verify_arrival) honors it instead of asking to confirm.
    fake_perception.detect.return_value = [
        Detection(label="milk", box=[10, 10, 60, 60], score=0.8)
    ]
    arrived_resp = VLMResponse(action=VLMAction.ARRIVED, guidance="found", question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=arrived_resp):
        client.post(f"/session/{sid}/photo",
                    files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
        r2 = client.post(f"/session/{sid}/photo",
                         files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    assert r2.status_code == 409
