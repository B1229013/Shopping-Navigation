import io
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient
from server.server import app
from server.models import VLMAction, VLMResponse
from server.perception import Detection


def _jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), color=(0, 0, 255)).save(buf, "JPEG")
    return buf.getvalue()


def test_get_annotated_photo():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find milk"}).json()["session_id"]
    fake_p = MagicMock()
    fake_p.detect.return_value = [Detection(label="x", box=[0, 0, 5, 5], score=0.5)]
    move = VLMResponse(action=VLMAction.MOVE, guidance="g", question=None, vlm_summary="s")
    with patch("server.server.get_perception", return_value=fake_p), \
         patch("server.server.vlm_decide", return_value=move):
        client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")})
    r = client.get(f"/session/{sid}/photo/0.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 100


def test_get_annotated_photo_missing():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=[]):
        sid = client.post("/session", json={"goal": "x"}).json()["session_id"]
    r = client.get(f"/session/{sid}/photo/99.jpg")
    assert r.status_code == 404
