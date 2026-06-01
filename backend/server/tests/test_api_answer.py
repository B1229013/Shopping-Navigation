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


def test_answer_after_ask_advances_state():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        r = client.post("/session", json={"goal": "find the milk"})
    sid = r.json()["session_id"]

    fake_perception = MagicMock()
    fake_perception.detect.return_value = [Detection(label="shelf", box=[0, 0, 10, 10], score=0.5)]
    ask_resp = VLMResponse(action=VLMAction.ASK, guidance="left or right?",
                           question="Are you near checkout?", vlm_summary="ambiguous")
    move_resp = VLMResponse(action=VLMAction.MOVE, guidance="go left",
                            question=None, vlm_summary="left side")

    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=ask_resp):
        client.post(f"/session/{sid}/photo",
                    files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})

    with patch("server.server.vlm_decide", return_value=move_resp):
        r2 = client.post(f"/session/{sid}/answer", json={"answer": "no"})
    assert r2.status_code == 200
    assert r2.json()["action"] == "MOVE"
    assert r2.json()["guidance"] == "go left"


def test_answer_without_pending_question_409():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=[]):
        r = client.post("/session", json={"goal": "x"})
    sid = r.json()["session_id"]
    r2 = client.post(f"/session/{sid}/answer", json={"answer": "yes"})
    assert r2.status_code == 409


def test_answer_unknown_session_404():
    client = TestClient(app)
    r = client.post("/session/nope/answer", json={"answer": "x"})
    assert r.status_code == 404
