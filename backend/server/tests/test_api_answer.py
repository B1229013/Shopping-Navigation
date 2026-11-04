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


def _session_at_arrival_question(client):
    """A session whose last photo was a downgraded ARRIVED: the 有／沒有 question is open."""
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find the milk"}).json()["session_id"]
    fake_perception = MagicMock()
    fake_perception.detect.return_value = [Detection(label="shelf", box=[0, 0, 10, 10], score=0.5)]
    arrived = VLMResponse(action=VLMAction.ARRIVED, guidance="here", question=None, vlm_summary="milk")
    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=arrived):
        r = client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    assert r.json()["action"] == "ASK" and r.json()["confirm_question"] is True
    return sid


def test_yes_to_arrival_question_needs_no_second_confirm():
    client = TestClient(app)
    sid = _session_at_arrival_question(client)
    r = client.post(f"/session/{sid}/answer", json={"answer": "有"})
    assert r.status_code == 200
    body = r.json()
    # finished like POST /confirm "confirmed": the item is found and the phase moves
    # on (checkout here), instead of an ARRIVED that the app has to confirm again
    assert body["action"] == "MOVE"
    assert body["phase"] == "checkout"
    assert body["visited_item_names"] == ["find the milk"]
    assert client.post(f"/session/{sid}/confirm", json={"kind": "confirmed"}).status_code == 409
    # and the next photo is taken, not refused with 409 confirm_pending
    fake_perception = MagicMock()
    fake_perception.detect.return_value = []
    move = VLMResponse(action=VLMAction.MOVE, guidance="go", question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=move):
        r2 = client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    assert r2.status_code == 200


def test_no_to_arrival_question_keeps_searching():
    client = TestClient(app)
    sid = _session_at_arrival_question(client)
    r = client.post(f"/session/{sid}/answer", json={"answer": "沒有"})
    assert r.status_code == 200
    assert r.json()["action"] == "MOVE"
    assert r.json()["question"] is None
