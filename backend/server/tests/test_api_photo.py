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
    # place="" = explore mode: never touch a live Neo4j map from unit tests
    with patch("server.server.decompose_goal", return_value=["milk"]):
        r = client.post("/session", json={"goal": "find the milk", "place": ""})
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


# ---- landmark words in goal_objects must not count as "arrived" ------------

def _start_session_with_objects(client, goal, objects) -> str:
    with patch("server.server.decompose_goal", return_value=objects):
        r = client.post("/session", json={"goal": goal, "place": ""})
    return r.json()["session_id"]


def _post_arrived_photo(client, sid, detections):
    fake_perception = MagicMock()
    fake_perception.detect.return_value = detections
    arrived_resp = VLMResponse(action=VLMAction.ARRIVED, guidance="就在您的右手邊",
                               question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=arrived_resp):
        return client.post(f"/session/{sid}/photo",
                           files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})


def test_arrived_on_landmark_only_is_downgraded_to_ask():
    # decompose_goal returns section/landmark words alongside the product; a
    # confident "cooler" is NOT evidence that the milk is in view.
    client = TestClient(app)
    sid = _start_session_with_objects(
        client, "find the milk", ["milk", "milk carton", "dairy section", "cooler"])
    r = _post_arrived_photo(client, sid, [
        Detection(label="cooler", box=[10, 10, 60, 60], score=0.9),
        Detection(label="shelf", box=[0, 0, 50, 50], score=0.9),
    ])
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "ASK"
    assert body["question"]


def test_arrived_on_product_variant_stays_arrived():
    # ...but a confident detection of the product itself (or a variant) does count.
    client = TestClient(app)
    sid = _start_session_with_objects(
        client, "find the milk", ["milk", "milk carton", "dairy section", "cooler"])
    r = _post_arrived_photo(client, sid, [
        Detection(label="milk carton", box=[10, 10, 60, 60], score=0.9),
    ])
    assert r.status_code == 200
    assert r.json()["action"] == "ARRIVED"


def test_vlm_prompt_receives_context_landmarks_separately():
    client = TestClient(app)
    sid = _start_session_with_objects(
        client, "find the milk", ["milk", "milk carton", "dairy section", "cooler"])
    fake_perception = MagicMock()
    fake_perception.detect.return_value = [Detection(label="shelf", box=[0, 0, 50, 50], score=0.7)]
    move = VLMResponse(action=VLMAction.MOVE, guidance="walk", question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=move) as decide:
        client.post(f"/session/{sid}/photo",
                    files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    kwargs = decide.call_args.kwargs
    assert kwargs["goal_objects"] == ["milk", "milk carton"]
    assert kwargs["context_objects"] == ["dairy section", "cooler"]


def test_vlm_only_mode_weak_goal_detection_is_not_rescued_by_crop_verify():
    # Mode B: the VLM perceives a low-score "泡麵" (0.41) and would confirm its own
    # crop; that self-confirmation must not turn into ARRIVED.
    from server.vlm import VLMPerception, VLMDetectedObject
    client = TestClient(app)
    sid = _start_session_with_objects(client, "泡麵 x1", ["泡麵", "零食區"])
    perception = VLMPerception(scene_description="snack aisle", detections=[
        VLMDetectedObject(label="泡麵", bbox=[516, 832, 792, 1152], score=0.41),
        VLMDetectedObject(label="left shelf", bbox=[0, 288, 516, 1520], score=0.98),
    ], ocr_texts=[])
    arrived = VLMResponse(action=VLMAction.ARRIVED, guidance="泡麵就在您前方", question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=None), \
         patch("server.server._vlm_perceive", return_value=perception), \
         patch("server.server.vlm_decide", return_value=arrived), \
         patch("server.server.GOAL_CROP_VERIFY", True), \
         patch("server.server.verify_goal_detection", return_value=True) as crop:
        r = client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["action"] == "ASK"
    crop.assert_not_called()        # below the score floor → not even worth asking
