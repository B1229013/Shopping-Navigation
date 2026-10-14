"""An unsure localization asks for a second photo, and the next photo is combined with it."""
import io
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from server import server as srv
from server.models import VLMAction, VLMResponse
from server.perception import Detection
from server.visual_localization import LocalizationResult


def _jpg():
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(1).random((300, 200)) * 255).astype(np.uint8)).convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def _result(conf, ev):
    return LocalizationResult(matched_nid=3, confidence=conf, method="grid+v2", reasoning="", image_evidence=ev)


def test_unsure_photo_asks_for_second_and_next_photo_combines():
    client = TestClient(srv.app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "milk", "place": ""}).json()["session_id"]
    perception = MagicMock()
    perception.detect.return_value = [Detection(label=f"shelf{i}", box=[0, 0, 50, 50], score=0.8) for i in range(4)]
    resp = VLMResponse(action=VLMAction.MOVE, guidance="go", question=None, vlm_summary="")
    calls = []

    def fake_early(s, sid_, *a, **kw):
        calls.append(kw.get("extra_evidence"))
        return _result(0.2, {3: 1.0}) if len(calls) == 1 else _result(0.7, {3: 1.0})

    with patch("server.server.get_perception", return_value=perception), \
         patch("server.server.vlm_decide", return_value=resp), \
         patch("server.server._run_early_localization", side_effect=fake_early), \
         patch("server.server._build_route_context", return_value=(None, None)):
        first = client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")},
                            data={"pdr_x": "1.0", "pdr_y": "1.0"}).json()
        second = client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")},
                             data={"pdr_x": "1.5", "pdr_y": "1.0"}).json()
    assert first["second_photo_hint"] and calls[0] is None
    assert calls[1] == {3: 1.0}                      # first photo's evidence carried over
    assert second["second_photo_hint"] is None


def _post(client, sid, **data):
    return client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")},
                       data=data).json()


def _run(turns, action=VLMAction.MOVE):
    """turns: [(form data, detections or None)] → (responses, extra_evidence per localization)."""
    client = TestClient(srv.app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "milk", "place": ""}).json()["session_id"]
    shelves = [Detection(label=f"shelf{i}", box=[0, 0, 50, 50], score=0.8) for i in range(4)]
    perception = MagicMock()
    resp = VLMResponse(action=action, guidance="go", question=None, vlm_summary="")
    calls = []

    def fake_early(s, sid_, *a, **kw):
        calls.append(kw.get("extra_evidence"))
        return _result(0.2, {3: 1.0})

    out = []
    with patch("server.server.get_perception", return_value=perception), \
         patch("server.server.vlm_decide", return_value=resp), \
         patch("server.server._run_early_localization", side_effect=fake_early), \
         patch("server.server._build_route_context", return_value=(None, None)):
        for data, dets in turns:
            perception.detect.return_value = dets or shelves
            out.append(_post(client, sid, **data))
    return out, calls


def test_no_second_photo_without_a_known_phone_position():
    # gallery photo, or a phone that sends no step position: "same spot" can't be checked
    for first in ({"source": "gallery", "pdr_x": "1", "pdr_y": "1"}, {}):
        out, calls = _run([(first, None), ({"pdr_x": "1", "pdr_y": "1"}, None)])
        assert out[0]["second_photo_hint"] is None
        assert calls == [None, None]


def test_second_photo_not_combined_when_movement_is_unknown():
    out, calls = _run([({"pdr_x": "1", "pdr_y": "1"}, None), ({}, None)])
    assert out[0]["second_photo_hint"] and calls == [None, None]
    out, calls = _run([({"pdr_x": "1", "pdr_y": "1"}, None),
                       ({"source": "gallery", "pdr_x": "1", "pdr_y": "1"}, None)])
    assert calls == [None, None]


def test_second_photo_only_asked_while_walking():
    out, _ = _run([({"pdr_x": "1", "pdr_y": "1"}, None)], action=VLMAction.ASK)
    assert out[0]["second_photo_hint"] is None


def test_blocked_retake_keeps_the_second_photo_request():
    person = [Detection(label="person", box=[0, 0, 300, 300], score=0.9),
              Detection(label="shelf", box=[0, 350, 50, 400], score=0.8)]
    out, calls = _run([({"pdr_x": "1", "pdr_y": "1"}, None),
                       ({"pdr_x": "1", "pdr_y": "1"}, person),     # blocked: not localized at all
                       ({"pdr_x": "1.5", "pdr_y": "1"}, None)])
    assert out[1]["retake_hint"] and len(calls) == 2
    assert calls[1] == {3: 1.0}                                    # third photo still combines
