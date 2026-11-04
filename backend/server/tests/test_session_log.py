"""Per-session records: output/sessions/<id>/turns.jsonl (one line per turn) and
session.json (latest state), written alongside the photos as a log."""
import io
import json
import math
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server import config, session_log
from server import server as srv
from server.models import VLMAction, VLMResponse
from server.perception import Detection
from server.visual_localization import LocalizationResult


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    monkeypatch.setattr(config, "OUTPUT_ROOT", root)     # photos (ensure_output_dir)
    monkeypatch.setattr(srv, "OUTPUT_ROOT", root)        # the records
    return root


def _jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(127, 127, 127)).save(buf, "JPEG")
    return buf.getvalue()


def _lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


# ── module helpers ──────────────────────────────────────────────────────

def test_append_turn_is_json_lines_and_cleans_values(tmp_path):
    session_log.append_turn(tmp_path, {"kind": "photo", "x": float("nan"), "nodes": {3, 1}, "ev": {7: 1.234567}})
    session_log.append_turn(tmp_path, {"kind": "confirm"})
    lines = _lines(tmp_path / "turns.jsonl")
    assert [l["kind"] for l in lines] == ["photo", "confirm"]
    assert lines[0]["x"] is None and lines[0]["nodes"] == [1, 3] and lines[0]["ev"] == {"7": 1.2346}
    assert "time" in lines[0] and "t" in lines[0]


def test_logging_failure_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    session_log.append_turn(blocker / "sub", {"kind": "photo"})       # cannot create → warning only
    session_log.write_session(blocker / "sub", object())


def test_localization_record_keeps_evidence_and_candidates():
    lr = LocalizationResult(matched_nid=7, confidence=0.81, method="server_dino+v2", reasoning="r",
                            top_candidates=[(7, 0.81, "x"), (3, 0.1, "y")], estimated_xy=(1.0, 2.0),
                            image_evidence={7: 2.0, 3: 1.5})
    rec = session_log.localization_record(lr, sigma=2.0, reachable={3, 7}, server_evidence={7: 2.0})
    assert rec["result"]["matched_nid"] == 7 and rec["result"]["top_candidates"] == [[7, 0.81], [3, 0.1]]
    assert rec["image_evidence"] == {"7": 2.0, "3": 1.5}
    assert rec["image_evidence_source"] == "server_dino" and rec["reachable_nodes"] == [3, 7]
    assert session_log.localization_record(None)["result"] is None


# ── through the API ─────────────────────────────────────────────────────

def test_start_photo_confirm_are_recorded(out_root):
    client = TestClient(srv.app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find the milk", "place": ""}).json()["session_id"]
    perception = MagicMock()
    perception.detect.return_value = [Detection(label="milk", box=[10, 10, 60, 60], score=0.8)]
    arrived = VLMResponse(action=VLMAction.ARRIVED, guidance="here", question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=perception), \
         patch("server.server.vlm_decide", return_value=arrived):
        r = client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")},
                        data={"pdr_x": "1.5", "pdr_y": "2.5", "map_x": "3", "map_y": "4",
                              "dist_since_fix": "6", "fix_node": "25", "heading": "90"})
    assert r.status_code == 200
    client.post(f"/session/{sid}/confirm", json={"kind": "confirmed"})

    d = out_root / sid
    lines = _lines(d / "turns.jsonl")
    assert [l["kind"] for l in lines] == ["start", "photo", "confirm"]
    photo = lines[1]
    assert photo["request"] == {"source": "camera", "pdr_x": 1.5, "pdr_y": 2.5, "map_x": 3.0, "map_y": 4.0,
                                "dist_since_fix": 6.0, "fix_node": 25, "heading": 90.0,
                                "dino_node_hint": None, "dino_topk": None}
    assert photo["response"]["action"] == "ARRIVED"
    assert photo["localization"] is None            # explore mode: no map, nothing localized
    assert lines[2]["request"] == {"kind": "confirmed"}
    state = json.loads((d / "session.json").read_text(encoding="utf-8"))
    assert state["session_id"] == sid and state["goal"] == "find the milk"
    assert (d / "photo" / "0.jpg").exists()          # photos still land next to the records


def test_photo_turn_carries_the_localization_record(out_root):
    client = TestClient(srv.app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find the milk", "place": ""}).json()["session_id"]
    s = srv._store.get(sid)
    loc = {"result": {"matched_nid": 7, "confidence": 0.8}, "sensor_sigma_m": 1.3}

    async def fake_handle(*args, **kwargs):
        s.last_loc_log = loc
        return srv.TurnResponse(action=VLMAction.MOVE, guidance="go", node_id=0)

    with patch("server.server._handle_photo", side_effect=fake_handle):
        client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")},
                    data={"source": "gallery"})
    photo = _lines(out_root / sid / "turns.jsonl")[-1]
    assert photo["localization"] == loc and photo["request"]["source"] == "gallery"


def test_unknown_session_writes_nothing(out_root):
    client = TestClient(srv.app)
    r = client.post("/session/nope/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")})
    assert r.status_code == 404
    assert not (out_root / "nope").exists()
