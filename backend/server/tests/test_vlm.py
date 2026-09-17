import io
import json
from unittest.mock import patch, MagicMock
from PIL import Image
from server.vlm import decide, warm_up
from server.models import VLMAction


def _make_jpeg(tmp_path):
    """Create a valid 8x8 JPEG file and return its path."""
    p = tmp_path / "p.jpg"
    img = Image.new("RGB", (8, 8), color=(128, 128, 128))
    img.save(str(p), format="JPEG")
    return p


def _mock_openai(content_text: str):
    """Return a mock requests.Response matching OpenAI chat completions format."""
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "choices": [{"message": {"content": content_text}}],
    }
    r.raise_for_status = lambda: None
    return r


def test_decide_parses_arrived(tmp_path):
    img = _make_jpeg(tmp_path)
    payload = json.dumps({"action": "ARRIVED", "guidance": "milk on right",
                          "question": None, "vlm_summary": "dairy"})
    with patch("server.vlm.requests.post", return_value=_mock_openai(payload)):
        resp = decide(image_path=str(img), goal="milk", goal_objects=["milk"],
                      topomap_summary="", detections_summary="milk x1",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.ARRIVED
    assert resp.guidance == "milk on right"
    assert resp.question is None


def test_decide_falls_back_on_unparseable(tmp_path):
    img = _make_jpeg(tmp_path)
    with patch("server.vlm.requests.post", return_value=_mock_openai("hmm let me think...")):
        resp = decide(image_path=str(img), goal="g", goal_objects=[],
                      topomap_summary="", detections_summary="",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.MOVE


def test_decide_falls_back_on_exception(tmp_path):
    img = _make_jpeg(tmp_path)
    with patch("server.vlm.requests.post", side_effect=Exception("boom")):
        resp = decide(image_path=str(img), goal="g", goal_objects=[],
                      topomap_summary="", detections_summary="",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.MOVE


def test_decide_includes_prior_answer_block(tmp_path):
    img = _make_jpeg(tmp_path)
    payload = json.dumps({"action": "MOVE", "guidance": "ok", "question": None, "vlm_summary": "s"})
    captured = {}

    def fake_post(url, **kw):
        captured["body"] = kw["json"]
        return _mock_openai(payload)

    with patch("server.vlm.requests.post", side_effect=fake_post):
        decide(image_path=str(img), goal="g", goal_objects=[],
               topomap_summary="", detections_summary="",
               prior_question="Are you in dairy?", prior_answer="yes")
    content_parts = captured["body"]["messages"][0]["content"]
    prompt_text = "".join(
        p["text"] for p in content_parts if p.get("type") == "text"
    )
    assert "Are you in dairy?" in prompt_text
    assert "yes" in prompt_text


def test_decide_uses_openai_format(tmp_path):
    img = _make_jpeg(tmp_path)
    payload = json.dumps({"action": "MOVE", "guidance": "ok", "question": None, "vlm_summary": "s"})
    captured = {}

    def fake_post(url, **kw):
        captured["body"] = kw["json"]
        return _mock_openai(payload)

    with patch("server.vlm.requests.post", side_effect=fake_post):
        decide(image_path=str(img), goal="g", goal_objects=[],
               topomap_summary="", detections_summary="",
               prior_question=None, prior_answer=None)
    assert "model" in captured["body"]
    assert "messages" in captured["body"]


def test_decide_retries_once_then_parses(tmp_path):
    img = _make_jpeg(tmp_path)
    good = json.dumps({"action": "MOVE", "guidance": "ok2", "question": None, "vlm_summary": "s"})
    responses = [_mock_openai("garbage, no json here"), _mock_openai(good)]
    with patch("server.vlm.requests.post", side_effect=responses) as mock_post:
        resp = decide(image_path=str(img), goal="g", goal_objects=[],
                      topomap_summary="", detections_summary="",
                      prior_question=None, prior_answer=None)
    assert resp.guidance == "ok2"
    assert mock_post.call_count == 2


def test_warm_up_posts_to_openai():
    with patch("server.vlm.requests.post", return_value=_mock_openai("{}")) as mp:
        warm_up()
    assert mp.called


def test_warm_up_swallows_errors():
    with patch("server.vlm.requests.post", side_effect=Exception("api down")):
        warm_up()  # must not raise
