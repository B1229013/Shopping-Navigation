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


# ---- prompt separates the product from nearby-landmark context -------------

from server.vlm import _build_prompt


def _prompt(**kw):
    base = dict(goal="找到：milk", goal_objects=["milk", "milk carton"],
                topomap_summary="", detections_summary="cooler (85%) - right middle, near",
                prior_question=None, prior_answer=None)
    base.update(kw)
    return _build_prompt(**base)


def test_prompt_lists_context_landmarks_as_not_the_target():
    text = _prompt(context_objects=["dairy section", "cooler"])
    # product line carries only the product words...
    assert "要找的物品特徵：milk, milk carton" in text
    # ...and landmarks are named separately, explicitly marked as not-arrival evidence
    assert "dairy section, cooler" in text
    assert "不代表已到達" in text


def test_prompt_without_context_has_no_landmark_line():
    text = _prompt()
    assert "不代表已到達" not in text


# ---- perception must survive a truncated reply ------------------------------

from server.vlm import _parse_perception, perceive


def test_parse_perception_salvages_complete_items_from_truncated_json():
    # A rich scene overflowed the token cap: the reply stops mid-object.
    text = ('{"scene_description":"aisle","detections":['
            '{"label":"aisle sign 3","box":[0.38,0.24,0.52,0.39],"score":0.98},'
            '{"label":"aisle sign 4","box":[0.62,0.24,0.82,0.42],"score":0.97},'
            '{"label":"left shelf","box":[0.0,0.34,0.3')
    p = _parse_perception(text, 1000, 1000)
    assert p is not None
    assert [d.label for d in p.detections] == ["aisle sign 3", "aisle sign 4"]


def test_parse_perception_salvages_when_truncated_inside_ocr():
    text = ('{"scene_description":"aisle","detections":[{"label":"sign","box":[0.1,0.1,0.2,0.2],"score":0.9}],'
            '"ocr_texts":[{"text":"3","box":[0.4,0.3,0.45,0.35],"score":0.9},{"text":"4","box":[0.6')
    p = _parse_perception(text, 1000, 1000)
    assert p is not None
    assert [d.label for d in p.detections] == ["sign"]
    assert [t.text for t in p.ocr_texts] == ["3"]


def test_perceive_requests_a_larger_token_budget_than_decide(tmp_path):
    img = _make_jpeg(tmp_path)
    reply = '{"scene_description":"x","detections":[],"ocr_texts":[]}'
    with patch("server.vlm.requests.post", return_value=_mock_openai(reply)) as post:
        perceive(str(img), "找到：泡麵", ["泡麵"], 8, 8)
    body = post.call_args.kwargs["json"]
    assert body["max_completion_tokens"] >= 1500


def test_perceive_prompt_asks_for_goal_label_in_goal_language(tmp_path):
    # goal words are Chinese; a "yogurt cups" label would never match 優格 in the gate
    img = _make_jpeg(tmp_path)
    reply = '{"scene_description":"x","detections":[],"ocr_texts":[]}'
    with patch("server.vlm.requests.post", return_value=_mock_openai(reply)) as post:
        perceive(str(img), "找到：優格", ["優格", "優酪乳"], 8, 8)
    text = post.call_args.kwargs["json"]["messages"][0]["content"][-1]["text"]
    assert "label it exactly as one of the goal names" in text
    assert "優格" in text
