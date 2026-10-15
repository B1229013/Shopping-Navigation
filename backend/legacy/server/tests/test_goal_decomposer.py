from unittest.mock import patch, MagicMock
from server.goal_decomposer import decompose_goal


def _mock_ollama_response(text: str):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": text}
    resp.raise_for_status = lambda: None
    return resp


def test_decompose_parses_comma_separated():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_ollama_response("milk, dairy, fridge, carton, shelf")):
        objs = decompose_goal("find the milk")
    assert objs == ["milk", "dairy", "fridge", "carton", "shelf"]


def test_decompose_strips_whitespace_and_lowercases():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_ollama_response("  Milk , DAIRY ,fridge")):
        objs = decompose_goal("find the milk")
    assert objs == ["milk", "dairy", "fridge"]


def test_decompose_falls_back_when_response_garbage():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_ollama_response("Sure! Here are 50 items: " + "x," * 50)):
        objs = decompose_goal("find the cereal")
    assert "cereal" in objs


def test_decompose_falls_back_on_exception():
    with patch("server.goal_decomposer.requests.post", side_effect=Exception("boom")):
        objs = decompose_goal("find the cookies")
    assert "cookies" in objs
