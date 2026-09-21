from unittest.mock import patch, MagicMock
from server.goal_decomposer import decompose_goal


def _mock_openai_response(text: str):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
    }
    resp.raise_for_status = lambda: None
    return resp


def test_decompose_parses_comma_separated():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_openai_response("milk, dairy, fridge, carton, shelf")):
        objs = decompose_goal("find the milk")
    assert objs == ["milk", "dairy", "fridge", "carton", "shelf"]


def test_decompose_strips_whitespace_and_lowercases():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_openai_response("  Milk , DAIRY ,fridge")):
        objs = decompose_goal("find the milk")
    assert objs == ["milk", "dairy", "fridge"]


def test_decompose_falls_back_when_response_garbage():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_openai_response("Sure! Here are 50 items: " + "x," * 50)):
        objs = decompose_goal("find the cereal")
    assert "cereal" in objs


def test_decompose_falls_back_on_exception():
    with patch("server.goal_decomposer.requests.post", side_effect=Exception("boom")):
        objs = decompose_goal("find the cookies")
    assert "cookies" in objs


# ---- target vs. context split ----------------------------------------------
# decompose_goal deliberately returns landmark/section words ("dairy section",
# "cooler") so the detector can look for context. Those words must never be
# treated as the product itself by the arrival gate.

from server.goal_decomposer import split_goal_objects


def test_split_keeps_product_and_its_variants_as_targets():
    targets, context = split_goal_objects(
        "find the milk",
        ["milk", "milk carton", "milk bottle", "dairy section", "cooler", "dairy sign"],
    )
    assert targets == ["milk", "milk carton", "milk bottle"]
    assert context == ["dairy section", "cooler", "dairy sign"]


def test_split_handles_shopping_list_goal_with_quantities():
    # iOS sends "name xN, name xN"
    targets, context = split_goal_objects(
        "牛奶 x2, 麵包 x1",
        ["牛奶", "牛奶盒", "乳製品區", "冷藏櫃", "麵包", "烘焙區"],
    )
    assert targets == ["牛奶", "麵包", "牛奶盒"]
    assert context == ["乳製品區", "冷藏櫃", "烘焙區"]


def test_split_always_includes_goal_items_even_if_decompose_omitted_them():
    targets, context = split_goal_objects("cereal", ["breakfast aisle", "shelf sign"])
    assert targets == ["cereal"]
    assert context == ["breakfast aisle", "shelf sign"]


def test_split_extra_target_terms_are_targets():
    # GOAL_CLASS_MAP expansions (e.g. 冰箱 -> refrigerator/fridge) are true synonyms
    targets, context = split_goal_objects(
        "冰箱", ["冰箱", "refrigerator", "fridge", "kitchen", "door"],
        extra_target_terms=["refrigerator", "fridge"],
    )
    assert targets == ["冰箱", "refrigerator", "fridge"]
    assert context == ["kitchen", "door"]
