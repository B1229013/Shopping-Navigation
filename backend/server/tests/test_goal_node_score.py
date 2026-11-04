"""_goal_node_score — which reference node a goal item ("牛奶") is resolved to at session start."""
from types import SimpleNamespace

from server.server import _goal_node_score

MILK_TERMS = ["牛奶", "鮮奶", "milk", "鮮乳"]


def _obj(label, ocr=None):
    return SimpleNamespace(label=label, ocr_text=ocr)


def test_more_matching_text_beats_a_single_match():
    # A7: the shelf of milk drinks (one 「milk」) used to tie with the milk fridge and win
    # by coming first; the fridge reads 「Milk」「Milk」「鮮乳」.
    shelf = [_obj("牛奶飲品 milk drink", "milk"), _obj("燕麥奶飲品 oat milk drink", "OATLY"),
             _obj("乳飲品 dairy drink"), _obj("果汁 juice"), _obj("杏仁奶 almond milk drink")]
    fridge = [_obj("牛奶盒插圖 milk carton illustration", "Milk"), _obj("牛奶瓶插圖 milk bottle illustration", "Milk"),
              _obj("右側中層盒裝乳品 right middle milk cartons", "鮮乳")]
    assert _goal_node_score(fridge, MILK_TERMS) > _goal_node_score(shelf, MILK_TERMS)


def test_label_matches_decide_when_no_text_matches():
    two = [_obj("優格杯 yogurt cup"), _obj("杯裝優格 yogurt cups")]
    one = [_obj("優格飲料瓶 yogurt drink bottle")]
    assert _goal_node_score(two, ["優格", "yogurt"]) > _goal_node_score(one, ["優格", "yogurt"])


def test_single_character_text_matches_nothing():
    # "e" is a substring of "instant noodle" — it must not count as the noodle section
    assert _goal_node_score([_obj("區域分類標示 section sign", "E")], ["泡麵", "instant noodle"]) == (0.0, 0.0)


def test_empty_label_matches_nothing():
    assert _goal_node_score([_obj("", None)], MILK_TERMS) == (0.0, 0.0)


def test_text_scores():
    exact = _goal_node_score([_obj("招牌 sign", "milk")], MILK_TERMS)
    part = _goal_node_score([_obj("招牌 sign", "透明鮮乳")], MILK_TERMS)
    assert exact[0] == 1.0 and part[0] == 0.8
