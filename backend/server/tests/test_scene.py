from types import SimpleNamespace

from server.scene import format_detections, detection_sides, format_ocr, match_ocr_to_goal


def _det(label, box, score):
    return SimpleNamespace(label=label, box=box, score=score)


def _ocr(text, conf, bbox):
    return SimpleNamespace(text=text, confidence=conf, bbox=bbox)


def test_format_detections_reports_side_and_depth():
    # 100x100 image; box in the right third, large area => near
    det = _det("refrigerator", [70, 10, 95, 90], 0.91)  # area=25*80/10000=0.20
    out = format_detections([det], 100, 100)
    assert "refrigerator" in out
    assert "right" in out
    assert "near" in out


def test_format_detections_left_and_far():
    det = _det("sign", [2, 2, 12, 12], 0.5)  # left third, area=100/10000=0.01 => far
    out = format_detections([det], 100, 100)
    assert "left" in out
    assert "far" in out


def test_format_detections_empty_is_none_marker():
    assert format_detections([], 100, 100) == "(none)"


def test_detection_sides_returns_unique_horizontal_sides():
    dets = [
        _det("a", [5, 5, 15, 15], 0.9),    # left
        _det("b", [45, 45, 55, 55], 0.9),  # center
        _det("c", [80, 5, 95, 15], 0.9),   # right
        _det("d", [2, 80, 10, 95], 0.9),   # left again
    ]
    assert detection_sides(dets, 100) == ["left", "center", "right"]


def test_format_ocr_includes_text_and_side():
    ocr = _ocr("Dairy", 0.98, [[5, 5], [20, 5], [20, 15], [5, 15]])  # left
    out = format_ocr([ocr], 100, 100)
    assert "Dairy" in out
    assert "left" in out


def test_format_ocr_empty_marker():
    assert format_ocr([], 100, 100) == "(no text detected)"


def _ocr_at(text):
    return _ocr(text, 0.95, [[0, 0], [10, 0], [10, 10], [0, 10]])


def test_match_ocr_to_goal_via_english_synonym():
    matches = match_ocr_to_goal([_ocr_at("Dairy")], ["milk"])
    assert matches
    assert any("Dairy" in m for m in matches)


def test_match_ocr_to_goal_direct_substring():
    assert match_ocr_to_goal([_ocr_at("Frozen Foods")], ["frozen"])


def test_match_ocr_to_goal_chinese_synonym():
    assert match_ocr_to_goal([_ocr_at("牛奶")], ["milk"])


def test_match_ocr_to_goal_no_match():
    assert match_ocr_to_goal([_ocr_at("Exit")], ["milk"]) == []


def test_format_detections_accepts_dicts():
    # the answer handler caches detections as dicts (Detection.__dict__)
    det = {"label": "freezer", "box": [70, 10, 95, 90], "score": 0.8}
    out = format_detections([det], 100, 100)
    assert "freezer" in out
    assert "right" in out
