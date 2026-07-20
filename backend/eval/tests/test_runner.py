from eval.metrics import summarize
from eval.runner import records_from, format_report, make_template


def test_make_template_creates_blank_labels_per_photo():
    tpl = make_template(["a.jpg", "b.jpg"], goal="milk")
    assert tpl == [
        {"photo": "a.jpg", "goal": "milk", "expected_action": None, "is_goal_frame": None},
        {"photo": "b.jpg", "goal": "milk", "expected_action": None, "is_goal_frame": None},
    ]


def test_records_from_merges_labels_into_predictions_by_photo():
    labels = [{"photo": "a.jpg", "goal": "milk", "expected_action": "MOVE", "is_goal_frame": False}]
    predictions = [
        {"photo": "a.jpg", "predicted_action": "ARRIVED", "guidance": "Turn left",
         "detection_sides": ["left"], "json_valid": True, "arrival_corroborated": False},
        {"photo": "b.jpg", "predicted_action": "MOVE", "guidance": "forward"},  # no label
    ]
    recs = records_from(labels, predictions)
    assert len(recs) == 2
    a = next(r for r in recs if r.photo == "a.jpg")
    assert a.expected_action == "MOVE"
    assert a.is_goal_frame is False
    assert a.predicted_action == "ARRIVED"
    b = next(r for r in recs if r.photo == "b.jpg")
    assert b.expected_action is None  # unlabeled prediction tolerated


def test_format_report_contains_headline_metrics():
    recs = records_from(
        [{"photo": "a.jpg", "goal": "milk", "expected_action": "MOVE", "is_goal_frame": False}],
        [{"photo": "a.jpg", "predicted_action": "MOVE", "guidance": "go", "json_valid": True}],
    )
    text = format_report(summarize(recs))
    assert "false-arrival" in text.lower()
    assert "100.0%" in text  # success rate: 1/1 labeled correct
