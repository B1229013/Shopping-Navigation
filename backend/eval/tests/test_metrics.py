from eval.metrics import (
    TurnRecord,
    success_rate,
    false_arrival_rate,
    direction_sanity,
    json_validity_rate,
    arrival_corroboration_rate,
    summarize,
)


def test_success_rate_counts_only_labeled_records():
    records = [
        TurnRecord(photo="a.jpg", goal="milk", predicted_action="MOVE", expected_action="MOVE"),
        TurnRecord(photo="b.jpg", goal="milk", predicted_action="ARRIVED", expected_action="MOVE"),
        TurnRecord(photo="c.jpg", goal="milk", predicted_action="ASK"),  # unlabeled, ignored
    ]
    # 1 of 2 labeled records correct
    assert success_rate(records) == 0.5


def test_false_arrival_rate_is_arrived_on_non_goal_frames():
    records = [
        TurnRecord(photo="a.jpg", goal="milk", predicted_action="ARRIVED", is_goal_frame=False),  # false arrival
        TurnRecord(photo="b.jpg", goal="milk", predicted_action="MOVE", is_goal_frame=False),      # fine
        TurnRecord(photo="c.jpg", goal="milk", predicted_action="ARRIVED", is_goal_frame=True),    # true arrival
        TurnRecord(photo="d.jpg", goal="milk", predicted_action="MOVE"),                           # unlabeled, ignored
    ]
    # non-goal frames: a, b. One of them wrongly said ARRIVED => 0.5
    assert false_arrival_rate(records) == 0.5


def test_false_arrival_rate_zero_when_no_non_goal_frames():
    records = [TurnRecord(photo="a.jpg", goal="milk", predicted_action="ARRIVED", is_goal_frame=True)]
    assert false_arrival_rate(records) == 0.0


def test_direction_sanity_matches_guidance_direction_to_detected_side():
    records = [
        # says left, something detected on left => sane
        TurnRecord(photo="a.jpg", goal="milk", predicted_action="MOVE",
                   guidance="Turn left toward the dairy.", detection_sides=["left", "center"]),
        # says right, nothing detected on right => not sane
        TurnRecord(photo="b.jpg", goal="milk", predicted_action="MOVE",
                   guidance="Head right down the aisle.", detection_sides=["left"]),
        # no direction word => excluded from denominator
        TurnRecord(photo="c.jpg", goal="milk", predicted_action="MOVE",
                   guidance="Walk forward and take another photo.", detection_sides=["right"]),
        # not a MOVE => excluded
        TurnRecord(photo="d.jpg", goal="milk", predicted_action="ASK",
                   guidance="Is it on your left?", detection_sides=["right"]),
    ]
    # directional MOVE records: a (sane), b (not) => 0.5
    assert direction_sanity(records) == 0.5


def test_json_validity_rate_counts_parseable_responses():
    records = [
        TurnRecord(photo="a.jpg", goal="milk", predicted_action="MOVE", json_valid=True),
        TurnRecord(photo="b.jpg", goal="milk", predicted_action="MOVE", json_valid=False),
    ]
    assert json_validity_rate(records) == 0.5


def test_json_validity_rate_empty_is_one():
    assert json_validity_rate([]) == 1.0


def test_arrival_corroboration_rate_only_over_arrived_predictions():
    records = [
        TurnRecord(photo="a.jpg", goal="milk", predicted_action="ARRIVED", arrival_corroborated=True),
        TurnRecord(photo="b.jpg", goal="milk", predicted_action="ARRIVED", arrival_corroborated=False),
        TurnRecord(photo="c.jpg", goal="milk", predicted_action="MOVE"),  # excluded
    ]
    assert arrival_corroboration_rate(records) == 0.5


def test_summarize_reports_metrics_with_denominator_counts():
    records = [
        TurnRecord(photo="a.jpg", goal="milk", predicted_action="ARRIVED",
                   is_goal_frame=True, expected_action="ARRIVED", arrival_corroborated=True),
        TurnRecord(photo="b.jpg", goal="milk", predicted_action="ARRIVED",
                   is_goal_frame=False, expected_action="MOVE", arrival_corroborated=False),
    ]
    s = summarize(records)
    assert s["n_turns"] == 2
    assert s["n_labeled"] == 2
    assert s["n_non_goal"] == 1
    assert s["success_rate"] == 0.5
    assert s["false_arrival_rate"] == 1.0
