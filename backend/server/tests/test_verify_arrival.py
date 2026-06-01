from types import SimpleNamespace

from server.scene import verify_arrival
from server.models import VLMResponse, VLMAction


def _resp(action, guidance="here", question=None):
    return VLMResponse(action=action, guidance=guidance, question=question, vlm_summary="loc")


def _det(label, score):
    return SimpleNamespace(label=label, box=[0, 0, 10, 10], score=score)


GOAL = ["refrigerator", "fridge"]


def test_non_arrived_passes_through_unchanged():
    r = _resp(VLMAction.MOVE, "go left")
    out = verify_arrival(r, detections=[], ocr_matches=[], goal_objects=GOAL, min_score=0.35)
    assert out.action == VLMAction.MOVE
    assert out.guidance == "go left"


def test_arrived_with_strong_goal_detection_stays_arrived():
    dets = [_det("refrigerator", 0.80)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35)
    assert out.action == VLMAction.ARRIVED


def test_arrived_without_corroboration_downgrades_to_ask():
    dets = [_det("door", 0.90)]  # not goal-related
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35)
    assert out.action == VLMAction.ASK
    assert out.question  # a confirm question is set


def test_arrived_with_weak_detection_downgrades():
    dets = [_det("refrigerator", 0.20)]  # goal-related but below min_score
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35)
    assert out.action == VLMAction.ASK


def test_arrived_with_ocr_match_stays_arrived():
    out = verify_arrival(_resp(VLMAction.ARRIVED), [], ocr_matches=["refrigerator"], goal_objects=GOAL, min_score=0.35)
    assert out.action == VLMAction.ARRIVED


def test_confirmed_arrival_passes_through():
    # user already answered a confirm question -> trust the ARRIVED, no loop
    out = verify_arrival(_resp(VLMAction.ARRIVED), [], [], GOAL, 0.35, prior_was_confirm=True)
    assert out.action == VLMAction.ARRIVED
