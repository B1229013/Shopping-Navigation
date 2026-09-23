from types import SimpleNamespace

import pytest

from server.scene import verify_arrival
from server.models import VLMResponse, VLMAction


def _resp(action, guidance="here", question=None):
    return VLMResponse(action=action, guidance=guidance, question=question, vlm_summary="loc")


def _det(label, score, box=None):
    return SimpleNamespace(label=label, box=box if box is not None else [0, 0, 10, 10], score=score)


GOAL = ["refrigerator", "fridge"]

# Image the distance-aware tests below pretend the photo is: 100x100 = 10000 px².
# NEAR_AREA_RATIO is 0.15, so a box needs >= 1500 px² to count as "near".
IMG_W = IMG_H = 100
NEAR_BOX = [30, 30, 70, 70]   # 40x40 = 1600 px² -> ratio 0.16 -> near
FAR_BOX = [40, 40, 60, 60]    # 20x20 =  400 px² -> ratio 0.04 -> far, centred

# The 5-bucket horizontal grid over a 100px-wide image is 20px per bucket, so a
# box centre picks its bucket by which fifth of the width it lands in.
FAR_LEFT_BOX = [0, 40, 10, 60]      # cx=5   -> far-left  -> 左手邊
LEFT_BOX = [20, 40, 40, 60]         # cx=30  -> left      -> 左手邊
CENTER_BOX = [40, 40, 60, 60]       # cx=50  -> center    -> 正前方
RIGHT_BOX = [60, 40, 80, 60]        # cx=70  -> right     -> 右手邊
FAR_RIGHT_BOX = [90, 40, 100, 60]   # cx=95  -> far-right -> 右手邊


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


# --------------------------------------------------------------------------
# Distance-aware path: the branches that only run when img_w/img_h are known.
# Both production call sites in server.py pass the photo's real dimensions, so
# these — not the legacy no-dims rules above — are what users actually hit.
# --------------------------------------------------------------------------


def test_near_goal_detection_with_dims_stays_arrived():
    dets = [_det("refrigerator", 0.80, NEAR_BOX)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                         img_w=IMG_W, img_h=IMG_H)
    assert out.action == VLMAction.ARRIVED
    assert out.guidance == "here"  # untouched, not rewritten by any downgrade


@pytest.mark.parametrize("box,side_zh", [
    (FAR_LEFT_BOX, "左手邊"),
    (LEFT_BOX, "左手邊"),
    (CENTER_BOX, "正前方"),
    (RIGHT_BOX, "右手邊"),
    (FAR_RIGHT_BOX, "右手邊"),
])
def test_far_goal_detection_downgrades_to_move_naming_the_side(box, side_zh):
    # The goal is visible but small (far away): we know where it is, so the user
    # gets a direction to walk in rather than a question.
    dets = [_det("refrigerator", 0.80, box)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                         img_w=IMG_W, img_h=IMG_H)
    assert out.action == VLMAction.MOVE
    assert side_zh in out.guidance
    # the other two sides must not leak into the guidance
    for other in {"左手邊", "正前方", "右手邊"} - {side_zh}:
        assert other not in out.guidance
    assert out.question is None
    assert out.vlm_summary == "loc"  # the VLM's scene summary survives the downgrade


def test_far_goal_detection_accepts_dict_detections():
    # The /answer call site replays s.last_detections, which are plain dicts
    # (server.py stores postprocess_detections output), so the far branch has to
    # read boxes out of dicts as well as objects.
    dets = [{"label": "refrigerator", "score": 0.80, "box": RIGHT_BOX}]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                         img_w=IMG_W, img_h=IMG_H)
    assert out.action == VLMAction.MOVE
    assert "右手邊" in out.guidance


def test_near_detection_wins_over_an_earlier_far_one():
    # A far goal box seen first must not suppress a genuinely near one later in
    # the list: ARRIVED stands.
    dets = [_det("refrigerator", 0.80, LEFT_BOX), _det("fridge", 0.90, NEAR_BOX)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                         img_w=IMG_W, img_h=IMG_H)
    assert out.action == VLMAction.ARRIVED


def test_below_threshold_far_detection_is_not_treated_as_visible():
    # Score below min_score is not evidence of anything, so this is the
    # "no evidence" case (ASK), not the "visible but far" case (MOVE).
    dets = [_det("refrigerator", 0.20, LEFT_BOX)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                         img_w=IMG_W, img_h=IMG_H)
    assert out.action == VLMAction.ASK
    assert out.question is not None


def test_ocr_match_with_dims_asks_instead_of_a_question_free_move():
    # A sign means the right zone, not the item. The downgrade must still leave
    # the user a way to finish the leg: a MOVE with question=None sets neither
    # pending_question nor pending_arrival, so POST /session/{id}/answer and
    # /confirm would both 409 — and when the detector has no class for the
    # product every later photo lands in this same branch, so the user could
    # never finish. The yes/no question is that escape hatch.
    out = verify_arrival(_resp(VLMAction.ARRIVED), [], ocr_matches=["refrigerator"],
                         goal_objects=GOAL, min_score=0.35, img_w=IMG_W, img_h=IMG_H)
    assert out.action == VLMAction.ASK
    assert out.question is not None
    assert out.vlm_summary == "loc"
    # _ocr_hint's own message — without this the branch is indistinguishable from
    # the no-evidence _confirm_question case and could be deleted unnoticed.
    assert "標示" in out.guidance
    assert "看起來可能已經到了" not in out.guidance


def test_ocr_match_with_unusable_detection_still_asks():
    # Detection present but below min_score -> no usable detection at all, so the
    # OCR sign is the only evidence and the same escape hatch must be offered.
    dets = [_det("refrigerator", 0.10, NEAR_BOX), _det("door", 0.99, NEAR_BOX)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, ocr_matches=["dairy"],
                         goal_objects=GOAL, min_score=0.35, img_w=IMG_W, img_h=IMG_H)
    assert out.action == VLMAction.ASK
    assert out.question is not None
    assert "標示" in out.guidance


def test_no_evidence_with_dims_asks_for_confirmation():
    dets = [_det("door", 0.99, NEAR_BOX)]  # near, but nothing to do with the goal
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                         img_w=IMG_W, img_h=IMG_H)
    assert out.action == VLMAction.ASK
    assert out.question is not None
    # _confirm_question, not _ocr_hint: no sign was read, so the message must not
    # claim one was.
    assert "看起來可能已經到了" in out.guidance
    assert "標示" not in out.guidance


def test_goal_verified_false_asks_even_with_a_near_detection():
    # The crop check looked at this exact box and said no: that explicit
    # rejection overrides an otherwise-convincing near detection.
    dets = [_det("refrigerator", 0.95, NEAR_BOX)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, ocr_matches=["refrigerator"],
                         goal_objects=GOAL, min_score=0.35, img_w=IMG_W, img_h=IMG_H,
                         goal_verified=False)
    assert out.action == VLMAction.ASK
    assert out.question is not None


def test_goal_verified_true_keeps_arrived_without_other_evidence():
    # A confirmed crop is strong enough on its own — no detection, no OCR.
    out = verify_arrival(_resp(VLMAction.ARRIVED), [], [], GOAL, 0.35,
                         img_w=IMG_W, img_h=IMG_H, goal_verified=True)
    assert out.action == VLMAction.ARRIVED
    assert out.guidance == "here"


def test_prior_was_confirm_beats_every_dims_downgrade():
    # The user is answering the confirm question right now; nothing (not even a
    # far-only detection that would otherwise become MOVE) may re-downgrade it.
    dets = [_det("refrigerator", 0.80, LEFT_BOX)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                         img_w=IMG_W, img_h=IMG_H, prior_was_confirm=True)
    assert out.action == VLMAction.ARRIVED
    assert out.guidance == "here"


# --------------------------------------------------------------------------
# Regression guard: callers that omit img_w/img_h (offline eval harness, the
# older unit tests) must keep the pre-merge, distance-blind behaviour.
# --------------------------------------------------------------------------


def test_legacy_no_dims_ignores_distance():
    # Same tiny box that becomes a MOVE once dimensions are known.
    dets = [_det("refrigerator", 0.80, FAR_BOX)]
    out = verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35)
    assert out.action == VLMAction.ARRIVED


def test_legacy_partial_dims_ignore_distance():
    # Only one dimension known is not enough to judge area -> legacy rule.
    dets = [_det("refrigerator", 0.80, FAR_BOX)]
    assert verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                          img_w=IMG_W, img_h=0).action == VLMAction.ARRIVED
    assert verify_arrival(_resp(VLMAction.ARRIVED), dets, [], GOAL, 0.35,
                          img_w=0, img_h=IMG_H).action == VLMAction.ARRIVED


def test_legacy_no_dims_ocr_match_stays_arrived_not_ask():
    # Same inputs that ASK once dimensions are known must still pass through.
    out = verify_arrival(_resp(VLMAction.ARRIVED), [], ocr_matches=["refrigerator"],
                         goal_objects=GOAL, min_score=0.35)
    assert out.action == VLMAction.ARRIVED
    assert out.question is None
    # the VLM's own guidance is passed through untouched — no downgrade helper ran
    assert out.guidance == "here"
