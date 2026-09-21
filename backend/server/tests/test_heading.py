"""Tests for the heading estimation and relative direction module."""
from __future__ import annotations

import math

import pytest

from server.heading import (
    DirectionSlot,
    DirectionalPhoto,
    RelativeDirection,
    RelativeInstruction,
    _angle_diff,
    _normalize_angle,
    best_matching_slot,
    compute_relative_direction,
    convert_leg_to_relative,
    estimate_heading,
    heading_between_nodes,
    relative_direction_text,
    slot_headings,
)


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

class TestAngleHelpers:
    def test_normalize_positive(self):
        assert _normalize_angle(90) == 90
        assert _normalize_angle(0) == 0
        assert _normalize_angle(359.9) == pytest.approx(359.9)

    def test_normalize_wrap(self):
        assert _normalize_angle(360) == 0
        assert _normalize_angle(450) == 90
        assert _normalize_angle(-90) == 270
        assert _normalize_angle(-180) == 180

    def test_angle_diff_same(self):
        assert _angle_diff(90, 90) == 0

    def test_angle_diff_positive(self):
        # 180 - 90 = +90 (turn right)
        assert _angle_diff(180, 90) == pytest.approx(90)

    def test_angle_diff_negative(self):
        # 90 - 180 = -90 (turn left)
        assert _angle_diff(90, 180) == pytest.approx(-90)

    def test_angle_diff_wraparound(self):
        # 10 - 350 should be +20, not -340
        assert _angle_diff(10, 350) == pytest.approx(20)
        # 350 - 10 should be -20, not +340
        assert _angle_diff(350, 10) == pytest.approx(-20)


# ---------------------------------------------------------------------------
# Slot headings
# ---------------------------------------------------------------------------

class TestSlotHeadings:
    def test_north_facing(self):
        slots = slot_headings(0)
        assert slots[DirectionSlot.FRONT] == pytest.approx(0)
        assert slots[DirectionSlot.RIGHT] == pytest.approx(90)
        assert slots[DirectionSlot.BACK] == pytest.approx(180)
        assert slots[DirectionSlot.LEFT] == pytest.approx(270)

    def test_east_facing(self):
        slots = slot_headings(90)
        assert slots[DirectionSlot.FRONT] == pytest.approx(90)
        assert slots[DirectionSlot.RIGHT] == pytest.approx(180)
        assert slots[DirectionSlot.BACK] == pytest.approx(270)
        assert slots[DirectionSlot.LEFT] == pytest.approx(0)

    def test_wraps_around(self):
        slots = slot_headings(300)
        assert slots[DirectionSlot.RIGHT] == pytest.approx(30)
        assert slots[DirectionSlot.BACK] == pytest.approx(120)


# ---------------------------------------------------------------------------
# Heading estimation
# ---------------------------------------------------------------------------

class TestEstimateHeading:
    def _make_photo(self, slot, heading, labels, ocr=None):
        objects = [{"label": l, "ocr_text": ""} for l in labels]
        if ocr:
            for o, t in zip(objects, ocr):
                o["ocr_text"] = t
        return DirectionalPhoto(
            slot=DirectionSlot(slot),
            heading_deg=heading,
            photo_file=f"test_{slot}.jpg",
            objects=objects,
        )

    def test_matches_best_direction(self):
        photos = [
            self._make_photo("front", 0, ["shelf", "rice", "aisle"]),
            self._make_photo("right", 90, ["refrigerator", "milk"]),
            self._make_photo("back", 180, ["checkout", "cashier"]),
            self._make_photo("left", 270, ["fruit", "oranges"]),
        ]
        # User sees shelf and rice → should match front (0°)
        heading = estimate_heading({"shelf", "rice"}, set(), photos)
        assert heading == pytest.approx(0)

    def test_ocr_boosts_match(self):
        photos = [
            self._make_photo("front", 0, ["shelf"], ["米"]),
            self._make_photo("right", 90, ["shelf"], ["牛奶"]),
            self._make_photo("back", 180, ["shelf"], ["餅乾"]),
            self._make_photo("left", 270, ["shelf"], ["啤酒"]),
        ]
        # All have "shelf" — OCR "米" breaks the tie
        heading = estimate_heading({"shelf"}, {"米"}, photos)
        assert heading == pytest.approx(0)

    def test_returns_none_if_no_photos(self):
        assert estimate_heading({"shelf"}, set(), []) is None

    def test_best_matching_slot(self):
        photos = [
            self._make_photo("front", 0, ["shelf", "rice"]),
            self._make_photo("right", 90, ["refrigerator"]),
            self._make_photo("back", 180, ["checkout"]),
            self._make_photo("left", 270, ["fruit"]),
        ]
        slot = best_matching_slot({"refrigerator", "milk"}, set(), photos)
        assert slot == DirectionSlot.RIGHT


# ---------------------------------------------------------------------------
# Relative direction computation
# ---------------------------------------------------------------------------

class TestRelativeDirection:
    def test_straight(self):
        assert compute_relative_direction(0, 0) == RelativeDirection.STRAIGHT
        assert compute_relative_direction(0, 30) == RelativeDirection.STRAIGHT
        assert compute_relative_direction(350, 10) == RelativeDirection.STRAIGHT

    def test_right(self):
        assert compute_relative_direction(0, 90) == RelativeDirection.RIGHT
        assert compute_relative_direction(0, 100) == RelativeDirection.RIGHT

    def test_left(self):
        assert compute_relative_direction(0, 270) == RelativeDirection.LEFT
        assert compute_relative_direction(0, 260) == RelativeDirection.LEFT

    def test_behind(self):
        assert compute_relative_direction(0, 180) == RelativeDirection.BEHIND
        assert compute_relative_direction(0, 175) == RelativeDirection.BEHIND
        assert compute_relative_direction(90, 270) == RelativeDirection.BEHIND

    def test_slight_right(self):
        assert compute_relative_direction(0, 60) == RelativeDirection.SLIGHT_RIGHT

    def test_slight_left(self):
        assert compute_relative_direction(0, 300) == RelativeDirection.SLIGHT_LEFT

    def test_sharp_right(self):
        assert compute_relative_direction(0, 150) == RelativeDirection.SHARP_RIGHT

    def test_sharp_left(self):
        assert compute_relative_direction(0, 210) == RelativeDirection.SHARP_LEFT

    def test_wraparound(self):
        # Facing 350°, target 20° → delta = +30° → straight
        assert compute_relative_direction(350, 20) == RelativeDirection.STRAIGHT
        # Facing 10°, target 280° → delta = -90° → left
        assert compute_relative_direction(10, 280) == RelativeDirection.LEFT


# ---------------------------------------------------------------------------
# Direction text
# ---------------------------------------------------------------------------

class TestDirectionText:
    def test_zh(self):
        assert relative_direction_text(RelativeDirection.RIGHT, "zh") == "右轉"
        assert relative_direction_text(RelativeDirection.STRAIGHT, "zh") == "直走"
        assert relative_direction_text(RelativeDirection.BEHIND, "zh") == "迴轉"

    def test_en(self):
        assert relative_direction_text(RelativeDirection.LEFT, "en") == "Turn left"
        assert relative_direction_text(RelativeDirection.STRAIGHT, "en") == "Go straight"


# ---------------------------------------------------------------------------
# Heading between nodes
# ---------------------------------------------------------------------------

class TestHeadingBetweenNodes:
    def test_north(self):
        # (0,0) → (0,10) = north = 0°
        assert heading_between_nodes(0, 0, 0, 10) == pytest.approx(0)

    def test_east(self):
        # (0,0) → (10,0) = east = 90°
        assert heading_between_nodes(0, 0, 10, 0) == pytest.approx(90)

    def test_south(self):
        # (0,0) → (0,-10) = south = 180°
        assert heading_between_nodes(0, 0, 0, -10) == pytest.approx(180)

    def test_west(self):
        # (0,0) → (-10,0) = west = 270°
        assert heading_between_nodes(0, 0, -10, 0) == pytest.approx(270)

    def test_northeast(self):
        h = heading_between_nodes(0, 0, 10, 10)
        assert h == pytest.approx(45)

    def test_same_point(self):
        assert heading_between_nodes(5, 5, 5, 5) == 0.0


# ---------------------------------------------------------------------------
# Convert leg to relative instructions
# ---------------------------------------------------------------------------

class TestConvertLeg:
    def test_simple_path(self):
        # Path: A(0,0) → B(0,10) → C(10,10)
        # User faces north (0°)
        # A→B: heading=0° (north) → relative=straight
        # B→C: heading=90° (east) → relative=right
        positions = {1: (0, 0), 2: (0, 10), 3: (10, 10)}
        result = convert_leg_to_relative([1, 2, 3], positions, user_heading=0.0)

        assert len(result) == 2
        assert result[0].direction == RelativeDirection.STRAIGHT
        assert result[0].text_zh == "直走"
        assert result[1].direction == RelativeDirection.RIGHT
        assert result[1].text_zh == "右轉"

    def test_heading_updates_along_path(self):
        # A(0,0) → B(0,10) → C(-10,10)
        # User faces east (90°)
        # A→B: heading=0° (north), from 90° → left
        # B→C: heading=270° (west), from 0° (updated) → left again
        positions = {1: (0, 0), 2: (0, 10), 3: (-10, 10)}
        result = convert_leg_to_relative([1, 2, 3], positions, user_heading=90.0)

        assert result[0].direction == RelativeDirection.LEFT
        assert result[1].direction == RelativeDirection.LEFT
        # After first step user faces 0°, after second faces 270°
        assert result[0].new_heading == pytest.approx(0)
        assert result[1].new_heading == pytest.approx(270)

    def test_single_edge(self):
        positions = {1: (0, 0), 2: (10, 0)}
        result = convert_leg_to_relative([1, 2], positions, user_heading=90.0)
        assert len(result) == 1
        assert result[0].direction == RelativeDirection.STRAIGHT

    def test_empty_path(self):
        result = convert_leg_to_relative([1], {1: (0, 0)}, user_heading=0.0)
        assert result == []

    def test_with_distances(self):
        positions = {1: (0, 0), 2: (0, 10)}
        distances = {(1, 2): 5.5}
        result = convert_leg_to_relative(
            [1, 2], positions, user_heading=0.0, distances=distances
        )
        assert result[0].distance_m == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Merging per-edge instructions into human steps (with distance)
# ---------------------------------------------------------------------------

from server.heading import merge_instructions, next_instruction_text


# Corridor heading north with two aisles on the right; target is in the 2nd aisle.
_POS = {10: (0, 0), 11: (0, 6), 12: (0, 12), 13: (5, 12)}
_DIST = {(10, 11): 6.0, (11, 12): 6.0, (12, 13): 5.0}


class TestMergeInstructions:
    def test_straight_run_is_collapsed_with_summed_distance_and_passed_nodes(self):
        raw = convert_leg_to_relative([10, 11, 12, 13], _POS, user_heading=0.0, distances=_DIST)
        steps = merge_instructions(raw)
        assert len(steps) == 2
        assert steps[0].direction == RelativeDirection.STRAIGHT
        assert steps[0].distance_m == pytest.approx(12.0)
        assert steps[0].passed_nodes == 1          # went through node 11
        assert steps[0].to_node == 12
        assert steps[1].direction == RelativeDirection.RIGHT
        assert steps[1].distance_m == pytest.approx(5.0)
        assert steps[1].passed_nodes == 0

    def test_turn_absorbs_following_straight_edges(self):
        pos = {12: (0, 12), 13: (5, 12), 14: (9, 12)}
        raw = convert_leg_to_relative([12, 13, 14], pos, user_heading=0.0,
                                      distances={(12, 13): 5.0, (13, 14): 4.0})
        steps = merge_instructions(raw)
        assert len(steps) == 1
        assert steps[0].direction == RelativeDirection.RIGHT
        assert steps[0].distance_m == pytest.approx(9.0)
        assert steps[0].passed_nodes == 1

    def test_step_text_mentions_distance_and_intersections(self):
        raw = convert_leg_to_relative([10, 11, 12, 13], _POS, user_heading=0.0, distances=_DIST)
        steps = merge_instructions(raw)
        assert steps[0].text_zh == "直走約 12 公尺（經過 1 個路口）"
        assert steps[1].text_zh == "右轉後直走約 5 公尺"

    def test_step_text_without_distance_is_plain(self):
        raw = convert_leg_to_relative([10, 11, 12, 13], _POS, user_heading=0.0)  # no distances
        steps = merge_instructions(raw)
        assert steps[0].text_zh == "直走（經過 1 個路口）"
        assert steps[1].text_zh == "右轉"

    def test_empty(self):
        assert merge_instructions([]) == []


class TestNextInstructionText:
    def test_straight_then_turn_tells_how_far_before_turning(self):
        raw = convert_leg_to_relative([10, 11, 12, 13], _POS, user_heading=0.0, distances=_DIST)
        assert next_instruction_text(merge_instructions(raw)) == \
            "直走約 12 公尺（經過 1 個路口），然後右轉"

    def test_turn_first(self):
        raw = convert_leg_to_relative([12, 13], _POS, user_heading=0.0, distances=_DIST)
        assert next_instruction_text(merge_instructions(raw)) == "右轉後直走約 5 公尺"

    def test_single_straight_without_distance(self):
        raw = convert_leg_to_relative([10, 11], _POS, user_heading=0.0)
        assert next_instruction_text(merge_instructions(raw)) == "直走"

    def test_empty(self):
        assert next_instruction_text([]) == ""
