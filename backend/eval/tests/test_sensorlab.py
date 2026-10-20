"""eval/sensorlab.MapPositionEstimator must behave like the app's MapPositionEstimator.kt —
these are app/src/test/.../MapPositionEstimatorTest.kt, case for case."""
import pytest

from eval.sensorlab import MapPositionEstimator as E, _bearing, _rotate, _wrap

approx = lambda v: pytest.approx(v, abs=1e-6)


def walk(e, a, b, steps=10):
    for i in range(1, steps + 1):
        e.ingest((a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps))


def fixed():
    e = E()
    e.ingest((0, 0))
    e.apply_fix((0, 0), 1.0, (0, 0))
    return e


def test_no_map_position_until_first_fix():
    e = E()
    e.ingest((0, 0)); e.ingest((0, 5))
    assert e.map_position is None


def test_first_fix_snaps_and_later_steps_move():
    e = E()
    e.ingest((0, 0))
    assert e.apply_fix((10, 20), 0.4, (0, 0)) == "accepted"
    e.ingest((0, 3))
    assert e.map_position == (approx(10), approx(23)) and e.distance_since_fix == approx(3)


def test_low_confidence_and_jumps():
    e = E()
    e.ingest((0, 0))
    assert e.apply_fix((1, 1), 0.2, (0, 0)) == "rejected_low_confidence"
    e.apply_fix((0, 0), 1.0, (0, 0))
    assert e.apply_fix((1, 1), 0.5, (0, 0)) == "rejected_low_confidence"
    assert e.apply_fix((30, 0), 0.7, (0, 0)) == "rejected_jump"
    assert e.apply_fix((30, 0), 0.9, (0, 0)) == "accepted"


def test_jump_measured_at_capture():
    e = fixed()
    walk(e, (0, 0), (0, 20)); walk(e, (0, 20), (0, 30))
    assert e.apply_fix((0, 21), 0.7, (0, 20)) == "accepted"
    e.ingest((0, 30))
    assert e.step_scale == approx(1.015) and e.map_position[1] == approx(21 + 10 * 1.015)


def test_straight_stretch_learns_heading_and_scale():
    e = fixed()
    walk(e, (0, 0), (0, 10))
    e.apply_fix(_rotate((0, 12), 10), 0.9, (0, 10))
    assert e.heading_offset == approx(3.0) and e.step_scale == approx(1.06)


def test_step_scale_clamped():
    e = fixed()
    walk(e, (0, 0), (0, 10))
    e.apply_fix((0, 40), 0.9, (0, 10))
    assert e.step_scale == approx(1.09)


def test_align_heading():
    e = E()
    e.align_heading(93.7, 0.0)
    assert e.map_heading(93.7) == approx(0) and e.map_heading(183.7) == approx(90)
    e = fixed()
    e.align_heading(90.0, 0.0)
    e.ingest((5, 0))
    assert e.map_position == (approx(0), approx(5))
    e = E()
    e.align_heading(10.0, 0.0)
    e.absorb_raw_heading_jump(75.0)
    assert e.map_heading(85.0) == approx(0)


def test_two_stretches_far_off():
    e = fixed()
    walk(e, (0, 0), (0, 10))
    assert e.apply_fix((10, 0), 0.9, (0, 10)) == "accepted" and e.heading_offset == approx(0)
    walk(e, (0, 10), (0, 20))
    e.apply_fix((20, 0), 0.9, (0, 20))
    assert e.heading_offset == approx(90)
    e = fixed()
    walk(e, (0, 0), (0, 10)); e.apply_fix((10, 0), 0.9, (0, 10))
    walk(e, (0, 10), (0, 20)); e.apply_fix((0, -10), 0.9, (0, 20))
    assert e.heading_offset == approx(0)


def test_helpers():
    assert E().map_heading(-10) == approx(350) and E().map_heading(370) == approx(10)
    assert _bearing((0, 1)) == approx(0) and _bearing((1, 0)) == approx(90)
    assert _rotate((0, 1), 90) == (approx(1), approx(0))
    assert _wrap(190) == approx(-170) and _wrap(-180) == approx(180)
