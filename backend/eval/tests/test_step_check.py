"""eval/step_check: walking with no counted step is reported, standing still is not walking."""
import json
import math

from eval import step_check as SC

HEADER = "timestamp_ms,total_steps,step_detected,accel_x,accel_y,accel_z\n"


def log(tmp_path, steps_counted_until_s):
    """8 s standing, then 14 s walking at 1.8 Hz; the app counts steps only until the given time."""
    lines, n = [], 0
    for i in range(0, 22000, 20):
        s = i / 1000
        walking = s >= 8
        a = 9.81 + (1.5 * math.sin(2 * math.pi * 1.8 * s) if walking else 0.0)
        if walking and s < steps_counted_until_s and i % 556 < 20:
            n += 1
        lines.append(f"{1_700_000_000_000 + i},{n},{n},0,0,{a:.4f}")
    p = tmp_path / "sensor_log.csv"
    p.write_text("# SensorLab Export\n" + HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_frozen_walking_is_reported(tmp_path):
    r = SC.check_log(log(tmp_path, steps_counted_until_s=16))
    assert r["walking_stretches"] == 3
    assert r["app"]["frozen_stretches"] == 1 and r["app"]["frozen_at_s"] == [16.0]
    assert r["app"]["steps_while_standing"] == 0 and r["app"]["standing_stretches"] >= 4


def test_counted_walk_is_not_frozen(tmp_path):
    r = SC.check_log(log(tmp_path, steps_counted_until_s=99))
    assert r["app"]["frozen_stretches"] == 0
    assert abs(r["expected_walking_steps"] - 12 * 1.8) <= 2      # 3 stretches × 4 s


def test_stuck_legs(tmp_path):
    def rec(point, label, xy, steps, ms):
        return {"meta": {"point": point, "label": label}, "truth_xy": xy,
                "sensor_lab": {"steps": steps, "ms": ms}}
    rows = [rec("1/3", "25", [0, 0], 5, 1), rec("1/3", "25", [0, 0], 5, 2),     # same point, 2nd photo
            rec("1/5", "11", [0, 5], 5, 3),                                     # walked 5 m, no step
            rec("1/7", "26", [0, 9], 12, 4),                                    # walked, counted
            rec("2/4", "37", [50, 0], 12, 5)]                                   # new walk: not a leg
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    legs, stuck = SC.stuck_legs(p)
    assert legs == 2 and stuck == [{"from": "25", "to": "11", "point": "1/5", "true_m": 5.0}]
