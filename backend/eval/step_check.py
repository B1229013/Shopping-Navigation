"""Is the step sensor really moving when the user walks? Two checks on data we already record.

1. Sensor Lab export (sensor_log.csv, ~50 Hz): the walk is cut into 4-second stretches. A
   stretch is *walking* when |a| varies (std ≥ 0.6 m/s²) with a step rhythm (autocorrelation
   ≥ 0.2 at a 0.38–0.83 s period); the rhythm also gives its expected step count. Reported:
   walking stretches where no step was counted (the position froze), and steps counted while
   clearly standing (std < 0.3 over 2 s) — for the app's count (total_steps) and for the
   phone's own step detector alone (step_detected).
2. Localization results with a Sensor Lab replay (eval/results/image_only_*_photos.jsonl):
   legs between consecutive test points where the step count did not change although the
   true positions are more than 2 m apart.

    python -m eval.step_check <export folder or sensor_log.csv>...
    python -m eval.step_check --results eval/results/image_only_lab1019_vlm_photos.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

WINDOW_MS = 4000
WALK_STD = 0.6
STAND_STD = 0.3
MIN_LEG_M = 2.0


def read_log(path) -> Tuple[List[int], List[float], List[int], List[int]]:
    """(t ms from start, |a|, times of app steps, times of step-detector events)."""
    path = Path(path)
    if path.is_dir():
        path = path / "sensor_log.csv"
    rows = list(csv.DictReader(l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")))
    t0 = int(rows[0]["timestamp_ms"])
    t = [int(r["timestamp_ms"]) - t0 for r in rows]
    mag = [math.sqrt(sum(float(r[k]) ** 2 for k in ("accel_x", "accel_y", "accel_z"))) for r in rows]

    def events(col):
        vals = [int(r.get(col) or 0) for r in rows]
        return [t[i] for i in range(1, len(t)) for _ in range(max(0, vals[i] - vals[i - 1]))]

    return t, mag, events("total_steps"), events("step_detected")


def _resample(t, v, fs):
    """Linear resample at fs Hz; None across gaps > 300 ms."""
    out, j, T = [], 0, 0.0
    while T <= t[-1]:
        while j + 1 < len(t) and t[j + 1] < T:
            j += 1
        if j + 1 < len(t) and t[j + 1] - t[j] < 300:
            out.append(v[j] + (T - t[j]) / max(1, t[j + 1] - t[j]) * (v[j + 1] - v[j]))
        else:
            out.append(None)
        T += 1000 / fs
    return out


def walking_stretches(t, mag, fs=50) -> List[Tuple[int, float]]:
    """(start ms, expected steps) of every 4 s stretch; expected 0 = not walking."""
    x, n = _resample(t, mag, fs), int(WINDOW_MS * fs / 1000)
    out = []
    for s in range(0, len(x) - n, n):
        w = x[s:s + n]
        if any(v is None for v in w):
            continue
        m = sum(w) / n
        w = [v - m for v in w]
        energy = sum(v * v for v in w)
        steps = 0.0
        if math.sqrt(energy / n) >= WALK_STD:
            corr, lag = max((sum(w[i] * w[i + L] for i in range(n - L)) / energy, L)
                            for L in range(int(fs / 2.6), int(fs / 1.2) + 1))
            if corr >= 0.2:
                steps = WINDOW_MS / 1000 * fs / lag
        out.append((int(s * 1000 / fs), steps))
    return out


def standing_steps(t, mag, steps) -> Tuple[int, int]:
    """(2 s stretches clearly standing, steps counted in them)."""
    windows = counted = s = 0
    while s < len(t):
        e = s
        while e < len(t) and t[e] - t[s] < 2000:
            e += 1
        seg = mag[s:e]
        if len(seg) > 20:
            m = sum(seg) / len(seg)
            if math.sqrt(sum((v - m) ** 2 for v in seg) / len(seg)) < STAND_STD:
                windows += 1
                counted += sum(1 for x in steps if t[s] <= x < t[s] + 2000)
        s = e
    return windows, counted


def check_log(path) -> Dict:
    t, mag, app, hw = read_log(path)
    stretches = walking_stretches(t, mag)
    walking = [(s, n) for s, n in stretches if n]

    def report(steps):
        frozen = [s for s, _ in walking if not any(s <= x < s + WINDOW_MS for x in steps)]
        standing, while_standing = standing_steps(t, mag, steps)
        return {"steps": len(steps), "frozen_stretches": len(frozen), "frozen_at_s": [s / 1000 for s in frozen],
                "standing_stretches": standing, "steps_while_standing": while_standing}

    return {"duration_s": t[-1] / 1000, "walking_stretches": len(walking),
            "expected_walking_steps": round(sum(n for _, n in walking)),
            "app": report(app), "step_detector": report(hw)}


def stuck_legs(results_jsonl) -> Tuple[int, List[Dict]]:
    """(legs checked, legs > MIN_LEG_M apart with no step) between consecutive test points."""
    recs = [json.loads(l) for l in Path(results_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    first: Dict[str, Dict] = {}
    for r in sorted((r for r in recs if r.get("sensor_lab") and r.get("truth_xy")), key=lambda r: r["sensor_lab"]["ms"]):
        first.setdefault(r["meta"]["point"], r)
    legs, stuck, prev = 0, [], None
    for r in first.values():
        if prev and r["meta"]["point"].split("/")[0] == prev["meta"]["point"].split("/")[0]:
            true_m = math.dist(r["truth_xy"], prev["truth_xy"])
            if true_m > MIN_LEG_M:
                legs += 1
                if r["sensor_lab"]["steps"] == prev["sensor_lab"]["steps"]:
                    stuck.append({"from": prev["meta"]["label"], "to": r["meta"]["label"],
                                  "point": r["meta"]["point"], "true_m": round(true_m, 1)})
        prev = r
    return legs, stuck


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="*", help="Sensor Lab export folders or sensor_log.csv files")
    ap.add_argument("--results", action="append", default=[], help="image_only_*_photos.jsonl with sensor_lab")
    args = ap.parse_args(argv)
    for p in args.logs:
        r = check_log(p)
        print(f"{p}: {r['duration_s']:.0f} s, {r['walking_stretches']} walking stretches of 4 s "
              f"(~{r['expected_walking_steps']} steps by rhythm)")
        for name in ("app", "step_detector"):
            x = r[name]
            print(f"  {name:13s} {x['steps']:4d} steps · walking with no step: {x['frozen_stretches']}"
                  f"/{r['walking_stretches']} {x['frozen_at_s']} · steps while standing: "
                  f"{x['steps_while_standing']} in {x['standing_stretches']} × 2 s")
    for p in args.results:
        legs, stuck = stuck_legs(p)
        print(f"{p}: {len(stuck)}/{legs} legs > {MIN_LEG_M:g} m with no step")
        for s in stuck:
            print(f"  {s['point']:>6}  {s['from']} → {s['to']}  {s['true_m']} m")


if __name__ == "__main__":
    main()
