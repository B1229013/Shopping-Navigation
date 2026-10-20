"""Sensor Lab exports as localization test data, and what the phone would have sent.

A Sensor Lab export folder holds sensor_log.csv (~50 Hz rows: timestamp_ms, total_steps,
pdr_x/pdr_y, cal_yaw_deg, …, waypoint) and photos/wp_<N>_<dir>_<ms>.jpg. A test walk sets
a named marker ("25", "40-44", "入口") before each photo group; the group itself appears
as marker "照片<N>". The labels CSV (eval/data/lab*_labels.csv) maps (folder, group) to the
answer.

`replay` drives a port of the phone's MapPositionEstimator
(app/.../navigation/MapPositionEstimator.kt) over the raw PDR, the way NavigationViewModel
does: 「我在入口」 at the "入口" marker (position (0,0), facing map 0°, last fix = the
nearest node), every step ingested, and after each test point a photo fix at the point's
TRUE position (oracle: the previous localization is assumed right, so one wrong photo
cannot drag the next). For every photo it returns map_x/y, dist_since_fix and fix_node as
the phone would have sent them, plus the map heading of the phone at the shutter.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

START_XY = (0.0, 0.0)     # maps/neo4j_links: entrance start, heading 0° = facing into the store
START_HEADING = 0.0
NORTH_JUMP_DEG = 20.0     # cal_yaw jumping this much between two ~20 ms samples = north found
_PHOTO = re.compile(r"wp_(\d+)_([a-z]+)_(\d+)\.jpg$")


@dataclass
class Session:
    folder: Path
    t: List[int]                       # timestamp_ms per row
    raw: List[Tuple[float, float]]     # pdr_x, pdr_y
    yaw: List[float]                   # cal_yaw_deg
    steps: List[int]
    marks: List[Tuple[int, str]]       # (row index, waypoint name)
    photos: Dict[int, Dict[str, Tuple[int, Path]]] = field(default_factory=dict)  # group → dir → (ms, path)

    def row_at(self, ms: int) -> int:
        """Last row at or before ms (binary search)."""
        lo, hi = 0, len(self.t) - 1
        if ms <= self.t[0]:
            return 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.t[mid] <= ms:
                lo = mid
            else:
                hi = mid - 1
        return lo


def read_session(folder) -> Session:
    folder = Path(folder)
    lines = (folder / "sensor_log.csv").read_text(encoding="utf-8").splitlines()
    body = [l for l in lines if not l.startswith("#")]
    s = Session(folder, [], [], [], [], [])
    for i, r in enumerate(csv.DictReader(body)):
        s.t.append(int(r["timestamp_ms"]))
        s.raw.append((float(r["pdr_x"]), float(r["pdr_y"])))
        s.yaw.append(float(r["cal_yaw_deg"]))
        s.steps.append(int(r["total_steps"]))
        if r.get("waypoint"):
            s.marks.append((i, r["waypoint"]))
    for f in (folder / "photos").iterdir():
        m = _PHOTO.match(f.name)
        if m:
            s.photos.setdefault(int(m.group(1)), {})[m.group(2)] = (int(m.group(3)), f)
    return s


# ── the phone's MapPositionEstimator, line for line ────────────────────────

def _wrap(d):
    d = d % 360.0
    if d > 180:
        d -= 360.0
    if d <= -180:
        d += 360.0
    return d


def _bearing(v):
    return math.degrees(math.atan2(v[0], v[1]))


def _rotate(v, deg):
    r = math.radians(deg)
    return (v[0] * math.cos(r) + v[1] * math.sin(r), -v[0] * math.sin(r) + v[1] * math.cos(r))


class MapPositionEstimator:
    min_confidence = 0.6
    min_first_fix_confidence = 0.3
    max_jump_m = 15.0
    jump_override_confidence = 0.85
    min_calibration_distance_m = 8.0
    learning_rate = 0.3
    max_heading_correction_deg = 45.0
    large_correction_agreement_deg = 30.0
    step_scale_min, step_scale_max = 0.7, 1.3

    def __init__(self):
        self.map_position = None
        self.heading_offset = 0.0
        self.step_scale = 1.0
        self.distance_since_fix = 0.0
        self.last_raw = None
        self.base_map = self.base_raw = None
        self.raw_path_since_base = 0.0
        self.pending_large = None

    def align_heading(self, raw_heading, map_heading):
        self.heading_offset = _wrap(map_heading - raw_heading)
        self.pending_large = None

    def absorb_raw_heading_jump(self, jump):
        self.heading_offset = _wrap(self.heading_offset - jump)
        if self.pending_large is not None:
            self.pending_large = _wrap(self.pending_large - jump)

    def ingest(self, raw):
        prev, self.last_raw = self.last_raw, raw
        if prev is None:
            return
        d = math.dist(raw, prev)
        if d <= 0:
            return
        self.raw_path_since_base += d
        if self.map_position is None:
            return
        st = _rotate((raw[0] - prev[0], raw[1] - prev[1]), self.heading_offset)
        self.map_position = (self.map_position[0] + st[0] * self.step_scale,
                             self.map_position[1] + st[1] * self.step_scale)
        self.distance_since_fix += d * self.step_scale

    def apply_fix(self, node, confidence, raw):
        needed = self.min_first_fix_confidence if self.map_position is None else self.min_confidence
        if confidence < needed:
            return "rejected_low_confidence"
        since = (self.last_raw[0] - raw[0], self.last_raw[1] - raw[1]) if self.last_raw else (0.0, 0.0)
        if self.map_position is not None:
            back = _rotate(since, self.heading_offset)
            at = (self.map_position[0] - back[0] * self.step_scale, self.map_position[1] - back[1] * self.step_scale)
            jump = math.dist(node, at)
            if jump > self.max_jump_m and confidence < self.jump_override_confidence:
                return "rejected_jump"
        path_to_capture = max(self.raw_path_since_base - math.hypot(*since), 0.0)
        keep_base = False
        if self.base_map is not None and self.base_raw is not None:
            map_vec = (node[0] - self.base_map[0], node[1] - self.base_map[1])
            raw_vec = (raw[0] - self.base_raw[0], raw[1] - self.base_raw[1])
            map_d, raw_d = math.hypot(*map_vec), math.hypot(*raw_vec)
            straight = path_to_capture > 0 and raw_d / path_to_capture > 0.8
            keep_base = straight and (map_d < self.min_calibration_distance_m or raw_d < self.min_calibration_distance_m)
            if map_d >= self.min_calibration_distance_m and raw_d >= self.min_calibration_distance_m and straight:
                observed = _wrap(_bearing(map_vec) - _bearing(raw_vec))
                if abs(_wrap(observed - self.heading_offset)) <= self.max_heading_correction_deg:
                    self.heading_offset = _wrap(self.heading_offset + self.learning_rate * _wrap(observed - self.heading_offset))
                    self.pending_large = None
                elif self.pending_large is not None and abs(_wrap(observed - self.pending_large)) <= self.large_correction_agreement_deg:
                    self.heading_offset = _wrap(self.pending_large + _wrap(observed - self.pending_large) / 2)
                    self.pending_large = None
                else:
                    self.pending_large = observed
                ratio = min(max(map_d / raw_d, self.step_scale_min), self.step_scale_max)
                self.step_scale = min(max(self.step_scale + self.learning_rate * (ratio - self.step_scale),
                                          self.step_scale_min), self.step_scale_max)
        self.map_position = node
        if keep_base:
            self.raw_path_since_base = path_to_capture
        else:
            self.base_map, self.base_raw, self.raw_path_since_base = node, raw, 0.0
        self.last_raw = raw
        self.distance_since_fix = 0.0
        return "accepted"

    def map_heading(self, raw_heading):
        return (raw_heading + self.heading_offset) % 360.0


# ── replay ─────────────────────────────────────────────────────────────────

def replay(session: Session, points, start_node: int):
    """points: [(group, truth_xy, truth_node)] in shooting order. Returns
    ({(group, dir): {...phone inputs...}}, {"north_jumps": n, "start_found": bool})."""
    shots = sorted((ms, g, d) for g, dirs in session.photos.items() for d, (ms, _) in dirs.items())
    truth = {g: (xy, n) for g, xy, n in points}
    first_ms = {g: min(ms for ms, _ in session.photos[g].values()) for g in truth}
    last_ms = {g: max(ms for ms, _ in session.photos[g].values()) for g in truth}
    start_row = next((i for i, name in session.marks if name == "入口"), None)

    est = MapPositionEstimator()
    fix_node = None
    north_jumps = 0
    out = {}
    pending_fix = sorted((last_ms[g], g) for g in truth)
    shot_i = fix_i = 0
    for i in range(len(session.t)):
        if i > 0 and abs(_wrap(session.yaw[i] - session.yaw[i - 1])) > NORTH_JUMP_DEG:
            est.absorb_raw_heading_jump(_wrap(session.yaw[i] - session.yaw[i - 1]))
            north_jumps += 1
        est.ingest(session.raw[i])
        if i == start_row:
            est.apply_fix(START_XY, 1.0, session.raw[i])
            est.align_heading(session.yaw[i], START_HEADING)
            fix_node = start_node
        nxt = session.t[i + 1] if i + 1 < len(session.t) else float("inf")
        while shot_i < len(shots) and shots[shot_i][0] < nxt:
            ms, g, d = shots[shot_i]
            pos = est.map_position
            out[(g, d)] = {
                "ms": ms, "raw": list(session.raw[i]), "yaw_raw": session.yaw[i], "steps": session.steps[i],
                "map_xy": list(pos) if pos else None,
                "dist_since_fix": est.distance_since_fix if pos else None,
                "fix_node": fix_node if pos else None,
                "map_heading": est.map_heading(session.yaw[i]) if pos else None,
                "heading_offset": est.heading_offset, "step_scale": est.step_scale,
            }
            shot_i += 1
        while fix_i < len(pending_fix) and pending_fix[fix_i][0] < nxt:
            g = pending_fix[fix_i][1]
            xy, node = truth[g]
            est.apply_fix(tuple(xy), 1.0, session.raw[session.row_at(first_ms[g])])
            fix_node = node
            fix_i += 1
    return out, {"north_jumps": north_jumps, "start_found": start_row is not None}
