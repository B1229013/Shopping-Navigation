"""Fuse three localization signals into one ranking: DINOv2 : sensor : grid.

- **dino**   — the phone's on-device DINOv2 top-K cosine similarities
  (DINOLocalizationService.swift). Cosines between any two store photos sit
  in a narrow high band, so they are min-max normalised over the top-K the
  phone sent: best → 1, K-th → 0, nodes outside the top-K → 0.
- **sensor** — the phone's PDR position in the reference-map frame (only sent
  once a photo fix has calibrated the PDR offset). Scored as exp(-d / 8 m),
  the same decay the phone uses in NodeCoordinateStore.fuseDINOWithPDR.
- **grid**   — the existing object/grid matcher score, already in [0, 1].

fused = Σ wᵢ·sᵢ / Σ wᵢ over the signals actually present, so a missing signal
drops out instead of dragging every node down. With neither dino nor sensor
present the grid candidates are returned untouched.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

SENSOR_DECAY_M = 8.0

Candidate = Tuple[int, float, str]


@dataclass(frozen=True)
class FusionWeights:
    dino: float
    sensor: float
    grid: float


def parse_weights(raw: str) -> FusionWeights:
    """"5,1,1" → FusionWeights(dino=5, sensor=1, grid=1)."""
    parts = [float(p) for p in raw.split(",")]
    if (len(parts) != 3 or not all(math.isfinite(p) and p >= 0 for p in parts)
            or sum(parts) == 0):
        raise ValueError(f"fusion weights must be 3 non-negative numbers 'dino,sensor,grid': {raw!r}")
    return FusionWeights(*parts)


def parse_dino_topk(raw: Optional[str], hint: Optional[int]) -> Dict[int, float]:
    """Phone's `dino_topk` form field (JSON [{"nid", "sim"}, ...]) → {nid: sim}.

    Older app builds only send `dino_node_hint`; treat that as a single top-1.
    """
    if raw:
        try:
            return {int(e["nid"]): float(e["sim"]) for e in json.loads(raw)}
        except (ValueError, TypeError, KeyError):
            log.warning("ignoring malformed dino_topk: %.120s", raw)
            return {}
    return {hint: 1.0} if hint is not None else {}


def sensor_sigma(dist_since_fix: Optional[float], sigma0: float, drift_rate: float) -> Optional[float]:
    """Expected sensor position error (m): the snap error at the last photo fix plus
    PDR drift that grows with distance walked since then."""
    if dist_since_fix is None:
        return None
    return sigma0 + drift_rate * max(dist_since_fix, 0.0)


def dino_gap(dino_sims: Dict[int, float]) -> Optional[float]:
    """Cosine gap between DINOv2's best and second-best node — its own confidence.
    On the leave-one-out test, top-1 was right 4 % of the time with a gap < 0.01 and
    40 % with a gap > 0.05."""
    if len(dino_sims) < 2:
        return None
    a, b = sorted(dino_sims.values(), reverse=True)[:2]
    return a - b


def adaptive_weights(base: FusionWeights, sigma: Optional[float], n_objects: Optional[int],
                     *, c: float, cap: float, n_full: float,
                     gap: Optional[float] = None, gap_full: float = 0.0) -> FusionWeights:
    """Weights that follow how trustworthy each signal is for *this* photo.

    - sensor: c/σ, capped — trusted right after a photo fix, faded out as drift grows;
      `base.sensor` when σ is unknown.
    - grid:   scaled by min(1, n_objects/n_full) — few detected objects means the
      object match is mostly luck. n_full=0 disables the scaling.
    - dino:   scaled by min(1, gap/gap_full) — DINOv2 is only worth trusting when its
      best node clearly beats the runner-up. gap_full=0 or gap None disables it.
    """
    ws = base.sensor if sigma is None else (cap if sigma <= 0 else min(cap, c / sigma))
    wg = base.grid
    if n_full > 0 and n_objects is not None:
        wg *= min(1.0, n_objects / n_full)
    wd = base.dino
    if gap_full > 0 and gap is not None:
        wd *= min(1.0, max(gap, 0.0) / gap_full)
    return FusionWeights(dino=wd, sensor=ws, grid=wg)


def estimate_position(candidates: List[Candidate], node_positions: Dict[int, Tuple[float, float]],
                      k: int = 3) -> Optional[Tuple[float, float]]:
    """Position between nodes: mean of the top-k node positions, weighted by how far
    each score is above the (k+1)-th — a clear winner pins it to its node, a tie
    puts it between the tied nodes."""
    top = [c for c in candidates if c[0] in node_positions][:k + 1]
    if not top:
        return None
    floor = top[k][1] if len(top) > k else 0.0
    ws = [(max(score - floor, 0.0), nid) for nid, score, _ in top[:k]]
    total = sum(w for w, _ in ws)
    if total <= 0:
        return node_positions[top[0][0]]
    x = sum(w * node_positions[n][0] for w, n in ws) / total
    y = sum(w * node_positions[n][1] for w, n in ws) / total
    return (x, y)


def _minmax(values: Dict[int, float]) -> Dict[int, float]:
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def fuse_candidates(
    grid_candidates: List[Candidate],
    dino_sims: Dict[int, float],
    sensor_pos: Optional[Tuple[float, float]],
    node_positions: Dict[int, Tuple[float, float]],
    weights: FusionWeights,
) -> List[Candidate]:
    use_dino = bool(dino_sims) and weights.dino > 0
    use_sensor = sensor_pos is not None and weights.sensor > 0
    if not (use_dino or use_sensor):
        return grid_candidates

    grid = {nid: (min(max(score, 0.0), 1.0), reason) for nid, score, reason in grid_candidates}
    dino = _minmax(dino_sims) if use_dino else {}
    total_w = weights.grid + (weights.dino if use_dino else 0.0) + (weights.sensor if use_sensor else 0.0)

    fused: List[Candidate] = []
    for nid in list(grid) + [n for n in dino if n not in grid]:
        g, grid_reason = grid.get(nid, (0.0, ""))
        d = dino.get(nid, 0.0)
        s = 0.0
        if use_sensor and nid in node_positions:
            nx, ny = node_positions[nid]
            s = math.exp(-math.hypot(nx - sensor_pos[0], ny - sensor_pos[1]) / SENSOR_DECAY_M)
        score = (weights.grid * g
                 + (weights.dino * d if use_dino else 0.0)
                 + (weights.sensor * s if use_sensor else 0.0)) / total_w
        parts = [f"dino={d:.2f}" if use_dino else None,
                 f"sensor={s:.2f}" if use_sensor else None,
                 f"grid={g:.2f}"]
        reason = "fused[" + " ".join(p for p in parts if p) + "]"
        if grid_reason:
            reason += f" {grid_reason}"
        fused.append((nid, score, reason))

    fused.sort(key=lambda c: -c[1])
    return fused


# ── v2: probabilistic fusion over every node ──────────────────────────────
#
# score_n = k(σ)·image_n − d_n² / (2σ²) + reach_n          posterior = softmax(score)
#
#   image_n  = DINOv2_n · min(1, gap/0.05) + grid_n + AISLE_WEIGHT · aisle_n
#              (+ the previous photo's image_n when a second photo was asked for)
#   d_n      = distance from the phone's map position to node n; σ = expected error
#   reach_n  = −REACH_PENALTY for nodes not walkable from the last fix in time
#
# Every node is a candidate (the sensor can pick a node no image matcher listed), the
# sensor term sharpens as σ shrinks, and the image weight k(σ) grows as the sensor
# drifts. The posterior is a real probability, so `confidence` means what it says.
# Parameters come from eval/fusion_v2.py (walk scenarios on the held-out photos).

IMAGE_K_MIN = 0.25         # image weight right after a fix (σ ≤ 2 m)
IMAGE_K_MAX = 4.0          # image weight once the sensor has drifted (σ ≥ 5 m)
IMAGE_K_NO_SENSOR = 2.0    # no map position yet: image only (calibrated confidence)
AISLE_WEIGHT = 1.0          # only used with a hand-checked maps/aisles.json
REACH_PENALTY = 3.0


def image_weight(sigma: Optional[float]) -> float:
    if sigma is None:
        return IMAGE_K_NO_SENSOR
    return min(IMAGE_K_MAX, max(IMAGE_K_MIN, IMAGE_K_MIN + (IMAGE_K_MAX - IMAGE_K_MIN) * (sigma - 2.0) / 3.0))


def image_evidence(grid_candidates: List[Candidate], dino_sims: Dict[int, float],
                   aisle: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    """Per-node image evidence (0..2, + aisle) from this photo alone: DINOv2 and grid
    at 4 : 1 (the best image-only ratio in the sweep), DINOv2 scaled by its own gap."""
    ev: Dict[int, float] = {}
    if dino_sims:
        gap = dino_gap(dino_sims)
        factor = 1.0 if gap is None else min(1.0, max(gap, 0.0) / 0.05)
        for nid, v in _minmax(dino_sims).items():
            ev[nid] = ev.get(nid, 0.0) + 1.6 * factor * v
    for nid, score, _ in grid_candidates:
        ev[nid] = ev.get(nid, 0.0) + 0.4 * min(max(score, 0.0), 1.0)
    for nid, v in (aisle or {}).items():
        ev[nid] = ev.get(nid, 0.0) + AISLE_WEIGHT * v
    return ev


def fuse_v2(evidence: Dict[int, float], node_positions: Dict[int, Tuple[float, float]], *,
            sensor_pos: Optional[Tuple[float, float]] = None, sigma: Optional[float] = None,
            reachable: Optional[set] = None) -> List[Candidate]:
    """Posterior over every node → [(nid, probability, reason)], best first."""
    k = image_weight(sigma if sensor_pos is not None else None)
    nids = list(node_positions)
    scores = []
    for nid in nids:
        s = k * evidence.get(nid, 0.0)
        parts = [f"img={evidence.get(nid, 0.0):.2f}"]
        if sensor_pos is not None and sigma:
            x, y = node_positions[nid]
            d = math.hypot(x - sensor_pos[0], y - sensor_pos[1])
            s -= d * d / (2 * sigma * sigma)
            parts.append(f"d={d:.1f}m")
        if reachable is not None and nid not in reachable:
            s -= REACH_PENALTY
            parts.append("unreachable")
        scores.append((nid, s, parts))
    top = max(s for _, s, _ in scores)
    weights = [math.exp(s - top) for _, s, _ in scores]
    total = sum(weights)
    out = [(nid, w / total, f"v2[k={k:.1f} σ={sigma or 0:.1f} " + " ".join(parts) + "]")
           for (nid, _, parts), w in zip(scores, weights)]
    out.sort(key=lambda c: -c[1])
    return out
