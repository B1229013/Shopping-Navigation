"""Route planning on TopoGraphV2 — A* shortest path, multi-goal TSP."""
from __future__ import annotations

import itertools
import math
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from server.config import normalize_label
from server.topomap_v2 import TopoGraphV2, ETYPE_WALKWAY, NTYPE_OBJECT

log = logging.getLogger(__name__)

_DIST_RE = re.compile(r"([\d.]+)\s*m\)")

# Label normalization values that are too generic to serve as landmarks
_GENERIC_LABEL_NORMS = {
    "shelf", "rack", "box", "bag", "package", "sign", "label",
    "display", "cart", "basket", "ceiling", "floor", "wall",
    "pillar", "light", "aisle", "door",
}


@dataclass
class RouteResult:
    path: List[int]
    target_photo_id: int
    distance: float
    hops: int


def _get_pdr_xy(node_data: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract PDR coordinates from node, checking both top-level and sensor_data."""
    x = node_data.get("pdr_x")
    y = node_data.get("pdr_y")
    if x is not None and y is not None:
        return x, y
    sd = node_data.get("sensor_data", {})
    if sd:
        x = sd.get("data_pdr_x") or sd.get("pdr_x")
        y = sd.get("data_pdr_y") or sd.get("pdr_y")
        if x is not None and y is not None:
            return x, y
    return None, None


def _parse_edge_distance(edge_data: dict) -> Optional[float]:
    """Extract distance from edge's distance_m field or direction string."""
    dm = edge_data.get("distance_m")
    if dm is not None and dm > 0:
        return dm
    direction = edge_data.get("direction", "")
    m = _DIST_RE.search(direction)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _build_photo_undirected(topo: TopoGraphV2) -> nx.Graph:
    """Build undirected graph of photo nodes connected by walkway edges."""
    ug = nx.Graph()
    ug.add_nodes_from(topo.all_photo_nodes())
    for u, v, d in topo.graph.edges(data=True):
        if d.get("etype") == ETYPE_WALKWAY and u != v:
            w = _parse_edge_distance(d) or 1.0
            if w <= 0:
                w = 1.0
            if ug.has_edge(u, v):
                if w < ug[u][v]["weight"]:
                    ug[u][v]["weight"] = w
            else:
                ug.add_edge(u, v, weight=w)
    return ug


def _heuristic(topo: TopoGraphV2, u: int, v: int) -> float:
    """Euclidean heuristic from PDR coordinates. Returns 0 when unavailable (admissible)."""
    nu = topo.graph.nodes.get(u, {})
    nv = topo.graph.nodes.get(v, {})
    ux, uy = _get_pdr_xy(nu)
    vx, vy = _get_pdr_xy(nv)
    if ux is not None and uy is not None and vx is not None and vy is not None:
        return math.sqrt((ux - vx) ** 2 + (uy - vy) ** 2)
    return 0.0


def plan_route(
    topo: TopoGraphV2,
    source_photo_id: int,
    goal_photo_ids: List[int],
) -> Optional[RouteResult]:
    """Find shortest path from source to nearest goal photo node.

    Uses A* with Euclidean heuristic when PDR coordinates exist on nodes,
    otherwise degrades to Dijkstra (heuristic=0, still optimal).
    """
    if not goal_photo_ids:
        return None
    if source_photo_id in goal_photo_ids:
        return RouteResult(path=[source_photo_id], target_photo_id=source_photo_id,
                           distance=0.0, hops=0)

    ug = _build_photo_undirected(topo)
    if source_photo_id not in ug:
        log.warning("route: source photo_id=%d not in walkway graph", source_photo_id)
        return None

    best: Optional[RouteResult] = None
    for gid in goal_photo_ids:
        if gid not in ug:
            continue
        try:
            path = nx.astar_path(
                ug, source_photo_id, gid,
                heuristic=lambda u, v: _heuristic(topo, u, v),
                weight="weight",
            )
            dist = sum(ug[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))
            if best is None or dist < best.distance:
                best = RouteResult(path=path, target_photo_id=gid,
                                   distance=round(dist, 2), hops=len(path) - 1)
        except nx.NetworkXNoPath:
            continue

    return best


def _summarize_photo_objects(
    topo: TopoGraphV2, photo_id: int, max_items: int = 4,
    goal_objects: Optional[List[str]] = None,
) -> List[str]:
    """Extract short landmark labels for a photo node, filtering out generics.

    When goal_objects is provided, goal-matching objects are listed first.
    """
    objs = topo.photo_objects(photo_id)

    if goal_objects:
        goal_norms = {normalize_label(g) for g in goal_objects}
        goal_lows = {g.lower() for g in goal_objects}

        def _is_goal_match(o: dict) -> bool:
            ln = (o.get("label_norm") or "").lower()
            raw = (o.get("label") or "").lower()
            if ln in goal_norms:
                return True
            return any(g in raw or raw in g for g in goal_lows if g)

        objs = sorted(objs, key=lambda o: (0 if _is_goal_match(o) else 1))

    landmarks: List[str] = []
    seen_norms: set[str] = set()
    for o in objs:
        label_norm = (o.get("label_norm") or "").lower()
        if label_norm in _GENERIC_LABEL_NORMS or label_norm in seen_norms:
            continue
        seen_norms.add(label_norm)
        raw = o.get("label", "")
        # Prefer the Chinese portion before the English for brevity
        zh = raw.split(" ")[0] if raw else ""
        # If zh is very long (full description), try to extract just the key noun
        if len(zh) > 8:
            # Labels are like "左側貨架上的茶罐 tea tins" — grab the tail noun
            for sep in ("的", "上"):
                if sep in zh:
                    zh = zh.rsplit(sep, 1)[-1]
                    break
        landmarks.append(zh if zh else raw)
        if len(landmarks) >= max_items:
            break
    return landmarks


def _describe_node(
    topo: TopoGraphV2, photo_id: int,
    goal_objects: Optional[List[str]] = None,
) -> str:
    """One-line description of a waypoint: region (if set) + key objects."""
    node = topo.graph.nodes.get(photo_id, {})
    region = node.get("region") or ""
    landmarks = _summarize_photo_objects(topo, photo_id, max_items=3,
                                         goal_objects=goal_objects)
    if region and landmarks:
        return f"{region}（{'、'.join(landmarks)}）"
    if region:
        return region
    if landmarks:
        return "、".join(landmarks)
    return f"位置 P{topo._photo_sequence_rank(photo_id)}"


def build_route_waypoints(
    topo: TopoGraphV2, route: RouteResult,
    goal_objects: Optional[List[str]] = None,
) -> List[dict]:
    """Build waypoint info for each photo node along the route."""
    waypoints = []
    for i, pid in enumerate(route.path):
        node = topo.graph.nodes.get(pid, {})
        landmarks = _summarize_photo_objects(topo, pid, max_items=6,
                                             goal_objects=goal_objects)
        rank = topo._photo_sequence_rank(pid)
        role = "start" if i == 0 else ("goal" if i == len(route.path) - 1 else "waypoint")
        waypoints.append({
            "photo_id": pid,
            "rank": rank,
            "role": role,
            "region": node.get("region", ""),
            "objects": landmarks,
            "description": _describe_node(topo, pid, goal_objects=goal_objects),
        })
    return waypoints


def _relative_turn(prev_heading: float, next_heading: float) -> str:
    """Return a human-readable turn instruction from heading change."""
    diff = (next_heading - prev_heading + 360) % 360
    if diff <= 30 or diff >= 330:
        return "直走"
    elif diff < 70:
        return "稍微右轉"
    elif diff <= 110:
        return "右轉"
    elif diff < 160:
        return "向右後方走"
    elif diff <= 200:
        return "迴轉"
    elif diff < 250:
        return "向左後方走"
    elif diff <= 290:
        return "左轉"
    else:
        return "稍微左轉"


def _heading_between(topo: TopoGraphV2, a: int, b: int) -> Optional[float]:
    """Compute heading (degrees, 0=north) from node a to node b using PDR coords."""
    ad = topo.graph.nodes.get(a, {})
    bd = topo.graph.nodes.get(b, {})
    ax, ay = _get_pdr_xy(ad)
    bx, by = _get_pdr_xy(bd)
    if ax is None or ay is None or bx is None or by is None:
        return None
    dx, dy = bx - ax, by - ay
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360


def _edge_distance(topo: TopoGraphV2, a: int, b: int) -> Optional[float]:
    """Get distance between two adjacent nodes."""
    for _, v, d in topo.graph.out_edges(a, data=True):
        if v == b and d.get("etype") == ETYPE_WALKWAY:
            return _parse_edge_distance(d)
    for _, v, d in topo.graph.out_edges(b, data=True):
        if v == a and d.get("etype") == ETYPE_WALKWAY:
            return _parse_edge_distance(d)
    return None


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings (0..180)."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# A lateral step shorter than this (metres) is a grid-alignment jog, not a real
# turn — absorb it into the current leg instead of emitting a turn for it.
_LEG_JOG_MIN_M = 2.0
# Bearings within this many degrees belong to the same straight leg.
_LEG_SAME_DIR_DEG = 30.0


def _route_legs(topo: TopoGraphV2, path: List[int]) -> List[dict]:
    """Collapse a node path into straight legs.

    Consecutive segments heading roughly the same way are merged, and short
    (<2m) jogs are absorbed without starting a new leg — this stops the
    grid-aligned waypoints from producing spurious "右轉，左轉，右轉" chatter.
    Returns [{end_node, bearing, dist}, ...]; empty if no PDR headings exist.
    """
    legs: List[dict] = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        heading = _heading_between(topo, a, b)
        d = _edge_distance(topo, a, b)
        d = d if d and d > 0 else 1.0
        if heading is None:
            if legs:
                legs[-1]["dist"] += d
                legs[-1]["end_node"] = b
            continue
        if legs and (_angle_diff(legs[-1]["bearing"], heading) <= _LEG_SAME_DIR_DEG
                     or d < _LEG_JOG_MIN_M):
            # Same direction, or a short jog: extend current leg, keep its bearing.
            legs[-1]["dist"] += d
            legs[-1]["end_node"] = b
        else:
            legs.append({"end_node": b, "bearing": heading, "dist": d})
    return legs


def format_route_guidance(
    topo: TopoGraphV2,
    route: RouteResult,
    source_rank: int,
    target_rank: int,
    goal_objects: Optional[List[str]] = None,
    user_heading: Optional[float] = None,
) -> str:
    """Generate step-by-step walking guidance with turn directions.

    ``user_heading`` is the direction the user is actually facing (degrees,
    0=north). When provided, the first turn is computed relative to it, so a goal
    behind or to the side of the user yields a real opening turn instead of an
    incorrect "直走".
    """
    if route.hops == 0:
        return "你已經在目標位置！"

    path = route.path
    goal_desc = _describe_node(topo, route.target_photo_id, goal_objects=goal_objects)

    legs = _route_legs(topo, path)
    if not legs:
        # No PDR headings — fall back to simple distance guidance.
        return f"往前走約 {route.distance:.0f}m（{route.hops} 步），目標：{goal_desc}"

    steps: List[str] = []
    prev_heading = user_heading
    for idx, leg in enumerate(legs):
        is_last = idx == len(legs) - 1
        if prev_heading is not None:
            turn = _relative_turn(prev_heading, leg["bearing"])
            if turn != "直走":
                steps.append(turn)
        dist_text = f"約 {leg['dist']:.0f}m" if leg["dist"] >= 2 else ""
        if is_last:
            if dist_text:
                steps.append(f"直走{dist_text}即可到達{goal_desc}")
            else:
                steps.append(f"前方即是{goal_desc}")
        else:
            landmark = _describe_node(topo, leg["end_node"], goal_objects=goal_objects)
            steps.append(f"直走{dist_text}到{landmark}" if dist_text
                         else f"直走到{landmark}")
        prev_heading = leg["bearing"]

    return "，".join(steps)


# ── Direction computation ──

_DIRECTION_ANGLE_OFFSET = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": 270.0,
}


def _get_front_heading(node_data: dict) -> Optional[float]:
    """Return the stored 'front' capture heading (degrees, 0=north/+y) if present.

    This is the real direction the front photo faced, imported from the map's
    per-direction photo headings. Checks top-level and sensor_data.

    NOTE: 0.0 is a valid heading (facing north), so we test ``is not None`` and
    must never treat a 0 heading as "missing" (an earlier bug fell back to
    guessing from walkway edges, which pointed the wrong way).
    """
    for container in (node_data, node_data.get("sensor_data", {}) or {}):
        h = container.get("heading_deg")
        if h is not None:
            return float(h) % 360.0
    return None


def _infer_front_heading_deg(topo: TopoGraphV2, photo_id: int) -> Optional[float]:
    """Determine the 'front' heading at a photo node.

    Prefers the stored capture heading (``sensor_data.heading_deg``), which is
    the actual compass direction the front photo faced. Only when that is
    unavailable does it fall back to guessing from walkway edges (the direction
    of an adjacent node), which is a rough approximation.
    """
    node_data = topo.graph.nodes.get(photo_id, {})

    # Preferred: the real stored front heading from the map's directional photos.
    stored = _get_front_heading(node_data)
    if stored is not None:
        return stored

    nx, ny = _get_pdr_xy(node_data)
    if nx is None or ny is None:
        return None

    # Fallback: infer from walkway geometry. Try outgoing walkway edge first.
    for _, v, d in topo.graph.out_edges(photo_id, data=True):
        if d.get("etype") != ETYPE_WALKWAY or v == photo_id:
            continue
        vdata = topo.graph.nodes.get(v, {})
        vx, vy = _get_pdr_xy(vdata)
        if vx is not None and vy is not None:
            dx, dy = vx - nx, vy - ny
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                return math.degrees(math.atan2(dx, dy)) % 360

    # Fall back to incoming walkway edge (previous node to this node)
    for u, _, d in topo.graph.in_edges(photo_id, data=True):
        if d.get("etype") != ETYPE_WALKWAY or u == photo_id:
            continue
        udata = topo.graph.nodes.get(u, {})
        ux, uy = _get_pdr_xy(udata)
        if ux is not None and uy is not None:
            dx, dy = nx - ux, ny - uy
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                return math.degrees(math.atan2(dx, dy)) % 360

    return None


def user_facing_heading(
    topo: TopoGraphV2, photo_id: int, matched_direction: Optional[str],
) -> Optional[float]:
    """The direction the user is actually facing (degrees, 0=north).

    That is the node's front heading rotated by the matched view direction
    (front/right/back/left) the query photo aligned to. Returns None when the
    front heading is unavailable OR ``matched_direction`` is empty (facing
    unknown) — in that case ``format_route_guidance`` omits the opening turn
    rather than guessing one that may fight the camera view.
    """
    if not matched_direction:
        return None
    fh = _infer_front_heading_deg(topo, photo_id)
    if fh is None:
        return None
    off = _DIRECTION_ANGLE_OFFSET.get(matched_direction, 0.0)
    return (fh + off) % 360.0


def compute_relative_direction(
    topo: TopoGraphV2,
    current_photo_id: int,
    next_photo_id: int,
    matched_direction: str,
) -> Optional[str]:
    """Compute the relative direction from the user to the next waypoint.

    Returns a human-readable direction string like '左前方', '右後方', '正前方', '後方'.
    Returns None if coordinates are unavailable OR the user's facing is unknown
    (``matched_direction`` empty). When facing is unknown we must NOT assert a
    turn — an over-confident "往後方走" that contradicts what the camera sees is
    worse than no direction at all; the caller falls back to landmarks only.
    """
    if not matched_direction:
        return None
    cur_data = topo.graph.nodes.get(current_photo_id, {})
    nxt_data = topo.graph.nodes.get(next_photo_id, {})
    cx, cy = _get_pdr_xy(cur_data)
    tx, ty = _get_pdr_xy(nxt_data)
    if cx is None or cy is None or tx is None or ty is None:
        return None

    dx, dy = tx - cx, ty - cy
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return "正前方"

    target_bearing = math.degrees(math.atan2(dx, dy)) % 360

    front_heading = _infer_front_heading_deg(topo, current_photo_id)
    if front_heading is None:
        return None

    dir_offset = _DIRECTION_ANGLE_OFFSET.get(matched_direction, 0.0)
    user_heading = (front_heading + dir_offset) % 360

    relative = (target_bearing - user_heading + 360) % 360

    if relative <= 30 or relative >= 330:
        return "正前方"
    elif relative < 60:
        return "右前方"
    elif relative <= 120:
        return "右方"
    elif relative < 150:
        return "右後方"
    elif relative <= 210:
        return "後方"
    elif relative < 240:
        return "左後方"
    elif relative <= 300:
        return "左方"
    else:
        return "左前方"


# ── Multi-goal TSP ──

@dataclass
class TspResult:
    order: List[int]
    total_distance: float
    per_leg: List[float] = field(default_factory=list)


def _shortest_distance(ug: nx.Graph, a: int, b: int) -> float:
    """Shortest-path distance between two nodes. Returns inf if unreachable."""
    try:
        return nx.dijkstra_path_length(ug, a, b, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def solve_tsp(
    topo: TopoGraphV2,
    start_photo_id: int,
    goal_photo_ids: List[int],
) -> Optional[TspResult]:
    """Open-path TSP: find the best visit order starting from start_photo_id.

    Uses brute-force permutation (fine for < 10 goals).
    For >= 10 goals, falls back to greedy nearest-neighbor.
    """
    if not goal_photo_ids:
        return None

    reachable = [g for g in goal_photo_ids if g != start_photo_id]
    if not reachable:
        return TspResult(order=[], total_distance=0.0, per_leg=[])

    ug = _build_photo_undirected(topo)
    if start_photo_id not in ug:
        return None

    nodes = [start_photo_id] + reachable
    n = len(nodes)
    dist_cache: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = _shortest_distance(ug, nodes[i], nodes[j])
            dist_cache[(nodes[i], nodes[j])] = d
            dist_cache[(nodes[j], nodes[i])] = d

    def tour_cost(perm: Tuple[int, ...]) -> Tuple[float, List[float]]:
        legs = []
        prev = start_photo_id
        for idx in perm:
            d = dist_cache.get((prev, idx), float("inf"))
            legs.append(d)
            prev = idx
        return sum(legs), legs

    if len(reachable) <= 8:
        best_cost = float("inf")
        best_perm: Tuple[int, ...] = tuple(reachable)
        best_legs: List[float] = []
        for perm in itertools.permutations(reachable):
            cost, legs = tour_cost(perm)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
                best_legs = legs
    else:
        # Greedy nearest-neighbor for larger sets
        remaining = set(reachable)
        best_perm_list: List[int] = []
        best_legs = []
        current = start_photo_id
        while remaining:
            nearest = min(remaining, key=lambda g: dist_cache.get((current, g), float("inf")))
            d = dist_cache.get((current, nearest), float("inf"))
            best_perm_list.append(nearest)
            best_legs.append(d)
            current = nearest
            remaining.remove(nearest)
        best_perm = tuple(best_perm_list)
        best_cost = sum(best_legs)

    if best_cost == float("inf"):
        return None

    goal_id_to_sub_idx = {gid: i for i, gid in enumerate(goal_photo_ids)}
    order = [goal_id_to_sub_idx[pid] for pid in best_perm if pid in goal_id_to_sub_idx]

    return TspResult(
        order=order,
        total_distance=round(best_cost, 2),
        per_leg=[round(d, 2) for d in best_legs],
    )
