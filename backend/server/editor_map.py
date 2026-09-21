"""Hand-corrected PDR waypoint map → one walkable store graph the phone can follow.

The map comes from the PDR Waypoint Editor (``backend/maps/*.editor.json``): a few
recorded walks whose points were grid-snapped by hand. Here they become a single
graph — consecutive points of a walk are edges, and any two waypoints closer than
``link_radius`` (different walks crossing the same aisle) are linked too — with
metre lengths on every edge. Product / sign labels are merged in from the Neo4j
reference map by proximity, so "牛奶" resolves to a waypoint, and a route is a
polyline plus turn list that the iOS PathFollower walks with the phone's sensors.

Coordinate frame: metres, origin at the entrance, heading 0° = +y, 90° = +x
(identical to the iOS MotionPathTracker and to heading.heading_between_nodes).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from server.heading import convert_leg_to_relative, merge_instructions, next_instruction_text

LINK_RADIUS_M = 1.5        # walks passing within this distance are joined
DEDUPE_EPS_M = 0.05        # consecutive points closer than this are one waypoint
LABEL_MAX_DIST_M = 2.0     # a Neo4j node farther than this from every waypoint stays unmapped
SIMPLIFY_EPS_M = 0.5       # polyline jogs shorter than this are folded into the next edge
TURN_PENALTY_M = 3.0       # a route "costs" this many extra metres per turn (≥ TURN_MIN_DEG)
TURN_MIN_DEG = 45.0
SIGN_HINTS = ("區", "吊牌", "標示", "招牌", "標籤牌", "sign", "category", "section", "aisle")


@dataclass
class EditorGraph:
    place: str
    graph: nx.Graph
    entrance: int
    _walk_index: Dict[tuple, int] = field(default_factory=dict)
    _neo_to_wp: Dict[int, int] = field(default_factory=dict)

    def waypoint_for_neo(self, nid: int) -> Optional[int]:
        return self._neo_to_wp.get(nid)

    def position(self, wp: int) -> tuple[float, float]:
        n = self.graph.nodes[wp]
        return n["x"], n["y"]

    def nodes_payload(self) -> List[dict]:
        return [{"id": wp, "x": d["x"], "y": d["y"], "walk": d["walk"], "products": d["products"]}
                for wp, d in sorted(self.graph.nodes(data=True))]

    def edges_payload(self) -> List[dict]:
        return [{"from": a, "to": b, "length": round(d["length"], 2)}
                for a, b, d in self.graph.edges(data=True)]


def waypoint_id(g: EditorGraph, walk: str, idx: int) -> int:
    return g._walk_index[(walk, idx)]


def build_graph(data: dict, link_radius: float = LINK_RADIUS_M,
                dedupe_eps: float = DEDUPE_EPS_M) -> EditorGraph:
    graph = nx.Graph()
    walk_index: Dict[tuple, int] = {}
    next_id = 0
    for walk in data.get("walks", []):
        name = walk["name"]
        prev_id: Optional[int] = None
        kept = 0
        for x, y in walk["points"]:
            x, y = float(x), float(y)
            if prev_id is not None:
                px, py = graph.nodes[prev_id]["x"], graph.nodes[prev_id]["y"]
                if math.hypot(x - px, y - py) < dedupe_eps:
                    continue
            wp = next_id
            next_id += 1
            graph.add_node(wp, x=x, y=y, walk=name, idx=kept, products=[], neo_nids=[])
            walk_index[(name, kept)] = wp
            kept += 1
            if prev_id is not None:
                graph.add_edge(prev_id, wp, length=math.hypot(x - graph.nodes[prev_id]["x"],
                                                              y - graph.nodes[prev_id]["y"]))
            prev_id = wp

    # Join walks that pass through the same spot so the whole store is one graph.
    nodes = list(graph.nodes(data=True))
    for i, (a, da) in enumerate(nodes):
        for b, db in nodes[i + 1:]:
            if graph.has_edge(a, b):
                continue
            d = math.hypot(da["x"] - db["x"], da["y"] - db["y"])
            if d <= link_radius:
                graph.add_edge(a, b, length=d)

    # A waypoint lying beside another walk's *edge* (a corridor recorded as one
    # long segment) joins both of that edge's endpoints, so the corridor can be
    # entered/left there instead of forcing a U-turn at the far end. Pass-through
    # traffic still prefers the original straight edge (triangle inequality).
    for wp, d in list(graph.nodes(data=True)):
        for a, b, ed in list(graph.edges(data=True)):
            if wp in (a, b) or graph.nodes[a]["walk"] == d["walk"]:
                continue
            ax, ay = graph.nodes[a]["x"], graph.nodes[a]["y"]
            bx, by = graph.nodes[b]["x"], graph.nodes[b]["y"]
            vx, vy = bx - ax, by - ay
            seg2 = vx * vx + vy * vy
            if seg2 == 0:
                continue
            t = ((d["x"] - ax) * vx + (d["y"] - ay) * vy) / seg2
            if not 0.05 < t < 0.95:
                continue   # near an endpoint → the point-to-point rule above covers it
            px, py = ax + t * vx, ay + t * vy
            if math.hypot(d["x"] - px, d["y"] - py) <= link_radius:
                for end in (a, b):
                    if not graph.has_edge(wp, end):
                        graph.add_edge(wp, end, length=math.hypot(d["x"] - graph.nodes[end]["x"],
                                                                  d["y"] - graph.nodes[end]["y"]))

    entrance = min(graph.nodes, key=lambda n: math.hypot(graph.nodes[n]["x"], graph.nodes[n]["y"])) \
        if graph.number_of_nodes() else 0
    return EditorGraph(place=data.get("place", ""), graph=graph, entrance=entrance,
                       _walk_index=walk_index)


def nearest_waypoint(g: EditorGraph, x: float, y: float) -> int:
    return min(g.graph.nodes, key=lambda n: math.hypot(g.graph.nodes[n]["x"] - x,
                                                        g.graph.nodes[n]["y"] - y))


def merge_labels(g: EditorGraph, neo_nodes: List[dict], max_dist: float = LABEL_MAX_DIST_M) -> List[int]:
    """Attach each Neo4j node's labels to the nearest waypoint within ``max_dist``.

    ``neo_nodes``: [{"nid", "x", "y", "labels": [...]}, ...]. Returns the nids that
    were too far from every waypoint (the recorded walks never went there).
    """
    unmapped: List[int] = []
    for n in neo_nodes:
        wp = nearest_waypoint(g, n["x"], n["y"])
        if math.hypot(g.graph.nodes[wp]["x"] - n["x"], g.graph.nodes[wp]["y"] - n["y"]) > max_dist:
            unmapped.append(n["nid"])
            continue
        node = g.graph.nodes[wp]
        for label in n.get("labels", []):
            if label and label not in node["products"]:
                node["products"].append(label)
        node["neo_nids"].append(n["nid"])
        g._neo_to_wp[n["nid"]] = wp
    return unmapped


def merge_from_ref_map(g: EditorGraph, ref_map, max_dist: float = LABEL_MAX_DIST_M) -> List[int]:
    """Adapter for server.neo4j_client.RefMap: labels = object labels + OCR text."""
    neo_nodes = []
    for nid, node in ref_map.photos.items():
        labels: List[str] = []
        for o in node.objects:
            for t in (o.label, o.ocr_text):
                t = (t or "").strip()
                if t and t not in labels:
                    labels.append(t)
        neo_nodes.append({"nid": nid, "x": node.pdr_x, "y": node.pdr_y, "labels": labels})
    return merge_labels(g, neo_nodes, max_dist=max_dist)


def _is_sign(label: str) -> bool:
    low = label.lower()
    return any(h in low for h in SIGN_HINTS)


def find_goal_waypoint(g: EditorGraph, goal_item: str) -> Optional[int]:
    """Waypoint whose merged labels mention ``goal_item`` (case-insensitive substring).

    A category/aisle sign naming the item ("牛奶區吊牌") is the section itself and
    counts 3×; plain product mentions (milk powder, milk candy on a snack shelf…)
    count 1×, so scattered mentions don't outvote the real section.
    """
    needle = (goal_item or "").lower().strip()
    if not needle:
        return None
    best, best_score = None, 0
    for wp, d in g.graph.nodes(data=True):
        score = sum((3 if _is_sign(p) else 1) for p in d["products"] if needle in p.lower())
        if score > best_score:
            best, best_score = wp, score
    return best


def _turn_deg(g: EditorGraph, a: int, b: int, c: int) -> float:
    """Absolute heading change (degrees) when walking a→b→c."""
    (ax, ay), (bx, by), (cx, cy) = g.position(a), g.position(b), g.position(c)
    h1 = math.degrees(math.atan2(bx - ax, by - ay))
    h2 = math.degrees(math.atan2(cx - bx, cy - by))
    d = (h2 - h1 + 180) % 360 - 180
    return abs(d)


def plan(g: EditorGraph, start: int, goal: int,
         turn_penalty: float = TURN_PENALTY_M) -> List[int]:
    """Shortest route in metres, with each turn costing ``turn_penalty`` extra.

    On a grid of aisles many routes tie on distance; without the penalty the
    planner may pick a zigzag through every aisle. Dijkstra over (node, came_from)
    states makes the turn cost part of the edge cost.
    """
    if start not in g.graph or goal not in g.graph:
        return []
    if start == goal:
        return [start]
    import heapq
    best: Dict[tuple, float] = {(start, None): 0.0}
    prev: Dict[tuple, tuple] = {}
    heap = [(0.0, start, None)]
    while heap:
        cost, node, came = heapq.heappop(heap)
        if cost > best.get((node, came), math.inf):
            continue
        if node == goal:
            path, state = [], (node, came)
            while state is not None:
                path.append(state[0])
                state = prev.get(state)
            return path[::-1]
        for nxt in g.graph.neighbors(node):
            if nxt == came:
                continue
            step = g.graph.edges[node, nxt]["length"]
            if came is not None and _turn_deg(g, came, node, nxt) >= TURN_MIN_DEG:
                step += turn_penalty
            state = (nxt, node)
            if cost + step < best.get(state, math.inf):
                best[state] = cost + step
                prev[state] = (node, came)
                heapq.heappush(heap, (cost + step, nxt, node))
    return []


def _simplified(g: EditorGraph, path: List[int], eps: float = SIMPLIFY_EPS_M) -> List[int]:
    """Drop waypoints that sit within ``eps`` of the previous kept one (the tiny
    cross-link jog where two walks meet) so they don't become spurious turns."""
    if not path:
        return path
    kept = [path[0]]
    for wp in path[1:-1]:
        px, py = g.position(kept[-1])
        x, y = g.position(wp)
        if math.hypot(x - px, y - py) >= eps:
            kept.append(wp)
    if len(path) > 1:
        kept.append(path[-1])
    return kept


def route_payload(g: EditorGraph, path: List[int], user_heading: Optional[float] = None) -> dict:
    """Polyline + turn list + total distance for a planned path."""
    if len(path) < 2:
        pos = [list(g.position(path[0]))] if path else []
        return {"path": path, "polyline": pos, "turns": [], "distance_m": 0.0, "next_instruction_zh": ""}

    distance = sum(g.graph.edges[a, b]["length"] for a, b in zip(path, path[1:]))
    simple = _simplified(g, path)
    positions = {wp: g.position(wp) for wp in simple}
    edge_len = {}
    for a, b in zip(simple, simple[1:]):
        (ax, ay), (bx, by) = positions[a], positions[b]
        edge_len[(a, b)] = math.hypot(bx - ax, by - ay)

    if user_heading is None:  # face along the first edge → the first step reads "直走"
        (ax, ay), (bx, by) = positions[simple[0]], positions[simple[1]]
        user_heading = math.degrees(math.atan2(bx - ax, by - ay)) % 360

    steps = merge_instructions(convert_leg_to_relative(simple, positions, user_heading, distances=edge_len))
    turns = [{
        "direction": st.direction.value,
        "text_zh": st.text_zh,
        "at": [round(v, 2) for v in positions[st.from_node]],
        "to": [round(v, 2) for v in positions[st.to_node]],
        "distance_m": round(st.distance_m, 2),
        "passed_nodes": st.passed_nodes,
    } for st in steps]

    return {
        "path": path,
        "polyline": [[round(x, 2), round(y, 2)] for x, y in (positions[wp] for wp in simple)],
        "turns": turns,
        "distance_m": round(distance, 2),
        "next_instruction_zh": next_instruction_text(steps),
    }


def load_from_file(path: str | Path) -> EditorGraph:
    return build_graph(json.loads(Path(path).read_text(encoding="utf-8")))
