"""Path planner: A* shortest path + Held-Karp TSP for multi-target routing.

Solves the problem:
    current_position → [optimally ordered targets] → checkout → exit

The visit order is NOT the input list order — it is computed to minimise total
travel cost on the topological graph.  After arriving at each target the caller
should re-plan with the remaining targets (dynamic re-planning, approach B) so
mid-route changes (add / remove / reorder) are trivially supported.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A* on a topological graph
# ---------------------------------------------------------------------------

def astar(
    graph: nx.DiGraph,
    start: int,
    goal: int,
    *,
    weight: str = "weight",
) -> Optional[List[int]]:
    """A* shortest path on a weighted directed graph.

    Falls back to uniform weight = 1 for edges missing the *weight* attribute.
    Returns the node-id path [start, ..., goal], or None if unreachable.
    """
    if start == goal:
        return [start]

    # Ensure every edge has a numeric weight (default 1)
    for u, v, data in graph.edges(data=True):
        if weight not in data:
            data[weight] = 1.0

    try:
        # networkx A* — heuristic defaults to 0, equivalent to Dijkstra
        # on a graph without coordinates.  If nodes later get (x, y) we can
        # supply an Euclidean heuristic for a speedup.
        return list(nx.astar_path(graph, start, goal, weight=weight))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def astar_cost(
    graph: nx.DiGraph,
    start: int,
    goal: int,
    *,
    weight: str = "weight",
) -> float:
    """Return the A* path cost, or +inf if unreachable."""
    if start == goal:
        return 0.0
    for u, v, data in graph.edges(data=True):
        if weight not in data:
            data[weight] = 1.0
    try:
        return nx.astar_path_length(graph, start, goal, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return math.inf


# ---------------------------------------------------------------------------
# Pair-wise distance matrix (cached per call)
# ---------------------------------------------------------------------------

def _build_distance_matrix(
    graph: nx.DiGraph,
    nodes: Sequence[int],
    weight: str = "weight",
) -> Dict[Tuple[int, int], float]:
    """Pre-compute shortest-path cost for all ordered (i, j) pairs."""
    dist: Dict[Tuple[int, int], float] = {}
    for a in nodes:
        for b in nodes:
            if a != b:
                dist[(a, b)] = astar_cost(graph, a, b, weight=weight)
    return dist


# ---------------------------------------------------------------------------
# Held-Karp DP — exact TSP (optimal for ≤ 15 targets)
# ---------------------------------------------------------------------------

def _held_karp(
    dist: Dict[Tuple[int, int], float],
    start: int,
    targets: List[int],
    end: int,
) -> Tuple[List[int], float]:
    """Exact shortest Hamiltonian path: start → permutation(targets) → end.

    Uses bitmask DP (Held-Karp).  Complexity O(2^n · n²) where n = len(targets).
    Perfectly tractable for n ≤ 15 (a shopping list rarely exceeds that).

    Returns (ordered_targets, total_cost).  If any leg is unreachable the
    cost is +inf and the order is arbitrary.
    """
    n = len(targets)
    if n == 0:
        return [], dist.get((start, end), math.inf)

    if n == 1:
        cost = dist.get((start, targets[0]), math.inf) + dist.get((targets[0], end), math.inf)
        return list(targets), cost

    # Brute-force for very small n (simpler, same speed)
    if n <= 5:
        return _brute_force(dist, start, targets, end)

    # Held-Karp DP
    INF = math.inf
    ALL = (1 << n) - 1

    # dp[mask][i] = min cost to visit the subset encoded by *mask* ending at targets[i]
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    # Base: start → each target individually
    for i in range(n):
        c = dist.get((start, targets[i]), INF)
        dp[1 << i][i] = c

    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask][last] == INF:
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                leg = dist.get((targets[last], targets[nxt]), INF)
                new_cost = dp[mask][last] + leg
                if new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = last

    # Find the best last target before going to *end*
    best_cost = INF
    best_last = -1
    for i in range(n):
        total = dp[ALL][i] + dist.get((targets[i], end), INF)
        if total < best_cost:
            best_cost = total
            best_last = i

    if best_last == -1:
        return list(targets), INF

    # Trace back the order
    order_indices: List[int] = []
    mask = ALL
    cur = best_last
    while cur != -1:
        order_indices.append(cur)
        prev = parent[mask][cur]
        mask ^= (1 << cur)
        cur = prev
    order_indices.reverse()

    ordered = [targets[i] for i in order_indices]
    return ordered, best_cost


def _brute_force(
    dist: Dict[Tuple[int, int], float],
    start: int,
    targets: List[int],
    end: int,
) -> Tuple[List[int], float]:
    best_cost = math.inf
    best_perm: List[int] = list(targets)
    for perm in permutations(targets):
        cost = dist.get((start, perm[0]), math.inf)
        for i in range(len(perm) - 1):
            cost += dist.get((perm[i], perm[i + 1]), math.inf)
        cost += dist.get((perm[-1], end), math.inf)
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)
    return best_perm, best_cost


# ---------------------------------------------------------------------------
# Public API — RoutePlan
# ---------------------------------------------------------------------------

@dataclass
class RouteLeg:
    """One segment of the full route: from_node → to_node."""
    from_node: int
    to_node: int
    path: List[int]          # full A* path including both endpoints
    cost: float
    actions: List[str]       # human-readable directions for each edge in path
    purpose: str             # "target" | "checkout" | "exit"


@dataclass
class RoutePlan:
    """Complete optimised route from current position through all targets,
    then checkout, then exit."""
    legs: List[RouteLeg] = field(default_factory=list)
    visit_order: List[int] = field(default_factory=list)   # target nodes in optimal order
    total_cost: float = math.inf
    checkout_node: Optional[int] = None
    exit_node: Optional[int] = None

    @property
    def full_path(self) -> List[int]:
        """Concatenated node sequence (no duplicates at leg boundaries)."""
        if not self.legs:
            return []
        result = list(self.legs[0].path)
        for leg in self.legs[1:]:
            result.extend(leg.path[1:])  # skip first node (= prev leg's last)
        return result

    @property
    def all_directions(self) -> List[str]:
        """Flat list of turn-by-turn directions across all legs."""
        dirs: List[str] = []
        for leg in self.legs:
            dirs.extend(leg.actions)
        return dirs

    def to_dict(self) -> dict:
        return {
            "visit_order": self.visit_order,
            "total_cost": self.total_cost,
            "checkout_node": self.checkout_node,
            "exit_node": self.exit_node,
            "full_path": self.full_path,
            "legs": [
                {
                    "from": leg.from_node,
                    "to": leg.to_node,
                    "path": leg.path,
                    "cost": leg.cost,
                    "actions": leg.actions,
                    "purpose": leg.purpose,
                }
                for leg in self.legs
            ],
        }


def plan_route(
    graph: nx.DiGraph,
    start: int,
    targets: List[int],
    checkout: Optional[int] = None,
    exit_node: Optional[int] = None,
    *,
    weight: str = "weight",
) -> RoutePlan:
    """Plan an optimised route: start → targets (best order) → checkout → exit.

    - *targets* may be empty (just go to checkout then exit).
    - *checkout* may be None (skip checkout, go straight to exit after targets).
    - *exit_node* may be None (route ends at last target / checkout).
    - Duplicate target nodes are collapsed.

    Returns a RoutePlan with the full path broken into legs.
    """
    # Deduplicate targets while preserving first-seen order
    seen = set()
    unique_targets: List[int] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique_targets.append(t)

    # Build the fixed suffix (nodes after all targets)
    suffix: List[int] = []
    if checkout is not None:
        suffix.append(checkout)
    if exit_node is not None:
        suffix.append(exit_node)

    # Remove targets that coincide with suffix nodes — they will be visited
    # as part of the fixed tail anyway, so including them in the TSP causes
    # a wasteful round-trip (visit the node as a target, leave for the next
    # target, then come back for checkout/exit).
    suffix_set = set(suffix)
    absorbed = [t for t in unique_targets if t in suffix_set]
    unique_targets = [t for t in unique_targets if t not in suffix_set]
    if absorbed:
        log.info("Targets absorbed into suffix (same node): %s", absorbed)

    # All key nodes for distance matrix
    key_nodes_set = {start} | set(unique_targets) | set(suffix)
    key_nodes = list(key_nodes_set)
    dist = _build_distance_matrix(graph, key_nodes, weight=weight)

    # Determine the "end" node for TSP — the first node of the suffix,
    # or start if there's no suffix and no targets (degenerate case).
    if suffix:
        tsp_end = suffix[0]
    elif unique_targets:
        # No checkout/exit: the TSP just optimises target order,
        # last target is the final destination.
        tsp_end = None
    else:
        return RoutePlan()  # nothing to do

    # Solve TSP for the targets
    if tsp_end is not None:
        ordered_targets, _tsp_cost = _held_karp(dist, start, unique_targets, tsp_end)
    else:
        # No fixed endpoint — try every target as last stop
        best_cost = math.inf
        best_order: List[int] = list(unique_targets)
        for candidate_last in unique_targets:
            remaining = [t for t in unique_targets if t != candidate_last]
            if remaining:
                order, cost_part = _held_karp(dist, start, remaining, candidate_last)
                order.append(candidate_last)
            else:
                order = [candidate_last]
                cost_part = dist.get((start, candidate_last), math.inf)
            if cost_part < best_cost:
                best_cost = cost_part
                best_order = order
        ordered_targets = best_order

    # Build the full node sequence, collapsing consecutive duplicates
    # (e.g. last target == checkout, or start == first target)
    raw_waypoints = [start] + ordered_targets + suffix
    waypoints: List[int] = [raw_waypoints[0]]
    for w in raw_waypoints[1:]:
        if w != waypoints[-1]:
            waypoints.append(w)

    # Build legs
    legs: List[RouteLeg] = []
    total_cost = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        path = astar(graph, a, b, weight=weight)
        cost = dist.get((a, b), astar_cost(graph, a, b, weight=weight))

        # Collect edge actions for directions
        actions: List[str] = []
        if path and len(path) > 1:
            for j in range(len(path) - 1):
                edge_data = graph.edges.get((path[j], path[j + 1]), {})
                action = edge_data.get("action", f"Move from {path[j]} to {path[j+1]}")
                actions.append(action)

        # Determine purpose
        if b == exit_node:
            purpose = "exit"
        elif b == checkout:
            purpose = "checkout"
        else:
            purpose = "target"

        leg = RouteLeg(
            from_node=a,
            to_node=b,
            path=path or [a, b],
            cost=cost,
            actions=actions,
            purpose=purpose,
        )
        legs.append(leg)
        total_cost += cost

    plan = RoutePlan(
        legs=legs,
        visit_order=ordered_targets,
        total_cost=total_cost,
        checkout_node=checkout,
        exit_node=exit_node,
    )

    log.info(
        "Route planned: %s → %s | cost=%.1f | legs=%d",
        start,
        " → ".join(str(n) for n in ordered_targets + suffix),
        total_cost,
        len(legs),
    )
    return plan


# ---------------------------------------------------------------------------
# Dynamic re-planning helper
# ---------------------------------------------------------------------------

def replan_after_arrival(
    graph: nx.DiGraph,
    current_node: int,
    remaining_targets: List[int],
    checkout: Optional[int] = None,
    exit_node: Optional[int] = None,
    *,
    weight: str = "weight",
) -> RoutePlan:
    """Re-plan the route after arriving at a target (or after a mid-route change).

    This is the same as plan_route but named explicitly for clarity.
    Call it whenever:
      - A target is reached (remove it from remaining_targets)
      - The user adds a new target mid-route
      - The user removes a target mid-route
      - Visual localization corrects the current position
    """
    return plan_route(
        graph, current_node, remaining_targets,
        checkout=checkout, exit_node=exit_node,
        weight=weight,
    )
