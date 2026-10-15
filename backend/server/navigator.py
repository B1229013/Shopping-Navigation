"""Navigation engine: converts object/location queries into step-by-step directions.

Supports both single-target lookup (legacy) and multi-target optimised routing
with dynamic re-planning (A* + Held-Karp TSP via path_planner).
"""
from __future__ import annotations

import logging
from typing import List, Optional

import networkx as nx

from server.path_planner import RoutePlan, plan_route, replan_after_arrival
from server.store_knowledge import LocationMatch, find_product
from server.store_map import (
    NODE_ENTRANCE,
    NODE_LOBBY,
    get_store_topomap,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-target navigation (legacy, kept for backward compatibility)
# ---------------------------------------------------------------------------

def navigate_to_product(
    query: str,
    start_node: int = NODE_LOBBY,
) -> Optional[dict]:
    """Find an object/location and return step-by-step navigation directions.

    Returns a dict with:
        query, matched_products, location, location_zh,
        directions, aisle_number, zone, confidence
    Or None if no match found.
    """
    matches = find_product(query)
    if not matches:
        return None

    best: LocationMatch = matches[0]
    topo = get_store_topomap()

    target_node = best.aisle_number
    if target_node is None:
        return None

    try:
        path = nx.shortest_path(topo.graph, source=start_node, target=target_node)
    except nx.NetworkXNoPath:
        return None

    directions: List[str] = []
    if start_node == NODE_LOBBY:
        directions.append("Start at the main lobby sofa area (大廳沙發區)")
    elif start_node == NODE_ENTRANCE:
        directions.append("Start at the entrance (入口)")

    for i in range(len(path) - 1):
        edge_data = topo.graph.edges[path[i], path[i + 1]]
        directions.append(edge_data["action"])

    directions.append(
        f"You've arrived at {best.display_en} — look for {best.matched_keyword}"
    )

    matched_products = list(dict.fromkeys(m.matched_keyword for m in matches[:5]))

    return {
        "query": query,
        "matched_products": matched_products,
        "location": best.display_en,
        "location_zh": best.display_zh,
        "directions": directions,
        "aisle_number": best.aisle_number,
        "zone": best.zone_name,
        "confidence": round(best.confidence, 3),
    }


# ---------------------------------------------------------------------------
# Multi-target navigation (new)
# ---------------------------------------------------------------------------

def resolve_targets(queries: List[str]) -> List[dict]:
    """Resolve a list of search queries to map node IDs.

    Returns a list of dicts, each with:
        query, node_id, display_en, display_zh, confidence, matched_keyword
    Items that cannot be found are included with node_id = None.
    """
    results = []
    for q in queries:
        matches = find_product(q)
        if matches:
            best = matches[0]
            results.append({
                "query": q,
                "node_id": best.aisle_number,
                "display_en": best.display_en,
                "display_zh": best.display_zh,
                "confidence": round(best.confidence, 3),
                "matched_keyword": best.matched_keyword,
            })
        else:
            results.append({
                "query": q,
                "node_id": None,
                "display_en": None,
                "display_zh": None,
                "confidence": 0.0,
                "matched_keyword": None,
            })
    return results


def plan_multi_target_route(
    graph: nx.DiGraph,
    start_node: int,
    target_nodes: List[int],
    checkout_node: Optional[int] = None,
    exit_node: Optional[int] = None,
) -> RoutePlan:
    """Plan an optimised route through multiple targets, then checkout, then exit.

    This is the main entry point for multi-target navigation.
    The order of target_nodes does NOT determine the visit order — the planner
    finds the shortest total path.
    """
    return plan_route(
        graph, start_node, target_nodes,
        checkout=checkout_node, exit_node=exit_node,
    )


def replan_route(
    graph: nx.DiGraph,
    current_node: int,
    remaining_targets: List[int],
    checkout_node: Optional[int] = None,
    exit_node: Optional[int] = None,
) -> RoutePlan:
    """Re-plan after arriving at a target or after a mid-route change.

    Call when:
      - A target is reached → remove it and re-plan from current position
      - User adds/removes a target → update the list and re-plan
      - Visual localization corrects the position → re-plan from corrected node
    """
    return replan_after_arrival(
        graph, current_node, remaining_targets,
        checkout=checkout_node, exit_node=exit_node,
    )
