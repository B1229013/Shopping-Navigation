"""Navigation engine: converts object/location queries into step-by-step directions."""
from __future__ import annotations

from typing import List, Optional

import networkx as nx

from server.store_knowledge import LocationMatch, find_product
from server.store_map import (
    NODE_ENTRANCE,
    NODE_LOBBY,
    get_store_topomap,
)


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
