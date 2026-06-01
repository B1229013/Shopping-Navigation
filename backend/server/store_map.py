"""Pre-built topological map of the CSIE Department Office from photo survey."""
from __future__ import annotations

from server.topomap import TopoMap
from server.store_knowledge import ROOMS

# Node ID assignments (matching ROOMS):
#   0 = Entrance Hallway
#   1 = Main Lobby (Sofa Area)          ← starting position
#   2 = Coffee Station (mini-fridge #1)
#   3 = Department Chair Office
#   4 = Admin / Reception Area (refrigerator #2 - silver)
#   5 = Professor Corridor A (Left)
#   6 = Professor Corridor B (Right)
#   7 = Filing Cabinet Corridor
#   8 = Refrigerator Area (refrigerator #3 - white, near fire hydrant)
#   9 = Professor Corridor C (Back)

NODE_ENTRANCE = 0
NODE_LOBBY = 1
NODE_COFFEE = 2
NODE_CHAIR_OFFICE = 3
NODE_ADMIN = 4
NODE_CORRIDOR_A = 5
NODE_CORRIDOR_B = 6
NODE_CABINET_CORRIDOR = 7
NODE_FRIDGE_AREA = 8
NODE_CORRIDOR_C = 9

# Keep old names for import compatibility
NODE_BACK_WALL = NODE_FRIDGE_AREA
NODE_LEFT_FRESH = NODE_COFFEE
NODE_LEFT_HOTPOT = NODE_ADMIN
NODE_ENTRANCE_COMPAT = NODE_ENTRANCE


def build_store_topomap() -> TopoMap:
    """Build the complete topological map of the CSIE Department Office.

    Returns a TopoMap with 10 nodes representing key locations and edges
    encoding walking directions through the office.
    """
    topo = TopoMap()

    # --- Add all nodes ---
    for room in ROOMS:
        nid = topo.add_node(
            photo_path="",
            detected=room.landmarks_en[:4],
            summary=f"{room.name_en} ({room.name_zh})",
        )
        assert nid == room.node_id

    # --- Edges: Entrance ↔ Main Lobby ---
    topo.add_edge(NODE_ENTRANCE, NODE_LOBBY,
                  action="Walk forward through the entrance into the main lobby")
    topo.add_edge(NODE_LOBBY, NODE_ENTRANCE,
                  action="Walk back toward the entrance hallway")

    # --- Edges: Main Lobby ↔ Coffee Station ---
    topo.add_edge(NODE_LOBBY, NODE_COFFEE,
                  action="Turn left and walk toward the coffee station along the left wall")
    topo.add_edge(NODE_COFFEE, NODE_LOBBY,
                  action="Walk back to the main lobby sofa area")

    # --- Edges: Coffee Station ↔ Department Chair Office ---
    topo.add_edge(NODE_COFFEE, NODE_CHAIR_OFFICE,
                  action="Continue past the coffee machine to the Department Chair's office")
    topo.add_edge(NODE_CHAIR_OFFICE, NODE_COFFEE,
                  action="Walk back toward the coffee station")

    # --- Edges: Main Lobby ↔ Admin Area ---
    topo.add_edge(NODE_LOBBY, NODE_ADMIN,
                  action="Walk toward the back-right of the lobby to the admin/reception area")
    topo.add_edge(NODE_ADMIN, NODE_LOBBY,
                  action="Walk back to the main lobby")

    # --- Edges: Coffee Station ↔ Professor Corridor A ---
    topo.add_edge(NODE_COFFEE, NODE_CORRIDOR_A,
                  action="Walk past the coffee station into the left professor corridor")
    topo.add_edge(NODE_CORRIDOR_A, NODE_COFFEE,
                  action="Walk back toward the coffee station")

    # --- Edges: Admin Area ↔ Professor Corridor B ---
    topo.add_edge(NODE_ADMIN, NODE_CORRIDOR_B,
                  action="Walk from the admin area into the right professor corridor")
    topo.add_edge(NODE_CORRIDOR_B, NODE_ADMIN,
                  action="Walk back toward the admin/reception area")

    # --- Edges: Professor Corridor A ↔ Filing Cabinet Corridor ---
    topo.add_edge(NODE_CORRIDOR_A, NODE_CABINET_CORRIDOR,
                  action="Continue down the corridor to the filing cabinet area")
    topo.add_edge(NODE_CABINET_CORRIDOR, NODE_CORRIDOR_A,
                  action="Walk back through the left professor corridor")

    # --- Edges: Filing Cabinet Corridor ↔ Refrigerator Area ---
    topo.add_edge(NODE_CABINET_CORRIDOR, NODE_FRIDGE_AREA,
                  action="Walk to the end of the filing cabinet corridor toward the refrigerator")
    topo.add_edge(NODE_FRIDGE_AREA, NODE_CABINET_CORRIDOR,
                  action="Walk back from the refrigerator into the cabinet corridor")

    # --- Edges: Filing Cabinet Corridor ↔ Professor Corridor C ---
    topo.add_edge(NODE_CABINET_CORRIDOR, NODE_CORRIDOR_C,
                  action="Turn and walk into the back professor corridor")
    topo.add_edge(NODE_CORRIDOR_C, NODE_CABINET_CORRIDOR,
                  action="Walk back to the filing cabinet corridor")

    # --- Edges: Professor Corridor C ↔ Coffee Station (loop back) ---
    topo.add_edge(NODE_CORRIDOR_C, NODE_COFFEE,
                  action="Continue down the corridor back toward the coffee station and lobby")
    topo.add_edge(NODE_COFFEE, NODE_CORRIDOR_C,
                  action="Walk past the coffee station into the back professor corridor")

    # --- Edges: Professor Corridor B ↔ Professor Corridor C ---
    topo.add_edge(NODE_CORRIDOR_B, NODE_CORRIDOR_C,
                  action="Continue down the right corridor toward the back professor offices")
    topo.add_edge(NODE_CORRIDOR_C, NODE_CORRIDOR_B,
                  action="Walk back along the corridor toward the right professor offices")

    return topo


# Module-level singleton
_store_map: TopoMap | None = None


def get_store_topomap() -> TopoMap:
    """Get or create the singleton environment topological map."""
    global _store_map
    if _store_map is None:
        _store_map = build_store_topomap()
    return _store_map
