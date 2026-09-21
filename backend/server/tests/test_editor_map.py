"""Editor map: the hand-corrected PDR waypoints become one walkable store graph,
Neo4j product labels are merged onto the nearest waypoint, and a route is a
polyline the phone can follow."""
import math

import pytest

from server import editor_map as em


def _two_walks():
    # walk A: north up a corridor; walk B: crosses it at (0, 6) and continues east
    return {
        "place": "test",
        "walks": [
            {"name": "A", "points": [[0, 0], [0, 6], [0, 12]]},
            {"name": "B", "points": [[-6, 6.3], [0.2, 6.1], [6, 6]]},   # passes 0.22 m from A's (0,6)
        ],
    }


class TestBuildGraph:
    def test_sequential_edges_carry_metre_lengths(self):
        g = em.build_graph(_two_walks())
        a0, a1 = em.waypoint_id(g, "A", 0), em.waypoint_id(g, "A", 1)
        assert g.graph.has_edge(a0, a1)
        assert g.graph.edges[a0, a1]["length"] == pytest.approx(6.0)

    def test_walks_that_cross_are_linked(self):
        g = em.build_graph(_two_walks())
        a1, b1 = em.waypoint_id(g, "A", 1), em.waypoint_id(g, "B", 1)
        assert g.graph.has_edge(a1, b1)          # 0.22 m apart -> linked
        assert g.graph.edges[a1, b1]["length"] == pytest.approx(math.hypot(0.2, 0.1), abs=1e-6)

    def test_far_apart_points_are_not_linked(self):
        g = em.build_graph(_two_walks())
        a0, b0 = em.waypoint_id(g, "A", 0), em.waypoint_id(g, "B", 0)
        assert not g.graph.has_edge(a0, b0)

    def test_duplicate_consecutive_points_are_collapsed(self):
        g = em.build_graph({"place": "t", "walks": [{"name": "A", "points": [[0, 0], [0, 0], [0, 5]]}]})
        assert g.graph.number_of_nodes() == 2

    def test_entrance_is_the_waypoint_nearest_the_origin(self):
        g = em.build_graph(_two_walks())
        assert g.entrance == em.waypoint_id(g, "A", 0)


class TestMergeLabels:
    def _neo(self):
        return [
            {"nid": 26, "x": 0.4, "y": 11.6, "labels": ["麵包 bread", "價格牌 129"]},   # 0.57 m from A2
            {"nid": 3,  "x": 5.5, "y": 6.4,  "labels": ["牛奶 milk"]},                  # 0.64 m from B2
            {"nid": 99, "x": 20,  "y": 20,   "labels": ["far away"]},                   # nothing near
        ]

    def test_labels_land_on_nearest_waypoint_within_radius(self):
        g = em.build_graph(_two_walks())
        unmapped = em.merge_labels(g, self._neo(), max_dist=2.0)
        a2 = em.waypoint_id(g, "A", 2)
        assert g.graph.nodes[a2]["products"] == ["麵包 bread", "價格牌 129"]
        assert g.waypoint_for_neo(26) == a2
        assert unmapped == [99]

    def test_goal_item_resolves_by_substring_on_products(self):
        g = em.build_graph(_two_walks())
        em.merge_labels(g, self._neo())
        assert em.find_goal_waypoint(g, "牛奶") == em.waypoint_id(g, "B", 2)
        assert em.find_goal_waypoint(g, "泡麵") is None


class TestRoute:
    def test_shortest_path_uses_cross_link(self):
        g = em.build_graph(_two_walks())
        path = em.plan(g, em.waypoint_id(g, "A", 0), em.waypoint_id(g, "B", 2))
        assert path == [em.waypoint_id(g, "A", 0), em.waypoint_id(g, "A", 1),
                        em.waypoint_id(g, "B", 1), em.waypoint_id(g, "B", 2)]

    def test_route_payload_has_polyline_turns_and_distance(self):
        g = em.build_graph(_two_walks())
        em.merge_labels(g, [{"nid": 3, "x": 5.5, "y": 6.4, "labels": ["牛奶 milk"]}])
        path = em.plan(g, g.entrance, em.find_goal_waypoint(g, "牛奶"))
        r = em.route_payload(g, path, user_heading=0.0)
        assert r["polyline"][0] == [0.0, 0.0] and r["polyline"][-1] == [6.0, 6.0]
        assert r["distance_m"] == pytest.approx(6.0 + math.hypot(0.2, 0.1) + math.hypot(5.8, 0.1), abs=0.01)
        # north 6 m, then the cross-link + east leg are one right turn
        assert r["turns"][0]["direction"] == "straight"
        assert r["turns"][1]["direction"] in ("right", "slight_right")
        assert r["turns"][1]["at"] == [0.0, 6.0]

    def test_nearest_waypoint(self):
        g = em.build_graph(_two_walks())
        assert em.nearest_waypoint(g, 0.3, 5.5) == em.waypoint_id(g, "A", 1)


class TestStraightPreference:
    def _grid(self):
        # 2x3 grid, 4 m spacing: (0,0)-(4,0)-(8,0) / (0,4)-(4,4)-(8,4), all edges present.
        # From (0,0) to (8,4) the zigzag and the L-shaped route are both 12 m long.
        return {"place": "t", "walks": [
            {"name": "row0", "points": [[0, 0], [4, 0], [8, 0]]},
            {"name": "row1", "points": [[0, 4], [4, 4], [8, 4]]},
            {"name": "col0", "points": [[0, 0.2], [0, 4.2]]},
            {"name": "col1", "points": [[4, 0.2], [4, 4.2]]},
            {"name": "col2", "points": [[8, 0.2], [8, 4.2]]},
        ]}

    def test_planner_prefers_the_route_with_fewer_turns(self):
        g = em.build_graph(self._grid())
        start, goal = em.waypoint_id(g, "row0", 0), em.waypoint_id(g, "row1", 2)
        r = em.route_payload(g, em.plan(g, start, goal))
        assert r["distance_m"] == pytest.approx(12.0, abs=0.6)
        assert len(r["turns"]) == 2            # straight along the row, ONE turn, straight — not a zigzag


class TestGoalScoring:
    def test_category_sign_beats_scattered_product_mentions(self):
        g = em.build_graph(_two_walks())
        em.merge_labels(g, [
            {"nid": 1, "x": 0, "y": 12, "labels": ["紅牛奶粉罐", "牛奶糖", "KLIM 純牛奶 milk powder"]},  # 3 mentions, no sign
            {"nid": 2, "x": 6, "y": 6,  "labels": ["牛奶區吊牌 milk category sign", "牛奶紙盒"]},      # sign + 1
        ])
        assert em.find_goal_waypoint(g, "牛奶") == em.waypoint_id(g, "B", 2)


class TestEdgeLinking:
    def test_point_lying_on_another_walks_edge_joins_it(self):
        # Walk L is one long 20 m corridor edge; walk T ends 0.4 m beside its midpoint.
        g = em.build_graph({"place": "t", "walks": [
            {"name": "L", "points": [[0, 0], [0, 20]]},
            {"name": "T", "points": [[5, 10], [0.4, 10]]},
        ]})
        t1 = em.waypoint_id(g, "T", 1)
        l0, l1 = em.waypoint_id(g, "L", 0), em.waypoint_id(g, "L", 1)
        assert g.graph.has_edge(t1, l0) and g.graph.has_edge(t1, l1)
        # so walking the corridor north can turn into T without a U-turn
        path = em.plan(g, l0, em.waypoint_id(g, "T", 0))
        assert path == [l0, t1, em.waypoint_id(g, "T", 0)]

    def test_point_far_from_the_edge_is_not_joined(self):
        g = em.build_graph({"place": "t", "walks": [
            {"name": "L", "points": [[0, 0], [0, 20]]},
            {"name": "T", "points": [[5, 10], [3, 10]]},
        ]})
        assert not g.graph.has_edge(em.waypoint_id(g, "T", 1), em.waypoint_id(g, "L", 0))
