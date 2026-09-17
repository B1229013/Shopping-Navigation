"""Tests for the path planner: A* + TSP multi-target routing."""
from __future__ import annotations

import math

import networkx as nx
import pytest

from server.path_planner import (
    RoutePlan,
    astar,
    astar_cost,
    plan_route,
    replan_after_arrival,
)


# ---------------------------------------------------------------------------
# Fixture: a small weighted directed graph
#
#     0 ──(2)──> 1 ──(3)──> 2
#     |          |          |
#    (1)       (1)        (4)
#     v          v          v
#     3 ──(1)──> 4 ──(2)──> 5
#
# (number) = edge weight.  All edges are bidirectional with equal weight.
# ---------------------------------------------------------------------------

@pytest.fixture
def graph():
    g = nx.DiGraph()
    edges = [
        (0, 1, 2), (1, 0, 2),
        (1, 2, 3), (2, 1, 3),
        (0, 3, 1), (3, 0, 1),
        (1, 4, 1), (4, 1, 1),
        (2, 5, 4), (5, 2, 4),
        (3, 4, 1), (4, 3, 1),
        (4, 5, 2), (5, 4, 2),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, weight=w, action=f"Go from {u} to {v}")
    return g


# ---------------------------------------------------------------------------
# A* basic tests
# ---------------------------------------------------------------------------

class TestAstar:
    def test_same_node(self, graph):
        assert astar(graph, 3, 3) == [3]
        assert astar_cost(graph, 3, 3) == 0.0

    def test_direct_edge(self, graph):
        path = astar(graph, 0, 1)
        assert path == [0, 1]
        assert astar_cost(graph, 0, 1) == 2.0

    def test_shortest_path(self, graph):
        # 0→5: cheapest is 0→3→4→5 = 1+1+2 = 4
        path = astar(graph, 0, 5)
        cost = astar_cost(graph, 0, 5)
        assert cost == 4.0
        assert path == [0, 3, 4, 5]

    def test_unreachable(self):
        g = nx.DiGraph()
        g.add_node(0)
        g.add_node(1)
        assert astar(g, 0, 1) is None
        assert astar_cost(g, 0, 1) == math.inf

    def test_default_weight(self):
        """Edges without explicit weight should default to 1."""
        g = nx.DiGraph()
        g.add_edge(0, 1, action="go")
        g.add_edge(1, 2, action="go")
        assert astar_cost(g, 0, 2) == 2.0


# ---------------------------------------------------------------------------
# TSP / plan_route tests
# ---------------------------------------------------------------------------

class TestPlanRoute:
    def test_single_target(self, graph):
        plan = plan_route(graph, start=0, targets=[5])
        assert plan.visit_order == [5]
        assert plan.total_cost == 4.0  # 0→3→4→5
        assert plan.full_path[0] == 0
        assert plan.full_path[-1] == 5

    def test_single_target_with_exit(self, graph):
        plan = plan_route(graph, start=0, targets=[2], exit_node=3)
        # 0→2: 0→1→2 cost 5.  2→3: 2→1→4→3 cost 5.  Total 10.
        assert plan.visit_order == [2]
        assert plan.full_path[0] == 0
        assert plan.full_path[-1] == 3

    def test_multi_target_order_optimised(self, graph):
        # Targets: 5 and 1.  Starting at 0.
        # If we went 0→5→1: 4 + 3 = 7
        # If we went 0→1→5: 2 + 3 = 5  ← better
        plan = plan_route(graph, start=0, targets=[5, 1])
        assert plan.visit_order == [1, 5]
        assert plan.total_cost == 5.0

    def test_multi_target_with_checkout_and_exit(self, graph):
        # Targets: 2, 4.  checkout=5, exit=0.
        plan = plan_route(graph, start=0, targets=[2, 4],
                          checkout=5, exit_node=0)
        assert plan.full_path[0] == 0
        assert plan.full_path[-1] == 0
        # Last two stops should be checkout(5) → exit(0)
        assert 5 in plan.full_path
        # The route must visit both targets before checkout
        path = plan.full_path
        idx_2 = path.index(2)
        idx_4 = path.index(4)
        idx_5 = path.index(5)
        assert idx_2 < idx_5
        assert idx_4 < idx_5

    def test_empty_targets_with_exit(self, graph):
        # No targets, just go to exit
        plan = plan_route(graph, start=0, targets=[], exit_node=3)
        assert plan.visit_order == []
        assert plan.full_path == [0, 3]
        assert plan.total_cost == 1.0

    def test_duplicate_targets_collapsed(self, graph):
        plan = plan_route(graph, start=0, targets=[5, 5, 5])
        assert plan.visit_order == [5]

    def test_legs_have_actions(self, graph):
        plan = plan_route(graph, start=0, targets=[5])
        assert len(plan.legs) == 1
        assert len(plan.legs[0].actions) > 0
        assert all(isinstance(a, str) for a in plan.legs[0].actions)

    def test_to_dict(self, graph):
        plan = plan_route(graph, start=0, targets=[1, 5], exit_node=0)
        d = plan.to_dict()
        assert "visit_order" in d
        assert "legs" in d
        assert "total_cost" in d
        assert isinstance(d["legs"], list)


# ---------------------------------------------------------------------------
# Dynamic re-planning tests
# ---------------------------------------------------------------------------

class TestReplan:
    def test_replan_removes_visited(self, graph):
        # Original: 0 → [1, 5] → exit(3)
        plan1 = plan_route(graph, start=0, targets=[1, 5], exit_node=3)
        assert 1 in plan1.visit_order
        assert 5 in plan1.visit_order

        # Arrived at first target (1), re-plan from node 1
        plan2 = replan_after_arrival(graph, current_node=1,
                                     remaining_targets=[5], exit_node=3)
        assert plan2.visit_order == [5]
        assert plan2.full_path[0] == 1
        assert plan2.full_path[-1] == 3

    def test_replan_with_added_target(self, graph):
        # Was going to [5], now also need [2]
        plan = replan_after_arrival(graph, current_node=1,
                                   remaining_targets=[5, 2], exit_node=0)
        assert set(plan.visit_order) == {5, 2}
        assert plan.full_path[0] == 1
        assert plan.full_path[-1] == 0

    def test_replan_all_done(self, graph):
        # All targets visited, just go to exit
        plan = replan_after_arrival(graph, current_node=5,
                                   remaining_targets=[], exit_node=0)
        assert plan.visit_order == []
        assert plan.full_path[0] == 5
        assert plan.full_path[-1] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_start_is_target(self, graph):
        plan = plan_route(graph, start=0, targets=[0], exit_node=3)
        assert plan.full_path[0] == 0
        assert plan.full_path[-1] == 3

    def test_start_is_exit(self, graph):
        plan = plan_route(graph, start=0, targets=[5], exit_node=0)
        assert plan.full_path[0] == 0
        assert plan.full_path[-1] == 0

    def test_many_targets(self):
        """Test with more targets to exercise Held-Karp DP (> 5 nodes)."""
        g = nx.DiGraph()
        # Simple chain: 0 → 1 → 2 → ... → 9
        for i in range(9):
            g.add_edge(i, i + 1, weight=1, action=f"{i}→{i+1}")
            g.add_edge(i + 1, i, weight=1, action=f"{i+1}→{i}")

        plan = plan_route(g, start=0, targets=[2, 4, 6, 8, 1, 3], exit_node=9)
        # Optimal order on a line: visit in ascending order
        assert plan.visit_order == [1, 2, 3, 4, 6, 8]
        assert plan.total_cost == 9.0  # 0→1→2→3→4→6→8→9

    def test_no_targets_no_exit(self, graph):
        plan = plan_route(graph, start=0, targets=[])
        assert plan.legs == []

    def test_target_same_as_checkout_absorbed(self, graph):
        """Bug fix: when a target node == checkout node, the target should be
        absorbed into the suffix — no wasteful round-trip back to it."""
        # Target=5, checkout=5, exit=0.  Without the fix this would produce
        # ..→5→(other)→5→0 with a duplicate visit to node 5.
        plan = plan_route(graph, start=0, targets=[5], checkout=5, exit_node=0)
        # Node 5 is absorbed — visit_order should be empty (the target IS checkout)
        assert plan.visit_order == []
        assert plan.full_path[0] == 0
        assert plan.full_path[-1] == 0
        # Node 5 should appear exactly once in the path (as checkout, not as
        # both a target leg and a checkout leg)
        assert plan.full_path.count(5) == 1

    def test_target_same_as_exit_absorbed(self, graph):
        """Target node == exit node should be absorbed."""
        plan = plan_route(graph, start=0, targets=[3], exit_node=3)
        assert plan.visit_order == []
        assert plan.full_path[0] == 0
        assert plan.full_path[-1] == 3

    def test_target_same_as_checkout_with_other_targets(self, graph):
        """One target matches checkout, others don't — only the matching one
        is absorbed."""
        plan = plan_route(graph, start=0, targets=[1, 5],
                          checkout=5, exit_node=0)
        # Target 5 absorbed into checkout; target 1 remains
        assert plan.visit_order == [1]
        assert plan.full_path[0] == 0
        assert plan.full_path[-1] == 0
        # Route: 0→1→…→5(checkout)→…→0(exit)
        path = plan.full_path
        assert path.index(1) < path.index(5)

    def test_consecutive_duplicate_waypoints_collapsed(self, graph):
        """When last target == checkout, the waypoint list should not have
        two consecutive identical nodes generating a zero-cost leg."""
        # Target=5, checkout=4, exit=0.
        # Path to 5 goes through 4 (0→3→4→5), then back to 4 for checkout.
        # But if we set target=4, checkout=4: old code would try
        # waypoints = [0, 4, 4, 0] generating a useless 4→4 leg.
        # With the absorption fix, target 4 is absorbed.  But we also test
        # the dedup at the waypoint level.
        plan = plan_route(graph, start=3, targets=[4], checkout=4, exit_node=0)
        assert plan.visit_order == []  # absorbed
        # No zero-cost legs
        for leg in plan.legs:
            assert leg.from_node != leg.to_node

    def test_start_equals_first_target_no_empty_leg(self, graph):
        """start == first target should not produce an empty 0-cost leg."""
        plan = plan_route(graph, start=0, targets=[0, 5], exit_node=3)
        for leg in plan.legs:
            assert leg.from_node != leg.to_node
