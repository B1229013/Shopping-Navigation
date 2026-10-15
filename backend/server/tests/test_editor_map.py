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


def test_far_node_still_gets_a_position_but_donates_no_labels():
    """A Neo4j node whose recorded walk passed a few metres off the editor walks
    must still map to a waypoint — otherwise localizing there yields no route at
    all (4 of 6 photos in session 7c600e4a). Its product labels stay out, so
    goal resolution keeps the tighter radius."""
    g = em.build_graph({"walks": [{"name": "w1", "points": [[0, 0], [0, 5], [0, 10]]}]})
    neo = [
        {"nid": 1, "x": 0.5, "y": 5.0, "labels": ["優格"]},     # 0.5 m: close, labels count
        {"nid": 2, "x": 3.0, "y": 5.0, "labels": ["狗飼料"]},   # 3 m: position only
        {"nid": 3, "x": 30.0, "y": 5.0, "labels": ["電視"]},    # 30 m: nowhere near
    ]

    unmapped = em.merge_labels(g, neo)

    assert unmapped == [3]
    assert g.waypoint_for_neo(1) is not None
    assert g.waypoint_for_neo(2) is not None          # routes can start here
    assert g.waypoint_for_neo(3) is None
    products = g.graph.nodes[g.waypoint_for_neo(2)]["products"]
    assert "優格" in products and "狗飼料" not in products


# ── Route graph built from the Neo4j reference map ───────────────────────

from server.neo4j_client import RefMap, RefObject, RefPhotoNode


def _obj(label, ocr=""):
    return RefObject(label=label, label_norm=label, score=0.9, ocr_text=ocr, role="商品", grid_cell="c")


def _neo_map():
    """L-shaped aisle: 1 (0,2) → 2 (0,8) → 3 (5,8); 4 sits 0.4 m from 1 but has no WALKWAY."""
    m = RefMap(place="test")
    for nid, (x, y), objs, nbrs in [
        (1, (0, 2), [_obj("貨架")], [2]),
        (2, (0, 8), [_obj("牛奶區吊牌")], [3]),           # WALKWAY stored one-way
        (3, (5, 8), [_obj("優格", ocr="特價")], []),
        (4, (0.4, 2), [_obj("推車")], []),
    ]:
        m.photos[nid] = RefPhotoNode(nid=nid, photo_file=f"{nid}.jpg", pdr_x=x, pdr_y=y, heading_deg=0.0,
                                     session="s", total_steps=0, total_distance_m=0.0, objects=objs,
                                     neighbor_nids=nbrs)
    return m


class TestBuildFromRefMap:
    def test_waypoints_are_neo4j_nodes_in_neo4j_coordinates(self):
        g = em.build_from_ref_map(_neo_map())
        assert set(g.graph.nodes) == {1, 2, 3, 4}
        assert g.position(2) == (0.0, 8.0)
        assert all(g.waypoint_for_neo(n) == n for n in (1, 2, 3, 4))   # no nearest-point mapping

    def test_walkways_are_walkable_both_ways_with_metre_lengths(self):
        g = em.build_from_ref_map(_neo_map())
        assert g.graph.has_edge(2, 1) and g.graph.has_edge(3, 2)
        assert g.graph.edges[1, 2]["length"] == pytest.approx(6.0)
        assert em.plan(g, 3, 1) == [3, 2, 1]

    def test_labels_are_the_nodes_own_objects_and_ocr(self):
        g = em.build_from_ref_map(_neo_map())
        assert g.graph.nodes[3]["products"] == ["優格", "特價"]
        assert em.find_goal_waypoint(g, "牛奶") == 2

    def test_extra_links_and_virtual_entrance(self):
        extra = {"links": [{"from": 1, "to": 4}, {"from": 4, "to": 99}],   # 99: unknown → ignored
                 "entrance": {"x": 0, "y": 0, "links": [1]}}
        g = em.build_from_ref_map(_neo_map(), extra)
        assert g.graph.has_edge(1, 4) and 99 not in g.graph
        assert g.entrance == em.ENTRANCE_ID and g.position(em.ENTRANCE_ID) == (0.0, 0.0)
        assert g.waypoint_for_neo(em.ENTRANCE_ID) is None          # a route start, never a photo fix
        assert em.find_goal_waypoint(g, "牛奶") == 2
        route = em.route_payload(g, em.plan(g, g.entrance, 3))
        assert route["path"] == [em.ENTRANCE_ID, 1, 2, 3]
        assert route["distance_m"] == pytest.approx(2 + 6 + 5)

    def test_without_entrance_the_node_nearest_the_origin_is_the_entrance(self):
        assert em.build_from_ref_map(_neo_map()).entrance == 1


def test_route_graph_source_neo4j_reads_place_links_file(tmp_path, monkeypatch):
    import json
    from server import server as srv
    links = tmp_path / "neo4j_links"
    links.mkdir()
    (links / "other.json").write_text(json.dumps({"place": "elsewhere", "links": [{"from": 1, "to": 3}]}),
                                      encoding="utf-8")
    (links / "test.json").write_text(json.dumps({"place": "test", "links": [{"from": 1, "to": 4}],
                                                 "entrance": {"x": 0, "y": 0, "links": [1]}}),
                                     encoding="utf-8")

    class _FakeNeo4j:
        def load_reference_map(self, place):
            return _neo_map() if place == "test" else RefMap(place=place)

    monkeypatch.setattr(srv, "EDITOR_MAP_DIR", tmp_path)
    monkeypatch.setattr(srv._config, "ROUTE_MAP_SOURCE", "neo4j")
    monkeypatch.setattr(srv, "get_neo4j", lambda: _FakeNeo4j())
    monkeypatch.setattr(srv, "_editor_graphs", {})
    g = srv._editor_graph_for("test")
    assert g.graph.has_edge(1, 4) and not g.graph.has_edge(1, 3)
    assert g.entrance == em.ENTRANCE_ID
    assert srv._editor_graph_for("test") is g                     # cached
    assert srv._editor_graph_for("nowhere") is None               # no Neo4j map → no route
    assert "nowhere" not in srv._editor_graphs                    # …and retried next time
