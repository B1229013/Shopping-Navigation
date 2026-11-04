"""server/map_v2.py — Sensor Lab → topomap.json → V2 rows, checked against the A7 map's format."""
import json

import pytest

from server import map_v2 as m


CSV = """\
# SensorLab Export — 2026-10-16
# posture=handheld
timestamp_ms,elapsed_sec,total_steps,step_detected,pdr_x,pdr_y,cal_yaw_deg,waypoint
1,0.0,0,0,0.0,0.0,0.0,
2,1.0,2,2,0.0,1.5,0.0,
3,2.0,4,4,0.0,3.0,0.0,
4,9.0,4,4,0.0,3.0,0.0,"照片1"
5,10.0,6,6,1.5,3.0,90.0,
6,11.0,8,8,3.0,3.0,90.0,
7,19.0,8,8,3.0,3.0,90.0,"照片2"
"""
PHOTOS = [f"wp_1_{s}_100{i}.jpg" for i, s in enumerate(m.SLOTS)] + \
         [f"wp_2_{s}_200{i}.jpg" for i, s in enumerate(m.SLOTS[:3])] + ["notes.txt"]


def test_parse_sensor_log_skips_comments():
    rows = m.parse_sensor_log(CSV)
    assert len(rows) == 7
    assert rows[3] == {"elapsed_sec": 9.0, "total_steps": 4, "pdr_x": 0.0, "pdr_y": 3.0, "waypoint": "照片1"}


def test_extract_waypoints_position_heading_and_photos():
    wps, warnings = m.extract_waypoints(m.parse_sensor_log(CSV), PHOTOS)
    assert [w["wp_num"] for w in wps] == [1, 2]
    w1, w2 = wps
    assert (w1["x"], w1["y"]) == (0.0, 3.0) and w1["heading_deg"] == 0.0      # walked +y
    assert (w2["x"], w2["y"]) == (3.0, 3.0) and w2["heading_deg"] == 90.0     # walked +x
    assert w1["photos"]["front"] == "wp_1_front_1000.jpg"
    assert w2["headings"] == {"front": 90.0, "right": 180.0, "back": 270.0}
    assert any("3/4" in w for w in warnings)


def test_extract_waypoints_without_stamp_is_skipped():
    rows = [r for r in m.parse_sensor_log(CSV) if r["waypoint"] != "照片2"]
    wps, warnings = m.extract_waypoints(rows, PHOTOS)
    assert [w["wp_num"] for w in wps] == [1]
    assert any("照片2" in w for w in warnings)


def test_first_node_heading_looks_forward():
    rows = [{"pdr_x": 0.0, "pdr_y": 0.0, "waypoint": "照片1", "total_steps": 0, "elapsed_sec": 0},
            {"pdr_x": -2.0, "pdr_y": 0.0, "waypoint": "", "total_steps": 3, "elapsed_sec": 1}]
    assert m.walking_heading(rows, 0) == 270.0


def test_build_topomap_matches_builder_format():
    wps, _ = m.extract_waypoints(m.parse_sensor_log(CSV), PHOTOS)
    topo = m.build_topomap("測試店", [("01", wps), ("02", wps[:1])])
    assert [n["id"] for n in topo["nodes"]] == [1, 2, 3]
    n3 = topo["nodes"][2]
    assert (n3["set"], n3["wp_num"], n3["uuid"], n3["x"], n3["y"]) == ("02", 1, "set02_wp1", 0.0, 3.0)
    assert set(n3["photos"]) == set(m.SLOTS) and set(n3["headings"]) == set(m.SLOTS)
    assert topo["edges"] == [{"from": 1, "to": 2, "set": "01"}]
    assert topo["stats"] == {"total_nodes": 3, "sets": {"01": 2, "02": 1}}


def _confirmed():
    four = {s: f"{s}.jpg" for s in m.SLOTS}
    return {
        "place": "測試店",
        "nodes": [
            {"id": 1, "x": 0.004, "y": 3, "sources": [{"set": "01", "wp_num": 1, "uuid": "set01_wp1", "photos": four}]},
            {"id": 2, "x": 3, "y": 3, "sources": [
                {"set": "01", "wp_num": 2, "uuid": "set01_wp2", "photos": four, "headings": {"front": 90.0}},
                {"set": "03", "wp_num": 5, "uuid": "set03_wp5", "photos": {"front": "f3.jpg"}}]},
        ],
        "edges": [{"from": 1, "to": 2, "distance": 3.0}, {"from": 2, "to": 1, "distance": 3.0}],
    }


def test_upload_rows_follow_a7_conventions():
    topo = {"nodes": [{"uuid": "set01_wp1", "set": "01", "wp_num": 1, "headings": {"front": 10.0, "left": 280.0}}]}
    rows = m.upload_rows(_confirmed(), topo, "測試店")
    w1, w2 = rows["waypoints"]
    assert w1 == {"place": "測試店", "nid": 1, "pdr_x": 0.0, "pdr_y": 3.0, "session": "set01_wp1",
                  "heading_deg": 0.0, "total_steps": 0, "total_distance_m": 0.0}
    assert w2["session"] == "set01_wp2+set03_wp5"
    slots = {(p["nid"], p["slot"]): p for p in rows["photos"]}
    assert slots[(1, "front")] == {"nid": 1, "slot": "front", "photo_file": "set01/front.jpg", "heading_deg": 10.0}
    assert slots[(1, "left")]["heading_deg"] == 280.0
    assert slots[(2, "set01_front")]["heading_deg"] == 90.0      # headings carried by the builder win
    assert slots[(2, "set03_front")]["photo_file"] == "set03/f3.jpg"
    assert len(rows["photos"]) == 4 + 4 + 1
    assert rows["walkways"] == [{"from": 1, "to": 2, "distance_m": 3.0}]   # reverse duplicate dropped


def test_upload_rows_loop_closure_slots_stay_unique():
    four = {s: f"{s}.jpg" for s in m.SLOTS}
    confirmed = {"nodes": [{"id": 5, "x": 0, "y": 0, "sources": [
        {"set": "01", "wp_num": 5, "photos": four}, {"set": "01", "wp_num": 26, "photos": four},
        {"set": "02", "wp_num": 3, "photos": {"front": "f.jpg"}}]}], "edges": []}
    rows = m.upload_rows(confirmed, None, "系辦")
    slots = [p["slot"] for p in rows["photos"]]
    assert len(slots) == len(set(slots)) == 9
    assert {"set01wp5_front", "set01wp26_left", "set02_front"} <= set(slots)
    assert rows["waypoints"][0]["session"] == "set01_wp5+set01_wp26+set02_wp3"
    # the server reads the direction after the first "_"
    assert {s.split("_", 1)[1] for s in slots} == set(m.SLOTS)


def test_upload_rows_rejects_dangling_edge():
    bad = _confirmed()
    bad["edges"].append({"from": 1, "to": 9, "distance": 1})
    with pytest.raises(ValueError):
        m.upload_rows(bad, None, "測試店")


def test_photo_local_path():
    assert m.photo_local_path("set01/F05_front.jpg") == ("01", "F05_front.jpg")
    with pytest.raises(ValueError):
        m.photo_local_path("nofolder.jpg")


def test_parse_vlm_items_filters_like_a7():
    reply = "好的：\n" + json.dumps({"scene": "x", "items": [
        {"zh": "走道編號牌", "en": "aisle number sign", "conf": 0.9, "box": [0.1, 0.1, 0.2, 0.2]},
        {"zh": "大貨架", "en": "big shelf", "conf": 0.9, "box": [0.0, 0.0, 0.8, 0.3]},   # too wide
        {"zh": "x", "en": "", "conf": 0.9, "box": [0.1, 0.1, 0.2, 0.2]},                 # label too short
        {"zh": "反框", "en": "flipped", "conf": 0.5, "box": [0.4, 0.4, 0.3, 0.3]},        # gets sorted
        {"zh": "沒框", "en": "no box", "conf": 0.5},
    ]}, ensure_ascii=False)
    items = m.parse_vlm_items(reply)
    assert [i["zh"] for i in items] == ["走道編號牌", "反框"]
    assert items[1]["box"] == [0.3, 0.3, 0.4, 0.4]
    assert m.parse_vlm_items("not json") == []


def test_merge_vlm_calls_drops_near_duplicates():
    a = {"zh": "牛奶", "en": "milk", "conf": 0.9, "box": [0.1, 0.1, 0.2, 0.2]}
    a2 = {"zh": "牛奶", "en": "Milk", "conf": 0.7, "box": [0.12, 0.1, 0.22, 0.2]}
    far = {"zh": "牛奶", "en": "milk", "conf": 0.8, "box": [0.8, 0.8, 0.9, 0.9]}   # other corner: kept
    b = {"zh": "收銀台", "en": "cashier", "conf": 0.8, "box": [0.4, 0.4, 0.5, 0.5]}
    assert m.merge_vlm_calls([[a, b], [a2, far]]) == [a, b, far]


def test_attach_ocr_by_centre():
    items = [{"box": [0.0, 0.0, 0.3, 0.3]}, {"box": [0.5, 0.5, 0.9, 0.9]}]
    ocr = [["冷藏食品", [0.05, 0.05, 0.2, 0.1]], ["P", [0.6, 0.6, 0.7, 0.7]], ["P", [0.7, 0.7, 0.8, 0.8]],
           ["停車場", [0.75, 0.6, 0.85, 0.7]]]
    assert m.attach_ocr(items, ocr) == ["冷藏食品", "P / 停車場"]


def test_object_props_matches_a7_object():
    # An Object from "A7家樂福（9/16)": bbox [0.76, 0.62, 0.89, 0.79] → bottom-right, 2 cells, mid
    o = m.object_props({"zh": "紙箱堆疊", "en": "stacked cardboard boxes", "conf": 0.874,
                        "box": [0.76, 0.62, 0.89, 0.79]})
    assert o == {
        "label": "紙箱堆疊 stacked cardboard boxes", "label_norm": "紙箱堆疊 stacked cardboard boxes",
        "score": 0.87, "ocr_text": "", "bbox": "[0.76, 0.62, 0.89, 0.79]",
        "grid_cell": "bottom-right", "grid_cells": '["mid-right", "bottom-right"]',
        "piece_count": 2, "depth_class": "mid", "source": "rebuild",
    }
    assert m.object_props({"zh": "DM 招牌", "en": "DM sign", "box": [0.1, 0.1, 0.2, 0.2]})["label_norm"] == "dm 招牌 dm sign"


def _obj(cells, depth):
    return {"grid_cells": json.dumps(cells), "depth_class": depth}


def test_adjacent_pairs_match_a7_relationship():
    # The first ADJACENT of "A7家樂福（9/16)" (cardboard boxes → display table), verbatim
    objs = [_obj(["mid-right", "bottom-right"], "mid"),
            _obj(["mid-center", "mid-right", "bottom-center", "bottom-right"], "near"),
            _obj(["top-left"], "far")]
    pairs = m.adjacent_pairs(objs)
    assert [(p["i"], p["j"]) for p in pairs] == [(0, 1), (1, 2)]
    assert pairs[0]["props"] == {
        "cell_pairs": '[{"a": "mid-right", "b": "mid-center", "dir": "left"}, '
                      '{"a": "mid-right", "b": "mid-right", "dir": "overlap"}, '
                      '{"a": "mid-right", "b": "bottom-center", "dir": "below-left"}, '
                      '{"a": "mid-right", "b": "bottom-right", "dir": "below"}, '
                      '{"a": "bottom-right", "b": "mid-center", "dir": "above-left"}, '
                      '{"a": "bottom-right", "b": "mid-right", "dir": "above"}, '
                      '{"a": "bottom-right", "b": "bottom-center", "dir": "left"}, '
                      '{"a": "bottom-right", "b": "bottom-right", "dir": "overlap"}]',
        "horizontal_count": 2, "vertical_count": 2, "diagonal_count": 2,
        "overlap": True, "depth_diff": "closer",
    }
    assert pairs[1]["props"]["depth_diff"] == "farther" and pairs[1]["props"]["overlap"] is False


# ── DINOv2 merge suggestions ──

import numpy as np


def _vec(rng, base=None, noise=0.03):
    v = rng.normal(size=64) if base is None else base + noise * rng.normal(size=64)
    return v / np.linalg.norm(v)


def _walk(set_id, start_id, n):
    return [{"id": start_id + i, "set": set_id, "wp_num": i + 1,
             "photos": {s: f"w{set_id}_{i + 1}_{s}.jpg" for s in m.SLOTS}} for i in range(n)]


def _emb_for(nodes, rng, same_as=None):
    """Random photos per node; same_as {node_id: other_id} → near-copies of the other's photos."""
    emb = {}
    by_id = {n["id"]: n for n in nodes}
    for n in sorted(nodes, key=lambda n: n["id"] in (same_as or {})):
        for s in m.SLOTS:
            src = (same_as or {}).get(n["id"])
            emb[m.photo_key(n, s)] = _vec(rng, emb[m.photo_key(by_id[src], s)]) if src else _vec(rng)
    return emb


def test_suggest_finds_loop_and_cross_walk_matches():
    rng = np.random.default_rng(0)
    a, b = _walk("01", 1, 8), _walk("02", 9, 4)
    # walk 01 comes back to its 2nd waypoint at the end; walk 02 passes 01-4 and 01-6
    emb = _emb_for(a + b, rng, same_as={8: 2, 10: 4, 12: 6})
    s = m.suggest_merges(a + b, emb)
    assert [(i["a"], i["b"]) for i in s["loops"]] == [(2, 8)]
    assert sorted((i["a"], i["b"]) for i in s["matches"]) == [(4, 10), (6, 12)]
    assert s["uncertain"] == []
    assert all(i["score"] > 0.9 and i["runner_up"] < 0.5 for i in s["loops"] + s["matches"])


def test_suggest_cross_walk_match_uses_the_loop_group():
    rng = np.random.default_rng(1)
    a, b = _walk("01", 1, 6), _walk("02", 7, 3)
    # 01-6 closes the loop on 01-1; 02-2 is that same spot → one clear match to the group
    emb = _emb_for(a + b, rng, same_as={6: 1, 8: 1})
    s = m.suggest_merges(a + b, emb)
    assert [(i["a"], i["b"]) for i in s["loops"]] == [(1, 6)]
    assert [(i["a"], i["b"]) for i in s["matches"]] == [(1, 8)]


def test_suggest_ambiguous_match_is_uncertain():
    rng = np.random.default_rng(2)
    a, b = _walk("01", 1, 6), _walk("02", 7, 2)
    # 01-3 and 01-5 look alike (same kind of corridor) and 02-1 looks like both
    emb = _emb_for(a + b, rng, same_as={5: 3, 7: 3})
    s = m.suggest_merges(a + b, emb)
    assert s["matches"] == []
    assert {(i["a"], i["b"]) for i in s["uncertain"]} >= {(3, 7)} or {(i["a"], i["b"]) for i in s["uncertain"]} >= {(5, 7)}


def test_suggest_ignores_neighbours_on_the_same_walk():
    rng = np.random.default_rng(3)
    a = _walk("01", 1, 5)
    emb = _emb_for(a, rng, same_as={2: 1, 3: 2})     # consecutive waypoints look alike
    assert m.suggest_merges(a, emb)["loops"] == []


# ── CSV trajectory ──

def test_walk_track_segments_and_fractions():
    rows = m.parse_sensor_log(CSV)
    wps, _ = m.extract_waypoints(rows, PHOTOS)
    assert [w["row"] for w in wps] == [3, 6]
    tr = m.walk_track(rows, wps, min_step=0.1)
    assert tr["pts"][0] == [0.0, 0.0] and tr["pts"][-1] == [3.0, 3.0]
    assert len(tr["pts"]) == len(tr["seg"]) == len(tr["t"])
    # before the first waypoint → -1; between wp1 and wp2 → 0 with growing fraction; at wp2 → 1
    pairs = list(zip(tr["pts"], tr["seg"], tr["t"]))
    assert all(s == -1 for p, s, t in pairs if p[1] < 3.0)
    mid = [(p, t) for p, s, t in pairs if s == 0]
    assert mid[0] == ([0.0, 3.0], 0.0) and ([1.5, 3.0], 0.5) in mid
    assert pairs[-1][1] == 1


def test_walk_stats():
    assert m.walk_stats(m.parse_sensor_log(CSV)) == {"steps": 8, "distance_m": 6.0, "duration_s": 19.0}


def test_build_topomap_records_walks_and_tracks():
    rows = m.parse_sensor_log(CSV)
    wps, _ = m.extract_waypoints(rows, PHOTOS)
    info = {"03": {"source": "/data/walkA", "stats": m.walk_stats(rows), "track": m.walk_track(rows, wps)}}
    topo = m.build_topomap("店", [("03", wps)], info)
    assert topo["walks"] == [{"set": "03", "source": "/data/walkA", "nodes": 2,
                              "steps": 8, "distance_m": 6.0, "duration_s": 19.0}]
    assert set(topo["tracks"]) == {"03"}


def test_suggest_waypoint_between_two_waypoints_of_another_walk():
    rng = np.random.default_rng(4)
    a, b = _walk("01", 1, 6), _walk("02", 7, 3)
    emb = _emb_for(a + b, rng, same_as={4: 3, 9: 5})      # 01-3/01-4 alike (one corridor); 02-3 = 01-5
    for s in m.SLOTS:                                      # 02-2 stands between 01-3 and 01-4
        mix = emb[m.photo_key(a[2], s)] + emb[m.photo_key(a[3], s)]
        emb[m.photo_key(b[1], s)] = mix / np.linalg.norm(mix)
    out = m.suggest_merges(a + b, emb)
    assert [(i["node"], i["a"], i["b"]) for i in out["between"]] == [(8, 3, 4)]
    assert [(i["a"], i["b"]) for i in out["matches"]] == [(5, 9)]       # unaffected
    assert all(8 not in (i["a"], i["b"]) for i in out["matches"] + out["uncertain"])
