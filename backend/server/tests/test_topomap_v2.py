"""拓樸地圖 v2（topomap_v2.py）單元測試。"""
import math
import tempfile
from pathlib import Path

import pytest

from server.topomap_v2 import (
    ETYPE_ADJACENT, ETYPE_SAME_OBJECT, ETYPE_WALKWAY,
    GRID_CELLS, ROLE_PRODUCT, ROLE_SIGN, TopoGraphV2,
    classify_role, contains_etype, estimate_distance_m, grid_cell_from_box,
    grid_cell_color, grid_row_col, is_contains_etype,
)


# ── 節點與邊的基本行為 ─────────────────────────────────────────────

def test_add_photo_and_object_node_creates_contains_edge():
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("photo0.jpg")
    o = topo.add_object_node(p, "洗衣精", box=[10, 10, 100, 100])
    assert topo.graph.nodes[p]["ntype"] == "photo"
    assert topo.graph.nodes[o]["ntype"] == "object"
    edge_data = topo.graph.get_edge_data(p, o)
    assert any(is_contains_etype(d["etype"]) for d in edge_data.values())


def test_classify_role_defaults_to_product_and_detects_sign():
    assert classify_role("洗衣精") == ROLE_PRODUCT
    assert classify_role("出口標示") == ROLE_SIGN
    assert classify_role("random shelf item", has_nameplate=True) == ROLE_SIGN


def test_estimate_distance_m_bounds():
    # 佔滿整張照片的物件應該回傳最小距離
    d_near = estimate_distance_m([0, 0, 800, 600], 800, 600)
    # 極小物件應該回傳最大距離（被 clamp）
    d_far = estimate_distance_m([0, 0, 2, 2], 800, 600)
    assert 0.3 <= d_near <= d_far <= 15.0


# ── 子圖建立（第一、二種邊）────────────────────────────────────────

def test_build_subgraph_from_detections_sorts_left_to_right_and_links_adjacent():
    topo = TopoGraphV2()
    p = topo.add_photo_node("photo0.jpg")
    detections = [
        {"label": "B", "box": [300, 0, 400, 100], "score": 0.9},
        {"label": "A", "box": [0, 0, 100, 100], "score": 0.9},
        {"label": "C", "box": [500, 0, 600, 100], "score": 0.9},
    ]
    object_ids = topo.build_subgraph_from_detections(p, detections, img_w=800, img_h=600)
    labels_in_order = [topo.graph.nodes[oid]["label"] for oid in object_ids]
    assert labels_in_order == ["A", "B", "C"]

    # 相鄰邊只連接排序後相鄰的兩兩物件（A-B, B-C），A-C 不應直接相連
    a, b, c = object_ids
    assert topo.graph.has_edge(a, b)
    assert topo.graph.has_edge(b, c)
    assert not topo.graph.has_edge(a, c)
    adjacent_edges = [d for d in topo.graph.get_edge_data(a, b).values() if d["etype"] == ETYPE_ADJACENT]
    assert len(adjacent_edges) == 1
    assert "A" in adjacent_edges[0]["relation"] and "B" in adjacent_edges[0]["relation"]


# ── 跨照片同物件關聯（第三種邊）────────────────────────────────────

def test_link_same_objects_between_photos_matches_by_label_and_ocr():
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")

    dets1 = [{"label": "洗衣精", "box": [0, 0, 100, 100], "nameplate_text": ""}]
    dets2 = [{"label": "洗衣精", "box": [0, 0, 100, 100], "nameplate_text": ""}]
    objs1 = topo.build_subgraph_from_detections(p1, dets1, 800, 600)
    objs2 = topo.build_subgraph_from_detections(p2, dets2, 800, 600)

    linked = topo.link_same_objects_between_photos(objs1, objs2)
    assert len(linked) == 1
    a, b, score = linked[0]
    same_object_edges = [
        d for d in topo.graph.get_edge_data(a, b).values() if d["etype"] == ETYPE_SAME_OBJECT
    ]
    assert len(same_object_edges) == 1


def test_link_same_objects_no_match_when_labels_differ():
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    objs1 = topo.build_subgraph_from_detections(
        p1, [{"label": "洗衣精", "box": [0, 0, 100, 100]}], 800, 600)
    objs2 = topo.build_subgraph_from_detections(
        p2, [{"label": "衛生紙", "box": [0, 0, 100, 100]}], 800, 600)
    linked = topo.link_same_objects_between_photos(objs1, objs2)
    assert linked == []


# ── 走道邊與使用者位置移動（第四種邊）──────────────────────────────

def test_walkway_edge_and_move_to():
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    topo.add_walkway_edge(p1, p2, direction="直走 3 公尺")

    topo.set_start_position(p1)
    topo.move_to(p2)
    assert topo.current_position == p2
    assert topo.position_history == [p1, p2]


def test_move_to_without_walkway_raises():
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")  # 沒有建立走道邊
    topo.set_start_position(p1)
    with pytest.raises(ValueError):
        topo.move_to(p2)


# ── 虛假照片節點（未走過的路）──────────────────────────────────────

def test_virtual_branch_and_resolve():
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    vid = topo.add_virtual_branch(p1, direction="右側走道")
    assert topo.graph.nodes[vid]["is_virtual"] is True
    walkway_edges = [
        d for d in topo.graph.get_edge_data(p1, vid).values() if d["etype"] == ETYPE_WALKWAY
    ]
    assert len(walkway_edges) == 1

    topo.resolve_virtual_branch(vid, "real_photo.jpg")
    assert topo.graph.nodes[vid]["is_virtual"] is False
    assert topo.graph.nodes[vid]["photo_path"] == "real_photo.jpg"


# ── 序列化與場所持久化 ─────────────────────────────────────────────

def test_to_dict_from_dict_roundtrip_preserves_structure():
    topo = TopoGraphV2("測試場所")
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    topo.add_walkway_edge(p1, p2, direction="直走")
    objs = topo.build_subgraph_from_detections(
        p1, [{"label": "A", "box": [0, 0, 10, 10]}, {"label": "B", "box": [20, 0, 30, 10]}], 800, 600)
    topo.set_start_position(p1)
    topo.move_to(p2)

    data = topo.to_dict()
    restored = TopoGraphV2.from_dict(data)

    assert restored.place_name == "測試場所"
    assert restored.graph.number_of_nodes() == topo.graph.number_of_nodes()
    assert restored.graph.number_of_edges() == topo.graph.number_of_edges()
    assert restored.current_position == p2
    assert restored.position_history == [p1, p2]


def test_save_load_for_place_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        topo = TopoGraphV2("萬家福量販 新店店")
        p1 = topo.add_photo_node("photo0.jpg")
        topo.set_start_position(p1)
        topo.save_for_place(root)

        loaded = TopoGraphV2.load_for_place("萬家福量販 新店店", root)
        assert loaded.graph.number_of_nodes() == 1
        assert loaded.current_position == p1

        # 不存在的場所應該回傳一張全新的空地圖，而不是報錯
        empty = TopoGraphV2.load_for_place("從沒建過的地方", root)
        assert empty.graph.number_of_nodes() == 0


# ── 摘要與繪圖（至少確保不會噴例外、格式正確）──────────────────────

def test_summarize_for_vlm_walks_walkway_path():
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    topo.add_walkway_edge(p1, p2, direction="直走 3 公尺")
    topo.build_subgraph_from_detections(p1, [{"label": "入口標示牌", "box": [0, 0, 10, 10]}], 800, 600)
    topo.build_subgraph_from_detections(p2, [{"label": "洗衣精", "box": [0, 0, 10, 10]}], 800, 600)

    summary = topo.summarize_for_vlm(current_photo_id=p2)
    assert "入口標示牌" in summary
    assert "直走 3 公尺" in summary
    assert "洗衣精" in summary


def test_link_same_objects_requires_spatial_evidence_for_common_labels():
    """對應真實資料觀察到的問題：像 trash/desk/plant 這種在很多張照片都
    重複出現的通用標籤，如果畫面位置差很多，不該只憑標籤相同就判定成
    同一個實體（否則地圖會被大量誤連的邊弄得很亂）。
    """
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    # 同樣是 "trash"，但一個在畫面左上、一個在畫面右下 → 位置桶完全不同
    objs1 = topo.build_subgraph_from_detections(
        p1, [{"label": "trash", "box": [0, 0, 50, 50]}], 800, 600)
    objs2 = topo.build_subgraph_from_detections(
        p2, [{"label": "trash", "box": [700, 500, 780, 580]}], 800, 600)
    linked = topo.link_same_objects_between_photos(objs1, objs2)
    assert linked == []  # 只有標籤相同（1.0分）不足以通過新的門檻（1.6）


def test_link_same_objects_still_links_common_label_when_position_matches():
    """但如果位置也吻合（很可能真的是同一個實體被連續兩張照片拍到），
    就應該要連起來，不能矯枉過正。
    """
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    objs1 = topo.build_subgraph_from_detections(
        p1, [{"label": "trash", "box": [0, 0, 50, 50]}], 800, 600)
    objs2 = topo.build_subgraph_from_detections(
        p2, [{"label": "trash", "box": [10, 10, 60, 60]}], 800, 600)
    linked = topo.link_same_objects_between_photos(objs1, objs2)
    assert len(linked) == 1


def test_structural_role_excluded_from_default_linking():
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    objs1 = topo.build_subgraph_from_detections(
        p1, [{"label": "door", "box": [0, 0, 50, 50]}], 800, 600)
    objs2 = topo.build_subgraph_from_detections(
        p2, [{"label": "door", "box": [10, 10, 60, 60]}], 800, 600)
    assert topo.graph.nodes[objs1[0]]["role"] == "環境"
    linked = topo.link_same_objects_between_photos(objs1, objs2)
    assert linked == []  # 預設排除結構物件，即使標籤與位置都吻合
    linked_allowed = topo.link_same_objects_between_photos(objs1, objs2, allow_structural=True)
    assert len(linked_allowed) == 1  # 明確允許時才會連


def test_parse_turn_degrees_recognizes_keywords():
    assert TopoGraphV2._parse_turn_degrees("請直走，然後拍照。") == 0.0
    assert TopoGraphV2._parse_turn_degrees("請左轉，然後拍照。") == -90.0
    assert TopoGraphV2._parse_turn_degrees("請右轉，然後拍照。") == 90.0
    assert TopoGraphV2._parse_turn_degrees("請迴轉，這是死路。") == 180.0
    assert TopoGraphV2._parse_turn_degrees("") == 0.0


def test_parse_turn_degrees_recognizes_turn_around_phrasing():
    """對應真實資料觀察到的問題：VLM 產生的『請轉身，然後沿著走廊返回』
    沒被辨識成迴轉，導致地圖沒有正確折返。"""
    assert TopoGraphV2._parse_turn_degrees("請轉身，然後沿著走廊返回，並拍另一張照片。") == 180.0
    assert TopoGraphV2._parse_turn_degrees("請往回走。") == 180.0
    assert TopoGraphV2._parse_turn_degrees("請掉頭。") == 180.0


def test_parse_distance_m_extracts_number():
    assert TopoGraphV2._parse_distance_m("請直走 3 公尺") == 3.0
    assert TopoGraphV2._parse_distance_m("走 2.5 米") == 2.5
    assert TopoGraphV2._parse_distance_m("請直走，然後拍照") is None


def test_dead_reckoning_origin_is_first_photo():
    """第一張照片（id 最小）永遠是原點 (0,0)，符合『先把第一張照片定為
    原點』的需求。"""
    topo = TopoGraphV2()
    p0 = topo.add_photo_node("p0.jpg")
    p1 = topo.add_photo_node("p1.jpg")
    topo.add_walkway_edge(p0, p1, direction="請直走 3 公尺。")
    pos, _, _ = topo._layout_positions()
    assert pos[p0] == (0.0, 0.0)


def test_double_turn_around_cancels_out_like_real_session_data():
    """對應真實資料：同一段路線連續出現兩次『請轉身』，兩次迴轉應該互相
    抵銷，最後方向跟迴轉前一樣（而不是被忽略導致沒有任何變化——這兩者
    數值上看起來很像，但背後的計算過程不同，這裡直接驗證計算過程有真的
    把每一次迴轉都算進去）。"""
    topo = TopoGraphV2()
    p0 = topo.add_photo_node("p0.jpg")
    p1 = topo.add_photo_node("p1.jpg")
    p2 = topo.add_photo_node("p2.jpg")
    p3 = topo.add_photo_node("p3.jpg")
    topo.add_walkway_edge(p0, p1, direction="請向右轉，然後繼續走。")
    topo.add_walkway_edge(p1, p2, direction="請轉身，然後沿著走廊返回。")
    topo.add_walkway_edge(p2, p3, direction="請轉身，然後沿著走廊返回。")

    _, heading, _ = topo._dead_reckon_tree()
    assert heading[p1] == 90.0
    assert heading[p2] == 270.0   # 90 + 180
    assert heading[p3] == 450.0   # 90 + 180 + 180，跟 90 度朝向相同（差 360）
    assert (heading[p3] - heading[p1]) % 360 == 0.0


def test_dead_reckoning_follows_turns_not_a_straight_line():
    """路線是「直走→右轉→直走」時，畫出來的形狀應該要轉 90 度，
    而不是被硬拉成一條直線。"""
    topo = TopoGraphV2()
    p0 = topo.add_photo_node("p0.jpg")
    p1 = topo.add_photo_node("p1.jpg")
    p2 = topo.add_photo_node("p2.jpg")
    p3 = topo.add_photo_node("p3.jpg")
    topo.add_walkway_edge(p0, p1, direction="請直走 3 公尺。")
    topo.add_walkway_edge(p1, p2, direction="請右轉，然後拍照。")
    topo.add_walkway_edge(p2, p3, direction="請直走 3 公尺。")

    pos, _, _ = topo._layout_positions()
    x0, y0 = pos[p0]
    x1, y1 = pos[p1]
    x2, y2 = pos[p2]
    x3, y3 = pos[p3]

    # P0→P1 段跟 P2→P3 段方向應該互相垂直（右轉 90 度），
    # 而不是三個節點被硬排在同一條水平線上。
    seg_a = (x1 - x0, y1 - y0)
    seg_b = (x3 - x2, y3 - y2)
    dot = seg_a[0] * seg_b[0] + seg_a[1] * seg_b[1]
    assert abs(dot) < 1e-6  # 兩段方向垂直，內積應為 0


def test_virtual_branch_direction_offsets_perpendicular_to_travel():
    """虛假分支節點標示「往右側」時，位置應該偏到主線的側邊，
    而不是疊在主線正上面。"""
    topo = TopoGraphV2()
    p0 = topo.add_photo_node("p0.jpg")
    p1 = topo.add_photo_node("p1.jpg")
    topo.add_walkway_edge(p0, p1, direction="請直走 3 公尺。")
    vid = topo.add_virtual_branch(p1, direction="往右側走道，尚未走訪")

    pos, _, _ = topo._layout_positions()
    x1, y1 = pos[p1]
    xv, yv = pos[vid]
    # 主線方向 (走 p0→p1) 幾乎是沿 y 軸；分支應該明顯偏移到 x 軸方向，
    # 而不是跟主線同一個 x 座標往上/下延伸。
    assert abs(xv - x1) > 0.5


def _boxes_overlap(pos_a, size_a, pos_b, size_b) -> bool:
    ax, ay = pos_a
    aw, ah = size_a
    bx, by = pos_b
    bw, bh = size_b
    return (abs(ax - bx) * 2 < (aw + bw)) and (abs(ay - by) * 2 < (ah + bh))


def test_object_rows_do_not_overlap_when_corridor_is_axis_aligned():
    """對應真實資料觀察到的問題：走道方向剛好是水平或垂直（heading 是
    0/90/180/270 度）時，物件方塊很寬，如果列與列的間距只用固定值，
    寬的方塊會跟隔壁列疊在一起。這裡用夠多物件（會分成多列）加上長
    標籤，確認不管哪一列的方塊彼此都不重疊。"""
    topo = TopoGraphV2()
    p0 = topo.add_photo_node("p0.jpg")
    p1 = topo.add_photo_node("p1.jpg")
    topo.add_walkway_edge(p0, p1, direction="請直走 3 公尺。")  # heading 維持 0 度（正南北向）
    long_labels = [
        "cabinet bookshelf shelf", "desktop computer water dispenser",
        "fire extinguisher", "bulletin board board", "trash", "door",
        "reception", "table desk", "sofa chair", "plant",
    ]
    dets = [{"label": lbl, "box": [i * 10, 0, i * 10 + 40, 40]} for i, lbl in enumerate(long_labels)]
    obj_ids = topo.build_subgraph_from_detections(p1, dets, 800, 600)

    pos, sizes, _ = topo._layout_positions()
    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            a, b = obj_ids[i], obj_ids[j]
            assert not _boxes_overlap(pos[a], sizes[a], pos[b], sizes[b]), (
                f"物件節點 {a} 和 {b} 的方塊重疊了"
            )


def test_photo_label_uses_sequence_number_not_internal_id():
    """對應使用者疑問：P0、P8、P15 這種內部 id 讓人搞不清楚是第幾張照片，
    改成畫圖用的標籤直接顯示「第幾張照片」。"""
    topo = TopoGraphV2()
    p0 = topo.add_photo_node("p0.jpg")
    topo.build_subgraph_from_detections(
        p0, [{"label": f"obj{i}", "box": [i, 0, i + 5, 5]} for i in range(6)], 800, 600)
    p1 = topo.add_photo_node("p1.jpg")  # 這個節點的內部 id 會因為上面插入了 6 個物件節點而跳號
    assert topo._photo_label(p0) == "P1"
    assert topo._photo_label(p1) == "P2"


def test_same_object_edge_only_drawn_once_not_twice():
    """對應使用者截圖：同物件關聯邊資料上雙向各存一條（方便查詢），但畫圖
    只應該畫一條弧線，不能變成兩條分開的弧疊在一起。"""
    topo = TopoGraphV2()
    p0 = topo.add_photo_node("p0.jpg")
    p1 = topo.add_photo_node("p1.jpg")
    topo.add_walkway_edge(p0, p1, direction="請直走 3 公尺。")
    o0 = topo.build_subgraph_from_detections(p0, [{"label": "A", "box": [0, 0, 10, 10]}], 800, 600)[0]
    o1 = topo.build_subgraph_from_detections(p1, [{"label": "A", "box": [0, 0, 10, 10]}], 800, 600)[0]
    topo.add_same_object_edge(o0, o1)

    # 資料上確實雙向各有一條（這是設計上故意的，方便查詢鄰居）
    same_object_data_edges = [
        (u, v) for u, v, d in topo.graph.edges(data=True) if d.get("etype") == ETYPE_SAME_OBJECT
    ]
    assert len(same_object_data_edges) == 2

    # 但畫圖的時候，去重後應該只剩一條無序節點對
    pairs = {frozenset(e) for e in same_object_data_edges}
    assert len(pairs) == 1


def test_no_global_overlap_even_when_path_folds_back():
    """對應真實資料觀察到的問題：路線右轉、直走、然後連續兩次『轉身返回』
    時，折返點在空間上會跟很久以前經過的節點靠在一起。這裡驗證『任兩個
    節點方塊』（不只是樹狀結構上直接相鄰的父子節點）都不會重疊，
    就算物件群很寬也一樣。"""
    topo = TopoGraphV2()
    ids = [topo.add_photo_node(f"p{i}.jpg") for i in range(6)]
    directions = [
        "請繼續沿著走廊直走。",
        "請向右轉，然後繼續走。",
        "請繼續沿著走廊直走。",
        "請轉身，然後沿著走廊返回。",
        "請轉身，然後沿著走廊返回。",
    ]
    for i, direction in enumerate(directions):
        topo.add_walkway_edge(ids[i], ids[i + 1], direction=direction)

    long_labels = [
        "cabinet bookshelf shelf", "desktop computer water dispenser",
        "fire extinguisher", "bulletin board board", "trash", "door", "reception",
    ]
    for pid in ids:
        dets = [{"label": lbl, "box": [i * 10, 0, i * 10 + 40, 40]} for i, lbl in enumerate(long_labels)]
        topo.build_subgraph_from_detections(pid, dets, 800, 600)

    pos, sizes, _ = topo._layout_positions()
    nodes = list(pos.keys())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            assert not _boxes_overlap(pos[a], sizes[a], pos[b], sizes[b]), (
                f"節點 {a} 和 {b} 的方塊重疊了（折返情境下的全域防重疊沒有生效）"
            )


# ── 第一種邊：九宮格子型態 ──────────────────────────────────────────

def test_grid_cell_from_box_covers_all_nine_cells():
    img_w, img_h = 900, 600
    cases = {
        "top_left": [0, 0, 50, 50],
        "top_center": [400, 0, 500, 50],
        "top_right": [850, 0, 900, 50],
        "left": [0, 275, 50, 325],
        "center": [400, 275, 500, 325],
        "right": [850, 275, 900, 325],
        "bottom_left": [0, 550, 50, 600],
        "bottom_center": [400, 550, 500, 600],
        "bottom_right": [850, 550, 900, 600],
    }
    for expected_cell, box in cases.items():
        assert grid_cell_from_box(box, img_w, img_h) == expected_cell
    assert set(cases.keys()) == set(GRID_CELLS)


def test_grid_cell_from_box_defaults_to_center_when_no_box():
    assert grid_cell_from_box([], 800, 600) == "center"
    assert grid_cell_from_box([0, 0, 10, 10], 0, 0) == "center"


def test_contains_etype_and_is_contains_etype_roundtrip():
    for cell in GRID_CELLS:
        etype = contains_etype(cell)
        assert etype == f"contains_{cell}"
        assert is_contains_etype(etype)
    assert contains_etype("not_a_real_cell") == "contains_center"
    assert not is_contains_etype("walkway")
    assert not is_contains_etype("adjacent")


def test_build_subgraph_from_detections_assigns_correct_grid_cell_etype():
    topo = TopoGraphV2()
    p = topo.add_photo_node("photo0.jpg")
    # 800x600 的照片，框在左上角
    obj_ids = topo.build_subgraph_from_detections(
        p, [{"label": "A", "box": [0, 0, 50, 50]}], 800, 600)
    oid = obj_ids[0]
    edge_data = topo.graph.get_edge_data(p, oid)
    etypes = [d["etype"] for d in edge_data.values()]
    assert contains_etype("top_left") in etypes
    assert topo.graph.nodes[oid]["grid_cell"] == "top_left"


# ── 第四種邊：單向有向邊 + 終點拍照方向 ──────────────────────────────

def test_walkway_edge_is_single_directed_not_bidirectional():
    """對應這次的需求：第四種邊改成單向，不再自動雙向各存一條。"""
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    topo.add_walkway_edge(p1, p2, direction="請直走")
    assert topo.graph.has_edge(p1, p2)
    assert not topo.graph.has_edge(p2, p1)
    walkway_edges = [
        (u, v) for u, v, d in topo.graph.edges(data=True) if d.get("etype") == ETYPE_WALKWAY
    ]
    assert walkway_edges == [(p1, p2)]


def test_set_terminal_facing_creates_self_loop_and_clears_previous():
    """路徑終點沒有下一步可以承載拍照方向，改用指向自己的自我迴圈邊；
    每次呼叫都要清掉舊的終點標記，同一時間只能有一個終點。"""
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    topo.add_walkway_edge(p1, p2, direction="請直走")

    topo.set_terminal_facing(p1, direction="")
    self_loops = [(u, v) for u, v, d in topo.graph.edges(data=True)
                  if d.get("etype") == ETYPE_WALKWAY and u == v]
    assert self_loops == [(p1, p1)]

    # 換終點之後，舊的終點標記應該被清掉，只剩新的
    topo.set_terminal_facing(p2, direction="")
    self_loops = [(u, v) for u, v, d in topo.graph.edges(data=True)
                  if d.get("etype") == ETYPE_WALKWAY and u == v]
    assert self_loops == [(p2, p2)]


def test_render_png_handles_terminal_facing_marker_without_crashing():
    topo = TopoGraphV2()
    p1 = topo.add_photo_node("photo0.jpg")
    p2 = topo.add_photo_node("photo1.jpg")
    topo.add_walkway_edge(p1, p2, direction="請直走")
    topo.build_subgraph_from_detections(p2, [{"label": "A", "box": [0, 0, 10, 10]}], 800, 600)
    topo.set_start_position(p1)
    topo.move_to(p2)
    topo.set_terminal_facing(p2)
    png_bytes = topo.render_png()
    assert isinstance(png_bytes, bytes) and len(png_bytes) > 0


def test_render_png_returns_nonempty_bytes_even_for_empty_graph():
    topo = TopoGraphV2()
    png_bytes = topo.render_png()
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0

    p1 = topo.add_photo_node("photo0.jpg")
    topo.add_object_node(p1, "測試物件", box=[0, 0, 10, 10])
    png_bytes2 = topo.render_png()
    assert len(png_bytes2) > 0


# ── 對應「箭頭被方塊蓋住」「物件沒有照九宮格排列」的回歸測試 ──────────

def test_object_layout_follows_grid_cell_column_order():
    """對應使用者回報：物件節點的視覺排列要真的貼近九宮格的左右關係，
    同一列（row 相同）裡，九宮格判定在左邊的物件，畫出來也要在比較
    靠負方向（沿走道側邊）的位置，不能隨便排。

    （第十一輪把「欄」的展開方向從沿走道前進方向改成沿走道側邊
    ——對應使用者要求的「順時針轉 90 度、九宮格上方對齊圖片上方」
    ——所以這裡投影用的軸也從 row_axis 改成 perp_axis。）"""
    topo = TopoGraphV2()
    p = topo.add_photo_node("photo0.jpg")
    # 三個物件，img 900x600，都在「中」那一排（top/bottom 都不是），
    # 但水平位置分別對應左/中/右
    dets = [
        {"label": "right_one", "box": [850, 275, 900, 325]},
        {"label": "left_one", "box": [0, 275, 50, 325]},
        {"label": "center_one", "box": [400, 275, 500, 325]},
    ]
    obj_ids = topo.build_subgraph_from_detections(p, dets, 900, 600)
    pos, sizes, heading = topo._layout_positions()
    by_label = {topo.graph.nodes[oid]["label"]: oid for oid in obj_ids}

    px, py = pos[p]
    h = heading.get(p, 0.0)
    import math
    perp_axis = (math.cos(math.radians(h)), -math.sin(math.radians(h)))

    def proj(oid):
        ox, oy = pos[oid]
        return (ox - px) * perp_axis[0] + (oy - py) * perp_axis[1]

    assert proj(by_label["left_one"]) < proj(by_label["center_one"]) < proj(by_label["right_one"])


def test_object_layout_follows_grid_cell_row_order():
    """同理，row（上/中/下）也要反映在離照片節點的遠近上——但方向要
    跟原照片的上下關係一致，不能顛倒。

    （第十八輪把堆疊方向改成順著拍攝方向（+row_axis）之後，第十九輪
    修正了一個上下顛倒的 bug：堆疊統一往同一個方向延伸時，「下排最
    靠近照片節點、上排疊最遠」才會讓畫出來的上下關係跟原照片一致
    ——上排離照片節點最遠，所以這裡驗證 top_one 的 depth_dist 最大、
    bottom_one 最小。）"""
    topo = TopoGraphV2()
    p = topo.add_photo_node("photo0.jpg")
    dets = [
        {"label": "bottom_one", "box": [400, 550, 500, 600]},
        {"label": "top_one", "box": [400, 0, 500, 50]},
        {"label": "middle_one", "box": [400, 275, 500, 325]},
    ]
    obj_ids = topo.build_subgraph_from_detections(p, dets, 900, 600)
    pos, sizes, heading = topo._layout_positions()
    by_label = {topo.graph.nodes[oid]["label"]: oid for oid in obj_ids}

    import math
    px, py = pos[p]
    h = heading.get(p, 0.0)
    depth_axis = (math.sin(math.radians(h)), math.cos(math.radians(h)))  # +row_axis（拍攝方向）

    def depth_dist(oid):
        ox, oy = pos[oid]
        return (ox - px) * depth_axis[0] + (oy - py) * depth_axis[1]

    assert depth_dist(by_label["bottom_one"]) < depth_dist(by_label["middle_one"]) < depth_dist(by_label["top_one"])


def test_walkway_arrow_shrink_never_consumes_whole_edge():
    """對應使用者回報：箭頭被節點方塊蓋住看不見。驗證縮排量的計算方式
    不會把整條邊吃光（如果縮排量 >= 邊長，箭頭畫出來會完全消失）。"""
    topo = TopoGraphV2()
    ids = [topo.add_photo_node(f"p{i}.jpg") for i in range(4)]
    long_labels = ["cabinet bookshelf shelf", "desktop computer water dispenser"]
    for i, pid in enumerate(ids):
        topo.build_subgraph_from_detections(
            pid, [{"label": lbl, "box": [j * 10, 0, j * 10 + 40, 40]} for j, lbl in enumerate(long_labels)],
            800, 600,
        )
    for i in range(len(ids) - 1):
        topo.add_walkway_edge(ids[i], ids[i + 1], direction="請直走。")

    pos, sizes, _ = topo._layout_positions()
    pt_per_unit = topo._INCH_PER_DATA_UNIT * 72.0

    def shrink_pt(node_id, dx, dy):
        w, h = sizes[node_id]
        return ((w / 2.0) * abs(dx) + (h / 2.0) * abs(dy)) * pt_per_unit

    import math
    for u, v in zip(ids, ids[1:]):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        seg_len_units = math.hypot(x1 - x0, y1 - y0) or 1.0
        dx, dy = (x1 - x0) / seg_len_units, (y1 - y0) / seg_len_units
        shrink_total_pt = shrink_pt(u, dx, dy) + shrink_pt(v, dx, dy)
        seg_len_pt = seg_len_units * pt_per_unit
        assert shrink_total_pt < seg_len_pt, f"邊 {u}->{v} 的箭頭會被兩端縮排完全吃光"


# ────────────────────────────────────────────────────────────────
# 九宮格色卡 / grid_cell_color()
# ────────────────────────────────────────────────────────────────
def test_grid_cell_color_all_nine_cells_are_distinct():
    """九種九宮格代號要對應到九種不同的顏色，顏色卡才有辨識度。"""
    colors = {cell: grid_cell_color(cell) for cell in GRID_CELLS}
    assert len(set(colors.values())) == 9


def _hex_to_luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def test_grid_cell_color_gets_darker_from_top_to_bottom_within_same_column():
    """同一欄（左/中/右）由上到下顏色要依序變深，對應設計說明「深淺分九種」。"""
    for col_name, cells in (
        ("left", ("top_left", "left", "bottom_left")),
        ("center", ("top_center", "center", "bottom_center")),
        ("right", ("top_right", "right", "bottom_right")),
    ):
        luminances = [_hex_to_luminance(grid_cell_color(c)) for c in cells]
        assert luminances[0] > luminances[1] > luminances[2], (
            f"欄「{col_name}」的顏色沒有隨列數（上→中→下）遞減，"
            f"luminance={luminances}"
        )


def test_grid_cell_color_matches_row_col_from_grid_row_col():
    """顏色函式跟排版用的 grid_row_col() 要吃同一套 row/col 定義，
    確認物件節點外框顏色（grid_cell_color）跟畫面位置（grid_row_col）
    對應的是同一格，不會兩邊各自解讀出不同的格子。"""
    for cell in GRID_CELLS:
        row, col = grid_row_col(cell)
        assert 0 <= row <= 2 and 0 <= col <= 2
    # 同一欄（col 相同）的顏色互不相同、同一列（row 相同）的顏色也互不相同，
    # 代表色相（欄）與深淺（列）兩個維度都真的有區分度，不是退化成常數。
    by_col: dict = {}
    for cell in GRID_CELLS:
        row, col = grid_row_col(cell)
        by_col.setdefault(col, set()).add(grid_cell_color(cell))
    for col, colors in by_col.items():
        assert len(colors) == 3, f"col={col} 的三種深淺顏色沒有互不相同：{colors}"


def test_walkway_edges_do_not_cross_unrelated_node_boxes_when_path_folds_back():
    """對應使用者回報：路線折返時，有些走道邊會被剛好卡在路徑中間的
    無關節點方塊擋住。第十四輪把要求改回「整條線（不只箭頭）都不能被
    無關節點蓋住」，這裡檢查完整線段，不只是靠近終點的箭頭區域。"""
    topo = TopoGraphV2()
    ids = [topo.add_photo_node(f"p{i}.jpg") for i in range(6)]
    directions = [
        "請繼續沿著走廊直走。",
        "請向右轉，然後繼續走。",
        "請繼續沿著走廊直走。",
        "請轉身，然後沿著走廊返回。",
        "請轉身，然後沿著走廊返回。",
    ]
    for i, direction in enumerate(directions):
        topo.add_walkway_edge(ids[i], ids[i + 1], direction=direction)

    long_labels = [
        "cabinet bookshelf shelf", "desktop computer water dispenser",
        "fire extinguisher", "bulletin board board", "trash", "door", "reception",
    ]
    for pid in ids:
        dets = [{"label": lbl, "box": [i * 10, 0, i * 10 + 40, 40]} for i, lbl in enumerate(long_labels)]
        topo.build_subgraph_from_detections(pid, dets, 800, 600)

    pos, sizes, _ = topo._layout_positions()

    owner = {}
    for pid in topo.all_photo_nodes():
        owner[pid] = pid
    for oid in topo.all_object_nodes():
        for u, v, d in topo.graph.edges(data=True):
            if is_contains_etype(d.get("etype", "")) and v == oid:
                owner[oid] = u
                break

    walkway_edges = [(u, v) for u, v, d in topo.graph.edges(data=True)
                      if d.get("etype") == ETYPE_WALKWAY and u != v]
    all_nodes = list(pos.keys())

    def seg_hits_box(x0, y0, x1, y1, bx, by, bw, bh, steps=60):
        for i in range(steps + 1):
            t = i / steps
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            if abs(x - bx) <= bw / 2.0 and abs(y - by) <= bh / 2.0:
                return True
        return False

    for u, v in walkway_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        for n in all_nodes:
            if owner.get(n, n) in (owner.get(u, u), owner.get(v, v)):
                continue
            bx, by = pos[n]
            bw, bh = sizes[n]
            assert not seg_hits_box(x0, y0, x1, y1, bx, by, bw, bh), (
                f"走道邊 {u}->{v} 被無關節點 {n} 的方塊擋住了"
            )


def test_object_cluster_width_stays_bounded_as_object_count_grows():
    """對應使用者回報：物件節點排起來像一直線，不像九宮格。真實資料裡
    物件幾乎都落在同一列（中），這裡驗證：即使同一欄物件數量增加，
    物件群整體寬度（沿走道側邊方向，也就是欄展開的方向）也不會跟著
    線性增加——多出來的物件應該是往「遠離照片」的方向堆疊得更深，
    而不是沿走道側邊排得更寬。

    （第十一輪把欄的展開方向從 row_axis 改成 perp_axis，所以這裡的
    寬度改用 perp_axis 投影來量。）"""
    import math

    def cluster_width(n_objects: int) -> float:
        topo = TopoGraphV2()
        p = topo.add_photo_node("photo0.jpg")
        # 全部物件都落在「中」那一列、水平位置隨機分散在左/中/右，
        # 模擬真實資料「垂直方向集中、水平方向分散」的分布
        dets = [
            {"label": f"item{i}", "box": [(i % 5) * 150, 275, (i % 5) * 150 + 60, 325]}
            for i in range(n_objects)
        ]
        topo.build_subgraph_from_detections(p, dets, 900, 600)
        pos, sizes, heading = topo._layout_positions()
        obj_ids = topo.all_object_nodes()
        px, py = pos[p]
        h = heading.get(p, 0.0)
        perp_axis = (math.cos(math.radians(h)), -math.sin(math.radians(h)))

        def proj(oid):
            ox, oy = pos[oid]
            return (ox - px) * perp_axis[0] + (oy - py) * perp_axis[1]

        projections = [proj(oid) for oid in obj_ids]
        return max(projections) - min(projections)

    width_6 = cluster_width(6)
    width_18 = cluster_width(18)
    # 物件數量增加 3 倍，寬度不應該跟著等比例增加（舊排法會線性變寬，
    # 新排法應該只有三欄的寬度上限，頂多因為物件本身文字寬度不同小幅變化）
    assert width_18 < width_6 * 1.8, (
        f"物件群寬度隨物件數量大幅增加（6 個時 {width_6:.2f}，18 個時 {width_18:.2f}），"
        "疑似退化成沿走道方向一路排開的舊行為"
    )


# ── 第十三輪：走道邊直線可被蓋住、箭頭不可被蓋住；轉彎邊要能分辨 ──────

def test_resolve_edge_node_crossings_pushes_nodes_blocking_middle_of_line_too():
    """對應使用者第十四輪的要求：走道邊的直線部分也不能被無關節點蓋住
    （不是只保護箭頭端）。驗證擋在直線正中間的無關節點一樣會被推開，
    不會被放過。"""
    topo = TopoGraphV2()

    # 節點 100 卡在 1→2 連線的正中間：應該被推開
    pos_mid = {1: (0.0, 0.0), 2: (0.0, 10.0), 100: (0.0, 5.0)}
    sizes = {1: (1.0, 1.0), 2: (1.0, 1.0), 100: (1.0, 1.0)}
    cluster_members = {1: [1], 2: [2], 100: [100]}
    moved_mid = topo._resolve_edge_node_crossings(pos_mid, sizes, cluster_members, [(1, 2)])
    assert moved_mid is True
    assert pos_mid[100] != (0.0, 5.0)

    # 節點 100 卡在靠近終點：一樣應該被推開
    pos_tip = {1: (0.0, 0.0), 2: (0.0, 10.0), 100: (0.0, 9.5)}
    moved_tip = topo._resolve_edge_node_crossings(pos_tip, sizes, cluster_members, [(1, 2)])
    assert moved_tip is True
    assert pos_tip[100] != (0.0, 9.5)


def test_render_png_succeeds_with_turn_edges_and_terminal_marker():
    """對應使用者要求：走道邊如果這一步有轉彎，要能跟直走的邊區分開來
    （改用不同顏色＋轉彎標籤）。這裡至少確保混合了直走／右轉／終點朝向
    標記的場景畫得出來、不會噴例外，圖例的轉彎邊判斷邏輯（has_turn_edge）
    也確實依資料正確觸發。"""
    topo = TopoGraphV2()
    ids = [topo.add_photo_node(f"p{i}.jpg") for i in range(3)]
    topo.add_walkway_edge(ids[0], ids[1], direction="請直走。")
    topo.add_walkway_edge(ids[1], ids[2], direction="請右轉，然後繼續走。")
    topo.set_terminal_facing(ids[2])
    for pid in ids:
        topo.build_subgraph_from_detections(pid, [{"label": "door", "box": [0, 0, 40, 40]}], 800, 600)

    has_turn_edge = any(
        u != v and d.get("etype") == ETYPE_WALKWAY and TopoGraphV2._parse_turn_degrees(d.get("direction", "") or "")
        for u, v, d in topo.graph.edges(data=True)
    )
    assert has_turn_edge is True

    png_bytes = topo.render_png()
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_walkway_arrow_connects_to_photo_node_box_not_whole_cluster():
    """對應使用者回報：第四種邊（走道邊）沒有跟照片節點相連，而是跟
    整個群（照片＋它掛的物件節點）相連——這偏離了走道邊的定義：走道邊
    代表照片節點跟照片節點之間的走道，箭頭理應連到照片節點本身。

    上一輪把縮排量改成看「整個群」的範圍，雖然解決了箭頭穿過物件群的
    視覺問題，卻讓箭頭常常在離照片節點還很遠的地方就停住，等於連到
    物件群的邊界而不是照片節點。這裡驗證縮排量只看照片節點自己的
    方塊，不會因為掛了很多／很大的物件節點就被拉遠——不管掛多少物件，
    縮排量都應該只等於照片節點自己方塊的投影半寬（加上固定的緩衝）。"""
    topo = TopoGraphV2()
    a = topo.add_photo_node("a.jpg")
    b = topo.add_photo_node("b.jpg")
    topo.add_walkway_edge(a, b, direction="請直走 5 公尺。")
    # b 掛一堆又大又深的物件（同一欄堆很深），確保「整個群」的範圍
    # 遠比 b 自己的照片方塊大上很多
    dets = [{"label": f"very long cabinet bookshelf item {i}", "box": [400, 275, 500, 325]}
            for i in range(6)]
    topo.build_subgraph_from_detections(b, dets, 900, 600)

    pos, sizes, heading = topo._layout_positions()
    x0, y0 = pos[a]
    x1, y1 = pos[b]
    seg_len = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / seg_len, (y1 - y0) / seg_len

    pt_per_data_unit = topo._INCH_PER_DATA_UNIT * 72.0

    def box_shrink_pt(node_id, ddx, ddy):
        w, h = sizes[node_id]
        return ((w / 2.0) * abs(ddx) + (h / 2.0) * abs(ddy)) * pt_per_data_unit + 2.0

    expected_shrink_b = box_shrink_pt(b, -dx, -dy)

    # 更直接的驗證：物件群整體的範圍（用物件節點本身的座標算）要遠比
    # b 自己方塊的縮排量大，證明這個情境真的會讓「整個群」版本的縮排
    # 跟「只看照片節點」版本的縮排產生巨大落差，確保這個測試情境有
    # 意義（不是隨便測不會出錯的東西）。
    object_ids = topo.all_object_nodes()
    px, py = pos[b]
    max_object_reach = max(
        abs((pos[oid][0] - px) * (-dx) + (pos[oid][1] - py) * (-dy)) for oid in object_ids
    )
    assert max_object_reach * pt_per_data_unit > expected_shrink_b * 3, (
        "這個測試情境裡物件群的範圍不夠大，沒辦法有效區分「只縮排照片"
        "節點」跟「縮排整個群」這兩種行為的差異"
    )

    # 直接檢查 render_png 的原始碼裡沒有「整個群」的縮排邏輯——確保
    # 沒有不小心又改回去用 cluster 範圍縮排（那樣會讓箭頭連不到照片
    # 節點本身，見上面的說明）。
    import inspect
    src = inspect.getsource(TopoGraphV2.render_png)
    assert "_cluster_shrink_pt" not in src, "render_png 又用回「整個群」的縮排邏輯了，箭頭會連不到照片節點本身"


def test_walkway_arrows_stay_visible_across_real_layout():
    """對應使用者回報「照片跟照片之間的邊沒有成功相連」。用一個接近
    真實情境的多節點、多物件場景（含折返）驗證：每條走道邊只縮排照片
    節點自己的方塊之後，剩下的可見長度都要是正的，不能被縮排吃光。"""
    topo = TopoGraphV2()
    ids = [topo.add_photo_node(f"p{i}.jpg") for i in range(6)]
    directions = [
        "請繼續沿著走廊直走 5 公尺。",
        "請向右轉，然後繼續走 3 公尺。",
        "請繼續沿著走廊直走 4 公尺。",
        "請轉身，然後沿著走廊返回 3 公尺，並拍另一張照片。",
        "請轉身，然後沿著走廊返回 3 公尺，並拍另一張照片。",
    ]
    for i, direction in enumerate(directions):
        topo.add_walkway_edge(ids[i], ids[i + 1], direction=direction)

    long_labels = [
        "cabinet bookshelf shelf unit", "desktop computer water dispenser stand",
        "fire extinguisher box", "bulletin board notice board", "trash bin", "door frame",
    ]
    for pid in ids:
        dets = [{"label": lbl, "box": [j * 60, 200 + j * 5, j * 60 + 120, 260 + j * 5]}
                for j, lbl in enumerate(long_labels)]
        topo.build_subgraph_from_detections(pid, dets, 1200, 800)

    pos, sizes, heading = topo._layout_positions()
    pt_per_data_unit = topo._INCH_PER_DATA_UNIT * 72.0

    def box_shrink_pt(node_id, ddx, ddy):
        w, h = sizes[node_id]
        return ((w / 2.0) * abs(ddx) + (h / 2.0) * abs(ddy)) * pt_per_data_unit + 2.0

    walkway_edges = [(u, v) for u, v, d in topo.graph.edges(data=True)
                      if d.get("etype") == ETYPE_WALKWAY and u != v]
    for u, v in walkway_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        seg_pt = seg_len * pt_per_data_unit
        dx, dy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
        shrink_a = box_shrink_pt(u, dx, dy)
        shrink_b = box_shrink_pt(v, dx, dy)
        visible = seg_pt - shrink_a - shrink_b
        assert visible > 0, (
            f"走道邊 {u}->{v} 的可見長度變成 {visible:.1f}pt（<=0），"
            "邊會被縮排吃光、看起來像沒有連起來"
        )


def test_walkway_arrow_zorder_is_below_node_boxes_so_nodes_never_get_covered():
    """對應使用者要求：不要讓節點被箭頭遮擋。驗證 render_png 裡走道邊
    （含轉彎起點的拍攝方向小箭頭、路徑終點的拍照方向標記）用的 zorder
    確實比節點方塊低，這樣不管箭頭的路徑上有沒有經過節點，節點方塊都
    會畫在箭頭上面、內容不會被蓋住。"""
    import inspect
    render_src = inspect.getsource(TopoGraphV2.render_png)
    assert "_WALKWAY_ZORDER = 2.5" in render_src
    # 節點文字（render_png 裡直接呼叫 ax.text）用的 zorder 要比走道邊高，
    # 箭頭才會被壓在節點下面。
    assert "zorder=4" in render_src
    # 節點方塊本身的 zorder 定義在 _draw_node_box 裡。
    box_src = inspect.getsource(TopoGraphV2._draw_node_box)
    assert "zorder=3" in box_src


# ── 第二十輪：角度復原、縮短照片節點間距 ──────────────────────────

def test_turn_angle_is_restored_even_when_anti_overlap_would_distort_it():
    """對應使用者回報：P10 看起來不是 90 度右轉。確認資料本身是對的
    （direction 文字寫「請向右轉」，_parse_turn_degrees 應該解析成 90
    度），且死算推位算出來的角度也是精準的 90 度；驗證即使路線後面接著
    折返（觸發全域防重疊），這個沒有真正衝突的父子節點角度，最後畫出來
    還是要維持原本死算推位算出來的角度，不能被防重疊的推擠帶歪。"""
    topo = TopoGraphV2()
    ids = [topo.add_photo_node(f"p{i}.jpg") for i in range(7)]
    directions = [
        "請沿著走廊直走，然後拍另一張照片。",
        "請沿著走廊直走，然後拍另一張照片。",
        "請向右轉，然後沿著走廊繼續走，然後拍另一張照片。",
        "請繼續沿著走廊直走，然後拍另一張照片。",
        "請轉身，然後沿著走廊返回，並拍另一張照片。",
        "請轉身，然後沿著走廊返回，並拍另一張照片。",
    ]
    for i, direction in enumerate(directions):
        topo.add_walkway_edge(ids[i], ids[i + 1], direction=direction)

    long_labels = [
        "cabinet bookshelf shelf unit", "desktop computer water dispenser stand",
        "fire extinguisher box", "bulletin board notice board", "trash bin", "door frame",
    ]
    for pid in ids:
        dets = [{"label": lbl, "box": [j * 60, 200 + j * 5, j * 60 + 120, 260 + j * 5]}
                for j, lbl in enumerate(long_labels)]
        topo.build_subgraph_from_detections(pid, dets, 1200, 800)

    raw_xy, heading, parent_of = topo._dead_reckon_tree()
    pos, sizes, heading2 = topo._layout_positions()

    # 找出「請向右轉」那一步對應的父子節點（ids[2] -> ids[3]）
    p, n = ids[2], ids[3]

    def angle(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        return math.degrees(math.atan2(dx, dy))

    raw_angle = angle(raw_xy[p], raw_xy[n])
    final_angle = angle(pos[p], pos[n])
    assert abs(raw_angle - 90.0) < 1e-6, f"死算推位本身角度就不是 90 度：{raw_angle}"
    assert abs(final_angle - raw_angle) < 1e-6, (
        f"最終畫出來的角度（{final_angle:.2f}°）跟死算推位算出來的角度"
        f"（{raw_angle:.2f}°）不一致，疑似被全域防重疊的推擠帶歪了"
    )


def test_photo_node_spacing_is_tighter_with_reduced_min_gap_multiplier():
    """對應使用者要求：在不造成節點被遮擋的情況下縮短照片節點之間的
    間距。驗證新的間距（min_gap 倍率 1.0）確實比舊的（2.2）短，同時
    仍然維持沒有任何節點重疊。"""
    topo = TopoGraphV2()
    ids = [topo.add_photo_node(f"p{i}.jpg") for i in range(4)]
    for i in range(3):
        topo.add_walkway_edge(ids[i], ids[i + 1], direction="請繼續沿著走廊直走。")
    # 掛很多物件、讓物件群的範圍夠大，確保 min_dist（受 min_gap 倍率
    # 影響）才是限制間距的關鍵因素，不是死算推位本身的距離（沒寫明確
    # 距離時，死算推位的預設步長很短，物件群一多很容易被 min_dist 蓋過）
    dets = [{"label": f"very long cabinet bookshelf item {i}", "box": [400, 275, 500, 325]}
            for i in range(6)]
    for pid in ids:
        topo.build_subgraph_from_detections(pid, dets, 900, 600)

    pos, sizes, heading = topo._layout_positions()

    import itertools
    def overlap(a, b):
        ax, ay = pos[a]; aw, ah = sizes[a]
        bx, by = pos[b]; bw, bh = sizes[b]
        return abs(ax - bx) < (aw + bw) / 2.0 - 1e-6 and abs(ay - by) < (ah + bh) / 2.0 - 1e-6

    nodes = list(pos.keys())
    assert not any(overlap(a, b) for a, b in itertools.combinations(nodes, 2))

    # 跟用舊倍率（2.2）算出來的間距比較，確認真的變短了
    pos_loose, _, _ = topo._layout_positions(min_gap=0.45 * 2.2)
    dist_tight = math.hypot(pos[ids[1]][0] - pos[ids[0]][0], pos[ids[1]][1] - pos[ids[0]][1])
    dist_loose = math.hypot(pos_loose[ids[1]][0] - pos_loose[ids[0]][0],
                             pos_loose[ids[1]][1] - pos_loose[ids[0]][1])
    assert dist_tight < dist_loose


# ── 找不到路徑的節點（leftover）排版：擺到版面外，不能疊到主圖 ──────

def test_leftover_node_without_walkway_is_placed_outside_main_map_bounds():
    """沒有走道邊連到主要分量的照片節點（例如外部呼叫端接了一個獨立
    節點進來，locate_v2.py 的除錯圖就是這樣用），排版時應該被擺到整張
    地圖版面外、留足夠安全距離的空白區域，不能疊到主路徑或任何物件群
    上——早期版本只是貼著起點挪一點點小偏移，起點旁邊本來就有自己的
    物件群展開，小偏移量完全不夠，會直接疊到主路徑的節點/物件上。
    """
    topo = TopoGraphV2("測試場所")
    ids = [topo.add_photo_node(f"p{i}.jpg") for i in range(3)]
    for i in range(2):
        topo.add_walkway_edge(ids[i], ids[i + 1], direction="請繼續直走。")
    for pid in ids:
        topo.build_subgraph_from_detections(
            pid, [{"label": "desk", "box": [100, 100, 300, 300]}], 900, 600,
        )

    # 完全沒有走道邊連接的獨立節點（模擬 locate_v2.py 除錯圖的查詢節點）
    leftover = topo.add_photo_node("lonely.jpg")
    topo.build_subgraph_from_detections(
        leftover, [{"label": "sofa", "box": [100, 100, 300, 300]}], 900, 600,
    )

    pos, sizes, _heading = topo._layout_positions()

    # 算出「除了 leftover 以外」所有節點的邊界框
    other_ids = [n for n in pos if n != leftover]
    xs_min = min(pos[n][0] - sizes[n][0] / 2 for n in other_ids)
    xs_max = max(pos[n][0] + sizes[n][0] / 2 for n in other_ids)
    ys_min = min(pos[n][1] - sizes[n][1] / 2 for n in other_ids)
    ys_max = max(pos[n][1] + sizes[n][1] / 2 for n in other_ids)

    lx, ly = pos[leftover]
    lw, lh = sizes[leftover]
    leftover_box = (lx - lw / 2, lx + lw / 2, ly - lh / 2, ly + lh / 2)

    # leftover 的方塊跟「其餘所有節點的邊界框」不能有任何重疊
    no_overlap = (
        leftover_box[1] < xs_min or leftover_box[0] > xs_max
        or leftover_box[3] < ys_min or leftover_box[2] > ys_max
    )
    assert no_overlap, (
        f"leftover 節點的方塊 {leftover_box} 跟主圖邊界框 "
        f"x=({xs_min},{xs_max}) y=({ys_min},{ys_max}) 有重疊"
    )


# ── OCR 關聯到最近物件（build_subgraph_from_detections 的 ocr_items）──

def test_ocr_items_are_associated_to_nearest_object_by_position():
    """`ocr_items` 這個參數過去接了但完全沒被使用——物件要有 OCR 文字，
    必須是物件字典本身就帶著 "nameplate_text"，光傳一份獨立的 OCR
    清單進來不會自動關聯。這個測試驗證關聯邏輯真的有生效：一筆 OCR
    結果的座標離哪個物件最近，就該被寫進那個物件的 ocr_text。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    detections = [
        {"label": "cabinet", "box": [0, 0, 100, 100], "score": 0.9},
        {"label": "sign", "box": [500, 500, 600, 600], "score": 0.8},
    ]
    ocr_items = [
        {"text": "乳製品區", "bbox": [[510, 510], [590, 510], [590, 590], [510, 590]]},
    ]
    object_ids = topo.build_subgraph_from_detections(p, detections, 900, 600, ocr_items=ocr_items)

    objs = {topo.graph.nodes[oid]["label"]: topo.graph.nodes[oid] for oid in object_ids}
    assert objs["sign"]["ocr_text"] == "乳製品區"
    assert objs["cabinet"]["ocr_text"] == ""


def test_structural_object_with_associated_ocr_is_classified_as_sign_not_structure():
    """一扇門本身會被 `classify_role()` 判定成環境結構物，但如果透過
    `ocr_items` 關聯到了 OCR 文字（例如貼著房間名稱的門牌），角色應該
    變成「標示牌」，不是「環境結構物」——`classify_role()` 判斷角色時，
    有沒有 OCR 文字的優先順序比標籤本身的關鍵字更高。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    detections = [
        {"label": "door", "box": [0, 0, 100, 100], "score": 0.9},
    ]
    ocr_items = [
        {"text": "會議室", "bbox": [[10, 10], [90, 10], [90, 90], [10, 90]]},
    ]
    object_ids = topo.build_subgraph_from_detections(p, detections, 900, 600, ocr_items=ocr_items)

    door_obj = topo.graph.nodes[object_ids[0]]
    assert door_obj["ocr_text"] == "會議室"
    assert door_obj["role"] == ROLE_SIGN


def test_ocr_association_does_not_overwrite_existing_nameplate_text():
    """物件字典自己本來就帶著 "nameplate_text"（例如某些 VLM 輸出格式
    本來就會直接把文字寫在物件自己身上）時，不該被 `ocr_items` 的關聯
    結果覆蓋掉。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    detections = [
        {"label": "sign", "box": [0, 0, 100, 100], "score": 0.9, "nameplate_text": "原本的文字"},
    ]
    ocr_items = [
        {"text": "不該蓋過去的文字", "bbox": [[10, 10], [90, 10], [90, 90], [10, 90]]},
    ]
    object_ids = topo.build_subgraph_from_detections(p, detections, 900, 600, ocr_items=ocr_items)
    assert topo.graph.nodes[object_ids[0]]["ocr_text"] == "原本的文字"


def test_ocr_association_ties_prefer_sign_like_labels():
    """VLM 模式常見：好幾個不同物件共用完全相同的座標（粗略模板框）。
    這種情況下 OCR 文字距離「打平手」，不該隨便選排最前面那個，而是
    優先關聯給標籤本身比較像招牌／告示牌的物件。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    same_box = [100, 100, 200, 200]
    detections = [
        {"label": "door", "box": same_box, "score": 0.9},
        {"label": "sign", "box": same_box, "score": 0.8},
    ]
    ocr_items = [
        {"text": "會議室", "bbox": [[100, 100], [200, 100], [200, 200], [100, 200]]},
    ]
    object_ids = topo.build_subgraph_from_detections(p, detections, 900, 600, ocr_items=ocr_items)
    objs = {topo.graph.nodes[oid]["label"]: topo.graph.nodes[oid] for oid in object_ids}
    assert objs["sign"]["ocr_text"] == "會議室"
    assert objs["door"]["ocr_text"] == ""


def test_ocr_association_skips_three_way_ties_instead_of_guessing():
    """回歸測試：對應一次真實踩到的問題。VLM 模式底下，一整面牆的
    裝飾性布條/海報文字（例如比賽得獎感謝布條），常常跟好幾個不相干
    的物件（水桶、櫃子、咖啡機）剛好共用同一個粗略模板框——3 個以上
    物件打平手時，這段文字通常是背景裝飾，不是專門描述某一個特定
    物件的門牌，硬猜一個（例如「優先選標籤裡有關鍵字的」規則在這種
    情況下沒有任何一個候選是招牌類標籤，只能退化成「選排最前面的」）
    容易猜錯，反而讓後續定位比對出現「trophies↔bucket」這種標籤對不上
    卻因為 OCR 文字剛好對上而配對成功的異常結果。3 個以上物件打平手時
    應該直接跳過，不猜。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    same_box = [768, 480, 1176, 1120]
    detections = [
        {"label": "door", "box": [24, 480, 432, 1120], "score": 0.9},
        {"label": "bucket", "box": same_box, "score": 0.8},
        {"label": "cabinet", "box": same_box, "score": 0.9},
        {"label": "coffee machine", "box": same_box, "score": 0.8},
    ]
    ocr_items = [
        {"text": "InnoServe", "bbox": [[768, 480], [1176, 480], [1176, 1120], [768, 1120]]},
    ]
    object_ids = topo.build_subgraph_from_detections(p, detections, 900, 600, ocr_items=ocr_items)
    objs = {topo.graph.nodes[oid]["label"]: topo.graph.nodes[oid] for oid in object_ids}
    assert objs["bucket"]["ocr_text"] == ""
    assert objs["cabinet"]["ocr_text"] == ""
    assert objs["coffee machine"]["ocr_text"] == ""
    assert objs["door"]["ocr_text"] == ""


# ── 查無所屬的 OCR 文字：存到照片節點的 ambient_ocr_texts，不整個丟棄 ──

def test_unassigned_ocr_text_is_stored_on_photo_node_not_discarded():
    """3 個以上物件打平手、判斷「查無所屬」的文字，不該憑空消失——
    應該被存到這張照片節點本身的 `ambient_ocr_texts` 屬性，供之後可能
    的「照片對照片，比對有沒有讀到同一句背景文字」這類用途使用（目前
    比對邏輯還沒有用到這個欄位，這裡只驗證資料有沒有正確保留）。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    same_box = [768, 480, 1176, 1120]
    detections = [
        {"label": "door", "box": [24, 480, 432, 1120], "score": 0.9},
        {"label": "bucket", "box": same_box, "score": 0.8},
        {"label": "cabinet", "box": same_box, "score": 0.9},
        {"label": "coffee machine", "box": same_box, "score": 0.8},
    ]
    ocr_items = [
        {"text": "InnoServe", "bbox": [[768, 480], [1176, 480], [1176, 1120], [768, 1120]]},
    ]
    topo.build_subgraph_from_detections(p, detections, 900, 600, ocr_items=ocr_items)
    assert topo.graph.nodes[p]["ambient_ocr_texts"] == ["InnoServe"]


def test_assigned_ocr_text_is_not_also_stored_as_ambient():
    """正常關聯成功的文字（例如 2 個物件打平手、成功選到 sign），只會
    出現在那個物件的 ocr_text 裡，不該重複出現在 ambient_ocr_texts。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    same_box = [100, 100, 200, 200]
    detections = [
        {"label": "door", "box": same_box, "score": 0.9},
        {"label": "sign", "box": same_box, "score": 0.8},
    ]
    ocr_items = [
        {"text": "會議室", "bbox": [[100, 100], [200, 100], [200, 200], [100, 200]]},
    ]
    topo.build_subgraph_from_detections(p, detections, 900, 600, ocr_items=ocr_items)
    assert topo.graph.nodes[p]["ambient_ocr_texts"] == []


def test_photo_node_without_ambient_ocr_texts_defaults_to_empty_list():
    """沒有傳 `ocr_items`（例如既有、還沒重新建圖的舊地圖資料），照片
    節點也該有這個欄位、預設是空清單，不會讓讀取端因為 KeyError 掛掉。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    assert topo.graph.nodes[p]["ambient_ocr_texts"] == []


def test_ambient_ocr_texts_survive_save_and_load():
    """存檔、讀檔一輪，ambient_ocr_texts 這個新欄位要能正確保留下來
    （驗證序列化沒有遺漏這個欄位，`to_dict()`/`from_dict()` 是通用地
    存/讀節點所有屬性，理論上不用特別處理，這裡直接測一次確認）。
    """
    topo = TopoGraphV2("測試場所")
    p = topo.add_photo_node("p.jpg")
    same_box = [768, 480, 1176, 1120]
    detections = [
        {"label": "door", "box": [24, 480, 432, 1120], "score": 0.9},
        {"label": "bucket", "box": same_box, "score": 0.8},
        {"label": "cabinet", "box": same_box, "score": 0.9},
        {"label": "coffee machine", "box": same_box, "score": 0.8},
    ]
    ocr_items = [
        {"text": "InnoServe", "bbox": [[768, 480], [1176, 480], [1176, 1120], [768, 1120]]},
    ]
    topo.build_subgraph_from_detections(p, detections, 900, 600, ocr_items=ocr_items)

    restored = TopoGraphV2.from_dict(topo.to_dict())
    assert restored.graph.nodes[p]["ambient_ocr_texts"] == ["InnoServe"]

