"""照片定位（locate_v2.py）單元測試。"""
import json
from pathlib import Path

import pytest

from server.locate_v2 import (
    LocalizationCandidate,
    _compute_label_rarity_weights,
    format_localization_report,
    localize_photo,
    render_localization_debug_png,
)
from server.topomap_v2 import TopoGraphV2

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ── 小型合成地圖（方便控制變因）────────────────────────────────────

def _build_toy_topo() -> TopoGraphV2:
    """三個照片節點的小地圖：
    - P (id=photo_a) 有 冰箱 / 標示牌「乳製品區」/ 手推車
    - photo_b 有 冰箱 / 標示牌「生鮮區」/ 貨架（跟 photo_a 標籤不同、位置也不同）
    - photo_c 完全不相關（電腦、椅子、書架）
    """
    topo = TopoGraphV2("玩具測試場所")
    photo_a = topo.add_photo_node("a.jpg")
    photo_b = topo.add_photo_node("b.jpg")
    photo_c = topo.add_photo_node("c.jpg")

    dets_a = [
        {"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9},
        {"label": "sign", "box": [350, 50, 500, 150], "score": 0.8, "nameplate_text": "乳製品區"},
        {"label": "cart", "box": [600, 400, 750, 600], "score": 0.7},
    ]
    dets_b = [
        {"label": "refrigerator", "box": [700, 200, 950, 500], "score": 0.9},
        {"label": "sign", "box": [50, 50, 200, 150], "score": 0.8, "nameplate_text": "生鮮區"},
        {"label": "shelf", "box": [300, 400, 450, 600], "score": 0.7},
    ]
    dets_c = [
        {"label": "computer", "box": [50, 50, 200, 200], "score": 0.9},
        {"label": "chair", "box": [300, 300, 450, 500], "score": 0.9},
        {"label": "bookshelf", "box": [600, 100, 750, 400], "score": 0.9},
    ]

    topo.build_subgraph_from_detections(photo_a, dets_a, img_w=800, img_h=600)
    topo.build_subgraph_from_detections(photo_b, dets_b, img_w=800, img_h=600)
    topo.build_subgraph_from_detections(photo_c, dets_c, img_w=800, img_h=600)
    topo.add_walkway_edge(photo_a, photo_b, direction="請直走")
    topo.add_walkway_edge(photo_b, photo_c, direction="請直走")
    topo.set_terminal_facing(photo_c)
    return topo, photo_a, photo_b, photo_c


# ── 基本比對行為 ─────────────────────────────────────────────────

def test_localize_exact_match_ranks_top():
    topo, photo_a, photo_b, photo_c = _build_toy_topo()
    query_dets = [
        {"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9},
        {"label": "sign", "box": [350, 50, 500, 150], "score": 0.8, "nameplate_text": "乳製品區"},
        {"label": "cart", "box": [600, 400, 750, 600], "score": 0.7},
    ]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    assert result.best is not None
    assert result.best.photo_id == photo_a
    assert result.best.score > result.candidates[1].score  # 明顯領先第二名
    assert result.candidates[1].photo_id == photo_b  # 同樣有冰箱，第二相似


def test_localize_low_confidence_when_query_unrelated():
    topo, photo_a, photo_b, photo_c = _build_toy_topo()
    query_dets = [
        {"label": "swimming pool", "box": [0, 0, 100, 100], "score": 0.9},
        {"label": "palm tree", "box": [200, 200, 300, 300], "score": 0.9},
    ]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    assert result.warning is not None
    assert result.best is None or result.best.score < 0.15


def test_localize_empty_detections_returns_warning_and_no_candidates():
    topo, *_ = _build_toy_topo()
    result = localize_photo(topo, [], img_w=800, img_h=600)
    assert result.candidates == []
    assert result.warning is not None
    assert result.query_object_count == 0


def test_localize_excludes_virtual_branch_nodes():
    topo, photo_a, photo_b, photo_c = _build_toy_topo()
    vid = topo.add_virtual_branch(photo_b, direction="往右側走道", note="沒走過")
    query_dets = [{"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9}]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    assert all(c.photo_id != vid for c in result.candidates)


def test_localize_top_k_limits_results():
    topo, *_ = _build_toy_topo()
    query_dets = [{"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9}]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600, top_k=1)
    assert len(result.candidates) == 1


def test_localize_does_not_mutate_original_topo():
    topo, *_ = _build_toy_topo()
    n_nodes_before = topo.graph.number_of_nodes()
    n_edges_before = topo.graph.number_of_edges()
    query_dets = [{"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9}]
    localize_photo(topo, query_dets, img_w=800, img_h=600)
    assert topo.graph.number_of_nodes() == n_nodes_before
    assert topo.graph.number_of_edges() == n_edges_before


def test_localize_structural_objects_excluded_by_default():
    """glass door 這種環境／結構物件，預設不該被拿來當定位依據
    （到處都有玻璃門，用它配對容易誤判——跟 link_same_objects_between_photos
    的預設行為一致）。"""
    topo = TopoGraphV2("測試場所2")
    photo_a = topo.add_photo_node("a.jpg")
    photo_b = topo.add_photo_node("b.jpg")
    topo.build_subgraph_from_detections(
        photo_a, [{"label": "glass door", "box": [0, 0, 100, 100], "score": 0.9}], 800, 600,
    )
    topo.build_subgraph_from_detections(
        photo_b, [{"label": "computer", "box": [0, 0, 100, 100], "score": 0.9}], 800, 600,
    )
    query_dets = [{"label": "glass door", "box": [0, 0, 100, 100], "score": 0.9}]

    result_default = localize_photo(topo, query_dets, img_w=800, img_h=600)
    assert result_default.best is None or result_default.best.matched_count == 0

    result_allow = localize_photo(topo, query_dets, img_w=800, img_h=600, allow_structural=True)
    assert result_allow.best is not None
    assert result_allow.best.photo_id == photo_a
    assert result_allow.best.matched_count == 1


def test_localize_candidate_carries_photo_metadata():
    topo, photo_a, *_ = _build_toy_topo()
    # 只給「冰箱」一項不夠鑑別力（photo_a / photo_b 都有冰箱，拿掉水平
    # 位置比對之後，兩者垂直位置又剛好一樣，單靠這個物件本來就分不出
    # 是哪一間——這是拿掉不可靠的水平比對後合理的真實限制，不是 bug）。
    # 這裡只是要測 metadata 欄位有沒有正確帶出來，所以額外加上帶 OCR
    # 招牌文字的 sign，確保有足夠證據明確指向 photo_a。
    query_dets = [
        {"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9},
        {"label": "sign", "box": [350, 50, 500, 150], "score": 0.8, "nameplate_text": "乳製品區"},
    ]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    best = result.best
    assert best is not None
    assert best.photo_id == photo_a
    assert best.photo_file == "a.jpg"
    assert best.photo_label.startswith("P")
    assert best.query_object_count == 2


def test_localize_position_mismatch_does_not_block_matching():
    """核心回歸測試（小規模、獨立資料版本）：查詢物件的標籤跟某節點
    完全一樣，但畫面位置刻意設在完全不同的地方（模擬用不同角度重新
    拍的照片）。修正前的邏輯會因為「位置不相近」讓配對數變成 0；
    修正後畫面位置只是加分項，標籤相同就應該配對成功。
    """
    topo, photo_a, photo_b, photo_c = _build_toy_topo()
    # 標籤跟 photo_a 完全一樣，但每個物件的畫面位置都換到對角
    query_dets = [
        {"label": "refrigerator", "box": [600, 50, 750, 150], "score": 0.9},   # 原本在左邊，這次在右上
        {"label": "sign", "box": [50, 400, 200, 550], "score": 0.8, "nameplate_text": "乳製品區"},  # 原本在右邊，這次在左下
        {"label": "cart", "box": [50, 50, 150, 150], "score": 0.7},             # 原本在右下，這次在左上
    ]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    assert result.best is not None
    assert result.best.photo_id == photo_a
    assert result.best.matched_count > 0, "畫面位置改變後配對數變成 0，回歸到舊有的 bug 了"


def test_localize_horizontal_mirror_does_not_hurt_score():
    """回歸測試：對應這次真實提出的問題——同一個地點，第二次拍攝時
    站位往旁邊移一點、或改成從對面拍，畫面上物件的水平位置（甚至左右
    順序）會整個不一樣，但這不該影響配對信心。這裡直接把 x 座標左右
    鏡像（x → img_w - x），模擬「從正對面拍」這種水平完全顛倒的極端
    情況，垂直座標（y）維持不變；分數應該要跟完全沒鏡像時一樣高，
    因為現在的比對邏輯根本不看水平位置。
    """
    topo, photo_a, *_ = _build_toy_topo()
    img_w = 800

    def mirror_x(box):
        x1, y1, x2, y2 = box
        return [img_w - x2, y1, img_w - x1, y2]

    original_dets = [
        {"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9},
        {"label": "sign", "box": [350, 50, 500, 150], "score": 0.8, "nameplate_text": "乳製品區"},
        {"label": "cart", "box": [600, 400, 750, 600], "score": 0.7},
    ]
    mirrored_dets = [
        {**d, "box": mirror_x(d["box"])} for d in original_dets
    ]

    result_original = localize_photo(topo, original_dets, img_w=img_w, img_h=600)
    result_mirrored = localize_photo(topo, mirrored_dets, img_w=img_w, img_h=600)

    assert result_original.best.photo_id == photo_a
    assert result_mirrored.best.photo_id == photo_a
    assert result_mirrored.best.matched_count == result_original.best.matched_count
    assert result_mirrored.best.score == result_original.best.score, (
        "水平鏡像（模擬從對面拍攝）不該影響信心分數——如果分數不一樣，"
        "代表比對邏輯還是有偷偷用到水平位置。"
    )


def test_extract_vertical_component_ignores_horizontal_and_distance():
    from server.locate_v2 import _extract_vertical_component

    # 兩種現存格式都要能正確抽出垂直分量，且水平/距離資訊完全不影響結果
    assert _extract_vertical_component("left,middle") == "middle"
    assert _extract_vertical_component("right,middle") == "middle"
    assert _extract_vertical_component("left middle, near") == "middle"
    assert _extract_vertical_component("right middle, far") == "middle"
    assert _extract_vertical_component("center,top") == "top"
    assert _extract_vertical_component("") == ""
    assert _extract_vertical_component("garbage") == ""


def test_localize_extra_interstitial_object_does_not_break_matching():
    """回歸測試：地圖裡 A 在上、B 在下（例如冰箱在上三分之一、標示牌在
    下三分之一），新照片多值測到一個原本沒被辨識出來的物件 C，夾在
    A、B 中間。只要 A、B 兩個物件本身的偵測框沒有變，各自的上/中/下
    分類就不該受 C 有沒有被偵測到影響——position 是每個物件各自獨立
    算的，不是「A 相對於 B」這種兩兩之間的關係，所以 C 的出現不該讓
    A、B 被判定成不同地點。
    """
    topo = TopoGraphV2("測試場所3")
    mapped_node = topo.add_photo_node("mapped.jpg")
    topo.build_subgraph_from_detections(
        mapped_node,
        [
            {"label": "refrigerator", "box": [300, 20, 500, 150], "score": 0.9},   # 上三分之一
            {"label": "sign", "box": [300, 450, 500, 580], "score": 0.9, "nameplate_text": "乳製品區"},  # 下三分之一
        ],
        800, 600,
    )

    # 查詢照片：A、B 座標完全不變，只是多偵測到一個原本沒找到的 C，
    # 夾在兩者中間（中三分之一）。
    query_dets = [
        {"label": "refrigerator", "box": [300, 20, 500, 150], "score": 0.9},
        {"label": "poster", "box": [300, 260, 500, 340], "score": 0.6},  # 新偵測到的 C，夾在中間
        {"label": "sign", "box": [300, 450, 500, 580], "score": 0.9, "nameplate_text": "乳製品區"},
    ]

    result = localize_photo(topo, query_dets, img_w=800, img_h=600, top_k=1)
    assert result.best is not None
    assert result.best.photo_id == mapped_node
    assert result.best.matched_count == 2, (
        "A、B 兩個物件都應該正常配對成功，不該因為多了一個 C 就配對失敗"
    )


def test_localize_vertical_bucket_off_by_one_gets_partial_credit():
    """回歸測試：構圖範圍不同時，物件實際位置沒變，但因為畫面涵蓋的
    垂直範圍不同，量到的三等分格子可能差一格（例如原本落在最上三分之
    一，構圖稍微拉遠、往下帶一點之後變成中間三分之一）。這種「差一格」
    的情況該給部分加分，不該直接當作沒對上（跟「差兩格」，也就是上下
    兩極端，要有差別待遇）。
    """
    from server.locate_v2 import _score_pair

    label_weights = {"refrigerator": 0.7}

    exact = {"label_norm": "refrigerator", "ocr_text": "", "position": "left,top"}
    off_by_one = {"label_norm": "refrigerator", "ocr_text": "", "position": "left,middle"}
    off_by_two = {"label_norm": "refrigerator", "ocr_text": "", "position": "left,bottom"}

    score_exact, _ = _score_pair(exact, exact, label_weights)
    score_adjacent, _ = _score_pair(exact, off_by_one, label_weights)
    score_opposite, _ = _score_pair(exact, off_by_two, label_weights)

    assert score_exact > score_adjacent > score_opposite, (
        "完全相同格 > 差一格（部分加分） > 差兩格（不加分），三者的分數"
        "應該要有這個順序，不能是二選一。"
    )
    assert score_opposite == label_weights["refrigerator"], (
        "差兩格（top vs bottom）不該有任何位置加分，只剩標籤配對的分數"
    )


# ── 圖結構驗證（垂直順序一致性）───────────────────────────────────

def test_structure_consistency_penalizes_contradictory_arrangement():
    """核心情境：物件種類、OCR 都對得上，但空間排列彼此矛盾（查詢照片
    「滅火器在上、垃圾桶在下」，候選節點卻是反過來），信心分數應該
    明顯低於「排列也一致」的情況——單看物件種類、覆蓋率兩者會一樣高，
    只有加了結構驗證才分得出來。
    """
    def build_topo(order_b_above_a: bool):
        topo = TopoGraphV2("結構驗證測試")
        node = topo.add_photo_node("n.jpg")
        if order_b_above_a:
            dets = [
                {"label": "trash can", "box": [300, 20, 500, 150], "score": 0.9},   # 垃圾桶在上
                {"label": "fire extinguisher", "box": [300, 450, 500, 580], "score": 0.9},  # 滅火器在下
            ]
        else:
            dets = [
                {"label": "fire extinguisher", "box": [300, 20, 500, 150], "score": 0.9},  # 滅火器在上
                {"label": "trash can", "box": [300, 450, 500, 580], "score": 0.9},          # 垃圾桶在下
            ]
        topo.build_subgraph_from_detections(node, dets, 800, 600)
        return topo, node

    # 查詢照片：滅火器在上、垃圾桶在下
    query_dets = [
        {"label": "fire extinguisher", "box": [300, 20, 500, 150], "score": 0.9},
        {"label": "trash can", "box": [300, 450, 500, 580], "score": 0.9},
    ]

    topo_consistent, node_consistent = build_topo(order_b_above_a=False)  # 節點也是滅火器在上
    topo_contradictory, node_contradictory = build_topo(order_b_above_a=True)  # 節點是垃圾桶在上（反過來）

    result_consistent = localize_photo(topo_consistent, query_dets, img_w=800, img_h=600)
    result_contradictory = localize_photo(topo_contradictory, query_dets, img_w=800, img_h=600)

    c1 = result_consistent.best
    c2 = result_contradictory.best
    assert c1.matched_count == c2.matched_count == 2, "兩邊物件種類都應該配對成功，數量要一樣"
    assert c1.structure_consistency > c2.structure_consistency, (
        "排列一致的候選節點，結構一致性應該明顯高於排列矛盾的候選節點"
    )
    assert c2.structure_consistency == 0.25, (
        "兩個物件、只有一組矛盾的組合，套用平滑公式 (0+0.5)/(1+1) 應該是 0.25，"
        "不是直接歸零（樣本數只有 1 組時，平滑會避免過度武斷）"
    )
    assert c1.score > c2.score, (
        "排列一致的候選節點，最終信心分數應該明顯高於排列矛盾的候選節點"
        "（即使物件種類、配對數都一樣）"
    )


def test_structure_consistency_insufficient_evidence_gets_penalized_not_neutral():
    """只配對到 1 個物件時，湊不出任何『可比較的組合』——依照設計，這種
    證據不足的情況要扣分（乘數 0.5），不是維持中立的 1.0。
    """
    from server.locate_v2 import _CONSISTENCY_PRIOR_CONSISTENT, _CONSISTENCY_PRIOR_TOTAL

    topo, photo_a, *_ = _build_toy_topo()
    query_dets = [{"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9}]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    best = result.best
    assert best.matched_count == 1
    expected = _CONSISTENCY_PRIOR_CONSISTENT / _CONSISTENCY_PRIOR_TOTAL
    assert best.structure_consistency == expected
    assert best.structure_consistency < 1.0, "證據不足不該給滿分 1.0"
    assert best.structure_consistency > 0.0, "證據不足也不該直接歸零（沒有直接證據說它是錯的）"


def test_structure_consistency_ignores_tied_vertical_buckets():
    """兩個物件在任一邊的照片裡剛好同一格（分不出上下），這組組合要被
    忽略，不能被當成「矛盾」扣分，也不能被當成「一致」加分。
    """
    topo = TopoGraphV2("同格測試")
    node = topo.add_photo_node("n.jpg")
    # 節點裡兩個物件都在同一個垂直格（都在中間）
    topo.build_subgraph_from_detections(
        node,
        [
            {"label": "fire extinguisher", "box": [100, 250, 200, 350], "score": 0.9},
            {"label": "trash can", "box": [500, 250, 600, 350], "score": 0.9},
        ],
        800, 600,
    )
    # 查詢照片：這兩個物件分得出明確的上下順序
    query_dets = [
        {"label": "fire extinguisher", "box": [300, 20, 500, 150], "score": 0.9},
        {"label": "trash can", "box": [300, 450, 500, 580], "score": 0.9},
    ]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    best = result.best
    assert best.matched_count == 2
    # 節點端兩個物件同格 → 這組組合被忽略，等同於沒有可比較的組合。
    # 但配對到的物件數（2）不算太少，這是「物件剛好沒有垂直差異可比」
    # 而不是「證據太少」，應該套用溫和懲罰 _CONSISTENCY_NO_SPREAD_PENALTY，
    # 不是配對數 <2 才會用到的較重懲罰。
    from server.locate_v2 import _CONSISTENCY_NO_SPREAD_PENALTY
    assert best.structure_consistency == _CONSISTENCY_NO_SPREAD_PENALTY


def test_format_localization_report_contains_expected_fields():
    topo, *_ = _build_toy_topo()
    query_dets = [{"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9}]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    report = format_localization_report(result, topo)
    assert topo.place_name in report
    assert "信心分數" in report
    assert "P" in report


def test_format_localization_report_has_no_emoji_in_warning():
    """警告文字要用純文字（「警告：」），不要用表情符號。"""
    topo, *_ = _build_toy_topo()
    unrelated_dets = [{"label": "swimming pool", "box": [0, 0, 100, 100], "score": 0.9}]
    result = localize_photo(topo, unrelated_dets, img_w=800, img_h=600)
    assert result.warning is not None
    report = format_localization_report(result, topo)
    assert "警告：" in report
    assert "⚠" not in report


def test_format_localization_report_handles_no_candidates():
    topo, *_ = _build_toy_topo()
    result = localize_photo(topo, [], img_w=800, img_h=600)
    report = format_localization_report(result, topo)
    assert "沒有任何候選節點" in report


# ── 使用真實建圖資料的整合測試 ──────────────────────────────────────
# 這份 fixture 是從真實導航資料建出的完整地圖（17 個照片節點、92 個節點、
# 218 條邊），用來驗證定位演算法在真實規模、真實標籤分布（含大量像
# "trash"/"desk"/"plant" 這種到處都是的通用標籤）下依然能正確運作。

@pytest.fixture(scope="module")
def real_topo() -> TopoGraphV2:
    return TopoGraphV2.load(FIXTURES_DIR / "real_place_topomap.json")


def _dets_from_existing_node(topo: TopoGraphV2, photo_id: int) -> list[dict]:
    """把地圖裡某個照片節點現有的物件，轉成一份「查詢照片偵測結果」，
    模擬「使用者站在同一個位置又拍了一張類似的照片」。

    直接把節點已經算好的 "position" 字串原樣帶入 det 字典的 "position"
    欄位——`build_subgraph_from_detections()` 看到 detections 裡已經有
    "position" 就會直接採用、不會用 box+img 重新計算（見
    `position_bucket()` 呼叫處的 `det.get("position") or ...`），這樣就
    不需要知道原始照片的真實寬高也能重建出跟原節點一致的畫面位置線索。
    """
    dets = []
    for o in topo.photo_objects(photo_id):
        dets.append({
            "label": o["label"],
            "box": o.get("box") or [0, 0, 10, 10],
            "score": o.get("score", 0.9),
            "position": o.get("position", ""),
            "nameplate_text": o.get("ocr_text", ""),
        })
    return dets


def test_localize_real_map_self_recognition(real_topo: TopoGraphV2):
    """拿真實地圖裡某個節點自己的物件當「查詢照片」，定位結果最高分
    應該就是它自己（在 17 個候選節點裡排第一）。"""
    photo_nodes = real_topo.all_photo_nodes()
    target = sorted(photo_nodes)[2]  # 第三張照片節點（id 由小到大）

    query_dets = _dets_from_existing_node(real_topo, target)
    assert len(query_dets) > 0, "測試資料的這個節點應該要有物件，請確認 fixture"

    result = localize_photo(real_topo, query_dets, img_w=1600, img_h=1200, top_k=17)
    assert result.best is not None
    assert result.best.photo_id == target
    assert result.best.score > 0.3


def test_localize_real_map_with_different_screen_positions_still_works(real_topo: TopoGraphV2):
    """回歸測試：對應到一次真實回報的 bug。

    這個測試刻意**不**沿用原節點的 "position" 欄位，而是把每個物件的
    畫面位置改成跟原本不一樣（模擬使用者用不同角度／距離重新拍了一張
    照片，同樣的東西這次出現在畫面的別的地方）。舊版比對邏輯要求標籤
    相同「而且」畫面位置也要相近才算數，這種情況下配對數會變成 0
    （這正是真實使用時回報的問題）。新版邏輯把畫面位置改成加分項、
    用標籤稀有度取代位置當主要判準，這裡驗證修正後即使畫面位置完全
    對不上，只要標籤跟建圖時一致，仍然能正確定位、配對數不再是 0。
    """
    photo_nodes = sorted(real_topo.all_photo_nodes())
    target = photo_nodes[0]  # 有 bulletin board/table/trophies/plants/lockers/sofa 等有鑑別力的物件
    dets = _dets_from_existing_node(real_topo, target)
    assert len(dets) > 0

    # 把每個物件的 "position" 全部改成同一個、跟原本大機率不一樣的值，
    # 模擬「這次拍照角度完全不同，所有東西出現的畫面位置都變了」。
    shuffled_dets = []
    for i, det in enumerate(dets):
        d = dict(det)
        d["position"] = "right bottom, far" if det.get("position") != "right bottom, far" else "left top, near"
        shuffled_dets.append(d)

    result = localize_photo(real_topo, shuffled_dets, img_w=1600, img_h=1200, top_k=17)
    assert result.best is not None
    assert result.best.photo_id == target
    assert result.best.matched_count > 0, (
        "畫面位置全部改變後配對數變成 0，代表比對邏輯又退化成依賴位置"
        "門檻了（這是之前真實踩到的 bug）。"
    )


def test_localize_real_map_partial_query_still_ranks_source_node_highest(real_topo: TopoGraphV2):
    """只拿某節點「比較有鑑別力」的一部分物件當查詢照片（模擬拍照角度
    不同、只拍到部分物件），排名最高的仍然應該是原本那個節點。

    刻意挑該節點裡標籤稀有度較高（比較不通用）的一半物件，而不是照
    原始順序盲目切一半——如果切出來的子集剛好全部都是「桌子」「垃圾
    桶」這種到處都有、好幾個節點共用同一組標籤的通用物件，那組合本來
    就無法唯一定位（例如這份真實地圖裡節點 0 跟節點 8 剛好都有
    「bulletin board」+「table」），這是資訊量不足的天花板，不是這支
    程式的 bug，測試不該對那種情況硬性要求排第一。
    """
    photo_nodes = sorted(real_topo.all_photo_nodes())
    target = photo_nodes[0]
    full_dets = _dets_from_existing_node(real_topo, target)
    if len(full_dets) < 2:
        pytest.skip("這個節點物件數太少，不適合做部分子集測試")

    label_weights = _compute_label_rarity_weights(real_topo, photo_nodes)
    full_dets.sort(
        key=lambda d: label_weights.get(d.get("label", "").lower(), 0.0), reverse=True,
    )
    half_dets = full_dets[: max(len(full_dets) // 2, 1)]

    result = localize_photo(real_topo, half_dets, img_w=1600, img_h=1200, top_k=17)
    assert result.best is not None
    assert result.best.photo_id == target


def test_localize_real_map_all_candidates_are_non_virtual_real_nodes(real_topo: TopoGraphV2):
    photo_nodes = sorted(real_topo.all_photo_nodes())
    target = photo_nodes[0]
    query_dets = _dets_from_existing_node(real_topo, target)
    result = localize_photo(real_topo, query_dets, img_w=1600, img_h=1200, top_k=100)
    for c in result.candidates:
        assert not real_topo.graph.nodes[c.photo_id].get("is_virtual")


def test_localize_real_map_does_not_mutate_fixture(real_topo: TopoGraphV2):
    n_nodes_before = real_topo.graph.number_of_nodes()
    n_edges_before = real_topo.graph.number_of_edges()
    photo_nodes = sorted(real_topo.all_photo_nodes())
    query_dets = _dets_from_existing_node(real_topo, photo_nodes[0])
    localize_photo(real_topo, query_dets, img_w=1600, img_h=1200)
    assert real_topo.graph.number_of_nodes() == n_nodes_before
    assert real_topo.graph.number_of_edges() == n_edges_before


# ── 分數公式（v3：雙向覆蓋率調和平均）回歸測試 ────────────────────────
# 對應一次真實回報的狀況：查詢照片偵測到很多物件（構圖較廣，看到不少
# 「原本那張建圖照片根本沒拍到、但不代表配錯」的東西），配對到的雖然
# 是幾個很有鑑別力的物件（信心都接近滿分），舊公式（除以查詢物件數）
# 卻把分數壓得很低，容易讓人誤以為配對失敗。

def test_localize_partial_but_distinctive_match_scores_reasonably_high(real_topo: TopoGraphV2):
    """只從查詢照片裡給 3 個高辨識度的物件（模擬只有這幾個對得上，其餘
    物件都是這張新照片多看到的東西），分數不應該因為「查詢物件數比對到
    的多很多」被壓到警告門檻以下——只要這幾個配對本身夠強，就該有
    合理高的信心分數。"""
    target = 83  # 有 desk/chair/sofa/trophies 等高辨識度物件的真實節點
    target_objects = {o["label"]: o for o in real_topo.photo_objects(target)}
    assert {"desk", "chair", "sofa"} <= set(target_objects.keys())

    query_dets = [
        {"label": "desk", "box": [10, 10, 50, 50], "score": 0.9,
         "position": target_objects["desk"].get("position", "")},
        {"label": "chair", "box": [900, 900, 950, 950], "score": 0.9,
         "position": target_objects["chair"].get("position", "")},
        {"label": "sofa", "box": [500, 20, 600, 100], "score": 0.9,
         "position": target_objects["sofa"].get("position", "")},
        # 加幾個這張新照片多看到、但原本那張建圖照片沒拍到的東西——
        # 不該因為這些「額外看到的東西」拖累分數。
        {"label": "backpack", "box": [1, 1, 20, 20], "score": 0.6},
        {"label": "umbrella", "box": [2, 2, 30, 30], "score": 0.6},
    ]

    result = localize_photo(real_topo, query_dets, img_w=1600, img_h=1200, top_k=17)
    assert result.best is not None
    assert result.best.photo_id == target
    # desk/chair/sofa 這三個物件在真實地圖裡剛好都在同一個垂直格（辦公室
    # 家具高度差不多很合理），湊不出任何「可比較的上下順序組合」——但
    # 配對到 3 個物件不算少，這是「物件沒有垂直差異可比」而不是「證據
    # 不足」，套用溫和懲罰 _CONSISTENCY_NO_SPREAD_PENALTY（不是配對數
    # < 2 才用到的較重懲罰），分數應該維持在合理高的水準。
    from server.locate_v2 import _CONSISTENCY_NO_SPREAD_PENALTY
    assert result.best.structure_consistency == _CONSISTENCY_NO_SPREAD_PENALTY
    assert result.best.score >= 0.35, (
        f"三個高辨識度物件配對成功，分數卻只有 {result.best.score:.0%}，"
        "代表分數公式又退化成被「查詢物件數量」拖累了。"
    )


def test_localize_tiny_node_single_generic_match_does_not_inflate_score():
    """反過來的情況：候選節點本身物件很少（只有一個通用標籤的物件），
    配對到一個很弱的通用標籤時，分數不該虛高（不能因為分母本身很小
    就讓 recall_node 衝到接近 100%）。"""
    topo = TopoGraphV2("小節點測試")
    tiny_node = topo.add_photo_node("tiny.jpg")
    topo.build_subgraph_from_detections(
        tiny_node, [{"label": "trash can", "box": [0, 0, 50, 50], "score": 0.9}], 800, 600,
    )
    rich_node = topo.add_photo_node("rich.jpg")
    topo.build_subgraph_from_detections(
        rich_node,
        [
            {"label": "desk", "box": [0, 0, 100, 100], "score": 0.9},
            {"label": "chair", "box": [200, 0, 300, 100], "score": 0.9},
            {"label": "computer", "box": [400, 0, 500, 100], "score": 0.9},
        ],
        800, 600,
    )

    # 查詢照片有很多物件（模擬真實情境：新照片構圖廣，看到很多東西），
    # 只有一個弱弱的通用標籤剛好跟 tiny_node 一樣。
    query_dets = [
        {"label": "trash can", "box": [700, 500, 750, 550], "score": 0.6},
        {"label": "monitor", "box": [10, 10, 100, 100], "score": 0.7},
        {"label": "printer", "box": [200, 10, 300, 100], "score": 0.7},
        {"label": "cabinet", "box": [400, 10, 500, 100], "score": 0.7},
    ]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600, top_k=5)
    tiny_candidate = next(c for c in result.candidates if c.photo_id == tiny_node)
    assert tiny_candidate.score < 0.5, (
        f"物件很少的節點靠單一通用標籤配對，分數不該虛高（目前 "
        f"{tiny_candidate.score:.0%}）——recall_node 分母太小時應該被 "
        "recall_query（調和平均的另一邊）拉住。"
    )


def test_localize_candidate_reports_coverage_fields():
    topo, photo_a, *_ = _build_toy_topo()
    query_dets = [{"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9}]
    result = localize_photo(topo, query_dets, img_w=800, img_h=600)
    best = result.best
    assert 0.0 <= best.node_coverage <= 1.0
    assert 0.0 <= best.query_coverage <= 1.0


# ── 除錯用視覺化 ─────────────────────────────────────────────────

def test_render_localization_debug_png_returns_valid_png_bytes():
    topo, photo_a, *_ = _build_toy_topo()
    query_dets = [
        {"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9},
        {"label": "sign", "box": [350, 50, 500, 150], "score": 0.8, "nameplate_text": "乳製品區"},
    ]
    png_bytes, result = render_localization_debug_png(topo, query_dets, img_w=800, img_h=600)
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG 檔案的標準開頭 magic bytes
    assert len(png_bytes) > 100
    assert result.best is not None
    assert result.best.photo_id == photo_a


def test_render_localization_debug_png_does_not_mutate_original_topo():
    topo, *_ = _build_toy_topo()
    n_nodes_before = topo.graph.number_of_nodes()
    n_edges_before = topo.graph.number_of_edges()
    query_dets = [{"label": "refrigerator", "box": [50, 200, 300, 500], "score": 0.9}]
    render_localization_debug_png(topo, query_dets, img_w=800, img_h=600)
    assert topo.graph.number_of_nodes() == n_nodes_before
    assert topo.graph.number_of_edges() == n_edges_before


def test_render_localization_debug_png_handles_empty_detections():
    """沒有偵測到任何物件時，還是要能畫出（至少畫出原本的地圖），
    不能整個崩潰。"""
    topo, *_ = _build_toy_topo()
    png_bytes, result = render_localization_debug_png(topo, [], img_w=800, img_h=600)
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert result.candidates == []


def test_render_localization_debug_png_on_real_map(real_topo: TopoGraphV2):
    """在真實規模的地圖（92 節點）上也要能正常畫出來，不只是玩具地圖。"""
    photo_nodes = sorted(real_topo.all_photo_nodes())
    query_dets = _dets_from_existing_node(real_topo, photo_nodes[0])
    png_bytes, result = render_localization_debug_png(
        real_topo, query_dets, img_w=1600, img_h=1200, top_k=5,
    )
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert result.best is not None
