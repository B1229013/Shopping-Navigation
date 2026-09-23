"""
locate_v2.py — 拓樸地圖 v2 的「照片定位」推理
Photo localization against a place-level TopoGraphV2

════════════════════════════════════════════════════════════════════════════
用途 (Purpose)
════════════════════════════════════════════════════════════════════════════
`topomap_v2.py` / `build_topomap_v2.py` 負責「把導航資料建成地圖」；這個
模組反過來做「給一張新照片的偵測結果，推理它最可能對應到地圖裡哪個
（哪幾個）照片節點」，用來測試「目前建出來的拓樸地圖，是否真的能拿來做
定位」這件事。

跟 `build_topomap_v2.py` 一樣，這裡也不呼叫任何 AI 模型、不需要啟動
server ——輸入是「已經算好的偵測結果」（格式跟 `detections_{node_id}.json`
相同），只吃資料、不吃照片本身的像素內容。要跑真正的偵測，由呼叫端
（例如 `locate_photo_v2.py` 這支 CLI）自己決定要不要接系統既有的
GroundingDINO/OCR pipeline。

════════════════════════════════════════════════════════════════════════════
比對邏輯 (Matching logic) — v2：不直接沿用 link_same_objects_between_photos()
════════════════════════════════════════════════════════════════════════════
最早的版本直接重用 `TopoGraphV2.link_same_objects_between_photos()`（建圖
時「跨照片同物件關聯」用的評分函式）。實測後發現這樣行不通：那個函式
自己的文件開頭就寫明是給「通常是相鄰的照片」用的——同一次導航裡連續
拍的兩張照片，拍攝角度、距離、構圖都差不多，所以它用「兩個物件在各自
照片畫面上的九宮格位置是否相近」當作重要佐證（標籤相同才 1.0 分，還
不夠通過門檻 1.6，一定要位置也對上）。

但「定位」的查詢照片是**獨立拍的一張全新照片**，跟建圖時的照片角度、
距離、構圖幾乎不可能一致——同一張桌子，在建圖照片裡可能在畫面左側，
在查詢照片裡可能在畫面正中央。硬用「畫面位置是否相近」當门檻，會讓
幾乎所有本來該配對成功的物件都因為位置對不上而被判定「不是同一個」，
最後配對數變成 0（這是實測踩到的真實 bug，不是假設）。

因此這裡改用另一套權重，畫面位置只當**加分項**、不當門檻，並且用
「標籤在整張地圖裡有多罕見」取代「畫面位置」來抑制通用標籤（垃圾桶、
桌子、盆栽…到處都有）造成的亂配對——這跟建圖時擔心的是同一件事
（通用標籤配對不可靠），只是換一種更適合「非連續照片」的解法：

    單一配對分數 = 標籤相同 × rarity_weight(該標籤在地圖裡的稀有度)
                  + OCR/招牌文字相同 × 1.2   （文字不受拍攝角度影響，是最強訊號）
                  + 垂直位置相近           × 0.15（純加分，不是門檻；只比垂直）

這個「畫面位置」的加分項後來又再修正過一次：一開始是水平＋垂直一起
比（例如整串 "left,middle" 直接比對是否相同），但水平位置（物件在
畫面左/中/右哪一邊）跟拍攝時的左右站位、鏡頭朝向強相關——同一個物件，
攝影者往左站一點、或换個方向面對它拍，畫面上的水平位置就不一樣，
甚至左右順序會整個顛倒（原本朝北拍在畫面右邊，改成朝南從對面拍就會
跑到畫面左邊）。查詢照片是獨立拍的，跟建圖當時的站位、朝向不會一致，
拿水平位置或物件間的左右關係當比對依據，比較的是兩個沒有共同基準的
量，只會產生誤導。垂直位置（物件大概在畫面上/中/下）則相對穩定，只要
兩次拍攝時手機拿的高度、俯仰角不要差太多，物件在畫面上下方向的相對
位置通常還是大致吻合，所以現在**只比垂直位置**，完全不使用水平位置或
左右關係，見 `_extract_vertical_component()` 的說明。

`rarity_weight`：一個標籤出現在地圖裡的照片節點越多（例如「垃圾桶」
到處都是），權重就越接近下限（預設 0.15）；只出現在一、兩個節點的標籤
（例如「咖啡機」、「獎盃」），權重接近上限 1.0。這樣「配對到一個到處
都有的通用物件」跟「配對到一個地圖上獨一無二的物件」對信心分數的貢獻
就不一樣，不需要再靠位置門檻來區分。

做法：把這張新照片當成一個「臨時的照片節點」接進地圖的一份副本裡，用
建圖時同一套 `build_subgraph_from_detections()` 建出它的物件子圖（保證
label 正規化、九宮格、position bucket 等邏輯跟正式建圖時完全一致），
再用上面這套權重，對地圖裡每一個「真實」（非虛假分支）照片節點做貪婪
配對。

════════════════════════════════════════════════════════════════════════════
分數怎麼算 (Score aggregation) — v3：雙向覆蓋率的調和平均，不是「除以較大邊」
════════════════════════════════════════════════════════════════════════════
最早的做法是「配對分數總和 ÷ max(查詢物件數, 候選節點物件數)」，實測
發現這樣會系統性壓低「其實配對得很準」的候選：查詢照片是新拍的，構圖
往往比建圖時那張照片更廣，會多看到好幾個「這個節點的舊照片根本沒拍到、
但也不代表配錯」的東西——拿這些「新照片多看到的東西」的數量去當分母，
等於懲罰了一個本來很準的配對。

改成算「雙向覆蓋率」的調和平均（F1 分數的概念）：

    node_weight_total  = Σ 候選節點裡每個物件的標籤稀有度權重
    query_weight_total = Σ 查詢照片裡每個物件的標籤稀有度權重
    matched_weight     = Σ 配對成功的每一組物件的配對分數（原始值，不壓縮）

    recall_node  = matched_weight / node_weight_total   （這個候選節點的內容，被查詢照片證實了多少）
    recall_query = matched_weight / query_weight_total  （查詢照片看到的東西，被這個候選節點解釋了多少）

    score = 2 × recall_node × recall_query / (recall_node + recall_query)

用調和平均（不是單看 recall_node 或只看 recall_query）是為了同時防兩種
偏差：
  - 只看 recall_node 的話，一個「本身物件很少」的候選節點（例如只有一個
    垃圾桶）配對到一個很弱的通用標籤，recall_node 就可能衝到接近 100%
    （因為分母本身就很小），變成明明證據薄弱卻顯得信心滿滿。
  - 只看 recall_query 的話，物件數量本來就多的候選節點會佔便宜，跟原本
    想避免的問題一樣。
兩個都要顧到，調和平均會被「比較低的那個」拉住，任何一邊證據不夠都
沒辦法灌出虛高分數。

這個函式**不會修改呼叫端傳進來的 TopoGraphV2**（內部對整份地圖做
`copy.deepcopy` 之後才動作），可以放心對正式地圖重複呼叫做測試，不用
擔心把查詢用的臨時節點意外留在正式資料裡。

════════════════════════════════════════════════════════════════════════════
圖結構驗證 (Structural consistency) — 只比對物件種類還不夠
════════════════════════════════════════════════════════════════════════════
只比對「有哪些物件」還有一個漏洞沒防到：物件種類、OCR 都對得上，不
代表這些物件的**空間排列**也對得上。例如查詢照片是「滅火器在上、垃圾
桶在下」，某個候選節點卻是「垃圾桶在上、滅火器在下」——兩邊物件種類
一模一樣，配對數、覆蓋率都會很高，但這其實是兩個不同的場景，只是剛好
物件清單重疊。

`_pairwise_vertical_consistency()` 補這個洞：對所有配對成功的物件兩兩
一組，檢查「查詢照片裡誰在上面」跟「候選節點裡誰在上面」是否一致，
算出一個 0~1 的乘數直接乘進最終分數。刻意用**垂直順序**、不是地圖裡
既有的 `adjacent` 邊（那個邊是照水平座標排的，等於「誰在誰左邊」，
已經證實會隨拍攝站位、朝向改變甚至鏡像顛倒，見前面「比對邏輯」一節）。

判斷規則（用例子說明）：假設配對到 A、B 兩組物件——

| 查詢照片 | 候選節點 | 判定 |
|---|---|---|
| A=上, B=下 | A=上, B=下 | 一致 |
| A=上, B=下 | A=下, B=上 | 矛盾 |
| A=中, B=中 | 不論候選節點是什麼 | 忽略不計（查詢照片裡分不出上下） |
| A=上, B=下 | A=中, B=中 | 忽略不計（候選節點裡分不出上下） |

只有「兩邊都分得出上下順序」的組合才列入統計，任一邊同格（分不出順序）
的組合直接跳過。最終乘數 = (一致組合數 + 0.5) / (可比較組合數 + 1)——
這個 +0.5 / +1 是貝氏平滑：完全沒有可比較組合時（配對到的物件太少、
或全部同格），乘數落在 0.5，比「完全一致」(1.0) 低（證據不足要打折），
比「完全矛盾」(趨近 0) 高（沒有直接證據說它是錯的，只是無法驗證）。

這個乘數會反映在 `LocalizationCandidate.structure_consistency` 欄位，
文字報告裡也會列出來，方便判斷「分數低是因為物件配得少，還是物件配得
多但排列彼此矛盾」。

════════════════════════════════════════════════════════════════════════════
除錯用視覺化 (Debug visualization)
════════════════════════════════════════════════════════════════════════════
`render_localization_debug_png()` 把查詢照片畫成地圖上的一個**獨立**
節點：它跟前幾名候選節點配對到的物件會用「同物件關聯」邊連起來（分數
越高的候選，看起來就跟查詢節點物件連得越密），但**不會**用走道邊把
查詢節點接進地圖的路徑拓樸——查詢照片不是使用者真的走到、拍下的照片，
只要接一條走道邊，不管接在哪個候選節點後面，排版都會把它畫成「路徑
接下去的某一步」，容易誤導成「這是導航會走到的下一個地方」。所以查詢
節點完全不連走道邊，靠物件配對線本身跟地圖產生視覺關聯，讀圖的人一眼
就能看出它是外加上去的比對節點，不是路徑上真正的一站。

直接重用 `TopoGraphV2.render_png()` 本身（沒有改動 render_png 一行
程式碼）——這裡只是在一份用完即丟的地圖副本上，多接幾條「同物件關聯」
邊再呼叫它，畫出來的風格、圖例跟平常看到的拓樸地圖完全一致。查詢節點
因為沒有走道邊連到主要路徑，會被 `_dead_reckon_tree()` 內建的「沒被
走道邊連到主要分量的節點」後備邏輯自動獨立擺在地圖的一角。

════════════════════════════════════════════════════════════════════════════
已知限制
════════════════════════════════════════════════════════════════════════════
- rarity 權重是根據「目前這張地圖」動態算的，同一個標籤在不同地圖（節點
  數、物件分布不同）權重會不一樣，這是刻意的（跟地圖規模無關的絕對
  稀有度沒有意義，只有「相對這張地圖來說罕不罕見」才有意義）。
- 這是「單張照片 vs. 整張地圖」的比對，沒有利用「使用者剛剛在哪個節點」
  這種時序/連續性資訊來縮小搜尋範圍或加權——如果之後想做「已知大概
  在哪一段路，只是要確認精確節點」，可以在呼叫端先用
  `topo.current_position` 或 `position_history` 篩選候選節點清單，
  再傳進來（`localize_photo()` 沒有限制候選節點必須是整張地圖）。
- 沒有用到視覺特徵向量（跟 `link_same_objects_between_photos()` 的限制
  相同），純粹是標籤／OCR／畫面位置的規則式比對——兩個標籤相同但其實
  是不同實體的東西（例如地圖上兩間都有的「白板」），只靠標籤還是分不
  出來，這是規則式方法的天花板，真的要解決需要視覺特徵比對。
"""
from __future__ import annotations

import copy
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from server.config import LABEL_SYNONYMS
from server.topomap_v2 import ROLE_STRUCTURE, TopoGraphV2

_log = logging.getLogger(__name__)

# ── Embedding cache for semantic label matching ──

_embedding_cache: Dict[str, np.ndarray] = {}


def _get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """Call OpenAI-compatible embedding API for a batch of texts."""
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://air.cgu.edu.tw/cgullmapi/v1")
    r = requests.post(
        f"{base_url}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "text-embedding-3-small", "input": texts},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()["data"]
    return [np.array(d["embedding"], dtype=np.float32) for d in sorted(data, key=lambda x: x["index"])]


def ensure_embeddings(labels: List[str]) -> None:
    """Pre-compute and cache embeddings for labels not yet cached."""
    missing = [l for l in labels if l and l not in _embedding_cache]
    if not missing:
        return
    unique = list(dict.fromkeys(missing))
    batch_size = 256
    for i in range(0, len(unique), batch_size):
        batch = unique[i:i + batch_size]
        try:
            embs = _get_embeddings_batch(batch)
            for text, emb in zip(batch, embs):
                _embedding_cache[text] = emb / np.linalg.norm(emb)
        except Exception as e:
            _log.warning("Embedding batch failed: %s", e)


def _embedding_sim(a: str, b: str) -> float:
    """Cosine similarity between two labels using cached embeddings."""
    ea = _embedding_cache.get(a)
    eb = _embedding_cache.get(b)
    if ea is None or eb is None:
        return 0.0
    return float(np.dot(ea, eb))

# 單一配對的理論最高分（raw label 1.0 + OCR 1.5 + grid 0.25 = 2.75，
# 再乘以 importance 5.0 = 13.75）。只用來當安全上限參考，不用來壓縮
# 分數到 0~1（v3 改用雙向覆蓋率的調和平均）。
_MAX_PAIR_SCORE = 15.0

_OCR_MATCH_WEIGHT = 1.5
_OCR_SUBSTRING_WEIGHT = 1.0

# ── 九宮格空間匹配 (3×3 grid cell matching) ──
_GRID_BONUS_SAME = 0.25
_GRID_BONUS_ADJACENT = 0.15

_GRID_ADJACENT = {
    "top-left":      ["top-center", "mid-left", "mid-center"],
    "top-center":    ["top-left", "top-right", "mid-center", "mid-left", "mid-right"],
    "top-right":     ["top-center", "mid-right", "mid-center"],
    "mid-left":      ["top-left", "top-center", "mid-center", "bottom-left", "bottom-center"],
    "mid-center":    ["top-left", "top-center", "top-right",
                      "mid-left", "mid-right",
                      "bottom-left", "bottom-center", "bottom-right"],
    "mid-right":     ["top-right", "top-center", "mid-center", "bottom-right", "bottom-center"],
    "bottom-left":   ["mid-left", "mid-center", "bottom-center"],
    "bottom-center": ["bottom-left", "bottom-right", "mid-center", "mid-left", "mid-right"],
    "bottom-right":  ["mid-right", "mid-center", "bottom-center"],
}

# ── 五級物件重要性加權 (5-level feature importance) ──
_IMPORTANCE_LANDMARK = 5.0
_IMPORTANCE_SIGN = 3.0
_IMPORTANCE_EQUIPMENT = 2.0
_IMPORTANCE_NORMAL = 1.0
_IMPORTANCE_GENERIC = 0.3

_SIGN_WORDS = frozenset({
    '標示', '指示', '告示', '看板', '招牌', '吊牌', '吊旗', 'sign', 'banner',
})
_ZONE_MARKERS = frozenset({
    '走道', '區域', '出口', 'exit', 'aisle', 'zone',
    '結帳', '收銀', '手扶梯', '電扶梯', '入口',
})
_CATEGORY_SIGNS = frozenset({
    '堅果', '海苔', '飲料', '零食', '冷凍', '冷藏', '生鮮', '日用',
    '烘焙', '麵包', '肉品', '海鮮', '水果', '蔬菜', '花卉',
    '咖啡', '茶', '乳品', '奶粉', '保健', '清潔', '衛生',
    '寵物', '酒', '啤酒', '調味', '罐頭', '餅乾', '早餐',
    '麵條', '即食', '米', '南北貨', '護理', '洗髮', '染髮',
})
_EQUIP_WORDS = frozenset({
    '冷藏展示櫃', '冷凍展示櫃', '冷凍櫃', '冰櫃', '冰箱',
    '展示櫃', '貨架', '櫃台', '磅秤', '烤箱',
    'refrigerat', 'freezer', 'display case', 'counter',
})
_GENERIC_WORDS = frozenset({
    '促銷立牌', '紅色促銷', '促銷', 'price tag', 'promotional',
    'price sign', '價牌', '價格',
})


def _classify_importance(label: str) -> Tuple[float, str]:
    combined = label.lower()
    for kw in _ZONE_MARKERS:
        if kw in combined:
            return _IMPORTANCE_LANDMARK, "landmark"
    has_sign = any(sw in combined for sw in _SIGN_WORDS)
    has_category = any(cw in combined for cw in _CATEGORY_SIGNS)
    if has_sign and has_category:
        return _IMPORTANCE_LANDMARK, "landmark"
    if has_sign:
        return _IMPORTANCE_SIGN, "sign"
    for kw in _EQUIP_WORDS:
        if kw in combined:
            return _IMPORTANCE_EQUIPMENT, "equipment"
    for kw in _GENERIC_WORDS:
        if kw in combined:
            return _IMPORTANCE_GENERIC, "generic"
    return _IMPORTANCE_NORMAL, "normal"


# 垂直分量的順序，用來算兩個分量差幾格（見 _vertical_rank() 的說明）。
_VERTICAL_ORDER = ("top", "middle", "bottom")


def _extract_vertical_component(position: str) -> str:
    """從 position 字串裡只挑出「垂直」（上/中/下）這個分量，忽略水平
    （左/中/右）分量跟距離（near/far）分量。

    為什麼要這樣做：`position` 目前實際上有兩種格式混用——
    - 系統自己算出來的（`position_bucket()`）：逗號分隔 "left,middle"
    - VLM 兩階段模式產生的：空白分隔 h/v、逗號再接距離 "left middle, near"
    不管哪種格式，水平分量都跟拍攝時的左右站位、鏡頭朝向強相關：同一個
    物件，攝影者往左站一點、或换個方向面對它拍，該物件在畫面上的水平
    位置就會不一樣，甚至左右順序整個顛倒（例如原本朝北拍，物件在畫面
    右邊；改成朝南、從對面拍，物件會跑到畫面左邊）。查詢照片是使用者
    獨立拍的一張新照片，跟建圖當時的站位、朝向不會一致，拿水平位置
    （或物件之間的左右關係）當比對依據，本質上是在比較兩個沒有共同
    基準的量，這種比對只會產生誤導，這裡刻意完全不使用。

    垂直分量（物件大概在畫面上/中/下）則相對穩定：只要兩次拍攝時手機
    拿的高度、俯仰角不要差太多，物件在畫面上下方向的相對位置通常還是
    大致吻合的，所以只保留這個分量當加分依據。
    """
    if not position:
        return ""
    v_words = ("top", "middle", "bottom")
    # 先看逗號前那一段（可能是 "left middle" 這種空白分隔的 h+v 組合，
    # 也可能只是單獨的 "left" 或 "middle"），逐一 token 找垂直詞彙。
    for part in position.split(","):
        for token in part.strip().split():
            if token in v_words:
                return token
    return ""


def _vertical_rank(v: str) -> Optional[int]:
    """把垂直分量轉成順序（top=0, middle=1, bottom=2），方便算「差幾格」。
    無法辨識就回傳 None。
    """
    try:
        return _VERTICAL_ORDER.index(v)
    except ValueError:
        return None


_SYNONYM_LOOKUP: Dict[str, str] = {}
for _canon, _variants in LABEL_SYNONYMS.items():
    for _v in _variants:
        _SYNONYM_LOOKUP[_v.lower()] = _canon.lower()

_SEMANTIC_SIM_DISCOUNT = 0.6

# 原文比對的權重上限：跟 IDF 稀有度權重的上限 (_LABEL_WEIGHT_CEIL=1.0)
# 對齊，原文匹配到的分數直接用這個值（不走 IDF），因為原文匹配本身就
# 是強證據，不需要再被 IDF 壓低。
_RAW_LABEL_MATCH_WEIGHT = 1.0
_RAW_LABEL_SIM_THRESHOLD = 0.4


def _raw_label_similarity(la: str, lb: str) -> float:
    """Compute similarity between two raw (un-normalized) labels.

    Raw labels are the original VLM output, e.g. "冷藏展示櫃 refrigerated
    display case".  Two calls to the same VLM looking at the same object
    produce labels that differ slightly but share most characters/words.
    This function captures that natural similarity without going through
    normalize_label(), preserving discriminative details like brand names,
    aisle numbers, and product categories that normalization would collapse.
    """
    if not la or not lb:
        return 0.0
    la_l, lb_l = la.lower(), lb.lower()
    if la_l == lb_l:
        return 1.0

    scores = []

    # Word overlap (Jaccard) — good for mixed zh/en labels
    words_a = set(la_l.split())
    words_b = set(lb_l.split())
    if words_a and words_b:
        scores.append(len(words_a & words_b) / len(words_a | words_b))

    # Character overlap (Jaccard) — good for Chinese labels where
    # word boundaries are ambiguous
    strip_chars = {" ", "/", "／", "(", ")", "（", "）", "「", "」", ",", "，"}
    chars_a = set(la_l) - strip_chars
    chars_b = set(lb_l) - strip_chars
    if chars_a and chars_b:
        scores.append(len(chars_a & chars_b) / len(chars_a | chars_b) * 0.85)

    # Substring containment — one label is a shorter form of the other
    if len(la_l) >= 3 and len(lb_l) >= 3:
        if la_l in lb_l or lb_l in la_l:
            scores.append(min(len(la_l), len(lb_l)) / max(len(la_l), len(lb_l)))

    # Number match — aisle numbers, product codes are highly discriminative
    nums_a = set(re.findall(r'\d+', la_l))
    nums_b = set(re.findall(r'\d+', lb_l))
    if nums_a and nums_b:
        if nums_a & nums_b:
            scores.append(0.8)
        else:
            scores.append(-0.3)  # different numbers → penalty

    # Category keyword match — both labels mention the same product category
    cats_a = {cw for cw in _CATEGORY_SIGNS if cw in la_l}
    cats_b = {cw for cw in _CATEGORY_SIGNS if cw in lb_l}
    if cats_a and cats_b and cats_a & cats_b:
        scores.append(0.85)

    return max(scores) if scores else 0.0


def _semantic_similarity(la: str, lb: str) -> float:
    """Compute a 0~1 semantic similarity between two normalized labels.

    Uses three signals (ported from classmate's approach):
    1. Synonym resolution — both map to the same canonical label → 0.9
    2. Word overlap — Jaccard on word tokens
    3. Character overlap — Jaccard on characters (helps with zh labels)
    4. Substring containment
    """
    if not la or not lb:
        return 0.0

    # Synonym check
    canon_a = _SYNONYM_LOOKUP.get(la, "")
    canon_b = _SYNONYM_LOOKUP.get(lb, "")
    if canon_a and canon_a == canon_b:
        return 0.9

    # Word overlap (Jaccard)
    words_a = set(la.split())
    words_b = set(lb.split())
    if words_a and words_b:
        word_jaccard = len(words_a & words_b) / len(words_a | words_b)
    else:
        word_jaccard = 0.0

    # Character overlap (good for Chinese labels)
    chars_a = set(la) - {" ", "/", "(", ")", "（", "）"}
    chars_b = set(lb) - {" ", "/", "(", ")", "（", "）"}
    if chars_a and chars_b:
        char_jaccard = len(chars_a & chars_b) / len(chars_a | chars_b)
    else:
        char_jaccard = 0.0

    # Substring containment
    substr_bonus = 0.0
    if len(la) >= 3 and len(lb) >= 3:
        if la in lb or lb in la:
            substr_bonus = 0.7

    return max(word_jaccard, char_jaccard, substr_bonus)


def _score_pair(da: dict, db: dict, label_weights: Dict[str, float]) -> Tuple[float, bool]:
    """算一組（查詢物件, 候選節點物件）的配對分數。

    跟舊版 `link_same_objects_between_photos()` 最大的差別：畫面位置
    在這裡只是加分項，不是門檻——查詢照片是獨立拍的，跟建圖照片的
    構圖、角度大機率對不上，硬性要求位置相近會讓幾乎所有配對都失敗
    （見模組開頭「比對邏輯」的說明）。用標籤稀有度取代位置，做為抑制
    通用標籤（垃圾桶、桌子…）亂配對的主要手段。

    這個加分項也**只比垂直位置**（畫面上/中/下），不比水平位置（左/中/
    右）——見 `_extract_vertical_component()` 的說明：水平位置、物件間
    的左右關係會隨拍攝站位、朝向改變甚至鏡像顛倒，不是可靠的比對依據，
    只有垂直位置相對穩定。

    垂直位置的比對本身也不是「完全相同格才算、差一格就當作沒對上」這種
    二選一：三等分的格線是依「這張照片的畫面高度」切的，如果兩次拍攝的
    構圖範圍不一樣（例如第二次為了拍到更多東西而站遠一點、或鏡頭角度
    稍微往下帶），同一個實際位置沒有變的物件，仍然可能因為分母（畫面
    涵蓋的垂直範圍）不同而被分到相鄰的另一格——這不代表物件真的移動了，
    只是量測基準跟著構圖變了。所以「差一格」（top↔middle 或
    middle↔bottom）給部分加分，只有「差兩格」（top↔bottom，兩個最極端
    的狀況，比較不像是構圖誤差、更像是真的不同物件）才不加分。

    Returns:
        (score, has_semantic_evidence)：has_semantic_evidence 代表這組
        配對有沒有「標籤相同」或「OCR/招牌文字相同」其中一項——畫面
        位置永遠不能單獨讓一組配對成立（不然「標籤完全不同、純粹畫面
        位置剛好落在同一個九宮格」這種巧合也會被當成配對，見
        `test_localize_low_confidence_when_query_unrelated` 這個回歸
        測試踩到的狀況）。
    """
    score = 0.0
    has_semantic_evidence = False

    # ── Layer 1: Raw label similarity (VLM-to-VLM) ──
    # Compare original VLM output text before normalization.  This
    # preserves discriminative details (brand names, aisle numbers,
    # product categories) that normalize_label() would collapse into
    # coarse categories like "sign" or "shelf".
    raw_a, raw_b = da.get("label", ""), db.get("label", "")
    raw_sim = _raw_label_similarity(raw_a, raw_b) if (raw_a and raw_b) else 0.0

    if raw_sim >= _RAW_LABEL_SIM_THRESHOLD:
        score += _RAW_LABEL_MATCH_WEIGHT * raw_sim
        has_semantic_evidence = True

    # ── Layer 2: Normalized label matching (fallback) ──
    # Only used when raw label similarity is too low — handles cases
    # where VLM descriptions are completely different in wording but
    # refer to the same category (e.g. "飲料冰箱" vs "beverage cooler"
    # both normalize to "refrigerator").
    if not has_semantic_evidence:
        la, lb = da.get("label_norm", ""), db.get("label_norm", "")
        if la and la == lb:
            score += label_weights.get(la, _LABEL_WEIGHT_FLOOR)
            has_semantic_evidence = True
        elif la and lb:
            sim = _semantic_similarity(la, lb)
            if sim >= 0.5:
                weight = label_weights.get(la, label_weights.get(lb, _LABEL_WEIGHT_FLOOR))
                score += weight * sim * _SEMANTIC_SIM_DISCOUNT
                has_semantic_evidence = True

    # ── Layer 3: OCR / signboard text ──
    oa, ob = da.get("ocr_text", ""), db.get("ocr_text", "")
    if oa and ob:
        if oa == ob:
            score += _OCR_MATCH_WEIGHT
            has_semantic_evidence = True
        elif len(oa) >= 2 and len(ob) >= 2 and (oa in ob or ob in oa):
            score += _OCR_SUBSTRING_WEIGHT
            has_semantic_evidence = True

    return score, has_semantic_evidence

# 標籤稀有度權重的下限／上限：即使一個標籤在地圖裡每個節點都出現
# （最不具鑑別力），配對到還是給一點點分數（不是 0）——因為就算標籤
# 通用，「同一種通用物件」還是比「完全配不到任何東西」更像是同一個
# 地方，只是這種配對不該單獨決定排名。
_LABEL_WEIGHT_FLOOR = 0.05
_LABEL_WEIGHT_CEIL = 1.0

# 單一配對分數低於這個值就不列入候選配對（避免位置巧合命中這種雜訊
# 配對混進「配對到的物件」清單裡，讓報告看起來像是配對到了什麼）。
_MIN_PAIR_SCORE_FLOOR = 0.05

# 算 recall_node / recall_query 時，分母（權重總和）加一個極小值避免
# 除以 0（候選節點或查詢照片完全沒有可用物件的極端情況）。
# 算 recall_node / recall_query 時，分母（權重總和）加一個平滑項——不是
# 為了避免除以 0（那個用 _WEIGHT_EPSILON 處理），是為了避免「候選節點
# 本身物件數很少（例如只有一個），隨便配對到一個就變成 100% 覆蓋率」
# 這種小樣本灌水。平滑值大致相當於「還缺一個中等稀有度物件才能真正
# 說服人」的量。
_COVERAGE_SMOOTHING = 1.2
_WEIGHT_EPSILON = 1e-6

# 圖結構驗證（垂直順序一致性）用的平滑／懲罰參數。
#
# 完全沒有可比較的組合時，要區分兩種完全不同的情況，不能套用同一個
# 懲罰值：
#   (a) 配對到的物件本身就太少（少於 2 個，連一組『兩兩比較』都湊不
#       出來）——證據量本身就不足，套用比較重的懲罰
#       _CONSISTENCY_TOO_FEW_MATCHES_PENALTY。
#   (b) 配對到的物件不算少（2 個以上），只是這些物件剛好都落在同一個
#       垂直格，沒有上下差異可以比對——這通常是物件本身的幾何特性
#       （例如辦公室的桌子、椅子、沙發本來就常常在差不多的高度），
#       不代表配對品質有問題，只是這組物件剛好沒有「順序」這個額外
#       資訊可用，只給溫和的懲罰 _CONSISTENCY_NO_SPREAD_PENALTY，不能
#       跟情況 (a) 一樣重罰，不然就是在懲罰場景本身的幾何特性，而不是
#       懲罰配對品質。
_CONSISTENCY_TOO_FEW_MATCHES_PENALTY = 0.5
_CONSISTENCY_NO_SPREAD_PENALTY = 0.85

# 真的有「可比較的組合」時（至少一組配對在兩張照片裡都分得出上下），
# 用貝氏平滑算一致比例：完全一致 → 接近 1.0；完全矛盾 → 接近 0；
# 中間值則反映「多少組合是矛盾的」。
_CONSISTENCY_PRIOR_CONSISTENT = 0.5
_CONSISTENCY_PRIOR_TOTAL = 1.0


# ══════════════════════════════════════════════════════════════════════════
# 結果資料結構 / Result data structures
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectMatch:
    """一組被判定為「同一物件」的查詢照片物件 ↔ 候選節點物件配對。"""
    query_object_id: int
    query_label: str
    node_object_id: int
    node_label: str
    pair_score: float


@dataclass
class LocalizationCandidate:
    """地圖裡的一個候選照片節點，附上跟查詢照片的比對結果。"""
    photo_id: int
    photo_rank: int            # 第幾張照片（跟 render_png() 的 P# 對應，從 1 開始）
    photo_label: str           # 例如 "P5"
    photo_path: str
    photo_file: str
    region: Optional[str]
    score: float                # 0~1，越高代表越可能是這個位置（雙向覆蓋率的調和平均 × 結構一致性）
    node_coverage: float         # 0~1，這個節點的內容被查詢照片證實了多少
    query_coverage: float        # 0~1，查詢照片看到的東西被這個節點解釋了多少
    structure_consistency: float  # 0~1，配對到的物件彼此上下順序有多一致（圖結構驗證）
    matched_count: int
    query_object_count: int
    node_object_count: int
    matched_direction: str = ""
    matches: List[ObjectMatch] = field(default_factory=list)


@dataclass
class LocalizationResult:
    query_object_count: int
    candidates: List[LocalizationCandidate]
    warning: Optional[str] = None

    @property
    def best(self) -> Optional[LocalizationCandidate]:
        """分數最高的候選節點；沒有任何候選時回傳 None。"""
        return self.candidates[0] if self.candidates else None


# ══════════════════════════════════════════════════════════════════════════
# 主要函式 / Main function
# ══════════════════════════════════════════════════════════════════════════

def _compute_label_rarity_weights(work: TopoGraphV2, candidate_photo_ids: List[int]) -> Dict[str, float]:
    """算出地圖裡每個標籤的「稀有度權重」：出現在越少照片節點的標籤，
    權重越接近 1.0（配對到很有鑑別力）；出現在越多節點的標籤，權重越
    接近下限 `_LABEL_WEIGHT_FLOOR`（配對到幾乎不能證明什麼，到處都有）。

    只統計「有幾個不同的照片節點包含這個標籤」（不管同一節點裡出現
    幾次），避免「同一節點裡有五個垃圾桶」把權重拉低成好像地圖到處
    都是垃圾桶一樣。
    """
    doc_count = max(len(candidate_photo_ids), 1)
    node_frequency: Dict[str, int] = {}
    for pid in candidate_photo_ids:
        labels_in_this_node = {
            o.get("label_norm") or o.get("label", "")
            for o in work.photo_objects(pid)
        }
        labels_in_this_node.discard("")
        for label in labels_in_this_node:
            node_frequency[label] = node_frequency.get(label, 0) + 1

    weights: Dict[str, float] = {}
    log_n = math.log(doc_count + 1)
    for label, freq in node_frequency.items():
        idf = math.log(doc_count / freq + 1) / log_n
        weights[label] = max(_LABEL_WEIGHT_FLOOR, min(idf, _LABEL_WEIGHT_CEIL))
    return weights


def _is_excluded_structural(obj: dict, allow_structural: bool) -> bool:
    """判斷一個結構物（門/牆/窗等）該不該被排除在比對之外。

    預設(`allow_structural=False`)排除所有結構物，因為這類物件到處
    都有，單靠「這是一扇門」配對到，證明力太弱。但**帶著 OCR 文字的
    結構物是例外**——例如一扇貼著房間名稱牌子的門，那段文字本身就是
    很有鑑別力的證據（不輸任何家具），不該因為物件類別是「門」就被
    一律擋掉。所以規則是：「單純的結構物」排除，「結構物 + OCR 文字」
    放行；`allow_structural=True` 時則完全不排除任何結構物（連沒有
    文字的也放行，適合想全面比較的情境）。
    """
    if allow_structural:
        return False
    if obj.get("role") != ROLE_STRUCTURE:
        return False
    has_ocr = bool(obj.get("ocr_text"))
    return not has_ocr


def _normalize_grid_cell(gc: str) -> str:
    """Convert topomap grid_cell format to classmate's format for _GRID_ADJACENT lookup.

    topomap uses: "left", "center", "right", "top_left", "bottom_center", ...
    classmate/imported uses: "mid-left", "mid-center", "mid-right", "top-left", ...
    """
    if not gc:
        return "mid-center"
    gc = gc.replace("_", "-")
    if gc in ("left", "center", "right"):
        gc = f"mid-{gc}"
    return gc


def _grid_compatible(gc_a: str, gc_b: str) -> float:
    if not gc_a or not gc_b:
        return 1.0
    if gc_a == gc_b:
        return 1.0
    if gc_b in _GRID_ADJACENT.get(gc_a, []):
        return 0.7
    return 0.0


def _direction_matches(ref_dir: str, query_dir: str) -> bool:
    if not query_dir or not ref_dir:
        return False
    return (ref_dir == query_dir
            or ref_dir.endswith("_" + query_dir)
            or ref_dir.startswith(query_dir + "_"))


_GRID_ORDER = ["top-left", "top-center", "top-right",
               "mid-left", "mid-center", "mid-right",
               "bottom-left", "bottom-center", "bottom-right"]


def _box_to_grid_cell(box) -> str:
    if not box or len(box) != 4:
        return "mid-center"
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "mid")
    return f"{row}-{col}"


def _wgm_similarity(ql: str, rl: str) -> float:
    """Label similarity for weighted_grid_match (classmate's formula)."""
    ql_low, rl_low = ql.lower(), rl.lower()
    if ql_low == rl_low:
        return 1.0
    scores: List[float] = []
    ql_words = set(ql_low.split())
    rl_words = set(rl_low.split())
    if ql_words and rl_words:
        common = ql_words & rl_words
        if common:
            scores.append(len(common) / max(len(ql_words), len(rl_words)))
    strip = {" ", "/", "／", "（", "）", "「", "」"}
    ql_chars = set(ql_low) - strip
    rl_chars = set(rl_low) - strip
    if ql_chars and rl_chars:
        common_c = ql_chars & rl_chars
        if common_c:
            scores.append(len(common_c) / max(len(ql_chars), len(rl_chars)) * 0.8)
    q_nums = set(re.findall(r"\d+", ql_low))
    r_nums = set(re.findall(r"\d+", rl_low))
    if q_nums and r_nums:
        scores.append(0.9 if q_nums & r_nums else -0.3)
    q_cats = {cw for cw in _CATEGORY_SIGNS if cw in ql_low}
    r_cats = {cw for cw in _CATEGORY_SIGNS if cw in rl_low}
    if q_cats and r_cats and q_cats & r_cats:
        scores.append(0.85)
    if len(ql_low) >= 2 and len(rl_low) >= 2:
        if ql_low in rl_low or rl_low in ql_low:
            scores.append(min(len(ql), len(rl)) / max(len(ql), len(rl)))
    return max(scores) if scores else 0.0


def _weighted_grid_match(
    query_grid: Dict[str, list],
    ref_grid: Dict[str, list],
    use_embedding: bool = False,
) -> float:
    """Classmate's weighted_grid_match: per-cell greedy matching with
    adjacent-cell tolerance (0.7 penalty).
    Score = Σ(weight × sim) / Σ(weight).

    When use_embedding=True, uses cosine similarity from cached embeddings
    instead of string-based matching.
    """
    total_weighted = 0.0
    total_possible = 0.0

    for cell in _GRID_ORDER:
        q_items = query_grid.get(cell, [])
        r_same = ref_grid.get(cell, [])
        r_adj = []
        for ac in _GRID_ADJACENT.get(cell, []):
            for ri in ref_grid.get(ac, []):
                r_adj.append({**ri, "_p": 0.7})
        all_r = [{**ri, "_p": 1.0} for ri in r_same] + r_adj

        if not q_items or not all_r:
            total_possible += sum(it["weight"] for it in q_items)
            continue

        used: set = set()
        for qi in q_items:
            ql, qw = qi["label"], qi["weight"]
            best_s, best_i = 0.0, -1
            for ri_i, ri in enumerate(all_r):
                k = (id(ri), ri["label"])
                if k in used:
                    continue
                if use_embedding:
                    raw_sim = _embedding_sim(ql, ri["label"])
                else:
                    raw_sim = _wgm_similarity(ql, ri["label"])
                s = raw_sim * ri.get("_p", 1.0)
                if s > best_s:
                    best_s, best_i = s, ri_i
            threshold = 0.35 if use_embedding else 0.25
            if best_i >= 0 and best_s > threshold:
                used.add((id(all_r[best_i]), all_r[best_i]["label"]))
                total_weighted += qw * best_s
            total_possible += qw

    return total_weighted / max(total_possible, 1e-6)


def _build_query_grid(detections: List[dict]) -> Dict[str, list]:
    """Build a 3×3 grid from raw detection items (VLM output format)."""
    grid: Dict[str, list] = {}
    for det in detections:
        box = det.get("box", [])
        if not box or len(box) != 4:
            continue
        if max(box) > 1.0:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw >= 0.55 or bh >= 0.55:
            continue
        cell = _box_to_grid_cell(box)
        label = det.get("label", "")
        weight, cat = _classify_importance(label)
        grid.setdefault(cell, []).append({"label": label, "weight": weight, "cat": cat})
    return grid


def _build_ref_grids(
    work: TopoGraphV2,
    photo_id: int,
    allow_structural: bool,
) -> Dict[str, Dict[str, list]]:
    """Build per-direction reference grids from a photo node's objects."""
    grids: Dict[str, Dict[str, list]] = {}
    for o in work.photo_objects(photo_id):
        if _is_excluded_structural(o, allow_structural):
            continue
        raw_dir = o.get("direction", "front")
        base_dir = raw_dir
        for d in ("front", "back", "left", "right"):
            if d in raw_dir:
                base_dir = d
                break
        cell = _normalize_grid_cell(o.get("grid_cell", ""))
        label = o.get("label", "")
        weight, cat = _classify_importance(label)
        grids.setdefault(base_dir, {}).setdefault(cell, []).append(
            {"label": label, "weight": weight, "cat": cat}
        )
    return grids


def _best_wgm_score(
    query_grid: Dict[str, list],
    ref_grids: Dict[str, Dict[str, list]],
    use_embedding: bool = False,
) -> Tuple[float, str]:
    """Try matching query grid against all direction slots, return (best_score, best_direction)."""
    best = 0.0
    best_dir = ""
    for _dir, ref_grid in ref_grids.items():
        s = _weighted_grid_match(query_grid, ref_grid, use_embedding=use_embedding)
        if s > best:
            best = s
            best_dir = _dir
    return best, best_dir


def _greedy_match_objects(
    work: TopoGraphV2,
    query_object_ids: List[int],
    node_object_ids: List[int],
    label_weights: Dict[str, float],
    allow_structural: bool,
    min_pair_score: float,
    query_direction: str = "",
    use_grid_filter: bool = False,
) -> List[ObjectMatch]:
    """在查詢物件跟某個候選節點的物件之間，用貪婪配對找出「同一物件」。

    use_grid_filter=True 時，只允許同格或相鄰格配對（相鄰格打 70% 折），
    適用於已經過方向過濾的物件集（同方向內 grid_cell 有意義）。
    """
    candidates: List[Tuple[float, int, int]] = []
    for a in query_object_ids:
        da = work.graph.nodes[a]
        if _is_excluded_structural(da, allow_structural):
            continue
        gc_a = da.get("grid_cell", "") if use_grid_filter else ""
        for b in node_object_ids:
            db = work.graph.nodes[b]
            if _is_excluded_structural(db, allow_structural):
                continue
            gc_discount = 1.0
            if gc_a:
                gc_discount = _grid_compatible(gc_a, db.get("grid_cell", ""))
                if gc_discount == 0.0:
                    continue
            s, has_evidence = _score_pair(da, db, label_weights)
            if gc_a:
                s *= gc_discount
            if has_evidence and s >= min_pair_score:
                candidates.append((s, a, b))

    candidates.sort(key=lambda t: t[0], reverse=True)
    used_a, used_b = set(), set()
    matches: List[ObjectMatch] = []
    for s, a, b in candidates:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        matches.append(ObjectMatch(
            query_object_id=a,
            query_label=work.graph.nodes[a].get("label", ""),
            node_object_id=b,
            node_label=work.graph.nodes[b].get("label", ""),
            pair_score=s,
        ))
    return matches


_DISC_RARE_THRESHOLD = 0.35
_DISC_OCR_PAIR_SCORE = 1.0
_DISC_ALL_COMMON_PENALTY = 0.55


def _discriminative_quality(
    work: TopoGraphV2, matches: List[ObjectMatch], label_weights: Dict[str, float],
) -> float:
    """Compute a multiplier reflecting how discriminative the matched evidence is.

    When all matched labels are very common (shelf, aisle, snack — appear in
    nearly every map photo), the evidence is inherently ambiguous regardless
    of the coverage score.  This multiplier penalises such weak evidence so
    the final score better reflects actual confidence.
    """
    if not matches:
        return _DISC_ALL_COMMON_PENALTY

    has_ocr = any(m.pair_score >= _DISC_OCR_PAIR_SCORE for m in matches)
    if has_ocr:
        return 1.0

    max_w = max(
        label_weights.get(
            work.graph.nodes[m.query_object_id].get("label_norm", ""),
            _LABEL_WEIGHT_FLOOR,
        )
        for m in matches
    )
    if max_w >= _DISC_RARE_THRESHOLD:
        return 1.0

    return _DISC_ALL_COMMON_PENALTY


def _pairwise_vertical_consistency(work: TopoGraphV2, matches: List[ObjectMatch]) -> float:
    """圖結構驗證：檢查配對成功的物件們，彼此的「上下順序」在查詢照片
    跟候選節點裡是否一致，回傳一個 0~1 的乘數，直接乘進最終信心分數。

    刻意用「垂直順序」而不是地圖裡既有的 `adjacent` 邊（那個邊是照
    水平座標排的，等於是「誰在誰左邊」）——水平關係已經證實會隨拍攝
    站位、朝向改變甚至鏡像顛倒，不可靠；垂直順序相對穩定，適合拿來做
    這種結構一致性檢查。

    做法：對每兩組配對 (i, j)，各自看查詢照片裡 i、j 誰在上面、候選
    節點裡 i、j 誰在上面，兩者一致就算「相符」，不一致就算「矛盾」。
    任一邊 i、j 剛好在同一個垂直格（分不出上下）的組合直接跳過、不列
    入統計（見模組使用說明或對話紀錄裡的例子表格）。

    完全沒有可比較組合時，**區分兩種情況**（這是實測踩到問題後修正的
    地方，一開始沒有分開處理）：
    - 配對到的物件本身太少（< 2 個，連一組「兩兩比較」都湊不出來）：
      證據量真的不足，套用比較重的懲罰 `_CONSISTENCY_TOO_FEW_MATCHES_PENALTY`。
    - 配對到的物件不算少，只是剛好都在同一個垂直格（例如辦公室的
      桌子、椅子、沙發本來就常常差不多高）：這是物件本身的幾何特性，
      不代表配對品質有問題，只給溫和懲罰
      `_CONSISTENCY_NO_SPREAD_PENALTY`——不然就是在懲罰場景的幾何
      特性，而不是懲罰配對品質，會把明明配對得很準的候選節點分數
      壓得不合理地低。

    真的有可比較組合時，才用貝氏平滑算「一致比例」：完全一致 → 接近
    1.0；完全矛盾 → 接近 0。
    """
    ranks: List[Tuple[Optional[int], Optional[int]]] = []
    for m in matches:
        qa = work.graph.nodes[m.query_object_id]
        nb = work.graph.nodes[m.node_object_id]
        rq = _vertical_rank(_extract_vertical_component(qa.get("position", "")))
        rn = _vertical_rank(_extract_vertical_component(nb.get("position", "")))
        ranks.append((rq, rn))

    consistent = 0
    comparable = 0
    for i in range(len(ranks)):
        rq_i, rn_i = ranks[i]
        if rq_i is None or rn_i is None:
            continue
        for j in range(i + 1, len(ranks)):
            rq_j, rn_j = ranks[j]
            if rq_j is None or rn_j is None:
                continue
            if rq_i == rq_j or rn_i == rn_j:
                continue  # 任一邊分不出上下順序，這組組合忽略不計
            comparable += 1
            if (rq_i - rq_j > 0) == (rn_i - rn_j > 0):
                consistent += 1

    if comparable == 0:
        if len(matches) < 2:
            return _CONSISTENCY_TOO_FEW_MATCHES_PENALTY
        return _CONSISTENCY_NO_SPREAD_PENALTY

    return (consistent + _CONSISTENCY_PRIOR_CONSISTENT) / (comparable + _CONSISTENCY_PRIOR_TOTAL)


_OCR_TOKEN_SPLIT = re.compile(r'[/,，、:：;\s]+')
_OCR_GLOBAL_EXACT_WEIGHT = 0.25
_OCR_GLOBAL_SUBSTR_WEIGHT = 0.15
_OCR_GLOBAL_MAX_BONUS = 0.5


def _tokenize_ocr(text: str) -> Set[str]:
    if not text:
        return set()
    tokens = _OCR_TOKEN_SPLIT.split(text)
    result = set()
    for t in tokens:
        t = t.strip()
        if len(t) < 2:
            continue
        if re.match(r'^走道\d+$', t):
            continue
        result.add(t)
    return result


def _global_ocr_bonus(
    query_ocr_raw: List[str],
    node_ocr_raw: List[str],
) -> float:
    if not query_ocr_raw or not node_ocr_raw:
        return 0.0
    q_tokens: Set[str] = set()
    for t in query_ocr_raw:
        q_tokens |= _tokenize_ocr(t)
    n_tokens: Set[str] = set()
    for t in node_ocr_raw:
        n_tokens |= _tokenize_ocr(t)
    if not q_tokens or not n_tokens:
        return 0.0
    bonus = 0.0
    for qt in q_tokens:
        for nt in n_tokens:
            if qt == nt:
                bonus += _OCR_GLOBAL_EXACT_WEIGHT
                break
            if len(qt) >= 2 and len(nt) >= 2 and (qt in nt or nt in qt):
                bonus += _OCR_GLOBAL_SUBSTR_WEIGHT
                break
    return min(bonus, _OCR_GLOBAL_MAX_BONUS)


def _weighted_object_total(
    work: TopoGraphV2, object_ids: List[int], label_weights: Dict[str, float], allow_structural: bool,
) -> float:
    """算一組物件的標籤稀有度權重總和，當作 recall 分母。"""
    total = 0.0
    for oid in object_ids:
        obj = work.graph.nodes[oid]
        if _is_excluded_structural(obj, allow_structural):
            continue
        label = obj.get("label_norm", "")
        total += label_weights.get(label, _LABEL_WEIGHT_FLOOR)
    return total


def _localize_internal(
    topo: TopoGraphV2,
    detections: List[dict],
    img_w: int,
    img_h: int,
    *,
    ocr_items: Optional[List[dict]],
    top_k: int,
    min_pair_score: float,
    allow_structural: bool,
    low_confidence_threshold: float,
    query_direction: str = "",
    raw_vlm_items: Optional[List[dict]] = None,
    use_embedding: bool = False,
) -> Tuple[TopoGraphV2, Optional[int], LocalizationResult]:
    """`localize_photo()` 的實作本體，額外回傳內部用的地圖副本跟查詢節點
    id，讓 `render_localization_debug_png()` 可以重用同一次比對結果來畫
    圖，不用重算一次（也保證報告數字跟畫出來的圖絕對一致）。
    """
    work = copy.deepcopy(topo)

    if not detections:
        return work, None, LocalizationResult(
            query_object_count=0, candidates=[],
            warning="這張照片沒有偵測到任何物件，無法比對定位。",
        )

    query_photo_id = work.add_photo_node("", note="__locate_query__")
    work.graph.nodes[query_photo_id]["is_query"] = True
    query_object_ids = work.build_subgraph_from_detections(
        query_photo_id, detections, img_w, img_h, ocr_items=ocr_items,
    )

    if not query_object_ids:
        return work, query_photo_id, LocalizationResult(
            query_object_count=0, candidates=[],
            warning="這張照片的偵測結果沒有建出任何可用的物件節點，無法比對定位。",
        )

    candidate_photo_ids = [
        pid for pid in work.all_photo_nodes()
        if pid != query_photo_id and not work.graph.nodes[pid].get("is_virtual")
    ]

    label_weights = _compute_label_rarity_weights(work, candidate_photo_ids)
    query_weight_total = _weighted_object_total(
        work, list(query_object_ids), label_weights, allow_structural,
    )

    query_ocr_raw: List[str] = []
    for oid in query_object_ids:
        ocr = work.graph.nodes[oid].get("ocr_text", "")
        if ocr:
            query_ocr_raw.append(ocr)

    query_grid: Optional[Dict[str, list]] = None
    if raw_vlm_items:
        query_grid = _build_query_grid(raw_vlm_items)

    candidates: List[LocalizationCandidate] = []
    for pid in candidate_photo_ids:
        node_objects = work.photo_objects(pid)
        node_object_ids = [o["id"] for o in node_objects]

        matches = _greedy_match_objects(
            work, list(query_object_ids), node_object_ids,
            label_weights, allow_structural, min_pair_score,
            query_direction=query_direction,
        ) if node_object_ids else []

        node_weight_total = _weighted_object_total(work, node_object_ids, label_weights, allow_structural)
        matched_weight = sum(m.pair_score for m in matches)

        recall_node = matched_weight / max(node_weight_total + _COVERAGE_SMOOTHING, _WEIGHT_EPSILON)
        recall_query = matched_weight / max(query_weight_total + _COVERAGE_SMOOTHING, _WEIGHT_EPSILON)
        if recall_node + recall_query > 0:
            coverage_score = 2 * recall_node * recall_query / (recall_node + recall_query)
        else:
            coverage_score = 0.0
        coverage_score = min(coverage_score, 1.0)

        node_ocr_raw: List[str] = [
            o.get("ocr_text", "") for o in node_objects if o.get("ocr_text")
        ]
        ocr_bonus = _global_ocr_bonus(query_ocr_raw, node_ocr_raw)

        consistency = _pairwise_vertical_consistency(work, matches)

        disc = _discriminative_quality(work, matches, label_weights)
        f1_score = min((coverage_score + ocr_bonus) * consistency * disc, 1.0)

        dir_f1_score = 0.0
        best_dir_f1_direction = ""
        for try_dir in ("front", "back", "left", "right"):
            same_dir_oids = [
                o["id"] for o in node_objects
                if _direction_matches(o.get("direction", ""), try_dir)
                and not _is_excluded_structural(o, allow_structural)
            ]
            if not same_dir_oids:
                continue
            dir_matches = _greedy_match_objects(
                work, list(query_object_ids), same_dir_oids,
                label_weights, allow_structural, min_pair_score,
            )
            dir_node_wt = _weighted_object_total(
                work, same_dir_oids, label_weights, allow_structural,
            )
            dir_matched_wt = sum(m.pair_score for m in dir_matches)
            dir_recall_n = dir_matched_wt / max(dir_node_wt + _COVERAGE_SMOOTHING, _WEIGHT_EPSILON)
            dir_recall_q = dir_matched_wt / max(query_weight_total + _COVERAGE_SMOOTHING, _WEIGHT_EPSILON)
            if dir_recall_n + dir_recall_q > 0:
                dir_cov = 2 * dir_recall_n * dir_recall_q / (dir_recall_n + dir_recall_q)
            else:
                dir_cov = 0.0
            dir_cov = min(dir_cov, 1.0)
            dir_cons = _pairwise_vertical_consistency(work, dir_matches)
            dir_disc = _discriminative_quality(work, dir_matches, label_weights)
            dir_ocr_same = [
                o.get("ocr_text", "") for o in node_objects
                if o.get("ocr_text")
                and _direction_matches(o.get("direction", ""), try_dir)
            ]
            dir_ocr_bonus = _global_ocr_bonus(query_ocr_raw, dir_ocr_same)
            this_dir_score = min((dir_cov + dir_ocr_bonus) * dir_cons * dir_disc, 1.0)
            if this_dir_score > dir_f1_score:
                dir_f1_score = this_dir_score
                best_dir_f1_direction = try_dir

        wgm_score = 0.0
        wgm_dir = ""
        if query_grid:
            ref_grids = _build_ref_grids(work, pid, allow_structural)
            if ref_grids:
                wgm_score, wgm_dir = _best_wgm_score(query_grid, ref_grids, use_embedding=use_embedding)

        if query_grid:
            score = wgm_score
            best_matched_dir = wgm_dir
        else:
            score = max(f1_score, dir_f1_score)
            best_matched_dir = best_dir_f1_direction

        photo_node = work.graph.nodes[pid]
        candidates.append(LocalizationCandidate(
            photo_id=pid,
            photo_rank=work._photo_sequence_rank(pid),
            photo_label=work._photo_label(pid),
            photo_path=photo_node.get("photo_path", ""),
            photo_file=photo_node.get("photo_file", ""),
            region=photo_node.get("region"),
            score=round(score, 4),
            node_coverage=round(min(recall_node, 1.0), 4),
            query_coverage=round(min(recall_query, 1.0), 4),
            structure_consistency=round(consistency, 4),
            matched_count=len(matches),
            query_object_count=len(query_object_ids),
            node_object_count=len(node_object_ids),
            matched_direction=best_matched_dir,
            matches=sorted(matches, key=lambda m: m.pair_score, reverse=True),
        ))

    candidates.sort(key=lambda c: (-c.score, -c.matched_count, c.photo_id))
    top = candidates[:top_k] if top_k >= 0 else candidates

    warning = None
    if not top or top[0].score < low_confidence_threshold:
        warning = (
            f"最高信心分數偏低（< {low_confidence_threshold:.0%}），比對證據薄弱："
            "可能是這張照片不在目前地圖涵蓋的範圍內，或跟建圖時的偵測結果"
            "差異太大。以下仍列出分數最高的候選，僅供參考，不建議直接採信。"
        )

    result = LocalizationResult(
        query_object_count=len(query_object_ids), candidates=top, warning=warning,
    )
    return work, query_photo_id, result


def localize_photo(
    topo: TopoGraphV2,
    detections: List[dict],
    img_w: int,
    img_h: int,
    *,
    ocr_items: Optional[List[dict]] = None,
    top_k: int = 5,
    min_pair_score: float = _MIN_PAIR_SCORE_FLOOR,
    allow_structural: bool = False,
    low_confidence_threshold: float = 0.15,
    query_direction: str = "",
    raw_vlm_items: Optional[List[dict]] = None,
    use_embedding: bool = False,
) -> LocalizationResult:
    """給一張新照片的偵測結果，推理它最可能對應到 `topo` 裡的哪個照片節點。

    Args:
        topo: 要比對的場所地圖（不會被修改，見模組開頭說明）。
        detections: 這張新照片的偵測結果。
        img_w, img_h: 這張新照片的寬高（像素）。
        ocr_items: 這張新照片的 OCR 結果（可選）。
        top_k: 回傳前幾名候選節點。
        min_pair_score: 單一組物件配對的最低分數門檻。
        allow_structural: 是否允許無 OCR 的環境結構物參與比對。
        low_confidence_threshold: 觸發低信心警告的門檻。
        query_direction: 查詢照片的拍攝方向（front/back/left/right），
            用於九宮格空間過濾——同方向的參考物件才比對 grid_cell。
        raw_vlm_items: VLM 原始輸出 items（含 box/zh/en），用於
            weighted_grid_match 計分——繞過 graph 正規化，保留原始鑑別力。
        use_embedding: 使用 embedding cosine similarity 取代字串比對。

    Returns:
        LocalizationResult，`candidates` 已依信心分數由高到低排序。
    """
    _work, _query_id, result = _localize_internal(
        topo, detections, img_w, img_h,
        ocr_items=ocr_items, top_k=top_k, min_pair_score=min_pair_score,
        allow_structural=allow_structural, low_confidence_threshold=low_confidence_threshold,
        query_direction=query_direction,
        raw_vlm_items=raw_vlm_items,
        use_embedding=use_embedding,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════
# 除錯用視覺化 / Debug visualization
# ══════════════════════════════════════════════════════════════════════════

def render_localization_debug_png(
    topo: TopoGraphV2,
    detections: List[dict],
    img_w: int,
    img_h: int,
    *,
    ocr_items: Optional[List[dict]] = None,
    top_k: int = 5,
    min_pair_score: float = _MIN_PAIR_SCORE_FLOOR,
    allow_structural: bool = False,
    low_confidence_threshold: float = 0.15,
    link_top_n: int = 3,
) -> Tuple[bytes, LocalizationResult]:
    """把查詢照片畫成地圖上的一個節點，方便直接用眼睛檢查比對結果。

    做法：重用 `_localize_internal()` 算出來的同一份地圖副本（保證圖上
    看到的跟文字報告的數字是同一次計算結果，不會兜不起來），把它跟
    前 `link_top_n` 名候選節點配對到的物件，用「同物件關聯」邊連起來
    （分數越高的候選，看起來就跟查詢節點物件連得越密），但**刻意不**
    用走道邊把查詢節點接進地圖的路徑拓樸——查詢照片不是使用者真的走
    到、拍下的照片，跟任何節點連一條走道邊都會讓它看起來像是路徑的
    一部分（不管接在哪個節點後面，排版都會把它畫成「路徑接下去的
    某一步」），容易誤導成「這是導航會走到的下一個地方」。查詢節點
    因此完全不連走道邊，只靠物件配對線（同物件關聯邊）跟地圖產生
    視覺上的關聯，讀圖的人一眼就能看出它是外加的比對節點，不是路徑
    上真正的一站。

    完全沒有修改 `TopoGraphV2.render_png()` 本身一行程式碼——這裡只是
    在一份用完即丟的地圖副本上，多接幾條「同物件關聯」邊再呼叫它，
    畫出來的視覺風格、圖例會跟平常看到的拓樸地圖 PNG 完全一致。查詢
    節點因為沒有走道邊連到主要路徑，會被 `_dead_reckon_tree()` 內建的
    「沒被走道邊連到主要分量的節點」後備邏輯獨立擺在地圖的一角，不會
    插進主路徑的直線序列裡。

    Args:
        （跟 `localize_photo()` 相同的參數，見該函式說明）
        link_top_n: 要把查詢節點的配對物件連到前幾名候選節點上（分數
            相近的前幾名一起畫出來，比較容易看出「其實不確定是哪一個」
            這種情況；預設 3）。

    Returns:
        (png_bytes, result)：PNG 圖片內容，跟這次比對的完整結果（跟直接
        呼叫 `localize_photo()` 拿到的東西一樣，不用另外再呼叫一次）。
    """
    work, query_photo_id, result = _localize_internal(
        topo, detections, img_w, img_h,
        ocr_items=ocr_items, top_k=top_k, min_pair_score=min_pair_score,
        allow_structural=allow_structural, low_confidence_threshold=low_confidence_threshold,
    )

    if query_photo_id is None or not result.candidates:
        # 沒有查詢節點可畫（偵測結果是空的）或完全沒有候選節點，直接把
        # 原本的地圖畫出來就好，至少讓使用者看到地圖本身長怎樣。
        return work.render_png(), result

    work.current_position = query_photo_id  # 借用「使用者目前位置」的紅星徽章標示查詢節點

    for candidate in result.candidates[:max(link_top_n, 0)]:
        for m in candidate.matches:
            work.add_same_object_edge(
                m.query_object_id, m.node_object_id, confidence=min(m.pair_score / _MAX_PAIR_SCORE, 1.0),
            )

    return work.render_png(), result


# ══════════════════════════════════════════════════════════════════════════
# 文字報告 / Human-readable report
# ══════════════════════════════════════════════════════════════════════════

def format_localization_report(result: LocalizationResult, topo: TopoGraphV2) -> str:
    """把 `localize_photo()` 的結果轉成人看的文字報告（CLI 用）。"""
    lines: List[str] = []
    lines.append(f"場所：{topo.place_name or '(未命名場所)'}")
    lines.append(f"這張照片偵測到 {result.query_object_count} 個可用於比對的物件。")

    if result.warning:
        lines.append("")
        lines.append(f"警告：{result.warning}")

    if not result.candidates:
        lines.append("")
        lines.append("沒有任何候選節點（地圖可能是空的，或這張照片完全沒有可比對的物件）。")
        return "\n".join(lines)

    lines.append("")
    lines.append("最可能的位置（由高到低排序）：")
    for rank, c in enumerate(result.candidates, start=1):
        lines.append(
            f"  {rank}. {c.photo_label}（節點 id={c.photo_id}）"
            f" — 信心分數 {c.score:.0%}"
            f"（配對到 {c.matched_count} 個物件："
            f"該節點既有內容獲驗證比例為 {c.node_coverage:.0%}，"
            f"本次偵測結果獲對應比例為 {c.query_coverage:.0%}，"
            f"物件空間排列一致性為 {c.structure_consistency:.0%}）"
        )
        if c.photo_file:
            lines.append(f"       照片檔案：{c.photo_file}")
        if c.region:
            lines.append(f"       區域：{c.region}")
        if c.matches:
            shown = c.matches[:5]
            match_strs = [f"{m.query_label}↔{m.node_label}({m.pair_score:.1f})" for m in shown]
            suffix = " ..." if len(c.matches) > len(shown) else ""
            lines.append(f"       配對物件：{', '.join(match_strs)}{suffix}")
    return "\n".join(lines)
