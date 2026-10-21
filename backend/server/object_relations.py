"""object_relations.py — 物件與物件的上下左右關係，用在定位。

一張照片裡兩個物件的關係由 bbox（正規化 0..1，[x1, y1, x2, y2]，y 向下）決定，兩軸分開看：

- 上下：兩個框在垂直方向幾乎不重疊（重疊 ≤ 較矮那個的 30%）→ 上面那個 ABOVE 下面那個。
- 左右：兩個框在水平方向幾乎不重疊 → 左邊那個 LEFT_OF 右邊那個。
- 重疊太多或中心差太小 → 那一軸沒有關係（同一層／同一欄），不拿來比。

斜對角的兩個物件兩軸都有關係（左上方）。建圖端（build_object_relations.py 寫進 Neo4j 的
ABOVE / BELOW / LEFT_OF / RIGHT_OF）與定位端（relation_score）用同一個 pair_relation，
兩邊算出來的關係才對得上。

定位時（locate_fusion.server_image_evidence）拿查詢照片的物件和每個節點「最像的那張方向照片」
（DINOv2＋物件＋OCR 選出來的）的物件比：先依名稱相似度一對一配對，再看配到的每兩個物件，
在查詢照片和參考照片裡的上下、左右關係是否一致。

左右的權重比上下低：上下關係（吊牌在貨架上面、上層在下層上面）站在哪裡拍都一樣，左右會隨
站位與遠近（視差）改變，換個朝向甚至左右對調（legacy/定位方法彙整與測試規劃.md：
舊版不分軸的 graph matching 沒有幫助）。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Box = Sequence[float]

MAX_OVERLAP = 0.30      # 某一軸重疊超過較短那邊的 30% → 這一軸沒有關係
MIN_CENTER_GAP = 0.03   # 中心差小於畫面的 3% → 這一軸沒有關係
NEIGHBOURS = 4          # 建圖時每個物件只連最近的 4 個（Neo4j 邊數；定位比對用全部配對）


def _valid(box: Optional[Box]) -> bool:
    return bool(box) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def _axis(a1: float, a2: float, b1: float, b2: float) -> int:
    """-1：a 在 b 前面（上／左），1：a 在 b 後面，0：這一軸沒有關係。"""
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    if overlap > MAX_OVERLAP * min(a2 - a1, b2 - b1):
        return 0
    gap = (b1 + b2) / 2 - (a1 + a2) / 2
    if abs(gap) < MIN_CENTER_GAP:
        return 0
    return -1 if gap > 0 else 1


@dataclass(frozen=True)
class PairRelation:
    v: int   # -1：a 在 b 上面（a ABOVE b），1：a 在 b 下面，0：同一層
    h: int   # -1：a 在 b 左邊（a LEFT_OF b），1：a 在 b 右邊，0：同一欄


def pair_relation(a: Box, b: Box) -> PairRelation:
    if not (_valid(a) and _valid(b)):
        return PairRelation(0, 0)
    return PairRelation(v=_axis(a[1], a[3], b[1], b[3]), h=_axis(a[0], a[2], b[0], b[2]))


def neighbour_pairs(boxes: Sequence[Optional[Box]], k: int = NEIGHBOURS) -> List[Tuple[int, int]]:
    """(i, j)，i < j：每個物件跟它最近的 k 個（框與框的距離）。建圖寫 Neo4j 用。"""
    def gap(a: Box, b: Box) -> float:
        dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
        dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
        return math.hypot(dx, dy)

    idx = [i for i, b in enumerate(boxes) if _valid(b)]
    out = set()
    for i in idx:
        near = sorted((gap(boxes[i], boxes[j]), j) for j in idx if j != i)[:k]
        out.update((min(i, j), max(i, j)) for _, j in near)
    return sorted(out)


def label_rarity(labels_per_photo: Iterable[Iterable[str]]) -> Dict[str, float]:
    """{標籤: 0..1}——出現在越多張參考照片的標籤（價格標籤、貨架）越接近 0，只出現一兩張的接近 1。"""
    photos = [set(ls) for ls in labels_per_photo]
    n = len(photos)
    df = Counter(l for ls in photos for l in ls)
    if n < 2:
        return {l: 1.0 for l in df}
    return {l: math.log((n + 1) / (d + 1)) / math.log(n + 1) for l, d in df.items()}


def relation_score(
    query: Sequence[Tuple[str, Box]],
    ref: Sequence[Tuple[str, Box]],
    similarity: Callable[[str, str], float],
    *,
    rarity: Optional[Dict[str, float]] = None,
    min_sim: float = 0.55,
    w_vertical: float = 1.0,
    w_horizontal: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """0..1：查詢照片與一張參考照片的物件上下左右排列有多一致。

    query / ref：[(標籤, bbox)]。名稱相似度 ≥ min_sim 的一對一配對（相似度高的先配），
    配到的每兩對物件、兩軸各比一次：兩邊都有關係且相同 → 一致，相反 → 矛盾。
    權重＝兩個配對的相似度 × 兩個參考物件的稀有度 × 那一軸的權重。
    分數＝(一致 − 矛盾) / (一致 + 矛盾 + smooth)，最低 0；smooth 讓只有一兩組可比的照片拿不到高分。
    """
    q = [(l, b) for l, b in query if l and _valid(b)]
    r = [(l, b) for l, b in ref if l and _valid(b)]
    if len(q) < 2 or len(r) < 2:
        return 0.0
    cand = []
    for i, (ql, _) in enumerate(q):
        for j, (rl, _) in enumerate(r):
            s = similarity(ql, rl)
            if s >= min_sim:
                cand.append((s, i, j))
    cand.sort(reverse=True)
    used_q, used_r, matched = set(), set(), []
    for s, i, j in cand:
        if i in used_q or j in used_r:
            continue
        used_q.add(i)
        used_r.add(j)
        w = s * ((rarity or {}).get(r[j][0], 1.0))
        matched.append((i, j, w))
    if len(matched) < 2:
        return 0.0
    agree = disagree = 0.0
    for x in range(len(matched)):
        i1, j1, w1 = matched[x]
        for y in range(x + 1, len(matched)):
            i2, j2, w2 = matched[y]
            qr = pair_relation(q[i1][1], q[i2][1])
            rr = pair_relation(r[j1][1], r[j2][1])
            w = w1 * w2
            for qa, ra, wa in ((qr.v, rr.v, w_vertical), (qr.h, rr.h, w_horizontal)):
                if qa and ra and wa:
                    if qa == ra:
                        agree += w * wa
                    else:
                        disagree += w * wa
    return max(0.0, (agree - disagree) / (agree + disagree + smooth))
