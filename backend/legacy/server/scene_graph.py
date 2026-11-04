"""scene_graph.py — 物件鄰接圖（scene graph）的共用建構邏輯。

匯入建圖端（import_neo4j_to_topomap）與定位查詢端（eval / locate）必須用
**同一套**鄰接邊建構，graph matching（UniGoal 式的 S_E / S_T）才公平。
把這段邏輯集中在這裡，兩邊共用。

一組物件（同一張照片、同一方向）→ kNN 稀疏鄰接圖，每條邊帶：
  rel_type  : left/right/above/below（由中心相對位置，影像座標 y 向下）
  depth_rel : nearer/farther/same（由 depth_class）
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

_COL_CENTER = {"left": 0.17, "center": 0.5, "right": 0.83}
_ROW_CENTER = {"top": 0.17, "mid": 0.5, "bottom": 0.83}
_REL_INV = {"right": "left", "left": "right", "above": "below", "below": "above"}
_DEPTH_RANK = {"near": 0, "mid": 1, "far": 2}
_DEPTH_INV = {"nearer": "farther", "farther": "nearer", "same": "same", "": ""}

DEFAULT_K = 3


def center(box: Optional[List[float]], grid_cell: str = "") -> Tuple[float, float]:
    """物件中心 (x, y)。有 bbox 用 bbox 中心；否則退回九宮格欄/列近似中心。"""
    if box and len(box) == 4:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    parts = (grid_cell or "mid-center").split("-")
    row = parts[0] if len(parts) == 2 else "mid"
    col = parts[1] if len(parts) == 2 else "center"
    return (_COL_CENTER.get(col, 0.5), _ROW_CENTER.get(row, 0.5))


def rel_type(ca: Tuple[float, float], cb: Tuple[float, float]) -> str:
    """b 相對 a 的主方位（取水平/垂直位移較大的那軸）。"""
    dx, dy = cb[0] - ca[0], cb[1] - ca[1]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "below" if dy > 0 else "above"


def rel_inv(rel: str) -> str:
    return _REL_INV.get(rel, rel)


def depth_rel(da: str, db: str) -> str:
    """b 相對 a 的深度關係。"""
    if da not in _DEPTH_RANK or db not in _DEPTH_RANK:
        return ""
    d = _DEPTH_RANK[db] - _DEPTH_RANK[da]
    return "farther" if d > 0 else ("nearer" if d < 0 else "same")


def depth_inv(rel: str) -> str:
    return _DEPTH_INV.get(rel, rel)


def build_adjacency(objects: List[dict], k: int = DEFAULT_K) -> List[dict]:
    """對一組物件（同照片同方向）建 kNN 鄰接邊。

    Args:
        objects: [{"id":.., "box":[..]|None, "grid_cell":.., "depth_class":..}, ...]
        k: 每個物件連最近的 k 個鄰居。

    Returns:
        無向邊清單（a<b 去重）：
        [{"a":id, "b":id, "rel_type":str, "depth_rel":str,
          "rel_type_inv":str, "depth_rel_inv":str}, ...]
        rel_type 是「b 相對 a」，rel_type_inv 是「a 相對 b」。
    """
    n = len(objects)
    if n < 2:
        return []
    centers = [center(o.get("box"), o.get("grid_cell", "")) for o in objects]

    pairs = set()
    for i in range(n):
        ci = centers[i]
        order = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: (centers[j][0] - ci[0]) ** 2 + (centers[j][1] - ci[1]) ** 2,
        )
        for j in order[:k]:
            pairs.add((min(i, j), max(i, j)))

    edges = []
    for i, j in pairs:
        ra = rel_type(centers[i], centers[j])
        da = depth_rel(objects[i].get("depth_class", ""), objects[j].get("depth_class", ""))
        edges.append({
            "a": objects[i]["id"], "b": objects[j]["id"],
            "rel_type": ra, "depth_rel": da,
            "rel_type_inv": rel_inv(ra), "depth_rel_inv": depth_inv(da),
        })
    return edges
