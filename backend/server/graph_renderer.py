"""繪製「目標圖」與「累積場景圖」的 PNG 圖片。

本檔案同時包含偵測結果的後處理流程：NMS 去重、相鄰同類框合併、
相鄰相似標籤合併、跨步驟實體追蹤、空間關係推論、OCR 與門牌關聯。

這份實作參考了 navigate_one_by_one.py（單張照片逐步導航腳本）的繪圖邏輯，
並將其中較成熟的部分（跨步實體精準比對、觀察點方框動態高度、
物件節點三欄式排版、狀態表情符號標記）移植過來，
同時保留 APPNAV 後端（server.py）實際使用的資料結構：
    - observation dict 使用 "vlm_summary"（單一描述文字），而非
      navigate_one_by_one.py 的 "interactions"（多輪使用者問答）。
    - detection dict 使用 "status" 欄位（"arrived" / "false_positive" /
      "wrong_instance"）標記使用者確認結果，而非 observation 層級的
      "arrived_here" / "wrong_instance" 旗標。
"""
from __future__ import annotations

import contextlib
import io as _io
import re
from collections import Counter, defaultdict
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# ── 中文字型輔助（跨平台：Windows / macOS）──────────────────────────────

def _cjk():
    """尋找系統中可用的中文字型，避免圖片中的中文顯示成方框。"""
    from matplotlib.font_manager import FontProperties, fontManager
    available = {f.name for f in fontManager.ttflist}
    for name in [
        "Microsoft JhengHei", "Microsoft YaHei",   # Windows
        "Heiti TC", "LiHei Pro", "Hiragino Sans GB",
        "Hiragino Sans CNS", "Hiragino Sans",       # macOS
        "Arial Unicode MS",
    ]:
        if name in available:
            return FontProperties(family=name)
    return FontProperties()


# ── 已知目標 → 偵測類別對照表 ───────────────────────────────────────────
# 使用者輸入的目標（例如「電腦」）不一定與 GroundingDINO/OCR 認得的詞彙一致，
# 這裡把常見目標展開成同義詞、英文詞、以及可能出現在門牌/標示上的關鍵字。

GOAL_CLASS_MAP: dict[str, dict] = {
    "電腦": {
        "classes":   ["電腦", "桌上型電腦", "筆記型電腦", "螢幕", "鍵盤", "滑鼠",
                      "computer", "laptop", "desktop computer", "monitor",
                      "keyboard", "mouse", "screen"],
        "ocr_parts": ["電腦", "桌機", "筆電", "本機", "computer", "PC", "desktop", "laptop"],
    },
    "冰箱": {
        "classes":   ["冰箱", "refrigerator", "fridge", "冷藏"],
        "ocr_parts": ["冰箱", "refrigerator", "fridge"],
    },
    "飲水機": {
        "classes":   ["飲水機", "water dispenser", "water cooler", "water fountain"],
        "ocr_parts": ["飲水機", "water dispenser"],
    },
    "印表機": {
        "classes":   ["印表機", "printer", "laser printer"],
        "ocr_parts": ["印表機", "printer"],
    },
    "辦公室": {
        "classes":   ["辦公室", "教師辦公室", "office", "nameplate", "name plate",
                      "door sign", "plaque", "office door"],
        "ocr_parts": [],
    },
}


def augment_goal_classes(goal: str, goal_objects: list[str]) -> list[str]:
    """用 GOAL_CLASS_MAP 展開 goal_objects，讓偵測 prompt 涵蓋同義詞。"""
    extra: list[str] = []
    for key, info in GOAL_CLASS_MAP.items():
        if key in goal:
            extra.extend(info["classes"])
    merged = list(dict.fromkeys(goal_objects + extra))
    return merged


def get_goal_ocr_parts(goal: str) -> list[str]:
    """回傳用來比對 OCR 文字是否符合目標的關鍵字（含目標原文）。"""
    parts: list[str] = [goal]
    for key, info in GOAL_CLASS_MAP.items():
        if key in goal:
            parts.extend(info["ocr_parts"])
    return list(dict.fromkeys(parts))


# ── 門牌/標示 ID 抽取 ────────────────────────────────────────────────────

_DOOR_LABELS = {"door", "office door"}
_NAMEPLATE_LIKE_LABELS = _DOOR_LABELS | {"sign", "plaque", "nameplate", "name plate"}


def _nameplate_id(nameplate_text: str) -> str:
    """從 OCR 門牌文字中取出有意義的 ID（優先抓連續中文字，如姓名+職稱）。

    例："陳仁暉 Ph.D. 助理教授" → "陳仁暉助理教授"
    """
    segments = re.findall(r'[一-鿿]{2,}', nameplate_text)
    if segments:
        return "".join(segments[:2])[:8]
    # 找不到中文字時，退而求其次去除英數符號雜訊
    cleaned = re.sub(r'[A-Za-z0-9\s\.%"\'_,\-]+', '', nameplate_text).strip()
    return cleaned[:8]


def assign_detection_ids(detections: list[dict]) -> None:
    """為每個偵測結果加上唯一的 'id' 欄位（就地修改）。

    規則（依優先順序）：
      1. 門/標示類物件若有 OCR 門牌文字 → 用門牌文字當 id（例如 "陳仁暉教授"）。
         若該 id 已被使用（同一張照片有兩個門牌抓到相同文字片段），
         則加上 _1, _2 後綴避免 id 衝突。
      2. 同一 label 只出現一次 → 直接用 label 當 id。
      3. 同一 label 出現多次且沒有門牌文字 → 加上 _1, _2 後綴。
    """
    label_count = Counter(d["label"] for d in detections)
    label_idx: dict[str, int] = {}
    used_ids: set[str] = set()

    for d in detections:
        lbl = d["label"]
        nameplate = d.get("nameplate_text", "")

        if nameplate and any(kw in lbl.lower() for kw in _NAMEPLATE_LIKE_LABELS):
            base_id = _nameplate_id(nameplate) or lbl
            if base_id and base_id not in used_ids:
                d["id"] = base_id
                used_ids.add(base_id)
                continue
            suffix = 1
            while f"{base_id}_{suffix}" in used_ids:
                suffix += 1
            d["id"] = f"{base_id}_{suffix}"
            used_ids.add(d["id"])
            continue

        if label_count[lbl] == 1:
            d["id"] = lbl
        else:
            label_idx[lbl] = label_idx.get(lbl, 0) + 1
            d["id"] = f"{lbl}_{label_idx[lbl]}"
        used_ids.add(d["id"])


# ── NMS（去除重疊框）──────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0


def apply_nms(detections: list[dict], iou_thr: float = 0.50) -> list[dict]:
    """依信心分數排序後，去除與高分框重疊過多（IoU > 閾值）的低分框。"""
    dets = sorted(detections, key=lambda d: -d["score"])
    keep, suppressed = [], set()
    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(d)
        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            if _iou(d["box"], dets[j]["box"]) > iou_thr:
                suppressed.add(j)
    return keep


# ── 合併相鄰的同標籤框 ───────────────────────────────────────────────────

def _boxes_nearby(a: list, b: list, gap_ratio: float = 0.6) -> bool:
    """兩框是否重疊，或間距相對框身尺寸而言足夠小（視為同一物件被切成兩塊）。"""
    h_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    v_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    min_side = min(a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1])
    thr = gap_ratio * min_side
    return h_gap <= thr and v_gap <= thr


def _merge_cluster(detections: list[dict], cluster: list[int]) -> dict:
    boxes = [detections[i]["box"] for i in cluster]
    merged_box = [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]
    best = max(cluster, key=lambda i: detections[i]["score"])
    merged_det = dict(detections[best])
    merged_det["box"] = [round(x, 1) for x in merged_box]
    return merged_det


def merge_adjacent_same_label(detections: list[dict]) -> list[dict]:
    """合併「標籤完全相同」且框相鄰的偵測（例如同一張桌子被切成兩個框）。"""
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(detections):
        by_label[d["label"]].append(i)

    result = []
    for label, idxs in by_label.items():
        clusters: list[list[int]] = []
        for idx in idxs:
            merged_into = None
            for cluster in clusters:
                if any(_boxes_nearby(detections[idx]["box"], detections[c]["box"]) for c in cluster):
                    cluster.append(idx)
                    merged_into = cluster
                    break
            if merged_into is None:
                clusters.append([idx])

        for cluster in clusters:
            if len(cluster) == 1:
                result.append(detections[cluster[0]])
            else:
                result.append(_merge_cluster(detections, cluster))

    result.sort(key=lambda d: -d["score"])
    return result


def _label_matches(a: str, b: str) -> bool:
    """兩個標籤是否代表同一類物件（處理如 'chair sofa' 這種複合標籤）。"""
    if a == b:
        return True
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    return bool(a_words & b_words)


def merge_adjacent_similar_label(detections: list[dict]) -> list[dict]:
    """合併「標籤相似（有共同詞彙）」且框相鄰的偵測。

    例如 GroundingDINO 可能對同一個櫃檯同時輸出 "counter reception desk"
    和 "reception desk" 兩個框，這裡用 union-find 把它們合併成一個。
    """
    n = len(detections)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if (_label_matches(detections[i]["label"], detections[j]["label"])
                    and _boxes_nearby(detections[i]["box"], detections[j]["box"])):
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    result = []
    for idxs in groups.values():
        if len(idxs) == 1:
            result.append(detections[idxs[0]])
        else:
            result.append(_merge_cluster(detections, idxs))

    result.sort(key=lambda d: -d["score"])
    return result


# ── 跨步驟實體追蹤 ───────────────────────────────────────────────────────

def _norm_center(box: list, img_w: int, img_h: int) -> tuple[float, float]:
    return ((box[0] + box[2]) / (2 * img_w),
            (box[1] + box[3]) / (2 * img_h))


def _norm_iou(a: list, b: list, wa: int, ha: int, wb: int, hb: int) -> float:
    """兩張不同尺寸照片的框，先各自正規化到 [0,1] 座標系再算 IoU。"""
    a_n = [a[0]/wa, a[1]/ha, a[2]/wa, a[3]/ha]
    b_n = [b[0]/wb, b[1]/hb, b[2]/wb, b[3]/hb]
    ix1, iy1 = max(a_n[0], b_n[0]), max(a_n[1], b_n[1])
    ix2, iy2 = min(a_n[2], b_n[2]), min(a_n[3], b_n[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a_n[2]-a_n[0])*(a_n[3]-a_n[1]) + (b_n[2]-b_n[0])*(b_n[3]-b_n[1]) - inter
    return inter / ua if ua else 0.0


def tag_same_entity(detections: list[dict], img_w: int, img_h: int,
                    prev_dets: list[dict], prev_w: int, prev_h: int,
                    pos_thresh: float = 0.25) -> None:
    """標記 same_entity=True，代表這個偵測「跟上一步是同一個物件」。

    比對規則：標籤相符，且（正規化中心點距離 < pos_thresh 或 正規化 IoU > 0.05）。
    採用一對一比對（每個上一步的偵測只能被配對一次），並取距離最近者，
    避免同一個上一步物件被多個本步物件重複認領。

    同時記錄 same_entity_prev_uid = 上一步偵測的 'id'，讓場景圖能精準畫出
    跨步驟連線，而不必再靠標籤比對去猜是哪一個節點（標籤相同但有多個實例時，
    單靠標籤比對容易連錯對象）。
    """
    used_prev: set[int] = set()
    for det in detections:
        nc = _norm_center(det["box"], img_w, img_h)
        best_idx, best_pd, best_dist = None, None, float("inf")
        for pi, pd in enumerate(prev_dets):
            if pi in used_prev:
                continue
            if not _label_matches(det["label"], pd["label"]):
                continue
            pnc = _norm_center(pd["box"], prev_w, prev_h)
            dx = abs(nc[0] - pnc[0])
            dy = abs(nc[1] - pnc[1])
            center_ok = dx < pos_thresh and dy < pos_thresh
            iou_ok = _norm_iou(det["box"], pd["box"], img_w, img_h, prev_w, prev_h) > 0.05
            if center_ok or iou_ok:
                dist = (dx**2 + dy**2) ** 0.5
                if dist < best_dist:
                    best_dist, best_idx, best_pd = dist, pi, pd
        if best_pd is not None:
            used_prev.add(best_idx)
            det["same_entity"] = True
            det["same_entity_prev_uid"] = best_pd.get("id", best_pd["label"])


# ── 空間關係推論（例如「在桌子上」）─────────────────────────────────────

_SURFACE_KEYWORDS = {"table", "desk", "cabinet", "counter", "shelf",
                     "sofa", "chair", "glass table", "windowsill"}


def infer_spatial_context(detections: list[dict]) -> None:
    """為每個偵測加上 'context' 欄位，描述它放在哪個平面上（例如「在cabinet上」）。"""
    for det in detections:
        det.setdefault("context", "")
        lbl_lower = det["label"].lower()
        if any(s in lbl_lower for s in _SURFACE_KEYWORDS):
            continue  # 平面本身不需要 context
        x1a, y1a, x2a, y2a = det["box"]
        cx_a = (x1a + x2a) / 2
        best_surface, best_overlap = None, 0.0
        for other in detections:
            if other is det:
                continue
            if not any(s in other["label"].lower() for s in _SURFACE_KEYWORDS):
                continue
            x1b, y1b, x2b, y2b = other["box"]
            # 物件的水平中心要落在平面範圍內，且底部要貼齊或略高於平面頂部
            if not (x1b < cx_a < x2b):
                continue
            if not (y1b <= y2a <= y2b + (y2b - y1b) * 0.35):
                continue
            overlap = min(x2a, x2b) - max(x1a, x1b)
            if overlap > best_overlap:
                best_overlap = overlap
                best_surface = other["label"]
        if best_surface:
            det["context"] = f"在{best_surface}上"


# ── OCR 文字 ↔ 門的空間關聯 ──────────────────────────────────────────────

def associate_ocr_to_doors(detections: list[dict],
                           ocr_with_bbox: list[dict]) -> None:
    """把靠近門框的 OCR 文字（門牌）關聯到對應的門偵測上。

    ocr_with_bbox: [{"text": str, "confidence": float,
                      "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]

    排除規則（避免誤把門上的海報/公告當成門牌）：
      - OCR 框與門框重疊超過 50% → 視為貼在門上的東西，不是門牌，跳過。
      - OCR 框寬度 > 門寬 50% 或高度 > 門高 30% → 太大，不像門牌。
    """
    for det in detections:
        det.setdefault("nameplate_text", "")
        det.setdefault("nameplate_conf", 0.0)
        if not any(kw in det["label"].lower() for kw in _DOOR_LABELS):
            continue
        dx1, dy1, dx2, dy2 = det["box"]
        door_h = dy2 - dy1
        door_w = dx2 - dx1

        texts, best_conf = [], 0.0
        for ocr in ocr_with_bbox:
            bbox_pts = ocr["bbox"]
            ox1 = min(p[0] for p in bbox_pts)
            oy1 = min(p[1] for p in bbox_pts)
            ox2 = max(p[0] for p in bbox_pts)
            oy2 = max(p[1] for p in bbox_pts)
            ocr_w = ox2 - ox1
            ocr_h = oy2 - oy1
            ocr_cy = (oy1 + oy2) / 2
            conf = ocr["confidence"]

            overlap_x = max(0.0, min(ox2, dx2) - max(ox1, dx1))
            overlap_y = max(0.0, min(oy2, dy2) - max(oy1, dy1))
            overlap_ratio = (overlap_x * overlap_y) / max(ocr_w * ocr_h, 1)
            if overlap_ratio > 0.50:
                continue
            if ocr_w > door_w * 0.50 or ocr_h > door_h * 0.30:
                continue

            # 門牌通常在門框上半部，且水平方向緊貼門框兩側
            v_ok = dy1 <= ocr_cy <= dy1 + door_h * 0.75
            h_gap = max(0.0, max(ox1 - dx2, dx1 - ox2))
            if h_gap < door_w * 1.5 and v_ok and conf >= 0.35:
                texts.append(ocr["text"])
                best_conf = max(best_conf, conf)

        if texts:
            det["nameplate_text"] = " ".join(texts)
            det["nameplate_conf"] = best_conf


# ── 完整後處理流程 ───────────────────────────────────────────────────────

def postprocess_detections(
    det_dicts: list[dict],
    img_w: int,
    img_h: int,
    ocr_with_bbox: Optional[list[dict]] = None,
    prev_dets: Optional[list[dict]] = None,
    prev_w: int = 0,
    prev_h: int = 0,
    skip_nms: bool = False,
) -> list[dict]:
    """依序執行：NMS → 同標籤合併 → 相似標籤合併 → OCR-門關聯 →
    空間關係推論 → 跨步實體追蹤 → ID 指派。

    注意：跨步實體追蹤（tag_same_entity）需要用到「上一步」偵測結果的
    'id' 欄位，而上一步的資料是在上一次呼叫本函式時就已經指派好 id，
    所以這裡的呼叫順序沒有問題（本步驟的 id 要到最後才指派）。

    ``skip_nms``：VLM-only 模式（PERCEPTION_ENABLED=0）下的 bbox 是 VLM
    自己粗略估計的區域，不是 GroundingDINO 的精確框——NMS/合併這幾步都
    假設框很準確、靠 IoU 重疊比例判斷「是不是同一個物件」，用在粗略框上
    容易誤刪、誤合併成不同的物件，所以這個模式下全部跳過，只做跟框準確度
    無關的 OCR 關聯/空間推論/跨步追蹤。
    """
    if not det_dicts:
        return det_dicts

    if not skip_nms:
        det_dicts = apply_nms(det_dicts)
        det_dicts = merge_adjacent_same_label(det_dicts)
        det_dicts = merge_adjacent_similar_label(det_dicts)

    if ocr_with_bbox:
        associate_ocr_to_doors(det_dicts, ocr_with_bbox)

    infer_spatial_context(det_dicts)

    if prev_dets and prev_w > 0 and prev_h > 0:
        tag_same_entity(det_dicts, img_w, img_h, prev_dets, prev_w, prev_h)

    assign_detection_ids(det_dicts)
    return det_dicts


# ── 目標分類輔助 ─────────────────────────────────────────────────────────

def _is_goal(label: str, goal_objects: List[str]) -> bool:
    label_l = label.lower()
    for g in goal_objects:
        if g.lower() in label_l or label_l in g.lower():
            return True
    return False


# ── 偵測節點顏色與狀態標記（6 種狀態）───────────────────────────────────
# 優先順序：使用者已確認的狀態（到達/誤判/同種非目標）最優先顯示，
# 其次是「目標候選」（尚待確認，即使它跨步被追蹤也優先顯示為紅色候選，
# 而不是被同一實體的藍色蓋掉），最後才是同一實體追蹤與一般地標。

def _det_color_and_tag(det: dict, goal_objects: List[str]) -> tuple[str, str, str, str]:
    """回傳 (facecolor, edgecolor, textcolor, text_tag)。

    text_tag 使用純文字標記（不用 emoji），避免在部分字型/環境下
    無法渲染而顯示成缺字方框。
    """
    status = det.get("status", "")
    if status == "arrived":
        return "#d7ccc8", "#6d4c41", "#3e2723", "[到達]"      # 棕：正式找到
    if status == "false_positive":
        return "#eeeeee", "#757575", "#424242", "[誤判]"      # 灰：誤判
    if status == "wrong_instance":
        return "#e8d5f5", "#7b1fa2", "#4a0072", "[非目標]"    # 紫：同種非目標
    if _is_goal(det.get("label", ""), goal_objects):
        return "#f8d7da", "#dc3545", "#721c24", "[候選]"      # 紅：目標候選
    if det.get("same_entity"):
        return "#cce5ff", "#004085", "#003060", "[追蹤]"      # 藍：同一實體
    return "#d4edda", "#28a745", "#155724", ""                 # 綠：一般地標


# ── 目標圖（3 節點：地點 → 目標 → 目標物）───────────────────────────────

def render_goal_graph(
    goal: str,
    goal_objects: List[str],
    dst_path: str,
    place: str = "室內",
) -> None:
    """畫出簡單的三節點目標圖：地點 -(located_in)-> 目標 -(target_of)-> 目標物。"""
    fp = _cjk()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.axis("off")

    goal_text = f"尋找{goal}"
    ax.set_title(f"目標圖 Goal Graph\n目標文字：{goal_text}",
                 fontsize=13, fontweight="bold", color="#222",
                 fontproperties=fp, pad=10)

    def node(cx, cy, lines, type_lbl, fc, tc="#333", ec="#aaa"):
        box = FancyBboxPatch((cx - 1.1, cy - 0.48), 2.2, 0.96,
                             boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        for i, ln in enumerate(lines):
            ax.text(cx, cy + 0.12 - i * 0.26, ln,
                    ha="center", va="center", fontsize=10,
                    color=tc, fontproperties=fp, zorder=3)
        ax.text(cx, cy - 0.35, f"(type: {type_lbl})",
                ha="center", fontsize=7.5, color="#888", zorder=3)

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
        if lbl:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, lbl,
                    fontsize=8, color="#666", ha="left", va="center")

    lns = [goal_text[:10], goal_text[10:]] if len(goal_text) > 10 else [goal_text]
    node(4.5, 2.5, lns, "goal", "#d0e4f7", "#1a4a7a", "#5b9bd5")
    node(4.5, 4.2, [place], "place", "#d4edda", "#155724", "#28a745")
    target_label = goal if len(goal) <= 10 else goal[:9] + "…"
    node(7.5, 1.2, [target_label], "target", "#f8d7da", "#721c24", "#dc3545")

    arrow(4.5, 3.0, 4.5, 3.75, "located_in")
    arrow(5.3, 2.1, 6.4, 1.5, "target_of")

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        plt.tight_layout()
        plt.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ── 場景圖（累積式、多步驟）──────────────────────────────────────────────
# 版面配置說明：
#   - 每一列（row）代表一個觀察點（照片），高度依「觀察方框需要的文字高度」
#     與「該步偵測到的物件數量」動態決定，避免文字被切掉或版面過度擁擠。
#   - 物件節點依三欄排列：
#       x ≈ 5.0  → 歷史步驟的物件（灰暗、較小）
#       x ≈ 8.0  → 目前步驟「新出現」的物件
#       x ≈ 12.5 → 目前步驟「跨步追蹤到的同一實體」
#     三欄式排版是為了不讓「新物件」與「同一實體物件」擠在同一欄造成混亂
#     （這點是從 navigate_one_by_one.py 移植過來的改進，舊版只用二欄）。

def _wrap_text(text: str, line_w: int, max_lines: int = 8) -> list[str]:
    """把文字換行到每行約 line_w 個字，超過 max_lines 行則截斷加刪節號。"""
    text = text.replace("\n", " ").strip()
    lines: list[str] = []
    while text and len(lines) < max_lines:
        if len(text) <= line_w:
            lines.append(text)
            text = ""
            break
        pos = text.rfind(" ", 0, line_w)
        if pos < line_w // 3:
            pos = line_w
        lines.append(text[:pos])
        text = text[pos:].lstrip()
    if text and lines:
        lines[-1] = lines[-1].rstrip() + "…"
    return lines or [""]


def _obs_box_h(vlm_summary: str, ocr_hit: bool, is_current: bool) -> float:
    """計算觀察點方框需要的高度，讓 VLM 描述文字能完整顯示而不被裁切。"""
    HEADER_H = 0.62   # 標題「觀察點 N」+ 照片檔名
    LINE_H = 0.20
    PAD = 0.15
    line_w = 22 if is_current else 18
    max_lines = 8 if is_current else 4
    clean = vlm_summary.replace("\n", " ").strip()
    n_lines = len(_wrap_text(clean, line_w, max_lines)) if clean else 0
    extra = 0.20 if ocr_hit else 0.0
    return max(1.54, HEADER_H + extra + n_lines * LINE_H + PAD)


def _row_h_for(n_dets: int, is_current: bool, obs_box_h: float) -> float:
    """列高需同時滿足：容納觀察方框的高度、容納所有物件節點不重疊。"""
    if is_current:
        NODE_H, GAP = 0.58, 0.24
    else:
        NODE_H, GAP = 0.44, 0.20
    min_for_obs = max(2.20, 2 * obs_box_h - 1.54 + 0.50)
    return max(min_for_obs, n_dets * (NODE_H + GAP) + 0.70)


def render_scene_graph(
    observations: List[dict],
    goal: str,
    goal_objects: List[str],
    dst_path: str,
    corrections: Optional[List[str]] = None,
) -> None:
    """畫出累積場景圖：由上到下依序排列每個觀察點，右側接出該步偵測到的物件節點。

    observations 每筆需含：
        step, photo, detections, vlm_summary, ocr_texts, false_positive(可選)
    corrections（可選）：使用者對「已到達」判斷的修正紀錄，若有提供則顯示於標題。
    """
    fp = _cjk()
    n_obs = len(observations)
    if n_obs == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "尚無觀察資料", ha="center", va="center",
                fontsize=14, fontproperties=fp)
        ax.axis("off")
        fig.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return

    current_step = observations[-1].get("step", n_obs)
    ocr_goal_parts = get_goal_ocr_parts(goal)

    # 舊資料可能沒有 'id' 欄位（例如尚未跑過完整後處理），補上預設值避免出錯
    for obs in observations:
        for d in obs.get("detections", []):
            if "id" not in d:
                d["id"] = d["label"]

    def _ocr_hit(obs: dict) -> bool:
        texts = obs.get("ocr_texts", [])
        return any(part.lower() in t.lower() for t in texts for part in ocr_goal_parts)

    # 依觀察方框所需高度，動態計算每一列的列高
    obs_box_heights = [
        _obs_box_h(obs.get("vlm_summary", ""), _ocr_hit(obs), obs.get("step", 0) == current_step)
        for obs in observations
    ]
    row_heights = [
        _row_h_for(len(obs.get("detections", [])),
                   obs.get("step", 0) == current_step, obs_box_heights[i])
        for i, obs in enumerate(observations)
    ]
    total_h = sum(row_heights)
    fig_h = max(6, total_h + 2.5)

    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    corr_note = f"  注意：{corrections[-1]}" if corrections else ""
    ax.set_title(
        f"場景圖 Scene Graph — 累積至步驟 {current_step}{corr_note}\n"
        f"([到達]=正式找到  [候選]=目標候選  [非目標]=同種非目標  [誤判]=誤判  [追蹤]=同一實體  綠=地標)",
        fontsize=11, fontweight="bold", color="#222",
        fontproperties=fp, pad=8)

    obs_x = 1.8

    # 由上而下計算每列中心 y 座標；y_cursor 代表「目前列的頂部」
    obs_y: dict[int, float] = {}
    y_cursor = fig_h - 1.3
    for i, obs in enumerate(observations):
        rh_i = row_heights[i]
        s = obs.get("step", i + 1)
        obs_y[s] = y_cursor - rh_i / 2
        y_cursor -= rh_i

    node_data: list[dict] = []

    for i, obs in enumerate(observations):
        s = obs.get("step", i + 1)
        y = obs_y[s]
        rh = row_heights[i]
        is_current = (s == current_step)
        photo = obs.get("photo", "")
        vlm_summary = obs.get("vlm_summary", "")
        dets = obs.get("detections", [])
        is_false_pos = obs.get("false_positive", False)
        ocr_hit = _ocr_hit(obs)
        box_h = obs_box_heights[i]

        fc = "#ffeeba" if is_false_pos else ("#d0e4f7" if is_current else "#e8f0f8")
        ec = "#ffc107" if is_false_pos else ("#5b9bd5" if is_current else "#a0b8cc")
        lw = 2.0 if is_current else (2.0 if is_false_pos else 1.2)

        # 觀察方框：頂端固定在 y+0.77，向下延伸 box_h（動態高度）
        box_top = y + 0.77
        box_bot = box_top - box_h
        box_center_y = box_top - box_h / 2

        box = FancyBboxPatch((obs_x - 1.5, box_bot), 3.0, box_h,
                             boxstyle="round,pad=0.08",
                             facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(box)

        label_txt = f"觀察點 {s}" + (" [誤判]" if is_false_pos else "")
        ax.text(obs_x, box_top - 0.20, label_txt, ha="center", va="center",
                fontsize=12, fontproperties=fp,
                color="#1a4a7a" if is_current else "#4a6a8a",
                fontweight="bold" if is_current else "normal", zorder=4)
        ax.text(obs_x, box_top - 0.38, photo,
                ha="center", va="center", fontsize=8.5, color="#777", zorder=4)

        text_cursor = box_top - 0.55
        if ocr_hit:
            ax.text(obs_x, text_cursor, f"[OCR 命中] {goal}",
                    ha="center", va="center", fontsize=8.5,
                    color="#155724", fontproperties=fp, zorder=4)
            text_cursor -= 0.20

        if vlm_summary:
            line_w = 22 if is_current else 18
            max_lines = 8 if is_current else 4
            for line in _wrap_text(vlm_summary, line_w, max_lines):
                ax.text(obs_x, text_cursor, line,
                        ha="center", va="center", fontsize=7,
                        color="#444", fontproperties=fp, zorder=4)
                text_cursor -= 0.20

        if i < n_obs - 1:
            ns = observations[i + 1].get("step", i + 2)
            ny = obs_y[ns]
            ax.annotate("", xy=(obs_x, ny + 0.77),
                        xytext=(obs_x, box_bot),
                        arrowprops=dict(arrowstyle="->", color="#7a9fc0", lw=1.3))

        if not dets:
            continue

        n = len(dets)
        spacing = rh / (n + 1)
        fs = 11.0 if is_current else 8.0
        MIN_H = 0.52 if is_current else 0.38
        node_h = max(MIN_H, min(MIN_H * 1.2, spacing - 0.14))
        char_w_est = 0.140 if is_current else 0.098

        for j, det in enumerate(dets):
            lbl = det.get("label", "")
            uid = det.get("id", lbl)
            score = det.get("score", 0)
            context = det.get("context", "")
            nameplate = det.get("nameplate_text", "")
            same_entity = det.get("same_entity", False)
            dy = y + rh / 2 - spacing * (j + 1)

            # 三欄式排版：歷史步驟固定 x=5.0；目前步驟依「是否為跨步同一實體」
            # 分成「新物件」(x=8.0) 與「同一實體」(x=12.5) 兩欄，避免擠在一起。
            if is_current:
                ox = 12.5 if same_entity else 8.0
            else:
                ox = 5.0

            if nameplate and any(kw in lbl.lower() for kw in _DOOR_LABELS):
                np_short = nameplate[:10]
                display_lbl = f"{uid}:{np_short}"
            elif context:
                ctx_s = context if len(context) <= 9 else context[:8] + "…"
                display_lbl = f"{uid} ({ctx_s})"
            else:
                display_lbl = uid
            _mc = 26 if is_current else 17
            if len(display_lbl) > _mc:
                display_lbl = display_lbl[:_mc - 1] + "…"

            node_w = min(
                6.0 if is_current else 3.4,
                max(4.2 if is_current else 2.3,
                    len(display_lbl) * char_w_est + 0.55)
            )

            fc2, ec2, tc, tag = _det_color_and_tag(det, goal_objects)

            nb = FancyBboxPatch((ox - node_w / 2, dy - node_h / 2), node_w, node_h,
                                boxstyle="round,pad=0.05",
                                facecolor=fc2, edgecolor=ec2,
                                linewidth=1.4 if is_current else 0.8,
                                alpha=1.0 if is_current else 0.6, zorder=2)
            ax.add_patch(nb)

            label_y = dy + node_h * 0.18
            conf_y = dy - node_h * 0.22
            tagged_lbl = f"{tag} {display_lbl}" if tag else display_lbl
            ax.text(ox, label_y, tagged_lbl,
                    ha="center", va="center", fontsize=fs,
                    color=tc, fontproperties=fp, zorder=3)
            ax.text(ox, conf_y, f"conf: {score:.3f}",
                    ha="center", va="center", fontsize=fs - 1.5,
                    color="#999", zorder=3)

            edge_color = "#5b9bd5" if is_current else "#ccc"
            edge_lw = 0.9 if is_current else 0.4
            ax.annotate("", xy=(ox - node_w / 2, dy),
                        xytext=(obs_x + 1.5, box_center_y),
                        arrowprops=dict(arrowstyle="->", color=edge_color, lw=edge_lw))

            node_data.append({"label": lbl, "uid": uid, "step": s, "x": ox, "y": dy,
                              "node_w": node_w, "context": context})

    # ── 跨步同一實體連線（依 label 分色 + 扇形展開避免重疊）───────────
    # 優先用 same_entity_prev_uid 精準比對上一步的節點；若舊資料沒有這個
    # 欄位（尚未跑過新版 tag_same_entity），才退回用標籤比對。
    _ARC_COLOURS = ["#004085", "#7b1fa2", "#1a7a40", "#b34700", "#005f73", "#8b0000"]
    _label_idx: dict[str, int] = {}

    for i, obs in enumerate(observations[1:], 1):
        prev_step_n = observations[i - 1].get("step", i)
        curr_step_n = obs.get("step", i + 1)
        for det in obs.get("detections", []):
            if not det.get("same_entity"):
                continue
            det_uid = det.get("id", det["label"])
            prev_uid = det.get("same_entity_prev_uid")

            if prev_uid:
                prev_n = next((n for n in node_data
                               if n["step"] == prev_step_n and n["uid"] == prev_uid), None)
            else:
                prev_n = next((n for n in node_data
                               if n["step"] == prev_step_n
                               and _label_matches(n["label"], det["label"])), None)
            curr_n = next((n for n in node_data
                           if n["step"] == curr_step_n and n["uid"] == det_uid), None)
            if not (prev_n and curr_n):
                continue

            lbl_key = det["label"]
            if lbl_key not in _label_idx:
                _label_idx[lbl_key] = len(_label_idx)
            idx = _label_idx[lbl_key]
            colour = _ARC_COLOURS[idx % len(_ARC_COLOURS)]

            sx = prev_n["x"] + prev_n["node_w"] / 2
            sy = prev_n["y"]
            ex = curr_n["x"] - curr_n["node_w"] / 2
            ey = curr_n["y"]

            same_col = abs(prev_n["x"] - curr_n["x"]) < 1.0
            if same_col:
                rad = -(0.30 + idx * 0.18)
            else:
                rad = -(0.08 + idx * 0.04)

            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.2,
                                        linestyle="dashed", alpha=0.75,
                                        connectionstyle=f"arc3,rad={rad:.2f}"),
                        zorder=1)

    # ── 空間關係連線「在…上」──────────────────────────────────────────
    curr_obs_obj = next((o for o in observations
                         if o.get("step") == current_step), None)
    if curr_obs_obj:
        for det in curr_obs_obj.get("detections", []):
            ctx = det.get("context", "")
            if not ctx:
                continue
            surface_lbl = ctx.replace("在", "").replace("上", "").strip()
            curr_n = next((n for n in node_data
                           if n["step"] == current_step
                           and n["label"] == det["label"]), None)
            surf_n = next((n for n in node_data
                           if n["step"] == current_step
                           and surface_lbl in n["label"].lower()), None)
            if curr_n and surf_n:
                ax.annotate("",
                    xy=(surf_n["x"] + surf_n["node_w"] / 2, surf_n["y"]),
                    xytext=(curr_n["x"] + curr_n["node_w"] / 2, curr_n["y"]),
                    arrowprops=dict(arrowstyle="->", color="#1a7a40", lw=0.8,
                                    linestyle="dotted", alpha=0.65,
                                    connectionstyle="arc3,rad=0.4"),
                    zorder=1)
                mid_x = max(curr_n["x"], surf_n["x"]) + curr_n["node_w"] / 2 + 0.3
                mid_y = (curr_n["y"] + surf_n["y"]) / 2
                ax.text(mid_x, mid_y, "在…上", fontsize=7,
                        color="#1a7a40", ha="left", va="center",
                        fontproperties=fp, zorder=2)

    legend_items = [
        mpatches.Patch(facecolor="#d7ccc8", edgecolor="#6d4c41", label="[到達] 正式找到（導航結束）"),
        mpatches.Patch(facecolor="#f8d7da", edgecolor="#dc3545", label="[候選] 目標候選（待確認）"),
        mpatches.Patch(facecolor="#e8d5f5", edgecolor="#7b1fa2", label="[非目標] 同種非目標（用戶確認）"),
        mpatches.Patch(facecolor="#eeeeee", edgecolor="#757575", label="[誤判] 誤判（與目標完全無關）"),
        mpatches.Patch(facecolor="#cce5ff", edgecolor="#004085", label="[追蹤] 同一實體（跨步追蹤）"),
        mpatches.Patch(facecolor="#d4edda", edgecolor="#28a745", label="地標物件"),
        Line2D([0], [0], color="#004085", lw=1.2, linestyle="dashed",
               label="- - 跨步同一實體追蹤"),
        Line2D([0], [0], color="#1a7a40", lw=1.0, linestyle="dotted",
               label="... 空間關係（在…上）"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=10,
              facecolor="white", edgecolor="#ccc", prop=fp)

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        plt.tight_layout()
        plt.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
