"""
topomap_v2.py — 拓樸地圖 v2（支援子圖結構的場所地圖）
Topological Map v2 — place-level map with sub-graph structure

════════════════════════════════════════════════════════════════════════════
背景 (Background)
════════════════════════════════════════════════════════════════════════════
這是「新增拓樸地圖功能」筆記中所描述的資料結構實作。與現有的
`topomap.py`（單一節點型態、單一邊型態，只服務單次導航的 VLM 提示文字）不同，
這個模組是「場所等級」的地圖，設計目標是：

    1. 建圖或導航結束後，後端自動執行（不是使用者操作的功能）
    2. 同一個場所（例如「資工系系辦」「萬家福量販 新店店」）的地圖可以
       跨多次導航持續更新、重複使用
    3. 節點分為兩種、邊分為四大類（其中第一種邊細分成 9 種子型態），
       並以「一張照片 + 其辨識到的物件」構成一個子圖 (sub-graph)

────────────────────────────────────────────────────────────────────────────
節點型態 (Node Types)
────────────────────────────────────────────────────────────────────────────
1. 照片節點 PHOTO   — 拍照的位置，存照片路徑；同時是子圖的根節點
2. 物件節點 OBJECT  — 某張照片中辨識到的物件，新增「作用」屬性
                       （例如：商品、標示牌、環境／結構物）

另外還有「使用者目前位置」：不是圖上的節點型態，而是一個指向 PHOTO 節點的
指標（初始為 None），會沿著第四種邊（走道）移動，並記錄移動路徑。

────────────────────────────────────────────────────────────────────────────
邊型態 (Edge Types)
────────────────────────────────────────────────────────────────────────────
1. CONTAINS  （子圖內部邊）photo → object，細分成 9 種子型態
              （etype 是 "contains_top_left"、"contains_center"…等九選一，
              見 GRID_CELLS／`contains_etype()`），依物件的偵測框中心落在
              照片九宮格（左上、中上、右上、左、中、右、左下、中下、右下）
              的哪一格決定，同時記錄距離（有感測器就用感測器，沒有則用
              程式估計，見 estimate_distance_m()）
2. ADJACENT  （相鄰物件邊）object ↔ object（同一張照片內，左至右排序後的
              相鄰物件），記錄相鄰關係的文字說明
3. SAME_OBJECT（跨照片同物件邊）object ↔ object（不同照片），代表兩個
              物件節點是同一個實體，用來建立不同照片間的相對位置關係
4. WALKWAY   （走道邊）photo → photo，**單向有向邊**：方向同時代表「拍攝
              起點那張照片時的拍照方向」。路徑最後一張照片沒有下一步可以
              承載這個方向，改用指向自己的自我迴圈邊表示，見
              `set_terminal_facing()`。

────────────────────────────────────────────────────────────────────────────
已知待解問題（沿用筆記內容，尚未完整解決，這裡先給出局部因應方案）
────────────────────────────────────────────────────────────────────────────
- 區域呈現：目前用 `region` 屬性標記照片節點所屬區域（例如「乳製品區」），
  不是完整的區域子圖，只能算是暫時方案。
- 沒走的路 → 用 `add_virtual_branch()` 建立「虛假照片節點」
  (is_virtual=True，沒有照片與距離資訊)，未來補拍到真正照片時用
  `resolve_virtual_branch()` 無條件更新成真的照片節點。
- 前後照片沒有共同物體 → 目前 `link_same_objects_between_photos()`
  找不到配對時就不會生成第三種邊，尚無解法，呼叫端可自行注意
  回傳的配對數量是否為 0。
- 貨架上物體被視為整體、使用者走出已建地圖範圍 → 尚未處理，留待後續。
"""
from __future__ import annotations

import colorsys
import io
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import networkx as nx  # noqa: E402


def _configure_cjk_font() -> None:
    """讓 render_png() 畫出來的中文字不要變成方框。

    matplotlib 預設字型不含中文字形，這裡嘗試尋找系統上常見的中文字型
    （Noto Sans CJK / 文泉驛等），找不到就靜默略過，圖片仍會產生，
    只是中文字可能顯示不完整（不影響資料本身的正確性）。
    """
    import matplotlib.font_manager as fm
    candidates = [
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans TC",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "PingFang TC",
        "Microsoft JhengHei", "SimHei", "Heiti TC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


_configure_cjk_font()

try:
    # 沿用既有的標籤同義詞正規化（例如 "fridge" → "refrigerator"），
    # 讓跨照片物件關聯與角色判斷更準確。若無法載入（例如獨立測試環境
    # 缺少完整套件），退回一個極簡版本，維持本模組可獨立執行。
    from server.config import normalize_label  # type: ignore
except Exception:  # pragma: no cover - 獨立執行時的保底
    def normalize_label(label: str) -> str:
        return (label or "").strip().lower()


# ══════════════════════════════════════════════════════════════════════════
# 常數 / Constants
# ══════════════════════════════════════════════════════════════════════════

# 節點型態 / Node types
NTYPE_PHOTO = "photo"
NTYPE_OBJECT = "object"

# 邊型態 / Edge types（對應筆記的第一～四種邊）
# 第一種邊（CONTAINS）現在細分成 9 種子型態，見下面的 GRID_CELLS／
# contains_etype()／is_contains_etype()，這裡不再有單一的 ETYPE_CONTAINS。
ETYPE_ADJACENT = "adjacent"        # 第二種邊：object ↔ object（同照片）
ETYPE_SAME_OBJECT = "same_object"  # 第三種邊：object ↔ object（跨照片）
ETYPE_WALKWAY = "walkway"          # 第四種邊：photo → photo（單向，方向＝拍照方向）

# 物件「作用」分類關鍵字（雙語）。命中即視為標示牌，其餘預設為商品。
# 可依場所自行擴充（例如賣場 vs 系辦）。
ROLE_SIGN = "標示牌"
ROLE_PRODUCT = "商品"
ROLE_STRUCTURE = "環境"  # 牆、天花板、地板等背景結構物

_SIGN_KEYWORDS = {
    "exit", "entrance", "checkout", "restroom", "toilet", "sign", "signage",
    "notice", "label", "nameplate", "doorplate",
    "出口", "入口", "收銀", "結帳", "洗手間", "廁所", "指示", "標示", "招牌", "門牌",
}
_STRUCTURE_KEYWORDS = {
    "wall", "ceiling", "floor", "door", "window",
    "牆", "天花板", "地板", "門", "窗",
}

# 距離估計（沒有感測器資料時的保底方案）
# 用單眼相機常見的「面積比例反平方根」近似公式：
#     distance ≈ calib / sqrt(area_ratio)
# calib 為經驗校正常數（公尺），需依實際手機鏡頭/場景校正，這裡先給一個
# 保守預設值。真正部署時應優先使用手機感測器（如 ARCore/ARKit 的深度、
# 或是簡單的陀螺儀+步數估計），這裡只是「真的沒有感測器」時的備案。
_DEFAULT_DISTANCE_CALIB = 1.4
_MIN_DISTANCE_M = 0.3
_MAX_DISTANCE_M = 15.0


def estimate_distance_m(
    box: List[float], img_w: int, img_h: int,
    calib_constant: float = _DEFAULT_DISTANCE_CALIB,
) -> float:
    """沒有手機感測器資料時，用偵測框大小粗估拍攝點到物件的距離（公尺）。

    這是暫時的保底公式，不是精確量測。物件在畫面中佔比越大代表越近。
    之後如果拿得到手機感測器資料（例如 ARCore 的深度、或加速度計推算的
    移動距離），應直接改用感測器數值，並把 contains 邊上的
    `distance_source` 標記為 "sensor" 而不是 "estimated"。
    """
    if not box or len(box) != 4 or not img_w or not img_h:
        return _MIN_DISTANCE_M
    x1, y1, x2, y2 = box
    area_ratio = max(((x2 - x1) * (y2 - y1)) / max(img_w * img_h, 1), 1e-6)
    dist = calib_constant / math.sqrt(area_ratio)
    return round(min(max(dist, _MIN_DISTANCE_M), _MAX_DISTANCE_M), 2)


def classify_role(label: str, has_nameplate: bool = False) -> str:
    """依標籤文字（與是否帶有門牌/標示文字）判斷物件節點的「作用」屬性。"""
    norm = normalize_label(label)
    if has_nameplate:
        return ROLE_SIGN
    if any(kw in norm for kw in _SIGN_KEYWORDS):
        return ROLE_SIGN
    if any(kw in norm for kw in _STRUCTURE_KEYWORDS):
        return ROLE_STRUCTURE
    return ROLE_PRODUCT


def _bbox_center_x(box: List[float]) -> float:
    if not box or len(box) != 4:
        return 0.0
    return (box[0] + box[2]) / 2.0


def _ocr_bbox_center(ocr_item: dict) -> Optional[tuple]:
    """算一筆 OCR 結果的中心點座標，相容兩種常見格式：四點多邊形
    [[x,y],[x,y],[x,y],[x,y]]（VLM/EasyOCR 常見格式），或簡單矩形
    [x1,y1,x2,y2]。抓不到座標就回傳 None。
    """
    bbox = ocr_item.get("bbox")
    if not bbox:
        return None
    try:
        if isinstance(bbox[0], (list, tuple)):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return sum(xs) / len(xs), sum(ys) / len(ys)
        if len(bbox) == 4:
            return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    except (TypeError, IndexError, ZeroDivisionError):
        pass
    return None


def _associate_ocr_with_detections(detections: List[dict], ocr_items: List[dict]) -> List[str]:
    """把 `ocr_items`（獨立的 OCR 結果清單）關聯到最近的物件，寫進該
    物件字典的 "nameplate_text" 欄位（原地修改 `detections`）。

    這一步是必要的：下面 `build_subgraph_from_detections()` 讀取物件
    的 OCR 文字，只看 `det.get("nameplate_text") or det.get("ocr_text")`
    ——也就是說物件字典本身必須先帶著這個欄位，光是把 `ocr_items` 傳
    進函式參數並不會自動幫忙關聯（這個參數過去一直是「接了但沒用」的
    狀態）。物件本身如果已經帶著 "nameplate_text"／"ocr_text"（例如
    某些 VLM 輸出格式本來就會直接把文字寫在物件自己身上），不會被
    覆蓋；每筆 OCR 只會關聯給距離最近的一個物件，避免同一段文字被
    好幾個物件借用，稀釋掉判斷力。

    距離「打平手」時（VLM 模式常見：好幾個不同物件共用同一個粗略模板
    框，中心點完全一樣），不是隨便選排最前面那個，而是優先選標籤本身
    比較像是「會承載文字」的物件（招牌、告示牌類），其次才照原本順序
    ——同樣是打平手，一段文字關聯到「sign」會比關聯到「door」更合理。

    Returns:
        「查無所屬」的文字清單——3 個以上物件打平手時（見下方說明），
        這段文字不會被硬塞給任何一個物件，但也不是完全沒有價值，回傳
        給呼叫端（`build_subgraph_from_detections()`）存到照片節點本身
        的 `ambient_ocr_texts` 欄位，供之後可能的「照片對照片，比對有
        沒有讀到同一句背景文字」這類用途使用（目前定位比對邏輯
        `locate_v2.py` 還沒有使用這個欄位，純粹先把資料留著）。
    """
    if not ocr_items or not detections:
        return []

    text_bearing_keywords = (
        "sign", "board", "plaque", "notice", "menu", "poster",
        "directional", "招牌", "標示", "告示",
    )

    det_centers = []
    for d in detections:
        box = d.get("box")
        if box and len(box) == 4:
            det_centers.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
        else:
            det_centers.append(None)

    unassigned_texts: List[str] = []

    for ocr_item in ocr_items:
        text = str(ocr_item.get("text", "")).strip()
        if not text:
            continue
        ocr_center = _ocr_bbox_center(ocr_item)
        if ocr_center is None:
            continue

        # 先找出「距離最近」的最小距離值，再從所有打平手的候選裡，
        # 依「標籤是否像招牌／告示牌」優先排序，選出最終要關聯的物件。
        best_dist = None
        for center in det_centers:
            if center is None:
                continue
            dist = (center[0] - ocr_center[0]) ** 2 + (center[1] - ocr_center[1]) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
        if best_dist is None:
            unassigned_texts.append(text)
            continue

        tied_indices = [
            i for i, center in enumerate(det_centers)
            if center is not None
            and (center[0] - ocr_center[0]) ** 2 + (center[1] - ocr_center[1]) ** 2 == best_dist
        ]

        # 3 個以上不同物件打平手（VLM 模式常見：一大段裝飾性布條/海報
        # 文字，跟好幾個不相干的物件剛好共用同一個粗略模板框），這種
        # 情況通常代表這段文字是牆面裝飾、不是專門描述某一個特定物件
        # 的門牌/標示牌，硬要猜一個容易猜錯（實測踩到：「InnoServe」
        # 得獎感謝布條的文字，被誤植到跟它毫不相干的「bucket」身上，
        # 導致後續定位比對出現「trophies↔bucket」這種標籤對不上、卻
        # 因為 OCR 文字剛好對上而配對成功的異常結果）。這種情況乾脆
        # 不猜、直接跳過，比硬猜一個然後猜錯安全，改記到「查無所屬」
        # 清單裡。只有 2 個物件打平手時（例如一扇門跟旁邊的標示牌），
        # 才用「優先選招牌類標籤」這條規則去判斷。
        if len(tied_indices) >= 3:
            unassigned_texts.append(text)
            continue

        best_idx = min(
            tied_indices,
            key=lambda i: (
                0 if any(kw in detections[i].get("label", "").lower() for kw in text_bearing_keywords) else 1,
                i,
            ),
        )

        det = detections[best_idx]
        if not (det.get("nameplate_text") or det.get("ocr_text")):
            det["nameplate_text"] = text
        else:
            # 這個物件已經有文字了(例如同一輪次別的 OCR 項目先佔用)，
            # 這筆就算沒地方去，一樣記到「查無所屬」清單，不要憑空消失。
            unassigned_texts.append(text)

    return unassigned_texts


def _bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """計算兩個 [x1,y1,x2,y2] 框的 IoU，用於同張照片內物件去重／關聯。"""
    if not box_a or not box_b:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _third(frac: float) -> int:
    """把 0~1 的比例分成三等分（0=左/上, 1=中, 2=右/下）。"""
    if frac < 1 / 3:
        return 0
    if frac < 2 / 3:
        return 1
    return 2


_H_LABELS = ("left", "center", "right")
_V_LABELS = ("top", "middle", "bottom")


def position_bucket(box: List[float], img_w: int, img_h: int) -> str:
    """依偵測框中心點，算出物件在畫面中的粗略位置（水平,垂直）。

    現有系統實際輸出的 detections_*.json 裡「position」欄位常常是空字串
    （只有兩階段 VLM 模式才會填），所以不能只依賴那個欄位判斷跨照片物件
    是否為同一實體。這裡改成不管有沒有拿到 position 欄位，都直接用
    bbox + 照片寬高自己算一份，確保 `link_same_objects_between_photos()`
    永遠有一個可用的空間線索，而不是退化成「純標籤比對」。
    """
    if not box or len(box) != 4 or not img_w or not img_h:
        return ""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = _H_LABELS[_third(cx / img_w)]
    v = _V_LABELS[_third(cy / img_h)]
    return f"{h},{v}"


# ── 九宮格（第一種邊的 9 種子型態）─────────────────────────────────
# 把照片切成 3×3 的九宮格，第一種邊（photo → object）依物件的偵測框中心
# 落在哪一格，分成 9 種子型態，型態字串格式固定為 "contains_<cell>"。
GRID_CELLS: Tuple[str, ...] = (
    "top_left", "top_center", "top_right",
    "left", "center", "right",
    "bottom_left", "bottom_center", "bottom_right",
)

# 給人看／給文件用的中文對照
GRID_CELL_ZH: Dict[str, str] = {
    "top_left": "左上", "top_center": "中上", "top_right": "右上",
    "left": "左", "center": "中", "right": "右",
    "bottom_left": "左下", "bottom_center": "中下", "bottom_right": "右下",
}

CONTAINS_ETYPE_PREFIX = "contains_"


def contains_etype(cell: str) -> str:
    """把九宮格代號轉成第一種邊的 etype 字串，例如 "top_left" → "contains_top_left"。
    給不認得的代號會退回 "center"，確保一定回傳合法的九種之一。"""
    if cell not in GRID_CELLS:
        cell = "center"
    return f"{CONTAINS_ETYPE_PREFIX}{cell}"


def is_contains_etype(etype: str) -> bool:
    """判斷某個 etype 字串是不是第一種邊（九種子型態的任何一種）。"""
    return isinstance(etype, str) and etype.startswith(CONTAINS_ETYPE_PREFIX)


def grid_cell_from_box(box: List[float], img_w: int, img_h: int) -> str:
    """依偵測框中心點，算出物件落在照片九宮格的哪一格（見 GRID_CELLS）。
    算不出來（沒有框或沒有照片寬高）時預設回傳 "center"。
    """
    if not box or len(box) != 4 or not img_w or not img_h:
        return "center"
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    h = _H_LABELS[_third(cx / img_w)]
    v = _V_LABELS[_third(cy / img_h)]
    return h if v == "middle" else f"{v}_{h}"


def grid_row_col(cell: str) -> Tuple[int, int]:
    """把九宮格代號拆成 (row, col)，row/col 都是 0/1/2：
    row：0=上、1=中、2=下；col：0=左、1=中、2=右。
    畫拓樸地圖時，物件節點會依這個 (row, col) 排位置，讓視覺上的排列
    盡量貼近物件原本在照片九宮格裡的位置，不是隨便排。
    """
    if cell.startswith("top_"):
        row, col_name = 0, cell[len("top_"):]
    elif cell.startswith("bottom_"):
        row, col_name = 2, cell[len("bottom_"):]
    else:
        row, col_name = 1, cell  # "left" / "center" / "right"（中間那排，沒有前綴）
    col = {"left": 0, "center": 1, "right": 2}.get(col_name, 1)
    return row, col


# ── 九宮格代號 → 顏色（給畫圖用，讓「物件在照片裡的哪個位置」一眼看得出）──
# 設計：col（左/中/右）決定色相（藍／紫／橘，色相環上刻意分開避免混淧），
# row（上/中/下）決定飽和度＋明度（上排淺、下排深）——也就是使用者說的
# 「顏色深淺分九種」：同一欄的上中下用同一色相、深淺不同一眼就能分出遠近，
# 不同欄則靠色相分開，兩個維度疊在一起剛好對應九宮格的兩個軸。
# 額外的巧合：照片裡「上排」通常是畫面較遠處、「下排」較靠近鏡頭，淺→深
# 的漸層順便也呼應了「越靠近底部（下排）越深」的直覺。
_GRID_COL_HUE = (0.58, 0.80, 0.04)     # 左＝藍、中＝紫、右＝橘紅
_GRID_ROW_SAT = (0.35, 0.55, 0.78)     # 上排飽和度低（淺）→下排飽和度高（深）
_GRID_ROW_VAL = (0.93, 0.82, 0.62)     # 上排明度高（亮）→下排明度低（暗）


def grid_cell_color(cell: str) -> str:
    """依九宮格代號算出一個十六進位顏色，色相對應左右、深淺對應上下。

    給 `render_png()` 畫 contains 邊、物件節點外框用，讓使用者不用查文字
    屬性，光看顏色就能大致判斷這個物件在原始照片裡偏左/中/右、偏上/中/下。
    """
    row, col = grid_row_col(cell)
    r, g, b = colorsys.hsv_to_rgb(_GRID_COL_HUE[col], _GRID_ROW_SAT[row], _GRID_ROW_VAL[row])
    return "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))


def _slugify(text: str) -> str:
    """把場所名稱（可能含中文/空白/斜線）轉成安全的資料夾名稱。"""
    text = unicodedata.normalize("NFKC", text or "unknown_place").strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unknown_place"


# ══════════════════════════════════════════════════════════════════════════
# 主要類別 / Main class
# ══════════════════════════════════════════════════════════════════════════

class TopoGraphV2:
    """場所等級的拓樸地圖，內部用 networkx.MultiDiGraph 儲存。

    使用 MultiDiGraph 是因為同一對節點之間可能同時存在不同型態的邊
    （雖然目前設計下同一對節點通常只會有一種邊型態，但保留彈性）。
    每個節點/邊都有 `ntype` / `etype` 屬性標記型態，方便序列化與繪圖。
    """

    def __init__(self, place_name: str = "") -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._next_id: int = 0
        self.place_name: str = place_name
        # 使用者目前位置：指向某個 PHOTO 節點的 id（尚未開始導航時為 None）
        self.current_position: Optional[int] = None
        # 使用者移動路徑（依序記錄走過的 PHOTO 節點 id）
        self.position_history: List[int] = []
        self.created_at: str = datetime.utcnow().isoformat()
        self.updated_at: str = self.created_at

    # ── 內部工具 ─────────────────────────────────────────────────────
    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()

    # ────────────────────────────────────────────────────────────────
    # 節點新增 / Add nodes
    # ────────────────────────────────────────────────────────────────
    def add_photo_node(
        self,
        photo_path: str,
        *,
        timestamp: Optional[str] = None,
        sensor_data: Optional[dict] = None,
        region: Optional[str] = None,
        is_virtual: bool = False,
        note: str = "",
        ambient_ocr_texts: Optional[List[str]] = None,
        photos: Optional[Dict[str, str]] = None,
    ) -> int:
        """新增照片節點（子圖的根節點）。

        Args:
            photo_path: 主照片路徑（通常是 front，向下相容）。
            sensor_data: 手機感測器原始資料（若有）。
            region: 所屬區域標籤。
            is_virtual: 是否為虛假照片節點。
            ambient_ocr_texts: 查無所屬的 OCR 文字清單。
            photos: 四方向照片路徑，例如
                {"front": "photos/wp_1_front_xxx.jpg",
                 "right": "photos/wp_1_right_xxx.jpg", ...}。
                若提供，photo_path 仍保留為主照片（向下相容）。
        """
        nid = self._new_id()
        self.graph.add_node(
            nid,
            ntype=NTYPE_PHOTO,
            photo_path=photo_path,
            photo_file=Path(photo_path).name if photo_path else "",
            timestamp=timestamp or datetime.utcnow().isoformat(),
            sensor_data=sensor_data or {},
            region=region,
            is_virtual=is_virtual,
            note=note,
            ambient_ocr_texts=list(ambient_ocr_texts) if ambient_ocr_texts else [],
            photos=dict(photos) if photos else {},
        )
        self._touch()
        return nid

    def add_object_node(
        self,
        photo_id: int,
        label: str,
        box: Optional[List[float]] = None,
        *,
        score: float = 0.0,
        position: str = "",
        ocr_text: str = "",
        role: Optional[str] = None,
        crop_path: str = "",
        grid_cell: Optional[str] = None,
        direction: str = "",
    ) -> int:
        """新增物件節點，並自動建立第一種邊（photo → object）。

        role 若未指定，會依 classify_role() 自動判斷（商品 / 標示牌 / 環境）。
        grid_cell 是這個物件落在照片九宮格的哪一格（見 GRID_CELLS）。
        direction: 這個物件來自哪個方向的照片（front/right/back/left），
            搭配四方向照片節點使用。空字串表示未指定或單方向照片。
        """
        if photo_id not in self.graph or self.graph.nodes[photo_id].get("ntype") != NTYPE_PHOTO:
            raise ValueError(f"photo_id {photo_id} 不是有效的照片節點")

        resolved_role = role or classify_role(label, has_nameplate=bool(ocr_text))
        oid = self._new_id()
        self.graph.add_node(
            oid,
            ntype=NTYPE_OBJECT,
            label=label,
            label_norm=normalize_label(label),
            box=list(box) if box else [],
            score=score,
            position=position,
            ocr_text=ocr_text,
            role=resolved_role,
            photo_id=photo_id,
            grid_cell=grid_cell or "center",
            direction=direction,
        )
        # 自動建立第一種邊（photo → object）。這裡還不知道照片寬高，所以先用
        # 保底距離；`build_subgraph_from_detections()` 會在算出正確的估計值後
        # 用同一個 edge key 覆蓋掉這條邊的距離與九宮格型態。
        self.add_contains_edge(photo_id, oid, grid_cell=grid_cell or "center")
        self._touch()
        return oid

    # ────────────────────────────────────────────────────────────────
    # 邊新增 / Add edges
    # ────────────────────────────────────────────────────────────────
    def add_contains_edge(
        self, photo_id: int, object_id: int, *,
        distance_m: Optional[float] = None, distance_source: str = "estimated",
        grid_cell: str = "center",
    ) -> None:
        """第一種邊：photo → object，記錄距離。

        現在分成九種子型態（見模組開頭 GRID_CELLS／`contains_etype()`），
        依物件落在照片九宮格的哪一格決定 etype 是 "contains_top_left"、
        "contains_center"…等九選一。

        distance_source: "sensor"（來自手機感測器）或 "estimated"（用
        estimate_distance_m() 粗估）。若呼叫時沒給 distance_m，會自動用
        物件框大小估計。
        """
        if distance_m is None:
            photo = self.graph.nodes[photo_id]
            obj = self.graph.nodes[object_id]
            sensor = photo.get("sensor_data") or {}
            if "ranging_m" in sensor:
                distance_m = float(sensor["ranging_m"])
                distance_source = "sensor"
            else:
                # 沒有 img size 資訊時無法用面積估計，退回保底值
                distance_m = _MIN_DISTANCE_M
                distance_source = "estimated"
            obj_box = obj.get("box")
            if obj_box:
                # img_w/img_h 由呼叫端在 build_subgraph_from_detections 內
                # 直接算好距離傳進來；這裡是防呆備援。
                distance_m = distance_m
        self.graph.add_edge(
            photo_id, object_id, key=f"contains_{object_id}",
            etype=contains_etype(grid_cell), grid_cell=grid_cell,
            distance_m=distance_m, distance_source=distance_source,
        )
        self._touch()

    def add_adjacent_edge(self, object_a: int, object_b: int, relation: str = "") -> None:
        """第二種邊：object ↔ object（同一張照片內的相鄰物件）。

        雙向各建立一條邊，relation 文字可由 relation_fn 產生（見
        build_subgraph_from_detections），也可以留空之後再由 LLM 補上。
        """
        self.graph.add_edge(
            object_a, object_b, key=f"adjacent_{object_a}_{object_b}",
            etype=ETYPE_ADJACENT, relation=relation,
        )
        self.graph.add_edge(
            object_b, object_a, key=f"adjacent_{object_b}_{object_a}",
            etype=ETYPE_ADJACENT, relation=relation,
        )
        self._touch()

    def add_same_object_edge(self, object_a: int, object_b: int, confidence: float = 1.0,
                              method: str = "label+position") -> None:
        """第三種邊：object ↔ object（跨照片、代表同一個實體）。"""
        self.graph.add_edge(
            object_a, object_b, key=f"same_{object_a}_{object_b}",
            etype=ETYPE_SAME_OBJECT, confidence=confidence, method=method,
        )
        self.graph.add_edge(
            object_b, object_a, key=f"same_{object_b}_{object_a}",
            etype=ETYPE_SAME_OBJECT, confidence=confidence, method=method,
        )
        self._touch()

    def add_walkway_edge(
        self, photo_a: int, photo_b: int, *,
        direction: str = "", distance_m: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        """第四種邊：photo → photo（走道），單向有向邊。

        邊的方向（從 photo_a 指向 photo_b）同時代表**拍攝 photo_a 那張
        照片時的拍照方向**：使用者在 photo_a 的位置面向這個方向拍照，
        然後照這個方向走到 photo_b。

        以前這裡會自動雙向各存一條（方便查詢鄰居），但拍照方向本來就是
        「單一方向」的概念，雙向反而會混淆「這條邊代表哪張照片的方向」，
        所以改成只存一條有向邊。圖上找鄰居（例如死算推位時的 BFS）改成
        用 `_walkway_undirected()` 內部另外建一張無向圖來查，不需要靠
        資料本身雙向存。

        最後一張照片（路徑終點）沒有下一步可以承載它的拍照方向，需要另外
        呼叫 `set_terminal_facing()` 建一條自我迴圈的邊來表示，見該方法
        說明。
        """
        edge_data = dict(etype=ETYPE_WALKWAY, direction=direction, distance_m=distance_m)
        if steps is not None:
            edge_data["steps"] = steps
        self.graph.add_edge(
            photo_a, photo_b, key=f"walkway_{photo_a}_{photo_b}",
            **edge_data,
        )
        self._touch()

    def _clear_terminal_facing(self) -> None:
        """移除所有『代表拍照方向』的自我迴圈邊（如果有的話）。"""
        to_remove = [
            (u, v, k) for u, v, k, d in self.graph.edges(keys=True, data=True)
            if u == v and d.get("etype") == ETYPE_WALKWAY
        ]
        for u, v, k in to_remove:
            self.graph.remove_edge(u, v, key=k)

    def set_terminal_facing(self, photo_id: int, direction: str = "") -> None:
        """把某個照片節點標記為目前路徑的終點，用一條指向自己的走道邊
        （自我迴圈）代表這張照片本身的拍攝方向。

        因為第四種邊的方向意義是「代表 photo_a 的拍照方向，同時指向下一
        個節點」，路徑最後一張照片沒有下一步可以承載這個方向，所以用
        自我迴圈表示「這裡是終點，面向這個方向拍的」。

        每次呼叫都會先清掉舊的終點標記（同一張地圖同一時間只會有一個
        終點），所以每次建圖／銜接新的一段路線之後，只要呼叫一次
        `set_terminal_facing(最新的終點)` 就好，不用自己先清舊的。

        direction 留空的話，代表「跟走到這裡的方向一樣，沒有再轉」
        （渲染時會用死算推位算出的朝向，不需要額外解析文字）。
        """
        self._assert_photo(photo_id)
        self._clear_terminal_facing()
        self.graph.add_edge(
            photo_id, photo_id, key=f"facing_{photo_id}",
            etype=ETYPE_WALKWAY, direction=direction, distance_m=0.0,
        )
        self._touch()

    # ────────────────────────────────────────────────────────────────
    # 高階建圖流程 / High-level build helpers
    # ────────────────────────────────────────────────────────────────
    def build_subgraph_from_detections(
        self,
        photo_id: int,
        detections: List[dict],
        img_w: int,
        img_h: int,
        *,
        ocr_items: Optional[List[dict]] = None,
        relation_fn: Optional[Callable[[dict, dict], str]] = None,
        distance_calib: float = _DEFAULT_DISTANCE_CALIB,
        direction: str = "",
    ) -> List[int]:
        """把一張照片的偵測結果，一次建成該照片的完整子圖：

            物件節點 + 第一種邊（contains）+ 第二種邊（adjacent，左至右排序）

        detections 格式沿用現有系統 `detections_{node_id}.json` 的
        "detections" 欄位：[{"label","score","box","position",
        "nameplate_text", ...}, ...]

        direction: 這張照片的拍攝方向（front/right/back/left），
            會存到每個物件節點的 direction 屬性。

        Returns:
            依左至右排序後的物件節點 id 清單（方便呼叫端做跨照片關聯）。
        """
        ocr_items = ocr_items or []
        # 把獨立的 OCR 結果關聯到最近的物件（見 _associate_ocr_with_detections()
        # 說明）——修正之前「ocr_items 接了但沒真的用來關聯」的問題。
        # 原地修改 detections 裡每個字典（補上 nameplate_text），不影響
        # 呼叫端傳進來的原始物件參照以外的任何東西。回傳值是「查無所屬」
        # 的文字（牆面裝飾之類，沒辦法安全歸屬給某個特定物件），存到
        # 這張照片節點本身的 ambient_ocr_texts 屬性，不要整個丟掉——
        # 目前比對邏輯還沒有使用這份資料，純粹先保留，見
        # `_associate_ocr_with_detections()` 開頭的說明。
        unassigned_ocr_texts = _associate_ocr_with_detections(detections, ocr_items)
        if photo_id in self.graph.nodes:
            existing = self.graph.nodes[photo_id].get("ambient_ocr_texts", [])
            self.graph.nodes[photo_id]["ambient_ocr_texts"] = existing + unassigned_ocr_texts

        # 依 bbox 中心 x 座標由左至右排序，符合筆記「最好照著左至右排」的要求
        sorted_dets = sorted(detections, key=lambda d: _bbox_center_x(d.get("box", [])))

        object_ids: List[int] = []
        for det in sorted_dets:
            label = det.get("label", "unknown")
            box = det.get("box", [])
            nameplate = det.get("nameplate_text", "") or det.get("ocr_text", "")
            # 現有系統的 detections_*.json 常常沒有 "position" 欄位
            # （GroundingDINO 模式下就是空的），這裡自己算一份保底，
            # 讓跨照片關聯有真正的空間線索可用，見 position_bucket() 說明。
            position = det.get("position") or position_bucket(box, img_w, img_h)
            cell = grid_cell_from_box(box, img_w, img_h)
            oid = self.add_object_node(
                photo_id, label, box,
                score=det.get("score", 0.0),
                position=position,
                ocr_text=nameplate,
                grid_cell=cell,
                direction=direction,
            )
            distance = estimate_distance_m(box, img_w, img_h, calib_constant=distance_calib)
            self.add_contains_edge(photo_id, oid, distance_m=distance, distance_source="estimated", grid_cell=cell)
            object_ids.append(oid)

        # 相鄰物件邊：只連接排序後的相鄰兩兩物件（不做全連接，避免圖太密）
        for a, b in zip(object_ids, object_ids[1:]):
            data_a = self.graph.nodes[a]
            data_b = self.graph.nodes[b]
            if relation_fn is not None:
                relation = relation_fn(data_a, data_b)
            else:
                relation = f"{data_a['label']} 在 {data_b['label']} 的左側"
            self.add_adjacent_edge(a, b, relation=relation)

        self._touch()
        return object_ids

    def link_same_objects_between_photos(
        self,
        photo_a_objects: List[int],
        photo_b_objects: List[int],
        *,
        min_score: float = 1.6,
        allow_structural: bool = False,
    ) -> List[Tuple[int, int, float]]:
        """在兩張（通常是相鄰的）照片之間，關聯「同一個物件」的節點對。

        關聯策略仿照 ConceptGraphs 的物件關聯做法（語意相似度 + 幾何/位置
        相似度加總後貪婪配對），但因為這裡沒有 3D 點雲，改用：

            score = 標籤是否相同(正規化後)             * 1.0
                  + 招牌/OCR 文字是否相同               * 1.0
                  + 畫面位置是否相近（水平+垂直都符合）  * 0.6
                  + 畫面位置只有一軸相近                 * 0.3

        大於等於 min_score 才視為同一物件，並各自只能配對一次
        （貪婪、由高分到低分依序配對）。

        ⚠️ 重要限制：現有系統輸出的偵測結果常常是「trash」「desk」
        「plant」「cabinet」這種在很多張照片裡都會重複出現的通用標籤，
        如果只看「標籤相同」就判定是同一實體，會產生大量錯誤配對
        （把不同位置的好幾個垃圾桶都連成同一個垃圾桶），畫出來的地圖
        會非常亂。所以這裡：
          1. 預設排除「環境／結構物」角色（門、牆、窗…，見 classify_role）
             的物件，因為這類物件幾乎每張照片都會出現，用標籤配對幾乎
             一定會誤判；真的需要關聯的話可以傳 allow_structural=True。
          2. min_score 預設拉高到 1.6，代表「只有標籤相同」（1.0分）
             不足以判定為同一物件，還需要至少一項空間位置佐證。
        這仍然是規則式的近似解法，不是像 ConceptGraphs 那樣用 CLIP
        語意向量 + 3D 幾何做嚴謹關聯，真的要做到更準確，需要之後補上
        視覺特徵比對。

        Returns:
            [(object_id_a, object_id_b, score), ...] 已建立 same_object 邊的配對。
        """
        candidates: List[Tuple[float, int, int]] = []
        for a in photo_a_objects:
            da = self.graph.nodes[a]
            if not allow_structural and da.get("role") == ROLE_STRUCTURE:
                continue
            for b in photo_b_objects:
                db = self.graph.nodes[b]
                if not allow_structural and db.get("role") == ROLE_STRUCTURE:
                    continue
                score = 0.0
                if da["label_norm"] and da["label_norm"] == db["label_norm"]:
                    score += 1.0
                if da.get("ocr_text") and da["ocr_text"] == db.get("ocr_text"):
                    score += 1.0
                pos_a, pos_b = da.get("position", ""), db.get("position", "")
                if pos_a and pos_b:
                    if pos_a == pos_b:
                        score += 0.6
                    elif pos_a.split(",")[0] == pos_b.split(",")[0] or \
                            pos_a.split(",")[-1] == pos_b.split(",")[-1]:
                        score += 0.3
                if score > 0:
                    candidates.append((score, a, b))

        candidates.sort(key=lambda t: t[0], reverse=True)
        used_a, used_b = set(), set()
        linked: List[Tuple[int, int, float]] = []
        for score, a, b in candidates:
            if a in used_a or b in used_b:
                continue
            if score < min_score:
                continue
            self.add_same_object_edge(a, b, confidence=min(score / 2.6, 1.0))
            used_a.add(a)
            used_b.add(b)
            linked.append((a, b, score))
        return linked

    # ────────────────────────────────────────────────────────────────
    # 虛假照片節點（沒走過的路）/ Virtual branch nodes
    # ────────────────────────────────────────────────────────────────
    def add_virtual_branch(
        self,
        from_photo_id: int,
        *,
        direction: str = "",
        note: str = "偵測到分支但尚未走訪",
        flank_object_ids: Optional[List[int]] = None,
    ) -> int:
        """建立一個「虛假照片節點」，代表照片中看到但使用者沒有走過的路。

        虛假節點沒有真正的照片與距離資訊；只用走道邊 (walkway) 連到
        觸發它的照片節點，並可選擇性地連到「走道兩旁的物件節點」
        （用 CONTAINS 邊但不記距離），方便之後定位。
        """
        vid = self.add_photo_node("", is_virtual=True, note=note)
        self.add_walkway_edge(from_photo_id, vid, direction=direction, distance_m=None)
        for oid in (flank_object_ids or []):
            self.graph.add_edge(
                vid, oid, key=f"contains_virtual_{oid}",
                etype=contains_etype("center"), grid_cell="center",
                distance_m=None, distance_source="none (virtual)",
            )
        self._touch()
        return vid

    def resolve_virtual_branch(self, virtual_photo_id: int, real_photo_path: str,
                                sensor_data: Optional[dict] = None) -> None:
        """未來真的拍到那條路的照片時，把虛假節點「無條件更新」成真節點。

        依筆記：「如果未來有獲得那個走道的照片和辨識資訊就無條件更新」。
        """
        node = self.graph.nodes[virtual_photo_id]
        if not node.get("is_virtual"):
            raise ValueError(f"節點 {virtual_photo_id} 不是虛假照片節點，不能用這個方法更新")
        node["photo_path"] = real_photo_path
        node["photo_file"] = Path(real_photo_path).name
        node["is_virtual"] = False
        node["note"] = ""
        node["timestamp"] = datetime.utcnow().isoformat()
        if sensor_data:
            node["sensor_data"] = sensor_data
        self._touch()

    # ────────────────────────────────────────────────────────────────
    # 使用者目前位置 / Current user position
    # ────────────────────────────────────────────────────────────────
    def set_start_position(self, photo_id: int) -> None:
        """設定導航起點（通常是本次導航第一張照片對應的節點）。"""
        self._assert_photo(photo_id)
        self.current_position = photo_id
        self.position_history = [photo_id]
        self._touch()

    def move_to(self, photo_id: int, *, require_walkway: bool = True) -> None:
        """使用者沿第四種邊移動到另一個照片節點，並記錄移動路徑。

        require_walkway=True 時，會檢查目前位置與目標之間是否真的存在
        走道邊，避免記錄出不合理的跳躍（例如定位錯誤造成的瞬移）。
        """
        self._assert_photo(photo_id)
        if require_walkway and self.current_position is not None:
            has_edge = self.graph.has_edge(self.current_position, photo_id)
            edge_types = [
                d.get("etype") for d in self.graph.get_edge_data(
                    self.current_position, photo_id, default={},
                ).values()
            ] if has_edge else []
            if ETYPE_WALKWAY not in edge_types:
                raise ValueError(
                    f"節點 {self.current_position} 與 {photo_id} 之間沒有走道邊，"
                    "無法直接移動（如果使用者真的走到地圖外，需要另外處理，"
                    "見模組開頭『已知待解問題』）。"
                )
        self.current_position = photo_id
        self.position_history.append(photo_id)
        self._touch()

    def _assert_photo(self, photo_id: int) -> None:
        if photo_id not in self.graph or self.graph.nodes[photo_id].get("ntype") != NTYPE_PHOTO:
            raise ValueError(f"{photo_id} 不是有效的照片節點")

    # ────────────────────────────────────────────────────────────────
    # 查詢輔助 / Query helpers
    # ────────────────────────────────────────────────────────────────
    def get_subgraph(self, photo_id: int) -> nx.MultiDiGraph:
        """回傳某照片節點的子圖（自己 + 其物件節點 + 兩者間的邊）。"""
        object_ids = [
            v for _, v, d in self.graph.out_edges(photo_id, data=True)
            if is_contains_etype(d.get("etype", ""))
        ]
        return self.graph.subgraph([photo_id] + object_ids).copy()

    def photo_objects(self, photo_id: int) -> List[dict]:
        """回傳某照片節點底下所有物件節點的資料（依 contains 邊，九種子型態都算）。"""
        return [
            {"id": v, **self.graph.nodes[v]}
            for _, v, d in self.graph.out_edges(photo_id, data=True)
            if is_contains_etype(d.get("etype", ""))
        ]

    def all_photo_nodes(self) -> List[int]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("ntype") == NTYPE_PHOTO]

    def all_object_nodes(self) -> List[int]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("ntype") == NTYPE_OBJECT]

    def summarize_for_vlm(self, current_photo_id: Optional[int] = None) -> str:
        """比照舊版 topomap.py 的介面，產生給 VLM 看的文字摘要（沿走道邊走）。"""
        photo_nodes = self.all_photo_nodes()
        if not photo_nodes:
            return "尚未建立任何位置。"
        current_photo_id = current_photo_id if current_photo_id is not None else self.current_position
        if current_photo_id is None:
            current_photo_id = photo_nodes[0]

        walkway_graph = nx.DiGraph()
        walkway_graph.add_nodes_from(photo_nodes)
        for u, v, d in self.graph.edges(data=True):
            if d.get("etype") == ETYPE_WALKWAY:
                walkway_graph.add_edge(u, v, **d)

        try:
            start = next(n for n in walkway_graph.nodes if walkway_graph.in_degree(n) == 0)
        except StopIteration:
            start = photo_nodes[0]

        try:
            path = nx.shortest_path(walkway_graph, source=start, target=current_photo_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [current_photo_id]

        parts: List[str] = []
        for i, nid in enumerate(path):
            objs = [o["label"] for o in self.photo_objects(nid)]
            label = ", ".join(objs[:5]) if objs else f"位置 {nid}"
            if i == 0:
                parts.append(f"起點看到：{label}")
            else:
                edge_data = walkway_graph.get_edge_data(path[i - 1], nid) or {}
                direction = edge_data.get("direction", "(未知方向)")
                parts.append(f"{direction}，抵達看到：{label}")
        return "。".join(parts) + "。"

    # ────────────────────────────────────────────────────────────────
    # 序列化 / Serialization
    # ────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        nodes = [{"id": n, **data} for n, data in self.graph.nodes(data=True)]
        edges = [
            {"from": u, "to": v, "key": k, **data}
            for u, v, k, data in self.graph.edges(keys=True, data=True)
        ]
        return {
            "place_name": self.place_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "next_id": self._next_id,
            "current_position": self.current_position,
            "position_history": self.position_history,
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopoGraphV2":
        g = cls(place_name=data.get("place_name", ""))
        for node in data.get("nodes", []):
            node = dict(node)
            nid = node.pop("id")
            g.graph.add_node(nid, **node)
        for edge in data.get("edges", []):
            edge = dict(edge)
            u, v, k = edge.pop("from"), edge.pop("to"), edge.pop("key")
            g.graph.add_edge(u, v, key=k, **edge)
        g._next_id = data.get("next_id", (max(g.graph.nodes) + 1) if g.graph.nodes else 0)
        g.current_position = data.get("current_position")
        g.position_history = data.get("position_history", [])
        g.created_at = data.get("created_at", g.created_at)
        g.updated_at = data.get("updated_at", g.updated_at)
        g._renormalize_labels()
        return g

    def _renormalize_labels(self) -> None:
        """Re-apply normalize_label to all object nodes so that label_norm
        stays consistent with the current LABEL_SYNONYMS table, even if the
        map was built with an older version of the table."""
        for nid, ndata in self.graph.nodes(data=True):
            if ndata.get("ntype") == NTYPE_OBJECT and "label" in ndata:
                ndata["label_norm"] = normalize_label(ndata["label"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TopoGraphV2":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ── 依場所存取（persistence registry）──────────────────────────
    # 讓同一個場所的地圖可以跨多次導航累積、更新，符合筆記
    # 「第一次跟每次導航開始時需新增的：目前建圖、導航的場所」的需求。
    @staticmethod
    def _place_path(place_name: str, root: str | Path) -> Path:
        return Path(root) / _slugify(place_name) / "topomap.json"

    @classmethod
    def load_for_place(cls, place_name: str, root: str | Path) -> "TopoGraphV2":
        """讀取某場所已存在的地圖；不存在就回傳一張全新的空地圖。"""
        p = cls._place_path(place_name, root)
        if p.exists():
            g = cls.load(p)
            g.place_name = place_name
            return g
        return cls(place_name=place_name)

    def save_for_place(self, root: str | Path) -> Path:
        p = self._place_path(self.place_name, root)
        self.save(p)
        return p

    # ────────────────────────────────────────────────────────────────
    # 視覺化 / Rendering
    # ────────────────────────────────────────────────────────────────

    # 配色（同一色系但照片／物件節點區分明顯，比預設的飽和藍橙更柔和）
    _COLOR_PHOTO = "#264653"       # 深藍綠 — 照片節點
    _COLOR_PHOTO_TEXT = "#FFFFFF"
    _COLOR_PHOTO_EDGE = "#12262E"
    _COLOR_OBJECT = "#E9C46A"      # 暖金色 — 物件節點
    _COLOR_OBJECT_TEXT = "#264653"
    _COLOR_OBJECT_EDGE = "#B08B3A"
    _COLOR_VIRTUAL = "#D9D9D9"     # 灰色 — 虛假照片節點
    _COLOR_VIRTUAL_EDGE = "#7A7A7A"
    _COLOR_WALKWAY = "#E76F51"     # 珊瑚橘 — 走道邊（直走，沒有轉彎）
    _COLOR_WALKWAY_TURN = "#4C6EF5"  # 靛藍 — 走道邊（這一步有轉彎，方向跟拍照點相對位置容易對不起來，特別標示）
    _COLOR_ADJACENT = "#2A9D8F"    # 青綠 — 相鄰物件邊
    _COLOR_SAME_OBJECT = "#6A4C93"  # 紫色 — 跨照片同物件邊
    _COLOR_CURRENT = "#E63946"     # 紅色 — 使用者目前位置徽章

    # render_png() 會把畫布的「英吋」尺寸直接設成「資料座標範圍 ×
    # _INCH_PER_DATA_UNIT」，讓文字（用 pt 為單位）跟節點方塊（用資料座標
    # 為單位）的比例永遠一致，不管圖有多大、有幾個節點，都不會跑掉。
    # 換算式：1 pt = 1/72 英吋 = (1/72)/_INCH_PER_DATA_UNIT 個資料座標單位。
    # 這個常數兩邊（_label_box_size 算方塊大小、render_png 算畫布大小）
    # 一定要共用同一個，否則文字跟方塊的比例就會對不上。
    _INCH_PER_DATA_UNIT = 0.62
    _DATA_UNITS_PER_PT = 1.0 / (72.0 * _INCH_PER_DATA_UNIT)

    # 節點文字字級（pt）。兩處用到（_layout_positions 算方塊大小、
    # render_png 實際畫文字）都要用同一組常數，否則文字大小跟方塊大小
    # 會對不上。之前分別是 11/9，使用者反應字太小，這次調大一點。
    _PHOTO_FONT = 13.5
    _OBJECT_FONT = 11.0

    _CJK_RE = re.compile(r"[\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF]")

    @classmethod
    def _char_width(cls, ch: str) -> float:
        """粗略估計一個字元佔標準字級的比例（中文字全形，較寬；英數窄一些）。"""
        if cls._CJK_RE.match(ch):
            return 1.0
        if ch == " ":
            return 0.32
        return 0.56

    @classmethod
    def _label_box_size(cls, text: str, font_size: float) -> Tuple[float, float]:
        """依文字內容與字級，算出剛好能完整放下文字的方塊寬高（資料座標單位）。

        這是為了解決「節點的字不會突出節點且完整呈現」的需求：不再用固定
        大小的圓點／方塊硬塞文字，而是反過來先量文字、再決定節點多大。
        寬高換算用的是 `_DATA_UNITS_PER_PT`，跟 `render_png()` 決定畫布
        英吋大小用的是同一個常數，所以文字實際畫出來的大小會跟這裡算出的
        方塊大小一致（不會忽大忽小、跟畫布縮放脫鉤）。
        """
        text = text or ""
        char_units = sum(cls._char_width(c) for c in text) or 1.0
        per_char = font_size * cls._DATA_UNITS_PER_PT
        width = char_units * per_char + 0.28
        height = font_size * cls._DATA_UNITS_PER_PT * 1.7 + 0.18
        return max(width, 0.55), max(height, 0.42)

    def _walkway_undirected(self) -> nx.Graph:
        """把走道邊（第四種邊）轉成無向圖，方便找主要動線。"""
        ug = nx.Graph()
        ug.add_nodes_from(self.all_photo_nodes())
        for u, v, d in self.graph.edges(data=True):
            if d.get("etype") == ETYPE_WALKWAY:
                ug.add_edge(u, v)
        return ug

    def _photo_label(self, photo_id: int) -> str:
        """照片節點顯示的文字：用「第幾張照片」的順序編號（P1, P2, P3...），
        不是圖裡的內部節點 id。

        內部 id 是「圖裡所有節點（照片＋物件）共用一個流水號」，每張照片
        中間會插入好幾個物件節點，所以內部 id 常常是 0, 8, 15, 21...
        這種跳號的數字——這是正常的，不是 bug，但對使用者來說很難懂
        「這是第幾張照片」，所以畫圖用的標籤改成只算照片節點本身的順序。
        """
        d = self.graph.nodes[photo_id]
        rank = self._photo_sequence_rank(photo_id)
        return f"P{rank}" + ("（未走訪）" if d.get("is_virtual") else "")

    def _photo_sequence_rank(self, photo_id: int) -> int:
        """回傳這個照片節點是所有照片節點裡（依內部 id 由小到大排）第幾個，
        從 1 開始編號。內部 id 由小到大排，正好就是實際拍照的時間順序
        （因為節點 id 本來就是依建立順序遞增配發的）。"""
        ordered = sorted(self.all_photo_nodes())
        try:
            return ordered.index(photo_id) + 1
        except ValueError:
            return photo_id

    def _layout_positions(
        self, photo_font: float = _PHOTO_FONT, object_font: float = _OBJECT_FONT, min_gap: float = 0.45,
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]], Dict[int, float]]:
        """把「照片節點沿走道排成一直線、物件節點吊掛在自己照片節點下方」
        設計成座標，取代預設的 spring_layout。

        這版同時會回傳每個節點的方塊尺寸（見 `_label_box_size`），並且：
        - 節點之間的間距是依「兩邊節點方塊的實際寬度」算出來的，
          不是固定值，所以文字比較長的節點不會跟隔壁重疊；
        - 每張照片底下的物件先分好幾列（zig-zag）打包、同一列內的物件
          彼此的間距也是依文字寬度算的，物件之間、跟隔壁照片的物件群
          之間都不會互相重疊。

        Returns:
            (pos, sizes, heading)：pos 是 {node_id: (x, y)}，sizes 是
            {node_id: (width, height)}，heading 是 {photo_id: 度數}（死算
            推位算出來的拍照朝向，畫走道箭頭／終點拍照方向標記時要用）。
        """
        photo_nodes = self.all_photo_nodes()
        pos: Dict[int, Tuple[float, float]] = {}
        sizes: Dict[int, Tuple[float, float]] = {}
        if not photo_nodes:
            return pos, sizes, {}

        for n in photo_nodes:
            sizes[n] = self._label_box_size(self._photo_label(n), photo_font)
        for n in self.all_object_nodes():
            sizes[n] = self._label_box_size(self.graph.nodes[n].get("label", ""), object_font)

        # ── 1. 以第一張照片（id 最小）為原點，沿走道邊的方向文字
        #        （直走／左轉／右轉／迴轉，還有看得到的話就用距離數字）
        #        做「死算推位」(dead reckoning)：這樣地圖的形狀會盡量貼近
        #        使用者實際走的路線轉折，而不是硬排成一條直線。
        #        沒有感測器精確方向/距離，所以角度只能抓文字裡的轉彎關鍵字，
        #        距離抓不到數字時就用一個固定的預設步長代替——這仍然只是
        #        「大概的形狀」，不是量出來的真座標，但比純直線更接近實際
        #        場景分布。
        raw_xy, heading, parent_of = self._dead_reckon_tree()

        # ── 2. 每張照片底下的物件依「九宮格 (row, col)」排位置：
        #        col 0/1/2（左/中/右）決定物件掛在照片節點哪一「欄」，
        #        欄與欄之間並排（三欄寬度固定，不會因為物件數量變多而
        #        越排越寬）；row 0/1/2（上/中/下）決定同一欄內由近到遠
        #        的堆疊順序——同一欄如果有好幾個物件，是往「遠離照片」
        #        的方向一個疊一個往外掛，下排（row=2）最靠近照片節點，
        #        上排（row=0）疊最遠。
        #
        #        欄的左右位置沿著「走道側邊」（perp_axis）展開，堆疊的
        #        遠近沿著「拍攝方向」（+row_axis，也就是照片節點的朝向、
        #        跟走道前進方向同一個值）展開——物件是照片裡拍到的東西，
        #        實際擺放位置理應在鏡頭前方，不是鏡頭後方，所以改成往
        #        「跟拍攝方向同向」的地方掛，比較接近物件實際的擺放位置。
        #        （第十二輪原本用 -row_axis／背對拍攝方向，是為了避免
        #        物件群一路往下一個照片節點的方向擴張過去；改成正方向
        #        後，如果跟下一個節點距離太近，交給既有的全域防重疊
        #        （`_resolve_cluster_overlaps` / `_resolve_edge_node_crossings`）
        #        處理，不需要犧牲「貼近實際擺放位置」這個更重要的目標。）
        #
        #        「下排最靠近、上排疊最遠」這個順序是刻意跟堆疊方向反過來
        #        配的：堆疊統一往同一個方向（+row_axis）延伸出去，如果
        #        還是「上排最靠近、下排疊最遠」（沿用改方向之前的順序），
        #        整串疊出去之後，上排的物件反而會落在畫面上比較靠近照片
        #        的那一端、下排落在比較遠的那一端——因為堆疊只往一個方向
        #        延伸，「離照片節點的遠近」自動就會對應到畫面上的某個
        #        方向，順序沒有跟著改方向一起翻過來的話，畫出來的上下
        #        關係就會是顛倒的（這正是這一輪要修正的問題）。改成
        #        「下排最靠近」之後，上排會落在堆疊延伸出去最遠的一端，
        #        跟畫面上「離照片節點越遠＝原照片裡越上面」的方向一致，
        #        堆疊出來的九宮格上下關係才會跟原照片相符，不會顛倒。
        #
        #        真實資料裡物件九成都落在同一列（中），如果同一列的物件
        #        用並排方式展開會排成一條很長的直線，所以固定三欄、
        #        往外堆疊，不管同一欄有幾個物件，整體寬度都只等於三欄
        #        寬度總和，物件多的欄就往外疊得深一點，畫面上會是一塊
        #        分三欄、有寬有高的方塊，比較貼近「九宮格」的直覺。
        #        堆疊/欄寬需要的實際距離都要等知道走道朝向後才能算
        #        （因為節點方塊不會轉，矩形投影到任意方向的公式一樣
        #        適用；欄寬跟堆疊深度分別投影到 perp_axis／row_axis）。──
        LANE_GAP = 0.3  # 三欄之間的間距
        row_margin = 0.22  # 同一欄內，物件與物件之間的額外間距
        obj_rel_layout: Dict[int, List[Tuple[int, float, float]]] = {}  # photo_id -> [(obj_id, depth_offset, lane_offset)]
        cluster_half_width: Dict[int, float] = {}
        cluster_reach: Dict[int, float] = {}  # 用來估計整個物件群「最遠可能延伸多少」，給步驟 3 抓安全間距用（保守的圓形估計，只在有轉彎、無法用方向性數值時當備案）
        cluster_forward_reach: Dict[int, float] = {}  # 物件群沿「自己拍攝方向」最遠伸到多遠——因為物件都往拍攝方向堆疊，這個方向才會有東西
        photo_row_axis_extent: Dict[int, float] = {}  # 照片節點自己的方塊沿自己 row_axis（拍攝方向）的半寬——物件群往拍攝方向延伸，反方向（背對拍攝方向）理論上不會有任何物件，只有照片節點自己的方塊要留空間

        def _row_axis_extent(heading_deg: float, box_w: float, box_h: float) -> float:
            """box 投影到 row_axis（沿走道前進/反方向）之後的半寬，用來算
            同一欄內堆疊需要留多少距離。"""
            hr = math.radians(heading_deg)
            ax, ay = abs(math.sin(hr)), abs(math.cos(hr))
            return (box_w / 2.0) * ax + (box_h / 2.0) * ay

        def _perp_axis_extent(heading_deg: float, box_w: float, box_h: float) -> float:
            """box 投影到 perp_axis（走道側邊）之後的半寬，用來算欄與欄
            之間需要留多少距離。"""
            hr = math.radians(heading_deg)
            ax, ay = abs(math.cos(hr)), abs(math.sin(hr))
            return (box_w / 2.0) * ax + (box_h / 2.0) * ay

        for pnode in photo_nodes:
            objs = sorted(self.photo_objects(pnode), key=lambda o: o["id"])
            lanes: List[List[dict]] = [[], [], []]
            for o in objs:
                row, col = grid_row_col(o.get("grid_cell", "center"))
                lanes[col].append((row, o))
            for c in range(3):
                # 同一欄先依 row 排，越先排到的物件離照片節點越近（見下面
                # 步驟 4，cum 是累加距離，排第一個的 cum 最小）。
                #
                # 這裡刻意由「下→中→上」排（row 由大到小），不是單純的
                # 「上→中→下」：因為堆疊方向這一輪改成順著拍攝方向
                # （+row_axis，見步驟 4 的說明），如果還是「上排最先＝
                # 上排離照片節點最近」，整串疊起來之後，上排反而會落在
                # 堆疊的近端、下排落在遠端——堆疊往同一個方向延伸時，
                # 畫面上看起來就是上下顛倒的（下排的物件看起來在上面，
                # 上排的物件看起來在下面）。改成「下排最先＝下排離照片
                # 節點最近」，下排會落在堆疊的近端、上排落在遠端，堆疊
                # 延伸出去後，上排的物件才會真的落在畫面上比較「上面」
                # 的位置，跟原照片的上下關係一致。
                lanes[c].sort(key=lambda item: (-item[0], item[1]["id"]))

            ph = heading.get(pnode, 0.0)
            lane_width = [
                max((_perp_axis_extent(ph, *sizes[o["id"]]) * 2.0 for _, o in lanes[c]), default=0.0)
                for c in range(3)
            ]
            cols_present = [c for c in range(3) if lanes[c]]
            total_w = sum(lane_width[c] for c in cols_present) + LANE_GAP * max(len(cols_present) - 1, 0)
            lane_center: Dict[int, float] = {}
            cur = -total_w / 2.0
            for c in range(3):
                if not lanes[c]:
                    continue
                lane_center[c] = cur + lane_width[c] / 2.0
                cur += lane_width[c] + LANE_GAP

            base_extent = _row_axis_extent(ph, *sizes[pnode])
            rel_positions: List[Tuple[int, float, float]] = []
            max_lane_depth = 0.0
            for c in range(3):
                items = lanes[c]
                if not items:
                    continue
                prev_extent = base_extent
                cum = 0.0
                for idx, (_row, o) in enumerate(items):
                    this_extent = _row_axis_extent(ph, *sizes[o["id"]])
                    cum = (prev_extent + row_margin + this_extent) if idx == 0 else \
                        (cum + prev_extent + row_margin + this_extent)
                    prev_extent = this_extent
                    rel_positions.append((o["id"], cum, lane_center[c]))
                max_lane_depth = max(max_lane_depth, cum)

            obj_rel_layout[pnode] = rel_positions
            cluster_half_width[pnode] = total_w / 2.0
            cluster_reach[pnode] = max(total_w / 2.0, max_lane_depth)
            cluster_forward_reach[pnode] = max(base_extent, max_lane_depth)
            photo_row_axis_extent[pnode] = base_extent

        def _extent(node_id: int) -> float:
            return max(sizes[node_id][0] / 2.0, cluster_reach.get(node_id, 0.0))

        # ── 3. 把死算推位算出來的「原始位置」依樹狀結構（父節點→子節點）
        #        重新量距離：方向（角度）完全保留死算推位算出來的轉折，
        #        但距離會拉長到至少「兩邊節點方塊/物件群的安全間距」，
        #        避免距離抓得太短時兩個節點疊在一起。
        #
        #        安全間距這裡改成盡量用「方向性」的數值，不是不管方向
        #        一律套用同一個保守的圓形範圍——物件都是往「自己的拍攝
        #        方向」堆疊，某個方向可能因為疊了很多物件而伸得很遠
        #        （見 `cluster_forward_reach`），但反方向（背對拍攝方向）
        #        理論上完全不會有物件，只需要留照片節點自己方塊的空間
        #        （見 `photo_row_axis_extent`）。如果還是不分方向、兩端
        #        都套用「物件群可能伸多遠」的保守估計，會導致「子節點
        #        自己那側其實沒有物件」也被迫留出一大段空間，把兩個
        #        照片節點的距離拉得比實際需要的長很多（對應使用者回報：
        #        P16、P17 之間的間距明顯偏長——P17 自己剛好有 6 個物件
        #        疊在同一欄，`cluster_reach` 因此很大，但那些物件是往
        #        P17 的拍攝方向、也就是遠離 P16 的方向延伸，並不會佔用
        #        P16、P17 之間的空間，用不分方向的保守估計去抓間距是
        #        不必要的浪費）。
        #
        #        只有在這條邊「沒有轉彎」（父節點跟子節點的拍攝方向
        #        相同）時才套用方向性的估計：這種情況下，父節點→子節點
        #        的直線方向剛好就是父節點自己的拍攝方向，父節點的物件群
        #        往同一個方向延伸，`cluster_forward_reach` 量的正是這個
        #        方向上的實際距離，是準確的；子節點的方塊背對這個方向
        #        （物件都往另一邊、也就是繼續往前的方向延伸），只需要
        #        自己方塊的半寬。如果這條邊「有轉彎」，父節點的物件群
        #        实際延伸方向跟這條邊的方向不一致（物件是照父節點自己的
        #        拍攝方向堆疊，不是照這條邊轉彎後的新方向），方向性估計
        #        不準，這種情況下退回原本不分方向的保守估計（`_extent`），
        #        避免低估間距造成重疊。──
        for n in photo_nodes:
            if n not in raw_xy:
                # 沒被走道邊連到（理論上少見，例如只有單張照片）：退回原點
                raw_xy[n] = (0.0, 0.0)
                heading[n] = 0.0

        root = min(photo_nodes)
        pos[root] = raw_xy[root]
        # 依 BFS 距離根節點的順序調整，確保先處理父節點再處理子節點
        order = sorted(photo_nodes, key=lambda n: self._tree_depth(n, parent_of, root))
        for n in order:
            if n == root:
                continue
            p = parent_of.get(n)
            if p is None or p not in pos:
                pos[n] = raw_xy[n]
                continue
            px, py = pos[p]
            rx, ry = raw_xy[n][0] - raw_xy.get(p, (0.0, 0.0))[0], raw_xy[n][1] - raw_xy.get(p, (0.0, 0.0))[1]
            raw_dist = math.hypot(rx, ry) or 1.0
            ux, uy = rx / raw_dist, ry / raw_dist

            no_turn_here = abs(((heading.get(n, 0.0) - heading.get(p, 0.0) + 180.0) % 360.0) - 180.0) < 1e-6
            if no_turn_here:
                p_reach = cluster_forward_reach.get(p, _extent(p))
                n_reach = photo_row_axis_extent.get(n, _extent(n))
            else:
                p_reach = _extent(p)
                n_reach = _extent(n)
            min_dist = p_reach + min_gap * 1.0 + n_reach
            actual_dist = max(raw_dist, min_dist)
            pos[n] = (px + ux * actual_dist, py + uy * actual_dist)

        # ── 4. 物件節點的絕對座標：吊掛方向依該照片節點的「走道朝向」
        #        旋轉——欄的左右位置沿走道側邊（perp_axis）展開，同一欄
        #        內由近到遠的堆疊沿「拍攝方向」（+row_axis，跟走道前進
        #        方向同一個值）展開，比較貼近物件在真實空間裡的擺放
        #        位置（物件是鏡頭拍到的東西，理應在鏡頭前方）。實際的
        #        堆疊深度跟欄位置已經在步驟 2 用同一個 heading 算好了，
        #        這裡只要把相對座標轉成絕對座標即可。物件群因此可能會
        #        跟路徑上的下一個節點比較靠近，交給步驟 5 的全域防重疊
        #        處理，不會真的疊在一起。──
        for pnode in photo_nodes:
            px, py = pos[pnode]
            h = math.radians(heading.get(pnode, 0.0))
            row_axis = (math.sin(h), math.cos(h))       # 沿走道前進方向（＝拍攝方向）
            perp_axis = (math.cos(h), -math.sin(h))     # 走道側邊（固定選一側，讓同一張圖裡方向一致）
            for oid, depth_offset, lane_offset in obj_rel_layout.get(pnode, []):
                ox = px + row_axis[0] * depth_offset + perp_axis[0] * lane_offset
                oy = py + row_axis[1] * depth_offset + perp_axis[1] * lane_offset
                pos[oid] = (ox, oy)

        # ── 5. 全域防重疊：路線折返時，樹狀結構上不相鄰的兩個節點群
        #        （例如折返點跟折返前經過的節點）在空間上可能剛好靠在
        #        一起。前面幾步只確保「父節點→子節點」這種直接相鄰的
        #        關係不會疊到，折返造成的「非相鄰但空間上靠近」不會被
        #        擋到，所以這裡再做一輪全域檢查：把每個照片節點跟它自己
        #        的物件群當成一個整體（移動的時候一起移動，不破壞群內
        #        排列），檢查任兩群是否重疊，重疊就把兩群沿造成重疊最少
        #        的方向推開，反覆幾輪直到不再重疊為止。──
        cluster_members: Dict[int, List[int]] = {
            pnode: [pnode] + [oid for oid, _, _ in obj_rel_layout.get(pnode, [])]
            for pnode in photo_nodes
        }
        walkway_pairs = [(u, v) for u, v, d in self.graph.edges(data=True)
                          if d.get("etype") == ETYPE_WALKWAY and u != v]
        # 兩種問題交替修正幾輪：「群跟群整塊重疊」跟「走道邊被無關節點擋住」
        # 有可能互相牽動（修好一個，可能讓另一個重新出現），所以不是各修
        # 一次就結束，而是反覆跑到兩邊都穩定為止。這一輪把預設間距縮短
        # 之後，節點彼此擠得更緊，兩種修正機制有時會卡進「你推過去、我
        # 推回來」的無限循環（防重疊把某個群拉回來，剛好又把它推回走道
        # 邊的路徑上，兩邊各自的修正都沒錯，只是彼此的目標互相牴觸）——
        # 這裡加一個循環偵測：把每輪修正完的座標四捨五入後當成一個「狀態
        # 指紋」，如果同一個指紋重複出現，代表卡進循環了，之後的走道邊
        # 穿越修正改用加大的推擠幅度（每偵測到一次循環就再加大），直到
        # 真的被推到不會再跟防重疊的拉力打架的地方為止。
        crossing_margin = 0.3
        seen_signatures: set = set()
        for _ in range(30):
            self._resolve_cluster_overlaps(pos, sizes, cluster_members)
            moved = self._resolve_edge_node_crossings(
                pos, sizes, cluster_members, walkway_pairs, margin=crossing_margin
            )
            if not moved:
                break
            signature = tuple(sorted((k, round(v[0], 2), round(v[1], 2)) for k, v in pos.items()))
            if signature in seen_signatures:
                crossing_margin *= 1.8  # 陷入循環，加大推擠幅度試著跳出
            seen_signatures.add(signature)

        # ── 6. 角度復原：第 5 步的全域防重疊，有時候會為了解開路線折返
        #        造成的衝突，連帶把「根本沒有衝突」的父子節點對之間的
        #        角度也一起推歪（距離不變，方向跑掉）——死算推位算出來
        #        的轉折角度其實是對的（例如資料寫「請向右轉」，死算推位
        #        算出來就是精準的 90 度），被防重疊的推擠影響後，畫出來
        #        會變成不是 90 度，容易讓人誤會轉彎角度算錯了。
        #
        #        這裡嘗試把每個節點跟它父節點之間的角度修正回死算推位的
        #        原始角度（距離維持防重疊調整後的值，不縮短，只調整
        #        方向），但**只有修正後重新檢查、確定不會造成任何新的
        #        重疊或走道邊穿越時才會真的套用**，不然維持原樣——寧可
        #        角度不夠精準，也不要為了修角度又把好不容易解開的重疊
        #        問題找回來。修正一個節點時，會把它「以自己為根的整個
        #        子樹」（子樹裡所有照片節點，連同它們各自的物件群）一起
        #        剛體平移，子樹內部的相對排列不會被打亂。
        children_of: Dict[int, List[int]] = {}
        for child, parent in parent_of.items():
            if child in pos and parent in pos:
                children_of.setdefault(parent, []).append(child)

        def _subtree_photo_nodes(start: int) -> List[int]:
            result = [start]
            stack = [start]
            while stack:
                cur = stack.pop()
                for ch in children_of.get(cur, []):
                    result.append(ch)
                    stack.append(ch)
            return result

        def _has_overlap() -> bool:
            items = list(pos.keys())
            for i in range(len(items)):
                ax, ay = pos[items[i]]
                aw, ah = sizes[items[i]]
                for j in range(i + 1, len(items)):
                    bx, by = pos[items[j]]
                    bw, bh = sizes[items[j]]
                    if abs(ax - bx) < (aw + bw) / 2.0 - 1e-6 and abs(ay - by) < (ah + bh) / 2.0 - 1e-6:
                        return True
            return False

        def _has_crossing() -> bool:
            node_owner: Dict[int, int] = {}
            for owner, members in cluster_members.items():
                for m in members:
                    node_owner[m] = owner
            for u, v in walkway_pairs:
                if u not in pos or v not in pos:
                    continue
                owner_u, owner_v = node_owner.get(u, u), node_owner.get(v, v)
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                seg_len = math.hypot(x1 - x0, y1 - y0)
                if seg_len < 1e-6:
                    continue
                for node_id in pos:
                    owner = node_owner.get(node_id, node_id)
                    if owner in (owner_u, owner_v):
                        continue
                    bx, by = pos[node_id]
                    bw, bh = sizes[node_id]
                    for i in range(41):
                        t = i / 40.0
                        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                        if abs(x - bx) <= bw / 2.0 and abs(y - by) <= bh / 2.0:
                            return True
            return False

        for n in order:
            if n == root:
                continue
            p = parent_of.get(n)
            if p is None or p not in pos or n not in raw_xy or p not in raw_xy:
                continue
            rx, ry = raw_xy[n][0] - raw_xy[p][0], raw_xy[n][1] - raw_xy[p][1]
            raw_len = math.hypot(rx, ry)
            if raw_len < 1e-9:
                continue
            target_ux, target_uy = rx / raw_len, ry / raw_len

            px, py = pos[p]
            cx, cy = pos[n]
            cur_len = math.hypot(cx - px, cy - py)
            if cur_len < 1e-9:
                continue

            full_new_x, full_new_y = px + target_ux * cur_len, py + target_uy * cur_len
            full_dx, full_dy = full_new_x - cx, full_new_y - cy
            if abs(full_dx) < 1e-6 and abs(full_dy) < 1e-6:
                continue  # 角度本來就沒被扭曲，不用修

            subtree_members = [
                m for sp in _subtree_photo_nodes(n) for m in cluster_members.get(sp, [sp])
            ]
            backup = {m: pos[m] for m in subtree_members if m in pos}

            # 先試完全修正（角度百分之百對齊死算推位），空間不夠、會重新
            # 造成重疊或穿越的話，改試比較小幅度的修正（75%／50%／25%，
            # 相當於朝目標角度轉一部分而不是轉到底），取「能安全套用的
            # 最大幅度」——比起「完全修正失敗就整個放棄、維持原本被
            # 防重疊推歪的角度」，能修正多少算多少，通常還是比完全不修
            # 更接近死算推位算出來的真實角度。
            applied = False
            for frac in (1.0, 0.75, 0.5, 0.25):
                dx, dy = full_dx * frac, full_dy * frac
                for m in subtree_members:
                    if m in pos:
                        pos[m] = (backup[m][0] + dx, backup[m][1] + dy)
                if not (_has_overlap() or _has_crossing()):
                    applied = True
                    break
            if not applied:
                for m, old_xy in backup.items():
                    pos[m] = old_xy  # 連 25% 的修正都不安全，放棄這次修正，維持原樣

        return pos, sizes, heading

    @staticmethod
    def _cluster_bbox(pos: Dict[int, Tuple[float, float]], sizes: Dict[int, Tuple[float, float]],
                       members: List[int]) -> Tuple[float, float, float, float]:
        xs_min = min(pos[m][0] - sizes[m][0] / 2.0 for m in members)
        xs_max = max(pos[m][0] + sizes[m][0] / 2.0 for m in members)
        ys_min = min(pos[m][1] - sizes[m][1] / 2.0 for m in members)
        ys_max = max(pos[m][1] + sizes[m][1] / 2.0 for m in members)
        return xs_min, xs_max, ys_min, ys_max

    def _resolve_cluster_overlaps(
        self,
        pos: Dict[int, Tuple[float, float]],
        sizes: Dict[int, Tuple[float, float]],
        cluster_members: Dict[int, List[int]],
        max_iterations: int = 80,
        margin: float = 0.25,
    ) -> None:
        """反覆偵測、消除「照片節點＋物件群」這種整塊之間的重疊（原地修改
        pos）。每個照片節點跟它自己的物件群一起當作剛體移動，只調整
        群與群之間的相對位置，不會打亂群內部（物件相對於自己照片節點）
        的排列方式。

        這是路線有折返（U 型甚至繞回原地）時的最後一道防線：死算推位
        跟父子節點間距只保證「直接走過去的下一步」不會疊到，折返後
        跟很久以前經過的節點在空間上靠在一起的情況要靠這裡處理。
        """
        photo_ids = list(cluster_members.keys())
        for _ in range(max_iterations):
            moved = False
            for i in range(len(photo_ids)):
                for j in range(i + 1, len(photo_ids)):
                    a, b = photo_ids[i], photo_ids[j]
                    members_a, members_b = cluster_members[a], cluster_members[b]
                    ax1, ax2, ay1, ay2 = self._cluster_bbox(pos, sizes, members_a)
                    bx1, bx2, by1, by2 = self._cluster_bbox(pos, sizes, members_b)
                    overlap_x = min(ax2, bx2) - max(ax1, bx1)
                    overlap_y = min(ay2, by2) - max(ay1, by1)
                    if overlap_x <= 0 or overlap_y <= 0:
                        continue  # 沒有真的重疊

                    moved = True
                    center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
                    center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                    # 往「重疊量比較小」的那個方向推開，移動距離最少
                    if overlap_x < overlap_y:
                        shift = overlap_x / 2.0 + margin
                        dx = shift if center_a[0] <= center_b[0] else -shift
                        delta_a, delta_b = (-dx, 0.0), (dx, 0.0)
                    else:
                        shift = overlap_y / 2.0 + margin
                        dy = shift if center_a[1] <= center_b[1] else -shift
                        delta_a, delta_b = (0.0, -dy), (0.0, dy)

                    for m in members_a:
                        pos[m] = (pos[m][0] + delta_a[0], pos[m][1] + delta_a[1])
                    for m in members_b:
                        pos[m] = (pos[m][0] + delta_b[0], pos[m][1] + delta_b[1])
            if not moved:
                break

    def _resolve_edge_node_crossings(
        self,
        pos: Dict[int, Tuple[float, float]],
        sizes: Dict[int, Tuple[float, float]],
        cluster_members: Dict[int, List[int]],
        walkway_pairs: List[Tuple[int, int]],
        margin: float = 0.3,
        samples: int = 40,
    ) -> bool:
        """檢查每條走道邊是不是被「不相關的節點方塊」擋在中間。

        `_resolve_cluster_overlaps` 只檢查群跟群整塊有沒有重疊，但路線
        折返、繞回附近時，常常是兩個在路徑上離得很遠的照片節點，投影到
        2D 座標後剛好排在同一直線附近——某張照片（或它的物件）並沒有跟
        邊的兩端「整塊重疊」，卻剛好卡在邊的路徑正中間，這裡另外處理。

        做法：對每條走道邊，沿整條線段取樣，檢查有沒有落進某個不屬於
        這條邊兩端節點所屬群的節點方塊裡；有的話，把「擋路的那個節點
        所屬的整個群」沿著「垂直於這條邊」的方向推開，直到不再擋路
        為止。

        同一個群如果有好幾個物件（例如同一欄堆疊很深）都擋在路徑上，
        一次算出「能同時清掉這幾個擋路物件」的推擠量（取這幾個物件
        各自需要的推擠量的最大值），一次推完，不是逐一物件各推一點點
        ——群裡的物件是沿著自己這欄的方向排成一長串，這條走道邊如果跟
        這欄的方向不平行，逐一小幅推擠容易變成「推開了這個物件，換下
        一個物件又擋到」，跟 `_resolve_cluster_overlaps` 互相拉扯陷入
        來回震盪、永遠收斂不了。

        Returns:
            這一輪有沒有真的移動過任何節點（呼叫端用來判斷還要不要
            再跑一輪跟 `_resolve_cluster_overlaps` 交替修正）。
        """
        node_to_cluster: Dict[int, int] = {}
        for owner, members in cluster_members.items():
            for m in members:
                node_to_cluster[m] = owner

        moved_any = False
        for u, v in walkway_pairs:
            if u not in pos or v not in pos:
                continue
            owner_u = node_to_cluster.get(u, u)
            owner_v = node_to_cluster.get(v, v)
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len < 1e-6:
                continue
            ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
            perp = (-uy, ux)

            # 先把「擋路的物件」依所屬的群分組，同一群一次算出足以清掉
            # 所有擋路物件的推擠量，而不是逐一物件各自推一次。
            hits_by_owner: Dict[int, List[int]] = {}
            for node_id in list(pos.keys()):
                owner = node_to_cluster.get(node_id, node_id)
                if owner in (owner_u, owner_v):
                    continue  # 邊自己兩端所屬的群，本來就會很靠近，不用閃避
                bx, by = pos[node_id]
                bw, bh = sizes[node_id]
                hit = False
                for i in range(samples + 1):
                    t = i / samples
                    x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                    if abs(x - bx) <= bw / 2.0 and abs(y - by) <= bh / 2.0:
                        hit = True
                        break
                if hit:
                    hits_by_owner.setdefault(owner, []).append(node_id)

            for owner, hit_nodes in hits_by_owner.items():
                moved_any = True
                # 這裡刻意用「這個群的所有成員」（不是只有這一輪偵測到
                # 擋路的那幾個）去算需要的清空距離：同一群裡不同物件的
                # 方塊大小不一樣，隨著群被推來推去，不同輪換不同物件
                # 剛好卡在線上，如果每次只看「這一輪擋路的是誰」來決定
                # 推多遠，可能推開了小的、換大的卡上去，永遠都在跟
                # `_resolve_cluster_overlaps` 拉扯收斂不了；改成一次用
                # 全部成員裡最寬的算清空距離，一次推到位。
                all_members = cluster_members.get(owner, hit_nodes)
                shift = max(
                    (sizes[m][0] / 2.0) * abs(perp[0]) + (sizes[m][1] / 2.0) * abs(perp[1])
                    for m in all_members if m in sizes
                ) + margin
                # 用「這個群裡所有擋路物件的平均位置」判斷該往哪一側推，
                # 比只看單一物件更能代表整個群相對於這條邊的位置。
                avg_bx = sum(pos[nid][0] for nid in hit_nodes) / len(hit_nodes)
                avg_by = sum(pos[nid][1] for nid in hit_nodes) / len(hit_nodes)
                side = (avg_bx - x0) * perp[0] + (avg_by - y0) * perp[1]
                direction = 1.0 if side >= 0 else -1.0
                dx, dy = perp[0] * shift * direction, perp[1] * shift * direction
                for m in cluster_members.get(owner, hit_nodes):
                    pos[m] = (pos[m][0] + dx, pos[m][1] + dy)

        return moved_any

    # 方向文字關鍵字 → 轉彎角度（度，正值＝右轉，模仿指北針方向）。
    _TURN_LEFT_KEYWORDS = ("左轉", "向左", "往左", "靠左")
    _TURN_RIGHT_KEYWORDS = ("右轉", "向右", "往右", "靠右")
    _TURN_BACK_KEYWORDS = (
        "迴轉", "回轉", "掉頭", "回頭", "轉身", "折返", "往回走", "退回",
        "turn around", "turn back",
    )
    _DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:公尺|米|m\b)")

    @classmethod
    def _parse_turn_degrees(cls, text: str) -> float:
        """從走道邊的方向文字（自然語言，VLM 產生，不是固定詞彙表）粗略判斷
        轉彎角度。沒有羅盤/陀螺儀資料，只能抓字面上的轉彎關鍵字，抓不到
        （例如純粹的「直走」「請繼續直走」）就當作沒有轉彎。
        """
        if not text:
            return 0.0
        if any(k in text for k in cls._TURN_BACK_KEYWORDS):
            return 180.0
        has_left = any(k in text for k in cls._TURN_LEFT_KEYWORDS)
        has_right = any(k in text for k in cls._TURN_RIGHT_KEYWORDS)
        if has_left and not has_right:
            return -90.0
        if has_right and not has_left:
            return 90.0
        return 0.0  # 兩個方向都提到、或都沒提到：無法判斷，當作直走

    @classmethod
    def _parse_distance_m(cls, text: str) -> Optional[float]:
        """從方向文字裡抓一個「幾公尺」的數字，抓不到就回傳 None
        （呼叫端會退回用預設步長）。"""
        if not text:
            return None
        m = cls._DISTANCE_RE.search(text)
        return float(m.group(1)) if m else None

    _UNITS_PER_METER = 2.0     # 抓得到公尺數時，1 公尺畫成多少資料座標單位
    _DEFAULT_STEP_UNITS = 3.2  # 抓不到公尺數（大多數情況）時的預設步長

    def _walkway_edge_data(self, u: int, v: int) -> dict:
        for a, b in ((u, v), (v, u)):
            data = self.graph.get_edge_data(a, b) or {}
            for d in data.values():
                if d.get("etype") == ETYPE_WALKWAY:
                    return d
        return {}

    def _dead_reckon_tree(self) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, float], Dict[int, int]]:
        """以「第一張照片（id 最小）」為原點、沿走道邊做死算推位，把每個
        照片節點走道邊組成的圖當成一棵樹（用 BFS 展開，遇到還沒走過的
        節點才展開，天然就是使用者實際走過的分支結構，包括虛假分支節點）。

        Returns:
            (xy, heading, parent_of)：
            - xy：{node_id: (x, y)}，死算推位算出的原始座標（還沒做防重疊調整）
            - heading：{node_id: 走到這個節點時面向的角度（度，0＝起點面向的方向）}
            - parent_of：{node_id: 是從哪個節點走過來的}，根節點沒有這個 key
        """
        photo_nodes = self.all_photo_nodes()
        xy: Dict[int, Tuple[float, float]] = {}
        heading: Dict[int, float] = {}
        parent_of: Dict[int, int] = {}
        if not photo_nodes:
            return xy, heading, parent_of

        root = min(photo_nodes)
        xy[root] = (0.0, 0.0)
        heading[root] = 0.0
        ug = self._walkway_undirected()
        visited = {root}
        queue = [root]
        while queue:
            cur = queue.pop(0)
            if cur not in ug:
                continue
            for nb in sorted(ug.neighbors(cur)):
                if nb in visited:
                    continue
                edge_data = self._walkway_edge_data(cur, nb)
                text = edge_data.get("direction", "") or ""
                turn = self._parse_turn_degrees(text)
                dist_m = self._parse_distance_m(text)
                step = dist_m * self._UNITS_PER_METER if dist_m else self._DEFAULT_STEP_UNITS
                h = heading[cur] + turn
                rad = math.radians(h)
                x = xy[cur][0] + step * math.sin(rad)
                y = xy[cur][1] + step * math.cos(rad)
                xy[nb] = (x, y)
                heading[nb] = h
                parent_of[nb] = cur
                visited.add(nb)
                queue.append(nb)

        # 沒被走道邊連到主要分量的節點（理論上少見，例如外部呼叫端接了一個
        # 沒有走道邊的獨立節點進來）：擺到整張地圖版面「外面」，跟所有已經
        # 排好的節點、物件群保持足夠安全距離，不要疊在主圖上面——早期版本
        # 只是貼著起點（root）挪一點點小偏移，起點旁邊本來就有自己的物件群
        # 展開，這個小偏移量完全不夠，會直接疊到主路徑的節點/物件上，嚴重
        # 影響可讀性。改成算出目前所有節點座標的邊界框，把這些節點排在
        # 邊界框下方留一大段安全間距的地方，橫向排開一列，保證落在先前
        # 完全沒有東西的空白區域。
        leftover = [n for n in photo_nodes if n not in visited]
        if leftover:
            anchor = root if root in xy else next(iter(xy), root)
            ah = heading.get(anchor, 0.0)
            all_x = [p[0] for p in xy.values()]
            all_y = [p[1] for p in xy.values()]
            margin = self._DEFAULT_STEP_UNITS * 4  # 留夠寬，蓋過物件群可能展開的範圍
            base_x = min(all_x)
            base_y = min(all_y) - margin
            spacing = self._DEFAULT_STEP_UNITS * 5  # 節點之間也要留夠物件群的空間，避免互相重疊
            for i, n in enumerate(leftover):
                xy[n] = (base_x + i * spacing, base_y)
                heading[n] = ah
                parent_of[n] = anchor
        return xy, heading, parent_of

    @staticmethod
    def _tree_depth(node: int, parent_of: Dict[int, int], root: int, _cache: Optional[dict] = None) -> int:
        """算某節點在死算推位那棵樹裡離根節點多少步（用來確保排版時
        一定先算完父節點的最終座標，才去算子節點的）。"""
        depth = 0
        cur = node
        seen = {cur}
        while cur != root and cur in parent_of:
            cur = parent_of[cur]
            depth += 1
            if cur in seen:  # 防呆：理論上樹狀結構不會有環
                break
            seen.add(cur)
        return depth

    def _draw_node_box(self, ax, x: float, y: float, w: float, h: float, *,
                        face: str, edge: str, dashed: bool = False, linewidth: float = 1.3) -> None:
        box = mpatches.FancyBboxPatch(
            (x - w / 2.0, y - h / 2.0), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=linewidth, edgecolor=edge, facecolor=face,
            linestyle="dashed" if dashed else "solid", zorder=3,
        )
        ax.add_patch(box)

    def render_png(self) -> bytes:
        """把拓樸地圖畫成 PNG（照片＋物件＋所有邊，唯一一種畫法）。

        節點是依文字內容自動算出剛好包住文字的方塊（見 `_label_box_size`），
        彼此間距也是依方塊實際大小算的，所以文字不會被截斷、也不會跟
        鄰居重疊。畫布的英吋大小是直接從排版算出的資料座標範圍換算過來
        （用跟 `_label_box_size` 相同的比例常數），確保文字大小跟節點方塊
        的比例不會因為圖的大小不同而跑掉。使用者目前位置改用該照片節點
        右上角的一顆小紅星徽章標示，而不是蓋在節點正中央的大星星。

        以前這裡還有一個 `mode="backbone"`，只畫照片節點跟走道邊、不畫
        物件，用來先看整體路線走向。評估後拿掉了：它畫的東西完全是
        `render_png()` 的子集，沒有任何獨有資訊，只是省略細節的簡化版；
        系統實際運作（定位、導航）只需要 `topomap.json` 這份圖資料本身，
        PNG 只是給人看的視覺化，不是任何程式邏輯的輸入，「總覽版」純粹
        是人工除錯時的方便功能，不是系統運作必要的東西，所以拿掉了，
        改成只有一種畫法，程式碼也少一層 `mode` 分支要維護。
        """
        if self.graph.number_of_nodes() == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "(空地圖)", ha="center", va="center")
            ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            return buf.getvalue()

        pos, sizes, heading = self._layout_positions()

        photo_nodes = self.all_photo_nodes()
        object_nodes = self.all_object_nodes()
        virtual_nodes = [n for n in photo_nodes if self.graph.nodes[n].get("is_virtual")]
        real_photo_nodes = [n for n in photo_nodes if n not in virtual_nodes]
        has_facing_marker = any(
            u == v and d.get("etype") == ETYPE_WALKWAY for u, v, d in self.graph.edges(data=True)
        )
        has_turn_edge = any(
            u != v and d.get("etype") == ETYPE_WALKWAY and self._parse_turn_degrees(d.get("direction", "") or "")
            for u, v, d in self.graph.edges(data=True)
        )

        # 畫布範圍依「這張圖實際會畫出來」的所有節點（照片＋物件）計算。
        visible_ids = list(pos.keys())
        x_min = min(pos[n][0] - sizes[n][0] / 2 for n in visible_ids)
        x_max = max(pos[n][0] + sizes[n][0] / 2 for n in visible_ids)
        y_min = min(pos[n][1] - sizes[n][1] / 2 for n in visible_ids)
        y_max = max(pos[n][1] + sizes[n][1] / 2 for n in visible_ids)
        pad = 0.4
        data_w = max(x_max - x_min + pad * 2, 1.0)
        data_h = max(y_max - y_min + pad * 2, 1.0)

        legend_count = (2 + (1 if virtual_nodes else 0) + 3 + 1
                        + (1 if has_facing_marker else 0) + (1 if has_turn_edge else 0))  # +1 是目前位置
        legend_ncol = min(legend_count, 3)
        legend_rows = math.ceil(legend_count / legend_ncol)

        margin_in = 0.5   # 給邊緣留的英吋邊界
        title_in = 0.42   # 給標題留的英吋高度
        legend_in = 0.26 * legend_rows + 0.16  # 圖例橫條高度依列數動態調整，
                           # 畫在標題正下方、圖的正上方，不會蓋到節點
                           # ——之前是疊在圖的左上角，小圖時會蓋住節點
        # 九宮格色卡（3x3 色塊＋上中下/左中右標籤）：畫在圖例橫條下方、
        # 主圖上方，只在真的有物件節點時才保留這塊高度。
        show_grid_legend = bool(object_nodes)
        grid_legend_in = 1.28 if show_grid_legend else 0.0
        MIN_FIG_W = 8.6    # 圖例橫排最少需要的寬度，圖本身內容較窄時也要保留這麼寬，
                           # 否則圖例文字會被截斷（畫布不夠寬，legend 換行/裁切）
        axes_w_in = data_w * self._INCH_PER_DATA_UNIT
        fig_w = max(axes_w_in + margin_in * 2, MIN_FIG_W)
        fig_h = (data_h * self._INCH_PER_DATA_UNIT + margin_in * 2
                 + title_in + legend_in + grid_legend_in)

        fig = plt.figure(figsize=(fig_w, fig_h))
        # 繪圖區寬度固定用 axes_w_in（維持跟 _label_box_size 一致的英吋/資料
        # 座標比例），畫布因圖例而被撐寬時，讓繪圖區在多出的寬度中置中，
        # 不要整個被拉伸變形。
        left_margin_total = fig_w - axes_w_in
        left = (left_margin_total / 2.0) / fig_w
        right = left + axes_w_in / fig_w
        bottom = margin_in / fig_h
        top = 1.0 - (title_in + legend_in + grid_legend_in) / fig_h
        ax = fig.add_axes((left, bottom, max(right - left, 0.05), max(top - bottom, 0.05)))
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

        # ── 邊（先畫「非走道」的邊，讓節點方塊蓋在上面，邊看起來才像
        #    連到節點邊緣）──
        #
        #    adjacent/same_object 這兩種邊在資料裡是「雙向各存一條」（方便
        #    沿任一方向查詢鄰居），但畫圖時同一對節點只需要畫一條線／一條
        #    弧就好，兩條方向相反的邊如果都畫出來，會變成兩條分開的弧線
        #    （尤其是 same_object 這種有弧度的線，方向相反時弧會往兩邊彎，
        #    疊在一起看起來像一對眼睛）。這裡統一在畫之前先依「無序節點
        #    對」去重，只保留一條。
        def _dedup_unordered_pairs(pairs):
            seen = set()
            result = []
            for u, v in pairs:
                key = frozenset((u, v))
                if key in seen:
                    continue
                seen.add(key)
                result.append((u, v))
            return result

        # contains 邊依「目標物件的九宮格位置」上色（而不是統一灰色），
        # 顏色跟物件節點外框用同一套 grid_cell_color()：色相＝左中右、
        # 深淺＝上中下，讓邊本身也能傳達物件在照片裡的相對位置，
        # 跟物件節點的排版位置互相印證。同一種顏色的邊一起畫（效能較好，
        # 也讓 matplotlib 圖層順序穩定）。
        contains_edges_by_color: Dict[str, List[Tuple[int, int]]] = {}
        for u, v, d in self.graph.edges(data=True):
            if not is_contains_etype(d.get("etype", "")):
                continue
            cell = self.graph.nodes[v].get("grid_cell", "center")
            contains_edges_by_color.setdefault(grid_cell_color(cell), []).append((u, v))
        for color, edgelist in contains_edges_by_color.items():
            nx.draw_networkx_edges(self.graph, pos, edgelist=edgelist, ax=ax,
                                    edge_color=color, style="dotted", width=1.1, alpha=0.8, arrows=False)

        adjacent_edges = _dedup_unordered_pairs(
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("etype") == ETYPE_ADJACENT
        )
        if adjacent_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=adjacent_edges, ax=ax,
                                    edge_color=self._COLOR_ADJACENT, width=1.0, alpha=0.7, arrows=False)

        same_object_edges = _dedup_unordered_pairs(
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("etype") == ETYPE_SAME_OBJECT
        )
        if same_object_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=same_object_edges, ax=ax,
                                    edge_color=self._COLOR_SAME_OBJECT, style="dashed", width=1.0, alpha=0.5,
                                    connectionstyle="arc3,rad=0.25", arrows=True, arrowstyle="-")

        # ── 節點方塊（依文字大小畫，不用固定大小的圓點/方塊）──
        #
        #    刻意排在走道邊之前畫：走道邊（下面）縮排要留多少，需要知道
        #    每個照片節點「連同它自己掛的所有物件節點」實際佔了多大範圍
        #    ——先把節點都畫出來、位置都定案，縮排的計算才有完整的資訊
        #    可以用（雖然縮排計算只需要 pos/sizes 數值、不需要真的等圖
        #    畫出來，但把「先決定所有節點在哪裡，再畫箭頭」的邏輯順序
        #    對應到程式碼順序，比較不容易漏算）。
        for n in real_photo_nodes:
            x, y = pos[n]
            w, h = sizes[n]
            self._draw_node_box(ax, x, y, w, h, face=self._COLOR_PHOTO, edge=self._COLOR_PHOTO_EDGE)
            ax.text(x, y, self._photo_label(n), ha="center", va="center",
                    fontsize=self._PHOTO_FONT, color=self._COLOR_PHOTO_TEXT, zorder=4)
        for n in virtual_nodes:
            x, y = pos[n]
            w, h = sizes[n]
            self._draw_node_box(ax, x, y, w, h, face=self._COLOR_VIRTUAL, edge=self._COLOR_VIRTUAL_EDGE, dashed=True)
            ax.text(x, y, self._photo_label(n), ha="center", va="center",
                    fontsize=self._PHOTO_FONT, color=self._COLOR_PHOTO, zorder=4)
        for n in object_nodes:
            x, y = pos[n]
            w, h = sizes[n]
            cell = self.graph.nodes[n].get("grid_cell", "center")
            tag_color = grid_cell_color(cell)
            # 外框改用九宮格顏色（比原本固定的暖金色外框更粗），跟 contains
            # 邊、九宮格色卡用同一套色彩對照，讓使用者不用查資料就能從
            # 顏色直接判斷這個物件在原始照片裡偏左/中/右、偏上/中/下。
            self._draw_node_box(ax, x, y, w, h, face=self._COLOR_OBJECT, edge=tag_color, linewidth=2.4)
            ax.text(x, y, self.graph.nodes[n].get("label", ""), ha="center", va="center",
                    fontsize=self._OBJECT_FONT, color=self._COLOR_OBJECT_TEXT, zorder=4)

        # ── 走道邊（第一種邊）：等所有節點都畫完才畫。
        #
        #    這一輪的重點：走道邊代表「照片節點跟照片節點之間的走道」，
        #    箭頭的起訖點依定義就是要連到照片節點本身，不是連到某個
        #    物件節點或整個群的外緣。上一輪把縮排量改成看「整個群」
        #    （照片＋它掛的物件節點）的範圍，雖然解決了箭頭穿過物件群
        #    的視覺問題，但也讓箭頭常常在離照片節點還很遠的地方就停住
        #    ——箭頭連到的是物件群的邊界，不是照片節點，偏離了走道邊
        #    「連接照片節點」的定義。改回**縮排只看照片節點自己的
        #    方塊**，箭頭一定會畫到緊貼照片節點的方塊邊緣才停，確保
        #    邊真正連到的是照片節點本身。
        #
        #    照片節點自己的物件群仍然可能出現在箭頭路徑附近（第十八輪
        #    把物件群改成往「拍攝方向」——也就是跟走道前進方向同側
        #    ——堆疊之後，比較容易跟「下一個」節點的方向重疊），這不
        #    影響「邊連到照片節點」這件事——箭頭的終點依然是照片節點
        #    的方塊邊緣，只是路徑中段可能會經過自己這一群的某個物件
        #    附近。這裡改成**走道邊的 zorder 比節點方塊低**（見
        #    `_draw_single_arrow` 的 `zorder=2.5`），節點方塊畫在箭頭
        #    上面，箭頭經過任何節點（不管是自己這一群的物件，還是理論
        #    上殘餘的個案）時會被該節點的方塊蓋住一小段，節點本身永遠
        #    完整看得到、不會被箭頭線條蓋到內容——這是這一輪的重點：
        #    「不遮擋節點」優先於「箭頭本身處處可見」，箭頭被自己的
        #    物件蓋住一小段是可以接受的，跟終點的連接處（緊貼照片節點
        #    方塊邊緣）並不會被蓋到，因為物件群跟照片節點之間本來就
        #    留有間距（見步驟 2 的 `row_margin`），這一小段間距通常就
        #    足夠讓箭頭終點附近保持淨空。
        #
        #    路徑上「非相鄰」的第三個群恰好擋在路徑中間的情況（折返時
        #    偶爾發生），另外交給 `_resolve_edge_node_crossings` 在排版
        #    階段把擋路的群推開處理（見 `_layout_positions`）——這個
        #    處理跟 zorder 沒有關係，是真的把節點挪開，避免無關的節點
        #    群整個卡在路徑中間看起來很奇怪（不只是視覺上被蓋住的問題）。
        walkway_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("etype") == ETYPE_WALKWAY and u != v
        ]
        facing_markers = [
            u for u, v, d in self.graph.edges(data=True) if d.get("etype") == ETYPE_WALKWAY and u == v
        ]
        pt_per_data_unit = self._INCH_PER_DATA_UNIT * 72.0

        def _box_shrink_pt(node_id, dx: float, dy: float) -> float:
            """照片節點自己方塊沿 (dx,dy) 方向的投影半寬（points），用來
            當箭頭在這一端要縮排多少——縮到照片節點自己的方塊邊緣，
            確保箭頭真的連到照片節點本身（第四種邊的定義），不是連到
            某個物件節點或整個群的外緣。"""
            w, h = sizes[node_id]
            half_extent = (w / 2.0) * abs(dx) + (h / 2.0) * abs(dy)
            return half_extent * pt_per_data_unit + 2.0  # 多留 2pt 緩衝，貼著邊緣看起來才乾淨

        def _draw_single_arrow(x0, y0, x1, y1, shrinkA, shrinkB, *, color, lw, mutation_scale, zorder) -> None:
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                 shrinkA=shrinkA, shrinkB=shrinkB, mutation_scale=mutation_scale),
                zorder=zorder,
            )

        # 節點方塊的 zorder 是 3（見下面畫節點的段落），這裡用比它低的
        # 2.5，確保箭頭經過任何節點時會被該節點的方塊蓋住，節點本身
        # 不會被箭頭線條蓋到內容。
        _WALKWAY_ZORDER = 2.5

        for u, v in walkway_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            seg_len = math.hypot(x1 - x0, y1 - y0) or 1.0
            dx, dy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
            text = self._walkway_edge_data(u, v).get("direction", "") or ""
            turn = self._parse_turn_degrees(text)
            color = self._COLOR_WALKWAY_TURN if turn != 0.0 else self._COLOR_WALKWAY
            _draw_single_arrow(
                x0, y0, x1, y1,
                _box_shrink_pt(u, dx, dy), _box_shrink_pt(v, dx, dy),
                color=color, lw=2.6, mutation_scale=16, zorder=_WALKWAY_ZORDER,
            )
            if turn != 0.0:
                # 這一步「有轉彎」：走道邊本身（上面畫的）代表轉彎之後、
                # 走到下一張照片的方向，不等於起點這張照片實際拍攝的
                # 方向。額外在起點旁邊畫一小段拍攝方向指示（跟路徑終點
                # 的拍照方向標記同一種畫法），明確標出「這裡的照片實際
                # 是面向哪個方向拍的」，跟移動方向的箭頭區分開來。
                fh = math.radians(heading.get(u, 0.0))
                fdx, fdy = math.sin(fh), math.cos(fh)
                stub_len = 1.1
                _draw_single_arrow(
                    x0, y0, x0 + fdx * stub_len, y0 + fdy * stub_len,
                    _box_shrink_pt(u, fdx, fdy), 0.0,
                    color=self._COLOR_WALKWAY_TURN, lw=1.6, mutation_scale=11, zorder=_WALKWAY_ZORDER,
                )

        for node_id in facing_markers:
            x, y = pos[node_id]
            rad = math.radians(heading.get(node_id, 0.0))
            dx, dy = math.sin(rad), math.cos(rad)
            stub_len = 1.3
            _draw_single_arrow(
                x, y, x + dx * stub_len, y + dy * stub_len,
                _box_shrink_pt(node_id, dx, dy), 0.0,
                color=self._COLOR_WALKWAY, lw=2.2, mutation_scale=14, zorder=_WALKWAY_ZORDER,
            )

        # ── 使用者目前位置：畫在該照片節點右上角的小星星徽章，
        #    不蓋住節點本身的文字（舊版是蓋在正中央的巨大星星）──
        if self.current_position is not None and self.current_position in pos:
            x, y = pos[self.current_position]
            w, h = sizes[self.current_position]
            ax.scatter([x + w / 2.0], [y + h / 2.0], marker="*", s=320,
                       c=self._COLOR_CURRENT, edgecolors="white", linewidths=0.8, zorder=6)

        legend_handles = [
            mpatches.Patch(color=self._COLOR_PHOTO, label="照片節點"),
            mpatches.Patch(color=self._COLOR_WALKWAY, label="走道邊·直走（箭頭＝拍照方向）"),
        ]
        if has_turn_edge:
            legend_handles.append(
                mpatches.Patch(color=self._COLOR_WALKWAY_TURN, label="走道邊·這步有轉彎（起點旁短箭頭＝實際拍攝方向）")
            )
        if virtual_nodes:
            legend_handles.append(mpatches.Patch(color=self._COLOR_VIRTUAL, label="虛假照片節點（未走訪）"))
        if has_facing_marker:
            legend_handles.append(
                plt.Line2D([0], [0], color=self._COLOR_WALKWAY, lw=2.2, marker=">",
                           markersize=6, label="路徑終點的拍照方向")
            )
        legend_handles += [
            mpatches.Patch(facecolor=self._COLOR_OBJECT, edgecolor=self._COLOR_OBJECT_EDGE,
                            label="物件節點（外框顏色＝九宮格位置，見下方色卡）"),
            mpatches.Patch(color=self._COLOR_ADJACENT, label="相鄰物件邊"),
            mpatches.Patch(color=self._COLOR_SAME_OBJECT, label="跨照片同物件邊"),
        ]
        legend_handles.append(
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=self._COLOR_CURRENT,
                       markeredgecolor="white", markersize=13, label="使用者目前位置（右上角星號）")
        )
        # 圖例畫在標題正下方、圖的正上方的獨立橫條裡（fig.legend，不是
        # ax.legend），這樣不管圖多小都不會疊到節點上。
        fig.legend(handles=legend_handles, loc="upper center", ncol=legend_ncol,
                   fontsize=7.3, framealpha=0.95,
                   bbox_to_anchor=(0.5, 1.0 - title_in / fig_h))
        fig.suptitle(f"拓樸地圖：{self.place_name or '(未命名場所)'}",
                     fontsize=12, y=1.0 - (title_in * 0.35) / fig_h)

        # ── 九宮格色卡：畫在圖例橫條下方、主圖上方的獨立橫幅裡，用 3x3
        #    色塊＋「上/中/下」「左/中/右」標籤，把 contains 邊跟物件節點
        #    外框用的顏色對照表直接畫出來，不用另外查文件才知道顏色代表
        #    哪一格。色塊本身用固定英吋大小（不跟著資料座標縮放），確保
        #    圖不管大小，色卡永遠一樣好辨識。──
        if show_grid_legend:
            strip_bottom = top
            strip_top = 1.0 - (title_in + legend_in) / fig_h
            strip_h_fig = strip_top - strip_bottom

            swatch_in = 0.30
            swatch_gap_in = 0.05
            grid_w_in = swatch_in * 3 + swatch_gap_in * 2
            grid_h_in = grid_w_in
            label_pad_in = 0.24   # 左側留給「上/中/下」列標籤、上方留給「左/中/右」欄標籤
            block_w_in = grid_w_in + label_pad_in
            block_h_in = grid_h_in + label_pad_in

            gx0 = margin_in / fig_w
            gy0 = strip_bottom + max((strip_h_fig - block_h_in / fig_h) / 2.0, 0.0)
            grid_ax = fig.add_axes((gx0, gy0, block_w_in / fig_w, block_h_in / fig_h))
            grid_ax.set_xlim(0, block_w_in)
            grid_ax.set_ylim(0, block_h_in)
            grid_ax.axis("off")

            row_zh = ("上", "中", "下")   # row 0/1/2，對應照片裡的上/中/下
            col_zh = ("左", "中", "右")   # col 0/1/2，對應照片裡的左/中/右
            row_prefix = ("top", "", "bottom")
            for r in range(3):
                y = label_pad_in + (2 - r) * (swatch_in + swatch_gap_in)
                grid_ax.text(label_pad_in * 0.32, y + swatch_in / 2.0, row_zh[r],
                              ha="center", va="center", fontsize=7.5, color="#444444")
                for c in range(3):
                    col_name = ("left", "center", "right")[c]
                    cell = col_name if not row_prefix[r] else f"{row_prefix[r]}_{col_name}"
                    x = label_pad_in + c * (swatch_in + swatch_gap_in)
                    grid_ax.add_patch(mpatches.FancyBboxPatch(
                        (x, y), swatch_in, swatch_in,
                        boxstyle="round,pad=0.01,rounding_size=0.03",
                        linewidth=0.6, edgecolor="#666666", facecolor=grid_cell_color(cell),
                    ))
                    if r == 0:
                        grid_ax.text(x + swatch_in / 2.0, label_pad_in + grid_h_in + 0.02, col_zh[c],
                                      ha="center", va="bottom", fontsize=7.5, color="#444444")
            # 色卡右側的文字說明（放在同一條橫幅裡，色卡佔用的寬度之外）
            caption_x = gx0 + block_w_in / fig_w + 0.012
            caption_y = strip_bottom + strip_h_fig / 2.0
            fig.text(caption_x, caption_y,
                      "九宮格色卡：物件在原始照片中的位置\n"
                      "（色相＝左／中／右，深淺＝上／中／下，越深＝越靠照片下緣）",
                      ha="left", va="center", fontsize=7.3, color="#444444")

        ax.axis("off")
        # 不呼叫 set_aspect/autoscale_view：畫布英吋大小已經直接照資料座標
        # 範圍（乘上 _INCH_PER_DATA_UNIT）算好，兩者比例本來就是對的，
        # 再讓 matplotlib 自動調整反而會破壞這個精確對應。

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()
