"""
scene.py — 場景空間格式化與到達驗證模組 / Scene Spatial Formatting & Arrival Verification

功能說明 (Purpose):
    純工具模組（不載入任何 AI 模型），負責：
    1. 將偵測/OCR 的 bounding box 轉換成人類可讀的空間描述
       Convert detection/OCR bounding boxes into human-readable spatial descriptions
       例: "refrigerator (91%) - right middle, near"
    2. OCR 文字與目標的比對（含同義詞擴展）
       Match OCR text against goal (with synonym expansion)
    3. VLM 判定 ARRIVED 時的閘門驗證（防止無證據的誤判）
       Gate ARRIVED decisions to prevent false claims without evidence

被誰呼叫 (Called by):
    - server.py: format_detections(), format_ocr() → 組合 VLM prompt
    - server.py: match_ocr_to_goal() → 檢查 OCR 是否命中目標
    - server.py: verify_arrival() → ARRIVED 閘門驗證

依賴 (Dependencies):
    - difflib: 模糊字串比對（內建模組）
    - models.py: VLMAction, VLMResponse 資料結構

設計原則 (Design principle):
    不匯入 torch/模型，保持輕量，方便單元測試
    No torch/model imports, stays lightweight, easy to unit test
"""
from __future__ import annotations

import difflib

from server.models import VLMAction, VLMResponse


# ══════════════════════════════════════════════════════════════════════════════
# 同義詞表 / Synonym Dictionary
# ══════════════════════════════════════════════════════════════════════════════

# 雙語（英文/中文）類別同義詞，讓 OCR 辨識到的標示文字能對應到導航目標
# Bilingual (EN/ZH) category synonyms so OCR'd signs can match navigation goals
# 例: 目標是 "milk" → OCR 看到 "dairy" 或 "乳製品" 也算命中
# Example: goal is "milk" → OCR seeing "dairy" or "乳製品" also counts as match
SYNONYMS: dict[str, list[str]] = {
    "milk": ["dairy", "refrigerated", "牛奶", "鮮奶", "乳製品"],
    "cheese": ["dairy", "refrigerated", "起司", "乳酪", "乳製品"],
    "egg": ["eggs", "dairy", "refrigerated", "蛋", "雞蛋"],
    "eggs": ["egg", "dairy", "refrigerated", "蛋", "雞蛋"],
    "refrigerator": ["fridge", "freezer", "冰箱", "冷藏", "冷凍"],
    "fire extinguisher": ["extinguisher", "滅火器", "消防"],
    "bread": ["bakery", "baked", "麵包", "烘焙"],
    "vegetable": ["produce", "vegetables", "fresh", "蔬菜", "生鮮"],
    "fruit": ["produce", "fruits", "fresh", "水果", "生鮮"],
    "produce": ["vegetables", "fruit", "fresh", "蔬果", "生鮮"],
    "frozen": ["freezer", "frozen foods", "冷凍"],
    "meat": ["butcher", "肉", "肉品", "生鮮"],
    "drink": ["beverage", "beverages", "drinks", "soda", "juice", "water", "飲料"],
    "snack": ["snacks", "零食", "餅乾"],
    "checkout": ["cashier", "checkout", "register", "結帳", "收銀"],
    "exit": ["出口"],
    "entrance": ["入口"],
    "toilet": ["restroom", "washroom", "wc", "廁所", "洗手間"],
    "衛生紙": ["toilet paper", "tissue", "tissues", "家用", "清潔", "家用清潔", "紙巾"],
    "toilet paper": ["tissue", "tissues", "衛生紙", "紙巾", "家用", "清潔", "家用清潔"],
    "tissue": ["tissues", "toilet paper", "衛生紙", "紙巾", "面紙"],
    "洗衣精": ["laundry", "detergent", "家用", "清潔", "家用清潔"],
    "清潔用品": ["cleaning", "household", "家用", "清潔", "家用清潔"],
}

# 模糊比對閾值：OCR 文字與目標詞的相似度 >= 此值即視為匹配
# Fuzzy match threshold: OCR text vs goal term similarity >= this value = match
OCR_MATCH_RATIO = 0.85

# 物件面積佔整張照片的比例 >= 此值 → 判定為「近處」
# Object area / image area >= this → classified as "near"
NEAR_AREA_RATIO = 0.15

# 空間位置標籤 / Spatial position labels
_HORIZONTAL = ("left", "center", "right")     # 水平三等分 / horizontal thirds
_VERTICAL = ("top", "middle", "bottom")       # 垂直三等分 / vertical thirds


# ══════════════════════════════════════════════════════════════════════════════
# 空間位置計算 / Spatial Position Calculation
# ══════════════════════════════════════════════════════════════════════════════

def _attr(item, name):
    """
    通用欄位讀取器：支援 dict 和 dataclass 兩種格式
    Universal field reader: supports both dict and dataclass formats

    設計原因 (Why):
        偵測結果有時是 Detection dataclass，有時是 dict，
        這個函式讓後續程式碼不用區分
        Detection results are sometimes Detection dataclass, sometimes dict.
        This function lets downstream code work with either.
    """
    return item[name] if isinstance(item, dict) else getattr(item, name)


def _third(frac: float) -> int:
    """
    將 0~1 的比例值分成三等分 / Divide a 0~1 fraction into thirds

    0.0 ~ 0.33 → 0 (left/top)
    0.33 ~ 0.66 → 1 (center/middle)
    0.66 ~ 1.0 → 2 (right/bottom)
    """
    if frac < 1 / 3:
        return 0
    if frac < 2 / 3:
        return 1
    return 2


def _horizontal_side(center_x: float, img_w: int) -> str:
    """
    根據物件中心 x 座標判斷在畫面的左/中/右
    Determine left/center/right based on object center x coordinate

    例: center_x=100, img_w=300 → 100/300=0.33 → "center"
    """
    if not img_w:
        return "center"
    return _HORIZONTAL[_third(center_x / img_w)]


def _position(box: list[float], img_w: int, img_h: int) -> tuple[str, str, str]:
    """
    從偵測框計算三維空間位置描述 / Calculate 3D spatial description from detection box

    參數 (Parameters):
        box:   [x1, y1, x2, y2] 偵測框座標 / detection box coordinates
        img_w: 影像寬度 / image width
        img_h: 影像高度 / image height

    回傳 (Returns):
        (horizontal, vertical, depth) 三元組
        - horizontal: "left" / "center" / "right"（依中心 x 位置）
        - vertical:   "top" / "middle" / "bottom"（依中心 y 位置）
        - depth:      "near" / "far"（依面積佔比判斷遠近）

    深度判斷邏輯 (Depth estimation logic):
        物件在照片中佔的面積越大 → 離相機越近
        用面積佔比 >= NEAR_AREA_RATIO (15%) 作為「近處」的閾值
        Larger area ratio in photo → closer to camera
        Area ratio >= NEAR_AREA_RATIO (15%) = "near"
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # 中心座標 / center coordinates

    # 水平位置 / Horizontal position
    horizontal = _horizontal_side(cx, img_w)

    # 垂直位置 / Vertical position
    vertical = _VERTICAL[_third(cy / img_h)] if img_h else "middle"

    # 深度估計：面積佔比 / Depth estimation: area ratio
    area_ratio = (abs(x2 - x1) * abs(y2 - y1)) / (img_w * img_h) if img_w and img_h else 0.0
    depth = "near" if area_ratio >= NEAR_AREA_RATIO else "far"

    return horizontal, vertical, depth


# ══════════════════════════════════════════════════════════════════════════════
# 格式化輸出（給 VLM prompt 用）/ Formatting (for VLM prompt)
# ══════════════════════════════════════════════════════════════════════════════

def format_detections(detections, img_w: int, img_h: int) -> str:
    """
    將偵測結果格式化為帶空間位置的文字描述 / Format detections with spatial positions

    範例輸出 (Example output):
        "refrigerator (91%) - right middle, near; door (85%) - left top, far"

    這段文字會被放入 VLM 的 prompt 中，讓 VLM 知道：
    - 畫面中有什麼物件
    - 每個物件在畫面的哪個位置
    - 每個物件離相機多遠
    This text goes into the VLM prompt so it knows:
    - What objects are in the frame
    - Where each object is positioned
    - How far each object is from camera
    """
    if not detections:
        return "(none)"
    parts = []
    for d in detections:
        h, v, depth = _position(_attr(d, "box"), img_w, img_h)
        parts.append(f"{_attr(d, 'label')} ({_attr(d, 'score'):.0%}) - {h} {v}, {depth}")
    return "; ".join(parts)


def _bbox_center(bbox: list[list[float]]) -> tuple[float, float]:
    """
    計算四邊形 bbox 的中心點 / Calculate center of quadrilateral bbox

    OCR 的 bbox 是四個角點，取 x 和 y 的平均值作為中心
    OCR bbox has four corners, average x and y for center
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def format_ocr(ocr_results, img_w: int, img_h: int) -> str:
    """
    將 OCR 結果格式化為帶位置的文字描述 / Format OCR results with position

    範例輸出 (Example output):
        '"E301" (95%) - right; "實驗室" (87%) - center'

    OCR 只標示水平位置（左/中/右），因為文字通常沿水平方向排列，
    垂直位置對導航參考價值較低
    OCR only shows horizontal position (left/center/right) because text is
    usually horizontal, vertical position is less useful for navigation
    """
    if not ocr_results:
        return "(no text detected)"
    parts = []
    for r in ocr_results:
        cx, _ = _bbox_center(r.bbox)
        side = _horizontal_side(cx, img_w)
        parts.append(f'"{r.text}" ({r.confidence:.0%}) - {side}')
    return "; ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# OCR 目標比對 / OCR Goal Matching
# ══════════════════════════════════════════════════════════════════════════════

def _goal_terms(goal_objects) -> set[str]:
    """
    擴展目標詞為完整的同義詞集合 / Expand goal words into full synonym set

    流程 (Process):
        1. 每個目標詞加入集合
        2. 查 SYNONYMS 表，加入所有同義詞
        3. 反向查找：如果目標詞本身是某個 key 的同義詞，也加入該 key 的所有同義詞

    範例 (Example):
        goal_objects = ["milk"]
        → {"milk", "dairy", "refrigerated", "牛奶", "鮮奶", "乳製品",
           "cheese", "起司", "乳酪", "egg", ...}

    這個擴展確保：OCR 看到 "乳製品" 能匹配到目標 "milk"
    This expansion ensures: OCR seeing "乳製品" matches goal "milk"
    """
    terms: set[str] = set()
    for g in goal_objects:
        gl = g.lower().strip()
        if not gl:
            continue
        terms.add(gl)
        # 正向查找：目標詞的同義詞 / Forward lookup: synonyms of goal word
        terms.update(s.lower() for s in SYNONYMS.get(gl, []))
        # 反向查找：如果目標詞是別人的同義詞 / Reverse lookup: if goal is someone's synonym
        for key, syns in SYNONYMS.items():
            low = [s.lower() for s in syns]
            if gl == key or gl in low:
                terms.add(key)
                terms.update(low)
    return terms


def match_ocr_to_goal(ocr_results, goal_objects) -> list[str]:
    """
    找出 OCR 結果中與目標相關的文字 / Find OCR results related to the goal

    比對策略 (Matching strategy):
        對每個 OCR 文字，逐一與擴展後的目標詞比對：
        1. 子字串包含：term in text 或 text in term
           Substring containment
        2. 模糊比對：SequenceMatcher 相似度 >= 0.85
           Fuzzy match: SequenceMatcher ratio >= 0.85
        任一條件成立即視為命中
        Either condition = match

    回傳 (Returns):
        命中的描述列表，如 ['"乳製品" (relates to your goal "milk")']
        List of match descriptions

    重要性 (Importance):
        在商店/室內環境中，標示牌是最可靠的位置信號
        OCR 命中比物件偵測更可信（標示不會被誤偵測）
        In stores/indoor, signs are the most reliable location signal
        OCR match is more trustworthy than object detection
    """
    terms = _goal_terms(goal_objects)
    out: list[str] = []
    seen: set[str] = set()  # 去重 / deduplicate

    for r in ocr_results:
        text = r.text.lower().strip()
        if not text:
            continue
        for term in terms:
            # 三種比對方式 / Three matching methods
            hit = (term in text or           # 目標詞是 OCR 文字的子字串 / term is substring of text
                   text in term or           # OCR 文字是目標詞的子字串 / text is substring of term
                   difflib.SequenceMatcher(None, term, text).ratio() >= OCR_MATCH_RATIO)  # 模糊比對 / fuzzy
            if hit:
                if r.text not in seen:
                    out.append(f'"{r.text}" (relates to your goal "{term}")')
                    seen.add(r.text)
                break  # 一個 OCR 結果只匹配一次 / Each OCR result matches at most once
    return out


def confirm_zone_by_ocr(ocr_texts, goal_objects) -> bool:
    """
    用 OCR 文字確認是否在目標區域 / Confirm if in target zone via OCR text

    與 match_ocr_to_goal 類似，但：
    Similar to match_ocr_to_goal but:
        - 輸入是純文字列表（非 OCRResult 物件）/ Input is plain text list (not OCRResult)
        - 回傳布林值（是/否）/ Returns boolean (yes/no)
        - 用在 arrival 驗證的輔助判斷 / Used as supplementary arrival verification

    應用場景 (Use case):
        VLM 判定 ARRIVED，但偵測分數不夠高時，
        如果 OCR 確認在目標區域 → 增強 ARRIVED 的可信度
        When VLM says ARRIVED but detection score is borderline,
        OCR zone confirmation strengthens the ARRIVED claim
    """
    terms = _goal_terms(goal_objects)
    for text in ocr_texts:
        t = (text or "").lower().strip()
        if not t:
            continue
        for term in terms:
            if term in t or t in term or \
                    difflib.SequenceMatcher(None, term, t).ratio() >= OCR_MATCH_RATIO:
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 到達驗證閘門 / Arrival Verification Gate
# ══════════════════════════════════════════════════════════════════════════════

def _label_matches_goal(label: str, goal_objects) -> bool:
    """
    檢查偵測標籤是否與目標相關 / Check if detection label relates to goal

    用子字串雙向包含判斷：
    Uses bidirectional substring containment:
        "refrigerator" vs "fridge" → False（不包含）
        "milk bottle" vs "milk" → True（milk in milk bottle）
    """
    label = label.lower()
    for g in goal_objects:
        g = g.lower()
        if g and (g in label or label in g):
            return True
    return False


def verify_arrival(resp: VLMResponse, detections, ocr_matches, goal_objects,
                   min_score: float, prior_was_confirm: bool = False,
                   goal_verified: bool | None = None) -> VLMResponse:
    """
    ARRIVED 閘門驗證：防止 VLM 在沒有足夠證據時錯誤宣告到達
    ARRIVED gate: prevent VLM from falsely declaring arrival without evidence

    為什麼需要這個 (Why needed):
        VLM 有時會「過度樂觀」，看到遠處模糊的相似物件就宣告到達。
        這個閘門要求至少有一種客觀證據才放行 ARRIVED。
        VLM sometimes "over-optimistically" declares arrival seeing a distant
        similar object. This gate requires at least one objective evidence.

    閘門邏輯 (Gate logic):
        VLM 說 ARRIVED → 檢查以下條件（任一成立即放行）：
        VLM says ARRIVED → check these conditions (any one = pass):

        1. goal_verified == True（裁切驗證通過）
           Crop verification passed → strongest evidence, always pass

        2. goal_verified == False（裁切驗證明確否認）
           Crop verification explicitly denied → always block, downgrade to ASK

        3. goal_verified == None（沒做裁切驗證）→ 檢查其他證據：
           No crop verification → check other evidence:
           a. 有目標物件偵測且分數 >= min_score → 放行
              Goal object detected with score >= min_score → pass
           b. OCR 匹配到目標相關文字 → 放行
              OCR matched goal-related text → pass
           c. 都沒有 → 降級為 ASK（請使用者確認）
              Neither → downgrade to ASK (ask user to confirm)

    特殊情況 (Special case):
        prior_was_confirm == True：使用者已經在回答確認問題了，
        此時 VLM 再次說 ARRIVED 應該信任，避免無限迴圈
        User is already answering a confirm question,
        trust VLM's ARRIVED to avoid infinite loop

    參數 (Parameters):
        resp:              VLM 的回應 / VLM response
        detections:        當前照片的偵測結果 / Current photo detections
        ocr_matches:       OCR 匹配結果列表 / OCR match results
        goal_objects:       目標物件詞列表 / Goal object word list
        min_score:          最低偵測分數閾值 / Minimum detection score threshold
        prior_was_confirm:  上一輪是否是確認問題 / Was last turn a confirm question
        goal_verified:      裁切驗證結果 / Crop verification result
                           True=確認 / False=否認 / None=沒做

    回傳 (Returns):
        原樣回傳（ARRIVED 放行）或降級為 ASK 的 VLMResponse
        Original (ARRIVED passed) or downgraded ASK VLMResponse
    """
    # 非 ARRIVED 的回應直接放行 / Non-ARRIVED responses pass through
    if resp.action != VLMAction.ARRIVED or prior_was_confirm:
        return resp

    # 裁切驗證結果優先 / Crop verification takes priority
    if goal_verified is True:
        return resp          # 裁切確認 → 放行 / Crop confirmed → pass
    if goal_verified is False:
        return _confirm_question(resp)  # 裁切否認 → 降級 / Crop denied → downgrade

    # 檢查偵測和 OCR 證據 / Check detection and OCR evidence
    detection_ok = any(
        _attr(d, "score") >= min_score and _label_matches_goal(_attr(d, "label"), goal_objects)
        for d in detections
    )
    if detection_ok or ocr_matches:
        return resp          # 有客觀證據 → 放行 / Objective evidence → pass

    # 沒有足夠證據 → 降級為確認問題
    # Insufficient evidence → downgrade to confirmation question
    return _confirm_question(resp)


def _confirm_question(resp: VLMResponse) -> VLMResponse:
    """
    將 ARRIVED 降級為 ASK 確認問題
    Downgrade ARRIVED to an ASK confirmation question

    保留原本的 vlm_summary（場景描述），只改 action 和 question
    Keeps original vlm_summary (scene description), only changes action and question
    """
    return VLMResponse(
        action=VLMAction.ASK,
        guidance="I think you may have reached it, but I'm not certain.",
        question="Can you confirm you can see what you're looking for right in front of you?",
        vlm_summary=resp.vlm_summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 反向閘門：MOVE 升級 / Reverse Gate: MOVE Upgrade
# ══════════════════════════════════════════════════════════════════════════════

# 反向閘門的偵測分數閾值 / Reverse gate detection score threshold
REVERSE_GATE_MIN_SCORE = 0.50

def check_missed_arrival(
    resp: VLMResponse,
    detections,
    ocr_matches: list,
    goal_objects,
    min_score: float = REVERSE_GATE_MIN_SCORE,
) -> VLMResponse:
    """
    反向閘門：VLM 判 MOVE 但有強烈到達證據時升級為 ARRIVED 或 ASK
    Reverse gate: upgrade MOVE to ARRIVED/ASK when strong arrival evidence exists

    為什麼需要這個 (Why needed):
        verify_arrival() 只在 VLM 回 ARRIVED 時介入（防止誤報），
        但 VLM 有時太保守，看到目標就在眼前卻回 MOVE（漏報）。
        這個反向閘門在 verify_arrival() 之後執行，檢查是否有遺漏的到達。
        verify_arrival() only acts on ARRIVED (prevents false positives),
        but VLM is sometimes too conservative (false negatives).
        This reverse gate catches missed arrivals.

    升級條件 (Upgrade conditions):
        1. VLM 回應是 MOVE（非 ARRIVED/ASK）
        2. 偵測中有目標物件且分數 >= min_score
        3. 該偵測的 position 標籤含 "near"

        若同時有 OCR 匹配 → 升級為 ARRIVED（雙重證據）
        若只有偵測 → 升級為 ASK（讓使用者確認）

    參數 (Parameters):
        resp:          VLM 的回應 / VLM response
        detections:    偵測結果（dict 或 dataclass，需有 position 欄位）
                       Detections (dict or dataclass, must have position field)
        ocr_matches:   OCR 匹配結果 / OCR match results
        goal_objects:  目標物件詞列表 / Goal object word list
        min_score:     最低偵測分數閾值 / Minimum detection score threshold

    回傳 (Returns):
        原樣回傳（MOVE 維持）或升級為 ARRIVED/ASK 的 VLMResponse
        Original (MOVE kept) or upgraded ARRIVED/ASK VLMResponse
    """
    # 只處理 MOVE / Only handle MOVE
    if resp.action != VLMAction.MOVE:
        return resp

    # 檢查是否有「近處的目標物件」偵測
    # Check for "near goal object" detection
    def _get_pos(d):
        """Get position string from dict or dataclass."""
        if isinstance(d, dict):
            return d.get("position", "")
        return getattr(d, "position", "")

    has_near_goal = any(
        _attr(d, "score") >= min_score
        and "near" in _get_pos(d).lower()
        and _label_matches_goal(_attr(d, "label"), goal_objects)
        for d in detections
    )

    if not has_near_goal:
        return resp

    # 有 near goal + OCR 匹配 → 雙重證據，升級為 ARRIVED
    # Near goal + OCR match → dual evidence, upgrade to ARRIVED
    if ocr_matches:
        return VLMResponse(
            action=VLMAction.ARRIVED,
            guidance=resp.guidance,
            question=None,
            vlm_summary=resp.vlm_summary,
        )

    # 有 near goal 但無 OCR → 單一證據，升級為 ASK
    # Near goal but no OCR → single evidence, upgrade to ASK
    return VLMResponse(
        action=VLMAction.ASK,
        guidance=resp.guidance,
        question="偵測到目標物品就在附近，請確認您是否已經到達？",
        vlm_summary=resp.vlm_summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 輔助工具 / Utility Functions
# ══════════════════════════════════════════════════════════════════════════════

def detection_sides(detections, img_w: int) -> list[str]:
    """
    列出偵測到物件的水平方向（去重，保持順序）
    List horizontal sides where objects were detected (deduplicated, ordered)

    用途 (Use case):
        評估工具用來檢查 VLM 的 MOVE 指引是否指向有物件的方向
        Evaluation harness checks if MOVE guidance points to a side with objects

    範例 (Example):
        detections 在畫面右邊和中間 → ["right", "center"]
    """
    sides: list[str] = []
    for d in detections:
        x1, _, x2, _ = _attr(d, "box")
        side = _horizontal_side((x1 + x2) / 2, img_w)
        if side not in sides:
            sides.append(side)
    return sides
