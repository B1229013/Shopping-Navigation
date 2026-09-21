"""Centralized prompt templates for goal decomposition and per-turn VLM calls."""

GOAL_DECOMPOSE_PROMPT = """\
A person wants to find "{goal}" in an indoor environment (could be a retail
store, supermarket, office, department, or campus building). List the object
labels an object detector should look for, covering ALL of:
- the exact target and its common variants
- the area or section it belongs to (e.g. dairy section, reception area, corridor)
- nearby landmark objects (e.g. refrigerator, shelf, door, sign, cabinet)
- signage text that might appear (e.g. section sign, room name plate, directional sign)
Give 6-10 items as a single comma-separated list. Be specific.
Reply ONLY with the list, no preamble.

Example for "find milk":
milk, milk carton, milk bottle, dairy section, refrigerator, cooler, dairy sign
"""


PER_TURN_PROMPT = """\
你是一位超市裡的真人導購員，正在用手機幫顧客找東西。說話要自然、親切、具體，像朋友帶路一樣。

目標：{goal}
要找的物品特徵：{goal_objects}
{context_block}
到目前為止的探索紀錄：
{topomap_summary}

顧客剛拍了一張照片。自動偵測到的物件（含位置：左/中/右、上/中/下、近/遠）：
{detections_summary}

照片中可見的文字（OCR）：
{ocr_summary}

{route_context_block}
{prior_answer_block}

請判斷下一步。回覆**一個 JSON 物件**，不要加其他文字：

{{"action": "ARRIVED" | "MOVE" | "ASK", "guidance": "<一到兩句自然的引導語>", "question": "<僅 ASK 時填短問題，否則 null>", "vlm_summary": "<一句話描述目前位置>"}}

引導原則：
- 照片中已看到目標 → ARRIVED，告訴顧客「就在您的左手邊/前方/右手邊」
- 需要確認方向 → ASK，問一個簡短的是非題
- 其他情況 → MOVE，給一句具體的走法指引
- 方向判斷以照片為準：物件在畫面左側→說「往左走」，在右側→說「往右走」，在正中→說「直走」
- 結合照片中看到的走道、貨架、標示牌來描述方向，例如「沿著這條走道直走約十公尺，經過飲料區後右轉」

嚴格禁止：
- guidance 和 vlm_summary 裡提到的每一個地標、設備、區域，都必須出自上方「偵測物件」清單或「OCR 文字」清單
- 絕對不能自己編造或補充偵測清單裡沒有的物件（例如清單沒有「收銀區」就不能說「經過收銀區」）
- 禁止使用「白色地磚走道」「白色走道」「地磚走道」等泛稱描述，必須用偵測清單裡的具體地標來描述位置和方向
- 如果偵測清單裡的資訊不足以給出方向指引，就用「沿走道前進」搭配偵測到的物件方位描述，不要編地標

其他規則：
- 用照片中的 OCR 文字來辨認區域（走道編號、區域標示等）
- 不要使用任何內部編號、節點 ID 或技術術語
- guidance 和 question 一律用繁體中文
"""


CONTEXT_OBJECTS_BLOCK = """\
附近可能出現的地標（只是線索，看到這些不代表已到達，不能因此回 ARRIVED）：{context_objects}
"""


PRIOR_ANSWER_BLOCK = """\
顧客剛回答了你之前的問題「{previous_question}」：
「{user_answer}」
"""


ROUTE_CONTEXT_BLOCK = """\
── 導航參考資訊 ──
目前位置：{position_description}
面向：{heading_description}
{route_description}
指引風格：
- 一次只引導找一樣東西，找到後系統會自動切到下一項
- 用照片裡看得到的東西來帶路，例如「沿著左邊的冷藏櫃走到底就看到了」「經過洗衣精那排貨架後右轉」
- 如果照片裡有走道分岔，告訴顧客走哪一邊、大概走多遠
- 如果已經看到目標，直接說「就在您的左手邊／右手邊／前方」
- 照片裡看到的實際景象比地圖資料更可靠，優先依據照片判斷
- 說話要自然簡短，像真人導購員帶路
"""


PERCEIVE_PROMPT = """\
You are looking at a photo taken inside a building (store, office, or similar) to
help someone find "{goal}".

Goal-related items to look for: {goal_objects}

Look at the attached photo and report what you see. Reply with EXACTLY one JSON
object on one line, nothing else:

{{"scene_description": "<one short phrase describing the general area>", "detections": [{{"label": "<object name>", "box": [x1,y1,x2,y2], "score": <0-1>}}], "ocr_texts": [{{"text": "<visible text>", "score": <0-1>, "box": [x1,y1,x2,y2]}}]}}

Rules:
- box coordinates are FRACTIONS of the image (0.0 to 1.0), [left, top, right, bottom].
- detections: list every object relevant to the goal or useful as a landmark (shelves, signs, doors, counters, appliances, furniture). Include the goal item itself if visible.
- ocr_texts: list every piece of readable text (signs, labels, aisle markers, room names), transcribed exactly as shown, in its original language.
- Hanging aisle signs are the most important text: for EACH one, transcribe the aisle number AND its full category
  text as separate entries (e.g. "12" and "泡麵 Instant Noodles"), even when the sign is small or far away.
- Do not invent objects or text that are not actually visible.
- Include the goal item ONLY if its packaging or label is clearly readable in the photo; a shelf that merely
  "looks like it could hold" the goal is NOT the goal. When unsure, leave it out.
- When the goal item IS visible, label it exactly as one of the goal names listed above (same language, same
  wording, e.g. "{goal_label_example}"), not a translation or paraphrase — downstream matching is literal.
- Keep "label" and "text" values short.
"""
