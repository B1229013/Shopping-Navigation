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
You are guiding a person through an indoor environment using their phone camera.

Goal: "{goal}"
Goal-related items to look for: {goal_objects}

What's happened so far:
{topomap_summary}

The person just uploaded the attached photo. In it, automatic detection found
(each item lists its position in the frame - left/center/right, top/middle/bottom,
and near/far):
{detections_summary}

Text visible in the photo (OCR), each with its side of the frame:
{ocr_summary}

{route_context_block}
{prior_answer_block}

Decide the next step. Reply with EXACTLY one JSON object on one line, nothing else:

{{"action": "ARRIVED" | "MOVE" | "ASK", "guidance": "<one or two sentences>", "question": "<short question, only if ASK, else null>", "vlm_summary": "<one phrase summarizing the location>"}}

Rules:
- ARRIVED only if the goal item is clearly visible in the photo (point at it in `guidance`).
- ASK if you cannot decide between two plausible directions and a yes/no answer would resolve it.
- MOVE otherwise. Tell the user a concrete direction (e.g., "turn left and walk down the corridor, then take another photo").
- USE the listed positions to give correct directions: if a goal-related item or sign is on the left, say turn left; if it is on the right, say turn right; if it is centered and near, say go straight toward it.
- Do NOT invent details not in the photo or detections.
- USE the OCR text to identify specific places, sections, signs, and labels. The text tells you WHERE you are (e.g., "Dairy", "Exit", a room name, a directional sign).
- Do NOT invent or assume aisle numbers, room numbers, or location names that are not visible in the photo or OCR text. Only reference locations you can see evidence for.
- ALWAYS reply the "guidance" and "question" fields in Traditional Chinese (繁體中文).
"""


PRIOR_ANSWER_BLOCK = """\
The person just answered your earlier question "{previous_question}" with:
"{user_answer}"
"""


ROUTE_CONTEXT_BLOCK = """\
── 導航上下文（來自預建地圖與路線規劃）──
使用者目前位置：{position_description}
使用者目前面朝：{heading_description}
{route_description}
重要指引原則：
- 一次只引導使用者找「目前要找的商品」，找到後系統會自動切換到下一項
- 用使用者能理解的方式描述方向，例如「往前走到底」「左轉」「右轉走到飲料區」
- 結合照片中看到的標示、招牌、商品來描述位置，不要使用任何內部編號
- 如果照片中已經看到目標商品，直接告訴使用者「就在你前方/左邊/右邊」
- 給出具體、簡短的一到兩句指引，不要含糊或冗長
- 位置信心值低時，以照片中的實際景象為主要判斷依據，地圖位置作為參考
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
- Do not invent objects or text that are not actually visible.
- Keep "label" and "text" values short.
"""
