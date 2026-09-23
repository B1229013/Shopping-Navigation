"""Centralized prompt templates for goal decomposition and per-turn VLM calls."""

GOAL_DECOMPOSE_PROMPT = """\
A person wants to find "{goal}" in an indoor environment (could be a retail
store, supermarket, office, department, or campus building). List the object
labels an object detector should look for, covering ALL of:
- the exact target and its common variants, in BOTH English and Traditional Chinese
- the area or section it belongs to (e.g. dairy section, 乳製品區, frozen food, 冷凍食品)
- nearby landmark objects (e.g. refrigerator, shelf, door, sign, cabinet)
- signage text that might appear in Chinese (e.g. 乳製品, 冷藏食品, 收銀台)
- signage text that might appear in English (e.g. Dairy, Frozen, Cashier)
- well-known brand names that belong to this product category (e.g. for chips: Lay's, Pringles, Doritos, 樂事, 品客)
Give 10-15 items as a single comma-separated list. Include BOTH English and Chinese for each concept. Be specific — avoid generic words like "shelf", "door", "product".
Reply ONLY with the list, no preamble.

Example for "牛奶":
milk, milk carton, milk bottle, 牛奶, 鮮奶, 鮮乳, dairy section, 乳製品, refrigerator, cooler, 冷藏食品
Example for "洋芋片":
chips, potato chips, snack bags, 洋芋片, 薯片, Lay's, 樂事, Pringles, 品客, Doritos, 多力多滋, snack aisle, 零食
"""


PER_TURN_PROMPT = """\
You are guiding a person through an indoor environment using their phone camera.

Goal: "{goal}"
Goal-related items to look for: {goal_objects}
{progress}
{route_info}
What's happened so far:
{topomap_summary}

The person just uploaded the attached photo. In it, automatic detection found
(each item lists its position in the frame - left/center/right, top/middle/bottom,
and near/far):
{detections_summary}

Text visible in the photo (OCR), each with its side of the frame:
{ocr_summary}

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

MAP ROUTE GUIDANCE:
- If "地圖路線參考" is provided above, it is a route from a known indoor map: use it for the overall PATH, the DISTANCES, and the sequence of LANDMARKS to head toward.
- The turn wording in it ("方向指引：在你的左方/右後方…" and "左轉/右轉/往後方走") is only an ESTIMATE of which way you are currently facing, and it can be WRONG.
- Decide the IMMEDIATE turn from the PHOTO, not from that wording: if the goal or the next landmark is clearly visible in some direction, go that way. When the photo and the map's stated direction conflict, TRUST THE PHOTO.
- NEVER put contradictory directions in one guidance (e.g. do not say "往右後方走" and "朝前方前進" together) — give ONE coherent instruction based on what you actually see.
- Match visible signs, objects, and layout against the expected landmarks to confirm the user is on track; if the suggested direction is a wall or dead end, say so and suggest an alternative.

AISLE SIGN NAVIGATION (critical for supermarkets):
- Overhead aisle signs show aisle numbers and product categories (e.g., "走道8: 服飾, 洗衣"). COMPARE them with the goal — if signs show UNRELATED categories, suggest changing direction.
- If 3+ steps have passed without goal-related signs, STRONGLY suggest trying a different direction.
- SUPERMARKET LAYOUT: dairy/milk is in refrigerated perimeter sections; toilet paper in household/cleaning; produce near entrance.
- Track aisle number progression to understand your direction through the store.

DEAD END / WALL DETECTION:
- If you see a wall, blocked path, or store perimeter with no forward passage, the user reached a DEAD END — tell them to TURN AROUND.

CRITICAL DETECTION OVERRIDE:
- If detections_summary contains an item whose label matches the goal (or a close variant) AND it is tagged "near" with confidence >= 70%, you MUST reply ARRIVED. Do NOT override the structured "near" tag based on visual impression — the tag is computed from bounding-box area and is more reliable than visual depth judgment, especially when items are behind glass doors or display cases.
- Similarly, if OCR text matches the goal item AND the detection is "near", declare ARRIVED.
- Trust the structured data over your own visual depth impression.
"""


PRIOR_ANSWER_BLOCK = """\
The person just answered your earlier question "{previous_question}" with:
"{user_answer}"
"""


# ── Two-stage VLM prompts (when GroundingDINO + EasyOCR are disabled) ──

VLM_PERCEPTION_PROMPT = """\
You are analyzing a photo taken inside an indoor environment (store, office, campus, etc.).
The photo is {img_w}x{img_h} pixels.

Items the user is looking for: {goal_objects}

Carefully examine the photo and return EXACTLY one JSON object (nothing else):

{{"detections": [{{"label": "object name", "score": 0.65, "bbox": [120, 300, 450, 800], "position": "left/center/right top/middle/bottom, near/far"}}], "ocr_texts": [{{"text": "visible text", "score": 0.8, "bbox": [100, 200, 350, 260], "position": "left/center/right top/middle/bottom"}}], "aisle_signs": [{{"aisle_number": 6, "categories": ["飲料", "零食"], "position": "center top, far"}}], "scene_description": "one or two sentences describing the overall layout"}}

Rules for detections:
- List ALL visible objects. Include BOTH goal-related items AND generic indoor objects (door, shelf, sign, chair, table, counter, refrigerator, aisle, cart, etc.)
- IMPORTANT: Labels MUST be in English. Use simple, common English nouns (e.g., "refrigerator", "shelf", "sign", "cart", "milk", "bread"). Do NOT use Chinese for labels. This is critical for matching against the indoor map database.
- IMPORTANT: When a shelf or display rack is visible, also identify the SPECIFIC PRODUCTS on it (e.g., "tea bottles", "coffee cans", "soda bottles", "snack bags", "milk cartons"). Do NOT just say "shelf" — list what is ON the shelf.
- IMPORTANT: If overhead hanging signs with aisle numbers or section names are visible, report them as separate detections with label "sign" and include the text in "nameplate_text". These signs are CRITICAL for localization — they tell us exactly where the user is.
- If a product label or packaging is clearly readable, use the actual product name you can read. Do NOT guess or copy example names — only report text you can actually see in the photo.
- Use CONSISTENT label names: always use the same word for the same type of object. Prefer these standard labels: refrigerator, shelf, sign, cart, door, wall, ceiling, floor, aisle, pillar, entrance, exit, escalator, checkout, bottle, can, fruit, vegetable, meat, milk, bread, snack, rice, noodle.
- score: your confidence 0.0~1.0. Use the FULL range: 0.9+ only if very clear, 0.5~0.8 if partially visible or uncertain, below 0.5 if guessing. Do NOT default everything to the same value.
- bbox: [x1, y1, x2, y2] in absolute pixel coordinates (the image is {img_w}x{img_h}). x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the object's bounding box. Estimate the tightest rectangle that contains the object. Each object MUST have its own distinct bbox even if objects overlap or are on the same shelf — do NOT give multiple objects the same bbox.
- position: describe where the object is in the frame. Use the format: "left/center/right" + "top/middle/bottom" + ", near/far". Example: "right middle, near". This is a coarse backup for the bbox.
- near vs far: judge this by how much of the frame the object fills and how much fine detail (small text, texture) you can actually make out — "near" means it's close enough to reach out and touch, filling a large part of the frame with fine print legible; "far" means it's small in the frame, part of a distant row/shelf, or its fine details are not clearly legible even if you can name it. Do NOT default to "near" just because you can identify what the object is — you can often recognize a product from across an aisle.

Rules for ocr_texts:
- List ALL visible text: aisle signs, nameplates, labels, posters, price tags, directional signs, product names on packaging, aisle numbers, section headers.
- CRITICAL FOR LOCALIZATION: The most valuable text for determining WHERE the user is includes:
  * Overhead/hanging aisle signs with numbers (e.g., "8", "15") and categories (e.g., "飲料", "零食")
  * Section headers on walls or shelves (e.g., "冷凍食品", "生鮮", "日用品")
  * Price tags with numbers (e.g., "99", "135", "199")
  * Store section signs (e.g., "自助收銀", "EXIT", "入口")
  READ AND REPORT ALL OF THESE, even if they seem unimportant. Numbers and section names are the primary way we determine the user's location in the store.
- IMPORTANT: Read product labels and price tags on shelves — these identify what products are available.
- Include both English and Chinese (Traditional/Simplified).
- ACCURACY: Report EXACTLY the text you see, character by character. Do NOT substitute similar words — "衛生棉" and "衛生紙" are DIFFERENT products. Do NOT change text to match the goal items. If you cannot read it clearly, lower the score instead of guessing.
- If the text you read matches (or closely resembles) one of the goal items {goal_objects}, transcribe it EXACTLY as printed — do not paraphrase or translate it into a different phrasing. This exact text is later matched against the goal, so an exact transcription is what lets the system recognize it.
- ONLY report text that you can literally SEE as printed/written characters in the image. Do NOT infer or translate from icons, symbols, or pictograms. A green running-man exit sign is NOT OCR text "出口" unless the characters "出口" are actually printed on it. If you see a symbol/icon but no readable characters, do NOT add it to ocr_texts.
- bbox: [x1, y1, x2, y2] in absolute pixel coordinates (the image is {img_w}x{img_h}). The bounding box should tightly enclose the visible text region. Each text entry MUST have its own distinct bbox.
- position: where in the frame. Use the same format as detections, including the same near/far judgment described above (based on how much of the frame it fills and how legible it is, not just whether you can read it). Example: "center top, near"

Rules for aisle_signs:
- If you see ANY overhead hanging sign that indicates an aisle number and/or product categories, add an entry to aisle_signs.
- aisle_number: the aisle number as an integer (e.g., 6, 14). Set to null if only categories are visible without a number.
- categories: a list of product category names shown on the sign, in the ORIGINAL language as printed (e.g., ["飲料", "零食"], ["進口食品", "泡麵"]). Empty list if only a number is visible.
- position: where the sign is in the frame, same format as detections.
- A single physical sign often shows BOTH a number AND categories — combine them into ONE aisle_signs entry. Do NOT split them.
- If you see MULTIPLE different aisle signs (e.g., aisle 6 on the left and aisle 7 on the right), create SEPARATE entries for each.
- This field is CRITICAL for localization. Even if you already reported the number or category text in ocr_texts, ALSO add it to aisle_signs with the structured format.

Rules for scene_description:
- Describe the layout: corridor/aisle/room, what is on each side, the path ahead.
- Mention any directional cues (arrows, signs pointing somewhere).
- Keep it to 1-2 sentences, factual, no guessing.
"""


VLM_NAVIGATION_PROMPT = """\
You are guiding a person through an indoor environment using their phone camera.

Goal: "{goal}"
Goal-related items to look for: {goal_objects}
{progress}
{route_info}

What's happened so far:
{topomap_summary}

Scene overview:
{scene_description}

The person just uploaded the attached photo. Analysis found these objects
(each with its position in the frame):
{detections_summary}

Text visible in the photo:
{ocr_summary}

{prior_answer_block}

Decide the next step. Reply with EXACTLY one JSON object on one line, nothing else:

{{"action": "ARRIVED" | "MOVE" | "ASK", "guidance": "<one or two sentences>", "question": "<short question, only if ASK, else null>", "vlm_summary": "<one phrase summarizing the location>"}}

Rules:
- ARRIVED only if the goal item is clearly visible AND tagged "near" (or otherwise filling a meaningful part of the frame with fine details legible) — point at it in `guidance`. This includes a close-up of the item itself — the goal item does NOT need to be sitting on a visible shelf or rack. A detected label or OCR text that directly names the goal item (or its brand/packaging) is enough to declare ARRIVED, but ONLY when it is tagged "near". A "far" match on the goal item is NOT arrival — see the next rule.
- If the goal item's name or brand appears in detections or OCR but is tagged "far" (or otherwise small/distant in the frame), do NOT declare ARRIVED. This is strong evidence you are heading the right way — respond MOVE, tell the user to walk toward it, and take another photo once closer.
- ASK if you cannot decide between two plausible directions and a yes/no answer would resolve it.
- MOVE otherwise. Tell the user a concrete direction (e.g., "turn left and walk down the corridor, then take another photo").
- USE the listed positions to give correct directions: if a goal-related item or sign is on the left, say turn left; if on the right, say turn right; if centered and near, say go straight.
- USE the OCR text for TWO purposes, and check both every time: (1) identify places, sections, signs — the text tells you WHERE you are; AND (2) confirm WHAT you found — if any OCR text or detected label matches the goal item's name or brand, treat that as direct evidence (ARRIVED if "near", MOVE-toward-it if "far"), on equal footing with (1), not secondary to it.
- PRODUCT MATCH STRATEGY: Whether or not a shelf/rack is visible, check every detected label and every OCR text against the goal item. If the goal product's name or brand is among them AND tagged "near", declare ARRIVED. If tagged "far", guide the user toward it instead. If a shelf/rack is visible but the specific products on it are unclear, guide the user to move closer and take another photo.
- Do NOT invent details not in the photo or detection results.
- Do NOT invent or assume aisle numbers, room numbers, or location names not visible in the photo or OCR.
- ALWAYS reply the "guidance" and "question" fields in Traditional Chinese (繁體中文).

MAP ROUTE GUIDANCE:
- If "地圖路線參考" is provided above, it is a route from a known indoor map: use it for the overall PATH, the DISTANCES, and the sequence of LANDMARKS to head toward.
- The turn wording in it ("方向指引：在你的左方/右後方…" and "左轉/右轉/往後方走") is only an ESTIMATE of which way you are currently facing, and it can be WRONG.
- Decide the IMMEDIATE turn from the PHOTO, not from that wording: if the goal or the next landmark is clearly visible in some direction, go that way. When the photo and the map's stated direction conflict, TRUST THE PHOTO.
- NEVER put contradictory directions in one guidance (e.g. do not say "往右後方走" and "朝前方前進" together) — give ONE coherent instruction based on what you actually see.
- Match visible signs, objects, and layout against the expected landmarks to confirm the user is on track; if the suggested direction is a wall or dead end, say so and suggest an alternative.

AISLE SIGN NAVIGATION (critical for supermarkets):
- Overhead aisle signs show aisle numbers and product categories (e.g., "走道8: 服飾, 洗衣", "走道 12, 13"). These tell you EXACTLY which section you are in.
- COMPARE the sign categories with the goal. If the signs show categories UNRELATED to the goal (e.g., looking for 牛奶 but signs say 服飾/洗衣/零食), you are in the WRONG area — suggest changing direction, turning around, or looking for signs that match the goal category.
- Use aisle number progression to track direction: if numbers are increasing (3→5→7), you are moving one direction; if they start decreasing or you see much higher numbers, consider whether the goal section has been passed.

DEAD END / WALL DETECTION:
- If you see a solid wall, orange/colored wall panel, or the path is clearly blocked with no forward passage, the user has reached a DEAD END. Do NOT say "continue straight" — tell them to TURN AROUND or try a different aisle.
- Signs of a dead end: wall filling the center/far area, no visible aisle ahead, store perimeter wall, loading area, or employee-only door.

CRITICAL DETECTION OVERRIDE:
- If detections_summary contains an item whose label matches the goal (or a close variant, e.g. milk/鮮乳/鮮奶 for 牛奶) AND it is tagged "near" with confidence >= 70%, you MUST reply ARRIVED. Do NOT override the structured "near" tag based on your own visual depth impression — the "near"/"far" tag is computed from bounding-box area and is MORE RELIABLE than visual judgment, especially when items are behind glass doors, inside refrigerated display cases, or on the other side of transparent barriers.
- Similarly, if OCR text matches the goal item AND the corresponding detection is "near", you MUST declare ARRIVED.
- This rule takes PRIORITY over your visual impression. Trust the structured data.
"""


# ── Alternative perception prompt (繁中、比例座標、精簡偵測) ──

PERCEIVE_PROMPT = """\
你正在分析一張超市內部的照片，用來定位使用者的位置。

使用者正在尋找: {goal_objects}

仔細觀察照片，回報你**確實看到**的**具體**物品。回覆格式為一個 JSON 物件，不要加其他文字：

{{"scene": "<用繁體中文描述這是什麼區域>", "items": [{{"zh": "<繁體中文名稱>", "en": "<English name>", "conf": <0.0-1.0 信心值>, "box": [x1,y1,x2,y2]}}]}}

嚴格規則：
- box 座標是圖片的比例值（0.0 到 1.0），格式 [左, 上, 右, 下]。框要緊密包住物件。
- 禁止大框：寬度和高度都不應超過圖片的50%。
- 只回報你**確實看到**的具體物件，不要猜測。
- 回報具體商品、品牌、設備、標示牌文字、地標。
- 特別注意走道上方的吊牌、區域標示牌（如「走道5」「堅果海苔」等分類標示），這些是重要的定位特徵。
- 標示牌上的文字內容要完整辨識，寫在 zh 欄位。
- 不要回報背景（地板、天花板、牆壁、燈光）。
- 寧可少報也不要報錯。5-12個精確的偵測最好。
"""
