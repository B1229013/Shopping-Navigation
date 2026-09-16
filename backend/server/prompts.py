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

AISLE SIGN NAVIGATION (critical for supermarkets):
- Overhead aisle signs show aisle numbers and product categories (e.g., "走道8: 服飾, 洗衣"). COMPARE them with the goal — if signs show UNRELATED categories, suggest changing direction.
- If 3+ steps have passed without goal-related signs, STRONGLY suggest trying a different direction.
- SUPERMARKET LAYOUT: dairy/milk is in refrigerated perimeter sections; toilet paper in household/cleaning; produce near entrance.
- Track aisle number progression to understand your direction through the store.

DEAD END / WALL DETECTION:
- If you see a wall, blocked path, or store perimeter with no forward passage, the user reached a DEAD END — tell them to TURN AROUND.
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

{{"detections": [{{"label": "object name", "score": 0.65, "position": "left/center/right top/middle/bottom, near/far"}}], "ocr_texts": [{{"text": "visible text", "score": 0.8, "position": "left/center/right top/middle/bottom"}}], "scene_description": "one or two sentences describing the overall layout"}}

Rules for detections:
- List ALL visible objects. Include BOTH goal-related items AND generic indoor objects (door, shelf, sign, chair, table, counter, refrigerator, aisle, cart, etc.)
- IMPORTANT: When a shelf or display rack is visible, also identify the SPECIFIC PRODUCTS on it (e.g., "tea bottles", "coffee cans", "soda bottles", "snack bags", "milk cartons"). Do NOT just say "shelf" — list what is ON the shelf.
- If a product label or packaging is clearly readable, use the actual product name you can read. Do NOT guess or copy example names — only report text you can actually see in the photo.
- score: your confidence 0.0~1.0. Use the FULL range: 0.9+ only if very clear, 0.5~0.8 if partially visible or uncertain, below 0.5 if guessing. Do NOT default everything to the same value.
- position: IMPORTANT — describe precisely where the object is in the frame. Use the format: "left/center/right" + "top/middle/bottom" + ", near/far". Example: "right middle, near". This is the primary way we locate objects, so be accurate.
- near vs far: judge this by how much of the frame the object fills and how much fine detail (small text, texture) you can actually make out — "near" means it's close enough to reach out and touch, filling a large part of the frame with fine print legible; "far" means it's small in the frame, part of a distant row/shelf, or its fine details are not clearly legible even if you can name it. Do NOT default to "near" just because you can identify what the object is — you can often recognize a product from across an aisle.

Rules for ocr_texts:
- List ALL visible text: aisle signs, nameplates, labels, posters, price tags, directional signs, product names on packaging.
- IMPORTANT: Read product labels and price tags on shelves — these identify what products are available.
- Include both English and Chinese (Traditional/Simplified).
- ACCURACY: Report EXACTLY the text you see, character by character. Do NOT substitute similar words — "衛生棉" and "衛生紙" are DIFFERENT products. Do NOT change text to match the goal items. If you cannot read it clearly, lower the score instead of guessing.
- If the text you read matches (or closely resembles) one of the goal items {goal_objects}, transcribe it EXACTLY as printed — do not paraphrase or translate it into a different phrasing. This exact text is later matched against the goal, so an exact transcription is what lets the system recognize it.
- ONLY report text that you can literally SEE as printed/written characters in the image. Do NOT infer or translate from icons, symbols, or pictograms. A green running-man exit sign is NOT OCR text "出口" unless the characters "出口" are actually printed on it. If you see a symbol/icon but no readable characters, do NOT add it to ocr_texts.
- position: where in the frame. Use the same format as detections, including the same near/far judgment described above (based on how much of the frame it fills and how legible it is, not just whether you can read it). Example: "center top, near"

Rules for scene_description:
- Describe the layout: corridor/aisle/room, what is on each side, the path ahead.
- Mention any directional cues (arrows, signs pointing somewhere).
- Keep it to 1-2 sentences, factual, no guessing.
"""


VLM_NAVIGATION_PROMPT = """\
You are guiding a person through an indoor environment using their phone camera.

Goal: "{goal}"
Goal-related items to look for: {goal_objects}

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

AISLE SIGN NAVIGATION (critical for supermarkets):
- Overhead aisle signs show aisle numbers and product categories (e.g., "走道8: 服飾, 洗衣", "走道 12, 13"). These tell you EXACTLY which section you are in.
- COMPARE the sign categories with the goal. If the signs show categories UNRELATED to the goal (e.g., looking for 牛奶 but signs say 服飾/洗衣/零食), you are in the WRONG area — suggest changing direction, turning around, or looking for signs that match the goal category.
- Use aisle number progression to track direction: if numbers are increasing (3→5→7), you are moving one direction; if they start decreasing or you see much higher numbers, consider whether the goal section has been passed.

DEAD END / WALL DETECTION:
- If you see a solid wall, orange/colored wall panel, or the path is clearly blocked with no forward passage, the user has reached a DEAD END. Do NOT say "continue straight" — tell them to TURN AROUND or try a different aisle.
- Signs of a dead end: wall filling the center/far area, no visible aisle ahead, store perimeter wall, loading area, or employee-only door.
"""
