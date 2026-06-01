"""Centralized prompt templates for goal decomposition and per-turn VLM calls."""

GOAL_DECOMPOSE_PROMPT = """\
A customer wants to find "{goal}" in a retail store. List the object labels an
object detector should look for, covering ALL of:
- the exact product and its common variants (e.g. milk, milk bottle, milk carton)
- the store section it belongs to (e.g. dairy section, refrigerated aisle)
- nearby landmark objects (e.g. refrigerator, cooler, freezer door)
- aisle signage words (e.g. dairy sign, aisle 5 sign)
Give 6-10 items as a single comma-separated list. Be specific to a store.
Reply ONLY with the list, no preamble.

Example for "find milk":
milk, milk carton, milk bottle, dairy section, refrigerated aisle, refrigerator, cooler, dairy sign
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
- USE the OCR text to identify specific places, sections, aisle names, signs, and labels. The text tells you WHERE you are (e.g., "Dairy", "Aisle 5", "Exit", "Checkout").
"""


PRIOR_ANSWER_BLOCK = """\
The person just answered your earlier question "{previous_question}" with:
"{user_answer}"
"""
