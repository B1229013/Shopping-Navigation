"""Enrich GroundingDINO detections with OCR sign categories, then build the map.

In a real store the strongest zone signal is the *signage* (aisle category
names + numbers), not GroundingDINO's generic object labels. This merges OCR
text (store_ocr.json) into the detection objects (store_detections.json) as
canonical category tokens so Jaccard clustering and zone labels reflect actual
store sections (snacks, kitchenware, candy, cashier, ...).

Run (WSL unigoal env):
    python build_store_map.py
Then it renders store_topomap.png via generate_topomap.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Chinese/English sign fragment -> canonical category token. Matched as a
# substring against each OCR string (case-insensitive for latin text).
CATEGORY_SIGNS = {
    # front / perishables
    "冷藏": "chilled-food", "冷凍": "frozen-food", "火鍋": "hotpot",
    "taiwan beer": "beer", "分解茶": "tea", "早餐": "breakfast",
    "南北貨": "dry-goods", "米": "rice",
    # packaged / grocery
    "進口食品": "imported-food", "罐頭": "canned-food", "canned": "canned-food",
    "甜餅乾": "biscuits", "sweet biscuit": "biscuits", "夾心餅": "biscuits",
    "泡芙": "puffs", "泡麵": "instant-noodles",
    # snacks
    "洋芋片": "snacks", "點心": "snacks", "snack": "snacks",
    "米果": "rice-crackers", "蘇打餅": "crackers",
    # non-food
    "洗衣": "laundry", "服飾": "clothing",
    "家庭保健": "family-health", "family hygiene": "family-health",
    "廚房用品": "kitchenware", "kitchenwa": "kitchenware",
    "身體": "body-hair-care", "髮類": "body-hair-care", "body": "body-hair-care",
    "紙尿褲": "hygiene", "衛生棉": "hygiene", "diaper": "hygiene",
    # sweets / nuts / drinks / checkout
    "糖果": "candy", "candy": "candy", "巧克力": "chocolate", "chocolate": "chocolate",
    "堅果": "nuts", "杏仁": "nuts",
    "茶|汽水": "tea-soda", "汽水": "tea-soda", "soda": "tea-soda",
    "咖啡": "coffee-juice", "果汁": "coffee-juice", "juice": "coffee-juice",
    "肉乾": "jerky-seaweed", "海苔": "jerky-seaweed", "seaweed": "jerky-seaweed",
    "收銀": "cashier", "cashier": "cashier", "carrefour": "store-brand",
}


def ocr_tokens(ocr_strings: list[str]) -> dict[str, float]:
    """Map a photo's OCR strings to canonical category tokens (token->best conf)."""
    tokens: dict[str, float] = {}
    for text, conf in ocr_strings:
        low = text.lower()
        for frag, token in CATEGORY_SIGNS.items():
            f = frag.lower()
            if f in low or frag in text:
                if conf > tokens.get(token, 0.0):
                    tokens[token] = conf
    return tokens


def aisle_numbers(ocr_strings: list[str]) -> list[int]:
    """Extract plausible aisle numbers (1..16) from OCR, ignoring prices."""
    nums = set()
    for text, _ in ocr_strings:
        for m in re.findall(r"\b(\d{1,2})\b", text):
            n = int(m)
            if 1 <= n <= 16:
                nums.add(n)
    return sorted(nums)


def main() -> int:
    det = json.load(open(ROOT / "store_detections.json", encoding="utf-8"))
    ocr = json.load(open(ROOT / "store_ocr.json", encoding="utf-8"))

    for photo in det["photos"]:
        toks = ocr_tokens(ocr.get(photo["filename"], []))
        for token, conf in toks.items():
            photo["objects"].append(
                {"label": token, "score": round(float(conf), 3), "box": [0, 0, 0, 0]}
            )
        photo["summary"] = ", ".join(f"{o['label']}({o['score']})" for o in photo["objects"])

    enriched = ROOT / "store_detections_enriched.json"
    json.dump(det, open(enriched, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote {enriched.name}")
    for photo in det["photos"]:
        labs = sorted({o["label"] for o in photo["objects"]})
        ais = aisle_numbers(ocr.get(photo["filename"], []))
        print(f"  {photo['filename']:22s} aisles={ais} {labs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
