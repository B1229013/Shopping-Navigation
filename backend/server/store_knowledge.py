"""Static environment knowledge for the CSIE Department Office (資訊工程學系 系辦)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RoomInfo:
    node_id: int
    name_zh: str
    name_en: str
    categories_en: List[str] = field(default_factory=list)
    landmarks_en: List[str] = field(default_factory=list)
    landmarks_zh: List[str] = field(default_factory=list)


@dataclass
class LocationMatch:
    location_type: str  # "room" or "zone"
    aisle_number: Optional[int]  # kept for API compat, maps to node_id
    zone_name: Optional[str]
    display_en: str
    display_zh: str
    matched_keyword: str
    confidence: float


# ---------------------------------------------------------------------------
# Room / Zone data (from photo survey of the department office)
# ---------------------------------------------------------------------------
ROOMS: List[RoomInfo] = [
    RoomInfo(
        node_id=0,
        name_zh="入口走廊",
        name_en="Entrance Hallway",
        categories_en=["Entrance", "Hallway"],
        landmarks_en=["department banner", "glass door", "hallway", "entrance"],
        landmarks_zh=["入口", "走廊", "資訊工程學系"],
    ),
    RoomInfo(
        node_id=1,
        name_zh="大廳（沙發區）",
        name_en="Main Lobby (Sofa Area)",
        categories_en=["Lobby", "Lounge", "Waiting Area"],
        landmarks_en=["sofa", "leather sofa", "coffee table", "glass table",
                       "flowers", "wooden cabinets", "lockers", "bulletin board",
                       "trophy", "brochure rack"],
        landmarks_zh=["沙發", "大廳", "置物櫃", "獎杯", "佈告欄"],
    ),
    RoomInfo(
        node_id=2,
        name_zh="咖啡區",
        name_en="Coffee Station",
        categories_en=["Coffee", "Pantry", "Kitchen"],
        landmarks_en=["espresso machine", "coffee grinder", "coffee maker",
                       "mini fridge", "small refrigerator", "counter", "cabinet",
                       "glass cabinet"],
        landmarks_zh=["咖啡機", "磨豆機", "小冰箱", "流理台"],
    ),
    RoomInfo(
        node_id=3,
        name_zh="主任室",
        name_en="Office of Department Chair",
        categories_en=["Chair Office", "Director Office"],
        landmarks_en=["department chair", "director office", "meeting room",
                       "yellow chairs", "round table", "blinds"],
        landmarks_zh=["主任室", "系主任"],
    ),
    RoomInfo(
        node_id=4,
        name_zh="行政辦公區",
        name_en="Admin / Reception Area",
        categories_en=["Admin", "Reception", "Office"],
        landmarks_en=["reception desk", "printer", "copier", "photocopier",
                       "mosaic wall", "patterned wallpaper", "computer desk",
                       "water dispenser"],
        landmarks_zh=["行政區", "影印機", "飲水機", "辦公桌"],
    ),
    RoomInfo(
        node_id=5,
        name_zh="教授走廊 A（左側）",
        name_en="Professor Corridor A (Left)",
        categories_en=["Corridor", "Professor Offices"],
        landmarks_en=["professor office", "fire extinguisher", "bird decals",
                       "swallow decals", "yellow raincoat", "name plates"],
        landmarks_zh=["教授辦公室", "滅火器", "走廊"],
    ),
    RoomInfo(
        node_id=6,
        name_zh="教授走廊 B（右側）",
        name_en="Professor Corridor B (Right)",
        categories_en=["Corridor", "Professor Offices"],
        landmarks_en=["professor office", "name plates", "fire extinguisher",
                       "trash buckets", "blue buckets"],
        landmarks_zh=["教授辦公室", "走廊"],
    ),
    RoomInfo(
        node_id=7,
        name_zh="檔案櫃走廊",
        name_en="Filing Cabinet Corridor",
        categories_en=["Filing", "Storage", "Corridor"],
        landmarks_en=["glass-front cabinets", "filing cabinets", "wooden cabinets",
                       "storage corridor"],
        landmarks_zh=["檔案櫃", "置物櫃", "走廊"],
    ),
    RoomInfo(
        node_id=8,
        name_zh="冰箱區（消防栓旁）",
        name_en="Refrigerator Area (near Fire Hydrant)",
        categories_en=["Refrigerator", "Storage"],
        landmarks_en=["refrigerator", "fridge", "white refrigerator",
                       "fire hydrant", "fire extinguisher", "hydrant box"],
        landmarks_zh=["冰箱", "白色冰箱", "消防栓", "滅火器"],
    ),
    RoomInfo(
        node_id=9,
        name_zh="教授走廊 C（後段）",
        name_en="Professor Corridor C (Back)",
        categories_en=["Corridor", "Professor Offices"],
        landmarks_en=["professor office", "name plates", "trash buckets",
                       "teaching facility", "poster"],
        landmarks_zh=["教授辦公室", "教具室", "走廊"],
    ),
]

# Map of professor offices to their corridor node for reference
PROFESSOR_OFFICES = {
    "黃崇源": {"en": "Chung-Yuan Huang", "node": 5},
    "主任室": {"en": "Department Chair", "node": 3},
    "吳齊人": {"en": "Chi-Jen Wu", "node": 6},
    "林仲志": {"en": "Chong-Chih Lin", "node": 6},
    "李季青": {"en": "Chi-Ching Lee", "node": 6},
    "許文宏": {"en": "Ai-Liang Hsu", "node": 6},
    "吳巴琳": {"en": "Ba-Lin Wu", "node": 6},
    "謝萬雲": {"en": "Wann-Yun Shieh", "node": 9},
    "陳仁匯": {"en": "Jen-Hui Chen", "node": 9},
    "陳光武": {"en": "Guang-Wu Chen", "node": 9},
    "魏志達": {"en": "Jyh-Da Wei", "node": 9},
    "張哲維": {"en": "Che-Wei Chang", "node": 9},
    "李春良": {"en": "Chun-Liang Lee", "node": 9},
}


# ---------------------------------------------------------------------------
# Product / Object search
# ---------------------------------------------------------------------------
def _is_cjk(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def find_product(query: str) -> List[LocationMatch]:
    """Find where an object/room is located in the environment.

    Accepts Chinese or English queries. Returns a ranked list of matches.
    """
    query_lower = query.lower().strip()
    results: List[LocationMatch] = []

    for room in ROOMS:
        best_score = 0.0
        best_keyword = ""

        if _is_cjk(query_lower):
            search_lists = [
                ([room.name_zh], 1.0),
                (room.landmarks_zh, 0.9),
            ]
        else:
            search_lists = [
                ([room.name_en], 1.0),
                (room.categories_en, 0.95),
                (room.landmarks_en, 0.9),
            ]

        for word_list, base_score in search_lists:
            for keyword in word_list:
                kw_lower = keyword.lower()
                if query_lower == kw_lower:
                    score = base_score
                elif query_lower in kw_lower or kw_lower in query_lower:
                    score = base_score * 0.8
                else:
                    ratio = difflib.SequenceMatcher(None, query_lower, kw_lower).ratio()
                    if ratio > 0.55:
                        score = base_score * ratio * 0.7
                    else:
                        continue

                if score > best_score:
                    best_score = score
                    best_keyword = keyword

        if best_score > 0:
            results.append(LocationMatch(
                location_type="room",
                aisle_number=room.node_id,
                zone_name=None,
                display_en=room.name_en,
                display_zh=room.name_zh,
                matched_keyword=best_keyword,
                confidence=best_score,
            ))

    # Also search professor offices
    for prof_zh, info in PROFESSOR_OFFICES.items():
        prof_en = info["en"].lower()
        score = 0.0
        keyword = ""
        if _is_cjk(query_lower):
            if query_lower in prof_zh or prof_zh in query_lower:
                score = 0.9
                keyword = prof_zh
        else:
            if query_lower in prof_en or prof_en in query_lower:
                score = 0.9
                keyword = info["en"]
            else:
                ratio = difflib.SequenceMatcher(None, query_lower, prof_en).ratio()
                if ratio > 0.55:
                    score = 0.9 * ratio * 0.7
                    keyword = info["en"]

        if score > 0:
            node = info["node"]
            room_info = next((r for r in ROOMS if r.node_id == node), None)
            if room_info:
                results.append(LocationMatch(
                    location_type="room",
                    aisle_number=node,
                    zone_name=None,
                    display_en=f"{info['en']}'s Office (in {room_info.name_en})",
                    display_zh=f"{prof_zh} 教授辦公室",
                    matched_keyword=keyword,
                    confidence=score,
                ))

    results.sort(key=lambda m: m.confidence, reverse=True)
    return results


def get_all_aisles() -> list:
    """Return a summary of all rooms/zones for API compatibility."""
    return [
        {
            "number": r.node_id,
            "categories_zh": [r.name_zh],
            "categories_en": r.categories_en,
        }
        for r in ROOMS
    ]


def get_all_zones() -> list:
    """Return a summary of all zones."""
    return [
        {
            "name": f"room_{r.node_id}",
            "display_name_zh": r.name_zh,
            "display_name_en": r.name_en,
            "categories_zh": [r.name_zh],
            "categories_en": r.categories_en,
        }
        for r in ROOMS
    ]
