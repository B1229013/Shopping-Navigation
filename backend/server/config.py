"""Configuration constants and paths.

Paths are auto-detected for both WSL and Windows environments.
Model weights and configs are resolved relative to the project root.
"""
import os
from pathlib import Path

# Project root (where this repo lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root (Shopping-Navigation/.env)
_dotenv = PROJECT_ROOT.parent / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.split("#")[0].strip()
            os.environ.setdefault(k.strip(), v)

# Model weights — check local project first, then WSL paths
_LOCAL_MODELS = PROJECT_ROOT / "models"
_WSL_MODELS = Path("/home/user/UniGoal/data/models")

# GDINO_BACKBONE: "swint" (default, faster) or "swinb" (more accurate, slower)
GDINO_BACKBONE = os.environ.get("GDINO_BACKBONE", "swint")

_WEIGHTS_MAP = {
    "swint": "groundingdino_swint_ogc.pth",
    "swinb": "groundingdino_swinb_cogcoor.pth",
}
_weights_file = _WEIGHTS_MAP.get(GDINO_BACKBONE, _WEIGHTS_MAP["swint"])

if (_LOCAL_MODELS / _weights_file).exists():
    GROUNDINGDINO_WEIGHTS = _LOCAL_MODELS / _weights_file
elif (_WSL_MODELS / _weights_file).exists():
    GROUNDINGDINO_WEIGHTS = _WSL_MODELS / _weights_file
else:
    GROUNDINGDINO_WEIGHTS = _LOCAL_MODELS / _weights_file

# GroundingDINO config — auto-detect from pip site-packages
_GDINO_REPO = Path(os.environ.get(
    "GROUNDINGDINO_REPO",
    str(PROJECT_ROOT.parent / "GroundingDINO")
))

_CONFIG_MAP = {
    "swint": "GroundingDINO_SwinT_OGC.py",
    "swinb": "GroundingDINO_SwinB_cfg.py",
}
_config_file = _CONFIG_MAP.get(GDINO_BACKBONE, _CONFIG_MAP["swint"])

def _gdino_site_packages_config() -> Path:
    try:
        import groundingdino
        return Path(groundingdino.__file__).parent / "config" / _config_file
    except ImportError:
        return Path("_not_installed_")

_GDINO_CONFIG_CANDIDATES = [
    _gdino_site_packages_config(),
    _GDINO_REPO / "groundingdino" / "config" / _config_file,
    PROJECT_ROOT.parent.parent / "GroundingDINO" / "groundingdino" / "config" / _config_file,
]
GROUNDINGDINO_CONFIG = next(
    (p for p in _GDINO_CONFIG_CANDIDATES if p.exists()),
    _GDINO_CONFIG_CANDIDATES[0],
)

# SAM weights (optional — not loaded by default to stay under VRAM budget)
SAM_WEIGHTS = _LOCAL_MODELS / "sam_vit_h_4b8939.pth"

# Output
OUTPUT_ROOT = PROJECT_ROOT / "output" / "sessions"

# LLM services — "openai" (default), "gemini", or "ollama".
VLM_BACKEND = os.environ.get("VLM_BACKEND", "openai")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# CGU LLM gateway — OpenAI-compatible: one key (CGU_API_KEY) unlocks every model;
# only the endpoint differs by model type. See server/llm_gateway.py.
CGU_API_KEY = os.environ.get("CGU_API_KEY", "")
CGU_BASE_URL = os.environ.get("CGU_BASE_URL", "https://air.cgu.edu.tw/cgullmapi/v1")

# The existing VLM backend reuses the CGU key/URL when OPENAI_* are unset, so
# running VLM_BACKEND=openai against the gateway needs only CGU_API_KEY in .env.
# Setting OPENAI_API_KEY/OPENAI_BASE_URL explicitly still targets real OpenAI.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or CGU_API_KEY
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or (CGU_BASE_URL if CGU_API_KEY else "https://api.openai.com/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")

# Server
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

# Photo paths like "set01/UUID_front.jpg" are resolved as "{root}/01/UUID_front.jpg"
# (strip the "set" prefix from the first path component).
TOPO_PHOTOS_ROOT = os.environ.get("TOPO_PHOTOS_ROOT", "")

# Optional: load topo_v2 map from a local JSON file instead of Neo4j.
# When set, the server loads this JSON first; Neo4j is still available for other uses.
TOPO_MAP_JSON = os.environ.get("TOPO_MAP_JSON", "")

# Optional: JSON topomap for photo_path fallback when Neo4j paths are unresolvable.
# When the Neo4j map has paths from another machine (e.g. Mac paths), this JSON
# provides correct local relative paths keyed by photo_id.
TOPO_PHOTOS_JSON = os.environ.get(
    "TOPO_PHOTOS_JSON",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "carrefour_a7_imported.json"),
)

# Lazy-loaded fallback: photo_id → photo_path from JSON
_json_photo_fallback: dict = {}
_json_fallback_loaded: bool = False


def _load_json_photo_fallback() -> dict:
    """Load photo_id → photo_path mapping from JSON topomap (once)."""
    global _json_photo_fallback, _json_fallback_loaded
    if _json_fallback_loaded:
        return _json_photo_fallback
    _json_fallback_loaded = True
    import json
    import logging
    _log = logging.getLogger(__name__)
    json_path = TOPO_PHOTOS_JSON
    if not json_path or not os.path.isfile(json_path):
        _log.info("TOPO_PHOTOS_JSON not found (%s), no photo fallback", json_path)
        return _json_photo_fallback
    try:
        with open(json_path, encoding="utf-8-sig") as f:
            data = json.load(f)
        for node in data.get("nodes", []):
            if node.get("ntype") == "photo":
                nid = node.get("id")
                pp = node.get("photo_path", "")
                if nid is not None and pp:
                    _json_photo_fallback[int(nid)] = pp
                # Also store 4-direction photos
                for direction, dp in node.get("photos", {}).items():
                    if dp:
                        _json_photo_fallback[(int(nid), direction)] = dp
        _log.info("Loaded JSON photo fallback: %d entries from %s",
                  len(_json_photo_fallback), json_path)
    except Exception as e:
        _log.warning("Failed to load JSON photo fallback: %s", e)
    return _json_photo_fallback


def resolve_topo_photo(photo_path: str, photo_id: int = None) -> "Optional[Path]":
    """Resolve a topo photo_path to an actual file on disk.

    Tries the original path first, then falls back to TOPO_PHOTOS_ROOT
    with the 'set' prefix stripped (e.g. set01/file.jpg → 01/file.jpg).
    If photo_id is given and all attempts fail, tries looking up the JSON
    topomap fallback for an alternative path.
    """
    import re
    import logging
    from typing import Optional
    _log = logging.getLogger(__name__)
    if not photo_path:
        _log.warning("resolve_topo_photo: empty photo_path")
        return None
    p = Path(photo_path)
    if p.is_file():
        return p
    root = TOPO_PHOTOS_ROOT
    if not root:
        _log.warning("resolve_topo_photo: TOPO_PHOTOS_ROOT not set, photo_path=%s", photo_path)
        return None
    root_path = Path(root)
    if not root_path.is_dir():
        _log.warning("resolve_topo_photo: root dir not found: %s", root_path)
        return None
    candidate = root_path / photo_path
    if candidate.is_file():
        return candidate
    stripped = re.sub(r"^set(\d+)", r"\1", photo_path)
    if stripped != photo_path:
        candidate = root_path / stripped
        if candidate.is_file():
            return candidate
    # Also try stripping set prefix with path separator normalization
    stripped_fwd = re.sub(r"^set(\d+)[/\\]", r"\1/", photo_path)
    if stripped_fwd != photo_path:
        candidate = root_path / stripped_fwd
        if candidate.is_file():
            return candidate
    fname = Path(photo_path).name
    for child in root_path.iterdir():
        if child.is_dir():
            candidate = child / fname
            if candidate.is_file():
                return candidate

    # ── JSON fallback: try alternative photo_path from JSON by photo_id ──
    if photo_id is not None:
        fallback = _load_json_photo_fallback()
        alt_path = fallback.get(photo_id, "")
        if alt_path and alt_path != photo_path:
            result = _resolve_path_with_root(alt_path, root_path)
            if result is not None:
                return result

    _log.warning("resolve_topo_photo: not found after all attempts, photo_path=%s root=%s", photo_path, root_path)
    return None


def _resolve_path_with_root(photo_path: str, root_path: Path) -> "Optional[Path]":
    """Try resolving a photo_path against root_path (helper for fallback)."""
    import re
    candidate = root_path / photo_path
    if candidate.is_file():
        return candidate
    stripped = re.sub(r"^set(\d+)", r"\1", photo_path)
    if stripped != photo_path:
        candidate = root_path / stripped
        if candidate.is_file():
            return candidate
    stripped_fwd = re.sub(r"^set(\d+)[/\\]", r"\1/", photo_path)
    if stripped_fwd != photo_path:
        candidate = root_path / stripped_fwd
        if candidate.is_file():
            return candidate
    fname = Path(photo_path).name
    for child in root_path.iterdir():
        if child.is_dir():
            candidate = child / fname
            if candidate.is_file():
                return candidate
    return None


# Detection thresholds (TASK 1 — raised for busy store shelves; retry lower only if nothing found)
PERCEPTION_ENABLED = os.environ.get("PERCEPTION_ENABLED", "1") != "0"
GROUNDINGDINO_BOX_THRESHOLD = 0.30
GROUNDINGDINO_TEXT_THRESHOLD = 0.25
GROUNDINGDINO_BOX_THRESHOLD_FALLBACK = 0.20
SAM_TOP_K_BOXES = 10

# Generic indoor objects — always detected alongside goal_objects for spatial context
GENERIC_INDOOR_OBJECTS = [
    "door", "cabinet", "chair", "table", "desk", "sofa", "sign",
    "fire extinguisher", "plant", "printer", "whiteboard", "monitor",
    "computer", "bookshelf", "window", "counter", "service counter",
    "bulletin board", "notice board",
    "refrigerator", "water dispenser", "shelf", "trash can", "poster",
]

# Supermarket-specific objects — for PX Mart (全聯) and similar stores
# Keep list short (<15) with distinct terms to avoid GroundingDINO label merging
SUPERMARKET_OBJECTS = [
    "shelf", "refrigerator", "freezer",
    "shopping cart", "basket",
    "cash register", "conveyor belt",
    "sign", "poster", "price tag",
    "door", "bottle", "fruit", "vegetable", "bread",
]

# TASK 3 — crop-verify a goal detection with the VLM before trusting it.
# Disabled under test mode so the offline suite never makes a network call.
GOAL_CROP_VERIFY = os.environ.get("GOAL_CROP_VERIFY", "1") != "0" and not os.environ.get("UNIGOAL_TEST_MODE")

# Minimum detection score for an ARRIVED to count as corroborated by perception.
# Below this (and with no OCR sign match), ARRIVED is downgraded to a confirm question.
ARRIVED_MIN_DETECTION_SCORE = float(os.environ.get("ARRIVED_MIN_DETECTION_SCORE", "0.35"))

# OCR
OCR_ENABLED = os.environ.get("OCR_ENABLED", "1") != "0"
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "en,ch_tra").split(",")
OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.3"))
OCR_MAX_RESULTS = int(os.environ.get("OCR_MAX_RESULTS", "15"))
# OCR backend: "openai" (gpt-4o-mini vision via /chat/completions, default) or
# "easyocr" (local EasyOCR engine). The OpenAI backend reuses OPENAI_API_KEY /
# OPENAI_BASE_URL so OCR and VLM share one key and both run on gpt-4o-mini.
OCR_BACKEND = os.environ.get("OCR_BACKEND", "openai").strip().lower()
# OCR model — defaults to the VLM model so both default to gpt-4o-mini; override
# with OPENAI_OCR_MODEL if you want OCR on a different OpenAI model.
OPENAI_OCR_MODEL = os.environ.get("OPENAI_OCR_MODEL", OPENAI_MODEL)

# Timeouts
VLM_TIMEOUT_S = int(os.environ.get("VLM_TIMEOUT_S", "120"))
GOAL_DECOMPOSE_TIMEOUT_S = 30


# TASK 2 — Store-specific label normalization. GroundingDINO labels the same
# physical object differently across photos ("milk bottle" vs "milk carton" vs
# "milk"), which splits one real zone into several during Jaccard clustering.
# normalize_label() collapses every variant to a canonical token before Jaccard.
LABEL_SYNONYMS = {
    # Store sections (EN + ZH)
    "dairy": ["dairy", "milk section", "refrigerated dairy", "dairy aisle",
              "乳製品", "乳品", "鮮乳區", "奶類"],
    "produce": ["produce", "vegetables", "fruits", "fresh produce", "greens",
                "生鮮", "蔬果", "蔬菜", "水果", "青菜"],
    "frozen": ["frozen", "frozen foods", "freezer section", "frozen aisle",
               "冷凍", "冷凍食品", "冷凍區"],
    "bakery": ["bakery", "bread section", "baked goods", "bread aisle",
               "烘焙", "麵包區", "糕餅"],
    "beverages": ["beverages", "beverage", "drinks", "sodas", "water aisle", "juice",
                  "bottled drink", "various beverage",
                  "飲料", "飲品", "果汁", "汽水"],
    "checkout": ["checkout", "cashier", "register", "cash register", "till",
                 "checkout counter", "self checkout", "self-checkout",
                 "收銀", "結帳", "收銀台", "自助結帳", "自助收銀"],
    # Common products (EN + ZH)
    "milk": ["milk", "milk bottle", "milk carton", "whole milk", "skim milk",
             "牛奶", "鮮奶", "鮮乳", "奶"],
    "bread": ["bread", "loaf", "bread loaf", "sandwich bread", "pastri",
              "麵包", "吐司", "土司"],
    "refrigerator": ["refrigerator", "fridge", "cooler", "refrigerated case",
                     "refrigerated display", "refrigerated section",
                     "freezer", "refrigerated",
                     "冰箱", "冷藏", "冷藏櫃", "冷凍櫃", "冰櫃"],
    "shelf": ["shelf", "shelves", "shelving", "rack", "display rack",
              "store shelf", "gondola shelf", "product display",
              "貨架", "架子", "展示架", "陳列架", "層架"],
    "cart": ["cart", "shopping cart", "trolley", "basket", "shopping basket",
             "購物車", "手推車", "推車", "購物籃", "菜籃"],
    "sign": ["sign", "aisle sign", "store sign", "hanging sign",
             "exit sign", "directional sign", "price tag", "banner",
             "標示", "標誌", "招牌", "指示牌", "走道標示", "價格標籤", "吊牌"],
    # Supermarket landmarks (EN + ZH)
    "bottle": ["bottle", "bottles", "water bottle", "plastic bottle",
               "soda bottle", "wine bottle", "beer bottle", "tea bottle",
               "shampoo bottle", "liquor bottle", "detergent bottle",
               "瓶子", "水瓶", "寶特瓶", "塑膠瓶"],
    "can": ["canned", "can ", " can", "cans", "canned food", "tin",
            "beer can", "soda can", "fire extinguisher",
            "罐頭", "罐裝", "鐵罐"],
    "fruit": ["fruit", "fruits", "apple", "banana", "orange", "grape",
              "水果", "蘋果", "香蕉", "柳丁", "橘子", "葡萄"],
    "vegetable": ["vegetable", "lettuce", "cabbage", "carrot", "tomato",
                  "蔬菜", "生菜", "高麗菜", "紅蘿蔔", "番茄"],
    "meat": ["meat", "pork", "beef", "chicken", "fish",
             "肉", "豬肉", "牛肉", "雞肉", "魚", "肉品"],
    "entrance": ["entrance", "exit", "doorway", "gate",
                 "入口", "出口", "大門", "門口", "通道口"],
    "pillar": ["pillar", "column", "post", "support column",
               "柱子", "圓柱", "支柱"],
    "conveyor": ["conveyor belt", "conveyor",
                 "輸送帶", "傳送帶"],
    # Additional common indoor objects (ZH→EN mapping)
    "toilet_paper": ["toilet paper", "tissue", "tissue paper", "paper towel",
                     "衛生紙", "面紙", "紙巾", "廁紙", "捲筒衛生紙"],
    "snack": ["snack", "chips", "potato chips", "snack bag", "cracker",
              "snack pack", "snack box",
              "零食", "洋芋片", "薯片", "餅乾"],
    "rice": ["rice", "rice bag",
             "米", "白米", "稻米"],
    "noodle": ["noodle", "instant noodle", "ramen", "pasta",
               "麵", "泡麵", "即食麵", "義大利麵", "拉麵"],
    "oil": ["oil", "cooking oil", "vegetable oil", "olive oil",
            "油", "食用油", "沙拉油", "橄欖油"],
    "sauce": ["sauce", "soy sauce", "ketchup", "condiment",
              "醬", "醬油", "番茄醬", "調味料"],
    "detergent": ["detergent", "laundry detergent", "cleaning product",
                  "洗衣精", "清潔劑", "洗碗精", "洗潔精"],
    "aisle": ["aisle", "corridor", "walkway", "passage", "shopping aisle",
              "走道", "通道", "走廊"],
    "floor": ["floor", "tile floor", "地板", "地面", "磁磚"],
    "ceiling": ["ceiling", "ceiling light", "天花板", "頂部"],
    "wall": ["wall", "牆", "牆壁", "牆面"],
    "door": ["door", "glass door", "門", "大門", "玻璃門"],
    "escalator": ["escalator", "elevator", "stairs", "staircase",
                  "電扶梯", "手扶梯", "電梯", "樓梯"],
    "restroom": ["restroom", "toilet", "bathroom", "washroom",
                 "洗手間", "廁所", "化妝室"],
    "box": ["boxes", "boxed product", "cardboard box", "stacked box",
            "packaged good", "箱子", "紙箱"],
    "pet_food": ["pet food", "dog food", "cat food",
                 "寵物食品", "狗糧", "貓糧"],
    "plant": ["plant", "flower", "hanging plant",
              "植物", "花", "盆栽"],
    "cup": ["cup", "plastic cup", "paper cup", "paper plate",
            "杯子", "紙杯", "紙盤"],
}


def normalize_label(label: str) -> str:
    """Collapse a raw GroundingDINO label to its canonical store token.

    Idempotent: each canonical key lists itself among its variants, so applying
    this twice is a no-op. Unknown labels are returned lowercased and unchanged.
    """
    label_lower = label.lower()
    for canonical, variants in LABEL_SYNONYMS.items():
        if any(v in label_lower for v in variants):
            return canonical
    return label_lower


def ensure_output_dir(session_id: str) -> Path:
    p = OUTPUT_ROOT / session_id
    (p / "photo").mkdir(parents=True, exist_ok=True)
    (p / "annotated").mkdir(parents=True, exist_ok=True)
    return p
