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

# LLM services — "gemini" (default), "openai", or "ollama"
VLM_BACKEND = os.environ.get("VLM_BACKEND", "gemini")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")

# Server
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

# Detection thresholds (TASK 1 — raised for busy store shelves; retry lower only if nothing found)
PERCEPTION_ENABLED = os.environ.get("PERCEPTION_ENABLED", "1") != "0"
GROUNDINGDINO_BOX_THRESHOLD = 0.30
GROUNDINGDINO_TEXT_THRESHOLD = 0.25
GROUNDINGDINO_BOX_THRESHOLD_FALLBACK = 0.20
SAM_TOP_K_BOXES = 10

# Generic indoor objects — always detected alongside goal_objects for spatial context
GENERIC_INDOOR_OBJECTS = [
    # 同學的 DEFAULT_CLASSES（19 項）
    "door", "cabinet", "chair", "table", "desk", "sofa", "sign",
    "fire extinguisher", "plant", "printer", "whiteboard", "monitor",
    "computer", "bookshelf", "window", "counter", "reception",
    "bulletin board", "notice board",
    # 額外補充
    "refrigerator", "water dispenser", "shelf", "trash can", "poster",
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

# Timeouts
VLM_TIMEOUT_S = int(os.environ.get("VLM_TIMEOUT_S", "120"))
GOAL_DECOMPOSE_TIMEOUT_S = 30


# TASK 2 — Store-specific label normalization. GroundingDINO labels the same
# physical object differently across photos ("milk bottle" vs "milk carton" vs
# "milk"), which splits one real zone into several during Jaccard clustering.
# normalize_label() collapses every variant to a canonical token before Jaccard.
LABEL_SYNONYMS = {
    # Store sections
    "dairy": ["dairy", "milk section", "refrigerated dairy", "dairy aisle"],
    "produce": ["produce", "vegetables", "fruits", "fresh produce", "greens"],
    "frozen": ["frozen", "frozen foods", "freezer section", "frozen aisle"],
    "bakery": ["bakery", "bread section", "baked goods", "bread aisle"],
    "beverages": ["beverages", "drinks", "sodas", "water aisle", "juice"],
    "checkout": ["checkout", "cashier", "register", "cash register", "till"],
    # Common products
    "milk": ["milk", "milk bottle", "milk carton", "whole milk", "skim milk"],
    "bread": ["bread", "loaf", "bread loaf", "sandwich bread"],
    "refrigerator": ["refrigerator", "fridge", "cooler", "refrigerated case", "freezer"],
    "shelf": ["shelf", "shelving", "rack", "display rack", "store shelf"],
    "cart": ["cart", "shopping cart", "trolley", "basket"],
    "sign": ["sign", "aisle sign", "store sign", "label", "price tag"],
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
