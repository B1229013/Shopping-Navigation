"""Configuration constants and paths.

Paths are auto-detected for both WSL and Windows environments.
Model weights and configs are resolved relative to the project root.
"""
import os
from pathlib import Path

# Project root (where this repo lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env — check project root first, then parent (Shopping-Navigation/.env)
for _dotenv in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:
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
# Hand-corrected PDR waypoint maps (see server/editor_map.py); one *.json per place
EDITOR_MAP_DIR = PROJECT_ROOT / "maps"
# Where /path routes come from: "neo4j" = the Neo4j reference map itself (one set of
# coordinates with localization) + maps/neo4j_links/<place>.json for missing links and
# the entrance; "editor" = the hand-corrected maps/*.json walks.
ROUTE_MAP_SOURCE = os.environ.get("ROUTE_MAP_SOURCE", "neo4j").strip().lower()
# Building a new store map (server/map_v2.py): one workspace folder per store under this root
# (MAP_BUILD_ROOT/<店名>/ holds topomap.json, <set>/ photos, topomap_confirmed.json, …; see
# server/map_build.workspace_for), and an optional Neo4j place drawn under the builder as a
# reference (e.g. an older map of the same store). MAP_BUILD_DIR is the old name of the root.
MAP_BUILD_ROOT = Path(os.environ.get("MAP_BUILD_ROOT", "") or os.environ.get("MAP_BUILD_DIR", "")
                      or PROJECT_ROOT / "output" / "mapbuild")
MAP_BUILD_REF_PLACE = os.environ.get("MAP_BUILD_REF_PLACE", "").strip()
# Folders the builder's 「路線」 dialog lists Sensor Lab exports from (os.pathsep-separated)
MAP_BUILD_SOURCES = [Path(p) for p in os.environ.get("MAP_BUILD_SOURCES", "").split(os.pathsep) if p.strip()] \
    or [PROJECT_ROOT / "legacy" / "testPDR", PROJECT_ROOT / "testPDR"]

# LLM service — any OpenAI-compatible chat API (OpenAI, or the CGU gateway via OPENAI_BASE_URL)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# gpt-5.4-mini: the model the team runs (CGU gateway). Any OpenAI-compatible model works;
# vlm._generate sends temperature 0.2. The VLM eval results (eval/results/*_vlm*.json) used it too.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Neo4j Aura
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
NEO4J_PLACE = os.environ.get("NEO4J_PLACE", "A7家樂福（9/16)")

# Server
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

# Detection thresholds (TASK 1 — raised for busy store shelves; retry lower only if nothing found)
PERCEPTION_ENABLED = os.environ.get("PERCEPTION_ENABLED", "0") != "0"
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
# Occluders — also detected (Mode A) so photo_quality can warn when they block the view;
# dropped from localization items (server._build_localization_items). Kept out of
# GENERIC_INDOOR_OBJECTS, which also filters goal words ("find a shopping cart").
OCCLUDER_PROMPTS = ["person", "shopping cart"]

# TASK 3 — crop-verify a goal detection with the VLM before trusting it.
# Disabled under test mode so the offline suite never makes a network call.
GOAL_CROP_VERIFY = os.environ.get("GOAL_CROP_VERIFY", "1") != "0" and not os.environ.get("UNIGOAL_TEST_MODE")

# Minimum detection score for an ARRIVED to count as corroborated by perception.
# Below this (and with no OCR sign match), ARRIVED is downgraded to a confirm question.
ARRIVED_MIN_DETECTION_SCORE = float(os.environ.get("ARRIVED_MIN_DETECTION_SCORE", "0.50"))

# OCR
OCR_ENABLED = os.environ.get("OCR_ENABLED", "0") != "0"
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "en,ch_tra").split(",")
OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.3"))
OCR_MAX_RESULTS = int(os.environ.get("OCR_MAX_RESULTS", "15"))

# Grid size for spatial matching (3 = 3×3, 4 = 4×4, 5 = 5×5)
GRID_N = int(os.environ.get("GRID_N", "4"))

# Localization fusion weights "dino,sensor,grid" (see server/loc_fusion.py).
# Only applies when the phone sends DINOv2 top-K and/or a calibrated map position;
# "0,0,1" restores grid-only localization.
LOC_FUSION_WEIGHTS = os.environ.get("LOC_FUSION_WEIGHTS", "4,1,1")

# Adaptive fusion (eval/fusion_sweep.py policy): when the phone sends the distance
# walked since its last photo fix, the sensor weight follows the expected PDR error
#   σ = LOC_SENSOR_SIGMA0 + LOC_DRIFT_RATE · distance,  w_sensor = min(CAP, C / σ)
# and the dino / grid weights come from LOC_FUSION_WEIGHTS. LOC_DRIFT_RATE should be
# re-measured with SensorTestView on known routes.
LOC_FUSION_ADAPTIVE = os.environ.get("LOC_FUSION_ADAPTIVE", "1") != "0"
LOC_SENSOR_SIGMA0 = float(os.environ.get("LOC_SENSOR_SIGMA0", "1.0"))
LOC_DRIFT_RATE = float(os.environ.get("LOC_DRIFT_RATE", "0.05"))
LOC_SENSOR_C = float(os.environ.get("LOC_SENSOR_C", "40"))
LOC_SENSOR_CAP = float(os.environ.get("LOC_SENSOR_CAP", "40"))
# Scale the grid weight by min(1, detected_objects / N); 0 = off (the sweep found no gain).
LOC_GRID_NFULL = float(os.environ.get("LOC_GRID_NFULL", "0"))
# Scale the DINOv2 weight by min(1, (top1 − top2 cosine) / G); 0 = off.
LOC_DINO_GAP_FULL = float(os.environ.get("LOC_DINO_GAP_FULL", "0.05"))

# v2 fusion (loc_fusion.fuse_v2): posterior over every node from image evidence, the
# phone's map position (Gaussian, σ from distance walked) and walkability from the last
# fix. Replaces the weighted average above whenever the phone sends DINOv2 / position.
LOC_FUSION_V2 = os.environ.get("LOC_FUSION_V2", "1") != "0"
# k(σ) "k_min,k_max,k_no_sensor" for the phone's DINOv2 top-K evidence (iOS):
# image weight at σ ≤ 2 m, at σ ≥ 5 m, and before the first fix (loc_fusion.ImageK).
LOC_PHONE_IMAGE_K = os.environ.get("LOC_PHONE_IMAGE_K", "0.25,4,2")
# Ask for a second photo (turn ~90° on the spot) when the best node's probability is
# below this; the next photo within SECOND_PHOTO_WINDOW_S and < SECOND_PHOTO_MAX_MOVE_M
# of walking adds its image evidence to this one's.
SECOND_PHOTO_BELOW = float(os.environ.get("SECOND_PHOTO_BELOW", "0.4"))
SECOND_PHOTO_WINDOW_S = float(os.environ.get("SECOND_PHOTO_WINDOW_S", "120"))
SECOND_PHOTO_MAX_MOVE_M = float(os.environ.get("SECOND_PHOTO_MAX_MOVE_M", "3"))
# Places whose aisle-number signs feed localization (server/aisle_signs.py), "|"-separated.
# The row table and the "≥ 3 numbers = central walkway" rule were checked only on this
# store's layout (eval/route1_eval.py); add another store after checking it there.
AISLE_SIGN_PLACES = {p.strip() for p in os.environ.get("AISLE_SIGN_PLACES", NEO4J_PLACE).split("|")
                     if p.strip()}

# Server-side DINOv2 image evidence (server/locate_fusion.server_image_evidence), for
# phones that do not run DINOv2 themselves (Android): when a photo arrives without
# dino_topk, the server embeds it (facebook/dinov2-base, needs torch + transformers)
# and fuses DINOv2 : objects : OCR at 1 : 0.1 : 0.3 (locate_fusion.W_V/W_N/W_O) against DINO_SIDECAR_PATH (one
# 768-d vector per reference photo, keyed by photo file name). Missing model or
# sidecar → falls back to grid matching as before.
LOC_SERVER_DINO = os.environ.get("LOC_SERVER_DINO", "1") != "0"
DINO_SIDECAR_PATH = os.environ.get("DINO_SIDECAR_PATH",
                                   str(Path(__file__).resolve().parent.parent / "dino_sidecar.pkl"))
# k(σ) "k_min,k_max,k_no_sensor" for that evidence in fuse_v2. From 5-fold cross-
# validation (grouped by node) over 268 held-out 0916 photos with simulated walks:
# average top-1 68.0 % vs 43.1 % with the phone-evidence k (0.25,4,2) and 65.7 %
# image-only. k_no_sensor 8 names a node for 98.5 % of first photos (2 → 27.6 %).
LOC_SERVER_IMAGE_K = os.environ.get("LOC_SERVER_IMAGE_K", "32,64,8")
# Object-to-object 上下左右 relations (server/object_relations.py) added to the server image
# evidence: fused_n += W · relation consistency with node n's best directional photo.
# On (team decision 2026-10-22) at 0.05, the weight picked on held-out walks; 0 = off.
# On the 376 lab1019 photos it is not a clear gain: 44.7 % Top-1 without it, 42.6–44.9 % when W
# is picked on the other walks (best +1 photo, with worse 1-hop and error):
# eval/results/object_relations_lab1019.md.
LOC_RELATION_WEIGHT = float(os.environ.get("LOC_RELATION_WEIGHT", "0.05"))
# Load the map's ADJACENT object pairs (map_v2, 3×3 grid cells) for the grid matcher's
# spatial bonus (visual_localization.SPATIAL_BONUS_COEFF; from APPNAV_ios fe4c851, with its
# direction reading fixed). Only the grid path uses it: a phone that sends dino_topk (iOS)
# and maps without DINOv2; server-side DINOv2 does not. On (team decision 2026-10-22);
# 0 = off. On the 376 lab1019 photos it lowered the grid ranking (Top-1 13.6 → 11.2 % at
# 0.10) and moved the phone-DINOv2 path by +0.8 pt (37.5 → 38.3 %, 3 photos) —
# eval/results/relations_compare_lab1019.md.
LOC_ADJACENT = os.environ.get("LOC_ADJACENT", "1") != "0"

# Fail at startup, not on every photo, if either k(σ) setting is malformed.
from server.loc_fusion import parse_image_k as _parse_image_k  # noqa: E402
_parse_image_k(LOC_PHONE_IMAGE_K)
_parse_image_k(LOC_SERVER_IMAGE_K)

# Grid-matching weight overrides, e.g. "landmark=6,ocr=3,rarity_mode=idf"
# (keys: see visual_localization.apply_weight_overrides). Empty = code defaults.
LOC_GRID_WEIGHTS = os.environ.get("LOC_GRID_WEIGHTS", "")

# VLM multi-round perception (3 = run 3 rounds, keep stable detections appearing 2+/3)
VLM_ROUNDS = int(os.environ.get("VLM_ROUNDS", "1"))

# VLM 3rd pass — extra perception pass focusing on distinctive features
VLM_PASS3 = os.environ.get("VLM_PASS3", "1") != "0"

# Hybrid embedding similarity (hybrid_max = max(char_sim, embedding_cosine_sim)) for object
# labels. The reference map's label vectors are fetched once and kept in EMBEDDING_CACHE_PATH
# (.npz, or the older .json); a photo's labels are embedded per photo (one small request).
# Until 2026-10 the cache only existed under REF_PHOTO_ROOT, which deployments do not set, so
# label matching was character-only (eval/results/weight_sweep_lab1019_*: 44.7% → 48.1%).
EMBEDDING_ENABLED = os.environ.get("EMBEDDING_ENABLED", "1") != "0"
EMBEDDING_CACHE_PATH = os.environ.get("EMBEDDING_CACHE_PATH") or (
    str(Path(os.environ["REF_PHOTO_ROOT"]) / "檢索圖片" / "embedding_cache" / "label_embeddings.json")
    if os.environ.get("REF_PHOTO_ROOT") else str(PROJECT_ROOT / "output" / "label_embeddings.npz"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
# Where the vectors come from: by default the VLM's own service and key (OPENAI_BASE_URL /
# OPENAI_API_KEY), so a CGU-gateway key is never sent to api.openai.com. If that service has
# no /embeddings endpoint, point these two at OpenAI; a failed request leaves the labels on
# character similarity only.
EMBEDDING_BASE_URL = (os.environ.get("EMBEDDING_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY") or OPENAI_API_KEY

# VLM Re-ranking (optional, adds ~10s per photo for improved accuracy)
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "0") != "0"
REF_PHOTO_ROOT = os.environ.get("REF_PHOTO_ROOT", "")
OPENAI_BACKUP_KEY = os.environ.get("OPENAI_BACKUP_KEY", "")
RERANK_GAP_THRESHOLD = float(os.environ.get("RERANK_GAP_THRESHOLD", "0.05"))

# Timeouts
VLM_TIMEOUT_S = int(os.environ.get("VLM_TIMEOUT_S", "120"))
# Seconds to wait after an HTTP 429 (rate limited) before the one retry. Short: the photo
# request runs on the server's event loop, so the wait stalls every session.
VLM_RATE_LIMIT_WAIT_S = float(os.environ.get("VLM_RATE_LIMIT_WAIT_S", "5"))
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
