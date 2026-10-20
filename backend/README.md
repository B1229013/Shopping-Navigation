# 智慧購物導航 — 後端導航伺服器 (Navigation Backend Server)

本目錄為「智慧購物導航」系統的 **Python 後端導航伺服器**，對應設計文件書 §3.2「軟硬體組件設計說明」。負責整個賣場導航的核心運算：**視覺感知 → 拓樸定位 → 多模態語意推論 → 導航決策**。Android App 為前端薄客戶端，僅負責拍照與顯示，所有 AI 運算皆在本伺服器執行。

> **隱私設計**：視覺語言模型透過本地端 **Ollama** 運行，顧客上傳的照片不送往任何雲端 API，亦無任何雲端金鑰。

---

## 系統定位（在整體架構中的角色）

```
┌─────────────────────────────┐         ┌──────────────────────────────────────────┐
│  Android App（客戶端）       │  HTTP   │  Python 後端導航伺服器（本目錄）            │
│  CameraX 擷取 / OpenCV 前處理 │ ──────► │  FastAPI                                    │
│  AR 箭頭與文字導引顯示        │ ◄────── │  GroundingDINO + EasyOCR + Ollama VLM       │
│                             │  JSON   │  NetworkX 拓樸地圖 + 目標拆解 + 路徑決策     │
└─────────────────────────────┘         └──────────────────────────────────────────┘
```

- **測試階段（每間賣場執行一次）**：以所有賣場環境照片一次性預建拓樸地圖，供後續所有顧客 session 重複使用。
- **執行階段（每位顧客）**：顧客每步上傳一張照片，伺服器回傳逐步導航指令（MOVE / ASK / ARRIVED）。

---

## 核心模組

| 模組 | 說明 | 主要檔案 |
|------|------|----------|
| 視覺感知 — 物件偵測 | GroundingDINO SwinT-OGC，零樣本偵測貨架商品、冰箱、門口、走道、標示物等地標 | `server/perception.py` |
| 視覺感知 — 文字萃取 | EasyOCR 讀取走道招牌、區域名稱與貨架標籤（賣場最強的區域辨識訊號） | `server/ocr.py` |
| 拓樸地圖與定位 | NetworkX 無向圖，記錄「使用者在哪裡」與「走過哪裡」，動態建立節點與邊 | `server/topomap.py` |
| 決策與大腦推理 (VLM) | 本地 Ollama 上的 LLaMA 3.2 Vision 11B，整合影像、偵測結果、OCR 與地圖摘要，輸出導航動作 | `server/vlm.py` |
| 目標拆解 | 將自然語言目標（如「找牛奶」）拆解為可偵測的視覺地標清單（如 `["milk", "dairy", "fridge"]`） | `server/goal_decomposer.py` |
| 場景整合 / 抵達驗證 | 整理偵測與 OCR 摘要、比對招牌與目標、把關 ARRIVED 防止誤判抵達 | `server/scene.py` |
| 影像標註 | 在照片上繪製偵測框與 OCR 區域，回傳標註後影像 | `server/annotator.py` |
| Session 狀態 | 每場導航 session 的目標、歷史、目前節點等狀態 | `server/session.py` |
| API 服務 | FastAPI REST 介面 | `server/server.py`、`server/run_server.py` |

**決策輸出**（`server/vlm.py`）標準化為三種導航動作之一：
- **MOVE** — 提供往前或轉彎的具體方向。
- **ASK** — 遇到岔路或不確定時向使用者提問。
- **ARRIVED** — 已抵達目標（需通過偵測分數 / OCR 招牌 / 裁切複驗等把關才成立）。

---

## API 端點

| 動作 | 端點 |
|------|------|
| 建立 session（輸入商品目標） | `POST /session` — body：`{"goal": "find milk"}` → 回傳 `session_id` |
| 上傳一張照片取得導航 | `POST /session/{id}/photo` — multipart 影像 → 回傳 action + guidance + 標註影像 URL |
| 回答 ASK 問題 | `POST /session/{id}/answer` — body：`{"answer": "yes"}` |
| 取得目前拓樸地圖 | `GET /session/{id}/map?format=json\|png` |
| 取得 session 狀態 | `GET /session/{id}` |
| 取得標註後影像 | `GET /session/{id}/photo/{node_id}.jpg` |
| 健康檢查 | `GET /health` |

---

## 環境需求

- **Python 3.12**
- **伺服器硬體**（設計文件書 §3.2）：8 GB VRAM、16+ 執行緒 CPU。
- **Ollama** 本地服務並已拉取 `llama3.2-vision` 模型。
- **GroundingDINO 權重**：`models/groundingdino_swint_ogc.pth`（請另行下載，勿入庫）。

---

## 安裝與啟動

```bash
# 1) 安裝伺服器執行階段相依套件
pip install -r requirements-server.txt

# 2) 安裝 Ollama（https://ollama.com）並啟動服務，拉取視覺模型
ollama pull llama3.2-vision        # 約 7–8 GB，首次拉取較久
ollama serve                       # 預設監聽 127.0.0.1:11434

# 3) 下載 GroundingDINO 權重（zero-shot 物件偵測模型），放到 models/groundingdino_swint_ogc.pth
#    權重來源：https://github.com/IDEA-Research/GroundingDINO（檔名 groundingdino_swint_ogc.pth）
#    並 clone GroundingDINO 以取得設定檔 GroundingDINO_SwinT_OGC.py（路徑由 server/config.py 自動偵測）
#    EasyOCR 模型會於首次執行時自動下載，無需手動處理

# 4) 啟動伺服器（於本 backend/ 目錄下執行）
python -m server.run_server
# 或
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

> 建圖管線（下節）另需 ML 相依套件：`torch`、`groundingdino`、`matplotlib`、`pyvis`、`pillow`、`pillow-heif` 等；建議於具 GPU 的環境（如 WSL conda 環境）執行。

---

## 使用方式 (Usage)

伺服器啟動後，一次顧客導航流程如下（以 `curl` 為例）：

```bash
# 1) 建立 session，輸入要找的商品
curl -X POST http://localhost:8000/session \
  -H "Content-Type: application/json" \
  -d '{"goal": "find milk"}'
# → { "session_id": "...", "guidance": "...", "goal_objects": ["milk","dairy",...] }

# 2) 每走一步上傳一張照片，取得導航動作
curl -X POST http://localhost:8000/session/<session_id>/photo \
  -F "photo=@step1.jpg"
# → { "action": "MOVE|ASK|ARRIVED", "guidance": "...", "question": "...",
#     "annotated_photo_url": "/session/<id>/photo/0.jpg" }

# 3) 若 action 為 ASK，回答後繼續
curl -X POST http://localhost:8000/session/<session_id>/answer \
  -H "Content-Type: application/json" \
  -d '{"answer": "yes"}'
```

- `action = MOVE` → 顯示 `guidance` 指引顧客移動，再拍下一張照片。
- `action = ASK` → 顯示 `question`，由顧客回答（`/answer`）。
- `action = ARRIVED` → 已抵達目標，session 結束。

> Android App 端由 `NavigationScreen` 對應呼叫上述端點（目前 App 的 `processImageForModel()` 仍為佔位實作，待串接此後端）。

---

## 建圖管線（測試階段）

以賣場照片一次性建立拓樸地圖，依序執行：

| 步驟 | 腳本 | 功能 |
|------|------|------|
| OCR | `ocr_dump.py` | 對所有賣場照片跑 EasyOCR，輸出 `store_ocr.json` |
| 偵測 + 富化 | `build_store_map.py` | 將 OCR 招牌類別併入 GroundingDINO 偵測結果作為標準化標籤 |
| 建圖（序列） | `generate_topomap.py` | Jaccard 分群成區域、建立節點與邊、輸出拓樸地圖（PNG/HTML） |
| 建圖（分支樓面） | `build_store_graph.py` | 以主走道 + 分支走道（1–13）+ 周界生鮮區的真實賣場拓樸建模 |
| 精簡渲染 | `render_clean.py` | 蛇形版面、每區一張卡片的精簡可讀地圖 |
| 互動式 HTML | `build_html_map.py` | Pyvis 可拖曳節點的互動式 HTML 地圖，便於核對實際樓面 |

```bash
python generate_topomap.py --sim-threshold 0.40
```

---

## Neo4j 拓樸地圖（V2 Schema）

系統使用 **Neo4j Aura** 雲端圖資料庫儲存預建的參考地圖。V2 schema 適用於所有場域（A7家樂福、系辦等）。

### 節點類型

| 節點 | 說明 | 主要屬性 |
|------|------|----------|
| `Waypoint` | 地圖上的一個位置點 | `place`, `nid`, `pdr_x`, `pdr_y`, `heading_deg`, `session`, `total_steps`, `total_distance_m` |
| `DirPhoto` | 某個 Waypoint 某方向的照片 | `photo_file`, `heading_deg` |
| `Object` | 照片中偵測到的物件 | `label`, `label_norm`, `score`, `bbox`(normalized 0-1), `grid_cell`, `grid_cells`, `piece_count`, `depth_class`, `ocr_text`, `source` |

### 關係類型

| 關係 | 說明 | 屬性 |
|------|------|------|
| `HAS_PHOTO` | Waypoint → DirPhoto | `slot`（front/back/left/right） |
| `DETECTED` | DirPhoto → Object | （無屬性） |
| `WALKWAY` | Waypoint → Waypoint | `direction`, `distance_m`, `steps` |
| `ADJACENT` | Object → Object（同一張照片內） | `cell_pairs`, `overlap`, `vertical_count`, `horizontal_count`, `diagonal_count`, `depth_diff` |

### ADJACENT 空間關係

ADJACENT 記錄同一張照片內兩個物件之間的空間關係，用於定位時的 **spatial bonus** 加分。

**DB 儲存格式**（`cell_pairs`）：每個物件先映射到 3×3 九宮格（top/mid/bottom × left/center/right），再計算所有格子對之間的方向。

```json
{
  "cell_pairs": [
    {"a": "top-center", "b": "mid-right", "dir": "below-right"},
    {"a": "top-center", "b": "top-right", "dir": "right"}
  ],
  "overlap": false,
  "depth_diff": "closer"
}
```

**Server 讀取時**（`neo4j_client.py`）：從 `cell_pairs` 提取方向，轉換為簡化格式供定位比對：

```python
{
  "v_rel": "above" | "below" | "same_row",
  "h_rel": "left" | "right" | "same_col",
  "overlap": True | False,
  "depth_diff": "closer" | "farther" | "same_level"
}
```

**定位比對時**（`visual_localization.py`）：
1. Query 照片偵測物件 → `compute_query_spatial()` 從 bbox center 直接算出物件對的 `v`/`h`/`overlap`
2. `_spatial_direction_match()` 比對 query 方向 vs. DB 中的方向
3. 匹配成功的物件對獲得 spatial bonus，上限由 `SPATIAL_BONUS_COEFF`（目前 **0.10**，即最多 +10%）控制

---

## 地圖建立流程

### 1. 資料收集與拓樸校正

手動在場域中沿路線拍攝四方向照片（front/back/left/right），記錄 PDR 座標，產出 `topology_map.json`。手動校正後成為 `topology_map_corrected.json`。

### 2. 上傳骨架至 Neo4j

```bash
python upload_topo_neo4j.py --skip-vlm
```

建立 Waypoint + DirPhoto + WALKWAY 骨架。`--skip-vlm` 跳過物件偵測（由下一步獨立處理）。

### 3. 物件偵測與 ADJACENT 建立

```bash
cd scripts/vlm_test
DATA_ROOT=/path/to/photos python3 build_object_graph.py --rounds 3
```

- 讀取 Neo4j 中已有的 DirPhoto 節點
- 對每張照片執行 VLM 兩階段偵測（`gpt-5.4-mini`）
- `--rounds 3`：多輪穩定性過濾（出現 2+/3 輪的物件才保留）
- 建立 Object 節點 + ADJACENT 邊（物件間空間關係）
- `DATA_ROOT` 指向照片目錄，`NEO4J_PLACE` 設定場域名稱

---

## 視覺定位（Visual Localization）

Server 啟動時從 Neo4j 載入完整的參考地圖（`neo4j_client.py`），包含 Waypoint、DirPhoto、Object、WALKWAY 和 **ADJACENT** 空間關係。

定位流程（`visual_localization.py`）：

1. **VLM 感知**：上傳照片 → VLM 偵測物件與 OCR 文字
2. **Grid Matching**：Query 物件 vs. 參考照片物件，以 label 語意相似度 + 稀有度加權比對
3. **Spatial Bonus**：若配對到 ≥2 個物件，檢查它們的空間排列是否與 DB 中 ADJACENT 記錄一致，一致則加分
4. **DINOv2 Re-ranking**（若啟用）：用視覺特徵重排序 top candidates
5. **輸出**：最佳匹配的 Waypoint nid + confidence

### 定位相關參數

| 參數 | 值 | 位置 | 說明 |
|------|-----|------|------|
| `SPATIAL_BONUS_COEFF` | 0.10 | `visual_localization.py:208` | ADJACENT 空間比對最大加分比例 |
| Semantic similarity threshold | 0.25 | `visual_localization.py:513` | 物件配對最低語意相似度 |
| Spatial direction match threshold | 0.3 | `visual_localization.py:543` | 方向匹配分數需 > 0.3 才算 hit |

---

## 測試

```bash
# 單元測試（離線、不打網路；設定 UNIGOAL_TEST_MODE）
pytest

# 評估 / 回放 harness
#   見 eval/ 目錄（runner.py、replay.py、metrics.py）
```

---

## 設定（環境變數，詳見 `server/config.py`）

| 變數 | 預設 | 說明 |
|------|------|------|
| `NEO4J_URI` | — | Neo4j Aura 連線 URI（如 `neo4j+s://xxx.databases.neo4j.io`） |
| `NEO4J_USER` | — | Neo4j 使用者名稱 |
| `NEO4J_PASSWORD` | — | Neo4j 密碼 |
| `NEO4J_PLACE` | — | 目前場域名稱（須與 DB 中 `place` 完全一致，如 `系辦`） |
| `OPENAI_API_KEY` | — | OpenAI API Key（VLM 偵測用） |
| `OPENAI_MODEL` | `gpt-5.4-mini` | VLM 模型名稱 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API endpoint |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 服務位址（本地 VLM 模式） |
| `OLLAMA_MODEL` | `llama3.2-vision` | 本地視覺語言模型名稱 |
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `8000` | 伺服器位址與埠 |
| `OCR_ENABLED` | `1` | 是否啟用 EasyOCR |
| `OCR_LANGUAGES` | `en,ch_tra` | OCR 語言（英文 + 繁體中文） |
| `GROUNDINGDINO_BOX_THRESHOLD` | `0.40` | 偵測框信心閾值 |
| `GOAL_CROP_VERIFY` | `1` | ARRIVED 前裁切目標框交由 VLM 複驗 |
| `ARRIVED_MIN_DETECTION_SCORE` | `0.35` | ARRIVED 成立所需的最低偵測佐證分數 |

---

## 目錄結構

```
backend/
├── server/                     # FastAPI 後端核心
│   ├── server.py               # REST API 端點
│   ├── run_server.py           # uvicorn 啟動點
│   ├── neo4j_client.py         # Neo4j Aura HTTP 連線 + V1/V2 schema 自動偵測
│   ├── visual_localization.py  # Grid Matching + Spatial Bonus + DINOv2 定位
│   ├── grid_utils.py           # 九宮格計算（compute_grid_cells）
│   ├── vlm.py                  # VLM 多階段物件偵測（OpenAI / Ollama）
│   ├── perception.py           # GroundingDINO 物件偵測
│   ├── ocr.py                  # EasyOCR 文字辨識
│   ├── goal_decomposer.py      # 目標 → 視覺地標清單
│   ├── topomap.py              # NetworkX 拓樸地圖
│   ├── scene.py                # 場景整合與抵達驗證
│   ├── annotator.py            # 偵測框 / OCR 標註
│   ├── session.py              # session 狀態
│   ├── prompts.py / models.py / navigator.py / editor_map.py
│   └── tests/                  # pytest 單元測試
├── upload_topo_neo4j.py        # 拓樸骨架上傳至 Neo4j（Waypoint + DirPhoto + WALKWAY）
├── maps/                       # 場域拓樸地圖 JSON（editor 格式）
├── eval/                       # 評估 / 回放 harness
└── requirements-server.txt     # 伺服器執行階段相依套件

scripts/vlm_test/
├── build_object_graph.py       # Object + ADJACENT 重建（VLM 偵測 + Neo4j 寫入）
└── ...
```
