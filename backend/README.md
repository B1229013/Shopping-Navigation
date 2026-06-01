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

# 2) 安裝並啟動 Ollama，拉取視覺模型
ollama pull llama3.2-vision

# 3) 下載 GroundingDINO 權重，放到 models/groundingdino_swint_ogc.pth
#    （並備妥 GroundingDINO 設定檔，路徑由 server/config.py 自動偵測）

# 4) 啟動伺服器（於本 backend/ 目錄下執行）
python -m server.run_server
# 或
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

> 建圖管線（下節）另需 ML 相依套件：`torch`、`groundingdino`、`matplotlib`、`pyvis`、`pillow`、`pillow-heif` 等；建議於具 GPU 的環境（如 WSL conda 環境）執行。

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
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 服務位址 |
| `OLLAMA_MODEL` | `llama3.2-vision` | 視覺語言模型名稱 |
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `8000` | 伺服器位址與埠 |
| `OCR_ENABLED` | `1` | 是否啟用 EasyOCR |
| `OCR_LANGUAGES` | `en,ch_tra` | OCR 語言（英文 + 繁體中文） |
| `GROUNDINGDINO_BOX_THRESHOLD` | `0.40` | 偵測框信心閾值（針對忙碌貨架調高以降低誤判） |
| `GOAL_CROP_VERIFY` | `1` | ARRIVED 前裁切目標框交由 VLM 複驗 |
| `ARRIVED_MIN_DETECTION_SCORE` | `0.35` | ARRIVED 成立所需的最低偵測佐證分數 |

---

## 目錄結構

```
backend/
├── server/                     # FastAPI 後端核心
│   ├── server.py               # REST API 端點
│   ├── run_server.py           # uvicorn 啟動點
│   ├── perception.py           # GroundingDINO 物件偵測
│   ├── ocr.py                  # EasyOCR 文字辨識
│   ├── vlm.py                  # Ollama VLM 導航決策（MOVE/ASK/ARRIVED）
│   ├── goal_decomposer.py      # 目標 → 視覺地標清單
│   ├── topomap.py              # NetworkX 拓樸地圖
│   ├── scene.py                # 場景整合與抵達驗證
│   ├── annotator.py            # 偵測框 / OCR 標註
│   ├── session.py              # session 狀態
│   ├── batch_mapper.py         # 批次照片處理 + Jaccard 分群
│   ├── config.py               # 閾值與路徑設定
│   ├── prompts.py / models.py / store_knowledge.py / store_map.py / navigator.py
│   └── tests/                  # pytest 單元測試
├── eval/                       # 評估 / 回放 harness
├── generate_topomap.py         # 拓樸地圖生成（建圖主入口）
├── build_store_map.py          # OCR 富化 + 建圖
├── build_store_graph.py        # 分支樓面拓樸建模
├── build_html_map.py           # Pyvis 互動式 HTML 地圖
├── render_clean.py             # 精簡地圖渲染
├── ocr_dump.py                 # 批次 OCR
└── requirements-server.txt     # 伺服器執行階段相依套件
```
