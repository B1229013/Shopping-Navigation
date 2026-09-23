# Shopping Navigation — 室內導航系統

**VLM（視覺語言模型）視覺推理**即時室內導航 App。
使用者邊走邊拍，系統透過物件偵測、文字辨識、VLM 決策與感測器航位推算，
逐步建立帶有步數/距離/航向的拓樸地圖，並提供導航指引。

---

## 系統功能綜覽

系統由 **Android App（前端）** 與 **Python FastAPI（後端）** 組成，後端為共用雲端服務，
兩端透過 HTTP/JSON 溝通。完整功能如下：

**導航（兩種模式）**

- **VLM 拍照導航** — 輸入目標（如「找冰箱」），拍照後由物件偵測 + OCR + VLM
  綜合判斷，逐步回傳 MOVE / ARRIVED / ASK 指引，並在照片上標註偵測結果。
- **Sensor-Nav / PDR 導航** — 開啟後 7 種感測器持續記錄步數與航向，邊走邊拍
  自動建圖，指引會結合最短路徑給出「還需幾步 / 幾公尺」的量化提示。

**地圖與定位**

- **拓樸地圖 v2** — 場所等級地圖，照片節點 + 物件節點子圖、多方向支援，可跨導航重用。
- **照片定位（locate_v2）** — 給一張新照片，推理它對應到既有地圖的哪個節點
  （embedding 相似度 + IDF 權重 + 網格比對）。
- **路徑規劃（route_planner）** — 地圖上的 A* 最短路徑與多目標 TSP 排序。
- **Neo4j 雲端地圖** — 地圖存入/讀出 Neo4j Aura，供多人共用同一份場所地圖。

**資料採集與工具**

- **離線採集模式** — 不需後端，照片 + PDR + 原始感測器全存手機，可匯出 ZIP。
- **感測器實驗室** — 8 種感測器即時錄製、CSV 匯出、歷史瀏覽與分享。
- **CGU LLM Gateway** — 一把 `CGU_API_KEY` 通吃 chat / vision / OCR / embedding。
- **雲端部署** — 附 Dockerfile 與 render.yaml，可一鍵部署到 Render。

| 功能 | 模組 | 說明 |
|------|------|------|
| VLM 拍照導航 | `server.py` + `vlm.py` | 拍照 → 物件偵測 + OCR + VLM 決策 → 逐步指引（MOVE/ARRIVED/ASK） |
| Sensor-Nav / PDR 導航 | `sensor_nav.py` + Android `PdrTracker` | PDR 計步 + AI 視覺融合，邊走邊拍自動建圖（含步數/距離/航向） |
| 拓樸地圖 v2 | `topomap_v2.py` | 場所等級地圖：照片節點 + 物件節點子圖、多方向、可跨多次導航重用 |
| 照片定位 | `locate_v2.py` | 給一張照片，推理它對應到已建地圖中哪個節點 |
| 路徑規劃 | `route_planner.py` | TopoGraphV2 上的 A* 最短路徑與多目標 TSP 排序 |
| Neo4j 雲端地圖 | `neo4j_map_store.py` | 把地圖存入/讀出 Neo4j Aura，供多人共用 |
| CGU LLM Gateway | `llm_gateway.py` | 長庚大學 OpenAI 相容 gateway：一把金鑰通吃 chat/vision/OCR/embedding |
| 離線採集 | Android `LocalSessionStore` | 不需後端，照片 + 感測器全存手機，可匯出 ZIP |
| 感測器實驗室 | Android `SensorLabScreen` | 8 種感測器即時錄製、CSV 匯出、歷史瀏覽與分享 |

> Sensor-Nav / PDR 的完整說明（感測器規格、PDR 演算法、資料格式、離線模式）見
> [`docs/sensor-nav-README.md`](docs/sensor-nav-README.md)。

---

## 如何操作（快速上手）

> 完整安裝步驟見下方 [從零開始安裝](#從零開始安裝)；此處為操作全貌。

**環境準備**

1. 啟動後端：`cd backend` → `.\start.ps1`（本機）或部署到 Render 取得雲端 URL。
2. 設定 App 連線：模擬器用預設 `http://10.0.2.2:8000/`；實體手機用
   `adb reverse tcp:8000 tcp:8000` 或在設定頁填入後端 URL。
3. 開啟 `http://<後端>/health` 回傳 `{"status":"ok"}` 即代表連線正常。

**操作流程 A — VLM 拍照導航**

1. App 進入「導航」頁 → 輸入目標（如「找冰箱」）→ 開始導航。
2. 拍照或選相簿照片 → 等待分析 → 依 MOVE/ARRIVED/ASK 指引移動。
3. 重複拍照直到 ARRIVED；過程自動建立拓樸地圖。

**操作流程 B — Sensor-Nav / PDR 導航**

1. 設定頁 → 「感測器導航」→ 輸入目的地 → 開始導航（授予相機與活動辨識權限）。
2. 邊走邊拍，畫面顯示步數/距離統計與小地圖軌跡，指引含「還需幾步」提示。
3. 確認到達後可儲存為可重用地圖；或切「離線採集模式」把資料存手機、之後匯出 ZIP。

**進階**

- 已建地圖可用照片定位（`locate_v2`）確認目前位置，或用路徑規劃（`route_planner`）
  排多目標最短路線；多人共用地圖時把地圖上傳至 Neo4j。

---

## 系統架構

```
┌──────────────────────────┐   HTTP/JSON    ┌──────────────────────────────────────┐
│  Android App (Kotlin)    │ ◄────────────► │  Python FastAPI Backend                │
│                          │   Retrofit     │                                        │
│  Camera / Gallery ───────┼─ JPEG ───────► │  ┌─ GroundingDINO (物件偵測)          │
│  PdrTracker (7 sensors) ─┼─ PDR JSON ───► │  ├─ EasyOCR (文字辨識)               │
│                          │                │  ├─ VLM (GPT-4o / Gemini / CGU)       │
│  NavigationScreen        │                │  ├─ Annotator (標註圖)                │
│  SensorNavScreen ◄───────┼── TurnResponse │  ├─ TopoMap / TopoMap v2 (拓樸地圖)   │
│  SensorLabScreen         │                │  ├─ locate_v2 (照片定位)             │
│  Settings / ...          │                │  ├─ route_planner (路徑規劃)         │
│                          │                │  └─ Neo4j / MapStore (地圖持久化)     │
└──────────────────────────┘                └──────────────────────────────────────┘
```

## 導航流程

### VLM 拍照導航（`/session`）

1. **建立 Session** — 使用者輸入目標（如「找冰箱」），VLM 分解出偵測關鍵字
2. **拍照上傳** — GroundingDINO 偵測物件、EasyOCR 讀取文字
3. **VLM 決策** — 綜合偵測結果 + 照片 + 歷史路徑，回傳 MOVE / ARRIVED / ASK
4. **標註圖** — 在照片上標示所有偵測框和 OCR 文字（含信心值）
5. **拓樸地圖** — 每一步自動建立並保存地圖（JSON + PNG）

### Sensor-Nav / PDR 導航（`/snav`）

1. **開始導航** — PdrTracker 註冊 7 種感測器，持續記錄步數/航向
2. **邊走邊拍** — 每次拍照打包 PDR 快照（steps/distance/heading/x/y）+ 原始感測器資料
3. **上傳融合** — 後端結合影像推理與 PDR 度量建立節點與邊
4. **步數指引** — 回傳結合最短路徑的步數/距離預估
5. **儲存地圖** — 可保存為可重用的場所地圖（含 PDR 度量）

---

## 從零開始安裝

### 前置需求

| 項目 | 版本 |
|------|------|
| Android Studio | Ladybug 以上（支援 AGP 9.0.0） |
| JDK | 17+ |
| Python | 3.10 ~ 3.11 |
| C++ Build Tools | Visual Studio Build Tools（安裝「使用 C++ 的桌面開發」，本機跑 GroundingDINO 才需要） |

### Step 1：Clone 專案

```bash
git clone <repo-url>
cd Shopping-Navigation
```

### Step 2：設定環境變數

在專案根目錄建立 `.env`（可參考 `.env.example`）：

```env
# Android App 連線用（改成你電腦的 IP，或雲端 URL）
BACKEND_URL=http://192.168.x.x:8000/

# VLM 後端選擇：openai / gemini / ollama
VLM_BACKEND=openai

# 長庚大學 LLM Gateway（一把金鑰通吃 chat/vision/OCR/embedding）
CGU_API_KEY=your-cgu-gateway-key

# OpenAI / 相容 API（CGU_API_KEY 為空時使用）
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# Gemini（選用）
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

# 感知模組開關（1=開，0=關；雲端部署建議都設 0）
PERCEPTION_ENABLED=1
OCR_ENABLED=1

# Neo4j Aura（選用，多人共用地圖時）
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

### Step 3：安裝 Python Backend

```powershell
cd backend

# 建立虛擬環境
python -m venv venv
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"

# 安裝基礎套件
pip install -r requirements-server.txt

# 安裝 CPU 版 PyTorch（覆蓋 CUDA 版，除非你有 ≥4GB VRAM 的 GPU）
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cpu

# 安裝 transformers（必須 <5，GroundingDINO 不相容 v5）
pip install "transformers>=4.40,<5"

# 安裝 GroundingDINO
pip install wheel setuptools
pip install groundingdino-py --no-build-isolation
```

> 若只走 VLM 路線（不用本機 GroundingDINO/EasyOCR），可設 `PERCEPTION_ENABLED=0`、
> `OCR_ENABLED=0`，並改裝輕量的 `requirements-cloud.txt`，就不需要 PyTorch 與模型權重。

### Step 4：下載模型權重（本機完整模式才需要）

下載 `groundingdino_swint_ogc.pth`（約 694MB）放到 `backend/models/`：

```
backend/
  models/
    groundingdino_swint_ogc.pth
```

下載連結：https://github.com/IDEA-Research/GroundingDINO/releases

### Step 5：啟動 Backend

```powershell
cd backend
.\start.ps1
```

或手動：

```powershell
cd backend
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m uvicorn server.server:app --host 0.0.0.0 --port 8000
```

啟動成功後，瀏覽器開 `http://localhost:8000/docs` 可看到 Swagger UI 上所有端點。

### Step 6：安裝 Android App

1. 用 Android Studio 開啟專案根目錄
2. 設定後端連線：
   - **模擬器**：預設 `http://10.0.2.2:8000/`，不用改
   - **實體手機 USB**：`adb reverse tcp:8000 tcp:8000`
   - **同網路 WiFi**：`.env` 設 `BACKEND_URL=http://你的IP:8000/`
   - App 內也可在設定頁動態切換後端 URL（`BackendConfig`）
3. 連接手機或模擬器，點 Run

### 連線測試

手機瀏覽器開啟 `http://<你的IP>:8000/health`，應回傳 `{"status":"ok"}`。

---

## 使用方式

### VLM 拍照導航

1. 開啟 App → 進入「導航」頁面
2. 輸入目標（如「找冰箱」）→ 按「開始導航」
3. 拍照或從相簿選取 → 等待分析
4. 依指引移動 → 再拍照 → 重複直到 ARRIVED

### Sensor-Nav / PDR 導航

1. 設定頁 → 「感測器導航」
2. 輸入目的地 → 開始導航（PdrTracker 持續記錄步數/航向）
3. 邊走邊拍，App 上傳照片 + PDR 資料，顯示步數指引與小地圖軌跡
4. 確認到達後可儲存地圖
5. 也可切換「離線採集模式」：不需後端，資料全存手機，之後匯出 ZIP

### 快速模式 vs 完整模式

在 `.env` 中切換：

| 模式 | 設定 | 速度 | 精度 |
|------|------|------|------|
| 快速（VLM only） | `PERCEPTION_ENABLED=0`, `OCR_ENABLED=0` | ~5 秒 | 中 |
| 完整（全模組） | `PERCEPTION_ENABLED=1`, `OCR_ENABLED=1` | ~60 秒 | 高 |

---

## API 端點

### 導航 Session（VLM 拍照導航）

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/session` | 建立導航 session，回傳 session_id 和 goal_objects |
| `POST` | `/session/{id}/photo` | 上傳照片，回傳導航指引（TurnResponse） |
| `POST` | `/session/{id}/answer` | 回答 VLM 提問 |
| `POST` | `/session/{id}/confirm` | 確認 / 否認抵達 |
| `POST` | `/session/{id}/confirm_location` | 確認定位候選節點 |
| `GET` | `/session/{id}` | 查詢 session 狀態和歷史 |
| `GET` | `/session/{id}/map` | 取得拓樸地圖（`?format=png` 取 PNG） |
| `GET` | `/session/{id}/photo/{n}.jpg` | 取得第 n 張標註照片 |
| `GET` | `/session/{id}/raw_photo/{node_id}` | 取得原始照片 |
| `GET` | `/session/{id}/topo_photo/{photo_id}` | 取得拓樸地圖照片 |

### Sensor-Nav（`/snav` router）

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/snav/session` | 建立感測器導航 session |
| `POST` | `/snav/{id}/photo` | 上傳照片 + PDR 資料（multipart） |
| `POST` | `/snav/{id}/answer` / `/confirm` | 回答提問 / 確認抵達 |
| `GET`/`POST`/`DELETE` | `/snav/maps...` | 儲存、列出、取得、刪除可重用地圖 |

> `/snav` 端點的完整資料格式見 [`docs/sensor-nav-README.md`](docs/sensor-nav-README.md)。

### 工具與診斷

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/ocr` | 單張照片 OCR |
| `GET` | `/sessions/list` | 列出所有 session |
| `GET` | `/sessions/{id}/debug` | session 除錯資訊 |
| `GET` | `/health` | 健康檢查 |

---

## 雲端部署（Render）

`backend/` 已備妥部署檔：

```
backend/
├── Dockerfile              # Python 3.11 slim，只裝核心依賴
├── .dockerignore
├── requirements-cloud.txt  # 無 torch/easyocr（image ~200MB）
└── render.yaml             # Render 一鍵部署設定
```

步驟：

1. 推到 GitHub
2. render.com → New Web Service → 連結 repo
3. Root Directory 設 `backend`
4. Environment 設 `CGU_API_KEY`（或 `OPENAI_API_KEY`），需要多人共用地圖時再設 `NEO4J_*`
5. Deploy → 取得 URL
6. `.env` 或 App 設定頁把 `BACKEND_URL` 指向 `https://你的app.onrender.com/`

雲端版停用 GroundingDINO + EasyOCR（`PERCEPTION_ENABLED=0`, `OCR_ENABLED=0`），
完全走 VLM 路線，免費方案即可運行。

---

## 輸出檔案

每個 session 的輸出保存在 `backend/output/sessions/{session_id}/`：

```
output/sessions/{session_id}/
  ├── photo/        原始照片（0.jpg, 1.jpg, ...）
  ├── annotated/    標註照片（綠框=物件偵測，青框=OCR文字，含信心值）
  ├── sensors/      Sensor-Nav 的原始感測器 JSON（含 PDR 摘要）
  ├── graph/        目標圖 / 場景圖 PNG
  └── map/          拓樸地圖（map_0.json / map_0.png / ...）

output/maps/
  └── {map_id}.json  可重用的持久化地圖（含 PDR 度量）
```

---

## 專案檔案說明

### Backend（`backend/server/`）

| 檔案 | 功能 |
|------|------|
| `server.py` | FastAPI 主程式。定義所有 API 端點，串接照片上傳→偵測→OCR→VLM→標註→地圖的完整流程 |
| `sensor_nav.py` | `/snav` router。PDR + AI 視覺融合導航：EXIF 修正、偵測、TopoMap 建圖、VLM 決策、步數指引、原始感測器儲存 |
| `config.py` | 全域設定。讀取 `.env`，定義模型路徑、偵測閾值、VLM 後端、OCR 語言、CGU/Neo4j 參數、label 正規化 |
| `vlm.py` | VLM 呼叫層。支援 OpenAI / Gemini / Ollama / CGU gateway，負責圖片壓縮、prompt 組裝、API 呼叫 |
| `llm_gateway.py` | 長庚大學 OpenAI 相容 gateway 共用 client：一把 `CGU_API_KEY` 路由 chat / vision / OCR / embedding |
| `perception.py` | GroundingDINO 物件偵測。依 goal_objects 關鍵字偵測物件，回傳 bounding box + 信心值 |
| `ocr.py` | EasyOCR 文字辨識。支援英文和繁體中文 |
| `annotator.py` | 標註圖產生器。繪製物件偵測框（綠）與 OCR 文字框（青），標示 label 和信心值 |
| `graph_renderer.py` | 目標圖 / 累積場景圖 PNG 繪製，含偵測後處理（NMS、相鄰框合併、跨步實體追蹤、空間關係推論） |
| `prompts.py` | Prompt 模板。goal 分解、每步導航、比例式 bbox 感知 prompt |
| `goal_decomposer.py` | 目標分解。呼叫 VLM 將使用者目標轉換為偵測關鍵字列表 |
| `scene.py` | 場景格式化。將偵測與 OCR 結果轉為 VLM 可讀文字（含空間位置：左/中/右、近/遠） |
| `topomap.py` | 拓樸地圖（v1）。NetworkX 動態圖：節點=拍照位置，邊=移動動作，供單次導航 |
| `topomap_v2.py` | 拓樸地圖 v2。場所等級地圖：照片節點 + 物件節點子圖、多方向、可跨導航重用 |
| `locate_v2.py` | 照片定位。給一張照片的偵測結果，推理最可能對應到地圖中哪個節點（embedding 相似度 + IDF 權重 + 網格比對） |
| `route_planner.py` | 路徑規劃。TopoGraphV2 上的 A* 最短路徑與多目標 TSP 排序 |
| `map_store.py` | 地圖 JSON 持久化：save / list / load / delete |
| `neo4j_map_store.py` | 把 TopoGraphV2 存入 / 讀出 Neo4j Aura，供多人共用 |
| `models.py` | 資料模型。API 的 request/response schema（Pydantic）與 VLM 回應格式 |
| `session.py` | Session 管理。保存每個導航工作的狀態、歷史和地圖 |
| `store_knowledge.py` / `store_map.py` / `navigator.py` | 靜態環境知識、預建地圖與靜態導航引擎（早期系辦 demo 用） |
| `batch_mapper.py` | 批次建圖工具。對資料夾照片批次執行 GroundingDINO 偵測 |
| `run_server.py` / `start.ps1` | 啟動腳本（設定 logging / 啟用 venv 並啟動 uvicorn） |

### Android App（`app/src/main/java/com/example/shopping/`）

| 檔案 | 功能 |
|------|------|
| `MainActivity.kt` | App 進入點與 composable 路由（含 sensor_nav） |
| `network/NavigationApi.kt` | Retrofit HTTP client（VLM 導航端點，OkHttp 30s connect / 300s read timeout） |
| `network/SensorNavApi.kt` | Retrofit service → `/snav` 端點 |
| `network/BackendConfig.kt` | 後端 URL 動態設定（可在 App 內切換） |
| `sensor/PdrTracker.kt` | PDR 引擎：7 感測器註冊、Weinberg 步長估計、原始緩衝、snapshot |
| `sensor/LocalSessionStore.kt` | 離線模式本地儲存：session/photo/sensor JSON、ZIP 匯出 |
| `viewmodel/NavigationViewModel.kt` | VLM 導航狀態管理 |
| `viewmodel/SensorNavViewModel.kt` | 整合 PDR + 上傳 + 離線模式 + 地圖管理 |
| `ui/screens/NavigationScreen.kt` | VLM 導航主畫面（CameraX、相簿、指引、標註圖） |
| `ui/screens/SensorNavScreen.kt` | Sensor-Nav 畫面：目標輸入 + 離線切換 + 導航（PDR 統計 + 小地圖軌跡） |
| `ui/screens/SensorLabScreen.kt` | 感測器實驗室：8 種感測器即時錄製、CSV 匯出、歷史瀏覽與分享 |
| `ui/screens/SettingsScreen.kt` | 設定頁面（含感測器導航入口、後端 URL 設定） |
| `ui/screens/HomeScreen.kt` / `ShoppingListScreen.kt` / `AIScreen.kt` / `LoginScreen.kt` / `HistoryScreen.kt` / `IngredientsScreen.kt` / `MainContainer.kt` / `NearbyStoresSheet.kt` / `NearbyStoresAr.kt` | 其他頁面（首頁、購物清單、AI 助手、登入、歷史、食材推薦、主容器、附近商店） |
| `ui/components/*.kt` | UI 動畫與互動元件 |
| `ui/utils/*.kt` | 商品分類、食物圖示等工具 |
| `model/*.kt` / `ui/theme/*.kt` | 資料模型與 Material3 主題 |

### 設定檔

| 檔案 | 功能 |
|------|------|
| `.env` / `.env.example` | 環境變數（VLM 後端、API Key、模組開關、Neo4j） |
| `gradle/libs.versions.toml` | Android 依賴版本管理 |
| `app/build.gradle.kts` | Android app 建置設定 |
| `app/src/main/AndroidManifest.xml` | Android 權限宣告（相機、網路、位置、ACTIVITY_RECOGNITION） |
| `app/src/main/res/xml/network_security_config.xml` | 允許 HTTP 明文連線（開發用） |
| `app/src/main/res/xml/file_paths.xml` | FileProvider path（含離線採集 snav_local） |
| `backend/requirements-server.txt` | 本機完整版 Python 套件清單 |
| `backend/requirements-cloud.txt` | 雲端輕量版（無 torch/easyocr） |
| `backend/Dockerfile` / `render.yaml` | 雲端部署設定 |

---

## 技術細節

### 偵測參數

| 參數 | 值 | 說明 |
|------|------|------|
| `GROUNDINGDINO_BOX_THRESHOLD` | 0.30 | 主要偵測信心門檻 |
| `GROUNDINGDINO_BOX_THRESHOLD_FALLBACK` | 0.20 | 未偵測到時降低重試 |
| `GROUNDINGDINO_TEXT_THRESHOLD` | 0.25 | 文字匹配門檻 |
| `OCR_MIN_CONFIDENCE` | 0.3 | OCR 最低信心值 |
| `OCR_LANGUAGES` | en, ch_tra | OCR 語言（英文 + 繁體中文） |

### PDR 演算法（Sensor-Nav）

- **步長估計**：Weinberg 方法 `stride = K × (a_max − a_min)^0.25`（K≈0.4667）
- **航向**：Game Rotation Vector（不含磁力計，室內不受磁場干擾）
- **位置更新**：`x += steps × stride × sin(yaw)`、`y += steps × stride × cos(yaw)`

> 詳見 [`docs/sensor-nav-README.md`](docs/sensor-nav-README.md)。

### 效能參考（CPU 模式，Intel i5 + MX550）

| 模組 | 耗時 |
|------|------|
| GroundingDINO | ~30-40 秒 |
| EasyOCR | ~20-30 秒 |
| VLM (GPT-4o) | ~3-5 秒 |
| 總計（完整模式） | ~55-80 秒 |
| 總計（快速模式 / 雲端 VLM only） | ~3-5 秒 |

### GPU 支援

系統會自動偵測 GPU VRAM：
- ≥ 3GB VRAM：自動使用 GPU（速度提升 5-10 倍）
- < 3GB VRAM：自動降回 CPU 模式

---

## 疑難排解

| 問題 | 解決方式 |
|------|----------|
| 手機連不到 server | 確認同一 Wi-Fi、IP 正確、防火牆允許 port 8000；或用 `adb reverse tcp:8000 tcp:8000` |
| `Perception unavailable` | 確認 `models/groundingdino_swint_ogc.pth` 存在，或設 `PERCEPTION_ENABLED=0` 走 VLM |
| `transformers` BertModel 錯誤 | 安裝 `transformers<5`：`pip install "transformers>=4.40,<5"` |
| 照片上傳 timeout | 改用快速模式（`PERCEPTION_ENABLED=0`）或增加手機端 timeout |
| GroundingDINO 安裝失敗 | 安裝 Visual Studio C++ Build Tools，設定 `$env:PYTHONUTF8="1"` |
| AGP 版本不相容 | `gradle/libs.versions.toml` 中 `agp` 設為 `9.0.0` |
| PDR 步數沒偵測到 | 確認 `ACTIVITY_RECOGNITION` 權限已授予；模擬器通常無硬體計步器 |
| CGU gateway 回 invalid key | `llm_gateway` 需要 base URL override，確認 `CGU_API_KEY` 與 gateway endpoint 正確 |

---

## 相關文件

- [`docs/sensor-nav-README.md`](docs/sensor-nav-README.md) — Sensor-Nav / PDR 完整說明
- [`backend/README_topomap_v2.md`](backend/README_topomap_v2.md) — 拓樸地圖 v2 資料結構
- [`backend/README_locate_v2.md`](backend/README_locate_v2.md) — 照片定位邏輯
- `http://localhost:8000/docs` — Swagger UI（所有 API 端點）
