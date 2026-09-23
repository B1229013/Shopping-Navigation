# Shopping Navigation — 室內導航系統

基於 VLM（視覺語言模型）的即時室內導航 App。使用者拍照上傳，系統透過物件偵測（GroundingDINO）、文字辨識（EasyOCR）和 VLM（GPT-4o / Gemini）分析環境，提供逐步導航指引。

## 系統架構

```
┌──────────────┐    HTTP/JSON     ┌────────────────────────────┐
│  Android App │ ◄──────────────► │  Python FastAPI Backend     │
│  (Kotlin)    │   Retrofit       │                            │
│              │                  │  ┌─ GroundingDINO (物件偵測)│
│  Camera ─────┼─ JPEG ─────────► │  ├─ EasyOCR (文字辨識)     │
│  Gallery ────┤                  │  ├─ VLM (GPT-4o/Gemini)    │
│              │                  │  ├─ Annotator (標註圖)      │
│  UI ◄────────┤◄── TurnResponse │  └─ TopoMap (拓撲地圖)      │
└──────────────┘                  └────────────────────────────┘
```

## 導航流程

1. **建立 Session** — 使用者輸入目標（如「找冰箱」），VLM 分解出偵測關鍵字
2. **拍照上傳** — GroundingDINO 偵測物件、EasyOCR 讀取文字
3. **VLM 決策** — 綜合偵測結果 + 照片 + 歷史路徑，回傳 MOVE / ARRIVED / ASK
4. **標註圖** — 在照片上標示所有偵測框和 OCR 文字（含信心值）
5. **拓撲地圖** — 每一步自動建立並保存地圖（JSON + PNG）

---

## 從零開始安裝

### 前置需求

| 項目 | 版本 |
|------|------|
| Android Studio | Ladybug 以上（支援 AGP 9.0.0） |
| JDK | 17+ |
| Python | 3.10 ~ 3.11 |
| C++ Build Tools | Visual Studio Build Tools（安裝「使用 C++ 的桌面開發」） |

### Step 1：Clone 專案

```bash
git clone <repo-url>
cd Shopping-Navigation
```

### Step 2：設定環境變數

在專案根目錄建立 `.env`：

```env
# Android App 連線用（改成你電腦的 IP）
BACKEND_URL=http://192.168.x.x:8000/

# VLM 後端選擇：openai 或 gemini
VLM_BACKEND=openai

# Gemini（選用）
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

# OpenAI / 相容 API（選用）
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 感知模組開關（1=開，0=關）
PERCEPTION_ENABLED=1
OCR_ENABLED=1
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

### Step 4：下載模型權重

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

啟動成功會看到：

```
INFO server.server: VLM backend: openai
INFO server.server: OpenAI model: gpt-4o
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6：安裝 Android App

1. 用 Android Studio 開啟專案根目錄
2. 確認 `.env` 中 `BACKEND_URL` 是你電腦的 IP（手機和電腦要在同一網路）
3. 連接手機或模擬器，點 Run

### 連線測試

手機瀏覽器開啟 `http://<你的IP>:8000/health`，應回傳 `{"status":"ok"}`。

---

## 使用方式

1. 開啟 App → 進入「導航」頁面
2. 輸入目標（如「找冰箱」、「找咖啡機」）→ 按「開始導航」
3. 拍照或從相簿選取照片 → 等待分析（約 50-80 秒，視 CPU 效能）
4. 依照指引移動 → 再拍照 → 重複直到 ARRIVED

### 快速模式 vs 完整模式

在 `.env` 中切換：

| 模式 | 設定 | 速度 | 精度 |
|------|------|------|------|
| 快速（VLM only） | `PERCEPTION_ENABLED=0`, `OCR_ENABLED=0` | ~5 秒 | 中 |
| 完整（全模組） | `PERCEPTION_ENABLED=1`, `OCR_ENABLED=1` | ~60 秒 | 高 |

---

## API 端點

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/session` | 建立導航 session，回傳 session_id 和 goal_objects |
| `POST` | `/session/{id}/photo` | 上傳照片，回傳導航指引 |
| `POST` | `/session/{id}/answer` | 回答 VLM 提問 |
| `GET` | `/session/{id}` | 查詢 session 狀態和歷史 |
| `GET` | `/session/{id}/map` | 取得拓撲地圖（JSON） |
| `GET` | `/session/{id}/map?format=png` | 取得拓撲地圖（PNG） |
| `GET` | `/session/{id}/photo/{n}.jpg` | 取得第 n 張標註照片 |
| `GET` | `/health` | 健康檢查 |

---

## 輸出檔案

每個 session 的輸出保存在 `backend/output/sessions/{session_id}/`：

```
output/sessions/{session_id}/
  ├── photo/        原始照片（0.jpg, 1.jpg, ...）
  ├── annotated/    標註照片（綠框=物件偵測，青框=OCR文字，含信心值）
  └── map/          拓撲地圖
        ├── map_0.json    第 0 步的地圖 JSON
        ├── map_0.png     第 0 步的地圖視覺化
        ├── map_1.json
        └── map_1.png
```

---

## 專案檔案說明

### Backend（`backend/server/`）

| 檔案 | 功能 |
|------|------|
| `server.py` | FastAPI 主程式。定義所有 API 端點，串接各模組，處理照片上傳→偵測→OCR→VLM→標註→地圖的完整流程 |
| `config.py` | 全域設定。讀取 `.env`，定義模型路徑、偵測閾值、VLM 後端選擇、OCR 語言等參數 |
| `vlm.py` | VLM 呼叫層。支援三種後端（Gemini / OpenAI / Ollama），負責圖片壓縮、prompt 組裝、API 呼叫 |
| `perception.py` | GroundingDINO 物件偵測。載入模型權重，根據 goal_objects 關鍵字在照片中偵測物件，回傳 bounding box + 信心值 |
| `ocr.py` | EasyOCR 文字辨識。支援英文和繁體中文，回傳偵測到的文字、信心值和位置 |
| `annotator.py` | 標註圖產生器。在照片上繪製物件偵測框（綠色）和 OCR 文字框（青色），標示 label 和信心值 |
| `prompts.py` | Prompt 模板。包含 goal 分解 prompt 和每一步導航 prompt，指示 VLM 如何回應 |
| `goal_decomposer.py` | 目標分解。呼叫 VLM 將使用者目標（如「找冰箱」）轉換為 GroundingDINO 偵測關鍵字列表 |
| `topomap.py` | 拓撲地圖。用 NetworkX 建立動態圖：節點=拍照位置，邊=移動動作。支援 JSON 匯出和 PNG 視覺化 |
| `scene.py` | 場景格式化。將偵測和 OCR 結果轉換為 VLM 可讀的文字描述（含空間位置：左/中/右、近/遠） |
| `models.py` | 資料模型。定義 API 的 request/response schema（Pydantic）和 VLM 回應格式 |
| `session.py` | Session 管理。記憶體中的 session store，保存每個導航工作的狀態、歷史和地圖 |
| `store_knowledge.py` | 靜態環境知識。CSIE 系辦的 10 個區域定義（房間名稱、地標物件、教授辦公室對照） |
| `store_map.py` | 靜態拓撲地圖。系辦的預建地圖（10 個節點及其連接關係） |
| `navigator.py` | 靜態導航引擎。根據 `store_knowledge.py` 的資料進行物件/地點搜尋和路徑規劃 |
| `batch_mapper.py` | 批次建圖工具。對一個資料夾的照片批次執行 GroundingDINO 偵測，產出偵測結果 JSON |
| `run_server.py` | 啟動腳本。設定 logging 並啟動 uvicorn |
| `start.ps1` | PowerShell 啟動腳本。自動啟用 venv 並啟動 server |

### Android App（`app/src/main/java/com/example/shopping/`）

| 檔案 | 功能 |
|------|------|
| `MainActivity.kt` | App 進入點 |
| `network/NavigationApi.kt` | Retrofit HTTP client。定義 API 介面和 OkHttp 設定（30s connect / 300s read timeout） |
| `ui/screens/NavigationScreen.kt` | 導航主畫面。整合 CameraX 拍照、相簿選取、API 呼叫、導航指引顯示、標註圖顯示 |
| `ui/screens/HomeScreen.kt` | 首頁 |
| `ui/screens/ShoppingListScreen.kt` | 購物清單頁面 |
| `ui/screens/AIScreen.kt` | AI 助手頁面 |
| `ui/screens/SettingsScreen.kt` | 設定頁面 |
| `ui/screens/LoginScreen.kt` | 登入頁面 |
| `ui/screens/MainContainer.kt` | 主容器（底部導航列） |
| `ui/screens/HistoryScreen.kt` | 歷史記錄 |
| `ui/screens/IngredientsScreen.kt` | 食材推薦 |
| `ui/screens/NearbyStoresSheet.kt` | 附近商店 |
| `ui/screens/NearbyStoresAr.kt` | AR 附近商店 |
| `ui/components/CinematicComponents.kt` | UI 動畫元件 |
| `ui/components/CinematicInteractiveComponents.kt` | 互動式 UI 元件 |
| `ui/utils/CategoryClassifier.kt` | 商品分類工具 |
| `ui/utils/FoodIcons.kt` | 食物圖示 |
| `model/*.kt` | 資料模型（購物項目、使用者設定等） |
| `ui/theme/*.kt` | Material3 主題設定 |

### 設定檔

| 檔案 | 功能 |
|------|------|
| `.env` | 環境變數（VLM 後端、API Key、模組開關） |
| `gradle/libs.versions.toml` | Android 依賴版本管理 |
| `app/build.gradle.kts` | Android app 建置設定 |
| `app/src/main/AndroidManifest.xml` | Android 權限宣告（相機、網路、位置） |
| `app/src/main/res/xml/network_security_config.xml` | 允許 HTTP 明文連線（開發用） |
| `backend/requirements-server.txt` | Python 套件清單 |
| `backend/.gitignore` | Git 忽略規則（venv、模型權重、輸出檔案） |

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

### 效能參考（CPU 模式，Intel i5 + MX550）

| 模組 | 耗時 |
|------|------|
| GroundingDINO | ~30-40 秒 |
| EasyOCR | ~20-30 秒 |
| VLM (GPT-4o) | ~3-5 秒 |
| 總計（完整模式） | ~55-80 秒 |
| 總計（快速模式） | ~3-5 秒 |

### GPU 支援

系統會自動偵測 GPU VRAM：
- ≥ 3GB VRAM：自動使用 GPU（速度提升 5-10 倍）
- < 3GB VRAM：自動降回 CPU 模式

---

## 疑難排解

| 問題 | 解決方式 |
|------|----------|
| 手機連不到 server | 確認同一 Wi-Fi、IP 正確、防火牆允許 port 8000 |
| `Perception unavailable` | 確認 `models/groundingdino_swint_ogc.pth` 存在 |
| `transformers` BertModel 錯誤 | 安裝 `transformers<5`：`pip install "transformers>=4.40,<5"` |
| 照片上傳 timeout | 改用快速模式（`PERCEPTION_ENABLED=0`）或增加手機端 timeout |
| GroundingDINO 安裝失敗 | 安裝 Visual Studio C++ Build Tools，設定 `$env:PYTHONUTF8="1"` |
| AGP 版本不相容 | `gradle/libs.versions.toml` 中 `agp` 設為 `9.0.0` |
