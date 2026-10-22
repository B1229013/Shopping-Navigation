# Shopping Navigation — 室內購物導航系統（iOS 版）

拍照就能在賣場裡導航的 iOS App（SwiftUI），搭配一個 Python FastAPI 後端。使用者輸入要找的商品後拍照，
後端用 **VLM 看懂照片 → 在店家的拓樸地圖上定位 → 規劃路線**，回傳逐步指引；手機在兩張照片之間
用 **PDR（計步＋航向）** 持續更新位置。App 另外有購物清單、預算、食材紀錄與 AI 助手等購物功能。

> **這是 `ios` 分支**：iOS App ＋ 和 Android 版（`android` 分支）**同一份**後端 `backend/`。
> 兩個分支只差在前端；後端請在 `android` 改，再把 `android` 合併進 `ios`（見下方「分支怎麼同步」）。

---

## 倉庫與分支

| 倉庫／分支 | 內容 |
|------|------|
| B1229013/APP-IOS-Android `android` | Android App ＋ 共用後端 `backend/`（內容來自 B1229023/APPNAV `main`） |
| **B1229013/APP-IOS-Android `ios`（本分支）** | iOS App ＋ 同一份後端。2026-10-21 由 B1222017/APPNAV_ios 的 `feature/localization-fusion-v2` 合併進來，iOS 的 commit 紀錄都保留 |
| B1229013/APP-IOS-Android `main` | 2026-09-23 的舊版（`android/`、`ios/`、`backend/` 三個資料夾），沒有動 |
| B1229023/APPNAV | Android ＋ 共用後端的整合倉庫（`android` 分支的來源；它的 `feat/object-relations`（物件關係）已合併進 `android` 與 `ios`，設定 `LOC_ADJACENT`、`LOC_RELATION_WEIGHT` 預設關閉） |
| B1222017/APPNAV_ios | 原本的 iOS 倉庫 |
| B1229013/Shopping-Navigation | Android，停在 2026-10-07 |

- 舊版分支在 B1229023/APPNAV 以 tag 保存：`archive/ranging-metric-distance`、`archive/sensorlab-pdr`、
  `archive/sensor-nav-pdr`、`archive/remove-dead-modules`；整合前的狀態是 `backup/before-consolidate`。

### 分支怎麼同步

後端只有一份：在 `android`（或從 `android` 開的分支）修改後端，再合併進 `ios`：

```bash
git switch ios
git merge android
# android 上 Android 檔案（app/、Gradle…）有改動時，會出現「deleted in ios, modified in android」衝突：
git rm -r -q app build.gradle.kts gradle gradle.properties gradlew gradlew.bat settings.gradle.kts google-services.json .idea 2>/dev/null
git commit
```

iOS 前端（`APPNAV_ios/`、`APPNAV_ios.xcodeproj/`）只在 `ios` 分支修改。

---

## 系統功能

**導航（主功能，「導航」頁）**

- **選店與起點** — 從 Neo4j 讀出可用的店（`GET /places`）；還沒拍照前可以按「📍 我在入口，從這裡開始」。
- **拍照定位** — VLM 讀出照片裡的物件、商品名與標示文字，再與店家地圖比對，融合
  **DINOv2 影像相似度 ＋ 物件名稱（中英＋語意向量）＋ OCR 文字 ＋ 手機感測器位置**，算出使用者在哪個節點、面向哪裡。
  DINOv2 由後端算（`server/dino_embed.py`、`locate_fusion.py`），和 Android 走同一條路。
  iOS 手機上的 Core ML DINOv2（`DINOLocalizationService`）還在，但預設關閉：Build Setting `ON_DEVICE_DINO=YES`
  才會在手機上算並上傳前幾名，而且只在「A7家樂福（9/16)」有效（手機裡的參考向量只有這間，`NodeCoordinateStore.place`）。
  lab1019 376 張實測（單張、不帶感測器）：後端路線 Top-1 44.7%，手機 DINOv2 路線 37.5%。
- **用照片校正感測器** — 定位成功的照片會把位置拉回節點，兩張照片之間走直線超過 8 m 時順便校正方向偏差與步長
  （`MapPositionEstimator`）。
- **路線規劃** — 在 Neo4j 的走道拓樸圖上規劃路線，小地圖與指令走同一張圖。
- **逐步指引** — 每張照片回傳 MOVE / ASK / ARRIVED；問「到了嗎？」時用「有／沒有」按鈕回答（和 Android 相同），其他問題用文字回答。
- **PDR 即時位置** — 兩張照片之間用計步與航向更新位置、沿路線前進；走超過 12 m 會提醒拍照校正。
- **照片品質** — 手機先擋模糊、太暗、手機沒拿直；後端再擋被人或推車擋住；信心不足時請使用者原地轉身再拍一張。

**建圖與研究工具**

- **Sensor Lab** — 在 Android App（`android` 分支）裡：全螢幕拍照＋感測器錄製與 PDR 紀錄，用來建新店的地圖。
- **拓樸地圖建置器**（`/topomap-builder`）— 把 Sensor Lab 的走路紀錄排成地圖，上傳到 Neo4j。
- **PDR Waypoint 編輯器**（根目錄 `pdr-waypoint-editor.html`）— 在平面圖上點出航點、Grid Snap、匯出座標 JSON。
- **定位評估**（`backend/eval/`）— 單張影像定位基準、權重掃描、逐張報告、VLM 重播（不呼叫 API）。
- **即時監看**（`/dashboard`）與 **定位診斷頁**（`/debug`、`/session/{id}/debug-page`）。

**購物功能**

- 登入（Firebase Auth）、購物清單、預算（收據 OCR）、歷史紀錄、食材紀錄、AI 助手（Groq）。

---

## 系統架構

```
┌────────────────────────────┐   HTTP/JSON   ┌──────────────────────────────────────────────┐
│ iOS App（SwiftUI）          │ ◄───────────► │ FastAPI 後端（backend/server）               │
│                            │   URLSession  │                                              │
│ NavigationCameraView 拍照 ──┼── JPEG ─────► │ VLM 感知（OpenAI 相容 API，最多 3 輪）         │
│ PhotoQualityChecker 拍前檢查 ┼── 位置、航向 ─► │ 定位融合：DINOv2 ＋ 物件 ＋ OCR ＋ 感測器       │
│ MapPositionEstimator（PDR） │ ◄─ 指引／路線 ─┤ 路線規劃（Neo4j 拓樸圖）                      │
│ PathFollower ＋ 小地圖       │               │ VLM 決策 → MOVE / ASK / ARRIVED              │
└────────────────────────────┘               └───────────────┬──────────────────────────────┘
                                                             │
                          ┌──────────────────────────────────┼─────────────────────┐
                          │ Neo4j Aura：店家地圖（節點、走道、方向照片、物件、DINOv2 向量）│
                          │ backend/maps：起點、額外走道邊、編輯器地圖                    │
                          └──────────────────────────────────────────────────────────┘

App 的 AI 助手（Groq）與收據 OCR（PaddleOCR）直接連各自的服務，不經過導航後端。
```

## 導航流程（`/session`）

1. **建立 session** — `POST /session`，帶目標（如「找優格」）和店名；後端把目標拆成要找的物件。
2. **選起點（可選）** — `POST /session/{id}/start`，從起點出發並重新規劃路線。
3. **拍照** — `POST /session/{id}/photo`，帶照片和手機的位置／航向：
   1. 照片品質檢查（模糊、太暗、遮擋）
   2. VLM 感知：物件、商品名、OCR 文字、店內區域（`VLM_ROUNDS`、`VLM_PASS3`；請在 `.env` 設 `VLM_PASS3=0`，原因見 [backend/README.md](backend/README.md)）
   3. 定位融合：伺服器端 DINOv2 找最像的參考照片，再加物件名稱、OCR、走道吊牌與感測器位置
   4. 依定位結果在 Neo4j 拓樸圖上規劃到目標的路線
   5. VLM 決策：根據照片與路線回傳 MOVE / ASK / ARRIVED 與中文指引
4. **回答與確認** — `POST /session/{id}/answer`（回答 ASK）、`POST /session/{id}/confirm`（確認抵達）。
5. **路線** — `GET /session/{id}/path` 給 App 畫小地圖與跟隨路線。

每次導航的定位紀錄存在 `backend/output/sessions/<session_id>/`（`session_log.py`）。

## 新店建圖流程

1. 用 Android App（`android` 分支）的 **Sensor Lab** 在店裡走一圈拍照（同時錄 PDR）。
2. `python build_map_v2_topomap.py` — 把走路紀錄整理成建圖工作區。
3. 開 `http://<後端>/topomap-builder` — 排節點、接走道、疊平面圖、確認。
4. `python upload_map_v2.py` — 上傳成 Neo4j 的 V2 地圖格式（`server/map_v2.py`）。
5. `python rebuild_objects_v2.py` — 對每張方向照片偵測物件並寫回 Neo4j。
6. `python store_dino_v2.py` — 算每張照片的 DINOv2 向量，存進 `dino_sidecar.pkl` 和 Neo4j。
7. 在建置器設定起點（位置、面向、照片），存在 `backend/maps/starts/<店名>/`。

---

## 安裝與啟動

### 前置需求

| 項目 | 版本 |
|------|------|
| Xcode | 最新穩定版；相機與感測器請用 iPhone 實機測試 |
| Python | 3.11 |
| Neo4j Aura | 店家地圖存在這裡（導航必需） |

### 1. 設定 `.env`（專案根目錄，可參考 `.env.example`）

```env
# ── 導航後端 ──
OPENAI_API_KEY=your-key            # VLM、物件名稱語意向量
OPENAI_BASE_URL=https://api.openai.com/v1   # 可改成任何 OpenAI 相容閘道
OPENAI_MODEL=gpt-4o
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
PERCEPTION_ENABLED=0               # 1 = 本機 GroundingDINO（需模型權重）
OCR_ENABLED=0                      # 1 = 本機 EasyOCR
```

### 2. 啟動後端

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-server.txt
pip install numpy pillow torch transformers   # 伺服器端 DINOv2 定位（沒裝會退回只用物件＋OCR）
.\start.ps1
```

`start.ps1` 會啟用 venv 並以 `uvicorn server.server:app --port 8000` 啟動。
瀏覽器開 `http://localhost:8000/docs` 可看到所有端點，`/health` 回傳 `{"status":"ok"}` 代表正常。

### 3. 安裝 App（iOS）

1. 用 Xcode 開啟 `APPNAV_ios.xcodeproj`，選 iPhone 後 `Cmd+R`。第一次開會自動抓 Swift Package（Firebase）。
2. **後端網址與金鑰在 Xcode Build Settings**（不在 `.env`，見 `APPNAV_ios/Utils/AppConfig.swift`）：
   target `APPNAV_ios` → Build Settings → 搜尋變數名稱。

   | 變數 | 用途 |
   |------|------|
   | `NAVIGATION_BACKEND_URL` | 導航後端網址；專案裡目前是 `http://172.20.10.3:8000`，請改成你跑後端那台電腦的 IP |
   | `GROQ_API_KEY` | AI 助手 |
   | `PADDLEOCR_API_URL` / `PADDLEOCR_ACCESS_TOKEN` | 收據 OCR |
   | `MAPS_API_KEY` | 地圖相關功能 |
   | `ON_DEVICE_DINO` | 設 `YES` 才在手機上跑 DINOv2（預設不設＝關閉，由後端算） |

3. **Firebase**：已附 `APPNAV_ios/GoogleService-Info.plist`；要換自己的專案就覆蓋這個檔案。
4. **DINOv2 在後端**：後端要裝 `torch`＋`transformers`（`LOC_SERVER_DINO=1`，預設開），見上方「啟動後端」。
   只有開 `ON_DEVICE_DINO` 時才需要 `APPNAV_ios/Models/DINOv2.mlpackage`（太大，沒有放進 git，要另外放進去）。
   節點座標 `node_coordinates.json`（A7，箭頭與「我在入口」用）和手機版參考向量 `dino_ref_embeddings.bin`、`dino_ref_meta.json` 在 `APPNAV_ios/Models/`。

### Docker（選用，尚未在目前版本驗證）

`backend/Dockerfile` 只裝 `requirements-cloud.txt`（不含 torch／easyocr），預設關閉本機偵測與 OCR。

```bash
cd backend
docker build -t shopping-nav-backend .
```

執行時以環境變數提供 `OPENAI_API_KEY`、`NEO4J_*`。

---

## API 端點

| 分類 | 端點 |
|------|------|
| 店家與起點 | `GET /places`、`GET /starts`、`GET /starts/photo`、`POST /starts` |
| 導航 session | `POST /session`、`POST /session/{id}/photo`、`POST /session/{id}/answer`、`POST /session/{id}/confirm`、`POST /session/{id}/start` |
| 路線與地圖 | `GET /session/{id}/path`、`GET /session/{id}/route`、`GET /session/{id}/map`、`POST /session/{id}/sensor-map`、`GET /reference-map` |
| 單張定位 | `POST /localize`、`POST /ocr` |
| 照片 | `GET /session/{id}/photo/{node_id}.jpg`、`GET /session/{id}/original/{node_id}.jpg`、`GET /reference-photo/{nid}.jpg` |
| 建圖工具 | `GET /topomap-builder` 與 `/topomap-*` 系列（來源上傳、走路紀錄、平面圖、建議、確認） |
| 診斷 | `GET /health`、`GET /dashboard`、`GET /debug`、`GET /sessions`、`GET /session/{id}`、`GET /session/{id}/debug`、`GET /session/{id}/debug-page`、`POST /test/config` |
| 感測器測試 | `POST /sensor-test`、`GET /sensor-test/{id}/plot.png` |

---

## 專案結構

```
.
├── APPNAV_ios/               iOS App（SwiftUI）
├── APPNAV_ios.xcodeproj/     Xcode 專案
├── scripts/
│   ├── swift_logic_tests/    不需要 Xcode 的 Swift 邏輯測試（MapPositionEstimator、PhotoQualityChecker）
│   └── vlm_test/             周的 VLM 定位測試、物件關係建圖、報告腳本（研究用）
├── backend/
│   ├── server/               FastAPI 後端（正式使用）
│   ├── eval/                 定位與導航評估（資料、結果、圖表）
│   ├── maps/                 起點照片、額外走道邊（neo4j_links）、編輯器地圖
│   ├── legacy/               舊後端（/snav、topomap_v2、locate_v2）與早期定位研究，只供研究腳本 import
│   ├── *.py                  建圖與資料處理腳本（見下表）
│   ├── dino_sidecar.pkl      參考照片的 DINOv2 向量
│   └── Dockerfile、requirements-*.txt、start.ps1
├── docs/                     設計文件書、測試計劃書、期中／期末報告、架構圖
├── 物件導向Project/            使用案例圖、使用案例描述、活動圖
└── pdr-waypoint-editor.html  PDR 航點編輯器
```

### 後端模組（`backend/server/`）

| 檔案 | 功能 |
|------|------|
| `server.py` | FastAPI 主程式：所有端點與拍照導航流程 |
| `config.py` | 讀 `.env`，所有開關與權重（`LOC_*`、`VLM_*`、`EMBEDDING_*`、`NEO4J_*`…） |
| `vlm.py` / `prompts.py` | VLM 呼叫與提示詞：感知（最多 3 輪）、導航決策、提問 |
| `goal_decomposer.py` | 把目標拆成要找的物件 |
| `visual_localization.py` | 照片物件／OCR 與 Neo4j 地圖比對（中英名稱、語意向量、稀有度權重、九宮格） |
| `locate_fusion.py` / `dino_embed.py` | 伺服器端 DINOv2 影像分數，與物件、OCR 融合（權重 1 : 0.1 : 0.3） |
| `loc_fusion.py` | 把影像證據與手機感測器位置合成機率排名 |
| `place_recognizer.py` | 判斷使用者在哪一間已知的店（目前沒有被其他模組使用） |
| `aisle_signs.py` | 從走道吊牌號碼縮小可能的位置 |
| `photo_quality.py` | 模糊、太暗、遮擋檢查 |
| `heading.py` | 估計使用者朝向，產生左轉／右轉指令 |
| `editor_map.py` / `path_planner.py` | 可走路線圖、A* 與多目標排序、轉彎清單、小地圖資料 |
| `neo4j_client.py` | Neo4j Aura HTTP 客戶端（讀地圖、照片、DINOv2 向量） |
| `map_v2.py` / `map_build.py` | 建新店地圖：Sensor Lab 資料 → 建圖工作區 → Neo4j V2 格式 |
| `start_points.py` | 每間店的起點（位置、面向、照片） |
| `session.py` / `session_log.py` | session 狀態（記憶體）與每次導航的紀錄（磁碟） |
| `scene.py` / `annotator.py` / `graph_renderer.py` | 給 VLM 的場景文字、標註圖、除錯用場景圖 |
| `perception.py` / `ocr.py` | 本機 GroundingDINO 與 EasyOCR（`PERCEPTION_ENABLED`、`OCR_ENABLED` 開啟時） |
| `topomap_builder.html` / `dashboard.html` / `debug_page.html` | 建圖工具、即時監看、定位診斷頁 |
| `tests/` | pytest（39 個測試檔） |

### 後端腳本（`backend/`）

| 檔案 | 功能 |
|------|------|
| `build_map_v2_topomap.py` | Sensor Lab 走路紀錄 → 建圖工作區 |
| `upload_map_v2.py` | 建置器確認後的地圖上傳到 Neo4j |
| `rebuild_objects_v2.py` | 重新偵測每張方向照片的物件並寫回 Neo4j |
| `store_dino_v2.py` | 計算並儲存每張方向照片的 DINOv2 向量 |
| `build_store_graph.py`、`build_store_map.py`、`build_html_map.py`、`generate_topomap.py`、`render_clean.py`、`ocr_dump.py` | 早期的拓樸圖產生與繪製工具 |

### iOS App（`APPNAV_ios/`）

| 檔案 | 功能 |
|------|------|
| `App/AppEntry.swift` | 進入點 |
| `Views/Navigation/NavigationEntryView.swift` | 導航入口：確認要找的商品、建立導航 session |
| `Views/Navigation/NavigationCameraView.swift` | 導航畫面：相機、拍照上傳、指引、提示橫幅、拍照說明、校正位置 |
| `Views/Navigation/RouteMapView.swift` | 小地圖與路線跟隨（`RouteFollowerModel`） |
| `Views/Navigation/DirectionArrowView.swift` | 方向箭頭 |
| `Navigation/NavigationSessionManager.swift` | 感測器 session：PDR、地圖位置、距離上次校正幾公尺 |
| `Navigation/MapPositionEstimator.swift` | 地圖上的位置估計；照片定位後校正位置、方向偏差、步長 |
| `Navigation/MotionPathTracker.swift` | 計步與航向（CoreMotion） |
| `Navigation/PathFollower.swift` | 沿路線前進與轉彎提示 |
| `Navigation/PhotoQualityChecker.swift` | 拍照前檢查模糊、太暗、手機沒拿直 |
| `Services/NavigationAPI.swift` | 導航後端 API（session、拍照、回答、確認、路線） |
| `Services/DINOLocalizationService.swift` | 手機上的 DINOv2（Core ML），預設關閉（`ON_DEVICE_DINO`） |
| `Services/NodeCoordinateStore.swift` | 節點座標（`Models/node_coordinates.json`） |
| `Views/Main/*.swift`、`Services/GroqService.swift`、`Services/OCRService.swift`、`Services/AuthService.swift` | 購物相關頁面、AI 助手、收據 OCR、登入 |
| `Views/Navigation/SensorTestView.swift`、`SensorComboTestView.swift`、`GPSLoggerView.swift`、`TopoMapTestView.swift` | 感測器與地圖測試頁 |

---

## 測試

```powershell
# 後端
cd backend
.\venv\Scripts\python.exe -m pytest -q

# iOS 定位邏輯（macOS，只要 Swift 命令列工具，不用 Xcode）
scripts/swift_logic_tests/run.sh

# iOS App 建置／測試（需要 Xcode）
xcodebuild -project APPNAV_ios.xcodeproj -scheme APPNAV_ios -destination 'generic/platform=iOS Simulator' build
```

目前後端 `test_vlm.py` 有 2 個檢查 perception 提示詞文字的測試未通過（整合前就存在）。

---

## 疑難排解

| 問題 | 解決方式 |
|------|----------|
| 手機連不到後端 | 確認同一 Wi-Fi、`NAVIGATION_BACKEND_URL` 的 IP 正確、防火牆允許 8000 |
| 選店清單是空的 | 確認 `.env` 的 `NEO4J_URI`／`NEO4J_PASSWORD`；沒有 Neo4j 時只會列出 `NEO4J_PLACE` |
| 定位只靠物件與文字 | 伺服器端 DINOv2 需要 `torch`＋`transformers` 與 `dino_sidecar.pkl`；`LOC_SERVER_DINO=0` 可關閉 |
| 不想每張照片多一次 embedding 請求 | 設 `EMBEDDING_ENABLED=0`，物件名稱改回只用字元相似度 |

---

## 相關文件

- [`backend/eval/README.md`](backend/eval/README.md) — 評估工具與指標
- [`backend/legacy/README_locate_v2.md`](backend/legacy/README_locate_v2.md)、[`backend/legacy/README_topomap_v2.md`](backend/legacy/README_topomap_v2.md) — 舊版定位與拓樸地圖（研究用）
- `docs/` — 設計文件書、測試計劃書、期中／期末報告
- `http://localhost:8000/docs` — Swagger UI
