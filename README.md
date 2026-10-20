# Shopping Navigation — 室內購物導航系統

拍照就能在賣場裡導航的 Android App，搭配一個 Python FastAPI 後端。使用者輸入要找的商品後拍照，
後端用 **VLM 看懂照片 → 在店家的拓樸地圖上定位 → 規劃路線**，回傳逐步指引；手機在兩張照片之間
用 **PDR（計步＋航向）** 持續更新位置。App 另外有購物清單、預算、食材紀錄與 AI 助手等購物功能。

開箱影片連結：https://drive.google.com/file/d/1nU0xep0huSmS2-fzG8ooJqlZE7_wVK6P/view?usp=sharing

---

## 倉庫與版本

| 倉庫 | 內容 | 狀態 |
|------|------|------|
| **B1229023/APPNAV**（本倉庫，`main`） | Android App ＋ 共用後端 `backend/` | **目前開發主線**（2026-10-20 整合三個倉庫） |
| B1229013/Shopping-Navigation | 同上，停在 2026-10-07 | 之後會同步成最後的版本 |
| B1222017/APPNAV_ios | iOS App（Swift） | iOS 繼續在那裡開發；後端改用本倉庫的 `backend/` |

- iOS 與 Android 共用同一套後端 API（`backend/`）。本倉庫已不再保留 iOS 程式的副本。
- 舊版分支以 tag 保存：`archive/ranging-metric-distance`、`archive/sensorlab-pdr`、
  `archive/sensor-nav-pdr`、`archive/remove-dead-modules`；整合前的狀態是 `backup/before-consolidate`。
- 舊版「感測器導航」（App 的 `SensorNav*`、後端 `/snav`）已由主導航取代並移除，程式保存在 `archive/sensor-nav-pdr`
  與 `backend/legacy/`。

---

## 系統功能

**導航（主功能，「AR 導航」頁）**

- **選店與起點** — 從 Neo4j 讀出可用的店（`GET /places`），可選「我在起點」對準地圖方向開始（`GET /starts`）。
- **拍照定位** — VLM 讀出照片裡的物件、商品名與標示文字，再與店家地圖比對，融合
  **DINOv2 影像相似度 ＋ 物件名稱（中英＋語意向量）＋ OCR 文字 ＋ 手機感測器位置**，算出使用者在哪個節點、面向哪裡。
- **路線規劃** — 在 Neo4j 的走道拓樸圖上規劃路線，小地圖與指令走同一張圖。
- **逐步指引** — 每張照片回傳 MOVE / ASK / ARRIVED；抵達時需地圖同意，並以「是／不是」按鈕確認。
- **PDR 即時位置** — 兩張照片之間用計步與航向更新位置、沿路線前進；照片定位後再校正。
- **照片品質** — 太暗、模糊、被擋住時會請使用者重拍；信心不足時請使用者原地轉身再拍一張。

**建圖與研究工具**

- **Sensor Lab** — 全螢幕拍照（倍率轉盤、點擊對焦）＋ 8 種感測器錄製與 PDR 紀錄，用來建新店的地圖。
- **拓樸地圖建置器**（`/topomap-builder`）— 把 Sensor Lab 的走路紀錄排成地圖，上傳到 Neo4j。
- **PDR Waypoint 編輯器**（根目錄 `pdr-waypoint-editor.html`）— 在平面圖上點出航點、Grid Snap、匯出座標 JSON。
- **定位評估**（`backend/eval/`）— 單張影像定位基準、權重掃描、逐張報告、VLM 重播（不呼叫 API）。
- **即時監看**（`/dashboard`）與 **定位診斷頁**（`/debug`、`/session/{id}/debug-page`）。

**購物功能**

- 購物清單（自動分類）、預算、歷史紀錄、食材紀錄、AI 助手、附近商店（含 AR 顯示）、收據掃描。

---

## 系統架構

```
┌────────────────────────────┐   HTTP/JSON   ┌──────────────────────────────────────────────┐
│ Android App（Kotlin）       │ ◄───────────► │ FastAPI 後端（backend/server）               │
│                            │   Retrofit    │                                              │
│ NavigationScreen  拍照 ─────┼── JPEG ─────► │ VLM 感知（OpenAI 相容 API，最多 3 輪）         │
│ MapPositionEstimator（PDR） │               │ 定位融合：DINOv2 ＋ 物件 ＋ OCR ＋ 感測器       │
│ PathFollower ＋ 小地圖       │ ◄─ 指引／路線 ─┤ 路線規劃（Neo4j 拓樸圖）                      │
│ Sensor Lab（建圖用）         │               │ VLM 決策 → MOVE / ASK / ARRIVED              │
└────────────────────────────┘               └───────────────┬──────────────────────────────┘
                                                             │
                          ┌──────────────────────────────────┼─────────────────────┐
                          │ Neo4j Aura：店家地圖（節點、走道、方向照片、物件、DINOv2 向量）│
                          │ backend/maps：起點、額外走道邊、編輯器地圖                    │
                          └──────────────────────────────────────────────────────────┘

App 的 AI 購物功能（分類、AI 助手、收據）另外直接連 CGU LLM 閘道，不經過導航後端。
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

1. 用 App 的 **Sensor Lab** 在店裡走一圈拍照（同時錄 PDR）。
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
| Android Studio | Ladybug 以上（AGP 9.0.0） |
| JDK | 17+ |
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

# ── Android App ──
BACKEND_URL=http://192.168.x.x:8000/
CGU_API_KEY=your-cgu-key           # 購物清單分類、AI 助手、收據（CGU 閘道）
CGU_BASE_URL=https://air.cgu.edu.tw/cgullmapi/v1
MAPS_API_KEY=your-google-maps-key  # 附近商店
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

### 3. 安裝 App

1. 用 Android Studio 開啟專案根目錄，連接手機後 Run。
2. 後端連線：模擬器預設 `http://10.0.2.2:8000/`；實體手機用 `adb reverse tcp:8000 tcp:8000`，
   或在 `.env` 設 `BACKEND_URL`，也可在 App 的設定頁切換。

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
├── app/                      Android App（Kotlin、Jetpack Compose）
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

### Android App（`app/src/main/java/com/example/shopping/`）

| 檔案 | 功能 |
|------|------|
| `MainActivity.kt` | 進入點與頁面路由（登入、主畫面、AR 導航、設定、Sensor Lab） |
| `ui/screens/NavigationScreen.kt` | 導航：選店與目標（`TeammateHomeScreen`）、相機、指引、PDR 跟隨 |
| `ui/screens/NavigationPanels.kt` | 導航畫面元件：狀態卡、上一張照片縮圖、購物清單、小地圖、拍照說明 |
| `ui/screens/ArGuidanceArrow.kt` | AR 方向箭頭 |
| `viewmodel/NavigationViewModel.kt` | 導航狀態：session、拍照上傳、定位、路線、抵達確認 |
| `navigation/MapPositionEstimator.kt` | PDR 在地圖上的位置估計（移植自 iOS） |
| `navigation/PathFollower.kt` | 沿路線前進與轉彎提示（移植自 iOS） |
| `sensor/PdrTracker.kt` | 計步與航向（Rotation Vector／Game Rotation Vector 北方校正） |
| `network/NavigationApi.kt` / `BackendConfig.kt` | 導航後端 API 與後端網址設定 |
| `ui/screens/SensorLabScreen.kt` | Sensor Lab：全螢幕拍照與感測器錄製（背景執行緒、約 50 Hz） |
| `ui/screens/MainContainer.kt` | 主畫面分頁（首頁、清單、食材、預算、歷史、AI），CGU 閘道請求 |
| `ui/screens/HomeScreen.kt`、`ShoppingListScreen.kt`、`IngredientsScreen.kt`、`HistoryScreen.kt`、`AIScreen.kt`、`LoginScreen.kt`、`SettingsScreen.kt`、`NearbyStoresSheet.kt`、`NearbyStoresAr.kt` | 購物相關頁面 |
| `ui/utils/CategoryClassifier.kt`、`LlmCategoryClassifier.kt` | 商品分類（例外清單 → LLM → 關鍵字） |

---

## 商品分類

每個商品屬於一個首頁分類按鈕，再由按鈕換算成預算類別：

| 首頁分類 | 預算類別 |
|------|------|
| 蔬果、肉品海鮮、蛋奶、主食、零食、調味料 | 食品（`Food`） |
| 飲品 | 飲品（`Beverages`） |
| 清潔用品、廚房用品 | 生活用品（`Groceries`） |
| 其他 | 其他（`Other`） |

判斷順序：**例外清單 → LLM → 關鍵字備案**。LLM 走 CGU 閘道的 `gpt-5.6-luna`（`CGU_CHAT_MODEL`），
AI 助手與收據掃描也用同一個模型。

- **何時分類**：手動新增、修改名稱、掃描收據時；App 啟動時把還沒分類的舊商品一次補完。
  結果存在 `shopping_list.json` 的 `homeCategory` 與 `location`，之後不會重複呼叫 LLM。
- **LLM 失敗時**：顯示「AI 分類失敗，已改用關鍵字分類」，改用關鍵字分類。
- **修正誤判**：在 `CategoryClassifier.kt` 的 `productOverrides` 加一行（例如 `"米酒" to HOME_SEASONING`），
  名稱包含該字串就套用，優先於 LLM。
- **實際打 LLM 測試**：`RUN_LLM_PROBE=1 ./gradlew :app:testDebugUnitTest --tests com.example.shopping.ui.utils.LlmClassificationProbe`。

---

## 測試

```powershell
# 後端
cd backend
.\venv\Scripts\python.exe -m pytest -q

# Android 單元測試
.\gradlew :app:testDebugUnitTest
```

目前後端 `test_vlm.py` 有 2 個檢查 perception 提示詞文字的測試未通過（整合前就存在）。

---

## 疑難排解

| 問題 | 解決方式 |
|------|----------|
| 手機連不到後端 | 確認同一 Wi-Fi、IP 正確、防火牆允許 8000；或 `adb reverse tcp:8000 tcp:8000` |
| 選店清單是空的 | 確認 `.env` 的 `NEO4J_URI`／`NEO4J_PASSWORD`；沒有 Neo4j 時只會列出 `NEO4J_PLACE` |
| 定位只靠物件與文字 | 伺服器端 DINOv2 需要 `torch`＋`transformers` 與 `dino_sidecar.pkl`；`LOC_SERVER_DINO=0` 可關閉 |
| 不想每張照片多一次 embedding 請求 | 設 `EMBEDDING_ENABLED=0`，物件名稱改回只用字元相似度 |
| AGP 版本不相容 | `gradle/libs.versions.toml` 中 `agp` 設為 `9.0.0` |

---

## 相關文件

- [`backend/eval/README.md`](backend/eval/README.md) — 評估工具與指標
- [`backend/legacy/README_locate_v2.md`](backend/legacy/README_locate_v2.md)、[`backend/legacy/README_topomap_v2.md`](backend/legacy/README_topomap_v2.md) — 舊版定位與拓樸地圖（研究用）
- `docs/` — 設計文件書、測試計劃書、期中／期末報告
- `http://localhost:8000/docs` — Swagger UI
