# 感測器導航系統（Sensor-Nav）

PDR 計步器 + AI 視覺融合的室內建圖與導航模組。  
邊走邊拍，自動建立帶有步數、距離、航向的拓撲地圖。

---

## 目錄

1. [系統架構](#系統架構)
2. [使用流程](#使用流程)
3. [感測器規格](#感測器規格)
4. [PDR 演算法](#pdr-演算法)
5. [檔案結構](#檔案結構)
6. [API 端點](#api-端點)
7. [資料格式](#資料格式)
8. [部署方式](#部署方式)
9. [離線模式](#離線模式)
10. [環境變數](#環境變數)

---

## 系統架構

```
┌─────────────────────────────────────────────────┐
│  Android App                                    │
│                                                 │
│  ┌────────────┐   ┌──────────────────────────┐  │
│  │ PdrTracker │   │ SensorNavViewModel       │  │
│  │            │──▶│  - captureAndUpload()    │  │
│  │ 7 sensors  │   │  - savePhotoLocally()    │  │
│  │ ~50Hz raw  │   │  - online / offline mode │  │
│  └────────────┘   └───────────┬──────────────┘  │
│                               │                 │
│  ┌────────────────────────────▼──────────────┐  │
│  │ SensorNavScreen                           │  │
│  │  - CameraX preview                       │  │
│  │  - PDR stats + mini trail map (Canvas)    │  │
│  │  - guidance panel + action buttons        │  │
│  └───────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────┘
                        │ multipart POST
                        │ (photo + pdr_data JSON)
                        ▼
┌─────────────────────────────────────────────────┐
│  FastAPI Backend (/snav/ router)                │
│                                                 │
│  sensor_nav.py                                  │
│   ├─ EXIF fix → resize → OCR → object detect   │
│   ├─ TopoMap: add node (PDR xy) + edge (steps) │
│   ├─ VLM navigation decision                   │
│   ├─ step guidance generation                   │
│   └─ raw sensor JSON storage                    │
│                                                 │
│  map_store.py                                   │
│   └─ save / load / list / delete maps (JSON)    │
└─────────────────────────────────────────────────┘
```

---

## 使用流程

### 連線模式（需後端）

```
設定頁 → 感測器導航 → 輸入目的地 → 開始導航
  │
  ▼
┌──────────────────────────────────────────────┐
│ 循環：                                       │
│  1. 使用者行走（PdrTracker 持續記錄）         │
│  2. 使用者拍照（CameraX 或從相簿選取）       │
│  3. App 取 PdrSnapshot + 上傳 photo + JSON   │
│  4. 後端回傳 guidance + action               │
│     - MOVE → 繼續走                          │
│     - ASK  → 顯示問題，等使用者回答          │
│     - ARRIVED → 顯示確認按鈕                 │
│  5. 回到步驟 1                               │
└──────────────────────────────────────────────┘
  │
  ▼ 確認到達後
儲存地圖（可選） → 結束
```

### 離線模式（不需後端）

```
設定頁 → 感測器導航 → 切換「離線採集模式」→ 輸入路線標記 → 開始採集
  │
  ▼
循環拍照：每次拍照存到手機
  photo + PDR snapshot + raw sensors → filesDir/snav_local/{id}/
  │
  ▼ 按「結束採集」
首頁列出所有紀錄 → 點分享按鈕匯出 ZIP
```

---

## 感測器規格

PdrTracker 註冊 7 種感測器：

| 感測器 | Android 常數 | 用途 | 緩衝頻率 |
|--------|-------------|------|---------|
| Accelerometer | `TYPE_ACCELEROMETER` | 步長估計（Weinberg） + 原始錄製 | ~50Hz（20ms 節流） |
| Gyroscope | `TYPE_GYROSCOPE` | 原始錄製（供後處理） | ~50Hz |
| Magnetometer | `TYPE_MAGNETIC_FIELD` | 磁場航向 + 原始錄製 | ~50Hz |
| Rotation Vector | `TYPE_ROTATION_VECTOR` | 原始錄製（含磁場） | 即時（無節流） |
| Game Rotation Vector | `TYPE_GAME_ROTATION_VECTOR` | PDR 航向（無磁干擾，室內穩定） | 即時 |
| Step Counter | `TYPE_STEP_COUNTER` | PDR 步數累計 + 觸發步長計算 | 事件驅動 |
| Step Detector | `TYPE_STEP_DETECTOR` | 每步時間戳錄製 | 事件驅動 |

### 權限

- `ACTIVITY_RECOGNITION`（Android Q+）：步數感測器需要此權限
- `CAMERA`：CameraX 拍照
- `AndroidManifest.xml` 已宣告，App 啟動時動態請求

---

## PDR 演算法

### 步長估計 — Weinberg 方法

```
stride = K × (a_max − a_min)^0.25
```

- `K` = 0.4667（預設值，可調）
- `a_max`, `a_min` = 上一步到這一步之間加速度向量模的最大與最小值
- 每次 `TYPE_STEP_COUNTER` 事件觸發時計算

### 航向 — Game Rotation Vector

使用 `TYPE_GAME_ROTATION_VECTOR`（不含磁力計，室內不受磁場干擾）：

```kotlin
val yaw = atan2(
    2.0 * (w * z + x * y),
    1.0 - 2.0 * (y * y + z * z)
)
```

### 位置更新

```kotlin
x += steps × stride × sin(yaw)
y += steps × stride × cos(yaw)
```

### Snapshot 機制

每次拍照時呼叫 `PdrTracker.snapshot()`：

1. 計算區段平均航向
2. 打包 PDR 摘要（steps, distance, heading, x, y）
3. 打包原始感測器緩衝（accel/gyro/mag/rotVec/gameRotVec/stepEvents）
4. 清空緩衝，開始下一個區段

---

## 檔案結構

### Android（新增）

```
app/src/main/java/com/example/shopping/
├── sensor/
│   ├── PdrTracker.kt           # PDR 引擎：7 感測器註冊、步長估計、原始緩衝、snapshot
│   └── LocalSessionStore.kt    # 離線模式本地儲存：session/photo/sensor JSON、ZIP 匯出
├── network/
│   └── SensorNavApi.kt         # Retrofit service → /snav/ 端點
├── viewmodel/
│   └── SensorNavViewModel.kt   # 整合 PDR + 上傳 + 離線模式 + 地圖管理
└── ui/screens/
    └── SensorNavScreen.kt      # UI：首頁（目標輸入+離線切換）+ 導航畫面
```

### Android（修改）

```
MainActivity.kt     # +2 composable routes: sensor_nav, sensor_nav_active
SettingsScreen.kt   # +1 入口卡片「感測器導航」
file_paths.xml      # +1 FileProvider path for snav_local
```

### Backend（新增）

```
backend/server/
├── sensor_nav.py    # FastAPI router /snav/：session, photo, answer, confirm, maps
└── map_store.py     # 地圖 JSON 持久化：save/load/list/delete
```

### Backend（修改）

```
server.py           # +2 行：import sensor_nav router → app.include_router()
```

---

## API 端點

所有端點在 `/snav/` prefix 下。

### Session

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/snav/session` | 建立導航 session |
| `POST` | `/snav/{id}/photo` | 上傳照片 + PDR 資料（multipart） |
| `POST` | `/snav/{id}/answer` | 回答 VLM 提問 |
| `POST` | `/snav/{id}/confirm` | 確認/否認抵達 |

### Maps

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/snav/maps/save` | 儲存地圖 |
| `GET` | `/snav/maps` | 列出所有地圖 |
| `GET` | `/snav/maps/{id}` | 取得地圖詳情 |
| `DELETE` | `/snav/maps/{id}` | 刪除地圖 |

### Static

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/snav/{id}/photo/{filename}` | 取得標註照片 |
| `GET` | `/snav/{id}/map` | 取得 session 地圖（JSON 或 PNG） |

---

## 資料格式

### 上傳：POST /snav/{id}/photo

Multipart form-data，兩個欄位：

| 欄位 | Content-Type | 說明 |
|------|-------------|------|
| `photo` | `image/jpeg` | 照片檔案 |
| `pdr_data` | `text/plain` | JSON 字串（見下） |

#### pdr_data JSON 結構

```json
{
  "steps": 8,
  "distance_m": 5.23,
  "heading_deg": 127.3,
  "pdr_x": 3.12,
  "pdr_y": -4.01,
  "total_steps": 42,
  "total_distance_m": 28.7,
  "raw_sensors": {
    "accel": [[elapsed_ms, x, y, z], ...],
    "gyro": [[elapsed_ms, x, y, z], ...],
    "mag": [[elapsed_ms, x, y, z], ...],
    "rot_vec": [[elapsed_ms, x, y, z, w], ...],
    "game_rot_vec": [[elapsed_ms, x, y, z, w], ...],
    "step_events": [elapsed_ms, ...],
    "sample_counts": {
      "accel": 150,
      "gyro": 150,
      "mag": 150,
      "rot_vec": 300,
      "game_rot_vec": 300,
      "step_events": 8
    }
  }
}
```

- `elapsed_ms`：相對於 PdrTracker 啟動時間的毫秒數
- 浮點值保留 4 位小數
- `steps`/`distance_m`/`heading_deg`：本區段（上次 snapshot 至今）
- `total_*`：全程累計
- `raw_sensors`：~50Hz 時序原始資料，每次 snapshot 清空

### 回應：SNavTurnResponse

```json
{
  "action": "MOVE",
  "guidance": "前方有走廊，向右轉可看到飲水機",
  "question": null,
  "node_id": 3,
  "annotated_photo_url": "/snav/abc12345/photo/3.jpg",
  "step_guidance": "已走 15 步（10.2m）　預估還需 ~20 步",
  "estimated_remaining_steps": 20,
  "estimated_remaining_distance": 14.5
}
```

- `action`：`MOVE`（繼續走）/ `ASK`（需回答問題）/ `ARRIVED`（判斷到達）
- `step_guidance`：結合 PDR 和圖最短路徑的步數預估文字

### 後端儲存結構

```
output/sessions/snav_{session_id}/
├── photo/
│   ├── 0.jpg                  # 原始照片
│   └── 1.jpg
├── annotated/
│   ├── 0.jpg                  # 標註照片（偵測框+導航橫幅）
│   └── 1.jpg
├── sensors/
│   ├── sensors_0.json         # 原始感測器資料（含 PDR 摘要）
│   └── sensors_1.json
├── map/
│   ├── map_0.json             # 每步的地圖快照
│   └── map_1.json
└── graph/
    └── goal_graph.png         # 目標分解圖

output/maps/
└── {map_id}.json              # 持久化地圖（含 PDR 度量）
```

### 離線模式本地儲存

```
filesDir/snav_local/{session_id}/
├── manifest.json              # session 總覽 + entry 索引
├── photos/
│   ├── photo_1.jpg
│   └── photo_2.jpg
└── sensors/
    ├── sensors_1.json         # PDR 摘要 + 原始感測器資料
    └── sensors_2.json
```

匯出時打包為 ZIP，透過 Android Share Sheet 分享。

### 持久化地圖 JSON

```json
{
  "map_id": "a1b2c3d4",
  "name": "3樓飲水機路線",
  "node_count": 5,
  "total_steps": 120,
  "total_distance_m": 85.3,
  "created_at": "2026-08-03T10:30:00",
  "goal": "飲水機",
  "nodes": [
    {
      "id": 0,
      "detected": ["door", "sign"],
      "ocr_texts": ["305"],
      "summary": "走廊入口",
      "pdr_x": 0.0,
      "pdr_y": 0.0
    }
  ],
  "edges": [
    {
      "from": 0,
      "to": 1,
      "action": "往右邊走廊前進",
      "steps": 25,
      "distance_m": 17.5,
      "heading_deg": 90.0
    }
  ]
}
```

---

## 部署方式

### 本機測試

```bash
cd backend
.\start.ps1
# 或
python -m uvicorn server.server:app --host 0.0.0.0 --port 8000
```

- **模擬器**：App 預設指向 `http://10.0.2.2:8000/`，不用改設定
- **實體手機 USB**：`adb reverse tcp:8000 tcp:8000`，App 自動連上
- **同網路 WiFi**：`.env` 設 `BACKEND_URL=http://你的IP:8000/`

### Render 雲端部署

已準備好部署檔案：

```
backend/
├── Dockerfile              # Python 3.11 slim，只裝核心依賴
├── requirements-cloud.txt  # 無 torch/easyocr（~200MB image）
└── render.yaml             # Render 一鍵部署設定
```

步驟：

1. 推到 GitHub
2. render.com → New Web Service → 連結 repo
3. Root Directory 設 `backend`
4. Environment 設 `CGU_API_KEY`（或 `OPENAI_API_KEY`）
5. Deploy → 取得 URL
6. `.env` 加 `BACKEND_URL=https://你的app.onrender.com/`

雲端版停用 GroundingDINO + EasyOCR（`PERCEPTION_ENABLED=0`, `OCR_ENABLED=0`），完全走 VLM 路線。免費方案可用。

### 本機快速驗證

瀏覽器開 `http://localhost:8000/docs`，Swagger UI 會顯示所有 `/snav/` 端點。

---

## 離線模式

不需後端，照片與感測器資料全部存在手機裡。

### 功能

- 首頁 Switch 切換「連線導航」/「離線採集」
- 離線模式下直接建立本地 session，不呼叫 API
- 每次拍照存 photo + PdrSnapshot + rawSensorJson 到 `filesDir/snav_local/`
- 按「結束採集」結束
- 首頁顯示所有本地紀錄，支援刪除和分享（ZIP 匯出）

### 匯出資料

**App 內匯出**：紀錄列表旁的分享按鈕 → ZIP → Android Share Sheet

**ADB 直接拉**：
```bash
adb shell run-as com.example.shopping tar -cf - files/snav_local | tar -xf - -C ./exported
```

**Android Studio**：Device Explorer → `data/data/com.example.shopping/files/snav_local/`

---

## 環境變數

### 後端

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `PERCEPTION_ENABLED` | `1` | `0` = 停用 GroundingDINO（雲端部署用） |
| `OCR_ENABLED` | `1` | `0` = 停用 EasyOCR（由 VLM 做文字偵測） |
| `VLM_BACKEND` | `openai` | `openai` / `gemini` / `ollama` |
| `CGU_API_KEY` | — | CGU LLM Gateway 金鑰 |
| `OPENAI_API_KEY` | — | OpenAI API 金鑰（CGU_API_KEY 為空時使用） |
| `OPENAI_MODEL` | `gpt-4o` | VLM 模型 |
| `SERVER_HOST` | `0.0.0.0` | 綁定位址 |
| `SERVER_PORT` | `8000` | 綁定埠號 |

### Android

| 變數（`.env`） | 說明 |
|---------------|------|
| `BACKEND_URL` | 後端 URL，例如 `http://10.0.2.2:8000/`（模擬器）或 Render URL |

---

## 故障排除

### 步數沒有偵測到

1. Logcat 過濾 `PdrTracker`，看是否有 `Registered: StepCounter`
   - 如果顯示 `NOT AVAILABLE: StepCounter` → 裝置無硬體計步器（模擬器通常沒有）
2. 確認 `ACTIVITY_RECOGNITION` 權限已授予（Android Q+）
3. 確認 PdrTracker 在權限授予後才啟動（已在最新版修正時序問題）

### 連線失敗

1. 確認後端啟動：`http://localhost:8000/docs` 是否正常
2. 模擬器：預設 `http://10.0.2.2:8000/` 應可連通
3. 實體手機：確認 `adb reverse tcp:8000 tcp:8000` 或 `.env` 的 `BACKEND_URL` 正確
4. 雲端部署：確認 `BACKEND_URL` 包含 `https://` 且末尾有 `/`

### 離線模式資料在哪

- App 私有目錄：`filesDir/snav_local/`
- 用分享按鈕匯出 ZIP，或 ADB / Device Explorer 直接存取
