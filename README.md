<div align="center">

# 🛒 智慧購物導航 (Smart Shopping Navigation)

**以 Android 為基礎的綜合型智慧購物助理 — 整合相機 OCR、雲端大型語言模型與賣場視覺導航**

![Kotlin](https://img.shields.io/badge/Kotlin-2.2.10-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white)
![Jetpack Compose](https://img.shields.io/badge/Jetpack_Compose-Material_3-4285F4?style=for-the-badge&logo=jetpackcompose&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Auth-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3--70B-F55036?style=for-the-badge)
![ML Kit](https://img.shields.io/badge/ML_Kit-Text_OCR-4285F4?style=for-the-badge&logo=google&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-REST-0062FF?style=for-the-badge)
![CameraX](https://img.shields.io/badge/CameraX-1.5.3-3DDC84?style=for-the-badge&logo=android&logoColor=white)
![Places](https://img.shields.io/badge/Google_Places-New-34A853?style=for-the-badge&logo=googlemaps&logoColor=white)
<br/>
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![GroundingDINO](https://img.shields.io/badge/GroundingDINO-Detection-555555?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-LLaMA_3.2_Vision-000000?style=for-the-badge&logo=ollama&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-Topomap-FF6F00?style=for-the-badge)

</div>

> 以手機相機擷取影像 + 雲端 LLM 進行收據/成分結構化與購物諮詢;賣場導航以拓樸地圖達成,**無需在賣場建置藍牙/Wi-Fi 定位硬體**。

---

## 📑 目錄 (Table of Contents)

- [專題簡介](#-專題簡介-overview)
- [系統特色](#-系統特色-features)
- [系統架構](#️-系統架構-architecture)
- [AI 應用與模型流程](#-ai-應用與模型流程-ai--model-flow)
- [核心功能](#-核心功能-core-features)
- [技術堆疊](#-技術堆疊-tech-stack)
- [設定金鑰](#-設定金鑰-api-keys)
- [安裝與使用](#-安裝與使用-installation--usage)
- [專案結構](#-專案結構-project-structure)
- [已知限制與容錯](#️-已知限制與容錯-limitations--robustness)

---

## 📖 專題簡介 (Overview)

「智慧購物導航」以 **Kotlin + Jetpack Compose** 開發,採**單一 Activity** 架構,搭配獨立的 **Python (FastAPI) 後端導航伺服器**([`backend/`](backend/README.md))處理賣場內視覺定位。

- **核心目標**:解決消費者在大型賣場找不到商品、繞遠路的問題,並一併管理預算與飲食健康。
- **兩大模組**:① 賣場導航(影像語意 + 拓樸地圖)、② AI 小助理(預算 / 健康 / 諮詢 / 附近門市)。

### 專題資訊
- **指導教授**:吳世琳 教授、陳嶽鵬 教授
- **專題成員**:B1229013 陳宜伶 · B1229020 何思顗 · B1229023 林依賢 · B1222017 周庠

---

## ✨ 系統特色 (Features)

| | 特色 | 說明 |
|---|------|------|
| 🧭 | **賣場視覺導航** | 手機拍照即定位,以拓樸地圖逐步引導找商品,免裝藍牙/Wi-Fi 信標 |
| 🤖 | **情境感知 AI 助理** | Groq LLaMA 3.3 取得即時購物 / 預算 / 健康快照,跨模組回答問題 |
| 🧾 | **收據秒記帳** | 拍收據 → ML Kit OCR → LLM 結構化 → 自動入帳 + 預算圓環圖 |
| 🥗 | **成分過敏偵測** | 拍成分表 → PaddleOCR → 比對個人過敏原 + 熱量/運動換算 |
| 📍 | **附近門市搜尋** | Google Places + GPS,清單與相機 AR 兩種呈現 |
| 🔒 | **資料本地化** | 購物/飲食/預算/個資僅存於裝置;導航模型可本地端 Ollama 推論 |

---

## 🏗️ 系統架構 (Architecture)

<div align="center">

<!-- 產生架構圖後置於 docs/architecture.png 並取消下一行註解 -->
<!-- ![系統架構圖](docs/architecture.png) -->

</div>

```
┌────────────────────────────────────────────────────────────┐
│  Android App (com.example.shopping) — 單一 Activity + Compose │
│  MainActivity → NavHost (起始頁 login)                       │
│   ├─ login           LoginScreen     (Firebase Auth + 信箱驗證)│
│   ├─ main_list       MainContainer   (六分頁底部導覽)          │
│   ├─ teammate_home   TeammateHomeScreen (導航前:清單+附近商店)│
│   ├─ ar_navigation   NavigationScreen   (CameraX 即時 + AR 標記)│
│   └─ settings        SettingsScreen     (個人資料 + 快取管理)  │
└────────────────────────────────────────────────────────────┘
            │ 外部服務 (Retrofit / SDK)
            ▼
  Groq API (LLaMA 3.3) │ ML Kit OCR │ PaddleOCR REST │ Firebase Auth │ Google Places
            │ (賣場視覺定位,整合進行中)
            ▼
┌────────────────────────────────────────────────────────────┐
│  Python 後端導航伺服器 backend/ — FastAPI                     │
│  GroundingDINO + EasyOCR + Ollama VLM + NetworkX 拓樸地圖     │
└────────────────────────────────────────────────────────────┘
```

### 狀態管理與持久化
- `MainContainer` 以 Material 3 **Scaffold** 統籌六個分頁,核心狀態:`shoppingItems`、`dietRecords`、`budgetTotalStr`。
- 持久化以 **kotlinx.serialization** 寫入 `filesDir`:`shopping_list.json`、`diet_records.json`、`monthly_budget.txt`、`user_profile.json`。
- 切換分頁採 `selectedTab` 索引 + `Crossfade`(單向資料流)。

---

## 🧠 AI 應用與模型流程 (AI & Model Flow)

系統的 AI 分為**雲端服務(App 端直接呼叫)**與**本地視覺導航管線(後端)**兩條路徑。

### A. App 端:OCR → LLM 結構化
```
拍照 ──▶ OCR (收據:ML Kit / 成分:PaddleOCR) ──▶ Groq LLaMA 3.3
                                                   │
            ┌──────────────────────────────────────┴───────────────┐
            ▼                                                        ▼
   收據 → TidiedReceiptResponse(品項/金額)              助理 → 注入 ShoppingContext
   → 入帳 + DonutChart 預算圖                            (庫存+預算+健康 JSON) → 跨模組回答
```

### B. 後端:賣場視覺導航(設計準則「先偵測 → 再讀字 → 後推論 → 拓樸定位」)
```
顧客照片 ─▶ GroundingDINO 物件偵測 ─▶ EasyOCR 招牌/標籤辨識 ─▶ Ollama VLM (LLaMA 3.2 Vision)
                                                                 │
            ┌────────────────────────────────────────────────────┘
            ▼
   NetworkX 拓樸地圖(節點=位置, 邊=走過路徑) ─▶ 導航動作:MOVE / ASK / ARRIVED
```
> 詳細模組、API 與啟動方式見 **[`backend/README.md`](backend/README.md)**。

---

## 🧩 核心功能 (Core Features)

底部導覽列共六個分頁:**首頁 / 清單 / 分析 / 預算 / 紀錄 / 助理**。

1. **首頁 (`HomeScreen`)** — 商品搜尋與快速加入清單,縮圖以 Coil 載入。
2. **購物清單 (`ShoppingListScreen`)** — 項目新增/編輯/勾選(`ShoppingItem`)。
3. **分析 / 飲食 (`IngredientsScreen`)** — 成分拍照 → **PaddleOCR REST**(`callPaddleOcr`) → 內建過敏原字典比對 + 內建 **MET 表**熱量/運動換算 → 存入 `diet_records.json`。
4. **預算與收據 (`BudgetScreen`,於 `MainContainer.kt`)** — 收據 → **ML Kit `TextRecognition`** → **Groq**(`response_format=json_object`)→ `TidiedReceiptResponse` → 入帳 + **`DonutChart`**。
5. **紀錄 (`HistoryScreen`)** — 依時間檢視消費 / 購物歷史。
6. **AI 助理 (`AIScreen`)** — **Groq `llama-3.3-70b-versatile`**,將 `ShoppingContext`(見 `model/ShoppingContext.kt`)注入系統提示,支援多輪情境對話。

**賣場導航 (`NavigationScreen`,路由 `ar_navigation`)** — 已實作 CameraX 即時預覽 + 每 3 秒取樣、GPS、`NearbyStoresAr` AR 店家標記、「選取模擬圖片」。**尚未串接**:每幀分析 `processImageForModel()` 目前為佔位實作,將串接 [`backend/`](backend/README.md) 導航伺服器。

**附近門市 (`NearbyStoresSheet` / `NearbyStoresAr`)** — Google Places SDK + `FusedLocationProviderClient`,haversine 計算距離。

**登入 / 設定** — Firebase Auth 雙模式 + Email 驗證攔截;設定管理個資(寫入 `user_profile.json`)與快取清除。

---

## 🛠️ 技術堆疊 (Tech Stack)

> 以 `app/build.gradle.kts` 與 `gradle/libs.versions.toml` 為準。

| 類別 | 技術項目 |
|------|----------|
| 語言 / UI | **Kotlin 2.2.10** · **Jetpack Compose** (Material 3, navigation-compose, material-icons-extended) |
| 影像載入 | **Coil 2.7.0** |
| 序列化 | **kotlinx.serialization** (本地 JSON 持久化) |
| 網路 | **Retrofit 2.9.0** + Gson + OkHttp logging-interceptor |
| 雲端 LLM | **Groq API** (`llama-3.3-70b-versatile`) — AI 助理 + 收據結構化 |
| 收據 OCR | **ML Kit Text Recognition 16.0.1** |
| 成分 OCR | **PaddleOCR**(自架 REST,`PADDLEOCR_ACCESS_TOKEN`) |
| 身份驗證 | **Firebase Auth** (BoM 33.9.0) + Analytics |
| 相機 | **CameraX 1.5.3** |
| 定位 / 地點 | **play-services-location 21.3.0** + **Places SDK (New) 4.1.0** |
| (相依但未呼叫) | Gemini generativeai 0.9.0(程式碼目前改用 Groq) |
| 後端導航 | FastAPI · GroundingDINO · EasyOCR · Ollama (LLaMA 3.2 Vision) · NetworkX — 見 [`backend/`](backend/README.md) |

---

## 🔑 設定金鑰 (API Keys)

金鑰透過專案根目錄 `.env` 載入並注入 `BuildConfig`(`.env` 已 `.gitignore`,請參考 `.env.example`):

```
MAPS_API_KEY=            # Google Places / Maps
GEMINI_API_KEY=          # (目前未實際使用)
GROQ_API_KEY=            # Groq LLM
PADDLEOCR_ACCESS_TOKEN=  # PaddleOCR REST
```

`MAPS_API_KEY` 另透過 manifestPlaceholder 注入 `AndroidManifest.xml` 供 Places 使用。

---

## 📲 安裝與使用 (Installation & Usage)

> ⚠️ 本節僅涵蓋 **Android App**。賣場視覺定位所用的 AI 模型(GroundingDINO + EasyOCR + Ollama VLM)**不在 App 內**,而在獨立的 **[`backend/`](backend/README.md)** 資料夾(獨立 Python / FastAPI 服務),其安裝、模型下載與啟動請見 **[`backend/README.md`](backend/README.md)**。

### 前置需求
- **Android Studio**(Hedgehog+)+ **JDK 11** · **Android SDK**(compileSdk 35)
- 實機或模擬器,**Android 7.0 (API 24) 以上**
- Firebase `google-services.json`(放入 `app/`)

### 安裝步驟
```bash
# 1) 取得原始碼
git clone https://github.com/B1229013/Shopping-Navigation.git
cd Shopping-Navigation

# 2) 設定金鑰:複製 .env.example 為 .env 並填入 4 把金鑰
#    放入 Firebase 設定檔 app/google-services.json

# 3) 建置 + 安裝(或在 Android Studio 直接按 ▶ Run)
gradlew.bat :app:assembleDebug    # Windows  (macOS/Linux: ./gradlew ...)
gradlew.bat :app:installDebug
```

### 模型 / 服務(App 端)
手機端**無需下載大型模型**,設定好金鑰即可:**Groq**(需 `GROQ_API_KEY`)、**ML Kit**(隨 Play Services 自動下載)、**PaddleOCR**(自架 REST)、**Places / Firebase**。

### 使用流程
`註冊→Email 驗證→登入` → 「清單」加商品 → 「分析」拍成分(過敏/熱量) → 「預算」拍收據(自動入帳) → 「助理」問問題 → 「賣場導航」確認清單後開始導航(需先啟動後端)。

---

## 📁 專案結構 (Project Structure)

```
Shopping-Navigation/
├── app/                                  # Android 客戶端
│   └── src/main/java/com/example/shopping/
│       ├── MainActivity.kt               # 單一 Activity + NavHost
│       ├── model/                        # ShoppingItem / DietRecord / UserProfile / ShoppingContext
│       └── ui/
│           ├── screens/                  # MainContainer(含 BudgetScreen) / Home / ShoppingList
│           │                             #   Ingredients / History / AI / Navigation / NearbyStores ...
│           ├── components/ · utils/ · theme/
│
├── backend/                              # Python 後端導航伺服器 (FastAPI) — 見 backend/README.md
│   ├── server/                           # GroundingDINO + EasyOCR + Ollama VLM + NetworkX
│   ├── eval/ · generate_topomap.py · build_store_*.py · ...
│   └── README.md
│
├── Map/ · .env.example · build.gradle.kts · settings.gradle.kts · README.md
```

---

## ⚠️ 已知限制與容錯 (Limitations & Robustness)

- **賣場視覺定位未完成**:`NavigationScreen` 逐幀分析為佔位,待串接 `backend/`。
- **OCR 光學條件**:光線不足、反光、字體過小會降低 ML Kit / PaddleOCR 成功率。
- **第三方 API 配額**:Groq、Places 受官方頻率/配額限制,需網路連線。
- **容錯**:LLM 呼叫以 `try-catch` 包覆;金鑰經 `.env`→`BuildConfig`(不入庫);資料僅存 `filesDir`;Email 未驗證帳號自動 `signOut()` 攔截。
