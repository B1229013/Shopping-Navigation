# 智慧購物導航 (Smart Shopping Navigation)

以 Android 為基礎的綜合型智慧購物助理應用程式,整合相機 OCR、雲端大型語言模型 (Groq / LLaMA 3.3) 與地點服務,協助消費者管理購物清單、預算、飲食健康,並提供賣場內導航(視覺定位後端整合進行中)。

---

## 專題簡介 (Project Overview)

「智慧購物導航」以 **Kotlin + Jetpack Compose** 開發,採**單一 Activity** 架構,搭配規劃中的 **Python (FastAPI) 後端導航伺服器**([`backend/`](backend/README.md))處理賣場內視覺定位。

- **核心目標**:解決消費者在大型賣場找不到商品、繞遠路的問題,並一併管理預算與飲食健康。
- **技術特色**:以手機相機擷取影像 + 雲端 LLM 進行收據/成分結構化與購物諮詢;賣場導航規劃以拓樸地圖達成,**無需在賣場建置藍牙/Wi-Fi 定位硬體**。

### 專題資訊
- **指導教授**:吳世琳 教授、陳嶽鵬 教授
- **專題成員**:
  - B1229013 陳宜伶
  - B1229020 何思顗
  - B1229023 林依賢
  - B1222017 周庠

---

## 系統架構 (Architecture)

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
- 持久化以 **kotlinx.serialization** 寫入 `filesDir` 下的本地檔:
  - `shopping_list.json`(購物清單)、`diet_records.json`(飲食紀錄)、`monthly_budget.txt`(月預算)、`user_profile.json`(個人資料 / 過敏原)。
- 切換分頁採 `selectedTab` 索引 + `Crossfade`;子畫面以參數接收 State 與 Callback(單向資料流)。

---

## 核心功能 (Core Features)

底部導覽列共六個分頁:**首頁 / 清單 / 分析 / 預算 / 紀錄 / 助理**。

### 1. 首頁 (`HomeScreen`)
- 商品搜尋與快速加入購物清單,商品縮圖以 **Coil** 載入(Unsplash)。
- 提供前往「清單」與「預算」分頁的捷徑。

### 2. 購物清單 (`ShoppingListScreen`)
- 購物項目的新增/編輯/勾選(`ShoppingItem`:名稱、數量、價格、是否已購、效期、店家、分類等)。

### 3. 分析 / 飲食健康 (`IngredientsScreen`)
- **成分 OCR**:相機拍照(`TakePicture` + `FileProvider`,暫存 `ocr_scan.jpg`)→ 呼叫 **PaddleOCR REST API**(`callPaddleOcr`,網址取自 `R.string.paddleocr_api_url`,權杖取自 `BuildConfig.PADDLEOCR_ACCESS_TOKEN`)取得成分文字。
- **過敏原 / 疾病警示**:以**內建過敏原字典**(`allergenMap`,含別名)比對使用者 `UserProfile.allergies`,命中時即時顯示紅色警告。
- **熱量與運動換算**:依個人資料計算建議熱量;以**內建 MET 表**(慢跑、走路、騎腳踏車等)將攝取熱量換算為所需運動分鐘。
- 結果封裝為 `DietRecord`(含熱量與碳水/糖/蛋白質/脂肪/鈉等),序列化存入 `diet_records.json`。

### 4. 預算與收據掃描 (`BudgetScreen`,定義於 `MainContainer.kt`)
- 拍照或選圖取得收據影像 → **ML Kit `TextRecognition`** 擷取原始文字。
- 將 OCR 文字送至 **Groq API**(`response_format = json_object`)結構化 → 反序列化為 `TidiedReceiptResponse`(`budget_entry` / `line_items`)。
- 解析後的品項可併入購物清單作為消費紀錄,並以 **圓環圖 `DonutChart`** 視覺化預算使用率。

### 5. 紀錄 (`HistoryScreen`)
- 依購買時間檢視消費 / 購物歷史。

### 6. AI 購物助理 (`AIScreen`)
- **Groq API**,模型 `llama-3.3-70b-versatile`,系統角色為繁體中文購物助理。
- **情境感知**:每次提問都將 `ShoppingContext`(庫存 + 預算 + 健康的 JSON 快照,見 `model/ShoppingContext.kt`)注入系統提示,可跨模組回答(如「這個月零食花了多少?」「清單會超過預算嗎?」)。
- 多輪對話:附帶最近 5 組歷史;以 `rememberCoroutineScope` 管理非同步呼叫。

### 賣場導航 (`NavigationScreen`,路由 `ar_navigation`)
- 由 `TeammateHomeScreen`(確認清單 + 「附近商店 1 公里內」)按「開始導航」進入。
- 已實作:**CameraX** 即時預覽 + `ImageAnalysis` **每 3 秒**取樣一幀、GPS 定位、`NearbyStoresAr` **AR 店家標記**疊加、以及「選取模擬圖片」測試入口。
- **尚未串接**:每幀影像分析 `processImageForModel()` 目前為佔位實作(僅記錄並關閉影像)。賣場視覺定位/逐步指引將串接 [`backend/`](backend/README.md) 之 Python 導航伺服器(GroundingDINO + EasyOCR + Ollama VLM + NetworkX)。

### 附近門市 (`NearbyStoresSheet` / `NearbyStoresAr`)
- **Google Places SDK (New)** + `FusedLocationProviderClient` 取得周邊門市,以 haversine 計算距離;提供清單式底部面板與相機 AR 標記兩種呈現。

### 登入 (`LoginScreen`) 與設定 (`SettingsScreen`)
- **Firebase Authentication**:登入/註冊雙模式,登入後檢查 `isEmailVerified`,未驗證則 `signOut()` 攔截;註冊後寄送驗證信。
- 設定:個人資料(性別、生日、身高體重、過敏原、疾病、活動量)寫入 `user_profile.json`;顯示 `filesDir`/`cacheDir` 占用並可清除快取。

---

## 技術堆疊 (Tech Stack) — 以 `app/build.gradle.kts` 為準

| 類別 | 技術項目 |
|------|----------|
| 語言 / UI | **Kotlin 2.2.10** · **Jetpack Compose** (Material 3, navigation-compose, material-icons-extended) |
| 影像載入 | **Coil 2.7.0** |
| 序列化 | **kotlinx.serialization** (本地 JSON 持久化) |
| 網路 | **Retrofit 2.9.0** + Gson + OkHttp logging-interceptor |
| 雲端 LLM | **Groq API** (`llama-3.3-70b-versatile`,`https://api.groq.com/openai/`) — AI 助理 + 收據結構化 |
| 收據 OCR | **ML Kit Text Recognition 16.0.1** |
| 成分 OCR | **PaddleOCR**(自架 REST API,以 `PADDLEOCR_ACCESS_TOKEN` 存取) |
| 身份驗證 | **Firebase Auth** (BoM 33.9.0) + Analytics |
| 相機 | **CameraX 1.5.3** (core / camera2 / lifecycle / view) |
| 定位 / 地點 | **play-services-location 21.3.0** + **Places SDK (New) 4.1.0** |
| (相依但未呼叫) | **Gemini generativeai 0.9.0**(保留為相依,程式碼目前改用 Groq) |
| 後端導航 | 見 [`backend/`](backend/README.md):FastAPI · GroundingDINO · EasyOCR · Ollama VLM · NetworkX |

### 執行環境
- Android **8.0+** 建議(實際 `minSdk = 24` / Android 7.0;`targetSdk = compileSdk = 35`)。
- 後置相機、GPS + 網路連線;需 Google Play Services。
- 開發:Android Studio、AGP 9.1.0、Kotlin 2.2.10。

---

## 設定金鑰 (API Keys)

金鑰透過專案根目錄的 `.env` 載入並注入 `BuildConfig`(`.env` 已列入 `.gitignore`,請參考 `.env.example` 建立):

```
MAPS_API_KEY=            # Google Places / Maps
GEMINI_API_KEY=          # (目前未實際使用)
GROQ_API_KEY=            # Groq LLM
PADDLEOCR_ACCESS_TOKEN=  # PaddleOCR REST
```

`MAPS_API_KEY` 另透過 manifestPlaceholder 注入 `AndroidManifest.xml` 供 Places 使用。

---

## 安裝與使用 (Installation & Usage)

> ⚠️ 本節僅涵蓋 **Android App**。賣場視覺定位所用的 AI 模型(GroundingDINO + EasyOCR + Ollama VLM)**不在 App 內**,而是放在獨立的 **[`backend/`](backend/README.md)** 資料夾(獨立的 Python / FastAPI 服務);其安裝、模型下載與啟動請見 **[`backend/README.md`](backend/README.md)**。

### 前置需求
- **Android Studio**(Hedgehog 或更新版)+ **JDK 11**
- **Android SDK**(compileSdk 35)
- 實機或模擬器,**Android 7.0 (API 24) 以上**
- Firebase 專案的 `google-services.json`(放入 `app/`)

### 安裝步驟
1. 取得原始碼:
   ```bash
   git clone https://github.com/B1229013/Shopping-Navigation.git
   cd Shopping-Navigation
   ```
2. **設定金鑰**:於專案根目錄將 `.env.example` 複製為 `.env`,填入 4 把金鑰(見上一節「設定金鑰」)。
3. 放入 Firebase 設定檔 `app/google-services.json`。
4. 以 Android Studio 開啟專案並讓 Gradle 同步;或用命令列建置:
   ```bash
   ./gradlew :app:assembleDebug      # macOS / Linux
   gradlew.bat :app:assembleDebug    # Windows
   ```
5. 安裝到裝置(或在 Android Studio 直接按 ▶ Run):
   ```bash
   ./gradlew :app:installDebug
   ```

### 模型 / 服務安裝(App 端)
本 App 的 AI 能力來自外部模型與服務,**手機端無需自行下載大型模型**,只要設定好金鑰即可:
- **Groq (LLaMA 3.3)** — 雲端呼叫,需 `GROQ_API_KEY`。用於 AI 助理與收據結構化。
- **ML Kit Text Recognition** — 隨 Google Play Services 提供,首次使用會自動下載辨識模組(收據 OCR)。
- **PaddleOCR** — 呼叫自架 REST 服務,需 `PADDLEOCR_ACCESS_TOKEN` 與 `R.string.paddleocr_api_url`(成分 OCR)。
- **Google Places / Firebase** — 需 `MAPS_API_KEY` 與 `app/google-services.json`。

### 使用流程
1. 首次啟動於 `login` 頁註冊,收驗證信完成 **Email 驗證**後登入。
2. 「**清單**」新增購物項目(「首頁」可快速搜尋加入)。
3. 「**分析**」拍攝食品成分 → 自動偵測過敏原並做熱量 / 運動換算。
4. 「**預算**」拍攝收據 → 自動結構化品項並更新預算圓環圖。
5. 「**助理**」以自然語言詢問購物 / 預算 / 健康(情境感知)。
6. 「**賣場導航**」:於 `teammate_home` 確認清單與附近商店後按「開始導航」進入相機 + AR 畫面。**賣場視覺定位需先啟動 [`backend/`](backend/README.md) 後端伺服器**。

---

## 專案結構 (Project Structure)

```
Shopping-Navigation/
├── app/                                  # Android 客戶端
│   └── src/main/java/com/example/shopping/
│       ├── MainActivity.kt               # 單一 Activity + NavHost
│       ├── model/                        # ShoppingItem / DietRecord / UserProfile
│       │                                 #   ShoppingContext / GeminiBudgetResponse
│       └── ui/
│           ├── screens/                  # 各畫面
│           │   ├── MainContainer.kt      # 六分頁容器 + BudgetScreen + Groq/OCR/DonutChart
│           │   ├── HomeScreen.kt         # 首頁
│           │   ├── ShoppingListScreen.kt # 清單
│           │   ├── IngredientsScreen.kt  # 分析(PaddleOCR + 過敏原 + MET)
│           │   ├── HistoryScreen.kt      # 紀錄
│           │   ├── AIScreen.kt           # 助理(Groq + ShoppingContext)
│           │   ├── NavigationScreen.kt   # 賣場導航(CameraX + AR;含 TeammateHomeScreen)
│           │   ├── NearbyStoresSheet.kt / NearbyStoresAr.kt
│           │   ├── LoginScreen.kt / SettingsScreen.kt
│           │   └── ...
│           ├── components/               # CinematicComponents / 互動元件
│           ├── utils/                    # CategoryClassifier / FoodIcons
│           └── theme/
│
├── backend/                              # Python 後端導航伺服器 (FastAPI)
│   ├── server/                           # GroundingDINO + EasyOCR + Ollama VLM + NetworkX
│   ├── eval/ · generate_topomap.py · build_store_*.py · ...
│   └── README.md
│
├── Map/                                  # 賣場地圖資料(執行期)
├── .env.example · build.gradle.kts · settings.gradle.kts · README.md
```

---

## 已知限制

- **賣場視覺定位未完成**:`NavigationScreen` 的逐幀分析為佔位實作,尚待串接 `backend/` 導航伺服器。
- **OCR 光學條件**:光線不足、包裝反光、字體過小會降低 ML Kit / PaddleOCR 成功率。
- **第三方 API 配額**:Groq、Places 受官方頻率與配額限制;需網路連線。
- **過敏原偵測**:以內建字典比對,涵蓋範圍有限,僅供參考非醫療建議。

## 容錯與安全

- **網路容錯**:LLM 呼叫以 `try-catch` 包覆,錯誤時於對話中回報而不致崩潰。
- **金鑰管理**:API 金鑰經 `.env` → `BuildConfig` 注入,`.env` 不入庫。
- **資料本地化**:購物 / 飲食 / 預算 / 個人資料僅存於 `filesDir`。
- **身份屏障**:Email 未驗證帳號自動 `signOut()` 攔截於登入頁面。
