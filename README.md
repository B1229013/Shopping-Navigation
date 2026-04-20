# 智慧購物導覽 (Smart Shopping Navigation)

以 Android 為基礎的綜合型智慧購物助理應用程式,整合電腦視覺、多模態大型語言模型與 OCR 技術,解決大型賣場內找不到商品、繞路、預算失控與飲食健康管理等問題。

---

## 專題簡介 (Project Overview)

「智慧購物導覽」是一款以 **Kotlin + Jetpack Compose** 開發的 Android 應用程式,透過整合人工智慧與機器學習,全面提升消費者的購物體驗與健康管理能力。

- **核心目標**:解決消費者在 IKEA、家樂福等大型賣場中找不到商品、為了買齊需要物品而繞遠路的問題。
- **技術特色**:透過手機鏡頭擷取第一人稱影像,利用多模態模型即時建立拓樸地圖,**無需在賣場額外建置藍牙或 Wi-Fi 定位硬體**。
- **主要應用模組**:
  - **賣場導航** — 影像語意解析 + A\* 演算法 + AR 箭頭引導。
  - **AI 小助理** — 預算管理、健康飲食、諮詢服務與附近門市搜尋。

### 專題資訊
- **指導教授**:吳世琳 教授、陳嶽鵬 教授
- **專題成員**:
  - B1229013 陳宜伶
  - B1229020 何思顗
  - B1229023 林依賢
  - B1222017 周庠

---

## 專題架構 (System Architecture)

### 軟體分層架構(單一 Activity Architecture)

```
┌──────────────────────────────────────────────────────┐
│  UI 表現層 (Presentation Layer)                      │
│  Jetpack Compose + NavHost                           │
│  MainContainer (Global State Holder)                 │
│  └─ shoppingItems / dietRecords / budgetTotalStr     │
├──────────────────────────────────────────────────────┤
│  應用邏輯層 (Application Logic Layer)                │
│  ├─ 建圖與導航:拓樸管理 + A* 路徑規劃                │
│  ├─ AI 助理:Retrofit + Groq API (LLaMA 3.3)         │
│  └─ 視覺處理:CameraX + Paddle OCR                   │
├──────────────────────────────────────────────────────┤
│  資料持久層 (Data Persistence Layer)                 │
│  kotlinx.serialization → 本地 JSON                   │
└──────────────────────────────────────────────────────┘
            ↕ 外部服務整合
  Gemini API │ Groq API │ Firebase Auth │ Google Places
```

### 路由結構 (NavHost)
```
login(起始頁,須通過 Email 驗證)
  └─ main_list
       ├─ teammate_home
       ├─ ar_navigation
       └─ settings
```

### 單向資料流 (UDF)
- `MainContainer` 以 Material 3 **Scaffold** 包覆主畫面,統一管理 `CenterAlignedTopAppBar` 與 `NavigationBar`。
- 內容區依 `selectedTab` 索引渲染子畫面;子畫面透過參數接收 State 與 Callback。
- 透過 `LaunchedEffect` 監聽狀態變化,自動序列化寫回本地檔案。

### 資料結構設計
- **拓樸圖**:以無向圖 `G = (V, E)` 儲存於 `map_cache.json`,包含語意節點、邊與路徑權重。
- **JSON 檔案系統**:`shopping_list.json`、`diet_records.json`、`monthly_budget.txt`。
- **UserProfile**:生理數據(身高、體重、性別)與過敏原清單,用於即時健康比對。

---

## 核心模組功能 (Core Features)

### 1. 導航與視覺定位子系統 (`ArNavigationScreen`)
- **影像串流管理**:CameraX `ImageAnalysis` 以每 3 秒一幀擷取關鍵影格。
- **影像前處理**:OpenCV 執行透視校正、二值化去雜訊與尺寸壓縮(壓縮為 640×640 Bitmap)。
- **語意辨識**:Gemini API 回傳 JSON 格式之區域名稱/編號(如 IKEA 藍底黃字標誌、樑柱編號、區域招牌)。
- **AR 引導呈現**:Compose Canvas 繪製 3D 方向箭頭,結合陀螺儀/加速度計修正視角偏差。

### 2. 路徑規劃模組 (Path Planning)
- **語意匹配**:NLP 將購物清單商品與拓樸地圖節點進行相似度匹配。
- **多目標排序**:Open TSP 安排「入口 → 各目標區域 → 結帳區」的最佳訪問順序。
- **A\* 演算法**:以實際步數權重 + 歐式距離啟發式估算,計算每段節點間最短路徑。
- **動態建圖**:未知地標且相似度低於閾值時,自動建立新節點並連接前一節點,實現「邊走邊記」。

### 3. AI 對話模組 (`AIScreen`)
- **指定模型**:`llama-3.3-70b-versatile`
- **系統角色**:繁體中文購物助理。
- **流程**:使用者輸入 → 封裝 `GroqRequest` → `https://api.groq.com/openai/v1/chat/completions` → 解析 `choices[0].message.content` → 更新 `chatHistory` (Pair<userMsg, aiMsg>)。
- **非同步**:以 `rememberCoroutineScope` 管理協程呼叫。

### 4. 預算收據掃描模組 (`BudgetScreen`)
- `ActivityResultContracts.TakePicture` / `PickVisualMedia` 取得影像。
- `decodeUriToBitmap` → Paddle OCR `TextRecognition` 擷取原始文字。
- Groq API 以 `response_format: json_object` 輸出合法 JSON → 反序列化為 `TidiedReceiptResponse`。
- 各 `line_item` 轉為 `ShoppingItem` (`isChecked = true`) 合併至主購物清單,等同直接記入消費。
- 以 **圓環圖 (DonutChart)** 視覺化預算使用狀況。

### 5. 成分 OCR 與飲食分析模組 (`IngredientsScreen`)
- **影像暫存**:FileProvider 建立 `ocr_scan.jpg`。
- **辨識流程**:`TakePicture` → `InputImage.fromFilePath` → `TextRecognition.getClient` → 去除換行符後填入 `ingredientText`。
- **過敏警示**:與 `UserProfile.allergens` 即時比對,相符時以 `AnimatedVisibility` 顯示紅色警告。
- **熱量與運動換算**:依性別/年齡查表得建議熱量;除以運動 MET 值(慢跑 8.1、走路 3.5、騎腳踏車 5.5)換算所需運動分鐘數。
- **持久化**:封裝為 `DietRecord`,序列化存入 `diet_records.json`。

### 6. 身份驗證模組 (`LoginScreen`)
- 整合 Firebase Authentication,以 `isSignUpMode` 切換登入/註冊模式。
- 登入成功後檢查 `user.isEmailVerified`,未驗證則 `auth.signOut()` 並攔截。
- 註冊成功後自動觸發 `sendEmailVerification`。

### 7. 設定模組 (`SettingsScreen`)
- **個人資料**:性別單選、`DatePickerDialog` 生日、身高/體重、過敏原輸入。
- **儲存空間管理**:`Formatter.formatFileSize` 顯示 `filesDir` 與 `cacheDir` 大小;一鍵 `deleteRecursively` 清除快取。

### 8. 效期預警機制
- 檢查商品條目是否填寫有效期限。
- 僅針對「未食用」且「庫存中」品項計算日期差距並推播提醒。
- 標記為已食用(Consumed)時自動解除監控。

### 9. 附近門市搜尋
- Google Places API (New) 即時檢索周邊門市資訊與距離,協助根據當前位置快速選定採購目標。

---

## 技術堆疊 (Tech Stack)

| 類別 | 技術項目 | 用途說明 |
|------|----------|----------|
| 程式語言 | **Kotlin** | 主要開發語言,支援協程非同步處理 |
| UI 框架 | **Jetpack Compose** | 宣告式介面 (Scaffold / NavHost) |
| 後端認證 | **Firebase Authentication** | 帳號管理與 Email/密碼登入,含 Email 安全驗證 |
| AI 語意對話 | **Groq API + LLaMA 3.3-70b** | AI 對話、過敏原、收據結構化解析 |
| AI 視覺導航 | **Gemini API** (gemini-3-flash) | 實景環境語意解析與標示牌辨識 |
| 電腦視覺 OCR | **Paddle OCR** | 食品成分與收據文字擷取 |
| 相機框架 | **CameraX** (Preview + ImageAnalysis) | AR 導航影像擷取(每 3 秒一幀) + OCR 拍照 |
| 網路層 | **Retrofit 2 + OkHttp + Gson** | REST API 請求與 JSON 解析 |
| 本地儲存 | **JSON (kotlinx.serialization)** | 購物清單、飲食紀錄、預算、拓樸地圖 |
| 圖片載入 | **Coil** | NavigationScreen 相簿圖片顯示 |
| 地點與定位 | **Google Places API (New)** | 附近門市檢索、空間定位與距離運算 |
| 影像前處理 | **OpenCV** | 透視校正、二值化、尺寸壓縮 |

### 執行環境需求
- Android 8.0 (Oreo, **API Level 26**) 以上
- 後置相機(≥ 1200 萬畫素,支援自動對焦)
- GPS 定位模組 + 陀螺儀 + 加速度計
- 穩定 4G/5G 或 Wi-Fi 連線
- 已安裝並更新 Google Play Services

### 開發環境
- Android Studio (Hedgehog 或更新版本),Target API Level 34
- Kotlin 1.9+
- Windows 11 / macOS Sequoia
- Intel Core i7 / Apple M2 以上,RAM ≥ 16GB
- 版本控制:Git / GitHub
- API 測試:Postman 或 Insomnia

---

## 專案結構 (Project Structure)

```
SmartShoppingNavigation/
├── app/
│   ├── src/main/
│   │   ├── java/com/project/smartshopping/
│   │   │   ├── MainActivity.kt              # 單一 Activity 進入點
│   │   │   ├── MainContainer.kt             # 全域 State Holder
│   │   │   │
│   │   │   ├── ui/screens/                  # Composable 畫面
│   │   │   │   ├── LoginScreen.kt           # 登入 / 註冊
│   │   │   │   ├── ShoppingListScreen.kt    # 購物清單
│   │   │   │   ├── BudgetScreen.kt          # 預算 + 收據 OCR
│   │   │   │   ├── IngredientsScreen.kt     # 成分 OCR + 飲食分析
│   │   │   │   ├── AIScreen.kt              # AI 對話助理
│   │   │   │   ├── ArNavigationScreen.kt    # AR 導航
│   │   │   │   ├── NearbyStoreScreen.kt     # 附近門市
│   │   │   │   └── SettingsScreen.kt        # 設定
│   │   │   │
│   │   │   ├── ui/components/               # 共用元件
│   │   │   │   ├── DonutChart.kt            # 預算圓環圖
│   │   │   │   └── AllergenAlert.kt         # 過敏警示
│   │   │   │
│   │   │   ├── navigation/                  # 拓樸導航核心
│   │   │   │   ├── TopologyManager.kt       # 拓樸地圖維護
│   │   │   │   ├── PathPlanner.kt           # A* + Open TSP
│   │   │   │   ├── VisionEngine.kt          # Gemini 語意辨識
│   │   │   │   └── ArOverlayRenderer.kt     # AR 箭頭繪製
│   │   │   │
│   │   │   ├── ocr/                         # OCR 模組
│   │   │   │   ├── PaddleOcrProcessor.kt
│   │   │   │   └── ImagePreprocessor.kt     # OpenCV 前處理
│   │   │   │
│   │   │   ├── network/                     # API 層
│   │   │   │   ├── GroqApiService.kt        # Retrofit 介面
│   │   │   │   ├── GeminiApiService.kt
│   │   │   │   ├── PlacesApiService.kt
│   │   │   │   └── dto/                     # 請求/回應 DTO
│   │   │   │       ├── GroqRequest.kt
│   │   │   │       └── TidiedReceiptResponse.kt
│   │   │   │
│   │   │   ├── auth/
│   │   │   │   └── FirebaseAuthManager.kt   # Firebase 驗證
│   │   │   │
│   │   │   ├── data/
│   │   │   │   ├── model/                   # 資料類別
│   │   │   │   │   ├── ShoppingItem.kt
│   │   │   │   │   ├── DietRecord.kt
│   │   │   │   │   ├── UserProfile.kt
│   │   │   │   │   └── TopologyNode.kt
│   │   │   │   └── repository/              # 本地 JSON 讀寫
│   │   │   │       ├── ShoppingRepository.kt
│   │   │   │       ├── DietRepository.kt
│   │   │   │       └── MapCacheRepository.kt
│   │   │   │
│   │   │   └── utils/
│   │   │       ├── MetCalculator.kt         # 運動熱量換算
│   │   │       └── ExpiryChecker.kt         # 效期偵測
│   │   │
│   │   ├── res/                             # 資源檔(layouts, drawables, values)
│   │   └── AndroidManifest.xml
│   │
│   └── build.gradle.kts
│
├── files/                                   # 本地持久化 (執行期產生於 filesDir)
│   ├── shopping_list.json                   # 購物清單
│   ├── diet_records.json                    # 飲食紀錄
│   ├── monthly_budget.txt                   # 月預算總額
│   └── map_cache.json                       # 拓樸地圖
│
├── cache/
│   └── ocr_scan.jpg                         # OCR 暫存影像
│
├── docs/
│   └── 設計文件書.docx                      # 系統設計文件
│
├── google-services.json                     # Firebase 設定
├── build.gradle.kts
├── settings.gradle.kts
└── README.md
```

---

## 已知限制

- **光學條件**:光線不足、包裝反光、字體過小會降低 OCR 與視覺特徵擷取成功率。
- **動態遮蔽**:人潮擁擠、大型推車遮蔽會造成定位延遲或遺失。
- **特徵匱乏**:倉儲區無標示紙箱或相似鋼架可能產生節點誤判。
- **API 配額**:Groq / Gemini 受官方頻率與配額限制。
- **硬體效能**:舊款裝置可能因算力/記憶體不足強制關閉;長時間使用會耗電與過熱。
- **導航迷失處理**:連續無法取得有效語意特徵時,系統暫停更新箭頭並提示使用者抬高手機環視標示。

## 容錯與安全

- **網路容錯**:`try-catch` 捕捉 `IOException` / `HttpException`,斷線時仍可操作已載入的本地資料。
- **OCR 修正**:辨識失敗時提供手動編輯欄位,再寫入 JSON。
- **協程調度**:所有耗時操作限定在 `Dispatcher.IO` / `Default`,避免阻塞主線程。
- **資料本地化**:敏感生理數據僅儲存於 `filesDir`,不上傳外部平台。
- **身份屏障**:Email 未驗證帳號自動 `signOut()` 攔截於登入頁面。
