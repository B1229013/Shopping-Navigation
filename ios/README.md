# 智慧購物導航 APPNAV_ios

一個 iOS 購物助理 App，整合預算管理、食材/購物清單、AI 助手，以及核心亮點功能——**賣場室內導航**（拍照 + 感測器融合 + VLM 語意推論，逐步指引使用者走到目標商品前）。

本 repo 包含兩個部分：

```
APPNAV_ios/
├── APPNAV_ios/          # iOS App（SwiftUI）
└── backend/             # Python 導航後端（FastAPI）
```

---

## 功能總覽

| 功能 | 說明 | 主要程式碼 |
|------|------|-----------|
| 登入 | 帳號登入/驗證 | `Views/Login/LoginView.swift`、`Services/AuthService.swift` |
| 首頁 | App 主要入口/總覽 | `Views/Main/HomeView.swift` |
| 預算管理 | 記帳、收據 OCR 辨識金額 | `Views/Main/BudgetView.swift`、`Services/OCRService.swift` |
| 食材管理 | 食材庫存/分類 | `Views/Main/IngredientsView.swift`、`Utils/CategoryClassifier.swift` |
| 購物清單 | 待買清單管理 | `Views/Main/ShoppingListView.swift`、`Models/ShoppingItem.swift` |
| 歷史紀錄 | 過去消費/購物紀錄 | `Views/Main/HistoryView.swift` |
| AI 助手 | 對話式 AI 功能（Groq） | `Views/Main/AIView.swift`、`Services/GroqService.swift` |
| **賣場導航** | 拍照上傳給後端 VLM，取得 MOVE / ASK / ARRIVED 導航指令；同時用手機感測器做室內路徑估計、偵測原地繞圈 | `Views/Navigation/`、`Navigation/`、`Services/NavigationAPI.swift` |
| 設定 | App 設定頁 | `Views/Settings/SettingsView.swift` |

---

## 架構

```
┌──────────────────────────────┐         ┌──────────────────────────────────────────┐
│  iOS App（本 repo）            │  HTTP   │  Python 後端導航伺服器（backend/）          │
│  AVFoundation 拍照 + 手電筒     │ ──────► │  FastAPI                                    │
│  CoreMotion 步數/姿態融合定位   │ ◄────── │  GroundingDINO + EasyOCR + VLM（多模態）    │
│  導航 UI / 上一張照片縮圖       │  JSON   │  NetworkX 拓樸地圖 + 目標拆解 + 路徑決策     │
└──────────────────────────────┘         └──────────────────────────────────────────┘
```

- 導航期間 App 會持續用 `CMPedometer`（步數/距離）+ `CMDeviceMotion`（融合姿態，取得 heading）估計使用者在賣場內的相對位置，並偵測「原地繞圈」提醒使用者。細節見 `APPNAV_ios/Navigation/MotionPathTracker.swift` 開頭的說明註解。
- 每次拍照會上傳到後端 `POST /session/{id}/photo`，後端結合物件偵測 + OCR + VLM 推論後回傳下一步指令（`MOVE`/`ASK`/`ARRIVED`）。後端細節與 API 文件見 [`backend/README.md`](backend/README.md)。

---

## 開發環境需求

- **Xcode**（建議最新穩定版），Deployment target 見專案設定
- iOS 實機或模擬器（相機/感測器相關功能建議用**實機**測試，模擬器沒有真的加速度計/陀螺儀）
- 一組 Firebase 專案（`GoogleService-Info.plist`，登入用 Firebase Auth）
- 後端（`backend/`）另需 Python 3.12，詳見其 README

---

## 設定與啟動（iOS App）

### 1. Firebase

專案已內附 `APPNAV_ios/GoogleService-Info.plist`。若要接自己的 Firebase 專案，去 [Firebase Console](https://console.firebase.google.com/) 建立專案、下載對應的 `GoogleService-Info.plist` 覆蓋掉即可。

### 2. API Key / 後端網址設定

App 內的第三方金鑰與後端網址**不是寫死在程式碼裡**，而是透過 Xcode Build Settings 注入到 `Info.plist`（見 `APPNAV_ios/Utils/AppConfig.swift`），對應的 build setting 變數如下：

| 變數 | 用途 | 目前狀態 |
|------|------|----------|
| `NAVIGATION_BACKEND_URL` | 導航後端網址 | 專案設定目前指向開發機的區網位址 `http://172.20.10.3:8000`，**請依你自己執行 `backend/` 的機器 IP 修改**（Xcode → 專案 target → Build Settings 搜尋 `NAVIGATION_BACKEND_URL`） |
| `GROQ_API_KEY` | AI 助手（Groq）金鑰 | 未預設值，需自行申請並填入 |
| `PADDLEOCR_API_URL` / `PADDLEOCR_ACCESS_TOKEN` | 收據 OCR 服務 | 未預設值，需自行申請並填入 |
| `MAPS_API_KEY` | 地圖相關功能 | 未預設值，需自行申請並填入 |

> **注意**：這些是 Xcode Build Settings，不是這個 repo 裡的檔案，所以不會意外把金鑰提交進 git。修改方式：Xcode 打開專案 → 點選 `APPNAV_ios` target → **Build Settings** → 搜尋對應變數名稱 → 填入你自己的值（或改用 `.xcconfig` 檔案管理，並把該檔加進 `.gitignore`）。

### 3. Build & Run

用 Xcode 開啟 `APPNAV_ios.xcodeproj`，選擇實機或模擬器，`Cmd+R` 執行即可。第一次開啟會自動解析 Swift Package 相依（Firebase 系列套件）。

也可以用命令列驗證是否能成功建置：
```bash
xcodebuild -project APPNAV_ios.xcodeproj -scheme APPNAV_ios \
  -destination 'generic/platform=iOS Simulator' build
```

---

## 啟動導航後端（backend/）

導航功能需要 `backend/` 這支 Python 伺服器在背景跑著，App 才有東西可以打。快速啟動：

```bash
cd backend
pip install -r requirements-server.txt
cp ../.env.example ../.env   # 若尚未建立 .env，填入你自己的金鑰
python -m server.run_server
```

`.env` 放在 repo 根目錄（跟這份 README 同一層），內容範例見 `.env.example`。**`.env` 已加入 `.gitignore`，裡面的金鑰不會被提交到 git，請勿手動移除這條規則。**

完整的後端架構、API 端點、環境變數說明、建圖流程，請見 [`backend/README.md`](backend/README.md)。

---

## 測試

```bash
# iOS（需要 Xcode 命令列工具）
xcodebuild -project APPNAV_ios.xcodeproj -scheme APPNAV_ios \
  -destination 'generic/platform=iOS Simulator' test

# 後端（於 backend/ 目錄下）
cd backend && pytest
```
