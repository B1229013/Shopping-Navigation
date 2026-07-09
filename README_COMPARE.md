# VLM × OCR 比較工具使用說明（`compare.html`）

透過長庚大學（CGU）OpenAI 相容 gateway，用**一把金鑰**同時比較多個視覺語言模型（VLM）與文字辨識（OCR）引擎。VLM 與 OCR **可獨立選擇**，並提供兩種模式：**批次比較**（整個資料夾）與**互動式導航**（逐步對話）。

工具是單一 HTML 檔，直接用瀏覽器開啟即可，金鑰只存在瀏覽器 localStorage。

---

## 1. 前置設定

### 1-1. 設定 CGU 金鑰（後端與工具共用）

在專案根目錄 `.env` 填入你的金鑰（此檔已被 `.gitignore` 忽略，**不會**進 git）：

```env
CGU_API_KEY=你的金鑰
CGU_BASE_URL=https://air.cgu.edu.tw/cgullmapi/v1
VLM_BACKEND=openai
OPENAI_MODEL=gpt-4o
```

> 後端會自動把 `CGU_API_KEY` 當成 `OPENAI_API_KEY` 使用（見 `backend/server/config.py`），所以正式後端與比較工具只需維護這一把金鑰。

### 1-2. 驗證 gateway 連線（可選）

```powershell
cd backend
python verify_gateway.py          # 需要 Python + openai 套件
# 或（免 Python）
powershell -File verify_gateway.ps1
```

會列出 gateway 上所有可用模型，並測試一個 OpenAI 模型與一個本地模型是否都能回應。

---

## 2. 開啟比較工具

直接用瀏覽器開啟根目錄的 `compare.html`（雙擊即可）。

1. 於 **CGU Gateway** 欄位貼上金鑰 → 按 **Load models**（會呼叫 `/v1/models` 動態載入所有模型）。
2. 為 **Slot A** 與 **Slot B** 各自選擇 **VLM** 與 **OCR**（兩者獨立）。
3. 在 **Item to find** 輸入要找的物品（例如 `milk`、`牛奶`、`bread`）。
4. 選擇模式並執行（見下方）。

---

## 3. 兩種模式

### 3-1. 批次模式（Batch — 整個資料夾）

適合對一批照片一次跑完、比較整體表現。

1. 上方 **Mode** 選 **Batch**。
2. **Choose folder…** 選照片資料夾（例如 `測試路線1照片`）。
3. 按 **Run comparison**。
4. 每張照片顯示 A / B 兩欄結果；上方 **Summary** 統計每個 Slot 的：可解析 JSON %、繁體中文輸出 %、平均延遲、動作一致率。
5. 按 **Save report (.html)** 匯出靜態報告。

### 3-2. 互動式模式（Interactive — 逐步導航）

模擬真實 App 的一步一步導航，**具對話記憶**。

1. 上方 **Mode** 選 **Interactive**。
2. **Attach one photo…** 附上一張照片（例如入口）。
3. 在訊息框輸入當前狀況（例如「我在入口」）。
4. 按 **Send step** → 每個 Slot 先跑該張照片的 OCR，再把文字注入 prompt，VLM 依**先前所有步驟的記憶**回覆該往哪走。
5. 走到下一個位置後，再附上新照片、輸入「我已照指示移動」，再按 **Send step**。
6. 需要重新開始一段導航時按 **Reset session** 清除記憶。

> 只有「最新一張照片」會送出，先前步驟以文字保留於對話歷史中 —— 與正式後端行為一致（當前照片 + 文字化路徑歷史）。

---

## 4. 模型分類與自動路由

Gateway 上的模型依 **id 是否含冒號 `:`** 決定送出格式，工具會**自動處理**：

| 類型 | 判斷 | 範例 | 送出格式 |
|------|------|------|----------|
| OpenAI 類 | id **無冒號** | `gpt-4o`, `gpt-5.4`, `gpt-5.4-mini` | OpenAI content 陣列（圖片用 `image_url`） |
| 本地 Ollama 類 | id **有冒號** | `deepseek-ocr:latest`, `glm-ocr:latest`, `gpt-oss:20b` | Ollama 原生格式（`content` 字串 + `images` 陣列） |
| 本地 Ollama VLM（直連） | 下拉選 Ollama 群組 | `llava:7b`, `llama3.2-vision` | 直接呼叫 `localhost:11434/api/chat` |
| EasyOCR（後端） | OCR 選 EasyOCR | — | 呼叫後端 `POST /ocr` |

**gpt-5.x** 需要 `max_completion_tokens`（不能用 `max_tokens`），工具也已自動切換。

### VLM 選項
- **本地 Ollama**：`llava:7b`（可用）、`llama3.2-vision`（見注意事項）。
- **CGU gateway chat 模型**：`gpt-4o`、`gpt-5.4` 等（速度快，建議互動式用）。

### OCR 選項
- **(no OCR)**：跳過 OCR，只跑 VLM。
- **EasyOCR（本地後端）**：需啟動後端（見第 5 節）。
- **CGU 專用 OCR**：`deepseek-ocr:latest`、`glm-ocr:latest`（慢，但專為 OCR）。
- **CGU vision-LLM 當 OCR**：`gpt-4o` 等（最快，約 2–3 秒；OCR 模型不可用時的替代）。

---

## 5. 啟用 EasyOCR（需要後端）

EasyOCR 在 Python 後端執行，工具透過新增的 `POST /ocr` 端點呼叫它。**需先啟動後端**：

```powershell
cd backend
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m uvicorn server.server:app --host 0.0.0.0 --port 8000
```

啟動後，`compare.html` 的 **EasyOCR backend URL** 保持 `http://localhost:8000` 即可。若後端未啟動，EasyOCR 選項會出現 `Failed to fetch`。

> 本端點沿用既有的 `OCR` 類別與設定（`OCR_LANGUAGES=en,ch_tra` 等），結果與正式導航流程一致。

---

## 6. 注意事項與已知問題

| 情況 | 說明 / 解法 |
|------|-------------|
| `llama3.2-vision` 無法載入（mllama 錯誤） | 目前安裝的 Ollama 版本不支援該架構；請改用 `llava:7b`。 |
| `deepseek-ocr` / `glm-ocr` 很慢或回 502 | 這兩個模型由 CGU 端的 Ollama 服務，首次載入慢（deepseek 約 110 秒、glm 約 31 秒），偶爾 upstream 502；急用時改選 `gpt-4o` 當 OCR。 |
| OCR 顯示 `Failed to fetch`（EasyOCR） | 後端未啟動，見第 5 節。 |
| VLM 用 `llava:7b` 很慢（約 23 秒/張） | CPU 推論所致；互動式建議改用 gateway 的 `gpt-4o`（約 2–3 秒）。 |
| Ollama 直連失敗 / CORS | 啟動 Ollama 時設 `OLLAMA_ORIGINS=*`。 |
| 新模型沒出現在下拉選單 | 重新按 **Load models**（`/v1/models` 動態載入，新模型會自動出現）。 |

---

## 7. 相關檔案

| 檔案 | 功能 |
|------|------|
| `compare.html` | 比較工具（批次 + 互動式），純瀏覽器執行 |
| `backend/server/llm_gateway.py` | 共用 gateway client：`chat` / `ocr` / `embed` / `run_on_image` / `list_models`，依模型類型自動路由 |
| `backend/server/config.py` | 新增 `CGU_API_KEY` / `CGU_BASE_URL`，並讓 `OPENAI_*` 自動沿用 |
| `backend/server/server.py` | 新增 `POST /ocr` 端點（包裝既有 EasyOCR）與 CORS |
| `backend/server/perception.py` | `ocr_on_detection()`：將偵測框裁切後送 gateway OCR/chat 比較 |
| `backend/verify_gateway.py` / `.ps1` | gateway 連線煙霧測試 |
| `.env` | CGU 金鑰（**已被 gitignore，勿提交**） |
| `.env.example` | 環境變數範例（不含真實金鑰） |
