# 智慧購物導航 — 後端導航伺服器 (Navigation Backend Server)

本目錄為「智慧購物導航」系統的 **Python 後端導航伺服器**，對應設計文件書 §3.2「軟硬體組件設計說明」。負責整個賣場導航的核心運算：**視覺感知 → 拓樸定位 → 多模態語意推論 → 導航決策**。Android App 為前端薄客戶端，僅負責拍照與顯示，所有 AI 運算皆在本伺服器執行。

> **隱私設計**：視覺語言模型透過本地端 **Ollama** 運行，顧客上傳的照片不送往任何雲端 API，亦無任何雲端金鑰。

---

## 系統定位（在整體架構中的角色）

```
┌─────────────────────────────┐         ┌──────────────────────────────────────────┐
│  Android App（客戶端）       │  HTTP   │  Python 後端導航伺服器（本目錄）            │
│  CameraX 擷取 / OpenCV 前處理 │ ──────► │  FastAPI                                    │
│  AR 箭頭與文字導引顯示        │ ◄────── │  GroundingDINO + EasyOCR + Ollama VLM       │
│                             │  JSON   │  NetworkX 拓樸地圖 + 目標拆解 + 路徑決策     │
└─────────────────────────────┘         └──────────────────────────────────────────┘
```

- **測試階段（每間賣場執行一次）**：以所有賣場環境照片一次性預建拓樸地圖，供後續所有顧客 session 重複使用。
- **執行階段（每位顧客）**：顧客每步上傳一張照片，伺服器回傳逐步導航指令（MOVE / ASK / ARRIVED）。

---

## 核心模組

| 模組 | 說明 | 主要檔案 |
|------|------|----------|
| 視覺感知 — 物件偵測 | GroundingDINO SwinT-OGC，零樣本偵測貨架商品、冰箱、門口、走道、標示物等地標 | `server/perception.py` |
| 視覺感知 — 文字萃取 | EasyOCR 讀取走道招牌、區域名稱與貨架標籤（賣場最強的區域辨識訊號） | `server/ocr.py` |
| 拓樸地圖與定位 | NetworkX 無向圖，記錄「使用者在哪裡」與「走過哪裡」，動態建立節點與邊 | `server/topomap.py` |
| 決策與大腦推理 (VLM) | 本地 Ollama 上的 LLaMA 3.2 Vision 11B，整合影像、偵測結果、OCR 與地圖摘要，輸出導航動作 | `server/vlm.py` |
| 目標拆解 | 將自然語言目標（如「找牛奶」）拆解為可偵測的視覺地標清單（如 `["milk", "dairy", "fridge"]`） | `server/goal_decomposer.py` |
| 場景整合 / 抵達驗證 | 整理偵測與 OCR 摘要、比對招牌與目標、把關 ARRIVED 防止誤判抵達 | `server/scene.py` |
| 影像標註 | 在照片上繪製偵測框與 OCR 區域，回傳標註後影像 | `server/annotator.py` |
| Session 狀態 | 每場導航 session 的目標、歷史、目前節點等狀態 | `server/session.py` |
| API 服務 | FastAPI REST 介面 | `server/server.py`、`server/run_server.py` |

**決策輸出**（`server/vlm.py`）標準化為三種導航動作之一：
- **MOVE** — 提供往前或轉彎的具體方向。
- **ASK** — 遇到岔路或不確定時向使用者提問。
- **ARRIVED** — 已抵達目標（需通過偵測分數 / OCR 招牌 / 裁切複驗等把關才成立）。

---

## API 端點

| 動作 | 端點 |
|------|------|
| 建立 session（輸入商品目標） | `POST /session` — body：`{"goal": "find milk"}` → 回傳 `session_id` |
| 上傳一張照片取得導航 | `POST /session/{id}/photo` — multipart 影像 → 回傳 action + guidance + 標註影像 URL |
| 回答 ASK 問題 | `POST /session/{id}/answer` — body：`{"answer": "yes"}` |
| 取得目前拓樸地圖 | `GET /session/{id}/map?format=json\|png` |
| 取得 session 狀態 | `GET /session/{id}` |
| 取得標註後影像 | `GET /session/{id}/photo/{node_id}.jpg` |
| 健康檢查 | `GET /health` |

---

## 環境需求

- **Python 3.12**
- **伺服器硬體**（設計文件書 §3.2）：8 GB VRAM、16+ 執行緒 CPU。
- **Ollama** 本地服務並已拉取 `llama3.2-vision` 模型。
- **GroundingDINO 權重**：`models/groundingdino_swint_ogc.pth`（請另行下載，勿入庫）。

---

## 安裝與啟動

```bash
# 1) 安裝伺服器執行階段相依套件
pip install -r requirements-server.txt

# 2) 安裝 Ollama（https://ollama.com）並啟動服務，拉取視覺模型
ollama pull llama3.2-vision        # 約 7–8 GB，首次拉取較久
ollama serve                       # 預設監聽 127.0.0.1:11434

# 3) 下載 GroundingDINO 權重（zero-shot 物件偵測模型），放到 models/groundingdino_swint_ogc.pth
#    權重來源：https://github.com/IDEA-Research/GroundingDINO（檔名 groundingdino_swint_ogc.pth）
#    並 clone GroundingDINO 以取得設定檔 GroundingDINO_SwinT_OGC.py（路徑由 server/config.py 自動偵測）
#    EasyOCR 模型會於首次執行時自動下載，無需手動處理

# 4) 啟動伺服器（於本 backend/ 目錄下執行）
python -m server.run_server
# 或
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

> 建圖管線（下節）另需 ML 相依套件：`torch`、`groundingdino`、`matplotlib`、`pyvis`、`pillow`、`pillow-heif` 等；建議於具 GPU 的環境（如 WSL conda 環境）執行。

---

## 使用方式 (Usage)

伺服器啟動後，一次顧客導航流程如下（以 `curl` 為例）：

```bash
# 1) 建立 session，輸入要找的商品
curl -X POST http://localhost:8000/session \
  -H "Content-Type: application/json" \
  -d '{"goal": "find milk"}'
# → { "session_id": "...", "guidance": "...", "goal_objects": ["milk","dairy",...] }

# 2) 每走一步上傳一張照片，取得導航動作
curl -X POST http://localhost:8000/session/<session_id>/photo \
  -F "photo=@step1.jpg"
# → { "action": "MOVE|ASK|ARRIVED", "guidance": "...", "question": "...",
#     "annotated_photo_url": "/session/<id>/photo/0.jpg" }

# 3) 若 action 為 ASK，回答後繼續
curl -X POST http://localhost:8000/session/<session_id>/answer \
  -H "Content-Type: application/json" \
  -d '{"answer": "yes"}'
```

- `action = MOVE` → 顯示 `guidance` 指引顧客移動，再拍下一張照片。
- `action = ASK` → 顯示 `question`，由顧客回答（`/answer`）。
- `action = ARRIVED` → 已抵達目標，session 結束。

> Android App 端由 `NavigationScreen` 對應呼叫上述端點（目前 App 的 `processImageForModel()` 仍為佔位實作，待串接此後端）。

---

## 建立新地圖（Neo4j V2 格式）

和 Neo4j 裡的「A7家樂福（9/16)」相同格式（`Waypoint → DirPhoto → Object`，加上 `WALKWAY`、`ADJACENT`），每間店做一次。共用邏輯與格式說明在 `server/map_v2.py`。

| 步驟 | 工具 | 產出 |
|------|------|------|
| 1. 現場收資料 | Android App → Sensor Lab：從入口開始錄，每個點位用「四方向拍照」拍前、右、後、左；分好幾趟走，每趟匯出一個 zip | `sensor_log.csv` + `photos/` |
| 2. 匯入路線 | 建圖頁的「店家」選單 →「＋ 新增店家…」，填店名、勾選 Sensor Lab 資料後按「重建地圖」；或 `python build_map_v2_topomap.py --place "<店名>" walk1.zip walk2.zip …` | 該店的工作資料夾：`01/`、`02/`…照片 + `topomap.json` |
| 3. 對齊、合併 | 啟動後端，打開 `http://localhost:8000/topomap-builder?place=<店名>`（不給 `place` 會開最近編輯的店）：對齊、合併後按「確認並儲存」 | 該店的 `topomap_confirmed.json` |
| 4. 上傳 | `python upload_map_v2.py --place "<店名>" --dry-run` 先檢查，沒問題再拿掉 `--dry-run` | `Waypoint`、`HAS_PHOTO`、`DirPhoto`、`WALKWAY` |
| 5. 產生物件 | `python rebuild_objects_v2.py --place "<店名>" --limit 2 --dry-run` 試跑，再完整執行 | `Object`、`DETECTED`、`ADJACENT` |
| 6. 設定 | 寫 `maps/neo4j_links/<店名>.json`（入口與補的連線，參考 `a7_carrefour.json`）；App 從 `/places` 選店 | |

- **每間店一個工作資料夾**：`<MAP_BUILD_ROOT>/<店名>/`（預設 `backend/output/mapbuild/<店名>/`；店名裡 Windows 不允許的字元會換成 `_`，例如「A7家樂福（9/16)」→「A7家樂福（9_16)」）。裡面有 `topomap.json`、`topomap_confirmed.json`、各路線照片、`dino_embeddings.pkl`、`floorplan.*`、`imports/`、`vlm_cache/`。建圖頁用頂端的「店家」選單切換，三支腳本依 `--place` 找資料夾（`--workspace` 可以另外指定）。舊的環境變數 `MAP_BUILD_DIR` 現在代表這個總資料夾。
- **參考底圖**：設定 `MAP_BUILD_REF_PLACE=<Neo4j place>`，建圖頁就會把那張地圖畫在底下（例如同一間店的舊地圖）。
- **路線…**（建圖頁頂端）：
  - 列出來源資料夾裡所有 Sensor Lab 匯出，也就是含 `sensor_log.csv` 的資料夾或 zip。來源資料夾預設是 `backend/legacy/testPDR`、`backend/testPDR`，可用 `MAP_BUILD_SOURCES` 設定；也可以直接上傳 zip（存到 `<工作資料夾>/imports/`）。
  - 勾選路線後按「重建地圖」，效果等同 `build_map_v2_topomap.py`。已在用的路線保留原本編號，新路線依錄製時間接在後面，編號不會重複使用。
- **CSV 軌跡**（頂端選單）：畫出 `sensor_log.csv` 每一步的 PDR 路線。「跟著節點」會隨對齊、合併一起變形；「原始座標」顯示 CSV 原本算出的結果。滑鼠移到路線圖例上，可以看到步數、距離、時間。
- **DINO 自動調整**（「對齊工具」最上面）：按「分析照片」後，會比對每個節點的四方向照片（DINOv2，向量快取在 `<工作資料夾>/dino_embeddings.pkl`），列出三類建議：
  - 閉環：同一趟走回原地。
  - 重疊：不同趟的同一個位置。
  - 不確定：預設不勾，按「看」可以並排比對兩個點的照片。

  按「套用勾選」會依序閉環合併（分攤誤差）、把其他路線最小平方旋轉加平移對上第一條、合併重疊點；「復原」一鍵還原。
- **合併**：點一個節點，按「合併：選另一個點…」，再點另一個節點。同一趟會自動用閉環合併；較早或第一條路線的點會保留位置。
- **插入路線**：節點站在另一條路線兩個節點之間的走道上時，點它，按「插入路線：選一條邊…」，再點那條邊。邊 a—b 會變成 a—這點—b，節點移到線上，照片保留在自己身上。DINO 分析也會自動找出這種情況，列在「在路線上」：最像的兩個點剛好是另一條路線上相鄰的節點。套用時會在對齊與合併之後插入，若離那條邊超過 5 m 則跳過。
- **平面圖對齊**（建圖頁右上角的「對齊工具」）：
  - 載入平面圖：圖檔會存成 `<工作資料夾>/floorplan.*`。可調整中心、寬度（公尺）、旋轉、透明度，也可以直接拖曳移動。
  - 整體調整節點：對象可選全部節點或某一趟。可以平移，或繞原點、對象中心、選取的節點旋轉；照片的拍攝方向會跟著轉。也可以勾選「拖曳任一節點時整體一起移動」。
  - 「復原移動」可以一步一步撤銷。
  - 編輯進度會自動存在瀏覽器裡，重新整理不會不見；「重設全部」會回到原始資料。
- **VLM**：使用 `.env` 的 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`。提示詞與 A7 相同，每張照片預設呼叫 3 次再去重（`--calls`）。結果快取在 `<工作資料夾>/vlm_cache/`，中斷後重跑不會重複付費。
- **OCR**：使用 EasyOCR，文字會掛到中心點落在其框內的物件上（`ocr_text`）。第一次載入模型較慢，可用 `--no-ocr` 略過。
- **安全**：本工具建立的 `DirPhoto` / `Object` 帶有 `place` 欄位。`upload_map_v2.py --replace` 與 `rebuild_objects_v2.py` 只會動這種 place，同學建的 A7 一律拒絕寫入。
- **還沒做**：新店照片的 DINOv2 sidecar（`dino_sidecar.pkl` 目前只有 A7）。少了它，新店定位會退回九宮格比對。

## 舊版建圖管線（legacy）

以賣場照片一次性建立拓樸地圖，依序執行：

| 步驟 | 腳本 | 功能 |
|------|------|------|
| OCR | `ocr_dump.py` | 對所有賣場照片跑 EasyOCR，輸出 `store_ocr.json` |
| 偵測 + 富化 | `build_store_map.py` | 將 OCR 招牌類別併入 GroundingDINO 偵測結果作為標準化標籤 |
| 建圖（序列） | `generate_topomap.py` | Jaccard 分群成區域、建立節點與邊、輸出拓樸地圖（PNG/HTML） |
| 建圖（分支樓面） | `build_store_graph.py` | 以主走道 + 分支走道（1–13）+ 周界生鮮區的真實賣場拓樸建模 |
| 精簡渲染 | `render_clean.py` | 蛇形版面、每區一張卡片的精簡可讀地圖 |
| 互動式 HTML | `build_html_map.py` | Pyvis 可拖曳節點的互動式 HTML 地圖，便於核對實際樓面 |

```bash
python generate_topomap.py --sim-threshold 0.40
```

---

## 測試

```bash
# 單元測試（離線、不打網路；設定 UNIGOAL_TEST_MODE）
pytest

# 評估 / 回放 harness
#   見 eval/ 目錄（runner.py、replay.py、metrics.py）
```

---

## 設定（環境變數，詳見 `server/config.py`）

| 變數 | 預設 | 說明 |
|------|------|------|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 服務位址 |
| `OLLAMA_MODEL` | `llama3.2-vision` | 視覺語言模型名稱 |
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `8000` | 伺服器位址與埠 |
| `OCR_ENABLED` | `1` | 是否啟用 EasyOCR |
| `OCR_LANGUAGES` | `en,ch_tra` | OCR 語言（英文 + 繁體中文） |
| `GROUNDINGDINO_BOX_THRESHOLD` | `0.40` | 偵測框信心閾值（針對忙碌貨架調高以降低誤判） |
| `GOAL_CROP_VERIFY` | `1` | ARRIVED 前裁切目標框交由 VLM 複驗 |
| `ARRIVED_MIN_DETECTION_SCORE` | `0.35` | ARRIVED 成立所需的最低偵測佐證分數 |

---

## 目錄結構

```
backend/
├── server/                     # FastAPI 後端核心
│   ├── server.py               # REST API 端點
│   ├── run_server.py           # uvicorn 啟動點
│   ├── perception.py           # GroundingDINO 物件偵測
│   ├── ocr.py                  # EasyOCR 文字辨識
│   ├── vlm.py                  # Ollama VLM 導航決策（MOVE/ASK/ARRIVED）
│   ├── goal_decomposer.py      # 目標 → 視覺地標清單
│   ├── topomap.py              # NetworkX 拓樸地圖
│   ├── scene.py                # 場景整合與抵達驗證
│   ├── annotator.py            # 偵測框 / OCR 標註
│   ├── session.py              # session 狀態
│   ├── batch_mapper.py         # 批次照片處理 + Jaccard 分群
│   ├── config.py               # 閾值與路徑設定
│   ├── prompts.py / models.py / store_knowledge.py / store_map.py / navigator.py
│   └── tests/                  # pytest 單元測試
├── eval/                       # 評估 / 回放 harness
├── generate_topomap.py         # 拓樸地圖生成（建圖主入口）
├── build_store_map.py          # OCR 富化 + 建圖
├── build_store_graph.py        # 分支樓面拓樸建模
├── build_html_map.py           # Pyvis 互動式 HTML 地圖
├── render_clean.py             # 精簡地圖渲染
├── ocr_dump.py                 # 批次 OCR
└── requirements-server.txt     # 伺服器執行階段相依套件
```
