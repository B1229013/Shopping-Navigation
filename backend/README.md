# 共用導航後端（backend）

iOS（B1222017/APPNAV_ios）與 Android（本倉庫 `app/`）共用的 **Python FastAPI 導航伺服器**。
使用者每拍一張照片，伺服器就做：**照片品質檢查 → VLM 感知 → 在店家地圖上定位 → 規劃路線 → VLM 決策**，
回傳 MOVE / ASK / ARRIVED 與中文指引。店家地圖存在 **Neo4j Aura**，VLM 走 **OpenAI 相容 API**
（`OPENAI_BASE_URL` 可指向 OpenAI 或其他相容閘道）。

> 照片會送到 `OPENAI_BASE_URL` 指定的 VLM 服務；物件名稱的語意向量也會呼叫同一個服務的 embedding。

---

## 架構

```
手機（iOS / Android）
  │  POST /session/{id}/photo：照片 ＋ PDR 位置／航向
  ▼
server.py ─┬─ photo_quality      模糊／太暗／遮擋 → 請使用者重拍
           ├─ vlm（感知）          物件、商品名、OCR 文字、店內區域（每張 1 輪：2 次呼叫，VLM_PASS3=1 時 3 次）
           ├─ 定位
           │   ├─ locate_fusion   伺服器端 DINOv2 ＋ 物件名稱 ＋ OCR（權重 1 : 0.1 : 0.3）
           │   ├─ visual_localization  物件／OCR 與地圖比對（中英名稱、語意向量、稀有度、九宮格）
           │   ├─ aisle_signs     走道吊牌號碼縮小範圍
           │   └─ loc_fusion      影像證據 × 手機感測器位置 → 機率排名
           ├─ editor_map / path_planner   在 Neo4j 拓樸圖上規劃路線、轉彎清單、小地圖
           ├─ heading            估計朝向，產生左轉／右轉
           └─ vlm（決策）          照片 ＋ 路線 → MOVE / ASK / ARRIVED
  ▲
  │  Neo4j：Waypoint、DirPhoto（四方向照片、DINOv2 向量）、Object、WALKWAY、ADJACENT
  │  maps/：每間店的起點與補充走道邊（neo4j_links）、起點照片（starts）
```

---

## 安裝與啟動

需求：Python 3.11、Neo4j Aura 帳號、OpenAI 相容 API 金鑰。

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-server.txt
pip install numpy pillow torch transformers   # 伺服器端 DINOv2（facebook/dinov2-base）
.\start.ps1                                    # = uvicorn server.server:app --host 0.0.0.0 --port 8000
```

環境變數放在專案根目錄的 `.env`：

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.4-mini
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
VLM_PASS3=0          # 關掉 VLM 感知第 3 次呼叫（見下方說明）
```

- **請設 `VLM_PASS3=0`**：VLM 感知每張照片第 1 次列物件、第 2 次補漏、第 3 次找獨特特徵。
  第 3 次的回覆格式和解析不合，2026-10 以前結果全部被丟掉；解析修好後在 lab1019（376 張）重新測，
  單張不帶感測器 Top-1 **開 47.6%、關 48.1%**，對定位沒有幫助，只多花一次 VLM 呼叫（約 4 秒）。
  程式預設仍是開（`1`），所以要在 `.env` 關掉；改完要重新啟動伺服器。
  （`eval/results/image_only_lab1019_vlm_replay*.md`；`python -m eval.image_only --vlm-replay` 可用快取重測，不需重新呼叫 VLM。）

- `http://localhost:8000/docs`：Swagger UI；`/health`：健康檢查。
- `/dashboard`：即時監看；`/debug`、`/session/{id}/debug-page`：定位診斷。
- 沒裝 torch／transformers 或沒有 DINOv2 參考向量時，定位會退回只用物件＋OCR 比對。
- 本機 GroundingDINO／EasyOCR 預設關閉（`PERCEPTION_ENABLED=0`、`OCR_ENABLED=0`），物件與文字都由 VLM 讀。

---

## 使用方式

```bash
# 1) 可用的店與起點
curl http://localhost:8000/places
curl "http://localhost:8000/starts?place=系辦"

# 2) 建立 session
curl -X POST http://localhost:8000/session -H "Content-Type: application/json" \
  -d '{"goal": "找優格", "place": "A7家樂福（9/16)"}'

# 3)（可選）從起點出發：x、y 是 /starts 回傳的地圖座標
curl -X POST http://localhost:8000/session/<id>/start -H "Content-Type: application/json" \
  -d '{"x": 0.0, "y": 0.0}'

# 4) 每走一段拍一張照片
curl -X POST http://localhost:8000/session/<id>/photo -F "photo=@step1.jpg"

# 5) 路線（小地圖）、回答提問、確認抵達
curl http://localhost:8000/session/<id>/path
curl -X POST http://localhost:8000/session/<id>/answer  -H "Content-Type: application/json" -d '{"answer": "是"}'
curl -X POST http://localhost:8000/session/<id>/confirm -H "Content-Type: application/json" -d '{"kind": "confirmed"}'
```

`/confirm` 的 `kind`：`confirmed`（到了）、`false_positive`（不是）、`wrong_instance`（不是這個）、
`repeated_misidentification`（連續認錯，改用提問）。

各端點的完整欄位（照片可附帶的 PDR 位置、航向、DINOv2 top-K 等）以 `/docs` 為準。
每次導航的紀錄寫在 `output/sessions/<session_id>/`（`server/session_log.py`）。

---

## API 端點

| 分類 | 端點 |
|------|------|
| 店家與起點 | `GET /places`、`GET /starts`、`GET /starts/photo`、`POST /starts` |
| 導航 session | `POST /session`、`POST /session/{id}/photo`、`POST /session/{id}/answer`、`POST /session/{id}/confirm`、`POST /session/{id}/start` |
| 路線與地圖 | `GET /session/{id}/path`、`GET /session/{id}/route`、`GET /session/{id}/map`、`POST /session/{id}/sensor-map`、`GET /reference-map` |
| 單張定位 | `POST /localize`、`POST /ocr` |
| 照片 | `GET /session/{id}/photo/{node_id}.jpg`、`GET /session/{id}/photo/{node_id}/{pass}.jpg`、`GET /session/{id}/original/{node_id}.jpg`、`GET /reference-photo/{nid}.jpg`、`GET /test-photo/{set}/{file}` |
| 建圖工具 | `GET /topomap-builder`、`GET /topomap-places`、`GET /existing-topomap`、`GET /topomap-data`、`GET\|POST /topomap-sources…`、`POST /topomap-walks`、`GET /topomap-suggest`、`GET\|POST\|DELETE /topomap-floorplan`、`POST /topomap-confirm` |
| 診斷 | `GET /health`、`GET /dashboard`、`GET /debug`、`GET /sessions`、`GET /session/{id}`、`GET /session/{id}/debug`、`GET /session/{id}/debug-page`、`POST /test/config` |
| 感測器測試 | `POST /sensor-test`、`GET /sensor-test/{id}/plot.png` |

---

## 主要設定（`server/config.py`）

| 變數 | 預設 | 說明 |
|------|------|------|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | — / `https://api.openai.com/v1` / `gpt-5.4-mini` | VLM 服務；`OPENAI_BACKUP_KEY` 為備用金鑰 |
| `VLM_ROUNDS` / `VLM_PASS3` / `VLM_TIMEOUT_S` | `1` / `1` / `120` | 感知輪數、每輪第 3 次呼叫（找特徵；**建議 `0`**，見上方）、逾時秒數 |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | — / `neo4j` / — | 店家地圖資料庫 |
| `NEO4J_PLACE` | `A7家樂福（9/16)` | 預設店名；Neo4j 連不上時 `/places` 只列這一間 |
| `ROUTE_MAP_SOURCE` | `neo4j` | 路線用 Neo4j 地圖＋`maps/neo4j_links`；`editor` 改用 `maps/*.json` 手繪路線 |
| `LOC_SERVER_DINO` | `1` | 伺服器端 DINOv2 影像分數（手機沒送 DINOv2 時） |
| `DINO_SIDECAR_PATH` | `dino_sidecar.pkl` | 參考照片的 DINOv2 向量（另有一份存在 Neo4j 的 `DirPhoto.dino`） |
| `LOC_SERVER_IMAGE_K` / `LOC_PHONE_IMAGE_K` | `32,64,8` / `0.25,4,2` | 影像證據在融合裡的權重 k(σ)：σ≤2 m、σ≥5 m、還沒定位過 |
| `LOC_FUSION_V2` | `1` | 影像 × 感測器的機率融合 |
| `LOC_RELATION_WEIGHT` | `0.05`（開） | 物件上下左右關係分數的權重（`server/object_relations.py`，用框算，伺服器端 DINOv2 那條路）；`0` 關閉。lab1019 實測沒有變好，見 `eval/results/object_relations_lab1019.md` |
| `LOC_ADJACENT` | `1`（開） | 讀地圖的 `ADJACENT`（九宮格相鄰物件）給格子比對加分（`SPATIAL_BONUS_COEFF` 0.10，手機送 DINOv2／沒有 DINOv2 的店才會用到）；`0` 關閉。lab1019 實測格子比對變差，見 `eval/results/relations_compare_lab1019.md` |
| `EMBEDDING_ENABLED` / `EMBEDDING_MODEL` | `1` / `text-embedding-3-small` | 物件名稱語意相似度；每張照片多一次 embedding 請求 |
| `EMBEDDING_CACHE_PATH` | `output/label_embeddings.npz` | 地圖標籤向量快取 |
| `AISLE_SIGN_PLACES` | `NEO4J_PLACE` | 哪些店使用走道吊牌號碼（用 `\|` 分隔） |
| `SECOND_PHOTO_BELOW` | `0.4` | 最佳節點機率低於此值時，請使用者原地轉身再拍一張 |
| `RERANK_ENABLED` | `0` | VLM 重排序候選節點（每張約多 10 秒） |
| `GOAL_CROP_VERIFY` / `ARRIVED_MIN_DETECTION_SCORE` | `1` / `0.50` | ARRIVED 前裁切目標交 VLM 複驗、最低偵測佐證分數 |
| `PERCEPTION_ENABLED` / `OCR_ENABLED` | `0` / `0` | 本機 GroundingDINO / EasyOCR |
| `MAP_BUILD_ROOT` / `MAP_BUILD_SOURCES` / `MAP_BUILD_REF_PLACE` | `output/mapbuild` / … / — | 建圖工作區、Sensor Lab 來源資料夾、參考底圖 |

---

## 建立新店地圖（Neo4j V2 格式）

和 Neo4j 裡的「A7家樂福（9/16)」相同格式（`Waypoint → DirPhoto → Object`，加上 `WALKWAY`、`ADJACENT`），每間店做一次。共用邏輯與格式說明在 `server/map_v2.py`。

| 步驟 | 工具 | 產出 |
|------|------|------|
| 1. 現場收資料 | Android App → Sensor Lab：從入口開始錄，每個點位用「四方向拍照」拍前、右、後、左；分好幾趟走，每趟匯出一個 zip | `sensor_log.csv` + `photos/` |
| 2. 匯入路線 | 建圖頁的「店家」選單 →「＋ 新增店家…」，填店名、勾選 Sensor Lab 資料後按「重建地圖」；或 `python build_map_v2_topomap.py --place "<店名>" walk1.zip walk2.zip …` | 該店的工作資料夾：`01/`、`02/`…照片 + `topomap.json` |
| 3. 對齊、合併 | 啟動後端，打開 `http://localhost:8000/topomap-builder?place=<店名>`（不給 `place` 會開最近編輯的店）：對齊、合併後按「確認並儲存」 | 該店的 `topomap_confirmed.json` |
| 4. 上傳 | `python upload_map_v2.py --place "<店名>" --dry-run` 先檢查，沒問題再拿掉 `--dry-run` | `Waypoint`、`HAS_PHOTO`、`DirPhoto`、`WALKWAY` |
| 5. 產生物件 | `python rebuild_objects_v2.py --place "<店名>" --limit 2 --dry-run` 試跑，再完整執行 | `Object`、`DETECTED`、`ADJACENT` |
| 6. DINOv2 向量 | `python store_dino_v2.py --place "<店名>" --dry-run`，再拿掉 `--dry-run`（`--no-neo4j` 只寫 sidecar） | `dino_sidecar.pkl`、`DirPhoto.dino` |
| 6b. 物件上下左右（選用） | `python build_object_relations.py --place "<店名>" --dry-run`，再拿掉 `--dry-run`；`--delete` 移除 | `ABOVE`、`BELOW`、`LEFT_OF`、`RIGHT_OF`（同一張照片、每個物件連最近 4 個） |
| 7. 起點與補充連線 | 建圖頁設定起點（位置、面向、照片）；Neo4j 缺的走道邊寫進 `maps/neo4j_links/<店名>.json`（參考 `系辦.json`） | `maps/neo4j_links/<店名>.json`、`maps/starts/<店名>/` |

- **每間店一個工作資料夾**：`<MAP_BUILD_ROOT>/<店名>/`（預設 `backend/output/mapbuild/<店名>/`；店名裡 Windows 不允許的字元會換成 `_`，例如「A7家樂福（9/16)」→「A7家樂福（9_16)」）。裡面有 `topomap.json`、`topomap_confirmed.json`、各路線照片、`dino_embeddings.pkl`、`floorplan.*`、`imports/`、`vlm_cache/`。建圖頁用頂端的「店家」選單切換，腳本依 `--place` 找資料夾（`--workspace` 可以另外指定）。舊的環境變數 `MAP_BUILD_DIR` 現在代表這個總資料夾。
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
- **安全**：本工具建立的 `DirPhoto` / `Object` 帶有 `place` 欄位。`upload_map_v2.py --replace`、`rebuild_objects_v2.py`、`store_dino_v2.py`、`build_object_relations.py` 只會動這種 place，同學建的 A7 一律拒絕寫入（`build_object_relations.py` 要加 `--allow-foreign` 才會寫）。
- **沒有 DINOv2 向量的店**：定位會退回只用物件＋OCR（九宮格）比對，所以建完圖記得做第 6 步。

---

## 測試與評估

```powershell
.\venv\Scripts\python.exe -m pytest -q     # server/tests、eval/tests（離線，不打 API）
```

目前 `server/tests/test_vlm.py` 有 2 個檢查感知提示詞文字的測試未通過（整合前就存在）。

`eval/` 是定位與導航評估：單張影像定位基準（`image_only.py`，可用 VLM 重播模式不呼叫 API）、
權重掃描（`fusion_sweep.py`）、逐張報告（`photo_report.py`）、Sensor Lab 實測資料（`sensorlab.py`）、
動作準確率 harness（`runner.py`、`replay.py`、`metrics.py`）。標註在 `eval/data/`，結果在 `eval/results/`，
說明見 [`eval/README.md`](eval/README.md)。

---

## 目錄結構

```
backend/
├── server/                     FastAPI 後端（正式使用）
│   ├── server.py               所有端點與拍照導航流程
│   ├── config.py               環境變數與預設值
│   ├── vlm.py / prompts.py     VLM 感知、決策與提示詞
│   ├── goal_decomposer.py      目標 → 要找的物件
│   ├── visual_localization.py  物件／OCR 與地圖比對
│   ├── locate_fusion.py        DINOv2 ＋ 物件 ＋ OCR 融合
│   ├── dino_embed.py           DINOv2 影像向量（lazy 載入）
│   ├── loc_fusion.py           影像 × 感測器機率融合
│   ├── aisle_signs.py          走道吊牌
│   ├── photo_quality.py        照片品質
│   ├── heading.py              朝向與左右轉
│   ├── editor_map.py           可走路線圖、轉彎清單、小地圖
│   ├── path_planner.py         A* 與多目標排序
│   ├── neo4j_client.py         Neo4j HTTP 客戶端
│   ├── map_v2.py / map_build.py  新店地圖格式與建圖工作區
│   ├── start_points.py         起點
│   ├── session.py / session_log.py  session 狀態與紀錄
│   ├── scene.py / annotator.py / graph_renderer.py  場景文字、標註圖、場景圖
│   ├── perception.py / ocr.py  本機 GroundingDINO / EasyOCR（預設關閉）
│   ├── place_recognizer.py / grid_utils.py / topomap.py / batch_mapper.py / sensor_test_render.py
│   ├── topomap_builder.html / dashboard.html / debug_page.html
│   └── tests/                  pytest
├── eval/                       評估工具、標註、結果、圖表
├── maps/                       neo4j_links（補充走道邊＋起點）、starts（起點照片）、編輯器路線
├── legacy/                     舊後端（/snav、topomap_v2、locate_v2）與早期定位研究，只供研究腳本 import
├── build_map_v2_topomap.py     Sensor Lab → 建圖工作區
├── upload_map_v2.py            確認後的地圖 → Neo4j
├── rebuild_objects_v2.py       方向照片物件偵測 → Neo4j
├── store_dino_v2.py            方向照片 DINOv2 向量 → sidecar／Neo4j
├── build_object_relations.py   方向照片裡物件的上下左右關係 → Neo4j（ABOVE／BELOW／LEFT_OF／RIGHT_OF）
├── dino_sidecar.pkl            參考照片 DINOv2 向量
├── build_store_graph.py / build_store_map.py / build_html_map.py / generate_topomap.py / render_clean.py / ocr_dump.py
│                               早期的拓樸圖產生與繪製工具（GroundingDINO／EasyOCR 建圖）
├── Dockerfile / requirements-cloud.txt   輕量 Docker 映像（不含 torch，尚未在目前版本驗證）
├── requirements-server.txt / start.ps1 / pytest.ini / conftest.py
└── README.md
```
