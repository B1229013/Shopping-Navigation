# 照片定位（locate_v2）— 使用說明

`locate_v2` 回答：**「給一張新照片的偵測結果，它最可能位於既有拓樸地圖（`topomap_v2` 格式）的哪個照片節點？」**
全程是規則式計分，不呼叫模型；模型只用在更早的「照片 → 偵測結果」那一步。

- 程式在 `backend/legacy/`，地圖格式見 [README_topomap_v2.md](README_topomap_v2.md)。
- 誰在用：`/snav` 後端（`backend/legacy/server/server.py`、`sensor_nav.py`）和下面的 CLI `locate_photo_v2.py`。
- 目前的導航後端 `backend/server/` 不用 `locate_v2`：定位改由 `locate_fusion.py`（DINOv2 ＋ 物件 ＋ OCR）
  與 `visual_localization.py`（Neo4j 地圖比對）負責，見 [定位檔案索引.md](定位檔案索引.md)。

## 檔案清單

```
backend/legacy/server/locate_v2.py             核心比對邏輯（localize_photo() /
                                                 render_localization_debug_png() /
                                                 format_localization_report()）
backend/legacy/locate_photo_v2.py              CLI（--detect-mode vlm/groundingdino/auto、
                                                 --detect-runs 多輪偵測合併）
backend/legacy/server/topomap_v2.py            地圖資料結構（建查詢子圖、OCR 關聯到物件）
backend/legacy/server/config.py                LABEL_SYNONYMS 同義詞表
backend/legacy/server/tests/test_locate_v2.py  單元測試（31 個，見下方「測試」）
```

## 比對邏輯

`localize_photo()` 先把查詢照片當成一個臨時照片節點接進地圖的**副本**（`copy.deepcopy`，不會改到傳進來的地圖），
用建圖同一套 `build_subgraph_from_detections()` 建出物件子圖（標籤正規化、OCR 關聯都一樣），
再對每個真實照片節點（非虛擬分支）算分。分數有兩條路：

### A. 有 VLM 原始物件框時：九宮格加權比對（WGM）

呼叫端傳 `raw_vlm_items`（`/snav` 後端和 `sensor_nav` 都會傳）時，分數＝`_best_wgm_score()`：

- 查詢照片和節點照片的物件各自依框的中心放進 3×3 格子（`top-left` … `bottom-right`）。
- 逐格貪婪配對：同格相似度 ×1.0、相鄰格 ×0.7；相似度預設是字串比對（門檻 0.25），
  `use_embedding=True` 時改用 embedding cosine（門檻 0.35）。
- 每個物件有重要性權重：地標 5、標示牌 3、設備 2、一般 1、通用物件 0.3（`_classify_importance()`）。
- 分數＝Σ(權重 × 相似度) ÷ Σ(查詢物件權重)；四個方向的參考照片各算一次取最高。

### B. 沒有原始物件框時（CLI）：物件配對 ＋ 雙向覆蓋率

**單一配對分數**（`_score_pair()`，依序試）：

```
1. 原始標籤相似度 ≥ 0.4      → + 1.0 × 相似度         （保留品牌、走道號碼這些細節）
2. 否則正規化標籤相同          → + 稀有度權重
   或語意相似度 ≥ 0.5          → + 稀有度權重 × 相似度 × 0.6
3. OCR 文字完全相同            → + 1.5；互相包含（≥ 2 字）→ + 1.0
```

至少要有「標籤」或「OCR」其中一項對上才算一組配對；畫面位置不參與單一配對分數。
沒有 OCR 文字的門／牆／窗等結構物不參與比對，帶 OCR 文字的（例如貼著房間牌子的門）照樣比對。

**稀有度權重**（`_compute_label_rarity_weights()`）：統計每個標籤出現在幾個不同的候選照片節點
（同一節點出現多次只算一次），用 log-IDF 換算：

```
N = 候選照片節點數，freq = 有這個標籤的節點數
weight = log(N / freq + 1) / log(N + 1)，限制在 0.05 ~ 1.0
```

只出現在一個節點的標籤權重 1.0；每個節點都有的標籤權重最低（例如 N = 59 時約 0.17），下限 0.05。

**貪婪配對**：所有候選配對依分數高到低排序，物件用過就不能再配第二次。

**雙向覆蓋率**（調和平均）：

```
Rn = 配對分數總和 / (節點所有物件的稀有度權重總和 + 1.2)
Rq = 配對分數總和 / (查詢照片所有物件的稀有度權重總和 + 1.2)
覆蓋率 = 2·Rn·Rq / (Rn + Rq)          ← 任一邊證據不夠就被拉低
```

**再乘上三個調整**：

- OCR 全域加分（`_global_ocr_bonus()`）：兩邊 OCR 文字切成詞，完全相同 +0.25、互相包含 +0.15，最多 +0.5。
- 上下排列一致性（`_pairwise_vertical_consistency()`）：配對成功的物件兩兩比較「誰在上面」，
  一致比例用貝氏平滑（先驗 0.5 / 1）；配對少於 2 組 → 0.5；有 2 組以上但都在同一個垂直格 → 0.85。
- 鑑別力（`_discriminative_quality()`）：配對裡沒有 OCR（單組分數 ≥ 1.0）、而且所有標籤的稀有度都 < 0.35
  （全是到處都有的東西）→ × 0.55。

```
節點分數 = min((覆蓋率 + OCR 加分) × 排列一致性 × 鑑別力, 1.0)
```

同一個節點也會只拿某一個方向（front/back/left/right）的物件再算一次，取兩者較高的分數。
最高分 < 0.15 時，結果會附上「信心偏低」的警告。完整常數與理由見 `server/locate_v2.py`。

## 偵測模式：GroundingDINO 或兩階段 VLM

系統依 GroundingDINO 有沒有成功載入，自動切換兩套偵測邏輯：

- **模式 A（GroundingDINO）**：逐一精確框選候選詞清單裡的物件
- **模式 B/C（兩階段 VLM）**：改用 LLM 直接用文字描述整個畫面，物件
  位置是「粗略九宮格＋近遠」文字描述，`box` 座標是依這段文字換算出來
  的固定模板，不是逐一量測的結果（VLM 沒辦法可靠輸出精確座標，這是
  刻意的設計取捨，不是資料有問題）

實測發現：拿模式 A 重新偵測、去跟模式 B/C 建出來的地圖節點比對，即使
畫面內容一樣，物件標籤的用詞、精細度也會差很多——不是同一套方法產生
的東西，比不起來，不是環境設定或版本問題。**建圖統一用模式 B/C（VLM），`locate_photo_v2.py` 的 `--detect-mode` 預設也
是 `vlm`**。

## 多輪偵測合併（`--detect-runs`）：降低 VLM 模式的信心分數波動

實測發現：同一張照片、同一支程式，VLM 模式連續跑三次，分別偵測到
5、7、7 個物件，其中一次漏讀了門上的 OCR 文字，導致最終信心分數在
55%~75% 之間跳動——核心物件（desk/chair/sofa 這種明顯的東西）三次都
配對到同一批，純粹是「這次有沒有順手多注意到一兩樣東西、有沒有讀到
某段文字」每次有出入。這是 VLM（語言模型）本身的特性，不是 bug——
GroundingDINO 模式已經驗證過「同一張圖連續跑兩次，結果位元級相同」，
只有 VLM 模式才有這個問題。

`--detect-runs N`（N>1）讓 CLI 真的把偵測重複跑 N 次（各自獨立呼叫
模型，花 N 倍時間／API 額度），再合併成一份查詢物件清單。合併規則
（`_merge_multi_run_detections()`）：對每個標籤，分別統計它在「每一輪
各自」出現幾次，取**單一輪次裡出現次數最多**的那一組當代表——不是把
各輪次的數量加起來（那樣會把「同一個實體因為每輪判斷位置有些微落差
而被誤拆分」的雜訊也一起加總，物件數只會越合併越膨脹，我們踩過這個
坑，見下方「設計上的取捨」）。只出現在部分輪次的標籤一樣會被保留（聯集，
不是交集），OCR 文字用「同標籤、數量一樣」的其他輪次依順位回填。

**實測效果**（同一張從 P17 位置、但角度略有不同拍攝的照片，各測 3 次）：

| | 分數範圍 | 波動幅度 |
|---|---|---|
| 單次（`--detect-runs 1`） | 55.1% ~ 74.7% | 19.6 個百分點 |
| 兩輪合併（`--detect-runs 2`） | 67.2% ~ 71.1% | **3.9 個百分點**（縮小約 80%） |
| 三輪合併（`--detect-runs 3`） | 57.8% ~ 71.1% | 13.3 個百分點（一次離群值：3 輪剛好都漏看同一樣東西，機率上偶爾會發生，不是邏輯錯誤） |

**兩輪合併（`--detect-runs 2`）目前實測下來報酬率最高**——用兩倍成本
換到波動大幅縮小的效果，三輪合併理論上該更穩定，但目前這批測試沒有
展現出比兩輪更好的效果（多測幾次應該會回到平均值附近）。不管有沒有
合併、合併幾輪，所有測試裡 P17 都正確排名第一、跟第二名差距至少
30 個百分點以上——**信心分數本身會波動，但「排名結果」始終穩定正確**，
實務上比較該看的是有沒有正確排第一、差距夠不夠大，不用糾結分數本身
的小數點差異。

## 快速測試

在 `backend/legacy` 底下執行（`locate_photo_v2.py` 會把這個資料夾加進 `sys.path`）：

```bash
cd backend/legacy

# 用現成的偵測結果 json（離線，不呼叫任何模型）
python locate_photo_v2.py \
    --map output/topomaps/資工系系辦/topomap.json \
    --detections output/sessions/<id>/annotated/detections_<n>.json \
    --photo output/sessions/<id>/photo/<n>.jpg \
    --top-k 5 \
    --json-out locate_queries/result.json \
    --debug-png locate_queries/debug.png

# 或讓 CLI 自己跑一次偵測（預設 VLM 模式；用 backend/legacy/server/config.py 的
# VLM_BACKEND 與對應的 API 金鑰）
python locate_photo_v2.py \
    --map output/topomaps/資工系系辦/topomap.json \
    --photo /path/to/new_photo.jpg \
    --detect-mode vlm \
    --goal "尋找電腦"   # 可選，VLM 模式才會用到

# VLM 模式每次觀察細節不完全一樣，想降低信心分數波動可以跑多輪合併
python locate_photo_v2.py \
    --map output/topomaps/資工系系辦/topomap.json \
    --photo /path/to/new_photo.jpg \
    --detect-runs 2
```

地圖要先用 `build_topomap_v2.py` 建好（預設存在 `backend/legacy/output/topomaps/<場所>/topomap.json`）。
`--detections` 格式跟 `output/sessions/<id>/annotated/detections_*.json` 相同（`{"detections": [...], "ocr": [...]}`）。
`--detect-mode` 還有 `groundingdino`（強制只用 GroundingDINO，失敗直接報錯）跟 `auto`（先試 GroundingDINO，
失敗才退回 VLM）。其他參數：`--min-score`（單組配對最低分，預設 0.05）、`--allow-structural`（結構物也參與比對）。

## 除錯用地圖圖片（`--debug-png`）

把查詢照片畫成地圖上的一個**獨立節點**（紅星標示），跟它配對到的物件
用細線連到對應的候選節點——但**不會**幫查詢節點接任何走道邊，接了不管
接在哪個候選節點後面，排版都會把它畫成「路徑接下去的某一步」，容易
誤導成「這是導航會走到的下一個地方」。所以查詢節點完全獨立於路徑之外
（`_dead_reckon_tree()` 找不到路徑的節點，會被擺到整張地圖版面外、留
足夠安全距離的空白區域，不會疊到主路徑或物件群上）。完全沒有改動
`TopoGraphV2.render_png()` 本身的繪圖邏輯。

## 設計上的取捨

| 問題 | 目前的做法 |
|---|---|
| 查詢照片是另外拍的，構圖、角度和建圖照片不同，用畫面位置當門檻會讓配對全部失敗 | 物件配對只看標籤和 OCR；位置只用在 WGM 的九宮格（相鄰格 ×0.7）和上下排列一致性 |
| 查詢照片多看到不相干的東西，被當成扣分 | 雙向覆蓋率取調和平均 |
| 物件很少的節點靠單一配對虛報高分 | 覆蓋率分母加平滑項（+1.2） |
| 物件種類對上但排列矛盾 | 上下排列一致性乘數 |
| 「證據不足」跟「證據夠但沒有上下差異」罰得一樣重 | 分開給 0.5 / 0.85 |
| 只配到到處都有的東西 | 鑑別力乘數 0.55 |
| 標籤詞彙不統一（sofa/couch、cabinet/cabinet shelf） | `LABEL_SYNONYMS` 同義詞表，取最長變體優先；另外先比原始標籤，保留品牌與號碼 |
| 查詢偵測模式（GroundingDINO）跟建圖模式（VLM）不一致，比不起來 | `--detect-mode` 預設 `vlm`，跟地圖用同一套方法 |
| OCR 文字沒有接到物件上 | 建子圖時把 OCR 關聯到最近的物件（多個物件共用同一個框時，優先給招牌類標籤） |
| 帶 OCR 文字的結構物被一律排除 | 只排除沒有文字的結構物 |
| VLM 每次觀察細節不同，分數波動大 | `--detect-runs N` 多輪合併：同標籤取單一輪次裡出現次數最多的那組 |
| 除錯圖裡查詢節點被誤認成路徑的一部分 | 查詢節點不接走道邊，擺到版面外的空白區域 |

## 實測紀錄（資工系系辦地圖）

下面的分數是在目前計分方式之前量的（當時單組配對有垂直位置加分、OCR 權重 1.2、稀有度下限 0.15），
現在重跑分數會不同；repo 裡沒有這批照片和地圖，無法直接重現。

- desk/chair/sofa 高辨識度物件：改成雙向覆蓋率前 12.5%，改後 56%。
- P15 節點只有 3 個物件（2 門 ＋ 1 個帶「會議室」牌子的 `sign`）：OCR 關聯修好前 24%，修好後 50%。
- 用 VLM 偵測查詢照片（`--detect-mode vlm`）測 P17：三次 77.7%～83.0%，每次都排第一，
  跟第二名（P1，38～41%）差 40 個百分點以上。

## 測試

```bash
cd backend/legacy
python -m pytest server/tests/test_locate_v2.py -q
```

31 個測試：22 個通過、2 個失敗、7 個錯誤。

- 7 個錯誤：整合測試要用 `server/tests/fixtures/real_place_topomap.json`（真實 92 節點地圖），這個檔案不在 repo 裡。
- 2 個失敗：`test_localize_vertical_bucket_off_by_one_gets_partial_credit`、
  `test_structure_consistency_penalizes_contradictory_arrangement`。這兩個測試期望單組配對分數有垂直位置加分、
  排列矛盾時分數較低；目前的 `_score_pair()` 已經不看位置，兩種情況得到一樣的分數。

## 還沒做的部分

- 沒有用「使用者剛剛在哪個節點」縮小範圍，對整張地圖全比對。（目前的導航後端用手機感測器位置做這件事：`server/loc_fusion.py`。）
- 沒有影像特徵，純規則式比對；兩個標籤相同但其實是不同實體的東西分不出來。（目前的導航後端用 DINOv2：`server/locate_fusion.py`。）
- 只回答「最可能是哪個節點」，沒有「介於哪兩個節點之間」；`topomap_v2` 地圖的走道邊大多沒有實際距離（`distance_m` 幾乎都是 `None`）。
- 既有地圖要用 `build_topomap_v2.py --fresh` 重建，才會有 OCR 關聯、同義詞這些修正。
- `--detect-mode groundingdino`／`auto` 還沒實測過。
- 多輪合併只看「單一輪次裡出現次數最多」，沒有用跨輪次的畫面位置互相佐證同一個實體。
