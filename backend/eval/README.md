# Navigation accuracy evaluation harness

Measures whether changes to the navigation pipeline actually help. Run it before
and after each change and compare the numbers.

## Metrics

| Metric | Needs labels? | Meaning | Good direction |
|---|---|---|---|
| `success_rate` | yes | predicted action == expected action | higher |
| `false_arrival_rate` | yes | said ARRIVED on a non-goal frame — the worst failure | **lower** |
| `direction_sanity` | no | MOVE guidance names a side where something was detected | higher |
| `json_validity_rate` | no | VLM response parsed (no fallback) | higher |
| `arrival_corroboration_rate` | no | ARRIVED backed by a detection / OCR match | higher |

Every metric is reported with its denominator count (`n=…`) so an empty sample is
not mistaken for a perfect (or terrible) score.

## Workflow

1. **Make a labels template** from a folder of photos (no models needed):

   ```
   python -m eval.runner make-template --photos path/to/photos --goal "find the milk" --out eval/labels.json
   ```

   Then hand-fill `expected_action` (`MOVE`/`ASK`/`ARRIVED`) and `is_goal_frame`
   (`true`/`false`) for each photo. See `labels.example.json` for the shape.

2. **Produce a predictions file** — a JSON list, one object per photo:

   ```json
   {"photo": "0.jpg", "predicted_action": "MOVE", "guidance": "...",
    "detection_sides": ["left"], "json_valid": true, "arrival_corroborated": false}
   ```

   (Live replay via `python -m eval.runner predict …` is enabled once the spatial
   formatter and arrival verifier land — see `replay.py`.)

3. **Score it:**

   ```
   python -m eval.runner score --labels eval/labels.json --predictions preds.json
   ```

## 單張影像定位基準（不靠感測器）

`eval/image_only.py` 用正式伺服器的定位路徑（`server._run_early_localization` →
`locate_fusion.server_image_evidence` → `visual_localization.localize(v2=True)`）逐張跑，
不帶任何手機資料（位置、步行距離、上次定位、手機 DINOv2、購物目標）：

```
python -m eval.image_only --set 0916                       # 268 張 0916 測試照片
python -m eval.image_only --set field1017                  # 2026-10-17 A7 實測 A 部分 24 張
python -m eval.image_only --set route1 --photos <測試路線1>  # 走道中間的照片
```

- 照片和標註：`eval/data/0916_labels.csv`、`field1017_labels.csv`、`route1_labels.csv`；
  照片路徑相對於 `--data-root`（預設 backend/，在 worktree 裡要指到主 checkout 的 backend/）。
- 輸出 `eval/results/image_only_<set>.json` 與 `.md`：Top-1／Top-5／1 hop 內、誤差（m）、
  信心 ≥0.3／≥0.6 的涵蓋率與命中率、正式閘門（品質／遮擋退件）、朝向、各階段耗時、失敗分類與
  最常混淆節點對；`prod` 與 `dino_only`（物件、OCR、走道號碼權重 0）兩組並列。
- 快取在 `output/eval/image_only/<set>/`：縮圖、`perception.json`（品質、OCR、GroundingDINO、
  遮擋、耗時）、`dino.json`（查詢照片 DINOv2 向量）、`refmap.pkl`（地圖快照）。只改融合 → 直接重跑；
  換 DINOv2 → `--reembed`；改偵測 → `--redetect`；離線 → `--map snapshot`。
- 偵測分兩輪跑（先全部 OCR、再全部 GroundingDINO），8 GB 筆電同時載兩個模型會換頁、耗時加倍。

### 帶感測器、Sensor Lab 測試資料

```
python -m eval.image_only --set lab1019 --data-root <主 checkout>/backend --sim
```

- `lab1019`：2026-10-19 三趟 Sensor Lab 測試（`output/eval_photos/1019_A7/1,2,3`），94 點（節點 65、中點 29），
  每點四方向，答案在 `eval/data/lab1019_labels.csv`（拍照前設的點位標記，更正寫在 note 欄）。
- 同一批照片跑「不帶感測器」與「帶感測器」：
  - `+replay`：`eval/sensorlab.py` 把手機的 `MapPositionEstimator`（Kotlin 版的單元測試已移植成
    `eval/tests/test_sensorlab.py`）套在這趟的原始 PDR 上：入口對齊、每點拍完後用正確位置修正一次（oracle），
    送出的 `map_x/y`、`dist_since_fix`、`fix_node` 和導航模式相同。
  - `+simσ`（`--sim`）：手機位置 = 真實位置 + σ = 1、2、4、8 m 的高斯誤差。
- 報表多了「照片類型 × 感測器」表、同一張照片帶／不帶感測器的配對比較、逐點計分（同一點四張只算一個樣本）。
- `eval/results/image_only_<set>_photos.jsonl`：每張照片一行，含所有偵測物件、OCR 文字、每個參考節點的
  DINOv2 cosine／物件／OCR／融合分數、走道號碼、伺服器影像證據、重播的手機輸入，以及每個版本的前五名與誤差。

### 計步有沒有在動（`step_check.py`）

```
python -m eval.step_check <Sensor Lab 匯出資料夾或 sensor_log.csv>...
python -m eval.step_check --results eval/results/image_only_lab1019_vlm_photos.jsonl
```

- Sensor Lab 紀錄：每 4 秒一段，|a| 有起伏（std ≥ 0.6 m/s²）又有步伐節奏（0.38–0.83 秒一步）的算「在走」，
  節奏也給出預期步數。列出「在走卻一步都沒算到」的段落（位置停住），和明顯站著（2 秒 std < 0.3）時算到的步數；
  App 的計數（total_steps）和手機內建計步器（step_detected）各一份。
- 定位結果：相鄰兩個測試點距離 > 2 m、步數卻沒變的路段。
- 2026-10-29：lab1019 有 6/86 段走了 2.4–5 m 卻 0 步；0819 那趟手持紀錄，內建計步器有 8/32 段在走卻沒算到，
  舊的備援偵測整趟只抓到約 10 步。Android `sensor/StepDetection.kt` 改寫備援偵測後，把同一份紀錄重播進
  `PdrTracker`：在走卻 0 步的段落 7/32 → 0/32，總步數 143 → 197（依節奏估約 189 步在走的段落），站著時多算 0 步。

## 物件上下左右關係（`object_relations_eval.py`）

重播 `image_only_lab1019_vlm_photos.jsonl`（不重跑 VLM、不需要照片），把
`server/object_relations.relation_score`（查詢照片與每個節點最像那張方向照片的物件，上下、左右排列是否一致）
以權重 W 加進伺服器的影像分數，比較 Top-1／Top-5／1 hop 內／誤差；W 另外用「兩條路線選、第三條測」交叉驗證。
先確認不加關係時每張照片的結果和原本那次執行一模一樣才繼續。

```
python -m eval.object_relations_eval                          # 有 OPENAI_API_KEY：用伺服器的名稱相似度
python -m eval.object_relations_eval --local-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

2026-10-21 結果（`eval/results/object_relations_lab1019.md`，本機多語模型配對名稱）：基準 Top-1 44.7%；
七種版本（上下＋左右、只上下、只左右、等權、九宮格粗略版＝現有 ADJACENT、四方向取最高、只重排前 5 名）
交叉驗證 42.6–44.9%，沒有一種比基準好。2026-10-22 團隊決定開啟，`LOC_RELATION_WEIGHT` 預設 0.05（交叉驗證選到的權重）。

## 物件關係兩個版本比較（`relations_compare.py`）

同一批照片（lab1019 376 張）跑真正的格子比對 `match_by_grid`，比較地圖 `ADJACENT` 關係的讀法與加分：
不讀、APPNAV_ios `fe4c851` 原樣（把 cell_pairs 的 dir 當成 a 自己的方向，等於上下左右全部讀反）、
方向修正（本分支 `neo4j_client.adjacent_relation`）。看格子比對自己，以及手機送 DINOv2 時的融合結果。

```
python -m eval.relations_compare --local-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

2026-10-21 結果（`eval/results/relations_compare_lab1019.md`）：格子比對 Top-1 不讀 13.6%、原樣 ×0.1 10.4%、
方向修正 ×0.1 11.2%（加分越大越差）；手機 DINOv2 37.5% → 原樣 37.2%、方向修正 38.3%（3 張，屬於誤差範圍）。
都沒有比不用關係好。2026-10-22 團隊決定開啟，`LOC_ADJACENT` 預設 1。

## Before / after

Keep a table in `docs/superpowers/specs/2026-05-25-navigation-accuracy-design.md`
recording each metric before and after every change in the plan.
