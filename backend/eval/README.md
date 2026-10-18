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

## Before / after

Keep a table in `docs/superpowers/specs/2026-05-25-navigation-accuracy-design.md`
recording each metric before and after every change in the plan.
