# Navigation accuracy evaluation harness（`/snav` 後端用）

和 [`backend/eval/`](../../eval/README.md) 的動作準確率 harness 是同一套程式（`runner.py`、`metrics.py`、
`replay.py`），差別只在 `import server...` 拿到哪個後端：在 `backend/legacy` 底下執行，量的是
`backend/legacy/server/`（`/snav` 後端）；在 `backend/` 底下執行，量的是目前的導航後端。

```
cd backend/legacy
python -m eval.runner …
python -m pytest eval/tests -q
```

## Metrics

| Metric | Needs labels? | Meaning | Good direction |
|---|---|---|---|
| `success_rate` | yes | predicted action == expected action | higher |
| `false_arrival_rate` | yes | said ARRIVED on a non-goal frame — the worst failure | **lower** |
| `direction_sanity` | no | MOVE guidance names a side (左／右, or left／right) where something was detected (`left` also counts `far-left`) | higher |
| `json_validity_rate` | no | VLM response parsed (not `vlm._FALLBACK`) | higher |
| `arrival_corroboration_rate` | no | ARRIVED backed by a detection / OCR match | higher |

Every metric is reported with its denominator count (`n=…`) so an empty sample is
not mistaken for a perfect (or terrible) score.

With `predict`, `arrival_corroboration_rate` is 100% by construction: `scene.verify_arrival`
already downgrades every ARRIVED without evidence. It only means something for predictions
files made some other way.

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

   Or replay the photos through the pipeline (GroundingDINO perception, EasyOCR when
   `OCR_ENABLED`, and the VLM decision — paid API calls; see `replay.py`):

   ```
   python -m eval.runner predict --photos path/to/photos --goal "find the milk" --out preds.json
   ```

3. **Score it:**

   ```
   python -m eval.runner score --labels eval/labels.json --predictions preds.json
   ```

## Before / after

每次改動前後各跑一次，比較同一個指標。
