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

## Before / after

Keep a table in `docs/superpowers/specs/2026-05-25-navigation-accuracy-design.md`
recording each metric before and after every change in the plan.
