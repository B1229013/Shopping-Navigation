"""Evaluation runner: build turn records, summarize, and report.

Pure helpers (``make_template``, ``records_from``, ``format_report``) carry the
logic and are unit-tested. ``main`` is thin file-I/O glue around them.

Usage
-----
Make a labels template from a folder of photos (no models needed)::

    python -m eval.runner make-template --photos path/to/photos --goal "find the milk" --out eval/labels.json

Then hand-fill ``expected_action`` and ``is_goal_frame`` in labels.json.

Score a predictions file against labels::

    python -m eval.runner score --labels eval/labels.json --predictions preds.json

``preds.json`` is a list of objects shaped like::

    {"photo": "0.jpg", "predicted_action": "MOVE", "guidance": "...",
     "detection_sides": ["left"], "json_valid": true, "arrival_corroborated": false}

Generate predictions by replaying photos through the live pipeline::

    python -m eval.runner predict --photos path/to/photos --goal "find the milk" --out preds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import TurnRecord, summarize

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def make_template(photo_names: list[str], goal: str) -> list[dict]:
    """Blank label rows, one per photo, for a human to fill in."""
    return [
        {"photo": name, "goal": goal, "expected_action": None, "is_goal_frame": None}
        for name in photo_names
    ]


def records_from(labels: list[dict], predictions: list[dict]) -> list[TurnRecord]:
    """Merge ground-truth labels into predictions, keyed by photo filename.

    Iterates predictions (a turn must have happened to be scored). A matching
    label contributes ``expected_action``/``is_goal_frame``/``goal``; an
    unlabeled prediction yields a record with those left as ``None``.
    """
    label_by_photo = {l["photo"]: l for l in labels}
    records: list[TurnRecord] = []
    for p in predictions:
        label = label_by_photo.get(p["photo"], {})
        records.append(TurnRecord(
            photo=p["photo"],
            goal=label.get("goal", "") or "",
            predicted_action=p["predicted_action"],
            guidance=p.get("guidance", ""),
            detection_sides=p.get("detection_sides", []),
            json_valid=p.get("json_valid", True),
            arrival_corroborated=p.get("arrival_corroborated", False),
            expected_action=label.get("expected_action"),
            is_goal_frame=label.get("is_goal_frame"),
        ))
    return records


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_report(summary: dict) -> str:
    """Render a summarize() dict as a readable, denominator-annotated report."""
    lines = [
        "Navigation accuracy report",
        f"  turns: {summary['n_turns']}  (labeled: {summary['n_labeled']})",
        f"  success rate:           {_pct(summary['success_rate']):>7}  (n={summary['n_labeled']})",
        f"  false-arrival rate:     {_pct(summary['false_arrival_rate']):>7}  (n={summary['n_non_goal']})   <- lower is better",
        f"  direction sanity:       {_pct(summary['direction_sanity']):>7}  (n={summary['n_directional']})",
        f"  json validity:          {_pct(summary['json_validity_rate']):>7}  (n={summary['n_turns']})",
        f"  arrival corroboration:  {_pct(summary['arrival_corroboration_rate']):>7}  (n={summary['n_arrived']})",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI glue (thin; the logic above is what's tested)
# --------------------------------------------------------------------------- #

def _list_photos(photo_dir: str) -> list[str]:
    return sorted(
        p.name for p in Path(photo_dir).iterdir()
        if p.suffix.lower() in _IMAGE_EXTS
    )


def _cmd_make_template(args: argparse.Namespace) -> None:
    template = make_template(_list_photos(args.photos), args.goal)
    Path(args.out).write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(template)} blank label rows to {args.out}. Fill in expected_action and is_goal_frame.")


def _cmd_score(args: argparse.Namespace) -> None:
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    records = records_from(labels, predictions)
    print(format_report(summarize(records)))


def _cmd_predict(args: argparse.Namespace) -> None:
    # Lazy import: replaying through the live pipeline needs the heavy model stack.
    from eval.replay import predict_over_photos

    predictions = predict_over_photos(args.photos, args.goal)
    Path(args.out).write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {args.out}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Navigation accuracy evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("make-template", help="emit a blank labels file for a photo folder")
    t.add_argument("--photos", required=True)
    t.add_argument("--goal", required=True)
    t.add_argument("--out", default="eval/labels.json")
    t.set_defaults(func=_cmd_make_template)

    s = sub.add_parser("score", help="score a predictions file against labels")
    s.add_argument("--labels", required=True)
    s.add_argument("--predictions", required=True)
    s.set_defaults(func=_cmd_score)

    p = sub.add_parser("predict", help="replay photos through the live pipeline")
    p.add_argument("--photos", required=True)
    p.add_argument("--goal", required=True)
    p.add_argument("--out", default="preds.json")
    p.set_defaults(func=_cmd_predict)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
