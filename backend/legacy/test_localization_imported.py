"""
test_localization_imported.py — 用匯入的地圖做 self-recognition 定位測試

對每個照片節點，取其物件清單當作查詢，測試 locate_v2 是否正確定位回自己。
這是 self-test（不需要 VLM 呼叫，不需要實際照片檔案），
用來驗證匯入的資料在我們的定位邏輯下是否可用。

Usage:
    python test_localization_imported.py carrefour_a7_imported.json
    python test_localization_imported.py carrefour_a7_imported.json --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.topomap_v2 import TopoGraphV2
from server.locate_v2 import localize_photo


def run_self_test(topo: TopoGraphV2, top_k: int = 5) -> dict:
    photo_ids = topo.all_photo_nodes()
    print(f"Testing {len(photo_ids)} photo nodes (self-recognition)...\n", flush=True)

    results = {"top1": 0, "top3": 0, "top5": 0, "total": 0, "failures": []}
    t_start = time.time()

    for i, pid in enumerate(photo_ids):
        objs = topo.photo_objects(pid)
        if not objs:
            continue

        # Build query detections from this node's objects
        query_dets = []
        for o in objs:
            det = {
                "label": o.get("label", ""),
                "box": o.get("box", []),
                "score": o.get("score", 0.5),
                "position": o.get("position", ""),
                "nameplate_text": o.get("ocr_text", ""),
            }
            if det["label"]:
                query_dets.append(det)

        if not query_dets:
            continue

        results["total"] += 1

        result = localize_photo(topo, query_dets, img_w=1024, img_h=768, top_k=top_k)

        if not result.candidates:
            results["failures"].append({
                "photo_id": pid, "rank": -1, "reason": "no candidates",
            })
            continue

        # Find where the correct node ranks
        ranked_ids = [c.photo_id for c in result.candidates]
        if pid in ranked_ids:
            rank = ranked_ids.index(pid) + 1
        else:
            rank = -1

        if rank == 1:
            results["top1"] += 1
        if 1 <= rank <= 3:
            results["top3"] += 1
        if 1 <= rank <= 5:
            results["top5"] += 1

        if rank != 1:
            best = result.best
            results["failures"].append({
                "photo_id": pid,
                "rank": rank,
                "best_match": best.photo_id if best else None,
                "best_score": round(best.score, 4) if best else 0,
                "query_objects": len(query_dets),
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(
                f"  [{i+1:3d}/{len(photo_ids)}] "
                f"Top-1: {results['top1']}/{results['total']} "
                f"({results['top1']/max(results['total'],1)*100:.1f}%) | "
                f"{elapsed:.1f}s",
                flush=True,
            )

    elapsed_total = time.time() - t_start
    total = max(results["total"], 1)

    print(f"\n{'='*60}", flush=True)
    print(f"Self-Recognition Test Results ({elapsed_total:.1f}s)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total tested: {results['total']}/{len(photo_ids)}", flush=True)
    print(f"  Top-1: {results['top1']}/{total} ({results['top1']/total*100:.1f}%)", flush=True)
    print(f"  Top-3: {results['top3']}/{total} ({results['top3']/total*100:.1f}%)", flush=True)
    print(f"  Top-5: {results['top5']}/{total} ({results['top5']/total*100:.1f}%)", flush=True)

    if results["failures"]:
        print(f"\n  Failures (not Top-1):", flush=True)
        for f in results["failures"][:20]:
            rank_str = f"rank {f['rank']}" if f["rank"] > 0 else "not found"
            print(
                f"    node {f['photo_id']}: {rank_str}"
                f" (best={f.get('best_match')}, score={f.get('best_score', 0):.4f},"
                f" qobj={f.get('query_objects', 0)})",
                flush=True,
            )
        if len(results["failures"]) > 20:
            print(f"    ... and {len(results['failures'])-20} more", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Self-recognition localization test")
    parser.add_argument("topomap_json", help="Path to imported topomap JSON")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for ranking (default: 5)")
    args = parser.parse_args()

    print(f"Loading {args.topomap_json}...", flush=True)
    topo = TopoGraphV2.load(args.topomap_json)
    print(f"  {len(topo.all_photo_nodes())} photos, {len(topo.all_object_nodes())} objects\n", flush=True)

    run_self_test(topo, top_k=args.top_k)


if __name__ == "__main__":
    main()
