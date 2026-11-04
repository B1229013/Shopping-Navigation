"""Sensor Lab walks → map-building workspace (photos per walk + topomap.json for the builder).

Step 1 of building a new store map in the shared V2 schema (server/map_v2.py). Each input is
one Sensor Lab export (the .zip, or a folder holding sensor_log.csv + photos/) recorded from
the store entrance. The builder's 「路線」 button does the same from the browser.

A walk already in the workspace keeps its number (01, 02, …); new ones get the next number,
so edits saved in the builder stay on the right waypoints.

Usage (from backend/):
    python build_map_v2_topomap.py --place "全聯 XX店" walk1.zip walk2.zip walk3.zip

The store's workspace is MAP_BUILD_ROOT/<店名>/ (backend/output/mapbuild/全聯 XX店/); then open
http://localhost:8000/topomap-builder?place=全聯 XX店 to align / merge the nodes and press 確認並儲存.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.config import MAP_BUILD_ROOT  # noqa: E402
from server.map_build import assign_sets, build_workspace, workspace_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="+", type=Path, help="Sensor Lab 匯出的 zip 或資料夾")
    ap.add_argument("--place", required=True, help="Neo4j 的 place 名稱（新店名）")
    ap.add_argument("--workspace", type=Path, default=None,
                    help=f"建圖工作資料夾（預設 {MAP_BUILD_ROOT}/<店名>）")
    ap.add_argument("--force", action="store_true", help="工作資料夾已有 topomap.json 時照樣重建")
    args = ap.parse_args()

    ws: Path = args.workspace or workspace_for(args.place, MAP_BUILD_ROOT)
    out = ws / "topomap.json"
    current = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    if current and not args.force:
        print(f"{out} 已存在；確定要重建請加 --force（已在用的路線會保留原本的編號）")
        return 1

    walks, next_set = assign_sets(current, args.walks)
    try:
        topo, report = build_workspace(args.place, walks, ws, next_set)
    except ValueError as e:
        print(e)
        return 1
    print("\n".join(report))
    if not topo["nodes"]:
        print("沒有任何拍照節點（Sensor Lab 要用「四方向拍照」記錄節點）")
        return 1
    print(f"\n已寫入 {out}：{topo['stats']['total_nodes']} 個節點、{len(topo['edges'])} 條邊")
    print(f"下一步：打開 http://localhost:8000/topomap-builder?place={args.place} 對齊、合併後按「確認並儲存」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
