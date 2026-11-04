"""測試目標搜尋：goal decompose + topo map 搜尋。

用法：
    py -3 test_goal_search.py                          # 預設 5 個目標
    py -3 test_goal_search.py 牛奶 餅乾                # 自訂目標
    py -3 test_goal_search.py --place 0908家樂福1 咖啡  # 指定地圖
"""
import sys
import argparse

sys.path.insert(0, ".")

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from server.neo4j_map_store import Neo4jMapStore
from server.topomap_v2 import NTYPE_OBJECT
from server.goal_decomposer import decompose_goal
from server.sensor_nav import _search_goal_in_topo


def main():
    parser = argparse.ArgumentParser(description="測試目標搜尋")
    parser.add_argument("goals", nargs="*", default=["衛生紙", "優格", "洋芋片", "結帳", "茶"])
    parser.add_argument("--place", default="A7家樂福", help="Neo4j 地圖名稱")
    parser.add_argument("--top", type=int, default=10, help="顯示前幾名")
    args = parser.parse_args()

    print(f"載入地圖: {args.place} ...")
    store = Neo4jMapStore()
    topo = store.download(args.place)
    store.close()

    photo_count = len(topo.all_photo_nodes())
    obj_count = len(topo.all_object_nodes())
    print(f"地圖: {photo_count} 張照片, {obj_count} 個物件\n")

    for goal in args.goals:
        print(f"{'='*70}")
        print(f"目標: {goal}")
        print(f"{'='*70}")

        keywords = decompose_goal(goal)
        core_cutoff = max(len(keywords) // 3, 3)
        print(f"核心關鍵字 ({core_cutoff}): {', '.join(keywords[:core_cutoff])}")
        print(f"輔助關鍵字 ({len(keywords)-core_cutoff}): {', '.join(keywords[core_cutoff:])}")

        results = _search_goal_in_topo(topo, goal, keywords)

        print(f"\n匹配 {len(results)} 張照片，前 {min(args.top, len(results))} 名:")
        for i, pid in enumerate(results[:args.top], 1):
            objs = topo.photo_objects(pid)
            obj_labels = [o.get("label", "") for o in objs]
            ocr_texts = [o.get("ocr_text", "") for o in objs if o.get("ocr_text")]
            info = ", ".join(obj_labels[:4])
            if ocr_texts:
                info += f" | OCR: {', '.join(ocr_texts[:3])}"
            print(f"  {i:2d}. P{pid:<3d} — {info}")
        print()


if __name__ == "__main__":
    main()
