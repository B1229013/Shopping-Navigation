"""verify_server_fusion.py — 驗證 server 進入點 fusion_localize_for_server（即時嵌入）。

走正式 server 會用的函式：對每張查詢照片「即時」算 DINO 向量（非快取）、
lazy 建索引、物件/OCR 比對、融合排名。確認 live 路徑可跑且準確率對得上。
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server.topomap_v2 import TopoGraphV2
from server.locate_fusion import fusion_localize_for_server
from eval_localization_real import TOPOMAP, GT_FILE
from eval_localization_single import load_query_photos
from eval_localization_gm_ocr import load_query_ocr


def main():
    topo = TopoGraphV2.load(str(TOPOMAP))
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    queries = []
    for key, v in gt.items():
        if v.get("ambiguous"):
            continue
        session, wp = key.rsplit("_wp", 1)
        for qdir, objs, _ in load_query_photos(session, wp):
            hits = glob.glob(str(HERE/"testPDR"/session/"photos"/f"wp_{wp}_{qdir}_*.jpg"))
            if not hits:
                continue
            det_dicts = [{"label": o["label"]} for o in objs]
            ocr_items = [{"text": t} for t in load_query_ocr(session, wp, qdir)]
            queries.append({"gt": v["node_id"], "img": hits[0], "det": det_dicts, "ocr": ocr_items})

    t1 = t3 = t5 = n = 0
    t0 = time.time()
    for i, q in enumerate(queries):
        ranked = fusion_localize_for_server(topo, q["img"], q["det"], q["ocr"], top_k=5)
        if ranked is None:
            print("fusion 回傳 None（DINO/sidecar 不可用）→ 會退回 WGM"); return
        order = [pid for pid, _ in ranked]
        n += 1
        if order and order[0] == q["gt"]: t1 += 1
        if q["gt"] in order[:3]: t3 += 1
        if q["gt"] in order[:5]: t5 += 1
        if (i+1) % 30 == 0:
            print(f"  ...{i+1}/{len(queries)} ({time.time()-t0:.0f}s) 累計Top-1 {t1}/{n}", flush=True)

    print(f"\nserver 進入點 fusion_localize_for_server（即時 DINO 嵌入，查詢 {n}，耗時 {time.time()-t0:.0f}s）")
    print(f"Top-1 {t1/n*100:.1f}%   Top-3 {t3/n*100:.1f}%   Top-5 {t5/n*100:.1f}%")


if __name__ == "__main__":
    main()
