"""embed_dino.py — 計算 DINOv2 影像 embedding，存硬碟快取（給融合用）。
鍵＝影像路徑（跟 clip_cache 同一套鍵），方便直接替換視覺訊號。
"""
from __future__ import annotations
import glob, json, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server.topomap_v2 import TopoGraphV2
from eval_localization_real import TOPOMAP, GT_FILE
from eval_localization_single import load_query_photos, _base_dir
from eval_visual import _map_img

DINO_CACHE = HERE / "benchmark_results" / "realworld_0916" / "dino_cache.pkl"


def main():
    topo = TopoGraphV2.load(str(TOPOMAP))
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    photos_of = {n: topo.graph.nodes[n].get("photos") or {} for n in topo.graph.nodes
                 if topo.graph.nodes[n].get("ntype") == "photo"}
    # 地圖影像路徑（每個節點每個方向）
    paths = set()
    nodes_dirs = defaultdict(set)
    for nid in topo.graph.nodes:
        dd = topo.graph.nodes[nid]
        if dd.get("ntype") == "object":
            nodes_dirs[dd.get("photo_id")].add(_base_dir(dd.get("direction", "")))
    for pid, dirs in nodes_dirs.items():
        for bdir in dirs:
            p = _map_img(photos_of.get(pid, {}), bdir)
            if p: paths.add(str(p))
    # 查詢影像路徑
    for key, v in gt.items():
        if v.get("ambiguous"): continue
        session, wp = key.rsplit("_wp", 1)
        for qdir, _objs, _ in load_query_photos(session, wp):
            hits = glob.glob(str(HERE/"testPDR"/session/"photos"/f"wp_{wp}_{qdir}_*.jpg"))
            if hits: paths.add(hits[0])
    paths = sorted(paths)
    print(f"要計算 DINOv2 embedding 的影像數: {len(paths)}", flush=True)

    import torch
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image, ImageOps
    print("載入 DINOv2 (facebook/dinov2-base)…", flush=True)
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base"); model.eval()

    cache = {}
    if DINO_CACHE.exists():
        try: cache = pickle.load(open(DINO_CACHE, "rb"))
        except Exception: cache = {}
    todo = [p for p in paths if p not in cache]
    print(f"尚未算的: {len(todo)}", flush=True)
    t0 = time.time()
    for i, p in enumerate(todo):
        try:
            im = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
            inp = proc(images=im, return_tensors="pt")
            with torch.no_grad():
                out = model(**inp)
            v = out.last_hidden_state[:, 0]        # CLS token
            v = v / v.norm(dim=-1, keepdim=True)
            cache[p] = v[0].numpy().astype(np.float32)
        except Exception as e:
            cache[p] = None
            print(f"  失敗 {p}: {e}", flush=True)
        if (i+1) % 50 == 0:
            print(f"  ...{i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
            DINO_CACHE.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(cache, open(DINO_CACHE, "wb"))
    DINO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cache, open(DINO_CACHE, "wb"))
    ok = sum(1 for v in cache.values() if v is not None)
    print(f"完成：{ok}/{len(cache)} 張有 embedding，存到 {DINO_CACHE}")


if __name__ == "__main__":
    main()
