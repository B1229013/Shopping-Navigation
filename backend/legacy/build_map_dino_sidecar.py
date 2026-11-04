"""build_map_dino_sidecar.py — 產生地圖 DINO 向量 sidecar（server 用，不需原圖）。
鍵＝photo_file（topo 節點 photos dict 裡的 'setXX/name.jpg' 字串）→ DINO 向量。
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server.topomap_v2 import TopoGraphV2
from eval_localization_real import TOPOMAP
from eval_visual import ARCHIVE

DINO_CACHE = HERE / "benchmark_results" / "realworld_0916" / "dino_cache.pkl"
SIDECAR = HERE / "carrefour_a7_dino.pkl"


def main():
    dino = pickle.load(open(DINO_CACHE, "rb"))  # keyed by archive abs path (str)
    topo = TopoGraphV2.load(str(TOPOMAP))
    sidecar = {}
    miss = 0
    for nid in topo.graph.nodes:
        dd = topo.graph.nodes[nid]
        if dd.get("ntype") != "photo":
            continue
        for slot, pf in (dd.get("photos") or {}).items():
            if not pf or "/" not in pf:
                continue
            fo, nm = pf.split("/", 1)
            apath = str(ARCHIVE / fo.replace("set", "") / nm)
            emb = dino.get(apath)
            if emb is not None:
                sidecar[pf] = emb
            else:
                miss += 1
    pickle.dump(sidecar, open(SIDECAR, "wb"))
    print(f"sidecar 存到 {SIDECAR}：{len(sidecar)} 張有向量，缺 {miss}")


if __name__ == "__main__":
    main()
