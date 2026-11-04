"""eval_fusion_v2_compare.py — 把 iOS fusion-v2 的「影像 + 感測器 + 可達性」機率融合，
套在我們 0916 家樂福 268 張真實測試照片上，比較不同影像分數的效果。

影像分數來自快取 benchmark_results/realworld_0916/dino_sigs.pkl（DINO 視覺、物件、OCR，
每個 node×方向一筆），與 locate_fusion 線上版同一套訊號（Top-1 可重現 65.7%）。

感測器是模擬的（照 iOS eval/fusion_v2.py 的做法）：
  - 上次定位點 = 從正解節點沿走道隨機往回走 L 公尺
  - 手機位置 = 真實拍照位置 test_xy + 高斯誤差，σ = 1 + 0.05·L（drift 2 = 實際漂移是假設的兩倍）

fuse_v2（iOS backend/server/loc_fusion.py）：
  score_n = k(σ)·image_n − d_n²/(2σ²) − 3·[n 從上次定位點走不到]，取最大者
  k(σ) = 0.25（σ≤2m）… 4.0（σ≥5m）；沒有感測器時 = 2.0

另外用網格搜尋 + 按正解節點分組的 5 折交叉驗證重新調 k(σ) 的上下限（k_min, k_max），
表中「CV 調 k」那一列是 out-of-fold 結果（每張照片用沒看過它的那 4 折選出的參數）。

  venv/Scripts/python.exe eval_fusion_v2_compare.py [--reps 20]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_localization_real import GT_FILE
from eval_localization_single import load_query_photos

SIGS = HERE / "benchmark_results" / "realworld_0916" / "dino_sigs.pkl"
TOPO = HERE / "carrefour_a7_imported.json"

# ── iOS loc_fusion.py 常數 ──
IMAGE_K_MIN, IMAGE_K_MAX, IMAGE_K_NO_SENSOR, REACH_PENALTY = 0.25, 4.0, 2.0, 3.0


def image_weight(sigma):
    if sigma is None:
        return IMAGE_K_NO_SENSOR
    return min(IMAGE_K_MAX, max(IMAGE_K_MIN, IMAGE_K_MIN + (IMAGE_K_MAX - IMAGE_K_MIN) * (sigma - 2.0) / 3.0))


def mm(a):
    r = a.max() - a.min()
    return (a - a.min()) / r if r > 0 else np.zeros_like(a)


def load():
    ds = pickle.load(open(SIGS, "rb"))
    pids = np.array(ds["pids"])
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    txy = []
    for key, v in gt.items():
        if v.get("ambiguous"):
            continue
        s, wp = key.rsplit("_wp", 1)
        for _ in load_query_photos(s, wp):
            txy.append((v["node_id"], v["test_xy"]))
    assert [t[0] for t in txy] == [d[0] for d in ds["data"]], "查詢順序與快取對不上"

    topo = json.load(open(TOPO, encoding="utf-8"))
    photo = {n["id"]: n for n in topo["nodes"] if n.get("ntype") == "photo"}
    node_ids = sorted(photo)
    idx = {n: i for i, n in enumerate(node_ids)}
    N = len(node_ids)
    xy = np.array([[photo[n]["sensor_data"]["pdr_x"], photo[n]["sensor_data"]["pdr_y"]] for n in node_ids])
    g = nx.Graph()
    g.add_nodes_from(range(N))
    for e in topo["edges"]:
        if e.get("etype") == "walkway":
            a, b = idx[e["from"]], idx[e["to"]]
            g.add_edge(a, b, w=float(np.linalg.norm(xy[a] - xy[b])))
    W = np.full((N, N), np.inf)
    for i, dd in nx.all_pairs_dijkstra_path_length(g, weight="w"):
        for j, v in dd.items():
            W[i, j] = v
    nbr = np.eye(N, dtype=bool)
    for a, b in g.edges:
        nbr[a, b] = nbr[b, a] = True

    # 每張查詢 → 每個節點的訊號（節點內各方向取最大）
    Q = len(ds["data"])
    P = np.array([idx[p] for p in pids])
    vis = np.full((Q, N), -np.inf); nod = np.full((Q, N), -np.inf); fused = np.full((Q, N), -np.inf)
    for q, (_, v, n, o) in enumerate(ds["data"]):
        f = 5 * mm(v) + 0.5 * mm(n) + 0.5 * o           # locate_fusion 線上版
        for arr, src in ((vis[q], v), (nod[q], mm(n)), (fused[q], f)):
            np.maximum.at(arr, P, src)
    truth = np.array([idx[t[0]] for t in txy])
    true_xy = np.array([t[1] for t in txy], float)
    return dict(xy=xy, g=g, W=W, nbr=nbr, vis=vis, nod=nod, fused=fused, truth=truth, txy=true_xy)


def evidences(st):
    """各種『影像證據』（單張照片、不含感測器），範圍都大約 0..2，對齊 iOS image_evidence。"""
    vis = st["vis"]
    Q, N = vis.shape
    # iOS 原式：DINO 只取前 10 名做 min-max，再乘「第一名與第二名差距/0.05」（上限 1），×1.6
    dino_v2 = np.zeros((Q, N))
    for q in range(Q):
        order = np.argsort(-vis[q])[:10]
        top = vis[q, order]
        gap = top[0] - top[1]
        dino_v2[q, order] = 1.6 * min(1.0, max(gap, 0) / 0.05) * mm(top)
    ios_like = dino_v2 + 0.4 * st["nod"]              # 物件分數代替 iOS 的格網分數
    ours = 2 * np.array([mm(r) for r in st["fused"]])  # 你的融合，縮放到 0..2
    return {"iOS 影像式(DINO+物件)": ios_like, "你的融合(5:0.5:0.5)": ours}


def walk_back(g, xy, rng, t, L):
    cur, prev, acc = t, None, 0.0
    while acc < L:
        nb = [j for j in g.neighbors(cur) if j != prev] or list(g.neighbors(cur))
        if not nb:
            break
        nxt = nb[rng.integers(len(nb))]
        acc += float(np.linalg.norm(xy[cur] - xy[nxt]))
        prev, cur = cur, nxt
    return cur, acc


def metrics(st, score):
    top = score.argmax(1)
    ok = top == st["truth"]
    near = ok | st["nbr"][st["truth"], top]
    err = np.linalg.norm(st["xy"][top] - st["txy"], axis=1)
    return np.array([ok.mean() * 100, near.mean() * 100, err.mean()])


def per_query(st, score):
    """每張查詢的 (答對, 正解或相鄰, 誤差 m)，給交叉驗證用。"""
    top = score.argmax(1)
    ok = top == st["truth"]
    near = ok | st["nbr"][st["truth"], top]
    err = np.linalg.norm(st["xy"][top] - st["txy"], axis=1)
    return np.stack([ok * 100.0, near * 100.0, err], 1)


SCENARIOS = [(L, drift) for drift in (1, 2) for L in (0, 10, 20, 40, 80)]


def simulate(st, L, drift, rng):
    """一次模擬：感測器分數、可達性懲罰、σ（與影像權重無關，可重複套不同 k）。"""
    truth, xy, txy = st["truth"], st["xy"], st["txy"]
    prev = [walk_back(st["g"], xy, rng, t, L) if L > 0 else (t, 0.0) for t in truth]
    f = np.array([p[0] for p in prev]); Lw = np.array([p[1] for p in prev])
    sig = 1 + 0.05 * Lw
    pos = txy + rng.normal(0, 1, txy.shape) * (sig * drift)[:, None]
    d = np.linalg.norm(pos[:, None, :] - xy[None], axis=2)
    return dict(d=d, sig=sig,
                sensor=-d ** 2 / (2 * sig[:, None] ** 2),
                reach=np.where(st["W"][f] <= 1.5 * Lw[:, None] + 8, 0.0, -REACH_PENALTY))


def k_of(sig, k_min, k_max):
    """iOS image_weight 的形狀：σ≤2 m 時 k_min，σ≥5 m 時 k_max，中間線性。"""
    return np.clip(k_min + (k_max - k_min) * (sig - 2.0) / 3.0, min(k_min, k_max), max(k_min, k_max))[:, None]


def fused_score(sim, img, k_min=IMAGE_K_MIN, k_max=IMAGE_K_MAX, reach=True):
    s = sim["sensor"] + k_of(sim["sig"], k_min, k_max) * img
    return s + sim["reach"] if reach else s


# ── k(σ) 網格搜尋 + 按節點分組的 5 折交叉驗證 ──
K_MIN_GRID = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
K_MAX_GRID = [4, 8, 16, 32, 64, 128, 256, 512]
N_FOLDS = 5


def node_folds(truth, n_folds=N_FOLDS, seed=0):
    """同一個正解節點的照片放同一折（避免同地點照片同時出現在訓練與驗證）。"""
    nodes = np.unique(truth)
    np.random.default_rng(seed).shuffle(nodes)
    fold_of_node = {n: i % n_folds for i, n in enumerate(nodes)}
    return np.array([fold_of_node[t] for t in truth])


def tune_k(st, img, sims):
    """回傳 (各組合, 每組合每情境每查詢的 [答對, 相鄰, 誤差])。"""
    combos = [(a, b) for a in K_MIN_GRID for b in K_MAX_GRID if b >= a]
    table = {}
    for c in combos:
        table[c] = {sc: np.mean([per_query(st, fused_score(s, img, *c)) for s in reps], 0)
                    for sc, reps in sims.items()}
    return combos, table


def objective(table, c, mask):
    """所有情境平均 Top-1，誤差當次要（同分時選誤差小的）。"""
    m = np.mean([v[mask] for v in table[c].values()], axis=(0, 1))
    return m[0] - 0.01 * m[2]


def cross_validate(st, table, combos):
    folds = node_folds(st["truth"])
    chosen = np.empty(len(st["truth"]), dtype=object)
    report = []
    for k in range(N_FOLDS):
        tr, va = folds != k, folds == k
        best = max(combos, key=lambda c: objective(table, c, tr))
        for q in np.where(va)[0]:
            chosen[q] = best
        val = np.mean([v[va, 0].mean() for v in table[best].values()])
        report.append((k, best, int(va.sum()), val))
    oof = {sc: np.mean([table[chosen[q]][sc][q] for q in range(len(chosen))], 0) for sc in SCENARIOS}
    full_best = max(combos, key=lambda c: objective(table, c, np.ones(len(st["truth"]), bool)))
    return report, oof, full_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", default=str(HERE / "benchmark_results" / "realworld_0916" / "fusion_v2_compare.json"))
    args = ap.parse_args()
    st = load()
    ev = evidences(st)
    ours = ev["你的融合(5:0.5:0.5)"]
    floor = np.linalg.norm(st["xy"][st["truth"]] - st["txy"], axis=1).mean()
    print(f"{len(st['truth'])} 張測試照片；就算選中正解節點，平均誤差也有 {floor:.2f} m（測試點離節點的距離）")
    print("格式：Top-1 % / 正解或相鄰節點 % / 平均誤差 m\n")
    res = {"no sensor": {}}
    print("【沒有感測器（第一次定位）】")
    for name, img in ev.items():
        m = metrics(st, img); res["no sensor"][name] = m.round(2).tolist()
        print(f"  {name:28s} {m[0]:5.1f} / {m[1]:5.1f} / {m[2]:5.2f}")

    # 模擬只做一次，所有變體與 k 組合共用同一批隨機情境
    rng = np.random.default_rng(0)
    sims = {(L, drift): [simulate(st, L, drift, rng) for _ in range(args.reps)] for L, drift in SCENARIOS}

    print(f"\n調 k(σ)：k_min ∈ {K_MIN_GRID}，k_max ∈ {K_MAX_GRID}，"
          f"{N_FOLDS} 折（按正解節點分組），目標 = 10 個情境平均 Top-1")
    combos, table = tune_k(st, ours, sims)
    report, oof, full_best = cross_validate(st, table, combos)
    for k, best, n, val in report:
        print(f"  第 {k + 1} 折：選 k_min={best[0]:g}, k_max={best[1]:g}　驗證 {n} 張，平均 Top-1 {val:.1f}%")
    print(f"  全部資料最佳：k_min={full_best[0]:g}, k_max={full_best[1]:g}"
          f"（iOS 原值 k_min={IMAGE_K_MIN:g}, k_max={IMAGE_K_MAX:g}）")
    res["cv"] = {"folds": [{"fold": k + 1, "k_min": b[0], "k_max": b[1], "n": n, "val_top1": round(v, 2)}
                           for k, b, n, v in report],
                 "full_data_best": {"k_min": full_best[0], "k_max": full_best[1]}}

    cv_name = "你的融合 +感測器+可達 (CV 調 k)"
    for L, drift in SCENARIOS:
        reps = sims[(L, drift)]
        rows = {"只用感測器": np.mean([metrics(st, -s["d"]) for s in reps], 0)}
        for name, img in ev.items():
            rows[f"{name} +感測器"] = np.mean([metrics(st, fused_score(s, img, reach=False)) for s in reps], 0)
            rows[f"{name} +感測器+可達"] = np.mean([metrics(st, fused_score(s, img)) for s in reps], 0)
        rows[cv_name] = oof[(L, drift)]
        key = f"{L}m x{drift}"
        res[key] = {n: m.round(2).tolist() for n, m in rows.items()}
        print(f"\n【上次定位後走 {L} m，漂移 ×{drift}，σ≈{1 + 0.05 * L:.1f} m】")
        for n, m in rows.items():
            print(f"  {n:32s} {m[0]:5.1f} / {m[1]:5.1f} / {m[2]:5.2f}")

    def avg(name):
        return np.mean([res[f"{L}m x{d}"][name] for L, d in SCENARIOS], 0)
    print("\n【10 個情境平均】")
    for n in ("只用感測器", "你的融合(5:0.5:0.5) +感測器+可達", cv_name):
        m = avg(n)
        print(f"  {n:32s} {m[0]:5.1f} / {m[1]:5.1f} / {m[2]:5.2f}")
    m = res["no sensor"]["你的融合(5:0.5:0.5)"]
    print(f"  {'只用影像（你的融合）':32s} {m[0]:5.1f} / {m[1]:5.1f} / {m[2]:5.2f}")

    Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n結果存到 {args.json}")


if __name__ == "__main__":
    main()
