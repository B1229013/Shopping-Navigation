"""Real-photo test on test route 1: DINOv2 alone vs DINOv2 + aisle signs (+ sensor).

Photos 15–30 of 測試路線1 were taken walking down the signed central walkway toward the
entrance. Their approximate true positions come from the aisle signs in each photo
(columns node_a / node_b / fraction in eval/data/route1_labels.csv — estimated, please
check). Signs are read with EasyOCR (free, on this machine) and turned into evidence by
the server's own server/aisle_signs.py; the phone's sensor position is simulated around
the true position. No VLM calls (grid matching is left out).

  python -m eval.route1_eval --photos <dir> --out <dir> [--ocr-cache ocr.json] [--walk]

--walk adds the same walk table as eval/fusion_v2.py on these real photos: for each
photo a random walk of L metres back along the map gives the last photo fix, and the
phone's position is the true position plus Gaussian error (σ = 1 m + 5% × L, or twice
that). Rows: before (the previous weighted average), after (fuse_v2 + reachable spots,
with and without the aisle signs) and the sensor alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LABELS = Path(__file__).resolve().parent / "data" / "route1_labels.csv"


def _read_signs(photos, names, cache_path):
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text())
    import easyocr
    from PIL import Image, ImageOps
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    out = {}
    for name in names:
        im = ImageOps.exif_transpose(Image.open(Path(photos) / name)).convert("RGB")
        im.thumbnail((1600, 1600))
        res = reader.readtext(np.asarray(im), allowlist="0123456789")
        out[name] = {"size": im.size,
                     "items": [{"text": t, "conf": float(c), "box": [[float(x), float(y)] for x, y in b]}
                               for b, t, c in res]}
    if cache_path:
        Path(cache_path).write_text(json.dumps(out))
    return out


def main():
    from server import aisle_signs as AS
    from server.config import NEO4J_PLACE
    from server.loc_fusion import image_weight
    from server.neo4j_client import get_neo4j
    from eval.real_photos import _dino_topk, _ref_embeddings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ocr-cache")
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    global REPS
    REPS = args.reps
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(LABELS, encoding="utf-8")) if r.get("node_a")]
    m = get_neo4j().load_reference_map(NEO4J_PLACE)
    pos = {n: np.array([p.pdr_x, p.pdr_y]) for n, p in m.photos.items()}
    nbr = {n: set(p.neighbor_nids) for n, p in m.photos.items()}
    nodes = sorted(pos)
    idx = {n: i for i, n in enumerate(nodes)}
    xy = np.array([pos[n] for n in nodes])
    emb, owner = _ref_embeddings()
    signs = _read_signs(args.photos, [r["photo"] for r in rows], args.ocr_cache)
    rng = np.random.default_rng(0)

    results, per_photo = {}, []
    for r in rows:
        a = int(r["node_a"])
        b = int(r["node_b"]) if r.get("node_b") else None
        f = float(r.get("fraction") or 0)
        t = pos[a] + (f * (pos[b] - pos[a]) if b else 0)
        ok_set = {a} | ({b} if b else set())

        dino = _dino_topk(Path(args.photos) / r["photo"], emb, owner)
        vals = sorted(dino.values(), reverse=True)
        lo, hi = min(vals), max(vals)
        img = np.zeros(len(nodes))
        for n, v in dino.items():
            img[idx[n]] = 1.6 * min(1.0, (vals[0] - vals[1]) / 0.05) * (v - lo) / (hi - lo + 1e-9)

        s = signs[r["photo"]]
        ocr = [SimpleNamespace(text=i["text"], confidence=i["conf"], bbox=i["box"]) for i in s["items"]]
        numbers = AS.numbers_from_ocr(ocr, *s["size"])
        ev = AS.sign_evidence(numbers, m, NEO4J_PLACE)
        A = np.zeros(len(nodes))
        for n, v in ev.items():
            A[idx[n]] = v

        picks = {}
        for sig in [None, 2.0, 4.0, 8.0]:
            k = image_weight(sig)
            if sig is None:
                variants = {"DINOv2": k * img, "DINOv2 + signs": k * (img + A)}
            else:
                d2 = ((xy - (t + rng.normal(0, sig, 2))) ** 2).sum(1)
                variants = {"sensor": -d2, "DINOv2 + sensor": k * img - d2 / (2 * sig * sig),
                            "DINOv2 + signs + sensor": k * (img + A) - d2 / (2 * sig * sig)}
            for name, score in variants.items():
                top = nodes[int(np.argmax(score))]
                key = f"{name} | sensor {sig:g} m" if sig else f"{name} | no sensor"
                picks[key] = top
                results.setdefault(key, []).append((top in ok_set, top in ok_set or bool(nbr[top] & ok_set),
                                                    float(np.linalg.norm(pos[top] - t))))
        per_photo.append({"photo": r["photo"], "truth": [a, b, f], "truth_xy": t.tolist(),
                          "numbers": {str(k): v for k, v in numbers.items()}, "picks": picks})

    summary = {k: [round(100 * float(np.mean([x[0] for x in v])), 1), round(100 * float(np.mean([x[1] for x in v])), 1),
                   round(float(np.mean([x[2] for x in v])), 1)] for k, v in results.items()}
    for k, (r1, r2, e) in summary.items():
        print(f"{k:40s} right {r1:5.1f}%  right-or-next {r2:5.1f}%  error {e:4.1f} m")
    (out / "route1_results.json").write_text(json.dumps({"summary": summary, "photos": per_photo}, indent=1))
    _figures(args.photos, signs, per_photo, m, out)
    if args.walk:
        _walk_table(rows, signs, m, emb, owner, args.photos, out)


REPS = 20


def _walk_table(rows, signs, m, emb, owner, photos, out):
    import networkx as nx
    from server import aisle_signs as AS
    from server.config import NEO4J_PLACE
    from server.loc_fusion import image_weight
    from eval.real_photos import _dino_topk

    nodes = sorted(m.photos)
    idx = {n: i for i, n in enumerate(nodes)}
    pos = np.array([[m.photos[n].pdr_x, m.photos[n].pdr_y] for n in nodes])
    N = len(nodes)
    g = nx.Graph()
    for n in nodes:
        for k in m.photos[n].neighbor_nids:
            if k in idx:
                g.add_edge(idx[n], idx[k], w=float(np.linalg.norm(pos[idx[n]] - pos[idx[k]])))
    W = np.full((N, N), np.inf)
    for i, dd in nx.all_pairs_dijkstra_path_length(g, weight="w"):
        for j, v in dd.items():
            W[i, j] = v
    nbr = {i: set(g.neighbors(i)) for i in range(N)}

    P = []   # per photo: truth set, true xy, nearest node, D, gap factor, sign evidence
    for r in rows:
        a, b = int(r["node_a"]), (int(r["node_b"]) if r.get("node_b") else None)
        f = float(r.get("fraction") or 0)
        t = pos[idx[a]] + (f * (pos[idx[b]] - pos[idx[a]]) if b else 0)
        ok = {idx[a]} | ({idx[b]} if b else set())
        dino = _dino_topk(Path(photos) / r["photo"], emb, owner)
        vals = sorted(dino.values(), reverse=True)
        lo, hi = min(vals), max(vals)
        D = np.zeros(N)
        for n, v in dino.items():
            D[idx[n]] = (v - lo) / (hi - lo + 1e-9)
        fac = min(1.0, (vals[0] - vals[1]) / 0.05)
        sgn = signs[r["photo"]]
        ocr = [SimpleNamespace(text=i["text"], confidence=i["conf"], bbox=i["box"]) for i in sgn["items"]]
        ev = AS.sign_evidence(AS.numbers_from_ocr(ocr, *sgn["size"]), m, NEO4J_PLACE)
        A = np.zeros(N)
        for n, v in ev.items():
            A[idx[n]] = v
        near = int(np.argmin(np.linalg.norm(pos - t, axis=1)))
        strict = idx[a] if (b is None or f <= 0.5) else idx[b]
        area = "walkway" if "from aisle signs" in (r.get("notes") or "") else "front"
        P.append(dict(ok=ok, t=t, near=near, D=D, fac=fac, A=A, photo=r["photo"], truth=(a, b),
                      strict=strict, area=area))

    rng = np.random.default_rng(0)

    def walk_back(i, L):
        cur, prev, acc = i, None, 0.0
        while acc < L:
            nb = [j for j in g.neighbors(cur) if j != prev] or list(g.neighbors(cur))
            nxt = nb[rng.integers(len(nb))]
            acc += W[cur, nxt]
            prev, cur = cur, nxt
        return cur, acc

    table = {}
    map_picks = {}
    for drift in [1, 2]:
        for L in [0, 10, 20, 40, 80]:
            acc = {}
            for _ in range(REPS):
                for p in P:
                    f_node, Lw = walk_back(p["near"], L) if L > 0 else (p["near"], 0.0)
                    sig = 1 + 0.05 * Lw
                    sensor = p["t"] + rng.normal(0, sig * drift, 2)
                    d = np.linalg.norm(pos - sensor, axis=1)
                    reach = np.where(W[f_node] <= 1.5 * Lw + 8, 0.0, -3.0)
                    k = image_weight(sig)
                    img = 1.6 * p["fac"] * p["D"]
                    ws, wd = min(40, 40 / sig), 4 * p["fac"]
                    cand = p["D"] > 0
                    before = np.where(cand, (wd * p["D"] + ws * np.exp(-d / 8)) / (wd + ws), -np.inf)
                    # signs only: every node tied at the best sign score is equally likely
                    signs_only = p["A"] + (rng.random(N) * 1e-6 if p["A"].max() > 0 else rng.random(N))
                    variants = {
                        "dino": (img, None),
                        "signs": (signs_only, None),
                        "dino_signs": (img + p["A"], None),
                        "before": (before, None),
                        "after": (k * (img + p["A"]) - d ** 2 / (2 * sig ** 2) + reach, "post"),
                        "after_nosigns": (k * img - d ** 2 / (2 * sig ** 2) + reach, "post"),
                        "sensor": (-d, None),
                    }
                    if L == 10 and drift == 1 and _ == 0:
                        map_picks.setdefault(p["photo"], {})
                        for name, (sc, _kind) in variants.items():
                            map_picks[p["photo"]][name] = nodes[int(np.argmax(sc))]
                    for name, (sc, kind) in variants.items():
                        top = int(np.argmax(sc))
                        if kind == "post":
                            pr = np.exp(sc - sc.max())
                            conf = pr.max() / pr.sum()
                        else:
                            conf = float(np.clip(sc.max(), 0, 1)) if name == "before" else None
                        right = top in p["ok"]
                        rec = (right, right or bool(nbr[top] & p["ok"]), float(np.linalg.norm(pos[top] - p["t"])),
                               None if conf is None else (conf >= 0.6, right), top == p["strict"])
                        acc.setdefault(name, []).append(rec)
                        acc.setdefault(f"{name}@{p['area']}", []).append(rec)
            cell = {}
            for name, v in acc.items():
                applied = [x[3] for x in v if x[3] is not None and x[3][0]]
                cell[name] = [round(100 * float(np.mean([x[0] for x in v])), 1),
                              round(100 * float(np.mean([x[1] for x in v])), 1),
                              round(float(np.mean([x[2] for x in v])), 2),
                              round(100 * float(np.mean([x[1] for x in applied])), 1) if applied else None,
                              round(100 * float(np.mean([x[4] for x in v])), 1)]
            table[f"{L}m x{drift}"] = cell
            print(f"walked {L:2d} m drift x{drift}: " + " | ".join(
                f"{n} strict {c[4]:.0f} / either {c[0]:.0f} / nbr {c[1]:.0f} / {c[2]:.1f}m"
                for n, c in cell.items() if n.startswith("after")))
    (out / "route1_walk.json").write_text(json.dumps(table, indent=1))
    _method_maps(P, map_picks, m, pos, nodes, out)


def _method_maps(P, map_picks, m, pos, nodes, out):
    """Where each method put each real photo (walked 10 m since the last fix, normal drift)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.report_figures import GREEN, RED, _draw_map
    plt.rcParams["font.family"] = ["Heiti TC", "PingFang TC", "Arial Unicode MS", "sans-serif"]
    xy = {n: tuple(pos[i]) for i, n in enumerate(nodes)}
    edges = {tuple(sorted((a, b))) for a, nd in m.photos.items() for b in nd.neighbor_nids if b in xy}
    panels = [("dino", "① 只用 DINOv2\nDINOv2 only"), ("signs", "② 只用走道編號\nAisle signs only"),
              ("dino_signs", "③ DINOv2 + 走道編號\nDINOv2 + signs"),
              ("after", "④ 全部：DINOv2 + 走道編號 + 感測器\nAll: DINOv2 + signs + sensor")]
    walkway = [p for p in P if p["area"] == "walkway"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 9.5))
    for ax, (key, title) in zip(axes, panels):
        _draw_map(ax, xy, edges)
        right, errs = 0, []
        for p in walkway:
            t = p["t"]
            pick = map_picks[p["photo"]][key]
            ok = pick == nodes[p["strict"]]          # only the nearest spot counts
            right += ok
            q = np.array(xy[pick])
            errs.append(float(np.linalg.norm(q - t)))
            ax.scatter(*t, marker="*", s=170, color=GREEN if ok else "black", zorder=5)
            if not ok:
                ax.annotate("", xy=q, xytext=t, arrowprops=dict(arrowstyle="-|>", color=RED, alpha=0.6, lw=1.4), zorder=4)
        ax.set_title(f"{title}\n答對 right {right}/{len(walkway)} · 平均差 mean error {np.mean(errs):.1f} m", fontsize=12.5)
    fig.suptitle(f"中央走道 {len(walkway)} 張實拍，各方法猜的位置（④ 的感測器：上次定位後走 10 m）。星號＝真實位置（綠＝最近的點答對），紅箭頭＝指向錯誤答案\n"
                 f"{len(walkway)} real walkway photos, where each method put them (④ sensor: 10 m walked since the last fix). "
                 "Star = true position (green = nearest spot right), red arrow → wrong answer", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "route1_methods.png", dpi=100)
    plt.close(fig)

    # ④ only, every photo, strict scoring (only the nearest spot counts)
    fig, ax = plt.subplots(figsize=(11, 12))
    _draw_map(ax, xy, edges)
    right, errs = 0, []
    for p in P:
        t = p["t"]
        pick = map_picks[p["photo"]]["after"]
        ok = pick == nodes[p["strict"]]
        right += ok
        q = np.array(xy[pick])
        errs.append(float(np.linalg.norm(q - t)))
        ax.scatter(*t, marker="*", s=230, color=GREEN if ok else "black", zorder=5)
        ax.annotate(p["photo"].replace(".jpg", ""), t, xytext=(-14, 6), textcoords="offset points",
                    fontsize=9, color="#333333", zorder=6)
        if not ok:
            ax.annotate("", xy=q, xytext=t, arrowprops=dict(arrowstyle="-|>", color=RED, alpha=0.7, lw=1.5), zorder=4)
    ax.set_title(f"全店 {len(P)} 張實拍，現在的系統（DINOv2 + 走道編號 + 感測器，上次定位後走 10 m）\n"
                 f"All {len(P)} real photos, current system (10 m walked since the last fix): "
                 f"exactly right {right}/{len(P)} · mean error {np.mean(errs):.1f} m\n"
                 "星號＝真實位置（綠＝完全答對，只算最近的點）　紅箭頭＝指向錯誤答案　數字＝照片編號", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "route1_all_map.png", dpi=105)
    plt.close(fig)


def _figures(photos, signs, per_photo, m, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    from eval.report_figures import BLUE, GREEN, GREY, RED, _draw_map
    plt.rcParams["font.family"] = ["Heiti TC", "PingFang TC", "Arial Unicode MS", "sans-serif"]
    from server import aisle_signs as AS

    # 1. what the sign reader saw in each photo
    n = len(per_photo)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.3 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, p in zip(axes.flat, per_photo):
        im = ImageOps.exif_transpose(Image.open(Path(photos) / p["photo"])).convert("RGB")
        im.thumbnail((1600, 1600))
        w, h = im.size
        ax.imshow(im.crop((0, 0, w, int(h * 0.62))))
        nums = {int(k): v for k, v in p["numbers"].items()}
        k = max(range(1, 16), key=lambda k: nums.get(2 * k - 1, 0) + nums.get(2 * k, 0)) if nums else None
        for it in signs[p["photo"]]["items"]:
            xs = [q[0] for q in it["box"]]
            ys = [q[1] for q in it["box"]]
            try:
                v = int(it["text"])
            except ValueError:
                continue
            if v not in nums:
                continue
            near = k and v in (2 * k - 1, 2 * k)
            ax.add_patch(plt.Rectangle((min(xs), min(ys)), max(xs) - min(xs), max(ys) - min(ys), fill=False,
                                       ec=GREEN if near else BLUE, lw=2.5 if near else 1.2))
        pair = f"{2 * k - 1} | {2 * k}" if k else "—"
        ax.set_title(f"{p['photo']}  最近的牌 nearest {pair}  · 讀到 {len(nums)} 個號碼", fontsize=10.5)
    fig.suptitle("自動讀到的走道編號（綠框＝最近的一對，決定在哪一排；讀到 3 個以上＝在中央走道）\n"
                 "Aisle numbers read automatically (green = nearest pair → row; 3+ numbers → central walkway)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "route1_signs.jpg", dpi=85, pil_kwargs={"quality": 82})
    plt.close(fig)

    # 2. where each method put each photo
    nodes = {nid: (p.pdr_x, p.pdr_y) for nid, p in m.photos.items()}
    edges = {tuple(sorted((a, b))) for a, p in m.photos.items() for b in p.neighbor_nids if b in nodes}
    fig, axes = plt.subplots(1, 2, figsize=(15, 9.5))
    for ax, key, title in [(axes[0], "DINOv2 | no sensor", "只用 DINOv2  DINOv2 only"),
                           (axes[1], "DINOv2 + signs | no sensor", "DINOv2 + 走道編號  DINOv2 + aisle signs")]:
        _draw_map(ax, nodes, edges)
        right = 0
        for p in per_photo:
            t = np.array(p["truth_xy"])
            pick = p["picks"][key]
            q = np.array(nodes[pick])
            ok = pick in {p["truth"][0], p["truth"][1]}
            right += ok
            ax.scatter(*t, marker="*", s=160, color=GREEN if ok else "black", zorder=5)
            if not ok:
                ax.annotate("", xy=q, xytext=t, arrowprops=dict(arrowstyle="-|>", color=RED, alpha=0.6, lw=1.4), zorder=4)
        err = np.mean([np.linalg.norm(np.array(nodes[p["picks"][key]]) - np.array(p["truth_xy"])) for p in per_photo])
        ax.set_title(f"{title}\n答對 right {right}/{len(per_photo)} · 平均差 mean error {err:.1f} m", fontsize=13)
    fig.suptitle("測試路線 1 的 16 張實拍（沒有感測器）：星號＝真實位置（綠＝答對），紅箭頭＝指向錯誤答案\n"
                 "16 real route-1 photos, no sensor: star = true position (green = right), red arrow → wrong answer",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "route1_maps.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
