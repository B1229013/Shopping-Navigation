"""Result pictures for the localization / sensor report (bilingual labels).

  python -m eval.report_figures --cache <fusion_sweep cache.json> --photos <walk photo dir>
                                --walk-sim <compiled scripts/swift_logic_tests/walk_sim> --out <dir>

Writes:
  map_examples.png     held-out photos on the store map: true node vs 5:1:1 vs adaptive
  dino_ambiguity.png   real walk photos and where DINOv2's top-5 nodes are on the map
  photo_quality.png    the server's blur / darkness check on a real walk photo
  occlusion.png        the occlusion check on real walk photos (people found by a local
                       COCO detector standing in for the server's detector)
  walk_correction.png  a simulated walk through the phone's MapPositionEstimator:
                       drifting PDR vs photo-corrected position
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # noqa: E402

plt.rcParams["font.family"] = ["Heiti TC", "PingFang TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

GREEN, RED, BLUE, ORANGE, GREY = "#1a9850", "#d73027", "#2166ac", "#f28e2b", "#9e9e9e"


def _graph(cache):
    nodes = {int(n): (v["x"], v["y"]) for n, v in cache["nodes"].items()}
    edges = {tuple(sorted((int(n), m))) for n, v in cache["nodes"].items() for m in v["nbrs"] if m in nodes}
    return nodes, edges


def _draw_map(ax, nodes, edges, label_nodes=()):
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color="#dddddd", lw=1.2, zorder=1)
    xy = np.array(list(nodes.values()))
    ax.scatter(xy[:, 0], xy[:, 1], s=10, color="#bdbdbd", zorder=2)
    for n in label_nodes:
        ax.annotate(f"WP{n}", nodes[n], xytext=(4, 4), textcoords="offset points", fontsize=8, color="#555555")
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ── 1. held-out examples on the map ────────────────────────────────────────

def _picks_state(cache, sigma=2.0):
    """Fixed 5:1:1 vs adaptive rankings for every held-out photo at one sensor error."""
    from eval.fusion_sweep import _prepare
    from server.loc_fusion import dino_gap
    node_ids, xy, G, D, gmask, dmask, truth, dist, nbr, txy = _prepare(cache)
    rng = np.random.default_rng(1000 * 0 + int(sigma * 10))
    pos = txy + rng.normal(0, sigma, size=(len(truth), 2))
    S = np.exp(-np.linalg.norm(pos[:, None, :] - xy[None, :, :], axis=2) / 8.0)
    cand = gmask | dmask

    def rank(wd, ws, wg):
        wd = np.broadcast_to(np.asarray(wd, float), (len(truth),))[:, None]
        sc = (wg * G + wd * D + ws * S) / (wg + wd + ws)
        return np.where(cand, sc, -np.inf)

    gaps = np.array([dino_gap({int(k): v for k, v in e["dino"].items()}) or 0 for e in cache["queries"]])
    return dict(node_ids=node_ids, xy=xy, truth=truth, dist=dist, nbr=nbr, pos=pos,
                fixed=rank(5, 1, 1), adapt=rank(4 * np.clip(gaps / 0.05, 0, 1), min(40, 40 / sigma), 1))


def error_map(cache, out):
    st = _picks_state(cache)
    nodes, edges = _graph(cache)
    xy, truth, dist = st["xy"], st["truth"], st["dist"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 9.5))
    stats = {}
    for ax, key, name in [(axes[0], "fixed", "固定 5:1:1 Fixed 5:1:1"), (axes[1], "adapt", "自適應權重 Adaptive")]:
        pick = st[key].argmax(1)
        ok = pick == truth
        _draw_map(ax, nodes, edges)
        for q in np.where(~ok)[0]:
            a, b = xy[truth[q]], xy[pick[q]]
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=RED, alpha=0.35, lw=1.1,
                                                              shrinkA=0, shrinkB=2), zorder=3)
        hits = np.bincount(truth[ok], minlength=len(xy))
        nz = hits > 0
        ax.scatter(xy[nz, 0], xy[nz, 1], s=25 + 40 * hits[nz], color=GREEN, alpha=0.85, zorder=4)
        err = dist[truth, pick]
        stats[key] = (int(ok.sum()), float(err.mean()))
        ax.set_title(f"{name}\n答對 correct {ok.sum()}/{len(truth)}（{100 * ok.mean():.0f}%）· 平均差 mean error {err.mean():.1f} m",
                     fontsize=13)
    fig.suptitle("所有 260 張照片：綠點＝答對（越大越多張），紅箭頭＝從真實位置指到錯誤答案\n"
                 "All 260 held-out photos: green = correct (bigger = more photos), red arrow = true position → wrong answer "
                 "(sensor error 2 m)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "error_map.png", dpi=110)
    plt.close(fig)
    return stats


def map_examples(cache, out, name="map_examples.png", n_good=3, n_bad=1, skip_good=0, skip_bad=0, cols=2):
    st = _picks_state(cache)
    node_ids, xy, truth, dist, nbr, pos = (st[k] for k in ("node_ids", "xy", "truth", "dist", "nbr", "pos"))
    fixed, adapt = st["fixed"], st["adapt"]
    nodes, edges = _graph(cache)
    f1, a1 = fixed.argmax(1), adapt.argmax(1)
    good = [q for q in range(len(truth)) if a1[q] == truth[q] and f1[q] != truth[q] and dist[truth[q], f1[q]] > 8]
    bad = [q for q in range(len(truth)) if a1[q] != truth[q] and not nbr[truth[q], a1[q]]]
    uniq, seen = [], set()
    for q in good:
        if truth[q] not in seen:
            uniq.append(q)
            seen.add(truth[q])
    picks = uniq[skip_good:skip_good + n_good] + bad[skip_bad:skip_bad + n_bad]
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.75 * rows))
    for ax, q in zip(axes.flat, picks):
        t, f, a = node_ids[truth[q]], node_ids[f1[q]], node_ids[a1[q]]
        order = np.argsort(-adapt[q])[:4]
        sc = adapt[q, order]
        w = np.clip(sc[:3] - sc[3], 0, None)
        est = (w[:, None] * xy[order[:3]]).sum(0) / w.sum() if w.sum() > 0 else xy[order[0]]
        _draw_map(ax, nodes, edges, label_nodes={t, f, a})
        ax.scatter(*xy[truth[q]], marker="*", s=420, color=GREEN, zorder=5, label="真實位置 True position")
        ax.scatter(*pos[q], marker="^", s=110, color=ORANGE, zorder=4, label="感測器位置（誤差 2 m）Sensor (2 m error)")
        ax.scatter(*xy[f1[q]], marker="X", s=200, color=RED, zorder=6, label="固定 5:1:1 的答案 Fixed 5:1:1")
        ax.scatter(*xy[a1[q]], marker="o", s=300, facecolors="none", edgecolors=BLUE, linewidths=2.5, zorder=6,
                   label="自適應權重的答案 Adaptive")
        ax.scatter(*est, marker="D", s=60, color=BLUE, zorder=7, label="節點間估計位置 Estimated position")
        photo = cache["queries"][q]["photo"].split("/")[-1]
        slot = photo.rsplit("_", 1)[-1].replace(".jpg", "")
        ef, ea = dist[truth[q], f1[q]], dist[truth[q], a1[q]]
        verdict = "✓" if a == t else "✗"
        ax.set_title(f"WP{t} 的 {slot} 照片 photo\n5:1:1 → WP{f}（差 {ef:.0f} m）   自適應 Adaptive → WP{a}（差 {ea:.0f} m）{verdict}",
                     fontsize=11)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle("同一張照片，固定 5:1:1 與自適應權重的定位結果\nSame photo, fixed 5:1:1 vs adaptive weights "
                 "(held-out reference photo, sensor error 2 m)", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out / name, dpi=110 if cols > 2 else 130)
    plt.close(fig)
    return [(cache["queries"][q]["photo"], node_ids[truth[q]], node_ids[f1[q]], node_ids[a1[q]]) for q in picks]


# ── 2. DINOv2 ambiguity on real walk photos ────────────────────────────────

def dino_ambiguity(cache, photos_dir, names, out):
    from eval.real_photos import _dino_topk, _ref_embeddings
    nodes, edges = _graph(cache)
    emb, owner = _ref_embeddings()
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 5.2 * len(names)),
                             gridspec_kw={"width_ratios": [0.75, 1.25]})
    rows = []
    for (ax_img, ax_map), name in zip(np.atleast_2d(axes), names):
        p = Path(photos_dir) / name
        im = ImageOps.exif_transpose(Image.open(p))
        im.thumbnail((900, 900))
        ax_img.imshow(im)
        ax_img.axis("off")
        top = list(_dino_topk(p, emb, owner).items())[:5]
        gap = top[0][1] - top[1][1]
        ax_img.set_title(f"測試路線1 {name}", fontsize=12)
        _draw_map(ax_map, nodes, edges)
        for rank, (n, s) in enumerate(top):
            ax_map.scatter(*nodes[n], s=380 - 60 * rank, color=BLUE, alpha=1 - 0.15 * rank, zorder=5)
            # another candidate just to the right on the same row → put this label on the left
            crowded = any(0 < nodes[m][0] - nodes[n][0] < 15 and abs(nodes[m][1] - nodes[n][1]) < 2
                          for m, _ in top if m != n)
            ax_map.annotate(f"#{rank + 1} WP{n}  {s:.3f}", nodes[n], xytext=(-10 if crowded else 10, -4),
                            textcoords="offset points", fontsize=10, color="#1f3b5c", zorder=6,
                            ha="right" if crowded else "left",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        spread = max(np.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]) for a, _ in top for b, _ in top)
        ax_map.set_title(f"DINOv2 前五名：第一、二名只差 {gap:.3f}，最遠相隔 {spread:.0f} m\n"
                         f"Top 5: #1 beats #2 by {gap:.3f}; candidates up to {spread:.0f} m apart", fontsize=11)
        rows.append((name, top, gap, spread))
    fig.suptitle("DINOv2 在真實照片上常常分不出是哪一個節點\nDINOv2 often cannot tell nodes apart on real walk photos",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "dino_ambiguity.png", dpi=120)
    plt.close(fig)
    return rows


# ── 3. blur / darkness check ───────────────────────────────────────────────

def photo_quality_fig(photos_dir, name, out):
    from server.photo_quality import MIN_BRIGHTNESS, MIN_SHARPNESS, assess_image
    src = ImageOps.exif_transpose(Image.open(Path(photos_dir) / name)).convert("RGB")
    variants = [("原圖 Original", src),
                ("輕微模糊 Slight blur", src.filter(ImageFilter.GaussianBlur(6))),
                ("嚴重模糊 Heavy blur", src.filter(ImageFilter.GaussianBlur(12))),
                ("太暗 Too dark", ImageEnhance.Brightness(src).enhance(0.25))]
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.4))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for ax, (label, im) in zip(axes, variants):
            f = Path(tmp) / "v.jpg"
            im.save(f, "JPEG", quality=90)
            q = assess_image(str(f))
            show = im.copy()
            show.thumbnail((700, 700))
            ax.imshow(show)
            ax.axis("off")
            ok = q.hint is None
            ax.set_title(f"{label}\n清晰度 sharpness {q.sharpness:.0f} · 亮度 brightness {q.brightness:.0f}",
                         fontsize=11)
            ax.text(0.5, -0.04, "通過 PASS" if ok else f"重拍 RETAKE\n{q.hint}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=11, color=GREEN if ok else RED, fontweight="bold")
            rows.append((label, q.sharpness, q.brightness, q.hint))
    fig.suptitle(f"伺服器的模糊／亮度檢查（門檻：清晰度 < {MIN_SHARPNESS:.0f} 或亮度 < {MIN_BRIGHTNESS:.0f} 就要求重拍）\n"
                 f"Server blur / darkness check on walk photo {name} (retake below sharpness "
                 f"{MIN_SHARPNESS:.0f} or brightness {MIN_BRIGHTNESS:.0f})", fontsize=13)
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    fig.savefig(out / "photo_quality.png", dpi=120)
    plt.close(fig)
    return rows


# ── 4. occlusion ───────────────────────────────────────────────────────────

def occlusion_fig(photos_dir, out):
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from types import SimpleNamespace
    from server.photo_quality import MAX_OCCLUDED_FRACTION, assess_occlusion
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.6).eval()
    prep = weights.transforms()

    def people(im):
        with torch.no_grad():
            r = model([prep(im)])[0]
        return [b.tolist() for b, l in zip(r["boxes"], r["labels"]) if int(l) == 1]

    scored = []
    for p in sorted(Path(photos_dir).glob("*.jpg"), key=lambda p: (len(p.stem), p.stem)):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.thumbnail((1000, 1000))
        boxes = people(im)
        occ = assess_occlusion([SimpleNamespace(label="person", box=b) for b in boxes], *im.size)
        scored.append((occ.fraction, p.name, im, boxes))
    scored.sort(key=lambda r: -r[0])
    frac, name, im, boxes = scored[0]
    # the same scene with the person close to the camera: crop around the largest person
    if boxes:
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cw, ch = (x2 - x1) * 1.6, (y2 - y1) * 1.25
        cx = (x1 + x2) / 2
        crop = im.crop((max(0, cx - cw / 2), max(0, y1 - 0.1 * ch), min(im.width, cx + cw / 2), min(im.height, y1 + ch)))
    else:
        crop = im
    cases = [(f"{name}：照片中人最多的一張\nWalk photo with the most people", im),
             ("同一張裁切成人站在鏡頭前（模擬）\nSame photo cropped: person right in front (simulated)", crop)]
    clear = next(r for r in reversed(scored) if r[0] == 0)
    cases.insert(0, (f"{clear[1]}：沒有人\nNo one in view", clear[2]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    rows = []
    for ax, (label, img) in zip(axes, cases):
        bxs = people(img)
        occ = assess_occlusion([SimpleNamespace(label="person", box=b) for b in bxs], *img.size)
        ax.imshow(img)
        for b in bxs:
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=RED, lw=2.5))
        ax.axis("off")
        bad = occ.fraction >= MAX_OCCLUDED_FRACTION
        ax.set_title(label, fontsize=11)
        ax.text(0.5, -0.03, f"人佔畫面 people cover {100 * occ.fraction:.0f}%\n"
                + ("重拍：畫面被擋住，不用來校正位置\nRETAKE — not used to correct position" if bad
                   else "通過 PASS"), transform=ax.transAxes, ha="center", va="top", fontsize=11,
                color=RED if bad else GREEN, fontweight="bold")
        rows.append((label.split("\n")[0], len(bxs), occ.fraction, bad))
    fig.suptitle(f"遮擋檢查：人或推車蓋住 ≥ {100 * MAX_OCCLUDED_FRACTION:.0f}% 畫面就要求重拍\n"
                 f"Occlusion check: retake when people or carts cover ≥ {100 * MAX_OCCLUDED_FRACTION:.0f}% "
                 "of the frame (red boxes: detected people)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out / "occlusion.png", dpi=110)
    plt.close(fig)
    return rows


# ── 5. simulated walk through MapPositionEstimator ─────────────────────────

def walk_fig(cache, sim_bin, out, bias=10.0, scale=0.9, fix_every_m=12.0):
    import networkx as nx
    nodes, edges = _graph(cache)
    g = nx.Graph()
    for a, b in edges:
        g.add_edge(a, b, w=float(np.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])))
    lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="w"))
    a, b = max(((a, b) for a in lengths for b in lengths[a]), key=lambda ab: lengths[ab[0]][ab[1]])
    route = nx.dijkstra_path(g, a, b, weight="w")
    verts = [nodes[n] for n in route]
    fixes, acc = [0], 0.0
    for i in range(1, len(verts)):
        acc += np.hypot(verts[i][0] - verts[i - 1][0], verts[i][1] - verts[i - 1][1])
        if acc >= fix_every_m:
            fixes.append(i)
            acc = 0.0

    def run(fix_idx):
        inp = "\n".join(f"{x},{y}" for x, y in verts) + "\n" + "\n".join(f"FIX {i}" for i in fix_idx) + "\n"
        txt = subprocess.run([sim_bin, str(bias), str(scale), "7"], input=inp, capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()
        cols = txt[0].split(",")
        return [dict(zip(cols, l.split(","))) for l in txt[1:]]

    with_fix, start_only = run(fixes), run([0])
    T = np.array([[float(r["true_x"]), float(r["true_y"])] for r in with_fix])
    E = np.array([[float(r["est_x"]), float(r["est_y"])] for r in with_fix])
    R = np.array([[float(r["est_x"]), float(r["est_y"])] for r in start_only])
    fx = np.array([[float(r["true_x"]), float(r["true_y"])] for r in with_fix if r["fix"] == "1"])
    walked = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(T, axis=0), axis=1))])
    err_raw, err_fix = np.linalg.norm(R - T, axis=1), np.linalg.norm(E - T, axis=1)

    fig, ax = plt.subplots(figsize=(11, 12.5))
    _draw_map(ax, nodes, edges)
    ax.plot(T[:, 0], T[:, 1], color="black", lw=2.5, label="真實路徑 True path", zorder=3)
    ax.plot(R[:, 0], R[:, 1], color=RED, lw=2, ls="--",
            label=f"只靠感測器（方向偏 {bias:.0f}°、步長短 {100 * (1 - scale):.0f}%）Sensors only", zorder=4)
    ax.plot(E[:, 0], E[:, 1], color=BLUE, lw=2, label="感測器 + 照片校正 Sensors + photo fixes", zorder=5)
    ax.scatter(fx[:, 0], fx[:, 1], marker="s", s=90, color=BLUE, edgecolors="white", zorder=6,
               label=f"拍照定位點（約每 {fix_every_m:.0f} m 一張）Photo fix")
    for i, (x, y) in enumerate(fx):
        ax.annotate(f"{i + 1}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=10, color=BLUE,
                    fontweight="bold")
    ax.scatter(*T[0], marker="o", s=120, color="black", zorder=7)
    ax.annotate("起點 Start", T[0], xytext=(8, -14), textcoords="offset points", fontsize=11)
    ax.annotate("終點 End", T[-1], xytext=(8, 6), textcoords="offset points", fontsize=11)
    last = with_fix[-1]
    box = (f"走了 walked {walked[-1]:.0f} m\n"
           f"終點誤差 end error：只靠感測器 {err_raw[-1]:.1f} m → 有照片校正 {err_fix[-1]:.1f} m\n"
           f"平均誤差 mean error：{err_raw.mean():.1f} m → {err_fix.mean():.1f} m\n"
           f"學到的方向修正 learned heading offset：{float(last['heading_offset']):.1f}°（真實 {bias:.0f}°）\n"
           f"學到的步長倍率 learned step scale：{float(last['step_scale']):.2f}（真實 {1 / scale:.2f}）")
    fig.text(0.54, 0.02, box, fontsize=10.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc"))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.02, 0.03), fontsize=10, frameon=False)
    ax.set_title("模擬走路：照片定位把累積誤差拉回來\nSimulated walk: photo fixes pull the drifting sensor position back "
                 "(phone's MapPositionEstimator code)", fontsize=13)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out / "walk_correction.png", dpi=120)
    plt.close(fig)
    return dict(walked=walked[-1], end_raw=err_raw[-1], end_fix=err_fix[-1], mean_raw=err_raw.mean(),
                mean_fix=err_fix.mean(), offset=float(last["heading_offset"]), scale=float(last["step_scale"]),
                fixes=len(fx), route=route)


def error_map_v2(cache, out, walked=10.0):
    """Before (v1 weighted average) vs after (v2 posterior + reach) on the same walk scenario."""
    from eval import fusion_v2 as V
    st = V._setup(cache)
    sc = V.scenario_scores(st, walked, 1, np.random.default_rng(0), {"aisle": []})
    nodes, edges = _graph(cache)
    xy, truth, dist = st["xy"], st["truth"], st["dist"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 9.5))
    stats = {}
    for ax, key, name in [(axes[0], "v1 (before)", "修改前 Before"), (axes[1], "v2 +reach", "修改後 After")]:
        pick = sc[key].argmax(1)
        ok = pick == truth
        _draw_map(ax, nodes, edges)
        for q in np.where(~ok)[0]:
            ax.annotate("", xy=xy[pick[q]], xytext=xy[truth[q]],
                        arrowprops=dict(arrowstyle="-|>", color=RED, alpha=0.35, lw=1.1, shrinkA=0, shrinkB=2), zorder=3)
        hits = np.bincount(truth[ok], minlength=len(xy))
        nz = hits > 0
        ax.scatter(xy[nz, 0], xy[nz, 1], s=25 + 40 * hits[nz], color=GREEN, alpha=0.85, zorder=4)
        err = dist[truth, pick]
        stats[key] = (int(ok.sum()), float(err.mean()))
        ax.set_title(f"{name}\n答對 correct {ok.sum()}/{len(truth)}（{100 * ok.mean():.0f}%）· 平均差 mean error {err.mean():.1f} m",
                     fontsize=13)
    fig.suptitle(f"同樣 260 張照片，上次定位後走了 {walked:.0f} m：綠點＝答對，紅箭頭＝真實位置 → 錯誤答案\n"
                 f"Same 260 photos, {walked:.0f} m walked since the last photo fix: green = correct, "
                 "red arrow = true position → wrong answer", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "error_map_v2.png", dpi=110)
    plt.close(fig)
    return stats


def node_map(cache, out):
    """Every spot with its number, for filling in the aisle table."""
    nodes, edges = _graph(cache)
    fig, ax = plt.subplots(figsize=(11, 13))
    _draw_map(ax, nodes, edges)
    for n, (x, y) in nodes.items():
        ax.scatter(x, y, s=260, color="white", edgecolors=BLUE, linewidths=1.5, zorder=4)
        ax.annotate(f"{n}", (x, y), ha="center", va="center", fontsize=8.5, color="#1f3b5c", zorder=5,
                    fontweight="bold")
    ax.set_title("賣場地圖上的 59 個點（WP 編號）\n59 spots on the store map (WP numbers) — origin (0, 0) = entrance",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "node_map.png", dpi=110)
    plt.close(fig)


def _sheet(n, cols=6, w=2.6, h=5.0):
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows))
    for ax in axes.flat:
        ax.axis("off")
    return fig, axes.flat


def _walk_photos(photos_dir):
    return sorted(Path(photos_dir).glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))


def quality_all(photos_dir, out):
    from server.photo_quality import assess_image
    files = _walk_photos(photos_dir)
    fig, axes = _sheet(len(files))
    res = []
    for ax, p in zip(axes, files):
        q = assess_image(str(p))
        im = ImageOps.exif_transpose(Image.open(p))
        im.thumbnail((360, 360))
        ax.imshow(im)
        ok = q.hint is None
        ax.set_title(f"{p.name}\n清晰 {q.sharpness:.0f} · 亮度 {q.brightness:.0f}", fontsize=10)
        ax.text(0.5, -0.03, "通過 PASS" if ok else "重拍 RETAKE", transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold", color=GREEN if ok else RED)
        res.append((p.name, round(q.sharpness), round(q.brightness), ok))
    fig.suptitle(f"測試路線 1 全部 {len(files)} 張照片的模糊／亮度檢查：{sum(r[3] for r in res)} 張通過\n"
                 f"Blur / brightness check on all {len(files)} route-1 photos: {sum(r[3] for r in res)} pass "
                 "(retake below sharpness 150 or brightness 45)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "quality_all.jpg", dpi=90, pil_kwargs={"quality": 82})
    plt.close(fig)
    return res


def occlusion_all(photos_dir, out):
    import torch
    from types import SimpleNamespace
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from server.photo_quality import MAX_OCCLUDED_FRACTION, assess_occlusion
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.6).eval()
    prep = weights.transforms()
    files = _walk_photos(photos_dir)
    fig, axes = _sheet(len(files))
    res = []
    for ax, p in zip(axes, files):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.thumbnail((800, 800))
        with torch.no_grad():
            r = model([prep(im)])[0]
        boxes = [b.tolist() for b, l in zip(r["boxes"], r["labels"]) if int(l) == 1]
        occ = assess_occlusion([SimpleNamespace(label="person", box=b) for b in boxes], *im.size)
        ax.imshow(im)
        for b in boxes:
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=RED, lw=1.8))
        bad = occ.fraction >= MAX_OCCLUDED_FRACTION
        ax.set_title(f"{p.name}\n{len(boxes)} 人 people · {100 * occ.fraction:.0f}%", fontsize=10)
        ax.text(0.5, -0.03, "重拍 RETAKE" if bad else "通過 PASS", transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold", color=RED if bad else GREEN)
        res.append((p.name, len(boxes), round(100 * occ.fraction, 1), bad))
    n_people = sum(1 for r in res if r[1])
    fig.suptitle(f"測試路線 1 全部 {len(files)} 張照片的遮擋檢查：{n_people} 張有人，最多佔 "
                 f"{max(r[2] for r in res):.0f}% 畫面，全部通過（門檻 30%）\n"
                 f"Occlusion check on all {len(files)} route-1 photos: people in {n_people}, at most "
                 f"{max(r[2] for r in res):.0f}% of the frame, all pass (limit 30%; red boxes: detected people)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "occlusion_all.jpg", dpi=90, pil_kwargs={"quality": 82})
    plt.close(fig)
    return res


def dino_all(photos_dir, out):
    from eval.real_photos import _dino_topk, _ref_embeddings
    emb, owner = _ref_embeddings()
    files = _walk_photos(photos_dir)
    fig, axes = _sheet(len(files), h=5.4)
    res = []
    for ax, p in zip(axes, files):
        top = list(_dino_topk(p, emb, owner).items())[:3]
        gap = top[0][1] - top[1][1]
        wd = 4 * min(1.0, gap / 0.05)
        im = ImageOps.exif_transpose(Image.open(p))
        im.thumbnail((360, 360))
        ax.imshow(im)
        ax.set_title(p.name, fontsize=10)
        txt = "\n".join(f"#{i + 1} WP{n}  {v:.3f}" for i, (n, v) in enumerate(top))
        ax.text(0.5, -0.02, f"{txt}\ngap {gap:.3f} → w {wd:.1f}/4", transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5, color=RED if gap < 0.02 else GREEN)
        res.append((p.name, top[0][0], round(gap, 4), round(wd, 2)))
    n_amb = sum(1 for r in res if r[2] < 0.02)
    fig.suptitle(f"DINOv2 在測試路線 1 的前三名：{n_amb}/{len(files)} 張第一、二名差距 < 0.02（紅字），權重自動調低\n"
                 f"DINOv2 top 3 on route-1 photos: {n_amb}/{len(files)} have #1 vs #2 gap < 0.02 (red), "
                 "so their DINOv2 weight w is cut (full weight 4)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "dino_all.jpg", dpi=90, pil_kwargs={"quality": 82})
    plt.close(fig)
    return res


def walk_frequency(cache, sim_bin, out, bias=10.0, scale=0.9):
    import networkx as nx
    nodes, edges = _graph(cache)
    g = nx.Graph()
    for a, b in edges:
        g.add_edge(a, b, w=float(np.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])))
    lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="w"))
    a, b = max(((a, b) for a in lengths for b in lengths[a]), key=lambda ab: lengths[ab[0]][ab[1]])
    verts = [nodes[n] for n in nx.dijkstra_path(g, a, b, weight="w")]

    def fixes_every(m):
        if m is None:
            return [0]
        out_, acc = [0], 0.0
        for i in range(1, len(verts)):
            acc += np.hypot(verts[i][0] - verts[i - 1][0], verts[i][1] - verts[i - 1][1])
            if acc >= m:
                out_.append(i)
                acc = 0.0
        return out_

    def run(fix_idx):
        inp = "\n".join(f"{x},{y}" for x, y in verts) + "\n" + "\n".join(f"FIX {i}" for i in fix_idx) + "\n"
        txt = subprocess.run([sim_bin, str(bias), str(scale), "7"], input=inp, capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()
        cols = txt[0].split(",")
        return [dict(zip(cols, l.split(","))) for l in txt[1:]]

    fig, axes = plt.subplots(1, 4, figsize=(18, 7.5))
    res = []
    for ax, every in zip(axes, [None, 24, 12, 6]):
        rows = run(fixes_every(every))
        T = np.array([[float(r["true_x"]), float(r["true_y"])] for r in rows])
        E = np.array([[float(r["est_x"]), float(r["est_y"])] for r in rows])
        F = np.array([[float(r["true_x"]), float(r["true_y"])] for r in rows if r["fix"] == "1"]).reshape(-1, 2)
        err = np.linalg.norm(E - T, axis=1)
        _draw_map(ax, nodes, edges)
        ax.plot(T[:, 0], T[:, 1], color="black", lw=2.2, zorder=3)
        ax.plot(E[:, 0], E[:, 1], color=BLUE if every else RED, lw=1.8, ls="-" if every else "--", zorder=4)
        if len(F):
            ax.scatter(F[:, 0], F[:, 1], marker="s", s=55, color=BLUE, edgecolors="white", zorder=5)
        label = "只在起點拍一張 start photo only" if every is None else f"每 {every} m 一張 photo every {every} m"
        ax.set_title(f"{label}\n{len(F)} 張 photos · 平均差 mean {err.mean():.1f} m · 最大 max {err.max():.1f} m",
                     fontsize=11.5)
        res.append((every, len(F), round(float(err.mean()), 2), round(float(err.max()), 2)))
    fig.suptitle(f"拍照越頻繁，感測器誤差越小（同一條 62 m 路線，方向偏 {bias:.0f}°、步長短 {100 * (1 - scale):.0f}%）\n"
                 "More frequent photos keep the sensor error small (same 62 m route; black = true path, "
                 "blue = corrected, red dashed = sensors only)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "walk_frequency.png", dpi=110)
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--walk-sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = json.loads(Path(args.cache).read_text())
    want = set(args.only or ["map", "map2", "errmap", "dino", "dinoall", "quality", "qualityall",
                             "occlusion", "occlusionall", "walk", "walkfreq"])
    results = {}
    if "map" in want:
        results["map"] = map_examples(cache, out)
    if "errmap2" in want:
        results["errmap2"] = error_map_v2(cache, out)
    if "nodemap" in want:
        node_map(cache, out)
    if "map2" in want:
        results["map2"] = map_examples(cache, out, name="map_examples_more.png", n_good=6, n_bad=2,
                                       skip_good=3, skip_bad=1, cols=4)
    if "errmap" in want:
        results["errmap"] = error_map(cache, out)
    if "dinoall" in want:
        results["dinoall"] = dino_all(args.photos, out)
    if "qualityall" in want:
        results["qualityall"] = quality_all(args.photos, out)
    if "occlusionall" in want:
        results["occlusionall"] = occlusion_all(args.photos, out)
    if "walkfreq" in want:
        results["walkfreq"] = walk_frequency(cache, args.walk_sim, out)
    if "dino" in want:
        results["dino"] = dino_ambiguity(cache, args.photos, ["17.jpg", "25.jpg"], out)
    if "quality" in want:
        results["quality"] = photo_quality_fig(args.photos, "10.jpg", out)
    if "occlusion" in want:
        results["occlusion"] = occlusion_fig(args.photos, out)
    if "walk" in want:
        results["walk"] = walk_fig(cache, args.walk_sim, out)
    print(json.dumps(results, ensure_ascii=False, default=str, indent=1))


if __name__ == "__main__":
    main()
