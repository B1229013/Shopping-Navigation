"""Per-photo localization report: why each query photo landed where it did.

  python -m eval.photo_report --set lab1019 --perception vlm --data-root <backend> --refs <reference photo root>
                              [--out DIR] [--variant prod]

For every query photo (no phone sensors), from the image_only caches only — no VLM call,
no model run except the label similarity the server itself computes:

  the photo, its VLM objects and texts; the top-5 nodes of the server's posterior and,
  if missing from them, the true node — each with the reference photo that scored it
  (locate_fusion's best directional photo = server_best_photo), its place on the map,
  DINOv2 cosine, the object pairs that matched (query label → map label, similarity ≥
  NODE_SIM_THRESHOLD, greedy one-to-one as _node_score) and their score, the shared OCR
  texts with their IDF weights and the OCR score, aisle-sign evidence, fused score,
  image evidence (0..2) and posterior probability.

Every number is recomputed with locate_fusion / loc_fusion's own functions and checked
against the server's own result (pick and posterior) for the same photo.

Output (--out, default <data-root>/output/eval_photos/<set>_report): index.html (summary,
how the method works), nodes.html, between.html, img/ (thumbnails), matches.jsonl (the
same details as data). matches.jsonl is also copied to eval/results/.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from eval import image_only as io

BACKEND = Path(__file__).resolve().parents[1]
DIRS = ("front", "right", "back", "left")


def _thumb(src, dst, size):
    from PIL import Image, ImageOps
    if not dst.is_file():
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        im.thumbnail((size, size))
        im.save(dst, "JPEG", quality=78)
    return dst


def _ref_path(root, photo_file):
    sub, name = photo_file.split("/", 1)
    for p in (root / sub.replace("set", "") / name, root / sub / name, root / photo_file):
        if p.is_file():
            return p
    return None


class Explainer:
    """locate_fusion.best_photo_per_node, term by term, for one query photo."""

    def __init__(self, ref_map, place):
        from server import config as C
        from server import locate_fusion as LF
        self.LF, self.place, self.ref_map = LF, place, ref_map
        embs = LF._reference_embeddings(ref_map, LF._load_sidecar(C.DINO_SIDECAR_PATH))
        self.index = LF._index_for(ref_map, embs)
        self.E = self.index["entries"]

    def pairs(self, q_labels, r_labels):
        """_node_score with its matched pairs."""
        cand = []
        for i, ql in enumerate(q_labels):
            for j, rl in enumerate(r_labels):
                s = self.LF._embedding_sim(ql, rl)
                if s >= self.LF.NODE_SIM_THRESHOLD:
                    cand.append((s, i, j))
        cand.sort(reverse=True)
        uq, ur, out = set(), set(), []
        for s, i, j in cand:
            if i in uq or j in ur:
                continue
            uq.add(i); ur.add(j)
            out.append((q_labels[i], r_labels[j], s))
        score = sum(s for *_, s in out) / len(q_labels) if q_labels else 0.0
        return score, out

    def explain(self, q_emb, labels, texts, aisle):
        LF, E, idx = self.LF, self.E, self.index
        q_ocr = sorted({LF._norm_ocr(t) for t in texts} - {""})
        idf = {t: idx["idf"].get(t, idx["def_idf"]) for t in q_ocr}
        denom = sum(idf.values())
        vis = np.array([float(np.dot(q_emb, e["emb"])) if e["emb"] is not None else 0.0 for e in E])
        nod, details = np.zeros(len(E)), []
        ocr = np.zeros(len(E))
        for k, e in enumerate(E):
            sc, pr = self.pairs(labels, e["labels"])
            nod[k] = sc
            shared = sorted(set(q_ocr) & e["ocrset"])
            ocr[k] = sum(idf[t] for t in shared) / denom if shared and denom else 0.0
            details.append((pr, shared))
        dmax = {}
        for k, e in enumerate(E):
            if e["emb"] is not None and vis[k] > dmax.get(e["nid"], (-9.0, None))[0]:
                dmax[e["nid"]] = (float(vis[k]), e["slot"])
        dino_top5 = [[n, round(c, 4), s] for n, (c, s) in sorted(dmax.items(), key=lambda kv: -kv[1][0])[:5]]
        mm = lambda a: (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else a * 0
        vmm, nmm = mm(vis), mm(nod)
        fused = LF.W_V * vmm + LF.W_N * nmm + LF.W_O * ocr
        best = {}
        for k, e in enumerate(E):
            if e["nid"] not in best or fused[k] > fused[best[e["nid"]]]:
                best[e["nid"]] = k
        scores = {n: float(fused[k]) for n, k in best.items()}
        lo, hi = min(scores.values()), max(scores.values())
        ev = {n: (2.0 * (s - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0) + 1.0 * aisle.get(n, 0.0)
              for n, s in scores.items()}
        per = {}
        for n, k in best.items():
            e = E[k]
            pr, shared = details[k]
            per[n] = {
                "photo_file": e["photo"].photo_file if e["photo"] is not None else None, "slot": e["slot"],
                "dino_cos": float(vis[k]), "dino_mm": float(vmm[k]),
                "objects_score": float(nod[k]), "objects_mm": float(nmm[k]),
                "object_pairs": [[q, r, round(s, 3)] for q, r, s in pr],
                "ocr_score": float(ocr[k]), "ocr_shared": [[t, round(idf[t], 3)] for t in shared],
                "fused": float(fused[k]), "aisle": float(aisle.get(n, 0.0)), "evidence": ev[n],
            }
        return per, {"ocr_tokens": [[t, round(idf[t], 3)] for t in q_ocr], "ocr_idf_total": round(denom, 3),
                     "dino_only_top5": dino_top5}


def _mini_map(ref_map, truth_xy, ok, ranked, size=230):
    xy = {n: (p.pdr_x, p.pdr_y) for n, p in ref_map.photos.items()}
    xs, ys = [p[0] for p in xy.values()], [p[1] for p in xy.values()]
    x0, x1, y0, y1 = min(xs) - 2, max(xs) + 2, min(ys) - 2, max(ys) + 2
    s = size / max(x1 - x0, y1 - y0)
    P = lambda p: (round((p[0] - x0) * s, 1), round((y1 - p[1]) * s, 1))
    w, h = round((x1 - x0) * s), round((y1 - y0) * s)
    out = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' class=map>"]
    seen = set()
    for a, p in ref_map.photos.items():
        for b in p.neighbor_nids:
            if b in xy and (b, a) not in seen:
                seen.add((a, b))
                (ax, ay), (bx, by) = P(xy[a]), P(xy[b])
                out.append(f"<line x1={ax} y1={ay} x2={bx} y2={by} class=e />")
    for n, p in xy.items():
        cx, cy = P(p)
        out.append(f"<circle cx={cx} cy={cy} r=2.2 class=n><title>{n}</title></circle>")
    for i, n in enumerate(ranked[:5], 1):
        cx, cy = P(xy[n])
        out.append(f"<circle cx={cx} cy={cy} r={7 if i == 1 else 5} class='c{1 if i == 1 else 2}'><title>#{i} 節點 {n}</title></circle>"
                   f"<text x={cx + 6} y={cy - 4} class=t>{i}</text>")
    tx, ty = P(truth_xy)
    out.append(f"<rect x={tx - 5} y={ty - 5} width=10 height=10 class=truth><title>正確位置</title></rect>")
    out.append("</svg>")
    return "".join(out)


METHOD_HTML = """
<h2>目前的定位方法（正式伺服器、部署中的設定，不帶感測器）</h2>
<ol>
<li><b>前處理</b>：照片轉正、長邊縮到 1600 px；太模糊或太暗直接請使用者重拍（<code>photo_quality</code>）。</li>
<li><b>感知</b>：三階段 VLM（gpt-5.4-mini）。第 1 次列出物件和文字，第 2 次補漏；第 3 次的回覆格式和解析器不合，
目前全部被丟掉。物件名稱是中文，文字是照片上讀到的字串。人或推車擋住太多畫面就不定位。</li>
<li><b>DINOv2</b>（facebook/dinov2-base 的 CLS 向量）：和地圖 59 個節點、每個節點約 4 個方向、共約 260 張參考照片逐張算 cosine 相似度。</li>
<li><b>每張參考照片的融合分數</b> = 5 × DINOv2 + 0.5 × 物件 + 0.5 × 文字（<code>locate_fusion</code>）。
<ul><li>DINOv2：cosine 在所有參考照片間做 min-max（最像 1、最不像 0）。</li>
<li>物件：查詢物件和參考物件的字元相似度 ≥ 0.55 才算配對，一對一取最高，總和 ÷ 查詢物件數，再做 min-max。
地圖物件名稱是「中文 英文」、VLM 只給中文，所以幾乎配不到（報表裡物件分數多半是 0）。</li>
<li>文字：正規化後<b>完全相同</b>的字串才算，依稀有度（IDF）加權：共同文字的權重 ÷ 照片所有文字的權重，不做 min-max。</li></ul></li>
<li><b>節點分數</b>：每個節點取分數最高的那張參考照片（報表裡「對應的參考照片」，也決定使用者面向哪個方向），
全部節點正規化到 0–2，再加走道號碼證據（× 1）＝「影像證據」。</li>
<li><b>機率</b>：softmax(8 × 影像證據)（帶感測器時再扣掉與手機位置的距離項）。第一名就是定位結果，機率就是信心；低於 0.15 不給節點。</li>
</ol>
<h2>之前的定位方法</h2>
<ul>
<li><b>研究腳本（backend/legacy，0916 測試集 268 張）</b>：九宮格字串比對 28% → 換 embedding 33.6% → 物件＋OCR 41% →
CLIP 40%（融合 55%）→ DINOv2 單獨 60% → <b>DINOv2 融合 66%</b>（同樣 5 : 0.5 : 0.5，但物件和文字來自另一套單次 VLM 提示詞、
DINOv2 用原圖、沒有走道號碼）。</li>
<li><b>伺服器加入 DINOv2 之前</b>：九宮格比對（<code>match_by_grid</code>：物件落在 4×4 格子的位置＋OCR），
再和手機送來的 DINOv2 前幾名、PDR 位置加權平均（v1，DINOv2 : 感測器 : 九宮格 = 4 : 1 : 1）。Android 手機沒有 DINOv2，只剩九宮格。</li>
<li><b>現在</b>：Android 不算 DINOv2，改由伺服器算（上面第 3～4 步），再用 v2 機率融合；九宮格仍會執行，但在這條路上結果沒有被使用。</li>
</ul>
<h2>同一套方法在不同資料上的表現（不帶感測器、Top-1）</h2>
<ul>
<li>0916（和建圖同一天拍）：DINOv2 單獨 59.7%（正式路徑重算）。</li>
<li>lab1019（建圖後一個月拍，VLM 感知）：融合 44.7%（節點 50.0%、節點之間 32.8%），DINOv2 單獨 41.2%。</li>
</ul>"""

CSS = """body{font-family:system-ui,'Microsoft JhengHei',sans-serif;margin:16px;background:#f6f7f9;color:#1f2328;max-width:1400px}
h1,h2{margin:.4em 0}.card{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:12px;margin:0 0 16px}
.ok{color:#1a7f37;font-weight:600}.bad{color:#cf222e;font-weight:600}.near{color:#9a6700;font-weight:600}
.top{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}.q img{border-radius:6px}
.small{font-size:12px;color:#57606a}table{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}
td,th{border-top:1px solid #e5e8eb;padding:5px 6px;vertical-align:top;text-align:left}th{background:#f6f8fa}
tr.truthrow{background:#e9f7ec}tr.pick{background:#fff6e5}.ref img{border-radius:4px;display:block}
.map{background:#fbfbfb;border:1px solid #e5e8eb;border-radius:6px}.map .e{stroke:#d0d7de;stroke-width:1}
.map .n{fill:#afb8c1}.map .c1{fill:#cf222e;opacity:.85}.map .c2{fill:#fb8f44;opacity:.8}.map .t{font-size:10px;fill:#57606a}
.map .truth{fill:none;stroke:#1a7f37;stroke-width:2.5}.tag{display:inline-block;background:#eef1f4;border-radius:4px;padding:1px 5px;margin:1px;font-size:12px}
.m{background:#dafbe1}.nav a{margin-right:12px}code{background:#eef1f4;padding:1px 4px;border-radius:3px}"""


def _cand_row(rank, n, d, p, truth_ok, img_html, dist, w=(5.0, 0.5, 0.5)):
    cls = "truthrow" if truth_ok else ""
    aisle = f"<br>走道號碼 +{d['aisle']:.2f}" if d["aisle"] else ""
    objs = "".join(f"<div>{html.escape(q)} → {html.escape(r)} <b>{s:.2f}</b></div>" for q, r, s in d["object_pairs"]) or "<span class=small>無配對</span>"
    ocrs = "".join(f"<span class='tag m'>{html.escape(t)} ×{w:.2f}</span>" for t, w in d["ocr_shared"]) or "<span class=small>無共同文字</span>"
    return (f"<tr class='{cls}'><td>{rank}</td><td><b>節點 {n}</b>{' ✓正確' if truth_ok else ''}<div class=small>"
            f"離正確位置 {dist:.1f} m<br>方向照片 {html.escape(str(d['slot']))}</div></td><td class=ref>{img_html}</td>"
            f"<td><b>{p:.3f}</b></td><td>{d['evidence']:.2f}<div class=small>融合 {d['fused']:.2f}{aisle}</div></td>"
            f"<td>{d['dino_cos']:.3f}<div class=small>正規化 {d['dino_mm']:.2f} ×{w[0]:g}</div></td>"
            f"<td>{d['objects_score']:.2f}<div class=small>正規化 {d['objects_mm']:.2f} ×{w[1]:g}</div>{objs}</td>"
            f"<td>{d['ocr_score']:.2f}<div class=small>×{w[2]:g}</div>{ocrs}</td></tr>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=io.SETS)
    ap.add_argument("--perception", choices=("gdino", "vlm"), default="vlm")
    ap.add_argument("--data-root", default=str(BACKEND))
    ap.add_argument("--refs", required=True, help="folder holding the map's reference photos (01/, 02/, 03/ …)")
    ap.add_argument("--cache", help="default backend/output/eval/image_only/<set>")
    ap.add_argument("--out")
    ap.add_argument("--variant", default="prod", choices=io.VARIANTS + io.ABLATIONS)
    ap.add_argument("--limit", type=int, help="first N photos only (check the output)")
    ap.add_argument("--labels", choices=("zh", "bilingual"), default="zh",
                    help="bilingual = the VLM's object names as 「中文 English」 (eval/whatif)")
    ap.add_argument("--sim", choices=("char", "hybrid"), default="char",
                    help="hybrid = the server's label similarity with its embedding cache on "
                         "(vectors from eval/label_embeddings' cache; no API call)")
    ap.add_argument("--weights", default="5,0.5,0.5", help="DINOv2,objects,text (locate_fusion W_V,W_N,W_O)")
    args = ap.parse_args(argv)
    weights_arg = tuple(float(x) for x in args.weights.split(","))
    tag = "" if (args.labels, args.sim, weights_arg) == ("zh", "char", (5.0, 0.5, 0.5)) else \
        f"_{args.labels}_{args.sim}_{'-'.join(f'{w:g}' for w in weights_arg)}"

    import logging
    from server import config as C
    from server import server as srv
    from server.loc_fusion import fuse_v2, parse_image_k
    logging.getLogger().setLevel(logging.WARNING)
    C.RERANK_ENABLED = False
    cache = Path(args.cache) if args.cache else BACKEND / "output" / "eval" / "image_only" / args.set
    out = Path(args.out) if args.out else Path(args.data_root) / "output" / "eval_photos" / f"{args.set}_report{tag}"
    (out / "img").mkdir(parents=True, exist_ok=True)
    refs = Path(args.refs)

    ref_map, _ = io.load_map("snapshot", cache)
    srv.get_neo4j = lambda: io._MapSource(ref_map)
    items = io.load_items(args.set, ref_map, Path(args.data_root))
    if args.limit:
        items = [it for it in items if it["condition"] == "at_node"][:args.limit // 2] + \
                [it for it in items if it["condition"] == "between"][:args.limit - args.limit // 2]
    perc_all = io._load_json(cache / ("perception_vlm.json" if args.perception == "vlm" else "perception.json"))
    dino = io._load_json(cache / "dino.json")
    hops = io.hop_table(ref_map)
    xy = {n: (p.pdr_x, p.pdr_y) for n, p in ref_map.photos.items()}
    from server import locate_fusion as LF
    LF.W_V, LF.W_N, LF.W_O = weights_arg
    labels_fn = lambda p: p
    if args.labels == "bilingual":
        from eval.whatif import bilingual_labels as labels_fn
    if args.sim == "hybrid":
        from eval.label_embeddings import LabelEmbeddings
        from server import visual_localization as VL
        emb = LabelEmbeddings(cache / "label_embeddings.npz")
        embs = LF._reference_embeddings(ref_map, LF._load_sidecar(C.DINO_SIDECAR_PATH))
        need = {l for e in LF._index_for(ref_map, embs)["entries"] for l in e["labels"]} | \
               {d["label"] for it in items for d in labels_fn(perc_all[it["id"]])["detections"]}
        missing = [l for l in need if l not in emb.vec]
        if missing:
            sys.exit(f"{len(missing)} labels have no cached vector — run eval.weight_sweep --sim hybrid first")
        VL._emb_cache.clear()
        VL._emb_cache.update({l: emb.matrix([l])[0] for l in need})
        VL._emb_loaded = True        # the server's own hybrid_max path, cache on
        # speed: one (photo labels × map labels) matrix per photo instead of ~2×10⁵ single
        # _hybrid_max_sim calls; same values (weight_sweep checks this against _node_score)
        from eval.weight_sweep import SemanticSim
        ref_labels = sorted({l for e in LF._index_for(ref_map, embs)["entries"] for l in e["labels"]})
        semsim = SemanticSim("hybrid", emb, ref_labels)
        ridx = {l: j for j, l in enumerate(ref_labels)}
        real_hybrid = VL._hybrid_max_sim
    else:
        semsim = None
    ex = Explainer(ref_map, C.NEO4J_PLACE)
    k_img = parse_image_k(C.LOC_SERVER_IMAGE_K)
    from server import dino_embed
    embed = io.EmbedCache(cache / "dino.json", dino_embed.embed_image, dino_embed._MODEL_NAME)
    dino_embed.embed_image = embed      # the query vectors image_only already computed

    records, cards = [], {"at_node": [], "between": []}
    for i, it in enumerate(items, 1):
        p, q = perc_all.get(it["id"]), dino.get(it["id"])
        if p is None or q is None:
            sys.exit(f"{it['id']}: not in the image_only cache — run eval.image_only first")
        p = labels_fn(p)
        if semsim is not None:
            uniq = list(dict.fromkeys(d["label"] for d in p["detections"]))
            S, qidx = semsim.matrix(uniq), {l: i for i, l in enumerate(uniq)}

            def lookup(a, b, S=S, qidx=qidx):
                i, j = qidx.get(a), ridx.get(b)
                return float(S[i, j]) if i is not None and j is not None else real_hybrid(a, b)
            VL._hybrid_max_sim = lookup     # read at call time by locate_fusion and match_by_grid
        img = io.preprocess(it, cache / "img")
        embed.key = it["id"]
        with io.variant(args.variant):
            loc, _ = io.localize(srv, C.NEO4J_PLACE, p, img)
            aisle = io.evidence_detail(ref_map, p, None, C.NEO4J_PLACE)["aisle_evidence"]
            aisle = {int(k): v for k, v in aisle.items()} if C.AISLE_SIGN_PLACES else {}
            from server import locate_fusion as LF
            weights = (LF.W_V, LF.W_N, LF.W_O)
            per, qinfo = ex.explain(np.asarray(q["emb"], np.float32), [d["label"] for d in p["detections"]],
                                    [o["text"] for o in p["ocr"]], aisle)
        post = fuse_v2({n: d["evidence"] for n, d in per.items()}, xy, image_k=k_img)
        prob = {n: pr for n, pr, _ in post}
        ranked = [n for n, *_ in post]
        server_top = [int(c[0]) for c in loc.top_candidates]
        if ranked[:5] != server_top[:5] or abs(prob[ranked[0]] - loc.confidence) > 1e-6:
            sys.exit(f"{it['id']}: recomputed ranking {ranked[:5]} ≠ server {server_top[:5]} — explanation would be wrong")
        ok = it["ok"]
        show = ranked[:5] + [n for n in ok if n not in ranked[:5]]
        hit = ranked[0] in ok
        hop = min(hops[o].get(ranked[0], 99) for o in ok)
        rec = {"id": it["id"], "photo": str(it["path"]), "condition": it["condition"], "label": it["meta"].get("label"),
               "ok": ok, "truth_xy": it["truth_xy"], "variant": args.variant, "weights": weights,
               "pick": ranked[0], "hit1": hit, "hop": hop, "confidence": prob[ranked[0]],
               "truth_rank": min(ranked.index(o) for o in ok) + 1,
               "objects": [{"label": d["label"], "score": d["score"]} for d in p["detections"]],
               "texts": [{"text": o["text"], "confidence": o["confidence"]} for o in p["ocr"]], **qinfo,
               "candidates": [{"rank": ranked.index(n) + 1, "node": n, "xy": list(xy[n]), "posterior": prob[n],
                               "is_truth": n in ok, **per[n]} for n in show]}
        records.append(rec)

        qt = _thumb(img, out / "img" / (it["id"].replace("/", "__") + ".jpg"), 360)
        rows = []
        for n in show:
            d = per[n]
            rp = _ref_path(refs, d["photo_file"]) if d["photo_file"] else None
            im = (f"<img src='img/{_thumb(rp, out / 'img' / ('ref_' + Path(d['photo_file']).name), 200).name}' width=170>"
                  if rp else "<span class=small>找不到參考照片</span>")
            rows.append(_cand_row(ranked.index(n) + 1, n, d, prob[n], n in ok, im, math.dist(xy[n], it["truth_xy"]),
                                  weights))
        verdict = ("<span class=ok>對</span>" if hit else "<span class=near>差一格</span>" if hop <= 1
                   else f"<span class=bad>錯（差 {hop} 格）</span>")
        tags_o = "".join(f"<span class=tag>{html.escape(d['label'])}</span>" for d in p["detections"])
        tags_t = "".join(f"<span class=tag>{html.escape(t)}</span>" for t, _ in qinfo["ocr_tokens"])
        cards[it["condition"]].append(
            f"<div class=card id='{html.escape(it['id'])}'><h3>{html.escape(it['id'])}　答案 {html.escape(str(it['meta'].get('label')))}"
            f"　→ 猜 節點 {ranked[0]}（機率 {prob[ranked[0]]:.2f}）{verdict}　正確答案排第 {rec['truth_rank']}</h3>"
            f"<div class=top><div class=q><img src='img/{qt.name}' width=300></div>"
            f"<div>{_mini_map(ref_map, it['truth_xy'], ok, ranked)}<div class=small>綠框＝正確位置，紅＝第 1 名，橘＝第 2～5 名</div></div>"
            f"<div style='flex:1;min-width:280px'><b>VLM 看到的物件（{len(p['detections'])}）</b><div>{tags_o}</div>"
            f"<b>VLM 讀到的文字（{len(qinfo['ocr_tokens'])}）</b><div>{tags_t or '<span class=small>無</span>'}</div>"
            f"<div class=small>場景：{html.escape(p.get('scene') or '')}</div>"
            f"<b>DINOv2 單獨的前五名</b>（只看影像相似度，每個節點取最像的方向照片）<div>" + "".join(
                f"<span class='tag{' m' if n in ok else ''}'>{i_ + 1}. 節點 {n}　{c:.3f}　{html.escape(str(s))}</span>"
                for i_, (n, c, s) in enumerate(qinfo["dino_only_top5"])) +
            ("" if any(n in ok for n, *_ in qinfo["dino_only_top5"]) else "<span class=small>（正確節點不在 DINOv2 前五名）</span>") +
            "</div></div></div>"
            "<table><tr><th>名次</th><th>節點</th><th>對應的參考照片</th><th>機率</th><th>影像證據 0–2</th>"
            "<th>DINOv2 cosine</th><th>物件分數與配對</th><th>文字分數與共同文字（×權重）</th></tr>" + "".join(rows) + "</table></div>")
        print(f"\r  {i}/{len(items)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    with open(out / "matches.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(io._rounded(r, 4), ensure_ascii=False) + "\n")
    dst = BACKEND / "eval" / "results" / f"image_only_{args.set}{'_vlm' if args.perception == 'vlm' else ''}_matches{tag}.jsonl"
    shutil.copyfile(out / "matches.jsonl", dst)

    def summary(rs):
        n = len(rs)
        pct = lambda xs: f"{100 * sum(xs) / n:.1f}%" if n else "–"
        return (f"{n} 張｜Top-1 {pct([r['hit1'] for r in rs])}｜Top-5 {pct([r['truth_rank'] <= 5 for r in rs])}"
                f"｜1 格內 {pct([r['hop'] <= 1 for r in rs])}")

    nav = "<div class=nav><a href=index.html>總覽</a><a href=nodes.html>節點照片</a><a href=between.html>節點之間照片</a></div>"
    for cond, fname, title in (("at_node", "nodes.html", "節點照片"), ("between", "between.html", "節點和節點之間的照片")):
        rs = [r for r in records if r["condition"] == cond]
        (out / fname).write_text(
            f"<!doctype html><meta charset=utf-8><title>{title}</title><style>{CSS}</style>{nav}"
            f"<h1>{title}</h1><p>{summary(rs)}（版本 {args.variant}，不帶感測器）</p>" + "".join(cards[cond]),
            encoding="utf-8")
    (out / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>定位逐張報告</title><style>{CSS}</style>{nav}"
        f"<h1>{args.set} 逐張定位報告</h1><ul><li>全部：{summary(records)}</li>"
        f"<li><a href=nodes.html>節點照片</a>：{summary([r for r in records if r['condition'] == 'at_node'])}</li>"
        f"<li><a href=between.html>節點之間照片</a>：{summary([r for r in records if r['condition'] == 'between'])}</li></ul>"
        f"<p><b>這一版的設定</b>：物件名稱 {'「中文 English」' if args.labels == 'bilingual' else '中文（現行）'}；"
        f"物件相似度 {'字元與語意向量取較大值（伺服器開啟向量快取時）' if args.sim == 'hybrid' else '字元相似度（現行）'}；"
        f"DINOv2 : 物件 : 文字 = {weights[0]:g} : {weights[1]:g} : {weights[2]:g}"
        f"{'（現行正式設定）' if not tag else '（試算設定，伺服器尚未採用）'}。</p>"
        f"<p class=small>版本 {args.variant}，不帶感測器；每張照片的數字都用伺服器的函式重算，並確認前五名和機率與伺服器結果相同。</p>"
        + METHOD_HTML,
        encoding="utf-8")
    print(f"→ {out / 'index.html'}  ({len(records)} photos) ; {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
