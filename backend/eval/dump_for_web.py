"""dump_for_web.py — 產生「定位比較網頁」要用的完整結果 JSON（含物件關係明細）。

沿用 eval.image_only 的生產定位路徑（server._run_early_localization →
locate_fusion.server_image_evidence → visual_localization.localize(v2=True)），
每張查詢照片輸出：
  - 定位前五名（節點、融合分數＝信心、該候選「最像的方向照片」slot）
  - 查詢照片 VLM 抓到的物件（label＋bbox）與 OCR 文字
  - 查詢照片物件兩兩推出的上下左右關係
  - 每個候選節點（建資料庫時）該方向照片的物件與關係
  - 查詢 vs 候選「對上的物件配對」與「一致的上下左右關係」＝實際幫到定位的證據
同時把每張查詢照片輸出一張 1400px 的 web 用 JPG 到 <out>/photos_query/<tag>/。

需求：和你平常跑 eval 一樣的環境（venv + torch/transformers + DINOv2 + .env 的
Neo4j / OpenAI 都在線）。放到 backend/eval/ 底下，從 backend/ 執行。

用法（從 backend/）：
  # 正常角度（0916，沿用既有 labels 與 legacy/testPDR 原圖）
  python -m eval.dump_for_web --set 0916 \
      --out ~/Downloads/學校家樂福/定位比較

  # 非正常角度（節點之間，用已轉好的 JPG 資料夾）
  python -m eval.dump_for_web --tag 樂家康 \
      --photos ~/Downloads/學校家樂福/定位比較/photos_query/樂家康 \
      --out ~/Downloads/學校家樂福/定位比較

輸出：<out>/web_results_<tag>.json（tag 預設為 --set 或 --photos 資料夾名）。
把兩個 JSON 都放在 <out>（也就是已連入 Claude 的「定位比較」資料夾）即可，我會接著把網頁補滿。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def rel_name(axis, qa):
    if axis == "V":
        return "ABOVE" if qa < 0 else "BELOW"
    return "LEFT_OF" if qa < 0 else "RIGHT_OF"


def relations_from_objects(objs):
    """objs: [(label, bbox 0..1)] → [{"a","b","type"}]，和建圖 build_object_relations 同規則。"""
    from server.object_relations import neighbour_pairs, pair_relation
    boxes = [b for _, b in objs]
    out = []
    for i, j in neighbour_pairs(boxes):
        rel = pair_relation(boxes[i], boxes[j])
        a, b = objs[i][0], objs[j][0]
        if rel.v:
            up, down = (a, b) if rel.v < 0 else (b, a)
            out.append({"a": up, "b": down, "type": "ABOVE"})
        if rel.h:
            left, right = (a, b) if rel.h < 0 else (b, a)
            out.append({"a": left, "b": right, "type": "LEFT_OF"})
    return out


def verbose_relations(query, ref, sim, rarity, min_sim=0.55, wv=1.0, wh=0.5, smooth=1.0):
    """複刻 object_relations.relation_score，但回傳『對上的物件配對』與『一致/矛盾的關係』。"""
    from server.object_relations import pair_relation, _valid
    q = [(l, b) for l, b in query if l and _valid(b)]
    r = [(l, b) for l, b in ref if l and _valid(b)]
    res = {"matched_objects": [], "relations": [], "agree": 0.0, "disagree": 0.0, "score": 0.0}
    if len(q) < 2 or len(r) < 2:
        return res
    cand = []
    for i, (ql, _) in enumerate(q):
        for j, (rl, _) in enumerate(r):
            s = sim(ql, rl)
            if s >= min_sim:
                cand.append((s, i, j))
    cand.sort(reverse=True)
    uq, ur, matched = set(), set(), []
    for s, i, j in cand:
        if i in uq or j in ur:
            continue
        uq.add(i)
        ur.add(j)
        w = s * (rarity.get(r[j][0], 1.0) if rarity else 1.0)
        matched.append((i, j, w))
        res["matched_objects"].append({"q": q[i][0], "db": r[j][0], "sim": round(float(s), 3)})
    if len(matched) < 2:
        return res
    agree = dis = 0.0
    for x in range(len(matched)):
        i1, j1, w1 = matched[x]
        for y in range(x + 1, len(matched)):
            i2, j2, w2 = matched[y]
            qr = pair_relation(q[i1][1], q[i2][1])
            rr = pair_relation(r[j1][1], r[j2][1])
            w = w1 * w2
            for qa, ra, wa, axis in ((qr.v, rr.v, wv, "V"), (qr.h, rr.h, wh, "H")):
                if qa and ra and wa:
                    ok = (qa == ra)
                    if ok:
                        agree += w * wa
                    else:
                        dis += w * wa
                    res["relations"].append({"a": q[i1][0], "b": q[i2][0],
                                             "type": rel_name(axis, qa), "agree": ok})
    res["agree"] = round(agree, 3)
    res["disagree"] = round(dis, 3)
    res["score"] = round(max(0.0, (agree - dis) / (agree + dis + smooth)), 3)
    return res


def items_from_folder(folder):
    exts = (".jpg", ".jpeg", ".png")
    fs = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    return [{"id": f, "path": Path(folder) / f, "ok": [], "truth": None, "truth_xy": None,
             "condition": "between", "meta": {}, "heading": None, "sensor": None, "stored": False} for f in fs]


def save_web_jpg(src_path, dst_path, maxpx=1400):
    try:
        from PIL import Image, ImageOps
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except Exception:
            pass
        if dst_path.exists():
            return True
        im = Image.open(src_path)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((maxpx, maxpx))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path, "JPEG", quality=82)
        return True
    except Exception as e:
        print(f"  (web jpg 失敗 {src_path.name}: {e})", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", help="沿用 eval.image_only 的資料集（例如 0916），有 labels 與 truth")
    ap.add_argument("--photos", help="自訂查詢照片資料夾（無 labels，例如樂家康）")
    ap.add_argument("--tag", help="輸出標籤/組別名（預設取 --set 或資料夾名）")
    ap.add_argument("--data-root", default=str(BACKEND), help="--set 照片的根目錄（預設 backend/）")
    ap.add_argument("--out", required=True, help="輸出資料夾（= 已連入 Claude 的「定位比較」）")
    ap.add_argument("--map", default="neo4j", choices=["neo4j", "snapshot"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張（測試用）")
    ap.add_argument("--random", type=int, default=0, help="隨機抽 N 張（設 seed=42 可重現）")
    args = ap.parse_args()
    if not args.set and not args.photos:
        ap.error("要給 --set 或 --photos 其中一個")

    import random
    from eval import image_only as IO
    from server import config as C
    from server import dino_embed
    from server import server as srv
    from server import locate_fusion as LF
    from server import object_relations as R
    from server import photo_quality

    import logging
    logging.getLogger().setLevel(logging.WARNING)
    C.RERANK_ENABLED = False
    if not (C.LOC_FUSION_V2 and C.LOC_SERVER_DINO):
        sys.exit("LOC_FUSION_V2 與 LOC_SERVER_DINO 必須開（才是生產定位路徑）")

    tag = args.tag or args.set or Path(args.photos).name
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache = BACKEND / "output" / "eval" / "image_only" / (args.set or ("folder_" + tag))
    (cache / "img").mkdir(parents=True, exist_ok=True)

    ref_map, _ = IO.load_map(args.map, cache)
    srv.get_neo4j = lambda: IO._MapSource(ref_map)

    if args.set:
        items = IO.load_items(args.set, ref_map, Path(args.data_root).expanduser(), args.photos)
    else:
        items = items_from_folder(Path(args.photos).expanduser())
    if args.random and args.random < len(items):
        random.seed(42)
        items = sorted(random.sample(items, args.random), key=lambda x: x["id"])
        print(f"  隨機抽樣 {len(items)} 張 (seed=42)", file=sys.stderr)
    elif args.limit:
        items = items[:args.limit]
    if not items:
        sys.exit("找不到任何查詢照片")
    print(f"[{tag}] {len(items)} 張查詢照片", file=sys.stderr)

    imgs = {it["id"]: IO.preprocess(it, cache / "img") for it in items}
    perc_all = IO.perceive_all_vlm(list(imgs.items()), cache / "perception_vlm.json", srv)
    IO.preload_label_embeddings(cache, ref_map, {k: perc_all[k] for k in imgs})

    if not dino_embed.available():
        sys.exit("DINOv2 不可用 — pip install torch transformers")
    embed = IO.EmbedCache(cache / "dino.json", dino_embed.embed_image, dino_embed._MODEL_NAME)
    dino_embed.embed_image = embed

    embs = LF._reference_embeddings(ref_map, LF._load_sidecar(C.DINO_SIDECAR_PATH))
    index = LF._index_for(ref_map, embs)
    rarity = LF._rarity_for(index)
    sim = LF._embedding_sim

    # 節點 → 建庫方向照片的物件（label+bbox），給候選用
    node_photo_objs = {}  # nid -> {slot: [(label,bbox), ...], "_file": {slot: photo_file}}
    for nid, node in ref_map.photos.items():
        slots, files = {}, {}
        for dp in node.directional_photos:
            slots[dp.slot] = [((o.label_norm or o.label), o.bbox) for o in dp.objects if o.label]
            files[dp.slot] = dp.photo_file
        node_photo_objs[nid] = {"slots": slots, "files": files}

    photos_out = []
    photo_dir = out / "photos_query" / tag
    t0 = time.perf_counter()
    for i, it in enumerate(items, 1):
        img, perc = imgs[it["id"]], perc_all[it["id"]]
        w, h = perc["size"]
        embed.key = it["id"]
        try:
            loc, loc_log = IO.localize(srv, C.NEO4J_PLACE, perc, img)
        except Exception as e:
            print(f"  localize 失敗 {it['id']}: {e}", file=sys.stderr)
            loc = None
        q_emb = embed(str(img))

        q_labels = [d.get("label_loc") or d["label"] for d in perc["detections"]]
        q_ocr_toks = [o["text"] for o in perc["ocr"]]
        query_items = [((d.get("label_loc") or d["label"]),
                        [d["box"][0] / w, d["box"][1] / h, d["box"][2] / w, d["box"][3] / h])
                       for d in perc["detections"]
                       if not photo_quality.is_occluder(d.get("label_loc") or d["label"])]
        query_objects = [{"label": l, "bbox": [round(x, 4) for x in b]} for l, b in query_items]
        query_relations = relations_from_objects(query_items)

        best = LF.best_photo_per_node(index, q_emb, q_labels, q_ocr_toks) if q_emb is not None else {}

        top = [(int(c[0]), float(c[1])) for c in (loc.top_candidates or [])][:5] if loc else []
        top5 = []
        for rank, (nid, score) in enumerate(top, 1):
            e = best.get(nid)
            slot = None
            ref_objs = []
            ref_file = None
            if e:
                dp = e[1].get("photo")
                if dp is not None:
                    slot = dp.slot
                    ref_objs = [((o.label_norm or o.label), o.bbox) for o in dp.objects if o.label]
                    ref_file = dp.photo_file
            db_objects = [{"label": l, "bbox": [round(x, 4) for x in b]} for l, b in ref_objs]
            db_relations = relations_from_objects(ref_objs)
            matched = verbose_relations(query_items, ref_objs, sim, rarity)
            top5.append({"rank": rank, "node": nid, "score": round(score, 4),
                         "slot": slot, "ref_photo_file": ref_file,
                         "db_objects": db_objects, "db_relations": db_relations,
                         "matched": matched})

        # web 用查詢照片
        safe = it["id"].replace("/", "__")
        dst = photo_dir / (Path(safe).stem + ".jpg")
        save_web_jpg(Path(it["path"]), dst)

        photos_out.append({
            "id": f"{tag}/{it['id']}", "tag": tag,
            "query_image": f"photos_query/{tag}/{Path(safe).stem}.jpg",
            "condition": it.get("condition"),
            "direction": (it.get("meta") or {}).get("direction"),
            "truth": it.get("truth"),
            "pred": top[0][0] if top else None,
            "confidence": round(float(loc.confidence), 4) if loc else None,
            "answered": bool(loc and loc.matched_nid is not None),
            "matched_slot": loc.matched_slot if loc else None,
            "method": loc.method if loc else None,
            "image_evidence_max": round(max((loc.image_evidence or {}).values(), default=0.0), 4) if loc else 0.0,
            "n_objects": len(query_objects), "n_ocr": len(perc["ocr"]),
            "query_objects": query_objects,
            "query_ocr": [{"text": o["text"]} for o in perc["ocr"]],
            "query_relations": query_relations,
            "top5": top5,
        })
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  ({time.perf_counter()-t0:.0f}s)", file=sys.stderr, flush=True)

    result = {"tag": tag, "generated": time.strftime("%Y-%m-%d %H:%M"),
              "place": C.NEO4J_PLACE, "n": len(photos_out),
              "fusion_weights": [LF.W_V, LF.W_N, LF.W_O],
              "photos": photos_out}
    outfile = out / f"web_results_{tag}.json"
    outfile.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"→ {outfile}  ({len(photos_out)} 張)", file=sys.stderr)


if __name__ == "__main__":
    main()
