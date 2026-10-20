"""What-if runs on the image_only caches: change one thing about the server's inputs, keep
everything else (same VLM replies, same DINOv2 vectors, same production localization),
no sensors, no API call.

  python -m eval.whatif --set lab1019 --change bilingual_labels --data-root <backend>

Changes:
  bilingual_labels — the VLM's object names as 「中文 English」 (zh + en from the cached raw
                     replies), the map's own label format, instead of the zh-only label
                     vlm._parse_perception keeps.

Output: eval/results/whatif_<set>_<change>.md / .json, compared with the same variants
on the unchanged inputs (the image_only _photos.jsonl of the same set).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from eval import image_only as io

BACKEND = Path(__file__).resolve().parents[1]
VARIANTS = ("prod", "dino_only", "no_text", "objects_only")


def _items_from_calls(calls, kind):
    """The first reply of this pass that the server's parser accepts → its raw items."""
    from server import vlm
    for c in calls:
        if c.get("kind") != kind or "text" not in c:
            continue
        t = c["text"]
        start = t.find("{")
        if start < 0:
            continue
        obj = vlm._load_json_or_salvage(t[start:])
        if obj is None:
            continue
        return [it for it in obj.get("items", []) if it.get("box") and len(it["box"]) == 4]
    return []


def bilingual_labels(perc):
    """Same detections in the same order (pass 1 then pass 2, as _merge_perceptions), with
    「zh en」 labels. Checked against the cached zh labels so the order cannot drift."""
    items = _items_from_calls(perc["vlm_calls"], "pass1") + _items_from_calls(perc["vlm_calls"], "pass2")
    dets = perc["detections"]
    if len(items) != len(dets):
        raise ValueError(f"{len(items)} raw items vs {len(dets)} detections")
    out = []
    for it, d in zip(items, dets):
        zh, en = str(it.get("zh", "")).strip(), str(it.get("en", "")).strip()
        if (zh or en) != d["label"]:
            raise ValueError(f"order drift: {zh or en!r} vs {d['label']!r}")
        out.append({**d, "label": f"{zh} {en}".strip() if zh else en})
    return {**perc, "detections": out}


CHANGES = {"bilingual_labels": bilingual_labels}


def _summ(recs):
    s = io.summarize(recs)
    by = {c: v["top1"] for c, v in s["by_condition"].items()}
    return {"top1": s["top1"], "top5": s["top5"], "within_1_hop": s["within_1_hop"],
            "error_m_median": s["error_m"]["median"], "photos_with_signal": s["photos_with_signal"],
            "top1_at_node": by.get("at_node"), "top1_between": by.get("between")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=io.SETS)
    ap.add_argument("--change", required=True, choices=tuple(CHANGES))
    ap.add_argument("--data-root", default=str(BACKEND))
    ap.add_argument("--cache")
    args = ap.parse_args(argv)

    import logging
    from server import config as C
    from server import dino_embed
    from server import locate_fusion as LF
    from server import server as srv
    logging.getLogger().setLevel(logging.WARNING)
    C.RERANK_ENABLED = False
    cache = Path(args.cache) if args.cache else BACKEND / "output" / "eval" / "image_only" / args.set
    ref_map, _ = io.load_map("snapshot", cache)
    srv.get_neo4j = lambda: io._MapSource(ref_map)
    hops = io.hop_table(ref_map)
    items = io.load_items(args.set, ref_map, Path(args.data_root))
    perc_all = io._load_json(cache / "perception_vlm.json")
    embed = io.EmbedCache(cache / "dino.json", dino_embed.embed_image, dino_embed._MODEL_NAME)
    dino_embed.embed_image = embed
    LF.server_image_evidence = io.EvidenceMemo(LF.server_image_evidence)
    change = CHANGES[args.change]

    # baseline: the same variants on unchanged inputs, from the image_only run
    base_file = BACKEND / "eval" / "results" / f"image_only_{args.set}_vlm_photos.jsonl"
    base = {d["id"]: d for d in map(json.loads, base_file.read_text(encoding="utf-8").splitlines())}

    new = {v: [] for v in VARIANTS}
    old = {v: [] for v in VARIANTS}
    t0 = time.perf_counter()
    for i, it in enumerate(items, 1):
        p = perc_all[it["id"]]
        q = change(p)
        img = io.preprocess(it, cache / "img")
        embed.key = it["id"]
        for v in VARIANTS:
            with io.variant(v):
                loc, _ = io.localize(srv, C.NEO4J_PLACE, q, img)
            rec = io.score_record(it, loc, ref_map, hops)
            rec["evidence_max"] = max((loc.image_evidence or {}).values(), default=0.0)
            new[v].append(rec)
            b = base[it["id"]]["results"][v][0]
            if b["pred"] is None or loc is None:
                sys.exit("missing result")
            ob = {**rec, "pred": b["pred"], "hit1": b["pred"] in it["ok"], "top5": b["top5"],
                  "confidence": b["confidence"]}
            ob["hit5"] = any(n in it["ok"] for n, _ in b["top5"])
            ob["hop"] = min(hops[o].get(b["pred"], 99) for o in it["ok"])
            ob["within1"] = ob["hop"] <= 1
            import math
            ob["err_m"] = math.dist((ref_map.photos[b["pred"]].pdr_x, ref_map.photos[b["pred"]].pdr_y), it["truth_xy"])
            ob["est_err_m"] = b["est_err_m"]
            ob["evidence_max"] = 1.0 if v != "objects_only" else None
            old[v].append(ob)
        print(f"\r  {i}/{len(items)} {time.perf_counter() - t0:5.0f}s", end="", file=sys.stderr)
    print(file=sys.stderr)

    # object-match coverage: how many query objects reach the threshold with some map label
    ref_labels = sorted({l for e in LF.build_fusion_index(ref_map, {})["entries"] for l in e["labels"]})
    import random
    sample = random.Random(0).sample([(it["id"], j) for it in items for j in range(len(perc_all[it["id"]]["detections"]))],
                                     600)   # all ~12k × 4387 map labels is ~50M comparisons

    def coverage(make_labels):
        labels = {}
        hit = 0
        for pid, j in sample:
            lab = labels.setdefault(pid, make_labels(perc_all[pid]))[j]
            hit += any(LF._embedding_sim(lab, rl) >= LF.NODE_SIM_THRESHOLD for rl in ref_labels)
        return round(100 * hit / len(sample), 1), len(sample)
    cov_old = coverage(lambda p: [d["label"] for d in p["detections"]])
    cov_new = coverage(lambda p: [d["label"] for d in change(p)["detections"]])

    rep = {"set": args.set, "change": args.change, "photos": len(items),
           "objects_reaching_threshold_pct": {"before": cov_old, "after": cov_new},
           "variants": {v: {"before": _summ(old[v]), "after": _summ(new[v]),
                            "fixes": sum(a["hit1"] and not b["hit1"] for a, b in zip(new[v], old[v])),
                            "breaks": sum(b["hit1"] and not a["hit1"] for a, b in zip(new[v], old[v]))}
                        for v in VARIANTS}}
    obj_new = new["objects_only"]
    rep["variants"]["objects_only"]["before"]["photos_with_signal"] = None
    L = [f"# What-if：{args.change}（{args.set}，不帶感測器）", "",
         f"- {len(items)} 張；同一份 VLM 回覆與 DINOv2，只把照片的物件名稱改成「中文 English」，相似度仍是伺服器的字元相似度",
         f"- 照片物件中能和地圖任一物件配到（≥{LF.NODE_SIM_THRESHOLD}）的比例：{cov_old[0]}% → {cov_new[0]}%（隨機抽 {cov_old[1]} 個物件估計）",
         f"- 只有物件時，影像證據有訊號的照片：{rep['variants']['objects_only']['after']['photos_with_signal']}%（之前 17.8%）", "",
         "| 版本 | Top-1 前 → 後 | Top-5 前 → 後 | 1 格內 前 → 後 | 節點 Top-1 前 → 後 | 中點 Top-1 前 → 後 | 救回 / 弄錯 |",
         "|---|---|---|---|---|---|---|"]
    for v, s in rep["variants"].items():
        b, a = s["before"], s["after"]
        L.append(f"| {v} | {b['top1']} → {a['top1']}% | {b['top5']} → {a['top5']}% | {b['within_1_hop']} → {a['within_1_hop']}% | "
                 f"{b['top1_at_node']} → {a['top1_at_node']}% | {b['top1_between']} → {a['top1_between']}% | {s['fixes']} / {s['breaks']} |")
    out = BACKEND / "eval" / "results" / f"whatif_{args.set}_{args.change}"
    Path(f"{out}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(f"{out}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
