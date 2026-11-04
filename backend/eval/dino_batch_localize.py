"""DINOv2-only batch localization — no VLM/API key needed.

Computes DINOv2 embeddings for query photos and matches against the
reference map's embeddings to produce top-5 candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def load_ref_embeddings():
    """Load reference DINOv2 embeddings + node metadata from dino_sidecar.pkl."""
    sidecar_path = BACKEND / "dino_sidecar.pkl"
    meta_path = BACKEND.parent / "APPNAV_ios" / "Models" / "dino_ref_meta.json"

    side = pickle.load(open(sidecar_path, "rb"))
    meta = json.loads(meta_path.read_text())

    names = [e["file"] for e in meta]
    emb = np.stack([np.asarray(side[n], dtype=np.float32) for n in names])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    nids = np.array([e["nid"] for e in meta])
    slots = [e.get("slot", e.get("direction", "?")) for e in meta]
    return emb, nids, names, slots, meta


def load_topomap(path):
    """Load node positions and edges from topomap.json."""
    d = json.loads(Path(path).read_text())
    nodes = {}
    for n in d.get("nodes", []):
        nid = n["id"]
        nodes[nid] = {"x": n["x"], "y": n["y"], "sources": n.get("sources", [])}
    edges = d.get("edges", [])
    return nodes, edges


def bfs_distance(nodes, edges, src, dst):
    """Compute hop distance between two nodes via BFS."""
    if src == dst:
        return 0
    adj = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])
        adj.setdefault(e["to"], []).append(e["from"])
    visited = {src}
    queue = [(src, 0)]
    while queue:
        n, d = queue.pop(0)
        for nb in adj.get(n, []):
            if nb == dst:
                return d + 1
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, d + 1))
    return None


def euclidean_dist(nodes, a, b):
    if a not in nodes or b not in nodes:
        return None
    dx = nodes[a]["x"] - nodes[b]["x"]
    dy = nodes[a]["y"] - nodes[b]["y"]
    return round((dx * dx + dy * dy) ** 0.5, 2)


def photo_for_node(nodes_data, nid, slot=None):
    """Get the reference photo file for a node, preferring the given slot."""
    nd = nodes_data.get(nid)
    if not nd:
        return None
    sources = nd.get("sources", [])
    if not sources:
        return None
    src = sources[0]
    photos = src.get("photos", {})
    if slot and slot in photos:
        return photos[slot]
    for s in ["front", "right", "back", "left"]:
        if s in photos:
            return photos[s]
    return None


def localize_photos(query_dir, ref_emb, ref_nids, ref_names, ref_slots, ref_meta,
                    topo_nodes, topo_edges, tag, out_dir, limit=0):
    from server import dino_embed

    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in Path(query_dir).iterdir() if f.suffix.lower() in exts)
    if limit:
        files = files[:limit]

    print(f"[{tag}] {len(files)} query photos, {len(ref_emb)} reference embeddings", file=sys.stderr)

    # Find which set directories contain which photos
    set_dirs = {}
    for m in ref_meta:
        nid = m["nid"]
        nd = topo_nodes.get(nid)
        if nd and nd.get("sources"):
            s = nd["sources"][0].get("set", "01")
            set_dirs[nid] = s

    results = []
    t0 = time.perf_counter()

    for i, fpath in enumerate(files, 1):
        q_emb = dino_embed.embed_image(str(fpath))
        if q_emb is None:
            print(f"  SKIP {fpath.name}: DINOv2 embed failed", file=sys.stderr)
            continue

        # Cosine similarity against all reference photos
        sims = ref_emb @ q_emb

        # Aggregate by node (max sim per node)
        node_scores = {}
        node_best_idx = {}
        for j in range(len(sims)):
            nid = int(ref_nids[j])
            if nid not in node_scores or sims[j] > node_scores[nid]:
                node_scores[nid] = float(sims[j])
                node_best_idx[nid] = j

        # Sort by score and take top 5
        ranked = sorted(node_scores.items(), key=lambda x: -x[1])[:5]

        # Normalize scores to sum to 1 (softmax-like)
        scores_raw = np.array([s for _, s in ranked])
        exp_scores = np.exp((scores_raw - scores_raw.max()) * 10)  # temperature=0.1
        probs = exp_scores / exp_scores.sum()

        top5 = []
        for rank, ((nid, raw_score), prob) in enumerate(zip(ranked, probs), 1):
            best_j = node_best_idx[nid]
            slot = ref_slots[best_j]
            ref_file = ref_names[best_j]
            nd = topo_nodes.get(nid, {})
            ref_set = set_dirs.get(nid, "01")

            entry = {
                "rank": rank,
                "node": nid,
                "confidence": round(float(prob), 4),
                "raw_similarity": round(float(raw_score), 4),
                "x": nd.get("x"),
                "y": nd.get("y"),
                "ref_photo": f"0916測試/{ref_set}/{ref_file}" if ref_file else None,
                "ref_set": ref_set,
                "slot": slot,
                "db_objects": None,
                "db_relations": None,
                "matched": None,
            }
            top5.append(entry)

        pred_nid = ranked[0][0] if ranked else None

        results.append({
            "id": f"{tag}/{fpath.name}",
            "group": tag,
            "query_image": f"photos_query/{tag}/{fpath.name}",
            "condition": "between",
            "direction": None,
            "truth": None,
            "pred": pred_nid,
            "confidence": round(float(probs[0]), 4) if len(probs) > 0 else None,
            "answered": pred_nid is not None,
            "hit1": None,
            "hit5": None,
            "err_m": None,
            "n_objects": None,
            "n_ocr": None,
            "top5": top5,
            "query_objects": None,
            "query_ocr": None,
            "query_relations": None,
            "auto_class": None,
            "auto_hop": None,
            "auto_m": None,
            "pending_localization": False,
            "pending_detail": True,
        })

        if i % 10 == 0 or i == len(files):
            elapsed = time.perf_counter() - t0
            print(f"  {i}/{len(files)} ({elapsed:.0f}s) pred=WP{pred_nid} conf={probs[0]:.3f}",
                  file=sys.stderr, flush=True)

    return results


def localize_0916_set(ref_emb, ref_nids, ref_names, ref_slots, ref_meta,
                      topo_nodes, topo_edges, data_root, tag, out_dir, limit=0):
    """Localize 0916 test photos using leave-one-out (exclude the test photo's own node)."""
    from server import dino_embed
    from eval import image_only as IO
    from server import config as C

    # Load items from eval.image_only's dataset
    refmap_path = BACKEND / "output" / "eval" / "image_only" / "0916" / "refmap.pkl"
    if refmap_path.exists():
        ref_map = pickle.load(open(refmap_path, "rb"))
    else:
        ref_map, _ = IO.load_map("neo4j", BACKEND / "output" / "eval" / "image_only" / "0916")

    items = IO.load_items("0916", ref_map, Path(data_root).expanduser())
    if limit:
        items = items[:limit]
    print(f"[{tag}] {len(items)} test photos", file=sys.stderr)

    set_dirs = {}
    for m in ref_meta:
        nid = m["nid"]
        nd = topo_nodes.get(nid)
        if nd and nd.get("sources"):
            s = nd["sources"][0].get("set", "01")
            set_dirs[nid] = s

    results = []
    t0 = time.perf_counter()
    for i, it in enumerate(items, 1):
        img_path = it["path"]
        truth_nid = it.get("truth")

        q_emb = dino_embed.embed_image(str(img_path))
        if q_emb is None:
            continue

        sims = ref_emb @ q_emb
        node_scores = {}
        node_best_idx = {}
        for j in range(len(sims)):
            nid = int(ref_nids[j])
            if nid not in node_scores or sims[j] > node_scores[nid]:
                node_scores[nid] = float(sims[j])
                node_best_idx[nid] = j

        ranked = sorted(node_scores.items(), key=lambda x: -x[1])[:5]
        scores_raw = np.array([s for _, s in ranked])
        exp_scores = np.exp((scores_raw - scores_raw.max()) * 10)
        probs = exp_scores / exp_scores.sum()

        top5 = []
        for rank, ((nid, raw_score), prob) in enumerate(zip(ranked, probs), 1):
            best_j = node_best_idx[nid]
            slot = ref_slots[best_j]
            ref_file = ref_names[best_j]
            nd = topo_nodes.get(nid, {})
            ref_set = set_dirs.get(nid, "01")

            hop = bfs_distance(topo_nodes, topo_edges, truth_nid, nid) if truth_nid else None
            m_dist = euclidean_dist(topo_nodes, truth_nid, nid) if truth_nid else None

            entry = {
                "rank": rank,
                "node": nid,
                "confidence": round(float(prob), 4),
                "raw_similarity": round(float(raw_score), 4),
                "x": nd.get("x"),
                "y": nd.get("y"),
                "ref_photo": f"0916測試/{ref_set}/{ref_file}" if ref_file else None,
                "ref_set": ref_set,
                "slot": slot,
                "hop_to_truth": hop,
                "m_to_truth": m_dist,
                "db_objects": None,
                "db_relations": None,
                "matched": None,
            }
            top5.append(entry)

        pred_nid = ranked[0][0] if ranked else None
        hit1 = (pred_nid == truth_nid) if truth_nid else None
        hit5 = any(nid == truth_nid for nid, _ in ranked) if truth_nid else None

        hop1 = bfs_distance(topo_nodes, topo_edges, truth_nid, pred_nid) if (truth_nid and pred_nid) else None
        err_m = euclidean_dist(topo_nodes, truth_nid, pred_nid) if (truth_nid and pred_nid) else None

        auto_class = None
        if hop1 is not None:
            if hop1 <= 1:
                auto_class = "correct"
            elif hop1 <= 4:
                auto_class = "nearby"
            else:
                auto_class = "wrong"

        cond = it.get("condition", "at_node")
        direction = (it.get("meta") or {}).get("direction")

        results.append({
            "id": f"{tag}/{it['id']}",
            "group": tag,
            "query_image": None,
            "condition": cond,
            "direction": direction,
            "truth": truth_nid,
            "pred": pred_nid,
            "confidence": round(float(probs[0]), 4) if len(probs) > 0 else None,
            "answered": pred_nid is not None,
            "hit1": hit1,
            "hit5": hit5,
            "err_m": err_m,
            "n_objects": None,
            "n_ocr": None,
            "top5": top5,
            "query_objects": None,
            "query_ocr": None,
            "query_relations": None,
            "auto_class": auto_class,
            "auto_hop": hop1,
            "auto_m": err_m,
            "pending_localization": False,
            "pending_detail": True,
        })

        if i % 20 == 0 or i == len(items):
            elapsed = time.perf_counter() - t0
            acc = sum(1 for r in results if r["hit1"]) / len(results) if results else 0
            print(f"  {i}/{len(items)} ({elapsed:.0f}s) top1_acc={acc:.1%}",
                  file=sys.stderr, flush=True)

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos", help="Query photos folder (e.g. 樂家康 JPGs)")
    ap.add_argument("--set", help="Named set like '0916' (uses eval.image_only's dataset)")
    ap.add_argument("--tag", help="Output tag/group name")
    ap.add_argument("--topomap", required=True, help="Path to topomap.json")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.photos and not args.set:
        ap.error("Provide --photos or --set")

    tag = args.tag or args.set or Path(args.photos).name
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print("Loading reference embeddings...", file=sys.stderr)
    ref_emb, ref_nids, ref_names, ref_slots, ref_meta = load_ref_embeddings()
    print(f"  {len(ref_emb)} reference photos across {len(set(ref_nids.tolist()))} nodes", file=sys.stderr)

    print("Loading topomap...", file=sys.stderr)
    topo_nodes, topo_edges = load_topomap(args.topomap)
    print(f"  {len(topo_nodes)} nodes, {len(topo_edges)} edges", file=sys.stderr)

    if args.photos:
        results = localize_photos(
            args.photos, ref_emb, ref_nids, ref_names, ref_slots, ref_meta,
            topo_nodes, topo_edges, tag, out, args.limit)
    else:
        results = localize_0916_set(
            ref_emb, ref_nids, ref_names, ref_slots, ref_meta,
            topo_nodes, topo_edges, str(BACKEND), tag, out, args.limit)

    outfile = out / f"dino_results_{tag}.json"
    data = {
        "tag": tag,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "method": "DINOv2 embedding similarity",
        "n": len(results),
        "photos": results,
    }
    outfile.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {outfile} ({len(results)} photos)", file=sys.stderr)


if __name__ == "__main__":
    main()
