"""Export graph topology (node positions + edges) for the eval report viewer.

Usage (from backend/):
    python eval/export_graph.py lab1019
    python eval/export_graph.py xiban1015

Outputs: eval/results/<set>_graph.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
CACHE_BASE = BACKEND / "output" / "eval" / "image_only"

PLACE_MAP = {
    "lab1019": "A7家樂福（9/16)",
    "xiban1015": "系辦",
    "0916": "A7家樂福（9/16)",
    "field1017": "A7家樂福（9/16)",
}


def _load_ref_map(set_name: str):
    # pickle contains server.neo4j_client classes
    sys.path.insert(0, str(BACKEND))
    snap = CACHE_BASE / set_name / "refmap.pkl"
    if snap.exists():
        with open(snap, "rb") as f:
            return pickle.load(f)

    from dotenv import load_dotenv
    load_dotenv(BACKEND.parent / ".env")
    from server.neo4j_client import get_neo4j
    neo4j = get_neo4j()
    if neo4j is None:
        sys.exit("Neo4j not configured — fill NEO4J_* in the repo-root .env")
    ref_map = neo4j.load_reference_map(PLACE_MAP[set_name])
    if not ref_map.photos:
        sys.exit(f"Neo4j returned an empty map for {PLACE_MAP[set_name]!r}")
    snap.parent.mkdir(parents=True, exist_ok=True)
    with open(snap, "wb") as f:
        pickle.dump(ref_map, f)
    return ref_map


def export(set_name: str):
    if set_name not in PLACE_MAP:
        sys.exit(f"Unknown set: {set_name}. Choose from: {', '.join(PLACE_MAP)}")

    ref_map = _load_ref_map(set_name)
    nodes = {}
    edges = []
    seen = set()
    for nid, p in ref_map.photos.items():
        nodes[str(nid)] = {"x": round(p.pdr_x, 3), "y": round(p.pdr_y, 3),
                           "heading": round(p.heading_deg, 1) if p.heading_deg else None}
        for nb in p.neighbor_nids:
            edge = tuple(sorted((nid, nb)))
            if edge not in seen:
                seen.add(edge)
                edges.append(list(edge))

    out = {"place": PLACE_MAP[set_name], "set": set_name,
           "nodes": nodes, "edges": edges}
    dest = RESULTS / f"{set_name}_graph.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {dest}  ({len(nodes)} nodes, {len(edges)} edges)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python eval/export_graph.py <set_name>")
    for name in sys.argv[1:]:
        export(name)
