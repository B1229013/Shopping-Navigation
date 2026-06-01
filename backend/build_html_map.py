"""Interactive drag-to-arrange topological map (Pyvis / HTML) — CLAUDE.md TASK 17.

Open the resulting store_map.html in any browser. Each node is a store zone; drag
them to match the real floor plan. Hover a node for its full category list, aisle
range and photo list; hover an edge for the shared signage that links two zones.

Usage (WSL unigoal env):
    python build_html_map.py store_detections_enriched.json store_map.html [sim_threshold]
"""
from __future__ import annotations

import sys

from pyvis.network import Network

from generate_topomap import (
    load_detections, cluster_photos_into_zones, build_zone_info, build_edges,
)
from render_clean import zone_title, PRETTY, GENERIC


def main() -> int:
    inp = sys.argv[1] if len(sys.argv) > 1 else "store_detections_enriched.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "store_map.html"
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.40

    photos = load_detections(inp)
    zones = cluster_photos_into_zones(photos, sim_threshold=thr)
    zi = build_zone_info(zones)
    edges = build_edges(zones, zi)
    n = len(zi)

    net = Network(height="800px", width="100%", bgcolor="#0d1117",
                  font_color="#c9d1d9", directed=True)
    net.toggle_physics(False)  # fixed initial layout, fully draggable

    for i, info in zi.items():
        cats = [PRETTY.get(l, l.title()) for l, _ in info["all_objects"] if l not in GENERIC]
        is_start, is_end = (i == 0), (i == n - 1)
        color = "#f85149" if is_start else ("#3fb950" if is_end else "#1f6feb")
        tag = " ▶START" if is_start else (" ■END" if is_end else "")
        label = f"Zone {i}{tag}\n{zone_title(info)}"
        rng = info["photo_range"].replace(".jpg", "").replace("_", " ")
        tip = (f"Zone {i}{tag}\nCategories: {', '.join(cats) if cats else '(transit)'}\n"
               f"{len(info['photos'])} photos: {rng}")
        net.add_node(i, label=label, title=tip, color=color,
                     shape="box", x=i * 220, y=0, physics=False,
                     borderWidth=3, font={"size": 16, "multi": True})

    for a, b, shared in edges:
        sh = [PRETTY.get(s, s) for s in shared if s not in GENERIC][:6]
        net.add_edge(a, b, title=", ".join(sh) if sh else "adjacent",
                     color="#58a6ff", width=2)

    net.set_options('{"interaction":{"hover":true,"dragNodes":true},'
                    '"edges":{"smooth":false,"arrows":{"to":{"enabled":true}}}}')
    net.write_html(out, notebook=False)
    print(f"Saved {out}  ({n} zones, {len(edges)} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
