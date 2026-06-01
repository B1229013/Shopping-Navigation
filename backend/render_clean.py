"""Compact, readable topological-map renderer for the store walk.

The stock generate_topomap renderer sprawls vertically (per-zone text dumps) and
overflows matplotlib's pixel limit. This draws a clean snake layout with one tidy
card per zone: an auto title from its strongest sign categories, the aisle/photo
range, and its top category tags. Output stays < 2000px so it is directly viewable.

Usage (WSL unigoal env):
    python render_clean.py store_detections_enriched.json store_map_clean.png [sim_threshold]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from generate_topomap import (
    load_detections, cluster_photos_into_zones, build_zone_info, build_edges,
)

GENERIC = {"store", "person", "cart", "sign aisle sign", "cabinet shelf",
           "door", "fire extinguisher", "shelf", "sign"}

# Nice display names for canonical category tokens.
PRETTY = {
    "frozen-food": "Frozen", "chilled-food": "Chilled", "hotpot": "Hot Pot",
    "beer": "Beer", "tea": "Tea", "breakfast": "Breakfast", "dry-goods": "Dry Goods",
    "rice": "Rice", "imported-food": "Imported Food", "canned-food": "Canned Food",
    "biscuits": "Biscuits", "puffs": "Puffs", "instant-noodles": "Instant Noodles",
    "snacks": "Snacks", "rice-crackers": "Rice Crackers", "crackers": "Crackers",
    "laundry": "Laundry", "clothing": "Clothing", "family-health": "Family Health",
    "kitchenware": "Kitchenware", "body-hair-care": "Body/Hair Care", "hygiene": "Hygiene",
    "candy": "Candy", "chocolate": "Chocolate", "nuts": "Nuts", "tea-soda": "Tea/Soda",
    "coffee-juice": "Coffee/Juice", "jerky-seaweed": "Jerky/Seaweed", "cashier": "Cashier",
    "store-brand": "Carrefour", "refrigerator": "Fridges", "beer crate": "Beer Crates",
    "umbrella": "Japanese Sec.", "bread": "Bakery", "water bottle": "Water",
    "produce": "Produce", "milk": "Milk",
}


def zone_title(info) -> str:
    cats = [l for l, _ in info["all_objects"] if l not in GENERIC]
    pretty = [PRETTY.get(c, c.title()) for c in cats]
    return " / ".join(pretty[:3]) if pretty else "Transit"


def main() -> int:
    inp = sys.argv[1] if len(sys.argv) > 1 else "store_detections_enriched.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "store_map_clean.png"
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.40

    photos = load_detections(inp)
    zones = cluster_photos_into_zones(photos, sim_threshold=thr)
    zi = build_zone_info(zones)
    edges = build_edges(zones, zi)

    n = len(zi)
    per_row = 5
    pos = {}
    for i in range(n):
        row, col = divmod(i, per_row)
        if row % 2 == 1:
            col = per_row - 1 - col
        pos[i] = (col, -row)

    rows = (n + per_row - 1) // per_row
    fig, ax = plt.subplots(figsize=(3.6 * min(n, per_row), 3.4 * rows + 0.6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # edges first
    for a, b, shared in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
            color="#58a6ff", lw=2.0, alpha=0.8,
            shrinkA=46, shrinkB=46, zorder=1))
        sh = [s for s in shared if s not in GENERIC][:3]
        if sh:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12,
                    ", ".join(PRETTY.get(s, s) for s in sh),
                    ha="center", va="bottom", fontsize=6.5, color="#8b949e", style="italic")

    BW, BH = 0.92, 0.92
    for i, info in zi.items():
        x, y = pos[i]
        is_start = (i == 0)
        is_end = (i == n - 1)
        face = "#13261c" if is_end else ("#2d0d0d" if is_start else "#161b22")
        edge = "#3fb950" if is_end else ("#f85149" if is_start else "#30363d")
        ax.add_patch(FancyBboxPatch(
            (x - BW / 2, y - BH / 2), BW, BH,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=face, edgecolor=edge, lw=2.2, zorder=2))

        tag = "  ▶ START" if is_start else ("  ■ END" if is_end else "")
        ax.text(x, y + 0.36, f"Zone {i}{tag}", ha="center", va="center",
                fontsize=9.5, color="#ffffff", fontweight="bold", zorder=3)
        ax.text(x, y + 0.22, zone_title(info), ha="center", va="center",
                fontsize=8.5, color="#58a6ff", fontweight="bold", zorder=3)

        cats = [PRETTY.get(l, l.title()) for l, _ in info["all_objects"] if l not in GENERIC]
        body = ", ".join(cats) if cats else "(transit / no signage)"
        ax.text(x, y - 0.05, _wrap(body, 30), ha="center", va="center",
                fontsize=6.8, color="#c9d1d9", zorder=3, linespacing=1.3)
        rng = info["photo_range"].replace(".jpg", "").replace("_", " ")
        ax.text(x, y - BH / 2 + 0.10, f'{len(info["photos"])} photos\n{rng}', ha="center", va="bottom",
                fontsize=5.6, color="#6e7681", zorder=3, linespacing=1.2)

    ax.set_title(
        f"Carrefour store — topological map  |  {sum(len(i['photos']) for i in zi.values())} photos "
        f"→ {n} zones → {len(edges)} edges  (Jaccard sim ≥ {thr}, GroundingDINO + OCR signage)",
        fontsize=11, color="#58a6ff", fontweight="bold", pad=14)
    ax.set_xlim(-0.8, min(n, per_row) - 0.2)
    ax.set_ylim(-rows + 0.3, 0.85)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    print(f"Saved {out}")
    return 0


def _wrap(s: str, width: int) -> str:
    words, lines, cur = s.split(", "), [], ""
    for w in words:
        if len(cur) + len(w) + 2 > width and cur:
            lines.append(cur.rstrip(", "))
            cur = ""
        cur += w + ", "
    if cur:
        lines.append(cur.rstrip(", "))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
