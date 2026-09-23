"""Branched topological map of the store (floor-plan style), anchored at the entrance.

Unlike the photo-sequence clustering (which is a 1-D chain because the walk was
linear), this models the real store topology: a main walkway with the numbered
aisles 1-13 branching off it, the perimeter perishable sections around the
outside, and a loop closing back at the entrance.

Grounding:
  - Aisle NUMBERS (1-13) were read by EasyOCR from the aisle signs.
  - Each aisle's CATEGORY is the OCR section sign seen at that aisle (best effort
    for the middle aisles, where several signs were visible at once).
  - PERIMETER sections + the walk CONNECTIONS come from the photos in walk order.
  - Exact geometry is schematic; the topology (what connects to what) is faithful.

Outputs: store_graph.png  and  store_graph.html
Run (WSL unigoal env):  python build_store_graph.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from pyvis.network import Network

# aisle number -> (categories...). num 13 = far end (entered from the dairy/beer
# wall), num 1-2 = near the cashier/entrance.
AISLES = {
    13: ["Imported Food"],
    12: ["Canned Food", "Biscuits"],
    11: ["Instant Noodles", "Puffs"],
    10: ["Biscuits", "Snacks"],
    9:  ["Snacks / Chips"],
    8:  ["Laundry"],
    7:  ["Clothing"],
    6:  ["Family Health", "Body/Hair Care"],
    5:  ["Crackers", "Rice Crackers"],
    4:  ["Kitchenware"],
    3:  ["Candy", "Chocolate", "Nuts"],
    2:  ["Tea/Soda", "Coffee/Juice"],
    1:  ["Rice", "Breakfast", "Dry Goods"],
}

# perimeter / perishable nodes: id -> (label, x, y, kind)
PERIM = {
    "entrance": ("ENTRANCE\nProduce · Promo · Lanterns", -2.7, 0.0, "entrance"),
    "beverages": ("Beverages / Promo\nTea · Water pallets", -2.7, 2.1, "perim"),
    "dairybeer": ("Dairy Fridges & Taiwan Beer\n(front-aisle wall)", -1.6, 3.9, "perim"),
    "cashiers": ("CASHIERS / Checkout", 13.2, 1.4, "cashier"),
    "service": ("Service desk\n(Carrefour / GO JAPAN)", 9.8, -1.2, "perim"),
    "deli": ("Coffee · Chocolate · Deli", 6.6, -1.4, "perim"),
    "frozen": ("Frozen Islands\n& Japanese Section", 3.6, -1.4, "perim"),
    "milk": ("Milk / Chilled Island", 0.7, -1.0, "perim"),
}

# topological edges (id -> id); aisle nodes are "a13".."a1"
PERIM_EDGES = [
    ("entrance", "beverages"), ("beverages", "dairybeer"), ("dairybeer", "a13"),
    ("a2", "cashiers"), ("a1", "cashiers"),
    ("cashiers", "service"), ("service", "deli"), ("deli", "frozen"),
    ("frozen", "milk"), ("milk", "entrance"),
]


def aisle_pos(num: int) -> tuple[float, float]:
    return (float(13 - num), 3.0)  # a13 at x=0 (left) .. a1 at x=12 (right)


def build_graph() -> nx.Graph:
    G = nx.Graph()
    for num, cats in AISLES.items():
        x, y = aisle_pos(num)
        G.add_node(f"a{num}", label=f"Aisle {num}\n" + " / ".join(cats),
                   x=x, y=y, kind="aisle")
    for num in range(13, 1, -1):  # chain a13-a12-...-a1 (the main walkway)
        G.add_edge(f"a{num}", f"a{num-1}", kind="walkway")
    for nid, (label, x, y, kind) in PERIM.items():
        G.add_node(nid, label=label, x=x, y=y, kind=kind)
    for a, b in PERIM_EDGES:
        G.add_edge(a, b, kind="perim")
    return G


STYLE = {  # kind -> (face, edge)
    "entrance": ("#2d0d0d", "#f85149"),
    "cashier":  ("#13261c", "#3fb950"),
    "aisle":    ("#161b22", "#1f6feb"),
    "perim":    ("#1c2333", "#8957e5"),
}


def render_png(G: nx.Graph, out="store_graph.png") -> None:
    fig, ax = plt.subplots(figsize=(17, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # main walkway bar
    ax.plot([-0.4, 12.4], [3.0, 3.0], color="#30363d", lw=14, solid_capstyle="round", zorder=1)
    ax.text(6, 3.0, "MAIN WALKWAY", ha="center", va="center", fontsize=8,
            color="#6e7681", fontweight="bold", zorder=2)

    # perimeter loop edges
    for a, b in PERIM_EDGES:
        x1, y1 = (aisle_pos(int(a[1:])) if a.startswith("a") else PERIM[a][1:3])
        x2, y2 = (aisle_pos(int(b[1:])) if b.startswith("a") else PERIM[b][1:3])
        ax.plot([x1, x2], [y1, y2], color="#3a4250", lw=2.5, zorder=1)

    # aisle branches (drawn as tall boxes sticking UP from the walkway)
    for num, cats in AISLES.items():
        x, y = aisle_pos(num)
        ax.add_patch(FancyBboxPatch((x - 0.42, y + 0.15), 0.84, 1.7,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     facecolor=STYLE["aisle"][0], edgecolor=STYLE["aisle"][1],
                     lw=1.8, zorder=3))
        ax.text(x, y + 1.66, f"Aisle {num}", ha="center", va="top",
                fontsize=8.5, color="#58a6ff", fontweight="bold", zorder=4)
        ax.text(x, y + 1.34, "\n".join(cats), ha="center", va="top",
                fontsize=6.4, color="#c9d1d9", zorder=4, linespacing=1.3)

    # perimeter nodes
    for nid, (label, x, y, kind) in PERIM.items():
        face, edge = STYLE[kind]
        ax.add_patch(FancyBboxPatch((x - 1.05, y - 0.42), 2.1, 0.84,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor=face, edgecolor=edge, lw=2.2, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.2,
                color="#ffffff", fontweight="bold", zorder=4, linespacing=1.2)

    ax.set_title(
        "Carrefour store — topological map (entrance-anchored, aisles branch off the main walkway)\n"
        "aisle numbers from OCR · perimeter sections + connections from the walk · geometry schematic",
        fontsize=12, color="#58a6ff", fontweight="bold", pad=16)
    ax.set_xlim(-4.2, 14.8)
    ax.set_ylim(-2.4, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=140, facecolor="#0d1117", bbox_inches="tight")
    print(f"Saved {out}")


def render_html(G: nx.Graph, out="store_graph.html") -> None:
    net = Network(height="820px", width="100%", bgcolor="#0d1117",
                  font_color="#c9d1d9")
    net.toggle_physics(False)
    for nid, d in G.nodes(data=True):
        face, edge = STYLE[d["kind"]]
        net.add_node(nid, label=d["label"], shape="box",
                     x=d["x"] * 95, y=-d["y"] * 95, physics=False,
                     color={"background": face, "border": edge},
                     borderWidth=2, font={"size": 14, "multi": True})
    for a, b, d in G.edges(data=True):
        col = "#30363d" if d["kind"] == "walkway" else "#8957e5"
        w = 6 if d["kind"] == "walkway" else 2
        net.add_edge(a, b, color=col, width=w)
    net.set_options('{"interaction":{"hover":true,"dragNodes":true},'
                    '"edges":{"smooth":false}}')
    net.write_html(out, notebook=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    G = build_graph()
    render_png(G)
    render_html(G)
