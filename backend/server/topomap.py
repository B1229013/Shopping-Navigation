"""Topological map: nodes are photo locations, edges are movement actions."""
from __future__ import annotations

import io
from datetime import datetime
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402


class TopoMap:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._next_id: int = 0

    def add_node(
        self, photo_path: str, detected: List[str], summary: str,
        ocr_texts: Optional[List[str]] = None,
    ) -> int:
        nid = self._next_id
        self._next_id += 1
        self.graph.add_node(
            nid,
            photo_path=photo_path,
            detected=detected,
            summary=summary,
            ocr_texts=ocr_texts or [],
            timestamp=datetime.utcnow().isoformat(),
        )
        return nid

    def add_edge(self, from_id: int, to_id: int, action: str) -> None:
        self.graph.add_edge(from_id, to_id, action=action)

    def to_dict(self, current_node: Optional[int], goal_node: Optional[int]) -> dict:
        nodes = [
            {
                "id": nid,
                "photo": data["photo_path"],
                "detected": data["detected"],
                "ocr_texts": data.get("ocr_texts", []),
                "summary": data["summary"],
                "timestamp": data["timestamp"],
            }
            for nid, data in self.graph.nodes(data=True)
        ]
        edges = [
            {"from": u, "to": v, "action": data["action"]}
            for u, v, data in self.graph.edges(data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "current_node": current_node,
            "goal_node": goal_node,
        }

    def summarize_for_vlm(self, current_id: int) -> str:
        """Walk from start to current, produce <150-word prose summary for the VLM."""
        if self.graph.number_of_nodes() == 0:
            return "No locations visited yet."

        try:
            start = next(n for n in self.graph.nodes if self.graph.in_degree(n) == 0)
        except StopIteration:
            start = 0

        if start == current_id:
            data = self.graph.nodes[current_id]
            return f"You are at the starting location. Visible: {data['summary']}."

        try:
            path = nx.shortest_path(self.graph, source=start, target=current_id)
        except nx.NetworkXNoPath:
            path = [current_id]

        parts: List[str] = []
        for i, nid in enumerate(path):
            node = self.graph.nodes[nid]
            label = node["summary"] or ", ".join(node["detected"][:3]) or f"location {nid}"
            ocr = node.get("ocr_texts", [])
            if ocr:
                label += f' [signs: {", ".join(ocr[:3])}]'
            if i == 0:
                parts.append(f"Started at {label}")
            else:
                action = self.graph.edges[path[i - 1], nid]["action"]
                parts.append(f"{action}, arrived at {label}")
        return ". ".join(parts) + "."

    def render_png(self, current_id: Optional[int] = None) -> bytes:
        fig, ax = plt.subplots(figsize=(6, 6))
        if self.graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "(empty map)", ha="center", va="center")
        else:
            pos = nx.spring_layout(self.graph, seed=42)
            node_colors = ["red" if n == current_id else "lightblue" for n in self.graph.nodes]
            nx.draw(
                self.graph, pos, ax=ax, with_labels=True,
                node_color=node_colors, node_size=600, font_size=10,
            )
            edge_labels = {(u, v): d["action"][:20] for u, v, d in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=ax, font_size=8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
