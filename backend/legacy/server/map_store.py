"""Persistent map storage — save / list / load / delete topological maps as JSON."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from server.topomap import TopoMap


class MapStore:
    def __init__(self, data_dir: str = "output/maps") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Save ──

    def save(
        self,
        topomap: TopoMap,
        name: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        map_id = uuid.uuid4().hex[:8]
        meta = metadata or {}

        total_steps = 0
        total_distance = 0.0
        for _u, _v, data in topomap.graph.edges(data=True):
            total_steps += data.get("steps", 0) or 0
            total_distance += data.get("distance_m", 0.0) or 0.0

        nodes = []
        for nid, data in topomap.graph.nodes(data=True):
            nodes.append({
                "id": nid,
                "detected": data.get("detected", []),
                "ocr_texts": data.get("ocr_texts", []),
                "summary": data.get("summary", ""),
                "timestamp": data.get("timestamp", ""),
                "pdr_x": data.get("pdr_x"),
                "pdr_y": data.get("pdr_y"),
            })

        edges = []
        for u, v, data in topomap.graph.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "action": data.get("action", ""),
                "steps": data.get("steps"),
                "distance_m": data.get("distance_m"),
                "heading_deg": data.get("heading_deg"),
            })

        doc = {
            "map_id": map_id,
            "name": name,
            "node_count": topomap.graph.number_of_nodes(),
            "total_steps": total_steps,
            "total_distance_m": round(total_distance, 2),
            "created_at": datetime.utcnow().isoformat(),
            "goal": meta.get("goal", ""),
            "nodes": nodes,
            "edges": edges,
        }

        path = self._dir / f"{map_id}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return map_id

    # ── List ──

    def list_maps(self) -> List[Dict]:
        results = []
        for f in sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
                results.append({
                    "id": doc["map_id"],
                    "name": doc["name"],
                    "node_count": doc.get("node_count", 0),
                    "total_steps": doc.get("total_steps", 0),
                    "total_distance_m": doc.get("total_distance_m", 0.0),
                    "created_at": doc.get("created_at", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    # ── Load ──

    def load(self, map_id: str) -> Tuple[Optional[TopoMap], Optional[Dict]]:
        path = self._dir / f"{map_id}.json"
        if not path.exists():
            return None, None

        doc = json.loads(path.read_text(encoding="utf-8"))

        topo = TopoMap()
        id_map: Dict[int, int] = {}
        for node in doc.get("nodes", []):
            old_id = node["id"]
            nid = topo.add_node(
                photo_path="",
                detected=node.get("detected", []),
                summary=node.get("summary", ""),
                ocr_texts=node.get("ocr_texts"),
            )
            if node.get("pdr_x") is not None:
                topo.graph.nodes[nid]["pdr_x"] = node["pdr_x"]
                topo.graph.nodes[nid]["pdr_y"] = node["pdr_y"]
            id_map[old_id] = nid

        for edge in doc.get("edges", []):
            from_id = id_map.get(edge["from"])
            to_id = id_map.get(edge["to"])
            if from_id is not None and to_id is not None:
                topo.add_edge(from_id, to_id, action=edge.get("action", ""))
                edata = topo.graph.edges[from_id, to_id]
                if edge.get("steps") is not None:
                    edata["steps"] = edge["steps"]
                if edge.get("distance_m") is not None:
                    edata["distance_m"] = edge["distance_m"]
                if edge.get("heading_deg") is not None:
                    edata["heading_deg"] = edge["heading_deg"]

        meta = {
            "name": doc.get("name", ""),
            "goal": doc.get("goal", ""),
            "total_steps": doc.get("total_steps", 0),
            "total_distance_m": doc.get("total_distance_m", 0.0),
            "created_at": doc.get("created_at", ""),
        }
        return topo, meta

    # ── Delete ──

    def delete(self, map_id: str) -> bool:
        path = self._dir / f"{map_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
