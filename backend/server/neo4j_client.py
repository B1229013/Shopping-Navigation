"""Neo4j Aura client — pure HTTP API, zero extra dependencies.

Uses the Neo4j HTTP Cypher Transaction API to query the cloud topological map.
No `neo4j` pip package needed — just the `requests` library already in the project.

API docs: https://neo4j.com/docs/http-api/current/

Schema (as stored by the Android topomap builder):
  (:TopoNode:Photo {place, nid, photo_file, sensor_data_pdr_x, sensor_data_pdr_y,
                     sensor_data_heading_deg, sensor_data_session, timestamp, ...})
  (:TopoNode {ntype:"object", label, label_norm, score, ocr_text, role, grid_cell,
              photo_id, ...})
  (photo)-[:CONTAINS_*]->(object)
  (photo)-[:WALKWAY {direction, distance_m, steps}]->(photo)
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests as http

from server.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, NEO4J_PLACE

log = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class RefObject:
    """A detected object at a reference photo location."""
    label: str
    label_norm: str
    score: float
    ocr_text: str
    role: str
    grid_cell: str


@dataclass
class RefPhotoNode:
    """A reference photo location from the pre-built topological map."""
    nid: int
    photo_file: str
    pdr_x: float
    pdr_y: float
    heading_deg: float
    session: str
    total_steps: int
    total_distance_m: float
    objects: List[RefObject] = field(default_factory=list)
    neighbor_nids: List[int] = field(default_factory=list)


@dataclass
class RefMap:
    """The full reference topological map for one place."""
    place: str
    photos: Dict[int, RefPhotoNode] = field(default_factory=dict)
    walkway_edges: List[dict] = field(default_factory=list)


# ── HTTP-based Neo4j client ───────────────────────────────────────────────

def _bolt_to_query_url(bolt_uri: str, database: str = "neo4j") -> str:
    """Convert a bolt URI to the Neo4j Query API v2 endpoint.

    Aura Free blocks the old tx/commit endpoint (403) but exposes the
    newer Query API v2 on port 443:
      neo4j+s://xxx.databases.neo4j.io  →  https://xxx.databases.neo4j.io/db/<database>/query/v2
    """
    uri = bolt_uri.strip()
    if uri.startswith(("neo4j+s://", "bolt+s://", "neo4j+ssc://")):
        host = uri.split("://", 1)[1].rstrip("/")
        # Aura: standard HTTPS port 443 (no port suffix needed)
        if ":" in host:
            host = host.split(":")[0]
        return f"https://{host}/db/{database}/query/v2"
    elif uri.startswith(("neo4j://", "bolt://")):
        host = uri.split("://", 1)[1].rstrip("/")
        if ":" in host:
            host = host.split(":")[0]
        return f"http://{host}:7474/db/{database}/query/v2"
    elif uri.startswith("http"):
        base = uri.rstrip("/")
        if not base.endswith(f"/query/v2"):
            base += f"/db/{database}/query/v2"
        return base
    else:
        raise ValueError(f"Unrecognized Neo4j URI scheme: {uri}")


class Neo4jClient:
    """Queries Neo4j Aura via its HTTP Cypher Transaction API."""

    def __init__(self) -> None:
        self._endpoint: str = ""
        self._auth_header: str = ""
        self._connected: bool = False
        self._ref_map_cache: Optional[RefMap] = None

    def connect(self) -> bool:
        """Resolve the HTTP endpoint and verify connectivity."""
        if not NEO4J_URI or not NEO4J_PASSWORD:
            log.warning("Neo4j not configured (NEO4J_URI or NEO4J_PASSWORD empty)")
            return False
        try:
            self._endpoint = _bolt_to_query_url(NEO4J_URI, NEO4J_DATABASE)
            creds = base64.b64encode(
                f"{NEO4J_USER}:{NEO4J_PASSWORD}".encode()
            ).decode()
            self._auth_header = f"Basic {creds}"

            # Ping with a trivial query
            rows = self._query("RETURN 1 AS ok")
            if rows and rows[0].get("ok") == 1:
                self._connected = True
                log.info("Connected to Neo4j via Query API v2: %s", self._endpoint)
                return True
            else:
                log.error("Neo4j ping returned unexpected result: %s", rows)
                return False
        except Exception as e:
            log.error("Failed to connect to Neo4j: %s", e)
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._connected = False

    # ── Raw Cypher query over HTTP ────────────────────────────────────────

    def _query(
        self, cypher: str, parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher statement via the Query API v2 and return rows as dicts.

        POST /db/<database>/query/v2
        Request:  {"statement": "...", "parameters": {...}}
        Response: {"data": {"fields": ["col1","col2"], "values": [v1,v2,v3,v4,...]}}
        Values are flattened: every N consecutive items form one row (N = len(fields)).
        """
        body = {
            "statement": cypher,
            "parameters": parameters or {},
        }
        resp = http.post(
            self._endpoint,
            json=body,
            headers={
                "Authorization": self._auth_header,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        # Check for Cypher errors
        errors = payload.get("errors", [])
        if errors:
            msg = "; ".join(e.get("message", str(e)) for e in errors)
            raise RuntimeError(f"Neo4j query error: {msg}")

        # Parse Query API v2 response format
        # {"data": {"fields": ["col1","col2"], "values": [[v1,v2], [v3,v4], ...]}}
        data = payload.get("data", {})
        fields = data.get("fields", [])
        values = data.get("values", [])
        if not fields:
            return []

        rows: List[Dict[str, Any]] = []
        for row_vals in values:
            rows.append(dict(zip(fields, row_vals)))
        return rows

    # ── Load reference map ────────────────────────────────────────────────

    def load_reference_map(self, place: Optional[str] = None) -> RefMap:
        """Load the full reference topological map for the given place.

        Three Cypher queries: photo nodes → objects → walkway edges.
        Results are cached in memory after the first load.
        """
        place = place or NEO4J_PLACE
        if self._ref_map_cache and self._ref_map_cache.place == place:
            return self._ref_map_cache

        if not self._connected:
            log.warning("Neo4j not connected — returning empty reference map")
            return RefMap(place=place)

        ref = RefMap(place=place)

        # 1) Photo nodes
        rows = self._query("""
            MATCH (n:TopoNode {place: $place})
            WHERE n.ntype = 'photo'
            RETURN n.nid AS nid,
                   n.photo_file AS photo_file,
                   n.sensor_data_pdr_x AS pdr_x,
                   n.sensor_data_pdr_y AS pdr_y,
                   n.sensor_data_heading_deg AS heading,
                   n.sensor_data_session AS session,
                   n.sensor_data_total_steps AS steps,
                   n.sensor_data_total_distance_m AS dist
            ORDER BY n.nid
        """, {"place": place})

        for r in rows:
            nid = r["nid"]
            ref.photos[nid] = RefPhotoNode(
                nid=nid,
                photo_file=r["photo_file"] or "",
                pdr_x=r["pdr_x"] or 0.0,
                pdr_y=r["pdr_y"] or 0.0,
                heading_deg=r["heading"] or 0.0,
                session=r["session"] or "",
                total_steps=r["steps"] or 0,
                total_distance_m=r["dist"] or 0.0,
            )

        log.info("Loaded %d photo nodes for '%s'", len(ref.photos), place)

        # 2) Objects per photo node
        #    Objects live on separate :Photo nodes (not :TopoNode) linked via
        #    :CONTAINS → :Object.  Photo.node_id ≠ TopoNode.nid, but within
        #    each session the node counts match and both are ordered, so we
        #    build a node_id → nid mapping by session-internal position.

        # 2a) Build mapping: Photo.node_id → TopoNode.nid
        topo_by_session: Dict[str, List[int]] = {}
        for nid, node in sorted(ref.photos.items()):
            topo_by_session.setdefault(node.session, []).append(nid)

        photo_order_rows = self._query("""
            MATCH (p:Photo)-[:CONTAINS]->(o)
            WHERE NOT p:TopoNode AND p.place_name = $place
            WITH DISTINCT p
            RETURN p.node_id AS node_id, p.sensor_session AS session
            ORDER BY p.node_id
        """, {"place": place})

        photo_by_session: Dict[str, List[int]] = {}
        for r in photo_order_rows:
            sess = r["session"] or ""
            photo_by_session.setdefault(sess, []).append(r["node_id"])

        node_id_to_nid: Dict[int, int] = {}
        for sess, photo_ids in photo_by_session.items():
            topo_nids = topo_by_session.get(sess, [])
            for i, pid in enumerate(photo_ids):
                if i < len(topo_nids):
                    node_id_to_nid[pid] = topo_nids[i]

        # 2b) Load objects and map to TopoNode nids
        rows = self._query("""
            MATCH (p:Photo)-[:CONTAINS]->(obj:Object)
            WHERE NOT p:TopoNode AND p.place_name = $place
            RETURN p.node_id AS photo_node_id,
                   obj.label AS label,
                   obj.label_norm AS label_norm,
                   obj.score AS score,
                   obj.ocr_text AS ocr_text,
                   obj.role AS role,
                   obj.grid_cell AS grid_cell
        """, {"place": place})

        obj_count = 0
        for r in rows:
            nid = node_id_to_nid.get(r["photo_node_id"])
            if nid is not None and nid in ref.photos:
                ref.photos[nid].objects.append(RefObject(
                    label=r["label"] or "",
                    label_norm=r["label_norm"] or "",
                    score=r["score"] or 0.0,
                    ocr_text=r["ocr_text"] or "",
                    role=r["role"] or "",
                    grid_cell=r["grid_cell"] or "",
                ))
                obj_count += 1

        log.info("Loaded %d object detections across photo nodes", obj_count)

        # 3) Walkway edges (photo→photo)
        rows = self._query("""
            MATCH (a:TopoNode {place: $place, ntype: 'photo'})
                  -[r:WALKWAY]->(b:TopoNode {ntype: 'photo'})
            RETURN a.nid AS from_nid,
                   b.nid AS to_nid,
                   r.direction AS direction,
                   r.distance_m AS distance_m,
                   r.steps AS steps
        """, {"place": place})

        for r in rows:
            from_nid = r["from_nid"]
            to_nid = r["to_nid"]
            ref.walkway_edges.append({
                "from": from_nid,
                "to": to_nid,
                "direction": r["direction"] or "",
                "distance_m": r["distance_m"] or 0.0,
                "steps": r["steps"] or 0,
            })
            if from_nid in ref.photos:
                ref.photos[from_nid].neighbor_nids.append(to_nid)

        log.info("Loaded %d walkway edges", len(ref.walkway_edges))

        self._ref_map_cache = ref
        return ref

    def list_places(self) -> List[Dict[str, Any]]:
        """Return all distinct places in the topological map.

        Each dict: {"name": str, "photo_count": int}
        """
        if not self._connected:
            return []
        rows = self._query("""
            MATCH (t:TopoNode {ntype: 'photo'})
            RETURN t.place AS name, count(t) AS photo_count
            ORDER BY t.place
        """)
        return [{"name": r["name"], "photo_count": r["photo_count"]}
                for r in rows if r["name"]]

    def invalidate_cache(self) -> None:
        self._ref_map_cache = None

    # ── Query helpers ─────────────────────────────────────────────────────

    def get_photo_node(self, nid: int, place: Optional[str] = None) -> Optional[RefPhotoNode]:
        ref = self.load_reference_map(place)
        return ref.photos.get(nid)

    def get_neighbors(self, nid: int, place: Optional[str] = None) -> List[RefPhotoNode]:
        ref = self.load_reference_map(place)
        node = ref.photos.get(nid)
        if not node:
            return []
        return [ref.photos[n] for n in node.neighbor_nids if n in ref.photos]

    def get_shortest_path(self, from_nid: int, to_nid: int,
                          place: Optional[str] = None) -> Optional[List[dict]]:
        """Shortest walkway path between two photo nodes via Neo4j."""
        if not self._connected:
            return None
        place = place or NEO4J_PLACE
        rows = self._query("""
            MATCH (a:TopoNode {place: $place, nid: $from_nid}),
                  (b:TopoNode {place: $place, nid: $to_nid}),
                  path = shortestPath((a)-[:WALKWAY*]-(b))
            WITH nodes(path) AS ns, relationships(path) AS rs
            UNWIND range(0, size(rs)-1) AS i
            RETURN ns[i].nid AS from_nid,
                   ns[i+1].nid AS to_nid,
                   rs[i].direction AS direction,
                   rs[i].distance_m AS distance_m
        """, {"place": place, "from_nid": from_nid, "to_nid": to_nid})
        return rows if rows else None


# ── Module-level singleton ────────────────────────────────────────────────

_client: Optional[Neo4jClient] = None


def get_neo4j() -> Optional[Neo4jClient]:
    """Get or create the Neo4j client singleton. Returns None if unconfigured."""
    global _client
    if _client is not None:
        return _client if _client.is_connected else None
    _client = Neo4jClient()
    if _client.connect():
        return _client
    return None
