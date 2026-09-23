"""Neo4j Aura client — pure HTTP API, zero extra dependencies.

Uses the Neo4j HTTP Cypher Transaction API to query the cloud topological map.
No `neo4j` pip package needed — just the `requests` library already in the project.

API docs: https://neo4j.com/docs/http-api/current/

Supports two schemas (auto-detected per place):

Schema V2 (new — 4 directional photos per waypoint):
  (:Waypoint {place, nid, pdr_x, pdr_y, session, timestamp})
  (:Waypoint)-[:HAS_PHOTO {slot}]->(:DirPhoto {heading_deg, photo_file})
  (:DirPhoto)-[:DETECTED]->(:Object {label, label_norm, score, ocr_text, role, grid_cell})
  (:Waypoint)-[:WALKWAY {direction, distance_m, steps}]->(:Waypoint)

Schema V1 (legacy — 1 photo per node):
  (:TopoNode:Photo {place, nid, photo_file, sensor_data_pdr_x, ...})
  (:TopoNode:Photo)-[:CONTAINS]->(:Object {label, ...})
  (:TopoNode:Photo)-[:WALKWAY]->(:TopoNode:Photo)
"""
from __future__ import annotations

import base64
import logging
from collections import defaultdict
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
class DirectionalRef:
    """One of the 4 directional reference photos at a node (front/right/back/left)."""
    slot: str             # "front" | "right" | "back" | "left"
    heading_deg: float    # absolute heading of this directional photo
    photo_file: str
    objects: List[RefObject] = field(default_factory=list)


@dataclass
class RefPhotoNode:
    """A reference waypoint from the pre-built topological map."""
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
    directional_photos: List[DirectionalRef] = field(default_factory=list)


@dataclass
class RefMap:
    """The full reference topological map for one place."""
    place: str
    schema_version: int = 1
    photos: Dict[int, RefPhotoNode] = field(default_factory=dict)
    walkway_edges: List[dict] = field(default_factory=list)


# ── HTTP-based Neo4j client ───────────────────────────────────────────────

def _bolt_to_query_url(bolt_uri: str, database: str = "neo4j") -> str:
    """Convert a bolt URI to the Neo4j Query API v2 endpoint."""
    uri = bolt_uri.strip()
    if uri.startswith(("neo4j+s://", "bolt+s://", "neo4j+ssc://")):
        host = uri.split("://", 1)[1].rstrip("/")
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
        """Execute a Cypher statement via the Query API v2 and return rows as dicts."""
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

        errors = payload.get("errors", [])
        if errors:
            msg = "; ".join(e.get("message", str(e)) for e in errors)
            raise RuntimeError(f"Neo4j query error: {msg}")

        data = payload.get("data", {})
        fields = data.get("fields", [])
        values = data.get("values", [])
        if not fields:
            return []

        rows: List[Dict[str, Any]] = []
        for row_vals in values:
            rows.append(dict(zip(fields, row_vals)))
        return rows

    # ── Schema detection ─────────────────────────────────────────────────

    def _detect_schema(self, place: str) -> int:
        """Detect which schema a place uses. Returns 2 for new, 1 for legacy."""
        rows = self._query(
            "MATCH (w:Waypoint {place: $place}) RETURN count(w) AS cnt",
            {"place": place},
        )
        if rows and rows[0]["cnt"] > 0:
            return 2
        return 1

    # ── Load reference map ────────────────────────────────────────────────

    def load_reference_map(self, place: Optional[str] = None) -> RefMap:
        """Load the full reference topological map for the given place.

        Auto-detects schema version and loads accordingly.
        Results are cached in memory after the first load.
        """
        place = place or NEO4J_PLACE
        if self._ref_map_cache and self._ref_map_cache.place == place:
            return self._ref_map_cache

        if not self._connected:
            log.warning("Neo4j not connected — returning empty reference map")
            return RefMap(place=place)

        version = self._detect_schema(place)
        log.info("Detected schema V%d for '%s'", version, place)

        if version == 2:
            ref = self._load_v2(place)
        else:
            ref = self._load_v1(place)

        self._ref_map_cache = ref
        return ref

    # ── Schema V2: Waypoint → DirPhoto → Object ─────────────────────────

    def _load_v2(self, place: str) -> RefMap:
        ref = RefMap(place=place, schema_version=2)

        # 1) Waypoints
        rows = self._query("""
            MATCH (w:Waypoint {place: $place})
            RETURN w.nid AS nid,
                   w.pdr_x AS pdr_x,
                   w.pdr_y AS pdr_y,
                   w.heading_deg AS heading,
                   w.session AS session,
                   w.total_steps AS steps,
                   w.total_distance_m AS dist
            ORDER BY w.nid
        """, {"place": place})

        for r in rows:
            nid = r["nid"]
            ref.photos[nid] = RefPhotoNode(
                nid=nid,
                photo_file="",
                pdr_x=r["pdr_x"] or 0.0,
                pdr_y=r["pdr_y"] or 0.0,
                heading_deg=r["heading"] or 0.0,
                session=r["session"] or "",
                total_steps=r["steps"] or 0,
                total_distance_m=r["dist"] or 0.0,
            )

        log.info("V2: loaded %d waypoints for '%s'", len(ref.photos), place)

        # 2) Directional photos + their objects
        rows = self._query("""
            MATCH (w:Waypoint {place: $place})-[hp:HAS_PHOTO]->(p:DirPhoto)
            OPTIONAL MATCH (p)-[:DETECTED]->(obj:Object)
            RETURN w.nid AS nid,
                   hp.slot AS slot,
                   p.heading_deg AS heading,
                   p.photo_file AS photo_file,
                   obj.label AS label,
                   obj.label_norm AS label_norm,
                   obj.score AS score,
                   obj.ocr_text AS ocr_text,
                   obj.role AS role,
                   obj.grid_cell AS grid_cell
        """, {"place": place})

        # Group by (nid, slot) to build DirectionalRef with its objects
        dir_data: Dict[tuple, dict] = {}
        for r in rows:
            key = (r["nid"], r["slot"])
            if key not in dir_data:
                dir_data[key] = {
                    "slot": r["slot"],
                    "heading": r["heading"] or 0.0,
                    "photo_file": r["photo_file"] or "",
                    "objects": [],
                }
            if r["label"]:
                dir_data[key]["objects"].append(RefObject(
                    label=r["label"] or "",
                    label_norm=r["label_norm"] or "",
                    score=r["score"] or 0.0,
                    ocr_text=r["ocr_text"] or "",
                    role=r["role"] or "",
                    grid_cell=r["grid_cell"] or "",
                ))

        obj_count = 0
        for (nid, slot), d in dir_data.items():
            if nid not in ref.photos:
                continue
            node = ref.photos[nid]
            dr = DirectionalRef(
                slot=d["slot"],
                heading_deg=d["heading"],
                photo_file=d["photo_file"],
                objects=d["objects"],
            )
            node.directional_photos.append(dr)
            # Aggregate all directional objects into node.objects for localization
            node.objects.extend(d["objects"])
            obj_count += len(d["objects"])
            # Use front photo as the node's primary photo
            if d["slot"] == "front" and not node.photo_file:
                node.photo_file = d["photo_file"]

        # If no front photo was set, use any available photo
        for node in ref.photos.values():
            if not node.photo_file and node.directional_photos:
                node.photo_file = node.directional_photos[0].photo_file

        log.info("V2: loaded %d directional photos, %d objects",
                 len(dir_data), obj_count)

        # 3) Walkway edges
        rows = self._query("""
            MATCH (a:Waypoint {place: $place})-[r:WALKWAY]->(b:Waypoint)
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

        log.info("V2: loaded %d walkway edges", len(ref.walkway_edges))
        return ref

    # ── Schema V1 (legacy): TopoNode:Photo → Object ─────────────────────

    def _load_v1(self, place: str) -> RefMap:
        ref = RefMap(place=place, schema_version=1)

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

        log.info("V1: loaded %d photo nodes for '%s'", len(ref.photos), place)

        # 2) Objects per photo node
        rows = self._query("""
            MATCH (p:TopoNode {place: $place, ntype: 'photo'})-[:CONTAINS]->(obj:Object)
            RETURN p.nid AS nid,
                   obj.label AS label,
                   obj.label_norm AS label_norm,
                   obj.score AS score,
                   obj.ocr_text AS ocr_text,
                   obj.role AS role,
                   obj.grid_cell AS grid_cell
        """, {"place": place})

        obj_count = 0
        for r in rows:
            nid = r["nid"]
            if nid in ref.photos:
                ref.photos[nid].objects.append(RefObject(
                    label=r["label"] or "",
                    label_norm=r["label_norm"] or "",
                    score=r["score"] or 0.0,
                    ocr_text=r["ocr_text"] or "",
                    role=r["role"] or "",
                    grid_cell=r["grid_cell"] or "",
                ))
                obj_count += 1

        log.info("V1: loaded %d object detections", obj_count)

        # 3) Synthesize directional photos from single heading
        from server.heading import slot_headings

        pos_groups: Dict[tuple, List[int]] = defaultdict(list)
        for nid, node in ref.photos.items():
            key = (round(node.pdr_x, 1), round(node.pdr_y, 1))
            pos_groups[key].append(nid)

        for pos, nids in pos_groups.items():
            if len(nids) == 1:
                node = ref.photos[nids[0]]
                slots = slot_headings(node.heading_deg)
                for slot, hdeg in slots.items():
                    node.directional_photos.append(DirectionalRef(
                        slot=slot.value,
                        heading_deg=hdeg,
                        photo_file=node.photo_file,
                        objects=list(node.objects),
                    ))
            else:
                primary_nid = nids[0]
                primary_node = ref.photos[primary_nid]
                slots = slot_headings(primary_node.heading_deg)

                for slot, target_hdeg in slots.items():
                    best_nid = primary_nid
                    best_diff = 999.0
                    for nid in nids:
                        diff = abs(((ref.photos[nid].heading_deg - target_hdeg + 180) % 360) - 180)
                        if diff < best_diff:
                            best_diff = diff
                            best_nid = nid
                    matched_node = ref.photos[best_nid]
                    primary_node.directional_photos.append(DirectionalRef(
                        slot=slot.value,
                        heading_deg=matched_node.heading_deg,
                        photo_file=matched_node.photo_file,
                        objects=list(matched_node.objects),
                    ))

                for nid in nids[1:]:
                    ref.photos[nid].directional_photos = list(
                        primary_node.directional_photos
                    )

        dir_count = sum(len(n.directional_photos) for n in ref.photos.values())
        log.info("V1: synthesized %d directional slots across %d nodes",
                 dir_count, len(ref.photos))

        # 4) Walkway edges
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

        log.info("V1: loaded %d walkway edges", len(ref.walkway_edges))
        return ref

    # ── Place listing (works with both schemas) ──────────────────────────

    def list_places(self) -> List[Dict[str, Any]]:
        """Return all distinct places in the topological map."""
        if not self._connected:
            return []
        # V2 places
        v2 = self._query("""
            MATCH (w:Waypoint)
            RETURN w.place AS name, count(w) AS photo_count
        """)
        # V1 places
        v1 = self._query("""
            MATCH (t:TopoNode {ntype: 'photo'})
            RETURN t.place AS name, count(t) AS photo_count
        """)
        seen = set()
        result = []
        for r in v2 + v1:
            name = r.get("name")
            if name and name not in seen:
                seen.add(name)
                result.append({"name": name, "photo_count": r["photo_count"]})
        result.sort(key=lambda x: x["name"])
        return result

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
        ref = self.load_reference_map(place)
        # Use the right node label depending on schema
        label = "Waypoint" if ref.schema_version == 2 else "TopoNode"
        rows = self._query(f"""
            MATCH (a:{label} {{place: $place, nid: $from_nid}}),
                  (b:{label} {{place: $place, nid: $to_nid}}),
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
