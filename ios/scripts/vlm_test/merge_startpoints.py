"""Merge WP28 and WP34 into WP25 (all are PDR start points at same physical location)."""
import base64, os, requests, json

_host = os.environ.get("NEO4J_HOST", "")
_neo4j_user = _host.split('.')[0] if _host else ""
_neo4j_pass = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_ENDPOINT = f"https://{_host}/db/{_neo4j_user}/query/v2"
NEO4J_AUTH = "Basic " + base64.b64encode(f"{_neo4j_user}:{_neo4j_pass}".encode()).decode()
PLACE = os.environ.get("PLACE", "A7家樂福（9/16)")

def neo4j_query(cypher, params=None):
    resp = requests.post(NEO4J_ENDPOINT, json={"statement": cypher, "parameters": params or {}},
                         headers={"Authorization": NEO4J_AUTH, "Content-Type": "application/json",
                                  "Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", {})
    fields = data.get("fields", [])
    return [dict(zip(fields, row)) for row in data.get("values", [])]

PRIMARY = 25
ABSORBED = [28, 34]

# Step 1: Move DirPhotos from absorbed WPs to primary
for absorbed_nid in ABSORBED:
    photos = neo4j_query("""
        MATCH (w:Waypoint {place: $place, nid: $nid})-[hp:HAS_PHOTO]->(p:DirPhoto)
        RETURN hp.slot AS slot, elementId(p) AS pid
    """, {"place": PLACE, "nid": absorbed_nid})
    print(f"WP{absorbed_nid} has {len(photos)} DirPhotos")

    suffix = "_set02" if absorbed_nid == 28 else "_set03"
    for photo in photos:
        new_slot = photo["slot"] + suffix
        neo4j_query("""
            MATCH (w:Waypoint {place: $place, nid: $primary_nid})
            MATCH (p:DirPhoto) WHERE elementId(p) = $pid
            CREATE (w)-[:HAS_PHOTO {slot: $new_slot}]->(p)
        """, {"place": PLACE, "primary_nid": PRIMARY, "pid": photo["pid"], "new_slot": new_slot})
        print(f"  Moved {photo['slot']} -> {new_slot}")

    neo4j_query("""
        MATCH (w:Waypoint {place: $place, nid: $nid})-[hp:HAS_PHOTO]->()
        DELETE hp
    """, {"place": PLACE, "nid": absorbed_nid})
    print(f"  Deleted old HAS_PHOTO rels from WP{absorbed_nid}")

# Step 2: Move WALKWAY edges
for absorbed_nid in ABSORBED:
    out_edges = neo4j_query("""
        MATCH (w:Waypoint {place: $place, nid: $nid})-[r:WALKWAY]->(other:Waypoint)
        WHERE other.nid <> $primary_nid
        RETURN other.nid AS other_nid, r.distance AS dist
    """, {"place": PLACE, "nid": absorbed_nid, "primary_nid": PRIMARY})

    for e in out_edges:
        existing = neo4j_query("""
            MATCH (a:Waypoint {place: $place, nid: $pnid})-[r:WALKWAY]->(b:Waypoint {nid: $onid, place: $place})
            RETURN count(r) AS cnt
        """, {"place": PLACE, "pnid": PRIMARY, "onid": e["other_nid"]})
        if existing[0]["cnt"] == 0:
            neo4j_query("""
                MATCH (a:Waypoint {place: $place, nid: $pnid})
                MATCH (b:Waypoint {place: $place, nid: $onid})
                CREATE (a)-[:WALKWAY {distance: $dist}]->(b)
            """, {"place": PLACE, "pnid": PRIMARY, "onid": e["other_nid"], "dist": e["dist"] or 0})
            print(f"  Added WALKWAY WP{PRIMARY}->WP{e['other_nid']}")

    neo4j_query("""
        MATCH (w:Waypoint {place: $place, nid: $nid})-[r:WALKWAY]-()
        DELETE r
    """, {"place": PLACE, "nid": absorbed_nid})
    print(f"  Deleted WALKWAYs from WP{absorbed_nid}")

# Step 3: Delete absorbed waypoints
for absorbed_nid in ABSORBED:
    neo4j_query("""
        MATCH (w:Waypoint {place: $place, nid: $nid})
        DELETE w
    """, {"place": PLACE, "nid": absorbed_nid})
    print(f"Deleted WP{absorbed_nid}")

# Step 4: Update WP25 coordinates
neighbors = neo4j_query("""
    MATCH (w:Waypoint {place: $place, nid: 25})-[:WALKWAY]-(n:Waypoint)
    RETURN n.nid AS nid, n.pdr_x AS x, n.pdr_y AS y
""", {"place": PLACE})
print(f"\nWP25 neighbors: {[(n['nid'], n['x'], n['y']) for n in neighbors]}")

if neighbors:
    avg_x = sum(n["x"] for n in neighbors) / len(neighbors)
    avg_y = sum(n["y"] for n in neighbors) / len(neighbors)
    neo4j_query("""
        MATCH (w:Waypoint {place: $place, nid: 25})
        SET w.pdr_x = $x, w.pdr_y = $y
    """, {"place": PLACE, "x": round(avg_x, 1), "y": round(avg_y, 1)})
    print(f"Updated WP25 coords to ({avg_x:.1f}, {avg_y:.1f})")

# Verify
final = neo4j_query("""
    MATCH (w:Waypoint {place: $place})
    RETURN count(w) AS total_wps
""", {"place": PLACE})
print(f"\nFinal: {final[0]['total_wps']} waypoints")

wp25_photos = neo4j_query("""
    MATCH (w:Waypoint {place: $place, nid: 25})-[hp:HAS_PHOTO]->(p:DirPhoto)
    RETURN hp.slot AS slot ORDER BY hp.slot
""", {"place": PLACE})
print(f"WP25 now has {len(wp25_photos)} DirPhotos: {[p['slot'] for p in wp25_photos]}")
