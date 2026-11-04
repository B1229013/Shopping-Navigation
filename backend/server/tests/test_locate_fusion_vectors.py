"""locate_fusion: reference DINOv2 vectors come from the sidecar and/or DirPhoto.dino in Neo4j."""
import numpy as np

from server import locate_fusion as lf
from server.neo4j_client import DirectionalRef, RefMap, RefPhotoNode, _link_neighbors


def _ref_map(dinos):
    ref = RefMap(place="店", schema_version=2)
    for nid, (photo_file, dino) in enumerate(dinos, start=1):
        node = RefPhotoNode(nid=nid, photo_file=photo_file, pdr_x=0.0, pdr_y=0.0, heading_deg=0.0,
                            session="", total_steps=0, total_distance_m=0.0)
        node.directional_photos.append(DirectionalRef(slot="front", heading_deg=0.0, photo_file=photo_file,
                                                      dino=dino))
        ref.photos[nid] = node
    return ref


def setup_function():
    lf._combined_cache.update(key=None, embs=None)


def test_sidecar_only_is_returned_as_is():
    sidecar = {"a.jpg": np.ones(3, dtype=np.float32)}
    ref = _ref_map([("set01/a.jpg", None)])
    assert lf._reference_embeddings(ref, sidecar) is sidecar


def test_neo4j_vectors_without_sidecar():
    ref = _ref_map([("set01/a.jpg", [1.0, 0.0, 0.0]), ("set01/b.jpg", [0.0, 1.0, 0.0])])
    embs = lf._reference_embeddings(ref, None)
    assert set(embs) == {"set01/a.jpg", "set01/b.jpg"}
    assert embs["set01/b.jpg"].dtype == np.float32
    assert lf._lookup_emb(embs, "set01/a.jpg") is not None


def test_neo4j_vectors_win_and_result_is_cached():
    sidecar = {"a.jpg": np.zeros(3, dtype=np.float32), "z.jpg": np.ones(3, dtype=np.float32)}
    ref = _ref_map([("set01/a.jpg", [1.0, 0.0, 0.0])])
    embs = lf._reference_embeddings(ref, sidecar)
    assert set(embs) == {"a.jpg", "z.jpg", "set01/a.jpg"}
    # _lookup_emb tries the full photo_file first → the Neo4j vector
    assert lf._lookup_emb(embs, "set01/a.jpg").tolist() == [1.0, 0.0, 0.0]
    assert lf._reference_embeddings(ref, sidecar) is embs          # same object → index cache stays valid
    assert "a.jpg" in sidecar and len(sidecar) == 2                 # sidecar dict itself untouched


# ── WALKWAY neighbours: walkable both ways, whichever way the edge is stored ──

def test_link_neighbors_both_ways_once():
    ref = _ref_map([("a.jpg", None), ("b.jpg", None), ("c.jpg", None)])
    _link_neighbors(ref, 1, 2)          # stored 1→2
    _link_neighbors(ref, 2, 1)          # and also 2→1 (both directions in the database)
    _link_neighbors(ref, 3, 2)          # stored 3→2 only
    _link_neighbors(ref, 2, 2)          # self-loop ignored
    _link_neighbors(ref, 1, 99)         # unknown node ignored
    assert ref.photos[1].neighbor_nids == [2]
    assert ref.photos[2].neighbor_nids == [1, 3]
    assert ref.photos[3].neighbor_nids == [2]
