"""The photo re-ranker must compare the user's photo against EVERY directional
reference photo of each candidate (the user may face any way), and the
candidate set must include the nodes around the previous fix (the user cannot
have walked far since the last photo)."""
import json
import logging
from unittest.mock import patch, MagicMock

from PIL import Image

from server.neo4j_client import RefMap, RefPhotoNode, RefObject, DirectionalRef
from server import visual_localization as vl


def _obj(label):
    return RefObject(label=label, label_norm=label, score=0.9, ocr_text="", role="landmark", grid_cell="c")


def _jpg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path, "JPEG")


def _node_with_files(nid, root, slots=("front", "right", "back", "left"), neighbours=()):
    hdg = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}
    dps = []
    for sl in slots:
        rel = f"set01/{nid}_{sl}.jpg"
        _jpg(root / "01" / f"{nid}_{sl}.jpg")
        dps.append(DirectionalRef(slot=sl, heading_deg=hdg[sl], photo_file=rel, objects=[_obj("shelf")]))
    return RefPhotoNode(nid=nid, photo_file=f"set01/{nid}_front.jpg", pdr_x=0, pdr_y=0, heading_deg=0.0,
                        session="s", total_steps=0, total_distance_m=0.0, objects=[_obj("shelf")],
                        neighbor_nids=list(neighbours), directional_photos=dps)


def _post_returning(text):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    resp.raise_for_status = lambda: None
    return resp


def test_rerank_sends_every_directional_photo_of_each_candidate(tmp_path):
    m = RefMap(place="t")
    m.photos[1] = _node_with_files(1, tmp_path)                      # 4 views
    m.photos[2] = _node_with_files(2, tmp_path, slots=("front", "back"))  # 2 views
    _jpg(tmp_path / "q.jpg")
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return _post_returning('{"ranking": [2, 1], "reason": "r"}')

    with patch.object(vl.requests, "post", side_effect=fake_post):
        out = vl.vlm_rerank(str(tmp_path / "q.jpg"), [(1, 0.5, "a"), (2, 0.4, "b")], m,
                            ref_photo_root=str(tmp_path), api_key="k", api_base_url="http://x",
                            api_model="m")

    content = calls[0]["messages"][0]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 1 + 4 + 2                    # query + all views of both nodes
    labels = [c["text"] for c in content if c["type"] == "text"]
    # every view is labelled with its node number AND direction, numbering is per node
    assert any("候選 1" in t and "WP1" in t and "left" in t for t in labels)
    assert any("候選 2" in t and "WP2" in t and "back" in t for t in labels)
    assert not any("候選 3" in t for t in labels)
    # ranking indices are per NODE, so [2, 1] means node 2 first
    assert [nid for nid, _, _ in out] == [2, 1]


def test_rerank_candidates_add_neighbourhood_of_last_fix():
    """Top-5 from the word matcher ∪ nodes within 2 hops of the previous fix,
    top-5 first, deduped, capped."""
    m = RefMap(place="t")
    for nid in range(1, 12):
        m.photos[nid] = RefPhotoNode(nid=nid, photo_file="", pdr_x=0, pdr_y=0, heading_deg=0, session="s",
                                     total_steps=0, total_distance_m=0)
    m.photos[10].neighbor_nids = [11]        # 10 → 11
    m.photos[9].neighbor_nids = [10]         # 9 → 10 (reverse edge must count too)
    m.photos[11].neighbor_nids = [7]         # 2 hops from 10
    top = [(1, 0.5, "a"), (2, 0.4, "b"), (3, 0.3, "c"), (4, 0.2, "d"), (5, 0.1, "e")]

    merged = vl.rerank_candidates(top, m, hint_nid=10)

    nids = [n for n, _, _ in merged]
    # Word candidates lead, but slots are RESERVED for the neighbourhood: where the
    # user just stood beats the 5th-best word guess.
    assert nids[0] == 1
    assert 10 in nids and 11 in nids and 9 in nids          # the fix itself + both edge directions
    assert len(nids) <= vl.RERANK_MAX_NODES
    assert len(set(nids)) == len(nids)
    assert vl.rerank_candidates(top, m, hint_nid=None) == top
    assert vl.rerank_candidates([(10, 0.5, "x")], m, hint_nid=10)[0] == (10, 0.5, "x")   # no duplicate


def test_word_candidates_fill_the_slots_no_neighbour_needs():
    """With a hint that has no neighbours, the reserved slots go back to the word
    matcher rather than being wasted."""
    m = RefMap(place="t")
    for nid in range(1, 12):
        m.photos[nid] = RefPhotoNode(nid=nid, photo_file="", pdr_x=0, pdr_y=0, heading_deg=0, session="s",
                                     total_steps=0, total_distance_m=0)
    top = [(i, 1.0 / i, "w") for i in range(1, 9)]

    nids = [n for n, _, _ in vl.rerank_candidates(top, m, hint_nid=10)]

    assert len(nids) == vl.RERANK_MAX_NODES
    assert nids[0] == 1 and 10 in nids                      # hint kept, rest are word candidates


def test_neighbourhood_prefers_closer_hops_when_capped():
    m = RefMap(place="t")
    for nid in range(1, 30):
        m.photos[nid] = RefPhotoNode(nid=nid, photo_file="", pdr_x=0, pdr_y=0, heading_deg=0, session="s",
                                     total_steps=0, total_distance_m=0)
    m.photos[20].neighbor_nids = [21, 22]               # 1 hop
    m.photos[21].neighbor_nids = [23, 24, 25, 26]       # 2 hops
    top = [(1, 0.5, "a"), (2, 0.4, "b"), (3, 0.3, "c"), (4, 0.2, "d"), (5, 0.1, "e")]

    nids = [n for n, _, _ in vl.rerank_candidates(top, m, hint_nid=20)]

    assert len(nids) == vl.RERANK_MAX_NODES
    assert 20 in nids and 21 in nids and 22 in nids     # the fix and its 1-hop ring survive the cap
    assert 23 not in nids and 26 not in nids            # 2-hop ring yields to the word candidates


def test_server_hands_reranker_the_neighbourhood_of_last_fix(caplog):
    """upload_photo's early localization must augment the word-matcher's
    candidates with the previous fix's neighbourhood before re-ranking."""
    from server.session import Session
    from server.visual_localization import LocalizationResult
    from server import server as srv

    m = RefMap(place="t")
    for nid in (1, 2, 3, 40, 41):
        m.photos[nid] = RefPhotoNode(nid=nid, photo_file="", pdr_x=0, pdr_y=0, heading_deg=0, session="s",
                                     total_steps=0, total_distance_m=0)
    m.photos[40].neighbor_nids = [41]

    class _Neo:
        def load_reference_map(self, place): return m

    s = Session(id="t", goal="milk", goal_objects=["milk"], place="t")
    s.last_corrected_nid = 40                     # previous photo was fixed at WP40
    grid = LocalizationResult(matched_nid=1, confidence=0.6, method="grid", reasoning="",
                              ref_node=m.photos[1], top_candidates=[(1, 0.3, ""), (2, 0.2, ""), (3, 0.1, "")])
    seen = {}
    caplog.set_level(logging.INFO, logger="server.server")

    def fake_rerank(query_image_path, candidates, ref_map, **kw):
        seen["cands"] = [n for n, _, _ in candidates]
        return candidates

    with patch("server.server.get_neo4j", return_value=_Neo()), \
         patch("server.server._localize_photo", return_value=grid), \
         patch("server.server._vlm_rerank", side_effect=fake_rerank), \
         patch("server.config.RERANK_ENABLED", True), \
         patch("server.config.REF_PHOTO_ROOT", "/refs"):
        srv._run_early_localization(s, "sid", ["shelf"], [], image_path="p.jpg")

    assert seen["cands"][:3] == [1, 2, 3]
    assert 40 in seen["cands"] and 41 in seen["cands"]
    assert any("⏱ rerank:" in r.message for r in caplog.records)   # duration is logged


def test_rerank_payload_is_small_enough_for_a_phone_hotspot(tmp_path):
    """Field runs tether the Mac to the phone's hotspot, so the re-rank upload
    competes with the phone's own photo upload. At 8 nodes × 4 views × 512 px the
    request was ~1.9 MB and the proxy connection timed out (session a8a95d4d).
    Keep reference views small and the node count modest."""
    from server import visual_localization as vl

    assert vl.RERANK_MAX_NODES <= 6
    assert vl.REF_PHOTO_MAX_PX <= 384

    m = RefMap(place="t")
    for nid in (1, 2):
        m.photos[nid] = _node_with_files(nid, tmp_path)
    _jpg(tmp_path / "q.jpg")
    sizes = []

    def fake_post(url, json=None, headers=None, timeout=None):
        body = str(json)
        sizes.append(len(body))
        return _post_returning('{"ranking": [1, 2], "reason": "r"}')

    with patch.object(vl.requests, "post", side_effect=fake_post):
        vl.vlm_rerank(str(tmp_path / "q.jpg"), [(1, 0.5, "a"), (2, 0.4, "b")], m,
                      ref_photo_root=str(tmp_path), api_key="k", api_base_url="http://x",
                      api_model="m")

    # 8 tiny 8×8 JPEGs — the point is that reference views go through the shrinking
    # encoder, so the per-view budget stays where the constant says.
    assert sizes and sizes[0] > 0
