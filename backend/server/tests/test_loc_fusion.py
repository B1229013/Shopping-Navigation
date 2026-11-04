"""DINOv2 (phone) : sensor (PDR) : grid fusion for photo localization."""
import json
import math
from unittest.mock import patch

import pytest

from server.loc_fusion import FusionWeights, fuse_candidates, parse_dino_topk, parse_weights
from server.neo4j_client import RefMap, RefPhotoNode
from server import visual_localization as vl


def _node(nid, x, y):
    return RefPhotoNode(nid=nid, photo_file=f"{nid}.jpg", pdr_x=x, pdr_y=y, heading_deg=0.0,
                        session="s", total_steps=0, total_distance_m=0.0, objects=[])


def _positions():
    return {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (20.0, 0.0)}


# ── parsing ───────────────────────────────────────────────────────────────

def test_parse_dino_topk_reads_json_list():
    raw = json.dumps([{"nid": 2, "sim": 0.81}, {"nid": 1, "sim": 0.7}])
    assert parse_dino_topk(raw, None) == {2: 0.81, 1: 0.7}


def test_parse_dino_topk_falls_back_to_single_hint():
    assert parse_dino_topk(None, 5) == {5: 1.0}


def test_parse_dino_topk_ignores_garbage():
    assert parse_dino_topk("not json", None) == {}
    assert parse_dino_topk(json.dumps([{"nid": "x"}]), None) == {}


def test_parse_weights():
    assert parse_weights("5,1,1") == FusionWeights(dino=5.0, sensor=1.0, grid=1.0)
    with pytest.raises(ValueError):
        parse_weights("5,1")
    with pytest.raises(ValueError):
        parse_weights("-1,1,1")
    for bad in ("nan,1,1", "inf,1,1", "1,-inf,1", "0,0,0"):
        with pytest.raises(ValueError):
            parse_weights(bad)


# ── fusion ────────────────────────────────────────────────────────────────

W = FusionWeights(dino=5.0, sensor=1.0, grid=1.0)


def test_grid_only_is_unchanged():
    grid = [(1, 0.4, "grid a"), (2, 0.3, "grid b")]
    assert fuse_candidates(grid, {}, None, _positions(), W) == grid


def test_dino_outweighs_grid_at_5_1_1():
    grid = [(1, 0.5, "g"), (2, 0.3, "g")]
    dino = {2: 0.90, 1: 0.60}
    out = fuse_candidates(grid, dino, None, _positions(), W)
    assert out[0][0] == 2
    # node 2: (5*1.0 + 1*0.3) / 6 ; node 1: (5*0.0 + 1*0.5) / 6
    assert out[0][1] == pytest.approx(5.3 / 6)
    assert out[1][1] == pytest.approx(0.5 / 6)


def test_sensor_breaks_tie_between_equal_visual_scores():
    grid = [(1, 0.4, "g"), (3, 0.4, "g")]
    dino = {1: 0.8, 3: 0.8}
    out = fuse_candidates(grid, dino, (19.0, 0.0), _positions(), W)
    assert out[0][0] == 3


def test_sensor_score_is_exponential_decay():
    out = fuse_candidates([(2, 0.0, "g")], {}, (2.0, 0.0), _positions(),
                          FusionWeights(dino=0.0, sensor=1.0, grid=0.0))
    assert out[0][1] == pytest.approx(math.exp(-8.0 / 8.0))


def test_dino_only_node_becomes_candidate():
    out = fuse_candidates([(1, 0.5, "g")], {3: 0.9, 1: 0.5}, None, _positions(), W)
    assert out[0][0] == 3


def test_reason_lists_components():
    out = fuse_candidates([(1, 0.5, "grid stuff")], {1: 0.9, 2: 0.5}, (0.0, 0.0), _positions(), W)
    reason = out[0][2]
    assert "dino=" in reason and "sensor=" in reason and "grid=" in reason


def test_zero_weight_signal_is_ignored():
    grid = [(1, 0.5, "g"), (2, 0.3, "g")]
    out = fuse_candidates(grid, {2: 0.9, 1: 0.6}, None, _positions(),
                          FusionWeights(dino=0.0, sensor=1.0, grid=1.0))
    assert [c[0] for c in out] == [1, 2]


# ── localize() integration ────────────────────────────────────────────────

class _FakeNeo4j:
    def __init__(self, m): self._m = m
    def load_reference_map(self, place): return self._m


def _ref_map():
    m = RefMap(place="test")
    for nid, (x, y) in _positions().items():
        m.photos[nid] = _node(nid, x, y)
    return m


def test_localize_uses_dino_when_given():
    grid = [(1, 0.5, "g"), (2, 0.3, "g")]
    with patch.object(vl, "match_by_objects", return_value=grid):
        out = vl.localize(["shelf"], [], _FakeNeo4j(_ref_map()), place="test",
                          dino_sims={2: 0.9, 1: 0.6}, fusion_weights=W)
    assert out.matched_nid == 2
    assert "fused" in out.method


def test_localize_without_dino_keeps_grid_result():
    grid = [(1, 0.5, "g"), (2, 0.3, "g")]
    with patch.object(vl, "match_by_objects", return_value=grid):
        out = vl.localize(["shelf"], [], _FakeNeo4j(_ref_map()), place="test", fusion_weights=W)
    assert out.matched_nid == 1
    assert out.method == "object_set"


def test_localize_dino_rescues_empty_grid():
    with patch.object(vl, "match_by_objects", return_value=[]):
        out = vl.localize([], [], _FakeNeo4j(_ref_map()), place="test",
                          dino_sims={3: 0.9, 1: 0.5}, fusion_weights=W)
    assert out.matched_nid == 3


# ── /photo endpoint passes phone data through ────────────────────────────

def test_phone_inputs_from_form_fields():
    from server import server as srv
    payload = json.dumps([{"nid": 3, "sim": 0.88}])
    assert srv._phone_localization_inputs(payload, None, 1.5, 2.5, 12.0, 7) == {
        "dino_sims": {3: 0.88}, "sensor_pos": (1.5, 2.5), "dist_since_fix": 12.0, "fix_node": 7}
    # map position is only sent once the phone is calibrated; half a pair is ignored
    assert srv._phone_localization_inputs(None, None, 1.5, None, 12.0, 7) == {
        "dino_sims": {}, "sensor_pos": None, "dist_since_fix": None, "fix_node": None}


def test_run_early_localization_passes_phone_inputs_to_localize():
    from server import server as srv
    from server.session import Session
    s = Session(id="t", goal="milk", goal_objects=["milk"], place="test")
    result = vl.LocalizationResult(matched_nid=None, confidence=0.0, method="none", reasoning="")
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(_ref_map())), \
         patch.object(srv._config, "LOC_FUSION_V2", False), \
         patch("server.server._localize_photo", return_value=result) as loc:
        srv._run_early_localization(s, "sid", ["shelf"], [],
                                    dino_sims={3: 0.9}, sensor_pos=(1.0, 2.0), dist_since_fix=20.0)
    kw = loc.call_args.kwargs
    assert kw["dino_sims"] == {3: 0.9}
    assert kw["sensor_pos"] == (1.0, 2.0)
    w = kw["fusion_weights"]
    assert isinstance(w, FusionWeights)
    # adaptive: σ = 1 + 0.05·20 = 2 m → sensor weight 40/2 = 20
    assert w.sensor == pytest.approx(20.0)
    # a single DINOv2 entry has no runner-up → dino weight unscaled
    assert w.dino == 4.0


def test_v2_does_not_need_the_v1_weights():
    # v2 never reads LOC_FUSION_WEIGHTS, so a bad value there must not stop localization
    from server import server as srv
    from server.session import Session
    s = Session(id="t", goal="milk", goal_objects=["milk"], place="test")
    result = vl.LocalizationResult(matched_nid=None, confidence=0.0, method="none", reasoning="")
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(_ref_map())), \
         patch.object(srv._config, "LOC_FUSION_V2", True), \
         patch.object(srv._config, "LOC_FUSION_WEIGHTS", "not,a,number"), \
         patch("server.server._localize_photo", return_value=result) as loc:
        out = srv._run_early_localization(s, "sid", ["shelf"], [], dino_sims={3: 0.9})
    assert out is result and loc.call_args.kwargs["fusion_weights"] is None
    assert loc.call_args.kwargs["v2"] is True


def test_test_config_sets_fusion_weights():
    from fastapi.testclient import TestClient
    from server import config
    from server.server import app
    client = TestClient(app)
    with patch.object(config, "LOC_FUSION_WEIGHTS", "5,1,1"):
        r = client.post("/test/config", params={"fusion_weights": "3,1,2"})
        assert r.status_code == 200
        assert r.json()["current"]["fusion_weights"] == "3,1,2"
        assert client.post("/test/config", params={"fusion_weights": "3,1"}).status_code == 400


# ── adaptive weights / position estimate ─────────────────────────────────

from server.loc_fusion import adaptive_weights, estimate_position, sensor_sigma


def test_sensor_sigma_grows_with_distance():
    assert sensor_sigma(None, 1.0, 0.05) is None
    assert sensor_sigma(0, 1.0, 0.05) == 1.0
    assert sensor_sigma(40, 1.0, 0.05) == pytest.approx(3.0)


def test_adaptive_weights():
    base = FusionWeights(dino=4, sensor=1, grid=1)
    assert adaptive_weights(base, None, 10, c=40, cap=40, n_full=0) == FusionWeights(4, 1, 1)
    assert adaptive_weights(base, 0.5, 10, c=40, cap=40, n_full=0).sensor == 40       # capped
    assert adaptive_weights(base, 8.0, 10, c=40, cap=40, n_full=0).sensor == 5
    assert adaptive_weights(base, 2.0, 5, c=40, cap=40, n_full=10).grid == pytest.approx(0.5)
    assert adaptive_weights(base, 2.0, 5, c=40, cap=40, n_full=0, gap=0.01, gap_full=0.05).dino == pytest.approx(0.8)
    assert adaptive_weights(base, 2.0, 5, c=40, cap=40, n_full=0, gap=0.2, gap_full=0.05).dino == 4


def test_dino_gap():
    from server.loc_fusion import dino_gap
    assert dino_gap({1: 0.9, 2: 0.85, 3: 0.5}) == pytest.approx(0.05)
    assert dino_gap({1: 1.0}) is None


def test_estimate_position_between_tied_nodes():
    pos = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (50.0, 0.0), 4: (90.0, 0.0)}
    # node 1 and 2 tied far above 3 and 4 → halfway between 1 and 2
    x, y = estimate_position([(1, 0.9, ""), (2, 0.9, ""), (3, 0.1, ""), (4, 0.1, "")], pos)
    assert x == pytest.approx(5.0) and y == 0.0
    # clear winner → on the winner
    assert estimate_position([(3, 0.9, ""), (1, 0.1, ""), (2, 0.1, ""), (4, 0.1, "")], pos) == (50.0, 0.0)


def test_localize_returns_estimated_position():
    grid = [(1, 0.5, "g"), (2, 0.5, "g"), (3, 0.0, "g")]
    with patch.object(vl, "match_by_objects", return_value=grid):
        out = vl.localize(["shelf"], [], _FakeNeo4j(_ref_map()), place="test", fusion_weights=W)
    assert out.estimated_xy == pytest.approx((5.0, 0.0))


# ── v2 fusion ────────────────────────────────────────────────────────────

from server.loc_fusion import fuse_v2, image_evidence, image_weight


def test_image_weight_grows_with_sensor_error():
    assert image_weight(None) == 2.0
    assert image_weight(1.0) == image_weight(2.0) == 0.25
    assert image_weight(5.0) == image_weight(9.0) == 4.0
    assert 0.25 < image_weight(3.5) < 4.0


def test_v2_posterior_sums_to_one_and_sensor_adds_candidates():
    pos = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (20.0, 0.0)}
    ev = image_evidence([(1, 0.6, "g")], {1: 0.9, 2: 0.5})   # images only know nodes 1, 2
    post = fuse_v2(ev, pos, sensor_pos=(19.5, 0.0), sigma=1.0)
    assert abs(sum(p for _, p, _ in post) - 1.0) < 1e-9
    assert post[0][0] == 3                    # a precise sensor wins, though no image listed node 3
    no_sensor = fuse_v2(ev, pos)
    assert no_sensor[0][0] == 1


def test_v2_drifted_sensor_defers_to_images():
    pos = {1: (0.0, 0.0), 2: (3.0, 0.0), 3: (6.0, 0.0)}
    ev = image_evidence([(1, 1.0, "g")], {1: 0.95, 2: 0.6, 3: 0.5})
    assert fuse_v2(ev, pos, sensor_pos=(3.0, 0.0), sigma=8.0)[0][0] == 1
    assert fuse_v2(ev, pos, sensor_pos=(3.0, 0.0), sigma=1.0)[0][0] == 2


def test_v2_unreachable_nodes_are_penalised():
    pos = {1: (0.0, 0.0), 2: (2.0, 0.0)}
    ev = {1: 0.0, 2: 0.0}
    post = fuse_v2(ev, pos, sensor_pos=(1.0, 0.0), sigma=2.0, reachable={1})
    assert post[0][0] == 1 and "unreachable" in dict((n, r) for n, _, r in post)[2]


def test_localize_v2_returns_probability_and_evidence():
    grid = [(1, 0.5, "g"), (2, 0.3, "g")]
    with patch.object(vl, "match_by_objects", return_value=grid):
        out = vl.localize(["shelf"], [], _FakeNeo4j(_ref_map()), place="test", v2=True,
                          dino_sims={2: 0.9, 1: 0.6}, sensor_pos=(10.0, 0.0), sensor_sigma=1.0)
    assert out.matched_nid == 2 and "v2" in out.method
    assert 0 < out.confidence <= 1 and out.image_evidence[2] > out.image_evidence[1]


def test_second_photo_evidence_is_added():
    grid = [(1, 0.4, "g"), (2, 0.4, "g")]
    with patch.object(vl, "match_by_objects", return_value=grid):
        alone = vl.localize(["shelf"], [], _FakeNeo4j(_ref_map()), place="test", v2=True,
                            dino_sims={1: 0.8, 2: 0.8})
        both = vl.localize(["shelf"], [], _FakeNeo4j(_ref_map()), place="test", v2=True,
                           dino_sims={1: 0.8, 2: 0.8}, extra_evidence={2: 2.0})
    assert both.matched_nid == 2 and both.confidence > alone.confidence


def test_reachable_nodes_follow_the_walkway():
    from server import server as srv
    m = RefMap(place="t")
    for nid, (x, y), nbrs in [(1, (0, 0), [2]), (2, (5, 0), [1, 3]), (3, (40, 0), [2])]:
        n = _node(nid, x, y)
        n.neighbor_nids = nbrs
        m.photos[nid] = n
    assert srv._reachable_nodes(m, 1, 0.0) == {1, 2}          # 8 m budget
    assert srv._reachable_nodes(m, 1, 30.0) == {1, 2, 3}      # 53 m budget
    assert srv._reachable_nodes(m, None, 30.0) is None


def test_reachable_nodes_walk_one_way_links_both_ways():
    # the reference map stores most walkway links in one direction only
    from server import server as srv
    m = RefMap(place="t")
    for nid, (x, y), nbrs in [(1, (0, 0), []), (2, (5, 0), [1]), (3, (10, 0), [2])]:
        n = _node(nid, x, y)
        n.neighbor_nids = nbrs
        m.photos[nid] = n
    assert srv._reachable_nodes(m, 1, 10.0) == {1, 2, 3}
    assert srv._reachable_nodes(m, 3, 10.0) == {1, 2, 3}


def test_phone_inputs_drop_non_finite_numbers():
    from server import server as srv
    nan, inf = float("nan"), float("inf")
    assert srv._phone_localization_inputs(None, None, nan, 2.0, 5.0, 7)["sensor_pos"] is None
    assert srv._phone_localization_inputs(None, None, 1.0, inf, 5.0, 7)["sensor_pos"] is None
    for bad in (nan, inf, -3.0):
        out = srv._phone_localization_inputs(None, None, 1.0, 2.0, bad, 7)
        assert out["sensor_pos"] == (1.0, 2.0) and out["dist_since_fix"] is None


def test_parse_dino_topk_skips_non_finite():
    assert parse_dino_topk('[{"nid": 1, "sim": NaN}, {"nid": 2, "sim": 0.8}]', None) == {2: 0.8}
    assert parse_dino_topk('[{"nid": Infinity, "sim": 0.8}]', None) == {}


def test_localize_v2_spread_posterior_names_no_node():
    m = RefMap(place="test")
    for nid in range(1, 21):
        m.photos[nid] = _node(nid, float(nid), 0.0)
    with patch.object(vl, "match_by_objects", return_value=[]):
        out = vl.localize(["shelf"], [], _FakeNeo4j(m), place="test", v2=True,
                          dino_sims={n: 0.5 for n in range(1, 21)})
    assert out.matched_nid is None and out.ref_node is None
    assert out.confidence == pytest.approx(1 / 20)
    assert out.image_evidence is not None and len(out.top_candidates) == 5   # kept for a second photo


def test_aisle_signs_only_for_configured_places():
    from server import server as srv
    from server.session import Session
    result = vl.LocalizationResult(matched_nid=None, confidence=0.0, method="none", reasoning="")
    for place, used in [("test", True), ("other store", False)]:
        s = Session(id="t", goal="milk", goal_objects=["milk"], place=place)
        with patch("server.server.get_neo4j", return_value=_FakeNeo4j(_ref_map())), \
             patch.object(srv._config, "AISLE_SIGN_PLACES", {"test"}), \
             patch("server.server.aisle_signs.sign_evidence", return_value={1: 1.0}) as ev, \
             patch("server.server._localize_photo", return_value=result) as loc:
            srv._run_early_localization(s, "sid", ["shelf"], [], dino_sims={1: 0.9},
                                        sign_numbers={12: 0.02})
        assert ev.called is used
        assert loc.call_args.kwargs["aisle_evidence"] == ({1: 1.0} if used else None)


# ── k(σ) settings ─────────────────────────────────────────────────────────

from server.loc_fusion import ImageK, parse_image_k


def test_parse_image_k():
    assert parse_image_k("32,64,8") == ImageK(k_min=32.0, k_max=64.0, k_no_sensor=8.0)
    for bad in ("1,2", "1,2,nan", "0,4,2", "-1,4,2", "a,b,c"):
        with pytest.raises(ValueError):
            parse_image_k(bad)


def test_image_weight_follows_custom_k():
    ks = ImageK(k_min=32, k_max=64, k_no_sensor=8)
    assert image_weight(None, ks) == 8
    assert image_weight(1.0, ks) == image_weight(2.0, ks) == 32
    assert image_weight(5.0, ks) == image_weight(9.0, ks) == 64
    assert image_weight(3.5, ks) == pytest.approx(48)
    assert image_weight(2.0) == 0.25          # default stays the phone-evidence k


def test_fuse_v2_large_image_k_lets_the_photo_overrule_a_drifted_sensor():
    pos = {1: (0.0, 0.0), 2: (3.0, 0.0)}
    ev = {1: 2.0, 2: 1.8}                        # photo prefers 1, sensor sits on 2
    assert fuse_v2(ev, pos, sensor_pos=(3.0, 0.0), sigma=1.5)[0][0] == 2
    big = ImageK(k_min=32, k_max=64, k_no_sensor=8)
    assert fuse_v2(ev, pos, sensor_pos=(3.0, 0.0), sigma=1.5, image_k=big)[0][0] == 1
    # …but the sensor still decides a near-tie
    tie = {1: 2.0, 2: 1.98}
    assert fuse_v2(tie, pos, sensor_pos=(3.0, 0.0), sigma=1.5, image_k=big)[0][0] == 2


# ── server-side DINOv2 evidence (phones without on-device DINOv2) ────────

import pickle

import numpy as np

from server import locate_fusion
from server.neo4j_client import DirectionalRef, RefObject


def _unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _dino_map():
    m = RefMap(place="test")
    for nid, (x, y) in _positions().items():
        n = _node(nid, x, y)
        n.directional_photos = [DirectionalRef(
            slot="front", heading_deg=0.0, photo_file=f"set01/{nid}_front.jpg",
            objects=[RefObject(label=f"thing{nid}", label_norm=f"thing{nid}", score=0.9,
                               ocr_text="", role="商品", grid_cell="mid-center")])]
        m.photos[nid] = n
    return m


@pytest.fixture
def dino_files(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.pkl"
    # keyed by basename, as backend/dino_sidecar.pkl is
    sidecar.write_bytes(pickle.dumps({"1_front.jpg": _unit(1, 0, 0), "2_front.jpg": _unit(0, 1, 0),
                                      "3_front.jpg": _unit(0, 0, 1)}))
    photo = tmp_path / "q.jpg"
    photo.write_bytes(b"jpeg")
    monkeypatch.setattr(locate_fusion, "_sidecar", {"path": None, "embs": None})
    monkeypatch.setattr(locate_fusion, "_index_cache", {})
    return str(sidecar), str(photo)


def test_server_evidence_scales_fused_score_to_0_2(dino_files, monkeypatch):
    sidecar, photo = dino_files
    from server import dino_embed
    monkeypatch.setattr(dino_embed, "embed_image", lambda p: _unit(0.2, 1, 0))   # looks like node 2
    ev = locate_fusion.server_image_evidence(_dino_map(), photo, [], [], sidecar).scores
    assert set(ev) == {1, 2, 3}
    assert ev[2] == pytest.approx(2.0) and min(ev.values()) == pytest.approx(0.0)
    assert ev[1] > ev[3]


def _four_way_map():
    """Node 1 with four calibrated directional photos (front faces 30°)."""
    m = RefMap(place="test")
    n = _node(1, 0.0, 0.0)
    for slot, h in (("front", 30.0), ("right", 120.0), ("back", 210.0), ("left", 300.0)):
        n.directional_photos.append(DirectionalRef(
            slot=slot, heading_deg=h, photo_file=f"set01/1_{slot}.jpg",
            objects=[RefObject(label="shelf", label_norm="shelf", score=0.9,
                               ocr_text="", role="商品", grid_cell="mid-center")]))
    m.photos[1] = n
    m.photos[2] = _node(2, 10.0, 0.0)
    return m


def test_server_evidence_records_each_nodes_most_similar_photo(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.pkl"
    sidecar.write_bytes(pickle.dumps({"1_front.jpg": _unit(1, 0, 0, 0), "1_right.jpg": _unit(0, 1, 0, 0),
                                      "1_back.jpg": _unit(0, 0, 1, 0), "1_left.jpg": _unit(0, 0, 0, 1)}))
    photo = tmp_path / "q.jpg"
    photo.write_bytes(b"jpeg")
    monkeypatch.setattr(locate_fusion, "_sidecar", {"path": None, "embs": None})
    monkeypatch.setattr(locate_fusion, "_index_cache", {})
    from server import dino_embed
    monkeypatch.setattr(dino_embed, "embed_image", lambda p: _unit(0, 0.3, 1, 0))   # looks like 1/back
    ev = locate_fusion.server_image_evidence(_four_way_map(), photo, [], [], str(sidecar))
    assert ev.best_photo[1].slot == "back" and ev.best_photo[1].heading_deg == 210.0
    assert 2 not in ev.best_photo                  # no directional photos → no photo to name


def test_heading_comes_from_the_best_photo_not_label_overlap():
    m = _four_way_map()
    back = m.photos[1].directional_photos[2]
    # labels match every photo equally, so label overlap alone would pick "front"
    h, slot, conf = vl.estimate_node_heading(m.photos[1], ["shelf"], [], 0.9, best_photo=back)
    assert (h, slot, conf) == (210.0, "back", 0.8)
    with patch.object(vl, "match_by_objects", return_value=[]):
        out = vl.localize(["shelf"], [], _FakeNeo4j(m), place="test", v2=True,
                          server_evidence={1: 2.0, 2: 0.0}, server_best_photo={1: back},
                          server_image_k=ImageK(32, 64, 8))
    assert out.matched_nid == 1 and out.matched_slot == "back" and out.matched_heading == 210.0
    assert out.matched_photo == "set01/1_back.jpg"


def test_best_photo_on_uncalibrated_node_gives_slot_but_no_heading():
    m = _four_way_map()
    for dp in m.photos[1].directional_photos:
        dp.heading_deg = 0.0
    back = m.photos[1].directional_photos[2]
    assert vl.estimate_node_heading(m.photos[1], ["shelf"], [], 0.9, best_photo=back) == (None, "back", 0.0)


def test_server_evidence_unavailable_returns_none(dino_files, monkeypatch, tmp_path):
    sidecar, photo = dino_files
    from server import dino_embed
    monkeypatch.setattr(dino_embed, "embed_image", lambda p: None)            # no torch / model
    assert locate_fusion.server_image_evidence(_dino_map(), photo, [], [], sidecar) is None
    monkeypatch.setattr(dino_embed, "embed_image", lambda p: _unit(1, 0, 0))
    assert locate_fusion.server_image_evidence(_dino_map(), photo, [], [],
                                               str(tmp_path / "missing.pkl")) is None
    assert locate_fusion.server_image_evidence(_dino_map(), str(tmp_path / "nope.jpg"), [], [],
                                               sidecar) is None


def test_localize_v2_uses_server_evidence_with_its_own_k():
    with patch.object(vl, "match_by_objects", return_value=[(1, 0.9, "g")]):
        out = vl.localize(["shelf"], [], _FakeNeo4j(_ref_map()), place="test", v2=True,
                          server_evidence={1: 1.8, 2: 2.0, 3: 0.0},
                          server_image_k=ImageK(32, 64, 8))
    assert out.matched_nid == 2 and out.method == "server_dino+v2"
    assert out.image_evidence == {1: 1.8, 2: 2.0, 3: 0.0}      # grid is not added on top
    # no sensor → k_no_sensor 8: p(2) = 1 / (1 + e^(−8·0.2) + e^(−8·2))
    assert out.confidence == pytest.approx(1 / (1 + math.exp(-1.6) + math.exp(-16)))


def test_localize_v2_prefers_phone_dino_over_server_evidence():
    with patch.object(vl, "match_by_objects", return_value=[]):
        out = vl.localize(["shelf"], [], _FakeNeo4j(_ref_map()), place="test", v2=True,
                          dino_sims={1: 0.9, 3: 0.5}, server_evidence={2: 2.0})
    assert out.matched_nid == 1 and out.method == "object_set+v2"


def test_run_early_localization_adds_server_dino_without_phone_dino():
    from server import server as srv
    from server.session import Session
    s = Session(id="t", goal="milk", goal_objects=["milk"], place="test")
    result = vl.LocalizationResult(matched_nid=None, confidence=0.0, method="none", reasoning="")
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(_ref_map())), \
         patch.object(srv._config, "LOC_FUSION_V2", True), \
         patch.object(srv._config, "LOC_SERVER_DINO", True), \
         patch.object(srv._config, "LOC_SERVER_IMAGE_K", "32,64,8"), \
         patch("server.locate_fusion.server_image_evidence",
               return_value=locate_fusion.ServerImageEvidence({2: 2.0}, {2: "photo"})) as sev, \
         patch("server.server._localize_photo", return_value=result) as loc:
        srv._run_early_localization(s, "sid", ["shelf"], ["7"], image_path="q.jpg")
        kw = loc.call_args.kwargs
        assert kw["server_evidence"] == {2: 2.0} and kw["server_image_k"] == ImageK(32, 64, 8)
        assert kw["server_best_photo"] == {2: "photo"}
        assert sev.call_args.args[2:4] == (["shelf"], ["7"])
        # the phone sent DINOv2 top-K → the server does not embed the photo
        sev.reset_mock()
        srv._run_early_localization(s, "sid", ["shelf"], [], image_path="q.jpg", dino_sims={1: 0.9})
        assert not sev.called and "server_evidence" not in loc.call_args.kwargs


def test_run_early_localization_server_dino_off_or_failing_falls_back():
    from server import server as srv
    from server.session import Session
    s = Session(id="t", goal="milk", goal_objects=["milk"], place="test")
    result = vl.LocalizationResult(matched_nid=None, confidence=0.0, method="none", reasoning="")
    with patch("server.server.get_neo4j", return_value=_FakeNeo4j(_ref_map())), \
         patch.object(srv._config, "LOC_FUSION_V2", True), \
         patch.object(srv._config, "LOC_SERVER_DINO", True), \
         patch("server.locate_fusion.server_image_evidence", side_effect=RuntimeError("boom")), \
         patch("server.server._localize_photo", return_value=result) as loc:
        assert srv._run_early_localization(s, "sid", ["shelf"], [], image_path="q.jpg") is result
    assert "server_evidence" not in loc.call_args.kwargs
