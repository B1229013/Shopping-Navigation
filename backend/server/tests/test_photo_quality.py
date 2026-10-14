"""Blur / darkness / occlusion checks (server/photo_quality.py) and how /photo uses them."""
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageFilter
from fastapi.testclient import TestClient

from server import photo_quality as pq
from server.models import VLMAction, VLMResponse
from server.perception import Detection


def _textured(tmp_path, name="ok.jpg", blur=0, scale=1.0):
    rng = np.random.default_rng(0)
    a = (rng.random((600, 450)) * 255 * scale).astype(np.uint8)
    im = Image.fromarray(a).convert("RGB")
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    p = tmp_path / name
    im.save(p, "JPEG", quality=90)
    return str(p)


def test_sharp_bright_photo_passes(tmp_path):
    q = pq.assess_image(_textured(tmp_path))
    assert q.hint is None and q.sharpness > pq.MIN_SHARPNESS and q.brightness > pq.MIN_BRIGHTNESS


def test_blurred_photo_asks_retake(tmp_path):
    assert pq.assess_image(_textured(tmp_path, blur=12)).hint == pq.HINT_BLUR


def test_dark_photo_asks_for_flash(tmp_path):
    assert pq.assess_image(_textured(tmp_path, scale=0.2)).hint == pq.HINT_DARK


@pytest.mark.parametrize("label,expected", [
    ("person", True), ("shopping cart", True), ("遮擋物:人", True), ("購物車", True), ("人", True),
    ("personal care products", False), ("人參雞精", False), ("手套", False), ("shelf", False),
    ("handle", False),
    # head noun decides: a product or a sign that merely names people is not a blocker
    ("hand soap", False), ("customer service", False), ("customer service desk", False),
    ("顧客服務台", False), ("顧客服務", False), ("洗手乳", False), ("cart corral sign", False),
    ("person with shopping cart", True), ("hands holding phone", True), ("Shopper", True),
    ("手推車", True), ("推著購物車的顧客", True), ("遮擋物:推車", True), ("工作人員", True),
    ("顧客們", True), ("cart", True),
])
def test_is_occluder(label, expected):
    assert pq.is_occluder(label) is expected


def _det(label, box):
    return SimpleNamespace(label=label, box=box)


def test_large_person_is_occlusion():
    dets = [_det("person", [0, 0, 600, 800])] + [_det(f"shelf{i}", [0, 0, 10, 10]) for i in range(5)]
    o = pq.assess_occlusion(dets, 1000, 1000)
    assert o.fraction == pytest.approx(0.6 * 0.8, abs=0.03)
    assert o.hint == pq.HINT_OCCLUDED


def test_small_person_far_away_is_fine():
    dets = [_det("person", [500, 500, 550, 600])] + [_det(f"shelf{i}", [0, 0, 10, 10]) for i in range(5)]
    assert pq.assess_occlusion(dets, 1000, 1000).hint is None


def test_few_objects_alone_is_not_occlusion():
    assert pq.assess_occlusion([_det("shelf", [0, 0, 10, 10])], 100, 100).hint is None


def test_partly_blocked_and_few_objects():
    dets = [_det("person", [0, 0, 40, 40]), _det("shelf", [90, 90, 100, 100])]
    assert pq.assess_occlusion(dets, 100, 100).hint == pq.HINT_FEW_OBJECTS


# ── /photo endpoint ───────────────────────────────────────────────────────

def _jpg(blur=0):
    rng = np.random.default_rng(1)
    im = Image.fromarray((rng.random((400, 300)) * 255).astype(np.uint8)).convert("RGB")
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _session(client):
    with patch("server.server.decompose_goal", return_value=["milk"]):
        return client.post("/session", json={"goal": "find the milk", "place": ""}).json()["session_id"]


def test_blurred_upload_is_rejected_before_any_model_call():
    from server.server import app
    client = TestClient(app)
    sid = _session(client)
    perception = MagicMock()
    with patch("server.server.get_perception", return_value=perception), \
         patch("server.server.vlm_decide") as decide:
        r = client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(blur=12), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["retake_hint"] == pq.HINT_BLUR and body["node_id"] == -1
    perception.detect.assert_not_called()
    decide.assert_not_called()


def test_occluded_upload_sets_retake_hint_and_skips_localization():
    from server.server import app
    client = TestClient(app)
    sid = _session(client)
    perception = MagicMock()
    perception.detect.return_value = [
        Detection(label="person", box=[0, 0, 300, 300], score=0.9),
        Detection(label="shelf", box=[0, 350, 50, 400], score=0.8),
    ]
    resp = VLMResponse(action=VLMAction.MOVE, guidance="go", question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=perception), \
         patch("server.server.vlm_decide", return_value=resp), \
         patch("server.server._run_early_localization") as early, \
         patch("server.server._build_route_context",          # a session with a store map
               return_value=(None, "這張照片對不到地圖上的位置")):
        body = client.post(f"/session/{sid}/photo",
                           files={"photo": ("p.jpg", _jpg(), "image/jpeg")}).json()
    early.assert_not_called()          # no position, heading or route change from a blocked view
    assert body["retake_hint"] == pq.HINT_OCCLUDED
    assert body["next_instruction"] == pq.HINT_OCCLUDED   # not the "doesn't match the map" hint
    assert body["guidance"] == pq.HINT_OCCLUDED
    assert body["corrected_node_id"] is None


def test_gallery_upload_ignores_live_heading_and_position():
    from server import server as srv
    client = TestClient(srv.app)
    sid = _session(client)
    perception = MagicMock()
    perception.detect.return_value = [Detection(label=f"shelf{i}", box=[0, 0, 50, 50], score=0.8)
                                      for i in range(4)]
    resp = VLMResponse(action=VLMAction.MOVE, guidance="go", question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=perception), \
         patch("server.server.vlm_decide", return_value=resp), \
         patch("server.server._phone_localization_inputs", wraps=srv._phone_localization_inputs) as inputs, \
         patch("server.server._apply_phone_heading") as heading:
        client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")},
                    data={"source": "gallery", "heading": "90", "map_x": "1", "map_y": "2",
                          "dist_since_fix": "5"})
    assert inputs.call_args.args[2:] == (None, None, None, None)   # map_x, map_y, dist, fix_node
    assert heading.call_args.args[2] is None


def test_occluder_prompts_do_not_filter_a_shopping_cart_goal():
    from server.server import app
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=["shopping cart"]):
        body = client.post("/session", json={"goal": "shopping cart", "place": ""}).json()
    from server.server import _store
    assert _store.get(body["session_id"]).goal_objects == ["shopping cart"]


def test_the_goal_itself_is_not_an_occluder():
    cart = [_det("shopping cart", [0, 0, 100, 100]), _det("shelf", [0, 0, 5, 5])]
    assert pq.assess_occlusion(cart, 100, 100).hint == pq.HINT_OCCLUDED
    assert pq.assess_occlusion(cart, 100, 100, goals=["Shopping Cart"]).hint is None
