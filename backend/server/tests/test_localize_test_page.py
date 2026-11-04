"""The localization test page's endpoints: /localize-test runs the same localization as a
navigation photo (#11), and the reference-photo endpoints use the chosen store (#12) and
answer 404 for a missing photo (#13)."""
import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from server import server as srv
from server import visual_localization as vl
from server.tests.test_loc_fusion import _ref_map


class _Neo:
    def __init__(self):
        self.places = []
        self.m = _ref_map()

    def load_reference_map(self, place=None):
        self.places.append(place)
        return self.m


def _jpg(w=800, h=600):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="JPEG")
    return buf.getvalue()


def _perception():
    from server.vlm import VLMDetectedObject, VLMDetectedText, VLMPerception
    return VLMPerception(
        detections=[VLMDetectedObject(label="冷藏櫃", bbox=[100.0, 150.0, 500.0, 450.0], score=0.9,
                                      position="center", label_en="fridge")],
        ocr_texts=[VLMDetectedText(text="走道 7", score=0.9, bbox=[[10, 10], [90, 10], [90, 40], [10, 40]])])


def test_localize_test_runs_the_navigation_localization():
    neo = _Neo()
    result = vl.LocalizationResult(matched_nid=3, confidence=0.7, method="fused_v2", reasoning="",
                                   top_candidates=[(3, 0.7, "img"), (1, 0.2, "img")])
    with patch("server.server.get_neo4j", return_value=neo), \
         patch.object(srv._config, "VLM_ROUNDS", 1), patch.object(srv._config, "VLM_PASS3", False), \
         patch("server.server._vlm_perceive_two_pass", return_value=(_perception(), {})), \
         patch("server.server.get_ocr", return_value=None), \
         patch("server.server._run_early_localization", return_value=result) as loc:
        r = TestClient(srv.app).post("/localize-test", params={"ground_truth_nid": 3, "place": "系辦"},
                                     files={"photo": ("p.jpg", _jpg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    kw = loc.call_args.kwargs
    boxes = [it["box"] for it in kw["detected_items"] if it["box"]]
    assert boxes and all(0.0 <= v <= 1.0 for b in boxes for v in b)      # scaled like a navigation photo
    assert kw["ocr_items"] and kw["image_path"]
    body = r.json()
    assert body["matched"]["nid"] == body["top5"][0]["nid"] == 3        # one answer, not two
    assert body["ground_truth"]["rank"] == 1 and body["ground_truth"]["distance_m"] == 0.0
    assert body["top5"][0]["photo_url"] == "/reference-photo/3.jpg?place=%E7%B3%BB%E8%BE%A6"   # the chosen store


def test_reference_photos_use_the_chosen_store():
    neo = _Neo()
    client = TestClient(srv.app)
    with patch("server.server.get_neo4j", return_value=neo):
        client.get("/reference-photo/1.jpg", params={"place": "系辦"})
        client.get("/reference-dir-photo/1/front.jpg", params={"place": "系辦"})
    assert neo.places == ["系辦", "系辦"]


def test_missing_directional_photo_is_404():
    with patch("server.server.get_neo4j", return_value=_Neo()):
        r = TestClient(srv.app).get("/reference-dir-photo/1/left.jpg")
    assert r.status_code == 404
