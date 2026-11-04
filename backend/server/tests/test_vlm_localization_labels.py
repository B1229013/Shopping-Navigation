"""Pass-3 replies are parsed; localization gets 「中文 English」 names; label embeddings are
fetched, cached on disk and used by the label similarity."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from server import visual_localization as VL
from server.vlm import VLMDetectedObject, _parse_perception, localization_label


def test_items_keep_both_names_and_localization_joins_them():
    text = json.dumps({"scene": "s", "items": [
        {"zh": "區域招牌", "en": "area sign", "conf": 0.9, "box": [0.1, 0.1, 0.3, 0.2], "text": "生鮮"},
        {"zh": "", "en": "exit sign", "conf": 0.8, "box": [0.5, 0.1, 0.6, 0.2]},
    ]})
    p = _parse_perception(text, 1000, 1000)
    assert [d.label for d in p.detections] == ["區域招牌", "exit sign"]      # unchanged for everything else
    assert [localization_label(d) for d in p.detections] == ["區域招牌 area sign", "exit sign"]
    assert [t.text for t in p.ocr_texts] == ["生鮮"]


def test_localization_label_does_not_repeat_english():
    d = VLMDetectedObject(label="MILO 商品", bbox=[0, 0, 1, 1], score=1.0, label_en="milo")
    assert localization_label(d) == "MILO 商品"


def test_pass3_format_is_parsed():
    text = json.dumps({"scene_description": "store", "store_zone": "aisle",
                       "detections": [{"label": "barber pole sign", "box": [0.8, 0.3, 0.85, 0.4], "score": 0.9},
                                      {"label": "shopping cart", "box": [0, 0, 0, 0], "score": 0.0}],
                       "ocr_texts": [{"text": "專業剪髮", "score": 0.96, "box": [0.7, 0.1, 0.9, 0.2]},
                                     {"text": "", "score": 0.5, "box": [0.1, 0.1, 0.2, 0.2]}]})
    p = _parse_perception(text, 1000, 2000)
    assert [d.label for d in p.detections] == ["barber pole sign"]       # [0,0,0,0] box dropped
    assert p.detections[0].bbox == [800.0, 600.0, 850.0, 800.0]
    assert [t.text for t in p.ocr_texts] == ["專業剪髮"]
    assert p.ocr_texts[0].bbox[0] == [700.0, 200.0]


def _fake_post(dim=8):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(list(json["input"]))
        post.last = (url, headers["Authorization"])

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"index": i, "embedding": list(np.eye(dim)[hash(t) % dim] + 0.01)}
                                 for i, t in enumerate(json["input"])]}
        return R()
    return post, calls


def test_embeddings_are_fetched_persisted_and_reloaded(tmp_path, monkeypatch):
    from server import config
    path = tmp_path / "label_embeddings.npz"
    monkeypatch.setattr(config, "EMBEDDING_CACHE_PATH", str(path))
    monkeypatch.setattr(config, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(VL, "_emb_loaded", False)
    post, calls = _fake_post()
    monkeypatch.setattr(VL.requests, "post", post)

    assert VL.ensure_label_embeddings(["地圖物件 a", "地圖物件 b"], persist=True) == 2
    assert VL.ensure_label_embeddings(["地圖物件 a"], persist=True) == 0     # cached: no request
    assert VL.ensure_label_embeddings(["照片物件"]) == 1                      # photo labels: memory only
    assert len(calls) == 2 and path.is_file()

    # a fresh process loads the map labels from disk, not the photo's
    monkeypatch.setattr(VL, "_emb_cache", {})
    monkeypatch.setattr(VL, "_emb_persisted", set())
    monkeypatch.setattr(VL, "_emb_loaded", False)
    assert VL._load_embedding_cache()
    assert set(VL._emb_cache) == {"地圖物件 a", "地圖物件 b"}


def test_embeddings_go_to_the_configured_service(tmp_path, monkeypatch):
    # A CGU-gateway key must go to the gateway, never to api.openai.com.
    from server import config
    monkeypatch.setattr(config, "EMBEDDING_CACHE_PATH", str(tmp_path / "e.npz"))
    monkeypatch.setattr(config, "EMBEDDING_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setattr(config, "EMBEDDING_API_KEY", "gateway-key")
    monkeypatch.setattr(VL, "_emb_loaded", False)
    post, _ = _fake_post()
    monkeypatch.setattr(VL.requests, "post", post)
    assert VL.ensure_label_embeddings(["貨架"]) == 1
    assert post.last == ("https://gateway.example/v1/embeddings", "Bearer gateway-key")


def test_embedding_service_defaults_to_the_vlm_service():
    env = {**os.environ, "OPENAI_BASE_URL": "https://gateway.example/v1/", "OPENAI_API_KEY": "gateway-key",
           "EMBEDDING_BASE_URL": "", "EMBEDDING_API_KEY": ""}     # empty = unset, even if .env sets them
    out = subprocess.run(
        [sys.executable, "-c", "from server import config as c; print(c.EMBEDDING_BASE_URL, c.EMBEDDING_API_KEY)"],
        cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["https://gateway.example/v1", "gateway-key"]


def test_hybrid_similarity_uses_the_vectors(monkeypatch):
    v = np.ones(4, np.float32)
    monkeypatch.setattr(VL, "_emb_cache", {"區域招牌": v, "區域招牌 area sign": v * 2})
    assert VL._semantic_similarity("區域招牌", "區域招牌 area sign") < 0.55   # characters alone
    assert VL._hybrid_max_sim("區域招牌", "區域招牌 area sign") > 0.99
