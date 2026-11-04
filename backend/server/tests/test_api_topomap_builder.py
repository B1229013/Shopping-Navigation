"""/topomap-builder endpoints: every store reads and writes its own MAP_BUILD_ROOT/<店名>/ workspace."""
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

import server.config as config
from server.server import app

PLACE = "測試店"
P = {"place": PLACE}


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "mapbuild"
    r.mkdir()
    monkeypatch.setattr(config, "MAP_BUILD_ROOT", r)
    return r


@pytest.fixture
def ws(root):
    w = root / PLACE
    w.mkdir()
    return w


def test_topomap_data_and_confirm_use_the_store_workspace(ws):
    client = TestClient(app)
    assert client.get("/topomap-data", params=P).status_code == 404
    (ws / "topomap.json").write_text(json.dumps({"place": PLACE, "nodes": [], "edges": []}), encoding="utf-8")
    assert client.get("/topomap-data", params=P).json()["place"] == PLACE
    r = client.post("/topomap-confirm", json={"place": PLACE, "nodes": [{"id": 1}], "edges": []})
    assert r.status_code == 200 and PLACE in r.json()["message"]
    assert json.loads((ws / "topomap_confirmed.json").read_text(encoding="utf-8"))["nodes"] == [{"id": 1}]
    assert client.get("/topomap-data").status_code == 422          # which store? required


def test_two_stores_do_not_share_files(root):
    client = TestClient(app)
    for place in ("A7家樂福（9/16)", "全聯 XX店"):
        client.post("/topomap-confirm", json={"place": place, "nodes": [{"id": place}], "edges": []})
    a = json.loads((root / "A7家樂福（9_16)" / "topomap_confirmed.json").read_text(encoding="utf-8"))
    b = json.loads((root / "全聯 XX店" / "topomap_confirmed.json").read_text(encoding="utf-8"))
    assert a["nodes"] == [{"id": "A7家樂福（9/16)"}] and b["nodes"] == [{"id": "全聯 XX店"}]


def test_places_listing(root, ws):
    (ws / "topomap.json").write_text(json.dumps({"place": PLACE, "nodes": [{"id": 1}], "edges": []}), encoding="utf-8")
    (root / "空資料夾").mkdir()
    places = TestClient(app).get("/topomap-places").json()["places"]
    assert [(p["place"], p["nodes"], p["confirmed"]) for p in places] == [(PLACE, 1, False)]


def test_test_photo_serves_workspace_files_only(ws):
    (ws / "01").mkdir()
    (ws / "01" / "a.jpg").write_bytes(b"\xff\xd8jpg")
    (ws.parent / "secret.txt").write_text("x")
    client = TestClient(app)
    assert client.get("/test-photo/01/a.jpg", params=P).content == b"\xff\xd8jpg"
    assert client.get("/test-photo/01/missing.jpg", params=P).status_code == 404
    assert client.get("/test-photo/01/..%2F..%2Fsecret.txt", params=P).status_code == 404
    assert client.get("/test-photo/01/a.jpg", params={"place": "別家"}).status_code == 404


def test_existing_topomap_404_without_reference_place(ws, monkeypatch):
    monkeypatch.setattr(config, "MAP_BUILD_REF_PLACE", "")
    assert TestClient(app).get("/existing-topomap").status_code == 404


def test_floorplan_upload_replace_and_delete(ws):
    client = TestClient(app)
    assert client.get("/topomap-floorplan", params=P).status_code == 404
    r = client.post("/topomap-floorplan", params=P, files={"file": ("plan.png", b"\x89PNG one", "image/png")})
    assert r.status_code == 200
    assert client.get("/topomap-floorplan", params=P).content == b"\x89PNG one"
    client.post("/topomap-floorplan", params=P, files={"file": ("plan.jpg", b"\xff\xd8 two", "image/jpeg")})
    assert sorted(p.name for p in ws.glob("floorplan.*")) == ["floorplan.jpg"]   # old one replaced
    assert client.get("/topomap-floorplan", params=P).content == b"\xff\xd8 two"
    assert client.delete("/topomap-floorplan", params=P).status_code == 200
    assert client.get("/topomap-floorplan", params=P).status_code == 404


def test_floorplan_rejects_non_images(ws):
    r = TestClient(app).post("/topomap-floorplan", params=P, files={"file": ("x.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 415
    assert not list(ws.glob("floorplan.*"))


def test_topomap_suggest_embeds_once_and_caches(ws, monkeypatch):
    import numpy as np
    from server import dino_embed

    nodes = []
    for i, (wp, base) in enumerate([(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "A")], start=1):
        photos = {s: f"{base}{wp}_{s}.jpg" for s in ("front", "right", "back", "left")}
        nodes.append({"id": i, "set": "01", "wp_num": wp, "photos": photos})
        (ws / "01").mkdir(exist_ok=True)
        for f in photos.values():
            (ws / "01" / f).write_bytes(b"jpg")
    (ws / "topomap.json").write_text(json.dumps({"place": PLACE, "nodes": nodes, "edges": []}), encoding="utf-8")

    calls = []
    def fake_embed(path):
        calls.append(path)
        name = path.replace("\\", "/").rsplit("/", 1)[1]
        rng = np.random.default_rng(abs(hash((name[0], name.split("_")[1]))) % 2**32)
        v = rng.normal(size=16)
        return (v / np.linalg.norm(v)).astype(np.float32)
    monkeypatch.setattr(dino_embed, "available", lambda: True)
    monkeypatch.setattr(dino_embed, "embed_image", fake_embed)

    client = TestClient(app)
    first = client.get("/topomap-suggest", params=P).json()
    assert first["embedded"] == 20 and len(calls) == 20
    assert [(i["a"], i["b"]) for i in first["loops"]] == [(1, 5)]      # waypoints 1 and 5 share photos "A"
    assert (ws / "dino_embeddings.pkl").exists()
    second = client.get("/topomap-suggest", params=P).json()
    assert second["embedded"] == 0 and len(calls) == 20


def test_topomap_suggest_without_dino(ws, monkeypatch):
    from server import dino_embed
    (ws / "01").mkdir()
    (ws / "01" / "a.jpg").write_bytes(b"jpg")
    nodes = [{"id": 1, "set": "01", "wp_num": 1, "photos": {"front": "a.jpg"}}]
    (ws / "topomap.json").write_text(json.dumps({"nodes": nodes, "edges": []}), encoding="utf-8")
    monkeypatch.setattr(dino_embed, "available", lambda: False)
    assert TestClient(app).get("/topomap-suggest", params=P).status_code == 503


def test_walks_dialog_endpoints(root, tmp_path, monkeypatch):
    from server.tests.test_map_build import _folder
    from server.tests.test_map_v2 import CSV, PHOTOS

    data = tmp_path / "data"
    walk = _folder(data, "walkA")
    monkeypatch.setattr(config, "MAP_BUILD_SOURCES", [data])
    client = TestClient(app)

    listing = client.get("/topomap-sources").json()            # new store: nothing in use yet
    assert [s["name"] for s in listing["sources"]] == ["walkA"] and listing["walks"] == []

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sensor_log.csv", CSV)
        for p in PHOTOS:
            zf.writestr(f"photos/{p}", b"x")
    up = client.post("/topomap-sources/upload", params=P,
                     files={"file": ("phone walk.zip", buf.getvalue(), "application/zip")})
    assert up.status_code == 200 and up.json()["name"] == "phone walk"
    assert (root / PLACE / "imports" / "phone walk.zip").is_file()
    bad = client.post("/topomap-sources/upload", params=P, files={"file": ("x.zip", b"not a zip", "application/zip")})
    assert bad.status_code == 422

    r = client.post("/topomap-walks", json={"place": PLACE, "sources": [str(walk), up.json()["path"]]})
    assert r.status_code == 200 and r.json()["nodes"] == 4 and r.json()["folder"] == PLACE
    walks = client.get("/topomap-sources", params=P).json()["walks"]
    assert [(w["set"], w["nodes"]) for w in walks] == [("01", 2), ("02", 2)]

    # a second store gets its own folder; the zip uploaded under 測試店 is usable via from_place
    r2 = client.post("/topomap-walks", json={"place": "第二間", "sources": [up.json()["path"]], "from_place": PLACE})
    assert r2.status_code == 200 and (root / "第二間" / "topomap.json").is_file()
    assert json.loads((root / PLACE / "topomap.json").read_text(encoding="utf-8"))["stats"]["total_nodes"] == 4

    outside = tmp_path / "elsewhere"
    _folder(tmp_path, "elsewhere")
    assert client.post("/topomap-walks", json={"place": PLACE, "sources": [str(outside)]}).status_code == 422
    assert client.post("/topomap-walks", json={"place": "", "sources": [str(walk)]}).status_code == 422
