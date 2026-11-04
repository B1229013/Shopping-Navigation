"""Start points: saved by the map builder, read by the app (GET/POST /starts)."""
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from server import start_points
from server.server import app

PLACE = "測試店（1/2)"          # a "/" in the name, like A7家樂福（9/16)


def _workspace(tmp_path):
    ws = tmp_path / "mapbuild" / "測試店（1_2)"
    (ws / "01").mkdir(parents=True)
    Image.new("RGB", (1600, 1200), (200, 30, 30)).save(ws / "01" / "wp_1_front.jpg")
    return ws


def _client(tmp_path):
    maps = tmp_path / "maps"
    (maps / "neo4j_links").mkdir(parents=True)
    return maps, patch("server.server.EDITOR_MAP_DIR", maps), patch("server.config.MAP_BUILD_ROOT", tmp_path / "mapbuild")


def test_no_starts_is_an_empty_list(tmp_path):
    maps, p1, p2 = _client(tmp_path)
    with p1, p2:
        r = TestClient(app).get("/starts", params={"place": PLACE})
    assert r.status_code == 200 and r.json()["starts"] == []


def test_save_keeps_the_links_file_and_serves_the_photo(tmp_path):
    maps, p1, p2 = _client(tmp_path)
    _workspace(tmp_path)
    (maps / "neo4j_links" / "store.json").write_text(json.dumps(
        {"place": PLACE, "links": [{"from": 1, "to": 2}], "entrance": {"x": 0, "y": 0, "links": [1]}}), encoding="utf-8")
    with p1, p2:
        c = TestClient(app)
        r = c.post("/starts", params={"place": PLACE}, json={"starts": [
            {"name": "正門", "x": 1.234, "y": -2, "heading_deg": 370, "photo": {"set": "01", "file": "wp_1_front.jpg"}},
            {"name": "電梯口", "x": 5, "y": 6, "heading_deg": 90},
        ]})
        assert r.status_code == 200, r.text
        starts = r.json()["starts"]
        assert [s["name"] for s in starts] == ["正門", "電梯口"]
        assert starts[0]["x"] == 1.23 and starts[0]["heading_deg"] == 10.0
        assert starts[0]["photo_url"] and starts[1]["photo_url"] is None
        photo = c.get(starts[0]["photo_url"])
        assert photo.status_code == 200 and photo.headers["content-type"] == "image/jpeg"
        # the links file kept its other keys and wrote the starts into the same file
        data = json.loads((maps / "neo4j_links" / "store.json").read_text(encoding="utf-8"))
        assert data["links"] == [{"from": 1, "to": 2}] and data["entrance"]["links"] == [1]
        assert [s["name"] for s in data["starts"]] == ["正門", "電梯口"]
        # a start sent back as returned keeps its photo; dropping it deletes the photo
        first = starts[0]
        r = c.post("/starts", params={"place": PLACE}, json={"starts": [
            {"id": first["id"], "name": first["name"], "x": first["x"], "y": first["y"], "heading_deg": first["heading_deg"]}]})
        assert r.json()["starts"][0]["photo_url"]
        c.post("/starts", params={"place": PLACE}, json={"starts": []})
    assert not any(start_points.photo_dir(PLACE, maps).glob("*.jpg"))


def test_a_start_without_name_or_position_is_refused(tmp_path):
    maps, p1, p2 = _client(tmp_path)
    with p1, p2:
        c = TestClient(app)
        assert c.post("/starts", params={"place": PLACE}, json={"starts": [{"name": "", "x": 0, "y": 0, "heading_deg": 0}]}).status_code == 422
        assert c.post("/starts", params={"place": PLACE}, json={"starts": [{"name": "門"}]}).status_code == 422


def test_ids_stay_unique():
    used = set()
    a = start_points._clean({"id": "gate", "name": "A", "x": 0, "y": 0, "heading_deg": 0}, used)
    b = start_points._clean({"id": "gate", "name": "B", "x": 0, "y": 0, "heading_deg": 0}, used)
    c = start_points._clean({"name": "C", "x": 0, "y": 0, "heading_deg": 0}, used)
    assert len({a["id"], b["id"], c["id"]}) == 3
