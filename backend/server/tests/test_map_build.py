"""server/map_build.py — importing Sensor Lab exports into the map-building workspace."""
import json
import zipfile

import pytest

from server import map_build as mb
from server.tests.test_map_v2 import CSV, PHOTOS


def _folder(root, name, photos=PHOTOS):
    d = root / name
    (d / "photos").mkdir(parents=True)
    (d / "sensor_log.csv").write_text(CSV, encoding="utf-8")
    for p in photos:
        (d / "photos" / p).write_bytes(b"jpg-" + p.encode())
    return d


def _zip(root, name):
    z = root / f"{name}.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("sensor_log.csv", CSV)
        for p in PHOTOS:
            zf.writestr(f"photos/{p}", b"jpg-" + p.encode())
    return z


def test_assign_sets_keeps_ids_and_never_reuses_them(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    walks, nxt = mb.assign_sets(None, [a, b])
    assert walks == [("01", a), ("02", b)] and nxt == 3
    current = {"walks": [{"set": "01", "source": str(a)}, {"set": "02", "source": str(b)}], "next_set": 3}
    # drop a, add c: b stays 02, c gets 03 (not the freed 01)
    walks, nxt = mb.assign_sets(current, [c, b])
    assert walks == [("02", b), ("03", c)] and nxt == 4
    current = {"walks": [{"set": "02", "source": str(b)}], "next_set": 4}
    assert mb.assign_sets(current, [a])[0] == [("04", a)]


def test_build_workspace_from_folder_and_zip(tmp_path):
    src = tmp_path / "src"
    f = _folder(src, "walkA")
    z = _zip(src, "walkB")
    ws = tmp_path / "ws"
    topo, report = mb.build_workspace("店", [("01", f), ("02", z)], ws, 3)
    assert [n["uuid"] for n in topo["nodes"]] == ["set01_wp1", "set01_wp2", "set02_wp1", "set02_wp2"]
    assert sorted(p.name for p in (ws / "02").iterdir()) == sorted(PHOTOS[:-1])   # notes.txt skipped
    saved = json.loads((ws / "topomap.json").read_text(encoding="utf-8"))
    assert saved["next_set"] == 3 and [w["set"] for w in saved["walks"]] == ["01", "02"]
    assert saved["walks"][0]["source"] == str(f) and set(saved["tracks"]) == {"01", "02"}
    assert any("2 個拍照節點" in line for line in report)

    # rebuilding without walk 01 removes its photo folder; a stale photo in 02 is cleaned up
    (ws / "02" / "wp_9_front_1.jpg").write_bytes(b"old")
    mb.build_workspace("店", [("02", z)], ws, 3)
    assert not (ws / "01").exists()
    assert not (ws / "02" / "wp_9_front_1.jpg").exists()


def test_list_sources_finds_folders_zips_and_imports(tmp_path):
    root, ws = tmp_path / "data", tmp_path / "ws"
    _folder(root, "系辦")
    _folder(root / "group", "nested")
    _zip(root, "walkZ")
    (root / "not_a_walk").mkdir()
    (ws / "imports").mkdir(parents=True)
    _zip(ws / "imports", "uploaded")
    found = {e["name"]: e for e in mb.list_sources([root], ws)}
    assert set(found) == {"系辦", "nested", "walkZ", "uploaded"}
    assert found["系辦"]["waypoints"] == 2 and found["系辦"]["kind"] == "folder"
    assert found["walkZ"]["kind"] == "zip"
    assert found["系辦"]["recorded"] == ""          # the test CSV header has no export time


def test_place_folder_names():
    assert mb.place_folder("系辦") == "系辦"
    assert mb.place_folder("A7家樂福（9/16)") == "A7家樂福（9_16)"
    assert mb.place_folder('a:b*c?"d<e>f|g\h') == "a_b_c_d_e_f_g_h"
    assert mb.place_folder("  店名. ") == "店名"
    for bad in ("", "   ", ".."):
        with pytest.raises(ValueError):
            mb.place_folder(bad)
    assert mb.workspace_for("系辦", mb.Path("root")) == mb.Path("root") / "系辦"


def test_list_places(tmp_path):
    for name, place, confirmed in (("系辦", "系辦", True), ("A7_9_16", "A7（9/16)", False)):
        d = tmp_path / name
        d.mkdir()
        (d / "topomap.json").write_text(json.dumps({"place": place, "nodes": [{}, {}], "walks": [{}]}), encoding="utf-8")
        if confirmed:
            (d / "topomap_confirmed.json").write_text("{}", encoding="utf-8")
    (tmp_path / "空的").mkdir()
    got = {p["place"]: p for p in mb.list_places(tmp_path)}
    assert set(got) == {"系辦", "A7（9/16)"}
    assert got["系辦"]["confirmed"] and got["系辦"]["nodes"] == 2 and got["系辦"]["walks"] == 1
    assert got["A7（9/16)"]["folder"] == "A7_9_16"
    assert mb.list_places(tmp_path / "missing") == []
