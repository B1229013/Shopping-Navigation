"""Aisle-number sign parsing and the aisle table (server/aisle_signs.py)."""
from server import aisle_signs as A
from server.neo4j_client import DirectionalRef, RefMap, RefObject, RefPhotoNode


def _obj(label, text="", box=(0.4, 0.1, 0.6, 0.3)):
    return RefObject(label=label, label_norm=label, score=0.9, ocr_text=text, role="", grid_cell="c", bbox=list(box))


def test_numbers_only_from_aisle_signs():
    items = [{"label": "走道編號吊牌 aisle number sign", "text": "12", "box": [0.4, 0.1, 0.6, 0.3]},
             {"label": "價格標籤 price tag", "text": "12", "box": [0, 0, 0.1, 0.1]},
             {"label": "走道編號吊牌 5 aisle number sign", "box": [0.1, 0.1, 0.15, 0.15]}]
    nums = A.numbers_in_items(items)
    assert set(nums) == {12, 5} and nums[12] > nums[5]


def _map():
    m = RefMap(place="t")
    rows = {1: (-6.5, 22.0, "12"), 2: (-21.3, 22.0, "12"), 3: (-6.5, 8.3, "4"), 4: (-11.3, 22.0, "")}
    for nid, (x, y, sign) in rows.items():
        objs = [_obj("走道編號吊牌 aisle sign", sign)] if sign else [_obj("shelf")]
        m.photos[nid] = RefPhotoNode(nid=nid, photo_file=f"{nid}.jpg", pdr_x=x, pdr_y=y, heading_deg=0.0,
                                     session="s", total_steps=0, total_distance_m=0.0, objects=[],
                                     directional_photos=[DirectionalRef(slot="right", heading_deg=0.0,
                                                                        photo_file=f"{nid}_r.jpg", objects=objs)])
    return m


def test_build_table_rows_and_odd_even_pairing():
    t = A.build_table(_map())
    assert t[12] == [1, 2, 4] and t[11] == t[12]
    assert t[4] == [3]


def test_evidence_and_no_table_without_file(tmp_path, monkeypatch):
    t = A.build_table(_map())
    ev = A.evidence({12: 0.02}, t)
    assert ev == {1: 1.0, 2: 1.0, 4: 1.0}
    monkeypatch.setattr(A, "AISLES_FILE", tmp_path / "none.json")
    A._cache.clear()
    assert A.table_for("t", _map(), derive=False) is None
    A._cache.clear()
    assert A.table_for("t", _map())[12] == [1, 2, 4]       # derived from the reference photos


def test_table_from_file(tmp_path, monkeypatch):
    f = tmp_path / "aisles.json"
    f.write_text('{"t": {"12": [1, 4]}}', encoding="utf-8")
    monkeypatch.setattr(A, "AISLES_FILE", f)
    A._cache.clear()
    assert A.table_for("t", _map()) == {12: [1, 4]}


from types import SimpleNamespace


def _ocr(text, conf, x1, y1, x2, y2):
    return SimpleNamespace(text=text, confidence=conf, bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def test_numbers_from_ocr_keeps_confident_signs_high_in_the_frame():
    res = [_ocr("13", 1.0, 100, 100, 200, 160), _ocr("14", 0.95, 600, 100, 700, 160),
           _ocr("1", 0.7, 0, 0, 400, 400),           # low confidence misread
           _ocr("12", 1.0, 100, 900, 150, 950),      # low in the frame: a price, not a sign
           _ocr("99", 1.0, 100, 100, 150, 150)]      # not an aisle number
    nums = A.numbers_from_ocr(res, 1000, 1000)
    assert set(nums) == {13, 14} and nums[13] > 0


def _walkway_map():
    m = RefMap(place="w")
    spec = {1: (-6.5, 22.0, ["12"]), 40: (-11.3, 22.0, ["14", "13", "12", "11"]), 30: (-11.3, 25.5, []),
            4: (-6.5, 25.5, ["14"]), 45: (-21.3, 22.0, ["12"])}
    for nid, (x, y, signs) in spec.items():
        objs = [_obj("走道編號吊牌 aisle sign", t) for t in signs] or [_obj("shelf")]
        m.photos[nid] = RefPhotoNode(nid=nid, photo_file=f"{nid}.jpg", pdr_x=x, pdr_y=y, heading_deg=0.0,
                                     session="s", total_steps=0, total_distance_m=0.0, objects=[],
                                     directional_photos=[DirectionalRef(slot="front", heading_deg=0.0,
                                                                        photo_file=f"{nid}_f.jpg", objects=objs)])
    return m


def test_walkway_is_the_column_that_sees_a_sequence_of_signs():
    assert A.walkway_nodes(_walkway_map()) == {40, 30}


def test_sign_evidence_row_plus_walkway():
    m = _walkway_map()
    t = A.build_table(m)
    ev = A.sign_evidence({11: 0.002, 12: 0.002, 9: 0.0005}, m, table=t, walkway={40, 30})
    assert ev[40] == 2.0                      # in the 11|12 row and on the walkway
    assert ev[1] == ev[45] == 1.0             # same row, other walkways
    assert ev[30] == 1.5                      # walkway, one row over
    assert A.sign_evidence({12: 0.002}, m, table=t, walkway={40, 30})[40] == 1.0   # one sign: no walkway bonus
