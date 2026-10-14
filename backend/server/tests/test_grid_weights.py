"""Grid-matching weight overrides and the IDF rarity mode."""
import pytest

from server import visual_localization as vl


@pytest.fixture(autouse=True)
def _restore():
    saved = {k: getattr(vl, v) for k, v in vl._OVERRIDABLE.items()}
    yield
    vl.apply_weight_overrides(saved)


def _q(*labels):
    return {"0-0": [{"label": l, "weight": 1.0, "cat": "normal", "bbox_area": 0.02} for l in labels]}


def test_overrides_parse_and_apply():
    now = vl.apply_weight_overrides("landmark=7,ocr=2.5,rarity_mode=idf")
    assert now["landmark"] == 7.0 and vl.WEIGHT_LANDMARK == 7.0
    assert vl.WEIGHT_OCR == 2.5 and vl.RARITY_MODE == "idf"
    assert vl.classify_feature("走道標示")[0] == 7.0


def test_overrides_reject_unknown():
    with pytest.raises(ValueError):
        vl.apply_weight_overrides("nope=1")
    with pytest.raises(ValueError):
        vl.apply_weight_overrides("rarity_mode=other")


def test_cap_mode_gives_no_rarity_credit_to_exact_matches():
    rarity = {"rare": 1.0, "common": 0.03}
    vl.apply_weight_overrides({"rarity_mode": "cap"})
    # node A matches only the rare object, node B only the common one
    a, _, _ = vl.weighted_grid_match(_q("rare", "common"), _q("rare"), rarity)
    b, _, _ = vl.weighted_grid_match(_q("rare", "common"), _q("common"), rarity)
    assert a == pytest.approx(b)          # the bug: rarity changes nothing


def test_idf_mode_prefers_node_sharing_the_rare_object():
    rarity = {"rare": 1.0, "common": 0.03}
    vl.apply_weight_overrides({"rarity_mode": "idf"})
    a, _, _ = vl.weighted_grid_match(_q("rare", "common"), _q("rare"), rarity)
    b, _, _ = vl.weighted_grid_match(_q("rare", "common"), _q("common"), rarity)
    assert a > b
    full, _, _ = vl.weighted_grid_match(_q("rare", "common"), _q("rare", "common"), rarity)
    assert full <= 1.0 + 1e-9
