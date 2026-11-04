"""What the user is told around an arrival: a too-far ARRIVED that becomes MOVE (#8), the
instruction repeated after 「沒有」 (#9), and the area name of the target node (#10)."""
from types import SimpleNamespace

from server import server as srv
from server.models import VLMAction, VLMResponse
from server.neo4j_client import RefMap, RefPhotoNode, RefObject


def _session(item="泡麵"):
    leg = SimpleNamespace(to_node=7, purpose="target")
    return SimpleNamespace(route_plan=SimpleNamespace(legs=[leg]), current_leg_index=0, phase="shopping",
                           target_node_to_item={7: item}, target_objects=[item], goal_objects=[item])


ARRIVED = VLMResponse(action=VLMAction.ARRIVED, guidance="你已經到達泡麵區了！", question=None, vlm_summary="s")


# ── #8 too far from the goal on the map ──

def test_far_arrival_keeps_walking_without_arrival_words():
    out = srv._gate_far_arrival(_session(), ARRIVED, "App 認為…你已經看到「泡麵」了嗎？", 20.0)
    assert out.action == VLMAction.MOVE and out.question is None
    assert "到達" not in out.guidance and "泡麵" in out.guidance


def test_far_arrival_still_asks_when_the_vlm_checked_the_item():
    q = "App 認為你在目前的位置，離「泡麵」還有約 20 公尺。你已經看到「泡麵」了嗎？"
    out = srv._gate_far_arrival(_session(), ARRIVED, q, 20.0, goal_verified=True)
    assert out.action == VLMAction.ASK and out.question == q


def test_nearer_arrival_asks():
    out = srv._gate_far_arrival(_session(), ARRIVED, "q?", srv.ARRIVAL_ASK_MAX_M - 1)
    assert out.action == VLMAction.ASK and out.question == "q?"


# ── #9 instruction repeated after 「沒有」 ──

def test_resume_instruction_is_a_route_step_only():
    assert srv._resume_instruction("直走 5 公尺後右轉", None) == "直走 5 公尺後右轉"
    assert srv._resume_instruction(srv.LOCALIZE_FAILED_HINT, None) is None
    assert srv._resume_instruction("畫面被人或推車擋住了，請避開或換個角度再拍", "畫面被人或推車擋住了，請避開或換個角度再拍") is None
    assert srv._resume_instruction(None, None) is None


# ── #10 area name of the target node ──

def _obj(label, ocr=""):
    return RefObject(label=label, label_norm=label, score=0.9, ocr_text=ocr, role="", grid_cell="")


def _map(*objects):
    m = RefMap(place="t")
    m.photos[1] = RefPhotoNode(nid=1, photo_file="1.jpg", pdr_x=0, pdr_y=0, heading_deg=0.0, session="s",
                               total_steps=0, total_distance_m=0.0, objects=list(objects))
    return m


def test_area_name_keeps_the_aisle_number():
    m = _map(_obj("走道標示牌", "走道 7"), _obj("泡麵貨架"))
    assert srv._node_area_name(1, m, goal_keywords=["泡麵"]) == "「走道 7（泡麵）」"


def test_area_name_uses_the_item_found_at_this_node():
    m = _map(_obj("泡麵貨架"), _obj("冷藏櫃"))
    assert srv._node_area_name(1, m, goal_keywords=["牛奶", "泡麵"]) == "「泡麵」區"


def test_area_name_prefers_the_users_own_word():
    m = _map(_obj("Instant Noodles 貨架", "泡麵"))
    assert srv._node_area_name(1, m, goal_keywords=["泡麵"], goal_item="杯麵") == "「杯麵」區"
