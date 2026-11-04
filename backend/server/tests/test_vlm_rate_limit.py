"""VLM calls that hit a rate limit (HTTP 429): one short wait before the retry, none after the
last attempt — the photo request runs on the server's event loop, so a long sleep there
freezes every session (and four 90 s waits outlast the phone's 300 s upload timeout)."""
from unittest.mock import patch

from PIL import Image

from server import config, vlm


def _photo(tmp_path):
    p = tmp_path / "p.jpg"
    Image.new("RGB", (64, 48)).save(p)
    return str(p)


def _waits(call, error):
    with patch.object(vlm, "OPENAI_API_KEY", "k"), \
         patch.object(vlm, "_generate", side_effect=Exception(error)), \
         patch.object(vlm.time, "sleep") as sleep:
        call()
    return [c.args[0] for c in sleep.call_args_list]


def test_rate_limited_perceive_waits_once_before_the_retry(tmp_path):
    img = _photo(tmp_path)
    perceive = lambda: vlm.perceive(img, "找到：牛奶", ["牛奶"], 64, 48)
    assert _waits(perceive, "429 Client Error: Too Many Requests") == [config.VLM_RATE_LIMIT_WAIT_S]


def test_rate_limited_decide_waits_once_before_the_retry(tmp_path):
    img = _photo(tmp_path)
    decide = lambda: vlm.decide(img, "牛奶", ["牛奶"], "", "", None, None)
    assert _waits(decide, "429 Client Error: Too Many Requests") == [config.VLM_RATE_LIMIT_WAIT_S]


def test_other_errors_retry_without_waiting(tmp_path):
    img = _photo(tmp_path)
    perceive = lambda: vlm.perceive(img, "找到：牛奶", ["牛奶"], 64, 48)
    assert _waits(perceive, "500 Server Error") == []


def test_the_wait_is_short_by_default():
    assert config.VLM_RATE_LIMIT_WAIT_S <= 10
