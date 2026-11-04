"""The developer pages put session / VLM / OCR text into the page only through esc() (#17), and the
debug page gets its session id as a JS string literal, not pasted between quotes."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import server as srv

PAGES = Path(srv.__file__).parent
RAW = {
    "dashboard.html": ["${s.id}", "${s.goal_objects.slice", "${loc.reasoning", "${entry.vlm_guidance", "${data.goal_objects"],
    "debug_page.html": ["${d.label}", "${t.text}", "${reason}", "${loc.reasoning", "${name}", "${mg.scene"],
    "localize_test.html": ["${d.label}", "${t.text}", "${m.reasoning}", "${c.photo_url}", "${data.place}"],
}


@pytest.mark.parametrize("page", sorted(RAW))
def test_page_escapes_what_it_shows(page):
    html = (PAGES / page).read_text(encoding="utf-8")
    assert "function esc(" in html
    for raw in RAW[page]:
        assert raw not in html, f"{page}: {raw} goes into innerHTML unescaped"


def test_debug_page_session_id_cannot_break_out():
    r = TestClient(srv.app).get("/session/a';alert(1);'/debug-page")
    assert r.status_code == 200
    assert """const SID = "a';alert(1);'";""" in r.text
    assert "'a';alert(1);''" not in r.text
