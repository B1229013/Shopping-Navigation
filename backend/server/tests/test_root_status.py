"""Opening the server address in a browser must show its status, not a 404 —
the bare 404 reads as "server is broken" during field tests."""
from fastapi.testclient import TestClient

from server.server import app


def test_root_reports_status_not_404():
    with TestClient(app) as c:
        r = c.get("/")
    assert r.status_code == 200
    body = r.text
    assert "APPNAV" in body
    assert "/health" in body           # points at the machine-readable check
