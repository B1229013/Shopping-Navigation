"""Entry point: launch uvicorn for the FastAPI app."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import uvicorn

from server.config import OUTPUT_ROOT, SERVER_HOST, SERVER_PORT

LOG_PATH = OUTPUT_ROOT.parent / "server.log"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(log_path: Path = LOG_PATH) -> None:
    """Console as before, plus a rotating copy on disk (5 × 20 MB). The store runs
    happen while the terminal scrolls away; the per-photo localization/rerank
    lines are what every later "why did it say that?" question needs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    logging.getLogger(__name__).info("Log file: %s", log_path)


def main() -> None:
    configure_logging()
    uvicorn.run("server.server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)


if __name__ == "__main__":
    main()
