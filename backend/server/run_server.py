"""Entry point: launch uvicorn for the FastAPI app."""
from __future__ import annotations

import logging

import uvicorn

from server.config import SERVER_HOST, SERVER_PORT


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run("server.server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)


if __name__ == "__main__":
    main()
