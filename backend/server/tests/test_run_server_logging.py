"""The server must keep a copy of its log on disk: the field runs happen on a
phone while the terminal scrolls away, and every "why did it say that?" question
needs the per-photo rerank/localization lines afterwards."""
import logging

from server import run_server


def test_configure_logging_writes_to_file_and_console(tmp_path):
    log_path = tmp_path / "server.log"
    run_server.configure_logging(log_path)
    try:
        logging.getLogger("server.test").info("rerank triggered: WP1")
        for h in logging.getLogger().handlers:
            h.flush()
        assert "rerank triggered: WP1" in log_path.read_text(encoding="utf-8")
        kinds = {type(h).__name__ for h in logging.getLogger().handlers}
        assert "StreamHandler" in kinds                # terminal still gets it
        assert any("File" in k for k in kinds)        # and the file
    finally:
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)
            h.close()
