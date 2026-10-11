"""Configure standard library logging once, with UTC timestamps."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from pathlib import Path

FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Log to stderr and, when log_dir is given, to a file rotated at UTC midnight and kept for 14 days."""
    formatter = logging.Formatter(FORMAT, DATE_FORMAT)
    formatter.converter = time.gmtime
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "btc_survivor.log", when="midnight", utc=True, backupCount=14, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    root.setLevel(level)
