"""Raw feed recording as gzip JSON lines, and the reader that replay uses.

Each process start opens a new file, and files rotate at UTC midnight, so a power cut can only truncate the tail
of one file. The reader stops cleanly at a truncated tail.
"""

from __future__ import annotations

import gzip
import json
import logging
import zlib
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from btc_survivor.clock import MS_PER_DAY

log = logging.getLogger(__name__)

FLUSH_EVERY_MS = 10_000


def _file_name(ms: int) -> str:
    dt = datetime.fromtimestamp(ms // 1000, tz=timezone.utc)
    return f"{dt:%Y-%m-%d_%H%M%S}.jsonl.gz"


class Recorder:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._day: int | None = None
        self._last_flush = 0
        self.path: Path | None = None

    def write(self, kind: str, raw: str, recv_ms: int, conn_id: int) -> None:
        day = recv_ms // MS_PER_DAY
        if day != self._day:
            self._open(recv_ms, day)
        line = json.dumps({"t": recv_ms, "k": kind, "c": conn_id, "d": raw}, separators=(",", ":"))
        self._file.write(line + "\n")
        if recv_ms - self._last_flush >= FLUSH_EVERY_MS:
            self._file.flush()
            self._last_flush = recv_ms

    def _open(self, recv_ms: int, day: int) -> None:
        self.close()
        self.path = self.folder / _file_name(recv_ms)
        self._file = gzip.open(self.path, "at", encoding="utf-8")
        self._day = day
        self._last_flush = recv_ms

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def recording_files(paths: Iterable[Path]) -> list[Path]:
    """Expand folders into their recording files and sort everything by name, which is time order."""
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        files.extend(sorted(path.glob("*.jsonl.gz")) if path.is_dir() else [path])
    return sorted(files, key=lambda p: p.name)


def read_recording(paths: Iterable[Path]) -> Iterator[tuple[str, str, int, int]]:
    """Yield (kind, raw, recv_ms, conn_id) in recorded order. A truncated or corrupt tail ends that file early."""
    for path in recording_files(paths):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("%s: skipping a partial last line", path.name)
                        break
                    yield row["k"], row["d"], row["t"], row["c"]
        except (EOFError, zlib.error, gzip.BadGzipFile, OSError) as exc:
            log.warning("%s: recording ends early (%s)", path.name, exc)
