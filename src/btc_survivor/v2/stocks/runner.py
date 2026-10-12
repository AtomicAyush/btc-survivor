"""The stock process runner of PLAN_V2 section 14.2. Phase 13 builds its feed client, the websocket lock and the
raw recorder, which scripts/v2_record_stocks.py uses before the stock engine exists.

StockFeedClient reuses the version 1 FeedClient, with its backoff, health reset and resync, and takes BackoffSettings
in place of a version 1 FeedConfig, so feed/base.py stays untouched. Its own connection loop adds what version 2
needs, a send method for resubscribe frames, the stamp of the open connection, close reasons and log lines passed
through redact, one ERROR line an hour while Finnhub refuses the handshake with HTTP 401, and a direct connection
that never goes through a proxy. A failed write to the recording raises RecordingError, which ends the process
instead of passing for a lost connection.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import websockets
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from btc_survivor.clock import MS_PER_HOUR, LiveClock
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, FeedAdapter, FeedClient, Sink
from btc_survivor.recorder import Recorder
from btc_survivor.v2.config import StockFeedConfig
from btc_survivor.v2.redact import redact

log = logging.getLogger(__name__)

PING_INTERVAL_SECONDS = 20.0
PING_TIMEOUT_SECONDS = 20.0
OPEN_TIMEOUT_SECONDS = 10.0
CLOSE_TIMEOUT_SECONDS = 5.0
UNAUTHORIZED = 401


class LockError(RuntimeError):
    """Raised when another process holds the Finnhub websocket lock."""


@contextmanager
def websocket_lock(path: Path) -> Iterator[Path]:
    """Hold an exclusive, non blocking flock on path for the block, so one process alone owns the websocket the key
    allows. The file names the holder's process id while it holds the lock, for the error that a second process gets,
    and is emptied on release, so an old id never points at an unrelated process."""
    path = Path(path).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.pread(fd, 32, 0).decode("ascii", "replace").strip()
            holder = holder if holder.isdigit() else "unknown"
            raise LockError(f"another process, id {holder}, holds the Finnhub websocket lock {path}, and the key "
                            f"allows one connection, so this process exits") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()}\n".encode("ascii"), 0)
        try:
            yield path
        finally:
            os.ftruncate(fd, 0)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class BackoffSettings:
    """The two backoff fields the reused FeedClient reads, in place of a version 1 FeedConfig."""

    reconnect_backoff_start_seconds: float
    reconnect_backoff_max_seconds: float

    @classmethod
    def from_config(cls, feed: StockFeedConfig) -> BackoffSettings:
        return cls(feed.reconnect_backoff_start_seconds, feed.reconnect_backoff_max_seconds)


class StockFeedClient(FeedClient):
    """Keeps the one Finnhub websocket open, stamps every message on arrival and hands it to sink, resubscribing after
    each connect and pinging every 20 seconds with a 20 second timeout."""

    ping_interval = PING_INTERVAL_SECONDS
    ping_timeout = PING_TIMEOUT_SECONDS
    close_timeout = CLOSE_TIMEOUT_SECONDS

    def __init__(self, adapter: FeedAdapter, backoff: BackoffSettings, sink: Sink, rng: np.random.Generator,
                 clock: LiveClock | None = None, floor_ms: Callable[[], int] = lambda: 0) -> None:
        super().__init__(adapter, backoff, sink, rng, clock, floor_ms)  # type: ignore[arg-type]
        self.connected_ms: int | None = None
        self._refused_logged_ms: int | None = None
        # The library's debug lines show the handshake request, whose path holds the token.
        logging.getLogger("websockets").setLevel(logging.WARNING)

    def request_resync(self, reason: str) -> None:
        """Close the open connection once, keeping the first reason, and stop counting it as open, so the silence, error
        and pause rules act once per connection even while a close to a dead peer takes seconds."""
        if self._ws is None or self._resync is not None:
            return
        super().request_resync(reason)
        self.connected_ms = None

    async def send(self, text: str) -> bool:
        """Send one text frame on the open connection, such as a resubscribe, and say whether it went out."""
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(text)
        except websockets.exceptions.ConnectionClosed:
            return False
        return True

    def mark_healthy_if_open(self, now_ms: int, after_ms: int) -> bool:
        """Reset the backoff once the open connection has lasted after_ms, which the caller uses where price health
        cannot judge, outside the session or in the recorder."""
        if self.connected_ms is None or now_ms - self.connected_ms < after_ms or self.attempt == 0:
            return False
        self.mark_healthy()
        return True

    async def _loop(self, stop: asyncio.Event) -> None:
        """The version 1 loop plus the open stamp, redacted close reasons, the 401 rule, no proxy and a stop that
        arrives during the handshake, which ends the connection before any marker is written."""
        while not stop.is_set():
            self.conn_id += 1
            conn_id = self.conn_id
            opened = False
            detail = ""
            try:
                async with connect(self.adapter.url, max_size=self.adapter.max_message_bytes,
                                   open_timeout=OPEN_TIMEOUT_SECONDS, ping_interval=self.ping_interval,
                                   ping_timeout=self.ping_timeout, close_timeout=self.close_timeout,
                                   proxy=None) as ws:
                    if not stop.is_set():
                        opened = True
                        await self._receive(ws, conn_id, stop)
            except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
                detail = redact(f"{type(exc).__name__}: {exc}")
                if isinstance(exc, InvalidStatus) and exc.response.status_code == UNAUTHORIZED:
                    self._log_refused()
            finally:
                self._ws = None
                self.connected_ms = None
            reason = self._resync or detail or ("shutdown" if stop.is_set() else "closed by server")
            self._resync = None
            if opened:
                self.sink(CLOSE, reason, self._stamp(), conn_id)
            if stop.is_set():
                break
            delay = self.backoff_seconds()
            log.info("feed connection %d ended (%s), reconnecting in %.1f s", conn_id, reason, delay)
            try:
                await asyncio.wait_for(stop.wait(), delay)
            except asyncio.TimeoutError:
                pass

    async def _receive(self, ws: ClientConnection, conn_id: int, stop: asyncio.Event) -> None:
        """Mark the connection open, send every subscribe frame and hand each message to the sink until it ends."""
        self._ws = ws
        self._resync = None
        self.connected_ms = self._stamp()
        self.sink(OPEN, "", self.connected_ms, conn_id)
        log.info("feed connection %d open", conn_id)
        for message in self.adapter.subscribe_messages():
            await ws.send(message)
        async for raw in ws:
            text = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
            self.sink(MESSAGE, text, self._stamp(), conn_id)
            if self._resync is not None or stop.is_set():
                break

    def _log_refused(self) -> None:
        """Log one ERROR line an hour while the handshake is refused with HTTP 401, which the backoff keeps retrying."""
        now = self.clock.now_ms()
        if self._refused_logged_ms is None or now - self._refused_logged_ms >= MS_PER_HOUR:
            self._refused_logged_ms = now
            log.error("Finnhub refused the websocket handshake with HTTP 401, so the key may be wrong or revoked, "
                      "retrying with backoff")


class RecordingError(RuntimeError):
    """Raised when the recording cannot be written, such as on a full disk. It is not an OSError, so the feed loop
    never mistakes it for a lost connection and the process stops."""


class StockRecorder(Recorder):
    """The version 1 raw recorder with a flush the ticker calls every second, so a kill loses about a second at most,
    and write errors raised as RecordingError."""

    def write(self, kind: str, raw: str, recv_ms: int, conn_id: int) -> None:
        try:
            super().write(kind, raw, recv_ms, conn_id)
        except OSError as exc:
            raise RecordingError(f"writing the recording failed with {type(exc).__name__}") from exc

    def flush(self) -> None:
        try:
            if self._file is not None:
                self._file.flush()
        except OSError as exc:
            raise RecordingError(f"flushing the recording failed with {type(exc).__name__}") from exc

    @property
    def size_bytes(self) -> int:
        """The current file's size on disk, compressed, or 0 before the first line."""
        return self.path.stat().st_size if self.path is not None and self.path.exists() else 0
