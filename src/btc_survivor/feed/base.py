"""The exchange interface and the live websocket client.

An adapter turns raw messages into two kinds of output on one ordered stream, trades and order book updates,
plus notices. Adding an exchange means adding one module in this package with a create_adapter function.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import websockets
from websockets.asyncio.client import connect

from btc_survivor.clock import LiveClock
from btc_survivor.config import FeedConfig
from btc_survivor.types import FeedEvent, FeedNotice

log = logging.getLogger(__name__)

# Recorded line kinds. Replay feeds the same kinds back through decode.
OPEN, MESSAGE, CLOSE = "open", "message", "close"


class FeedAdapter(ABC):
    url: str
    max_message_bytes: int

    @abstractmethod
    def subscribe_messages(self) -> list[str]:
        """Text frames to send right after connecting."""

    @abstractmethod
    def on_open(self, conn_id: int) -> None:
        """Reset per connection parse state, such as the expected sequence number."""

    @abstractmethod
    def parse(self, raw: str, recv_ms: int, conn_id: int) -> list[FeedEvent]:
        """Turn one raw message into events. Depends only on the text, the receive time and parse state."""


def load_adapter(feed: FeedConfig) -> FeedAdapter:
    module = importlib.import_module(f"btc_survivor.feed.{feed.exchange}")
    return module.create_adapter(feed)


def decode(adapter: FeedAdapter, kind: str, raw: str, recv_ms: int, conn_id: int) -> list[FeedEvent]:
    """Turn one recorded or live line into events, the same way in live and replay."""
    if kind == OPEN:
        adapter.on_open(conn_id)
        return [FeedNotice(recv_ms, conn_id, "connected")]
    if kind == CLOSE:
        return [FeedNotice(recv_ms, conn_id, "disconnected", detail=raw)]
    return adapter.parse(raw, recv_ms, conn_id)


Sink = Callable[[str, str, int, int], None]


class FeedClient:
    """Keeps one websocket open, stamps every message on arrival and hands it to sink as (kind, raw, recv_ms, conn_id).

    Reconnects with exponential backoff and jitter. The backoff only resets once the caller reports Healthy,
    so a feed that keeps failing right after connecting backs off instead of hammering the exchange.
    """

    def __init__(self, adapter: FeedAdapter, feed: FeedConfig, sink: Sink, rng: np.random.Generator,
                 clock: LiveClock | None = None, floor_ms: Callable[[], int] = lambda: 0) -> None:
        self.adapter = adapter
        self.feed = feed
        self.sink = sink
        self.rng = rng
        self.clock = clock or LiveClock()
        self.floor_ms = floor_ms
        self.conn_id = 0
        self.attempt = 0
        self._ws = None
        self._resync: str | None = None

    def _stamp(self) -> int:
        return self.clock.stamp(self.floor_ms())

    def request_resync(self, reason: str) -> None:
        """Close the current connection so the loop reconnects and receives a fresh snapshot.

        With no connection open there is nothing to resync, because the next connection starts fresh anyway.
        """
        if self._ws is None:
            return
        self._resync = reason
        asyncio.get_running_loop().create_task(self._ws.close())

    def mark_healthy(self) -> None:
        self.attempt = 0

    def backoff_seconds(self) -> float:
        base = min(self.feed.reconnect_backoff_max_seconds,
                   self.feed.reconnect_backoff_start_seconds * 2 ** min(self.attempt, 30))
        self.attempt += 1
        return float(self.rng.uniform(base / 2, base))

    async def run(self, stop: asyncio.Event) -> None:
        """Run until stop is set. Setting stop also closes an open socket, so shutdown never waits on a silent feed."""
        closer = asyncio.get_running_loop().create_task(self._close_on(stop))
        try:
            await self._loop(stop)
        finally:
            closer.cancel()

    async def _close_on(self, stop: asyncio.Event) -> None:
        await stop.wait()
        if self._ws is not None:
            await self._ws.close()

    async def _loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            self.conn_id += 1
            conn_id = self.conn_id
            opened = False
            detail = ""
            try:
                async with connect(self.adapter.url, max_size=self.adapter.max_message_bytes, open_timeout=10,
                                   ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                    self._ws = ws
                    self._resync = None
                    opened = True
                    self.sink(OPEN, "", self._stamp(), conn_id)
                    for message in self.adapter.subscribe_messages():
                        await ws.send(message)
                    async for raw in ws:
                        text = raw if isinstance(raw, str) else raw.decode("utf-8")
                        self.sink(MESSAGE, text, self._stamp(), conn_id)
                        if self._resync is not None or stop.is_set():
                            break
            except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
                detail = f"{type(exc).__name__}: {exc}"
            finally:
                self._ws = None
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
