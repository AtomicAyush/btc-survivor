"""Applies feed events to the book, the candle builder and the health monitor, and decides when to resync.

A usable book is uncrossed and has both sides, so it always has a mid.

The engine and the development recorder both use this, so a recording made either way replays the same.
"""

from __future__ import annotations

from btc_survivor.feed.book import LocalBook
from btc_survivor.feed.candles import Candle, CandleBuilder
from btc_survivor.feed.health import HealthMonitor
from btc_survivor.types import BookDelta, BookSnapshot, FeedEvent, FeedNotice, Trade

LIFECYCLE = ("connected", "disconnected")


class Market:
    def __init__(self, depth: int, stale_after_ms: int, max_skew_ms: int) -> None:
        self.book = LocalBook(depth)
        self.candles = CandleBuilder()
        self.health = HealthMonitor(stale_after_ms, max_skew_ms)
        self._resync: str | None = None
        self._asked_this_connection = False

    def on_event(self, ev: FeedEvent) -> None:
        health = self.health
        if not (isinstance(ev, FeedNotice) and ev.kind in LIFECYCLE):
            health.on_message(ev.recv_ms, ev.exch_ms)
        if isinstance(ev, BookDelta):
            if self.book.valid:
                self.book.apply_changes(ev.changes)
                if not self.book.valid:
                    self._lose_book(ev.recv_ms, "crossed", "an update crossed the book")
                elif self.book.mid is None:
                    self._lose_book(ev.recv_ms, "one_sided", "an update emptied one side of the book")
                else:
                    health.on_book_update(ev.recv_ms)
        elif isinstance(ev, Trade):
            self.candles.add_trade(ev.price, ev.size)
        elif isinstance(ev, BookSnapshot):
            self.book.apply_snapshot(ev.bids, ev.asks)
            usable = self.book.valid and self.book.mid is not None
            problem = "crossed" if self.book.is_crossed() else "one_sided"
            if not usable:
                self.book.clear()
            health.on_snapshot(ev.recv_ms, usable, f"{len(ev.bids)} bids, {len(ev.asks)} asks", problem)
            if not usable:
                self._request(problem)
        elif ev.kind == "connected":
            self.book.clear()
            self._asked_this_connection = False
            health.on_connected(ev.recv_ms)
        elif ev.kind == "disconnected":
            self.book.clear()
            health.on_disconnected(ev.recv_ms, ev.detail)
        elif ev.kind == "gap":
            self._lose_book(ev.recv_ms, "gap", ev.detail)
        elif ev.kind in ("trade_gap", "error"):
            health.event(ev.recv_ms, ev.kind, ev.detail)
            if ev.kind == "error":
                self._request("error")

    def _lose_book(self, ms: int, kind: str, detail: str) -> None:
        self.book.clear()
        self.health.on_book_lost(ms, kind, detail)
        self._request(kind)

    def _request(self, reason: str) -> None:
        if self._resync is None and not self._asked_this_connection:
            self._resync = reason
            self._asked_this_connection = True

    def take_resync(self) -> str | None:
        """The reason a reconnect is wanted, at most once per connection. Live code closes the socket on it."""
        reason, self._resync = self._resync, None
        return reason

    def advance(self, now_ms: int) -> None:
        if self.health.check(now_ms) and self.health.connected:
            self._request("stale")

    def close_minute(self, boundary_ms: int) -> tuple[Candle, bool]:
        """Close the minute ending at boundary_ms and say whether it counts."""
        healthy = self.health.judge_minute(boundary_ms) and self.book.valid and self.book.mid is not None
        mid = self.book.mid if self.book.valid else None
        return self.candles.close(boundary_ms, mid, healthy), healthy
