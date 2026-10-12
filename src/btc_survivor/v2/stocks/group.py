"""The group of PLAN_V2 sections 4.8 and 14.2, one shared market layer driving the engines of both runs in stamp order.

Every live or recorded line reaches the group in stamp order. It first runs every minute boundary and persistence point
up to the line's stamp, so nothing stamped at or after a boundary is applied before that boundary closes. Lines of kind
news and news_heartbeat are dispatched here, before the version 1 decode, which would hand any other kind to the
adapter. A score line is decoded once and given to each engine in run order, and a heartbeat line goes to the shared
layer for news health. Every price line is decoded once, each event is applied once to the shared layer, and the same
frozen results reach both engines in run order, the new feed events first, then the order checks, then the fill
prints of a batch.

At a session boundary the shared layer closes the minute once and each engine steps the same MarketMinute. Persistence
points sit 30 seconds after every boundary, inside or outside the session, where each engine commits its rows and every
checkpoint interval its checkpoint, which carries the engine and a copy of the shared layer. A restart restores both
from their checkpoints, then resumes the shared layer and each engine, which rejects the orders that were in flight.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

from btc_survivor.clock import MS_PER_MINUTE, next_minute
from btc_survivor.feed.base import MESSAGE, FeedAdapter, decode
from btc_survivor.types import FeedEvent, FeedNotice
from btc_survivor.v2.stocks import checkpoint, records
from btc_survivor.v2.stocks.engine import PERSIST_OFFSET_MS, StockEngine
from btc_survivor.v2.stocks.finnhub import PrintBatch
from btc_survivor.v2.stocks.health import ERROR
from btc_survivor.v2.stocks.market import MarketLayer
from btc_survivor.v2.stocks.news_inputs import HEARTBEAT_LINE, NEWS_LINE, HeartbeatLine, ScoreLine

log = logging.getLogger(__name__)

# Counts the group keeps for the heartbeat log line, never a price or a size.
MESSAGES, PRINTS, NEWS_LINES, HEARTBEAT_LINES, UNREADABLE = "messages", "prints", "news", "heartbeats", "unreadable"
UNREADABLE_ERRORS = (ValueError, KeyError, TypeError, AttributeError)


class StockGroup:
    """One market layer and the engines of the runs it feeds, S1 then S2, moved forward only by line stamps and the
    ticker, never by the wall clock."""

    def __init__(self, market: MarketLayer, engines: Sequence[StockEngine], adapter: FeedAdapter,
                 now_ms: int) -> None:
        if not engines:
            raise ValueError("a group needs at least one engine")
        self.market = market
        self.engines = tuple(engines)
        self.adapter = adapter
        self.now_ms = now_ms
        self.next_boundary = next_minute(now_ms)
        self.next_persist = self.next_boundary + PERSIST_OFFSET_MS
        self.counts: Counter[str] = Counter()
        self._unreadable_logged: set[str] = set()
        for engine in self.engines:
            engine.checkpointer = self._checkpoint

    def _checkpoint(self, engine: StockEngine, at: int) -> tuple[str, bytes]:
        """The checkpoint of one engine with its copy of the shared layer, which resume compares across the runs."""
        return checkpoint.encode(self.market, engine, at)

    # Lines

    def on_line(self, kind: str, raw: str, recv_ms: int, conn_id: int) -> list[FeedEvent]:
        """One recorded or live line, returning the feed events it decoded to, none for a news line. An unreadable
        price line counts as a feed error, which asks for a reconnect, and an unreadable news line is counted."""
        self.advance_to(recv_ms)
        if kind in (NEWS_LINE, HEARTBEAT_LINE):
            self._news_line(kind, raw, recv_ms)
            return []
        try:
            events = decode(self.adapter, kind, raw, recv_ms, conn_id)
        except UNREADABLE_ERRORS as exc:
            self._unreadable(kind, exc)
            events = [FeedNotice(recv_ms, conn_id, ERROR, detail=f"unreadable message, {type(exc).__name__}")]
        if kind == MESSAGE:
            self.counts[MESSAGES] += 1
        for event in events:
            self.on_event(event)
        return events

    def _news_line(self, kind: str, raw: str, recv_ms: int) -> None:
        """Decode a news line once, a score line for every engine and a heartbeat line for the shared layer."""
        try:
            line = ScoreLine.decode(raw, recv_ms) if kind == NEWS_LINE else HeartbeatLine.decode(raw, recv_ms)
        except UNREADABLE_ERRORS as exc:
            self._unreadable(kind, exc)
            return
        if isinstance(line, ScoreLine):
            self.on_news(line)
        else:
            self.on_heartbeat(line)

    def _unreadable(self, kind: str, exc: Exception) -> None:
        """Count a line that cannot be decoded, logging its kind and error type once and never its text."""
        self.counts[UNREADABLE] += 1
        key = f"{kind} {type(exc).__name__}"
        if key not in self._unreadable_logged:
            self._unreadable_logged.add(key)
            log.error("a %s line could not be decoded (%s), counted from now on", kind, type(exc).__name__)

    def on_event(self, ev: FeedEvent) -> None:
        """Apply one decoded feed event to the shared layer, pass its feed events and order checks to every engine,
        then the fill prints of a batch."""
        self.advance_to(ev.recv_ms)
        prints = self.market.on_event(ev)
        self._settle(ev.recv_ms)
        if isinstance(ev, PrintBatch):
            self.counts[PRINTS] += len(ev.prints)
        if prints:
            for engine in self.engines:
                engine.on_prints(ev.recv_ms, prints)

    def on_news(self, line: ScoreLine) -> None:
        """One score line for every engine, each skipping a sequence number it already read."""
        self.advance_to(line.recv_ms)
        self.counts[NEWS_LINES] += 1
        for engine in self.engines:
            engine.on_news(line)

    def on_heartbeat(self, line: HeartbeatLine) -> None:
        """One heartbeat line for the news health of the minute it was received in."""
        self.advance_to(line.recv_ms)
        self.counts[HEARTBEAT_LINES] += 1
        self.market.on_heartbeat(line)

    # Time

    def advance_to(self, t: int) -> None:
        """Run every boundary and persistence point at or before t in time order, then every silence, close and open
        due by t, rejecting the orders they make unfillable."""
        t = max(t, self.now_ms)
        while True:
            if self.next_boundary <= self.next_persist:
                if self.next_boundary > t:
                    break
                self._boundary(self.next_boundary)
                self.next_boundary += MS_PER_MINUTE
            else:
                if self.next_persist > t:
                    break
                for engine in self.engines:
                    engine.persistence_point(self.next_persist)
                self.next_persist += MS_PER_MINUTE
        self.now_ms = t
        self.market.advance(t)
        self._settle(t)

    def _boundary(self, boundary_ms: int) -> None:
        """Close a session minute once in the shared layer and step every engine on it."""
        minute = self.market.close_minute(boundary_ms)
        if minute is not None:
            events = self.market.drain_events()
            for engine in self.engines:
                engine.on_events(events)
                engine.on_minute(minute)
        self._settle(boundary_ms)

    def _settle(self, now_ms: int) -> None:
        """Hand the shared layer's new feed events to every engine and reject the orders that can no longer fill."""
        events = self.market.drain_events()
        for engine in self.engines:
            engine.on_events(events)
            engine.check_orders(now_ms, self.market)

    # Process events

    def settling(self) -> bool:
        """Whether any order is pending on a symbol where it may still fill, which a shutdown waits for."""
        return any(self.market.rejection(bot.pending_order.symbol) is None
                   for engine in self.engines for _, bot in engine.pending.values())

    def local_only(self, ms: int, passed: bool, offending_sockets: int, tunnel_processes: int) -> None:
        """Write the local only check of section 13.2 to every run database as a local_only feed event."""
        self.advance_to(ms)
        for engine in self.engines:
            engine.feed_rows.append(records.local_only_row(ms, passed, offending_sockets, tunnel_processes))

    def event(self, ms: int, kind: str, detail: str) -> None:
        """Write one whole feed event of the process to every run database, such as the lock taken at start."""
        self.advance_to(ms)
        for engine in self.engines:
            engine.feed_rows.append(records.feed_event_row(ms, kind, None, detail))

    def resume(self, now_ms: int) -> None:
        """After a restart from checkpoints, resume the shared layer and then each engine, at now_ms."""
        self.market.resume(now_ms)
        for engine in self.engines:
            engine.resume(now_ms)
        self.advance_to(now_ms)

    def persist(self) -> None:
        """Commit every engine's rows with a checkpoint at the current time."""
        for engine in self.engines:
            engine.persist(self.now_ms)

    def finish(self, t: int | None = None) -> None:
        """Process everything up to t and write a final checkpoint of every engine, at shutdown or the end of a
        replay."""
        self.advance_to(self.now_ms if t is None else t)
        self.persist()
