"""The live program: one asyncio loop with the feed, a one second ticker, a heartbeat and clean shutdown.

All bot and ledger writes happen inside engine calls. This module only moves time forward, records raw messages,
writes runtime_status and the heartbeat line once a minute, and on SIGINT or SIGTERM lets pending fills settle,
writes a final checkpoint and exits with code 0.
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import signal
import statistics
import sys

from btc_survivor import records
from btc_survivor.bot import DEAD
from btc_survivor.clock import LiveClock, iso, minute_index
from btc_survivor.config import Config
from btc_survivor.engine import PERSIST_OFFSET_MS, Engine
from btc_survivor.feed.base import FeedAdapter, FeedClient, load_adapter
from btc_survivor.recorder import Recorder
from btc_survivor.seeds import reconnect_rng
from btc_survivor.store import Store

log = logging.getLogger(__name__)

TICK_SECONDS = 1.0


def memory_mb() -> tuple[float | None, float]:
    """Current resident memory where the system exposes it (Linux), and peak resident memory."""
    current = None
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            current = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return current, peak / 2**20 if sys.platform == "darwin" else peak / 1024


class Runner:
    def __init__(self, cfg: Config, engine: Engine, clock: LiveClock) -> None:
        self.cfg, self.engine, self.clock = cfg, engine, clock
        self.recorder = Recorder(cfg.data_dir / "raw") if cfg.feed.record_raw_messages else None
        self.shutdown = asyncio.Event()
        self.feed_stop = asyncio.Event()
        self.client = FeedClient(engine.adapter, cfg.feed, self.sink, reconnect_rng(cfg.run.master_seed), clock,
                                 floor_ms=lambda: engine.now_ms)
        self.status_minute: int | None = None
        self.lag_s = 0.0  # the worst lateness of the one second tick this minute, which measures loop backlog

    def sink(self, kind: str, raw: str, recv_ms: int, conn_id: int) -> None:
        if self.recorder is not None:
            self.recorder.write(kind, raw, recv_ms, conn_id)
        self.engine.on_line(kind, raw, recv_ms, conn_id)
        self.after_engine()

    def after_engine(self) -> None:
        """Pass any resync request to the socket, and reset the reconnect backoff once the feed is Healthy."""
        reason = self.engine.market.take_resync()
        if reason:
            self.client.request_resync(reason)
        if self.engine.market.health.is_healthy:
            self.client.mark_healthy()

    async def ticker(self) -> None:
        """Fires minute boundaries and persistence points even when no message arrives."""
        while not self.feed_stop.is_set():
            started = self.clock.monotonic()
            try:
                await asyncio.wait_for(self.feed_stop.wait(), TICK_SECONDS)
            except asyncio.TimeoutError:
                self.lag_s = max(self.lag_s, self.clock.monotonic() - started - TICK_SECONDS)
            now = self.clock.stamp(self.engine.now_ms)
            self.engine.advance_to(now)
            self.after_engine()
            minute = minute_index(now - PERSIST_OFFSET_MS)
            if minute != self.status_minute:
                self.status_minute = minute
                self.heartbeat(now)

    async def supervisor(self) -> None:
        """After a signal, wait up to the order latency for pending fills while the feed is Healthy, then stop."""
        await self.shutdown.wait()
        log.info("shutdown requested, %d orders pending", len(self.engine.pending))
        costs = self.cfg.costs
        deadline = self.clock.monotonic() + (costs.latency_ms + costs.latency_jitter_ms) / 1000
        while self.engine.pending and self.engine.market.health.is_healthy and self.clock.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self.feed_stop.set()

    def heartbeat(self, now: int, state: str | None = None) -> None:
        """One log line and one runtime_status row, also during stale periods."""
        engine, market = self.engine, self.engine.market
        book, health = market.book, market.health
        equities = [b.equity for b in engine.population.learners if b.state != DEAD]
        current, peak = memory_mb()
        age = None if health.last_msg_ms is None else (now - health.last_msg_ms) / 1000
        log.info("heartbeat feed=%s living=%d equity mean=%.2f min=%.2f max=%.2f deaths=%d rss=%s peak=%.1f MB "
                 "lag=%.0f ms last_message_age=%s s", health.state, len(equities), statistics.fmean(equities),
                 min(equities), max(equities), engine.deaths, "n/a" if current is None else f"{current:.1f}", peak,
                 self.lag_s * 1000, "n/a" if age is None else f"{age:.1f}")
        self.lag_s = 0.0
        engine.store.execute_now(records.UPSERT_STATUS, (
            iso(now), state or health.state, book.best_bid, book.best_ask, book.mid if book.valid else None,
            len(equities), engine.last_checkpoint_ts, current, peak))

    async def run(self, install_signals: bool = True) -> None:
        loop = asyncio.get_running_loop()
        signals = (signal.SIGINT, signal.SIGTERM) if install_signals else ()
        for sig in signals:
            loop.add_signal_handler(sig, self.shutdown.set)
        watcher = loop.create_task(self.supervisor())
        try:
            self.heartbeat(self.engine.now_ms)
            await asyncio.gather(self.client.run(self.feed_stop), self.ticker())
            # The final checkpoint runs while the handlers are still installed, so a second signal cannot cut it.
            self.engine.finish(self.clock.stamp(self.engine.now_ms))
            self.heartbeat(self.engine.now_ms, state="stopped")
        finally:
            watcher.cancel()
            for sig in signals:
                loop.remove_signal_handler(sig)
            if self.recorder is not None:
                self.recorder.close()


async def run_live(cfg: Config, adapter: FeedAdapter | None = None, clock: LiveClock | None = None,
                   ready: asyncio.Future | None = None, install_signals: bool = True) -> int:
    """Run until SIGINT or SIGTERM, then checkpoint and return 0. ready receives the runner, for tests."""
    clock = clock or LiveClock()
    store = Store.open(cfg.db_path)
    try:
        engine = Engine.start(cfg, store, adapter or load_adapter(cfg.feed), clock.now_ms())
        runner = Runner(cfg, engine, clock)
        if ready is not None:
            ready.set_result(runner)
        log.info("running %s %s with %d learners, data in %s", cfg.feed.exchange, cfg.feed.product,
                 cfg.population.size, cfg.data_dir)
        await runner.run(install_signals)
        log.info("stopped cleanly, last checkpoint %s", engine.last_checkpoint_ts)
    finally:
        store.close()
    return 0
