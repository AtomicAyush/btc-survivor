"""Development recorder for the live keyless feed. It never trades and never touches the database.

It runs the adapter, the book and the health monitor with the live reconnect rules, records every raw message,
and logs one line per minute with message rates and feed health, so busy hour rates and gap frequency can be
measured. Stop it with Ctrl+C.

    python scripts/record_feed.py --config config.toml [--minutes 60]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import numpy as np

from btc_survivor.clock import MS_PER_MINUTE, LiveClock, iso, next_minute
from btc_survivor.config import load_config
from btc_survivor.feed.base import MESSAGE, FeedClient, decode, load_adapter
from btc_survivor.feed.health import HEALTHY
from btc_survivor.feed.market import Market
from btc_survivor.logsetup import setup_logging
from btc_survivor.recorder import Recorder

log = logging.getLogger("record_feed")
RECONNECT_TAG = 7001


async def record(config_path: str, minutes: float | None) -> None:
    cfg = load_config(config_path)
    adapter = load_adapter(cfg.feed)
    market = Market(cfg.feed.book_depth_levels, int(cfg.feed.stale_after_seconds * 1000),
                    int(cfg.feed.max_clock_skew_seconds * 1000))
    recorder = Recorder(cfg.data_dir / "raw")
    clock = LiveClock()
    stop = asyncio.Event()
    stats = {"messages": 0, "bytes": 0, "changes": 0, "trades": 0}
    state = {"boundary": None, "minutes": 0}

    def advance_to(now: int) -> None:
        if state["boundary"] is None:
            state["boundary"] = next_minute(now)
        while now >= state["boundary"]:
            boundary = state["boundary"]
            candle, healthy = market.close_minute(boundary)  # judges staleness at the boundary itself
            if market.health.state == HEALTHY:
                client.mark_healthy()
            log.info("minute %s healthy=%s msgs=%d bytes=%d level_changes=%d trades=%d mid=%s exits_so_far=%d",
                     iso(boundary), healthy, stats["messages"], stats["bytes"], stats["changes"], stats["trades"],
                     f"{candle.mid:.2f}" if candle.mid else None, market.health.epoch)
            stats.update(messages=0, bytes=0, changes=0, trades=0)
            state["boundary"] += MS_PER_MINUTE
            state["minutes"] += 1
            if minutes is not None and state["minutes"] >= minutes:
                stop.set()
        market.advance(now)
        for ms, kind, detail in market.health.drain_events():
            log.info("feed event %s %s %s", iso(ms), kind, detail)
        reason = market.take_resync()
        if reason:
            client.request_resync(reason)

    def sink(kind: str, raw: str, recv_ms: int, conn_id: int) -> None:
        recorder.write(kind, raw, recv_ms, conn_id)
        advance_to(recv_ms)
        if kind == MESSAGE:
            stats["messages"] += 1
            stats["bytes"] += len(raw)
        for event in decode(adapter, kind, raw, recv_ms, conn_id):
            changes = getattr(event, "changes", None)
            if changes is not None:
                stats["changes"] += len(changes)
            elif hasattr(event, "trade_id"):
                stats["trades"] += 1
            market.on_event(event)
        reason = market.take_resync()
        if reason:
            client.request_resync(reason)

    rng = np.random.default_rng([cfg.run.master_seed, RECONNECT_TAG])
    client = FeedClient(adapter, cfg.feed, sink, rng, clock)

    async def ticker() -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), 1.0)
            except asyncio.TimeoutError:
                advance_to(clock.stamp())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    log.info("recording %s %s into %s", cfg.feed.exchange, cfg.feed.product, recorder.folder)
    try:
        await asyncio.gather(client.run(stop), ticker())
    finally:
        recorder.close()
        log.info("stopped, last file %s", recorder.path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the live keyless feed for replay and measurement.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--minutes", type=float, default=None, help="stop after this many minute boundaries")
    args = parser.parse_args()
    setup_logging("INFO")
    asyncio.run(record(args.config, args.minutes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
