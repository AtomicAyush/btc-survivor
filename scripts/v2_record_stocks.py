"""Record the raw Finnhub stock feed for the tape check and replay of PLAN_V2 Phase 13. It never trades, never writes a
database and never prints or logs a price, a size or a message body.

It reads the token only through local_secrets, installs redact before anything logs, refuses to start while a proxy
or certificate variable is set, takes the Finnhub websocket lock and keeps one direct connection open around the
clock, reconnecting with the backoff and the rules of section 4.6. Every raw message goes to gzip JSON lines in the
config's raw_dir with its receive stamp and connection id, beside open and close markers with their stamps and
reasons. Files rotate at each start and at UTC midnight and are flushed every second, and everything it writes is
readable by the owner alone. A heartbeat line each minute holds only counts, the connection state and the file size,
in data/logs/v2/record_stocks for the committed configs, outside data/v2. SIGTERM or Ctrl+C stops it cleanly with
exit code 0, an expected failure such as a held lock or a failed write exits with code 2, and a crash with code 1.

What the recording means for the tape check. Connection ids restart at 1 at each start and skip any id a refused
handshake used. A connection ends at its close marker, or after a kill or power cut at its last line before the next
open or the end of the files. A file started at UTC midnight continues the open connection with no open marker. A
process frozen for more than 10 seconds, by sleep or SIGSTOP, closes its connection with the reason "process paused
for N s" as soon as it runs again, so a pause always shows as a close and an open.

    python scripts/v2_record_stocks.py configs/v2/s1_prices.toml [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import signal
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, LiveClock
from btc_survivor.feed.base import MESSAGE, OPEN, decode
from btc_survivor.logsetup import setup_logging
from btc_survivor.seeds import reconnect_rng
from btc_survivor.v2 import output, redact, sessions
from btc_survivor.v2.config import ConfigError, StockConfig, load_stock_config
from btc_survivor.v2.local_secrets import FINNHUB, REFUSED, SecretsError, load_secrets
from btc_survivor.v2.stocks.finnhub import URL, FinnhubAdapter, PrintBatch
from btc_survivor.v2.stocks.runner import (BackoffSettings, LockError, RecordingError, StockFeedClient, StockRecorder,
                                           websocket_lock)

log = logging.getLogger("v2_record_stocks")

TICK_SECONDS = 1.0
PAUSE_MS = 10 * MS_PER_SECOND
PRE_OPEN_CHECK_MS = 10 * MS_PER_MINUTE
PRE_OPEN_SILENCE_MS = 5 * MS_PER_MINUTE
RESUBSCRIBE_AFTER_MS = 120 * MS_PER_SECOND
RESUBSCRIBE_EVERY_MS = 5 * MS_PER_MINUTE
COUNTS = ("messages", "bytes", "prints", "pings", "errors", "decode_errors", "pauses", "connects", "disconnects")
LOCAL_URL = re.compile(r"ws://127\.0\.0\.1:\d{1,5}")


def local_url(text: str) -> str:
    """Accept only a plain websocket address on 127.0.0.1, so a test flag can never send the token to another host."""
    if not LOCAL_URL.fullmatch(text):
        raise argparse.ArgumentTypeError("only ws://127.0.0.1:PORT is accepted, for a local fake server in tests")
    return text


def positive(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("must be above zero")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record the raw Finnhub stock feed. It prints and logs only counts, "
                                                 "states and file sizes, never a price, a size or a message body.")
    parser.add_argument("config", type=Path, help="the stock run config, such as configs/v2/s1_prices.toml")
    parser.add_argument("--dry-run", action="store_true", help="load the config and secrets, take and release the "
                                                               "lock, build the adapter and print a verdict, "
                                                               "without connecting")
    parser.add_argument("--log-dir", type=Path, help="the log folder, by default logs/v2/record_stocks two folders "
                                                     "above raw_dir, so data/logs/v2/record_stocks")
    parser.add_argument("--heartbeat-seconds", type=positive, default=60.0, help="seconds between heartbeat lines")
    parser.add_argument("--test-url", type=local_url, help="a local fake server for tests, only ws://127.0.0.1:PORT")
    return parser


def default_log_dir(cfg: StockConfig) -> Path:
    """logs/v2/record_stocks two folders above raw_dir, so under the gitignored data folder but outside data/v2,
    which holds the recordings that no assistant session reads."""
    return cfg.raw_dir.parent.parent / "logs" / "v2" / "record_stocks"


def refused_variables(environ: Mapping[str, str]) -> list[str]:
    """The proxy and certificate variables that are set, in any case, which could route the token through another
    host or trust a certificate authority the owner did not choose."""
    return sorted(name for name, value in environ.items() if name.upper() in REFUSED and value)


@dataclass
class Watched:
    """What the silence and resubscribe rules need about the open connection, and the open of the session whose pre
    open check already ran."""

    last_ms: int = 0
    printed: set[str] = field(default_factory=set)
    resubscribed: dict[str, int] = field(default_factory=dict)
    pre_open_checked: int = 0


def silence_reason(now_ms: int, last_ms: int, stale_ms: int, watched: Watched) -> str | None:
    """Why the open connection should be replaced for silence, or None. In session after stale_ms without a message,
    counted from the later of the last message and the open, and once per session, at the first check in the ten
    minutes before the open, after five silent minutes."""
    session = sessions.next_session(now_ms)
    if now_ms >= session.open_ms:
        if now_ms - max(last_ms, session.open_ms) >= stale_ms:
            return f"no message for {stale_ms / MS_PER_SECOND:g} s in session"
    elif now_ms >= session.open_ms - PRE_OPEN_CHECK_MS and watched.pre_open_checked != session.open_ms:
        watched.pre_open_checked = session.open_ms
        if now_ms - last_ms >= PRE_OPEN_SILENCE_MS:
            return "silent for 5 minutes before the open"
    return None


def quiet_symbols(now_ms: int, connected_ms: int, symbols: tuple[str, ...], printed: set[str],
                  resubscribed: dict[str, int]) -> list[str]:
    """Symbols to subscribe again, those with no print on this connection 120 seconds into the session while others
    print, each at most once every 5 minutes. A symbol that printed and went quiet is halted, not resubscribed."""
    session = sessions.session_at(now_ms)
    if session is None or not printed or now_ms - max(session.open_ms, connected_ms) < RESUBSCRIBE_AFTER_MS:
        return []
    return [s for s in symbols if s not in printed
            and (s not in resubscribed or now_ms - resubscribed[s] >= RESUBSCRIBE_EVERY_MS)]


def heartbeat_line(counts: Counter[str], symbols: int, state: str, conn_id: int, adapter: FinnhubAdapter,
                   recorder: StockRecorder) -> str:
    """One line of counts, the connection state and the file size, never a price, a size or a symbol's volume."""
    fields = [f"feed={state}", f"connection={conn_id}", *(f"{name}={counts[name]}" for name in COUNTS),
              f"symbols_printing={symbols}", f"dropped={sum(adapter.dropped.values())}",
              f"unknown={sum(adapter.unknown.values())}",
              f"file={recorder.path.name if recorder.path else 'none'}", f"file_bytes={recorder.size_bytes}"]
    return " ".join(fields)


async def record(cfg: StockConfig, adapter: FinnhubAdapter, recorder: StockRecorder, heartbeat_ms: int,
                 clock: LiveClock | None = None, stop: asyncio.Event | None = None) -> Counter[str]:
    """Record until stop is set, by default by SIGTERM or SIGINT, and return the total counts. Tests pass their own
    clock and stop."""
    loop = asyncio.get_running_loop()
    redact.install_asyncio_handler(loop)
    if stop is None:
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
    clock = clock or LiveClock()
    feed = cfg.stock_feed
    stale_ms = int(feed.feed_stale_after_seconds * MS_PER_SECOND)
    healthy_ms = int(feed.mark_healthy_outside_session_seconds * MS_PER_SECOND)
    interval: Counter[str] = Counter()
    total: Counter[str] = Counter()
    printing: set[str] = set()
    watched = Watched()
    failures: set[str] = set()

    def sink(kind: str, raw: str, recv_ms: int, conn_id: int) -> None:
        recorder.write(kind, raw, recv_ms, conn_id)
        if kind == MESSAGE:
            interval["messages"] += 1
            interval["bytes"] += len(raw.encode("utf-8"))
        if kind in (MESSAGE, OPEN):
            watched.last_ms = recv_ms
        if kind == OPEN:
            failures.clear()
        try:
            events = decode(adapter, kind, raw, recv_ms, conn_id)
        except Exception as exc:  # the line is recorded already, so a decode bug costs counts and never the recording
            interval["decode_errors"] += 1
            if type(exc).__name__ not in failures:
                failures.add(type(exc).__name__)
                log.error("could not decode a line on connection %d (%s), counted from now on", conn_id,
                          type(exc).__name__)
            return
        for event in events:
            if isinstance(event, PrintBatch):
                interval["prints"] += len(event.prints)
                printing.update(p.symbol for p in event.prints)
                watched.printed.update(p.symbol for p in event.prints)
            elif event.kind == "ping":
                interval["pings"] += 1
            elif event.kind == "error":
                interval["errors"] += 1
                log.warning("Finnhub sent an error message on connection %d, reconnecting", conn_id)
                client.request_resync("finnhub error")
            elif event.kind == "connected":
                interval["connects"] += 1
                watched.printed = set()
            elif event.kind == "disconnected":
                interval["disconnects"] += 1

    client = StockFeedClient(adapter, BackoffSettings.from_config(feed), redact.redacting_sink(sink),
                             reconnect_rng(cfg.run.master_seed), clock)

    async def watch(now: int) -> None:
        """Apply the silence and resubscribe rules to the open connection."""
        if client.connected_ms is None:
            return
        reason = silence_reason(now, max(watched.last_ms, client.connected_ms), stale_ms, watched)
        if reason is not None:
            log.warning("feed connection %d: %s, reconnecting", client.conn_id, reason)
            client.request_resync(reason)
            return
        for symbol in quiet_symbols(now, client.connected_ms, adapter.symbols, watched.printed, watched.resubscribed):
            watched.resubscribed[symbol] = now
            if await client.send(adapter.subscribe_frame(symbol)):
                log.info("subscribed again to %s, which has not printed on connection %d", symbol, client.conn_id)

    def beat() -> None:
        state = "connected" if client.connected_ms is not None else "disconnected"
        log.info("heartbeat %s", heartbeat_line(interval, len(printing), state, client.conn_id, adapter, recorder))
        total.update(interval)
        interval.clear()
        printing.clear()

    async def ticker() -> None:
        """Every second flush the file, replace a connection the process slept through, reset the backoff, apply the
        rules and write the heartbeat when due."""
        last = clock.stamp()
        next_beat = (last // heartbeat_ms + 1) * heartbeat_ms
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), TICK_SECONDS)
            except asyncio.TimeoutError:
                pass
            now = clock.stamp()
            recorder.flush()
            if now - last > PAUSE_MS:
                interval["pauses"] += 1
                log.warning("the process was paused for %d s, such as by sleep", (now - last) // MS_PER_SECOND)
                client.request_resync(f"process paused for {(now - last) // MS_PER_SECOND} s")
            last = now
            client.mark_healthy_if_open(now, healthy_ms)
            if not stop.is_set():
                await watch(now)
            if now >= next_beat:
                beat()
                next_beat = (now // heartbeat_ms + 1) * heartbeat_ms

    log.info("recording %d symbols into %s", len(adapter.symbols), recorder.folder)
    await asyncio.gather(client.run(stop), ticker())
    total.update(interval)
    return total


def dry_run_report(adapter: FinnhubAdapter) -> output.Report:
    return output.Report("record stocks dry run", (
        ("verdict", True),
        ("symbols", output.Count(len(adapter.symbols))),
        ("subscribe frames", output.Count(len(adapter.subscribe_messages()))),
        ("lock", output.Word("taken and released")),
    ))


def main(argv: list[str] | None = None) -> int:
    """Record, or with --dry-run check the setup, printing an expected failure as one error line with exit code 2."""
    args = build_parser().parse_args(argv)
    os.umask(0o077)  # recordings, logs and their folders are for the owner alone
    try:
        cfg = load_stock_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not cfg.stock_feed.record_raw_messages:
        print("error: [stock_feed] record_raw_messages is false, so there is nothing to record", file=sys.stderr)
        return 2
    setup_logging(cfg.run.log_level, None if args.dry_run else args.log_dir or default_log_dir(cfg))
    redact.install()
    lock_file = cfg.path(cfg.stock_feed.lock_file)
    with ExitStack() as held:
        try:
            refused = refused_variables(os.environ)
            if refused:
                raise ValueError(f"unset {', '.join(refused)} first, because the token goes only straight to Finnhub, "
                                 f"checked against the system's own certificate authorities")
            sessions.self_test()
            sessions.next_session(LiveClock().now_ms())  # the silence rules need a working calendar for today
            adapter = FinnhubAdapter(cfg.stock_feed.symbols, load_secrets(cfg.path(cfg.run.secrets_file)).get(FINNHUB),
                                     args.test_url or URL)
            held.enter_context(websocket_lock(lock_file))
        except (SecretsError, LockError, ValueError, OSError) as exc:
            print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
            return 2
        if args.dry_run:
            held.close()
            output.emit(dry_run_report(adapter))
            return 0
        recorder = StockRecorder(cfg.raw_dir)
        log.info("holding the websocket lock %s", lock_file)
        try:
            total = asyncio.run(record(cfg, adapter, recorder, int(args.heartbeat_seconds * MS_PER_SECOND)))
        except RecordingError as exc:
            log.error("%s, so the recorder stops", exc)
            return 2
        finally:
            with suppress(OSError):  # a file that cannot be closed only loses its trailer
                recorder.close()
    log.info("stopped, %s dropped=%d unknown=%d, last file %s", " ".join(f"{n}={total[n]}" for n in COUNTS),
             sum(adapter.dropped.values()), sum(adapter.unknown.values()),
             recorder.path.name if recorder.path else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
