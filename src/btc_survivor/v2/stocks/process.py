"""The stock process of PLAN_V2 sections 4.1, 7.1, 13.2 and 14.2, both runs in one asyncio loop on the one Finnhub
websocket, which it holds through the websocket lock of the frozen runner.

Every feed line is redacted, written to the raw recording and handed to the group, and so is every news line the
puller reads from news.db. Both take their stamps from one live clock with the group's time as floor, so recorded
order equals stamp order and a replay of the recording sees exactly what the engines saw. A one second ticker moves
the group forward, flushes the recording, applies the reconnect and resubscribe rules of section 4.6, reads news.db,
runs the local only check of section 13.2 in a worker thread ten minutes before each open and at a start inside a
session, once a UTC day deletes the recordings older than keep_raw_days as section 9.2 sets, and once a minute writes
runtime_status and symbol_status to both databases with a log line of states, counts and timings, never a price.

The news puller opens news.db read only, with mode=ro and never immutable, so it sees every later commit. It retries
an open that fails, as before the news process first creates the file, reopens on any error and whenever no new
heartbeat arrived for reopen_after_silent_seconds, and starts at the newest score at first launch, stored in meta, or
after a restart at the lower of the runs' checkpointed sequences, each engine skipping what it already read.

A start resumes both runs from their checkpoints, or launches them fresh with the launch meta and the news hashes of
news.db, and writes a lock and a code_commit event to both databases. SIGINT or SIGTERM lets pending fills settle while
their symbol is healthy, checkpoints both engines with the shared layer and exits with code 0. An expected failure,
such as a held lock or a refused resume, exits with code 2 and a crash with code 1.

    btc_survivor_v2 stocks configs/v2/s1_prices.toml configs/v2/s2_news.toml [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import ExitStack, suppress
from datetime import date, timedelta
from pathlib import Path

from btc_survivor.clock import MS_PER_DAY, MS_PER_MINUTE, MS_PER_SECOND, LiveClock, minute_index, parse_iso
from btc_survivor.feed.base import FeedAdapter
from btc_survivor.logsetup import setup_logging
from btc_survivor.runner import memory_mb
from btc_survivor.seeds import reconnect_rng
from btc_survivor.store import StoreError
from btc_survivor.types import FeedNotice
from btc_survivor.v2 import localcheck, output, redact, sessions
from btc_survivor.v2.config import ConfigError, StockConfig, load_run_pair
from btc_survivor.v2.local_secrets import FINNHUB, REFUSED, SecretsError, load_secrets
from btc_survivor.v2.localcheck import LocalCheck, LocalCheckError
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.news.db import NewsDbError
from btc_survivor.v2.sessions import CalendarError
from btc_survivor.v2.stocks import checkpoint, records
from btc_survivor.v2.stocks.checkpoint import ResumeError
from btc_survivor.v2.stocks.engine import StockEngine
from btc_survivor.v2.stocks.finnhub import URL, FinnhubAdapter
from btc_survivor.v2.stocks.group import MESSAGES, PRINTS, StockGroup
from btc_survivor.v2.stocks.health import ERROR, quiet_symbols, should_mark_healthy, silence_reason
from btc_survivor.v2.stocks.market import MarketLayer
from btc_survivor.v2.stocks.news_inputs import HEARTBEAT_LINE, NEWS_LINE, HeartbeatLine, ScoreLine
from btc_survivor.v2.stocks.runner import (BackoffSettings, LockError, RecordingError, StockFeedClient, StockRecorder,
                                           websocket_lock)
from btc_survivor.v2.stocks.store import StockStore, read_meta

log = logging.getLogger("btc_survivor.v2.stocks")

TICK_SECONDS = 1.0
PAUSE_MS = 10 * MS_PER_SECOND             # a tick this much later than the last means the process was frozen
LOCAL_CHECK_BEFORE_OPEN_MS = 10 * MS_PER_MINUTE
PULL_BATCH = 1000                         # rows of each news table read per tick at most
NEWS_CONN = 0                             # the connection id recorded with news lines
LOCK, CODE_COMMIT = "lock", "code_commit"
# The news.db meta keys of the hashes every heartbeat must carry, in the order news health compares them.
NEWS_HASHES = ("config_hash", "prompt_sha256", "schema_sha256")
LOCAL_URL = re.compile(r"ws://127\.0\.0\.1:\d{1,5}")
# The names the version 1 recorder gives its files, by the UTC day and time each was opened.
RECORDING_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})_\d{6}\.jsonl\.gz")

SCORES = """SELECT s.seq, s.item_seq, s.scorer, s.route, s.status, s.scored_at, i.source, i.kind, i.tickers
FROM item_scores s JOIN news_items i ON i.seq = s.item_seq WHERE s.seq > ? ORDER BY s.seq LIMIT ?"""
IMPACTS = "SELECT ticker, strength, is_new FROM score_impacts WHERE item_seq = ? AND scorer = ? ORDER BY ticker"
HEARTBEATS = "SELECT * FROM news_heartbeat WHERE seq > ? ORDER BY seq LIMIT ?"


# The news puller


class NewsPuller:
    """Reads new score and heartbeat rows from news.db and turns them into news lines without headline text."""

    def __init__(self, path: Path, score_seq: int, reopen_after_ms: int, heartbeat_seq: int | None = None,
                 batch: int = PULL_BATCH, opener: Callable[[Path], sqlite3.Connection] = news_db.open_reader) -> None:
        self.path = Path(path)
        self.score_seq = score_seq
        self.heartbeat_seq = heartbeat_seq
        self.reopen_after_ms = reopen_after_ms
        self.batch = batch
        self.opener = opener
        self.conn: sqlite3.Connection | None = None
        self.opens = 0
        self.failures = 0
        self.last_heartbeat_ms: int | None = None  # when the newest heartbeat line was read
        self._silent_since = 0
        self._failing = False

    def pull(self, now_ms: int) -> list[tuple[str, str]]:
        """The new news lines as (kind, text), scores first, reopening news.db after a silence of reopen_after_ms and
        returning nothing while it cannot be read."""
        if self.conn is not None and now_ms - self._silent_since >= self.reopen_after_ms:
            log.warning("no new news heartbeat for %d s, reopening news.db", (now_ms - self._silent_since) // 1000)
            self.close()
        try:
            if self.conn is None:
                self._open(now_ms)
            scores, score_seq = self._scores()
            beats = self.conn.execute(HEARTBEATS, (self.heartbeat_seq, self.batch)).fetchall()
        except (sqlite3.Error, NewsDbError, ValueError, KeyError) as exc:
            self._fail(exc)
            return []
        if self._failing:
            log.info("news.db can be read again")
            self._failing = False
        self.score_seq = score_seq
        lines = [(NEWS_LINE, line.encode()) for line in scores]
        if beats:
            self.heartbeat_seq = beats[-1]["seq"]
            self.last_heartbeat_ms = self._silent_since = now_ms
            lines += [(HEARTBEAT_LINE, HeartbeatLine.from_row(row).encode()) for row in beats]
        return lines

    def _open(self, now_ms: int) -> None:
        """Open news.db read only, starting the heartbeats after the newest one when no start was given."""
        self.conn = self.opener(self.path)
        self.opens += 1
        self._silent_since = now_ms
        if self.heartbeat_seq is None:
            self.heartbeat_seq = self.conn.execute("SELECT coalesce(max(seq), 0) FROM news_heartbeat").fetchone()[0]

    def _scores(self) -> tuple[list[ScoreLine], int]:
        """The score lines after the cursor in sequence order, each with its item's fields and its impacts."""
        rows = self.conn.execute(SCORES, (self.score_seq, self.batch)).fetchall()
        lines = [ScoreLine.from_rows(row, row, self.conn.execute(IMPACTS, (row["item_seq"], row["scorer"])).fetchall())
                 for row in rows]
        return lines, rows[-1]["seq"] if rows else self.score_seq

    def _fail(self, exc: Exception) -> None:
        """Close the connection after any error, logging the first of a streak by its type alone."""
        self.failures += 1
        if not self._failing:
            log.warning("news.db cannot be read (%s), retrying every tick", type(exc).__name__)
            self._failing = True
        self.close()

    def close(self) -> None:
        if self.conn is not None:
            with suppress(sqlite3.Error):
                self.conn.close()
            self.conn = None


def news_start(path: Path) -> tuple[dict[str, str], int]:
    """The meta of news.db and its newest score sequence, where a first launch starts reading."""
    conn = news_db.open_reader(path)
    try:
        return read_meta(conn), conn.execute("SELECT coalesce(max(seq), 0) FROM item_scores").fetchone()[0]
    finally:
        conn.close()


# Raw recordings


def prune_recordings(folder: Path, now_ms: int, keep_days: int) -> int:
    """Delete the recording files of UTC days more than keep_days before the day of now_ms, judged by the day in
    each name, since a file never crosses UTC midnight, and return how many went. Other files are left alone."""
    cutoff = date(1970, 1, 1) + timedelta(days=now_ms // MS_PER_DAY - keep_days)
    removed = 0
    for path in sorted(Path(folder).glob("*.jsonl.gz")):
        match = RECORDING_NAME.fullmatch(path.name)
        if match and date.fromisoformat(match.group(1)) < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


# The local only check


class LocalOnlySchedule:
    """When the local only check of section 13.2 is due, once per session, ten minutes before its open, or at once
    when the process starts inside a session or in those ten minutes."""

    def __init__(self) -> None:
        self.checked_open: int | None = None

    def due(self, now_ms: int) -> bool:
        session = sessions.next_session(now_ms)
        if session.open_ms == self.checked_open or now_ms < session.open_ms - LOCAL_CHECK_BEFORE_OPEN_MS:
            return False
        self.checked_open = session.open_ms
        return True


# The process


class StockProcess:
    """The group with its recorder, news puller, feed client and the work of each tick of the one second ticker.
    Recordings are pruned only when keep_raw_days is given, as the live command does, so a soak or a test replaying a
    long stretch keeps every file it wrote."""

    def __init__(self, group: StockGroup, puller: NewsPuller | None, clock: LiveClock,
                 recorder: StockRecorder | None = None, client: StockFeedClient | None = None,
                 check: Callable[[], LocalCheck] | None = None, executor: Executor | None = None,
                 keep_raw_days: int | None = None) -> None:
        cfg = group.engines[0].cfg
        feed = cfg.stock_feed
        self.group, self.puller, self.clock, self.recorder, self.client = group, puller, clock, recorder, client
        self.check = check or localcheck.run_check
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="local_only")
        self.local = LocalOnlySchedule()
        self.feed_stale_ms = round(feed.feed_stale_after_seconds * MS_PER_SECOND)
        self.healthy_after_ms = round(feed.mark_healthy_outside_session_seconds * MS_PER_SECOND)
        self.settle_seconds = (cfg.stock_costs.latency_ms + cfg.stock_costs.latency_jitter_ms) / MS_PER_SECOND
        self.lag_s = 0.0  # the worst lateness of the one second tick this minute, which measures loop backlog
        self.resubscribed: dict[str, int] = {}
        self._checking: Future | None = None
        self._pre_open_checked = 0
        self._last_tick: int | None = None
        self._status_minute: int | None = None
        self._beat_at: int | None = None
        self._counted: Counter[str] = Counter()
        self.keep_raw_days = keep_raw_days
        self._pruned_day: int | None = None

    def sink(self, kind: str, raw: str, recv_ms: int, conn_id: int) -> None:
        """Record one line, hand it to the group and ask for a reconnect when Finnhub sent an error or the line could
        not be read."""
        if self.recorder is not None:
            self.recorder.write(kind, raw, recv_ms, conn_id)
        events = self.group.on_line(kind, raw, recv_ms, conn_id)
        if self.client is not None and any(isinstance(ev, FeedNotice) and ev.kind == ERROR for ev in events):
            log.warning("feed connection %d sent an error or an unreadable message, reconnecting", conn_id)
            self.client.request_resync("finnhub error")

    def tick(self) -> list[str]:
        """One second of the ticker, returning the symbols whose subscribe frame should go out again."""
        now = self.clock.stamp(self.group.now_ms)
        if self.recorder is not None:
            self.recorder.flush()
        if self._last_tick is not None and now - self._last_tick > PAUSE_MS and self.client is not None:
            log.warning("the process was paused for %d s, such as by sleep", (now - self._last_tick) // 1000)
            self.client.request_resync(f"process paused for {(now - self._last_tick) // 1000} s")
        self._last_tick = now
        self.group.advance_to(now)
        resubscribe = self._watch(now) if self.client is not None else []
        if self.puller is not None:
            for kind, raw in self.puller.pull(now):
                self.sink(kind, raw, self.clock.stamp(self.group.now_ms), NEWS_CONN)
        self._local_only(now)
        self._prune(now)
        if minute_index(now) != self._status_minute:
            self._status_minute = minute_index(now)
            self.heartbeat(now)
        return resubscribe

    def _watch(self, now: int) -> list[str]:
        """The silence rule, the backoff reset and the quiet symbols of section 4.6 on the open connection."""
        client, health = self.client, self.group.market.health
        if client.connected_ms is None:
            return []
        last = max(health.last_msg_ms or 0, client.connected_ms)
        reason, self._pre_open_checked = silence_reason(now, last, self.feed_stale_ms, self._pre_open_checked)
        if reason is not None:
            log.warning("feed connection %d: %s, reconnecting", client.conn_id, reason)
            client.request_resync(reason)
            return []
        if should_mark_healthy(now, client.connected_ms, self.healthy_after_ms, health.any_healthy):
            client.mark_healthy()
        quiet = quiet_symbols(now, client.connected_ms, self.group.market.symbols, health.printed, self.resubscribed)
        for symbol in quiet:
            self.resubscribed[symbol] = now
        return quiet

    def _local_only(self, now: int) -> None:
        """Write a finished check to both databases and start the next one in the worker thread when it is due."""
        if self._checking is not None and self._checking.done():
            try:
                result = self._checking.result()
                verdict = (result.passed, result.offending_sockets, len(result.tunnels))
            except LocalCheckError as exc:
                log.error("the local only check could not run (%s), recorded as failed", exc)
                verdict = (False, 0, 0)
            self._checking = None
            log.info("local only check %s", "passed" if verdict[0] else "failed")
            self.group.local_only(self.clock.stamp(self.group.now_ms), *verdict)
        if self._checking is None and self.local.due(now):
            self._checking = self.executor.submit(self.check)

    def _prune(self, now: int) -> None:
        """Once a UTC day, delete the recordings older than keep_raw_days, logging only how many went."""
        if self.recorder is None or self.keep_raw_days is None or now // MS_PER_DAY == self._pruned_day:
            return
        self._pruned_day = now // MS_PER_DAY
        keep = self.keep_raw_days
        try:
            removed = prune_recordings(self.recorder.folder, now, keep)
        except OSError as exc:
            log.warning("old recordings could not be pruned (%s), trying again tomorrow", type(exc).__name__)
            return
        if removed:
            log.info("pruned %d recording files older than %d days", removed, keep)

    def heartbeat(self, now: int, state: str | None = None) -> None:
        """Write runtime_status and symbol_status to every run database and log one line of states, counts and
        timings, never a price."""
        market, health = self.group.market, self.group.market.health
        seconds = 60.0 if self._beat_at is None else max((now - self._beat_at) / MS_PER_SECOND, 1.0)
        counts = self.group.counts - self._counted
        self._beat_at, self._counted = now, Counter(self.group.counts)
        feed_state = state or ("disconnected" if not health.connected else "silent" if health.silent
                               else "delayed" if not health.delay_ok else "connected")
        session_state = "open" if health.session is not None else "closed"
        news_age = None
        if self.puller is not None and self.puller.last_heartbeat_ms is not None:
            news_age = (now - self.puller.last_heartbeat_ms) / MS_PER_SECOND
        rss, peak = memory_mb()
        symbols = []
        for i, symbol in enumerate(market.symbols):
            last = market.candles.last_eligible_ms(symbol)
            ref = float(market.last_ref[i])
            symbols.append((records.PUT_SYMBOL_STATUS, records.symbol_status_row(
                symbol, now, health.symbols[symbol].state, None if math.isnan(ref) else ref,
                None if last is None else (now - last) / MS_PER_SECOND)))
        living = []
        for engine in self.group.engines:
            living.append(len(engine.population.living_learners()))
            last_ckpt = None if engine.last_checkpoint_ts is None else parse_iso(engine.last_checkpoint_ts)
            engine.store.commit_now([(records.PUT_RUNTIME_STATUS, records.runtime_status_row(
                now, feed_state=feed_state, session_state=session_state, prints_per_s=counts[PRINTS] / seconds,
                msgs_per_s=counts[MESSAGES] / seconds, news_age_s=news_age, living=living[-1],
                last_checkpoint_ms=last_ckpt, rss_mb=rss, rss_peak_mb=peak)), *symbols])
        log.info("heartbeat feed=%s session=%s healthy_symbols=%d prints_per_s=%.1f msgs_per_s=%.1f news_age=%s "
                 "living=%s rss=%s peak=%.1f MB lag=%.0f ms", feed_state, session_state,
                 sum(s.state == "healthy" for s in health.symbols.values()), counts[PRINTS] / seconds,
                 counts[MESSAGES] / seconds, "n/a" if news_age is None else f"{news_age:.0f} s",
                 "/".join(map(str, living)), "n/a" if rss is None else f"{rss:.1f}", peak, self.lag_s * 1000)
        self.lag_s = 0.0

    def finish(self) -> None:
        """At shutdown, checkpoint both engines with the shared layer and write the stopped state."""
        self.group.finish(self.clock.stamp(self.group.now_ms))
        self.heartbeat(self.group.now_ms, state="stopped")

    def close(self) -> None:
        """Release the worker thread, the news.db connection and the recording."""
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.puller is not None:
            self.puller.close()
        if self.recorder is not None:
            with suppress(OSError):  # a file that cannot be closed only loses its trailer
                self.recorder.close()


async def serve(process: StockProcess, stop: asyncio.Event | None = None) -> None:
    """Run the feed and the ticker until stop is set, by default by SIGINT or SIGTERM, then let pending fills settle
    while their symbol is healthy, for at most the order latency, and checkpoint everything."""
    loop = asyncio.get_running_loop()
    redact.install_asyncio_handler(loop)
    signals: tuple[signal.Signals, ...] = ()
    if stop is None:
        stop, signals = asyncio.Event(), (signal.SIGINT, signal.SIGTERM)
        for sig in signals:
            loop.add_signal_handler(sig, stop.set)
    feed_stop = asyncio.Event()
    client, clock = process.client, process.clock

    async def ticker() -> None:
        while True:
            started = clock.monotonic()
            try:
                await asyncio.wait_for(feed_stop.wait(), TICK_SECONDS)
            except asyncio.TimeoutError:  # also when the stop lands as the wait times out
                process.lag_s = max(process.lag_s, clock.monotonic() - started - TICK_SECONDS)
            if feed_stop.is_set():
                return  # the final checkpoint does the last step, so nothing is recorded after the feed closes
            for symbol in process.tick():
                if await client.send(client.adapter.subscribe_frame(symbol)):
                    log.info("subscribed again to %s, which has not printed on connection %d", symbol, client.conn_id)

    async def supervisor() -> None:
        await stop.wait()
        log.info("shutdown requested, letting pending fills settle")
        deadline = clock.monotonic() + process.settle_seconds
        while process.group.settling() and clock.monotonic() < deadline:
            await asyncio.sleep(0.05)
        feed_stop.set()

    watcher = loop.create_task(supervisor())
    try:
        await asyncio.gather(client.run(feed_stop), ticker())
        # The final checkpoint runs while the handlers are still installed, so a second signal cannot cut it.
        process.finish()
    finally:
        watcher.cancel()
        for sig in signals:
            loop.remove_signal_handler(sig)


# Starting both runs


def code_commit(root: Path = Path(__file__).resolve().parents[4]) -> str:
    """The commit of the running code, marked dirty when tracked files changed, or unknown outside a git tree."""
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if head.returncode != 0 or dirty.returncode != 0:
        return "unknown"
    return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def open_runs(cfgs: Sequence[StockConfig], adapter: FeedAdapter, now_ms: int, commit: str,
              news_meta: Callable[[Path], tuple[Mapping[str, str], int]] = news_start) -> tuple[StockGroup, int]:
    """Resume both runs from their checkpoints, or launch them fresh at now_ms with their launch meta, and return the
    group with the score sequence the puller starts after. A first launch reads news.db's meta and newest score."""
    stores = [StockStore.open(cfg.db_path) for cfg in cfgs]
    try:
        restored = checkpoint.load(cfgs, stores)
        if restored is not None:
            group = StockGroup(*restored, adapter, now_ms)
            group.resume(now_ms)
        else:
            meta, start_seq = news_meta(cfgs[0].news_db_path)
            engines = []
            for cfg, store in zip(cfgs, stores):
                for key, value in records.launch_meta(cfg, now_ms, meta, start_seq, commit).items():
                    store.queue_meta(key, value)
                engine = StockEngine.fresh(cfg, store, now_ms)
                engine.news_seq = start_seq
                engines.append(engine)
            group = StockGroup(MarketLayer(cfgs[0], tuple(meta[key] for key in NEWS_HASHES)), engines, adapter,
                               now_ms)
        group.event(now_ms, LOCK, "holding the Finnhub websocket lock")
        group.event(now_ms, CODE_COMMIT, commit)
        group.persist()
    except BaseException:
        for store in stores:
            store.close()
        raise
    return group, min(engine.news_seq for engine in group.engines)


# The command


DESCRIPTION = ("Run the stock process for both runs on the one Finnhub websocket. It logs only states, counts and "
               "timings, never a price, a size or a fill.")
EXPECTED = (SecretsError, LockError, ConfigError, NewsDbError, StoreError, ResumeError, CalendarError, ValueError,
            OSError)


def local_url(text: str) -> str:
    """Accept only a plain websocket address on 127.0.0.1, so a test flag can never send the token to another host."""
    if not LOCAL_URL.fullmatch(text):
        raise argparse.ArgumentTypeError("only ws://127.0.0.1:PORT is accepted, for a local fake server in tests")
    return text


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The stocks arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("s1", metavar="S1", type=Path, help="the prices run config, configs/v2/s1_prices.toml")
    parser.add_argument("s2", metavar="S2", type=Path, help="the news run config, configs/v2/s2_news.toml")
    parser.add_argument("--dry-run", action="store_true", help="load the configs and secrets, take and release the "
                                                               "lock and print a verdict, without connecting or "
                                                               "opening a database")
    parser.add_argument("--log-dir", type=Path, help="the log folder, by default logs/v2/stocks two folders above "
                                                     "data_dir, so data/logs/v2/stocks")
    parser.add_argument("--test-url", type=local_url, help="a local fake server for tests, only ws://127.0.0.1:PORT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 stocks", description=DESCRIPTION)
    add_arguments(parser)
    return parser


def default_log_dir(cfg: StockConfig) -> Path:
    """logs/v2/stocks two folders above data_dir, so data/logs/v2/stocks for the committed configs."""
    return cfg.data_dir.parent.parent / "logs" / "v2" / "stocks"


def refused_variables(environ: Mapping[str, str]) -> list[str]:
    """The proxy and certificate variables that are set, in any case, which could route the token through another
    host or trust a certificate authority the owner did not choose."""
    return sorted(name for name, value in environ.items() if name.upper() in REFUSED and value)


def dry_run_report(cfgs: Sequence[StockConfig]) -> output.Report:
    return output.Report("stocks dry run", (
        ("verdict", True),
        ("runs", output.Count(len(cfgs))),
        ("symbols", output.Count(len(cfgs[0].stock_feed.symbols))),
        ("lock", output.Word("taken and released")),
    ))


def main(argv: list[str] | None = None) -> int:
    """Run the stock process from the command line."""
    return run(build_parser().parse_args(argv))


def run(args: argparse.Namespace) -> int:
    """Run the stock process, or with --dry-run check its setup, printing an expected failure as one error line with
    exit code 2."""
    os.umask(0o077)  # databases, recordings, logs and their folders are for the owner alone
    try:
        cfgs = load_run_pair(args.s1, args.s2)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cfg = cfgs[0]
    setup_logging(cfg.run.log_level, None if args.dry_run else args.log_dir or default_log_dir(cfg))
    redact.install()
    clock = LiveClock()
    with ExitStack() as held:
        try:
            refused = refused_variables(os.environ)
            if refused:
                raise ValueError(f"unset {', '.join(refused)} first, because the token goes only straight to Finnhub, "
                                 f"checked against the system's own certificate authorities")
            sessions.self_test()
            sessions.next_session(clock.now_ms())  # the session clock needs a working calendar for today
            token = load_secrets(cfg.path(cfg.run.secrets_file)).get(FINNHUB)
            adapter = FinnhubAdapter(cfg.stock_feed.symbols, token, args.test_url or URL)
            held.enter_context(websocket_lock(cfg.path(cfg.stock_feed.lock_file)))
            if args.dry_run:
                held.close()
                output.emit(dry_run_report(cfgs))
                return 0
            group, start_seq = open_runs(cfgs, adapter, clock.stamp(), code_commit())
            for engine in group.engines:
                held.callback(engine.store.close)
            recorder = StockRecorder(cfg.raw_dir) if cfg.stock_feed.record_raw_messages else None
        except EXPECTED as exc:
            print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
            return 2
        puller = NewsPuller(cfg.news_db_path, start_seq,
                            round(cfg.news_input.reopen_after_silent_seconds * MS_PER_SECOND))
        process = StockProcess(group, puller, clock, recorder, keep_raw_days=cfg.stock_feed.keep_raw_days)
        held.callback(process.close)
        process.client = StockFeedClient(adapter, BackoffSettings.from_config(cfg.stock_feed),
                                         redact.redacting_sink(process.sink), reconnect_rng(cfg.run.master_seed),
                                         clock, floor_ms=lambda: group.now_ms)
        log.info("running %d runs on %d symbols, news from sequence %d", len(cfgs), len(adapter.symbols), start_seq)
        try:
            asyncio.run(serve(process))
        except RecordingError as exc:
            log.error("%s, so the stock process stops", exc)
            return 2
    log.info("stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
