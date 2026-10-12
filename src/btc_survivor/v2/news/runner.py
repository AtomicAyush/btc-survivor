"""The news process of PLAN_V2 sections 7.1 and 14.3, collecting only until Phase 15 adds the scorers.

It sets umask 077, installs redact before anything logs, reads secrets only through local_secrets, refuses a proxy or
certificate variable, holds the Finnhub REST lock while Finnhub is enabled and opens news.db as its one writer. On every
start it writes the config hash, the prompt and schema hashes and both model ids to meta, with a config_changed event
when the hash moved, and starts its receive clock no earlier than the newest item, so seq order stays receive order even
when the wall clock stepped back across a restart. It then runs a task per channel, writes a news_heartbeat row every
heartbeat_seconds with a log line of counts once a minute, writes a pause event when the process was frozen, such as by
sleep, and prunes once a day. Items are written with their route and never scored here. SIGTERM or Ctrl+C lets polls in
flight finish for up to 20 seconds and exits with code 0, an expected failure such as a held lock exits with code 2, and
a crash with code 1. --dry-run loads the config and secrets, takes and releases the lock and opens nothing on the
network.

    btc_survivor_v2 news configs/v2/news.toml [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import os
import signal
import sqlite3
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path

from btc_survivor.clock import MS_PER_DAY, MS_PER_MINUTE, MS_PER_SECOND, LiveClock, iso, parse_iso
from btc_survivor.logsetup import setup_logging
from btc_survivor.v2 import output, redact
from btc_survivor.v2.config import ConfigError, NewsConfig, load_news_config
from btc_survivor.v2.local_secrets import REFUSED, SecretsError, load_secrets
from btc_survivor.v2.news import db, health
from btc_survivor.v2.news.budget import Budget, rules_from_config
from btc_survivor.v2.news.collector import Client, Collector, wait_or_stop
from btc_survivor.v2.news.db import NewsDbError
from btc_survivor.v2.news.http import HttpClient
from btc_survivor.v2.news.prefilter import Prefilter
from btc_survivor.v2.news.schedule import Schedule

log = logging.getLogger("btc_survivor.v2.news")

LOG_LEVEL = "INFO"
STOP_GRACE_SECONDS = 20.0
PRUNE_EVERY_SECONDS = MS_PER_DAY / MS_PER_SECOND
LOG_EVERY_MS = MS_PER_MINUTE
PAUSE_SLACK_MS = 10 * MS_PER_SECOND       # a heartbeat this much later than due means the process was frozen
COUNTS = ("new", "seed", "backlog", "filtered", "dup")


class LockError(RuntimeError):
    """Raised when another process holds the Finnhub REST lock."""


class CollectionFailed(RuntimeError):
    """Raised after a clean stop when a channel, heartbeat or pruning task crashed, so the process exits with code 1."""


@contextmanager
def rest_lock(path: Path) -> Iterator[Path]:
    """Hold an exclusive, non blocking flock on path for the block, so the news process is the only Finnhub REST
    client. Closing the file releases it, also when the process dies."""
    path = Path(path).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockError(f"another process holds the Finnhub REST lock {path}, and only one process may call "
                            f"Finnhub over REST, so this process exits") from None
        yield path
    finally:
        os.close(fd)


def refused_variables(environ: Mapping[str, str]) -> list[str]:
    """The proxy and certificate variables that are set, in any case, which could route the contact address or the
    token through another host or trust a certificate authority the owner did not choose."""
    return sorted(name for name, value in environ.items() if name.upper() in REFUSED and value)


def default_log_dir(config: NewsConfig) -> Path:
    """logs/v2/news two folders above data_dir, so data/logs/v2/news for the committed config."""
    return config.data_dir.parent.parent / "logs" / "v2" / "news"


def record_start(conn: sqlite3.Connection, config: NewsConfig, now_ms: int, channels: int) -> None:
    """Write the hashes and model ids to meta and the started event, with a config_changed event first when the
    config hash differs from the one the last start wrote."""
    previous = db.get_value(conn, "meta", "config_hash")
    current = config.config_hash
    with db.transaction(conn):
        db.set_values(conn, "meta", {"config_hash": current, "prompt_sha256": config.llm.prompt_sha256,
                                     "schema_sha256": config.llm.schema_sha256, "haiku_model": config.llm.model,
                                     "sonnet_model": config.llm.sonnet.model})
        if previous is not None and previous != current:
            db.insert(conn, "collector_events", {"ts": iso(now_ms), "kind": "config_changed",
                                                 "detail": f"config hash {previous[:12]} became {current[:12]}"})
        db.insert(conn, "collector_events", {"ts": iso(now_ms), "kind": "started", "detail": f"{channels} channels"})


def heartbeat_line(row: dict, counts: Counter[str]) -> str:
    """One log line of the heartbeat's states and the counts since start, never a headline."""
    polls = sum(n for name, n in counts.items() if name.startswith("polls "))
    failed = polls - counts["polls ok"] - counts["polls not_modified"]
    down = len(row["channels_down"].split(",")) if row["channels_down"] else 0
    fields = [f"healthy={row['healthy']}", f"polls={polls}", f"failed={failed}",
              *(f"{name}={counts[name]}" for name in COUNTS), f"waiting={row['waiting']}", f"channels_down={down}"]
    return " ".join(fields) + (f" reason={row['reason']}" if row["reason"] else "")


async def beat(config: NewsConfig, collector: Collector, heart: health.Heartbeat, start_ms: int,
               stop: asyncio.Event, wait: Callable[[asyncio.Event, float], Awaitable[bool]]) -> None:
    """Write a heartbeat row every heartbeat_seconds and a log line of counts once a minute, and a pause event when a
    beat comes far later than due, as after the laptop slept."""
    every_ms = config.news.heartbeat_seconds * MS_PER_SECOND
    clock = collector.clock
    last = clock.stamp()
    logged = last - LOG_EVERY_MS
    while True:
        now = clock.stamp()
        if now - last > every_ms + PAUSE_SLACK_MS:
            seconds = (now - last) // MS_PER_SECOND
            db.insert(collector.conn, "collector_events", {"ts": iso(now), "kind": "pause",
                                                           "detail": f"the process was paused for {seconds} s"})
            log.warning("the process was paused for %d s, such as by sleep", seconds)
        last = now
        row = heart.write(now, collector.clock_check.clean, collector.last_ok(), start_ms)
        if now - logged >= LOG_EVERY_MS:
            log.info("heartbeat %s", heartbeat_line(row, collector.counts))
            logged = now
        if await wait(stop, config.news.heartbeat_seconds):
            return


async def prune_daily(config: NewsConfig, conn: sqlite3.Connection, clock: LiveClock, stop: asyncio.Event,
                      wait: Callable[[asyncio.Event, float], Awaitable[bool]]) -> None:
    """Prune by retention at start and once a day."""
    retention = db.Retention.from_config(config)
    while True:
        deleted = db.prune(conn, clock.now_ms(), retention)
        log.info("pruned %s", " ".join(f"{table}={n}" for table, n in deleted.items()))
        if await wait(stop, PRUNE_EVERY_SECONDS):
            return


async def serve(config: NewsConfig, conn: sqlite3.Connection, client: Client, stop: asyncio.Event | None = None,
                clock: LiveClock | None = None, budget: Budget | None = None,
                wait: Callable[[asyncio.Event, float], Awaitable[bool]] = wait_or_stop) -> Counter[str]:
    """Collect until stop is set, by default by SIGTERM or SIGINT, then let tasks finish their poll for up to 20
    seconds, write the stopped event and return the counts. Tests pass their own client, stop, clock, budget and
    wait."""
    loop = asyncio.get_running_loop()
    redact.install_asyncio_handler(loop)
    if stop is None:
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
    clock = clock or LiveClock()
    newest = conn.execute("SELECT max(received_at) FROM news_items").fetchone()[0]
    if newest is not None:
        clock.stamp(parse_iso(newest))  # a wall clock stepped back across a restart never stamps an earlier item
    collector = Collector(config, conn, client, budget or Budget(rules_from_config(config)), clock, wait)
    heart = health.Heartbeat(config, conn, collector.schedule)
    start_ms = clock.stamp()
    record_start(conn, config, start_ms, len(collector.channels))
    log.info("collecting %d channels into %s", len(collector.channels), config.db_path.name)
    failed: list[str] = []

    async def guard(name: str, work: Awaitable[None]) -> None:
        try:
            await work
        except Exception:
            log.exception("%s failed, so the news process stops", name)
            failed.append(name)
            stop.set()

    tasks = [asyncio.create_task(guard(name, collector.run_channel(name, stop))) for name in collector.channels]
    tasks.append(asyncio.create_task(guard("heartbeat", beat(config, collector, heart, start_ms, stop, wait))))
    tasks.append(asyncio.create_task(guard("pruning", prune_daily(config, conn, clock, stop, wait))))
    await stop.wait()
    _, pending = await asyncio.wait(tasks, timeout=STOP_GRACE_SECONDS)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    db.insert(conn, "collector_events", {"ts": iso(clock.stamp()), "kind": "stopped",
                                         "detail": " ".join(f"{name}={collector.counts[name]}" for name in COUNTS)})
    if failed:
        raise CollectionFailed(f"{', '.join(failed)} failed")
    return collector.counts


DESCRIPTION = "Run the news process, which collects news into news.db. It logs only states, counts and timings."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The news arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("config", metavar="CONFIG", type=Path, help="the news config, such as configs/v2/news.toml")
    parser.add_argument("--dry-run", action="store_true", help="load the config and secrets, take and release the "
                                                               "lock and print a verdict, opening nothing on the "
                                                               "network")
    parser.add_argument("--log-dir", type=Path, help="the log folder, by default logs/v2/news two folders above "
                                                     "data_dir, so data/logs/v2/news")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 news", description=DESCRIPTION)
    add_arguments(parser)
    return parser


def dry_run_report(config: NewsConfig, channels: int, locked: bool) -> output.Report:
    sources = [config.finnhub.enabled, config.edgar.enabled, config.fed.enabled, config.bls.enabled]
    return output.Report("news dry run", (
        ("verdict", True),
        ("channels", output.Count(channels)),
        ("sources enabled", output.Count(sum(sources))),
        ("lock", output.Word("taken and released" if locked else "not needed")),
    ))


def main(argv: list[str] | None = None) -> int:
    """Run the news process from the command line."""
    return run(build_parser().parse_args(argv))


def run(args: argparse.Namespace) -> int:
    """Run the news process, or with --dry-run check its setup, printing an expected failure as one error line with
    exit code 2."""
    os.umask(0o077)  # news.db, its folder, the lock and the logs are for the owner alone
    try:
        config = load_news_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    setup_logging(LOG_LEVEL, None if args.dry_run else args.log_dir or default_log_dir(config))
    redact.install()
    with ExitStack() as held:
        try:
            refused = refused_variables(os.environ)
            if refused:
                raise ValueError(f"unset {', '.join(refused)} first, because the contact address and the token go "
                                 f"only straight to their hosts, checked against the system's own certificate "
                                 f"authorities")
            client = HttpClient.from_config(config, load_secrets(config.path(config.news.secrets_file)))
            channels = len(Schedule(config).channels)
            Prefilter(config)
            if config.finnhub.enabled:
                held.enter_context(rest_lock(config.path(config.news.rest_lock_file)))
            if args.dry_run:
                held.close()
                output.emit(dry_run_report(config, channels, config.finnhub.enabled))
                return 0
            conn = db.open_writer(config.db_path)
            held.callback(conn.close)
        except (SecretsError, LockError, NewsDbError, ValueError, OSError) as exc:
            print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
            return 2
        counts = asyncio.run(serve(config, conn, client))
    log.info("stopped, %s", " ".join(f"{name}={counts[name]}" for name in COUNTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
