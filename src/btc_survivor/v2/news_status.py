"""The news-status command of PLAN_V2 section 7.7, a week one and data quality report on news.db, which it opens read
only.

By default it prints counts and verdicts only, the status mix of polls and of items, the bytes received, the backlog
items and the channel hours that hold them, outage minutes with the longest outage, and the order and backlog
verdicts. The order verdict passes when received_at never decreases in seq order. The backlog verdict passes when
every backlog item follows a restart, a pause, a failed poll of its channel or an outage between its channel's
previous successful poll and its own. A pause counts over the whole time the process was frozen, which its event
ends, and an item whose previous successful poll is older than the kept polls needs a restart, pause or failed poll
before it. --detail adds requests per channel per day with their statuses, bytes and durations, backlog per channel
per hour, outage minutes per channel per day and, with --samples, kept and filtered headlines and near duplicate pairs
drawn with the report seed, for the owner's own terminal only.

    btc_survivor_v2 news-status [--config configs/v2/news.toml] [--days 7] [--detail [--samples 50]]
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from btc_survivor.clock import MS_PER_DAY, MS_PER_SECOND, LiveClock, iso, parse_iso
from btc_survivor.v2 import output
from btc_survivor.v2.config import ConfigError, NewsConfig, load_news_config
from btc_survivor.v2.news import db, health
from btc_survivor.v2.news.collector import CLOCK_SKEW, PARSE_ERROR, SUCCESS
from btc_survivor.v2.news.db import NewsDbError
from btc_survivor.v2.news.http import BLOCKED, HTTP_ERROR, NOT_MODIFIED, OK, RATE_LIMITED, TIMEOUT
from btc_survivor.v2.news.schedule import Schedule

DEFAULT_CONFIG = Path("configs/v2/news.toml")
POLL_STATUSES = (OK, NOT_MODIFIED, HTTP_ERROR, TIMEOUT, PARSE_ERROR, RATE_LIMITED, BLOCKED, CLOCK_SKEW)
ITEM_STATUSES = ("new", "seed", "backlog", "filtered")
EXPLAINING_EVENTS = ("started", "pause")
FROZEN = re.compile(r"process was paused for (\d+) s")
NEAR_SAMPLE_BELOW = 0.9


@dataclass
class Outages:
    """Outage minutes per channel and UTC day, and the longest run of one channel."""

    by_day: Counter[tuple[str, str]] = field(default_factory=Counter)
    longest: int = 0

    @property
    def total(self) -> int:
        return sum(self.by_day.values())


def order_violations(conn: sqlite3.Connection) -> int:
    """How many items have a received_at earlier than the item before them in seq order."""
    count, previous = 0, ""
    for (received_at,) in conn.execute("SELECT received_at FROM news_items ORDER BY seq"):
        count += received_at < previous
        previous = max(previous, received_at)
    return count


def mix(conn: sqlite3.Connection, sql: str, since: str, names: tuple[str, ...]) -> Counter[str]:
    """Rows per status since a time, with every expected status present."""
    counts: Counter[str] = Counter(dict.fromkeys(names, 0))
    counts.update(dict(conn.execute(sql, (since,)).fetchall()))
    return counts


def requests(conn: sqlite3.Connection, since: str) -> list[dict]:
    """Requests per channel per UTC day since a time, with their statuses, bytes and mean and longest durations,
    document fetches included under their own channels."""
    rows = conn.execute(
        "SELECT channel, substr(started_at, 1, 10), count(*), sum(status = 'ok'), sum(status = 'not_modified'), "
        "sum(status NOT IN ('ok', 'not_modified')), sum(http_status = 403), sum(http_status = 429), "
        "coalesce(sum(bytes), 0), CAST(round(avg(duration_ms)) AS INTEGER), max(duration_ms) FROM news_polls "
        "WHERE started_at >= ? "
        "GROUP BY 1, 2 ORDER BY 1, 2", (since,))
    names = ("channel", "day", "requests", "ok", "not_modified", "failed", "http_403", "http_429", "bytes",
             "mean_ms", "max_ms")
    return [dict(zip(names, row)) for row in rows]


def backlog_hours(conn: sqlite3.Connection, since: str) -> list[dict]:
    """Backlog items per channel per UTC hour since a time."""
    rows = conn.execute("SELECT channel, substr(received_at, 1, 13), count(*) FROM news_items WHERE status = 'backlog' "
                        "AND received_at >= ? GROUP BY 1, 2 ORDER BY 1, 2", (since,))
    return [{"channel": channel, "hour": f"{hour}:00Z", "backlog": n} for channel, hour, n in rows]


def _polls(conn: sqlite3.Connection, channel: str, before: str) -> list[tuple[int, int, str]]:
    """Every poll of one channel that started before a time, as start, cadence and status in seq order."""
    rows = conn.execute("SELECT started_at, cadence_s, status FROM news_polls WHERE channel = ? AND started_at < ? "
                        "ORDER BY seq", (channel, before))
    return [(parse_iso(started), cadence, status) for started, cadence, status in rows]


def outages(conn: sqlite3.Connection, schedule: Schedule, config: NewsConfig, since_ms: int, now_ms: int) -> Outages:
    """Outage minutes of every channel from since_ms to now_ms, by the rule of the heartbeat. A channel counts from
    its first poll, and between two successful polls, or after the last one, it is in outage while health says so."""
    news, found = config.news, Outages()
    for channel in schedule.channels:
        anchors = []
        for started, cadence, status in _polls(conn, channel.name, iso(now_ms)):
            if status in SUCCESS or not anchors:
                anchors.append((started, cadence))
        for (start, cadence), end in zip(anchors, [*(a[0] for a in anchors[1:]), now_ms]):
            if end <= since_ms or end - start <= max(news.outage_cadence_multiple * cadence,
                                                     news.outage_floor_seconds) * 1000:
                continue
            minutes = health.outage_minutes(schedule, channel.name, start, cadence, max(start, since_ms), end,
                                            news.outage_cadence_multiple, news.outage_floor_seconds)
            found.longest = max(found.longest, len(minutes))
            found.by_day.update((channel.name, iso(at)[:10]) for at in minutes)
    return found


def _spans(conn: sqlite3.Connection) -> list[tuple[int, int]]:
    """The time each start or pause event covers, from when the process froze to the event for a pause, which the
    heartbeat writes only after waking, and the instant itself for a start or a budget pause."""
    spans = []
    for ts, detail in conn.execute("SELECT ts, detail FROM collector_events WHERE kind IN (?, ?)", EXPLAINING_EVENTS):
        end = parse_iso(ts)
        frozen = FROZEN.search(detail or "")
        spans.append((end - int(frozen.group(1)) * MS_PER_SECOND if frozen else end, end))
    return spans


def unexplained_backlog(conn: sqlite3.Connection, schedule: Schedule, config: NewsConfig, since: str) -> list[int]:
    """The seqs of backlog items since a time that no restart, pause, failed poll or outage explains. Each is checked
    over the gap from its channel's previous successful poll to the poll that returned it, or over everything before
    it when that previous poll is no longer kept, which only a long failure, stop or pause allows."""
    news = config.news
    spans = _spans(conn)
    unexplained = []
    for seq, channel, received_at in conn.execute("SELECT seq, channel, received_at FROM news_items WHERE status = "
                                                  "'backlog' AND received_at >= ? ORDER BY seq", (since,)).fetchall():
        received = parse_iso(received_at)
        polls = [p for p in _polls(conn, channel, iso(received + 1)) if p[0] <= received]
        ok = [i for i, p in enumerate(polls) if p[2] in SUCCESS]
        if len(ok) < 2:
            previous, down = None, False
            failed = any(p[2] not in SUCCESS for p in polls[:ok[-1] if ok else len(polls)])
        else:
            (previous, cadence, _), own = polls[ok[-2]], polls[ok[-1]][0]
            failed = any(p[2] not in SUCCESS for p in polls[ok[-2]:ok[-1]])
            down = health.in_outage(schedule, channel, previous, cadence, own, news.outage_cadence_multiple,
                                    news.outage_floor_seconds)
        stopped = any(start <= received and (previous is None or end > previous) for start, end in spans)
        if not (failed or stopped or down):
            unexplained.append(seq)
    return unexplained


def samples(conn: sqlite3.Connection, config: NewsConfig, since: str, n: int) -> dict[str, list[dict]]:
    """Up to n kept and n filtered headlines and n near duplicate pairs with a Jaccard index from the threshold to
    0.9, each drawn in seq order with the report seed. A pair shows the original's headline and the duplicate's id,
    since news_dups keeps no headline."""
    rng = random.Random(config.news.master_seed)
    queries = {
        "kept headlines": ("SELECT seq, channel, status, relevance, title FROM news_items WHERE received_at >= ? "
                           "AND status != 'filtered' ORDER BY seq", (since,)),
        "filtered headlines": ("SELECT seq, channel, filter_reason AS reason, relevance, title FROM news_items "
                               "WHERE received_at >= ? AND status = 'filtered' ORDER BY seq", (since,)),
        "near duplicate pairs": ("SELECT d.item_seq AS seq, d.channel, d.vendor_id AS duplicate, "
                                 "round(d.jaccard, 3) AS jaccard, i.title FROM news_dups d JOIN news_items i "
                                 "ON i.seq = d.item_seq WHERE d.method = 'near' AND d.seen_at >= ? "
                                 "AND d.jaccard >= ? AND d.jaccard < ? ORDER BY d.id",
                                 (since, config.dedupe.near_dup_jaccard, NEAR_SAMPLE_BELOW)),
    }
    drawn = {}
    for name, (sql, params) in queries.items():
        cursor = conn.execute(sql, params)
        columns = [column[0] for column in cursor.description]
        found = [dict(zip(columns, row)) for row in cursor]
        drawn[name] = [found[i] for i in sorted(rng.sample(range(len(found)), min(n, len(found))))]
    return drawn


def build(conn: sqlite3.Connection, config: NewsConfig, now_ms: int, days: int,
          sample_count: int = 0) -> tuple[output.Report, list[output.Report]]:
    """The summary of counts and verdicts, and the detail sections for --detail."""
    schedule = Schedule(config)
    since_ms = now_ms - days * MS_PER_DAY
    since = iso(since_ms)
    polls = mix(conn, "SELECT status, count(*) FROM news_polls WHERE started_at >= ? GROUP BY status", since,
                POLL_STATUSES)
    items = mix(conn, "SELECT status, count(*) FROM news_items WHERE received_at >= ? GROUP BY status", since,
                ITEM_STATUSES)
    hours = backlog_hours(conn, since)
    sent = requests(conn, since)
    down = outages(conn, schedule, config, since_ms, now_ms)
    violations = order_violations(conn)
    unexplained = unexplained_backlog(conn, schedule, config, since)
    summary = (
        ("days", output.Count(days)),
        *((f"polls {status}", output.Count(polls[status])) for status in sorted(polls, key=_order(POLL_STATUSES))),
        *((f"items {status}", output.Count(items[status])) for status in sorted(items, key=_order(ITEM_STATUSES))),
        ("bytes received", output.Count(sum(row["bytes"] for row in sent))),
        ("backlog channel hours", output.Count(len(hours))),
        ("outage minutes", output.Count(down.total)),
        ("longest outage minutes", output.Count(down.longest)),
        ("channels with outage", output.Count(len({channel for channel, _ in down.by_day}))),
        ("order violations", output.Count(violations)),
        ("order verdict", violations == 0),
        ("unexplained backlog items", output.Count(len(unexplained))),
        ("backlog verdict", not unexplained),
    )
    sections = [
        output.Report("requests per channel per day", (), tuple(sent)),
        output.Report("backlog per channel per hour", (), tuple(hours)),
        output.Report("outage minutes per channel per day", (), tuple(
            {"channel": channel, "day": day, "minutes": n} for (channel, day), n in sorted(down.by_day.items()))),
    ]
    if sample_count:
        sections += [output.Report(name, (), tuple(rows))
                     for name, rows in samples(conn, config, since, sample_count).items()]
    return output.Report("news status", summary), sections


def _order(names: tuple[str, ...]):
    """A sort key that keeps the known statuses in their order and puts any other after them."""
    return lambda name: (names.index(name) if name in names else len(names), name)


def passed(summary: output.Report) -> bool:
    """Whether both verdicts of a summary passed."""
    return all(value for label, value in summary.summary if label.endswith("verdict"))


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def not_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return value


DESCRIPTION = "Print the week one and data quality report of news.db as counts and verdicts."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The news-status arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"the news config (default {DEFAULT_CONFIG})")
    parser.add_argument("--days", type=positive, default=7, help="the days the report covers, up to keep_polls_days")
    output.add_detail_flag(parser)
    parser.add_argument("--samples", type=not_negative, default=0,
                        help="sampled headlines of each kind, printed only with --detail")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 news-status", description=DESCRIPTION)
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the report from the command line."""
    return run(build_parser().parse_args(argv))


def run(args: argparse.Namespace) -> int:
    """Print the report, exiting 1 when a verdict fails and 2 on an expected error such as a missing database."""
    try:
        config = load_news_config(args.config)
        if args.days > config.news.keep_polls_days:
            raise ConfigError(f"--days is at most keep_polls_days, {config.news.keep_polls_days}, since older polls "
                              f"are pruned")
        conn = db.open_reader(config.db_path)
    except (ConfigError, NewsDbError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        summary, sections = build(conn, config, LiveClock().now_ms(), args.days, args.samples if args.detail else 0)
    finally:
        conn.close()
    if args.detail:
        output.emit(summary)
        for section in sections:
            output.emit(section, detail=True)
    else:
        rows = tuple(row for section in sections for row in section.rows)
        output.emit(output.Report(summary.title, summary.summary, rows))
    return 0 if passed(summary) else 1
