"""The news collector and the news process of PLAN_V2 sections 7.2 to 7.5 and 14.3, on a fake clock and a fake HTTP
client that serves the fixtures or fictional bodies in their shape, so no test reaches the network. received_at never
decreases and matches seq order across interleaved awaits, a first poll is all seed, ordinary polls at long or
slowing cadences give new, a restart gives backlog except for filings, a filing waits for its documents, clock skew
marks polls and the news process stops cleanly. The process collects only until [finbert] is pinned, and once it is
it scores with a fake model and a fake SDK client that sends nothing. Secrets are fake canaries. Instants use fixed UTC
offsets."""

import asyncio
import dataclasses
import hashlib
import heapq
import itertools
import json
import os
import re
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from itertools import pairwise

import numpy as np
import pytest

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, LiveClock, iso, parse_iso
from btc_survivor.v2.config import load_news_config
from btc_survivor.v2.local_secrets import ANTHROPIC, CONTACT, FINNHUB, REFUSED
from btc_survivor.v2.news import db, edgar, edgar_docs, rss, runner
from btc_survivor.v2.news.budget import Budget, rules_from_config
from btc_survivor.v2.news.collector import ChannelState, Collector, vendor_status
from btc_survivor.v2.news.http import Response, Validators
from helpers import ROOT

EDT, EST = timezone(timedelta(hours=-4)), timezone(timedelta(hours=-5))
NEWS = ROOT / "configs" / "v2" / "news.toml"
CONFIG = load_news_config(NEWS)
FIXTURES = ROOT / "tests" / "fixtures"
CANARY_TOKEN = "CANARYcollectorTOKEN0099"
CANARY_CONTACT = "canary.collector@example.invalid"
CANARY_KEY = "CANARYcollectorKEY0098"
SECOND, MINUTE = MS_PER_SECOND, MS_PER_MINUTE


def at(y, mo, d, h, mi, tz, s=0):
    return int(datetime(y, mo, d, h, mi, s, tzinfo=tz).timestamp()) * SECOND


MON_10 = at(2026, 10, 12, 10, 0, EDT)            # a Monday in regular hours and the filing window
MON_01 = at(2026, 10, 12, 1, 0, EDT)             # overnight
SAT_12 = at(2026, 10, 17, 12, 0, EDT)            # a Saturday

AAPL, GENERAL, ATOM, SUB = "finnhub_company_AAPL", "finnhub_general", "edgar_atom_8k", "edgar_sub_AAPL"
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet",
         "META": "Meta", "TSLA": "Tesla"}
GENERAL_PREFIX = "https://finnhub.io/api/v1/news?category=general"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000101/"
MAIN, EXHIBIT = ARCHIVE + "aapl-20261008.htm", ARCHIVE + "a8-kex991q4fy26.htm"
OLD, FILED, AMENDED = "0000320193-26-000101", "0000320193-26-000150", "0000320193-26-000151"
ACCEPTED = "2026-10-12T18:00:00.000Z"
MONETARY = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20261012a.htm"


def fixture(name):
    return (FIXTURES / name).read_bytes()


DOC_ROUTES = {
    "https://www.sec.gov/Archives/edgar/data/320193/": fixture("news/edgar_index_8k.htm"),
    MAIN: fixture("news/edgar_8k_main.htm"),
    EXHIBIT: fixture("news/edgar_ex99_1.htm"),
}


class FakeClock(LiveClock):
    """A wall clock the test sets, kept non decreasing by LiveClock.stamp."""

    def __init__(self, ms):
        super().__init__()
        self.ms = ms

    def now_ms(self):
        return self.ms


class FakeClient:
    """Answers each URL from routes, the longest matching prefix winning, after some event loop turns and an optional
    gate, with a Date header offset_ms away from the fake clock, or none when dated is false. An answer is bytes, an
    HTTP status, an exception to raise or a callable giving one of them."""

    def __init__(self, clock, routes=None):
        self.clock = clock
        self.routes = dict(routes or {})
        self.calls = []
        self.gates = {}
        self.offset_ms = 0
        self.dated = True
        self.turns = lambda: 1

    async def fetch(self, url, validators=None):
        self.calls.append((url, validators))
        if url in self.gates:
            await self.gates[url].wait()
        for _ in range(self.turns()):
            await asyncio.sleep(0)
        answer = self.routes[max((p for p in self.routes if url.startswith(p)), key=len)]
        answer = answer() if callable(answer) else answer
        if isinstance(answer, Exception):
            raise answer
        date = self.clock.ms + self.offset_ms if self.dated else None
        if isinstance(answer, int):
            status = {304: "not_modified", 403: "blocked", 429: "rate_limited"}.get(answer, "http_error")
            return Response(status, answer, b"", 0, 3, date, validators or Validators(), f"HTTP {answer}")
        return Response("ok", 200, answer, len(answer), 3, date, Validators(etag=f'"{len(self.calls)}"'))


def make(conn, clock, client, config=CONFIG):
    """A collector whose budget waits move the fake clock."""
    async def sleep(seconds):
        clock.ms += round(seconds * SECOND)
        await asyncio.sleep(0)
    return Collector(config, conn, client, Budget(rules_from_config(config), lambda: clock.ms, sleep), clock)


async def poll_at(collector, name, ms):
    collector.clock.ms = ms
    await collector.poll(name, ms)


async def poll_due(collector, name):
    """Move the clock to the channel's next due time and poll it there."""
    due = collector.next_poll(name, collector.clock.ms)
    collector.clock.ms = due
    await collector.poll(name, due)
    return due


def growing(render):
    """An answer that renders one more item on every call, with all earlier ones."""
    calls = itertools.count(1)
    return lambda: render(next(calls))


def finnhub_body(entries):
    return json.dumps(entries).encode()


def entry(news_id, headline, seconds, related="", summary=""):
    return {"category": "company", "datetime": seconds, "headline": headline, "id": news_id, "image": "",
            "related": related, "source": "Fixture Wire", "summary": summary,
            "url": f"https://news.example.com/fixture/{news_id}"}


def company_prefix(symbol):
    return f"https://finnhub.io/api/v1/company-news?symbol={symbol}&"


def company_render(symbol, clock):
    base = 1_000_000 * (list(NAMES).index(symbol) + 1)
    return lambda k: finnhub_body([entry(base + i, f"{NAMES[symbol]} fixture story number {i:03d}", clock.ms // SECOND,
                                         symbol) for i in range(k, 0, -1)])


def general_render(clock):
    return lambda k: finnhub_body([entry(9_000_000 + i, f"Fed fixture outlook number {i:03d}", clock.ms // SECOND)
                                   for i in range(k, 0, -1)])


def rss_body(items):
    """An RSS 2.0 feed of (title, link, ms) items, newest first."""
    rows = "".join(f"<item><title>{title}</title><link>{link}</link><guid>{link}</guid><description>{title}"
                   f"</description><pubDate>{format_datetime(datetime.fromtimestamp(ms / 1000, timezone.utc), True)}"
                   f"</pubDate></item>" for title, link, ms in items)
    return f'<?xml version="1.0" encoding="utf-8"?><rss version="2.0"><channel><title>Fixture</title>{rows}' \
           f'</channel></rss>'.encode()


def bls_render(feed, clock):
    return lambda k: rss_body([(f"Fixture {feed} release headline number {i:03d}",
                                f"https://www.bls.gov/news.release/archives/{feed}_{i:03d}.htm", clock.ms)
                               for i in range(k, 0, -1)])


def fed_render(path, title, clock):
    site = "https://www.federalreserve.gov/newsevents/"
    return lambda k: rss_body([(f"{title} number {i:03d}", f"{site}{path}{i:03d}a.htm", clock.ms)
                               for i in range(k, 0, -1)])


RENDERS = {
    AAPL: (company_prefix("AAPL"), lambda clock: company_render("AAPL", clock)),
    GENERAL: (GENERAL_PREFIX, general_render),
    "bls_cpi": (rss.bls_url("cpi"), lambda clock: bls_render("cpi", clock)),
    "fed_speeches": (rss.fed_url("speeches"), lambda clock: fed_render("speech/fixture", "Governor fixture speech "
                                                                                          "on topic", clock)),
    "fed_press_all": (rss.fed_url("press_all"), lambda clock: fed_render("pressreleases/other2026", "Federal Reserve "
                                                                         "fixture announcement", clock)),
}


def atom_body(accessions, form="8-K"):
    """The latest filings feed with one Apple filing per accession, newest first."""
    rows = "".join(
        f'<entry><title>{form} - Apple Inc. (0000320193) (Filer)</title>'
        f'<link rel="alternate" type="text/html" href="{edgar.index_url(320193, accession)}"/>'
        f'<summary type="html">Item 2.02: Results of Operations&lt;br&gt;Item 9.01: Exhibits</summary>'
        f'<updated>2026-10-12T09:59:00-04:00</updated>'
        f'<category scheme="https://www.sec.gov/" label="form type" term="{form}"/>'
        f'<id>urn:tag:sec.gov,2008:accession-number={accession}</id></entry>' for accession in accessions)
    return f'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Latest</title>' \
           f'{rows}</feed>'.encode()


def submissions_body(filings, cik=320193, name="Apple Inc."):
    """A submissions file whose filings.recent holds (accession, form, accepted, items) rows."""
    columns = list(zip(*filings)) if filings else [(), (), (), ()]
    recent = dict(zip(("accessionNumber", "form", "acceptanceDateTime", "items"), map(list, columns)))
    return json.dumps({"cik": f"{cik:010d}", "name": name, "filings": {"recent": recent}}).encode()


@pytest.fixture
def conn(tmp_path):
    connection = db.open_writer(tmp_path / "news.db")
    yield connection
    connection.close()


def items(conn):
    return [dict(row) for row in conn.execute("SELECT * FROM news_items ORDER BY seq")]


def polls(conn, channel):
    return [dict(row) for row in conn.execute("SELECT * FROM news_polls WHERE channel = ? ORDER BY seq", (channel,))]


def dups(conn):
    return [dict(row) for row in conn.execute("SELECT * FROM news_dups ORDER BY id")]


def events(conn, kind=None):
    rows = conn.execute("SELECT kind, detail FROM collector_events ORDER BY id").fetchall()
    return [tuple(row) for row in rows if kind is None or row[0] == kind]


# The receive order


def test_received_at_never_decreases_and_matches_seq_order_across_interleaved_awaits(conn):
    rng = np.random.default_rng(7)
    clock = FakeClock(MON_10)
    client = FakeClient(clock, DOC_ROUTES)
    for symbol in CONFIG.finnhub.company_symbols:
        client.routes[company_prefix(symbol)] = growing(company_render(symbol, clock))
    for name, (prefix, render) in RENDERS.items():
        if name != AAPL:
            client.routes[prefix] = growing(render(clock))
    for feed in ("empsit", "ppi", "jolts"):
        client.routes[rss.bls_url(feed)] = growing(bls_render(feed, clock))
    client.routes[edgar.ATOM_URL] = growing(lambda k: atom_body([f"0000320193-26-{200 + i:06d}"
                                                                 for i in range(k, 0, -1)]))
    for symbol, cik in CONFIG.edgar.ciks.items():
        client.routes[edgar.submissions_url(cik)] = submissions_body([], cik, f"Fixture {symbol} Inc.")

    def turns():
        clock.ms += int(rng.integers(-1500, 4000))       # the wall clock also steps back now and then
        return int(rng.integers(0, 6))

    client.turns = turns
    client.dated = False                                 # so only the order, never the clock check, can fail here
    collector = make(conn, clock, client)

    async def rounds():
        for n in range(5):
            clock.ms = due = MON_10 + n * 2 * MINUTE
            await asyncio.gather(*(collector.poll(name, due) for name in collector.channels
                                   if collector.schedule.cadence(name, due) is not None))

    asyncio.run(rounds())
    rows = items(conn)
    stamps = [row["received_at"] for row in rows]
    assert len(rows) == 15 * 5 and stamps == sorted(stamps)       # 15 growing feeds, one new item a round
    filings = [row for row in rows if row["kind"] == "8k" and row["status"] == "new"]
    assert len(filings) == 4 and all(row["detected_at"] < row["received_at"] for row in filings)
    assert Counter(row["status"] for row in rows) == {"seed": 15, "new": 60}


# Seed, new and backlog


def all_fixtures(clock):
    return FakeClient(clock, {
        RENDERS[AAPL][0]: fixture("finnhub/company_news_aapl.json"),
        GENERAL_PREFIX: fixture("finnhub/general_news.json"),
        edgar.ATOM_URL: fixture("news/edgar_atom_8k.xml"),
        edgar.submissions_url(320193): fixture("news/edgar_submissions_aapl.json"),
        rss.fed_url("press_all"): fixture("news/fed_press_all.xml"),
        rss.fed_url("speeches"): fixture("news/fed_speeches.xml"),
        rss.bls_url("cpi"): fixture("news/bls_cpi.rss"),
    })


FIXTURE_CHANNELS = (AAPL, GENERAL, ATOM, SUB, "fed_press_all", "fed_speeches", "bls_cpi")


def test_a_first_poll_is_all_seed_and_a_repoll_writes_nothing(conn):
    clock = FakeClock(MON_10)
    client = all_fixtures(clock)
    collector = make(conn, clock, client)

    async def each(ms):
        for name in FIXTURE_CHANNELS:
            await poll_at(collector, name, ms)

    asyncio.run(each(MON_10))
    rows = items(conn)
    assert len(rows) > 20 and {row["status"] for row in rows} == {"seed"} and {row["route"] for row in rows} == {"none"}
    assert {row["channel"] for row in rows} >= set(FIXTURE_CHANNELS) - {SUB}
    for name in FIXTURE_CHANNELS:
        assert db.get_value(conn, "meta", "first_poll." + name) is not None, name
        [row] = polls(conn, name)
        assert (row["status"], row["n_new"], row["n_backlog"]) == ("ok", 0, 0) and row["n_raw"] > 0, name
    assert [url for url, _ in client.calls if "Archives" in url or "pressreleases/monetary" in url] == []
    assert [(d["channel"], d["vendor_id"], d["method"]) for d in dups(conn) if d["method"] == "vendor_id"] == [
        (SUB, OLD, "vendor_id")]
    assert int(db.get_value(conn, "meta", "cursor.finnhub_general")) == max(
        int(row["vendor_id"]) for row in rows if row["channel"] == GENERAL)
    before, dup_rows = len(rows), len(dups(conn))
    asyncio.run(each(MON_10 + MINUTE))
    assert len(items(conn)) == before and len(dups(conn)) == dup_rows
    for name in FIXTURE_CHANNELS:
        assert polls(conn, name)[-1]["n_new"] == polls(conn, name)[-1]["n_dup"] == 0, name
    assert [v for url, v in client.calls[-len(FIXTURE_CHANNELS):] if url == rss.bls_url("cpi")] == [
        Validators(etag=f'"{FIXTURE_CHANNELS.index("bls_cpi") + 1}"')]


@pytest.mark.parametrize("channel, start, cadence", [
    ("bls_cpi", MON_10, 1800),
    ("bls_cpi", SAT_12, 7200),
    ("fed_speeches", SAT_12, 3600),
    (AAPL, MON_01, 600),
    (GENERAL, SAT_12, 600),
])
def test_ordinary_polls_at_long_cadences_give_status_new(conn, channel, start, cadence):
    clock = FakeClock(start)
    prefix, render = RENDERS[channel]
    collector = make(conn, clock, FakeClient(clock, {prefix: growing(render(clock))}))

    async def run():
        for _ in range(4):
            await poll_due(collector, channel)

    asyncio.run(run())
    starts = [parse_iso(row["started_at"]) for row in polls(conn, channel)]
    assert [b - a for a, b in pairwise(starts)] == [cadence * SECOND] * 3
    assert [(row["status"], row["route"]) for row in items(conn)] == [("seed", "none")] + [("new", "llm")] * 3


def test_a_cadence_slowing_from_15_to_1800_seconds_gives_status_new(conn):
    calendar = dataclasses.replace(CONFIG.calendar, cpi=("2026-10-14T08:30",))
    config = dataclasses.replace(CONFIG, calendar=calendar)
    clock = FakeClock(at(2026, 10, 14, 8, 44, EDT))
    collector = make(conn, clock, FakeClient(clock, {rss.bls_url("cpi"): growing(bls_render("cpi", clock))}), config)

    async def run():
        for _ in range(5):
            await poll_due(collector, "bls_cpi")

    asyncio.run(run())
    rows = polls(conn, "bls_cpi")
    assert [row["cadence_s"] for row in rows] == [15, 15, 15, 15, 1800]
    assert parse_iso(rows[-1]["started_at"]) - parse_iso(rows[-2]["started_at"]) == 1800 * SECOND
    assert [row["status"] for row in items(conn)] == ["seed", "new", "new", "new", "new"]


def test_a_cadence_quickening_into_a_release_window_gives_status_new(conn):
    """The gap from an 08:00 poll at 1,800 s to the 08:29 poll at 15 s is judged by the longer cadence."""
    calendar = dataclasses.replace(CONFIG.calendar, cpi=("2026-10-14T08:30",))
    config = dataclasses.replace(CONFIG, calendar=calendar)
    clock = FakeClock(at(2026, 10, 14, 8, 0, EDT))
    collector = make(conn, clock, FakeClient(clock, {rss.bls_url("cpi"): growing(bls_render("cpi", clock))}), config)

    async def run():
        await poll_at(collector, "bls_cpi", clock.ms)
        await poll_due(collector, "bls_cpi")

    asyncio.run(run())
    rows = polls(conn, "bls_cpi")
    assert [row["cadence_s"] for row in rows] == [1800, 15]
    assert parse_iso(rows[1]["started_at"]) == at(2026, 10, 14, 8, 29, EDT)
    assert [row["status"] for row in items(conn)] == ["seed", "new"]


def test_a_restart_restores_the_last_successful_poll_even_after_a_failure(conn):
    clock = FakeClock(MON_10)
    answers = [finnhub_body([]), 500]
    collector = make(conn, clock, FakeClient(clock, {RENDERS[AAPL][0]: lambda: answers.pop(0)}))

    async def run():
        await poll_at(collector, AAPL, MON_10)
        await poll_at(collector, AAPL, MON_10 + MINUTE)

    asyncio.run(run())
    state = make(conn, clock, FakeClient(clock)).states[AAPL]
    assert (state.ok_ms, state.ok_cadence_s, state.last_ms, state.seeded) == (MON_10, 60, MON_10 + MINUTE, True)


def test_a_restart_after_pruning_measures_from_the_last_success_kept_in_meta(conn):
    clock = FakeClock(at(2026, 10, 28, 13, 58, EDT))
    feed = []
    client = FakeClient(clock, {rss.fed_url("press_monetary"): lambda: rss_body(feed),
                                "https://www.federalreserve.gov/newsevents/pressreleases/monetary":
                                    fixture("news/fed_monetary_page.htm")})
    collector = make(conn, clock, client)

    async def window():
        for _ in range(88):
            await poll_due(collector, "fed_press_monetary")

    asyncio.run(window())
    december = at(2026, 12, 9, 13, 58, EST)
    db.prune(conn, december, db.Retention.from_config(CONFIG))
    assert polls(conn, "fed_press_monetary") == []
    clock.ms = december
    restarted = make(conn, clock, client)
    state = restarted.states["fed_press_monetary"]
    assert (state.ok_ms, state.ok_cadence_s) == (at(2026, 10, 28, 14, 19, EDT, 45), 15)
    feed.append(("Federal Reserve issues fixture FOMC statement", MONETARY, december))
    asyncio.run(poll_at(restarted, "fed_press_monetary", december))
    [row] = items(conn)
    assert (row["kind"], row["status"], row["route"]) == ("fed_monetary", "new", "llm")


def test_a_60_second_channel_after_a_20_minute_restart_gives_backlog_except_for_filings(conn):
    clock = FakeClock(MON_10)
    filings = [(OLD, "8-K", ACCEPTED, "2.02,9.01")]
    client = FakeClient(clock, {RENDERS[AAPL][0]: growing(company_render("AAPL", clock)),
                                edgar.submissions_url(320193): lambda: submissions_body(filings), **DOC_ROUTES})
    first = make(conn, clock, client)

    async def before():
        for ms in (MON_10, MON_10 + MINUTE):
            await poll_at(first, AAPL, ms)
            await poll_at(first, SUB, ms)

    asyncio.run(before())
    assert [row["status"] for row in items(conn)] == ["seed", "seed", "new"]
    restart = MON_10 + MINUTE + 20 * MINUTE
    filings[:0] = [(FILED, "8-K", ACCEPTED, "2.02,9.01"), (AMENDED, "8-K/A", ACCEPTED, "2.02")]
    clock.ms = restart
    second = make(conn, clock, client)
    assert second.states[AAPL].ok_ms == MON_10 + MINUTE and second.states[SUB].seeded

    async def after():
        await poll_at(second, AAPL, restart)
        await poll_at(second, SUB, restart)

    asyncio.run(after())
    rows = {row["vendor_id"]: row for row in items(conn)[3:]}
    assert [(r["kind"], r["status"], r["route"]) for r in rows.values()] == [
        ("media", "backlog", "none"), ("8k_a", "backlog", "none"), ("8k", "new", "llm")]
    assert rows[FILED]["detected_at"] is not None
    assert conn.execute("SELECT count(*) FROM news_bodies").fetchone()[0] == 1


def test_the_backlog_edge_is_the_longer_cadence_plus_the_gap_plus_any_budget_wait(conn):
    clock = FakeClock(MON_10)
    answers = []
    client = FakeClient(clock, {RENDERS[AAPL][0]: lambda: answers.pop(0)})
    render = company_render("AAPL", clock)
    collector = make(conn, clock, client)
    allowed = int((60 + CONFIG.news.backlog_gap_seconds) * SECOND)
    second = MON_10 + allowed
    third = second + allowed + 1
    pause = third + 620 * SECOND                         # a 429 here pauses Finnhub for 60 s
    fourth = pause + 5 * SECOND                          # the budget holds it 55 s, so it starts 680 s after third
    answers += [render(1), render(2), render(3), 429, render(4)]

    async def run():
        for ms in (MON_10, second, third, pause, fourth):
            await poll_at(collector, AAPL, ms)

    asyncio.run(run())
    assert [row["status"] for row in items(conn)] == ["seed", "new", "backlog", "new"]
    rows = polls(conn, AAPL)
    assert [row["status"] for row in rows] == ["ok", "ok", "ok", "rate_limited", "ok"]
    assert rows[-1]["budget_wait_ms"] == 55 * SECOND
    gap = parse_iso(rows[-1]["started_at"]) - parse_iso(rows[2]["started_at"])
    assert gap == 680 * SECOND and allowed < gap <= allowed + 55 * SECOND
    assert events(conn, "pause") == [("pause", "finnhub paused for 60 s after HTTP 429")]


def test_the_dormant_fomc_channel_gives_new_in_the_next_window_and_backlog_after_a_late_restart(conn):
    clock = FakeClock(at(2026, 10, 28, 13, 58, EDT))
    feed = []
    client = FakeClient(clock, {rss.fed_url("press_monetary"): lambda: rss_body(feed),
                                "https://www.federalreserve.gov/newsevents/pressreleases/monetary":
                                    fixture("news/fed_monetary_page.htm")})
    collector = make(conn, clock, client)

    async def window():
        for _ in range(88):
            await poll_due(collector, "fed_press_monetary")

    asyncio.run(window())
    assert parse_iso(polls(conn, "fed_press_monetary")[-1]["started_at"]) == at(2026, 10, 28, 14, 19, EDT, 45)
    december = at(2026, 12, 9, 13, 58, EST)
    assert collector.next_poll("fed_press_monetary", clock.ms) == december
    feed.append(("Federal Reserve issues fixture FOMC statement", MONETARY, december))

    asyncio.run(poll_due(collector, "fed_press_monetary"))
    [row] = items(conn)
    assert (row["kind"], row["status"], row["route"]) == ("fed_monetary", "new", "llm")
    late = at(2026, 12, 9, 14, 10, EST)
    feed.append(("Federal Reserve issues a second fixture statement", MONETARY.replace("a.htm", "b.htm"), late))
    clock.ms = late
    restarted = make(conn, clock, client)
    restarted.states["fed_press_monetary"] = dataclasses.replace(restarted.states["fed_press_monetary"],
                                                                 ok_ms=at(2026, 10, 28, 14, 19, EDT, 45))
    asyncio.run(poll_at(restarted, "fed_press_monetary", late))
    assert [(row["status"], row["route"]) for row in items(conn)] == [("new", "llm"), ("backlog", "none")]


def test_a_stale_vendor_time_gives_stale_vendor_from_every_scorer(conn):
    clock = FakeClock(MON_10)
    later = MON_10 + MINUTE
    day = 24 * 3600
    bodies = [finnhub_body([]), finnhub_body([
        entry(11, "Apple fixture supplier note one day old", later // SECOND - day, "AAPL"),
        entry(12, "Apple fixture factory report older than that", later // SECOND - day - 1, "AAPL"),
        {**entry(13, "Apple fixture keynote recap without a time", 0, "AAPL"), "datetime": None}])]
    collector = make(conn, clock, FakeClient(clock, {RENDERS[AAPL][0]: lambda: bodies.pop(0)}))

    async def run():
        await poll_at(collector, AAPL, MON_10)
        await poll_at(collector, AAPL, later)

    asyncio.run(run())
    rows = items(conn)
    assert [(row["status"], row["route"]) for row in rows] == [("new", "llm")] * 3
    hours = CONFIG.news.max_vendor_age_hours
    # The one guard every scorer applies before scoring, Haiku, Sonnet and FinBERT alike.
    assert [vendor_status(row["received_at"], row["vendor_ts"], hours) for row in rows] == [
        None, "stale_vendor", None]


# Held filings and releases


def test_a_filing_waits_for_its_documents_while_other_channels_skip_it(conn):
    clock = FakeClock(MON_10)
    atom, filings = [OLD], [(OLD, "8-K", ACCEPTED, "2.02,9.01")]
    client = FakeClient(clock, {edgar.ATOM_URL: lambda: atom_body(atom),
                                edgar.submissions_url(320193): lambda: submissions_body(filings), **DOC_ROUTES})
    collector = make(conn, clock, client)
    detected, seen, released = MON_10 + 30 * SECOND, MON_10 + 60 * SECOND, MON_10 + 90 * SECOND

    def stored():
        return conn.execute("SELECT count(*) FROM news_items WHERE vendor_id = ?", (FILED,)).fetchone()[0]

    async def scenario():
        await poll_at(collector, ATOM, MON_10)
        await poll_at(collector, SUB, MON_10)
        atom.insert(0, FILED)
        filings.insert(0, (FILED, "8-K", ACCEPTED, "2.02,9.01"))
        gate = client.gates[edgar.index_url(320193, FILED)] = asyncio.Event()
        task = asyncio.create_task(poll_at(collector, ATOM, detected))
        for _ in range(20):
            await asyncio.sleep(0)
        assert ("edgar", FILED) in collector.held and stored() == 0
        await poll_at(collector, SUB, seen)
        assert stored() == 0 and [d["vendor_id"] for d in dups(conn)] == [OLD]
        clock.ms = released
        gate.set()
        await task
        await poll_at(collector, SUB, released + MINUTE)

    asyncio.run(scenario())
    [row] = [row for row in items(conn) if row["vendor_id"] == FILED]
    assert (row["kind"], row["status"], row["route"], row["tickers"]) == ("8k", "new", "llm", "AAPL")
    assert row["detected_at"] == iso(detected) and row["received_at"] >= iso(released)
    assert row["items_8k"] == "2.02,9.01" and row["channel"] == ATOM
    body = dict(conn.execute("SELECT * FROM news_bodies WHERE item_seq = ?", (row["seq"],)).fetchone())
    expected = full_body()
    assert (body["text"], body["chars_total"], body["truncated"], body["doc_urls"]) == (
        expected.text, expected.chars_total, int(expected.truncated), expected.doc_urls)
    assert [(d["vendor_id"], d["channel"], d["seen_at"], d["method"]) for d in dups(conn)] == [
        (OLD, SUB, iso(MON_10), "vendor_id"), (FILED, SUB, iso(seen), "vendor_id")]
    assert [(p["status"], p["cadence_s"]) for p in polls(conn, "edgar_docs")] == [("ok", 0)] * 3


def held_filing(conn, routes):
    """Seed the Atom channel, then poll a new filing whose documents answer from routes, returning its body row."""
    clock = FakeClock(MON_10)
    atom = [OLD]
    client = FakeClient(clock, {edgar.ATOM_URL: lambda: atom_body(atom), **DOC_ROUTES, **routes})
    collector = make(conn, clock, client)

    async def run():
        await poll_at(collector, ATOM, MON_10)
        atom.insert(0, FILED)
        await poll_at(collector, ATOM, MON_10 + 30 * SECOND)

    asyncio.run(run())
    seq = conn.execute("SELECT seq FROM news_items WHERE vendor_id = ?", (FILED,)).fetchone()[0]
    return dict(conn.execute("SELECT * FROM news_bodies WHERE item_seq = ?", (seq,)).fetchone())


def test_a_filing_listed_twice_in_one_feed_is_one_item_without_a_duplicate_row(conn):
    clock = FakeClock(MON_10)
    atom = [OLD]
    collector = make(conn, clock, FakeClient(clock, {edgar.ATOM_URL: lambda: atom_body(atom), **DOC_ROUTES}))

    async def run():
        await poll_at(collector, ATOM, MON_10)
        atom[:0] = [FILED, FILED]
        await poll_at(collector, ATOM, MON_10 + 30 * SECOND)

    asyncio.run(run())
    assert [(row["vendor_id"], row["status"]) for row in items(conn)] == [(OLD, "seed"), (FILED, "new")]
    assert dups(conn) == [] and len(polls(conn, "edgar_docs")) == 3


def test_a_filing_whose_index_fails_twice_is_written_flagged_without_text(conn):
    body = held_filing(conn, {edgar.index_url(320193, FILED): 500})
    assert (body["text"], body["chars_total"], body["truncated"], body["doc_urls"]) == ("", 0, 1, "[]")
    assert [p["status"] for p in polls(conn, "edgar_docs")] == ["http_error", "http_error"]
    assert events(conn, "doc_failed") == [("doc_failed", f"{FILED} without its index")]


def test_a_filing_whose_main_document_fails_twice_keeps_its_exhibit_and_is_flagged(conn):
    body = held_filing(conn, {MAIN: 500})
    exhibit = edgar_docs.document(fixture("news/edgar_ex99_1.htm"), CONFIG.edgar.exhibit_max_chars)
    assert (body["text"], body["truncated"], json.loads(body["doc_urls"])) == (exhibit.text, 1, [EXHIBIT])
    assert [p["status"] for p in polls(conn, "edgar_docs")] == ["ok", "http_error", "http_error", "ok"]
    assert events(conn, "doc_failed") == [("doc_failed", f"{FILED} without its main document")]


def full_body():
    """The body of the fixture filing with both its documents."""
    return edgar_docs.body([
        (MAIN, edgar_docs.document(fixture("news/edgar_8k_main.htm"), CONFIG.edgar.main_doc_max_chars)),
        (EXHIBIT, edgar_docs.document(fixture("news/edgar_ex99_1.htm"), CONFIG.edgar.exhibit_max_chars))])


def test_a_document_that_fails_once_is_fetched_again(conn):
    answers = [500, fixture("news/edgar_8k_main.htm")]
    body = held_filing(conn, {MAIN: lambda: answers.pop(0)})
    expected = full_body()
    assert (body["text"], body["truncated"], body["doc_urls"]) == (expected.text, int(expected.truncated),
                                                                   expected.doc_urls)
    assert events(conn, "doc_failed") == []


def test_a_monetary_policy_release_waits_for_its_page(conn):
    clock = FakeClock(MON_10)
    feed = [("Federal Reserve Board announces a fixture survey", "https://www.federalreserve.gov/newsevents/"
                                                                 "pressreleases/other20261012a.htm", MON_10)]
    client = FakeClient(clock, {rss.fed_url("press_all"): lambda: rss_body(feed),
                                MONETARY: fixture("news/fed_monetary_page.htm")})
    collector = make(conn, clock, client)

    async def run():
        await poll_at(collector, "fed_press_all", MON_10)
        feed.insert(0, ("Federal Reserve issues fixture FOMC statement", MONETARY, MON_10 + MINUTE))
        await poll_at(collector, "fed_press_all", MON_10 + MINUTE)

    asyncio.run(run())
    row = items(conn)[-1]
    assert (row["kind"], row["status"], row["route"], row["tickers"]) == ("fed_monetary", "new", "llm", "MACRO")
    assert row["detected_at"] == iso(MON_10 + MINUTE)
    body = conn.execute("SELECT text, doc_urls FROM news_bodies WHERE item_seq = ?", (row["seq"],)).fetchone()
    expected = rss.fed_page_body(fixture("news/fed_monetary_page.htm"), MONETARY, CONFIG.fed.monetary_body_max_chars)
    assert tuple(body) == (expected.text, expected.doc_urls)
    assert [p["status"] for p in polls(conn, "fed_pages")] == ["ok"]


@pytest.mark.parametrize("broken", ["index", "main document", "exhibit"])
def test_a_filing_document_that_cannot_be_parsed_is_left_out_and_flagged(conn, monkeypatch, broken):
    def refuse(*args):
        raise AssertionError("unknown status keyword 'foo' in marked section")

    if broken == "index":
        monkeypatch.setattr(edgar_docs, "filing_documents", refuse)
    else:
        page, real = fixture("news/edgar_8k_main.htm" if broken == "main document" else "news/edgar_ex99_1.htm"), \
            edgar_docs.document
        monkeypatch.setattr(edgar_docs, "document", lambda body, *args: refuse() if body == page else real(body, *args))
    body = held_filing(conn, {})
    assert body["truncated"] == 1 and len(json.loads(body["doc_urls"])) == {"index": 0}.get(broken, 1)
    assert events(conn, "doc_failed") == [("doc_failed", f"{FILED} without its {broken}")]


def test_a_release_page_that_cannot_be_parsed_is_written_flagged_without_it(conn, monkeypatch):
    def refuse(*args):
        raise AssertionError("unknown status keyword 'foo' in marked section")

    monkeypatch.setattr(rss, "fed_page_body", refuse)
    clock = FakeClock(MON_10)
    feed = []
    collector = make(conn, clock, FakeClient(clock, {rss.fed_url("press_all"): lambda: rss_body(feed),
                                                     MONETARY: fixture("news/fed_monetary_page.htm")}))

    async def run():
        await poll_at(collector, "fed_press_all", MON_10)
        feed.append(("Federal Reserve issues fixture FOMC statement", MONETARY, MON_10 + MINUTE))
        await poll_at(collector, "fed_press_all", MON_10 + MINUTE)

    asyncio.run(run())
    [row] = items(conn)
    body = conn.execute("SELECT text, truncated FROM news_bodies WHERE item_seq = ?", (row["seq"],)).fetchone()
    assert (row["status"], tuple(body)) == ("new", ("", 1))
    assert events(conn, "doc_failed") == [("doc_failed", f"{MONETARY} without its page")]


# Deduplication


def test_a_first_poll_seeds_items_with_the_same_headline_without_duplicate_rows(conn):
    clock = FakeClock(MON_10)
    seconds = MON_10 // SECOND
    body = finnhub_body([entry(11, "Apple fixture headline told twice", seconds, "AAPL"),
                         entry(12, "Apple fixture headline told twice", seconds, "AAPL")])
    collector = make(conn, clock, FakeClient(clock, {RENDERS[AAPL][0]: body}))
    asyncio.run(poll_at(collector, AAPL, MON_10))
    assert [row["status"] for row in items(conn)] == ["seed", "seed"] and dups(conn) == []


def test_hash_and_near_duplicates_write_dup_rows_once_with_new_tickers(conn):
    clock = FakeClock(MON_10)
    seconds = MON_10 // SECOND
    bodies = [
        finnhub_body([entry(1, "Apple unveils a fixture device for schools - Reuters", seconds, "AAPL"),
                      entry(2, "Apple fixture supplier meeting ends early today", seconds, "AAPL")]),
        finnhub_body([entry(3, "APPLE UNVEILS A FIXTURE DEVICE FOR SCHOOLS", seconds, "AAPL"),
                      entry(4, "Apple fixture supplier meeting ends early", seconds, "AAPL",
                            summary="Microsoft attended the fixture meeting.")]),
    ]
    collector = make(conn, clock, FakeClient(clock, {RENDERS[AAPL][0]: lambda: bodies[min(len(calls), 1)]}))
    calls = []

    async def run():
        for n in range(3):
            await poll_at(collector, AAPL, MON_10 + n * MINUTE)
            calls.append(n)

    asyncio.run(run())
    assert [row["vendor_id"] for row in items(conn)] == ["1", "2"]
    found = [(d["vendor_id"], d["method"], d["jaccard"], d["new_tickers"]) for d in dups(conn)]
    assert found == [("3", "hash", None, None), ("4", "near", pytest.approx(6 / 7), "MSFT")]
    assert [p["n_dup"] for p in polls(conn, AAPL)] == [0, 2, 0]


def test_filings_with_the_same_title_are_both_kept(conn):
    clock = FakeClock(MON_10)
    atom = [OLD]
    collector = make(conn, clock, FakeClient(clock, {edgar.ATOM_URL: lambda: atom_body(atom), **DOC_ROUTES}))

    async def run():
        await poll_at(collector, ATOM, MON_10)
        atom[:0] = [AMENDED, FILED]
        await poll_at(collector, ATOM, MON_10 + 30 * SECOND)

    asyncio.run(run())
    rows = items(conn)
    assert len({row["title_hash"] for row in rows}) == 1
    assert [(row["vendor_id"], row["status"]) for row in rows] == [(OLD, "seed"), (AMENDED, "new"), (FILED, "new")]
    assert dups(conn) == []


def test_the_general_news_cursor_moves_forward_and_survives_a_restart(conn):
    clock = FakeClock(MON_10)
    client = FakeClient(clock, {GENERAL_PREFIX: growing(general_render(clock))})
    collector = make(conn, clock, client)

    async def run(target, minutes):
        for minute in minutes:
            await poll_at(target, GENERAL, MON_10 + minute * MINUTE)

    asyncio.run(run(collector, range(3)))
    assert [url[len(GENERAL_PREFIX):] for url, _ in client.calls] == ["&minId=0", "&minId=9000001",
                                                                     "&minId=9000002"]
    assert db.get_value(conn, "meta", "cursor.finnhub_general") == "9000003"
    asyncio.run(run(make(conn, clock, client), [3]))
    assert client.calls[-1][0].endswith("&minId=9000003")
    assert [row["status"] for row in items(conn)] == ["seed", "new", "new", "new"]


def test_validators_go_back_only_to_their_own_url_and_a_304_is_a_success(conn):
    clock = FakeClock(at(2026, 10, 12, 19, 58, EDT))      # 23:58 UTC, so the company news dates change soon
    answers = [fixture("news/fed_press_all.xml"), 304, b"<not a feed", 304]
    client = FakeClient(clock, {rss.fed_url("press_all"): lambda: answers.pop(0),
                                RENDERS[AAPL][0]: growing(company_render("AAPL", clock))})
    collector = make(conn, clock, client)
    start = clock.ms

    async def run():
        for n in range(4):
            await poll_at(collector, "fed_press_all", start + n * MINUTE)
        await poll_at(collector, AAPL, start)
        await poll_at(collector, AAPL, start + 4 * MINUTE)

    asyncio.run(run())
    sent = [v for url, v in client.calls if url == rss.fed_url("press_all")]
    assert sent == [None, Validators(etag='"1"'), Validators(etag='"1"'), Validators(etag='"1"')]
    assert [(p["status"], p["n_raw"]) for p in polls(conn, "fed_press_all")] == [
        ("ok", 4), ("not_modified", 0), ("parse_error", None), ("not_modified", 0)]
    assert collector.states["fed_press_all"].failures == 0
    company = [(url, v) for url, v in client.calls if url.startswith(RENDERS[AAPL][0])]
    assert company[0][0] != company[1][0] and company[1][1] is None


def test_failure_backoff_doubles_the_cadence_up_to_the_cap_and_never_polls_sooner_than_the_cadence(conn):
    clock = FakeClock(MON_10)
    collector = make(conn, clock, FakeClient(clock, {rss.bls_url("cpi"): 500}))
    for failures, delay_s in ((0, 60), (1, 120), (2, 240), (3, 480), (4, 900), (9, 900)):
        collector.states[AAPL] = ChannelState(last_ms=MON_10, last_cadence_s=60, failures=failures)
        assert collector.next_poll(AAPL, MON_10) == MON_10 + delay_s * SECOND, failures

    async def failures(n):
        return [await poll_due(collector, "bls_cpi") for _ in range(n)]

    due = asyncio.run(failures(3))
    assert [b - a for a, b in pairwise(due)] == [1800 * SECOND] * 2
    assert [p["status"] for p in polls(conn, "bls_cpi")] == ["http_error"] * 3
    assert collector.states["bls_cpi"].failures == 3


def test_a_403_pauses_every_sec_channel_for_900_seconds(conn):
    clock = FakeClock(MON_10)
    client = FakeClient(clock, {edgar.submissions_url(320193): 403, edgar.ATOM_URL: atom_body([OLD])})
    collector = make(conn, clock, client)
    asyncio.run(poll_at(collector, SUB, MON_10))
    assert polls(conn, SUB)[0]["status"] == "blocked"
    assert events(conn, "pause") == [("pause", "sec paused for 900 s after HTTP 403")]
    assert collector.next_poll(SUB, MON_10) == MON_10 + 120 * SECOND
    asyncio.run(poll_at(collector, ATOM, MON_10 + SECOND))
    [atom] = polls(conn, ATOM)
    assert (atom["status"], atom["budget_wait_ms"]) == ("ok", 899 * SECOND)
    assert atom["started_at"] == iso(MON_10 + 900 * SECOND)
    assert [kind for kind, _ in events(conn)] == ["pause", "budget_wait"]


# Clock skew


def test_clock_skew_marks_polls_failed_until_the_median_is_back_within_five_seconds(conn):
    clock = FakeClock(MON_10)
    client = FakeClient(clock, {RENDERS[AAPL][0]: growing(company_render("AAPL", clock))})
    client.offset_ms = 10 * SECOND
    collector = make(conn, clock, client)

    asyncio.run(poll_at(collector, AAPL, MON_10))
    assert polls(conn, AAPL)[-1]["status"] == "clock_skew" and items(conn) == []
    assert not collector.clock_check.clean and collector.states[AAPL].failures == 1
    assert not collector.states[AAPL].seeded
    assert collector.next_poll(AAPL, MON_10) == MON_10 + 120 * SECOND
    client.offset_ms = 0
    asyncio.run(poll_at(collector, AAPL, MON_10 + 120 * SECOND))
    assert polls(conn, AAPL)[-1]["status"] == "ok" and collector.clock_check.clean
    assert [row["status"] for row in items(conn)] == ["seed", "seed"]
    assert events(conn, "clock_skew") == [("clock_skew", "median Date header offset 10.0 s"),
                                          ("clock_skew", "clean again")]
    client.offset_ms = -6 * SECOND                       # the median of 10, 0, -6, -6 and -6 s is -6 s
    for n in (1, 2, 3):
        asyncio.run(poll_at(collector, AAPL, MON_10 + (120 + 60 * n) * SECOND))
    assert [p["status"] for p in polls(conn, AAPL)] == ["clock_skew", "ok", "ok", "ok", "clock_skew"]


# The channel loop and the news process


def test_a_channel_task_ends_on_stop_and_when_dormant_for_good(conn):
    calendar = dataclasses.replace(CONFIG.calendar, fomc_statement=())
    config = dataclasses.replace(CONFIG, calendar=calendar)
    clock = FakeClock(MON_10)
    client = FakeClient(clock, {rss.bls_url("cpi"): growing(bls_render("cpi", clock))})
    collector = make(conn, clock, client, config)

    async def run():
        stop = asyncio.Event()
        await asyncio.wait_for(collector.run_channel("fed_press_monetary", stop), 5)
        task = asyncio.create_task(collector.run_channel("bls_cpi", stop))
        for _ in range(50):
            await asyncio.sleep(0)
        stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(run())
    assert [p["status"] for p in polls(conn, "bls_cpi")] == ["ok"]
    assert polls(conn, "fed_press_monetary") == []


def test_a_channel_waits_in_slices_so_a_sleep_never_delays_its_next_poll(conn):
    """The event loop's clock stops while the Mac sleeps, so a wait ends late by the whole sleep. A wait of at most a
    minute, then a fresh look at the wall clock, polls a channel at once after waking."""
    clock = FakeClock(MON_01)
    client = FakeClient(clock, {RENDERS[AAPL][0]: growing(company_render("AAPL", clock))})
    waits = []

    async def wait(stop, seconds):
        waits.append(seconds)
        if len(waits) == 1:
            clock.ms += 3 * 3600 * SECOND             # the Mac slept through the first slice
        elif len(polls(conn, AAPL)) >= 2:
            stop.set()
            return True
        else:
            clock.ms += round(seconds * SECOND)
        await asyncio.sleep(0)
        return False

    async def sleep(seconds):
        clock.ms += round(seconds * SECOND)

    collector = Collector(CONFIG, conn, client, Budget(rules_from_config(CONFIG), lambda: clock.ms, sleep), clock,
                          wait)
    asyncio.run(asyncio.wait_for(collector.run_channel(AAPL, asyncio.Event()), 5))
    assert waits[0] == 60 and all(seconds <= 60 for seconds in waits)
    assert [parse_iso(row["started_at"]) for row in polls(conn, AAPL)] == [MON_01, MON_01 + 3 * 3600 * SECOND]


class VirtualTime:
    """A fake wall clock with waits that end when the driver moves the clock to them, so many minutes of polling run
    in a moment and in a fixed order."""

    def __init__(self, ms):
        self.clock = FakeClock(ms)
        self.sleepers = []
        self.order = itertools.count()

    async def wait(self, stop, seconds):
        if stop is not None and stop.is_set():
            return True
        future = asyncio.get_running_loop().create_future()
        wake = self.clock.ms + max(0, round(seconds * SECOND))
        heapq.heappush(self.sleepers, (wake, next(self.order), future, stop is not None))
        await future
        return stop is not None and stop.is_set()

    async def sleep(self, seconds):
        await self.wait(None, seconds)

    async def drive(self, task, stop, until_ms):
        """Move the clock from wake to wake until until_ms, then set stop and keep waking until task ends. After the
        stop only budget waits still move the clock, so no task sees a far later time."""
        while not task.done():
            for _ in range(20):
                await asyncio.sleep(0)
            if not self.sleepers:
                await asyncio.sleep(0.001)
                continue
            if self.sleepers[0][0] > until_ms:
                stop.set()
            wake, _, future, stoppable = heapq.heappop(self.sleepers)
            if not (stoppable and stop.is_set()):
                self.clock.ms = max(self.clock.ms, wake)
            if not future.done():
                future.set_result(None)


def small_config():
    return dataclasses.replace(
        CONFIG, finnhub=dataclasses.replace(CONFIG.finnhub, company_symbols=("AAPL",)),
        edgar=dataclasses.replace(CONFIG.edgar, enabled=False), bls=dataclasses.replace(CONFIG.bls, feeds=("cpi",)))


def serve(conn, client_routes, minutes, start=at(2026, 10, 12, 9, 59, EDT)):
    """Run the news process's serve on virtual time for some minutes and return its counts or its error."""
    config = small_config()
    time_ = VirtualTime(start)
    client = FakeClient(time_.clock, {prefix: render(time_.clock) for prefix, render in client_routes.items()})
    budget = Budget(rules_from_config(config), lambda: time_.clock.ms, time_.sleep)

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(runner.serve(config, conn, client, stop, time_.clock, budget, time_.wait))
        await time_.drive(task, stop, start + minutes * MINUTE)
        return await task

    return config, asyncio.run(run())


def small_routes():
    return {prefix: (lambda r: lambda clock: growing(r(clock)))(render) for prefix, render in RENDERS.values()}


def test_serve_writes_routed_items_without_scores_and_a_heartbeat_every_30_seconds(conn):
    config, counts = serve(conn, small_routes(), 10)
    rows = items(conn)
    assert counts["new"] > 0 and counts["seed"] == 5
    assert {(row["status"], row["route"]) for row in rows} == {("seed", "none"), ("new", "llm")}
    assert conn.execute("SELECT count(*) FROM item_scores").fetchone()[0] == 0
    assert len(polls(conn, AAPL)) == 11 and len(polls(conn, "fed_speeches")) == 2
    beats = [dict(row) for row in conn.execute("SELECT * FROM news_heartbeat ORDER BY seq")]
    assert len(beats) == 21
    assert [parse_iso(b["ts"]) - parse_iso(a["ts"]) for a, b in pairwise(beats)] == [30 * SECOND] * 20
    assert {(b["healthy"], b["reason"], b["channels_down"], b["config_hash"]) for b in beats} == {
        (0, "finbert is not on", "", config.config_hash)}
    assert [kind for kind, _ in events(conn)] == ["started", "stopped"]
    assert db.get_value(conn, "meta", "config_hash") == config.config_hash
    assert db.get_value(conn, "meta", "haiku_model") == config.llm.model
    assert db.get_value(conn, "meta", "sonnet_model") == config.llm.sonnet.model


def test_serve_writes_config_changed_when_the_hash_moved(conn):
    db.set_values(conn, "meta", {"config_hash": "0" * 64})
    serve(conn, small_routes(), 1)
    assert [kind for kind, _ in events(conn)] == ["config_changed", "started", "stopped"]


def test_serve_writes_a_pause_event_when_the_process_was_frozen(conn):
    start = at(2026, 10, 12, 9, 59, EDT)
    routes = small_routes()

    def freezing(clock):
        render = growing(general_render(clock))

        def answer():
            if clock.ms >= start + 3 * MINUTE and not freezing.done:
                freezing.done = True
                clock.ms += 10 * MINUTE              # the laptop slept during this request
            return render()
        return answer

    freezing.done = False
    routes[GENERAL_PREFIX] = freezing
    serve(conn, routes, 20, start)
    [(_, detail)] = events(conn, "pause")
    seconds = int(re.fullmatch(r"the process was paused for (\d+) s", detail).group(1))
    assert 600 <= seconds <= 660


def test_serve_never_stamps_an_item_before_the_newest_one_after_the_clock_stepped_back(conn):
    start = at(2026, 10, 12, 9, 59, EDT)
    newest = start + 3 * SECOND                       # written before a restart whose wall clock stepped back
    db.insert(conn, "news_items", {
        "source": "finnhub", "channel": GENERAL, "vendor_id": "planted", "kind": "media", "received_at": iso(newest),
        "title": "Fictional planted headline", "tickers": "", "relevance": "macro", "status": "new",
        "title_hash": "0123456789abcdef", "route": "llm"})
    serve(conn, small_routes(), 5, start)
    stamps = [row["received_at"] for row in items(conn)]
    assert len(stamps) > 5 and stamps == sorted(stamps) and stamps[0] == iso(newest)
    assert events(conn, "pause") == []


def test_a_crashing_channel_stops_the_process_with_an_error_after_a_clean_stop(conn):
    routes = small_routes()
    routes[rss.bls_url("cpi")] = lambda clock: (lambda: RuntimeError("fixture failure"))
    with pytest.raises(runner.CollectionFailed, match="bls_cpi failed"):
        serve(conn, routes, 5)
    assert [kind for kind, _ in events(conn)][-1] == "stopped"


def test_the_rest_lock_admits_one_holder(tmp_path):
    path = tmp_path / "lock" / "finnhub_rest.lock"
    with runner.rest_lock(path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        script = (f"import sys\nfrom btc_survivor.v2.news import runner\ntry:\n"
                  f"    with runner.rest_lock({str(path)!r}):\n        sys.exit(0)\n"
                  f"except runner.LockError:\n    sys.exit(3)\n")
        assert subprocess.run([sys.executable, "-c", script], timeout=60).returncode == 3
    with runner.rest_lock(path):
        pass


# The news process as a subprocess, with every socket closed and HttpClient.get answering from the fixtures


WRAPPER = r'''
import atexit, os, socket, sys, time
from pathlib import Path

def closed(*args, **kwargs):
    raise OSError("the network is closed in this test")

socket.socket.connect = closed
socket.socket.connect_ex = closed
socket.create_connection = closed
socket.getaddrinfo = closed

from btc_survivor.v2.news import http

FIXTURES = Path(os.environ["NEWS_FIXTURES"])
FILES = (
    ("https://finnhub.io/api/v1/company-news", "finnhub/company_news_aapl.json"),
    ("https://finnhub.io/api/v1/news", "finnhub/general_news.json"),
    ("https://www.sec.gov/cgi-bin/browse-edgar", "news/edgar_atom_8k.xml"),
    ("https://data.sec.gov/submissions/", "news/edgar_submissions_aapl.json"),
    ("https://www.federalreserve.gov/feeds/press_all", "news/fed_press_all.xml"),
    ("https://www.federalreserve.gov/feeds/press_monetary", "news/fed_press_monetary.xml"),
    ("https://www.federalreserve.gov/feeds/speeches", "news/fed_speeches.xml"),
    ("https://www.bls.gov/feed/", "news/bls_cpi.rss"),
)
CALLS = []
atexit.register(lambda: Path(os.environ["NEWS_CALLS"]).write_text(str(len(CALLS))))

def fake_get(self, url, validators=None):
    self.request(url, validators)
    CALLS.append(url)
    if os.environ.get("NEWS_BOOM") and url.startswith(os.environ["NEWS_BOOM"]):
        raise RuntimeError("fixture failure " + os.environ["NEWS_BOOM_TEXT"])
    body = (FIXTURES / next(name for prefix, name in FILES if url.startswith(prefix))).read_bytes()
    return http.Response(http.OK, 200, body, len(body), 1, int(time.time() * 1000), http.Validators())

http.HttpClient.get = fake_get

# With NEWS_SCORING set, the real pin check runs on the test's fake FinBERT files, then a fake model and a fake SDK
# client stand in for FinBERT and the API, so no model runs and no request is sent. The fake SDK fails every call,
# which the scorer books at its reservation.
if os.environ.get("NEWS_SCORING"):
    import anthropic
    from btc_survivor.v2.news import finbert, llm_client

    class FakeModel:
        def score(self, headline, text=None):
            return finbert.Probs(0.2, 0.1, 0.7, 1, 12)

    class FakeMessages:
        async def create(self, **request):
            raise anthropic.AnthropicError("the fake SDK sends nothing")

        async def count_tokens(self, **request):
            raise anthropic.AnthropicError("the fake SDK counts nothing")

    class FakeSdk:
        messages = FakeMessages()

        async def close(self):
            Path(os.environ["NEWS_CALLS"]).with_name("sdk_closed.txt").write_text("closed")

    finbert.Finbert.load = classmethod(lambda cls, config, session_factory=None: (finbert.verify(config),
                                                                                    FakeModel())[1])
    llm_client.sdk_client = lambda timeout_s: FakeSdk()

from btc_survivor.v2.news import runner
sys.exit(runner.main(sys.argv[1:]))
'''


def edit(text, section, key, value):
    """Set one key in one section of a TOML text, refusing to write if it is not there."""
    start = re.search(rf"(?m)^\[{re.escape(section)}\][ \t]*(#.*)?$", text).start()
    tail, found = re.subn(rf"(?m)^{re.escape(key)} = [^#\n]*", f"{key} = {value} ", text[start:], count=1)
    assert found == 1, (section, key)
    return text[:start] + tail


def process_config(folder, secrets=None, finnhub=True, pinned=False):
    """A secrets file of fake canaries and a copy of the news config whose every path points into folder, with one
    company and one CIK, and when pinned with fake FinBERT files in folder/models that [finbert] pins."""
    path = folder / "secrets.env"
    path.write_text(secrets if secrets is not None else f"{FINNHUB}={CANARY_TOKEN}\n{CONTACT}={CANARY_CONTACT}\n",
                    encoding="utf-8")
    path.chmod(0o600)
    text = NEWS.read_text(encoding="utf-8")
    for section, key, value in (
            ("news", "data_dir", '"data/news"'), ("news", "secrets_file", '"secrets.env"'),
            ("news", "rest_lock_file", '"lock/finnhub_rest.lock"'), ("finnhub", "company_symbols", '["AAPL"]'),
            ("finnhub", "enabled", "true" if finnhub else "false"), ("edgar", "ciks", "{ AAPL = 320193 }"),
            ("llm.caps", "ledger_file", '"ledger/llm_ledger.db"'),
            ("llm.caps", "ledger_lock_file", '"ledger/llm_ledger.lock"')):
        text = edit(text, section, key, value)
    if pinned:
        models = folder / "models"
        models.mkdir()
        manifest = {"revision": "0" * 40, "labels": ["positive", "negative", "neutral"],
                    "max_position_embeddings": 512}
        for role, name, data in (("model", "model.onnx", b"a fake model, never run"),
                                 ("tokenizer", "tokenizer.json", b'{"fake": true}')):
            (models / name).write_bytes(data)
            manifest[role] = {"sha256": hashlib.sha256(data).hexdigest()}
            text = edit(text, "finbert", f"{role}_file", f'"models/{name}"')
            text = edit(text, "finbert", f"{role}_sha256", f'"{manifest[role]["sha256"]}"')
        (models / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        text = edit(text, "finbert", "manifest_file", '"models/manifest.json"')
        text = edit(text, "finbert", "source_revision", f'"{manifest["revision"]}"')
    assert "~" not in text
    config = folder / "news.toml"
    config.write_text(text, encoding="utf-8")
    return config


def process_env(folder, **extra):
    env = {name: value for name, value in os.environ.items() if name.upper() not in REFUSED}
    return {**env, "HOME": str(folder), "NEWS_FIXTURES": str(FIXTURES), "NEWS_CALLS": str(folder / "calls.txt"),
            **extra}


def start(folder, *args, **extra):
    with (folder / "out.txt").open("w") as out, (folder / "err.txt").open("w") as err:
        return subprocess.Popen([sys.executable, "-c", WRAPPER, *args], stdout=out, stderr=err, cwd=folder,
                                env=process_env(folder, **extra))


def run_process(folder, *args, **extra):
    proc = start(folder, *args, **extra)
    proc.wait(timeout=60)
    return proc.returncode, (folder / "out.txt").read_text(), (folder / "err.txt").read_text()


def read_db(path, sql):
    conn = db.open_reader(path)
    try:
        return [tuple(row) for row in conn.execute(sql)]
    finally:
        conn.close()


def wait_for(check, timeout=40):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if check():
                return True
        except (db.NewsDbError, sqlite3.Error):
            pass
        time.sleep(0.1)
    return False


def test_dry_run_loads_config_and_secrets_and_opens_nothing_on_the_network(tmp_path):
    config = process_config(tmp_path)
    code, out, err = run_process(tmp_path, str(config), "--dry-run")
    assert (code, err) == (0, "")
    assert out.splitlines() == ["news dry run", "verdict: pass", "channels: 11", "sources enabled: 4",
                                "lock: taken and released", "scoring: waits for the finbert pins"]
    assert (tmp_path / "calls.txt").read_text() == "0"
    assert not (tmp_path / "data").exists() and CANARY_TOKEN not in out
    with runner.rest_lock(tmp_path / "lock" / "finnhub_rest.lock"):
        pass
    code, out, _ = run_process(tmp_path, str(process_config(tmp_path, finnhub=False)), "--dry-run")
    assert code == 0 and "lock: not needed" in out and "channels: 9" in out


@pytest.mark.parametrize("case", ["missing contact", "held lock", "proxy", "stock database"])
def test_expected_failures_exit_with_code_2_naming_no_value(tmp_path, case):
    secrets = f"{FINNHUB}={CANARY_TOKEN}\n" if case == "missing contact" else None
    config = process_config(tmp_path, secrets)
    extra = {"https_proxy": "http://127.0.0.1:9"} if case == "proxy" else {}
    if case == "stock database":
        folder = tmp_path / "data" / "news"
        folder.mkdir(parents=True)
        other = sqlite3.connect(folder / "news.db")
        other.executescript("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                            "INSERT INTO meta VALUES ('kind', 'stocks');")
        other.close()
    if case == "held lock":
        with runner.rest_lock(tmp_path / "lock" / "finnhub_rest.lock"):
            code, out, err = run_process(tmp_path, str(config), "--dry-run")
    else:
        code, out, err = run_process(tmp_path, str(config), *([] if case == "stock database" else ["--dry-run"]),
                                     "--log-dir", str(tmp_path / "logs"), **extra)
    assert code == 2 and out == "" and err.startswith("error: ")
    assert CANARY_TOKEN not in err and CANARY_CONTACT not in err
    expected = {"missing contact": f"{CONTACT} is missing from", "held lock": "holds the Finnhub REST lock",
                "proxy": "unset https_proxy first", "stock database": "meta.kind is 'stocks'"}[case]
    assert expected in err
    assert (tmp_path / "calls.txt").read_text() == "0"


def test_the_news_process_collects_beats_and_stops_cleanly_on_sigterm(tmp_path):
    config = process_config(tmp_path)
    logs = tmp_path / "logs"
    path = tmp_path / "data" / "news" / "news.db"
    proc = start(tmp_path, str(config), "--log-dir", str(logs))
    try:
        assert wait_for(lambda: read_db(path, "SELECT count(DISTINCT channel) FROM news_polls")[0][0] >= 10
                        and read_db(path, "SELECT count(*) FROM news_heartbeat")[0][0] >= 1), \
            (tmp_path / "err.txt").read_text()
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
    out, err = (tmp_path / "out.txt").read_text(), (tmp_path / "err.txt").read_text()
    assert proc.returncode == 0, err
    assert out == "" and "heartbeat healthy=0" in err
    assert [kind for (kind,) in read_db(path, "SELECT kind FROM collector_events ORDER BY id")] == ["started",
                                                                                                  "stopped"]
    loaded = load_news_config(config)
    beats = read_db(path, "SELECT healthy, reason, config_hash, channels_down FROM news_heartbeat")
    assert {beat[:3] for beat in beats} == {(0, "finbert is not on", loaded.config_hash)}
    assert set(read_db(path, "SELECT status, route FROM news_items")) == {("seed", "none")}
    assert read_db(path, "SELECT count(*) FROM item_scores") == [(0,)]
    assert int((tmp_path / "calls.txt").read_text()) >= 10
    for folder in (path.parent, logs, tmp_path / "lock"):
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700, folder
    for file in (path, logs / "btc_survivor.log", tmp_path / "lock" / "finnhub_rest.lock"):
        assert stat.S_IMODE(file.stat().st_mode) == 0o600, file
    written = [*path.parent.iterdir(), logs / "btc_survivor.log", tmp_path / "out.txt", tmp_path / "err.txt"]
    for file in written:
        data = file.read_bytes()
        assert CANARY_TOKEN.encode() not in data and CANARY_CONTACT.encode() not in data, file


def test_a_crash_is_logged_redacted_and_exits_with_code_1_after_a_clean_stop(tmp_path):
    config = process_config(tmp_path)
    path = tmp_path / "data" / "news" / "news.db"
    code, out, err = run_process(tmp_path, str(config), "--log-dir", str(tmp_path / "logs"),
                                 NEWS_BOOM="https://www.bls.gov/feed/", NEWS_BOOM_TEXT=CANARY_TOKEN)
    assert code == 1 and out == ""
    assert "fixture failure [redacted]" in err and CANARY_TOKEN not in err
    assert "failed, so the news process stops" in err
    assert [kind for (kind,) in read_db(path, "SELECT kind FROM collector_events ORDER BY id")] == ["started",
                                                                                                  "stopped"]
    assert CANARY_TOKEN not in (tmp_path / "logs" / "btc_survivor.log").read_text()


# The news process once [finbert] pins the export, with the fake model and SDK client of WRAPPER.


def scoring_secrets():
    return f"{FINNHUB}={CANARY_TOKEN}\n{ANTHROPIC}={CANARY_KEY}\n{CONTACT}={CANARY_CONTACT}\n"


def test_a_pinned_dry_run_checks_the_pins_the_key_and_the_finbert_files(tmp_path):
    config = process_config(tmp_path, scoring_secrets(), pinned=True)
    code, out, err = run_process(tmp_path, str(config), "--dry-run", NEWS_SCORING="1")
    assert (code, err) == (0, "")
    assert out.splitlines()[-1] == "scoring: pins checked"
    assert not (tmp_path / "data").exists() and not (tmp_path / "ledger").exists()


@pytest.mark.parametrize("case", ["missing key", "finbert mismatch", "unpinned prompt", "partly pinned"])
def test_a_pinned_start_that_cannot_score_exits_with_code_2_naming_no_value(tmp_path, case):
    config = process_config(tmp_path, scoring_secrets() if case != "missing key" else None, pinned=True)
    text = config.read_text(encoding="utf-8")
    if case == "unpinned prompt":
        config.write_text(edit(text, "llm", "prompt_sha256", '""'), encoding="utf-8")
    if case == "partly pinned":
        config.write_text(edit(text, "finbert", "model_sha256", '""'), encoding="utf-8")
    if case == "finbert mismatch":
        (tmp_path / "models" / "model.onnx").write_bytes(b"a model changed after it was pinned")
    code, out, err = run_process(tmp_path, str(config), "--log-dir", str(tmp_path / "logs"), NEWS_SCORING="1")
    assert code == 2 and out == "" and err.startswith("error: ")
    expected = {"missing key": f"missing from {tmp_path / 'secrets.env'}: {ANTHROPIC}",
                "finbert mismatch": "the model file model.onnx has sha256", "unpinned prompt": "prompt_sha256",
                "partly pinned": "[finbert] model_sha256 is not pinned"}[case]
    assert expected in err
    assert CANARY_TOKEN not in err and CANARY_KEY not in err and CANARY_CONTACT not in err
    assert (tmp_path / "calls.txt").read_text() == "0" and not (tmp_path / "data").exists()


def test_a_pinned_news_process_scores_beside_collection_and_stops_cleanly_on_sigterm(tmp_path):
    config = process_config(tmp_path, scoring_secrets(), pinned=True)
    logs = tmp_path / "logs"
    path = tmp_path / "data" / "news" / "news.db"
    proc = start(tmp_path, str(config), "--log-dir", str(logs), NEWS_SCORING="1")
    loaded = load_news_config(config)
    # The fake client fails every call, so Sonnet's row is failed once the live call has turned it on, and
    # skipped_off while the committed config keeps it off.
    sonnet = "failed" if loaded.llm.sonnet.enabled else "skipped_off"
    names = {loaded.llm.scorer_name: "failed", loaded.llm.sonnet.scorer_name: sonnet, "finbert_v1": "scored"}
    try:
        assert wait_for(lambda: read_db(path, "SELECT count(DISTINCT channel) FROM news_polls")[0][0] >= 10
                        and read_db(path, "SELECT count(*) FROM news_heartbeat")[0][0] >= 1), \
            (tmp_path / "err.txt").read_text()
        # One live official item, as the collector would write it, reaches every scorer of the running process.
        writer = db.open_writer(path)
        with db.transaction(writer):
            seq = db.insert(writer, "news_items", {
                "source": "fed", "channel": "fed_press_all", "vendor_id": "planted-1", "kind": "fed_press",
                "received_at": iso(int(time.time() * 1000)), "detected_at": None, "vendor_ts": None,
                "title": "Fictional board notice", "summary": "A fictional notice for this test.", "url": None,
                "publisher": None, "tickers": "MACRO", "relevance": "title", "status": "new", "filter_reason": None,
                "title_hash": "1" * 16, "items_8k": None, "route": "llm"})
        writer.close()
        scored = f"SELECT scorer, status FROM item_scores WHERE item_seq = {seq}"
        assert wait_for(lambda: dict(read_db(path, scored)) == names), (read_db(path, scored),
                                                                         (tmp_path / "err.txt").read_text())
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
    out, err = (tmp_path / "out.txt").read_text(), (tmp_path / "err.txt").read_text()
    assert proc.returncode == 0, err
    assert out == "" and "heartbeat healthy=1" in err and "collects only" not in err
    # The fake SDK fails the call, which is booked at its reservation and never sent again.
    calls = 2 if loaded.llm.sonnet.enabled else 1
    assert read_db(path, f"SELECT c.status, c.cost_basis, c.cost_usd = c.reserved_usd AND c.reserved_usd > 0 "
                         f"FROM llm_calls c JOIN llm_call_items i USING (call_id) WHERE i.item_seq = {seq}") == [
        ("server", "reserved", 1)] * calls
    beats = read_db(path, "SELECT healthy, reason, config_hash, haiku_state, sonnet_state, finbert_state "
                          "FROM news_heartbeat")
    assert set(beats) == {(1, None, loaded.config_hash, "on", "on" if loaded.llm.sonnet.enabled else "off", "on")}
    assert (tmp_path / "sdk_closed.txt").read_text() == "closed"
    ledger = tmp_path / "ledger" / "llm_ledger.db"
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600
    assert stat.S_IMODE(ledger.parent.stat().st_mode) == 0o700
    written = [*path.parent.iterdir(), *ledger.parent.iterdir(), logs / "btc_survivor.log", tmp_path / "out.txt",
               tmp_path / "err.txt"]
    for file in written:
        data = file.read_bytes()
        for canary in (CANARY_TOKEN, CANARY_KEY, CANARY_CONTACT):
            assert canary.encode() not in data, file
