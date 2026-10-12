"""The news collector of PLAN_V2 sections 7.2 to 7.5, which polls every enabled channel on its schedule and writes
each item once, in receive order, with its route and without a score.

Each channel polls at the times schedule.py gives, through the budget of its host and the one HTTP client, and a
failed poll comes due again after the capped backoff. The receive stamp comes from one clock that never goes
backwards, taken right after a body is read, and parsing, deduplication, the prefilter, the insert and the commit
follow with no await between them, so seq order equals received_at order. A channel's first successful poll writes
seeds, a poll after a gap longer than the cadences allow writes backlog, and a new 8-K or monetary policy release is
held until its documents arrive, while other channels skip it and their sightings become duplicate rows. While the
response Date headers disagree with the local clock, every poll is recorded as clock_skew and counts as failed.

Two choices go beyond the plan's text. Filings skip the headline hash and near duplicate steps, because their titles
repeat while the accession number already tells them apart. A document that fails twice, or that cannot be parsed,
leaves the item's body marked truncated, without that text, and writes a doc_failed event.

Waits run in slices of at most a minute, each followed by a fresh look at the wall clock, because the event loop's
clock stops while the Mac sleeps, so a long wait would otherwise end late by the whole sleep. The start and cadence of
each channel's last successful poll are also kept in meta, so a restart after news_polls was pruned, such as between
two FOMC statement days, still measures the gap from it.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import statistics
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Protocol, TypeVar

from btc_survivor.clock import MS_PER_HOUR, MS_PER_SECOND, LiveClock, iso, parse_iso
from btc_survivor.v2.config import NewsConfig
from btc_survivor.v2.news import db, dedupe, edgar, edgar_docs, finnhub_news, health, http, rss
from btc_survivor.v2.news.budget import Budget, failure_delay_s
from btc_survivor.v2.news.edgar_docs import Body
from btc_survivor.v2.news.http import Response, Validators
from btc_survivor.v2.news.prefilter import Prefilter, Tags, route
from btc_survivor.v2.news.rss import RawItem
from btc_survivor.v2.news.schedule import EDGAR, FED, FINNHUB, Channel, Schedule
from btc_survivor.v2.news.text import title_hash
from btc_survivor.v2.redact import redact

log = logging.getLogger(__name__)
T = TypeVar("T")

PARSE_ERROR, CLOCK_SKEW = "parse_error", "clock_skew"
SUCCESS = (http.OK, http.NOT_MODIFIED)
NEW, SEED, BACKLOG, FILTERED = "new", "seed", "backlog", "filtered"
DUP, SILENT, SKIPPED = "dup", "silent", "skipped"
STALE_VENDOR = "stale_vendor"
HELD_KINDS = ("8k", "fed_monetary")          # a new item of these kinds waits for its documents
TITLE_DEDUPE_EXEMPT = ("8k", "8k_a")         # the accession number already tells filings apart
DOC_ATTEMPTS = 2
DOC_CHANNELS = {EDGAR: "edgar_docs", FED: "fed_pages"}   # news_polls rows of document fetches, with cadence 0
FIRST_POLL = "first_poll."
CURSOR = "cursor.finnhub_general"
GENERAL = "finnhub_general"
LAST_OK = "last_ok."                         # meta keys of each channel's last successful poll
LONG_WAIT_MS = 60_000                        # a budget wait this long also writes a budget_wait event
WAIT_SLICE_S = 60.0                          # a channel reads the wall clock again at least this often
DETAIL_CHARS = 200


class Client(Protocol):
    async def fetch(self, url: str, validators: Validators | None = None) -> Response: ...


def vendor_status(received_at: str, vendor_ts: str | None, max_hours: float) -> str | None:
    """stale_vendor when an item's vendor time is more than max_hours before its receive time, the final status every
    scorer gives it without scoring, else None. An item without a vendor time is never stale."""
    if vendor_ts is None:
        return None
    return STALE_VENDOR if parse_iso(received_at) - parse_iso(vendor_ts) > max_hours * MS_PER_HOUR else None


def _parsed(parse: Callable[..., T], page: bytes | None, *args: object) -> T | None:
    """parse(page, *args), or None when the page did not arrive or could not be parsed, which a held item treats as a
    missing document instead of a crash that would recur at every restart."""
    if page is None:
        return None
    try:
        return parse(page, *args)
    except Exception as exc:  # a parser bug on one public page must never stop the collector
        log.warning("a fetched document could not be parsed (%s)", type(exc).__name__)
        return None


async def wait_or_stop(stop: asyncio.Event, seconds: float) -> bool:
    """Wait up to seconds and say whether stop was set, returning as soon as it is."""
    try:
        await asyncio.wait_for(stop.wait(), max(0.0, seconds))
    except asyncio.TimeoutError:
        return False
    return True


class ClockCheck:
    """The median difference between response Date headers and the local clock over the last responses, which is
    clean while it stays within the limit."""

    def __init__(self, max_seconds: float, samples: int) -> None:
        self.max_ms = max_seconds * MS_PER_SECOND
        self.offsets: deque[int] = deque(maxlen=samples)

    def add(self, date_ms: int | None, local_ms: int) -> None:
        """Count one response, if it carried a readable Date header."""
        if date_ms is not None:
            self.offsets.append(date_ms - local_ms)

    @property
    def offset_ms(self) -> float:
        return statistics.median(self.offsets) if self.offsets else 0.0

    @property
    def clean(self) -> bool:
        return abs(self.offset_ms) <= self.max_ms


@dataclass
class ChannelState:
    """What the collector keeps about one channel between polls, restored from news_polls and meta at start. The
    validators belong to the URL they came from, so a company news call for a new date never sends yesterday's."""

    last_ms: int | None = None
    last_cadence_s: int = 0
    ok_ms: int | None = None
    ok_cadence_s: int = 0
    failures: int = 0
    seeded: bool = False
    url: str = ""
    validators: Validators = field(default_factory=Validators)


@dataclass
class Held:
    """A new item waiting for its documents, with the channel and stamp of its first sighting and the first stamp of
    each other channel that saw it meanwhile."""

    raw: RawItem
    channel: str
    detected_ms: int
    tags: Tags
    sightings: dict[str, int] = field(default_factory=dict)


class Collector:
    """Polls the enabled channels of a news config into news.db through the one writer connection. Every method that
    writes runs on the event loop, and only poll and the document fetches await."""

    def __init__(self, config: NewsConfig, conn: sqlite3.Connection, client: Client, budget: Budget,
                 clock: LiveClock | None = None,
                 wait: Callable[[asyncio.Event, float], Awaitable[bool]] = wait_or_stop) -> None:
        self.config = config
        self.conn = conn
        self.client = client
        self.budget = budget
        self.clock = clock or LiveClock()
        self.wait = wait
        self.schedule = Schedule(config)
        self.prefilter = Prefilter(config)
        self.clock_check = ClockCheck(config.news.clock_skew_max_seconds, config.news.clock_skew_samples)
        self.channels = {channel.name: channel for channel in self.schedule.channels}
        self.states = {name: self._restore(name) for name in self.channels}
        self.held: dict[tuple[str, str], Held] = {}
        self.cursor = int(db.get_value(conn, "meta", CURSOR) or 0)
        self.counts: Counter[str] = Counter()
        self._seen: dict[str, list[dedupe.Seen]] = {}

    def _restore(self, name: str) -> ChannelState:
        """A channel's state from its latest poll, its latest successful poll, kept in meta too once news_polls has
        pruned it, and its first poll mark in meta."""
        state = ChannelState(seeded=db.get_value(self.conn, "meta", FIRST_POLL + name) is not None)
        last = self.conn.execute("SELECT started_at, cadence_s FROM news_polls WHERE channel = ? "
                                 "ORDER BY seq DESC LIMIT 1", (name,)).fetchone()
        ok = self.conn.execute("SELECT started_at, cadence_s FROM news_polls WHERE channel = ? AND status IN (?, ?) "
                               "ORDER BY seq DESC LIMIT 1", (name, *SUCCESS)).fetchone()
        kept = db.get_value(self.conn, "meta", LAST_OK + name)
        if ok is None and kept is not None:
            ok = kept.split()
        if last is not None:
            state.last_ms, state.last_cadence_s = parse_iso(last[0]), last[1]
        if ok is not None:
            state.ok_ms, state.ok_cadence_s = parse_iso(ok[0]), int(ok[1])
        return state

    def last_ok(self) -> dict[str, tuple[int, int] | None]:
        """The start and cadence of each channel's last successful poll, for the outage list."""
        return {name: None if s.ok_ms is None else (s.ok_ms, s.ok_cadence_s) for name, s in self.states.items()}

    def next_poll(self, name: str, now_ms: int) -> int | None:
        """When a channel polls next, at or after now_ms, by its schedule after a success and no sooner than the
        backoff after failures. None means it stays dormant for good."""
        state = self.states[name]
        if state.failures and state.last_ms is not None:
            delay_s = failure_delay_s(state.last_cadence_s, state.failures,
                                      self.config.news.failure_backoff_cap_seconds)
            now_ms = max(now_ms, state.last_ms + round(delay_s * MS_PER_SECOND))
        return self.schedule.next_due(name, state.last_ms, now_ms)

    async def run_channel(self, name: str, stop: asyncio.Event) -> None:
        """Poll one channel whenever it is due until stop is set or it stays dormant for good, waiting in slices of
        at most a minute and reading the wall clock after each, so a sleep of the Mac never delays a poll."""
        while not stop.is_set():
            now = self.clock.now_ms()
            due = self.next_poll(name, now)
            if due is None:
                log.info("channel %s stays dormant from now on", name)
                return
            if due > now:
                if await self.wait(stop, min(WAIT_SLICE_S, (due - now) / MS_PER_SECOND)):
                    return
                continue
            await self.poll(name, due)

    async def poll(self, name: str, due_ms: int) -> None:
        """Poll one channel now, as due at due_ms, write what it returned and then fetch the documents of any item it
        holds."""
        channel, state = self.channels[name], self.states[name]
        cadence = self.schedule.cadence(name, due_ms)
        if cadence is None:
            raise ValueError(f"{name} is dormant at its due time")
        url = self._url(channel)
        group = http.host_group(url)
        wait_ms = await self.budget.acquire(group)
        started = self.clock.stamp()
        response = await self.client.fetch(url, state.validators if state.url == url else None)
        received = self.clock.stamp()
        for key in self._receive(channel, state, url, cadence, group, started, received, wait_ms, response):
            await self._complete(key)

    def _receive(self, channel: Channel, state: ChannelState, url: str, cadence: int, group: str, started: int,
                 received: int, wait_ms: int, response: Response) -> list[tuple[str, str]]:
        """Everything a poll does once its body is read, with no await, ending with its news_polls row. It returns
        the items held for their documents."""
        self._observe(group, response, received, wait_ms, channel.name)
        status, detail, items = response.status, response.detail, []
        if response.ok and not self.clock_check.clean:
            status = CLOCK_SKEW
            detail = f"the median Date header offset is {self.clock_check.offset_ms / MS_PER_SECOND:.1f} s"
        elif status == http.OK:
            try:
                items = self._parse(channel, response.body)
            except Exception as exc:  # a body no parser can read is a failed poll, never a crash
                status, detail = PARSE_ERROR, f"{type(exc).__name__}: {exc}"
        counts: Counter[str] = Counter()
        held: list[tuple[str, str]] = []
        if status == http.OK:
            first = not state.seeded
            backlog = not first and self._backlog(channel.name, state, cadence, started, wait_ms)
            for raw in items:
                counts[self._item(channel.name, raw, received, first, backlog, held)] += 1
            state.url, state.validators = url, response.validators
            if channel.name == GENERAL:
                self._advance_cursor(items)
            if first:
                db.set_values(self.conn, "meta", {FIRST_POLL + channel.name: iso(started)})
                state.seeded = True
        if status in SUCCESS:
            state.ok_ms, state.ok_cadence_s, state.failures = started, cadence, 0
            db.set_values(self.conn, "meta", {LAST_OK + channel.name: f"{iso(started)} {cadence}"})
        else:
            state.failures += 1
            log.warning("poll of %s failed with %s, failure %d in a row", channel.name, status, state.failures)
        state.last_ms, state.last_cadence_s = started, cadence
        self.counts[f"polls {status}"] += 1
        self._poll_row(channel.name, started, cadence, status, response, wait_ms, detail,
                       len(items) if status in SUCCESS else None, counts)
        return held

    def _observe(self, group: str, response: Response, received: int, wait_ms: int, name: str) -> None:
        """Count a response toward the clock check and the budget, writing an event when the clock check changes,
        a host pauses or a wait was long."""
        was_clean = self.clock_check.clean
        self.clock_check.add(response.date_ms, received)
        if self.clock_check.clean != was_clean:
            offset = self.clock_check.offset_ms / MS_PER_SECOND
            detail = "clean again" if self.clock_check.clean else f"median Date header offset {offset:.1f} s"
            self._event(CLOCK_SKEW, detail)
            log.warning("clock check: %s", detail)
        if self.budget.report(group, response.http_status):
            pause_s = self.budget.paused_ms(group) / MS_PER_SECOND
            self._event("pause", f"{group} paused for {pause_s:.0f} s after HTTP {response.http_status}")
            log.warning("%s answered HTTP %s, so every %s channel pauses for %.0f s", name, response.http_status,
                        group, pause_s)
        if wait_ms >= LONG_WAIT_MS:
            self._event("budget_wait", f"{name} waited {wait_ms / MS_PER_SECOND:.1f} s for the {group} budget")

    def _event(self, kind: str, detail: str) -> None:
        db.insert(self.conn, "collector_events", {"ts": iso(self.clock.stamp()), "kind": kind,
                                                  "detail": redact(detail)[:DETAIL_CHARS]})

    def _poll_row(self, name: str, started: int, cadence: int, status: str, response: Response, wait_ms: int,
                  detail: str, n_raw: int | None, counts: Counter[str] | None = None) -> None:
        """The news_polls row of one poll or document fetch, its detail redacted, with item counts only when it
        succeeded."""
        row = {"channel": name, "started_at": iso(started), "finished_at": iso(self.clock.stamp()),
               "cadence_s": cadence, "status": status, "http_status": response.http_status,
               "bytes": response.wire_bytes, "duration_ms": response.duration_ms, "budget_wait_ms": wait_ms,
               "detail": redact(detail)[:DETAIL_CHARS] or None}
        if n_raw is not None and counts is not None:
            row.update(n_raw=n_raw, n_new=counts[NEW], n_backlog=counts[BACKLOG], n_dup=counts[DUP],
                       n_filtered=counts[FILTERED])
        db.insert(self.conn, "news_polls", row)

    def _backlog(self, name: str, state: ChannelState, cadence: int, started: int, wait_ms: int) -> bool:
        """Whether the time since the channel's previous successful poll, less any dormant stretch, exceeds the longer
        of the two cadences plus backlog_gap_seconds and this poll's budget wait. A channel whose previous success is
        no longer in news_polls is behind by definition."""
        if state.ok_ms is None:
            return True
        gap = health.active_gap_ms(self.schedule, name, state.ok_ms, state.ok_cadence_s, started)
        allowed = (max(state.ok_cadence_s, cadence) + self.config.news.backlog_gap_seconds) * MS_PER_SECOND
        return gap > allowed + wait_ms

    def _item(self, name: str, raw: RawItem, received: int, first: bool, backlog: bool,
              held: list[tuple[str, str]]) -> str:
        """Steps 1 to 6 of section 7.5 for one parsed item, inserting it, writing a duplicate row, holding it for its
        documents or writing nothing, and returning what happened."""
        key = (raw.source, raw.vendor_id)
        if key in self.held:
            if name != self.held[key].channel:
                self.held[key].sightings.setdefault(name, received)
            return SKIPPED
        known = self._known(raw.source, raw.vendor_id, name)
        if known is not None:
            seq, tickers, channels = known
            if dedupe.repoll(channels, name) == dedupe.SILENT:
                return SILENT
            self._dup(seq, name, raw, received, dedupe.VENDOR_ID, None, tickers, self.prefilter.tag(raw))
            return DUP
        tags = self.prefilter.tag(raw)
        if not first and raw.kind not in TITLE_DEDUPE_EXEMPT:  # a first poll is all seed, step 2 before steps 4 and 5
            found = dedupe.find_duplicate(raw.source, received, raw.title, self._recent(raw.source, received),
                                          self.config.dedupe)
            if found is not None:
                original = self.conn.execute("SELECT tickers FROM news_items WHERE seq = ?", (found.seq,)).fetchone()
                self._dup(found.seq, name, raw, received, found.method, found.jaccard, original[0], tags)
                return DUP
        if first:
            status = SEED
        elif tags.filtered:
            status = FILTERED
        elif backlog and raw.kind not in self.config.news.backlog_exempt_kinds:
            status = BACKLOG
        else:
            status = NEW
        if status == NEW and raw.kind in HELD_KINDS:
            self.held[key] = Held(raw, name, received, tags)
            held.append(key)
            return NEW
        self._insert(name, raw, received, None, status, tags, None)
        return status

    def _known(self, source: str, vendor_id: str, channel: str) -> tuple[int, str, set[str]] | None:
        """The original seq, its tickers and every channel that has returned a known vendor id, from its item or from
        the duplicate rows that carry the id, or None for an id never seen."""
        row = self.conn.execute("SELECT seq, channel, tickers FROM news_items WHERE source = ? AND vendor_id = ?",
                                (source, vendor_id)).fetchone()
        if row is not None:
            channels = {row[1]}
            if channel not in channels:
                channels |= {r[0] for r in self.conn.execute(
                    "SELECT channel FROM news_dups WHERE item_seq = ? AND vendor_id = ?", (row[0], vendor_id))}
            return row[0], row[2], channels
        rows = self.conn.execute("SELECT d.item_seq, d.channel, i.tickers FROM news_dups d JOIN news_items i "
                                 "ON i.seq = d.item_seq WHERE d.vendor_id = ? AND i.source = ? ORDER BY d.id",
                                 (vendor_id, source)).fetchall()
        if not rows:
            return None
        return rows[0][0], rows[0][2], {r[1] for r in rows}

    def _recent(self, source: str, received: int) -> list[dedupe.Seen]:
        """The items of one source inside the longer deduplication window, read once and then kept in step with
        every insert, since this process is the only writer."""
        window_ms = max(self.config.dedupe.hash_window_hours, self.config.dedupe.near_dup_window_hours) * MS_PER_HOUR
        if source not in self._seen:
            rows = self.conn.execute("SELECT seq, received_at, title FROM news_items WHERE source = ? "
                                     "AND received_at >= ? ORDER BY seq", (source, iso(received - window_ms)))
            self._seen[source] = [dedupe.seen(seq, source, parse_iso(at), title) for seq, at, title in rows]
        seen = self._seen[source]
        if seen and seen[0].received_ms < received - window_ms:
            self._seen[source] = seen = [s for s in seen if s.received_ms >= received - window_ms]
        return seen

    def _dup(self, seq: int, name: str, raw: RawItem, seen_ms: int, method: str, jaccard: float | None,
             original_tickers: str, tags: Tags) -> None:
        """One news_dups row pointing at the original, with the tickers the duplicate would have added, kept for
        measurement only."""
        extra = dedupe.new_tickers(tags.tickers, [t for t in original_tickers.split(",") if t])
        db.insert(self.conn, "news_dups", {"item_seq": seq, "channel": name, "vendor_id": raw.vendor_id,
                                           "seen_at": iso(seen_ms), "method": method, "jaccard": jaccard,
                                           "new_tickers": extra or None})
        self.counts[DUP] += 1

    def _insert(self, name: str, raw: RawItem, received: int, detected: int | None, status: str, tags: Tags,
                body: Body | None) -> int:
        """Insert one item with its route, and its body when it has one, in one transaction, returning its seq."""
        row = {
            "source": raw.source, "channel": name, "vendor_id": raw.vendor_id, "kind": raw.kind,
            "received_at": iso(received), "detected_at": None if detected is None else iso(detected),
            "vendor_ts": raw.vendor_ts, "title": raw.title, "summary": raw.summary or None, "url": raw.url or None,
            "publisher": raw.publisher or None, "tickers": tags.tickers_text, "relevance": tags.relevance,
            "status": status, "filter_reason": tags.filter_reason, "title_hash": title_hash(raw.title),
            "items_8k": raw.items_8k or None,
            "route": route(raw.source, raw.kind, status, tags.relevance, self.config.finnhub.send_to_llm),
        }
        with db.transaction(self.conn):
            seq = db.insert(self.conn, "news_items", row)
            if body is not None:
                db.insert(self.conn, "news_bodies", {"item_seq": seq, "text": body.text,
                                                     "chars_total": body.chars_total,
                                                     "truncated": int(body.truncated), "doc_urls": body.doc_urls})
        if raw.source in self._seen:
            self._seen[raw.source].append(dedupe.seen(seq, raw.source, received, raw.title))
        self.counts[status] += 1
        return seq

    async def _complete(self, key: tuple[str, str]) -> None:
        """Fetch a held item's documents, then stamp and insert it with its body, the first sighting kept as
        detected_at, and write a duplicate row for each other channel that saw it meanwhile."""
        held = self.held[key]
        try:
            body = await (self._filing_body(held.raw) if held.raw.kind == "8k" else self._page_body(held.raw))
        except BaseException:
            del self.held[key]
            raise
        received = self.clock.stamp()
        del self.held[key]
        seq = self._insert(held.channel, held.raw, received, held.detected_ms, NEW, held.tags, body)
        for name, seen_ms in held.sightings.items():
            self._dup(seq, name, held.raw, seen_ms, dedupe.VENDOR_ID, None, held.tags.tickers_text, held.tags)

    async def _filing_body(self, raw: RawItem) -> Body:
        """The text of an 8-K, its main document and first Exhibit 99.1 found through the filing index. A document
        that does not arrive or cannot be parsed is left out and named."""
        edgar_config = self.config.edgar
        docs = _parsed(edgar_docs.filing_documents, await self._document(raw.source, raw.url))
        if docs is None:
            return self._flag(raw, edgar_docs.body([]), ["index"])
        missing = [] if docs.main is not None else ["main document"]
        parts = []
        for label, url, max_chars in (("main document", docs.main, edgar_config.main_doc_max_chars),
                                      ("exhibit", docs.exhibit, edgar_config.exhibit_max_chars)):
            if url is None:
                continue
            doc = _parsed(edgar_docs.document, await self._document(raw.source, url), max_chars)
            if doc is None:
                missing.append(label)
            else:
                parts.append((url, doc))
        return self._flag(raw, edgar_docs.body(parts), missing)

    async def _page_body(self, raw: RawItem) -> Body:
        """The text of a monetary policy release page, flagged without it when it does not arrive or cannot be
        parsed."""
        page = await self._document(raw.source, raw.url)
        body = _parsed(rss.fed_page_body, page, raw.url, self.config.fed.monetary_body_max_chars)
        return body if body is not None else self._flag(raw, edgar_docs.body([]), ["page"])

    def _flag(self, raw: RawItem, body: Body, missing: list[str]) -> Body:
        """A body marked truncated, with a doc_failed event, when any document it needs did not arrive."""
        if not missing:
            return body
        self._event("doc_failed", f"{raw.vendor_id} without its {', '.join(missing)}")
        log.warning("a held %s item is written without its %s", raw.kind, ", ".join(missing))
        return replace(body, truncated=True)

    async def _document(self, source: str, url: str) -> bytes | None:
        """One document's body, in at most two attempts through the budget, each written as a news_polls row of the
        source's document channel, or None when both fail or the address is not one the client may call."""
        for _ in range(DOC_ATTEMPTS):
            try:
                group = http.host_group(url)
            except ValueError:
                return None
            wait_ms = await self.budget.acquire(group)
            started = self.clock.stamp()
            response = await self.client.fetch(url, None)
            received = self.clock.stamp()
            self._observe(group, response, received, wait_ms, DOC_CHANNELS[source])
            self._poll_row(DOC_CHANNELS[source], started, 0, response.status, response, wait_ms, response.detail,
                           None)
            if response.status == http.OK:
                return response.body
        return None

    def _url(self, channel: Channel) -> str:
        """The address a channel polls now."""
        finnhub, edgar_config = self.config.finnhub, self.config.edgar
        if channel.source == FINNHUB:
            if channel.symbol is not None:
                return finnhub_news.company_url(channel.symbol, self.clock.now_ms(), finnhub.lookback_days)
            return finnhub_news.general_url(finnhub.general_category, self.cursor)
        if channel.source == EDGAR:
            if channel.symbol is not None:
                return edgar.submissions_url(edgar_config.ciks[channel.symbol])
            return edgar.ATOM_URL
        if channel.source == FED:
            return rss.fed_url(channel.feed)
        return rss.bls_url(channel.feed)

    def _parse(self, channel: Channel, body: bytes) -> list[RawItem]:
        """The items of one body, by the channel's source."""
        finnhub, edgar_config = self.config.finnhub, self.config.edgar
        if channel.source == FINNHUB:
            if channel.symbol is not None:
                return finnhub_news.parse_news(body, finnhub.headline_max_chars, finnhub.summary_max_chars)
            return finnhub_news.parse_news(body, finnhub.headline_max_chars, finnhub.summary_max_chars,
                                           min_id=self.cursor, general=True)
        if channel.source == EDGAR:
            if channel.symbol is not None:
                return edgar.parse_submissions(body, channel.symbol, edgar_config.ciks[channel.symbol],
                                               edgar_config.forms)
            return edgar.parse_atom(body, edgar_config.ciks, edgar_config.forms)
        if channel.source == FED:
            return rss.fed_items(body, self.config.fed.monetary_link_pattern, self.config.fed.summary_max_chars)
        return rss.bls_items(body, self.config.bls.summary_max_chars)

    def _advance_cursor(self, items: list[RawItem]) -> None:
        """Keep the largest general news id seen in meta, the minId of the next call."""
        newest = finnhub_news.newest_id(items, self.cursor)
        if newest != self.cursor:
            db.set_values(self.conn, "meta", {CURSOR: str(newest)})
            self.cursor = newest
