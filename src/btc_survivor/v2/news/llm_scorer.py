"""One model scorer of PLAN_V2 sections 8.3 to 8.6, Haiku on every item routed to llm or Sonnet on the official ones.

Items come from news.db in seq order and wait in the scorer's batcher. Each call is counted exactly, or bounded by
its bytes when counting fails, checked against the size guard and reserved against the key ledger. It is put on
record in scorer_state before it is sent, sent once and never again, booked in the ledger, and written with the final
rows it decided in one transaction that also clears the record. A refused, cut off or unreadable answer is split in
halves recursively, an item an answer left out is requeued once and then marked malformed, and a batch never takes
more than max_calls_per_batch calls. Errors follow the failure table of section 8.5. While a model is paused, capped
or off, every item routed to it gets a final skipped_off or skipped_cap row at once, so FinBERT takes over. The
monthly cap and a spend limit belong to the key, so they turn both models off until the next UTC month, and nothing
else one model meets changes the other's state. Haiku also sends the daily probe between its batches, so one call is
in flight per model. A reservation prices the whole max_tokens a request sends, its thinking room included, and a
thinking block raises an alert only from a model whose thinking is disabled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import zlib
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, LiveClock, iso, parse_iso
from btc_survivor.v2.config import LlmConfig, NewsConfig
from btc_survivor.v2.news import db, health, probes
from btc_survivor.v2.news.batcher import Batch, Batcher, BatchSettings, Pending, split
from btc_survivor.v2.news.collector import vendor_status, wait_or_stop
from btc_survivor.v2.news.ledger import RESERVED, USAGE, Ledger, Prices, Reservation, byte_bound, cap_until, make_ref
from btc_survivor.v2.news.llm_client import (
    AUTH, BAD_REQUEST, MALFORMED as MALFORMED_CALL, MAX_TOKENS, MODEL_MISMATCH, OK, RATE_LIMITED, REFUSAL,
    SPEND_LIMIT, CallResult, sdk_version)
from btc_survivor.v2.news.llm_request import (
    BYTES, COUNT_TOKENS, DISABLED, Request, RequestTooLarge, ScorerSpec, build_request, check_size, request_item)
from btc_survivor.v2.news.llm_validate import MISSING, Impact, Validation, validate
from btc_survivor.v2.news.score_store import (
    EXPIRED, FAILED, IRRELEVANT, MALFORMED, REFUSED, SCORED, SKIPPED_CAP, SKIPPED_OFF, Final, write_call, write_finals)

log = logging.getLogger(__name__)

ON, OFF = health.ON, health.OFF
PAUSED, CAPPED = "paused", "capped"
HAIKU, SONNET = "haiku", "sonnet"            # the roles, and the prefixes of their scorer_state keys
SPLIT, PROBE, LOST = "split", "probe", "lost"
CAP_UNTIL = "cap_until"                      # the scorer_state key both models share
STATE_EVENT, ALERT_EVENT = "scorer_state", "scorer_alert"
POLL_SECONDS = 1.0
SKIPPED = {OFF: SKIPPED_OFF, PAUSED: SKIPPED_OFF, CAPPED: SKIPPED_CAP}
_SPLIT_ON = (REFUSAL, MAX_TOKENS, MALFORMED_CALL)
_STATE_AFTER = {MODEL_MISMATCH: OFF, BAD_REQUEST: OFF, AUTH: OFF, RATE_LIMITED: PAUSED, SPEND_LIMIT: CAPPED}
_ITEMS = """
SELECT i.seq, i.source, i.kind, i.status, i.route, i.title, i.summary, i.tickers, b.text AS body
FROM news_items i LEFT JOIN news_bodies b ON b.item_seq = i.seq WHERE i.seq IN ({})
"""


class NotLive(RuntimeError):
    """Raised before a call holding an item that is not a live item routed to this model, such as a seed, backlog
    or backfill item, or a kind Sonnet does not take, which section 8.8 never lets reach a model."""


class Client(Protocol):
    """What a scorer needs from llm_client.LlmClient, so tests pass a fake."""

    async def count_tokens(self, request: Mapping[str, Any]) -> int | None: ...

    async def send(self, request: Mapping[str, Any]) -> CallResult: ...


@dataclass
class Cap:
    """The monthly cap both models share, holding the instant until which both stay off once the cap refused a call
    or the API answered with a spend limit."""

    until_ms: int | None = None

    def active(self, now_ms: int) -> bool:
        return self.until_ms is not None and now_ms < self.until_ms


@dataclass(frozen=True)
class Prepared:
    """A request that passed the size guard, with the count its reservation rests on and that reservation."""

    request: Request
    counted: int
    basis: str
    reserved_usd: float


@dataclass(frozen=True)
class Sent:
    """One reserved call that went out and is booked in the ledger, at the cost of its usage or at its reservation."""

    prepared: Prepared
    reservation: Reservation
    started_ms: int
    result: CallResult
    cost_usd: float
    cost_basis: str


def booked(prices: Prices, model: str, result: CallResult, reserved_usd: float) -> tuple[float, str]:
    """What the ledger books for a call, the cost of the usage it returned, or its reservation when it returned no
    usage or came from another model, whose prices are unknown."""
    if result.usage is None or result.status == MODEL_MISMATCH:
        return reserved_usd, RESERVED
    return prices.cost(model, result.usage), USAGE


class Caller:
    """Counts, guards, reserves, sends and books one request at a time against the key ledger, for the scorers and
    scripts/llm_synthetic_check.py alike."""

    def __init__(self, client: Client, ledger: Ledger, prices: Prices, llm: LlmConfig, clock: LiveClock) -> None:
        self.client = client
        self.ledger = ledger
        self.prices = prices
        self.llm = llm
        self.clock = clock

    async def prepare(self, request: Request) -> Prepared:
        """Count a request exactly, or bound it by its bytes when counting fails, refuse it above the size guard with
        RequestTooLarge, and price its reservation. The API calls its count an estimate, so an exact count is
        reserved with reserve_overhead_tokens on top, while the count itself is what the call row records."""
        exact = await self.client.count_tokens(request.params())
        check_size(request, self.llm.max_input_tokens, exact)
        overhead = self.llm.caps.reserve_overhead_tokens
        if exact is None:
            counted, basis = byte_bound(request.params(), overhead), BYTES
            bound = counted
        else:
            counted, basis = exact, COUNT_TOKENS
            bound = exact + overhead
        spec = request.spec
        return Prepared(request, counted, basis, self.prices.reserve(spec.model, bound, spec.request_max_tokens))

    async def send(self, prepared: Prepared, ref: str, n_items: int,
                   on_reserved: Callable[[Reservation, int], None] | None = None) -> Sent | None:
        """Reserve and send one prepared request and book what it cost, or return None with nothing booked when the
        cap refuses it. on_reserved runs between the reservation and the send with no await before it, so a call is
        on record before it can be lost."""
        model = prepared.request.spec.model
        started = self.clock.stamp()
        reservation = self.ledger.reserve(model, ref, n_items, prepared.reserved_usd, started)
        if reservation is None:
            return None
        if on_reserved is not None:
            on_reserved(reservation, started)
        result = await self.client.send(prepared.request.params())
        cost, basis = booked(self.prices, model, result, prepared.reserved_usd)
        self.ledger.finish(reservation, cost if basis == USAGE else None)
        return Sent(prepared, reservation, started, result, cost, basis)


@dataclass(frozen=True)
class Decision:
    """What one answer means. finals maps a batch position to its final status and impacts, split asks for the items
    again in two halves, missing lists the positions the answer left out, and state is the model's next state word,
    with pause_ms for a pause."""

    status: str
    finals: Mapping[int, tuple[str, tuple[Impact, ...]]]
    split: bool = False
    missing: tuple[int, ...] = ()
    state: str | None = None
    pause_ms: int = 0
    validation: Validation | None = None


def decide(result: CallResult, ids: Sequence[str], pause_ms: int) -> Decision:
    """The action of section 8.5 for one call's result, with ids the batch positions it asked about. A single item
    is never split, so it ends refused or malformed."""
    alone = len(ids) == 1
    if result.status == OK:
        validation = validate(result.text or "", ids)
        if validation.whole is not None:
            return Decision(MALFORMED_CALL, {0: (MALFORMED, ())} if alone else {}, split=not alone,
                            validation=validation)
        finals: dict[int, tuple[str, tuple[Impact, ...]]] = {}
        missing = []
        for index, key in enumerate(ids):
            if key in validation.scored:
                impacts = validation.scored[key]
                finals[index] = (SCORED if impacts else IRRELEVANT, impacts)
            elif validation.failed[key] == MISSING:
                missing.append(index)
            else:
                finals[index] = (MALFORMED, ())
        return Decision(OK, finals, missing=tuple(missing), validation=validation)
    if result.status in _SPLIT_ON:
        final = REFUSED if result.status == REFUSAL else MALFORMED
        return Decision(result.status, {0: (final, ())} if alone else {}, split=not alone)
    state = _STATE_AFTER.get(result.status)
    pause = max(pause_ms, round((result.retry_after_s or 0) * MS_PER_SECOND)) if state == PAUSED else 0
    return Decision(result.status, {index: (FAILED, ()) for index in range(len(ids))}, state=state, pause_ms=pause)


def result_row(sent: Sent, status: str, finished_ms: int) -> dict[str, Any]:
    """The llm_calls fields a call learns once it has ended. response_z holds the raw answer, or the redacted error
    detail of a call that returned none, compressed."""
    result, usage = sent.result, sent.result.usage
    raw = result.raw_json
    if raw is None and result.error_detail:
        raw = json.dumps({"error_detail": result.error_detail})
    return {
        "finished_at": iso(finished_ms), "status": status, "response_model": result.response_model,
        "request_id": result.request_id, "stop_reason": result.stop_reason,
        "refusal_category": result.refusal_category, "http_status": result.http_status,
        "error_class": result.error_class, "thinking_blocks": result.thinking_blocks,
        "input_tokens": None if usage is None else usage.input_tokens,
        "cache_creation_input_tokens": None if usage is None else usage.cache_creation_input_tokens,
        "cache_read_input_tokens": None if usage is None else usage.cache_read_input_tokens,
        "output_tokens": None if usage is None else usage.output_tokens,
        "cost_usd": sent.cost_usd, "cost_basis": sent.cost_basis, "latency_ms": result.latency_ms,
        "response_z": None if raw is None else zlib.compress(raw.encode("utf-8")),
    }


@dataclass(frozen=True)
class Called:
    """A call that went out, its llm_calls id, whether its items go again in halves, and the items requeued."""

    call_id: int
    split: bool
    requeued: tuple[Pending, ...]


@dataclass
class _Tracks:
    """The calls each item of this process has been in, the last of them and the requeues it used."""

    attempts: Counter[int] = field(default_factory=Counter)
    last_call: dict[int, int] = field(default_factory=dict)
    requeues: Counter[int] = field(default_factory=Counter)

    def forget(self, seqs: Iterable[int]) -> None:
        for seq in seqs:
            self.attempts.pop(seq, None)
            self.last_call.pop(seq, None)
            self.requeues.pop(seq, None)


class LlmScorer:
    """The scorer of one model, Haiku or Sonnet, with its batcher, state and calls."""

    def __init__(self, role: str, spec: ScorerSpec, settings: BatchSettings, config: NewsConfig,
                 conn: sqlite3.Connection, clock: LiveClock, caller: Caller, cap: Cap, enabled: bool,
                 probe_items: Sequence[probes.Probe] = ()) -> None:
        self.role = role
        self.spec = spec
        self.name = spec.name
        self.llm = config.llm
        self.max_vendor_age_hours = config.news.max_vendor_age_hours
        self.batcher = Batcher(settings)
        self.conn = conn
        self.clock = clock
        self.caller = caller
        self.cap = cap
        self.enabled = enabled
        self.probe_items = tuple(probe_items)
        self.counts: Counter[str] = Counter()
        self._pause_ms = round(self.llm.rate_limit_pause_minutes * MS_PER_MINUTE)
        self._off = False
        self._paused_until = self._stored_ms(self.key("until"))
        self._written: str | None = None
        self._after = 0
        self._tracks = _Tracks()
        self._last_probe: str | None = None
        self._probe_loaded = False
        self._response_model = db.get_value(conn, "scorer_state", self.key("response_model"))
        kinds = settings.kinds
        marks = ", ".join("?" * len(kinds or ()))
        where = "" if kinds is None else f" AND i.source != 'finnhub' AND i.kind IN ({marks})"
        self._pending_sql = (f"SELECT i.seq, i.source, i.kind, i.route, i.status, i.received_at, i.vendor_ts "
                             f"FROM news_items i WHERE i.seq > ? AND i.route = 'llm' AND i.status = 'new'{where} "
                             f"AND NOT EXISTS (SELECT 1 FROM item_scores s WHERE s.item_seq = i.seq "
                             f"AND s.scorer = ?) ORDER BY i.seq")
        self._kinds = tuple(kinds or ())

    def key(self, name: str) -> str:
        """A scorer_state key of this model, such as haiku_state, haiku_until or haiku_call."""
        return f"{self.role}_{name}"

    def _stored_ms(self, key: str) -> int | None:
        value = db.get_value(self.conn, "scorer_state", key)
        return parse_iso(value) if value else None

    # State

    def word(self, now_ms: int) -> str:
        """The state word at now_ms, off while disabled or off until restart, then capped, paused or on."""
        if not self.enabled or self._off:
            return OFF
        if self.cap.active(now_ms):
            return CAPPED
        if self._paused_until is not None and now_ms < self._paused_until:
            return PAUSED
        return ON

    def sync(self, now_ms: int, reason: str | None = None) -> str:
        """The state word at now_ms, written to scorer_state with a scorer_state event and a log line whenever it
        changed, so a pause or the cap survives a restart."""
        word = self.word(now_ms)
        if word == self._written:
            return word
        until = {PAUSED: self._paused_until, CAPPED: self.cap.until_ms}.get(word)
        pausing = self._paused_until is not None and now_ms < self._paused_until
        values = {self.key("state"): word, self.key("until"): iso(self._paused_until) if pausing else ""}
        if word == CAPPED:
            values[CAP_UNTIL] = iso(self.cap.until_ms)
        detail = (f"{self.name} {word}" + (f" until {iso(until)}" if until is not None else "")
                  + (f" after {reason}" if reason else ""))
        with db.transaction(self.conn):
            db.set_values(self.conn, "scorer_state", values)
            db.insert(self.conn, "collector_events", {"ts": iso(now_ms), "kind": STATE_EVENT, "detail": detail})
        log.log(logging.INFO if word == ON or not self.enabled else logging.WARNING, "%s", detail)
        self._written = word
        return word

    def _alert(self, now_ms: int, detail: str) -> None:
        """Log one WARNING and write a scorer_alert event, which stops nothing."""
        db.insert(self.conn, "collector_events", {"ts": iso(now_ms), "kind": ALERT_EVENT, "detail": detail})
        log.warning("%s", detail)

    def _refusals_high(self) -> bool:
        """Whether refusals make more than refusal_pause_share of the window of this scorer's last
        refusal_window_calls calls, counted against the full window so a few early refusals never pause it."""
        window = self.llm.refusal_window_calls
        refused = self.conn.execute(
            "SELECT count(*) FROM (SELECT status FROM llm_calls WHERE scorer = ? ORDER BY call_id DESC LIMIT ?) "
            "WHERE status = ?", (self.name, window, REFUSAL)).fetchone()[0]
        return refused > self.llm.refusal_pause_share * window

    def _cap_refused(self) -> None:
        """The cap refused a reservation, so both models stay off until 00:00 UTC on the first of the next month."""
        now = self.clock.now_ms()
        self.cap.until_ms = cap_until(now)
        self.sync(now, "the monthly cap")

    def _after_call(self, sent: Sent, decision: Decision, now_ms: int) -> None:
        """Apply the state an answer calls for and raise the alerts of sections 8.1, 8.5 and 8.6."""
        result = sent.result
        if decision.state == OFF:
            self._off = True
            log.error("%s turned off until restart by %s, http %s, %s, %s", self.name, result.status,
                      result.http_status, result.error_class, result.error_detail)
        elif decision.state == PAUSED:
            self._paused_until = max(self._paused_until or 0, now_ms + decision.pause_ms)
        elif decision.state == CAPPED:
            self.cap.until_ms = cap_until(now_ms)
        usage = result.usage
        thought = 0 if usage is None else usage.thinking_tokens
        if self.spec.thinking == DISABLED and (result.thinking_blocks or thought):
            self._alert(now_ms, f"{self.name} answered with {result.thinking_blocks} thinking blocks and {thought} "
                                f"thinking tokens while its thinking is disabled")
        if self.role == SONNET and usage is not None and (usage.cache_creation_input_tokens
                                                          or usage.cache_read_input_tokens):
            self._alert(now_ms, f"{self.name} reported cache tokens, whose prices are not in the research")
        if sent.cost_usd > sent.prepared.reserved_usd:
            self._alert(now_ms, f"{self.name} call cost more than its reservation")
        if result.status == REFUSAL and self._refusals_high():
            self._paused_until = max(self._paused_until or 0, now_ms + self._pause_ms)
            self._alert(now_ms, f"{self.name} refused more than {self.llm.refusal_pause_share:.0%} of its last "
                                f"{self.llm.refusal_window_calls} calls, so it pauses")
        if result.status == OK and self._response_model is None:
            self._response_model = result.response_model
            db.set_values(self.conn, "scorer_state", {self.key("response_model"): result.response_model})
        self.sync(now_ms, result.status)

    # Rows

    def _final(self, item: Pending, status: str, impacts: tuple[Impact, ...] = ()) -> Final:
        """The final row of an item, with the calls it was in and the last of them."""
        tracks = self._tracks
        return Final(item.seq, item.route, status, tracks.attempts[item.seq], tracks.last_call.get(item.seq),
                     impacts)

    def _finish(self, items: Iterable[Pending], status: str) -> None:
        """Give items their final rows without a call, in one transaction."""
        finals = [self._final(item, status) for item in items]
        if not finals:
            return
        write_finals(self.conn, self.name, finals, self.clock.stamp())
        self._tracks.forget(final.item_seq for final in finals)
        self.counts[status] += len(finals)

    def base_row(self, kind: str, root_call_id: int | None, reservation: Reservation, started_ms: int,
                 n_items: int, prepared: Prepared | None) -> dict[str, Any]:
        """The llm_calls fields a call has before it is sent. A call lost without its record has no request hash or
        count, so request_sha256 is empty and the count null."""
        spec = self.spec
        sent = prepared is not None
        return {
            "root_call_id": root_call_id, "ledger_id": reservation.id, "kind": kind, "scorer": spec.name,
            "started_at": iso(started_ms), "n_items": n_items, "model": reservation.model,
            "sdk_version": sdk_version(), "prompt_sha256": self.llm.prompt_sha256,
            "schema_sha256": self.llm.schema_sha256, "request_sha256": prepared.request.sha256 if sent else "",
            "effort": spec.effort, "thinking": spec.thinking, "max_tokens": spec.request_max_tokens,
            "counted_tokens": prepared.counted if sent else None, "count_basis": prepared.basis if sent else None,
            "reserved_usd": reservation.reserved_usd, "price_table": self.caller.prices.table(),
        }

    def _record(self, base: dict[str, Any], seqs: Sequence[int]) -> None:
        """Put the call about to be sent on record, so a restart can book it lost and fail its items."""
        db.set_values(self.conn, "scorer_state", {self.key("call"): json.dumps({"row": base, "seqs": list(seqs)})})

    def recorded_call(self) -> dict[str, Any] | None:
        """The call this model had on record when the process stopped, if any."""
        value = db.get_value(self.conn, "scorer_state", self.key("call"))
        return json.loads(value) if value else None

    # Items

    def admit(self, now_ms: int) -> None:
        """Read the routed items that arrived since the last look, queue those this model takes while it is on, and
        give the rest of its items their final rows at once."""
        rows = self.conn.execute(self._pending_sql, (self._after, *self._kinds, self.name)).fetchall()
        word = self.word(now_ms)
        decided: dict[str, list[Pending]] = {}
        for row in rows:
            self._after = row["seq"]
            item = Pending.from_row(row)
            if not self.batcher.wants(item):
                continue
            if word == ON:
                status = self.batcher.admit(item, now_ms)
            else:
                status = vendor_status(item.received_at, item.vendor_ts, self.max_vendor_age_hours) or SKIPPED[word]
            if status is not None:
                decided.setdefault(status, []).append(item)
        for status, items in decided.items():
            self._finish(items, status)

    def _item_rows(self, items: Sequence[Pending]) -> dict[int, sqlite3.Row]:
        """The news_items fields and body text a request needs, by seq."""
        sql = _ITEMS.format(", ".join("?" * len(items)))
        return {row["seq"]: row for row in self.conn.execute(sql, [item.seq for item in items])}

    def _live(self, row: sqlite3.Row) -> bool:
        """The check of sections 7.6 and 8.8 just before a call, that an item is a new live item routed to llm and,
        for Sonnet, one of its official kinds."""
        if row["status"] != "new" or row["route"] != "llm":
            return False
        return not self._kinds or (row["source"] != "finnhub" and row["kind"] in self._kinds)

    def interrupted(self) -> int:
        """At start, fail every item that was in a finished call of this scorer and still has no final row, because a
        stop left it waiting for a split or its requeue, which can no longer be told apart from a second requeue."""
        rows = self.conn.execute(
            "SELECT ci.item_seq, i.route, count(*) AS calls, max(ci.call_id) AS last_call FROM llm_call_items ci "
            "JOIN llm_calls c ON c.call_id = ci.call_id JOIN news_items i ON i.seq = ci.item_seq "
            "WHERE c.scorer = ? AND NOT EXISTS (SELECT 1 FROM item_scores s WHERE s.item_seq = ci.item_seq "
            "AND s.scorer = ?) GROUP BY ci.item_seq ORDER BY ci.item_seq", (self.name, self.name)).fetchall()
        finals = [Final(row["item_seq"], row["route"], FAILED, row["calls"], row["last_call"]) for row in rows]
        if finals:
            write_finals(self.conn, self.name, finals, self.clock.stamp())
            self.counts[FAILED] += len(finals)
        return len(finals)

    # Calls

    async def _call(self, items: Sequence[Pending], kind: str, root: int | None) -> Called | None:
        """Send one call for items and write what it decided, or return None when the cap refused it. Raises
        NotLive or RequestTooLarge, with nothing reserved, when an item is not live or the request is above the size
        guard."""
        rows = self._item_rows(items)
        refused = [seq for seq, row in rows.items() if not self._live(row)]
        if refused or len(rows) != len(items):
            raise NotLive(f"{self.name} refused a call holding items {refused or 'missing'}, which are not live items "
                          f"routed to it")
        request = build_request(self.spec, [request_item(rows[item.seq], rows[item.seq]["body"], self.llm)
                                            for item in items])
        prepared = await self.caller.prepare(request)
        seqs = [item.seq for item in items]
        base: dict[str, Any] = {}

        def on_reserved(reservation: Reservation, started_ms: int) -> None:
            base.update(self.base_row(kind, root, reservation, started_ms, len(items), prepared))
            self._record(base, seqs)

        sent = await self.caller.send(prepared, make_ref(self.name, seqs), len(items), on_reserved)
        if sent is None:
            self._cap_refused()
            return None
        tracks = self._tracks
        tracks.attempts.update(seqs)
        decision = decide(sent.result, request.ids, self._pause_ms)
        finals, requeued = [], []
        for index, item in enumerate(items):
            if index in decision.finals:
                status, impacts = decision.finals[index]
                finals.append(self._final(item, status, impacts))
            elif index in decision.missing and tracks.requeues[item.seq] < self.llm.max_requeues:
                tracks.requeues[item.seq] += 1
                requeued.append(item)
            elif index in decision.missing:
                finals.append(self._final(item, MALFORMED))
        finished = self.clock.stamp()
        call_id = write_call(self.conn, {**base, **result_row(sent, decision.status, finished)}, seqs, finals,
                             finished, state={self.key("call"): ""})
        tracks.last_call.update(dict.fromkeys(seqs, call_id))
        tracks.forget(final.item_seq for final in finals)
        self.counts[f"call {decision.status}"] += 1
        self.counts.update(final.status for final in finals)
        self._after_call(sent, decision, finished)
        return Called(call_id, decision.split, tuple(requeued))

    async def process(self, batch: Batch, stop: asyncio.Event) -> None:
        """Score one batch, splitting it after a refused, cut off or unreadable answer and requeueing items an answer
        left out, within max_calls_per_batch calls. A stop ends it after the call in hand, and the items left are
        failed at the next start."""
        stack: list[tuple[Sequence[Pending], str]] = [(batch.items, batch.kind)]
        root: int | None = None
        calls = 0
        while stack:
            items, kind = stack.pop()
            word = self.word(self.clock.now_ms())
            if word != ON:
                self._finish([*items, *(item for chunk, _ in stack for item in chunk)], SKIPPED[word])
                return
            if calls >= self.llm.max_calls_per_batch:
                log.warning("%s used %d calls on one batch, so its last items end malformed", self.name, calls)
                self._finish([*items, *(item for chunk, _ in stack for item in chunk)], MALFORMED)
                return
            try:
                called = await self._call(items, kind, root)
            except NotLive as exc:
                log.error("%s", exc)
                self._finish(items, FAILED)
                continue
            except RequestTooLarge as exc:
                log.warning("%s %s", self.name, exc)
                if len(items) > 1:
                    first, second = split(items)
                    stack += [(second, SPLIT), (first, SPLIT)]
                else:
                    self._finish(items, FAILED)
                continue
            if called is None:
                stack.append((items, kind))
                continue
            calls += 1
            root = root if root is not None else called.call_id
            if called.split:
                first, second = split(items)
                stack += [(second, SPLIT), (first, SPLIT)]
            for item in called.requeued:
                self.batcher.requeue(item)
            if stop.is_set():
                return

    def probe_due(self, now_ms: int) -> bool:
        """Whether the daily probe is due, judged from the start of the last probe call on record, so a probe that
        failed or was lost is not sent again the same day."""
        if not self.probe_items:
            return False
        if not self._probe_loaded:
            self._last_probe = self.conn.execute("SELECT max(started_at) FROM llm_calls WHERE scorer = ? AND kind = ?",
                                                 (self.name, PROBE)).fetchone()[0]
            self._probe_loaded = True
        return probes.due(now_ms, self._last_probe, self.llm.probe_time_utc)

    async def probe(self) -> None:
        """Send the 20 probes as one reserved call and write their llm_probes rows, one per impact of each valid
        answer, with the call."""
        request = probes.probe_request(self.llm, self.probe_items)
        try:
            prepared = await self.caller.prepare(request)
        except RequestTooLarge as exc:
            log.warning("%s probe %s", self.name, exc)
            self._last_probe = iso(self.clock.now_ms())
            return
        base: dict[str, Any] = {}

        def on_reserved(reservation: Reservation, started_ms: int) -> None:
            base.update(self.base_row(PROBE, None, reservation, started_ms, len(self.probe_items), prepared))
            self._record(base, ())

        sent = await self.caller.send(prepared, make_ref(self.name, ()), len(self.probe_items), on_reserved)
        if sent is None:
            self._cap_refused()
            return
        decision = decide(sent.result, request.ids, self._pause_ms)
        valid = decision.validation if decision.status == OK else None
        rows = probes.probe_rows(base["started_at"], 0, self.probe_items, valid) if valid is not None else []
        finished = self.clock.stamp()
        write_call(self.conn, {**base, **result_row(sent, decision.status, finished)}, (), (), finished, rows,
                   state={self.key("call"): ""})
        self._last_probe = base["started_at"]
        self.counts[f"probe {decision.status}"] += 1
        self._after_call(sent, decision, finished)

    # The loop

    async def step(self, now_ms: int, stop: asyncio.Event) -> bool:
        """One pass at now_ms, returning whether it sent anything, so the loop waits only while idle."""
        word = self.sync(now_ms)
        self.admit(now_ms)
        if word != ON:
            self._finish(self.batcher.drain(), SKIPPED[word])
            return False
        if self.probe_due(now_ms):
            await self.probe()
            return True
        taken = self.batcher.take(now_ms)
        self._finish(taken.expired, EXPIRED)
        if taken.batch is None:
            return False
        await self.process(taken.batch, stop)
        return True

    async def run(self, stop: asyncio.Event,
                  wait: Callable[[asyncio.Event, float], Awaitable[bool]] = wait_or_stop) -> None:
        """Score until stop is set, finishing the call in hand, and look again every POLL_SECONDS while idle."""
        while not stop.is_set():
            busy = await self.step(self.clock.now_ms(), stop)
            if not busy and await wait(stop, POLL_SECONDS):
                return
