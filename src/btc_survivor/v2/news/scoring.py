"""The model scoring of the news process, PLAN_V2 sections 8.1 to 8.8, which news/runner.py runs beside collection.

It refuses to start unless the config pins the prompt and schema files, then builds the Haiku scorer, with the daily
probe, and the Sonnet scorer, which share one Caller on the key ledger and the one monthly cap. Before any item is
admitted it recovers what a stop or a crash left behind. Every call this data folder still had running in the ledger,
or on record in scorer_state, is booked lost at its reservation and gets an llm_calls row with status lost, its items
failed, and every item a stop left between the calls of a batch is failed too, so nothing is ever sent a second time.
A pause, the cap and each model's state live in scorer_state and survive a restart, while off until restart ends with
it.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from typing import Any

from btc_survivor.clock import LiveClock, iso, parse_iso
from btc_survivor.v2.config import NewsConfig
from btc_survivor.v2.news import db, probes
from btc_survivor.v2.news.batcher import BATCH, SINGLE, BatchSettings
from btc_survivor.v2.news.collector import wait_or_stop
from btc_survivor.v2.news.ledger import RESERVED, Ledger, Prices, Reservation, parse_ref
from btc_survivor.v2.news.llm_request import check_pins, haiku_spec, sonnet_spec
from btc_survivor.v2.news.llm_scorer import CAP_UNTIL, HAIKU, LOST, PROBE, SONNET, Caller, Cap, Client, LlmScorer
from btc_survivor.v2.news.score_store import FAILED, Final, call_recorded, has_final, write_call

log = logging.getLogger(__name__)


class Scoring:
    """Both model scorers of one news process."""

    def __init__(self, config: NewsConfig, conn: sqlite3.Connection, clock: LiveClock, client: Client,
                 ledger: Ledger) -> None:
        llm = config.llm
        check_pins(llm)
        self.conn = conn
        self.clock = clock
        self.caller = Caller(client, ledger, Prices.from_config(llm.prices), llm, clock)
        stored = db.get_value(conn, "scorer_state", CAP_UNTIL)
        self.cap = Cap(parse_iso(stored) if stored else None)
        self.haiku = LlmScorer(HAIKU, haiku_spec(llm), BatchSettings.haiku(config), config, conn, clock, self.caller,
                               self.cap, llm.enabled, probes.load_probes(llm.probe_file))
        self.sonnet = LlmScorer(SONNET, sonnet_spec(llm), BatchSettings.sonnet(config), config, conn, clock,
                                self.caller, self.cap, llm.sonnet.enabled)
        self.scorers = (self.haiku, self.sonnet)

    def words(self, now_ms: int | None = None) -> tuple[str, str]:
        """The state words of Haiku and Sonnet for the heartbeat."""
        now = self.clock.now_ms() if now_ms is None else now_ms
        return self.haiku.word(now), self.sonnet.word(now)

    def _scorer(self, name: str, model: str) -> LlmScorer:
        """The scorer a lost call belonged to, by its name, or by its model for a name an older config used."""
        for scorer in self.scorers:
            if scorer.name == name:
                return scorer
        return next(scorer for scorer in self.scorers if scorer.spec.model == model)

    def _unrecorded(self, reservation: Reservation, now_ms: int) -> dict[str, Any]:
        """The record of a lost call that never reached scorer_state, rebuilt from its ledger row, with the time of
        recovery as its start."""
        name, seqs = parse_ref(reservation.ref)
        kind = PROBE if not seqs else SINGLE if len(seqs) == 1 else BATCH
        scorer = self._scorer(name, reservation.model)
        row = scorer.base_row(kind, None, reservation, now_ms, reservation.n_items, None)
        return {"row": {**row, "scorer": name}, "seqs": list(seqs)}

    def _prior_calls(self, item_seq: int, scorer: str) -> int:
        return self.conn.execute("SELECT count(*) FROM llm_call_items ci JOIN llm_calls c ON c.call_id = ci.call_id "
                                 "WHERE ci.item_seq = ? AND c.scorer = ?", (item_seq, scorer)).fetchone()[0]

    def _write_lost(self, record: dict[str, Any], now_ms: int) -> None:
        """Write one lost call booked at its reservation, failing each of its items that has no final row."""
        row, seqs = record["row"], record["seqs"]
        scorer = row["scorer"]
        marks = ", ".join("?" * len(seqs))
        routes = dict(self.conn.execute(f"SELECT seq, route FROM news_items WHERE seq IN ({marks})", seqs).fetchall())
        finals = [Final(seq, routes[seq], FAILED, self._prior_calls(seq, scorer) + 1) for seq in seqs
                  if seq in routes and not has_final(self.conn, seq, scorer)]
        stamp = self.clock.stamp()
        write_call(self.conn, {**row, "status": LOST, "finished_at": iso(now_ms), "cost_usd": row["reserved_usd"],
                               "cost_basis": RESERVED}, seqs, finals, stamp)
        log.warning("%s call %s was lost when the process stopped, booked at its reservation, %d items failed",
                    scorer, row["ledger_id"], len(finals))

    def recover(self, now_ms: int) -> int:
        """Book every call this data folder lost and fail the items a stop left without a final row, returning the
        number of lost calls. Each lost call is written before its ledger row is marked lost, so recovery dying
        half way leaves the rest for the next start. It runs before any item is admitted and is safe to run twice."""
        records: dict[int, dict[str, Any]] = {}
        for scorer in self.scorers:
            record = scorer.recorded_call()
            if record is not None:
                records[record["row"]["ledger_id"]] = record
        for reservation in self.caller.ledger.running():
            records.setdefault(reservation.id, self._unrecorded(reservation, now_ms))
        lost = 0
        for ledger_id in sorted(records):
            if not call_recorded(self.conn, ledger_id):
                self._write_lost(records[ledger_id], now_ms)
                lost += 1
        self.caller.ledger.recover()   # only after every lost call is written, so a second crash loses none
        with db.transaction(self.conn):
            db.set_values(self.conn, "scorer_state", {scorer.key("call"): "" for scorer in self.scorers})
        failed = sum(scorer.interrupted() for scorer in self.scorers)
        if lost or failed:
            log.warning("recovered %d lost calls and failed %d interrupted items", lost, failed)
        return lost

    async def run(self, stop: asyncio.Event,
                  wait: Callable[[asyncio.Event, float], Awaitable[bool]] = wait_or_stop) -> None:
        """Recover, then run both scorers until stop is set."""
        self.recover(self.clock.stamp())
        await asyncio.gather(self.haiku.run(stop, wait), self.sonnet.run(stop, wait))
