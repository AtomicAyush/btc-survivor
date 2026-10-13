"""The score store of PLAN_V2 sections 8.5 and 9.1, the one write path of the model scorers.

Every item gets one final row per scorer and never a second. A write reads first, inside the same transaction, and
refuses an item that already has a final row for that scorer, and the unique key on item and scorer refuses it again
should the read ever be skipped. An llm_calls row is inserted once, when its call has ended or at restart for a call
the process lost, never at the start of a call, because a trigger aborts any update. It commits together with its
llm_call_items rows, the final rows the call decided with their impacts, a probe run's llm_probes rows and any
scorer_state values, so a crash leaves either all of them or none.

A model impact's strength folds its horizon and surprise into direction times confidence over 5, by weights fixed by
reasoning before launch and never fitted. The test pairs a score with each 30 minute return over the next session
hours, half its weight in the first 2, so drift per window is about the move still to come over how long it lasts. A
move over within minutes is mostly gone before the first evaluation minute, one lasting hours sits where the weight is,
and one building over days spreads thin, so the weights are 0.25, 1 and 0.5. (2 + surprise) / 5 keeps a routine item
at 0.4 and a major shock at 1, so strength stays between minus one and one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from btc_survivor.clock import iso
from btc_survivor.v2.news import db
from btc_survivor.v2.news.llm_request import position_id
from btc_survivor.v2.news.llm_validate import Impact

SCORED, IRRELEVANT, REFUSED, MALFORMED, FAILED = "scored", "irrelevant", "refused", "malformed", "failed"
SKIPPED_CAP, SKIPPED_OFF, EXPIRED, STALE_VENDOR, UNTAGGED = (
    "skipped_cap", "skipped_off", "expired", "stale_vendor", "untagged")
FINAL = (SCORED, IRRELEVANT, REFUSED, MALFORMED, FAILED, SKIPPED_CAP, SKIPPED_OFF, EXPIRED, STALE_VENDOR, UNTAGGED)


class ScoreStoreError(RuntimeError):
    """Raised for a second final row for one item and scorer, or a row the rules of section 9.1 forbid."""


@dataclass(frozen=True)
class Final:
    """The one final row of an item for a model scorer. call_id is the last call the item was in, which write_call
    fills for the rows a call decides, attempts counts the calls it was in, and impacts are given only for scored."""

    item_seq: int
    route: str
    status: str
    attempts: int = 0
    call_id: int | None = None
    impacts: tuple[Impact, ...] = ()


# The weight of each horizon in a model impact's strength, PLAN_V2 section 6.3.
HORIZON_WEIGHTS = {"minutes": 0.25, "hours": 1.0, "days": 0.5}


def strength(direction: int, confidence: int, horizon: str, surprise: int) -> float:
    """A model impact's strength, direction times confidence over 5 times the horizon's weight times (2 + surprise)
    over 5, worked out with one rounding, since every factor but the final division by 25 is exact."""
    return direction * confidence * HORIZON_WEIGHTS[horizon] * (2 + surprise) / 25


def impact_row(item_seq: int, scorer: str, impact: Impact) -> dict[str, Any]:
    """The score_impacts row of one validated impact."""
    return {"item_seq": item_seq, "scorer": scorer, "ticker": impact.ticker,
            "strength": strength(impact.direction, impact.confidence, impact.horizon, impact.surprise),
            "direction": impact.direction,
            "surprise": impact.surprise, "event_type": impact.event_type, "confidence": impact.confidence,
            "horizon": impact.horizon, "is_new": int(impact.is_new)}


def has_final(conn: sqlite3.Connection, item_seq: int, scorer: str) -> bool:
    """Whether an item already has its final row for a scorer."""
    found = conn.execute("SELECT 1 FROM item_scores WHERE item_seq = ? AND scorer = ?", (item_seq, scorer))
    return found.fetchone() is not None


def call_recorded(conn: sqlite3.Connection, ledger_id: int) -> bool:
    """Whether a call already has its llm_calls row, found by the id of its key ledger reservation."""
    return conn.execute("SELECT 1 FROM llm_calls WHERE ledger_id = ?", (ledger_id,)).fetchone() is not None


def _check(final: Final) -> None:
    """Refuse a status outside section 9.1, impacts on any status but scored, or scored without impacts, which a
    model answer with no impacts records as irrelevant."""
    if final.status not in FINAL:
        raise ScoreStoreError(f"{final.status!r} is not a final status")
    if final.status == SCORED and not final.impacts:
        raise ScoreStoreError("a scored row needs at least one impact, an answer without one is irrelevant")
    if final.impacts and final.status != SCORED:
        raise ScoreStoreError(f"a {final.status} row carries no impacts")
    if final.attempts < 0:
        raise ScoreStoreError("attempts are never negative")


def _insert_final(conn: sqlite3.Connection, scorer: str, final: Final, scored_at: str) -> int:
    """Insert one final row and its impacts inside the caller's transaction, refusing a second final row."""
    _check(final)
    if has_final(conn, final.item_seq, scorer):
        raise ScoreStoreError(f"item {final.item_seq} already has its final {scorer} row")
    seq = db.insert(conn, "item_scores", {"item_seq": final.item_seq, "scorer": scorer, "route": final.route,
                                          "status": final.status, "call_id": final.call_id,
                                          "attempts": final.attempts, "scored_at": scored_at})
    for impact in final.impacts:
        db.insert(conn, "score_impacts", impact_row(final.item_seq, scorer, impact))
    return seq


def write_finals(conn: sqlite3.Connection, scorer: str, finals: Iterable[Final], scored_at_ms: int) -> list[int]:
    """Insert final rows decided without a call, such as expired, stale_vendor or skipped_off, in one transaction,
    returning their seqs."""
    with db.transaction(conn):
        return [_insert_final(conn, scorer, final, iso(scored_at_ms)) for final in finals]


def write_call(conn: sqlite3.Connection, call: Mapping[str, Any], item_seqs: Sequence[int], finals: Iterable[Final],
               scored_at_ms: int, probe_rows: Iterable[Mapping[str, Any]] = (),
               state: Mapping[str, str] | None = None) -> int:
    """Insert one ended call with its items in batch order, the final rows it decided, any probe rows and any
    scorer_state values, in one transaction, returning its call_id. The finals and probe rows get that call_id."""
    if "call_id" in call:
        raise ScoreStoreError("call_id is given by the database")
    with db.transaction(conn):
        call_id = db.insert(conn, "llm_calls", call)
        for index, item_seq in enumerate(item_seqs):
            db.insert(conn, "llm_call_items", {"call_id": call_id, "item_seq": item_seq,
                                               "position": position_id(index)})
        for final in finals:
            _insert_final(conn, call["scorer"], replace(final, call_id=call_id), iso(scored_at_ms))
        for row in probe_rows:
            db.insert(conn, "llm_probes", {**row, "call_id": call_id})
        if state:
            db.set_values(conn, "scorer_state", state)
    return call_id
