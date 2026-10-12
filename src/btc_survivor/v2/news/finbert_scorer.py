"""The FinBERT scorer of PLAN_V2 sections 7.3 and 8.7, scorer finbert_v1, which gives every item routed to llm or
finbert one final row as it arrives, with no cost or cap.

Items are taken in seq order from the news process's writer connection on the event loop thread, and each is scored
in a worker thread, so polling never stalls. Before scoring, an item whose vendor time fails the 24 hour guard ends
stale_vendor, through the same vendor_status every scorer calls, and an item without tickers ends untagged, because
FinBERT cannot tell which company a text is about. A scored item gets one impact per ticker the prefilter fixed at
first receipt, the EDGAR filer for a filing and MACRO for Fed and BLS, each with strength p_pos minus p_neg. A text
the model cannot score ends failed, logged with the error class alone and never tried again. scored_at is stamped by
the news process's one clock right when the result exists, and the row and its impacts commit in one transaction with
no await in between, so seq order equals scored_at order. A restart picks up every routed item still without a row.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from btc_survivor.clock import LiveClock, iso
from btc_survivor.v2.config import NewsConfig
from btc_survivor.v2.news import db, health
from btc_survivor.v2.news.collector import vendor_status, wait_or_stop
from btc_survivor.v2.news.finbert import Probs
from btc_survivor.v2.news.prefilter import FINBERT, LLM

log = logging.getLogger(__name__)

SCORER = health.FINBERT_SCORER                # finbert_v1, the name news health and the score streams read
ROUTES = (LLM, FINBERT)
SCORED, FAILED, UNTAGGED = "scored", "failed", "untagged"
POLL_SECONDS = 1.0                            # how often an idle scorer looks for new items
BATCH_ITEMS = 100                             # items read per query

_PENDING = f"""
SELECT i.seq, i.route, i.received_at, i.vendor_ts, i.title, i.summary, i.tickers, b.text AS body
FROM news_items i LEFT JOIN news_bodies b ON b.item_seq = i.seq
WHERE i.seq > ? AND i.seq <= ? AND i.route IN ('{LLM}', '{FINBERT}')
  AND NOT EXISTS (SELECT 1 FROM item_scores s WHERE s.item_seq = i.seq AND s.scorer = ?)
ORDER BY i.seq LIMIT ?
"""


class Model(Protocol):
    """What the scorer needs from Finbert, so tests can pass a fake model."""

    def score(self, headline: str, text: str | None = None) -> Probs: ...


@dataclass(frozen=True)
class Item:
    """One routed item as FinBERT reads it. text is the stored body when the item has one, such as a filing or an
    FOMC statement, and otherwise its summary."""

    seq: int
    route: str
    received_at: str
    vendor_ts: str | None
    title: str
    text: str
    tickers: tuple[str, ...]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Item:
        return cls(row["seq"], row["route"], row["received_at"], row["vendor_ts"], row["title"],
                   row["body"] or row["summary"] or "", tuple(t for t in row["tickers"].split(",") if t))


def early_status(item: Item, max_vendor_age_hours: float) -> str | None:
    """The final status an item gets without running the model, stale_vendor before untagged, or None when it is
    scored."""
    return vendor_status(item.received_at, item.vendor_ts, max_vendor_age_hours) or (
        None if item.tickers else UNTAGGED)


def write(conn: sqlite3.Connection, item: Item, status: str, probs: Probs | None, attempts: int,
          scored_at_ms: int) -> int:
    """Insert the one final row of an item with its impacts, when scored, in one transaction, returning its seq. The
    unique key on item and scorer refuses a second final row."""
    row = {"item_seq": item.seq, "scorer": SCORER, "route": item.route, "status": status, "call_id": None,
           "attempts": attempts, "scored_at": iso(scored_at_ms),
           "p_pos": None if probs is None else probs.pos, "p_neg": None if probs is None else probs.neg,
           "p_neu": None if probs is None else probs.neu}
    with db.transaction(conn):
        seq = db.insert(conn, "item_scores", row)
        if probs is not None:
            for ticker in item.tickers:
                db.insert(conn, "score_impacts", {"item_seq": item.seq, "scorer": SCORER, "ticker": ticker,
                                                  "strength": probs.strength})
    return seq


class FinbertScorer:
    """Scores the routed items of one news process with a loaded model. Its state is on from the start, because the
    model was verified and loaded before the process began collecting."""

    def __init__(self, config: NewsConfig, conn: sqlite3.Connection, clock: LiveClock, model: Model,
                 to_thread: Callable[..., Awaitable[Probs]] = asyncio.to_thread) -> None:
        self.conn = conn
        self.clock = clock
        self.model = model
        self.to_thread = to_thread
        self.max_vendor_age_hours = config.news.max_vendor_age_hours
        self.state = health.ON
        self.counts: Counter[str] = Counter()
        self._after = 0

    def pending(self) -> list[Item]:
        """The next routed items after the last one handled that have no FinBERT row yet, in seq order. When none
        wait, every item up to the newest seq read first is done, so later looks start after it."""
        top = self.conn.execute("SELECT coalesce(max(seq), 0) FROM news_items").fetchone()[0]
        rows = self.conn.execute(_PENDING, (self._after, top, SCORER, BATCH_ITEMS)).fetchall()
        if not rows:
            self._after = max(self._after, top)
        return [Item.from_row(row) for row in rows]

    async def score(self, item: Item) -> str:
        """Give one item its final row and return its status."""
        status = early_status(item, self.max_vendor_age_hours)
        probs, attempts = None, 0
        if status is None:
            attempts = 1
            try:
                probs = await self.to_thread(self.model.score, item.title, item.text)
                status = SCORED
            except Exception as exc:  # one text the model cannot score must not stop the news process
                log.error("finbert could not score item %d (%s)", item.seq, type(exc).__name__)
                status = FAILED
        write(self.conn, item, status, probs, attempts, self.clock.stamp())
        self._after = item.seq
        self.counts[status] += 1
        return status

    async def run(self, stop: asyncio.Event,
                  wait: Callable[[asyncio.Event, float], Awaitable[bool]] = wait_or_stop) -> None:
        """Score items in seq order until stop is set, finishing the item in hand, and look for new ones every
        POLL_SECONDS while none wait."""
        while not stop.is_set():
            items = self.pending()
            for item in items:
                await self.score(item)
                if stop.is_set():
                    return
            if not items and await wait(stop, POLL_SECONDS):
                return
