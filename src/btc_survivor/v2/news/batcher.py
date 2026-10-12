"""The batching rule of PLAN_V2 section 8.3, pure, with the time passed in.

One Batcher serves one model. Finnhub items wait for Haiku in receive order, ties broken by item seq, and a batch goes
when flush_max_items wait or the oldest has waited flush_after_seconds, at any hour. An official item goes alone at
once to each model and never shares a call with a Finnhub item, and Sonnet takes only its official kinds. An item
reaches a model only while it is younger than max_queue_age_seconds and its vendor time passes the 24 hour guard, and
otherwise gets its final expired or stale_vendor status at once, so FinBERT never waits. While send_to_llm is false
no Finnhub item is batched. A split gives the first half floor(n / 2) items.
"""

from __future__ import annotations

import bisect
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from btc_survivor.clock import MS_PER_SECOND, parse_iso
from btc_survivor.v2.config import NewsConfig
from btc_survivor.v2.news.collector import NEW, vendor_status
from btc_survivor.v2.news.prefilter import FINNHUB, LLM

EXPIRED, SKIPPED_OFF = "expired", "skipped_off"
BATCH, SINGLE = "batch", "single"          # the kinds of llm_calls rows the batcher forms
T = TypeVar("T")


@dataclass(frozen=True)
class BatchSettings:
    """The rule for one model. Haiku takes every routed item, batching Finnhub items and taking the rest alone, and
    Sonnet takes only its official kinds, each alone. kinds is None for a model that takes every routed item."""

    kinds: tuple[str, ...] | None
    alone_kinds: tuple[str, ...]
    send_to_llm: bool
    flush_max_items: int
    flush_after_ms: int
    max_queue_age_ms: int
    max_vendor_age_hours: float

    @classmethod
    def haiku(cls, config: NewsConfig) -> BatchSettings:
        return cls._for(config, None)

    @classmethod
    def sonnet(cls, config: NewsConfig) -> BatchSettings:
        return cls._for(config, config.llm.sonnet.kinds)

    @classmethod
    def _for(cls, config: NewsConfig, kinds: tuple[str, ...] | None) -> BatchSettings:
        llm = config.llm
        return cls(kinds, llm.alone_kinds, config.finnhub.send_to_llm, min(llm.flush_max_items, llm.max_items_per_call),
                   round(llm.flush_after_seconds * MS_PER_SECOND), round(llm.max_queue_age_seconds * MS_PER_SECOND),
                   config.news.max_vendor_age_hours)


@dataclass(frozen=True)
class Pending:
    """An item routed to a model, with the fields of its news_items row the rule reads."""

    seq: int
    source: str
    kind: str
    route: str
    status: str
    received_at: str
    vendor_ts: str | None = None
    received_ms: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "received_ms", parse_iso(self.received_at))

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Pending:
        return cls(row["seq"], row["source"], row["kind"], row["route"], row["status"], row["received_at"],
                   row["vendor_ts"])

    @property
    def order(self) -> tuple[int, int]:
        """Receive order, ties broken by item seq."""
        return self.received_ms, self.seq


@dataclass(frozen=True)
class Batch:
    """One call's items in order, kind batch for Finnhub items and single for one official item."""

    kind: str
    items: tuple[Pending, ...]

    @property
    def seqs(self) -> tuple[int, ...]:
        return tuple(item.seq for item in self.items)


@dataclass(frozen=True)
class Taken:
    """What take found at one instant, the items that expired while waiting and the next batch due, if any."""

    expired: tuple[Pending, ...]
    batch: Batch | None


def split(items: Sequence[T]) -> tuple[Sequence[T], Sequence[T]]:
    """The two halves of a batch to retry after a refusal or a cut off answer, the first floor(n / 2) items long."""
    if len(items) < 2:
        raise ValueError("a single item cannot be split")
    half = len(items) // 2
    return items[:half], items[half:]


class Batcher:
    """The waiting items of one model, released by take in a fixed order."""

    def __init__(self, settings: BatchSettings) -> None:
        self.settings = settings
        self._waiting: list[Pending] = []
        self._seen: set[int] = set()

    def __len__(self) -> int:
        return len(self._waiting)

    def alone(self, item: Pending) -> bool:
        """Whether an item takes a call of its own, every official item and anything not from Finnhub."""
        return item.kind in self.settings.alone_kinds or item.source != FINNHUB

    def wants(self, item: Pending) -> bool:
        """Whether an item is routed to this model at all, any new llm item for Haiku and for Sonnet only one of its
        official kinds, never a Finnhub item."""
        if item.route != LLM or item.status != NEW:
            return False
        kinds = self.settings.kinds
        return kinds is None or (item.source != FINNHUB and item.kind in kinds)

    def admit(self, item: Pending, now_ms: int) -> str | None:
        """Queue an item routed to this model and return None, or return its final status at once, stale_vendor past
        the vendor guard, skipped_off for a Finnhub item while send_to_llm is false, or expired once
        max_queue_age_seconds have passed since its receipt."""
        if not self.wants(item):
            raise ValueError(f"item {item.seq} is not routed to this model")
        if item.seq in self._seen:
            raise ValueError(f"item {item.seq} was already admitted")
        self._seen.add(item.seq)
        status = vendor_status(item.received_at, item.vendor_ts, self.settings.max_vendor_age_hours)
        if status is not None:
            return status
        if item.source == FINNHUB and not self.settings.send_to_llm:
            return SKIPPED_OFF
        if now_ms - item.received_ms >= self.settings.max_queue_age_ms:
            return EXPIRED
        self._insert(item)
        return None

    def requeue(self, item: Pending) -> None:
        """Put back an item that an answer left out, for its one requeue, in its receive order."""
        if item.seq not in self._seen or any(waiting.seq == item.seq for waiting in self._waiting):
            raise ValueError(f"item {item.seq} was never admitted or is still waiting")
        self._insert(item)

    def _insert(self, item: Pending) -> None:
        bisect.insort(self._waiting, item, key=lambda waiting: waiting.order)

    def drain(self) -> tuple[Pending, ...]:
        """Remove every waiting item, for its final row when the model is paused, capped or off."""
        drained, self._waiting = tuple(self._waiting), []
        return drained

    def take(self, now_ms: int) -> Taken:
        """Expire every item that reached max_queue_age_seconds, then release the next due batch, an official item at
        once or the oldest Finnhub items once enough wait or the oldest has waited long enough. Of two due batches the
        one whose first item came first goes first."""
        age = self.settings.max_queue_age_ms
        expired = tuple(item for item in self._waiting if now_ms - item.received_ms >= age)
        self._waiting = [item for item in self._waiting if now_ms - item.received_ms < age]
        single = next((item for item in self._waiting if self.alone(item)), None)
        finnhub = [item for item in self._waiting if not self.alone(item)]
        due = finnhub and (len(finnhub) >= self.settings.flush_max_items
                           or now_ms - finnhub[0].received_ms >= self.settings.flush_after_ms)
        if due and (single is None or finnhub[0].order < single.order):
            batch = Batch(BATCH, tuple(finnhub[:self.settings.flush_max_items]))
        elif single is not None:
            batch = Batch(SINGLE, (single,))
        else:
            batch = None
        if batch is not None:
            taken = set(batch.seqs)
            self._waiting = [item for item in self._waiting if item.seq not in taken]
        return Taken(expired, batch)
