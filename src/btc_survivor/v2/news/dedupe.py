"""The deduplication steps of PLAN_V2 section 7.5 as pure decisions the collector applies inside its poll.

Step 1 compares vendor ids, step 4 the headline hash and step 5 the token sets, and every step stays within one
source. The collector passes the items of the same source already in news_items, seeds, backlog and filtered items
included, and a duplicate never changes its original, so it only writes a news_dups row pointing at it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from btc_survivor.clock import MS_PER_HOUR
from btc_survivor.v2.config import DedupeConfig
from btc_survivor.v2.news.text import title_hash, tokens

# What step 1 decides for a known vendor id, and the news_dups methods.
SILENT = "silent"
VENDOR_ID = "vendor_id"
HASH = "hash"
NEAR = "near"


@dataclass(frozen=True)
class Seen:
    """An item already in news_items, as the hash and near duplicate steps compare against it."""

    seq: int
    source: str
    received_ms: int
    title_hash: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class Duplicate:
    """The original an item duplicates, the news_dups method, and the Jaccard index for a near duplicate."""

    seq: int
    method: str
    jaccard: float | None = None


def seen(seq: int, source: str, received_ms: int, title: str) -> Seen:
    """A news_items row as the dedupe steps see it, from its stored title."""
    return Seen(seq, source, received_ms, title_hash(title), tokens(title))


def repoll(known_channels: Collection[str], channel: str) -> str | None:
    """Step 1 for a (source, vendor_id) that the channels in known_channels have already returned, which are the
    item's first channel plus the channel of every news_dups row with that vendor id. None means the id is unknown
    and the item goes on, SILENT means a repoll that writes nothing, and VENDOR_ID means one news_dups row."""
    if not known_channels:
        return None
    return SILENT if channel in known_channels else VENDOR_ID


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """The size of the intersection over the size of the union, 0 for two empty sets."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _within(received_ms: int, item: Seen, hours: int) -> bool:
    return 0 <= received_ms - item.received_ms <= hours * MS_PER_HOUR


def find_duplicate(source: str, received_ms: int, title: str, recent: Iterable[Seen],
                   settings: DedupeConfig) -> Duplicate | None:
    """Steps 4 and 5 for an item with a new vendor id. The earliest item of the same source with the same hash in the
    hash window wins, else the item of the same source in the near window with the highest Jaccard index of at
    least near_dup_jaccard, both token sets holding at least near_dup_min_tokens, ties going to the earliest."""
    same = sorted((item for item in recent if item.source == source), key=lambda item: item.seq)
    digest = title_hash(title)
    for item in same:
        if item.title_hash == digest and _within(received_ms, item, settings.hash_window_hours):
            return Duplicate(item.seq, HASH)
    words = tokens(title)
    if len(words) < settings.near_dup_min_tokens:
        return None
    best: Duplicate | None = None
    for item in same:
        if len(item.tokens) < settings.near_dup_min_tokens:
            continue
        score = jaccard(words, item.tokens)
        if (score >= settings.near_dup_jaccard and _within(received_ms, item, settings.near_dup_window_hours)
                and (best is None or score > best.jaccard)):
            best = Duplicate(item.seq, NEAR, score)
    return best


def new_tickers(duplicate_tickers: Sequence[str], original_tickers: Sequence[str]) -> str:
    """The duplicate's tickers its original lacks, comma separated for news_dups.new_tickers and empty for none, kept
    for measurement only, because an item's tickers are fixed at first receipt."""
    return ",".join(t for t in duplicate_tickers if t not in original_tickers)
