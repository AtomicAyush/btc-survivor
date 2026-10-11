"""Small shared data types. Feed events are exchange neutral, so only an adapter knows wire formats."""

from __future__ import annotations

from dataclasses import dataclass

BID = "bid"
ASK = "ask"

LONG = 1
FLAT = 0
SHORT = -1


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """A full replacement of the local book. Levels are (price, size) pairs."""

    recv_ms: int
    conn_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    exch_ms: int | None = None


@dataclass(frozen=True, slots=True)
class BookDelta:
    """Absolute size changes as (side, price, size), where size 0 removes the level."""

    recv_ms: int
    conn_id: int
    changes: list[tuple[str, float, float]]
    exch_ms: int | None = None


@dataclass(frozen=True, slots=True)
class Trade:
    recv_ms: int
    conn_id: int
    trade_id: int
    price: float
    size: float
    exch_ms: int | None = None


@dataclass(frozen=True, slots=True)
class FeedNotice:
    """Connection lifecycle and integrity notices, for example connected, heartbeat, gap or trade_gap."""

    recv_ms: int
    conn_id: int
    kind: str
    exch_ms: int | None = None
    detail: str = ""


FeedEvent = BookSnapshot | BookDelta | Trade | FeedNotice
