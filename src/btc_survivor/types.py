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


@dataclass(slots=True)
class Leg:
    """One market order inside an Order. A close leg always uses the quantity held when it fills."""

    action: str  # open or close
    position_side: int  # LONG or SHORT, the position this leg opens or closes
    qty: float = 0.0  # only used by open legs

    @property
    def taker_side(self) -> str:
        opening_long = self.position_side == LONG
        return "buy" if (self.action == "open") == opening_long else "sell"


@dataclass(slots=True)
class Order:
    bot_id: int
    legs: list[Leg]
    decided_ms: int
    due_ms: int
    ref_mid: float
    reason: str  # decision, stop, death, restart_flatten, gap_flatten
    forced: bool = False

    @property
    def latency_ms(self) -> int:
        return self.due_ms - self.decided_ms


@dataclass(frozen=True, slots=True)
class Fill:
    bot_id: int
    trade_id: int | None
    leg: str
    taker_side: str
    qty: float
    ref_mid: float
    decided_ms: int
    filled_ms: int | None
    book_avg_price: float | None
    price: float | None
    levels: int | None
    slippage_bps: float | None
    fee: float | None
    latency_ms: int
    status: str  # filled or rejected
    reason: str


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    trade_id: int
    side: int
    qty: float
    entry_price: float
    exit_price: float
    opened_ms: int
    closed_ms: int
    fee_open: float
    fee_close: float
    funding: float
    gross: float
    net: float
    slippage_bps_open: float
    slippage_bps_close: float
    exit_reason: str
    forced: bool
