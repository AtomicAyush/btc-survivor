"""Receive time one minute candles per symbol from Finnhub prints, by PLAN_V2 section 4.3.

The candle at boundary B covers receive stamps in [B - 60 s, B) and takes a print only when its receive stamp and its
vendor time both fall in the regular session, so pre and post market prints live only in the raw recording. Every
print a candle takes counts for volume, while only eligible prints set open, high, low and close, the reference
price, the median of the last 5 eligible prints of the session, and the tick rule behind buy and sell volume. A late
print or one with an excluded condition counts for volume only, and a replayed print or one more than 2 seconds
early counts for nothing, because its vendor time cannot be trusted. The reference buffer and the trade signs reset
at each open, and an empty minute repeats the previous close with zero volume and is never healthy.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, replace

from btc_survivor.clock import iso, next_minute
from btc_survivor.v2 import sessions
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks.finnhub import PrintBatch
from btc_survivor.v2.stocks.prints import (EARLY, ELIGIBLE, LATE, PriceRules, eligibility, in_session,
                                           is_replayed)

REFERENCE_PRINTS = 5
# Reasons a print enters no candle, counted on the builder.
REPLAYED, OUTSIDE = "replayed", "outside_session"


@dataclass(frozen=True, slots=True)
class StockCandle:
    """One symbol minute at boundary ts with session minute index k, the columns of the stock_candles table."""

    symbol: str
    ts: int
    k: int
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    ref_price: float | None
    volume: float
    buy_volume: float
    sell_volume: float
    prints: int
    price_prints: int
    late_prints: int
    valid: bool = False

    def judged(self, healthy: bool) -> StockCandle:
        """The candle with its minute's price health verdict, never valid without an eligible print."""
        return replace(self, valid=healthy and self.price_prints > 0)


class _Symbol:
    """One symbol's open candle, its reference buffer and its trade sign state."""

    def __init__(self) -> None:
        self.refs: deque[float] = deque(maxlen=REFERENCE_PRINTS)
        self.last_price: float | None = None
        self.sign = 0
        self.last_close: float | None = None
        self.last_eligible_ms: int | None = None
        self.clear()

    def clear(self) -> None:
        self.first = self.high = self.low = self.last = None
        self.volume = self.buy = self.sell = 0.0
        self.prints = self.price_prints = self.late_prints = 0

    def new_session(self) -> None:
        self.refs.clear()
        self.last_price = None
        self.sign = 0

    def add(self, price: float, size: float, kind: str, recv_ms: int) -> None:
        """Count a print for volume, and for price and the tick rule only when eligible."""
        self.volume += size
        self.prints += 1
        if kind == LATE:
            self.late_prints += 1
        if kind != ELIGIBLE:
            return
        if self.price_prints == 0:
            self.first = self.high = self.low = price
        else:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
        self.last = price
        self.price_prints += 1
        self.refs.append(price)
        self.last_eligible_ms = recv_ms
        # Plus one above the previous eligible print of the session, minus one below, the previous sign when equal.
        if self.last_price is not None and price != self.last_price:
            self.sign = 1 if price > self.last_price else -1
        self.last_price = price
        if self.sign > 0:
            self.buy += size
        elif self.sign < 0:
            self.sell += size

    def candle(self, symbol: str, boundary_ms: int, k: int) -> StockCandle:
        """Close the open candle, repeating the previous close when no eligible print came, and start the next."""
        if self.price_prints:
            self.last_close = self.last
            ohlc = (self.first, self.high, self.low, self.last)
        else:
            ohlc = (self.last_close,) * 4
        ref = statistics.median(self.refs) if self.refs else None
        candle = StockCandle(symbol, boundary_ms, k, *ohlc, ref, self.volume, self.buy, self.sell, self.prints,
                             self.price_prints, self.late_prints)
        self.clear()
        return candle


class StockCandles:
    """Candles for every symbol, fed print batches in receive order and closed at every session boundary in order,
    as the minute step does by advancing to each event's stamp before handling it."""

    def __init__(self, symbols: Sequence[str], rules: PriceRules) -> None:
        self.symbols = tuple(symbols)
        self.rules = rules
        self.session: Session | None = None
        self.connected_ms: int | None = None
        self.ignored: Counter[str] = Counter()
        self._state = {symbol: _Symbol() for symbol in self.symbols}
        self._end_ms: int | None = None
        self._closed_ms: int | None = None

    def on_connected(self, recv_ms: int) -> None:
        """Note the stamp of a connected notice, before which a print more than 1 second old counts as replayed."""
        self.connected_ms = recv_ms

    def last_eligible_ms(self, symbol: str) -> int | None:
        """The receive stamp of the symbol's last eligible session print, for freshness."""
        return self._state[symbol].last_eligible_ms

    def add(self, batch: PrintBatch) -> None:
        """Put each print of a batch in the open candle of its symbol, or count why it enters none."""
        self._check_order(batch.recv_ms)
        session = self._session_at(batch.recv_ms)
        for p in batch.prints:
            if is_replayed(p.vendor_ms, self.connected_ms):
                self.ignored[REPLAYED] += 1
            elif session is None or not in_session(session, batch.recv_ms, p.vendor_ms):
                self.ignored[OUTSIDE] += 1
            else:
                kind = eligibility(batch.recv_ms, p.vendor_ms, p.conditions, self.rules)
                if kind == EARLY:
                    self.ignored[EARLY] += 1
                else:
                    self._state[p.symbol].add(p.price, p.size, kind, batch.recv_ms)
                    self._end_ms = next_minute(batch.recv_ms)

    def close(self, boundary_ms: int) -> tuple[StockCandle, ...]:
        """Close the session minute ending at boundary_ms for every symbol in config order, not yet judged healthy."""
        session = sessions.session_of_boundary(boundary_ms)
        if session is None:
            raise ValueError(f"{iso(boundary_ms)} closes no session minute")
        if self._closed_ms is not None and boundary_ms <= self._closed_ms:
            raise ValueError(f"{iso(boundary_ms)} is not after the last closed boundary {iso(self._closed_ms)}")
        if self._end_ms is not None and self._end_ms != boundary_ms:
            raise ValueError(f"prints wait for the boundary {iso(self._end_ms)}, not {iso(boundary_ms)}")
        if session != self.session:
            self._enter(session)
        k = session.k(boundary_ms)
        self._end_ms = None
        self._closed_ms = boundary_ms
        return tuple(self._state[symbol].candle(symbol, boundary_ms, k) for symbol in self.symbols)

    def _check_order(self, recv_ms: int) -> None:
        """Refuse a batch received before the last closed boundary, or after a boundary its waiting prints need."""
        if self._closed_ms is not None and recv_ms < self._closed_ms:
            raise ValueError(f"a batch received at {iso(recv_ms)} belongs to a closed minute")
        if self._end_ms is not None and next_minute(recv_ms) != self._end_ms:
            raise ValueError(f"close the minute ending {iso(self._end_ms)} before a batch received at {iso(recv_ms)}")

    def _session_at(self, recv_ms: int) -> Session | None:
        """The session whose regular hours hold the receive stamp, entering it when it is a new one."""
        current = self.session
        if current is not None and current.open_ms <= recv_ms < current.close_ms:
            return current
        session = sessions.session_at(recv_ms)
        if session is not None:
            self._enter(session)
        return session

    def _enter(self, session: Session) -> None:
        """Start a session, which empties every reference buffer and resets every trade sign."""
        self.session = session
        for state in self._state.values():
            state.new_session()
