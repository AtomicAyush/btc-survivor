"""Local order book. Keeps full depth, because a cancel at the top pulls deeper levels into the top N."""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass

from btc_survivor.types import ASK, BID

_RESIDUE = 1e-12  # quantities left over by float rounding count as filled


@dataclass(frozen=True, slots=True)
class Walk:
    """Result of walking one side of the book for a quantity."""

    avg_price: float | None
    filled_qty: float
    unfilled_qty: float
    levels: int


class LocalBook:
    """Price to size dicts per side plus sorted price lists, with the best price at the end of each list.

    Asks are stored negated so both best prices sit at the list tail, where inserts and deletes are cheap.
    """

    def __init__(self, depth: int = 20) -> None:
        self.depth = depth
        self.valid = False
        self._sizes = {BID: {}, ASK: {}}
        self._keys: dict[str, list[float]] = {BID: [], ASK: []}

    def clear(self) -> None:
        self.valid = False
        self._sizes = {BID: {}, ASK: {}}
        self._keys = {BID: [], ASK: []}

    def apply_snapshot(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        """Replace the whole book. The book is valid afterwards unless it is crossed."""
        self.clear()
        for side, levels in ((BID, bids), (ASK, asks)):
            sizes = {p: s for p, s in levels if s > 0}
            self._sizes[side] = sizes
            self._keys[side] = sorted(sizes) if side == BID else sorted(-p for p in sizes)
        self.valid = not self.is_crossed()

    def apply_changes(self, changes: list[tuple[str, float, float]]) -> None:
        """Apply absolute size changes. Removing an absent level is ignored, and a crossing change invalidates the book."""
        for side, price, size in changes:
            sizes = self._sizes[side]
            keys = self._keys[side]
            key = price if side == BID else -price
            if size > 0:
                if price not in sizes:
                    insort(keys, key)
                sizes[price] = size
            elif price in sizes:
                del sizes[price]
                i = bisect_left(keys, key)
                del keys[i]
        if self.valid and self.is_crossed():
            self.valid = False

    def is_crossed(self) -> bool:
        bid, ask = self.best_bid, self.best_ask
        return bid is not None and ask is not None and bid >= ask

    @property
    def best_bid(self) -> float | None:
        keys = self._keys[BID]
        return keys[-1] if keys else None

    @property
    def best_ask(self) -> float | None:
        keys = self._keys[ASK]
        return -keys[-1] if keys else None

    @property
    def mid(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @property
    def spread_bps(self) -> float | None:
        bid, ask, mid = self.best_bid, self.best_ask, self.mid
        if mid is None:
            return None
        return (ask - bid) / mid * 1e4

    def top(self, side: str, n: int | None = None) -> list[tuple[float, float]]:
        """The best n levels of one side, best first, as (price, size)."""
        n = self.depth if n is None else n
        keys = self._keys[side]
        sizes = self._sizes[side]
        chosen = keys[-n:][::-1] if n > 0 else []
        if side == BID:
            return [(p, sizes[p]) for p in chosen]
        return [(-k, sizes[-k]) for k in chosen]

    def walk(self, taker_side: str, qty: float) -> Walk:
        """Fill qty against the visible top levels. A buy consumes asks and a sell consumes bids."""
        levels = self.top(ASK if taker_side == "buy" else BID)
        remaining = qty
        notional = 0.0
        used = 0
        for price, size in levels:
            if remaining <= _RESIDUE:
                break
            take = min(remaining, size)
            notional += take * price
            remaining -= take
            used += 1
        filled = qty - max(remaining, 0.0)
        if remaining <= _RESIDUE:
            filled, remaining = qty, 0.0
        avg = notional / filled if filled > 0 else None
        return Walk(avg_price=avg, filled_qty=filled, unfilled_qty=remaining, levels=used)

    def level_count(self, side: str) -> int:
        return len(self._keys[side])
