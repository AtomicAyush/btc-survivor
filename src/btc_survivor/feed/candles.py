"""One minute candles from the trade stream, and the shared minute series that features read.

A minute is labelled by the boundary that closes it, so the candle at M covers receive times in [M minus 60 s, M)
and its mid is the book mid at M.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from btc_survivor.clock import minute_index


@dataclass(frozen=True, slots=True)
class Candle:
    ts: int
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float
    trade_count: int
    mid: float | None
    valid: bool


class CandleBuilder:
    """Collects trades by receive time. A minute with no trades repeats the previous close and has zero volume."""

    def __init__(self) -> None:
        self._last_close: float | None = None
        self._reset()

    def _reset(self) -> None:
        self._open = self._high = self._low = self._close = None
        self._volume = 0.0
        self._count = 0

    def add_trade(self, price: float, size: float) -> None:
        if self._count == 0:
            self._open = self._high = self._low = price
        else:
            self._high = max(self._high, price)
            self._low = min(self._low, price)
        self._close = price
        self._volume += size
        self._count += 1

    def close(self, boundary_ms: int, mid: float | None, valid: bool) -> Candle:
        if self._count:
            candle = Candle(boundary_ms, self._open, self._high, self._low, self._close,
                            self._volume, self._count, mid, valid)
            self._last_close = self._close
        else:
            last = self._last_close
            candle = Candle(boundary_ms, last, last, last, last, 0.0, 0, mid, valid)
        self._reset()
        return candle


class MinuteSeries:
    """Ring arrays of minute mid, volume and validity, indexed by UTC minute so windows always line up with the clock."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.mid = np.full(capacity, np.nan)
        self.volume = np.zeros(capacity)
        self.valid = np.zeros(capacity, dtype=bool)
        self.last_index: int | None = None

    def append(self, boundary_ms: int, mid: float | None, volume: float, valid: bool) -> None:
        """Record one minute. Minutes skipped since the last call are recorded as invalid."""
        index = minute_index(boundary_ms)
        if self.last_index is not None:
            if index <= self.last_index:
                raise ValueError("minutes must be appended in increasing order")
            for missing in range(max(self.last_index + 1, index - self.capacity + 1), index):
                self._write(missing, None, 0.0, False)
        self._write(index, mid, volume, valid)
        self.last_index = index

    def _write(self, index: int, mid: float | None, volume: float, valid: bool) -> None:
        slot = index % self.capacity
        ok = valid and mid is not None
        self.mid[slot] = mid if ok else np.nan
        self.volume[slot] = volume if ok else 0.0
        self.valid[slot] = ok

    def window(self, end_index: int, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mid, volume and validity for the length minutes ending at end_index, oldest first.

        Minutes older than the ring or newer than the last append come back as invalid.
        """
        if length > self.capacity:
            raise ValueError("window longer than the series capacity")
        indexes = np.arange(end_index - length + 1, end_index + 1)
        slots = indexes % self.capacity
        known = (indexes <= (self.last_index if self.last_index is not None else -1)) & \
                (indexes > (self.last_index if self.last_index is not None else 0) - self.capacity)
        valid = self.valid[slots] & known
        mid = np.where(valid, self.mid[slots], np.nan)
        volume = np.where(valid, self.volume[slots], 0.0)
        return mid, volume, valid
