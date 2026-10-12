"""StockSeries, the per symbol minute series of PLAN_V2 section 3, indexed by the session minute index k.

It wraps the version 1 MinuteSeries, which keeps the reference price in its mid ring with the volume and validity, by
passing k minutes as its stamp, so a night, weekend or holiday adds no minutes. It adds buy and sell volume rings,
takes session starts from sessions.py, and holds at least 420 minutes, more than the 391 minutes of history the
longest feature reads.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from btc_survivor.clock import MS_PER_MINUTE
from btc_survivor.feed.candles import MinuteSeries
from btc_survivor.v2 import sessions
from btc_survivor.v2.stocks.candles import StockCandle

MIN_CAPACITY = 420


class SeriesWindow(NamedTuple):
    """The minutes ending at one k, oldest first, with zero volumes and a NaN price where a minute is not valid."""

    k: np.ndarray
    ref_price: np.ndarray
    volume: np.ndarray
    buy_volume: np.ndarray
    sell_volume: np.ndarray
    valid: np.ndarray


class StockSeries:
    """Rings of reference price, volume, buy and sell volume and validity per session minute index k."""

    def __init__(self, capacity: int = MIN_CAPACITY) -> None:
        if capacity < MIN_CAPACITY:
            raise ValueError(f"a stock series holds at least {MIN_CAPACITY} minutes, not {capacity}")
        self.capacity = capacity
        self.minutes = MinuteSeries(capacity)
        self.buy_volume = np.zeros(capacity)
        self.sell_volume = np.zeros(capacity)

    @property
    def last_k(self) -> int | None:
        return self.minutes.last_index

    @property
    def valid_total(self) -> int:
        """Healthy minutes ever appended, for the history a feature needs first."""
        return self.minutes.valid_total

    def append(self, k: int, ref_price: float | None, volume: float, buy_volume: float, sell_volume: float,
               valid: bool) -> None:
        """Record session minute k. Minutes skipped since the last call are invalid, as in MinuteSeries."""
        if k < 1:
            raise ValueError(f"session minute index {k} is below 1")
        self.minutes.append(k * MS_PER_MINUTE, ref_price, volume, valid)
        ok = valid and ref_price is not None
        self.buy_volume[k % self.capacity] = buy_volume if ok else 0.0
        self.sell_volume[k % self.capacity] = sell_volume if ok else 0.0

    def append_candle(self, candle: StockCandle) -> None:
        """Record a judged candle at its k."""
        self.append(candle.k, candle.ref_price, candle.volume, candle.buy_volume, candle.sell_volume, candle.valid)

    def window(self, end_k: int, length: int) -> SeriesWindow:
        """The length minutes ending at end_k. Minutes older than the ring, newer than the last append or skipped
        are not valid, and a stale buy or sell slot is masked by that validity."""
        ref, volume, valid = self.minutes.window(end_k, length)
        ks = np.arange(end_k - length + 1, end_k + 1)
        slots = ks % self.capacity
        return SeriesWindow(ks, ref, volume, np.where(valid, self.buy_volume[slots], 0.0),
                            np.where(valid, self.sell_volume[slots], 0.0), valid)

    @staticmethod
    def session_start(k: int) -> int:
        """The index k of the first minute, ending 09:31, of the session holding minute k."""
        return sessions.session_of_k(k).first_k
