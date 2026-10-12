"""The 10 price features of PLAN_V2 section 6.1 and their coverage, per symbol at each session boundary k.

Every feature reads the reference prices, volumes and trade signs that the candles built from eligible prints, through
a StockSeries indexed by the session minute index k, so a night, weekend or holiday adds nothing. In order they are
the 30 and 120 minute returns without the overnight move, the one session return with it, realized volatility over 60
minutes within the session, the volume ratio, the trade sign imbalance, the time since the open and to the close over
390, the overnight gap and the one session return relative to the index symbol. A return needs healthy reference
prices at both ends, volatility, the volume ratio and the imbalance need min_window_coverage of their minutes healthy,
feature 10 needs the index healthy at both ends, and one invalid input makes the symbol's whole vector invalid.

Everything reads the 391 minutes ending at k and nothing after it, so a series cut at k gives the same vector there as
the full series. The overnight gap takes the first healthy price of the session and the last healthy price of the
session before inside that window, which always holds both when the one session return is valid, since that return
needs a healthy price at k minus 390. The split guard of section 5.3 lives in broker.SymbolGuards, whose
features_allowed the market layer applies to these vectors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from btc_survivor.standardizer import Standardizer
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import FeaturesConfig, StockConfig
from btc_survivor.v2.stocks.series import SeriesWindow, StockSeries

N_PRICE_FEATURES = 10
FEATURE_NAMES = ("return_30", "return_120", "return_session", "volatility_60", "volume_ratio", "imbalance",
                 "since_open", "to_close", "overnight_gap", "relative_to_index")
VOLUME_EPS = 1e-8
# The yardstick of features 7 and 8, a full session.
DAY_MINUTES = sessions.FULL_SESSION_MINUTES


def gap_ratio(w: SeriesWindow, start: int) -> float | None:
    """The first healthy reference price of the session starting at start over the last healthy reference price of
    the session before it, both inside the window, or None when either is missing."""
    if start == 1:
        return None
    end = sessions.session_of_k(start).last_k
    before = sessions.session_of_k(start - 1).first_k
    today = w.valid & (w.k >= start) & (w.k <= end)
    previous = w.valid & (w.k >= before) & (w.k < start)
    if not (today.any() and previous.any()):
        return None
    return float(w.ref_price[np.argmax(today)] / w.ref_price[len(previous) - 1 - np.argmax(previous[::-1])])


class FeatureBuilder:
    """The raw vectors of every symbol at one boundary, from each symbol's StockSeries."""

    def __init__(self, cfg: FeaturesConfig, symbols: Sequence[str], index_symbol: str) -> None:
        if index_symbol not in symbols:
            raise ValueError(f"the index symbol {index_symbol} is not among the symbols")
        self.cfg = cfg
        self.symbols = tuple(symbols)
        self.index_symbol = index_symbol
        self.length = cfg.day_return_minutes + 1
        longest = max(max(cfg.return_windows_minutes) + 1, cfg.vol_window_minutes + 1, cfg.volume_baseline_minutes,
                      cfg.volume_window_minutes, cfg.tick_window_minutes)
        if longest > self.length:
            raise ValueError(f"a feature window of {longest} minutes is longer than the one session return")

    @classmethod
    def from_config(cls, cfg: StockConfig) -> FeatureBuilder:
        return cls(cfg.features, cfg.stock_feed.symbols, cfg.stock_feed.index_symbol)

    def compute(self, series: Mapping[str, StockSeries], k: int) -> dict[str, np.ndarray | None]:
        """Each symbol's 10 raw features at session minute k in config order, or None where its vector is invalid."""
        index_day = self._day_return(series[self.index_symbol].window(k, self.length))
        out: dict[str, np.ndarray | None] = {}
        for symbol in self.symbols:
            own = self._own(series[symbol].window(k, self.length), k)
            if own is None or (symbol != self.index_symbol and index_day is None):
                out[symbol] = None
                continue
            own.append(0.0 if symbol == self.index_symbol else own[2] - index_day)
            vector = np.array(own, dtype=float)
            out[symbol] = vector if np.all(np.isfinite(vector)) else None
        return out

    def _day_return(self, w: SeriesWindow) -> float | None:
        """Feature 3, the one session return including the overnight move, or None without both ends healthy."""
        n = self.cfg.day_return_minutes
        if not (w.valid[-1 - n] and w.valid[-1]):
            return None
        return 1e4 * math.log(w.ref_price[-1] / w.ref_price[-1 - n])

    def _own(self, w: SeriesWindow, k: int) -> list[float] | None:
        """Features 1 to 9 of one symbol, or None when any is invalid."""
        session = sessions.session_of_k(k)
        start = session.first_k
        minute = k - start + 1
        ratio = gap_ratio(w, start)
        if not w.valid[-1] or ratio is None:
            return None
        gap = 1e4 * math.log(ratio)
        out = []
        for n in self.cfg.return_windows_minutes:
            if not w.valid[-1 - n]:
                return None
            # A window that spans today's open leaves the overnight move out.
            out.append(1e4 * math.log(w.ref_price[-1] / w.ref_price[-1 - n]) - (gap if k - n < start else 0.0))
        for value in (self._day_return(w), self._volatility(w, start), self._volume_ratio(w), self._imbalance(w)):
            if value is None:
                return None
            out.append(value)
        out.extend((minute / DAY_MINUTES, (session.minutes - minute) / DAY_MINUTES, gap))
        return out

    def _covered(self, valid: np.ndarray, n: int) -> bool:
        return bool(valid[-n:].sum() >= self.cfg.min_window_coverage * n)

    def _volatility(self, w: SeriesWindow, start: int) -> float | None:
        """Feature 4, the standard deviation in bps of the one minute log returns within today's session."""
        n = self.cfg.vol_window_minutes
        ref, valid, ks = w.ref_price[-(n + 1):], w.valid[-(n + 1):], w.k[-(n + 1):]
        pairs = valid[1:] & valid[:-1] & (ks[1:] != start)
        if pairs.sum() < self.cfg.min_window_coverage * n or pairs.sum() < 2:
            return None
        return float(np.std(1e4 * np.log(ref[1:][pairs] / ref[:-1][pairs])))

    def _volume_ratio(self, w: SeriesWindow) -> float | None:
        """Feature 5, the log of the last 30 minutes' volume over the mean 30 minute volume of the trailing 390, each
        scaled up from its healthy minutes, clipped."""
        cfg = self.cfg
        recent_n, base_n = cfg.volume_window_minutes, cfg.volume_baseline_minutes
        if not (self._covered(w.valid, recent_n) and self._covered(w.valid, base_n)):
            return None
        recent = w.volume[-recent_n:][w.valid[-recent_n:]].mean() * recent_n
        baseline = w.volume[-base_n:][w.valid[-base_n:]].mean() * recent_n
        ratio = math.log((recent + VOLUME_EPS) / (baseline + VOLUME_EPS))
        return max(-cfg.volume_ratio_clip, min(cfg.volume_ratio_clip, ratio))

    def _imbalance(self, w: SeriesWindow) -> float | None:
        """Feature 6, buy minus sell volume over their sum across the last 30 minutes, zero with no signed volume."""
        n = self.cfg.tick_window_minutes
        if not self._covered(w.valid, n):
            return None
        buy, sell = float(w.buy_volume[-n:].sum()), float(w.sell_volume[-n:].sum())
        return (buy - sell) / (buy + sell) if buy + sell > 0 else 0.0


def make_standardizer(cfg: FeaturesConfig) -> Standardizer:
    """One symbol's version 1 Standardizer over the 10 features, its half life in healthy session minutes."""
    return Standardizer(N_PRICE_FEATURES, cfg.standardizer_half_life_hours * 60, cfg.min_samples, cfg.clip_z)
