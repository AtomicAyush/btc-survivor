"""The 11 shared features, computed once per minute from the minute series and the book at the boundary.

Order: returns over the four return windows, realized volatility over the two volatility windows, the volume
ratio, book imbalance, spread, and the sine and cosine of the UTC minute of the day. Everything reads only data
at or before the boundary, so a feature never sees the future.
"""

from __future__ import annotations

import math

import numpy as np

from btc_survivor.clock import MINUTES_PER_DAY, minute_index, minute_of_day
from btc_survivor.config import FeaturesConfig
from btc_survivor.feed.book import LocalBook
from btc_survivor.feed.candles import MinuteSeries
from btc_survivor.types import ASK, BID

N_FEATURES = 11
VOLUME_RATIO_CLIP = 3.0
VOLUME_WINDOW = 15
VOLUME_EPS = 1e-8


class FeatureBuilder:
    def __init__(self, cfg: FeaturesConfig, history_minutes_required: int) -> None:
        self.cfg = cfg
        self.history = history_minutes_required

    def compute(self, series: MinuteSeries, boundary_ms: int, book: LocalBook) -> np.ndarray | None:
        """The raw feature vector for the minute ending at boundary_ms, or None when it is not valid yet."""
        end = minute_index(boundary_ms)
        if series.valid_total_at(end) < self.history or not book.valid or book.mid is None:
            return None
        cover = self.cfg.min_window_coverage
        out = []
        longest = max(max(self.cfg.return_windows_minutes), max(self.cfg.vol_windows_minutes)) + 1
        mid, _, valid = series.window(end, max(longest, VOLUME_WINDOW))
        if not valid[-1]:
            return None
        for k in self.cfg.return_windows_minutes:
            if not valid[-1 - k]:
                return None
            out.append(1e4 * math.log(mid[-1] / mid[-1 - k]))
        for w in self.cfg.vol_windows_minutes:
            window_mid, window_ok = mid[-(w + 1):], valid[-(w + 1):]
            pairs = window_ok[1:] & window_ok[:-1]
            if pairs.sum() < cover * w or pairs.sum() < 2:
                return None
            returns = 1e4 * np.log(window_mid[1:][pairs] / window_mid[:-1][pairs])
            out.append(float(np.std(returns)))
        ratio = self._volume_ratio(series, end, cover)
        if ratio is None:
            return None
        out.append(ratio)
        out.append(_imbalance(book, self.cfg.imbalance_levels))
        out.append(book.spread_bps)
        angle = 2 * math.pi * minute_of_day(boundary_ms) / MINUTES_PER_DAY
        out.extend((math.sin(angle), math.cos(angle)))
        vector = np.array(out, dtype=float)
        return vector if np.all(np.isfinite(vector)) else None

    def _volume_ratio(self, series: MinuteSeries, end: int, cover: float) -> float | None:
        """Log of recent volume against its trailing baseline, both scaled from the valid minutes they hold."""
        baseline_len = self.cfg.volume_baseline_minutes
        _, volume, valid = series.window(end, baseline_len)
        recent_ok = valid[-VOLUME_WINDOW:]
        if recent_ok.sum() < cover * VOLUME_WINDOW or valid.sum() < cover * baseline_len:
            return None
        recent = volume[-VOLUME_WINDOW:][recent_ok].mean() * VOLUME_WINDOW
        baseline = volume[valid].mean() * VOLUME_WINDOW
        ratio = math.log((recent + VOLUME_EPS) / (baseline + VOLUME_EPS))
        return max(-VOLUME_RATIO_CLIP, min(VOLUME_RATIO_CLIP, ratio))


def _imbalance(book: LocalBook, levels: int) -> float:
    bids = sum(size for _, size in book.top(BID, levels))
    asks = sum(size for _, size in book.top(ASK, levels))
    total = bids + asks
    return (bids - asks) / total if total > 0 else 0.0
