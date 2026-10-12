"""PooledWindow and predict_rows of PLAN_V2 sections 3 and 6.4, where the version 1 MaturedWindow and MLP.predict do
not fit.

The window keeps a symbol column, holds the samples that matured over the last replay_window_minutes session minutes
from every symbol, up to that many minutes times the number of symbols, returns the newest n samples plus draws with
replacement, and divides each label by its own symbol's scale at batch time. predict_rows runs one forward pass over a
row per symbol, and adds the news part of the hidden pre activation only when some row has a nonzero news input, so S1
and S2 give bitwise equal outputs while their news inputs are zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from btc_survivor.model import MLP
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES

N_NEWS_INPUTS = 2


class PooledWindow:
    """A ring of samples in the order they matured, each with its features, label in bps, symbol and minute k."""

    def __init__(self, symbols: Sequence[str], window_minutes: int, n_inputs: int) -> None:
        self.symbols = tuple(symbols)
        self._index = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.window_minutes = window_minutes
        self.capacity = window_minutes * len(self.symbols)
        self.x = np.zeros((self.capacity, n_inputs))
        self.label = np.zeros(self.capacity)
        self.symbol = np.zeros(self.capacity, dtype=np.int64)
        self.minute = np.zeros(self.capacity, dtype=np.int64)
        self.start = 0
        self.count = 0

    @classmethod
    def from_config(cls, cfg: StockConfig) -> PooledWindow:
        return cls(cfg.stock_feed.symbols, cfg.model.replay_window_minutes, cfg.model.inputs)

    def add(self, k: int, symbol: str, x: np.ndarray, label_bps: float) -> None:
        """Add a sample that matured at session minute k, after dropping those older than the window."""
        index = self._index[symbol]
        x = np.asarray(x, dtype=float)
        if x.shape != self.x.shape[1:]:
            raise ValueError(f"a sample has shape {x.shape}, not ({self.x.shape[1]},)")
        if self.count and k < self.minute[self._slot(self.count - 1)]:
            raise ValueError(f"a sample at minute {k} is older than the newest in the window")
        self.evict(k)
        if self.count == self.capacity:
            self.start = (self.start + 1) % self.capacity
            self.count -= 1
        slot = self._slot(self.count)
        self.x[slot], self.label[slot], self.symbol[slot], self.minute[slot] = x, label_bps, index, k
        self.count += 1

    def evict(self, k: int) -> None:
        """Drop every sample that matured window_minutes or more session minutes before k."""
        while self.count and self.minute[self.start] <= k - self.window_minutes:
            self.start = (self.start + 1) % self.capacity
            self.count -= 1

    def column(self) -> tuple[str, ...]:
        """The symbol of every sample, oldest first."""
        return tuple(self.symbols[self.symbol[self._slot(i)]] for i in range(self.count))

    def batch(self, rng: np.random.Generator, newest: int, size: int,
              scales: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """The newest samples in the order they matured, plus size minus newest draws with replacement from the
        whole window, with each label divided by its own symbol's scale."""
        if not 0 <= newest <= self.count:
            raise ValueError(f"the window holds {self.count} samples, not the {newest} newest asked for")
        rows = [self._slot(self.count - newest + i) for i in range(newest)]
        if size > newest:
            rows.extend(self._slot(int(i)) for i in rng.integers(0, self.count, size - newest))
        idx = np.array(rows, dtype=np.int64)
        scale = np.array([scales[symbol] for symbol in self.symbols])
        return self.x[idx], self.label[idx] / scale[self.symbol[idx]]

    def _slot(self, i: int) -> int:
        """The ring slot of the i-th oldest sample."""
        return (self.start + i) % self.capacity


def network_inputs(z: np.ndarray, news: np.ndarray) -> np.ndarray:
    """One row of network inputs, the 10 price z scores followed by the company and macro news inputs."""
    return np.concatenate([z, news])


def predict_rows(model: MLP, X: np.ndarray, n_price: int = N_PRICE_FEATURES) -> np.ndarray:
    """One forward pass over the rows of X, the price part of the hidden pre activation plus the news part, which is
    skipped while every row's news inputs are zero."""
    pre = X[:, :n_price] @ model.W1[:, :n_price].T
    news = X[:, n_price:]
    if news.any():
        pre = pre + news @ model.W1[:, n_price:].T
    return np.tanh(pre + model.b1) @ model.W2 + model.b2[0]


def with_news_inputs(model: MLP, n_news: int = N_NEWS_INPUTS) -> MLP:
    """A copy of a network born at the price inputs with zero weight columns appended for the news inputs, as both
    runs build every learner."""
    wide = MLP(model.n_in + n_news, model.hidden)
    wide.W1[:, :model.n_in] = model.W1
    wide.b1[:] = model.b1
    wide.W2[:] = model.W2
    wide.b2[:] = model.b2
    return wide
