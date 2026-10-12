"""PooledWindow on synthetic samples: the symbol column, a window of 240 session minutes from every symbol, the
newest n samples plus draws, each label over its own symbol's scale, and S1 and S2 windows drawing the same rows."""

import numpy as np
import pytest

from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks.learning import PooledWindow
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s2_news.toml")
SYMBOLS = CFG.stock_feed.symbols
SCALES = {s: float(i + 1) for i, s in enumerate(SYMBOLS)}


def x(value, n=12):
    return np.full(n, float(value))


def filled(minutes, start_k=1000, symbols=SYMBOLS):
    """A window with one sample per symbol per minute, x and label both 1000 times the minute plus the slot."""
    w = PooledWindow(SYMBOLS, 240, 12)
    for k in range(start_k, start_k + minutes):
        for i, s in enumerate(symbols):
            w.add(k, s, x(1000 * k + i), 1000.0 * k + i)
    return w


def test_the_window_holds_240_session_minutes_of_every_symbol():
    w = PooledWindow.from_config(CFG)
    assert (w.window_minutes, w.capacity, w.x.shape) == (240, 1920, (1920, 12))
    assert len(SYMBOLS) == 8


def test_the_symbol_column_follows_the_samples():
    w = PooledWindow(SYMBOLS, 240, 12)
    w.add(5, "NVDA", x(1), 1.0)
    w.add(5, "AAPL", x(2), 2.0)
    w.add(6, "QQQ", x(3), 3.0)
    assert w.column() == ("NVDA", "AAPL", "QQQ")
    assert w.symbol[:3].tolist() == [2, 0, 7] and w.minute[:3].tolist() == [5, 5, 6]


def test_samples_older_than_240_session_minutes_leave():
    w = PooledWindow(SYMBOLS, 240, 12)
    w.add(1, "AAPL", x(1), 1.0)
    w.add(240, "MSFT", x(2), 2.0)
    assert w.column() == ("AAPL", "MSFT")
    w.add(241, "QQQ", x(3), 3.0)                    # minute 1 is now 240 minutes old
    assert w.column() == ("MSFT", "QQQ")
    w.evict(480)
    assert w.column() == ("QQQ",)
    full = filled(300)
    assert full.count == 1920 and set(full.minute[:full.count].tolist()) == set(range(1060, 1300))


def test_a_ring_overflow_drops_the_oldest():
    w = PooledWindow(("AAPL", "QQQ"), 2, 1)
    for i in range(6):
        w.add(10, "AAPL", np.array([float(i)]), float(i))
    assert w.count == 4 and sorted(w.label.tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_a_batch_is_the_newest_n_plus_draws_from_the_whole_window():
    w = filled(10)
    newest = 8
    X, y = w.batch(np.random.default_rng(3), newest, 16, SCALES)
    assert X.shape == (16, 12) and y.shape == (16,)
    # The eight samples that matured last, in the order they matured.
    assert X[:8, 0].tolist() == [1000.0 * 1009 + i for i in range(8)]
    draws = np.random.default_rng(3).integers(0, w.count, 8)
    order = sorted(range(w.count), key=lambda i: (w.minute[i], w.symbol[i]))
    assert X[8:, 0].tolist() == [w.x[order[i], 0] for i in draws]


def test_each_label_is_divided_by_its_own_symbols_scale():
    w = filled(3)
    X, y = w.batch(np.random.default_rng(0), 3, 40, SCALES)
    for row, target in zip(X, y):
        slot = int(row[0]) % 1000
        label = row[0]
        assert target == label / SCALES[SYMBOLS[slot]]
    assert len({int(r[0]) % 1000 for r in X}) > 4


def test_newest_alone_and_draws_alone():
    w = filled(2)
    X, _ = w.batch(np.random.default_rng(1), 16, 16, SCALES)
    assert X[:, 0].tolist() == [1000.0 * k + i for k in (1000, 1001) for i in range(8)]
    X, _ = w.batch(np.random.default_rng(1), 0, 5, SCALES)
    assert X.shape == (5, 12)
    with pytest.raises(ValueError, match="newest"):
        w.batch(np.random.default_rng(1), 17, 20, SCALES)


def test_bad_samples_are_refused():
    w = PooledWindow(SYMBOLS, 240, 12)
    with pytest.raises(KeyError):
        w.add(1, "BTC", x(0), 0.0)
    with pytest.raises(ValueError, match="shape"):
        w.add(1, "AAPL", np.zeros(10), 0.0)
    w.add(5, "AAPL", x(0), 0.0)
    with pytest.raises(ValueError, match="older"):
        w.add(4, "AAPL", x(0), 0.0)


def test_s1_and_s2_windows_draw_the_same_rows_and_differ_only_in_the_news_columns():
    s1, s2 = PooledWindow(SYMBOLS, 240, 12), PooledWindow(SYMBOLS, 240, 12)
    rng = np.random.default_rng(9)
    for k in range(100, 160):
        for s in SYMBOLS:
            z, label = rng.normal(size=10), float(rng.normal(0, 30))
            s1.add(k, s, np.concatenate([z, [0.0, 0.0]]), label)
            s2.add(k, s, np.concatenate([z, rng.uniform(-1, 1, 2)]), label)
    X1, y1 = s1.batch(np.random.default_rng(4), 8, 16, SCALES)
    X2, y2 = s2.batch(np.random.default_rng(4), 8, 16, SCALES)
    assert np.array_equal(X1[:, :10], X2[:, :10]) and np.array_equal(y1, y2)
    assert not X1[:, 10:].any() and X2[:, 10:].any()
