"""Price features on synthetic series: identical vectors from a full series and the same series cut at each minute
across nights, a weekend, a holiday, an early close and gaps, a hand check of every feature, the overnight gap and its
missing prices, and the coverage rules."""

import math
from datetime import date

import numpy as np
import pytest

from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.features import (FEATURE_NAMES, N_PRICE_FEATURES, FeatureBuilder, gap_ratio,
                                             make_standardizer)
from btc_survivor.v2.stocks.series import StockSeries
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml")
D1 = sessions.session_on(date(2026, 10, 13))
D2 = sessions.session_on(date(2026, 10, 14))
WED = sessions.session_on(date(2026, 11, 25))
EARLY = sessions.session_on(date(2026, 11, 27))        # 13:00 close, the day after Thanksgiving
SYMBOLS = ("AAPL", "QQQ")
DRIFT = {"AAPL": 1e-4, "MSFT": -4e-5, "QQQ": 5e-5}
JUMP = {"AAPL": 0.02, "MSFT": 0.0, "QQQ": -0.01}


def builder(symbols=SYMBOLS):
    return FeatureBuilder(CFG.features, symbols, "QQQ")


def price(symbol, k, first_k, jump_from):
    """A drift, a zigzag of 3 bps and an overnight jump at jump_from."""
    lp = math.log(100.0) + DRIFT[symbol] * (k - first_k) + 3e-4 * (k % 2) + (JUMP[symbol] if k >= jump_from else 0.0)
    return math.exp(lp)


def volume(k):
    return 10.0 + k % 7


def hand_series(first, last, symbols=SYMBOLS, invalid=None):
    """Every symbol over sessions first to last, healthy unless (symbol, k) is in invalid, with buy and sell volume
    at 0.6 and 0.3 of the volume, in series long enough to hold them all. Returns the series and every price."""
    invalid = invalid or set()
    series, prices = {}, {}
    for symbol in symbols:
        s = StockSeries(max(420, last.last_k - first.first_k + 1))
        for k in range(first.first_k, last.last_k + 1):
            p, v = price(symbol, k, first.first_k, last.first_k), volume(k)
            prices[symbol, k] = p
            s.append(k, p, v, 0.6 * v, 0.3 * v, (symbol, k) not in invalid)
        series[symbol] = s
    return series, prices


def expected(prices, symbol, k, session, previous_last):
    """Every feature of one symbol at k on a fully healthy hand series, from the definitions in section 6.1."""
    p = {j: prices[symbol, j] for j in range(k - 390, k + 1)}
    first = session.first_k
    minute = k - first + 1
    gap = 1e4 * math.log(prices[symbol, first] / prices[symbol, previous_last])
    f1 = 1e4 * math.log(p[k] / p[k - 30]) - (gap if minute <= 30 else 0.0)
    f2 = 1e4 * math.log(p[k] / p[k - 120]) - (gap if minute <= 120 else 0.0)
    f3 = 1e4 * math.log(p[k] / p[k - 390])
    returns = [1e4 * math.log(p[j] / p[j - 1]) for j in range(k - 59, k + 1) if j != first]
    f4 = float(np.std(returns))
    recent = sum(volume(j) for j in range(k - 29, k + 1))
    baseline = sum(volume(j) for j in range(k - 389, k + 1)) / 390 * 30
    f5 = math.log(recent / baseline)
    return [f1, f2, f3, f4, f5, 1 / 3, minute / 390, (session.minutes - minute) / 390, gap]


def test_the_ten_features_are_named_in_order():
    assert N_PRICE_FEATURES == len(FEATURE_NAMES) == 10
    assert FEATURE_NAMES[2] == "return_session" and FEATURE_NAMES[8] == "overnight_gap"


@pytest.mark.parametrize("minute", [1, 10, 30, 31, 120, 121, 200, 390])
def test_hand_check_of_every_feature(minute):
    series, prices = hand_series(D1, D2)
    k = D2.first_k + minute - 1
    out = builder().compute(series, k)
    aapl = expected(prices, "AAPL", k, D2, D1.last_k)
    qqq = expected(prices, "QQQ", k, D2, D1.last_k)
    assert out["AAPL"].tolist() == pytest.approx(aapl + [aapl[2] - qqq[2]], rel=1e-9, abs=1e-9)
    assert out["QQQ"].tolist() == pytest.approx(qqq + [0.0], rel=1e-9, abs=1e-9)
    assert out["AAPL"][5] == pytest.approx(1 / 3)


def test_returns_leave_out_the_overnight_move_only_while_the_window_spans_the_open():
    series, prices = hand_series(D1, D2)
    gap = 1e4 * math.log(prices["AAPL", D2.first_k] / prices["AAPL", D1.last_k])
    assert gap > 150                                       # the 2 percent jump at the open
    k = D2.first_k + 29                                    # 10:00, the 30 minute window starts at the close
    f = builder().compute(series, k)["AAPL"]
    assert f[0] == pytest.approx(1e4 * math.log(prices["AAPL", k] / prices["AAPL", k - 30]) - gap)
    assert f[2] == pytest.approx(1e4 * math.log(prices["AAPL", k] / prices["AAPL", k - 390]))
    assert abs(f[0]) < 50 and f[2] > 150                   # the session return keeps the move
    k = D2.first_k + 30
    f = builder().compute(series, k)["AAPL"]
    assert f[0] == pytest.approx(1e4 * math.log(prices["AAPL", k] / prices["AAPL", k - 30]))


def test_volatility_leaves_out_the_return_across_the_open():
    series, prices = hand_series(D1, D2)
    k = D2.first_k + 4
    returns = [1e4 * math.log(prices["AAPL", j] / prices["AAPL", j - 1]) for j in range(k - 59, k + 1)]
    f = builder().compute(series, k)["AAPL"]
    assert f[3] == pytest.approx(float(np.std([r for j, r in zip(range(k - 59, k + 1), returns) if j != D2.first_k])))
    assert f[3] < float(np.std(returns))


def test_times_since_the_open_and_to_the_close_on_an_early_close():
    series, prices = hand_series(WED, EARLY)
    assert EARLY.minutes == 210
    for minute in (1, 50, 180, 210):
        k = EARLY.first_k + minute - 1
        f = builder().compute(series, k)["AAPL"]
        assert (f[6], f[7]) == (minute / 390, (210 - minute) / 390)
        assert f.tolist()[:9] == pytest.approx(expected(prices, "AAPL", k, EARLY, WED.last_k), rel=1e-9, abs=1e-9)


def test_the_gap_takes_the_first_healthy_price_today_and_the_last_healthy_one_before():
    invalid = {("AAPL", D1.last_k), ("AAPL", D1.last_k - 1), ("AAPL", D2.first_k), ("AAPL", D2.first_k + 1)}
    series, prices = hand_series(D1, D2, invalid=invalid)
    k = D2.first_k + 100
    w = series["AAPL"].window(k, 391)
    ratio = prices["AAPL", D2.first_k + 2] / prices["AAPL", D1.last_k - 2]
    assert gap_ratio(w, D2.first_k) == pytest.approx(ratio, rel=1e-14)
    assert builder().compute(series, k)["AAPL"][8] == pytest.approx(1e4 * math.log(ratio))
    # Before the first healthy minute of the day it does not exist yet, and nothing later is read.
    assert gap_ratio(series["AAPL"].window(D2.first_k + 1, 391), D2.first_k) is None


def test_feature_9_is_invalid_without_healthy_reference_prices():
    invalid = {("AAPL", k) for k in range(D1.first_k, D1.last_k + 1)}
    series, _ = hand_series(D1, D2, invalid=invalid)
    k = D2.first_k + 200
    assert gap_ratio(series["AAPL"].window(k, 391), D2.first_k) is None
    out = builder().compute(series, k)
    assert out["AAPL"] is None and out["QQQ"] is not None
    # The very first session of the table has no session before it.
    s = StockSeries()
    s.append(1, 100.0, 1.0, 0.0, 0.0, True)
    assert gap_ratio(s.window(1, 391), 1) is None


def test_returns_need_healthy_prices_at_both_ends():
    k = D2.first_k + 200
    for back in (30, 120, 390, 0):
        series, _ = hand_series(D1, D2, invalid={("AAPL", k - back)})
        out = builder().compute(series, k)
        assert out["AAPL"] is None and out["QQQ"] is not None


def test_volatility_needs_80_percent_of_its_returns():
    k = D2.first_k + 200
    # A block of b unhealthy minutes inside the window removes b plus one of the 60 returns.
    ok, _ = hand_series(D1, D2, invalid={("AAPL", j) for j in range(k - 50, k - 39)})
    short, _ = hand_series(D1, D2, invalid={("AAPL", j) for j in range(k - 50, k - 38)})
    assert builder().compute(ok, k)["AAPL"] is not None
    assert builder().compute(short, k)["AAPL"] is None


def test_volume_ratio_and_imbalance_need_80_percent_of_their_minutes():
    k = D2.first_k + 200
    ok, _ = hand_series(D1, D2, invalid={("AAPL", j) for j in range(k - 20, k - 14)})
    short, _ = hand_series(D1, D2, invalid={("AAPL", j) for j in range(k - 20, k - 13)})
    f = builder().compute(ok, k)["AAPL"]
    assert f is not None and f[5] == pytest.approx(1 / 3)
    healthy = [j for j in range(k - 29, k + 1) if not k - 20 <= j < k - 14]
    assert f[4] == pytest.approx(math.log(np.mean([volume(j) for j in healthy]) * 30
                                          / (np.mean([volume(j) for j in range(k - 389, k + 1)
                                                      if not k - 20 <= j < k - 14]) * 30)))
    assert builder().compute(short, k)["AAPL"] is None
    # The baseline needs 312 of its 390 minutes.
    ok, _ = hand_series(D1, D2, invalid={("AAPL", j) for j in range(k - 380, k - 302)})
    short, _ = hand_series(D1, D2, invalid={("AAPL", j) for j in range(k - 380, k - 301)})
    assert builder().compute(ok, k)["AAPL"] is not None
    assert builder().compute(short, k)["AAPL"] is None


def test_volume_ratio_is_clipped_and_imbalance_is_zero_without_signed_volume():
    series = {}
    for symbol in SYMBOLS:
        s = StockSeries(D2.last_k - D1.first_k + 1)
        for k in range(D1.first_k, D2.last_k + 1):
            v = 0.0 if symbol == "AAPL" and k > D2.first_k + 170 else 1.0
            s.append(k, price(symbol, k, D1.first_k, D2.first_k), v, 0.0, 0.0, True)
        series[symbol] = s
    f = builder().compute(series, D2.first_k + 200)["AAPL"]
    assert f[4] == -3.0 and f[5] == 0.0
    # The baseline holds the last 30 minutes too, so the ratio can never pass ln(13) upward.
    assert CFG.features.volume_ratio_clip > math.log(390 / 30)


def test_feature_10_needs_the_index_healthy_at_both_ends():
    k = D2.first_k + 200
    series, _ = hand_series(D1, D2, invalid={("QQQ", k - 390)})
    out = builder().compute(series, k)
    assert out["AAPL"] is None and out["QQQ"] is None
    # An index vector invalid for another reason still gives feature 10.
    series, prices = hand_series(D1, D2, invalid={("QQQ", j) for j in range(k - 20, k - 13)})
    out = builder().compute(series, k)
    assert out["QQQ"] is None
    qqq = 1e4 * math.log(prices["QQQ", k] / prices["QQQ", k - 390])
    assert out["AAPL"][9] == pytest.approx(out["AAPL"][2] - qqq)


def test_vectors_first_become_valid_at_0931_of_the_second_session():
    series, _ = hand_series(D1, D2)
    for k in range(D1.first_k, D1.last_k + 1, 13):
        assert builder().compute(series, k) == {"AAPL": None, "QQQ": None}
    assert all(v is not None for v in builder().compute(series, D2.first_k).values())


def synthetic_candles(seed, symbols, first, last):
    """Seeded candles for every symbol from session first to last, with random unhealthy stretches, minutes with no
    reference price, a symbol missing for a whole session and skipped minutes, as a restart leaves them."""
    rng = np.random.default_rng(seed)
    days = [s for s in sessions.all_sessions() if first.first_k <= s.first_k <= last.first_k]
    out = []
    for symbol in symbols:
        p, healthy = float(rng.uniform(50, 500)), True
        missing = days[int(rng.integers(1, len(days)))]
        for day in days:
            p *= math.exp(rng.normal(0, 0.01))                 # the overnight move
            for k in range(day.first_k, day.last_k + 1):
                p *= math.exp(rng.normal(0, 0.0008))
                healthy = rng.random() > (0.004 if healthy else 0.25)
                if day is missing or rng.random() < 0.002:
                    continue                                   # never appended
                v = float(rng.integers(0, 5000))
                buy = float(rng.uniform(0, v))
                ref = None if rng.random() < 0.003 else round(p, 2)
                out.append(StockCandle(symbol, sessions.boundary_of_k(k), k, ref, ref, ref, ref, ref, v, buy,
                                       v - buy, 1, 1, 0, healthy and ref is not None))
    return sorted(out, key=lambda c: (c.k, symbols.index(c.symbol)))


def test_a_full_series_and_the_same_series_cut_at_each_minute_give_identical_vectors():
    symbols = ("AAPL", "MSFT", "QQQ")
    first = sessions.session_on(date(2026, 11, 20))            # Friday, then a weekend, Thanksgiving, an early close
    last = sessions.session_on(date(2026, 12, 1))
    candles = synthetic_candles(7, symbols, first, last)
    full = {s: StockSeries(capacity=last.last_k - first.first_k + 1) for s in symbols}
    for c in candles:
        full[c.symbol].append_candle(c)
    live = {s: StockSeries() for s in symbols}
    b = builder(symbols)
    by_k: dict[int, list[StockCandle]] = {}
    for c in candles:
        by_k.setdefault(c.k, []).append(c)
    valid = invalid = 0
    for k in range(first.first_k, last.last_k + 1):
        for c in by_k.get(k, []):
            live[c.symbol].append_candle(c)
        now, later = b.compute(live, k), b.compute(full, k)
        for symbol in symbols:
            if now[symbol] is None:
                assert later[symbol] is None
                invalid += 1
            else:
                assert np.array_equal(now[symbol], later[symbol])
                valid += 1
    assert valid > 2000 and invalid > 2000


def test_the_standardizer_counts_healthy_session_minutes():
    z = make_standardizer(CFG.features)
    assert z.min_samples == 30 and z.clip == 5.0
    assert z.stats.alpha == pytest.approx(1 - 0.5 ** (1 / 10080))
    assert z.stats.mean.shape == (10,)


def test_from_config_and_a_missing_index_symbol():
    b = FeatureBuilder.from_config(CFG)
    assert b.symbols == CFG.stock_feed.symbols and b.index_symbol == "QQQ" and b.length == 391
    with pytest.raises(ValueError, match="index symbol"):
        FeatureBuilder(CFG.features, ("AAPL",), "QQQ")
