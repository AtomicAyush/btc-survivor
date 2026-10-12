"""S1 and S2 born identical by PLAN_V2 section 10.3: equal weights and generator states after birth, bitwise identical
predictions while every batch has zero news inputs, S1's news weights staying zero, and divergence exactly at S2's first
training step whose batch holds a nonzero news input. Both engines step the same hand built MarketMinutes, and S2 alone
reads one scored headline."""

import re
import sqlite3
from datetime import date

import numpy as np
import pytest

from btc_survivor.clock import iso
from btc_survivor.model import MLP
from btc_survivor.seeds import bot_rng
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import parse_stock_config
from btc_survivor.v2.stocks.engine import StockEngine
from btc_survivor.v2.stocks.labels import fits, is_evaluation_minute
from btc_survivor.v2.stocks.market import MarketMinute, Sample
from btc_survivor.v2.stocks.news_inputs import Impact, ScoreLine
from btc_survivor.v2.stocks.population import StockPopulation, birth_differences
from btc_survivor.v2.stocks.store import StockStore
from helpers import ROOT

DAY = sessions.session_on(date(2026, 10, 20))
N = 8
PRICES = np.array([250.0, 500.0, 180.0, 220.0, 250.0, 700.0, 400.0, 600.0])
SD = 40.0
NEWS_MINUTE = 100  # the headline arrives 5 seconds after this boundary
FIRST_NEWS_BATCH = NEWS_MINUTE + 1 + 30  # the first record pushed after it matures 30 minutes later
HEADLINE_MS = DAY.boundary(NEWS_MINUTE) + 5_000


def config(folder, name):
    """A committed run config with its data folder moved into folder."""
    text = (ROOT / "configs" / "v2" / f"{name}.toml").read_text(encoding="utf-8")
    text, found = re.subn(r"(?m)^data_dir = [^#\n]*", f'data_dir = "data/{name}" ', text)
    assert found == 1
    return parse_stock_config(text, folder)


def pair(folder):
    return config(folder, "s1_prices"), config(folder, "s2_news")


def news_part(params):
    """The weights of the two news inputs inside a flat 225 parameter vector, the last two columns of W1."""
    return params[:16 * 12].reshape(16, 12)[:, 10:]


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def query(path, sql):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_s1_and_s2_are_born_identical_in_every_slot(tmp_path):
    c1, c2 = pair(tmp_path)
    p1, p2 = StockPopulation.fresh(c1, DAY.open_ms), StockPopulation.fresh(c2, DAY.open_ms)
    assert birth_differences(p1, p2) == []
    assert [b.bot_id for b in p1.learners] == list(range(1, 25))
    assert [t.bot_id for t in p1.twins] == list(range(25, 49))
    assert [b.symbol for b in p1.learners] == [c1.stock_feed.symbols[slot % 8] for slot in range(24)]
    for a, b in zip(p1.learners, p2.learners):
        rng = bot_rng(c1.run.master_seed, a.bot_id)
        narrow = MLP.random(10, 16, rng, 0.1)
        rng.integers(0, 180)
        assert a.model.n_in == 12 and a.model.size == 225
        assert np.array_equal(a.model.W1[:, :10], narrow.W1) and not news_part(a.model.params).any()
        assert a.rng.bit_generator.state == b.rng.bit_generator.state == rng.bit_generator.state
    for a, b in zip(p1.twins, p2.twins):
        assert a.rng.bit_generator.state == b.rng.bit_generator.state
    p2.learners[3].model.W1[0, 0] += 1e-12
    p2.twins[5].rng.random()
    assert birth_differences(p1, p2) == ["bot 4 has different weights", "bot 30 has a different generator state"]


def synthetic_minutes(seed):
    """A full session of minutes in which every symbol is healthy with a valid vector, news health holds, every label
    scale is SD, and a record is pushed for every symbol whenever it fits and matures 30 minutes later."""
    rng = np.random.default_rng(seed)
    pushed = {}
    ones = np.ones(N, dtype=bool)
    for minute in range(1, DAY.minutes + 1):
        boundary = DAY.boundary(minute)
        k = DAY.k(boundary)
        z = np.clip(rng.normal(size=(N, 10)), -5.0, 5.0)
        old = pushed.pop(k - 30, None)
        matured = () if old is None else tuple(Sample(i, k - 30, old[i], float(rng.normal(0.0, SD))) for i in range(N))
        push = tuple(range(N)) if fits(k, 30) else ()
        if push:
            pushed[k] = z
        yield MarketMinute(
            boundary_ms=boundary, k=k, session=DAY, candles=(), healthy=ones, ref=PRICES, last_ref=PRICES, z=z,
            valid=ones, label_sd=np.full(N, SD), shorts_ok=ones, news_ok=True, news_reason=None, channels_down=(),
            matured=matured, pushed=push, labels=(), linear=np.full(N, np.nan),
            evaluation=is_evaluation_minute(k, 30), flatten=(None,) * N, gaps=(1,) * N, spread_checks=(),
            healthy_minutes=minute, run_minutes=minute, run_minutes_total=minute)


def newest(window, n):
    """The symbol, maturity minute and news inputs of the newest n samples in a pooled window."""
    out = []
    for i in range(window.count - n, window.count):
        slot = (window.start + i) % window.capacity
        out.append((window.symbols[window.symbol[slot]], int(window.minute[slot]), window.x[slot, 10:].copy()))
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Both engines over one synthetic session, with what every learner held after each minute."""
    c1, c2 = pair(tmp_path_factory.mktemp("zero_init"))
    engines = [StockEngine.fresh(c, StockStore.open(c.db_path), DAY.open_ms - 60_000) for c in (c1, c2)]
    line = ScoreLine(1, 11, "finnhub", "media", "llm", "haiku_v1", "scored", HEADLINE_MS - 3_000, ("AAPL",),
                     (Impact("AAPL", 0.6, True),), recv_ms=HEADLINE_MS)
    log, batches = [], {}
    for minute, m in enumerate(synthetic_minutes(3), start=1):
        for engine in engines:
            engine.on_minute(m)
        if minute == NEWS_MINUTE:
            for engine in engines:
                engine.on_news(line)
        if minute in (FIRST_NEWS_BATCH - 1, FIRST_NEWS_BATCH):
            batches[minute] = newest(engines[1].window, N)
        log.append({
            "preds": [[b.preds.copy() for b in e.population.learners] for e in engines],
            "params": [[b.model.params.copy() for b in e.population.learners] for e in engines],
            "rng": [[b.rng.bit_generator.state for b in e.population.learners] for e in engines],
            "aapl": [e.inputs.current["AAPL"].copy() for e in engines],
            "linear_news": engines[1].linear_news.xy[11:].copy(),   # index 0 is the intercept, 11 and 12 the news
        })
    for engine in engines:
        engine.persist(DAY.close_ms + 30_000)
        engine.store.close()
    return engines, log, batches, (c1.db_path, c2.db_path)


def test_predictions_are_bitwise_identical_until_s2_first_trains_on_a_nonzero_news_input(runs):
    _, log, batches, _ = runs
    for minute, entry in enumerate(log, start=1):
        s1, s2 = entry["preds"]
        if minute < FIRST_NEWS_BATCH:
            assert all(same_bits(a, b) for a, b in zip(s1, s2)), minute
            assert all(same_bits(a, b) for a, b in zip(*entry["params"])), minute
        else:
            assert not any(same_bits(a, b) for a, b in zip(s1, s2)), minute
    # S2 predicted with a nonzero AAPL input for 30 minutes before any batch held one, and still matched S1.
    assert not log[NEWS_MINUTE - 1]["aapl"][1].any() and log[NEWS_MINUTE]["aapl"][1][0] > 0.25
    assert all(entry["aapl"][1][0] > 0 for entry in log[NEWS_MINUTE:FIRST_NEWS_BATCH - 1])
    assert all(not entry["aapl"][0].any() for entry in log)
    before, first = batches[FIRST_NEWS_BATCH - 1], batches[FIRST_NEWS_BATCH]
    assert not any(news.any() for _, _, news in before)
    assert {symbol for symbol, _, news in first if news.any()} == {"AAPL", "QQQ"}


def test_news_weights_stay_zero_in_s1_and_move_in_s2_only_from_that_step(runs):
    engines, log, _, _ = runs
    for minute, entry in enumerate(log, start=1):
        s1, s2 = entry["params"]
        assert not any(news_part(p).any() for p in s1), minute
        moved = [news_part(p).any() for p in s2]
        assert all(moved) if minute >= FIRST_NEWS_BATCH else not any(moved), minute
    s1, s2 = engines
    assert all(not news_part(b.model.params).any() for b in s1.population.learners)
    assert all(b.trained_minutes == DAY.minutes - 30 for e in engines for b in e.population.learners)


def test_linear_news_learns_only_from_the_news_inputs_stored_at_each_push(runs):
    engines, log, _, _ = runs
    assert engines[0].linear_news is None and engines[1].linear_news is not None
    for minute, entry in enumerate(log, start=1):
        moved = entry["linear_news"].any()
        assert moved if minute >= FIRST_NEWS_BATCH else not moved, minute


def test_generator_states_match_throughout_and_s1_never_stores_a_news_input(runs):
    engines, log, _, _ = runs
    assert all(a == b for entry in log for a, b in zip(*entry["rng"]))
    assert engines[0].window.count == engines[1].window.count == 240 * N
    assert not engines[0].window.x[:, 10:].any() and engines[1].window.x[:, 10:].any()
    assert np.array_equal(engines[0].window.x[:, :10], engines[1].window.x[:, :10])


def test_only_s2_keeps_the_headline_and_its_score_forecasts(runs):
    _, _, _, (s1, s2) = runs
    assert query(s1, "SELECT count(*) FROM news_seen") == [(0,)]
    assert query(s2, "SELECT score_seq, item_seq, recv_at, scorer, primary_use, fallback, tickers FROM news_seen") == \
        [(1, 11, iso(HEADLINE_MS), "haiku_v1", 1, 0, "AAPL")]
    assert query(s1, "SELECT DISTINCT predictor FROM stock_forecasts") == [("population",)]
    rows = query(s2, "SELECT symbol, min(prediction), max(items), max(inputs_max_recv_at) FROM stock_forecasts "
                     "WHERE predictor = 'news' AND prediction != 0 GROUP BY symbol ORDER BY symbol")
    assert [(symbol, items, newest) for symbol, _, items, newest in rows] == \
        [("AAPL", 1, iso(HEADLINE_MS)), ("QQQ", 1, iso(HEADLINE_MS))]
    assert all(lowest > 0 for _, lowest, _, _ in rows)
