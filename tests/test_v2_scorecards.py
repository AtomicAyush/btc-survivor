"""The chooser scorecard of PLAN_V2 section 11.8 and the day scorecard of section 11.9 on synthetic run databases.

Each fixture copies the committed S1 and S2 configs into a test folder, so their data lands there, and writes the runs
through the stock store and the row builders, with launch meta as the stock process records it. Every price, label and
forecast is synthetic. The chooser runs launch on Monday 19 October 2026 with seven sessions of labels and forecasts
from session 2, whose population forecasts are far below every cost except at planted minutes. The day runs launch on
Monday 16 November 2026, so the early close of 27 November is session 9, with candles from session 2 in the news run
alone and planted gaps, as the shared market layer writes the same candles to both runs. A reference written here from
the plan's words checks the module's figures, and traps check that no label, forecast or input after a minute moves
what is chosen at it. The stock process and the news process never load the scorecards.
"""

import ast
import contextlib
import dataclasses
import math
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from btc_survivor.cli.stats import block_test
from btc_survivor.clock import MS_PER_MINUTE, iso, parse_iso
from btc_survivor.types import FLAT, LONG, SHORT
from btc_survivor.v2 import scorecards, sessions, stats
from btc_survivor.v2.config import HAIKU_MODEL, SONNET_MODEL, load_run_pair
from btc_survivor.v2.scorecards import Scales, Source
from btc_survivor.v2.stocks import costs, records, store
from btc_survivor.v2.stocks.broker import StockBroker
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.labels import GAP, MATURED, PENDING, LabelRow
from btc_survivor.v2.stocks.store import StockStore
from helpers import ROOT
import v2_runkit as kit

SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
# MSFT trades at AAPL's price with AAPL's labels, so the two tie whenever their forecasts do.
PRICES = {"AAPL": 250.0, "MSFT": 250.0, "NVDA": 180.0, "AMZN": 220.0, "GOOGL": 250.0, "META": 700.0, "TSLA": 400.0,
          "QQQ": 600.0}
SPREAD = {"AAPL": 10.0, "MSFT": 10.0, "NVDA": 25.0, "AMZN": 15.0, "GOOGL": 15.0, "META": 20.0, "TSLA": 40.0,
          "QQQ": 8.0}
NEWS_META = {"config_hash": "a" * 64, "prompt_sha256": "b" * 64, "schema_sha256": "c" * 64,
             "haiku_model": HAIKU_MODEL, "sonnet_model": SONNET_MODEL}
COMMIT = "0123456789abcdef0123456789abcdef01234567"
LAGS, UPPER = 12, 1.645
SRC = ROOT / "src"
V2 = SRC / "btc_survivor" / "v2"


def run_days(launch, count):
    table = sessions.all_sessions()
    first = table.index(sessions.next_session(parse_iso(launch)))
    return table[first:first + count]


def write_run(config, launch, statements):
    """One run database with the meta the stock process records at launch and the given rows."""
    st = StockStore.open(config.db_path)
    for key, value in records.launch_meta(config, parse_iso(launch), NEWS_META, 1, COMMIT).items():
        st.queue_meta(key, value)
    for sql, params in statements:
        st.queue(sql, params)
    st.flush()
    st.close()


def label_rows(ms, symbol, entry, value):
    """The pending row and its settle, matured at value or gap for None, as the market layer writes them."""
    pending = LabelRow(ms, symbol, entry, None, None, PENDING)
    final = LabelRow(ms, symbol, entry, None, None, GAP) if value is None else \
        LabelRow(ms, symbol, entry, entry * math.exp(value / 1e4), value, MATURED)
    return [records.label_statement(pending), records.label_statement(final)]


def forecast(ms, symbol, predictor, value, received=None):
    return records.INSERT_FORECAST, records.forecast_row(ms, symbol, predictor, value, None, received)


def execute(path, sql, params=()):
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(sql, params)
    conn.close()


@contextlib.contextmanager
def opened(config, name=None):
    conn = store.open_reader(config.db_path)
    try:
        yield Source(name or config.run.name, conn, config)
    finally:
        conn.close()


def qty(config, side, price):
    return StockBroker.from_config(config).open_qty(side, price)


def reference_cost(config, symbol, side, price, minutes):
    """The round trip of the plan's words, costs.round_trip for the side, the broker's quantity and the minutes."""
    side = SHORT if side == SHORT else LONG
    return costs.round_trip(config.stock_costs, symbol, side, qty(config, side, price), price, minutes).total_bps


def moved(folder, tmp_path):
    """A copy of a built folder, whose configs still find their data inside it."""
    shutil.copytree(folder, tmp_path / "copy")
    return load_run_pair(tmp_path / "copy" / "configs" / "v2" / "s1_prices.toml",
                         tmp_path / "copy" / "configs" / "v2" / "s2_news.toml")


# The chooser runs


C_LAUNCH = "2026-10-19T13:25:00.000Z"
C_DAYS = run_days(C_LAUNCH, 7)
C_START, C_END = iso(C_DAYS[1].open_ms), iso(C_DAYS[6].close_ms)


def c_at(n, minute):
    return C_DAYS[n - 1].boundary(minute)


# Population forecasts planted far above cost, by session and minute. Every other forecast is far below every cost.
PLANTS = {
    (6, 30): {"AAPL": 40.0},  # AAPL has exactly 60 labels that exited by then
    (6, 90): {"AAPL": 40.0, "TSLA": -90.0},  # AAPL leads over its scale, TSLA in bps
    (6, 150): {"AAPL": 40.0, "MSFT": 40.0},  # a tie, AAPL first in config order
    (6, 180): {"AAPL": 40.0, "MSFT": 40.001},
    (6, 210): {"AAPL": 40.0},  # AAPL's label here ends as gap
    (7, 60): {"TSLA": -120.0},  # a short
}
GAP_PICK = (c_at(6, 210), "AAPL")
PLANTED = {c_at(n, m): picks for (n, m), picks in PLANTS.items()}


@dataclasses.dataclass(frozen=True)
class ChooserRuns:
    folder: Path
    s1: object
    s2: object
    labels: dict
    population: dict


def chooser_rows(rng):
    """Labels at every evaluation minute of sessions 1 to 7, MSFT's equal to AAPL's, and the population forecasts of
    sessions 2 to 7, with linear_prices and linear_news far below cost."""
    labels, population, others = {}, {}, []
    for n, day in enumerate(C_DAYS, 1):
        for minute in day.evaluation_minutes():
            ms = day.boundary(minute)
            z = rng.normal(size=len(SYMBOLS))
            for j, symbol in enumerate(SYMBOLS):
                value = SPREAD[symbol] * float(z[0 if symbol == "MSFT" else j])
                labels[ms, symbol] = None if (ms, symbol) == GAP_PICK else value
                if n >= 2:
                    population[ms, symbol] = PLANTED.get(ms, {}).get(symbol, 0.3 * float(rng.normal()))
                    others.append((ms, symbol, 0.3 * float(rng.normal()), 0.3 * float(rng.normal())))
    return labels, population, others


@pytest.fixture(scope="module")
def chooser(tmp_path_factory):
    folder = tmp_path_factory.mktemp("chooser")
    s1_path, s2_path, _ = kit.configs(folder)
    s1, s2 = load_run_pair(s1_path, s2_path)
    labels, population, others = chooser_rows(np.random.default_rng(8))
    shared = [row for (ms, symbol), value in labels.items() for row in label_rows(ms, symbol, PRICES[symbol], value)]
    shared += [forecast(ms, symbol, "population", f) for (ms, symbol), f in population.items()]
    shared += [forecast(ms, symbol, "linear_prices", f) for ms, symbol, f, _ in others]
    write_run(s1, C_LAUNCH, shared)
    write_run(s2, C_LAUNCH, shared + [forecast(ms, symbol, "linear_news", f) for ms, symbol, _, f in others])
    return ChooserRuns(folder, s1, s2, labels, population)


def chooser_cards(config, start=C_START, end=C_END, name=None, lags=LAGS):
    with opened(config, name) as source:
        return {card.predictor: card for card in scorecards.chooser_cards(source, start, end, lags, UPPER)}


def picks_at(card, name=scorecards.CHOOSER):
    return {ms: (symbol, side) for ms, symbol, side in card.trail[name]}


def exits_by(ms):
    return sessions.boundary_of_k(sessions.k_at(ms) + 30)


def reference_scale(labels, symbol, ms):
    """The spread of the symbol's matured labels whose exit is at or before ms and within 20 full sessions of it."""
    k = sessions.k_at(ms)
    chosen = [v for (t, s), v in labels.items()
              if s == symbol and v is not None and k - 20 * 390 < sessions.k_at(t) + 30 <= k]
    return float(np.std(chosen)) if len(chosen) >= 60 and np.std(chosen) > 0 else None


def reference_chooser(runs, config):
    """Every strategy's per minute value from the plan's words alone, at the minutes whose label exited by the end."""
    out = {name: [] for name in ("chooser", "random", "all", "raw", "index")}
    minutes = sorted({ms for ms, _ in runs.population if exits_by(ms) <= parse_iso(C_END)})
    for ms in minutes:
        offers = []
        for symbol in SYMBOLS:
            f, label = runs.population[ms, symbol], runs.labels[ms, symbol]
            cost = reference_cost(config, symbol, LONG if f >= 0 else SHORT, PRICES[symbol], 30)
            offers.append((symbol, f, cost, reference_scale(runs.labels, symbol, ms), label))

        def earn(symbol, side, f_cost=None):
            label = runs.labels[ms, symbol]
            cost = reference_cost(config, symbol, side, PRICES[symbol], 30) if f_cost is None else f_cost
            return None if label is None else side * label - cost

        for name, edge in (("chooser", lambda o: None if o[3] is None else (abs(o[1]) - o[2]) / o[3]),
                           ("raw", lambda o: abs(o[1]) - o[2])):
            ranked = [(edge(o), -i, o) for i, o in enumerate(offers) if edge(o) is not None and edge(o) > 0]
            top = max(ranked, default=None)
            out[name].append((ms, None, 0.0) if top is None else
                             (ms, top[2][0], earn(top[2][0], 1 if top[2][1] > 0 else -1, top[2][2])))
        rng = np.random.default_rng(np.random.SeedSequence(
            [config.run.master_seed, scorecards.CHOOSER_PICK, scorecards._tag(config.run.name), ms]))
        symbol = SYMBOLS[int(rng.integers(len(SYMBOLS)))]
        side = 1 if rng.integers(2) else -1
        out["random"].append((ms, symbol, earn(symbol, side)))
        each = [earn(s, 1 if f > 0 else -1, c) for s, f, c, _, _ in offers]
        each = [e for e in each if e is not None]
        out["all"].append((ms, None, sum(each) / len(each) if each else None))
        out["index"].append((ms, "QQQ", earn("QQQ", 1)))
    return out


def total(values):
    return math.fsum(v for _, _, v in values if v is not None)


# The scale


def test_the_scale_counts_matured_labels_that_exited_by_t_within_20_full_sessions():
    k = 100_000
    inside = [(k - 1000 + 10 * i, float(i % 7)) for i in range(59)] + [(k, 3.5)]
    scales = Scales({"AAPL": [(k - 20 * 390, 1000.0), *inside, (k + 1, -1e6)]})
    # The label exiting exactly 20 full sessions before t is out, the one exiting at t is in, one after t never is.
    assert scales.count("AAPL", k) == 60
    assert scales.at("AAPL", k) == float(np.std([v for _, v in inside]))
    assert Scales({"AAPL": [(k - 20 * 390 + 1, 1000.0), *inside[1:]]}).count("AAPL", k) == 60
    # 59 labels are too few, and so is no spread at all.
    assert Scales({"AAPL": inside[1:]}).at("AAPL", k) is None
    assert Scales({"AAPL": [(i, 2.0) for i in range(80)]}).at("AAPL", 100) is None
    assert scales.at("MSFT", k) is None


def test_the_scale_read_from_the_run_is_the_plan_s_at_every_minute(chooser):
    with opened(chooser.s2) as source:
        labels = scorecards.read_labels(source.conn, C_END, 30)
    for ms in (c_at(5, 360), c_at(6, 30), c_at(6, 90), c_at(7, 300)):
        for symbol in SYMBOLS:
            found, expected = labels.scales.at(symbol, sessions.k_at(ms)), reference_scale(chooser.labels, symbol, ms)
            assert found == (None if expected is None else pytest.approx(expected, rel=1e-12)), (iso(ms), symbol)
    # Session 6 opens with exactly 60 labels exited per symbol, and AAPL has one fewer once its gap would count.
    assert labels.scales.count("AAPL", sessions.k_at(c_at(6, 30))) == 60
    assert labels.scales.at("AAPL", sessions.k_at(c_at(5, 360))) is None


# The rule


def test_the_chooser_takes_the_largest_edge_over_the_scale_and_the_raw_forecast_the_largest_in_bps(chooser):
    card = chooser_cards(chooser.s2)["population"]
    chosen = picks_at(card)
    planted = {ms: pick for ms, pick in chosen.items() if pick[0] is not None}
    assert planted == {c_at(6, 30): ("AAPL", LONG), c_at(6, 90): ("AAPL", LONG), c_at(6, 150): ("AAPL", LONG),
                       c_at(6, 180): ("MSFT", LONG), c_at(6, 210): ("AAPL", LONG), c_at(7, 60): ("TSLA", SHORT)}
    # Without the scale the larger TSLA forecast wins in bps.
    assert picks_at(card, scorecards.RAW)[c_at(6, 90)] == ("TSLA", SHORT)
    chooser_row = card.strategies[0]
    assert (chooser_row["strategy"], chooser_row["picks"], chooser_row["no result"]) == ("chooser", 6, 1)
    assert card.minutes == 72 and chooser_row["flat"] == 66 and chooser_row["minutes"] == 71
    assert card.picks_per_symbol == {**{s: 0 for s in SYMBOLS}, "AAPL": 4, "MSFT": 1, "TSLA": 1}
    # Before session 6 no symbol has 60 labels, so 48 minutes are flat for want of any scale.
    assert card.unscaled_minutes == 48 and card.skipped == 48 * 8
    expected = reference_chooser(chooser, chooser.s2)
    assert [(ms, s) for ms, s, _ in expected["chooser"]] == [(ms, s) for ms, s, _ in card.trail["chooser"]]
    assert chooser_row["total bps"] == pytest.approx(total(expected["chooser"]), rel=1e-12)


def test_a_flat_minute_and_a_gap_pick_are_counted_and_the_gap_earns_nothing(chooser):
    card = chooser_cards(chooser.s2)["population"]
    row = card.strategies[0]
    results = [v for ms, s, v in reference_chooser(chooser, chooser.s2)["chooser"] if s is not None]
    assert results.count(None) == 1 and row["no result"] == 1
    scored = [v for v in results if v is not None]
    assert row["mean per pick bps"] == pytest.approx(sum(scored) / 5, rel=1e-12)
    assert row["mean per minute bps"] == pytest.approx(sum(scored) / 71, rel=1e-12)
    right = [side * chooser.labels[ms, symbol] > 0 for ms, symbol, side in card.trail["chooser"]
             if symbol is not None and chooser.labels[ms, symbol] is not None]
    assert len(right) == 5 and row["right share"] == sum(right) / 5
    # The gap minute and the minutes before any symbol had a scale, when the chooser could not act, are left out of
    # every difference.
    paired = card.minutes - card.unscaled_minutes - 1
    assert card.unscaled_minutes > 0 and all(d["minutes"] <= paired for d in card.differences)
    assert next(d for d in card.differences if d["comparison"] == "flat")["minutes"] == paired


def test_the_short_pick_pays_its_borrow_and_trades_whole_shares(chooser):
    config = chooser.s2
    long_trip = scorecards.trip_bps(config, "TSLA", LONG, 400.0, 30)
    short_trip = scorecards.trip_bps(config, "TSLA", SHORT, 400.0, 30)
    short = costs.round_trip(config.stock_costs, "TSLA", SHORT, 1.0, 400.0, 30)
    assert qty(config, SHORT, 400.0) == 1.0 and qty(config, LONG, 400.0) == 1.25
    assert short_trip == short.total_bps and short.notional == 400.0
    assert short.borrow_bps == costs.holding_cost_bps(config.stock_costs, SHORT, 30) > 0
    assert long_trip == costs.round_trip(config.stock_costs, "TSLA", LONG, 1.25, 400.0, 30).total_bps
    assert costs.round_trip(config.stock_costs, "TSLA", LONG, 1.25, 400.0, 30).borrow_bps == 0
    # A zero side is costed as a long, and the minutes set the borrow.
    assert scorecards.trip_bps(config, "TSLA", FLAT, 400.0, 30) == long_trip
    assert scorecards.trip_bps(config, "TSLA", SHORT, 400.0, 355) - short_trip == pytest.approx(
        costs.holding_cost_bps(config.stock_costs, SHORT, 355) - costs.holding_cost_bps(config.stock_costs, SHORT, 30))
    t = c_at(7, 60)
    card = chooser_cards(config, start=iso(t - 1), end=iso(c_at(7, 90)))["population"]
    assert card.minutes == 1 and card.trail["chooser"] == ((t, "TSLA", SHORT),)
    assert card.strategies[0]["total bps"] == pytest.approx(-chooser.labels[t, "TSLA"] - short_trip, rel=1e-12)


def test_the_comparisons_match_the_plan_s_words(chooser):
    for config in (chooser.s1, chooser.s2):
        card = chooser_cards(config)["population"]
        rows = {row["strategy"]: row for row in card.strategies}
        expected = reference_chooser(chooser, config)
        assert list(rows) == ["chooser", "random pick", "all symbols", "largest raw forecast", "QQQ long", "flat"]
        for name, key in (("random pick", "random"), ("all symbols", "all"), ("largest raw forecast", "raw"),
                          ("QQQ long", "index")):
            assert rows[name]["total bps"] == pytest.approx(total(expected[key]), rel=1e-12), name
        assert rows["random pick"]["picks"] == 72 and rows["all symbols"]["picks"] == 576
        assert rows["all symbols"]["no result"] == 1 and rows["QQQ long"]["picks"] == 72
        assert rows["flat"] == {"strategy": "flat", "minutes": 72, "picks": 0, "flat": 72, "no result": 0,
                                "right share": None, "mean per pick bps": None, "mean per minute bps": 0.0,
                                "total bps": 0.0}
        assert [(ms, s) for ms, s, _ in expected["random"]] == [(ms, s) for ms, s, _ in card.trail["random pick"]]


def test_the_chooser_minus_each_comparison_uses_the_version_1_newey_west_error_at_the_lags_given(chooser):
    expected = reference_chooser(chooser, chooser.s2)
    errors = []
    for lags in (12, 3):
        card = chooser_cards(chooser.s2, lags=lags)["population"]
        errors.append(card.differences[0]["se"])
        # The first minutes, before any symbol had 60 labels, are the unscaled ones and stay out of the differences.
        skip = card.unscaled_minutes
        assert skip > 0
        for comparison, key in (("random pick", "random"), ("all symbols", "all"), ("QQQ long", "index")):
            diffs = np.array([a - b for (_, _, a), (_, _, b) in zip(expected["chooser"][skip:], expected[key][skip:])
                              if a is not None and b is not None])
            mean, se, t = stats.clustered(diffs, lags)
            row = next(d for d in card.differences if d["comparison"] == comparison)
            assert row["minutes"] == len(diffs)
            assert row["mean bps"] == pytest.approx(mean, rel=1e-9) and row["se"] == pytest.approx(se, rel=1e-9)
            assert row["t"] == pytest.approx(t, rel=1e-9)
            assert row["upper bound"] == pytest.approx(mean + UPPER * se, rel=1e-9)
    assert errors[0] != errors[1]


def test_each_run_gets_its_bps_predictors_and_never_a_score(chooser):
    assert list(chooser_cards(chooser.s1)) == ["population", "linear_prices"]
    assert list(chooser_cards(chooser.s2)) == ["population", "linear_prices", "linear_news"]
    # Far below cost everywhere, linear_prices never picks.
    assert chooser_cards(chooser.s2)["linear_prices"].picks == 0


# Lookahead


def test_a_label_after_t_never_moves_the_scale_or_the_pick_at_t(chooser, tmp_path):
    s1, s2 = moved(chooser.folder, tmp_path)
    before = chooser_cards(s2)["population"]
    t = c_at(6, 90)
    # AAPL's own label at t exits 30 minutes after it. Were it in AAPL's scale at t, the scale would grow about 30
    # times and TSLA would win, so the pick must stay AAPL while the result follows the new label.
    for ms, symbol in [(t, "AAPL"), (c_at(6, 120), "AAPL"), (c_at(7, 30), "AAPL")]:
        execute(s2.db_path, "UPDATE stock_labels SET label_bps = 3000.0 WHERE ts = ? AND symbol = ?", (iso(ms), symbol))
    with opened(s2) as source:
        assert scorecards.read_labels(source.conn, C_END, 30).scales.at("AAPL", sessions.k_at(t)) == pytest.approx(
            reference_scale(chooser.labels, "AAPL", t), rel=1e-12)
    after = chooser_cards(s2)["population"]
    assert picks_at(after)[t] == ("AAPL", LONG) and picks_at(before)[t] == ("AAPL", LONG)
    assert [p for p in after.trail["chooser"] if p[0] <= t] == [p for p in before.trail["chooser"] if p[0] <= t]
    assert after.strategies[0]["total bps"] != before.strategies[0]["total bps"]


def test_a_later_forecast_never_moves_the_pick_at_t(chooser, tmp_path):
    s1, s2 = moved(chooser.folder, tmp_path)
    before = chooser_cards(s2)["population"]
    t, later = c_at(6, 90), c_at(6, 120)
    execute(s2.db_path, "UPDATE stock_forecasts SET prediction = -1000.0 WHERE ts = ? AND symbol = 'NVDA' AND "
            "predictor = 'population'", (iso(later),))
    after = chooser_cards(s2)["population"]
    assert picks_at(after)[later] == ("NVDA", SHORT) and picks_at(before)[later] == (None, FLAT)
    assert {ms: p for ms, p in picks_at(after).items() if ms != later} == \
        {ms: p for ms, p in picks_at(before).items() if ms != later}


def test_a_forecast_whose_inputs_arrived_after_its_minute_is_never_used(chooser, tmp_path):
    s1, s2 = moved(chooser.folder, tmp_path)
    t = c_at(6, 30)
    execute(s2.db_path, "UPDATE stock_forecasts SET inputs_max_recv_at = ? WHERE ts = ? AND symbol = 'AAPL' AND "
            "predictor = 'population'", (iso(t + 1), iso(t)))
    card = chooser_cards(s2)["population"]
    assert picks_at(card)[t] == (None, FLAT) and card.minutes == 72 and card.picks == 5


def test_the_forecasts_left_out_for_late_inputs_are_counted(chooser, tmp_path):
    s1, s2 = moved(chooser.folder, tmp_path)
    t = c_at(6, 30)
    with opened(s1) as prices, opened(s2) as news:
        assert scorecards.late_forecasts(prices, C_START, C_END) == scorecards.late_forecasts(news, C_START, C_END) == 0
    execute(s2.db_path, "UPDATE stock_forecasts SET inputs_max_recv_at = ? WHERE ts = ? AND symbol IN ('AAPL', 'MSFT') "
            "AND predictor IN ('population', 'linear_news')", (iso(t + 1), iso(t)))
    # The prices run never reads linear_news, so a late row of it there is not one the scorecards left out.
    execute(s1.db_path, *forecast(t, "AAPL", "linear_news", 1.0, t + 1))
    with opened(s1) as prices, opened(s2) as news:
        assert scorecards.late_forecasts(news, C_START, C_END) == 4
        assert scorecards.late_forecasts(news, iso(t), C_END) == 0  # rows at or before the start are outside
        assert scorecards.late_forecasts(prices, C_START, C_END) == 0
        assert scorecards.scored_predictors(news.config) == ("population", "linear_prices", "linear_news", "news")
        assert scorecards.scored_predictors(prices.config) == ("population", "linear_prices")


def test_a_minute_whose_label_has_not_ended_by_the_end_is_left_out(chooser):
    end = iso(c_at(7, 90))
    card = chooser_cards(chooser.s2, end=end)["population"]
    assert max(ms for ms, _, _ in card.trail["chooser"]) == c_at(7, 60) and card.minutes == 48 + 12 + 2


# The seeded random pick


def test_the_random_pick_is_seeded_by_the_master_seed_the_run_and_the_minute(chooser):
    first, again = (chooser_cards(chooser.s2)["population"].trail["random pick"] for _ in range(2))
    assert first == again
    other_run = chooser_cards(chooser.s1)["population"].trail["random pick"]
    assert [ms for ms, _, _ in other_run] == [ms for ms, _, _ in first] and other_run != first
    seed = dataclasses.replace(chooser.s2, run=dataclasses.replace(chooser.s2.run, master_seed=777))
    with opened(chooser.s2) as source:
        reseeded = scorecards.chooser_cards(Source(source.name, source.conn, seed), C_START, C_END, LAGS, UPPER)
    assert reseeded[0].trail["random pick"] != first
    # The report's name for a run never changes the draw, only the run recorded in its config.
    assert chooser_cards(chooser.s2, name="news run")["population"].trail["random pick"] == first
    # Each predictor of one run draws the same at one minute.
    assert chooser_cards(chooser.s2)["linear_news"].trail["random pick"] == first


# The day scorecard


D_LAUNCH = "2026-11-16T14:25:00.000Z"
D_DAYS = run_days(D_LAUNCH, 11)
D_END = iso(D_DAYS[9].boundary(385))  # the 15:55 flatten of session 10, 30 November


def d_at(n, minute):
    return D_DAYS[n - 1].boundary(minute)


# Candle defects planted in the news run, by session, symbol and session minutes.
INVALID = {(3, "AAPL"): range(100, 109), (4, "AAPL"): range(100, 110), (5, "MSFT"): range(30, 31),
           (2, "AAPL"): range(388, 391)}
MISSING = {(6, "NVDA"): range(385, 386), (6, "META"): range(1, 391)}
ZERO, ABSENT = (7, "AAPL"), (8, "MSFT")  # the news run's population is zero, and missing


@dataclasses.dataclass(frozen=True)
class DayRuns:
    folder: Path
    s1: object
    s2: object
    refs: dict  # (session, symbol, minute) -> reference price of a valid candle
    forecasts: dict  # (run, predictor) -> {(session, symbol): value}
    labels: dict


def day_rows(rng):
    refs, candles = {}, []
    for n, day in enumerate(D_DAYS[1:10], 2):
        for j, symbol in enumerate(SYMBOLS):
            path = PRICES[symbol] * np.exp(np.cumsum(rng.normal(0, 4e-4, day.minutes)))
            for minute, price in enumerate(path, 1):
                if minute in MISSING.get((n, symbol), ()):
                    continue
                valid = minute not in INVALID.get((n, symbol), ())
                if valid:
                    refs[n, symbol, minute] = float(price)
                ms = day.boundary(minute)
                candles.append((records.INSERT_CANDLE, records.candle_row(StockCandle(
                    symbol, ms, sessions.k_at(ms), *(float(price),) * 5, 10.0, 6.0, 4.0, 3, 3, 0, valid))))
    labels, statements = {}, []
    for day in D_DAYS[:10]:
        for minute in day.evaluation_minutes():
            for symbol in SYMBOLS:
                labels[day.boundary(minute), symbol] = SPREAD[symbol] * float(rng.normal())
                statements += label_rows(day.boundary(minute), symbol, PRICES[symbol], labels[day.boundary(minute),
                                                                                               symbol])
    forecasts = {}
    for run, predictors in (("s1", ("population", "linear_prices")),
                            ("s2", ("population", "linear_prices", "linear_news", "news"))):
        for predictor in predictors:
            values = {(n, s): float(rng.normal(0, 3)) for n in range(2, 11) for s in SYMBOLS}
            if (run, predictor) == ("s2", "population"):
                values[ZERO] = 0.0
                del values[ABSENT]
                values.update({(8, s): v / 100 for (n, s), v in values.items() if n == 8})  # far below any cost
            forecasts[run, predictor] = values
    return refs, candles, statements, forecasts, labels


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    folder = tmp_path_factory.mktemp("day")
    s1_path, s2_path, _ = kit.configs(folder)
    s1, s2 = load_run_pair(s1_path, s2_path)
    refs, candles, labels_sql, forecasts, labels = day_rows(np.random.default_rng(5))
    for config, run in ((s1, "s1"), (s2, "s2")):
        rows = [forecast(d_at(n, 30), s, p, v) for (r, p), values in forecasts.items() if r == run
                for (n, s), v in values.items()]
        # A later forecast never makes a call.
        rows += [forecast(d_at(n, 60), s, "population", 500.0) for n in range(2, 11) for s in SYMBOLS]
        write_run(config, D_LAUNCH, labels_sql + rows + (candles if run == "s2" else []))
    return DayRuns(folder, s1, s2, refs, forecasts, labels)


def day_card(runs, end=D_END, configs=None):
    configs = configs or (runs.s1, runs.s2)
    with contextlib.ExitStack() as stack:
        sources = [stack.enter_context(opened(c, name)) for c, name in zip(configs, ("prices run", "news run"))]
        return scorecards.day_card(sources, D_LAUNCH, end, UPPER)


def day_labels(runs, end=D_END):
    with opened(runs.s2) as source:
        return {(d.number, d.symbol): d for d in scorecards.day_labels(source, D_LAUNCH, end)}


def reference_day(runs, n, symbol):
    """The day label of the plan's words, its minutes and its morning move, or None where it fails."""
    day, refs = D_DAYS[n - 1], runs.refs
    exit_minute = day.minutes - 5
    entry, exit_price = refs.get((n, symbol, 30)), refs.get((n, symbol, exit_minute))
    gaps, run = 0, 0
    for minute in range(31, exit_minute):
        run = 0 if (n, symbol, minute) in refs else run + 1
        gaps = max(gaps, run)
    label = None if entry is None or exit_price is None or gaps >= 10 else 1e4 * math.log(exit_price / entry)
    previous = [m for m in range(D_DAYS[n - 2].minutes, 0, -1) if (n - 1, symbol, m) in refs]
    move = None if entry is None or not previous else 1e4 * math.log(entry / refs[n - 1, symbol, previous[0]])
    return label, exit_minute - 30, move


def test_the_day_label_runs_from_10_00_to_the_flatten_and_fails_by_cause(day):
    found = day_labels(day)
    assert sorted({n for n, _ in found}) == list(range(3, 11)) and len(found) == 64
    for (n, symbol), d in found.items():
        label, minutes, move = reference_day(day, n, symbol)
        assert d.label == (None if label is None else pytest.approx(label, rel=1e-12)), (n, symbol)
        assert d.minutes == minutes and d.move == (None if move is None else pytest.approx(move, rel=1e-12))
    # 27 November closes at 13:00, so its label runs 175 minutes to 12:55, and every full session's 355 to 15:55.
    assert {d.minutes for d in found.values() if d.day.isoformat() == "2026-11-27"} == {175}
    assert {d.minutes for d in found.values() if d.day.isoformat() != "2026-11-27"} == {355}
    assert found[9, "TSLA"].early and not found[8, "TSLA"].early
    assert {k: d.cause for k, d in found.items() if d.cause} == {
        (4, "AAPL"): scorecards.STRETCH, (5, "MSFT"): scorecards.NO_ENTRY, (6, "NVDA"): scorecards.NO_EXIT,
        (6, "META"): scorecards.NO_ENTRY}


def test_nine_invalid_minutes_still_count_and_the_stricter_rule_is_only_a_count(day):
    found = day_labels(day)
    assert found[3, "AAPL"].label is not None and not found[3, "AAPL"].strict
    assert found[4, "AAPL"].cause == scorecards.STRETCH
    assert sum(d.strict for d in found.values()) == 64 - 5
    card = day_card(day)
    assert card.labels[0] == {"scope": "with early closes", "sessions": 8, "symbol days": 64, "counted": 60,
                              "no entry price": 2, "no exit price": 1, "invalid stretch": 1,
                              "every minute valid": 59}
    assert (card.sessions, card.counted, card.failed) == (8, 60, 4)


def test_the_gap_rule_follows_max_resume_gap_minutes_recorded_at_launch(day):
    wider = dataclasses.replace(day.s2, persistence=dataclasses.replace(day.s2.persistence,
                                                                        max_resume_gap_minutes=11))
    with opened(wider) as source:
        found = {(d.number, d.symbol): d for d in scorecards.day_labels(source, D_LAUNCH, D_END)}
    assert found[4, "AAPL"].cause is None


def test_the_day_entry_is_10_00_whatever_the_forecast_horizon(day):
    shorter = dataclasses.replace(day.s2, time=dataclasses.replace(day.s2.time, horizon_minutes=15))
    with opened(day.s2) as base, opened(shorter) as other:
        before = scorecards.day_labels(base, D_LAUNCH, D_END)
        after = scorecards.day_labels(other, D_LAUNCH, D_END)
        assert after == before and {d.entry_minute for d in after} == {scorecards.DAY_ENTRY_MINUTE} == {30}
        calls = [c.positions for c in scorecards.forecast_calls(other, after)]
        assert calls == [c.positions for c in scorecards.forecast_calls(base, before)] and all(calls)
        assert scorecards.day_chooser(other, after, D_END).positions == \
            scorecards.day_chooser(base, before, D_END).positions


def test_morning_momentum_starts_from_the_previous_session_s_last_valid_reference_price(day):
    found = day_labels(day)
    # AAPL's last three minutes of session 2 were invalid, so session 3's move starts from minute 387.
    assert found[3, "AAPL"].move == pytest.approx(1e4 * math.log(day.refs[3, "AAPL", 30] / day.refs[2, "AAPL", 387]))
    # META has no candle in session 6, so session 7 has no momentum call and session 6 no entry.
    assert found[7, "META"].move is None and found[6, "META"].move is None
    card = {row["call"]: row for row in day_card(day).calls if row["scope"] == "with early closes"}
    assert card["morning momentum"]["calls"] == 60 - 1


def test_momentum_ignores_every_candle_after_10_00(day, tmp_path):
    s1, s2 = moved(day.folder, tmp_path)
    execute(s2.db_path, "UPDATE stock_candles SET ref_price = ref_price * 2 WHERE symbol = 'AAPL' AND ts > ? "
            "AND ts < ?", (iso(d_at(3, 30)), iso(d_at(3, 385))))
    with opened(s2) as source:
        found = {(d.number, d.symbol): d for d in scorecards.day_labels(source, D_LAUNCH, D_END)}
    assert found[3, "AAPL"].move == pytest.approx(day_labels(day)[3, "AAPL"].move, rel=1e-12)


def test_each_call_earns_its_side_times_the_day_label_less_the_round_trip_over_the_day(day):
    config = day.s2
    rows = {(row["scope"], row["call"]): row for row in day_card(day).calls}
    found = day_labels(day)
    held, coin, coin_short = [], [], 0
    for (n, symbol), d in found.items():
        if d.label is None:
            continue
        held.append(d.label - reference_cost(config, symbol, LONG, d.entry, d.minutes))
        rng = np.random.default_rng(np.random.SeedSequence(
            [config.run.master_seed, scorecards.DAY_COIN, D_DAYS[n - 1].first_k, scorecards._tag(symbol)]))
        side = LONG if rng.integers(2) else SHORT
        coin_short += side == SHORT
        coin.append(side * d.label - reference_cost(config, symbol, side, d.entry, d.minutes))
    assert coin_short > 0
    assert rows["with early closes", "hold long"]["total bps"] == pytest.approx(math.fsum(held), rel=1e-12)
    assert rows["with early closes", "seeded coin"]["total bps"] == pytest.approx(math.fsum(coin), rel=1e-12)
    assert rows["with early closes", "hold long"]["calls"] == 60 and rows["with early closes", "flat"] == {
        **rows["with early closes", "flat"], "calls": 0, "total bps": 0.0, "right share": None}
    # A short pays its borrow over the 355 or 175 minutes it is held.
    assert scorecards.trip_bps(config, "TSLA", SHORT, 400.0, 175) < scorecards.trip_bps(config, "TSLA", SHORT, 400.0,
                                                                                         355)


def test_forecast_calls_trade_by_sign_and_a_zero_or_missing_call_is_flat(day):
    rows = {(row["scope"], row["call"]): row for row in day_card(day).calls}
    found = day_labels(day)
    population = day.forecasts["s2", "population"]
    expected, right = [], 0
    for (n, symbol), d in found.items():
        f = population.get((n, symbol), 0.0)
        if d.label is None or f == 0:
            continue
        side = LONG if f > 0 else SHORT
        right += side * d.label > 0
        expected.append(side * d.label - reference_cost(day.s2, symbol, side, d.entry, d.minutes))
    row = rows["with early closes", "news run population"]
    assert found[ZERO].label is not None and found[ABSENT].label is not None
    assert row["calls"] == 60 - 2 == len(expected) and row["symbol days"] == 60 and row["days"] == 8
    assert row["total bps"] == pytest.approx(math.fsum(expected), rel=1e-12)
    assert row["right share"] == pytest.approx(right / len(expected))
    assert row["mean per call bps"] == pytest.approx(math.fsum(expected) / len(expected), rel=1e-12)
    # The prices run's calls come from its own forecasts at 10:00, never the 500 bps rows at 10:30.
    s1_row = rows["with early closes", "prices run population"]
    assert s1_row["calls"] == 60 and abs(s1_row["mean per call bps"]) < 200


def test_a_news_score_whose_inputs_arrived_after_10_00_makes_no_call(day, tmp_path):
    s1, s2 = moved(day.folder, tmp_path)
    execute(s2.db_path, "UPDATE stock_forecasts SET inputs_max_recv_at = ? WHERE ts = ? AND symbol = 'AAPL' AND "
            "predictor = 'news'", (iso(d_at(10, 30) + 1), iso(d_at(10, 30))))
    rows = {(row["scope"], row["call"]): row for row in day_card(day, configs=(s1, s2)).calls}
    assert rows["with early closes", "news run news"]["calls"] == 59
    assert rows["with early closes", "news run linear_news"]["calls"] == 60


def test_the_day_correlation_uses_the_version_1_block_test_with_one_block_per_session(day):
    found = day_labels(day)
    for call, values in (("news run news", day.forecasts["s2", "news"]),
                         ("prices run linear_prices", day.forecasts["s1", "linear_prices"])):
        rows = [(n, s, values[n, s], d.label) for (n, s), d in found.items()
                if d.label is not None and (n, s) in values]
        _, symbols, (p, y) = stats.pivot(*zip(*rows))
        products = stats.zscores(p) * stats.zscores(y)
        blocks = {i: list(row[~np.isnan(row)]) for i, row in enumerate(products)}
        test = block_test(blocks)
        means = np.array([np.mean(v) for v in blocks.values()])
        se = means.std(ddof=1) / math.sqrt(len(means))
        row = next(r for r in day_card(day).calls if r["call"] == call and r["scope"] == "with early closes")
        assert (row["sessions"], row["rows"]) == (8, 60)
        assert row["corr"] == pytest.approx(test["mean"], rel=1e-12) and row["t"] == pytest.approx(test["t"], rel=1e-12)
        assert row["se"] == pytest.approx(se, rel=1e-12)
        assert row["upper bound"] == pytest.approx(test["mean"] + UPPER * se)
        # Never the 12 lag error of the per minute statistics.
        assert row["se"] != pytest.approx(stats.clustered(np.nanmean(products, axis=1), 12)[1], rel=1e-6)
    hold = next(r for r in day_card(day).calls if r["call"] == "hold long")
    assert hold["corr"] is None and hold["left out"] == " ".join(sorted(SYMBOLS))


def test_the_day_chooser_ranks_the_news_run_population_over_its_scale_without_a_cost_gate(day):
    population = day.forecasts["s2", "population"]
    expected = {}
    for n in range(3, 11):
        ms = d_at(n, 30)
        ranked = []
        for i, symbol in enumerate(SYMBOLS):
            scale = reference_scale(day.labels, symbol, ms)
            if (n, symbol) in population and scale is not None and population[n, symbol] != 0:
                ranked.append((abs(population[n, symbol]) / scale, -i, symbol))
        if ranked:
            top = max(ranked)[2]
            expected[n, top] = LONG if population[n, top] > 0 else SHORT
    # The scale needs 60 labels, which sessions 1 to 5 give by 10:00 of session 6.
    assert sorted(n for n, _ in expected) == [6, 7, 8, 9, 10]
    with opened(day.s2, "news run") as source:
        call = scorecards.day_chooser(source, list(day_labels(day).values()), D_END)
    assert {key: side for key, (side, _) in call.positions.items()} == expected
    # Session 8's forecasts are all far below any cost and it still picks one.
    assert any(n == 8 for n, _ in call.positions)


def test_the_calls_are_the_plan_s_and_cover_sessions_3_to_the_end_with_and_without_early_closes(day):
    card = day_card(day)
    calls = [row["call"] for row in card.calls if row["scope"] == "with early closes"]
    assert calls == ["prices run population", "prices run linear_prices", "news run population",
                     "news run linear_prices", "news run linear_news", "news run news", "news run chooser",
                     "morning momentum", "hold long", "seeded coin", "flat"]
    assert [row["scope"] for row in card.labels] == ["with early closes", "without early closes"]
    without = card.labels[1]
    assert (without["sessions"], without["symbol days"], without["counted"]) == (7, 56, 52)
    rows = {(row["scope"], row["call"]): row for row in card.calls}
    assert rows["without early closes", "hold long"]["calls"] == 52 and rows["without early closes", "hold long"][
        "days"] == 7
    assert {row["scope"] for row in card.symbols} == set(scorecards.SCOPES)
    # One minute before session 10's flatten it is left out, and before session 3 there is nothing.
    assert day_card(day, end=iso(d_at(10, 384))).sessions == 7
    empty = day_card(day, end=iso(d_at(2, 390)))
    assert (empty.sessions, empty.counted, empty.failed) == (0, 0, 0)
    assert all(row["calls"] == 0 and row["corr"] is None for row in empty.calls)


def test_the_day_labels_come_from_the_news_run_and_the_prices_run_alone_still_scores(day):
    # The prices run holds no candles here, as the shared market layer would have given it the same ones.
    with opened(day.s1, "prices run") as source:
        alone = scorecards.day_card([source], D_LAUNCH, D_END, UPPER)
    assert alone.counted == 0 and [row["call"] for row in alone.calls][:2] == ["prices run population",
                                                                                   "prices run linear_prices"]
    assert scorecards.day_card([], D_LAUNCH, D_END, UPPER).calls == ()


# Isolation


def module_level_imports(path):
    """The modules a file imports at its top level, outside any function."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def every_import(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module or ''}.{a.name}" for a in node.names)
    return names


PROCESS_FILES = sorted([*(V2 / "stocks").glob("*.py"), *(V2 / "news").glob("*.py"), V2 / "__main__.py"])


@pytest.mark.parametrize("path", PROCESS_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_stock_or_news_process_module_imports_the_scorecards(path):
    assert not {n for n in every_import(path) if "scorecards" in n}
    assert "scorecards" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["evaluate.py", "dashboard.py"])
def test_evaluate_and_the_dashboard_import_the_scorecards_only_inside_a_function(name):
    path = V2 / name
    assert not {n for n in module_level_imports(path) if "scorecards" in n}
    assert "btc_survivor.v2.scorecards" in every_import(path)


def module_file(name):
    """The source file of a btc_survivor module name, or None for a name that is not one."""
    if name != "btc_survivor" and not name.startswith("btc_survivor."):
        return None
    base = SRC.joinpath(*name.split("."))
    return next((path for path in (base.with_suffix(".py"), base / "__init__.py") if path.is_file()), None)


def reachable(start):
    """Every btc_survivor module file that an import at any depth, one inside a function included, reaches from
    start, with the packages each passes through."""
    seen, todo = set(), [start]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in every_import(path):
            parts = name.split(".")
            todo += [found for i in range(1, len(parts) + 1)
                     if (found := module_file(".".join(parts[:i]))) is not None and found not in seen]
    return seen


@pytest.mark.parametrize("path", sorted([*(V2 / "stocks").glob("*.py"), *(V2 / "news").glob("*.py")]),
                         ids=lambda p: str(p.relative_to(SRC)))
def test_no_import_at_any_depth_from_a_stock_or_news_module_reaches_the_scorecards(path):
    found = reachable(path)
    assert not found & {V2 / name for name in ("scorecards.py", "evaluate.py", "dashboard.py", "__main__.py")}


def test_the_import_walk_follows_imports_inside_functions():
    assert all(V2 / "scorecards.py" in reachable(V2 / name) for name in ("evaluate.py", "dashboard.py"))
    assert V2 / "stocks" / "costs.py" in reachable(V2 / "scorecards.py")
    assert len(reachable(V2 / "stocks" / "process.py")) > 20


def test_starting_the_stock_or_the_news_process_never_loads_the_scorecards():
    code = ("import sys\n"
            "import btc_survivor.v2.__main__ as cli\n"
            "import btc_survivor.v2.stocks.process, btc_survivor.v2.news.runner, btc_survivor.v2.evaluate\n"
            "import btc_survivor.v2.dashboard\n"
            "cli.build_parser()\n"
            "print('btc_survivor.v2.scorecards' in sys.modules)\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "False"



def test_minutes_before_any_scale_stay_out_of_the_differences_but_are_counted(chooser):
    # Fixed before launch on 15 October 2026: the chooser cannot act before a symbol has a scale, while the comparisons
    # pay their costs, so those minutes would flatter it and are left out of the differences, counted apart.
    card = chooser_cards(chooser.s2)["population"]
    flat = next(d for d in card.differences if d["comparison"] == "flat")
    assert card.unscaled_minutes > 0
    assert flat["minutes"] < card.minutes - card.unscaled_minutes + 1
    assert next(r for r in card.strategies if r["strategy"] == "random pick")["minutes"] == card.minutes
