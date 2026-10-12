"""The outcomes fixed by arithmetic of PLAN_V2 section 12, worked by hand from the committed S1 config: the burn of a
quiet learner, its equity with no gross profit, and how long death would take."""

import math

import pytest

from btc_survivor.types import SHORT
from btc_survivor.v2 import evaluate
from btc_survivor.v2.config import parse_stock_config
from btc_survivor.v2.evaluate import (Run, burn_by_cutoff, burn_per_session, death_horizon_sessions, death_line,
                                      founder_bound, founder_equity, outcome_money, round_trip_dollars)
from btc_survivor.v2.stocks import costs
from helpers import ROOT

CONFIG = parse_stock_config((ROOT / "configs" / "v2" / "s1_prices.toml").read_text(encoding="utf-8"),
                            ROOT / "configs" / "v2")
COSTS = CONFIG.stock_costs
RENT = 1.16 * 365 / (252 * 390)  # dollars per usable session minute, 0.0043081
MIDDLE = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
WIDE = ("META", "TSLA")


def test_round_trips_are_the_plan_table():
    # Two legs of the allowance on 500 dollars, plus 0.02 of SEC fee and 0.01 of trading activity fee on the sale.
    assert round_trip_dollars(COSTS, "QQQ", 500, 30) == pytest.approx(2 * 1.5e-4 * 500 + 0.03)
    assert [round_trip_dollars(COSTS, s, 500, 30) for s in MIDDLE] == pytest.approx([2 * 2e-4 * 500 + 0.03] * 5)
    assert [round_trip_dollars(COSTS, s, 500, 30) for s in WIDE] == pytest.approx([2 * 3e-4 * 500 + 0.03] * 2)
    bps = [round(round_trip_dollars(COSTS, s, 500, 30) / 500 * 1e4, 6) for s in ("QQQ", "AAPL", "TSLA")]
    assert bps == [3.6, 4.6, 6.6]


def test_burn_per_session_and_by_the_cutoff_by_hand():
    assert costs.rent_per_minute(COSTS) == pytest.approx(RENT, rel=1e-12) and RENT == pytest.approx(0.0043081, abs=1e-7)
    assert costs.session_rent(COSTS) == pytest.approx(1.68, abs=0.001)
    burns = {s: burn_per_session(COSTS, s, 500, 30) for s in CONFIG.stock_feed.symbols}
    assert burns["QQQ"] == pytest.approx(390 * RENT + 2 * 0.18)
    assert [burns[s] for s in MIDDLE] == pytest.approx([390 * RENT + 2 * 0.23] * 5)
    assert [burns[s] for s in WIDE] == pytest.approx([390 * RENT + 2 * 0.33] * 2)
    assert [round(burns[s], 2) for s in ("QQQ", "AAPL", "META")] == [2.04, 2.14, 2.34]
    assert round(sum(burns.values()) / 8, 2) == 2.18
    # 40 full sessions of rent, and two round trips a session from the first trades of session 3 to session 40.
    rent, trips = burn_by_cutoff(COSTS, CONFIG.stock_feed.symbols, 500, 30, 15600)
    assert rent == pytest.approx(15600 * RENT) and round(rent, 2) == 67.21
    assert trips == pytest.approx(2 * 38 * (0.18 + 5 * 0.23 + 2 * 0.33) / 8) and round(trips) == 19
    assert round(rent + trips) == 86
    assert burn_by_cutoff(COSTS, CONFIG.stock_feed.symbols, 500, 30, 780)[1] == 0.0


def test_founder_equity_by_hand():
    # 15,600 usable minutes and 75 round trips of AAPL at 0.25 dollars each, with no gross profit.
    assert founder_equity(10000.0, RENT, 15600, 75 * 0.25) == pytest.approx(10000 - 15600 * RENT - 18.75)
    assert round(founder_equity(10000.0, RENT, 15600, 75 * 0.25), 2) == 9914.04
    # Two standard deviations of a 50 bps move on 500 dollars over 75 trades that all move together.
    assert founder_bound(75, 500) == pytest.approx(2 * math.sqrt(75) * 2.5) and 43 < founder_bound(75, 500) <= 45
    assert founder_bound(0, 500) == 0.0


def books(equity_offset, trips=75, usable=15600):
    paid = trips * 0.25
    return [{"bot_id": i, "home": "AAPL", "equity": founder_equity(10000.0, RENT, usable, paid) + offset,
             "rent": RENT * usable, "paid": paid, "trips": trips, "usable": usable}
            for i, offset in enumerate((equity_offset - 5, equity_offset + 5))]


def test_founder_equity_and_burn_are_judged_from_the_learners_books():
    run = Run("prices run", None, {}, CONFIG)
    outcomes = {o.name: o for o in outcome_money(run, books(30.0))}
    assert outcomes["founder equity"].matched is True and outcomes["burn"].matched is True
    assert outcomes["founder equity"].expected == f"{10000 - 15600 * RENT - 18.75:.2f} within 43.30"
    assert outcome_money(run, books(44.0))[1].matched is False
    # Paying 0.25 a round trip against the planned 0.23 is within a quarter, three times as much is not.
    heavy = [{**book, "paid": 3 * book["paid"]} for book in books(0.0)]
    assert outcome_money(run, heavy)[0].matched is False
    # Before any trade the bound is a cent, so rent alone must match to the cent.
    assert [o.matched for o in outcome_money(run, books(0.02, trips=0, usable=390))] == [True, False]
    assert [o.matched for o in outcome_money(run, books(0.004, trips=0, usable=390))] == [True, True]
    assert [o.matched for o in outcome_money(run, [])] == [None, None]


def test_the_death_threshold_and_horizon_by_hand():
    assert death_line(COSTS, "QQQ", 500, 30) == pytest.approx(0.18 + 390 * RENT) and \
        round(death_line(COSTS, "QQQ", 500, 30), 2) == 1.86
    assert round(death_line(COSTS, "META", 500, 30), 2) == 2.01
    # A whole share short of META at its planning price of 700 dollars is the dearest case.
    assert round(costs.death_line(COSTS, "META", SHORT, 1, 700.0, 30), 2) == 2.13
    # A threshold of about 2 dollars and a burn of about 2.2 a session take about 4,500 sessions.
    assert death_horizon_sessions(10000.0, 2.0, 2.2) == pytest.approx(9998 / 2.2)
    assert round(death_horizon_sessions(10000.0, 2.0, 2.2), -2) == 4500
    symbols = CONFIG.stock_feed.symbols
    threshold = sum(death_line(COSTS, s, 500, 30) for s in symbols) / 8
    burn = sum(burn_per_session(COSTS, s, 500, 30) for s in symbols) / 8
    assert 4500 < death_horizon_sessions(10000.0, threshold, burn) < 4600
    assert evaluate.ROUND_TRIPS_PER_SESSION == 2 and evaluate.FIRST_TRADE_SESSION == 3


def test_evaluate_prints_the_planned_figures_beside_the_outcomes():
    rows = {row["symbol"]: row for row in evaluate.planned_rows(CONFIG, 15600)}
    assert rows["QQQ"] == {"symbol": "QQQ", "round trip": "0.18", "burn per session": "2.04", "death threshold": "1.86",
                           "sessions to death": "4901"}
    assert [rows[s]["burn per session"] for s in ("AAPL", "TSLA")] == ["2.14", "2.34"]
    assert rows["all"] == {"symbol": "all", "rent by cutoff": "67.21", "round trips by cutoff": "18.91"}
