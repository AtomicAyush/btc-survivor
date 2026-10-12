"""The version 1 Account under stock costs, 5,000 seeded sequences of opens, closes, flips, stops, session closes,
session flattens across nights and weekends, rent, borrow, cent rounded fees, rejections and deaths, with the three
version 1 invariant lines and an independent cash ledger checked after every step."""

import math
from datetime import date

import numpy as np
import pytest

from btc_survivor.account import Account
from btc_survivor.clock import MS_PER_HOUR
from btc_survivor.types import LONG, SHORT, Leg
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks.bot import SESSION_CLOSE, SESSION_FLATTEN
from btc_survivor.v2.stocks.broker import SYMBOL_STALE, StockBroker
from btc_survivor.v2.stocks.finnhub import Print
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml")
COSTS = CFG.stock_costs
SYMBOLS = CFG.stock_feed.symbols
PLANNING = {"AAPL": 250.0, "MSFT": 500.0, "NVDA": 180.0, "AMZN": 220.0, "GOOGL": 250.0, "META": 700.0,
            "TSLA": 400.0, "QQQ": 600.0}
FIRST = sessions.session_on(date(2026, 10, 12))
TOL = 1e-6


def invariant_lines(acct, mark):
    """The three version 1 lines, as differences that must be zero."""
    unrealized = acct.unrealized(mark)
    equity = acct.equity(mark)
    cash_line = acct.cash - (acct.start_equity + acct.realized_gross - acct.fees - acct.funding - acct.rent
                             + acct.writeoff)
    equity_line = equity - (acct.cash + unrealized)
    loss_line = (acct.start_equity - equity) - (acct.fees + acct.funding + acct.rent + acct.stop_loss_losses
                                                + acct.other_losses - acct.writeoff - unrealized)
    return cash_line, equity_line, loss_line


def cents_up(amount):
    """The fee rule written out again, up to the next cent with a one cent minimum."""
    return max(0.01, math.ceil(amount * 100 - 1e-6) / 100)


def expected_fee(taker, qty, price):
    if taker != "sell":
        return 0.0
    sec = cents_up(qty * price * 20.60 / 1e6)
    taf = cents_up(min(qty * 0.000195, 9.79))
    return sec + taf


class Sequence:
    """One account with an independent shadow of its cash and position, driven by random steps."""

    def __init__(self, rng, symbol, start):
        self.rng = rng
        self.symbol = symbol
        self.acct = Account(start)
        self.broker = StockBroker.from_config(CFG)
        self.allowance = costs.allowance_bps(COSTS, symbol)
        self.price = PLANNING[symbol]
        self.session = FIRST
        self.minute = 30
        self.cash = start
        self.funded_ms = 0
        self.dead = False

    @property
    def now(self):
        return self.session.boundary(self.minute)

    def advance(self):
        """Move the price and the clock, sometimes past a night or a weekend, which accrues no rent."""
        self.price *= math.exp(self.rng.normal(0, 0.004))
        step = int(self.rng.integers(1, 40))
        if self.minute + step > self.session.minutes or self.rng.random() < 0.05:
            self.session = sessions.next_session(self.session.close_ms)
            self.minute = int(self.rng.integers(1, 60))
        else:
            self.minute += step

    def mark(self):
        acct = self.acct
        return costs.mark_price(self.price, acct.side, self.allowance) if acct.is_open else None

    def send(self, legs, reason):
        """Fill an order on one print after its latency, checking each fill's fees and the shadow ledger."""
        order = self.broker.make_order(1, self.symbol, legs, self.now, self.price, reason, self.rng)
        print_price = self.price * math.exp(self.rng.normal(0, 0.0005))
        fill = Print(self.symbol, print_price, 1.0, order.due_ms - int(self.rng.integers(0, 500)), ())
        held = (self.acct.side, self.acct.qty, self.acct.entry_price)
        ex = self.broker.execute(order, fill, order.due_ms, self.acct)
        for f in ex.fills:
            if f.status != "filled":
                continue
            assert f.fee == pytest.approx(expected_fee(f.taker_side, f.qty, f.price), abs=1e-12)
            assert round(f.fee * 100) == pytest.approx(f.fee * 100, abs=1e-9)
            if f.leg == "close":
                side, qty, entry = held
                if side == SHORT:
                    self.cash -= costs.borrow_bps_per_hour(COSTS) / 1e4 * qty * entry * \
                        (order.due_ms - self.funded_ms) / MS_PER_HOUR
                self.cash += side * qty * (f.price - entry) - f.fee
            else:
                self.cash -= f.fee
                self.funded_ms = order.due_ms
        return ex

    def step(self):
        acct, rng = self.acct, self.rng
        self.advance()
        op = int(rng.integers(0, 9))
        if op == 0 and not acct.is_open:
            side = LONG if rng.random() < 0.5 else SHORT
            self.send([Leg("open", side, self.broker.open_qty(side, self.price))], "decision")
        elif op == 1 and acct.is_open:
            self.send([Leg("close", acct.side)], str(rng.choice(["decision", "stop", SESSION_CLOSE])))
        elif op == 2 and acct.is_open:
            new = -acct.side
            self.send([Leg("close", acct.side), Leg("open", new, self.broker.open_qty(new, self.price))], "decision")
        elif op == 3 and acct.is_open:
            self.session = sessions.next_session(self.session.close_ms + 2 * 24 * MS_PER_HOUR * int(rng.integers(0, 2)))
            self.minute = 1
            ex = self.send([Leg("close", acct.side)], SESSION_FLATTEN)
            assert ex.closed.exit_reason == SESSION_FLATTEN
        elif op == 4:
            amount = costs.rent_per_minute(COSTS) * int(rng.integers(1, 391))
            acct.charge_rent(amount)
            self.cash -= amount
        elif op == 5 and acct.is_open:
            paid = acct.accrue_funding(self.now, costs.borrow_bps_per_hour(COSTS))
            if acct.side == SHORT:
                expected = costs.borrow_bps_per_hour(COSTS) / 1e4 * acct.qty * acct.entry_price * \
                    (self.now - self.funded_ms) / MS_PER_HOUR
                assert paid == pytest.approx(expected, rel=1e-9, abs=1e-15)
                self.cash -= expected
                self.funded_ms = self.now
            else:
                assert paid == 0.0
        elif op == 6 and acct.is_open:
            order = self.broker.make_order(1, self.symbol, [Leg("close", acct.side)], self.now, self.price, "stop",
                                           rng)
            before = acct.cash
            ex = self.broker.reject(order, SYMBOL_STALE, acct)
            assert ex.rejected == SYMBOL_STALE and acct.cash == before and acct.is_open
        elif op == 7:
            mark = self.mark()
            side = acct.side if acct.is_open else LONG
            qty = acct.qty if acct.is_open else self.broker.open_qty(LONG, self.price)
            line = costs.death_line(COSTS, self.symbol, side, qty, self.price, 30)
            if acct.equity(mark) < line or rng.random() < 0.02:
                if acct.is_open:
                    self.send([Leg("close", acct.side)], "death")
                self.cash += acct.clamp_cash()
                self.dead = True
        elif op == 8 and acct.is_open:
            self.send([Leg("close", acct.side)], "stop")


def test_three_invariant_lines_hold_after_every_step_of_5000_seeded_sequences():
    rng = np.random.default_rng(20261019)
    steps = deaths = writeoffs = shorts = flips = 0
    for index in range(5000):
        symbol = SYMBOLS[index % len(SYMBOLS)]
        start = float(rng.choice([10_000.0, 10_000.0, 40.0, 3.0]))
        seq = Sequence(rng, symbol, start)
        for _ in range(int(rng.integers(5, 40))):
            trades_before = seq.acct.trades
            seq.step()
            mark = seq.mark()
            lines = invariant_lines(seq.acct, mark)
            assert all(abs(x) <= TOL for x in lines), (index, lines)
            assert seq.acct.invariant_errors(mark) == []
            assert seq.acct.cash == pytest.approx(seq.cash, abs=TOL)
            steps += 1
            shorts += seq.acct.side == SHORT
            flips += seq.acct.trades > trades_before and seq.acct.is_open
            if seq.dead:
                deaths += 1
                writeoffs += seq.acct.writeoff > 0
                break
    assert steps > 50_000 and deaths > 500 and writeoffs > 50 and shorts > 1_000 and flips > 1_000
