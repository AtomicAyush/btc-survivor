import numpy as np
import pytest

from btc_survivor.account import Account, death_threshold
from btc_survivor.types import LONG, SHORT

HOUR = 3_600_000


def test_long_round_trip_profit_and_win():
    a = Account(1000.0)
    a.open(LONG, 2.0, 100.0, fee=0.5, ms=0, trade_id=1)
    assert a.equity(110.0) == pytest.approx(1000.0 - 0.5 + 20.0)
    closed = a.close(110.0, fee=0.6, ms=HOUR, exit_reason="signal", bps_per_hour=0.5)
    assert closed.gross == pytest.approx(20.0)
    assert closed.net == pytest.approx(20.0 - 1.1)
    assert a.cash == pytest.approx(1000.0 - 1.1 + 20.0)
    assert a.wins == 1 and a.trades == 1 and a.funding == 0.0  # longs pay no funding
    assert a.invariant_errors(None) == []


def test_short_profits_when_price_falls():
    a = Account(1000.0)
    a.open(SHORT, 1.0, 100.0, fee=0.0, ms=0, trade_id=1)
    assert a.unrealized(90.0) == pytest.approx(10.0)
    closed = a.close(90.0, fee=0.0, ms=0, exit_reason="signal", bps_per_hour=0.0)
    assert closed.gross == pytest.approx(10.0)


def test_win_is_judged_on_net_profit():
    a = Account(1000.0)
    a.open(LONG, 1.0, 100.0, fee=1.0, ms=0, trade_id=1)
    a.close(100.5, fee=1.0, ms=0, exit_reason="signal", bps_per_hour=0.0)
    assert a.wins == 0


def test_rent_reduces_cash():
    a = Account(1000.0)
    for _ in range(1440):
        a.charge_rent(1.16 / 1440)
    assert a.rent == pytest.approx(1.16)
    assert a.cash == pytest.approx(1000.0 - 1.16)
    assert a.invariant_errors(None) == []


def test_clamp_turns_negative_cash_into_writeoff():
    a = Account(10.0)
    a.open(LONG, 1.0, 100.0, fee=0.0, ms=0, trade_id=1)
    a.close(80.0, fee=0.5, ms=0, exit_reason="death", bps_per_hour=0.0)
    assert a.cash == pytest.approx(-10.5)
    assert a.clamp_cash() == pytest.approx(10.5)
    assert a.cash == 0.0 and a.writeoff == pytest.approx(10.5)
    assert a.invariant_errors(None) == []


def test_cause_of_death_is_the_largest_part_and_parts_add_up():
    a = Account(100.0)
    a.open(LONG, 1.0, 100.0, fee=3.0, ms=0, trade_id=1)
    a.close(95.0, fee=3.0, ms=0, exit_reason="stop", bps_per_hour=0.0)
    a.charge_rent(2.0)
    parts = a.breakdown()
    assert parts == {"fees": 6.0, "funding": 0.0, "rent": 2.0, "stop_losses": 5.0, "trading_losses": 0.0}
    assert a.death_cause() == "fees"
    assert sum(parts.values()) - a.writeoff == pytest.approx(a.start_equity - a.equity(None))


def _account_dying_of(cause):
    a = Account(100.0)
    if cause == "fees":
        a.open(LONG, 1.0, 100.0, fee=30.0, ms=0, trade_id=1)
        a.close(100.0, fee=30.0, ms=0, exit_reason="signal", bps_per_hour=0.0)
    elif cause == "funding":
        a.open(SHORT, 1.0, 100.0, fee=0.0, ms=0, trade_id=1, open_charge=1.0)
        a.close(100.0, fee=0.0, ms=10 * HOUR, exit_reason="signal", bps_per_hour=500.0)
    elif cause == "rent":
        a.charge_rent(50.0)
    elif cause == "stop_losses":
        a.open(LONG, 1.0, 100.0, fee=1.0, ms=0, trade_id=1)
        a.close(60.0, fee=1.0, ms=0, exit_reason="stop", bps_per_hour=0.0)
    elif cause == "trading_losses":
        a.open(SHORT, 1.0, 100.0, fee=1.0, ms=0, trade_id=1)
        a.close(140.0, fee=1.0, ms=0, exit_reason="signal", bps_per_hour=0.0)
    a.charge_rent(0.5)
    return a


@pytest.mark.parametrize("cause", ["fees", "funding", "rent", "stop_losses", "trading_losses"])
def test_each_cause_of_death_is_attributed(cause):
    a = _account_dying_of(cause)
    assert a.death_cause() == cause
    assert a.invariant_errors(None) == []


def test_death_threshold_formula():
    assert death_threshold(182.0, 500.0, 1.16) == pytest.approx(0.0182 * 500 + 1.16)


def test_misuse_is_rejected():
    a = Account(100.0)
    with pytest.raises(ValueError):
        a.close(100.0, 0.0, 0, "signal", 0.0)
    a.open(LONG, 1.0, 100.0, 0.0, 0, 1)
    with pytest.raises(ValueError):
        a.open(SHORT, 1.0, 100.0, 0.0, 0, 2)
    with pytest.raises(ValueError):
        a.unrealized(None)


def test_balance_invariant_holds_across_thousands_of_random_sequences():
    """Compare the account with an independent shadow ledger after every step of 5,000 random sequences."""
    rng = np.random.default_rng(12345)
    checked = 0
    for _ in range(5000):
        start = float(rng.choice([1000.0, 10000.0, 50.0]))
        a = Account(start)
        shadow_cash, price, now = start, 100.0, 0
        for _ in range(int(rng.integers(5, 40))):
            price = max(1.0, price * float(np.exp(rng.normal(0, 0.03))))
            now += int(rng.integers(1, 3 * HOUR))
            op = rng.integers(0, 5)
            funding_rate = float(rng.choice([0.0, 0.5, 3.0]))
            if op == 0 and not a.is_open:
                side = LONG if rng.random() < 0.5 else SHORT
                qty, fee = float(rng.uniform(0.01, 5.0)), float(rng.uniform(0, 2))
                charge = float(rng.uniform(0, 0.5)) if side == SHORT else 0.0
                a.open(side, qty, price, fee, now, trade_id=1, open_charge=charge)
                shadow_cash -= fee + charge
                shadow_side, shadow_qty, shadow_entry, shadow_last = side, qty, price, now
            elif op == 1 and a.is_open:
                fee = float(rng.uniform(0, 2))
                reason = str(rng.choice(["signal", "stop", "flip", "death"]))
                if shadow_side == SHORT:
                    shadow_cash -= funding_rate / 1e4 * shadow_qty * shadow_entry * (now - shadow_last) / HOUR
                a.close(price, fee, now, reason, funding_rate)
                shadow_cash += shadow_side * shadow_qty * (price - shadow_entry) - fee
            elif op == 2 and a.is_open:
                paid = a.accrue_funding(now, funding_rate)
                if shadow_side == SHORT:
                    expected = funding_rate / 1e4 * shadow_qty * shadow_entry * (now - shadow_last) / HOUR
                    assert paid == pytest.approx(expected)
                    shadow_cash -= expected
                shadow_last = now
            elif op == 3:
                rent = float(rng.uniform(0, 5))
                a.charge_rent(rent)
                shadow_cash -= rent
            elif op == 4 and not a.is_open and rng.random() < 0.1:
                shadow_cash += a.clamp_cash()
            mark = price if a.is_open else None
            assert a.cash == pytest.approx(shadow_cash, abs=1e-6)
            assert a.invariant_errors(mark) == []
            checked += 1
    assert checked > 50_000
