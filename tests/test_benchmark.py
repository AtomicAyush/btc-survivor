import pytest

from btc_survivor.benchmark import Anchor, buy_hold_equity, cash_equity, matched_equity

ANCHOR = Anchor(ms=0, ask=80_010.0, mid=80_000.0)


def test_buy_and_hold_pays_one_fee_at_the_launch_ask_and_marks_at_the_bid():
    qty = 10_000 * (1 - 0.009) / 80_010.0
    assert buy_hold_equity(ANCHOR, 88_000.0, 10_000.0, 90.0) == pytest.approx(qty * 88_000.0)
    assert buy_hold_equity(ANCHOR, 80_010.0, 10_000.0, 0.0) == pytest.approx(10_000.0)


def test_matched_curve_only_holds_the_notional():
    qty = 500 * (1 - 0.009) / 80_010.0
    assert matched_equity(ANCHOR, 72_000.0, 10_000.0, 500.0, 90.0) == pytest.approx(9_500.0 + qty * 72_000.0)


def test_cash_baseline_pays_rent_only_for_healthy_minutes():
    assert cash_equity(10_000.0, 1.44, 0) == 10_000.0
    assert cash_equity(10_000.0, 1.44, 1440) == pytest.approx(10_000.0 - 1.44)
    assert cash_equity(500.0, 1.16, 720) == pytest.approx(500.0 - 0.58)
