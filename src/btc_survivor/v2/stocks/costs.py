"""Stock trading costs by PLAN_V2 sections 5.1 to 5.4, the per symbol allowance, the regulatory fees on sales, borrow,
the round trip estimate behind the gate and the death threshold, the required edge, the holding cost a held position
must beat, rent and the Roll diagnostic.

There is no commission. The SEC Section 31 fee and the FINRA trading activity fee fall only on sales, the closing leg
of a long and the opening leg of a short, and each rounds up to the next cent per sale with a minimum of one cent.
Borrow accrues on a short's entry notional over calendar time through the version 1 funding accrual. Every figure here
is fixed before launch and never tuned, and the Roll diagnostic only reports.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

from btc_survivor.account import death_threshold
from btc_survivor.clock import MS_PER_HOUR
from btc_survivor.types import LONG, SHORT
from btc_survivor.v2.config import StockCostsConfig
from btc_survivor.v2.sessions import FULL_SESSION_MINUTES

CENT = 0.01
HOURS_PER_YEAR = 365 * 24
BUY, SELL = "buy", "sell"


def allowance_bps(costs: StockCostsConfig, symbol: str) -> float:
    """The per leg allowance in bps, the symbol's half spread plus the extra slippage."""
    return costs.half_spread_bps[symbol] + costs.extra_slippage_bps


def fill_price(print_price: float, taker_side: str, allowance: float) -> float:
    """A print moved against the bot by the allowance, up for a buy and down for a sell."""
    sign = 1.0 if taker_side == BUY else -1.0
    return print_price * (1.0 + sign * allowance / 1e4)


def mark_price(ref_price: float, side: int, allowance: float) -> float:
    """The reference price moved to the exit side by the allowance, a sell for a long and a buy for a short."""
    return fill_price(ref_price, SELL if side == LONG else BUY, allowance)


def round_fee(amount: float, round_up: bool = True) -> float:
    """A fee rounded up to the next cent with a minimum of one cent, or left as it is when rounding is off.

    The amount is first rounded to a millionth of a cent, so a float just above a whole cent never adds a cent.
    """
    if amount <= 0.0 or not round_up:
        return max(amount, 0.0)
    return max(CENT, math.ceil(round(amount * 100.0, 6)) / 100.0)


def sec_fee(costs: StockCostsConfig, proceeds: float) -> float:
    """The SEC Section 31 fee on the proceeds of one sale."""
    return round_fee(proceeds * costs.sec_fee_per_million / 1e6, costs.fee_round_up_to_cent)


def taf_fee(costs: StockCostsConfig, shares: float) -> float:
    """The FINRA trading activity fee on the shares of one sale, capped per trade before rounding."""
    raw = min(shares * costs.finra_taf_per_share, costs.finra_taf_max_per_trade)
    return round_fee(raw, costs.fee_round_up_to_cent)


class SaleFees(NamedTuple):
    sec: float
    taf: float

    @property
    def total(self) -> float:
        return self.sec + self.taf


NO_FEES = SaleFees(0.0, 0.0)


def leg_fees(costs: StockCostsConfig, taker_side: str, qty: float, price: float) -> SaleFees:
    """The regulatory fees of one leg, which only a sale pays."""
    if taker_side != SELL:
        return NO_FEES
    return SaleFees(sec_fee(costs, qty * price), taf_fee(costs, qty))


def borrow_bps_per_hour(costs: StockCostsConfig) -> float:
    """The yearly borrow rate in bps of entry notional per calendar hour, about 0.0114 at 1 percent."""
    return costs.short_borrow_pct_per_year * 100.0 / HOURS_PER_YEAR


def borrow_cost(costs: StockCostsConfig, notional: float, from_ms: int, to_ms: int) -> float:
    """Dollars of borrow on a short's entry notional between two instants, nights and weekends included."""
    return borrow_bps_per_hour(costs) / 1e4 * notional * max(0, to_ms - from_ms) / MS_PER_HOUR


class RoundTrip(NamedTuple):
    """A round trip estimate in bps of its notional, two legs of allowance, the fees of its one sale and borrow."""

    legs_bps: float
    fees_bps: float
    borrow_bps: float
    notional: float

    @property
    def total_bps(self) -> float:
        return self.legs_bps + self.fees_bps + self.borrow_bps

    @property
    def dollars(self) -> float:
        return self.total_bps / 1e4 * self.notional


def holding_cost_bps(costs: StockCostsConfig, side: int, horizon_minutes: int) -> float:
    """The cost in bps of keeping a position over one more horizon, nothing for a long and the borrow over the horizon
    for a short, about 0.0057 bps at 1 percent a year over 30 minutes. Rent is paid either way."""
    if side not in (LONG, SHORT):
        raise ValueError("a holding cost needs a side")
    return borrow_bps_per_hour(costs) * horizon_minutes / 60.0 if side == SHORT else 0.0


def round_trip(costs: StockCostsConfig, symbol: str, side: int, qty: float, price: float,
               horizon_minutes: int) -> RoundTrip:
    """The cost of opening and closing qty shares on side at price, with borrow over the horizon for a short."""
    if side not in (LONG, SHORT) or qty <= 0 or price <= 0:
        raise ValueError("a round trip needs a side, a positive quantity and a positive price")
    notional = qty * price
    fees = sec_fee(costs, notional) + taf_fee(costs, qty)
    borrow = holding_cost_bps(costs, side, horizon_minutes)
    return RoundTrip(2.0 * allowance_bps(costs, symbol), fees / notional * 1e4, borrow, notional)


def required_edge_bps(cost_bps: float, margin_sigma: float, label_sd_bps: float) -> float:
    """The edge a prediction must clear, the round trip cost plus the margin in label standard deviations."""
    return cost_bps + margin_sigma * label_sd_bps


def rent_per_minute(costs: StockCostsConfig) -> float:
    """The daily cost of capital spread over usable session minutes, 0.0043081 dollars at 1.16 a day."""
    return costs.daily_running_cost * 365 / (costs.trading_days_per_year * FULL_SESSION_MINUTES)


def session_rent(costs: StockCostsConfig) -> float:
    """One full session of rent, 1.68 dollars, the rent part of the death threshold."""
    return rent_per_minute(costs) * FULL_SESSION_MINUTES


def death_line(costs: StockCostsConfig, symbol: str, side: int, qty: float, price: float,
               horizon_minutes: int) -> float:
    """Equity below this means death, the round trip of the position on its actual notional plus a session of rent."""
    trip = round_trip(costs, symbol, side, qty, price, horizon_minutes)
    return death_threshold(trip.total_bps, trip.notional, session_rent(costs))


class RollAccumulator:
    """Running sums for Roll's estimate over one symbol's consecutive eligible print prices in one session, so a day
    of prints never has to be held in memory."""

    def __init__(self) -> None:
        self.prints = 0
        self.pairs = 0
        self._last_price: float | None = None
        self._last_change: float | None = None
        self._sum_now = self._sum_before = self._sum_product = 0.0

    def add(self, price: float) -> None:
        """Take the next eligible print price, as a log change in bps paired with the change before it."""
        self.prints += 1
        if self._last_price is not None:
            change = 1e4 * math.log(price / self._last_price)
            if self._last_change is not None:
                self.pairs += 1
                self._sum_now += change
                self._sum_before += self._last_change
                self._sum_product += change * self._last_change
            self._last_change = change
        self._last_price = price

    def half_spread_bps(self) -> float | None:
        """sqrt(-cov(dp_t, dp_t-1)) in bps, or None with fewer than two pairs or a covariance that is not negative."""
        if self.pairs < 2:
            return None
        n = self.pairs
        cov = (self._sum_product - self._sum_now * self._sum_before / n) / (n - 1)
        return math.sqrt(-cov) if cov < 0 else None


def roll_half_spread_bps(prices: Sequence[float]) -> float | None:
    """Roll's effective half spread in bps from a sequence of consecutive eligible print prices."""
    acc = RollAccumulator()
    for price in prices:
        acc.add(price)
    return acc.half_spread_bps()


class SpreadCheck(NamedTuple):
    """One spread_checks row, the Roll estimate against the symbol's half spread allowance. Report only."""

    symbol: str
    prints: int
    estimate_bps: float | None
    allowance_bps: float
    flagged: bool


def spread_check(costs: StockCostsConfig, symbol: str, acc: RollAccumulator) -> SpreadCheck:
    """The day's Roll estimate for a symbol, flagged when above its half spread allowance. It never changes costs."""
    estimate = acc.half_spread_bps()
    allowance = costs.half_spread_bps[symbol]
    return SpreadCheck(symbol, acc.prints, estimate, allowance, estimate is not None and estimate > allowance)
