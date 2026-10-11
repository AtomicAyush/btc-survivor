"""The books for one bot. Cash is a ledger of its own, so the invariant checks against the totals are real checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from btc_survivor.clock import MS_PER_HOUR
from btc_survivor.types import FLAT, LONG, SHORT, ClosedTrade

INVARIANT_TOLERANCE = 1e-6
CAUSES = ("fees", "funding", "rent", "stop_losses", "trading_losses")


@dataclass
class Account:
    start_equity: float
    cash: float = field(default=None)  # type: ignore[assignment]
    side: int = FLAT
    qty: float = 0.0
    entry_price: float = 0.0
    entry_fee: float = 0.0
    entry_slippage_bps: float = 0.0
    entry_ms: int = 0
    trade_id: int | None = None
    forced: bool = False
    position_funding: float = 0.0
    last_funding_ms: int = 0
    realized_gross: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    rent: float = 0.0
    stop_loss_losses: float = 0.0
    other_losses: float = 0.0
    writeoff: float = 0.0
    trades: int = 0
    wins: int = 0

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = self.start_equity

    @property
    def is_open(self) -> bool:
        return self.side != FLAT

    def open(self, side: int, qty: float, price: float, fee: float, ms: int, trade_id: int,
             slippage_bps: float = 0.0, open_charge: float = 0.0, forced: bool = False) -> None:
        """Open a position. Only the fee and any short opening charge leave cash; the charge counts as funding."""
        if self.is_open:
            raise ValueError("account already holds a position")
        if side not in (LONG, SHORT) or qty <= 0 or price <= 0:
            raise ValueError("invalid open")
        self.cash -= fee + open_charge
        self.fees += fee
        self.funding += open_charge
        self.side, self.qty, self.entry_price = side, qty, price
        self.entry_fee, self.entry_slippage_bps, self.entry_ms = fee, slippage_bps, ms
        self.trade_id, self.forced = trade_id, forced
        self.position_funding = open_charge
        self.last_funding_ms = ms

    def accrue_funding(self, now_ms: int, bps_per_hour: float) -> float:
        """Charge short funding on the entry notional for the time since the last accrual."""
        if self.side != SHORT or now_ms <= self.last_funding_ms:
            self.last_funding_ms = max(self.last_funding_ms, now_ms)
            return 0.0
        hours = (now_ms - self.last_funding_ms) / MS_PER_HOUR
        amount = bps_per_hour / 1e4 * self.qty * self.entry_price * hours
        self.cash -= amount
        self.funding += amount
        self.position_funding += amount
        self.last_funding_ms = now_ms
        return amount

    def close(self, price: float, fee: float, ms: int, exit_reason: str, bps_per_hour: float,
              slippage_bps: float = 0.0) -> ClosedTrade:
        """Close the whole position at price, settling gross profit and the closing fee into cash."""
        if not self.is_open:
            raise ValueError("no position to close")
        self.accrue_funding(ms, bps_per_hour)
        gross = self.side * self.qty * (price - self.entry_price)
        self.cash += gross - fee
        self.realized_gross += gross
        self.fees += fee
        if exit_reason == "stop":
            self.stop_loss_losses -= gross
        else:
            self.other_losses -= gross
        net = gross - self.entry_fee - fee - self.position_funding
        self.trades += 1
        if net > 0:
            self.wins += 1
        closed = ClosedTrade(
            trade_id=self.trade_id, side=self.side, qty=self.qty, entry_price=self.entry_price, exit_price=price,
            opened_ms=self.entry_ms, closed_ms=ms, fee_open=self.entry_fee, fee_close=fee,
            funding=self.position_funding, gross=gross, net=net, slippage_bps_open=self.entry_slippage_bps,
            slippage_bps_close=slippage_bps, exit_reason=exit_reason, forced=self.forced,
        )
        self.side, self.qty, self.entry_price, self.entry_fee = FLAT, 0.0, 0.0, 0.0
        self.entry_slippage_bps, self.trade_id, self.forced, self.position_funding = 0.0, None, False, 0.0
        return closed

    def charge_rent(self, amount: float) -> None:
        self.cash -= amount
        self.rent += amount

    def unrealized(self, mark_price: float | None) -> float:
        if not self.is_open:
            return 0.0
        if mark_price is None:
            raise ValueError("an open position needs a mark price")
        return self.side * self.qty * (mark_price - self.entry_price)

    def equity(self, mark_price: float | None) -> float:
        """Cash plus unrealized profit, with the position marked at the exit side price."""
        return self.cash + self.unrealized(mark_price)

    def clamp_cash(self) -> float:
        """At death, turn any negative cash into a recorded writeoff so the books still balance."""
        if self.cash < 0:
            amount = -self.cash
            self.writeoff += amount
            self.cash = 0.0
            return amount
        return 0.0

    def breakdown(self) -> dict[str, float]:
        """The five parts of the loss, which add up to start equity minus equity when flat and before writeoff."""
        return {
            "fees": self.fees,
            "funding": self.funding,
            "rent": self.rent,
            "stop_losses": self.stop_loss_losses,
            "trading_losses": self.other_losses,
        }

    def death_cause(self) -> str:
        parts = self.breakdown()
        return max(CAUSES, key=lambda c: parts[c])

    def invariant_errors(self, mark_price: float | None, tol: float = INVARIANT_TOLERANCE) -> list[str]:
        """Check the three balance identities from the plan within an absolute tolerance in dollars."""
        errors = []
        expected_cash = self.start_equity + self.realized_gross - self.fees - self.funding - self.rent + self.writeoff
        if abs(self.cash - expected_cash) > tol:
            errors.append(f"cash {self.cash!r} differs from ledger total {expected_cash!r}")
        unrealized = self.unrealized(mark_price)
        equity = self.equity(mark_price)
        if abs(equity - (self.cash + unrealized)) > tol:
            errors.append("equity differs from cash plus unrealized")
        losses = (self.fees + self.funding + self.rent + self.stop_loss_losses + self.other_losses
                  - self.writeoff - unrealized)
        if abs((self.start_equity - equity) - losses) > tol:
            errors.append(f"loss breakdown {losses!r} differs from start minus equity {self.start_equity - equity!r}")
        if abs(self.realized_gross + self.stop_loss_losses + self.other_losses) > tol:
            errors.append("realized gross differs from the negated stop and trading losses")
        return errors


def death_threshold(round_trip_cost_bps: float, notional: float, daily_rent: float) -> float:
    """Equity below this means the bot cannot pay for one more round trip and one more day."""
    return round_trip_cost_bps / 1e4 * notional + daily_rent
