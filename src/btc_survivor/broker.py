"""Simulated market orders against the local book. Never kinder than reality, and never talks to an exchange."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from btc_survivor.account import Account
from btc_survivor.config import CostsConfig
from btc_survivor.feed.book import LocalBook
from btc_survivor.types import LONG, SHORT, ClosedTrade, Fill, Leg, Order

QTY_DECIMALS = 8


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Round trip cost in bps and its parts, as used for the required edge and the death threshold."""

    fee_bps: float
    spread_bps: float
    impact_bps: float
    slippage_bps: float
    short_bps: float

    @property
    def total_bps(self) -> float:
        return 2 * self.fee_bps + self.spread_bps + 2 * self.impact_bps + 2 * self.slippage_bps + self.short_bps


def estimate_round_trip(book: LocalBook, notional: float, costs: CostsConfig, side: int,
                        horizon_minutes: int) -> CostEstimate | None:
    """Estimate the cost of opening and closing the fixed notional on side, or None if the book cannot fill it."""
    mid, bid, ask = book.mid, book.best_bid, book.best_ask
    if not book.valid or mid is None:
        return None
    qty = notional / mid
    walk = book.walk("buy" if side == LONG else "sell", qty)
    if walk.unfilled_qty > 0 or walk.avg_price is None:
        return None
    impact = (walk.avg_price - ask if side == LONG else bid - walk.avg_price) / mid * 1e4
    short_bps = 0.0
    if side == SHORT:
        short_bps = costs.short_open_fee_bps + costs.short_funding_bps_per_hour * horizon_minutes / 60.0
    return CostEstimate(
        fee_bps=costs.taker_fee_bps,
        spread_bps=book.spread_bps,
        impact_bps=max(impact, 0.0),
        slippage_bps=costs.extra_slippage_bps,
        short_bps=short_bps,
    )


@dataclass
class Execution:
    """What happened when an order was settled."""

    fills: list[Fill] = field(default_factory=list)
    closed: ClosedTrade | None = None
    opened_trade_id: int | None = None
    rejected: str | None = None


class Broker:
    """Creates orders with seeded latency and settles them against the book at or after their due time."""

    def __init__(self, costs: CostsConfig, notional: float) -> None:
        self.costs = costs
        self.notional = notional
        self.next_trade_id = 1

    def open_qty(self, ref_mid: float) -> float:
        return round(self.notional / ref_mid, QTY_DECIMALS)

    def make_order(self, bot_id: int, legs: list[Leg], decided_ms: int, ref_mid: float, reason: str,
                   rng: np.random.Generator, forced: bool = False) -> Order:
        """Latency is latency_ms plus a jitter drawn uniformly from 0 to latency_jitter_ms with the bot's generator."""
        jitter = int(rng.integers(0, self.costs.latency_jitter_ms + 1))
        due = decided_ms + self.costs.latency_ms + jitter
        return Order(bot_id=bot_id, legs=legs, decided_ms=decided_ms, due_ms=due, ref_mid=ref_mid,
                     reason=reason, forced=forced)

    def execute(self, order: Order, book: LocalBook, now_ms: int, account: Account) -> Execution:
        """Fill each leg in order by walking the book. A rejected leg stops the order, keeping earlier legs.

        Legs on the same side share the visible depth, so the second leg of a flip pays the deeper levels.
        """
        if now_ms < order.due_ms:
            raise ValueError("order is not due yet")
        result = Execution()
        has_open = any(leg.action == "open" for leg in order.legs)
        taken = {"buy": 0.0, "sell": 0.0}
        for leg in order.legs:
            reject = self._check(leg, book, account)
            qty = account.qty if leg.action == "close" else leg.qty
            walk = None
            if reject is None:
                before = taken[leg.taker_side]
                walk = book.walk(leg.taker_side, before + qty)
                if walk.unfilled_qty > 0 or walk.avg_price is None:
                    reject = "insufficient_depth"
            if reject is not None:
                result.fills.append(self._rejected(order, leg, qty, reject, account.trade_id))
                result.rejected = reject
                break
            book_avg = walk.avg_price
            if before > 0:
                book_avg = (walk.avg_price * (before + qty) - book.walk(leg.taker_side, before).avg_price * before) / qty
            taken[leg.taker_side] = before + qty
            sign = 1.0 if leg.taker_side == "buy" else -1.0
            price = book_avg * (1.0 + sign * self.costs.extra_slippage_bps / 1e4)
            fee = self.costs.taker_fee_bps / 1e4 * qty * price
            slippage = sign * (price - order.ref_mid) / order.ref_mid * 1e4
            if leg.action == "close":
                trade_id = account.trade_id
                exit_reason = _exit_reason(order.reason, has_open)
                result.closed = account.close(price, fee, now_ms, exit_reason,
                                              self.costs.short_funding_bps_per_hour, slippage_bps=slippage)
            else:
                trade_id = self.next_trade_id
                self.next_trade_id += 1
                charge = self.costs.short_open_fee_bps / 1e4 * qty * price if leg.position_side == SHORT else 0.0
                account.open(leg.position_side, qty, price, fee, now_ms, trade_id, slippage_bps=slippage,
                             open_charge=charge, forced=order.forced)
                result.opened_trade_id = trade_id
            result.fills.append(Fill(
                bot_id=order.bot_id, trade_id=trade_id, leg=leg.action, taker_side=leg.taker_side, qty=qty,
                ref_mid=order.ref_mid, decided_ms=order.decided_ms, filled_ms=now_ms,
                book_avg_price=book_avg, price=price, levels=walk.levels, slippage_bps=slippage, fee=fee,
                latency_ms=order.latency_ms, status="filled", reason=order.reason,
            ))
        return result

    def reject(self, order: Order, reason: str, account: Account) -> Execution:
        """Reject a whole order without touching the book, for example when the feed left Healthy."""
        leg = order.legs[0]
        qty = account.qty if leg.action == "close" else leg.qty
        return Execution(fills=[self._rejected(order, leg, qty, reason, account.trade_id)], rejected=reason)

    @staticmethod
    def _check(leg: Leg, book: LocalBook, account: Account) -> str | None:
        if not book.valid:
            return "book_invalid"
        if leg.action == "close" and (not account.is_open or account.side != leg.position_side):
            return "no_position"
        if leg.action == "open" and (account.is_open or leg.qty <= 0):
            return "already_open" if account.is_open else "zero_quantity"
        return None

    @staticmethod
    def _rejected(order: Order, leg: Leg, qty: float, reason: str, trade_id: int | None) -> Fill:
        return Fill(
            bot_id=order.bot_id, trade_id=trade_id if leg.action == "close" else None, leg=leg.action,
            taker_side=leg.taker_side, qty=qty, ref_mid=order.ref_mid, decided_ms=order.decided_ms, filled_ms=None,
            book_avg_price=None, price=None, levels=None, slippage_bps=None, fee=None,
            latency_ms=order.latency_ms, status="rejected", reason=reason,
        )


def _exit_reason(order_reason: str, has_open: bool) -> str:
    if order_reason == "decision":
        return "flip" if has_open else "signal"
    return order_reason
