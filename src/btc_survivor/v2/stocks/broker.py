"""Paper fills for stock orders on real prints by PLAN_V2 section 5.1, quantities, rejections, the short sale
restriction of section 5.3 and the split guard. Never kinder than reality, and never talks to a broker or exchange.

A decision at B takes the home symbol's reference price at B. The order is due at B plus latency_ms plus a uniform
jitter up to latency_jitter_ms from the bot's own generator, and fills at the first eligible print of its symbol
received at or after the due time with vendor time at or after B, moved against the bot by the symbol's allowance. A
flip fills both legs at one print, bots on one symbol share a print, and paper orders never move the market. Fees fall
only on sales, and borrow accrues through the version 1 funding accrual.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from btc_survivor.account import Account
from btc_survivor.types import SHORT, ClosedTrade, Leg
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockConfig, StockCostsConfig, TradingConfig
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks.finnhub import Print, PrintBatch
from btc_survivor.v2.stocks.prints import ELIGIBLE, PriceRules, eligibility, in_session, is_replayed

# Why a whole order is rejected before it fills.
SYMBOL_STALE, DISCONNECTED, SESSION_CLOSED = "symbol_not_healthy", "disconnected", "session_closed"
# Why one leg is rejected when it would fill.
NO_POSITION, ALREADY_OPEN, ZERO_QUANTITY = "no_position", "already_open", "zero_quantity"
# The feed event kind written when the split guard starts.
SPLIT_GUARD = "split_guard"


@dataclass(slots=True)
class StockOrder:
    """A market order on one symbol. A close leg always uses the quantity held when it fills."""

    bot_id: int
    symbol: str
    legs: list[Leg]
    decided_ms: int
    due_ms: int
    ref_price: float
    reason: str  # decision, stop, death, session_close, session_flatten, gap_flatten, restart_flatten, copy_exit
    forced: bool = False

    @property
    def latency_ms(self) -> int:
        return self.due_ms - self.decided_ms


@dataclass(frozen=True, slots=True)
class StockFill:
    """One fills row, the version 1 columns with book_avg_price and levels null, plus the stock columns."""

    bot_id: int
    trade_id: int | None
    leg: str
    taker_side: str
    qty: float
    ref_mid: float
    decided_ms: int
    filled_ms: int | None
    book_avg_price: float | None
    price: float | None
    levels: int | None
    slippage_bps: float | None
    fee: float | None
    latency_ms: int
    status: str  # filled or rejected
    reason: str
    symbol: str
    print_price: float | None
    allowance_bps: float
    notional: float | None
    sec_fee: float | None
    taf_fee: float | None

    def stock_columns(self) -> dict[str, float | str | None]:
        """The columns a stock fill adds to the version 1 row, as keyword arguments for records.fill_row."""
        return {"symbol": self.symbol, "print_price": self.print_price, "allowance_bps": self.allowance_bps,
                "notional": self.notional, "sec_fee": self.sec_fee, "taf_fee": self.taf_fee}


@dataclass
class StockExecution:
    """What happened when an order settled."""

    fills: list[StockFill] = field(default_factory=list)
    closed: ClosedTrade | None = None
    opened_trade_id: int | None = None
    rejected: str | None = None


def fill_prints(batch: PrintBatch, session: Session | None, rules: PriceRules,
                connected_ms: int | None) -> tuple[Print, ...]:
    """The prints of a batch that may fill an order, eligible, not replayed and in the regular session by both receive
    and vendor time, so a closing auction or post market print never fills. Computed once per batch."""
    if session is None:
        return ()
    return tuple(p for p in batch.prints
                 if not is_replayed(p.vendor_ms, connected_ms) and in_session(session, batch.recv_ms, p.vendor_ms)
                 and eligibility(batch.recv_ms, p.vendor_ms, p.conditions, rules) == ELIGIBLE)


def first_fill(order: StockOrder, recv_ms: int, prints: Sequence[Print]) -> Print | None:
    """The first print of the order's symbol with vendor time at or after the decision, when the batch was received
    at or after the due time, exactly at due included."""
    if recv_ms < order.due_ms:
        return None
    return next((p for p in prints if p.symbol == order.symbol and p.vendor_ms >= order.decided_ms), None)


def rejection(symbol_healthy: bool, connected: bool, session_open: bool) -> str | None:
    """Why a pending order must be rejected now, the session end first, then a dropped connection, then a stale
    symbol, or None while it may still fill."""
    if not session_open:
        return SESSION_CLOSED
    if not connected:
        return DISCONNECTED
    if not symbol_healthy:
        return SYMBOL_STALE
    return None


def floor_to(value: float, decimals: int) -> float:
    """Round down to decimals places, ignoring float noise far below the last place."""
    scale = 10 ** decimals
    return math.floor(round(value * scale, 6)) / scale


class StockBroker:
    """Creates orders with seeded latency and settles them on one print at or after their due time."""

    def __init__(self, cost_cfg: StockCostsConfig, trading: TradingConfig) -> None:
        self.costs = cost_cfg
        self.trading = trading
        self.borrow_bps_per_hour = costs.borrow_bps_per_hour(cost_cfg)
        self.next_trade_id = 1

    @classmethod
    def from_config(cls, cfg: StockConfig) -> StockBroker:
        return cls(cfg.stock_costs, cfg.trading)

    def open_qty(self, side: int, ref_price: float) -> float:
        """A long buys the notional rounded down to qty_decimals. A short sells the nearest whole number of shares,
        halves rounding up, with a one share minimum, because brokers do not short fractional shares."""
        raw = self.trading.notional / ref_price
        if side == SHORT and self.trading.short_whole_shares:
            return float(max(1, math.floor(raw + 0.5)))
        return floor_to(raw, self.trading.qty_decimals)

    def make_order(self, bot_id: int, symbol: str, legs: list[Leg], decided_ms: int, ref_price: float, reason: str,
                   rng: np.random.Generator, forced: bool = False) -> StockOrder:
        """Latency is latency_ms plus a jitter drawn uniformly from 0 to latency_jitter_ms with the bot's generator."""
        jitter = int(rng.integers(0, self.costs.latency_jitter_ms + 1))
        return StockOrder(bot_id, symbol, legs, decided_ms, decided_ms + self.costs.latency_ms + jitter, ref_price,
                          reason, forced)

    def execute(self, order: StockOrder, fill: Print, now_ms: int, account: Account) -> StockExecution:
        """Fill each leg in order at the one print. A rejected leg stops the order, keeping earlier legs."""
        if now_ms < order.due_ms:
            raise ValueError("order is not due yet")
        if fill.symbol != order.symbol or fill.vendor_ms < order.decided_ms:
            raise ValueError("the print cannot fill this order")
        result = StockExecution()
        has_open = any(leg.action == "open" for leg in order.legs)
        allowance = costs.allowance_bps(self.costs, order.symbol)
        for leg in order.legs:
            qty = account.qty if leg.action == "close" else leg.qty
            reject = _check(leg, account)
            if reject is not None:
                result.fills.append(self._rejected(order, leg, qty, reject, account.trade_id))
                result.rejected = reject
                break
            taker = leg.taker_side
            price = costs.fill_price(fill.price, taker, allowance)
            sale = costs.leg_fees(self.costs, taker, qty, price)
            fee = self.costs.commission_per_order + sale.total
            sign = 1.0 if taker == costs.BUY else -1.0
            slippage = sign * (price - order.ref_price) / order.ref_price * 1e4
            if leg.action == "close":
                trade_id = account.trade_id
                result.closed = account.close(price, fee, now_ms, _exit_reason(order.reason, has_open),
                                              self.borrow_bps_per_hour, slippage_bps=slippage)
            else:
                trade_id = self.next_trade_id
                self.next_trade_id += 1
                account.open(leg.position_side, qty, price, fee, now_ms, trade_id, slippage_bps=slippage,
                             forced=order.forced)
                result.opened_trade_id = trade_id
            result.fills.append(StockFill(
                bot_id=order.bot_id, trade_id=trade_id, leg=leg.action, taker_side=taker, qty=qty,
                ref_mid=order.ref_price, decided_ms=order.decided_ms, filled_ms=now_ms, book_avg_price=None,
                price=price, levels=None, slippage_bps=slippage, fee=fee, latency_ms=order.latency_ms,
                status="filled", reason=order.reason, symbol=order.symbol, print_price=fill.price,
                allowance_bps=allowance, notional=qty * price, sec_fee=sale.sec, taf_fee=sale.taf,
            ))
        return result

    def reject(self, order: StockOrder, reason: str, account: Account) -> StockExecution:
        """Reject a whole order without a print, for a stale symbol, a dropped connection or the session end."""
        leg = order.legs[0]
        qty = account.qty if leg.action == "close" else leg.qty
        return StockExecution(fills=[self._rejected(order, leg, qty, reason, account.trade_id)], rejected=reason)

    def _rejected(self, order: StockOrder, leg: Leg, qty: float, reason: str, trade_id: int | None) -> StockFill:
        return StockFill(
            bot_id=order.bot_id, trade_id=trade_id if leg.action == "close" else None, leg=leg.action,
            taker_side=leg.taker_side, qty=qty, ref_mid=order.ref_price, decided_ms=order.decided_ms,
            filled_ms=None, book_avg_price=None, price=None, levels=None, slippage_bps=None, fee=None,
            latency_ms=order.latency_ms, status="rejected", reason=reason, symbol=order.symbol, print_price=None,
            allowance_bps=costs.allowance_bps(self.costs, order.symbol), notional=None, sec_fee=None, taf_fee=None,
        )


def _check(leg: Leg, account: Account) -> str | None:
    if leg.action == "close" and (not account.is_open or account.side != leg.position_side):
        return NO_POSITION
    if leg.action == "open" and (account.is_open or leg.qty <= 0):
        return ALREADY_OPEN if account.is_open else ZERO_QUANTITY
    return None


def _exit_reason(order_reason: str, has_open: bool) -> str:
    """A decision closes as a flip or a signal, and every other order closes under its own reason."""
    if order_reason == "decision":
        return "flip" if has_open else "signal"
    return order_reason


@dataclass
class GuardState:
    """One symbol's guard state, plain values so a checkpoint can carry it."""

    day: str | None = None                   # the session date of the last healthy reference price
    last_price: float | None = None          # the last healthy reference price of that session
    prev_close: float | None = None          # the last healthy reference price of the session before today's
    restricted_through: str | None = None    # the last session date on which new short opens are refused
    split_k: int | None = None               # the first fresh minute after an overnight move above the limit


class SymbolGuards:
    """The short sale restriction and the split guard for every symbol, fed each healthy reference price in order.

    A reference price 10 percent or more below the previous session's last healthy reference price refuses new short
    opens for the rest of the session and the next, as SEC Rule 201 does, and cannot trigger without that price. An
    overnight move above max_overnight_gap_pct keeps the symbol's features invalid until 391 fresh session minutes
    exist.
    """

    def __init__(self, symbols: Sequence[str], drop_pct: float, max_gap_pct: float, history_minutes: int) -> None:
        self.drop = drop_pct / 100.0
        self.max_gap = max_gap_pct / 100.0
        self.history_minutes = history_minutes
        self.state = {symbol: GuardState() for symbol in symbols}

    @classmethod
    def from_config(cls, cfg: StockConfig) -> SymbolGuards:
        return cls(cfg.stock_feed.symbols, cfg.stock_costs.short_sale_restriction_drop_pct,
                   cfg.session.max_overnight_gap_pct, cfg.time.history_session_minutes)

    def on_price(self, symbol: str, session: Session, k: int, ref_price: float) -> list[tuple[str, str]]:
        """Take one healthy reference price at minute k of a session, returning any split_guard event as
        (kind, detail)."""
        s = self.state[symbol]
        events: list[tuple[str, str]] = []
        today = session.day.isoformat()
        if s.day != today:
            s.prev_close = s.last_price if s.day is not None and s.day == _previous_day(session) else None
            if s.prev_close is not None and abs(ref_price / s.prev_close - 1.0) > self.max_gap:
                s.split_k = k
                events.append((SPLIT_GUARD, f"{symbol} moved {100 * (ref_price / s.prev_close - 1):+.1f} percent "
                                            f"overnight, features wait {self.history_minutes} fresh session minutes"))
            s.day = today
        s.last_price = ref_price
        if s.prev_close is not None and ref_price <= s.prev_close * (1.0 - self.drop):
            s.restricted_through = sessions.next_session(session.close_ms).day.isoformat()
        return events

    def shorts_allowed(self, symbol: str, day: date) -> bool:
        """Whether a new short may open in the symbol on a session date."""
        through = self.state[symbol].restricted_through
        return through is None or day.isoformat() > through

    def features_allowed(self, symbol: str, k: int) -> bool:
        """Whether the split guard lets the symbol's features count at minute k."""
        split_k = self.state[symbol].split_k
        return split_k is None or k >= split_k + self.history_minutes - 1


def _previous_day(session: Session) -> str | None:
    """The date of the session before this one in the table, or None for the first."""
    if session.first_k == 1:
        return None
    return sessions.session_of_k(session.first_k - 1).day.isoformat()
