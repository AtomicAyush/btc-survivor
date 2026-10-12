"""One pooled stock learner by PLAN_V2 sections 4.7, 4.8, 5.4, 6.4 and 10.2, with the Warmup, Active and Dead states,
its minute step and the version 1 decision rule.

A learner predicts all eight symbols every usable minute in one forward pass, learns from every symbol, and trades only
its home symbol, the one at position slot % 8 in config order, deciding at the session minutes where k % 30 equals
slot % 30. Rent, the forced timer, training, predictions and decisions need a usable minute, price health plus news
health, while marking, borrow, death and the stop need only the home symbol healthy. Opens, flips and forced opens are
allowed from session minute 30 to 35 minutes before the close, and the engine sends every open position a closing order
from 5 minutes before the close. The margin and the stop are in label standard deviations of the symbol, and the stop
is fixed at entry. The forced timer counts usable minutes from the first Active minute, started up to 180 minutes early
by a delay drawn at birth, so first forced trades spread over the 3 hours after warmup ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from btc_survivor.account import Account
from btc_survivor.bot import ACTIVE, DEAD, POSITION_NAMES, WARMUP, decide
from btc_survivor.model import MLP, AdamW
from btc_survivor.seeds import bot_rng
from btc_survivor.skill import RollingSkill
from btc_survivor.types import LONG, SHORT, Leg
from btc_survivor.v2.config import SessionConfig, StockConfig
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks.broker import StockBroker, StockExecution, StockOrder
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES
from btc_survivor.v2.stocks.learning import PooledWindow, predict_rows, with_news_inputs

# Exit and decision reasons added by the session rules.
SESSION_CLOSE, SESSION_FLATTEN = "session_close", "session_flatten"
SESSION_WINDOW, SHORT_SALE_RESTRICTION = "session_window", "short_sale_restriction"
OPENS = ("open_long", "open_short")


@dataclass
class StockMinute:
    """The shared market layer's output for one session boundary, computed once and handed to every bot of a run.

    Arrays run over the symbols in config order. Rows hold the 10 standardized price features and the two news inputs,
    zero where a vector is invalid, and label_sd is NaN until a symbol's label scale is ready. Matured holds
    (symbol index, push minute k, label bps) for every sample that matured at this boundary, already added to the
    window, and window is None until training may start, once every label scale is ready.
    """

    boundary_ms: int
    k: int
    session: Session
    rows: np.ndarray
    valid: np.ndarray
    healthy: np.ndarray
    news_ok: bool
    ref: np.ndarray
    label_sd: np.ndarray
    shorts_ok: np.ndarray
    matured: tuple[tuple[int, int, float], ...] = ()
    window: PooledWindow | None = None

    @property
    def session_minute(self) -> int:
        return self.session.session_minute(self.boundary_ms)

    def usable(self, symbol: int) -> bool:
        """A symbol minute is usable when the symbol is healthy and news health held."""
        return bool(self.healthy[symbol]) and self.news_ok


@dataclass
class BotStep:
    decision: dict | None = None
    orders: list[StockOrder] = field(default_factory=list)
    died: bool = False
    stopped: bool = False


def opens_allowed(session: Session, session_minute: int, rules: SessionConfig) -> bool:
    """Whether an open, a flip or a forced open may go out, 10:00 to 15:25 ET, or 12:25 on an early close."""
    last = session.minutes - rules.last_open_minutes_before_close
    return rules.first_open_session_minute <= session_minute <= last


def session_close_due(session: Session, session_minute: int, rules: SessionConfig) -> bool:
    """Whether open positions get their session_close order, from 15:55 ET, or 12:55 on an early close."""
    return session_minute >= session.minutes - rules.flatten_minutes_before_close


class StockBot:
    def __init__(self, cfg: StockConfig, bot_id: int, slot: int, born_ms: int, model: MLP,
                 rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.bot_id, self.slot, self.born_ms = bot_id, slot, born_ms
        self.home = slot % len(cfg.stock_feed.symbols)
        self.symbol = cfg.stock_feed.symbols[self.home]
        self.model, self.rng = model, rng
        self.opt = AdamW(model.size, cfg.model.adam_beta1, cfg.model.adam_beta2)
        self.account = Account(cfg.trading.start_equity)
        self.warmup_minutes = round(cfg.time.warmup_hours * 60)
        self.force_minutes = round(cfg.trading.force_trade_after_hours * 60)
        self.state = WARMUP if self.warmup_minutes > 0 else ACTIVE
        self.head_start = 0
        self.trained_minutes = 0
        self.alive_minutes = 0
        self.minutes_since_open = 0
        self.preds = np.full(len(cfg.stock_feed.symbols), np.nan)
        self.pending_preds: dict[int, float] = {}
        self.skill = RollingSkill(cfg.model.skill_window_samples)
        self.pending_order: StockOrder | None = None
        self.stop_bps: float | None = None
        self._stop_at_send: float | None = None
        self.equity = cfg.trading.start_equity
        self.death_threshold: float | None = None
        self.rent_per_minute = costs.rent_per_minute(cfg.stock_costs)
        self.borrow_bps_per_hour = costs.borrow_bps_per_hour(cfg.stock_costs)
        self.allowance = costs.allowance_bps(cfg.stock_costs, self.symbol)

    @classmethod
    def newborn(cls, cfg: StockConfig, bot_id: int, slot: int, born_ms: int) -> StockBot:
        """A learner drawn as MLP.random at 10 inputs with two zero news columns appended, then its forced timer
        delay, so S1 and S2 make the same generator calls in every slot."""
        rng = bot_rng(cfg.run.master_seed, bot_id)
        model = MLP.random(N_PRICE_FEATURES, cfg.model.hidden_units, rng, cfg.model.init_output_scale)
        bot = cls(cfg, bot_id, slot, born_ms, with_news_inputs(model, cfg.model.inputs - N_PRICE_FEATURES), rng)
        bot.head_start = int(rng.integers(0, max(1, bot.force_minutes)))
        bot.minutes_since_open = bot.head_start if bot.state == ACTIVE else 0
        return bot

    def is_phase_minute(self, k: int) -> bool:
        horizon = self.cfg.time.horizon_minutes
        return k % horizon == self.slot % horizon

    def mark(self, ref_price: float | None) -> float | None:
        """The price the position would get if closed now, the reference price moved to the exit side."""
        if not self.account.is_open or ref_price is None or math.isnan(ref_price):
            return None
        return costs.mark_price(ref_price, self.account.side, self.allowance)

    def step(self, ctx: StockMinute, broker: StockBroker) -> BotStep:
        res = BotStep()
        self.preds = np.full(len(self.preds), np.nan)
        if self.state == DEAD or self._upkeep(ctx, broker, res):
            return res
        if ctx.news_ok and ctx.matured and ctx.window is not None:
            self._learn(ctx)
        self._predict(ctx)
        pred = self.preds[self.home]
        if not math.isnan(pred) and ctx.usable(self.home) and self.is_phase_minute(ctx.k):
            res.decision = self._decide(ctx, float(pred), res, broker)
        return res

    def _upkeep(self, ctx: StockMinute, broker: StockBroker, res: BotStep) -> bool:
        """Rent and the forced timer in a usable minute, then borrow, marking, the death check and the stop while the
        home symbol is healthy. Shared with the twins. True on death."""
        acct = self.account
        if ctx.usable(self.home):
            acct.charge_rent(self.rent_per_minute)
            self.alive_minutes += 1
            if self.state == ACTIVE:
                self.minutes_since_open += 1
        if not ctx.healthy[self.home]:
            return False
        ref = float(ctx.ref[self.home])
        acct.accrue_funding(ctx.boundary_ms, self.borrow_bps_per_hour)
        mark = self.mark(ref)
        self.equity = acct.equity(mark)
        if self._check_death(ctx, ref, broker, res):
            return True
        if acct.is_open and self.pending_order is None and mark is not None and self.stop_bps is not None:
            adverse = (acct.entry_price - mark) if acct.side == LONG else (mark - acct.entry_price)
            if adverse / acct.entry_price * 1e4 >= self.stop_bps:
                self._send(broker, [Leg("close", acct.side)], ctx.boundary_ms, ref, "stop", res)
                res.stopped = True
        return False

    def _check_death(self, ctx: StockMinute, ref: float, broker: StockBroker, res: BotStep) -> bool:
        """Equity below the round trip of the held position, or of a long if flat, on its actual notional plus a
        session of rent means death. A pending close becomes the death close."""
        acct = self.account
        side = acct.side if acct.is_open else LONG
        qty = acct.qty if acct.is_open else broker.open_qty(LONG, ref)
        threshold = costs.death_line(self.cfg.stock_costs, self.symbol, side, qty, ref, self.cfg.time.horizon_minutes)
        if self.equity >= threshold:
            return False
        self.death_threshold = threshold
        self.state = DEAD
        res.died = True
        if acct.is_open and self.pending_order is None:
            self._send(broker, [Leg("close", acct.side)], ctx.boundary_ms, ref, "death", res)
        return True

    def _learn(self, ctx: StockMinute) -> None:
        """Score home predictions whose labels matured, then take one AdamW step on the newly matured samples plus
        draws from the pooled window. A batch holding a label with no scale yet trains nothing."""
        cfg = self.cfg.model
        for symbol, k, label in ctx.matured:
            pred = self.pending_preds.pop(k, None) if symbol == self.home else None
            if pred is not None:
                self.skill.add(pred, label)
        scales = dict(zip(self.cfg.stock_feed.symbols, ctx.label_sd.tolist()))
        X, y = ctx.window.batch(self.rng, len(ctx.matured), cfg.batch_size, scales)
        if not np.isfinite(y).all():
            return
        _, grad = self.model.loss_and_grad(X, y, cfg.huber_delta)
        self.opt.step(self.model, grad, cfg.learning_rate, cfg.weight_decay, cfg.grad_clip)
        self.trained_minutes += 1
        if self.state == WARMUP and self.trained_minutes >= self.warmup_minutes:
            self.state = ACTIVE
            self.minutes_since_open = self.head_start

    def _predict(self, ctx: StockMinute) -> None:
        """Predict all symbols in one pass, in bps on each symbol's label scale, NaN where a symbol has no usable
        vector or no label scale yet."""
        horizon = self.cfg.time.horizon_minutes
        self.pending_preds = {k: p for k, p in self.pending_preds.items() if ctx.k - k < horizon}
        ok = ctx.valid & ~np.isnan(ctx.label_sd) & ctx.news_ok
        if not ok.any():
            return
        out = predict_rows(self.model, np.where(ok[:, None], ctx.rows, 0.0))
        self.preds = np.where(ok, out * np.where(ok, ctx.label_sd, 0.0), np.nan)
        if ok[self.home]:
            self.pending_preds[ctx.k] = float(self.preds[self.home])

    def _decide(self, ctx: StockMinute, pred: float, res: BotStep, broker: StockBroker) -> dict:
        trading, acct, h = self.cfg.trading, self.account, self.home
        position = acct.side
        ref, label_sd = float(ctx.ref[h]), float(ctx.label_sd[h])
        side = SHORT if pred < 0 and trading.allow_short else LONG
        trip = costs.round_trip(self.cfg.stock_costs, self.symbol, side, broker.open_qty(side, ref), ref,
                                self.cfg.time.horizon_minutes)
        margin = trading.margin_sigma * label_sd
        required = costs.required_edge_bps(trip.total_bps, trading.margin_sigma, label_sd)
        forced_due = self.state == ACTIVE and self.minutes_since_open >= self.force_minutes
        action, gate, used_forced = decide(position, pred, required, trading.allow_short, forced_due)
        row = {"bot_id": self.bot_id, "ts": ctx.boundary_ms, "symbol": self.symbol,
               "position_before": POSITION_NAMES[position], "ref_mid": ref, "prediction_bps": pred,
               "required_edge_bps": required, "cost_estimate_bps": trip.total_bps, "margin_bps": margin,
               "gate_passed": int(gate), "forced": 0, "action": "none"}
        if self.state == WARMUP:
            return {**row, "reason": f"warmup, {self.trained_minutes} of {self.warmup_minutes} training minutes"}
        if res.stopped:
            return {**row, "reason": "stop loss fired this minute"}
        if self.pending_order is not None:
            return {**row, "reason": "an order is still pending"}
        reason = _reason(action, pred, required, gate, used_forced, trading.allow_short)
        opens_short = action == "open_short" or (action == "flip" and position == LONG)
        if action in (*OPENS, "flip") and not opens_allowed(ctx.session, ctx.session_minute, self.cfg.session):
            action, used_forced, reason = ("close" if action == "flip" else "none"), False, SESSION_WINDOW
        elif opens_short and not ctx.shorts_ok[h]:
            action, used_forced, reason = ("close" if action == "flip" else "none"), False, SHORT_SALE_RESTRICTION
        legs = _legs(action, position, ref, broker)
        if legs:
            self._stop_at_send = trading.stop_loss_sigma * label_sd
            self._send(broker, legs, ctx.boundary_ms, ref, "decision", res, forced=used_forced)
        return {**row, "forced": int(used_forced), "action": action, "reason": reason}

    def send_close(self, now_ms: int, ref_price: float, broker: StockBroker, reason: str) -> StockOrder | None:
        """A closing order for the open position with no order pending, for the session close at 15:55, a session
        flatten after a carry, a gap or restart flatten, or a death close resent. None when there is nothing to send."""
        if not self.account.is_open or self.pending_order is not None:
            return None
        self.pending_order = broker.make_order(self.bot_id, self.symbol, [Leg("close", self.account.side)], now_ms,
                                               ref_price, reason, self.rng)
        return self.pending_order

    def _send(self, broker: StockBroker, legs: list[Leg], now: int, ref: float, reason: str, res: BotStep,
              forced: bool = False) -> None:
        order = broker.make_order(self.bot_id, self.symbol, legs, now, ref, reason, self.rng, forced=forced)
        self.pending_order = order
        res.orders.append(order)

    def on_execution(self, execution: StockExecution) -> None:
        """Called by the engine when the pending order settles. An open fixes the stop at entry and resets the forced
        timer."""
        self.pending_order = None
        if execution.opened_trade_id is not None:
            self.minutes_since_open = 0
            self.stop_bps = self._stop_at_send
        elif not self.account.is_open:
            self.stop_bps = None


def _legs(action: str, position: int, ref: float, broker: StockBroker) -> list[Leg]:
    """The legs of an action, opens sized from the reference price and closes from the quantity held at the fill."""
    if action in OPENS:
        side = LONG if action == "open_long" else SHORT
        return [Leg("open", side, broker.open_qty(side, ref))]
    if action == "close":
        return [Leg("close", position)]
    if action == "flip":
        return [Leg("close", position), Leg("open", -position, broker.open_qty(-position, ref))]
    return []


def _reason(action: str, pred: float, required: float, gate: bool, forced: bool, short_ok: bool) -> str:
    if forced:
        return f"forced by the exploration timer, prediction {pred:.2f} bps"
    if pred < 0 and not short_ok:
        return f"prediction {pred:.2f} bps is negative and shorts are disabled"
    if gate:
        return f"prediction {pred:.2f} bps clears required edge {required:.2f} bps"
    if pred == 0:
        return "prediction is exactly zero"
    return f"prediction {pred:.2f} bps is below required edge {required:.2f} bps"
