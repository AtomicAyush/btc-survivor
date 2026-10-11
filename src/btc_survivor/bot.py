"""One bot: the Warmup, Active and Dead states, its per minute step, and the decision rule.

Each healthy minute a bot pays rent, marks its position, checks death and its stop, learns from the sample that
matured this minute, predicts, and at its phase minute decides. The engine adds the shared stale gate.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from btc_survivor.account import Account, death_threshold
from btc_survivor.broker import Broker, Execution, estimate_round_trip
from btc_survivor.clock import MINUTES_PER_DAY, MS_PER_MINUTE, minute_index
from btc_survivor.config import Config
from btc_survivor.features import N_FEATURES
from btc_survivor.feed.book import LocalBook
from btc_survivor.genes import Genes, initial_genes
from btc_survivor.labels import Matured, MaturedWindow
from btc_survivor.model import MLP, AdamW
from btc_survivor.seeds import bot_rng
from btc_survivor.skill import RollingSkill
from btc_survivor.standardizer import LabelScale
from btc_survivor.types import FLAT, LONG, SHORT, Leg, Order

WARMUP, ACTIVE, DEAD = "warmup", "active", "dead"
POSITION_NAMES = {LONG: "long", SHORT: "short", FLAT: "flat"}


def decide(position: int, pred_bps: float, required_bps: float, allow_short: bool, forced: bool,
           skill_ok: bool = True) -> tuple[str, bool, bool]:
    """The pure decision rule, returning (action, real gate result, forced exploration used).

    Forced exploration applies only to a flat bot.
    """
    direction = LONG if pred_bps > 0 else SHORT if pred_bps < 0 else FLAT
    if direction == SHORT and not allow_short:
        direction = FLAT
    gate = direction != FLAT and abs(pred_bps) >= required_bps and skill_ok
    use_forced = forced and not gate and position == FLAT and direction != FLAT
    if gate or use_forced:
        if position == direction:
            return "hold", gate, use_forced
        if position == FLAT:
            return ("open_long" if direction == LONG else "open_short"), gate, use_forced
        return "flip", gate, use_forced
    return ("close" if position != FLAT else "none"), gate, False


@dataclass
class MinuteContext:
    """Shared state for one healthy minute, computed once and handed to every bot."""

    boundary_ms: int
    book: LocalBook
    z: np.ndarray | None
    matured: Matured | None
    window: MaturedWindow
    label_scale: LabelScale


@dataclass
class StepResult:
    decision: dict | None = None
    orders: list[Order] = field(default_factory=list)
    died: bool = False
    stopped: bool = False


class Bot:
    def __init__(self, cfg: Config, bot_id: int, slot: int, generation: int, born_ms: int, genes: Genes,
                 model: MLP, rng: np.random.Generator, origin: str = "random", parent_id: int | None = None) -> None:
        self.cfg = cfg
        self.bot_id, self.slot, self.generation, self.born_ms = bot_id, slot, generation, born_ms
        self.genes = genes.clamped(cfg.genes)
        self.model, self.rng, self.origin, self.parent_id = model, rng, origin, parent_id
        self.opt = AdamW(model.size, cfg.model.adam_beta1, cfg.model.adam_beta2)
        self.account = Account(cfg.trading.start_equity)
        hours = cfg.time.warmup_hours_inherited if origin == "mutated" else cfg.time.warmup_hours_random
        self.warmup_minutes = round(hours * 60)
        self.state = WARMUP if self.warmup_minutes > 0 else ACTIVE
        self.trained_minutes = 0
        self.alive_minutes = 0
        self.minutes_since_open = 0
        self.pending_preds: dict[int, float] = {}
        self.skill = RollingSkill(cfg.model.skill_window_samples)
        self.pending_order: Order | None = None
        self.hourly_equity: deque[float] = deque([cfg.trading.start_equity],
                                                 maxlen=cfg.population.fitness_window_hours + 1)
        self.equity = cfg.trading.start_equity
        self.death_threshold: float | None = None

    @classmethod
    def newborn(cls, cfg: Config, bot_id: int, slot: int, born_ms: int) -> "Bot":
        """A fully random bot with the initial genes."""
        rng = bot_rng(cfg.run.master_seed, bot_id)
        model = MLP.random(N_FEATURES, cfg.model.hidden_units, rng, cfg.model.init_output_scale)
        return cls(cfg, bot_id, slot, 0, born_ms, initial_genes(cfg.genes), model, rng)

    def is_phase_minute(self, boundary_ms: int) -> bool:
        horizon = self.cfg.time.horizon_minutes
        return minute_index(boundary_ms) % horizon == self.slot % horizon

    def mark_price(self, book: LocalBook) -> float | None:
        """The price the position would get if closed now, by walking the exit side for its quantity."""
        if not self.account.is_open:
            return None
        walk = book.walk("sell" if self.account.side == LONG else "buy", self.account.qty)
        return walk.avg_price if walk.avg_price is not None else book.mid

    def step(self, ctx: MinuteContext, broker: Broker) -> StepResult:
        cfg, acct, book, now = self.cfg, self.account, ctx.book, ctx.boundary_ms
        res = StepResult()
        if self.state == DEAD:
            return res
        acct.charge_rent(cfg.costs.daily_running_cost / MINUTES_PER_DAY)
        acct.accrue_funding(now, cfg.costs.short_funding_bps_per_hour)
        self.alive_minutes += 1
        self.minutes_since_open += 1
        mark = self.mark_price(book)
        self.equity = acct.equity(mark)
        if self.alive_minutes % 60 == 0:
            self.hourly_equity.append(self.equity)

        if self._check_death(book, now, broker, res):
            return res
        if acct.is_open and self.pending_order is None and mark is not None:
            adverse = (acct.entry_price - mark) if acct.side == LONG else (mark - acct.entry_price)
            if adverse / acct.entry_price * 100.0 >= cfg.trading.stop_loss_pct:
                self._send(broker, [Leg("close", acct.side)], now, book.mid, "stop", res)
                res.stopped = True

        if ctx.z is not None and ctx.matured is not None and ctx.label_scale.ready:
            self._learn(ctx)
        horizon_ms = cfg.time.horizon_minutes * MS_PER_MINUTE
        self.pending_preds = {t: p for t, p in self.pending_preds.items() if now - t < horizon_ms}

        pred = None
        if ctx.z is not None and ctx.label_scale.ready:
            pred = self.model.predict(ctx.z) * ctx.label_scale.scale
            self.pending_preds[now] = pred
        if pred is not None and self.is_phase_minute(now):
            res.decision = self._decide(now, book, pred, res, broker)
        return res

    def _check_death(self, book: LocalBook, now: int, broker: Broker, res: StepResult) -> bool:
        """Equity below one more round trip plus a day of rent means death. A pending close becomes the death close."""
        acct = self.account
        side = acct.side if acct.is_open else LONG
        est = estimate_round_trip(book, self.cfg.trading.notional, self.cfg.costs, side, self.cfg.time.horizon_minutes)
        if est is None:
            return False
        threshold = death_threshold(est.total_bps, self.cfg.trading.notional, self.cfg.costs.daily_running_cost)
        if self.equity >= threshold:
            return False
        self.death_threshold = threshold
        self.state = DEAD
        res.died = True
        if acct.is_open and self.pending_order is None:
            self._send(broker, [Leg("close", acct.side)], now, book.mid, "death", res)
        return True

    def _learn(self, ctx: MinuteContext) -> None:
        """Score the prediction made for the matured minute, then take one AdamW step on a replay batch."""
        cfg = self.cfg
        pred = self.pending_preds.pop(ctx.matured.minute_ms, None)
        if pred is not None:
            self.skill.add(pred, ctx.matured.label_bps)
        X, y = ctx.window.batch(self.rng, cfg.model.batch_size)
        _, grad = self.model.loss_and_grad(X, y / ctx.label_scale.scale, cfg.model.huber_delta)
        self.opt.step(self.model, grad, self.genes.lr, cfg.genes.weight_decay, cfg.model.grad_clip)
        self.trained_minutes += 1
        if self.state == WARMUP and self.trained_minutes >= self.warmup_minutes:
            self.state = ACTIVE

    def _decide(self, now: int, book: LocalBook, pred: float, res: StepResult, broker: Broker) -> dict:
        cfg, acct = self.cfg, self.account
        position = acct.side
        short_ok = cfg.trading.allow_short
        side = SHORT if pred < 0 and short_ok else LONG
        est = estimate_round_trip(book, cfg.trading.notional, cfg.costs, side, cfg.time.horizon_minutes)
        cost = est.total_bps if est is not None else math.inf
        required = cost + self.genes.margin_bps
        skill_ok = True
        if cfg.model.gate_on_skill:
            stats = self.skill.stats()
            skill_ok = self.skill.full and stats.corr is not None and stats.corr > cfg.model.skill_gate_min_corr
        forced_due = self.minutes_since_open >= cfg.trading.force_trade_after_hours * 60
        action, gate, used_forced = decide(position, pred, required, short_ok, forced_due, skill_ok)
        row = {"bot_id": self.bot_id, "ts": now, "position_before": POSITION_NAMES[position], "ref_mid": book.mid,
               "prediction_bps": pred, "required_edge_bps": required, "cost_estimate_bps": cost,
               "margin_bps": self.genes.margin_bps, "gate_passed": int(gate), "forced": 0, "action": "none"}
        if self.state == WARMUP:
            return {**row, "reason": f"warmup, {self.trained_minutes} of {self.warmup_minutes} training minutes"}
        if res.stopped:
            return {**row, "reason": "stop loss fired this minute"}
        if self.pending_order is not None:
            return {**row, "reason": "an order is still pending"}
        if est is None:
            return {**row, "reason": "the visible book cannot fill the notional"}
        reason = _reason(action, pred, required, gate, used_forced, skill_ok, short_ok, self.minutes_since_open)
        qty = broker.open_qty(book.mid)
        legs = {"open_long": [Leg("open", LONG, qty)], "open_short": [Leg("open", SHORT, qty)],
                "close": [Leg("close", position)],
                "flip": [Leg("close", position), Leg("open", -position, qty)]}.get(action, [])
        if legs:
            self._send(broker, legs, now, book.mid, "decision", res, forced=used_forced)
        return {**row, "gate_passed": int(gate), "forced": int(used_forced), "action": action, "reason": reason}

    def _send(self, broker: Broker, legs: list[Leg], now: int, mid: float, reason: str, res: StepResult,
              forced: bool = False) -> None:
        order = broker.make_order(self.bot_id, legs, now, mid, reason, self.rng, forced=forced)
        self.pending_order = order
        res.orders.append(order)

    def on_execution(self, execution: Execution) -> None:
        """Called by the engine when the pending order settles. Any open resets the forced exploration timer."""
        self.pending_order = None
        if execution.opened_trade_id is not None:
            self.minutes_since_open = 0


def _reason(action: str, pred: float, required: float, gate: bool, forced: bool, skill_ok: bool,
            short_ok: bool, idle_minutes: int) -> str:
    if forced:
        return f"forced after {idle_minutes / 60:.1f} h without opening, prediction {pred:.2f} bps"
    if pred < 0 and not short_ok:
        return f"prediction {pred:.2f} bps is negative and shorts are disabled"
    if not skill_ok:
        return "skill gate closed"
    if gate:
        return f"prediction {pred:.2f} bps clears required edge {required:.2f} bps"
    if pred == 0:
        return "prediction is exactly zero"
    return f"prediction {pred:.2f} bps is below required edge {required:.2f} bps"
