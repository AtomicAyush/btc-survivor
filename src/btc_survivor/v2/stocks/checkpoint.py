"""Checkpoints of the stock process by PLAN_V2 sections 4.4, 9.2 and 14.2.

Each run's checkpoint holds its engine and a copy of the shared market layer, in the version 1 layout of JSON for
scalars and generator states and one raw float64 blob for arrays, with no pickle, so identical states give identical
bytes. The shared part carries the series, standardizers, label scales, label queues, open evaluation labels,
linear_prices, the guards, the last healthy minutes and prices, the minute counts, news health and each symbol's
epoch. The engine part carries every bot with its weights, optimizer, generator, account, timers, skill rings, stop and
pending order, the dying bots, the pooled window, linear_news, the stored news inputs, the news sequence, the trade
ids and the owed flattens. S2's news state is rebuilt by adding its kept lines again with their receive stamps.

Candles, price health apart from its epochs, the heartbeat clock and the Roll accumulators are rebuilt fresh by the
shared layer's resume, which bumps every epoch, so a restart is a gap that no pending label survives.

Resume takes the newest checkpoint that every run holds and can read, rolling back the rows a run wrote after it, and
refuses when a config's experiment hash changed, when recomputing the checkpoint's session minute index from the
session table gives a different number, or when the runs' copies of the shared layer differ.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import deque
from datetime import date
from collections.abc import Sequence

import numpy as np

from btc_survivor import checkpoint as v1
from btc_survivor.account import Account
from btc_survivor.baselines import LinearBaseline
from btc_survivor.benchmark import Anchor
from btc_survivor.clock import MS_PER_MINUTE, iso, minute_floor, parse_iso
from btc_survivor.labels import Pending
from btc_survivor.model import MLP
from btc_survivor.standardizer import RunningStats
from btc_survivor.types import Leg
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.stocks.bot import StockBot
from btc_survivor.v2.stocks.broker import GuardState, StockOrder
from btc_survivor.v2.stocks.engine import StockEngine
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES
from btc_survivor.v2.stocks.labels import _Open  # the open evaluation row, which only a checkpoint rebuilds
from btc_survivor.v2.stocks.labels import PENDING, LabelRow
from btc_survivor.v2.stocks.learning import N_NEWS_INPUTS
from btc_survivor.v2.stocks.market import MarketLayer
from btc_survivor.v2.stocks.news_inputs import NewsInput, ScoreLine
from btc_survivor.v2.stocks import records
from btc_survivor.v2.stocks.population import StockPopulation
from btc_survivor.v2.stocks.store import StockStore
from btc_survivor.v2.stocks.twins import StockTwin

log = logging.getLogger(__name__)

KIND = "stocks"
VERSION = 1
MARKET, ENGINE = "market", "engine"
# The run meta keys holding the news hashes recorded at launch, which news health compares every heartbeat with.
LAUNCH_HASH_KEYS = ("news_config_hash", "prompt_sha256", "schema_sha256")


class ResumeError(RuntimeError):
    """Raised when the run databases cannot be resumed safely."""


def session_index(ms: int) -> int:
    """The session minute index k of the last session boundary at or before ms, which a resume recomputes from the
    session table to prove the table never changed under the run."""
    session = sessions.next_session(ms)
    if ms < session.open_ms + MS_PER_MINUTE:
        return session.first_k - 1
    return session.k(minute_floor(ms))


# Small pieces


def _put_stats(arrays: v1.Arrays, key: str, stats: RunningStats) -> int:
    arrays.put(f"{key}.mean", stats.mean)
    arrays.put(f"{key}.var", stats.var)
    return stats.n


def _load_stats(stats: RunningStats, n: int, arrays: dict[str, np.ndarray], key: str) -> None:
    stats.n, stats.mean, stats.var = n, arrays[f"{key}.mean"], arrays[f"{key}.var"]


def _put_linear(arrays: v1.Arrays, key: str, linear: LinearBaseline) -> dict:
    for name in ("xx", "xy", "coef"):
        arrays.put(f"{key}.{name}", getattr(linear, name))
    return {"n": linear.n, "weight": linear.weight}


def _load_linear(linear: LinearBaseline, d: dict, arrays: dict[str, np.ndarray], key: str) -> None:
    linear.n, linear.weight = d["n"], d["weight"]
    linear.xx, linear.xy, linear.coef = (arrays[f"{key}.{name}"] for name in ("xx", "xy", "coef"))


def _rows(values: list[np.ndarray], width: int) -> np.ndarray:
    """Equal length vectors stacked into rows, an empty table when there are none."""
    return np.array(values, dtype=float).reshape(len(values), width)


def dump_order(order: StockOrder | None) -> dict | None:
    if order is None:
        return None
    return {"bot_id": order.bot_id, "symbol": order.symbol, "legs": [[l.action, l.position_side, l.qty]
                                                                     for l in order.legs],
            "decided_ms": order.decided_ms, "due_ms": order.due_ms, "ref_price": order.ref_price,
            "reason": order.reason, "forced": order.forced}


def load_order(d: dict | None) -> StockOrder | None:
    if d is None:
        return None
    legs = [Leg(action, side, qty) for action, side, qty in d["legs"]]
    return StockOrder(d["bot_id"], d["symbol"], legs, d["decided_ms"], d["due_ms"], d["ref_price"], d["reason"],
                      d["forced"])


# The shared layer


def dump_market(market: MarketLayer, arrays: v1.Arrays, at: int) -> dict:
    """Everything a resumed shared layer restores, the series and statistics of every symbol by its position."""
    symbols = market.symbols
    arrays.put("market.last_ref", market.last_ref)
    series, standardizers, scales, queues, evaluation = [], [], [], [], []
    for i, symbol in enumerate(symbols):
        key = f"market.{i}"
        s = market.series[symbol]
        for name in ("mid", "volume", "valid", "cum_valid"):
            arrays.put(f"{key}.series.{name}", getattr(s.minutes, name))
        arrays.put(f"{key}.series.buy", s.buy_volume)
        arrays.put(f"{key}.series.sell", s.sell_volume)
        series.append({"last_index": s.minutes.last_index, "valid_total": s.minutes.valid_total})
        standardizers.append(_put_stats(arrays, f"{key}.standardizer", market.standardizers[symbol].stats))
        scales.append(_put_stats(arrays, f"{key}.label_scale", market.label_scales[symbol].stats))
        queue = market.queues[symbol].queue
        arrays.put(f"{key}.queue.z", _rows([p.z for p in queue.pending], N_PRICE_FEATURES))
        queues.append({"discarded": queue.discarded,
                       "pending": [[p.minute_ms, p.epoch, p.entry_mid] for p in queue.pending]})
        evaluation.append([[o.row.ts, o.row.entry_price, o.exit_k, o.healthy, o.epoch]
                           for o in market.evaluation[symbol]._open])
    return {
        "at": at, "symbols": list(symbols), "news_ok": market.news_ok, "session_day": market.session_day,
        "healthy_minutes": market.healthy_minutes, "run_minutes": market.run_minutes,
        "run_minutes_total": market.run_minutes_total,
        "last_healthy_k": [market.last_healthy_k[s] for s in symbols],
        "epochs": [market.health.symbols[s].epoch for s in symbols],
        "guards": [dataclasses.asdict(market.guards.state[s]) for s in symbols],
        "linear": _put_linear(arrays, "market.linear", market.linear),
        "series": series, "standardizers": standardizers, "label_scales": scales, "queues": queues,
        "evaluation": evaluation,
    }


def load_market(cfg: StockConfig, launch_hashes: tuple[str, str, str], d: dict,
                arrays: dict[str, np.ndarray]) -> MarketLayer:
    """A shared layer restored from its checkpoint part, before its resume rebuilds the rest and bumps every epoch."""
    market = MarketLayer(cfg, launch_hashes)
    if list(market.symbols) != d["symbols"]:
        raise ResumeError(f"the checkpoint covers {d['symbols']}, the config {list(market.symbols)}")
    market.news_ok, market.session_day = d["news_ok"], d["session_day"]
    market.healthy_minutes, market.run_minutes = d["healthy_minutes"], d["run_minutes"]
    market.run_minutes_total = d["run_minutes_total"]
    market.last_ref = arrays["market.last_ref"]
    _load_linear(market.linear, d["linear"], arrays, "market.linear")
    for i, symbol in enumerate(market.symbols):
        key = f"market.{i}"
        market.last_healthy_k[symbol] = d["last_healthy_k"][i]
        market.health.symbols[symbol].epoch = d["epochs"][i]
        market.guards.state[symbol] = GuardState(**d["guards"][i])
        minutes = market.series[symbol].minutes
        minutes.mid, minutes.volume = arrays[f"{key}.series.mid"], arrays[f"{key}.series.volume"]
        minutes.valid = arrays[f"{key}.series.valid"].astype(bool)
        minutes.cum_valid = arrays[f"{key}.series.cum_valid"].astype(np.int64)
        minutes.last_index, minutes.valid_total = d["series"][i]["last_index"], d["series"][i]["valid_total"]
        market.series[symbol].buy_volume = arrays[f"{key}.series.buy"]
        market.series[symbol].sell_volume = arrays[f"{key}.series.sell"]
        _load_stats(market.standardizers[symbol].stats, d["standardizers"][i], arrays, f"{key}.standardizer")
        _load_stats(market.label_scales[symbol].stats, d["label_scales"][i], arrays, f"{key}.label_scale")
        queue, z = market.queues[symbol].queue, arrays[f"{key}.queue.z"]
        queue.discarded = d["queues"][i]["discarded"]
        queue.pending = deque(Pending(minute_ms, epoch, z[j].copy(), entry)
                              for j, (minute_ms, epoch, entry) in enumerate(d["queues"][i]["pending"]))
        market.evaluation[symbol]._open = [
            _Open(LabelRow(ts, symbol, entry, None, None, PENDING), exit_k, healthy, epoch)
            for ts, entry, exit_k, healthy, epoch in d["evaluation"][i]]
    return market


# Bots and engines


def dump_bot(bot: StockBot, arrays: v1.Arrays) -> dict:
    """Everything that defines a learner or twin, its arrays in the shared blob under its id."""
    key = f"bot.{bot.bot_id}"
    arrays.put(f"{key}.params", bot.model.params)
    arrays.put(f"{key}.adam_m", bot.opt.m)
    arrays.put(f"{key}.adam_v", bot.opt.v)
    arrays.put(f"{key}.skill_pred", bot.skill.pred)
    arrays.put(f"{key}.skill_label", bot.skill.label)
    arrays.put(f"{key}.preds", bot.preds)
    return {
        "bot_id": bot.bot_id, "slot": bot.slot, "born_ms": bot.born_ms, "twin": isinstance(bot, StockTwin),
        "state": bot.state, "head_start": bot.head_start, "trained_minutes": bot.trained_minutes,
        "alive_minutes": bot.alive_minutes, "minutes_since_open": bot.minutes_since_open,
        "pending_preds": sorted(bot.pending_preds.items()), "skill_n": bot.skill.n, "skill_head": bot.skill.head,
        "adam_t": bot.opt.t, "stop_bps": bot.stop_bps, "stop_at_send": bot._stop_at_send, "equity": bot.equity,
        "death_threshold": bot.death_threshold, "account": dataclasses.asdict(bot.account),
        "pending_order": dump_order(bot.pending_order), "rng": bot.rng.bit_generator.state,
    }


def load_bot(cfg: StockConfig, d: dict, arrays: dict[str, np.ndarray]) -> StockBot:
    key = f"bot.{d['bot_id']}"
    rng = v1.rng_from_state(d["rng"])
    if d["twin"]:
        bot: StockBot = StockTwin(cfg, d["bot_id"], d["slot"], d["born_ms"], rng)
    else:
        model = MLP(cfg.model.inputs, cfg.model.hidden_units, params=arrays[f"{key}.params"])
        bot = StockBot(cfg, d["bot_id"], d["slot"], d["born_ms"], model, rng)
    bot.model.params[:] = arrays[f"{key}.params"]
    bot.opt.m, bot.opt.v, bot.opt.t = arrays[f"{key}.adam_m"], arrays[f"{key}.adam_v"], d["adam_t"]
    bot.skill.pred, bot.skill.label = arrays[f"{key}.skill_pred"], arrays[f"{key}.skill_label"]
    bot.skill.n, bot.skill.head = d["skill_n"], d["skill_head"]
    bot.preds = arrays[f"{key}.preds"]
    bot.state, bot.head_start, bot.trained_minutes = d["state"], d["head_start"], d["trained_minutes"]
    bot.alive_minutes, bot.minutes_since_open = d["alive_minutes"], d["minutes_since_open"]
    bot.pending_preds = {int(k): p for k, p in d["pending_preds"]}
    bot.stop_bps, bot._stop_at_send = d["stop_bps"], d["stop_at_send"]
    bot.equity, bot.death_threshold = d["equity"], d["death_threshold"]
    bot.account = Account(**d["account"])
    bot.pending_order = load_order(d["pending_order"])
    return bot


def dump_engine(engine: StockEngine, arrays: v1.Arrays) -> dict:
    """Everything one run restores beside the shared layer."""
    pop, window, inputs = engine.population, engine.window, engine.inputs
    for name in ("x", "label", "symbol", "minute"):
        arrays.put(f"window.{name}", getattr(window, name))
    arrays.put("inputs.current", _rows([inputs.current[s] for s in engine.symbols], N_NEWS_INPUTS))
    pushed = list(inputs._pushed)
    arrays.put("inputs.pushed", _rows([inputs._pushed[key] for key in pushed], N_NEWS_INPUTS))
    news = None
    if inputs.news is not None:
        news = {"lines": [[line.encode(), line.recv_ms] for line in inputs.news.kept_lines()],
                "ignored": inputs.news.streams.ignored, "last_ms": inputs.news._last_ms,
                "streams": [[stream, [list(inputs.streams[stream][s]) for s in engine.symbols]]
                            for stream in inputs.streams]}
    return {
        "bots": [dump_bot(bot, arrays) for bot in engine.all_bots()],
        "learners": [None if b is None else b.bot_id for b in pop.learners],
        "twins": [None if t is None else t.bot_id for t in pop.twins],
        "dying": [b.bot_id for b in engine.dying],
        "news_seq": engine.news_seq, "deaths": engine.deaths, "next_trade_id": engine.broker.next_trade_id,
        "sent": engine._sent, "pending": sorted([sent, bot_id] for bot_id, (sent, _) in engine.pending.items()),
        "owed": list(engine.owed.items()),
        "anchor": None if engine.anchor is None else [[a.ms, a.ask, a.mid] for a in engine.anchor],
        "carried": sorted(engine._carried),
        "decided": [[k, symbol, ids] for (k, symbol), ids in engine._decided.items()],
        "window": {"start": window.start, "count": window.count},
        "linear_news": None if engine.linear_news is None else _put_linear(arrays, "linear_news",
                                                                           engine.linear_news),
        "pushed": [list(key) for key in pushed], "news": news,
    }


def load_engine(cfg: StockConfig, store: StockStore, d: dict, arrays: dict[str, np.ndarray],
                checkpoint_ts: str | None = None) -> StockEngine:
    """An engine restored from its checkpoint part, before its resume rejects the orders that were in flight. The
    bot states last written come from the database, so the next persist repairs any a rollback left behind."""
    bots = {b["bot_id"]: load_bot(cfg, b, arrays) for b in d["bots"]}
    population = StockPopulation([None if i is None else bots[i] for i in d["learners"]],
                                 [None if i is None else bots[i] for i in d["twins"]])
    engine = StockEngine(cfg, store, population)
    engine.dying = [bots[i] for i in d["dying"]]
    engine.news_seq, engine.deaths = d["news_seq"], d["deaths"]
    engine.broker.next_trade_id = d["next_trade_id"]
    engine._sent = d["sent"]
    engine.pending = {bot_id: (sent, bots[bot_id]) for sent, bot_id in d["pending"]}
    engine.owed = {symbol: reason for symbol, reason in d["owed"]}
    engine.anchor = None if d["anchor"] is None else tuple(Anchor(*a) for a in d["anchor"])
    engine._carried = set(d["carried"])
    engine._decided = {(k, symbol): list(ids) for k, symbol, ids in d["decided"]}
    window = engine.window
    window.x, window.label = arrays["window.x"], arrays["window.label"]
    window.symbol = arrays["window.symbol"].astype(np.int64)
    window.minute = arrays["window.minute"].astype(np.int64)
    window.start, window.count = d["window"]["start"], d["window"]["count"]
    if engine.linear_news is not None:
        _load_linear(engine.linear_news, d["linear_news"], arrays, "linear_news")
    inputs = engine.inputs
    inputs.current = {s: arrays["inputs.current"][i].copy() for i, s in enumerate(engine.symbols)}
    inputs._pushed = {(k, symbol): arrays["inputs.pushed"][i].copy() for i, (k, symbol) in enumerate(d["pushed"])}
    if inputs.news is not None:
        news = d["news"]
        for raw, recv_ms in news["lines"]:
            inputs.news.add(ScoreLine.decode(raw, recv_ms))
        inputs.news.streams.ignored, inputs.news._last_ms = news["ignored"], news["last_ms"]
        inputs.streams = {stream: {s: NewsInput(*values) for s, values in zip(engine.symbols, rows)}
                          for stream, rows in news["streams"]}
    engine.last_checkpoint_ts = checkpoint_ts
    if store is not None:
        engine._stored_state = {r["bot_id"]: r["state"] for r in store.query("SELECT bot_id, state FROM bots")}
    return engine


# Whole checkpoints


def encode(market: MarketLayer, engine: StockEngine, at: int) -> tuple[str, bytes]:
    """One run's checkpoint at a persistence point, proven to read back before it is written."""
    arrays = v1.Arrays()
    meta = {"kind": KIND, "version": VERSION, "at": at, "run_name": engine.cfg.run.name,
            "experiment_hash": engine.cfg.experiment_hash, "session_k": session_index(at),
            MARKET: dump_market(market, arrays, at), ENGINE: dump_engine(engine, arrays)}
    meta_text, blob = v1.encode(meta, arrays)
    decode(meta_text, blob)
    return meta_text, blob


def decode(meta_text: str, blob: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """A stock checkpoint read back, raising ValueError for anything unreadable or of another kind."""
    meta, arrays = v1.decode(meta_text, blob)
    if meta.get("kind") != KIND or meta.get("version") != VERSION:
        raise ValueError(f"not a stock checkpoint of version {VERSION}")
    return meta, arrays


def shared_part(meta: dict, arrays: dict[str, np.ndarray]) -> tuple[str, dict[str, tuple]]:
    """The shared layer's part of a checkpoint as text and exact bytes, which both runs' checkpoints must share."""
    exact = {key: (value.shape, value.tobytes()) for key, value in arrays.items() if key.startswith(f"{MARKET}.")}
    return json.dumps(meta[MARKET], sort_keys=True), exact


def launch_hashes(store: StockStore) -> tuple[str, str, str] | None:
    """The news hashes a run recorded at launch, or None when any is missing."""
    values = tuple(store.get_meta(key) for key in LAUNCH_HASH_KEYS)
    return None if any(v is None for v in values) else values  # type: ignore[return-value]


def resume_problems(cfgs: Sequence[StockConfig], stores: Sequence[StockStore],
                    chosen: Sequence[tuple[dict, dict[str, np.ndarray]]]) -> list[str]:
    """Why the chosen checkpoints cannot be resumed, none when they can."""
    problems = []
    for cfg, (meta, _) in zip(cfgs, chosen):
        name = cfg.run.name
        if meta["experiment_hash"] != cfg.experiment_hash:
            problems.append(f"{name}: the config changed since the checkpoint, whose experiment hash "
                            f"{meta['experiment_hash'][:12]} is now {cfg.experiment_hash[:12]}")
        recomputed = session_index(meta["at"])
        if meta["session_k"] != recomputed:
            problems.append(f"{name}: the session table changed, the checkpoint at {iso(meta['at'])} was at session "
                            f"minute index {meta['session_k']} and the table now gives {recomputed}")
    first = shared_part(*chosen[0])
    if any(shared_part(*other) != first for other in chosen[1:]):
        problems.append("the copies of the shared market layer in the runs' checkpoints differ")
    hashes = [launch_hashes(store) for store in stores]
    if None in hashes:
        problems.append("a run database holds no news hashes from its launch")
    elif len(set(hashes)) > 1:
        problems.append("the runs recorded different news hashes at launch")
    return problems


def _readable(store: StockStore) -> list[tuple[str, dict, dict[str, np.ndarray]]]:
    """Every checkpoint of a run that reads back, newest first."""
    out = []
    for row in store.query("SELECT id, ts, meta, arrays FROM checkpoints ORDER BY id DESC"):
        try:
            meta, arrays = decode(row["meta"], row["arrays"])
        except ValueError as exc:
            log.error("checkpoint %d from %s cannot be read (%s), trying an older one", row["id"], row["ts"], exc)
            continue
        out.append((row["ts"], meta, arrays))
    return out


def load(cfgs: Sequence[StockConfig], stores: Sequence[StockStore]) -> tuple[MarketLayer, list[StockEngine]] | None:
    """The shared layer and every engine restored from the newest checkpoint all runs hold and can read, after
    rolling back what any run wrote later, or None when every run database is new. Nothing changes on a refusal."""
    found = [_readable(store) for store in stores]
    if not any(found):
        held = [str(cfg.db_path) for cfg, store in zip(cfgs, stores) if store.query("SELECT 1 FROM bots LIMIT 1")]
        if held:
            raise ResumeError(f"{', '.join(held)} holds bots but no readable checkpoint, refusing to seed a new "
                              f"population over old history")
        return None
    by_ts: list[dict[str, tuple[dict, dict[str, np.ndarray]]]] = [{} for _ in found]
    for rows, chosen_by_ts in zip(found, by_ts):
        for ts, meta, arrays in rows:  # newest first, so a shutdown checkpoint wins over a point at the same ms
            chosen_by_ts.setdefault(ts, (meta, arrays))
    common = next((ts for ts, _, _ in found[0] if all(ts in other for other in by_ts[1:])), None)
    if common is None:
        raise ResumeError("the run databases share no readable checkpoint, so the runs cannot resume together")
    chosen = [other[common] for other in by_ts]
    problems = resume_problems(cfgs, stores, chosen)
    if problems:
        raise ResumeError("refusing to resume\n  " + "\n  ".join(problems))
    rolled = False
    for cfg, store in zip(cfgs, stores):
        newest = store.query("SELECT ts FROM checkpoints ORDER BY id DESC LIMIT 1")[0]["ts"]
        if newest != common:
            store.rollback_after(common, iso(parse_iso(common) - cfg.time.horizon_minutes * MS_PER_MINUTE))
            log.warning("%s rolled back to its checkpoint from %s, which both runs hold", cfg.run.name, common)
            rolled = True
    market = load_market(cfgs[0], launch_hashes(stores[0]), chosen[0][0][MARKET], chosen[0][1])
    if rolled and market.session_day is not None:
        # a rollback deletes the row of the session open at the checkpoint, so every run gets it again as the
        # checkpoint left it, and a resume after that session closed still finds both runs' rows equal
        day = sessions.session_on(date.fromisoformat(market.session_day))
        row = records.session_row(day, market.healthy_minutes, market.run_minutes)
        for store in stores:
            store.commit_now([(records.PUT_SESSION, row)])
    engines = [load_engine(cfg, store, meta[ENGINE], arrays, common)
               for cfg, store, (meta, arrays) in zip(cfgs, stores, chosen)]
    return market, engines
