"""Row builders for the database tables, so the engine only decides what to write and when."""

from __future__ import annotations

from btc_survivor.account import Account
from btc_survivor.bot import POSITION_NAMES, Bot
from btc_survivor.clock import MS_PER_HOUR, iso
from btc_survivor.feed.candles import Candle
from btc_survivor.types import ClosedTrade, Fill

INSERT_BOT = ("INSERT INTO bots (bot_id, slot, parent_id, generation, born_at, state, margin_bps, lr, seed, origin) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
SET_BOT_STATE = "UPDATE bots SET state = ? WHERE bot_id = ?"
SET_BOT_DEAD = "UPDATE bots SET state = 'dead', died_at = ? WHERE bot_id = ?"
INSERT_LINEAGE = ("INSERT INTO lineage (bot_id, parent_id, generation, born_at, sigma_weights, sigma_genes, "
                  "margin_factor, lr_factor, parent_margin_bps, parent_lr, parent_fitness, parent_skill_corr) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_DECISION = ("INSERT INTO decisions (bot_id, ts, position_before, ref_mid, prediction_bps, required_edge_bps, "
                   "cost_estimate_bps, margin_bps, gate_passed, forced, action, reason) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
SET_LABEL = "UPDATE decisions SET label_bps = ? WHERE bot_id = ? AND ts = ? AND label_bps IS NULL"
INSERT_FILL = ("INSERT INTO fills (bot_id, trade_id, decided_at, filled_at, leg, side, qty, ref_mid, book_avg_price, "
               "price, levels_consumed, slippage_bps, fee, latency_ms, status, reason) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
OPEN_TRADE = ("INSERT INTO trades (id, bot_id, side, opened_at, qty, entry_price, fee_open, funding, "
              "slippage_bps_open, forced) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
CLOSE_TRADE = ("UPDATE trades SET closed_at = ?, exit_price = ?, fee_close = ?, funding = ?, slippage_bps_close = ?, "
               "pnl_gross = ?, pnl_net = ?, exit_reason = ? WHERE id = ?")
INSERT_SNAPSHOT = ("INSERT INTO equity_snapshots (bot_id, ts, cash, equity, unrealized, realized_gross, fees_total, "
                   "funding_total, rent_total, stop_loss_total, position, skill_corr, skill_dir_acc, skill_mae, "
                   "skill_n) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_DEATH = ("INSERT INTO deaths (bot_id, died_at, generation, cause, fees_total, funding_total, rent_total, "
                "stop_loss_total, other_losses_total, writeoff, threshold, final_equity, trades, lifespan_hours) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_CANDLE = ("INSERT INTO candles (ts, open, high, low, close, volume, trade_count, mid, valid) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_BENCHMARK = ("INSERT INTO benchmark (ts, btc_mid, buy_hold_equity, matched_equity, living_mean_equity, "
                    "living_median_equity, living, twin_mean_equity, cash_equity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_FEED_EVENT = "INSERT INTO feed_events (ts, kind, detail) VALUES (?, ?, ?)"
INSERT_FORECAST = "INSERT INTO forecasts (ts, predictor, prediction_bps) VALUES (?, ?, ?)"
SET_FORECAST_LABEL = "UPDATE forecasts SET label_bps = ? WHERE ts = ? AND label_bps IS NULL"
INSERT_CHECKPOINT = "INSERT INTO checkpoints (ts, schema_version, meta, arrays) VALUES (?, ?, ?, ?)"
PRUNE_CHECKPOINTS = "DELETE FROM checkpoints WHERE id NOT IN (SELECT id FROM checkpoints ORDER BY id DESC LIMIT ?)"
PRUNE_CANDLES = "DELETE FROM candles WHERE ts < ?"
UPSERT_STATUS = ("INSERT INTO runtime_status (id, ts, feed_state, best_bid, best_ask, mid, living, last_checkpoint_ts, "
                 "rss_mb, rss_peak_mb) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                 "ts = excluded.ts, feed_state = excluded.feed_state, best_bid = excluded.best_bid, "
                 "best_ask = excluded.best_ask, mid = excluded.mid, living = excluded.living, "
                 "last_checkpoint_ts = excluded.last_checkpoint_ts, rss_mb = excluded.rss_mb, "
                 "rss_peak_mb = excluded.rss_peak_mb")


def bot_row(bot: Bot, master_seed: int) -> tuple:
    return (bot.bot_id, bot.slot, bot.parent_id, bot.generation, iso(bot.born_ms), bot.state, bot.genes.margin_bps,
            bot.genes.lr, master_seed, bot.origin)


def lineage_row(d: dict) -> tuple:
    return (d["bot_id"], d["parent_id"], d["generation"], iso(d["born_ms"]), d["sigma_weights"], d["sigma_genes"],
            d["margin_factor"], d["lr_factor"], d["parent_margin_bps"], d["parent_lr"], d["parent_fitness"],
            d["parent_skill_corr"])


def decision_row(d: dict) -> tuple:
    return (d["bot_id"], iso(d["ts"]), d["position_before"], d["ref_mid"], d["prediction_bps"],
            d["required_edge_bps"], d["cost_estimate_bps"], d["margin_bps"], d["gate_passed"], d["forced"],
            d["action"], d["reason"])


def fill_row(f: Fill) -> tuple:
    return (f.bot_id, f.trade_id, iso(f.decided_ms), None if f.filled_ms is None else iso(f.filled_ms), f.leg,
            f.taker_side, f.qty, f.ref_mid, f.book_avg_price, f.price, f.levels, f.slippage_bps, f.fee,
            f.latency_ms, f.status, f.reason)


def open_trade_row(bot_id: int, acct: Account) -> tuple:
    """The trade row written when a position opens, read from the account right after the fill."""
    return (acct.trade_id, bot_id, POSITION_NAMES[acct.side], iso(acct.entry_ms), acct.qty, acct.entry_price,
            acct.entry_fee, acct.position_funding, acct.entry_slippage_bps, int(acct.forced))


def close_trade_row(c: ClosedTrade) -> tuple:
    return (iso(c.closed_ms), c.exit_price, c.fee_close, c.funding, c.slippage_bps_close, c.gross, c.net,
            c.exit_reason, c.trade_id)


def snapshot_row(bot: Bot, ms: int, mark: float | None) -> tuple:
    acct, stats = bot.account, bot.skill.stats()
    return (bot.bot_id, iso(ms), acct.cash, acct.equity(mark), acct.unrealized(mark), acct.realized_gross, acct.fees,
            acct.funding, acct.rent, acct.stop_loss_losses, POSITION_NAMES[acct.side], stats.corr, stats.dir_acc,
            stats.mae, stats.n)


def death_row(bot: Bot, ms: int) -> tuple:
    """Written once the bot is flat, after any negative cash has become a writeoff."""
    acct = bot.account
    return (bot.bot_id, iso(ms), bot.generation, acct.death_cause(), acct.fees, acct.funding, acct.rent,
            acct.stop_loss_losses, acct.other_losses, acct.writeoff, bot.death_threshold, acct.equity(None),
            acct.trades, (ms - bot.born_ms) / MS_PER_HOUR)


def candle_row(c: Candle) -> tuple:
    return (iso(c.ts), c.open, c.high, c.low, c.close, c.volume, c.trade_count, c.mid, int(c.valid))
