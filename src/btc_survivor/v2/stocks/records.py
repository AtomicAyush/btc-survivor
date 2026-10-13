"""Row builders for a stock run database of PLAN_V2 section 9.2, so the engines decide only what to write and when.

Times go in as UTC milliseconds and are written as ISO 8601 text with milliseconds and a trailing Z, as
btc_survivor.clock.iso writes them, and flags as 0 or 1. Where a version 1 table kept its shape, its statement and
builder are the version 1 ones, imported here so every writer of a run database takes its statements from this one
module. Comma separated lists, the tickers of a score line and the channels in outage, are written as news.db holds
them. The engine writes every fill with its two report only columns, while INSERT_FILL and fill_row write the section
9.2 columns alone and leave those two null.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Any

from btc_survivor import records as v1
from btc_survivor.account import Account
from btc_survivor.clock import iso
# Unchanged from version 1. A bot's state, a decision's label and a trade's close touch only version 1 columns, and
# equity_snapshots, deaths and checkpoints keep their shape.
from btc_survivor.records import (CLOSE_TRADE, INSERT_CHECKPOINT, INSERT_DEATH, INSERT_SNAPSHOT, PRUNE_CHECKPOINTS,
                                  SET_BOT_DEAD, SET_BOT_STATE, SET_LABEL, close_trade_row, death_row, snapshot_row)
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.redact import redact
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES

if TYPE_CHECKING:
    from btc_survivor.v2.stocks.broker import StockFill
    from btc_survivor.v2.stocks.labels import LabelRow

PENDING, MATURED, GAP = "pending", "matured", "gap"
LOCAL_ONLY = "local_only"
# The meta keys a run copies from news.db at launch, by its own name and by the news process's name.
NEWS_META = {"news_config_hash": "config_hash", "prompt_sha256": "prompt_sha256", "schema_sha256": "schema_sha256",
             "haiku_model": "haiku_model", "sonnet_model": "sonnet_model"}

INSERT_BOT = ("INSERT INTO bots (bot_id, slot, parent_id, generation, born_at, state, margin_bps, lr, seed, origin, "
              "home_symbol) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_DECISION = ("INSERT INTO decisions (bot_id, ts, position_before, ref_mid, prediction_bps, required_edge_bps, "
                   "cost_estimate_bps, margin_bps, gate_passed, forced, action, reason, symbol) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_FILL = ("INSERT INTO fills (bot_id, trade_id, decided_at, filled_at, leg, side, qty, ref_mid, book_avg_price, "
               "price, levels_consumed, slippage_bps, fee, latency_ms, status, reason, symbol, print_price, "
               "allowance_bps, notional, sec_fee, taf_fee) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_FILL_WITH_REPORTS = ("INSERT INTO fills (bot_id, trade_id, decided_at, filled_at, leg, side, qty, ref_mid, "
                            "book_avg_price, price, levels_consumed, slippage_bps, fee, latency_ms, status, reason, "
                            "symbol, print_price, allowance_bps, notional, sec_fee, taf_fee, print_after_due_ms, "
                            "news_window) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
OPEN_TRADE = ("INSERT INTO trades (id, bot_id, side, opened_at, qty, entry_price, fee_open, funding, "
              "slippage_bps_open, forced, symbol) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_FEED_EVENT = "INSERT INTO feed_events (ts, kind, symbol, detail) VALUES (?, ?, ?, ?)"
INSERT_CANDLE = ("INSERT INTO stock_candles (symbol, ts, session_date, session_k, open, high, low, close, ref_price, "
                 "volume, buy_volume, sell_volume, prints, price_prints, late_prints, valid) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
PRUNE_CANDLES = "DELETE FROM stock_candles WHERE ts < ?"
# The z score columns of price_z, the 10 price features in the order features.py computes them.
PRICE_Z_COLUMNS = ("z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10")
INSERT_PRICE_Z = ("INSERT INTO price_z (k, symbol, ts, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
PUT_SESSION = ("INSERT OR REPLACE INTO sessions (session_date, open_ts, close_ts, early_close, healthy_minutes, "
               "run_minutes) VALUES (?, ?, ?, ?, ?, ?)")
INSERT_LABEL = f"INSERT INTO stock_labels (ts, symbol, entry_price, status) VALUES (?, ?, ?, '{PENDING}')"
SETTLE_LABEL = (f"UPDATE stock_labels SET exit_price = ?, label_bps = ?, status = ? "
                f"WHERE ts = ? AND symbol = ? AND status = '{PENDING}'")
INSERT_FORECAST = ("INSERT INTO stock_forecasts (ts, symbol, predictor, prediction, items, inputs_max_recv_at) "
                   "VALUES (?, ?, ?, ?, ?, ?)")
INSERT_NEWS_SEEN = ("INSERT INTO news_seen (score_seq, item_seq, recv_at, scored_at, source, route, scorer, status, "
                    "tickers, primary_use, fallback) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
INSERT_NEWS_MINUTE = "INSERT INTO news_minutes (ts, healthy, reason, channels_down) VALUES (?, ?, ?, ?)"
PUT_SPREAD_CHECK = ("INSERT OR REPLACE INTO spread_checks (session_date, symbol, prints, roll_half_spread_bps, "
                    "allowance_bps, flagged) VALUES (?, ?, ?, ?, ?, ?)")
INSERT_BENCHMARK = ("INSERT INTO benchmark (ts, ew_buy_hold, qqq_buy_hold, matched_equity, living_mean_equity, "
                    "living_median_equity, living, twin_mean_equity, cash_equity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
PUT_RUNTIME_STATUS = ("INSERT OR REPLACE INTO runtime_status (id, ts, feed_state, session_state, prints_per_s, "
                      "msgs_per_s, news_age_s, living, last_checkpoint_ts, rss_mb, rss_peak_mb) "
                      "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
PUT_SYMBOL_STATUS = ("INSERT OR REPLACE INTO symbol_status (symbol, ts, state, ref_price, last_print_age_s) "
                     "VALUES (?, ?, ?, ?, ?)")


def _iso(ms: int | None) -> str | None:
    """The ISO text of a time that may be missing."""
    return None if ms is None else iso(ms)


def _csv(values: str | Sequence[str]) -> str:
    """A list as comma separated text, or text already in that form unchanged."""
    return values if isinstance(values, str) else ",".join(values)


def bot_row(*, bot_id: int, slot: int, home_symbol: str, born_ms: int, state: str, margin_sigma: float, lr: float,
            seed: int, origin: str, parent_id: int | None = None) -> tuple:
    """A learner or twin at birth. Without evolution the generation is always 0, and margin_bps keeps its version 1
    name while it holds the fixed margin in label standard deviations."""
    return (bot_id, slot, parent_id, 0, iso(born_ms), state, margin_sigma, lr, seed, origin, home_symbol)


def decision_row(d: Mapping[str, Any]) -> tuple:
    """A version 1 decision dict with ts in milliseconds plus its symbol, always the bot's home symbol."""
    return (*v1.decision_row(d), d["symbol"])


def fill_row(f: StockFill) -> tuple:
    """A stock fill, the version 1 Fill fields with no book, so book_avg_price and levels_consumed stay null, plus
    the stock columns. The print price is the real print before the allowance moved it, and the fees are null on a
    rejection."""
    return (*v1.fill_row(f), f.symbol, f.print_price, f.allowance_bps, f.notional, f.sec_fee, f.taf_fee)


def fill_row_with_reports(f: StockFill, news_window: bool | None) -> tuple:
    """A stock fill with its report only columns, how long after the due time its print traded, null on a rejection,
    and whether S2 received the first line of an item tagged with its symbol in the 30 minutes before its decision,
    null in S1, which reads no news."""
    return (*fill_row(f), f.print_after_due_ms, None if news_window is None else int(news_window))


def open_trade_row(bot_id: int, symbol: str, acct: Account) -> tuple:
    """The trade row written when a position opens, read from the account right after the fill."""
    return (*v1.open_trade_row(bot_id, acct), symbol)


def feed_event_row(ms: int, kind: str, symbol: str | None = None, detail: str | None = None) -> tuple:
    """One feed event with symbol None for a whole feed event, its detail redacted. A stock health event, whose
    fields come in this order, is written as feed_event_row(*event)."""
    return (iso(ms), kind, symbol, redact(detail) if detail else None)


def local_only_row(ms: int, passed: bool, offending_sockets: int, tunnel_processes: int) -> tuple:
    """The local_only event of section 13.2 with its verdict and counts as JSON, which evaluate reads for criterion
    1.8."""
    detail = json.dumps({"passed": passed, "offending_sockets": offending_sockets,
                         "tunnel_processes": tunnel_processes}, sort_keys=True)
    return feed_event_row(ms, LOCAL_ONLY, None, detail)


def candle_row(c: StockCandle) -> tuple:
    """One judged symbol minute with the New York date of its session."""
    return (c.symbol, iso(c.ts), sessions.session_of_k(c.k).day.isoformat(), c.k, c.open, c.high, c.low, c.close,
            c.ref_price, c.volume, c.buy_volume, c.sell_volume, c.prints, c.price_prints, c.late_prints,
            int(c.valid))


def session_row(session: Session, healthy_minutes: int, run_minutes: int) -> tuple:
    """A session with its minute counts, written again whenever they change and at the latest once it closes."""
    return (session.day.isoformat(), iso(session.open_ms), iso(session.close_ms), int(session.early_close),
            healthy_minutes, run_minutes)


def label_statement(row: LabelRow) -> tuple[str, tuple]:
    """The statement and values of one evaluation label, an insert for a pending row at its minute and the settle of
    that row when it ends matured or gap, so a settled row never changes again."""
    if row.status == PENDING:
        return INSERT_LABEL, (iso(row.ts), row.symbol, row.entry_price)
    if row.status not in (MATURED, GAP):
        raise ValueError(f"a label ends matured or gap, not {row.status!r}")
    return SETTLE_LABEL, (row.exit_price, row.label_bps, row.status, iso(row.ts), row.symbol)


def forecast_row(ms: int, symbol: str, predictor: str, prediction: float, items: int | None = None,
                 inputs_max_recv_ms: int | None = None) -> tuple:
    """One predictor's forecast at an evaluation minute, in bps or as a score, with the impacts behind a score and
    the newest receive stamp behind the row, which evaluate requires never to be later than ts."""
    return (iso(ms), symbol, predictor, prediction, items, _iso(inputs_max_recv_ms))


def price_z_row(k: int, ms: int, symbol: str, z: Sequence[float] | None) -> tuple:
    """The 10 price z scores of one usable symbol at the evaluation minute k ending at ms, as the learners and
    linear_prices saw them, all null where the symbol's vector was invalid and the learners saw zeros."""
    values = (None,) * N_PRICE_FEATURES if z is None else tuple(float(v) for v in z)
    if len(values) != N_PRICE_FEATURES:
        raise ValueError(f"price_z holds {N_PRICE_FEATURES} z scores, not {len(values)}")
    return (k, symbol, iso(ms), *values)


def news_seen_row(*, score_seq: int, item_seq: int, recv_ms: int, scored_at: str, source: str, route: str,
                  scorer: str, status: str, tickers: str | Sequence[str], primary_use: bool, fallback: bool) -> tuple:
    """A score line S2 read, counting from its receive stamp, with scored_at as news.db wrote it."""
    return (score_seq, item_seq, iso(recv_ms), scored_at, source, route, scorer, status, _csv(tickers),
            int(primary_use), int(fallback))


def news_minute_row(ms: int, reason: str | None, channels_down: str | Sequence[str]) -> tuple:
    """S2's news health of the minute ending at ms, healthy exactly when there is no reason, with the channels in
    outage from the newest heartbeat."""
    return (iso(ms), int(reason is None), reason, _csv(channels_down))


def spread_check_row(day: date, symbol: str, prints: int, roll_half_spread_bps: float | None,
                     allowance_bps: float) -> tuple:
    """A session's Roll diagnostic for one symbol, flagged when its estimate is above the allowance."""
    flagged = roll_half_spread_bps is not None and roll_half_spread_bps > allowance_bps
    return (day.isoformat(), symbol, prints, roll_half_spread_bps, allowance_bps, int(flagged))


def benchmark_row(ms: int, *, ew_buy_hold: float, qqq_buy_hold: float, matched_equity: float,
                  living_mean_equity: float, living_median_equity: float, living: int,
                  twin_mean_equity: float | None, cash_equity: float) -> tuple:
    """The benchmark equities beside the living learners' mean and median at ms."""
    return (iso(ms), ew_buy_hold, qqq_buy_hold, matched_equity, living_mean_equity, living_median_equity, living,
            twin_mean_equity, cash_equity)


def runtime_status_row(ms: int, *, feed_state: str, session_state: str, prints_per_s: float | None,
                       msgs_per_s: float | None, news_age_s: float | None, living: int,
                       last_checkpoint_ms: int | None, rss_mb: float | None, rss_peak_mb: float | None) -> tuple:
    """The one runtime_status row the heartbeat rewrites each minute."""
    return (iso(ms), feed_state, session_state, prints_per_s, msgs_per_s, news_age_s, living,
            _iso(last_checkpoint_ms), rss_mb, rss_peak_mb)


def symbol_status_row(symbol: str, ms: int, state: str, ref_price: float | None,
                      last_print_age_s: float | None) -> tuple:
    """One symbol's row of symbol_status, rewritten by the heartbeat each minute."""
    return (symbol, iso(ms), state, ref_price, last_print_age_s)


def launch_meta(config: StockConfig, launch_ms: int, news_meta: Mapping[str, str], news_start_seq: int,
                code_commit: str) -> dict[str, str]:
    """The meta values a run records once at launch, from its config, the news.db meta the news process wrote and
    the code commit, for Store.queue_meta. kind and schema_version come with the database, and S1 has no
    [evaluation], so evaluation_json and run_minute_rule are S2's alone."""
    missing = [name for name in NEWS_META.values() if not news_meta.get(name)]
    if missing:
        raise ValueError(f"news.db meta has no {', '.join(missing)}, so the news process has not started on it")
    values = {"run_name": config.run.name, "master_seed": str(config.run.master_seed), "launch_ts": iso(launch_ms),
              "config_toml": config.text, "experiment_hash": config.experiment_hash,
              **{ours: news_meta[theirs] for ours, theirs in NEWS_META.items()},
              "news_start_seq": str(news_start_seq), "code_commit": code_commit,
              "symbols_json": json.dumps(list(config.stock_feed.symbols))}
    if config.evaluation is not None:
        values["evaluation_json"] = json.dumps(dataclasses.asdict(config.evaluation), sort_keys=True)
        values["run_minute_rule"] = config.evaluation.run_minute_rule
    return values
