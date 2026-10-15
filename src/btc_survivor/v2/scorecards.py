"""The chooser scorecard of PLAN_V2 section 11.8 and the day scorecard of section 11.9, read only and descriptive.

Both come from rows the runs already store, stock_forecasts, stock_labels and stock_candles, with the session calendar
of v2/sessions.py, the table the sessions rows are written from, and each run's settings as its meta recorded them at
launch. They change nothing S1 or S2 does, carry no pass label and never decide a criterion, a verdict or an exit code.
evaluate and the private dashboard import this module only inside the functions that use it, because the console
script that starts the stock and news processes imports both of them, and neither process may ever load it.

The chooser. At every evaluation minute t of the window whose label ended by the end of the window, each symbol with a
forecast f of the predictor at t has a cost c, the round trip in bps of a position on the side of f over the horizon
from costs.round_trip with the run's own cost settings, for the quantity the broker's open_qty rule trades for the
notional at the symbol's reference price at t, so a short trades whole shares and pays its borrow. Its scale is the
standard deviation of its matured labels whose exit is at or before t, over the last 20 full sessions, and a symbol
with fewer than 60 such labels is skipped. The chooser picks the largest (|f| - c) / scale above zero, the earlier
symbol in config order on a tie, on the side of f, and stays flat when no edge is above zero. A pick earns the side
times its label minus c, and a pick whose label ended as gap earns nothing and is counted. It is set beside a seeded
random pick, every symbol with a forecast traded together, the largest |f| - c without the scale, the index symbol held
long and flat, at the same minutes, and the chooser minus each of them per minute is judged with the version 1
newey_west_se at the lags recorded at launch, with its one sided 95 percent upper bound.

The day scorecard. For each symbol and session from session 3 to the end of the window, the day label is 1e4 times
the log of the reference price at the flatten boundary, 15:55 ET or 12:55 on an early close, over the one at 10:00,
read from stock_candles. It counts when both boundaries have a reference price and no stretch of the symbol's minutes
between them is invalid for as long as the gap flatten needs, 10 minutes at max_resume_gap_minutes of 10. Each call,
taken only from rows at or before 10:00, trades the notional on its side from 10:00 to the flatten and earns the side
times the day label minus the round trip over those 355 or 175 minutes, and a call of zero or a missing call is flat.
Each call's correlation with the day label is pooled as in section 11.2 with its error from the version 1 block_test
over one block per session, never the 12 lag error. Every figure is given with early closes and without them.

Where the plan leaves a detail open the simplest reading is taken, as follows.

* The chooser covers the window of evaluate, from session 2 to the cutoff, and a minute counts once its label ended.
* The last 20 full sessions are the 7,800 session minutes up to t, counted on the session minute index as the cutoff
  counts 40 full sessions as 15,600 minutes, so the labels of t's own session that exited by t count and an early
  close counts at its real length. The scale is the population standard deviation, as evaluate's label scale is, and a
  scale of zero is skipped like a short history.
* A minute at which no symbol has a scale yet, in the first sessions, is flat by step 4 and is counted apart. It is
  left out of the chooser minus each comparison, as the plan fixed on 15 October 2026, because the chooser could not
  act there while every comparison paid its cost, which would flatter the chooser by about one cost a minute.
* The largest raw forecast ranks |f| - c over every symbol with a forecast, needing no scale, and is flat when none is
  above zero. The random pick draws a symbol from those with a forecast and then a side, both uniform.
* A forecast of exactly zero has no side, so it is flat wherever it would trade, and it is costed as a long.
* A right pick or call is one whose side times its label is above zero, before costs.
* A minute or symbol day enters a mean or a difference only when its value is known, flat counting as zero and a pick
  on a gap label leaving its minute out. Every symbol with a forecast traded together averages the symbols whose result
  is known, a zero forecast counting as zero and a gap label left out.
* A day boundary has a reference price when its candle is valid with one, as a matured stock_labels row takes its
  prices. The previous session is the calendar's, so with no valid candle in it momentum makes no call.
* Momentum's correlation uses its move in bps, as the forecasts' correlations use their values, and the day chooser's
  uses the side of its one pick a session.
* A forecast row whose newest input was received after its minute, a lookahead the audit of criterion 1.2 fails, is
  never used, and late_forecasts counts those rows so the reports show how many were left out.
* The day label enters at 10:00, session minute 30, which the plan fixes apart from the forecast horizon, so the day
  scorecard never reads its entry from horizon_minutes.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from btc_survivor.cli.stats import block_test
from btc_survivor.clock import iso, parse_iso
from btc_survivor.types import FLAT, LONG, SHORT
from btc_survivor.v2 import sessions, stats
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.sessions import FULL_SESSION_MINUTES
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks.broker import ORDER_JITTER, StockBroker
from btc_survivor.v2.stocks.labels import MATURED

SCALE_MINUTES = 20 * FULL_SESSION_MINUTES  # the last 20 full sessions, in session minutes
MIN_SCALE_LABELS = 60
FIRST_DAY_SESSION = 3  # the day scorecard's first session, once S1 and S2 have forecasts at 10:00
DAY_ENTRY_MINUTE = 30  # 10:00 ET, the day label's entry and the minute of every day call, fixed by section 11.9
# Seed tags after the order jitter tag of broker.py and the version 1 tags of seeds.py.
CHOOSER_PICK, DAY_COIN = ORDER_JITTER + 1, ORDER_JITTER + 2
BPS_PREDICTORS = ("population", "linear_prices", "linear_news")
NEWS_PREDICTOR = "news"
CHOOSER, RANDOM, ALL, RAW, FLAT_NAME = "chooser", "random pick", "all symbols", "largest raw forecast", "flat"
MOMENTUM, HOLD, COIN = "morning momentum", "hold long", "seeded coin"
NO_ENTRY, NO_EXIT, STRETCH = "no entry price", "no exit price", "invalid stretch"
SCOPES = ("with early closes", "without early closes")


@dataclass(frozen=True)
class Source:
    """One run's read only connection, the name a report gives it and the config it recorded at launch."""

    name: str
    conn: sqlite3.Connection
    config: StockConfig


# Shared pieces


def _tag(text: str) -> int:
    """A fixed integer for a name, such as a run or a symbol, for a seed sequence."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def side_of(value: float) -> int:
    """The side a forecast or a move points to, flat for exactly zero."""
    return LONG if value > 0 else SHORT if value < 0 else FLAT


@functools.lru_cache(maxsize=None)
def _k(ms: int) -> int:
    """The session minute index of a boundary that a row was written at."""
    k = sessions.k_at(ms)
    if k is None:
        raise ValueError(f"{iso(ms)} is not a session minute boundary")
    return k


def trip_bps(config: StockConfig, symbol: str, side: int, price: float, minutes: int) -> float:
    """The round trip in bps of the position the run opens on side for its notional at price, held minutes, with the
    quantity rule of the broker, a long rounded down to its decimals and a short in whole shares, a zero side costed
    as a long."""
    side = SHORT if side == SHORT else LONG
    qty = StockBroker(config.stock_costs, config.trading).open_qty(side, price)
    return costs.round_trip(config.stock_costs, symbol, side, qty, price, minutes).total_bps


def result(side: int, label: float, cost: float) -> float:
    """What a position earns in bps, the side times the label less its cost, nothing when flat."""
    return 0.0 if side == FLAT else side * label - cost


class Scales:
    """Each symbol's matured labels by the session minute index of their exit, for the scale at a minute."""

    def __init__(self, matured: dict[str, list[tuple[int, float]]]) -> None:
        self._by: dict[str, tuple[list[int], np.ndarray]] = {}
        for symbol, rows in matured.items():
            rows = sorted(rows)
            self._by[symbol] = ([k for k, _ in rows], np.array([v for _, v in rows], dtype=float))

    def count(self, symbol: str, k: int) -> int:
        """How many matured labels of the symbol exited at or before k within the last 20 full sessions."""
        exits, _ = self._by.get(symbol, ([], None))
        return bisect.bisect_right(exits, k) - bisect.bisect_right(exits, k - SCALE_MINUTES)

    def at(self, symbol: str, k: int) -> float | None:
        """The standard deviation of those labels, or None with fewer than 60 of them or none that differ."""
        if self.count(symbol, k) < MIN_SCALE_LABELS:
            return None
        exits, values = self._by[symbol]
        chosen = values[bisect.bisect_right(exits, k - SCALE_MINUTES):bisect.bisect_right(exits, k)]
        sd = float(np.std(chosen))
        return sd if sd > 0 else None


LABELS = "SELECT ts, symbol, entry_price, status, label_bps FROM stock_labels WHERE ts <= ?"


@dataclass(frozen=True)
class Labels:
    """A run's stock_labels rows up to the end, each with its entry price and its label when matured, and the scales
    from the matured ones."""

    rows: dict[tuple[int, str], tuple[float, float | None]]
    scales: Scales


def read_labels(conn: sqlite3.Connection, end: str, horizon: int) -> Labels:
    """Every label row at or before the end, a label counting only once matured, its exit horizon minutes later."""
    rows: dict[tuple[int, str], tuple[float, float | None]] = {}
    matured: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for ts, symbol, entry, status, value in conn.execute(LABELS, (end,)):
        ms = parse_iso(ts)
        done = status == MATURED and value is not None
        rows[ms, symbol] = (entry, value if done else None)
        if done:
            matured[symbol].append((_k(ms) + horizon, value))
    return Labels(rows, Scales(matured))


FORECASTS = ("SELECT ts, symbol, predictor, prediction, inputs_max_recv_at FROM stock_forecasts "
             "WHERE ts > ? AND ts <= ? AND predictor IN ({})")


def read_forecasts(conn: sqlite3.Connection, predictors: Sequence[str], after: str,
                   until: str) -> dict[str, dict[int, dict[str, float]]]:
    """Each predictor's forecasts after one time and at or before another by minute and symbol, leaving out a row
    whose newest input was received after its minute, which late_forecasts counts."""
    out: dict[str, dict[int, dict[str, float]]] = {p: defaultdict(dict) for p in predictors}
    sql = FORECASTS.format(", ".join("?" * len(predictors)))
    for ts, symbol, predictor, value, received in conn.execute(sql, (after, until, *predictors)):
        if received is None or received <= ts:
            out[predictor][parse_iso(ts)][symbol] = value
    return out


LATE = ("SELECT count(*) FROM stock_forecasts WHERE ts > ? AND ts <= ? AND predictor IN ({}) "
        "AND inputs_max_recv_at IS NOT NULL AND inputs_max_recv_at > ts")


def scored_predictors(config: StockConfig) -> tuple[str, ...]:
    """Every predictor either scorecard reads from a run, its bps predictors and in the news run its news score."""
    return chooser_predictors(config) + ((NEWS_PREDICTOR,) if config.news_input.use_news else ())


def late_forecasts(source: Source, after: str, until: str) -> int:
    """How many forecast rows of the predictors the scorecards read, after one time and at or before another, they
    leave out because the newest input was received after the row's minute. Criterion 1.2 fails any such row, and
    this shows how many the scorecards did without."""
    predictors = scored_predictors(source.config)
    sql = LATE.format(", ".join("?" * len(predictors)))
    return int(source.conn.execute(sql, (after, until, *predictors)).fetchone()[0])


def _mean(total: float, n: int) -> float | None:
    return total / n if n else None


# The chooser of section 11.8


@dataclass
class Tally:
    """One strategy over the minutes, its per minute values, and the trades it made with their results."""

    name: str
    minutes: int = 0  # minutes with a value, flat ones included
    flat: int = 0
    picks: int = 0  # trades made, those on a gap label included
    no_result: int = 0  # trades whose label ended as gap
    right: int = 0
    total: float = 0.0  # the running total of the per minute values, bps
    traded: float = 0.0  # the sum of the trades' results, bps
    by_symbol: Counter[str] | None = None

    def trade(self, symbol: str, side: int, label: float | None, cost: float) -> float | None:
        """Count one trade and give its result, or None on a gap label."""
        self.picks += 1
        if self.by_symbol is not None:
            self.by_symbol[symbol] += 1
        if label is None:
            self.no_result += 1
            return None
        self.right += side * label > 0
        earned = result(side, label, cost)
        self.traded += earned
        return earned

    def minute(self, value: float | None, flat: bool = False) -> float | None:
        """Count one minute's value, None when it is unknown."""
        self.flat += flat
        if value is not None:
            self.minutes += 1
            self.total += value
        return value

    def row(self) -> dict:
        scored = self.picks - self.no_result
        return {"strategy": self.name, "minutes": self.minutes, "picks": self.picks, "flat": self.flat,
                "no result": self.no_result, "right share": None if not scored else self.right / scored,
                "mean per pick bps": _mean(self.traded, scored),
                "mean per minute bps": _mean(self.total, self.minutes), "total bps": self.total}


@dataclass(frozen=True)
class Offer:
    """One symbol with a forecast at a minute, its side, cost, scale and label."""

    symbol: str
    forecast: float
    side: int
    cost: float
    scale: float | None
    label: float | None


def best(offers: Sequence[Offer], edge: Callable[[Offer], float | None]) -> Offer | None:
    """The offer with the largest edge above zero, the earlier one in config order on a tie, or None."""
    chosen, top = None, 0.0
    for offer in offers:
        value = edge(offer)
        if value is not None and value > top:
            chosen, top = offer, value
    return chosen


def scaled_edge(offer: Offer) -> float | None:
    """Step 3 of section 11.8, |f| - c over the scale, or None for a skipped symbol."""
    return None if offer.scale is None else (abs(offer.forecast) - offer.cost) / offer.scale


def raw_edge(offer: Offer) -> float:
    """The same edge in bps without the scale."""
    return abs(offer.forecast) - offer.cost


def random_pick(master_seed: int, run: str, minute_ms: int, offers: Sequence[Offer]) -> tuple[Offer, int]:
    """A symbol and then a side drawn uniformly from a generator seeded by the master seed, the run and the minute, so
    every evaluate draws the same."""
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, CHOOSER_PICK, _tag(run), minute_ms]))
    offer = offers[int(rng.integers(len(offers)))]
    return offer, LONG if rng.integers(2) else SHORT


@dataclass(frozen=True)
class ChooserCard:
    """One run's chooser on one bps predictor beside its comparisons."""

    run: str
    predictor: str
    minutes: int  # evaluation minutes with a forecast whose label ended by the end
    unscaled_minutes: int  # minutes at which no symbol with a forecast had a scale yet
    skipped: int  # symbol minutes skipped for fewer than 60 labels or no spread
    strategies: tuple[dict, ...]
    differences: tuple[dict, ...]
    picks_per_symbol: dict[str, int]
    # Each minute's pick of the chooser, the random pick and the largest raw forecast, as (minute, symbol, side), a
    # flat minute with no symbol, for checking the rule minute by minute.
    trail: dict[str, tuple[tuple[int, str | None, int], ...]]

    @property
    def picks(self) -> int:
        return self.strategies[0]["picks"]

    @property
    def flat(self) -> int:
        return self.strategies[0]["flat"]


def chooser_predictors(config: StockConfig) -> tuple[str, ...]:
    """The bps predictors a run writes, population and linear_prices, and linear_news in the news run."""
    return BPS_PREDICTORS if config.news_input.use_news else BPS_PREDICTORS[:2]


def _exit_ms(ms: int, horizon: int) -> int:
    return sessions.boundary_of_k(_k(ms) + horizon)


def chooser_card(source: Source, predictor: str, forecasts: dict[int, dict[str, float]], labels: Labels, end: str,
                 lags: int, upper_z: float) -> ChooserCard:
    """Section 11.8 for one run and predictor at every minute of forecasts whose label ended by the end."""
    cfg = source.config
    symbols, horizon, index = cfg.stock_feed.symbols, cfg.time.horizon_minutes, cfg.stock_feed.index_symbol
    end_ms = parse_iso(end)
    names = (CHOOSER, RANDOM, ALL, RAW, f"{index} long", FLAT_NAME)
    tallies = {name: Tally(name, by_symbol=Counter() if name == CHOOSER else None) for name in names}
    series: dict[str, list[float | None]] = {name: [] for name in names}
    trail: dict[str, list[tuple[int, str | None, int]]] = {name: [] for name in (CHOOSER, RANDOM, RAW)}
    minutes = unscaled = skipped = 0
    for ms in sorted(forecasts):
        if _exit_ms(ms, horizon) > end_ms:
            continue
        k = _k(ms)
        offers = []
        for symbol in symbols:
            if symbol in forecasts[ms] and (ms, symbol) in labels.rows:
                f = forecasts[ms][symbol]
                entry, label = labels.rows[ms, symbol]
                scale = labels.scales.at(symbol, k)
                skipped += scale is None
                offers.append(Offer(symbol, f, side_of(f), trip_bps(cfg, symbol, side_of(f), entry, horizon), scale,
                                    label))
        if not offers:
            continue
        minutes += 1
        scaled = any(o.scale is not None for o in offers)
        unscaled += not scaled
        values, picked = _choose(tallies, names, offers, cfg, ms, labels, horizon)
        if scaled:  # the chooser could not act at a minute with no scale, so it is left out of the differences
            for name in names:
                series[name].append(values[name])
        for name, (symbol, side) in picked.items():
            trail[name].append((ms, symbol, side))
    return ChooserCard(source.name, predictor, minutes, unscaled, skipped,
                       tuple(tallies[name].row() for name in names),
                       tuple(_difference(name, series[CHOOSER], series[name], lags, upper_z) for name in names[1:]),
                       {symbol: tallies[CHOOSER].by_symbol[symbol] for symbol in symbols},
                       {name: tuple(rows) for name, rows in trail.items()})


def _choose(tallies: dict[str, Tally], names: Sequence[str], offers: Sequence[Offer], cfg: StockConfig, ms: int,
            labels: Labels, horizon: int) -> tuple[dict[str, float | None], dict[str, tuple[str | None, int]]]:
    """Every strategy's value at one minute, counted in its tally, and the picks of those that pick."""
    chooser, random_name, all_name, raw_name, index_name, flat_name = names
    out: dict[str, float | None] = {}
    picked: dict[str, tuple[str | None, int]] = {}
    for name, offer in ((chooser, best(offers, scaled_edge)), (raw_name, best(offers, raw_edge))):
        tally = tallies[name]
        out[name] = tally.minute(0.0, flat=True) if offer is None else \
            tally.minute(tally.trade(offer.symbol, offer.side, offer.label, offer.cost))
        picked[name] = (None, FLAT) if offer is None else (offer.symbol, offer.side)
    drawn, side = random_pick(cfg.run.master_seed, cfg.run.name, ms, offers)
    cost = drawn.cost if side == drawn.side else trip_bps(cfg, drawn.symbol, side, labels.rows[ms, drawn.symbol][0],
                                                          horizon)
    out[random_name] = tallies[random_name].minute(tallies[random_name].trade(drawn.symbol, side, drawn.label, cost))
    picked[random_name] = (drawn.symbol, side)
    together, known = tallies[all_name], []
    for o in offers:  # a zero forecast is flat and earns nothing whatever its label
        earned = 0.0 if o.side == FLAT else together.trade(o.symbol, o.side, o.label, o.cost)
        if earned is not None:
            known.append(earned)
    out[all_name] = together.minute(sum(known) / len(known) if known else None)
    index = cfg.stock_feed.index_symbol
    entry, label = labels.rows.get((ms, index), (None, None))
    held = tallies[index_name]
    out[index_name] = held.minute(None if entry is None else held.trade(
        index, LONG, label, trip_bps(cfg, index, LONG, entry, horizon)))
    out[flat_name] = tallies[flat_name].minute(0.0, flat=True)
    return out, picked


def _difference(name: str, chooser: Sequence[float | None], other: Sequence[float | None], lags: int,
                upper_z: float) -> dict:
    """The chooser minus one comparison at the minutes both have a value, its mean with the version 1 newey_west_se
    and the one sided 95 percent upper bound."""
    diffs = np.array([a - b for a, b in zip(chooser, other) if a is not None and b is not None], dtype=float)
    judged = stats.clustered(diffs, lags)
    mean, se, t = judged if judged is not None else (None, None, None)
    return {"comparison": name, "minutes": len(diffs), "mean bps": mean, "se": se, "t": t,
            "upper bound": None if se is None else mean + upper_z * se}


def chooser_cards(source: Source, start: str, end: str, lags: int, upper_z: float) -> list[ChooserCard]:
    """The chooser of section 11.8 on every bps predictor of one run over the window after start and up to end."""
    predictors = chooser_predictors(source.config)
    forecasts = read_forecasts(source.conn, predictors, start, end)
    labels = read_labels(source.conn, end, source.config.time.horizon_minutes)
    return [chooser_card(source, p, forecasts[p], labels, end, lags, upper_z) for p in predictors]


# The day scorecard of section 11.9


@dataclass(frozen=True)
class DayLabel:
    """One symbol and session from 10:00 to the flatten, its label or why it failed, and the morning move."""

    number: int  # the session counted from launch as 1
    day: date
    early: bool
    symbol: str
    entry_minute: int
    minutes: int  # 355, or 175 on an early close
    entry: float | None
    label: float | None
    cause: str | None
    strict: bool  # every minute from entry to exit valid, the stricter rule of stock_labels
    move: float | None  # 1e4 * ln(ref at 10:00 / last valid ref of the previous session)


VALID_REFS = ("SELECT symbol, session_k, ref_price FROM stock_candles WHERE valid = 1 AND ref_price IS NOT NULL "
              "AND ts > ? AND ts <= ?")


def day_sessions(launch: str, end: str, flatten: int) -> list[tuple[int, int]]:
    """The calendar positions of sessions 3 on, counted from the launch session as 1, whose flatten boundary, flatten
    minutes before the close, is at or before the end, each with its number."""
    table = sessions.all_sessions()
    first = table.index(sessions.next_session(parse_iso(launch)))
    end_ms, out = parse_iso(end), []
    for i in range(first + FIRST_DAY_SESSION - 1, len(table)):
        if table[i].open_ms >= end_ms:
            break
        if table[i].boundary(table[i].minutes - flatten) <= end_ms:
            out.append((i, i - first + 1))
    return out


def day_labels(source: Source, launch: str, end: str) -> list[DayLabel]:
    """Every symbol day of sessions 3 to the end from the run's candles, a boundary needing a valid candle with a
    reference price, failing on a stretch of invalid minutes long enough for the gap flatten."""
    cfg = source.config
    flatten = cfg.session.flatten_minutes_before_close
    max_gap = cfg.persistence.max_resume_gap_minutes
    days = day_sessions(launch, end, flatten)
    if not days:
        return []
    table = sessions.all_sessions()
    refs: dict[str, dict[int, float]] = defaultdict(dict)
    for symbol, k, ref in source.conn.execute(VALID_REFS, (iso(table[days[0][0] - 1].open_ms), end)):
        refs[symbol][k] = ref
    order = {symbol: sorted(found) for symbol, found in refs.items()}
    out = []
    for i, number in days:
        session, previous = table[i], table[i - 1]
        entry_k = session.first_k + DAY_ENTRY_MINUTE - 1  # 10:00, never read from the forecast horizon
        exit_k = session.first_k + session.minutes - flatten - 1
        for symbol in cfg.stock_feed.symbols:
            found = refs.get(symbol, {})
            entry, exit_price = found.get(entry_k), found.get(exit_k)
            longest = run = 0
            for k in range(entry_k + 1, exit_k):
                run = 0 if k in found else run + 1
                longest = max(longest, run)
            cause = NO_ENTRY if entry is None else NO_EXIT if exit_price is None else \
                STRETCH if longest + 1 > max_gap else None
            ks = order.get(symbol, [])
            j = bisect.bisect_right(ks, previous.last_k) - 1  # the previous session's last valid minute
            last = found[ks[j]] if j >= 0 and ks[j] >= previous.first_k else None
            out.append(DayLabel(
                number, session.day, session.early_close, symbol, DAY_ENTRY_MINUTE, exit_k - entry_k, entry,
                None if cause else 1e4 * math.log(exit_price / entry), cause,
                all(k in found for k in range(entry_k, exit_k + 1)),
                None if entry is None or last is None else 1e4 * math.log(entry / last)))
    return out


@dataclass(frozen=True)
class Call:
    """One call of the day scorecard, its side and value per session number and symbol, and the run whose cost
    settings price it."""

    name: str
    config: StockConfig
    positions: dict[tuple[int, str], tuple[int, float]]


def forecast_calls(source: Source, days: Sequence[DayLabel]) -> list[Call]:
    """The run's bps predictors at 10:00, and in the news run its news score, by sign, with their values for the
    correlation."""
    predictors = scored_predictors(source.config)
    at = {(d.number, d.symbol): sessions.session_on(d.day).boundary(d.entry_minute) for d in days}
    if not at:
        return [Call(f"{source.name} {p}", source.config, {}) for p in predictors]
    found = read_forecasts(source.conn, predictors, iso(min(at.values()) - 1), iso(max(at.values())))
    calls = []
    for p in predictors:
        values = {key: found[p][ms][key[1]] for key, ms in at.items() if key[1] in found[p].get(ms, {})}
        calls.append(Call(f"{source.name} {p}", source.config, {key: (side_of(f), f) for key, f in values.items()}))
    return calls


def day_chooser(source: Source, days: Sequence[DayLabel], end: str) -> Call:
    """Section 11.8 at 10:00 on the news run's population, |f| over the symbol's 30 minute scale with no cost gate,
    one pick a session on the side of its forecast, flat when every forecast is zero."""
    labels = read_labels(source.conn, end, source.config.time.horizon_minutes)
    population = forecast_calls(source, days)[0].positions
    by_session: dict[int, list[DayLabel]] = defaultdict(list)
    for d in days:
        by_session[d.number].append(d)
    positions = {}
    for number, group in sorted(by_session.items()):
        k = sessions.session_on(group[0].day).first_k + DAY_ENTRY_MINUTE - 1  # 10:00
        offers = [Offer(d.symbol, population[number, d.symbol][1], population[number, d.symbol][0], 0.0,
                        labels.scales.at(d.symbol, k), None) for d in group if (number, d.symbol) in population]
        pick = best(offers, lambda o: None if o.scale is None else abs(o.forecast) / o.scale)
        if pick is not None:
            positions[number, pick.symbol] = (pick.side, float(pick.side))
    return Call(f"{source.name} {CHOOSER}", source.config, positions)


def market_calls(source: Source, days: Sequence[DayLabel]) -> list[Call]:
    """Morning momentum, every symbol held long and a seeded coin per symbol and session from the master seed."""
    seed = source.config.run.master_seed

    def coin(d: DayLabel) -> int:
        session = sessions.session_on(d.day)
        rng = np.random.default_rng(np.random.SeedSequence([seed, DAY_COIN, session.first_k, _tag(d.symbol)]))
        return LONG if rng.integers(2) else SHORT

    return [Call(MOMENTUM, source.config, {(d.number, d.symbol): (side_of(d.move), d.move) for d in days
                                           if d.move is not None}),
            Call(HOLD, source.config, {(d.number, d.symbol): (LONG, 1.0) for d in days}),
            Call(COIN, source.config, {(d.number, d.symbol): (coin(d), float(coin(d))) for d in days}),
            Call(FLAT_NAME, source.config, {})]


def block_skill(numbers: Sequence[int], symbols: Sequence[str], values: Sequence[float], labels: Sequence[float],
                upper_z: float) -> dict:
    """The pooled correlation of section 11.2 on one row per session and symbol, z scores within each symbol and
    their products averaged per session, with its error from the version 1 block_test over one block per session."""
    out = {"rows": 0, "sessions": 0, "corr": None, "se": None, "t": None, "upper bound": None, "left out": ""}
    if not numbers:
        return out
    _, names, (p, y) = stats.pivot(numbers, symbols, values, labels)
    keep = [j for j in range(len(names)) if stats.varies(p[:, j]) and stats.varies(y[:, j])]
    out["left out"] = " ".join(names[j] for j in range(len(names)) if j not in keep)
    products = stats.zscores(p[:, keep]) * stats.zscores(y[:, keep])
    blocks = {i: [float(v) for v in row[~np.isnan(row)]] for i, row in enumerate(products)}
    blocks = {i: v for i, v in blocks.items() if v}
    test = block_test(blocks)
    means = np.array([np.mean(v) for v in blocks.values()])
    se = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) >= 2 else None
    out.update({"rows": test["pairs"], "sessions": test["blocks"], "corr": test["mean"], "t": test["t"],
                "se": None if test["mean"] is None else se,
                "upper bound": None if test["mean"] is None else test["mean"] + upper_z * se})
    return out


@dataclass(frozen=True)
class DayCard:
    """The day scorecard, its label counts, each call per scope and each call per symbol."""

    sessions: int
    counted: int  # symbol days whose day label counts, with early closes
    failed: int
    labels: tuple[dict, ...]
    calls: tuple[dict, ...]
    symbols: tuple[dict, ...]


def _score(call: Call, days: Sequence[DayLabel], upper_z: float) -> tuple[dict, list[dict]]:
    """One call over the symbol days of one scope, and the same per symbol."""
    per: dict[str, dict] = {}
    rows: list[tuple[int, str, float, float]] = []
    no_result = 0
    for d in days:
        side, value = call.positions.get((d.number, d.symbol), (FLAT, None))
        if d.label is None:
            no_result += side != FLAT
            continue
        tally = per.setdefault(d.symbol, {"symbol days": 0, "calls": 0, "right": 0, "total": 0.0})
        tally["symbol days"] += 1
        if value is not None:
            rows.append((d.number, d.symbol, value, d.label))
        if side != FLAT:
            tally["calls"] += 1
            tally["right"] += side * d.label > 0
            tally["total"] += result(side, d.label, trip_bps(call.config, d.symbol, side, d.entry, d.minutes))

    def figures(t: dict) -> dict:
        return {"calls": t["calls"], "right share": _mean(t["right"], t["calls"]),
                "mean per call bps": _mean(t["total"], t["calls"]), "total bps": t["total"]}

    whole = {k: sum(t[k] for t in per.values()) for k in ("symbol days", "calls", "right", "total")}
    columns = [list(c) for c in zip(*rows)] if rows else [[], [], [], []]
    row = {"call": call.name, "days": len({d.number for d in days if d.label is not None}),
           "symbol days": whole["symbol days"], **figures(whole), "no result": no_result,
           **block_skill(*columns, upper_z=upper_z)}
    return row, [{"call": call.name, "symbol": s, "symbol days": t["symbol days"], **figures(t)}
                 for s, t in per.items()]


def day_card(sources: Sequence[Source], launch: str, end: str, upper_z: float) -> DayCard:
    """Section 11.9 from sessions 3 to the end, on the day labels of the news run's candles, which both runs share,
    or the prices run's when only it is given. The calls are each run's bps forecasts, the news run's news score and
    its chooser, morning momentum, holding each symbol long, a seeded coin and flat."""
    if not sources:
        return DayCard(0, 0, 0, (), (), ())
    market = next((s for s in sources if s.config.news_input.use_news), sources[-1])
    days = day_labels(market, launch, end)
    calls = [call for s in sources for call in forecast_calls(s, days)]
    calls += [day_chooser(market, days, end)] if market.config.news_input.use_news else []
    calls += market_calls(market, days)
    label_rows, call_rows, symbol_rows = [], [], []
    for scope in SCOPES:
        chosen = [d for d in days if scope == SCOPES[0] or not d.early]
        causes = Counter(d.cause for d in chosen)
        label_rows.append({"scope": scope, "sessions": len({d.number for d in chosen}), "symbol days": len(chosen),
                           "counted": causes[None], NO_ENTRY: causes[NO_ENTRY], NO_EXIT: causes[NO_EXIT],
                           STRETCH: causes[STRETCH], "every minute valid": sum(d.strict for d in chosen)})
        for call in calls:
            row, per_symbol = _score(call, chosen, upper_z)
            call_rows.append({"scope": scope, **row})
            symbol_rows += [{"scope": scope, **r} for r in per_symbol]
    first = label_rows[0]
    return DayCard(first["sessions"], first["counted"], first["symbol days"] - first["counted"], tuple(label_rows),
                   tuple(call_rows), tuple(symbol_rows))
