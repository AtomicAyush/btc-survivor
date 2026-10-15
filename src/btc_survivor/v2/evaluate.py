"""The evaluate command of PLAN_V2 section 11, the success criteria of section 11.6 and the outcomes fixed by arithmetic
of section 12, read from both stock run databases, news.db, the key ledger and the coverage records, all read only.

Settings come from the runs' meta at launch, the [evaluation] section from S2's evaluation_json and every cost, timer
and margin from config_toml, so a config file edited later changes nothing. A run minute is a session minute in which
news health held and at least one symbol was healthy, read from S2's news_minutes and valid candles, since only a
healthy symbol minute is valid. The cutoff is the minute S2 reaches final_healthy_session_minutes run minutes counted
from launch, and the window runs from the first minute of the session after the burn in to the cutoff. A forecast row
counts only when its label matured by the end, the cutoff or before it the newest run minute, so a later run never
changes a result, and before the cutoff every number is descriptive.

The confirmatory claim is a fixed sequence at one familywise level. H1 is predictor news against the label. Only when
H1 passes is H2 run, the partial test of stats.partial_skill on the same rows with news and label residualised on the
10 price z scores S2 stored in price_z, within each symbol, judged with the same clustered error and z value. Criterion
1.1 is coverage stability. The gate session's median volume ratio is the baseline, from the earliest full session with
a coverage record before launch, 12 October 2026, whether or not that session was clean, and the first clean full
session at or after each due session must keep its median ratio within a factor of 2 of it. That session may come
after the cutoff, so criterion 1.1 alone can change on a later run until its record exists. The trading gate has no
margin, and under the holding rule the owner approved on 13 October 2026 a learner keeps a position while its
prediction points the position's way past the holding cost, so the share of learner opens a learner made on its own,
and the round trips per session forced by the timer and made on the learner's own, are reported, never judged. The
founder equity bound grows with the horizons the learners held their positions, counted in session time, since one
round trip can now last several horizons.

By default it prints each criterion's verdict, each outcome matched or not, and counts, never a price, a forecast or a
statistic. --detail adds the problems behind each verdict, every predictor's pooled skill over all rows and event
rows, the incremental test, the squared error and paired comparisons, calibration, the twin tests with each learner's
share of long opens, the own trades, the round trips per session, the largest notional over equity, the learners'
mean equity with and without the dead, deaths, the outcomes, data quality, spend and the code commits, for the
owner's own terminal. Only H1 and H2 carry a pass label, and every other result is descriptive.

The chooser scorecard of section 11.8 and the day scorecard of section 11.9 come from v2/scorecards.py as descriptive
sections, their counts in the summary and their figures with --detail. They never change a criterion, a verdict or the
exit code, and a failure to build them is printed in their place.

    btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml [--detail] [--checks DIR] [--ledger FILE]
"""

from __future__ import annotations

import argparse
import bisect
import functools
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, iso, parse_iso
from btc_survivor.store import StoreError
from btc_survivor.types import LONG
from btc_survivor.v2 import output, redact, sessions, stats
from btc_survivor.v2.config import (MAX_MONTHLY_USD, ConfigError, EvaluationConfig, StockConfig, StockCostsConfig,
                                    load_run_pair, parse_stock_config)
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.news.db import NewsDbError
from btc_survivor.v2.news.health import FINBERT_SCORER
from btc_survivor.v2.news.ledger import SPEND_ALERT
from btc_survivor.v2.sessions import FULL_SESSION_MINUTES, CalendarError, Session
from btc_survivor.v2.stocks import costs, store, tapecheck
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.health import HASH_MISMATCH
from btc_survivor.v2.stocks.broker import SYMBOL_STALE
from btc_survivor.v2.stocks.labels import recompute_labels
from btc_survivor.v2.stocks.news_inputs import HAIKU_SCORER, SCORED_STATUSES, SONNET_SCORER
from btc_survivor.v2.stocks.population import TWIN as TWIN_ORIGIN
from btc_survivor.v2.stocks.records import LOCAL_ONLY, MATURED, NEWS_META, PRICE_Z_COLUMNS


class EvaluateError(RuntimeError):
    """Raised when a run has not launched, the two runs were not launched together, or a recorded setting is
    unusable."""


RUNS = ("prices run", "news run")
RUN_MINUTE_RULE = "news_healthy_and_any_symbol_healthy"
INVARIANT = "invariant_violation"  # the version 1 feed event of a broken invariant line
CODE_COMMIT = "code_commit"  # the feed event each start of the stock process writes with its commit
DEFAULT_LEDGER = Path("~/.config/btc_survivor/llm_ledger.db")
FALLBACK_CAVEAT = 0.25  # a fallback share above this of the items routed to Haiku in session is a caveat
EVENT_MS = 30 * MS_PER_MINUTE  # an event row has an item tagged with its symbol received this long before
BPS_PREDICTORS = ("population", "linear_prices", "linear_news")
OPENS = ("open_long", "open_short", "flip")
LEARNER, TWIN = "learner", "twin"
# The score lines whose receipt makes an event row for each score predictor, primary lines for the news inputs.
PRIMARY = "primary"
EVENT_LINES = {"news": PRIMARY, "news_company": PRIMARY, "news_macro": PRIMARY, "haiku": HAIKU_SCORER,
               "finbert_matched": HAIKU_SCORER, "finbert": FINBERT_SCORER, "sonnet": SONNET_SCORER,
               "haiku_official": SONNET_SCORER}
SCORE_PAIRS = (("haiku", "finbert_matched"), ("sonnet", "haiku_official"))
# The outcomes of section 12, judged by reasoning before launch.
ROUND_TRIPS_PER_SESSION = 2
MOVE_BPS = 50.0  # a 30 minute move behind the founder equity bound, per horizon a position was held
FIRST_TRADE_SESSION = 3
FIRST_TRADE_MINUTES = (60, 270)  # 10:30 to 13:30 ET, plus up to 29 minutes waiting for a phase minute
BURN_TOLERANCE = 0.25

Row = tuple[str, str, float, float]  # ts, symbol, prediction, label
Pair = tuple[str, str, float, float, float]  # ts, symbol, first prediction, second prediction, label


@dataclass(frozen=True)
class Run:
    """One run's read only connection, its meta, the config it recorded at launch and whether the config file still
    has the experiment hash recorded then."""

    name: str
    conn: sqlite3.Connection
    meta: dict[str, str]
    config: StockConfig
    file_matches: bool = True


@dataclass(frozen=True)
class Window:
    """Launch, the window's start, its end and whether that end is the cutoff."""

    launch: str
    start: str  # the open of the first window session, so rows after it count
    end: str  # the cutoff, or the newest run minute before it
    final: bool
    run_minutes: int


@dataclass(frozen=True)
class Check:
    """One condition of criterion 1 with its verdict and the problems behind it."""

    name: str
    passed: bool
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outcome:
    """One outcome of section 12 in one run, matched or not, or None until the run reaches it."""

    run: str
    name: str
    matched: bool | None
    expected: str
    actual: str


@dataclass(frozen=True)
class Comparison:
    """Two score predictors judged on exactly the rows both have, all of them or only the event rows, without a pass
    label."""

    first: str
    second: str
    subset: str
    rows: int
    first_skill: stats.PooledSkill
    second_skill: stats.PooledSkill
    left_out: tuple[str, ...] = ()  # symbols either side could not judge, dropped from both


@dataclass(frozen=True)
class Evaluation:
    """Everything evaluate found, the verdicts and counts the summary prints and the tables --detail adds."""

    window: Window
    settings: EvaluationConfig
    configs_match: bool
    checks: tuple[Check, ...]
    confirmatory: stats.PooledSkill
    labels_compared: int
    scored_rows: int
    comparisons: tuple[Comparison, ...]
    outcomes: tuple[Outcome, ...]
    deaths: tuple[dict, ...]
    twin_pairs: int
    fallback_share: float | None
    spend_alert_days: int
    tables: dict[str, list[dict]]
    incremental: stats.PooledSkill | None = None  # H2, None when H1 did not pass and H2 was not run
    without_scores: int = 0  # H1 rows H2 left out because S2 stored no complete price z scores for them
    scorecards: tuple[tuple[str, Any], ...] = ()  # the scorecards' summary counts, descriptive

    @property
    def valid(self) -> bool:
        return all(check.passed for check in self.checks)


# Settings and the window


def settings_from(meta: dict[str, str]) -> EvaluationConfig:
    """The [evaluation] section S2 recorded at launch, refusing a run without one or with another run minute rule."""
    if "evaluation_json" not in meta:
        raise EvaluateError("the news run's meta holds no evaluation_json, so it has not launched as S2")
    values = json.loads(meta["evaluation_json"])
    settings = EvaluationConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})
    if settings.run_minute_rule != RUN_MINUTE_RULE or meta.get("run_minute_rule") != RUN_MINUTE_RULE:
        raise EvaluateError(f"the recorded run minute rule is not {RUN_MINUTE_RULE}")
    return settings


@functools.cache
def _index() -> dict[date, int]:
    """The position of every session of the table, for counting sessions from launch."""
    return {session.day: i for i, session in enumerate(sessions.all_sessions())}


def session_of(ms: int) -> Session:
    """The session a boundary or an instant belongs to, the one in progress or else the next, so a close boundary
    belongs to the session it closes."""
    return sessions.next_session(ms - 1)


def session_number(ms: int, launch_ms: int) -> int:
    """The session of ms counted from the launch session as 1."""
    return _index()[session_of(ms).day] - _index()[sessions.next_session(launch_ms).day] + 1


def from_launch(launch_ms: int) -> tuple[Session, ...]:
    """Every session of the table from the launch session on."""
    return sessions.all_sessions()[_index()[sessions.next_session(launch_ms).day]:]


def nth_session(launch_ms: int, number: int) -> Session:
    """Session number n counted from the launch session as 1."""
    return from_launch(launch_ms)[number - 1]


RUN_MINUTES = ("SELECT ts FROM news_minutes WHERE healthy = 1 AND ts > ? "
               "INTERSECT SELECT ts FROM stock_candles WHERE valid = 1 ORDER BY ts")


def run_minutes(conn: sqlite3.Connection, launch: str) -> list[str]:
    """S2's run minutes after launch in order, news healthy with at least one valid symbol candle."""
    return [row[0] for row in conn.execute(RUN_MINUTES, (launch,))]


def find_window(minutes: Sequence[str], launch: str, settings: EvaluationConfig) -> Window:
    """The window from the open of the session after the burn in to the minute S2 reaches the final count of run
    minutes, or to its newest run minute before that."""
    final = settings.final_healthy_session_minutes
    reached = len(minutes) >= final
    end = minutes[final - 1] if reached else (minutes[-1] if minutes else launch)
    start = nth_session(parse_iso(launch), settings.burn_in_sessions + 1).open_ms
    return Window(launch, iso(start), end, reached, min(len(minutes), final))


def window_sessions(window: Window) -> list[Session]:
    """The sessions of the window in order, from the first after the burn in to the session of its end."""
    if window.end <= window.start:
        return []
    launch = parse_iso(window.launch)
    first, last = session_number(parse_iso(window.start) + 1, launch), session_number(parse_iso(window.end), launch)
    return [nth_session(launch, n) for n in range(first, last + 1)]


@functools.lru_cache(maxsize=None)
def label_end(ts: str, horizon: int) -> str:
    """When the label of the evaluation minute ts matures, horizon session minutes later."""
    return iso(sessions.boundary_of_k(sessions.k_at(parse_iso(ts)) + horizon))


# Rows


FORECASTS = """
SELECT f.predictor, f.ts, f.symbol, f.prediction, l.label_bps FROM stock_forecasts f
JOIN stock_labels l ON l.ts = f.ts AND l.symbol = f.symbol
WHERE l.status = ? AND f.ts > ? AND f.ts <= ? ORDER BY f.ts, f.symbol
"""


def scored_rows(conn: sqlite3.Connection, window: Window, horizon: int) -> dict[str, list[Row]]:
    """Every predictor's rows in the window with their matured labels, leaving out a label that matured after the
    end."""
    out: dict[str, list[Row]] = {}
    for predictor, ts, symbol, prediction, label in conn.execute(FORECASTS, (MATURED, window.start, window.end)):
        if label_end(ts, horizon) <= window.end:
            out.setdefault(predictor, []).append((ts, symbol, prediction, label))
    return out


def _columns(rows: Sequence[tuple], n: int) -> list[list]:
    """The n columns of rows as lists, empty lists when there are no rows."""
    return [list(column) for column in zip(*rows)] if rows else [[] for _ in range(n)]


def skill(rows: Sequence[Row], s: EvaluationConfig) -> stats.PooledSkill:
    """Pooled skill of section 11.2 with the lags and z values recorded at launch."""
    return stats.pooled_skill(*_columns(rows, 4), s.newey_west_lags, s.one_sided_z, s.upper_bound_z)


PRICE_Z = f"SELECT ts, symbol, {', '.join(PRICE_Z_COLUMNS)} FROM price_z WHERE ts > ? AND ts <= ?"


def price_scores(conn: sqlite3.Connection, window: Window) -> dict[tuple[str, str], tuple[float, ...]]:
    """The 10 price z scores a run stored for each evaluation minute and symbol in the window, leaving out a row
    whose vector was invalid and stored as nulls."""
    return {(ts, symbol): tuple(z) for ts, symbol, *z in conn.execute(PRICE_Z, (window.start, window.end))
            if None not in z}


def incremental(rows: Sequence[Row], scores: dict[tuple[str, str], tuple[float, ...]],
                s: EvaluationConfig) -> tuple[stats.PooledSkill, int]:
    """H2 on the rows of H1 that have price z scores, with the lags and z values recorded at launch, and how many
    rows of H1 had none."""
    chosen = [(ts, symbol, p, y, scores[ts, symbol]) for ts, symbol, p, y in rows if (ts, symbol) in scores]
    result = stats.partial_skill(*_columns(chosen, 5), s.newey_west_lags, s.one_sided_z, s.upper_bound_z)
    return result, len(rows) - len(chosen)


def paired(first: Sequence[Row], second: Sequence[Row]) -> list[Pair]:
    """The rows both forecasts have, with both predictions."""
    other = {(ts, symbol): prediction for ts, symbol, prediction, _ in second}
    return [(ts, symbol, p, other[ts, symbol], y) for ts, symbol, p, y in first if (ts, symbol) in other]


def squared_error(pairs: Sequence[Pair], s: EvaluationConfig) -> stats.PooledDM:
    """pooled_dm of section 11.3 on paired rows, a positive t favouring the first forecast."""
    return stats.pooled_dm(*_columns(pairs, 5), s.newey_west_lags)


def receipts(conn: sqlite3.Connection, window: Window) -> dict[str, dict[str, list[int]]]:
    """S2's receive stamps by event source and symbol, from the primary score lines and each scorer's scored or
    irrelevant lines received from 30 minutes before the window to its end."""
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    since = iso(parse_iso(window.start) - EVENT_MS)
    for recv, scorer, status, tickers, primary in conn.execute(
            "SELECT recv_at, scorer, status, tickers, primary_use FROM news_seen WHERE recv_at >= ? AND recv_at < ? "
            "ORDER BY recv_at", (since, window.end)):
        sources = ([PRIMARY] if primary else []) + ([scorer] if status in SCORED_STATUSES else [])
        for symbol in filter(None, tickers.split(",")):
            for source in sources:
                out[source][symbol].append(parse_iso(recv))
    return out


def events(rows: Sequence[tuple], stamps: dict[str, list[int]]) -> list:
    """The rows at whose minute an item tagged with the row's symbol had been received within the 30 minutes
    before."""
    out = []
    for row in rows:
        received, ms = stamps.get(row[1], []), parse_iso(row[0])
        i = bisect.bisect_left(received, ms - EVENT_MS)
        if i < len(received) and received[i] < ms:
            out.append(row)
    return out


def compare(rows: dict[str, list[Row]], stamps: dict[str, dict[str, list[int]]],
            s: EvaluationConfig) -> list[Comparison]:
    """haiku against finbert_matched and sonnet against haiku_official, each on exactly their paired rows and their
    paired event rows."""
    out = []
    for first, second in SCORE_PAIRS:
        both = paired(rows.get(first, []), rows.get(second, []))
        for subset, chosen in (("all", both), ("events", events(both, stamps.get(EVENT_LINES[first], {})))):
            judged = _both_sides(chosen, s)
            dropped = set(judged[0].left_out) | set(judged[1].left_out)
            if any(set(side.left_out) != dropped for side in judged):  # judge both sides on the same symbols
                chosen = [row for row in chosen if row[1] not in dropped]
                judged = _both_sides(chosen, s)
            out.append(Comparison(first, second, subset, len(chosen), *judged, tuple(sorted(dropped))))
    return out


def _both_sides(chosen: Sequence[Pair], s: EvaluationConfig) -> tuple[stats.PooledSkill, stats.PooledSkill]:
    return (skill([(t, n, a, y) for t, n, a, _, y in chosen], s), skill([(t, n, b, y) for t, n, _, b, y in chosen], s))


def comparison_row(c: Comparison) -> dict:
    """One paired comparison as a detail row, with no verdict."""
    return {"comparison": f"{c.first} against {c.second}", "rows": c.subset, "n": c.rows,
            "first corr": _f(c.first_skill.corr), "first t": _f(c.first_skill.t, ".2f"),
            "second corr": _f(c.second_skill.corr), "second t": _f(c.second_skill.t, ".2f"),
            "left out": " ".join(c.left_out)}


# Criterion 1


COVERAGE = "coverage stability"


def coverage_records(folder: Path) -> list[tuple[date, bool, float]]:
    """Every coverage record tape-check wrote in the checks folder, its session date, whether it was a clean session
    that was no early close, and its median ratio, by date. A malformed record stops evaluate."""
    try:
        records = tapecheck.read_records(folder)
    except tapecheck.TapeCheckError as exc:
        raise EvaluateError(str(exc)) from exc
    return sorted((r.session, r.clean and not r.early_close, r.median_ratio) for r in records)


def full_session(day: date) -> bool:
    """Whether a regular session was held on day and ran to the 16:00 ET close."""
    session = sessions.session_on(day)
    return session is not None and not session.early_close


def check_coverage(records: Sequence[tuple[date, bool, float]], window: Window, due: Sequence[int]) -> Check:
    """Criterion 1.1. The baseline is the median ratio of the gate session, the earliest full session with a coverage
    record before launch, clean or not, as the owner set on 12 October 2026, and for each due session that has
    closed, the first clean full session at or after it keeps its median ratio within a factor of 2 of the baseline,
    edges included. Before the cutoff a due session without a record yet is not yet due, because its record waits for
    the official figures and a clean session, and only at the cutoff does a missing record fail."""
    launch = parse_iso(window.launch)
    first = sessions.next_session(launch).day
    full = [(day, ratio) for day, clean, ratio in records if clean and full_session(day)]
    baseline = next((ratio for day, _, ratio in records if day < first and full_session(day)), None)
    if baseline is None:
        return Check(COVERAGE, False, ("no full session coverage record before launch, so no gate session",))
    if not baseline > 0:
        return Check(COVERAGE, False, ("the gate session's median ratio is not above zero",))
    problems = []
    for n in due:
        session = nth_session(launch, n)
        if session.close_ms > parse_iso(window.end):
            continue
        found = next(((day, ratio) for day, ratio in full if day >= session.day), None)
        if found is None:
            if window.final:
                problems.append(f"the coverage record at or after session {n} is missing")
        elif not tapecheck.within_factor(baseline, found[1]):
            problems.append(f"the median ratio of {found[0]}, the first clean session at or after session {n}, is "
                            f"{found[1] / baseline:.2f} times the gate session's")
    return Check(COVERAGE, not problems, tuple(problems))


def check_lookahead(runs: Sequence[Run], window: Window) -> Check:
    """No forecast row used an input received after its minute, and no score line counted before its scored_at."""
    problems = []
    span = (window.launch, window.end)
    for run in runs:
        problems += [f"{run.name} {p} {s} at {ts} used an input received at {r}" for ts, s, p, r in run.conn.execute(
            "SELECT ts, symbol, predictor, inputs_max_recv_at FROM stock_forecasts WHERE inputs_max_recv_at > ts "
            "AND ts > ? AND ts <= ?", span)]
        problems += [f"{run.name} score line {seq} received at {r} before its scored_at {sc}" for seq, r, sc in
                     run.conn.execute("SELECT score_seq, recv_at, scored_at FROM news_seen WHERE recv_at < scored_at "
                                      "AND recv_at > ? AND recv_at <= ?", span)]
    return Check("lookahead audit", not problems, tuple(problems))


CANDLES = ("SELECT symbol, ts, session_k, open, high, low, close, ref_price, volume, buy_volume, sell_volume, prints, "
           "price_prints, late_prints, valid FROM stock_candles WHERE ts > ? AND ts <= ? ORDER BY symbol, ts")


def recomputed_labels(conn: sqlite3.Connection, window: Window, horizon: int) -> dict[tuple[str, str], tuple]:
    """Every label from launch whose horizon ended by the end, recomputed from the stored candles alone."""
    candles: dict[str, list[StockCandle]] = defaultdict(list)
    for row in conn.execute(CANDLES, (window.launch, window.end)):
        candles[row[0]].append(StockCandle(row[0], parse_iso(row[1]), *row[2:14], bool(row[14])))
    out = {}
    for symbol, rows in candles.items():
        for label in recompute_labels(rows, horizon):
            ts = iso(label.ts)
            if label_end(ts, horizon) <= window.end:
                out[ts, symbol] = (label.status, label.entry_price, label.exit_price, label.label_bps)
    return out


def check_labels(runs: Sequence[Run], window: Window, horizon: int) -> tuple[Check, int]:
    """Every stored label whose horizon ended by the end equals the one recomputed from the candles, exactly, and
    how many were compared."""
    problems, compared = [], 0
    for run in runs:
        stored = {(ts, s): (status, entry, exit_price, label) for ts, s, status, entry, exit_price, label in
                  run.conn.execute("SELECT ts, symbol, status, entry_price, exit_price, label_bps FROM stock_labels "
                                   "WHERE ts > ? AND ts <= ?", (window.launch, window.end))
                  if label_end(ts, horizon) <= window.end}
        recomputed = recomputed_labels(run.conn, window, horizon)
        keys = sorted(stored.keys() | recomputed.keys())
        compared += len(keys)
        problems += [f"{run.name} label {s} at {ts} is {stored.get((ts, s))}, recomputed {recomputed.get((ts, s))}"
                     for ts, s in keys if stored.get((ts, s)) != recomputed.get((ts, s))]
    return Check("labels recomputed", not problems, tuple(problems)), compared


# Each comparison with the number of leading columns that name a row. A session counts once it closed and a label or a
# linear_prices row once its horizon ended, so a look between the two runs' flushes finds no difference.
IDENTICAL = {
    "sessions": ("SELECT session_date, healthy_minutes, run_minutes FROM sessions WHERE close_ts <= ?", 1),
    "labels": ("SELECT ts, symbol, entry_price, exit_price, label_bps, status FROM stock_labels WHERE ts <= ?", 2),
    "linear_prices rows": ("SELECT ts, symbol, prediction FROM stock_forecasts WHERE predictor = 'linear_prices' "
                           "AND ts <= ?", 2),
}


def check_identical(s1: Run, s2: Run, window: Window, horizon: int) -> Check:
    """S1 and S2 have the same run minutes per session, the same labels and the same linear_prices rows."""
    problems = []
    for name, (sql, width) in IDENTICAL.items():
        a, b = ({tuple(row) for row in run.conn.execute(sql, (window.end,))
                 if width == 1 or label_end(row[0], horizon) <= window.end} for run in (s1, s2))
        problems += [f"{name} differ at {' '.join(key)}" for key in sorted({row[:width] for row in a ^ b})]
    return Check("runs identical", not problems, tuple(problems))


def check_invariants(runs: Sequence[Run], window: Window) -> Check:
    """No invariant_violation feed event in either run."""
    problems = [f"{run.name} invariant violation at {ts}" for run in runs for (ts,) in run.conn.execute(
        "SELECT ts FROM feed_events WHERE kind = ? AND ts > ? AND ts <= ?", (INVARIANT, window.launch, window.end))]
    return Check("invariants", not problems, tuple(problems))


def check_one_config(s1: Run, s2: Run, news: sqlite3.Connection, window: Window) -> Check:
    """One config hash, one prompt hash and one model id per scorer, the ones recorded at launch, cover the run."""
    meta, span = s2.meta, (window.launch, window.end)
    problems = [f"the runs recorded different {key}" for key in NEWS_META if s1.meta.get(key) != meta.get(key)]
    for column in ("prompt_sha256", "schema_sha256"):
        problems += [f"a model call used {column} {value}" for (value,) in news.execute(
            f"SELECT DISTINCT {column} FROM llm_calls WHERE started_at > ? AND started_at <= ?", span)
            if value != meta.get(column)]
    models: dict[str, set[str]] = defaultdict(set)
    for scorer, model, response in news.execute(
            "SELECT DISTINCT scorer, model, response_model FROM llm_calls WHERE started_at > ? AND started_at <= ?",
            span):
        models[scorer].update(m for m in (model, response) if m)
    for scorer, found in sorted(models.items()):
        want = meta.get("sonnet_model" if scorer.startswith("sonnet") else "haiku_model")
        if found != {want}:
            problems.append(f"scorer {scorer} used model ids {', '.join(sorted(found))}, recorded {want}")
    launch_hashes = (meta.get("news_config_hash"), meta.get("prompt_sha256"), meta.get("schema_sha256"))
    problems += ["a heartbeat carried hashes other than those recorded at launch" for row in news.execute(
        "SELECT DISTINCT config_hash, prompt_sha256, schema_sha256 FROM news_heartbeat WHERE ts > ? AND ts <= ?", span)
        if tuple(row) != launch_hashes]
    mismatched = s2.conn.execute("SELECT count(*) FROM news_minutes WHERE reason = ? AND ts > ? AND ts <= ?",
                                 (HASH_MISMATCH, *span)).fetchone()[0]
    if mismatched:
        problems.append(f"{mismatched} minutes failed news health on a hash mismatch")
    return Check("one config prompt and model", not problems, tuple(problems))


def ledger_spend(path: Path, until: str | None = None) -> dict[str, dict[str, float]] | None:
    """The key ledger's spend per UTC month and model up to until, each call at its cost or else its reservation as
    the cap counts it, or None when there is no ledger file. Spend after the end never counts, so a later run cannot
    change a result. The ledger is opened read only."""
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT substr(ts, 1, 7), model, coalesce(cost_usd, reserved_usd) FROM reservations "
                            "WHERE ts <= ?", (until or "9999",))
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for month, model, usd in rows:
            grouped[month][model].append(usd)
    except sqlite3.DatabaseError as exc:
        raise EvaluateError(f"{path} is not a key ledger, {exc}") from exc
    finally:
        conn.close()
    return {month: {model: math.fsum(v) for model, v in models.items()} for month, models in grouped.items()}


def check_spend(spend: dict[str, dict[str, float]] | None, window: Window) -> Check:
    """The key ledger's spend of both models together stayed within the monthly cap in every UTC month of the run."""
    if spend is None:
        return Check("spend within cap", False, ("the key ledger is missing",))
    first, last = window.launch[:7], window.end[:7]
    problems = [f"{month} spent {math.fsum(models.values()):.2f} dollars" for month, models in sorted(spend.items())
                if first <= month <= last and math.fsum(models.values()) > MAX_MONTHLY_USD]
    return Check("spend within cap", not problems, tuple(problems))


def _passed(detail: str | None) -> bool:
    """Whether a local_only event's detail says passed."""
    try:
        return json.loads(detail or "{}").get("passed") is True
    except (ValueError, AttributeError):
        return False


def check_local_only(runs: Sequence[Run], window: Window) -> Check:
    """No failed local_only check, and a passed one in each run before every session of the window, after the
    previous close and no later than the open, or for a start inside the session no later than that run's first
    candle of it."""
    problems = []
    due = window_sessions(window)
    for run in runs:
        found = [(ts, _passed(detail)) for ts, detail in run.conn.execute(
            "SELECT ts, detail FROM feed_events WHERE kind = ? AND ts > ? AND ts <= ? ORDER BY ts",
            (LOCAL_ONLY, window.launch, window.end))]
        problems += [f"{run.name} local only check at {ts} failed" for ts, ok in found if not ok]
        passed = [parse_iso(ts) for ts, ok in found if ok]
        for session in due:
            before = sessions.all_sessions()[_index()[session.day] - 1].close_ms
            first = run.conn.execute("SELECT min(ts) FROM stock_candles WHERE session_date = ?",
                                     (session.day.isoformat(),)).fetchone()[0]
            latest = max(session.open_ms, parse_iso(first) if first else session.open_ms)
            if not any(before < ms <= latest for ms in passed):
                problems.append(f"{run.name} has no passed local only check before the session of {session.day}")
    return Check("local only checks", not problems, tuple(problems))


def check_commits(runs: Sequence[Run], window: Window) -> tuple[Check, tuple[str, ...]]:
    """Every code commit active in the run, the one recorded at launch and the one of every later start."""
    problems, listed = [], []
    for run in runs:
        listed.append(run.meta.get("code_commit", ""))
        if not listed[-1]:
            problems.append(f"{run.name} recorded no code_commit at launch")
        for ts, detail in run.conn.execute("SELECT ts, detail FROM feed_events WHERE kind = ? AND ts > ? AND ts <= ?",
                                           (CODE_COMMIT, window.launch, window.end)):
            listed.append(detail or "")
            if not detail:
                problems.append(f"{run.name} start at {ts} recorded no commit")
    return Check("code commits listed", not problems, tuple(problems)), tuple(sorted(set(filter(None, listed))))


# Section 12, by hand


def round_trip_dollars(cfg: StockCostsConfig, symbol: str, notional: float, horizon: int) -> float:
    """A long round trip of the notional, two legs of allowance and the cents of its one sale, the same at every
    planning price since both fees round up to a cent."""
    return costs.round_trip(cfg, symbol, LONG, 1.0, notional, horizon).dollars


def burn_per_session(cfg: StockCostsConfig, symbol: str, notional: float, horizon: int) -> float:
    """What a quiet learner pays in one full session, a session of rent and two round trips."""
    return costs.session_rent(cfg) + ROUND_TRIPS_PER_SESSION * round_trip_dollars(cfg, symbol, notional, horizon)


def burn_by_cutoff(cfg: StockCostsConfig, symbols: Sequence[str], notional: float, horizon: int,
                   minutes: int) -> tuple[float, float]:
    """The rent a quiet learner pays over minutes usable minutes from launch, and its round trips from the first
    trades of session 3 on, averaged over the home symbols."""
    trip = statistics.fmean(round_trip_dollars(cfg, s, notional, horizon) for s in symbols)
    sessions_trading = max(0.0, minutes / FULL_SESSION_MINUTES - (FIRST_TRADE_SESSION - 1))
    return costs.rent_per_minute(cfg) * minutes, ROUND_TRIPS_PER_SESSION * sessions_trading * trip


def founder_equity(start_equity: float, rent_per_minute: float, usable_minutes: float, costs_paid: float) -> float:
    """A learner's equity with no gross profit, start equity less rent per usable minute and the costs it paid."""
    return start_equity - rent_per_minute * usable_minutes - costs_paid


def founder_bound(horizons: float, notional: float) -> float:
    """Two standard deviations of the learners' mean gross profit if all their trades move together, a 50 bps move
    on the notional for each horizon a position was held, about 43 dollars at 75 trades of one horizon each and about
    76 at the 230 horizons a no skill learner held by the cutoff under the holding rule in a one off synthetic
    check."""
    return 2 * math.sqrt(horizons) * MOVE_BPS / 1e4 * notional


def session_ms_held(opened: str, closed: str) -> int:
    """The milliseconds of regular session hours between a trade's open and close, so a position carried overnight or
    over a weekend and flattened at the next open counts the session time it was held, not the night."""
    start, end = parse_iso(opened), parse_iso(closed)
    table = sessions.all_sessions()
    index, total = bisect.bisect_right(table, start, key=lambda s: s.close_ms), 0
    while index < len(table) and table[index].open_ms < end:
        total += max(0, min(end, table[index].close_ms) - max(start, table[index].open_ms))
        index += 1
    return total


def death_line(cfg: StockCostsConfig, symbol: str, notional: float, horizon: int) -> float:
    """A flat learner's death threshold, a long round trip of the notional plus one session of rent."""
    return costs.death_line(cfg, symbol, LONG, 1.0, notional, horizon)


def death_horizon_sessions(start_equity: float, threshold: float, burn: float) -> float:
    """How many sessions of quiet burn take a learner from start equity to its death threshold."""
    return (start_equity - threshold) / burn


def planned_rows(config: StockConfig, minutes: int) -> list[dict]:
    """The planned figures of section 12 from the recorded costs, per home symbol its round trip, burn per session,
    death threshold and sessions to death, then the rent and round trips a quiet learner pays by the cutoff."""
    cfg, trading, horizon = config.stock_costs, config.trading, config.time.horizon_minutes
    rows = []
    for symbol in config.stock_feed.symbols:
        trip = round_trip_dollars(cfg, symbol, trading.notional, horizon)
        burn, threshold = burn_per_session(cfg, symbol, trading.notional, horizon), \
            death_line(cfg, symbol, trading.notional, horizon)
        rows.append({"symbol": symbol, "round trip": _f(trip, ".2f"),
                     "burn per session": _f(burn, ".2f"), "death threshold": _f(threshold, ".2f"),
                     "sessions to death": _f(death_horizon_sessions(trading.start_equity, threshold, burn), ".0f")})
    rent, trips = burn_by_cutoff(cfg, config.stock_feed.symbols, trading.notional, horizon, minutes)
    return [*rows, {"symbol": "all", "rent by cutoff": _f(rent, ".2f"), "round trips by cutoff": _f(trips, ".2f")}]


# Section 12, from the rows


def _f(value: float | None, spec: str = ".4f") -> str:
    """A figure for a detail row, empty when there is none."""
    return "" if value is None else format(value, spec)


def usable_minutes(conn: sqlite3.Connection, news_ok: set[str], window: Window) -> dict[str, list[str]]:
    """Each symbol's usable minutes from launch to the end in order, valid with news health held."""
    out: dict[str, list[str]] = defaultdict(list)
    for symbol, ts in conn.execute("SELECT symbol, ts FROM stock_candles WHERE valid = 1 AND ts > ? AND ts <= ? "
                                   "ORDER BY ts", (window.launch, window.end)):
        if ts in news_ok:
            out[symbol].append(ts)
    return out


def _between(stamps: list[str], after: str, until: str) -> int:
    """How many sorted stamps fall after one time and at or before another."""
    return max(0, bisect.bisect_right(stamps, until) - bisect.bisect_right(stamps, after))


LEARNERS = "SELECT bot_id, home_symbol, born_at FROM bots WHERE origin != ? ORDER BY slot, bot_id"
SNAPSHOT = ("SELECT ts, equity, rent_total, fees_total, funding_total FROM equity_snapshots WHERE bot_id = ? "
            "AND ts <= ? ORDER BY ts DESC LIMIT 1")


def learner_books(run: Run, usable: dict[str, list[str]], window: Window) -> list[dict]:
    """Each learner's latest snapshot by the end with its usable home minutes, the costs it paid, the allowance
    between print and fill price plus fees and borrow, its closed round trips and the horizons they were held in all,
    counted in session time, all up to that snapshot. Under the holding rule one round trip can last several
    horizons."""
    out = []
    horizon_ms = run.config.time.horizon_minutes * MS_PER_MINUTE
    for bot_id, home, born in run.conn.execute(LEARNERS, (TWIN_ORIGIN,)).fetchall():
        snap = run.conn.execute(SNAPSHOT, (bot_id, window.end)).fetchone()
        if snap is None:
            continue
        ts, equity, rent, fees, borrow = snap
        allowance = run.conn.execute("SELECT coalesce(sum(abs(price - print_price) * qty), 0) FROM fills WHERE "
                                     "bot_id = ? AND status = 'filled' AND filled_at <= ?", (bot_id, ts)).fetchone()[0]
        held = run.conn.execute("SELECT opened_at, closed_at FROM trades WHERE bot_id = ? AND closed_at <= ?",
                                (bot_id, ts)).fetchall()
        out.append({"bot_id": bot_id, "home": home, "equity": equity, "rent": rent, "paid": allowance + fees + borrow,
                    "trips": len(held), "horizons": sum(session_ms_held(o, c) for o, c in held) / horizon_ms,
                    "usable": _between(usable[home], born, ts)})
    return out


def outcome_money(run: Run, books: list[dict]) -> list[Outcome]:
    """Burn and founder equity, the learners' mean rent and costs against the planned burn over the same usable
    minutes and round trips, and their mean equity against start equity less rent and costs paid, within a bound
    that grows with the horizons the learners held their positions."""
    cfg, trading, horizon = run.config.stock_costs, run.config.trading, run.config.time.horizon_minutes
    if not books:
        return [Outcome(run.name, name, None, "", "no learner snapshot yet") for name in ("burn", "founder equity")]
    rpm = costs.rent_per_minute(cfg)
    planned = statistics.fmean(rpm * b["usable"] + b["trips"] * round_trip_dollars(cfg, b["home"], trading.notional,
                                                                                    horizon) for b in books)
    burned = statistics.fmean(b["rent"] + b["paid"] for b in books)
    expected = statistics.fmean(founder_equity(trading.start_equity, rpm, b["usable"], b["paid"]) for b in books)
    actual = statistics.fmean(b["equity"] for b in books)
    bound = max(founder_bound(statistics.fmean(b["horizons"] for b in books), trading.notional), 0.01)
    return [Outcome(run.name, "burn", abs(burned - planned) <= BURN_TOLERANCE * planned, f"{planned:.2f} dollars",
                    f"{burned:.2f} dollars"),
            Outcome(run.name, "founder equity", abs(actual - expected) <= bound,
                    f"{expected:.2f} within {bound:.2f}", f"{actual:.2f}")]


def outcome_timeline(run: Run, books: list[dict], window: Window) -> list[Outcome]:
    """First trades in session 3 between about 10:30 and 13:30 ET, forced by the timer or made on the learner's own
    once warmup ends."""
    launch = parse_iso(window.launch)
    third = nth_session(launch, FIRST_TRADE_SESSION)
    out = []
    if window.end < iso(third.close_ms) or not books:
        out.append(Outcome(run.name, "first trades", None, "session 3 from 10:30 to 13:30 ET", "not reached"))
    else:
        firsts = {bot_id: parse_iso(ts) for bot_id, ts in run.conn.execute(
            "SELECT f.bot_id, min(f.decided_at) FROM fills f JOIN bots b ON b.bot_id = f.bot_id WHERE b.origin != ? "
            "AND f.leg = 'open' AND f.status = 'filled' AND f.decided_at <= ? GROUP BY f.bot_id",
            (TWIN_ORIGIN, window.end))}
        places = [(session_number(firsts[b["bot_id"]], launch),
                   session_of(firsts[b["bot_id"]]).session_minute(firsts[b["bot_id"]]))
                  if b["bot_id"] in firsts else (None, None) for b in books]
        lo, hi = FIRST_TRADE_MINUTES
        ok = all(n == FIRST_TRADE_SESSION and lo <= m <= hi for n, m in places)
        minutes = [m for n, m in places if m is not None]
        out.append(Outcome(run.name, "first trades", ok, f"session {FIRST_TRADE_SESSION} minutes {lo} to {hi}",
                           f"sessions {sorted({n for n, _ in places if n is not None})} minutes "
                           f"{min(minutes, default='')} to {max(minutes, default='')}"))
    return out


TRIPS = "SELECT forced, opened_at, closed_at FROM trades WHERE bot_id = ? AND closed_at > ? AND closed_at <= ?"


def round_trips(run: Run, usable: dict[str, list[str]], books: list[dict], window: Window) -> dict:
    """The learners' mean closed round trips per full session of usable home minutes from session 4, all of them,
    those the timer forced and those the learner opened on its own, descriptive since the zero margin lets own trades
    come beside the timer and restart it, and the holding rule keeps a position over several decisions, so section
    12's two forced trips a session is no longer a fixed outcome, with the mean session minutes those round trips were
    held."""
    steady = nth_session(parse_iso(window.launch), FIRST_TRADE_SESSION + 1)
    row = {"run": run.name, "from session": FIRST_TRADE_SESSION + 1, "learners": 0, "all": "", "forced": "",
           "own": "", "minutes held": "", "timer alone": f"at most about {ROUND_TRIPS_PER_SESSION}"}
    if window.end < iso(steady.close_ms):
        return row
    after, rates, held = iso(steady.open_ms), [], []
    for b in books:
        minutes = _between(usable[b["home"]], after, window.end)
        if minutes:
            trades = run.conn.execute(TRIPS, (b["bot_id"], after, window.end)).fetchall()
            forced = sum(1 for f, _, _ in trades if f)
            sessions_seen = minutes / FULL_SESSION_MINUTES
            rates.append((forced / sessions_seen, (len(trades) - forced) / sessions_seen))
            held += [session_ms_held(opened, closed) / MS_PER_MINUTE for _, opened, closed in trades]
    if rates:
        forced, own = (statistics.fmean(r[i] for r in rates) for i in (0, 1))
        row.update({"learners": len(rates), "all": _f(forced + own, ".2f"), "forced": _f(forced, ".2f"),
                    "own": _f(own, ".2f"), "minutes held": _f(statistics.fmean(held) if held else None, ".0f")})
    return row


STALE_CLOSE = ("SELECT 1 FROM fills WHERE bot_id = ? AND symbol = ? AND leg = 'close' AND status = 'rejected' "
               "AND reason = ? AND decided_at >= ? AND decided_at < ? LIMIT 1")


def outcome_closes(run: Run, deaths: int, window: Window) -> list[Outcome]:
    """No deaths, and every position flat at every close, except one whose session close was rejected because its
    symbol was stale, which is excused and listed as section 12 says."""
    carried, excused = [], []
    before = run.config.session.flatten_minutes_before_close * MS_PER_MINUTE
    for session in from_launch(parse_iso(window.launch)):
        close = iso(session.close_ms)
        if close > window.end:
            break
        for bot_id, symbol in run.conn.execute(
                "SELECT bot_id, symbol FROM trades WHERE opened_at < ? AND (closed_at IS NULL OR closed_at > ?)",
                (close, close)).fetchall():
            stale = run.conn.execute(STALE_CLOSE, (bot_id, symbol, SYMBOL_STALE, iso(session.close_ms - before),
                                                   close)).fetchone()
            (excused if stale else carried).append(f"bot {bot_id} {symbol} {session.day}")
    actual = ", ".join(carried) or "none"
    if excused:
        actual += f", excused after a close rejected as stale {', '.join(excused)}"
    return [Outcome(run.name, "no deaths", deaths == 0, "0", str(deaths)),
            Outcome(run.name, "flat at every close", not carried, "none carried but after a stale close", actual)]


def label_sds(run: Run, window: Window) -> dict[str, float]:
    """Each symbol's standard deviation of its matured labels in the window, the scale of the prediction spread."""
    labels: dict[str, list[float]] = defaultdict(list)
    for symbol, label in run.conn.execute("SELECT symbol, label_bps FROM stock_labels WHERE status = ? AND ts > ? "
                                          "AND ts <= ?", (MATURED, window.start, window.end)):
        labels[symbol].append(label)
    return {symbol: float(np.std(values)) for symbol, values in labels.items() if len(values) >= 2}


def own_trades(run: Run, window: Window) -> dict:
    """The margin recorded at launch, the learner decisions and opens in the window, how many opens the exploration
    timer forced and how many the learner made on its own once its prediction cleared the required edge, the own
    trade share, the share of decisions that cleared it and the spread of the predictions in label standard
    deviations, all descriptive, since with a margin of 0 a learner trades on its own whenever its prediction beats
    the cost. A flip counts as an open, a decision that kept a position under the holding rule does not, and the gate
    share counts every decision whose prediction cleared the full required edge, kept positions included."""
    rows = run.conn.execute(
        "SELECT d.action, d.forced, d.gate_passed, d.prediction_bps, d.symbol FROM decisions d JOIN bots b "
        "ON b.bot_id = d.bot_id WHERE b.origin != ? AND d.ts > ? AND d.ts <= ?",
        (TWIN_ORIGIN, window.start, window.end)).fetchall()
    forced = [bool(f) for action, f, *_ in rows if action in OPENS]
    own = forced.count(False)
    sds = label_sds(run, window)
    spreads = [p / sds[symbol] for *_, p, symbol in rows if sds.get(symbol)]
    return {"run": run.name, "margin sigma": f"{run.config.trading.margin_sigma:g}", "decisions": len(rows),
            "opens": len(forced), "forced opens": len(forced) - own,
            "own opens": own, "own share": _f(own / len(forced) if forced else None),
            "gate passed share": _f(sum(bool(g) for _, _, g, _, _ in rows) / len(rows) if rows else None),
            "prediction spread sd": _f(float(np.std(spreads)) if spreads else None)}


# Every open fill from launch to the end with the bot's latest equity snapshot at or before its decision.
OPEN_NOTIONALS = ("SELECT f.bot_id, f.decided_at, f.notional, b.origin, (SELECT e.equity FROM equity_snapshots e "
                  "WHERE e.bot_id = f.bot_id AND e.ts <= f.decided_at ORDER BY e.ts DESC LIMIT 1) "
                  "FROM fills f JOIN bots b ON b.bot_id = f.bot_id WHERE f.leg = 'open' AND f.status = 'filled' "
                  "AND f.notional IS NOT NULL AND f.decided_at > ? AND f.decided_at <= ? ORDER BY f.decided_at")


def notional_over_equity(run: Run, window: Window) -> dict:
    """The largest notional over equity of any open fill from launch to the end, learner or twin, with equity the
    bot's latest snapshot at or before the decision, or start equity before its first, about 0.05 by design."""
    start = run.config.trading.start_equity
    largest: tuple[float, int, str, str] | None = None
    fills = 0
    for bot_id, decided, notional, origin, equity in run.conn.execute(OPEN_NOTIONALS, (window.launch, window.end)):
        fills += 1
        equity = start if equity is None else equity
        ratio = notional / equity if equity > 0 else math.inf
        if largest is None or ratio > largest[0]:
            largest = (ratio, bot_id, TWIN if origin == TWIN_ORIGIN else LEARNER, decided)
    if largest is None:
        return {"run": run.name, "open fills": 0, "largest": "", "bot_id": "", "kind": "", "decided_at": ""}
    ratio, bot_id, kind, decided = largest
    return {"run": run.name, "open fills": fills, "largest": _f(ratio), "bot_id": bot_id, "kind": kind,
            "decided_at": decided}


def learner_equity(run: Run, window: Window) -> dict:
    """The learners born by the end, those still alive at their latest equity snapshot by the end, or start equity
    before their first, their mean, and the mean over every learner with each dead one at its final equity."""
    start = run.config.trading.start_equity
    dead = dict(run.conn.execute("SELECT d.bot_id, d.final_equity FROM deaths d JOIN bots b ON b.bot_id = d.bot_id "
                                 "WHERE b.origin != ? AND d.died_at <= ?", (TWIN_ORIGIN, window.end)).fetchall())
    alive = []
    for (bot_id,) in run.conn.execute("SELECT bot_id FROM bots WHERE origin != ? AND born_at <= ? ORDER BY bot_id",
                                      (TWIN_ORIGIN, window.end)).fetchall():
        if bot_id not in dead:
            snap = run.conn.execute(SNAPSHOT, (bot_id, window.end)).fetchone()
            alive.append(start if snap is None else snap[1])
    every = [*alive, *dead.values()]
    return {"run": run.name, "learners": len(every), "survivors": len(alive),
            "survivors mean equity": _f(statistics.fmean(alive) if alive else None, ".2f"),
            "all learners mean equity": _f(statistics.fmean(every) if every else None, ".2f")}


def outcome_pairing(s1: Run, s2: Run, window: Window) -> Outcome:
    """S1 and S2 population forecasts identical through session 1, diverging no earlier than session 2, a row only
    one run holds counting as a difference. No difference at all is not yet while session 2 is open and not matched
    once it has closed."""
    sql = "SELECT ts, symbol, prediction FROM stock_forecasts WHERE predictor = 'population' AND ts <= ?"
    first, second = ({(ts, s): p for ts, s, p in run.conn.execute(sql, (window.end,))} for run in (s1, s2))
    differ = min((key[0] for key in first.keys() | second.keys() if first.get(key) != second.get(key)), default=None)
    launch = parse_iso(window.launch)
    expected = "first difference in session 2 or later"
    if differ is None:
        closed = window.end >= iso(nth_session(launch, 2).close_ms)
        return Outcome("both runs", "pairing", False if closed else None, expected, "none yet")
    number = session_number(parse_iso(differ), launch)
    return Outcome("both runs", "pairing", number >= 2, expected, f"session {number}")


# Data quality and descriptive tables


def data_quality(s2: Run, news: sqlite3.Connection, window: Window) -> tuple[list[dict], float | None, int]:
    """The data quality rows of section 11.6, the fallback share in session and the days that raised the spend
    alert."""
    span, launch, end = (window.launch, window.end), parse_iso(window.launch), parse_iso(window.end)
    minutes = window.run_minutes
    elapsed = sum((min(s.close_ms, end) - max(s.open_ms, launch)) // MS_PER_MINUTE for s in from_launch(launch)
                  if s.open_ms < end)
    symbols = len(json.loads(s2.meta.get("symbols_json", "[]"))) or 1
    valid = s2.conn.execute("SELECT count(*) FROM stock_candles WHERE valid = 1 AND ts > ? AND ts <= ?",
                            span).fetchone()[0]
    rows = [{"measure": "session minutes", "value": elapsed}, {"measure": "run minutes", "value": minutes},
            {"measure": "healthy fraction", "value": _f(valid / (elapsed * symbols) if elapsed else None)},
            {"measure": "run minute share", "value": _f(minutes / elapsed if elapsed else None)}]
    ok = {ts for (ts,) in s2.conn.execute("SELECT ts FROM stock_candles WHERE valid = 1 AND ts > ? AND ts <= ?", span)}
    lost: Counter[str] = Counter()
    outage: Counter[str] = Counter()
    for ts, healthy, reason, down in s2.conn.execute("SELECT ts, healthy, reason, channels_down FROM news_minutes "
                                                     "WHERE ts > ? AND ts <= ?", span):
        if not healthy and ts in ok:
            lost[reason or "unknown"] += 1
        outage.update(filter(None, down.split(",")))
    rows += [{"measure": f"run minutes lost to {reason}", "value": n} for reason, n in sorted(lost.items())]
    rows += [{"measure": f"outage minutes {channel}", "value": n} for channel, n in sorted(outage.items())]
    lags: dict[str, list[float]] = defaultdict(list)
    for scorer, scored, received in news.execute(
            "SELECT s.scorer, s.scored_at, i.received_at FROM item_scores s JOIN news_items i ON i.seq = s.item_seq "
            "WHERE i.received_at > ? AND i.received_at <= ?", span):
        lags[scorer].append((parse_iso(scored) - parse_iso(received)) / MS_PER_SECOND)
    for scorer, values in sorted(lags.items()):
        rows += [{"measure": f"{scorer} scoring lag median s", "value": _f(statistics.median(values), ".1f")},
                 {"measure": f"{scorer} scoring lag p90 s", "value": _f(float(np.percentile(values, 90)), ".1f")}]
    rows += [{"measure": f"items from {source}", "value": n} for source, n in news.execute(
        "SELECT source, count(*) FROM news_items WHERE status = 'new' AND received_at > ? AND received_at <= ? "
        "GROUP BY source ORDER BY source", span)]
    flags = [fallback for recv, fallback in s2.conn.execute(
        "SELECT recv_at, fallback FROM news_seen WHERE primary_use = 1 AND route = 'llm' AND recv_at > ? "
        "AND recv_at <= ?", span) if sessions.session_at(parse_iso(recv)) is not None]
    share = sum(flags) / len(flags) if flags else None
    alert_days = news.execute("SELECT count(DISTINCT substr(ts, 1, 10)) FROM collector_events WHERE kind = ? "
                              "AND ts > ? AND ts <= ?", (SPEND_ALERT, *span)).fetchone()[0]
    rows += [{"measure": "fallback share in session", "value": _f(share)},
             {"measure": "days with the spend alert", "value": alert_days}]
    return rows, share, alert_days


def skill_row(run: str, predictor: str, subset: str, result: stats.PooledSkill, **extra: str) -> dict:
    """One predictor's pooled skill as a detail row."""
    return {"run": run, "predictor": predictor, "rows": subset, "n": result.rows, "minutes": result.minutes,
            "left out": " ".join(result.left_out), "corr": _f(result.corr), "se": _f(result.se),
            "t": _f(result.t, ".2f"), "detectable": _f(result.detectable), "upper bound": _f(result.upper_bound),
            **extra}


def verdict_word(result: stats.PooledSkill | None, window: Window) -> str:
    """Pass or fail at the cutoff, descriptive before it, and not run for an H2 that H1 never reached."""
    if result is None:
        return "not run"
    return ("pass" if result.passed else "fail") if window.final else "descriptive"


def incremental_rows(result: stats.PooledSkill | None, without: int, window: Window) -> list[dict]:
    """H2 as one detail row, the partial correlation of news and label beyond the 10 price z scores with its error,
    or a row saying it was not run."""
    if result is None:
        return [{"run": RUNS[1], "predictor": "news beyond price scores", "verdict": verdict_word(None, window)}]
    return [skill_row(RUNS[1], "news beyond price scores", "all", result, **{
        "rows without price scores": str(without), "verdict": verdict_word(result, window)})]


def dm_row(name: str, result: stats.PooledDM) -> dict:
    """One squared error comparison as a detail row."""
    return {"comparison": name, "n": result.rows, "minutes": result.minutes, "mse ratio": _f(result.mse_ratio),
            "mean": _f(result.mean), "t": _f(result.t, ".2f")}


def twin_test(run: Run, window: Window) -> tuple[dict, list[dict]]:
    """Learner minus twin gross profit for trades opened by one decision, paired by slot, symbol and decision time,
    one block per session through the version 1 block_test, and each learner's share of long opens."""
    pairs: dict[tuple[int, str, str], dict[str, float]] = {}
    longs: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for slot, origin, symbol, decided, side, gross in run.conn.execute(
            "SELECT b.slot, b.origin, f.symbol, f.decided_at, t.side, t.pnl_gross FROM fills f "
            "JOIN bots b ON b.bot_id = f.bot_id JOIN trades t ON t.id = f.trade_id WHERE f.leg = 'open' "
            "AND f.status = 'filled' AND f.decided_at > ? AND t.closed_at <= ?", (window.start, window.end)):
        kind = TWIN if origin == TWIN_ORIGIN else LEARNER
        pairs.setdefault((slot, symbol, decided), {})[kind] = gross
        if kind == LEARNER:
            longs[slot, symbol].append(side == "long")
    both = [(key, p) for key, p in sorted(pairs.items()) if len(p) == 2]
    launch = parse_iso(window.launch)
    test = stats.session_block_test([session_number(parse_iso(key[2]), launch) for key, _ in both],
                                    [p[LEARNER] for _, p in both], [p[TWIN] for _, p in both])
    shares = [{"run": run.name, "slot": slot, "symbol": symbol, "opens": len(v), "long share": _f(sum(v) / len(v))}
              for (slot, symbol), v in sorted(longs.items())]
    return test, shares


def twin_tables(runs: Sequence[Run], window: Window) -> tuple[list[dict], list[dict], int]:
    """The twin test of each run without a pass label, every learner's share of long opens, and the pairs tested."""
    tests, shares, pairs = [], [], 0
    for run in runs:
        test, mine = twin_test(run, window)
        pairs += test["pairs"]
        tests.append({"run": run.name, **{k: _f(v) if v is None or isinstance(v, float) else v
                                          for k, v in test.items() if k != "passed"}})
        shares += mine
    return tests, shares, pairs


def death_rows(run: Run, window: Window) -> list[dict]:
    """Every death by the end with its bot, kind, home symbol, cause and session counted from launch."""
    return [{"run": run.name, "bot_id": bot_id, "kind": TWIN if origin == TWIN_ORIGIN else LEARNER, "symbol": symbol,
             "died_at": died, "session": session_number(parse_iso(died), parse_iso(window.launch)), "cause": cause}
            for bot_id, died, cause, origin, symbol in run.conn.execute(
                "SELECT d.bot_id, d.died_at, d.cause, b.origin, b.home_symbol FROM deaths d JOIN bots b "
                "ON b.bot_id = d.bot_id WHERE d.died_at <= ? ORDER BY d.died_at, d.bot_id", (window.end,))]


# The scorecards of sections 11.8 and 11.9


SCORECARD_TITLES = ("chooser", "chooser against comparisons", "chooser picks per symbol", "day labels",
                    "day scorecard", "day scorecard per symbol", "scorecards")


def _cells(row: dict) -> dict:
    """A scorecard row for --detail, each figure to four places and a missing one empty."""
    return {key: _f(value) if value is None or isinstance(value, float) else value for key, value in row.items()}


def scorecard_tables(runs: Sequence[Run], window: Window,
                     s: EvaluationConfig) -> tuple[dict[str, list[dict]], tuple[tuple[str, Any], ...]]:
    """The chooser of section 11.8 on each run's bps predictors and the day scorecard of section 11.9 over the
    window, as detail tables, with their counts for the summary. Neither carries a pass label or decides anything, so
    a failure to build them is reported in their place and never changes a verdict or the exit code."""
    try:
        # Imported here and never at the top, because the console script that starts the stock and news processes
        # imports this module, and neither process may ever load the scorecards. It sits inside the try, so even a
        # module that fails to import changes no verdict and no exit code.
        from btc_survivor.v2 import scorecards
        sources = [scorecards.Source(run.name, run.conn, run.config) for run in runs]
        cards = [card for source in sources for card in scorecards.chooser_cards(
            source, window.start, window.end, s.newey_west_lags, s.upper_bound_z)]
        day = scorecards.day_card(sources, window.launch, window.end, s.upper_bound_z)
        late = [(source.name, scorecards.late_forecasts(source, window.start, window.end)) for source in sources]
    except Exception as exc:  # noqa: BLE001, a descriptive section must never change a verdict or the exit code
        return ({"scorecards": [{"problem": f"the scorecards could not be built, {type(exc).__name__} {exc}"}]},
                (("scorecards", output.Word("not computed")),))
    tables: dict[str, list[dict]] = defaultdict(list)
    lines: list[tuple[str, Any]] = []
    for c in cards:
        head = {"run": c.run, "predictor": c.predictor}
        for row in c.strategies:
            extra = {"no scale minutes": c.unscaled_minutes, "skipped symbol minutes": c.skipped} \
                if row["strategy"] == scorecards.CHOOSER else {}
            tables["chooser"].append(_cells({**head, **row, **extra}))
        tables["chooser against comparisons"] += [_cells({**head, **row}) for row in c.differences]
        tables["chooser picks per symbol"].append({**head, **c.picks_per_symbol})
        lines += [(f"{c.run} {c.predictor} chooser picks", output.Count(c.picks)),
                  (f"{c.run} {c.predictor} chooser flat minutes", output.Count(c.flat))]
    tables["day labels"] = [_cells(row) for row in day.labels]
    tables["day scorecard"] = [_cells(row) for row in day.calls]
    tables["day scorecard per symbol"] = [_cells(row) for row in day.symbols]
    lines += [("day scorecard sessions", output.Count(day.sessions)),
              ("day scorecard counted symbol days", output.Count(day.counted)),
              ("day scorecard failed symbol days", output.Count(day.failed))]
    lines += [(f"{name} scorecard forecasts left out for late inputs", output.Count(n)) for name, n in late]
    return dict(tables), tuple(lines)


# The whole evaluation


def open_run(name: str, config: StockConfig) -> Run:
    """One launched run, read only, with the config it recorded at launch."""
    conn = store.open_reader(config.db_path)
    try:
        meta = store.read_meta(conn)
        if "launch_ts" not in meta or "config_toml" not in meta:
            raise EvaluateError(f"the {name} has not launched, its meta holds no launch_ts or config_toml")
        return Run(name, conn, meta, parse_stock_config(meta["config_toml"], config.base_dir),
                   meta.get("experiment_hash") == config.experiment_hash)
    except BaseException:
        conn.close()
        raise


def skill_tables(runs: Sequence[Run], rows: dict[str, dict[str, list[Row]]],
                 stamps: dict[str, dict[str, list[int]]], window: Window, s: EvaluationConfig) -> dict[str, list[dict]]:
    """Every predictor's pooled skill over all its rows, and over its event rows for a score predictor of S2, with a
    verdict only on the confirmatory test, the calibration of the bps predictors and the squared error comparisons."""
    s1, s2 = runs
    tables: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        for predictor, chosen in sorted(rows[run.name].items()):
            result, extra = skill(chosen, s), {}
            if run is s2 and predictor == s.confirmatory_predictor:
                extra["verdict"] = verdict_word(result, window)
            tables["skill"].append(skill_row(run.name, predictor, "all", result, **extra))
            if run is s2 and predictor in EVENT_LINES:
                chosen_events = events(chosen, stamps.get(EVENT_LINES[predictor], {}))
                tables["skill"].append(skill_row(run.name, predictor, "events", skill(chosen_events, s)))
            if predictor in BPS_PREDICTORS:
                tables["calibration"] += [{"run": run.name, "predictor": predictor, "symbol": symbol,
                                           "slope": _f(slope)}
                                          for symbol, slope in stats.symbol_slopes(*_columns(chosen, 4)[1:]).items()]
        own = rows[run.name]
        pop = own.get("population", [])
        tables["squared error"] += [
            dm_row(f"{run.name} population against zero",
                   squared_error([(t, n, p, 0.0, y) for t, n, p, y in pop], s)),
            dm_row(f"{run.name} population against linear_prices",
                   squared_error(paired(pop, own.get("linear_prices", [])), s))]
    news_rows, price_rows = rows[s2.name], rows[s1.name]
    tables["squared error"] += [
        dm_row("news run population against prices run population",
               squared_error(paired(news_rows.get("population", []), price_rows.get("population", [])), s)),
        dm_row("linear_news against linear_prices",
               squared_error(paired(news_rows.get("linear_news", []), news_rows.get("linear_prices", [])), s))]
    return tables


def outcomes(s1: Run, s2: Run, window: Window) -> tuple[list[Outcome], list[dict]]:
    """Every outcome of section 12 in each run and the pairing of the two runs, with each run's round trips per
    session, which section 12 reports beside them without a verdict."""
    news_ok = {ts for (ts,) in s2.conn.execute("SELECT ts FROM news_minutes WHERE healthy = 1 AND ts > ? AND ts <= ?",
                                                (window.launch, window.end))}
    out, trips = [], []
    for run in (s1, s2):
        usable = usable_minutes(run.conn, news_ok, window)
        books = learner_books(run, usable, window)
        out += [*outcome_timeline(run, books, window), *outcome_money(run, books),
                *outcome_closes(run, len(death_rows(run, window)), window)]
        trips.append(round_trips(run, usable, books, window))
    return [*out, outcome_pairing(s1, s2, window)], trips


def trading_tables(runs: Sequence[Run], window: Window) -> dict[str, list[dict]]:
    """The descriptive trading tables of each run, its own trades, its largest notional over equity and its
    learners' mean equity with and without the dead."""
    return {"own trades": [own_trades(run, window) for run in runs],
            "notional over equity": [notional_over_equity(run, window) for run in runs],
            "learner equity": [learner_equity(run, window) for run in runs]}


def evaluate(s1: Run, s2: Run, news: sqlite3.Connection, checks_dir: Path, ledger: Path) -> Evaluation:
    """Every criterion, outcome and descriptive result of sections 11 and 12, from rows up to the end."""
    if s1.meta["launch_ts"] != s2.meta["launch_ts"]:
        raise EvaluateError("the prices run and the news run were not launched together")
    settings = settings_from(s2.meta)
    runs, horizon = (s1, s2), s2.config.time.horizon_minutes
    window = find_window(run_minutes(s2.conn, s2.meta["launch_ts"]), s2.meta["launch_ts"], settings)
    spend = ledger_spend(ledger, window.end)
    labels, compared = check_labels(runs, window, horizon)
    commits, listed = check_commits(runs, window)
    checks = (check_coverage(coverage_records(checks_dir), window, settings.tape_check_sessions),
              check_lookahead(runs, window), labels, check_identical(s1, s2, window, horizon),
              check_invariants(runs, window), check_one_config(s1, s2, news, window), check_spend(spend, window),
              check_local_only(runs, window), commits)
    rows = {run.name: scored_rows(run.conn, window, horizon) for run in runs}
    news_rows = rows[s2.name].get(settings.confirmatory_predictor, [])
    confirmatory = skill(news_rows, settings)
    h2, without = incremental(news_rows, price_scores(s2.conn, window), settings) if confirmatory.passed else (None, 0)
    stamps = receipts(s2.conn, window)
    tables = skill_tables(runs, rows, stamps, window, settings)
    tables["incremental test"] = incremental_rows(h2, without, window)
    comparisons = compare(rows[s2.name], stamps, settings)
    tables["paired comparisons"] = [comparison_row(c) for c in comparisons]
    tables["planned"] = planned_rows(s2.config, settings.final_healthy_session_minutes)
    tables["twins"], tables["long opens"], pairs = twin_tables(runs, window)
    tables.update(trading_tables(runs, window))
    tables["data quality"], share, alert_days = data_quality(s2, news, window)
    tables["spend"] = [{"month": month, "model": model, "dollars": _f(usd)}
                       for month, models in sorted((spend or {}).items()) for model, usd in sorted(models.items())]
    tables["commits"] = [{"commit": commit} for commit in listed]
    found, tables["round trips"] = outcomes(s1, s2, window)
    cards, card_lines = scorecard_tables(runs, window, settings)
    tables.update(cards)
    return Evaluation(window, settings, all(run.file_matches for run in runs), checks, confirmatory, compared,
                      sum(len(v) for r in rows.values() for v in r.values()), tuple(comparisons),
                      tuple(found), tuple(row for run in runs for row in death_rows(run, window)),
                      pairs, share, alert_days, dict(tables), h2, without, card_lines)


def evaluate_runs(s1_path: Path, s2_path: Path, checks: Path | None = None, ledger: Path | None = None) -> Evaluation:
    """Open both runs and news.db read only, evaluate them and close every connection. The coverage records default
    to the checks folder beside the run folders and the key ledger to its place under ~/.config/btc_survivor."""
    s1_config, s2_config = load_run_pair(s1_path, s2_path)
    conns: list[sqlite3.Connection] = []
    try:
        s1 = open_run(RUNS[0], s1_config)
        conns.append(s1.conn)
        s2 = open_run(RUNS[1], s2_config)
        conns.append(s2.conn)
        news = news_db.open_reader(s2_config.news_db_path)
        conns.append(news)
        return evaluate(s1, s2, news, checks or s2_config.data_dir.parent / "checks",
                        Path(ledger or DEFAULT_LEDGER).expanduser())
    finally:
        for conn in conns:
            conn.close()


# Output


def _verdict(matched: list[bool | None]) -> bool | output.Word:
    """Matched when every run matched, not matched when any did not, and not yet otherwise."""
    if any(m is False for m in matched):
        return False
    return output.Word("not yet") if any(m is None for m in matched) else True


def _test_line(result: stats.PooledSkill | None, window: Window) -> bool | output.Word:
    """A test's summary value, its verdict at the cutoff and a word before it or when it was not run."""
    word = verdict_word(result, window)
    return result.passed if word in ("pass", "fail") and result is not None else output.Word(word)


def _own_lines(ev: Evaluation) -> list[tuple[str, Any]]:
    """Each run's learner opens and own opens, counts only, since the own trade share is a descriptive figure."""
    return [line for row in ev.tables.get("own trades", ()) for line in (
        (f"{row['run']} learner opens", output.Count(row["opens"])),
        (f"{row['run']} own opens", output.Count(row["own opens"])))]


def report(ev: Evaluation) -> tuple[output.Report, list[output.Report]]:
    """The default summary of verdicts and counts, and the detail sections --detail adds."""
    w = ev.window
    lines: list[tuple[str, Any]] = [
        ("cutoff reached", w.final), ("run minutes", output.Count(w.run_minutes)),
        ("window sessions", output.Count(len(window_sessions(w)))), ("config files match launch", ev.configs_match)]
    for check in ev.checks:
        lines += [(check.name, check.passed), (f"{check.name} problems", output.Count(len(check.problems)))]
    caveat = ev.fallback_share is not None and ev.fallback_share > FALLBACK_CAVEAT
    lines += [("labels compared", output.Count(ev.labels_compared)), ("run valid", ev.valid),
              ("fallback caveat", output.Word("above a quarter" if caveat else "none")),
              ("spend alert days", output.Count(ev.spend_alert_days)),
              ("confirmatory rows", output.Count(ev.confirmatory.rows)),
              ("confirmatory test", _test_line(ev.confirmatory, w)),
              ("incremental rows", output.Count(0 if ev.incremental is None else ev.incremental.rows)),
              ("rows without price scores", output.Count(ev.without_scores)),
              ("incremental test", _test_line(ev.incremental, w)),
              ("scored rows", output.Count(ev.scored_rows)), ("paired comparisons", output.Count(len(ev.comparisons))),
              ("twin pairs", output.Count(ev.twin_pairs)), *_own_lines(ev), ("deaths", output.Count(len(ev.deaths)))]
    names = list(dict.fromkeys(o.name for o in ev.outcomes))
    lines += [(name, _verdict([o.matched for o in ev.outcomes if o.name == name])) for name in names]
    lines += list(ev.scorecards)
    criteria = [{"check": c.name, "verdict": "pass" if c.passed else "fail", "problem": problem}
                for c in ev.checks for problem in (c.problems or ("none",))]
    matched = [{"run": o.run, "outcome": o.name, "matched": "not yet" if o.matched is None else
                ("matched" if o.matched else "not matched"), "expected": o.expected, "actual": o.actual}
               for o in ev.outcomes]
    detail = [output.Report("criteria", (), tuple(criteria)), output.Report("outcomes", (), tuple(matched)),
              *(output.Report(title, (), tuple(ev.tables.get(title, ()))) for title in (
                  "planned", "skill", "incremental test", "squared error", "paired comparisons", "calibration",
                  "twins", "long opens", "own trades", "round trips", "notional over equity", "learner equity")),
              output.Report("deaths", (), ev.deaths),
              *(output.Report(title, (), tuple(ev.tables.get(title, ()))) for title in (
                  "data quality", "spend", "commits")),
              *(output.Report(title, (), tuple(ev.tables[title])) for title in SCORECARD_TITLES
                if title in ev.tables)]
    return output.Report("evaluate", tuple(lines)), detail


DESCRIPTION = ("Print the success criteria of PLAN_V2 section 11 and the outcomes of section 12 for the prices run and "
               "the news run, read from their databases, news.db, the key ledger and the coverage records, all read "
               "only. By default it prints only verdicts and counts.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The evaluate arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("s1", metavar="S1", type=Path, help="the prices run config, such as configs/v2/s1_prices.toml")
    parser.add_argument("s2", metavar="S2", type=Path, help="the news run config, such as configs/v2/s2_news.toml")
    parser.add_argument("--checks", type=Path, default=None,
                        help="the folder of coverage records that tape-check writes (default the checks folder "
                             "beside the run folders)")
    parser.add_argument("--ledger", type=Path, default=None, help=f"the key ledger (default {DEFAULT_LEDGER})")
    output.add_detail_flag(parser)


def build_parser() -> argparse.ArgumentParser:
    """The parser of python -m btc_survivor.v2.evaluate."""
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 evaluate", description=DESCRIPTION)
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    """Print the evaluation of both runs, exiting 0 when the run is valid so far and 1 when it is not."""
    ev = evaluate_runs(args.s1, args.s2, args.checks, args.ledger)
    summary, detail = report(ev)
    if args.detail:
        output.emit(summary)
        for section in detail:
            output.emit(section, detail=True)
    else:
        output.emit(output.Report(summary.title, summary.summary, tuple(r for s in detail for r in s.rows)))
    return 0 if ev.valid else 1


EXPECTED = (ConfigError, StoreError, NewsDbError, EvaluateError, CalendarError, OSError)


def main(argv: list[str] | None = None) -> int:
    """Run evaluate, printing an expected failure as one error line with exit code 2."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except EXPECTED as exc:
        print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
