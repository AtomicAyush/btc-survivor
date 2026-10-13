"""The whole path offline acceptance test of item 5 of the improvement plan, on synthetic data in a test folder.

The real news process scores fictional Finnhub items with a fake Anthropic client and a fake FinBERT model into a
temporary news.db, and the real stock process runs both rehearsal runs against a fake Finnhub on 127.0.0.1 over three
synthetic sessions, Tuesday 20 to Thursday 22 October 2026, with the clock and the fakes of v2_pathkit. It is killed
with SIGKILL in the middle of the second session, between the runs' checkpoint commits, and resumes from the
checkpoint both runs hold. Then the gate session's coverage record from the day before, evaluate, status and export run
through the command line. The tests assert no lookahead, S1 and S2 identical until S2 first received a nonzero news
score, labels equal to those recomputed from the candles, S2's news inputs nonzero, run minutes in every session,
learners opening on their own under the zero margin, learners keeping positions across decisions under the holding
rule, every fill on a recorded print traded at or after its due time, the same price_z rows in both runs and the
incremental test reading them. The whole path runs once for the module, in
about a minute, and the acceptance mark lets a quick run leave it out with -m "not acceptance".
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from btc_survivor.clock import iso, parse_iso
from btc_survivor.feed.base import MESSAGE
from btc_survivor.recorder import read_recording
from btc_survivor.v2 import evaluate, sessions
from btc_survivor.v2.config import load_run_pair
from btc_survivor.v2.news.health import FINBERT_SCORER
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.labels import MATURED, recompute_labels
from btc_survivor.v2.stocks.news_inputs import HAIKU_SCORER
from btc_survivor.v2.stocks.population import LEARNER, TWIN
from v2_pathkit import (CANARY, DAYS, FIRST_NEWS, HOLD, RESTART, SYMBOLS, cli, gate_record, news_config,
                        planned_news, query, recording_text, run_path, stock_configs)

pytestmark = pytest.mark.acceptance

OPENS = ("open_long", "open_short", "flip")


class PathRun(SimpleNamespace):
    """What the whole path left behind, shown by name alone in a failed assertion."""

    def __repr__(self) -> str:
        return "path_run"


@pytest.fixture(scope="module")
def path_run(tmp_path_factory):
    """The gate session's coverage record, then launch, three sessions with a kill and a resume and a clean stop, then
    evaluate, status and export on the result."""
    folder = tmp_path_factory.mktemp("acceptance")
    s1, s2 = stock_configs(folder)
    cfgs = load_run_pair(s1, s2)
    config = news_config(folder)
    gate = gate_record(folder, s1)
    result = run_path(folder, s1, s2, config, planned_news())
    ledger = config.path(config.llm.caps.ledger_file)
    commands = {
        "evaluate": cli(folder, "evaluate", str(s1), str(s2), "--checks", str(folder / "checks"), "--ledger",
                        str(ledger)),
        "status": cli(folder, "status", str(s1), str(s2)),
        "export s1": cli(folder, "export", str(s1)),
        "export s2": cli(folder, "export", str(s2)),
    }
    return PathRun(folder=folder, cfgs=cfgs, s1=cfgs[0].db_path, s2=cfgs[1].db_path, news=config.db_path,
                   ledger=ledger, gate=gate, commands=commands, **result)


# Reads


def first_nonzero_news(run) -> str:
    """When S2 first received a line its news inputs take with a nonzero impact, counted or not."""
    nonzero = set(query(run.news, "SELECT item_seq, scorer FROM score_impacts WHERE strength != 0"))
    received = [recv for recv, item, scorer in query(run.s2, "SELECT recv_at, item_seq, scorer FROM news_seen "
                                                             "WHERE primary_use = 1 ORDER BY recv_at")
                if (item, scorer) in nonzero]
    assert received, "S2 never received a nonzero news score"
    return received[0]


def candles(path: Path) -> dict[str, list[StockCandle]]:
    """Every stored candle by symbol, as evaluate reads them."""
    out: dict[str, list[StockCandle]] = defaultdict(list)
    for row in query(path, "SELECT symbol, ts, session_k, open, high, low, close, ref_price, volume, buy_volume, "
                           "sell_volume, prints, price_prints, late_prints, valid FROM stock_candles "
                           "ORDER BY symbol, ts"):
        out[row[0]].append(StockCandle(row[0], parse_iso(row[1]), *row[2:14], bool(row[14])))
    return out


def recorded_prints(folder: Path) -> dict[int, set[tuple[str, float, int]]]:
    """Every print the stock processes recorded, as (symbol, price, vendor time) by its receive stamp."""
    out: dict[int, set[tuple[str, float, int]]] = defaultdict(set)
    for kind, raw, recv_ms, _ in read_recording([folder / "data" / "raw_stocks"]):
        if kind == MESSAGE and raw.startswith('{"data"'):
            out[recv_ms].update((p["s"], p["p"], p["t"]) for p in json.loads(raw)["data"])
    return out


def summary(done: subprocess.CompletedProcess) -> dict[str, str]:
    """The label and value of each summary line of a command's default output."""
    return dict(line.split(": ", 1) for line in done.stdout.splitlines() if ": " in line)


# The tests


def test_the_whole_path_runs_resumes_after_the_kill_and_stops_cleanly(path_run):
    run = path_run
    assert run.gate.returncode == 0, run.gate.stdout + run.gate.stderr
    assert run.stopped == 0, run.errors
    assert "Traceback" not in run.errors and CANARY not in run.errors
    assert HOLD <= run.held_at < RESTART
    assert run.connections == run.token_paths == 2
    subscribe = sorted(json.dumps({"type": "subscribe", "symbol": s}, separators=(",", ":")) for s in SYMBOLS)
    assert all(sorted(frames[:8]) == subscribe for frames in run.frames)
    for path in (run.s1, run.s2):
        assert query(path, "SELECT feed_state FROM runtime_status") == [("stopped",)]
        kinds = [kind for kind, in query(path, "SELECT kind FROM feed_events ORDER BY id")]
        assert kinds.count("code_commit") == kinds.count("lock") == 2 and kinds.count("local_only") >= 4
        assert "resume" in kinds and kinds.count("connected") == 2
        assert query(path, "SELECT count(*) FROM fills WHERE reason = 'restart_flatten'")[0][0] > 0
    logs = "".join(p.read_text(encoding="utf-8") for p in (run.folder / "logs").glob("*.log"))
    assert "rehearsal_s1 rolled back to its checkpoint" in logs and "heartbeat feed=" in logs and CANARY not in logs
    text = recording_text(run.folder / "data" / "raw_stocks")
    assert CANARY not in text and "fixture report" not in text


def test_the_news_process_scored_with_the_fakes(path_run):
    run = path_run
    scorers = dict(query(run.news, "SELECT scorer, count(*) FROM item_scores GROUP BY scorer"))
    assert scorers[HAIKU_SCORER] > 20 and scorers[FINBERT_SCORER] > 20
    statuses = set(query(run.news, "SELECT scorer, status FROM item_scores"))
    assert {(HAIKU_SCORER, "scored"), (HAIKU_SCORER, "malformed"), (FINBERT_SCORER, "scored")} <= statuses
    kinds = dict(query(run.news, "SELECT kind, count(*) FROM llm_calls GROUP BY kind"))
    assert kinds.get("probe", 0) >= 2 and run.calls == sum(kinds.values()) and run.closed
    assert query(run.news, "SELECT count(*) FROM news_heartbeat WHERE healthy = 0") == [(0,)]
    assert query(run.ledger, "SELECT count(*) FROM reservations")[0][0] == run.calls
    assert run.counts["new"] > 20 and run.counts["seed"] == 16


def test_no_forecast_decision_or_score_line_looks_ahead(path_run):
    run = path_run
    for path in (run.s1, run.s2):
        assert query(path, "SELECT count(*) FROM stock_forecasts WHERE inputs_max_recv_at > ts") == [(0,)]
        assert query(path, "SELECT count(*) FROM news_seen WHERE recv_at < scored_at") == [(0,)]
        # A decision takes its symbol's reference price of the minute it closes, never a later one.
        joined = query(path, "SELECT count(*), sum(d.ref_mid = c.ref_price) FROM decisions d JOIN stock_candles c "
                             "ON c.symbol = d.symbol AND c.ts = d.ts")
        assert joined[0][0] == joined[0][1] == query(path, "SELECT count(*) FROM decisions")[0][0] > 0
    scored = dict(query(run.news, "SELECT seq, scored_at FROM item_scores"))
    seen = query(run.s2, "SELECT score_seq, recv_at, scored_at FROM news_seen")
    assert seen and all(scored[seq] == scored_at <= recv for seq, recv, scored_at in seen)
    assert query(run.s2, "SELECT count(*) FROM stock_forecasts WHERE inputs_max_recv_at IS NOT NULL")[0][0] > 0
    assert summary(run.commands["evaluate"])["lookahead audit"] == "pass"


def test_s1_and_s2_are_identical_until_s2_first_receives_a_nonzero_news_score(path_run):
    run = path_run
    cutoff = first_nonzero_news(run)
    assert iso(FIRST_NEWS) <= cutoff < iso(HOLD)
    compared = {
        "decisions": "SELECT * FROM decisions WHERE ts < ? ORDER BY id",
        "fills": "SELECT id, bot_id, trade_id, decided_at, filled_at, leg, side, qty, ref_mid, price, slippage_bps, "
                 "fee, latency_ms, status, reason, symbol, print_price, notional, print_after_due_ms FROM fills "
                 "WHERE decided_at < ? ORDER BY id",
        "trades": "SELECT id, bot_id, side, opened_at, qty, entry_price, fee_open, forced FROM trades "
                  "WHERE opened_at < ? ORDER BY id",
        "snapshots": "SELECT * FROM equity_snapshots WHERE ts < ? ORDER BY bot_id, ts",
        "forecasts": "SELECT ts, symbol, predictor, prediction FROM stock_forecasts WHERE predictor IN "
                     "('population', 'linear_prices') AND ts < ? ORDER BY ts, symbol, predictor",
        "benchmark": "SELECT * FROM benchmark WHERE ts < ? ORDER BY ts",
    }
    for name, sql in compared.items():
        before = query(run.s1, sql, (cutoff,))
        assert before and before == query(run.s2, sql, (cutoff,)), name
    # Once S2's news inputs are nonzero its learners train on them, so its forecasts leave S1's from then on.
    sql = "SELECT ts, symbol, prediction FROM stock_forecasts WHERE predictor = 'population'"
    first, second = ({(ts, s): p for ts, s, p in query(path, sql)} for path in (run.s1, run.s2))
    differ = sorted(key[0] for key in first.keys() | second.keys() if first.get(key) != second.get(key))
    assert differ and differ[0] >= cutoff


def test_labels_recompute_from_the_candles_in_both_runs(path_run):
    run = path_run
    horizon = run.cfgs[1].time.horizon_minutes
    stored = []
    for path in (run.s1, run.s2):
        rows = {(ts, s): (status, entry, exit_price, label) for ts, s, status, entry, exit_price, label in query(
            path, "SELECT ts, symbol, status, entry_price, exit_price, label_bps FROM stock_labels")}
        recomputed = {(iso(label.ts), symbol): (label.status, label.entry_price, label.exit_price, label.label_bps)
                      for symbol, found in candles(path).items() for label in recompute_labels(found, horizon)}
        end = parse_iso(query(path, "SELECT max(ts) FROM stock_candles")[0][0])

        def ended(key):
            return sessions.boundary_of_k(sessions.k_at(parse_iso(key[0])) + horizon) <= end

        assert {k: v for k, v in rows.items() if ended(k)} == {k: v for k, v in recomputed.items() if ended(k)}
        assert sum(value[0] == MATURED for value in recomputed.values()) > 100
        stored.append(rows)
    assert stored[0] == stored[1]
    assert summary(run.commands["evaluate"])["labels recomputed"] == "pass"


def test_s2_news_inputs_become_nonzero_and_s1_reads_no_news(path_run):
    run = path_run
    rows = query(run.s2, "SELECT predictor, sum(prediction != 0), sum(items) FROM stock_forecasts "
                         "WHERE predictor IN ('news', 'news_company', 'news_macro') GROUP BY predictor")
    assert {predictor for predictor, *_ in rows} == {"news", "news_company", "news_macro"}
    assert all(nonzero > 0 and items > 0 for _, nonzero, items in rows)
    assert query(run.s1, "SELECT count(*) FROM stock_forecasts WHERE predictor NOT IN ('population', "
                         "'linear_prices')") == [(0,)]
    assert query(run.s1, "SELECT count(*) FROM news_seen") == [(0,)]
    seen = dict(query(run.s2, "SELECT scorer, count(*) FROM news_seen GROUP BY scorer"))
    assert seen[HAIKU_SCORER] > 10 and seen[FINBERT_SCORER] > 10
    assert query(run.s2, "SELECT count(*) FROM news_seen WHERE fallback = 1")[0][0] > 0
    windows = dict(query(run.s2, "SELECT news_window, count(*) FROM fills WHERE status = 'filled' "
                                 "GROUP BY news_window"))
    assert None not in windows and windows.get(1, 0) > 0
    assert query(run.s1, "SELECT count(*) FROM fills WHERE news_window IS NOT NULL") == [(0,)]


def test_every_session_has_run_minutes_in_both_runs(path_run):
    run = path_run
    rows = [query(path, "SELECT session_date, healthy_minutes, run_minutes FROM sessions ORDER BY session_date")
            for path in (run.s1, run.s2)]
    assert rows[0] == rows[1]
    assert [day for day, *_ in rows[0]] == [d.day.isoformat() for d in DAYS]
    assert all(minutes > 300 for *_, minutes in rows[0])
    healthy = query(run.s2, "SELECT count(*) FROM news_minutes WHERE healthy = 1")[0][0]
    assert healthy >= sum(minutes for *_, minutes in rows[0])
    assert int(summary(run.commands["evaluate"])["run minutes"]) > 600


def test_learners_open_on_their_own_under_the_zero_margin(path_run):
    run = path_run
    for name, path in (("prices run", run.s1), ("news run", run.s2)):
        assert query(path, "SELECT DISTINCT margin_bps FROM bots") == [(0.0,)]
        assert query(path, "SELECT count(*) FROM decisions WHERE margin_bps != 0 "
                           "OR abs(required_edge_bps - cost_estimate_bps) > 1e-9") == [(0,)]
        own = query(path, "SELECT count(*) FROM decisions d JOIN bots b ON b.bot_id = d.bot_id WHERE b.origin != ? "
                          f"AND d.forced = 0 AND d.gate_passed = 1 AND d.action IN {OPENS}", (TWIN,))[0][0]
        assert own > 0 and query(path, "SELECT count(*) FROM decisions WHERE forced = 1")[0][0] > 0
        assert query(path, "SELECT count(*) FROM trades WHERE forced = 0")[0][0] > 0
        assert int(summary(run.commands["evaluate"])[f"{name} own opens"]) > 0


def test_learners_keep_positions_across_decisions_under_the_holding_rule(path_run):
    run = path_run
    for path in (run.s1, run.s2):
        holds = query(path, "SELECT b.origin, d.position_before, d.reason FROM decisions d JOIN bots b "
                            "ON b.bot_id = d.bot_id WHERE d.action = 'hold'")
        assert len(holds) > 20 and {origin for origin, *_ in holds} == {LEARNER}      # only learners keep
        assert all(reason.endswith(f"so the {before} is kept") for _, before, reason in holds)
        # A kept position sends no order, and trades kept across a decision end at a later close.
        assert query(path, "SELECT count(*) FROM decisions d JOIN fills f ON f.bot_id = d.bot_id AND "
                           "f.decided_at = d.ts WHERE d.action = 'hold'") == [(0,)]
        kept = query(path, "SELECT count(*) FROM trades t WHERE t.closed_at IS NOT NULL AND EXISTS (SELECT 1 FROM "
                           "decisions d WHERE d.bot_id = t.bot_id AND d.action = 'hold' AND d.ts > t.opened_at AND "
                           "d.ts < t.closed_at)")
        assert kept[0][0] > 0


def test_every_fill_is_on_a_recorded_print_traded_at_or_after_its_due_time(path_run):
    run = path_run
    prints = recorded_prints(run.folder)
    for path in (run.s1, run.s2):
        fills = query(path, "SELECT decided_at, filled_at, latency_ms, symbol, print_price, print_after_due_ms, "
                            "status FROM fills")
        filled = [f for f in fills if f[6] == "filled"]
        assert len(filled) > 20
        assert all(f[5] is None and f[1] is None for f in fills if f[6] != "filled")
        for decided, filled_at, latency, symbol, price, after, _ in filled:
            due = parse_iso(decided) + latency
            assert 1000 <= latency <= 1300 and after >= 0 and parse_iso(filled_at) >= due
            assert (symbol, price, due + after) in prints[parse_iso(filled_at)], (decided, symbol)


def test_price_z_rows_are_stored_alike_in_both_runs(path_run):
    run = path_run
    sql = "SELECT * FROM price_z ORDER BY k, symbol"
    rows = query(run.s1, sql)
    assert rows and rows == query(run.s2, sql)
    assert sum(row[3] is not None for row in rows) > 100
    news = set(query(run.s2, "SELECT ts, symbol FROM stock_forecasts WHERE predictor = 'news'"))
    assert {(ts, symbol) for _, symbol, ts, *_ in rows} == news
    ks = {(ts, s): k for s, ts, k in query(run.s1, "SELECT symbol, ts, session_k FROM stock_candles")}
    assert all(ks[ts, symbol] == k for k, symbol, ts, *_ in rows)


def test_the_incremental_test_runs_on_the_stored_price_z_rows(path_run):
    # H2 runs only once H1 passes, which a market whose prices ignore the news never gives, so it is called here on
    # the rows evaluate chose for H1.
    run = path_run
    news_run = evaluate.open_run("news run", run.cfgs[1])
    try:
        settings, launch = evaluate.settings_from(news_run.meta), news_run.meta["launch_ts"]
        window = evaluate.find_window(evaluate.run_minutes(news_run.conn, launch), launch, settings)
        rows = evaluate.scored_rows(news_run.conn, window, run.cfgs[1].time.horizon_minutes)["news"]
        scores = evaluate.price_scores(news_run.conn, window)
        result, without = evaluate.incremental(rows, scores, settings)
    finally:
        news_run.conn.close()
    assert len(rows) > 100 and without == sum((ts, symbol) not in scores for ts, symbol, *_ in rows) < len(rows)
    assert result.rows > 0 and result.symbols and result.t is not None


def test_evaluate_status_and_export_run_on_the_result(path_run):
    run = path_run
    prices = {str(p) for (p,) in query(run.s2, "SELECT print_price FROM fills WHERE print_price IS NOT NULL")}
    for name, done in run.commands.items():
        assert done.returncode == 0 and done.stderr == "", (name, done.stdout, done.stderr)
        assert not [price for price in prices if price in done.stdout], name
    evaluated = summary(run.commands["evaluate"])
    for check in ("run valid", "coverage stability", "runs identical", "invariants", "one config prompt and model",
                  "spend within cap", "local only checks", "code commits listed", "config files match launch"):
        assert evaluated[check] == "pass", (check, run.commands["evaluate"].stdout)
    assert int(evaluated["confirmatory rows"]) > 0 and int(evaluated["labels compared"]) > 0
    assert run.commands["evaluate"].stdout.splitlines()[-1].endswith("pass --detail in your own terminal to see them")
    status = summary(run.commands["status"])
    for prefix, path in (("prices run", run.s1), ("news run", run.s2)):
        assert status[f"{prefix} launch"] == "launched" and status[f"{prefix} feed state"] == "stopped"
        assert int(status[f"{prefix} run minutes"]) > 600 and int(status[f"{prefix} fills"]) > 0
        assert int(status[f"{prefix} decisions"]) == query(path, "SELECT count(*) FROM decisions")[0][0]
    for name, cfg in (("export s1", run.cfgs[0]), ("export s2", run.cfgs[1])):
        counts = summary(run.commands[name])
        folder = cfg.data_dir.parent / "exports" / cfg.run.name
        for table in ("decisions", "fills", "price_z", "stock_labels", "news_seen"):
            stored = query(cfg.db_path, f"SELECT count(*) FROM {table}")[0][0]
            lines = (folder / f"{table}.csv").read_text(encoding="utf-8").splitlines()
            assert int(counts[f"{table} rows"]) == stored == len(lines) - 1, (name, table)
