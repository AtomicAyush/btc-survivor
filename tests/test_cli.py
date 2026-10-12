"""The reporting commands against a seeded replay database, so every number comes from real engine rows."""

import csv
import sqlite3
import time

import pytest

from btc_survivor.__main__ import main
from btc_survivor.cli import evaluate, explain, export, lineage, status
from btc_survivor.replay import replay_lines
from btc_survivor.synthetic import Fault
from enginekit import deadly_config, synthetic
from helpers import dump_toml, example_dict

NOW = time.time_ns() // 1_000_000


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cli")
    replay_lines(deadly_config(folder / "unused"), synthetic(5, 7200, (Fault(1500, "disconnect", 400),)),
                 folder / "run")
    cfg = deadly_config(folder / "run")
    db = sqlite3.connect(cfg.db_path)
    yield cfg, db
    db.close()


def test_status_lists_every_living_learner_and_warns_when_nothing_is_running(run):
    cfg, db = run
    text = status.render(cfg, NOW)
    meta = __import__("json").loads(db.execute("SELECT meta FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0])
    for bot_id in meta["learners"]:
        assert any(line.split()[0] == str(bot_id) for line in text.splitlines()[3:] if line.strip())
    assert "stage: trading" in text and "the system is not running" in text
    assert "coin flip twins" in status.render(cfg, NOW, twins=True)


def test_explain_shows_a_death_whose_parts_add_up(run):
    cfg, db = run
    dead = db.execute("SELECT bot_id FROM deaths ORDER BY died_at LIMIT 1").fetchone()[0]
    text = explain.render(explain.life(cfg, dead, NOW))
    assert "the parts add up" in text and "The largest loss was fees" in text and "worst trades" in text
    alive = __import__("json").loads(db.execute("SELECT meta FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0])
    assert "alive and" in explain.render(explain.life(cfg, alive["learners"][0], NOW))
    with pytest.raises(LookupError):
        explain.life(cfg, 99_999, NOW)


def test_lineage_walks_back_to_a_random_ancestor(run):
    cfg, db = run
    young = db.execute("SELECT max(bot_id) FROM bots WHERE generation = 2").fetchone()[0]
    rows = lineage.chain(cfg, young, NOW)
    assert [r["generation"] for r in rows] == [2, 1, 0] and rows[-1]["origin"] == "random"
    assert "parent" in lineage.render(rows)


def test_export_writes_every_table_with_its_rows(run, tmp_path):
    cfg, db = run
    counts = export.export(cfg, tmp_path / "csv")
    for table, n in counts.items():
        with (tmp_path / "csv" / f"{table}.csv").open() as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == n + 1 == db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] + 1


def test_evaluate_reports_every_criterion_and_the_arithmetic(run):
    cfg, _ = run
    assert evaluate.evaluate(cfg, 28)["skill"]["skill"]["n"] == 0  # a two hour run is all burn in
    result = evaluate.evaluate(cfg, 28, burn_in_ms=0)
    assert not result["final"] and result["machinery"]["passed"]
    assert result["skill"]["skill"]["n"] > 20 and result["twins"]["pairs"] > 20
    ar = result["arithmetic"]
    assert result["twins"]["blocks"] == 1 and result["twins"]["threshold"] is None  # one 12 hour block so far
    assert ar["learner_opens"] > 0 and ar["forced_opens"] == ar["learner_opens"] and ar["gate_passes_on_opens"] == 0
    assert result["survival"]["learners"]["causes"] == {"fees": result["survival"]["learners"]["deaths"]}
    text = evaluate.render(result)
    assert "descriptive, the final day is day 28" in text and "Outcomes fixed by arithmetic" in text
    skill = result["skill"]["skill"]
    assert skill["upper_95"] == pytest.approx(skill["corr"] + 1.645 * skill["se"])
    assert skill["detectable_nw"] == pytest.approx(3.61 * skill["se"]) and skill["passed"] == (skill["t"] > 2.33)
    assert f"one sided 95 percent upper bound on the correlation {skill['upper_95']:+.4f}" in text
    assert f"a correlation of about {skill['detectable_nw']:.3f} from the realised Newey West error" in text
    assert f"{skill['detectable']:.3f} if the windows were independent" in text and "needs t above 2.33" in text


def test_commands_run_from_the_command_line(run, tmp_path, capsys):
    cfg, _ = run
    data = example_dict()
    data["run"]["data_dir"] = str(cfg.data_dir)
    config = tmp_path / "config.toml"
    config.write_text(dump_toml(data))
    for args in (["status"], ["evaluate"], ["lineage", "5"], ["explain", "5"], ["export", "--out", str(tmp_path / "x")]):
        assert main(["--config", str(config), *args]) == 0
    assert main(["--config", str(config), "explain", "99999"]) == 2
    missing = example_dict()
    missing["run"]["data_dir"] = str(tmp_path / "nothing")
    config.write_text(dump_toml(missing))
    assert main(["--config", str(config), "status"]) == 2
    assert "has this run been started" in capsys.readouterr().err


def test_export_covers_every_table(run, tmp_path):
    cfg, _ = run
    assert set(export.export(cfg, tmp_path / "all")) == {"bots", "trades", "fills", "decisions", "equity_snapshots",
                                                          "deaths", "lineage", "benchmark", "forecasts"}


def test_every_bot_explains_its_losses_and_lists_its_worst_trades_first(run):
    cfg, db = run
    for (bot_id,) in db.execute("SELECT bot_id FROM bots").fetchall():
        x = explain.life(cfg, bot_id, NOW)
        if x["parts"] is not None:
            assert "the parts add up" in explain.render(x), bot_id
        if x["worst"]:
            worst = db.execute("SELECT min(pnl_net) FROM trades WHERE bot_id = ? AND closed_at IS NOT NULL",
                               (bot_id,)).fetchone()[0]
            assert x["worst"][0]["pnl_net"] == worst


def test_explain_counts_a_writeoff_in_the_sum():
    x = {"bot": {"bot_id": 9, "origin": "random", "slot": 0, "generation": 0, "born_at": "2026-01-01T00:00:00.000Z",
                 "margin_bps": 8.0, "lr": 0.0001},
         "death": None, "lineage": None, "current": None, "start": 100.0, "equity": 0.0, "writeoff": 5.0,
         "unrealized": 0.0, "parts": {"fees": 90.0, "funding": 0.0, "rent": 5.0, "stop_loss": 0.0, "other_losses": 10.0},
         "trades": {"n": 1, "wins": 0, "net": -100.0}, "counts": [], "worst": []}
    assert "the parts add up" in explain.render(x)
    x["writeoff"] = 0.0
    assert "DO NOT ADD UP" in explain.render(x)


def test_positions_are_marked_at_the_live_exit_price_and_twins_have_no_forced_countdown(run):
    import json

    from btc_survivor.cli import data

    cfg, db = run
    meta = json.loads(db.execute("SELECT meta FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0])
    meta["checkpoint_ts"] = "2026-01-01T00:00:00.000Z"
    conn = data.connect(cfg.db_path)
    rows = {b["bot_id"]: b for b in data.bots(conn, cfg, meta, {"best_bid": 70_000.0, "best_ask": 90_000.0}, NOW)}
    conn.close()
    held = [d for d in meta["bots"] if d["account"]["side"]]
    assert held
    for d in held:
        acct = d["account"]
        mark = 70_000.0 if acct["side"] == 1 else 90_000.0
        assert rows[d["bot_id"]]["equity"] == pytest.approx(acct["cash"] + acct["side"] * acct["qty"] * (mark - acct["entry_price"]))
    assert all(b["forced_in_hours"] is None for b in rows.values() if b["kind"] == "twin")
    text = status.render(cfg, NOW, twins=True)
    for twin_id in meta["twins"]:
        assert any(line.split()[:1] == [str(twin_id)] for line in text.split("coin flip twins")[1].splitlines())


def test_burn_rate_is_the_daily_loss_over_the_last_day_of_snapshots(tmp_path):
    import sqlite3 as sql

    from btc_survivor.cli import data
    from btc_survivor.clock import MS_PER_HOUR, iso

    db = sql.connect(tmp_path / "b.db")
    db.execute("CREATE TABLE equity_snapshots (bot_id INTEGER, ts TEXT, equity REAL)")
    end = 1_791_000_000_000
    db.executemany("INSERT INTO equity_snapshots VALUES (?, ?, ?)", [
        (1, iso(end - 12 * MS_PER_HOUR), 100.0), (1, iso(end), 90.0),  # 10 dollars in half a day
        (2, iso(end - 30 * MS_PER_HOUR), 100.0), (2, iso(end - 2 * MS_PER_HOUR), 99.0), (2, iso(end), 98.0),
        (3, iso(end - 10 * 60_000), 100.0), (3, iso(end), 50.0),  # under an hour of data, no rate
    ])
    db.commit()
    db.row_factory = sql.Row
    rates = data.burn_rates(db, end)
    assert rates[1] == pytest.approx(20.0) and rates[2] == pytest.approx(12.0) and 3 not in rates


def test_evaluate_counts_the_right_rows_and_stops_at_the_final_day(run):
    cfg, db = run
    from btc_survivor.clock import minute_index, parse_iso

    result = evaluate.evaluate(cfg, 28, burn_in_ms=0)
    labelled = [ts for (ts,) in db.execute("SELECT ts FROM forecasts WHERE predictor = 'population' "
                                           "AND label_bps IS NOT NULL")]
    assert result["skill"]["skill"]["n"] == sum(minute_index(parse_iso(ts)) % cfg.time.horizon_minutes == 0
                                                 for ts in labelled)
    pairs = db.execute(
        "SELECT count(*) FROM fills a JOIN bots ba ON ba.bot_id = a.bot_id JOIN fills b ON b.decided_at = a.decided_at "
        "JOIN bots bb ON bb.bot_id = b.bot_id AND bb.slot = ba.slot JOIN trades ta ON ta.id = a.trade_id "
        "JOIN trades tb ON tb.id = b.trade_id WHERE a.leg = 'open' AND b.leg = 'open' AND a.status = 'filled' "
        "AND b.status = 'filled' AND ba.origin != 'control' AND bb.origin = 'control' "
        "AND ta.closed_at IS NOT NULL AND tb.closed_at IS NOT NULL").fetchone()[0]
    assert result["twins"]["pairs"] == pairs
    early = evaluate.evaluate(cfg, 0.02, burn_in_ms=0)  # 28 healthy minutes, long passed in this run
    cutoff = db.execute("SELECT ts FROM benchmark ORDER BY ts LIMIT 1 OFFSET 27").fetchone()[0]
    assert early["final"] and early["cutoff"] == cutoff and early["healthy_days"] == 0.02
    assert early["twins"]["pairs"] < pairs and early["survival"]["learners"]["deaths"] <= \
        db.execute("SELECT count(*) FROM deaths WHERE died_at <= ?", (cutoff,)).fetchone()[0]
    assert "final, rows up to" in evaluate.render(early)


def test_arithmetic_matches_surviving_founders(tmp_path):
    from btc_survivor.cli.data import threshold_estimate
    from enginekit import fast_config

    replay_lines(fast_config(tmp_path / "unused"), synthetic(4, 3600), tmp_path / "calm")
    cfg = fast_config(tmp_path / "calm")
    ar = evaluate.evaluate(cfg, 28)["arithmetic"]
    assert ar["founders"] == cfg.population.size and ar["founder_round_trips"] > 3
    assert abs(ar["founder_equity_mean"] - ar["expected_founder_equity"]) < 10
    assert ar["expected_first_death_day"] == pytest.approx(
        (cfg.trading.start_equity - threshold_estimate(cfg)) / ar["quiet_burn_per_day"])
