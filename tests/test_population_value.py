"""The self financing value of a population, from a hand made ledger and from a replayed run with deaths and births."""

import sqlite3
import sys
import time

import pytest

from btc_survivor.cli import data
from btc_survivor.cli.self_financing import daily, newest_snapshot, value_at
from btc_survivor.clock import MS_PER_MINUTE, iso
from btc_survivor.store import Store
from enginekit import START, deadly_config
from helpers import ROOT, dump_toml, example_dict

sys.path.insert(0, str(ROOT / "scripts"))
import population_value  # noqa: E402

M = MS_PER_MINUTE


def _at(minutes: float) -> str:
    return iso(START + int(minutes * M))


@pytest.fixture
def ledger(tmp_path):
    """Two founders with start equity 100 and a twin. Founder 1 dies at minute 10 owing 5 dollars, so its residual
    is minus 5, and newborn 4 takes its slot. Bot 4 dies at minute 20 with 30 dollars and newborn 5 takes the slot.
    The twin dies too and must never count."""
    cfg = deadly_config(tmp_path / "run")
    cfg.data_dir.mkdir(parents=True)
    Store.open(cfg.db_path).close()
    db = sqlite3.connect(cfg.db_path)
    bots = [(1, 0, _at(0), "random"), (2, 1, _at(0), "random"), (3, 0, _at(0), "control"),
            (4, 0, _at(10), "random"), (5, 0, _at(20), "mutated")]
    db.executemany("INSERT INTO bots (bot_id, slot, generation, born_at, state, margin_bps, lr, seed, origin) "
                   "VALUES (?, ?, 0, ?, 'active', 8, 0.0001, 1, ?)", [(b, s, t, o) for b, s, t, o in bots])
    deaths = [(1, iso(START + 10 * M + 1250), 0.0, 5.0), (4, iso(START + 20 * M + 1250), 30.0, 0.0),
              (3, iso(START + 12 * M), 0.0, 7.0)]
    db.executemany("INSERT INTO deaths VALUES (?, ?, 0, 'fees', 0, 0, 0, 0, 0, ?, 10, ?, 1, 1)",
                   [(b, t, w, f) for b, t, f, w in deaths])
    snaps = [(1, 5, 50.0), (2, 5, 70.0), (2, 15, 80.0), (4, 15, 60.0), (2, 25, 90.0), (5, 25, 95.0), (3, 5, 1.0)]
    db.executemany("INSERT INTO equity_snapshots VALUES (?, ?, 0, ?, 0, 0, 0, 0, 0, 0, 'flat', NULL, NULL, NULL, 0)",
                   [(b, _at(m), e) for b, m, e in snaps])
    db.executemany("INSERT INTO benchmark VALUES (?, 1, ?, 100, 0, 0, 2, NULL, ?)",
                   [(_at(m), 100 + 0.1 * m, 100 - 0.01 * m) for m in range(26)])
    db.commit()
    db.close()
    conn = data.connect(cfg.db_path)
    yield cfg, conn
    conn.close()


def test_the_value_counts_every_learner_ever_born_and_the_money_births_needed(ledger):
    cfg, conn = ledger
    v = value_at(conn, cfg)
    assert v["at"] == _at(25) == newest_snapshot(conn)
    assert (v["founders"], v["born"], v["births"], v["deaths"], v["living"]) == (2, 4, 2, 2, 2)
    assert v["living_equity"] == 185.0 and v["dead_residual"] == 25.0 and v["writeoffs"] == 5.0
    assert v["value"] == pytest.approx(185 + 25 - 200) == pytest.approx(2 * 100 + (-5 - 100) + (90 - 100) + (30 - 100)
                                                                         + (95 - 100))
    assert v["value_per_slot"] == pytest.approx(5.0) and v["return"] == pytest.approx(-0.95)
    assert v["outside_money"] == pytest.approx(175.0)  # minus 5 and 100 at minute 10, plus 30 and minus 100 at 20
    assert v["living_mean"] == pytest.approx(92.5)  # what the living mean shows, far above the 5 per slot
    assert v["buy_hold"] == pytest.approx(102.5) and v["buy_hold_return"] == pytest.approx(0.025)
    assert v["cash_return"] == pytest.approx(-0.0025)


def test_earlier_values_use_only_what_had_happened(ledger):
    cfg, conn = ledger
    before = value_at(conn, cfg, _at(5))
    assert before["births"] == 0 and before["outside_money"] == 0.0
    assert before["value_per_slot"] == pytest.approx(60.0) == before["living_mean"]  # no deaths, the two agree
    dying = value_at(conn, cfg, _at(10))  # founder 1 left its slot, its close has not filled, newborn 4 holds 100
    assert dying["deaths"] == 0 and dying["births"] == 1 and dying["value"] == pytest.approx(50 + 70 + 100 - 100)
    assert dying["outside_money"] == pytest.approx(100.0)
    middle = value_at(conn, cfg, _at(15))
    assert middle["value"] == pytest.approx(80 + 60 - 5 - 100) and middle["outside_money"] == pytest.approx(105.0)
    rows = daily(conn, cfg)
    assert [r["at"] for r in rows] == [_at(25)]  # under a day of data, only the newest value


def _expected(conn, cfg, at):
    """The value and outside money from plain SQL and a minute by minute pool, written apart from the module."""
    start = cfg.trading.start_equity
    first = conn.execute("SELECT min(born_at) FROM bots").fetchone()[0]
    living = 0.0
    for (bot_id,) in conn.execute("SELECT bot_id FROM bots b WHERE origin != 'control' AND born_at <= ? AND NOT EXISTS "
                                  "(SELECT 1 FROM deaths d WHERE d.bot_id = b.bot_id AND d.died_at <= ?)", (at, at)):
        row = conn.execute("SELECT equity FROM equity_snapshots WHERE bot_id = ? AND ts <= ? ORDER BY ts DESC LIMIT 1",
                           (bot_id, at)).fetchone()
        living += row[0] if row else start
    dead = conn.execute("SELECT coalesce(sum(final_equity - writeoff), 0) FROM deaths d JOIN bots b USING (bot_id) "
                        "WHERE b.origin != 'control' AND d.died_at <= ?", (at,)).fetchone()[0]
    births = [r[0][:16] for r in conn.execute("SELECT born_at FROM bots WHERE origin != 'control' AND born_at > ? "
                                              "AND born_at <= ?", (first, at))]
    residuals = [(r[0][:16], r[1]) for r in conn.execute(
        "SELECT d.died_at, d.final_equity - d.writeoff FROM deaths d JOIN bots b USING (bot_id) "
        "WHERE b.origin != 'control' AND d.died_at <= ?", (at,))]
    lowest = 0.0
    for minute in sorted({m for m in births} | {m for m, _ in residuals}):
        pool = sum(r for m, r in residuals if m <= minute) - start * sum(m <= minute for m in births)
        lowest = min(lowest, pool)
    return living + dead - start * len(births), -lowest


def test_a_replayed_run_with_deaths_and_births(selecting_run):
    cfg = selecting_run
    conn = data.connect(cfg.db_path)
    try:
        v = value_at(conn, cfg)
        assert v["deaths"] >= 5 and v["births"] >= 5 and v["founders"] == cfg.population.size
        value, outside = _expected(conn, cfg, v["at"])
        assert v["value"] == pytest.approx(value) and v["outside_money"] == pytest.approx(outside) and outside > 0
        assert v["value_per_slot"] < v["living_mean"]  # refilling at deaths flatters the living mean
        first_birth = conn.execute("SELECT min(born_at) FROM bots WHERE origin != 'control' AND born_at > "
                                   "(SELECT min(born_at) FROM bots)").fetchone()[0]
        early = conn.execute("SELECT max(ts) FROM equity_snapshots WHERE ts < ?", (first_birth,)).fetchone()[0]
        e = value_at(conn, cfg, early)
        mean = conn.execute("SELECT avg(equity) FROM equity_snapshots s JOIN bots b USING (bot_id) "
                            "WHERE s.ts = ? AND b.origin != 'control'", (early,)).fetchone()[0]
        assert e["births"] == 0 and e["value_per_slot"] == pytest.approx(mean) and e["outside_money"] == 0.0
        for row in daily(conn, cfg):
            value, outside = _expected(conn, cfg, row["at"])
            assert row["value"] == pytest.approx(value) and row["outside_money"] == pytest.approx(outside)
    finally:
        conn.close()


def test_the_dashboard_shows_the_same_value(selecting_run):
    from btc_survivor.dashboard import run_state

    state = run_state(selecting_run, time.time_ns() // 1_000_000)
    conn = data.connect(selecting_run.db_path)
    try:
        assert state["self_financing"] == value_at(conn, selecting_run)
    finally:
        conn.close()


def test_the_script_reads_a_copy_and_prints_the_value(selecting_run, tmp_path, capsys):
    copies = tmp_path / "copies"
    (copies / selecting_run.data_dir.name).mkdir(parents=True)
    source = sqlite3.connect(f"file:{selecting_run.db_path}?mode=ro", uri=True)
    target = sqlite3.connect(copies / selecting_run.data_dir.name / "btc_survivor.db")
    source.backup(target)
    source.close()
    target.close()
    settings = example_dict()
    settings["run"]["data_dir"] = str(tmp_path / "nowhere" / selecting_run.data_dir.name)
    settings["trading"].update(start_equity=100.0, notional=100.0)
    config = tmp_path / "config.toml"
    config.write_text(dump_toml(settings))
    assert population_value.main([str(config), "--copies", str(copies)]) == 0
    out = capsys.readouterr().out
    conn = data.connect(selecting_run.db_path)
    v = value_at(conn, selecting_run)
    conn.close()
    assert f"value over every learner ever born {v['value']:,.2f}" in out
    assert f"outside money injected by births {v['outside_money']:,.2f}" in out
    assert "self financing return" in out and "living learners only" in out
    assert not (tmp_path / "nowhere").exists()  # nothing was read from or made at the configured folder
