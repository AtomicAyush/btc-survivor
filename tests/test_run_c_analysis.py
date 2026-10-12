"""The run C analyses on a replayed run with deaths, births and parents with skill, and the verifier's numbers."""

import collections
import math
import sqlite3
import statistics
import sys

import pytest

from btc_survivor.cli import data
from btc_survivor.config import load_config
from btc_survivor.population import Population
from btc_survivor.replay import replay_lines
from btc_survivor.store import SCHEMA
from enginekit import fast_config, selecting_config, synthetic
from helpers import ROOT, dump_toml, example_dict

sys.path.insert(0, str(ROOT / "scripts"))
import run_c_analysis as rc  # noqa: E402

RUN_C = load_config(ROOT / "configs" / "run_c.toml")  # read for its settings only, its database is never opened


def recorded_config(data_dir):
    """Eight slots, a 4 minute horizon and a 12 minute timer, a whole number of horizons as in the live runs, and
    parents that must be 1.8 healthy minutes old with 2 closed trades. Births come in bursts a minute or two apart,
    so at some births a newborn is too young and at others it is old enough with no closed trade yet."""
    return fast_config(data_dir, trading={"start_equity": 100.0, "notional": 100.0, "force_trade_after_hours": 0.2},
                       time={"horizon_minutes": 4, "history_minutes_required": 16, "warmup_hours_random": 0.05,
                             "warmup_hours_inherited": 0.02},
                       costs={"fee_profile": "test", "taker_fees_bps": {"test": 400.0}},
                       population={"size": 8, "parent_min_age_hours": 0.03, "parent_min_trades": 2})


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A replay that records, at every tournament, the child's id, the parent's and the engine's own eligible pool."""
    pools = {}
    select = Population.select_parent

    def recording(self):
        eligible = [b.bot_id for b in self.eligible()]
        parent = select(self)
        if parent is not None:
            pools[self.next_bot_id] = (parent.bot_id, eligible)
        return parent

    folder = tmp_path_factory.mktemp("recorded")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Population, "select_parent", recording)
        replay_lines(recorded_config(folder / "unused"), synthetic(5, 3 * 3600), folder / "run_r")
    cfg = recorded_config(folder / "run_r")
    conn = data.connect(cfg.db_path)
    yield cfg, conn, rc.HealthyDays(conn), pools
    conn.close()


@pytest.fixture(scope="module")
def ledger(selecting_run):
    conn = data.connect(selecting_run.db_path)
    days = rc.HealthyDays(conn)
    yield selecting_run, conn, days, rc.opens(conn, selecting_run, days)
    conn.close()


def test_every_filled_open_is_measured_against_the_equity_before_it(ledger):
    cfg, conn, _, rows = ledger
    filled = conn.execute("SELECT count(*) FROM fills WHERE leg = 'open' AND status = 'filled'").fetchone()[0]
    assert len(rows) == filled > 20
    for o in rows:
        assert o["before"] == pytest.approx(o["notional"] / o["equity"])
        assert o["after"] == pytest.approx(o["notional"] / (o["equity"] - o["fee"]))
        snap = conn.execute("SELECT ts, equity FROM equity_snapshots WHERE bot_id = ? AND ts <= ? ORDER BY ts DESC "
                            "LIMIT 1", (o["bot_id"], o["ts"])).fetchone()
        assert o["equity"] <= snap["equity"] + 1e-9  # fees since the snapshot only lower it
    assert max(o["after"] for o in rows) > 2  # equity falls while the notional stays at 100


def _ledger(path, fills, snapshots):
    """A database with three bots, the given fills and the given equity snapshots, opened read only."""
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO bots VALUES (?, ?, NULL, 0, '2026-10-03T04:00:00.000Z', NULL, 'active', 0, 0, 0, ?)",
                   [(1, 0, "random"), (2, 1, "random"), (3, 0, "control")])
    db.executemany("INSERT INTO fills (id, bot_id, decided_at, leg, side, qty, ref_mid, price, fee, latency_ms, "
                   "status, reason) VALUES (?, ?, ?, ?, 'long', ?, 50000, 50000, ?, 1250, ?, ?)", fills)
    db.executemany("INSERT INTO equity_snapshots (bot_id, ts, cash, equity, unrealized, realized_gross, fees_total, "
                   "funding_total, rent_total, stop_loss_total, position, skill_n) "
                   "VALUES (?, ?, 0, ?, 0, 0, 0, 0, 0, 0, 'flat', 0)", snapshots)
    db.commit()
    db.close()
    return data.connect(path)


def test_equity_before_an_open_is_the_snapshot_less_every_fee_since(tmp_path):
    """Hand made ledger with exact answers.

    Bot 1 has a snapshot at 04:10, a fee before it that must not count, a rejected close in the snapshot minute, a
    stop close with a 2 dollar fee at 04:12 and a forced open of 50 dollars at 04:13, so it opens against 98, and a
    later snapshot that must not be used. Bot 2 flips in its snapshot minute, so its open comes after its close's 1.5
    dollar fee and is measured against 78.5. Bot 3 is a twin with no snapshot, measured against start equity, and its
    50 dollar fee at 04:11 must not leak into bot 1.
    """
    t = "2026-10-03T04:{:02d}:00.000Z".format
    fills = [(1, 1, t(9), "close", 0.001, 7.0, "filled", "decision"),
             (2, 3, t(1), "open", 0.002, 1.0, "filled", "decision"),
             (3, 1, t(10), "close", 0.001, 4.0, "rejected", "decision"),
             (4, 3, t(11), "close", 0.002, 50.0, "filled", "decision"),
             (5, 1, t(12), "close", 0.001, 2.0, "filled", "stop"),
             (6, 1, t(13), "open", 0.001, 3.0, "filled", "decision"),
             (7, 2, t(20), "close", 0.0008, 1.5, "filled", "decision"),
             (8, 2, t(20), "open", 0.0008, 1.5, "filled", "decision")]
    conn = _ledger(tmp_path / "ledger.db", fills, [(1, t(10), 100.0), (2, t(20), 80.0), (1, t(14), 1.0)])
    cfg = selecting_config(tmp_path / "unused")
    try:
        rows = rc.opens(conn, cfg, rc.HealthyDays(conn))
    finally:
        conn.close()
    assert [(o["bot_id"], o["kind"], o["ts"]) for o in rows] == [(3, "twin", t(1)), (1, "learner", t(13)),
                                                                  (2, "learner", t(20))]
    twin, stop, flip = rows
    assert twin["equity"] == pytest.approx(100.0) and twin["before"] == pytest.approx(1.0)
    assert twin["after"] == pytest.approx(100 / 99)
    assert stop["equity"] == pytest.approx(98.0) and stop["notional"] == pytest.approx(50.0)
    assert stop["before"] == pytest.approx(50 / 98) and stop["after"] == pytest.approx(50 / 95)
    assert flip["equity"] == pytest.approx(78.5) and flip["before"] == pytest.approx(40 / 78.5)
    assert flip["after"] == pytest.approx(40 / 77)


def _forced_spacings(conn, days):
    """Healthy minutes between each learner's successive filled forced opens."""
    index = {ts: i for i, ts in enumerate(days.minutes)}
    by_bot = collections.defaultdict(list)
    for bot_id, ts in conn.execute("SELECT f.bot_id, f.decided_at FROM fills f JOIN trades t ON t.id = f.trade_id "
                                   "JOIN bots b ON b.bot_id = f.bot_id WHERE b.origin != 'control' AND f.leg = 'open' "
                                   "AND f.status = 'filled' AND t.forced = 1 ORDER BY f.decided_at, f.id"):
        by_bot[bot_id].append(index[ts])
    return collections.Counter(b - a for opens in by_bot.values() for a, b in zip(opens, opens[1:]))


def test_forced_opens_come_the_timer_rounded_up_to_whole_horizons_apart(ledger, recorded):
    """The arithmetic's spacing is the engine's. A 12 minute timer on a 4 minute horizon gives exactly 12, as 720
    minutes on 30 and 240 minute horizons give 12 hours, and a 3 minute timer on a 2 minute horizon gives 4."""
    for cfg, conn, days in (ledger[:3], recorded[:3]):
        spacings = _forced_spacings(conn, days)
        assert sum(spacings.values()) > 40
        assert set(spacings) == {rc.forced_period_minutes(cfg)}
    assert rc.forced_period_minutes(recorded[0]) == 12 and rc.forced_period_minutes(ledger[0]) == 4
    for run in "abcde":
        cfg = load_config(ROOT / "configs" / f"run_{run}.toml")
        assert rc.forced_period_minutes(cfg) == 720
        assert rc.arithmetic_path(cfg, 5.0)["period_hours"] == 12


def test_a_cap_that_never_binds_keeps_every_death_day_and_a_tight_one_stops_at_the_first_open(ledger):
    cfg, conn, days, rows = ledger
    loose = rc.capped_deaths(conn, cfg, rows, 1e9, days)
    assert len(loose) == conn.execute("SELECT count(*) FROM deaths").fetchone()[0] >= 10
    assert all(d["capped_day"] == d["day"] and not d["cap_binds"] for d in loose)
    tight = rc.capped_deaths(conn, cfg, rows, 0.5, days)
    rent = cfg.costs.daily_running_cost
    for d in tight:
        first = next((o for o in rows if o["bot_id"] == d["bot_id"]), None)
        if first is None:
            assert d["capped_day"] == d["day"]
            continue
        threshold = conn.execute("SELECT threshold FROM deaths WHERE bot_id = ?", (d["bot_id"],)).fetchone()[0]
        assert d["cap_binds"] and d["capped_day"] == pytest.approx(first["day"] + (first["equity"] - threshold) / rent)
        assert d["capped_day"] > d["day"]  # living on rent alone outlasts trading at these fees


def test_the_arithmetic_path_of_run_c():
    """Expected costs and no market moves, first open 19 hours after launch, then one every 12 hours."""
    path = rc.arithmetic_path(RUN_C, 5.0)
    assert path["threshold"] == pytest.approx(10.26) and path["max_alive_leverage"] == pytest.approx(48.7, abs=0.05)
    expected = {2: (13.8, 242.85), 5: (21.3, 97.65), 10: (23.8, 49.25), 25: (25.8, 10.53)}
    for multiple, (day, equity) in expected.items():
        assert path["reach"][multiple][0] == pytest.approx(day, abs=0.05)
        assert path["reach"][multiple][1] == pytest.approx(equity, abs=0.01)
    day, equity, before, after = path["last"]
    assert (day, equity) == (pytest.approx(25.8, abs=0.05), pytest.approx(15.08, abs=0.01))
    assert before == pytest.approx(33.2, abs=0.05) and after == pytest.approx(47.5, abs=0.05)
    assert equity - path["leg"] > path["threshold"]  # it survives its last open and dies after the round trip
    assert path["death_day"] == pytest.approx(25.8, abs=0.05)
    assert path["capped_death_day"] == pytest.approx(92.7, abs=0.05)
    assert rc.arithmetic_path(RUN_C, 1e9)["capped_death_day"] == path["death_day"]
    deaths = {run: rc.arithmetic_path(load_config(ROOT / "configs" / f"run_{run}.toml"), 5.0)["death_day"]
              for run in "abde"}
    assert deaths == {"a": pytest.approx(516.3, abs=0.05), "b": pytest.approx(4236.8, abs=0.05),
                      "d": pytest.approx(4237.0, abs=0.05),  # evolution.md's 516 and 4,237 days
                      "e": pytest.approx(8619.8, abs=0.05)}  # run E pays rent alone, 9,998.84 dollars at 1.16 a day


def test_the_verifiers_spacing_of_12_25_hours_reproduces_his_arithmetic():
    path = rc.arithmetic_path(RUN_C, 5.0, period_hours=12.25)
    expected = {2: (14.1, 242.5), 5: (21.7, 97.2), 10: (24.3, 48.7), 25: (25.8, 19.6)}
    for multiple, (day, equity) in expected.items():
        assert path["reach"][multiple][0] == pytest.approx(day, abs=0.05)
        assert path["reach"][multiple][1] == pytest.approx(equity, abs=0.05)
    day, equity, before, after = path["last"]
    assert (day, equity) == (pytest.approx(26.3, abs=0.05), pytest.approx(14.48, abs=0.01))
    assert before == pytest.approx(34.5, abs=0.05) and after == pytest.approx(50.4, abs=0.05)
    assert path["death_day"] == pytest.approx(26.3, abs=0.05)


def _summary(days):
    means = {rho: float(sum(d) / len(d)) for rho, d in days.items()}
    sds = {rho: float((sum((x - means[rho]) ** 2 for x in d) / len(d)) ** 0.5) for rho, d in days.items()}
    assert not any(math.isnan(x) for d in days.values() for x in d)  # every simulated bot died
    return means, sds


def test_the_monte_carlo_with_the_engines_spacing():
    means, sds = _summary(rc.monte_carlo(RUN_C))
    assert means[0.0] == pytest.approx(25.83, abs=0.01) and sds[0.0] == pytest.approx(0.60, abs=0.01)
    assert means[0.1] == pytest.approx(26.16, abs=0.01) and sds[0.1] == pytest.approx(0.61, abs=0.01)
    assert means[0.3] == pytest.approx(26.85, abs=0.01) and sds[0.3] == pytest.approx(0.62, abs=0.01)
    assert means[0.1] - means[0.0] == pytest.approx(0.33, abs=0.01)


def test_the_monte_carlo_reproduces_the_verifier_at_his_spacing():
    means, sds = _summary(rc.monte_carlo(RUN_C, period_hours=12.25))
    assert means[0.0] == pytest.approx(26.32, abs=0.01) and sds[0.0] == pytest.approx(0.61, abs=0.01)
    assert means[0.1] == pytest.approx(26.66, abs=0.01) and sds[0.1] == pytest.approx(0.63, abs=0.01)
    assert means[0.3] == pytest.approx(27.36, abs=0.01)
    assert means[0.1] - means[0.0] == pytest.approx(0.34, abs=0.01)


def test_parents_are_compared_with_the_engines_own_eligible_pool(recorded):
    """The pool at each birth is exactly what Population.eligible held then, less the parent, and the means are
    recomputed here from the snapshots."""
    cfg, conn, days, pools = recorded
    births = rc.parents_against_population(conn, cfg, days)
    assert len(births) == len(pools) == conn.execute("SELECT count(*) FROM lineage").fetchone()[0] >= 4
    narrower = 0
    for b in births:
        parent, eligible = pools[b["bot_id"]]
        assert b["parent_id"] == parent and parent in eligible
        assert sorted(b["eligible_ids"]) == sorted(set(eligible) - {parent})
        living, _ = rc.pools_at(conn, cfg, days, b["born_at"], parent)
        narrower += len(living) > len(b["eligible_ids"])
        skills = []
        for bot_id in b["eligible_ids"]:
            row = conn.execute("SELECT skill_corr FROM equity_snapshots WHERE bot_id = ? AND ts <= ? "
                               "ORDER BY ts DESC LIMIT 1", (bot_id, b["born_at"])).fetchone()
            if row is not None and row[0] is not None:
                skills.append(row[0])
        assert b["eligible_n"] == len(skills)
        if skills:
            assert b["eligible_mean"] == pytest.approx(statistics.fmean(skills))
            assert b["eligible_median"] == pytest.approx(statistics.median(skills))
            if b["parent_skill"] is not None:
                share = sum(x < b["parent_skill"] for x in skills) / len(skills)
                assert b["parent_share_below"] == pytest.approx(share)
        else:
            assert b["eligible_mean"] is None and b["parent_share_below"] is None
    assert narrower  # the age and trade rules left a living learner out at some birth
    assert any(b["eligible_mean"] is not None and b["parent_skill"] is not None for b in births)


def test_the_eligible_pool_follows_the_engines_rules(tmp_path):
    """Hand made ledger with exact answers, a 15 minute age rule, a 2 trade rule and a birth at 04:30.

    Healthy minutes run from 04:00 to 04:30 with 04:10 to 04:14 missing. Founders 1, 5, 6 and 8 and twin 7 were born
    at 03:59:30.5, so founder 1 is old enough with three closed trades. Bot 2, born at 04:15, has stepped exactly 15
    healthy minutes, and bot 3, born at 04:16 with the same trades, only 14. Bot 4 has one closed trade and one open.
    Bot 10 has one trade closed before the birth and one closed in the birth minute after it. Founder 6 died at 04:20
    and bot 11 took its slot. Founder 5 dies in the birth minute and bot 9 takes its slot then, but its close was
    rejected and its death row is dated 04:31, so only the slot shows it gone. Founder 8 is the parent.
    """
    t = "2026-10-03T04:{:02d}:00.000Z".format
    founded = "2026-10-03T03:59:30.500Z"
    db = sqlite3.connect(tmp_path / "pool.db")
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO benchmark VALUES (?, 50000, 100, 100, 100, 100, 8, NULL, 100)",
                   [(t(m),) for m in range(31) if not 10 <= m <= 14])
    bots = [(1, 0, founded, "random"), (5, 1, founded, "random"), (6, 2, founded, "random"),
            (7, 1, founded, "control"), (8, 3, founded, "random"), (4, 4, t(5), "random"), (10, 5, t(9), "random"),
            (2, 6, t(15), "mutated"), (3, 7, t(16), "mutated"), (11, 2, t(20), "mutated"), (9, 1, t(30), "mutated")]
    db.executemany("INSERT INTO bots VALUES (?, ?, NULL, 0, ?, NULL, 'active', 0, 0, 0, ?)", bots)
    closed = {1: [2, 6, 20], 8: [3, 7, 21], 2: [20, 25], 3: [20, 25], 4: [8], 10: [20], 7: [5, 6, 7]}
    trades = [(bot, t(m - 1), t(m).replace(":00.000Z", ":01.250Z")) for bot, ms in closed.items() for m in ms]
    trades += [(4, t(25), None), (10, t(29), "2026-10-03T04:30:01.250Z")]
    db.executemany("INSERT INTO trades (bot_id, side, opened_at, closed_at, qty, entry_price, fee_open, "
                   "slippage_bps_open) VALUES (?, 'long', ?, ?, 0.001, 50000, 0.5, 0)", trades)
    db.executemany("INSERT INTO deaths VALUES (?, ?, 0, 'fees', 0, 0, 0, 0, 0, 0, 10, 9, 3, 0.5)",
                   [(5, "2026-10-03T04:31:01.250Z"), (6, "2026-10-03T04:20:01.250Z")])
    db.commit()
    db.close()
    cfg = fast_config(tmp_path / "unused", population={"parent_min_age_hours": 0.25, "parent_min_trades": 2})
    conn = data.connect(tmp_path / "pool.db")
    try:
        days = rc.HealthyDays(conn)
        assert rc.pools_at(conn, cfg, days, t(30), parent_id=8) == ([1, 4, 10, 2, 3, 11], [1, 2])
        assert rc.pools_at(conn, cfg, days, t(30)) == ([1, 8, 4, 10, 2, 3, 11], [1, 8, 2])
        assert rc.pools_at(conn, cfg, days, t(20))[0] == [1, 5, 8, 4, 10, 2, 3]  # 6 dies in this minute, 5 lives on
    finally:
        conn.close()


def test_parents_on_the_selecting_run(ledger):
    cfg, conn, days, _ = ledger
    births = rc.parents_against_population(conn, cfg, days)
    assert len(births) == conn.execute("SELECT count(*) FROM lineage").fetchone()[0] >= 3
    assert any(b["parent_skill"] is not None for b in births)
    for b in births:
        assert b["parent_skill"] == conn.execute("SELECT parent_skill_corr FROM lineage WHERE bot_id = ?",
                                                 (b["bot_id"],)).fetchone()[0]
        assert b["parent_id"] not in b["eligible_ids"]
        assert b["eligible_n"] <= len(b["eligible_ids"]) and b["eligible_n"] <= b["living_n"] + 1
        if b["parent_share_below"] is not None:
            assert 0 <= b["parent_share_below"] <= 1


def test_the_report_covers_all_five_analyses(ledger):
    cfg, conn, _, rows = ledger
    text = rc.render(conn, cfg, every_open=True)
    for heading in ("1. Notional over equity at every open", "2. The largest leverage", "3. Death days",
                    "4. The spread of death days", "5. Parents' skill against the eligible pool"):
        assert heading in text
    listed = [line for line in text.splitlines() if line.strip()[:2] == "20" and " notional " in line]
    assert len(listed) == len(rows)
    assert "with the cap" in text and "mutated births" in text and "correlation 0.1, death day mean" in text
    assert "parent skill mean" in text and "eligible pool mean" in text
    assert f"one every {rc.forced_period_minutes(cfg) / 60:g} hours as the engine spaces them" in text
    assert "one every 12.25 hours, a spacing given on the command line" in rc.render(conn, cfg, period_hours=12.25)
    dead = {r[0] for r in conn.execute("SELECT bot_id FROM deaths")}
    cap = min(o["before"] for o in rows if o["bot_id"] not in dead and o["kind"] == "learner") - 1e-9
    past = {o["bot_id"] for o in rows if o["kind"] == "learner" and o["before"] > cap} - dead
    assert past and f"{len(past)} living learners have had an open above {cap:g}x" in rc.render(conn, cfg, cap)


def test_the_script_reads_a_copy(selecting_run, tmp_path, capsys):
    copies = tmp_path / "copies"
    (copies / selecting_run.data_dir.name).mkdir(parents=True)
    source = sqlite3.connect(f"file:{selecting_run.db_path}?mode=ro", uri=True)
    target = sqlite3.connect(copies / selecting_run.data_dir.name / "btc_survivor.db")
    source.backup(target)
    source.close()
    target.close()
    settings = example_dict()
    settings["run"]["data_dir"] = str(tmp_path / "nowhere" / selecting_run.data_dir.name)
    config = tmp_path / "config.toml"
    config.write_text(dump_toml(settings))
    assert rc.main([str(config), "--copies", str(copies), "--cap", "3"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(selecting_run.data_dir.name) and "3x margin cap" in out
    assert "skipped, a quiet bot is expected to die on day" in out  # the example config has 10,000 dollars
    assert not (tmp_path / "nowhere").exists()
