"""The status command on seeded run databases, its default output of counts, state words, timings and the hash, the
rows --detail adds, and the refusal of a database of another kind."""

import sqlite3

import pytest

from btc_survivor.clock import parse_iso
from btc_survivor.v2 import redact, status
from btc_survivor.v2.config import load_run_pair
from btc_survivor.v2.stocks import store
from test_v2_store import contents, news_database, v1_database
import v2_runkit as kit

CANARIES = (str(kit.CANARY_PRICE), str(kit.CANARY_FILL))


class Clock:
    """A wall clock fixed at the seeded data's now."""

    def now_ms(self):
        return parse_iso(kit.NOW)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return kit.seeded(tmp_path_factory.mktemp("status"))


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(status, "LiveClock", Clock)


def run_status(capsys, *argv):
    code = status.main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


RUN_LINES = [
    "launch: launched",
    f"experiment hash: {kit.EXPERIMENT}",
    "feed state: healthy",
    "session state: closed",
    "heartbeat age: 60.000 s",
    "checkpoint age: 300.000 s",
    "living learners: 2",
    "living twins: 1",
    "deaths: 1",
    "open positions: 2",
    "closed trades: 2",
    "decisions: {decisions}",
    "forced decisions: {forced}",
    "gate passed decisions: 2",
    "fills: 6",
    "rejected fills: 1",
    "healthy symbols: 7",
    "other symbols: 1",
    "sessions: 4",
    "run minutes: 1552",
    "matured labels: 376",
    "pending labels: 8",
    "gap labels: 0",
    "forecast rows: {forecasts}",
    "unhealthy news minutes: {unhealthy}",
    "local only checks passed: 1",
    "local only checks failed: 1",
    "last local only check: fail",
]


def test_default_output_holds_only_counts_words_timings_and_the_hash(runs, capsys):
    code, out, err = run_status(capsys, runs.s1, runs.s2)
    prices = [f"prices run {line}".format(decisions=7, forced=2, forecasts=768, unhealthy=0) for line in RUN_LINES]
    news = [f"news run {line}".format(decisions=8, forced=3, forecasts=1152, unhealthy=1) for line in RUN_LINES]
    assert (code, err) == (0, "")
    hidden = "26 rows hidden, pass --detail in your own terminal to see them"
    assert out.splitlines() == ["status", *prices, *news, hidden]
    assert not [canary for canary in CANARIES if canary in out]


def test_detail_adds_symbol_bot_and_runtime_rows_for_each_run(runs, capsys):
    code, out, _ = run_status(capsys, runs.s1, runs.s2, "--detail")
    lines = out.splitlines()
    assert code == 0 and "hidden" not in out
    for title in ("prices run symbols", "prices run bots", "prices run runtime", "news run symbols", "news run bots",
                  "news run runtime"):
        assert lines.count(title) == 1, title
    assert str(kit.CANARY_PRICE) in out  # AAPL's last reference price, for the owner's own terminal
    bots = lines[lines.index("prices run bots") + 1:lines.index("prices run runtime")]
    assert bots[0].split() == ["bot_id", "slot", "kind", "home_symbol", "state", "equity", "trades", "position", "qty",
                               "entry_price"]
    assert [row.split()[:5] for row in bots[1:]] == [["1", "0", "learner", "AAPL", "active"],
                                                     ["2", "6", "learner", "TSLA", "active"],
                                                     ["3", "0", "twin", "AAPL", "active"],
                                                     ["4", "6", "twin", "TSLA", "dead"]]


def test_bot_rows_put_learners_first_with_equity_trades_and_position(runs):
    conn = store.open_reader(runs.s1_db)
    try:
        rows = status.bot_rows(conn)
    finally:
        conn.close()
    assert [(r["bot_id"], r["kind"], r["equity"], r["trades"], r["position"], r["qty"]) for r in rows] == [
        (1, "learner", 9993.0, 1, "flat", None), (2, "learner", 9994.0, 1, "long", 2.0),
        (3, "twin", 9995.0, 0, "short", 2.0), (4, "twin", 9996.0, 0, "flat", None)]
    assert rows[1]["entry_price"] == kit.CANARY_FILL


def test_a_never_launched_empty_run_reads_as_words_and_zeros(tmp_path, capsys):
    s1, s2, _ = kit.configs(tmp_path)
    for config in load_run_pair(s1, s2):
        store.StockStore.open(config.db_path).close()
    code, out, _ = run_status(capsys, s1, s2)
    lines = out.splitlines()
    assert code == 0 and lines[1:7] == ["prices run launch: not launched", "prices run experiment hash: not launched",
                                        "prices run feed state: unknown", "prices run session state: unknown",
                                        "prices run heartbeat age: never", "prices run checkpoint age: never"]
    assert "prices run last local only check: none" in lines and "news run decisions: 0" in lines
    assert lines[-1] == "news run last local only check: none"


def test_state_words_that_are_not_plain_words_read_as_unknown():
    assert str(status.word("healthy")) == "healthy"
    assert str(status.word("pre_open")) == "pre_open"
    assert str(status.word("Stale 5")) == "unknown" and str(status.word(None)) == "unknown"
    assert str(status.experiment({"experiment_hash": "1234"})) == "invalid"


def test_an_unreadable_local_only_detail_counts_as_failed(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE feed_events (id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, detail TEXT, symbol TEXT)")
    conn.executemany("INSERT INTO feed_events (ts, kind, detail) VALUES ('t', 'local_only', ?)",
                     [('{"passed": true}',), ("not json",), ('["passed"]',), ('{"passed": "yes"}',)])
    assert status.local_only_checks(conn) == (1, 3, False)


@pytest.mark.parametrize("make, problem", [(news_database, "meta.kind is 'news', not 'stocks'"),
                                           (v1_database, "it has no meta.kind")])
def test_status_refuses_a_news_or_version_1_database_and_leaves_it_unchanged(tmp_path, capsys, make, problem):
    runs = kit.seeded(tmp_path)
    runs.s1_db.unlink()
    for suffix in ("-wal", "-shm"):
        runs.s1_db.with_name(runs.s1_db.name + suffix).unlink(missing_ok=True)
    make(runs.s1_db)
    before = contents(runs.s1_db)
    code, out, err = run_status(capsys, runs.s1, runs.s2)
    assert (code, out) == (2, "")
    assert err.startswith("error: ") and "is not a stock run database" in err and problem in err
    assert contents(runs.s1_db) == before


def test_status_refuses_a_missing_database_without_making_one(tmp_path, capsys):
    s1, s2, _ = kit.configs(tmp_path)
    code, out, err = run_status(capsys, s1, s2)
    assert (code, out) == (2, "") and "stocks.db does not exist" in err
    assert not (tmp_path / "data").exists()


def test_status_refuses_the_runs_in_the_wrong_order(runs, capsys):
    code, _, err = run_status(capsys, runs.s2, runs.s1)
    assert code == 2 and "the first config must be the prices run" in err


def test_an_error_line_never_shows_a_secret(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    redact.register("fake-status-canary")
    s1, s2, _ = kit.configs(tmp_path / "fake-status-canary")
    code, _, err = run_status(capsys, s1, s2)
    assert code == 2 and "fake-status-canary" not in err and redact.MASK in err
