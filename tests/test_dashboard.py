"""The read only dashboard, its JSON view and the local server."""

import json
import math
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from btc_survivor.dashboard import clean, make_handler, page_version, run_state
from btc_survivor.replay import replay_lines
from enginekit import deadly_config, synthetic


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dash")
    replay_lines(deadly_config(folder / "unused"), synthetic(5, 3600), folder / "run")
    return deadly_config(folder / "run")


def test_run_state_reads_a_run_and_reports_a_missing_one(cfg, tmp_path):
    now = time.time_ns() // 1_000_000
    state = run_state(cfg, now)
    assert state["bots"] and 0 < len(state["equity_curve"]) <= 400
    assert state["skill"] == {} and state["decisions"] and state["fills"]  # an hour long run is all burn in
    assert "error" in run_state(deadly_config(tmp_path / "nothing"), now)


def test_clean_turns_infinities_into_null():
    assert clean({"a": [math.inf, 1.0, math.nan], "b": {"c": -math.inf}}) == {"a": [None, 1.0, None], "b": {"c": None}}


def test_server_serves_the_page_and_valid_json(cfg, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler([cfg, deadly_config(tmp_path / "nothing")]))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<title>BTC Survivor</title>" in page and "/api/state" in page
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert [r["name"] for r in state["runs"]] == ["run", "nothing"] and "error" in state["runs"][1]
        assert state["page_version"] == page_version()  # an open tab reloads itself when this changes
        reply = urllib.request.urlopen(base + "/api/state")
        assert "frame-ancestors 'none'" in reply.headers["Content-Security-Policy"]
        assert json.loads(reply.read())["now_ms"] == state["now_ms"]  # served from the five second cache
        assert str(tmp_path) not in json.dumps(state)  # errors never reveal paths on this machine
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/nope")
    finally:
        server.shutdown()
        server.server_close()


def test_reads_are_read_only_even_on_awkward_paths(cfg, tmp_path):
    import shutil
    import sqlite3

    from btc_survivor.cli import data

    conn = data.connect(cfg.db_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE x (a)")
    conn.close()
    odd = tmp_path / "a#b?c%d"
    odd.mkdir()
    copy = sqlite3.connect(odd / "btc_survivor.db")  # a plain copy with no WAL files, as after a clean close
    sqlite3.connect(cfg.db_path).backup(copy)
    copy.execute("PRAGMA journal_mode=DELETE")
    copy.close()
    assert not (odd / "btc_survivor.db-wal").exists()
    before = sorted(p.name for p in odd.iterdir())  # some SQLite builds leave an empty -shm from the copy itself
    conn = data.connect(odd / "btc_survivor.db")
    assert conn.execute("SELECT count(*) FROM bots").fetchone()[0] > 0
    conn.close()
    assert sorted(p.name for p in odd.iterdir()) == before  # reading created nothing
    shutil.rmtree(odd)


def test_server_binds_to_this_machine_only(monkeypatch, cfg):
    import btc_survivor.dashboard as dashboard

    seen = {}

    class Stub:
        def __init__(self, address, handler):
            seen["address"] = address

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(dashboard, "ThreadingHTTPServer", Stub)
    dashboard.serve([cfg], 8123)
    assert seen["address"] == ("127.0.0.1", 8123)


def test_chart_keeps_the_newest_point_and_skill_uses_the_criterion_rows(monkeypatch, cfg):
    import btc_survivor.dashboard as dashboard
    from btc_survivor.cli import data, evaluate

    monkeypatch.setattr(dashboard, "CHART_POINTS", 7)
    conn = data.connect(cfg.db_path)
    curve = dashboard.equity_curve(conn)
    newest = conn.execute("SELECT max(ts) FROM benchmark").fetchone()[0]
    assert len(curve) <= 8 and curve[-1]["ts"] == newest
    count = conn.execute("SELECT count(*) FROM benchmark").fetchone()[0]
    assert dashboard.chart_step(conn) == -(-count // 7) > 1  # minutes per point, shown on the Timing card
    monkeypatch.setattr(dashboard, "forecasts", lambda c, h: evaluate.forecasts(c, h, 0))
    skill = dashboard.skill_so_far(conn, cfg.time.horizon_minutes)
    expected = evaluate.criterion_skill(conn, cfg.time.horizon_minutes, 0)
    conn.close()
    assert skill["population"]["windows"] == expected["skill"]["n"] > 10
    assert skill["population"]["corr"] == pytest.approx(expected["skill"]["corr"])


def test_one_broken_run_never_blanks_the_others(monkeypatch, cfg):
    import btc_survivor.dashboard as dashboard

    monkeypatch.setattr(dashboard.data, "bots", lambda *a: (_ for _ in ()).throw(KeyError("learners")))
    state = dashboard.run_state(cfg, time.time_ns() // 1_000_000)
    assert "cannot be read" in state["error"] and "/" not in state["error"]
    assert dashboard.run_names([cfg, cfg, deadly_config(cfg.data_dir.parent / "other")]) == ["run 1", "run 2", "other"]
