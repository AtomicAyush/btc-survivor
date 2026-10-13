"""The read only dashboard, its JSON view and the local server."""

import json
import math
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from btc_survivor.dashboard import clean, make_handler, page_version, run_state
from btc_survivor.replay import replay_lines
import dashkit
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
    for key in ("t", "upper_95", "detectable_nw"):  # the same numbers evaluate prints beside the pass rule
        assert skill["population"][key] == pytest.approx(expected["skill"][key])
    assert skill["population"]["passed"] == expected["skill"]["passed"]


def test_one_broken_run_never_blanks_the_others(monkeypatch, cfg):
    import btc_survivor.dashboard as dashboard

    monkeypatch.setattr(dashboard.data, "bots", lambda *a: (_ for _ in ()).throw(KeyError("learners")))
    state = dashboard.run_state(cfg, time.time_ns() // 1_000_000)
    assert "cannot be read" in state["error"] and "/" not in state["error"]
    assert dashboard.run_names([cfg, cfg, deadly_config(cfg.data_dir.parent / "other")]) == ["run 1", "run 2", "other"]


# The decisions explorer

DECISION_COLUMNS = ("id", "bot_id", "ts", "position_before", "ref_mid", "prediction_bps", "required_edge_bps",
                    "cost_estimate_bps", "margin_bps", "gate_passed", "forced", "action", "reason", "label_bps")
FLIP_TS, HOLD_TS, ODD_TS = "2026-10-03T04:50:30.000Z", "2026-10-03T04:52:30.000Z", "2026-10-03T04:59:30.000Z"
OLD_TS, REJECTED_TS = "2026-10-03T02:00:00.000Z", "2026-10-03T04:44:30.000Z"
TWIN_OPEN_TS = "2026-10-03T04:57:30.000Z"
INSERTED = (FLIP_TS, HOLD_TS, ODD_TS, OLD_TS, TWIN_OPEN_TS)
ODD_REASON = "edge case 100% sure_thing with a \\ backslash"


@pytest.fixture(scope="module")
def rich(cfg, tmp_path_factory):
    """The replayed run plus a learner flip that passed the gate, a hold, an open the feed rejected, a reason full of
    LIKE wildcards, an old decision whose label never matured, and two twin decisions whose coin disagrees with the
    prediction they copied, a flip beside the learner's flip and an open. Their times sit between the replay's
    minutes, so no real row shares them."""
    import sqlite3

    folder = tmp_path_factory.mktemp("explore") / "rich"
    folder.mkdir()
    conn = sqlite3.connect(folder / "btc_survivor.db")
    src = sqlite3.connect(cfg.db_path)
    src.backup(conn)
    src.close()
    conn.execute("PRAGMA journal_mode=DELETE")
    decision = ("INSERT INTO decisions (bot_id, ts, position_before, ref_mid, prediction_bps, required_edge_bps, "
                "cost_estimate_bps, margin_bps, gate_passed, forced, action, reason, label_bps) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    conn.execute(decision, (1, FLIP_TS, "short", 80000.0, 25.0, 20.0, 12.0, 8.0, 1, 0, "flip",
                            "prediction 25.00 bps clears required edge 20.00 bps", 30.0))
    conn.execute(decision, (1, HOLD_TS, "long", 80100.0, 22.0, 20.0, 12.0, 8.0, 1, 0, "hold",
                            "prediction 22.00 bps clears required edge 20.00 bps", -5.0))
    conn.execute(decision, (2, ODD_TS, "flat", 80100.0, 1.0, 20.0, 12.0, 8.0, 0, 0, "none", ODD_REASON, None))
    conn.execute(decision, (3, REJECTED_TS, "flat", 80100.0, -2.0, 20.0, 12.0, 8.0, 0, 1, "open_short",
                            "forced by the exploration timer, prediction -2.00 bps", 4.0))
    conn.execute(decision, (2, OLD_TS, "flat", 80100.0, 0.1, 20.0, 12.0, 8.0, 0, 0, "none",
                            "prediction 0.10 bps is below required edge 20.00 bps", None))
    twin = conn.execute("SELECT min(bot_id) FROM bots WHERE origin = 'control' AND slot = "
                        "(SELECT slot FROM bots WHERE bot_id = 1)").fetchone()[0]
    conn.execute(decision, (twin, FLIP_TS, "long", 80000.0, 25.0, 20.0, 12.0, 8.0, 1, 0, "flip",
                            "twin of bot 1 copying flip, the coin chose short", 30.0))
    conn.execute(decision, (twin, TWIN_OPEN_TS, "flat", 80100.0, 6.0, 20.0, 12.0, 8.0, 0, 0, "open_short",
                            "twin of bot 1 copying open_long, the coin chose short", -12.0))
    trade = ("INSERT INTO trades (id, bot_id, side, opened_at, closed_at, qty, entry_price, exit_price, fee_open, "
             "fee_close, funding, slippage_bps_open, slippage_bps_close, pnl_gross, pnl_net, forced, exit_reason) "
             "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    conn.execute(trade, (9001, 1, "short", "2026-10-03T04:40:31.250Z", "2026-10-03T04:50:31.250Z", 0.01, 80200.0,
                         80010.0, 0.8, 0.8, 0.1, 0.5, 0.6, 1.9, 0.2, 0, "flip"))
    conn.execute(trade, (9002, 1, "long", "2026-10-03T04:50:31.250Z", None, 0.01, 80010.0, None, 0.8, None, 0.0,
                         0.7, None, None, None, 0, None))
    fill = ("INSERT INTO fills (bot_id, trade_id, decided_at, filled_at, leg, side, qty, ref_mid, book_avg_price, "
            "price, levels_consumed, slippage_bps, fee, latency_ms, status, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    conn.execute(fill, (1, 9001, FLIP_TS, "2026-10-03T04:50:31.250Z", "close", "buy", 0.01, 80000.0, 80005.0,
                        80010.0, 1, 0.6, 0.8, 1250, "filled", "decision"))
    conn.execute(fill, (1, 9002, FLIP_TS, "2026-10-03T04:50:31.250Z", "open", "buy", 0.01, 80000.0, 80006.0,
                        80010.0, 2, 0.7, 0.8, 1250, "filled", "decision"))
    conn.execute(fill, (1, 9001, FLIP_TS, "2026-10-03T04:50:35.000Z", "close", "buy", 0.01, 80000.0, 80005.0,
                        80010.0, 1, 0.6, 0.8, 1250, "filled", "stop"))  # another order in that minute, never shown
    conn.execute(fill, (3, None, REJECTED_TS, None, "open", "sell", 0.01, 80100.0, None, None, None, None, None, 1100,
                        "rejected", "feed_not_healthy"))
    conn.commit()
    conn.close()
    return deadly_config(folder)


@pytest.fixture(scope="module")
def explorer(rich, tmp_path_factory):
    """A server for the rich run and a run with no database, and a getter that returns (status, JSON body)."""
    missing = deadly_config(tmp_path_factory.mktemp("gone") / "nothing")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler([rich, missing]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path: str) -> tuple[int, dict]:
        try:
            reply = urllib.request.urlopen(base + path)
        except urllib.error.HTTPError as err:
            body = err.read().decode()
            assert "Traceback" not in body and 'File "' not in body and "/" not in json.loads(body)["error"]
            return err.code, json.loads(body)
        assert reply.headers["Content-Type"] == "application/json"
        assert "frame-ancestors 'none'" in reply.headers["Content-Security-Policy"]
        return reply.status, json.loads(reply.read())

    yield get
    server.shutdown()
    server.server_close()


def _all_rows(cfg) -> tuple[list[dict], str]:
    return dashkit.decision_rows(cfg)


def _expected(cfg, params: dict) -> tuple[list[int], dict, dict]:
    """The ids a request should return, newest first, its counts per action and its summary, worked out in Python."""
    return dashkit.expected(cfg, params)


def _fetch_all(get, params: dict, per: int = 200) -> tuple[list[dict], dict]:
    """Every row a filter returns from the rich run, walking its pages, and the first page's reply."""
    return dashkit.fetch_all(get, "rich", params, per)


FILTERS = dashkit.FILTERS


@pytest.mark.parametrize("params", FILTERS, ids=lambda p: "&".join(f"{k}={v}" for k, v in p.items()) or "none")
def test_every_filter_alone_and_combined_matches_python(explorer, rich, params):
    rows, reply = _fetch_all(explorer, params)
    ids, counts, summary = _expected(rich, params)
    assert [r["id"] for r in rows] == ids  # newest first, ties broken by the newer id
    assert reply["total"] == len(ids) and reply["summary"] == summary
    assert reply["action_counts"] == counts  # each action under the other filters, so the chips show what is there
    stamps = [(r["ts"], r["id"]) for r in rows]
    assert stamps == sorted(stamps, reverse=True)


def test_the_filters_are_not_vacuous(rich):
    """Each family of filters splits the seeded run, so the comparisons above test something."""
    total = len(_expected(rich, {})[0])
    for params in FILTERS[1:]:
        if len(params) == 1 and set(params) <= {"bot", "kind", "action", "forced", "gate", "outcome", "hours"}:
            assert 0 < len(_expected(rich, params)[0]) < total, params
    assert len(_expected(rich, {"q": "%"})[0]) == 1 and len(_expected(rich, {"q": "\\"})[0]) == 1
    assert 1 < len(_expected(rich, {"q": "_"})[0]) < total  # an escaped underscore matches only real underscores


def test_rows_carry_the_table_columns(explorer):
    code, reply = explorer("/api/decisions?run=rich&action=flip,hold,none&bot=1")
    assert code == 200 and reply["run"] == "rich" and reply["horizon_minutes"] == 2
    flip = next(r for r in reply["rows"] if r["action"] == "flip")
    assert {k: flip[k] for k in ("ts", "bot_id", "kind", "position_before", "prediction_bps", "required_edge_bps",
                                 "gate_passed", "forced", "reason", "label_bps", "label_state", "right")} == {
        "ts": FLIP_TS, "bot_id": 1, "kind": "learner", "position_before": "short", "prediction_bps": 25.0,
        "required_edge_bps": 20.0, "gate_passed": 1, "forced": 0, "label_bps": 30.0, "label_state": "matured",
        "right": True, "reason": "prediction 25.00 bps clears required edge 20.00 bps"}
    hold = next(r for r in reply["rows"] if r["action"] == "hold")
    assert hold["right"] is False and reply["filters"] == {"bot": 1, "actions": ["flip", "hold", "none"]}
    _, twins = explorer("/api/decisions?run=rich&kind=twin&per=1")
    assert twins["rows"][0]["kind"] == "twin"
    states = {r["label_state"] for r in _fetch_all(explorer, {})[0]}
    assert states == {"matured", "waiting", "missed"}


def test_paging_totals_and_the_last_page(explorer, rich):
    ids = _expected(rich, {})[0]
    code, first = explorer("/api/decisions?run=rich")
    assert code == 200 and first["per"] == 50 and len(first["rows"]) == 50  # 50 a page unless asked otherwise
    assert first["pages"] == -(-len(ids) // 50) and first["total"] == len(ids)
    _, second = explorer("/api/decisions?run=rich&per=7&page=2")
    assert [r["id"] for r in second["rows"]] == ids[7:14] and second["pages"] == -(-len(ids) // 7)
    _, last = explorer(f"/api/decisions?run=rich&per=7&page={second['pages']}")
    assert [r["id"] for r in last["rows"]] == ids[7 * (second["pages"] - 1):]
    _, beyond = explorer("/api/decisions?run=rich&per=7&page=99999")
    assert beyond["page"] == second["pages"] and beyond["rows"] == last["rows"]  # clamped to the last page


def test_empty_results(explorer):
    for query in ("bot=999999", "q=nothing+says+this", "from=2030-01-01", "action=hold&kind=twin"):
        code, reply = explorer("/api/decisions?run=rich&" + query)
        assert code == 200 and reply["rows"] == [] and reply["total"] == 0, query
        assert reply["pages"] == 1 and reply["page"] == 1
    _, reply = explorer("/api/decisions?run=rich&bot=999999")
    assert set(reply["action_counts"].values()) == {0} and set(reply["summary"].values()) == {0}


def test_detail_of_a_flip_joins_bot_fills_trades_and_label(explorer, rich):
    rows, _ = _all_rows(rich)
    flip = next(r for r in rows if r["ts"] == FLIP_TS)
    code, d = explorer(f"/api/decision?run=rich&id={flip['id']}")
    assert code == 200 and d["run"] == "rich" and d["horizon_minutes"] == 2
    assert set(d["decision"]) == set(DECISION_COLUMNS) and d["decision"]["action"] == "flip"
    assert {k: d["bot"][k] for k in ("bot_id", "generation", "origin", "kind", "slot")} == {
        "bot_id": 1, "generation": 0, "origin": "random", "kind": "learner", "slot": 0}
    assert d["bot"]["state"] in ("active", "dead", "dying", "warmup")
    assert [(f["leg"], f["trade_id"], f["price"], f["fee"], f["slippage_bps"], f["latency_ms"], f["filled_at"])
            for f in d["fills"]] == [("close", 9001, 80010.0, 0.8, 0.6, 1250, "2026-10-03T04:50:31.250Z"),
                                     ("open", 9002, 80010.0, 0.8, 0.7, 1250, "2026-10-03T04:50:31.250Z")]
    closed, opened = d["trades"]
    assert (closed["id"], closed["role"], closed["state"], closed["pnl_net"], closed["hold_minutes"]) == \
        (9001, "closed", "closed", 0.2, 10.0)
    assert closed["fees"] == pytest.approx(1.6)
    assert (opened["id"], opened["role"], opened["state"], opened["pnl_net"]) == (9002, "opened", "open", None)
    assert d["outcome"] == {"state": "matured", "label_bps": 30.0, "matures_at": "2026-10-03T04:52:30.000Z",
                            "right": True, "prediction_right": True, "coin_side": None, "position_side": "long",
                            "position_right": True}
    hold = next(r for r in rows if r["ts"] == HOLD_TS)
    _, h = explorer(f"/api/decision?run=rich&id={hold['id']}")
    assert h["fills"] == [] and h["trades"] == [] and h["outcome"]["prediction_right"] is False
    assert h["outcome"]["position_right"] is None and h["outcome"]["right"] is False


def test_detail_of_replayed_decisions(explorer, rich):
    from btc_survivor.cli import data

    rows, _ = _all_rows(rich)
    conn = data.connect(rich.db_path)
    statuses = []
    for r in rows:
        if r["action"] not in ("open_long", "open_short", "close") or r["ts"] in INSERTED:
            continue
        _, d = explorer(f"/api/decision?run=rich&id={r['id']}")
        want = [dict(f) for f in conn.execute("SELECT * FROM fills WHERE bot_id = ? AND decided_at = ? AND "
                                              "(reason = 'decision' OR status = 'rejected')", (r["bot_id"], r["ts"]))]
        assert [f["id"] for f in d["fills"]] == [f["id"] for f in want] and len(want) == 1
        fill = want[0]
        statuses.append(fill["status"])
        if fill["status"] == "rejected" and fill["leg"] == "open":
            assert d["trades"] == [] and d["fills"][0]["reason"] == fill["reason"]
            continue
        trade = dict(conn.execute("SELECT * FROM trades WHERE id = ?", (fill["trade_id"],)).fetchone())
        assert len(d["trades"]) == 1 and {k: d["trades"][0][k] for k in trade} == trade
        role = d["trades"][0]["role"]
        assert role == ("opened" if r["action"].startswith("open") else "closed" if fill["status"] == "filled"
                        else "close rejected")
        if role == "opened":
            assert d["trades"][0]["side"] == r["action"][5:] and d["outcome"]["position_side"] == trade["side"]
        pair = d["pair"]
        if r["origin"] == "control":
            assert pair is not None and f"twin of bot {pair['bot_id']}" in r["reason"]
        if pair is not None:
            _, back = explorer(f"/api/decision?run=rich&id={pair['id']}")
            assert back["pair"]["id"] == r["id"] and back["decision"]["ts"] == r["ts"]
    conn.close()
    assert statuses.count("rejected") >= 1 and statuses.count("filled") > 20  # both branches ran
    none = next(r for r in rows if r["action"] == "none" and r["origin"] != "control")
    _, d = explorer(f"/api/decision?run=rich&id={none['id']}")
    assert d["fills"] == [] and d["trades"] == [] and d["outcome"]["position_side"] is None


BAD = [
    "", "run=", "run=nope", "run=rich&bot=abc", "run=rich&bot=0", "run=rich&bot=-1", "run=rich&bot=1.5",
    "run=rich&bot=99999999999999999999", "run=rich&kind=robot", "run=rich&action=buy", "run=rich&action=none,buy",
    "run=rich&forced=2", "run=rich&gate=yes", "run=rich&outcome=maybe", "run=rich&q=" + "x" * 101,
    "run=rich&q=a%0Ab", "run=rich&from=yesterday", "run=rich&from=2026-13-01", "run=rich&to=2026-10-03+25:00",
    "run=rich&from=0001-01-01", "run=rich&from=2026-10-04&to=2026-10-03", "run=rich&hours=0", "run=rich&hours=abc",
    "run=rich&hours=99999", "run=rich&page=0", "run=rich&page=x", "run=rich&per=0", "run=rich&per=1000",
    "run=rich&foo=1", "run=rich&bot=1&bot=2", "run=rich&q=%FF", "run=rich&&bot=1", "run=rich&id=3",
    "run=rich&bot=%D9%A1",
]


@pytest.mark.parametrize("query", BAD)
def test_bad_parameters_are_refused_with_400(explorer, query):
    code, reply = explorer("/api/decisions?" + query)
    assert code == (404 if query == "run=nope" else 400), reply
    assert set(reply) == {"error"} and len(reply["error"]) < 200


@pytest.mark.parametrize("query", ["", "run=rich", "run=rich&id=abc", "run=rich&id=0", "run=rich&id=1&bot=1",
                                   "run=rich&id=1&id=2"])
def test_bad_detail_parameters_are_refused_with_400(explorer, query):
    code, reply = explorer("/api/decision?" + query)
    assert code == 400 and set(reply) == {"error"}


def test_missing_things_are_404(explorer):
    assert explorer("/api/decision?run=rich&id=999999")[0] == 404
    assert explorer("/api/decision?run=nope&id=1")[0] == 404
    code, reply = explorer("/api/decisions?run=nothing")
    assert code == 404 and "no database" in reply["error"]


def test_failures_are_one_sentence_and_slow_queries_stop(explorer, monkeypatch):
    import btc_survivor.dashboard as dashboard

    monkeypatch.setattr(dashboard, "decision_page", lambda *a: (_ for _ in ()).throw(KeyError("broken")))
    code, reply = explorer("/api/decisions?run=rich&q=failure")
    assert code == 500 and "cannot be read" in reply["error"]
    monkeypatch.undo()
    monkeypatch.setattr(dashboard, "QUERY_BUDGET_S", -1.0)
    monkeypatch.setattr(dashboard, "PROGRESS_STEPS", 1)
    code, reply = explorer("/api/decisions?run=rich&q=slow")
    assert code == 503 and "narrow the filters" in reply["error"]


def test_sql_stays_parameterised(rich, monkeypatch):
    """Every value travels as a parameter. No SQL text the explorer runs holds a value from the request."""
    import re

    import btc_survivor.dashboard as dashboard

    seen: list[tuple[str, tuple]] = []
    real = dashboard.data.connect

    class Recorder:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, params=()):
            seen.append((sql, tuple(params)))
            return self.conn.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self.conn, name)

    monkeypatch.setattr(dashboard.data, "connect", lambda path: Recorder(real(path)))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler([rich]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    hostile = "x' OR '1'='1; DROP TABLE decisions; --"
    try:
        query = urllib.parse.urlencode({"run": "rich", "bot": "31", "kind": "twin", "action": "close,flip",
                                        "forced": "0", "gate": "0", "outcome": "missed", "q": hostile,
                                        "from": "2026-10-03 04:41", "to": "2026-10-03 05:43", "hours": "37",
                                        "page": "3", "per": "39"})
        assert json.loads(urllib.request.urlopen(f"{base}/api/decisions?{query}").read())["total"] == 0
        urllib.request.urlopen(f"{base}/api/decisions?run=rich&kind=learner&outcome=waiting").read()
        rows, _ = _all_rows(rich)
        twin = next(r for r in rows if r["origin"] == "control" and r["action"] == "close")
        urllib.request.urlopen(f"{base}/api/decision?run=rich&id={twin['id']}").read()
    finally:
        server.shutdown()
        server.server_close()
    assert len(seen) >= 8
    for sql, _ in seen:
        text = re.sub(r" LIMIT 1$", "", sql.replace("ESCAPE '\\'", ""))  # the only fixed literals
        assert "'" not in text and "--" not in text and "DROP" not in text, sql
        assert not re.search(r"[1-9]", text), sql  # no bot id, page, size, hours or time was written into the text
    values = [v for _, params in seen for v in params]
    assert 31 in values and 39 in values and "%x' OR '1'='1; DROP TABLE decisions; --%" in values
    assert "%the coin chose short" in values and "control" in values  # how a twin's call is read, also a parameter
    assert "2026-10-03T04:41:00.000Z" in values and "2026-10-03T05:43:00.000Z" in values and twin["id"] in values


def test_actions_cover_everything_the_code_writes(rich):
    from btc_survivor.bot import decide
    from btc_survivor.dashboard import ACTIONS, ORDER_ACTIONS
    from btc_survivor.types import FLAT, LONG, SHORT

    written = {decide(pos, pred, req, short, forced, skill)[0] for pos in (LONG, FLAT, SHORT) for pred in (-5, 0, 5)
               for req in (1, 10) for short in (True, False) for forced in (True, False) for skill in (True, False)}
    assert written == set(ACTIONS) and set(ORDER_ACTIONS) < set(ACTIONS)
    rows, _ = _all_rows(rich)
    assert {r["action"] for r in rows} <= set(ACTIONS)


def test_a_long_run_stays_fast(tmp_path):
    """About four months of run A's decisions, 120,000 rows, answer every kind of request well inside the budget."""
    import random
    import sqlite3

    from btc_survivor.cli import data
    from btc_survivor.clock import iso
    from btc_survivor.dashboard import QUERY_BUDGET_S, decision_detail, decision_page, parse_filters
    from btc_survivor.store import Store
    from enginekit import START

    folder = tmp_path / "long"
    store = Store.open(folder / "btc_survivor.db")
    store.close()
    conn = sqlite3.connect(folder / "btc_survivor.db")
    conn.executemany("INSERT INTO bots (bot_id, slot, generation, born_at, state, margin_bps, lr, seed, origin) "
                     "VALUES (?, ?, 0, ?, 'active', 8, 0.001, 1, ?)",
                     [(b, (b - 1) % 20, iso(START), "control" if b > 20 else "random") for b in range(1, 41)])
    rng = random.Random(7)
    actions = ["none"] * 30 + ["open_long", "open_short", "close", "flip", "hold"]
    conn.executemany("INSERT INTO decisions (bot_id, ts, position_before, ref_mid, prediction_bps, required_edge_bps, "
                     "cost_estimate_bps, margin_bps, gate_passed, forced, action, reason, label_bps) "
                     "VALUES (?, ?, 'flat', 80000, ?, 20, 12, 8, ?, ?, ?, 'prediction is below required edge', ?)",
                     [(1 + i % 40, iso(START + i * 90_000), rng.gauss(0, 3), int(rng.random() < 0.05),
                       int(rng.random() < 0.05), rng.choice(actions), rng.gauss(0, 20) if i < 119_000 else None)
                      for i in range(120_000)])
    conn.executemany("INSERT INTO benchmark (ts, btc_mid, buy_hold_equity, matched_equity, living_mean_equity, "
                     "living_median_equity, living, cash_equity) VALUES (?, 80000, 1, 1, 1, 1, 20, 1)",
                     [(iso(START + h * 3_600_000),) for h in range(3001)])
    conn.commit()
    conn.close()
    cfg = deadly_config(folder)
    ro = data.connect(cfg.db_path)
    worst = 0.0
    for params in ({}, {"page": "2400"}, {"q": "below", "kind": "twin"}, {"outcome": "right", "action": "flip,hold"},
                   {"bot": "17", "hours": "24"}, {"from": iso(START + 3_000_000_000)[:16], "per": "200"}):
        started = time.monotonic()
        decision_page(ro, cfg, parse_filters(params))
        worst = max(worst, time.monotonic() - started)
    started = time.monotonic()
    decision_detail(ro, cfg, 119_999)
    worst = max(worst, time.monotonic() - started)
    ro.close()
    assert worst < QUERY_BUDGET_S / 2, worst


def test_the_page_offers_the_explorer():
    from btc_survivor.dashboard import PAGE

    page = PAGE.read_text()
    assert "/api/decisions?" in page and "/api/decision?" in page and 'id="decisions"' in page
    assert "<b>Decisions explorer.</b>" in page
    assert "a twin's is the side its coin chose" in page and "never shows forced" in page  # how twins are scored
    assert 'id="dx-hint"' in page and "DX_CHECK" in page  # a half typed time is held back, not sent


def test_the_page_says_what_forced_trades_are_for_and_whom_the_average_covers():
    """Learning does not depend on trading, the average learner covers the living only, and the self financing value
    and the skill bounds sit beside them, written so a page served by an older server still renders."""
    import re

    from btc_survivor.dashboard import PAGE
    from docstyle import prose_problems

    page = PAGE.read_text()
    assert "gathering experience" not in page
    assert "Learners train on every matured label whether or not they trade" in page
    assert "Forced trades exist so costs, twins, deaths and parent eligibility can be measured" in page
    assert "Average living learner" in page and "living learners only, refilled at deaths" in page
    assert ">Self financing<" in page and "sf.value_per_slot" in page and "sf.outside_money" in page
    assert '"restart the dashboard server to show it"' in page  # an older server sends no self_financing
    assert "sk.t !== undefined" in page and "sk.upper_95" in page and "sk.detectable_nw" in page
    assert '"mean learner"' not in page and "Mean learner" not in page
    help_text = page[page.index('<div class="help">'):page.index("</details>")]
    paragraphs = [re.sub(r"<[^>]+>", "", p) for p in re.findall(r"<p>(.*?)</p>", help_text, re.S)]
    assert len(paragraphs) >= 14 and prose_problems("\n".join(paragraphs)) == []


def test_run_state_carries_the_self_financing_value(cfg):
    from btc_survivor.cli import data
    from btc_survivor.cli.self_financing import value_at

    state = run_state(cfg, time.time_ns() // 1_000_000)
    conn = data.connect(cfg.db_path)
    try:
        assert state["self_financing"] == value_at(conn, cfg) and state["self_financing"]["births"] > 0
    finally:
        conn.close()


def test_the_page_names_its_own_icon():
    """An empty icon inside the page, which the policy's img-src data: allows, so a browser never asks for
    /favicon.ico and logs the 404 as a console error."""
    from btc_survivor.dashboard import PAGE, SECURITY_HEADERS

    assert '<link rel="icon" href="data:,">' in PAGE.read_text()
    assert "img-src 'self' data:" in SECURITY_HEADERS["Content-Security-Policy"]


def test_the_page_offers_a_time_zone_switch():
    """UTC or the viewer's own zone for every time on the page, remembered in the browser, while the server keeps
    UTC."""
    import re

    from btc_survivor.dashboard import PAGE

    page = PAGE.read_text()
    assert 'id="zone"' in page and 'data-zone="utc"' in page and 'data-zone="local"' in page
    assert 'localStorage.getItem("tz")' in page and 'localStorage.setItem("tz", zone)' in page
    assert "all times are UTC" not in page and "<b>Times.</b>" in page and "abbreviation" in page
    assert "UTC</th>" not in page and ")} UTC" not in page and "toISOString().slice(11" not in page  # no fixed labels
    assert "new Date(dxInstant(k, v)).toISOString()" in page  # a local from or to reaches the server as UTC
    assert "as of the checkpoint at" in page  # the Bots header says when its state is from
    assert "textTimes(r.detail)" in page  # a time quoted inside a feed event follows the switch too
    assert "at most a minute old" not in page and "written once a minute" in page  # runtime_status is once a minute
    script = page[page.index("<script>"):]
    blocks, depth, start = [], 0, None
    for match in re.finditer(r"try \{|\{|\}", script):  # the spans of every try block, by matching braces
        if match.group() == "try {" and start is None:
            start, depth = match.start(), 1
        elif start is not None:
            depth += 1 if match.group() != "}" else -1
            if depth == 0:
                blocks.append((start, match.end()))
                start = None
    uses = [m.start() for m in re.finditer(r"localStorage", script)]
    assert len(uses) >= 10
    assert all(any(a < u < b for a, b in blocks) for u in uses)  # the page still works where storage throws


NODE = shutil.which("node")
TIME_CHECK = r"""
zone = "local";
const year = Date.UTC(2026, 0, 1), out = {wrong: [], z: 0};
for (let t = year; t < year + 366 * 86400000; t += 900000) {
  for (const at of [t, t - 1, t + 15500]) for (const k of DX_TIMES) {
    const text = dxShow(k, new Date(at).toISOString());
    if (dxInstant(k, text) !== at || dxToServer(k, text) !== new Date(at).toISOString()) out.wrong.push([k, at, text]);
    if (text.endsWith("Z")) out.z++;
  }
}
out.pasted = [dxToServer("from", "2026-10-12T07:30:15.5Z"), dxToServer("to", "2026-10-12Z")];
out.typed = [dxToServer("from", "2026-10-12 07:30"), dxToServer("to", "2026-10-12")];
out.repeated = [dxShow("from", "2026-11-01T06:30:00.000Z"), dxShow("from", "2026-11-01T07:30:00.000Z")];
out.detail = textTimes("restored the checkpoint from 2026-10-12T17:20:30.000Z <b>");
zone = "utc";
out.detailUtc = textTimes("restored the checkpoint from 2026-10-12T17:20:30.000Z <b>");
out.asTyped = dxToServer("from", "2026-10-12 07:30");
console.log(JSON.stringify(out));
"""


def _page_times(page: str) -> str:
    """The page's own time code, the part that touches no element, for node to run."""
    script = page[page.index("<script>") + len("<script>"):page.index("</script>")]
    esc = next(line for line in script.splitlines() if line.startswith("const esc = "))
    times = script[script.index("// Times."):script.index("async function load()")]
    explorer = script[script.index("const DX_WHEN"):script.index("try {\n  const saved")]
    start = script.index("const ISO_IN_TEXT")
    detail = script[start:script.index("\n", script.index("const textTimes", start))]
    return "\n".join([esc, times, explorer, detail])


@pytest.mark.skipif(NODE is None, reason="node runs the page's own time code")
@pytest.mark.parametrize("tz", ["America/Chicago", "Europe/London", "Australia/Lord_Howe", "America/Santiago",
                                "Asia/Kolkata", "Pacific/Chatham", "UTC"])
def test_local_from_and_to_read_back_as_the_same_moment(tz):
    """Every from and to the page writes in Local reads back as the moment it came from, all year, the hour that
    repeats when the clocks go back included, so leaving a field or switching zone never moves a filter. A time ending
    in Z is UTC in either zone, so one pasted from a log means what it says."""
    from btc_survivor.dashboard import PAGE

    env = {**os.environ, "TZ": tz, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}
    run = subprocess.run([NODE, "-e", _page_times(PAGE.read_text()) + TIME_CHECK], env=env, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["wrong"] == []
    assert out["pasted"] == ["2026-10-12T07:30:15.500Z", "2026-10-12T23:59:59.999Z"]
    assert out["asTyped"] == "2026-10-12 07:30"  # in UTC a value goes to the server as typed
    assert out["detailUtc"] == "restored the checkpoint from 2026-10-12 17:20:30.000 UTC &lt;b&gt;"
    if tz in ("UTC", "Asia/Kolkata"):
        assert out["z"] == 0  # no hour repeats, so every time is written in the zone itself
    if tz == "America/Chicago":
        assert out["typed"] == ["2026-10-12T12:30:00.000Z", "2026-10-13T04:59:59.999Z"]
        assert out["repeated"] == ["2026-11-01 01:30", "2026-11-01T07:30Z"]  # the second 01:30 is 01:30 CST
        assert out["z"] == 24  # the repeated hour's 4 quarter hours, at 3 offsets, for from and to
        assert out["detail"] == "restored the checkpoint from 2026-10-12 12:20:30.000 CDT &lt;b&gt;"


def test_twins_are_scored_by_the_side_the_coin_chose(explorer):
    """A twin's row copies its learner's prediction, so its mark judges the coin's side. A close or wait has no side."""
    seeded = {r["ts"]: r for r in _fetch_all(explorer, {"kind": "twin"})[0] if r["ts"] in (FLIP_TS, TWIN_OPEN_TS)}
    flip, opened = seeded[FLIP_TS], seeded[TWIN_OPEN_TS]
    assert flip["prediction_bps"] * flip["label_bps"] > 0 and (flip["coin_side"], flip["right"]) == ("short", False)
    assert opened["prediction_bps"] * opened["label_bps"] < 0
    assert (opened["coin_side"], opened["right"]) == ("short", True)
    matured = _fetch_all(explorer, {"kind": "twin", "outcome": "matured"})[0]
    assert any(r["action"] == "close" for r in matured)  # the replay has twin closes to check
    assert all(r["right"] is None and r["coin_side"] is None for r in matured if r["action"] in ("close", "none"))
    assert all(r["coin_side"] is None for r in _fetch_all(explorer, {"kind": "learner"})[0])
    _, right = explorer("/api/decisions?run=rich&kind=twin&outcome=right&per=200")
    assert opened["id"] in [r["id"] for r in right["rows"]] and flip["id"] not in [r["id"] for r in right["rows"]]
    _, d = explorer(f"/api/decision?run=rich&id={opened['id']}")
    assert {k: d["outcome"][k] for k in ("right", "prediction_right", "coin_side")} == {
        "right": True, "prediction_right": False, "coin_side": "short"}
    _, d = explorer(f"/api/decision?run=rich&id={flip['id']}")
    assert (d["outcome"]["right"], d["outcome"]["prediction_right"], d["bot"]["kind"]) == (False, True, "twin")
    assert d["pair"]["bot_id"] == 1 and d["pair"]["action"] == "flip"  # beside its learner's flip in that minute


def test_coin_side_reads_what_a_twin_writes():
    """The coin's side comes from the action of an open and the reason of a flip, exactly as TwinBot.mirror words it."""
    import numpy as np

    from btc_survivor.broker import Broker
    from btc_survivor.config import load_config
    from btc_survivor.controls import TwinBot
    from btc_survivor.dashboard import coin_side
    from btc_survivor.feed.book import LocalBook
    from btc_survivor.types import LONG, SHORT
    from helpers import EXAMPLE_CONFIG

    cfg = load_config(EXAMPLE_CONFIG)
    book = LocalBook(20)
    book.apply_snapshot(bids=[(99.99 - i * 0.01, 50.0) for i in range(20)],
                        asks=[(100.01 + i * 0.01, 50.0) for i in range(20)])
    learner = {"bot_id": 1, "ts": 60_000, "position_before": "flat", "ref_mid": 100.0, "prediction_bps": 3.0,
               "required_edge_bps": 20.0, "cost_estimate_bps": 12.0, "margin_bps": 8.0, "gate_passed": 0,
               "forced": 0, "reason": "learner"}
    seen = set()
    for seed in range(8):
        for held in (None, LONG, SHORT):
            for action in ("open_long", "open_short", "flip", "close"):
                twin = TwinBot(cfg, 21, 0, 0, np.random.default_rng(seed))
                if held is not None:
                    twin.account.open(held, 1.0, 100.0, 0.0, 0, trade_id=1)
                row = twin.mirror({**learner, "action": action}, 1, book, Broker(cfg.costs, cfg.trading.notional))
                legs = row["order"].legs if "order" in row else []
                opens = [leg.position_side for leg in legs if leg.action == "open"]
                want = ("long" if opens[0] == LONG else "short") if opens else None
                assert coin_side(row["action"], row["reason"]) == want, row
                seen.add((row["action"], want))
    assert {("flip", "long"), ("flip", "short"), ("open_long", "long"), ("open_short", "short"),
            ("close", None), ("none", None)} <= seen


def test_each_answer_reads_one_snapshot(rich, tmp_path):
    """A checkpoint committed while an answer is being read never shows in half of it."""
    import shutil
    import sqlite3

    from btc_survivor.dashboard import read_run

    folder = tmp_path / "live"
    folder.mkdir()
    shutil.copy(rich.db_path, folder / "btc_survivor.db")
    writer = sqlite3.connect(folder / "btc_survivor.db")
    writer.execute("PRAGMA journal_mode=WAL")
    cfg = deadly_config(folder)
    count = "SELECT count(*) FROM decisions"

    def work(conn):
        before = conn.execute(count).fetchone()[0]
        writer.execute("INSERT INTO decisions (bot_id, ts, position_before, ref_mid, prediction_bps, "
                       "required_edge_bps, cost_estimate_bps, margin_bps, gate_passed, forced, action, reason) "
                       "VALUES (1, '2026-10-03T05:30:00.000Z', 'flat', 1, 0, 1, 1, 0, 0, 0, 'none', 'late')")
        writer.commit()
        return before, conn.execute(count).fetchone()[0]

    try:
        before, after = read_run(cfg, "live", work)
        assert after == before  # the commit landed between the two reads, and this answer never saw it
        assert read_run(cfg, "live", lambda conn: conn.execute(count).fetchone()[0]) == before + 1
    finally:
        writer.close()


def _raw(address, path: str, headers: list[str], method: str = "GET") -> tuple[int, bytes, bytes]:
    """Send one request with exactly these header lines and return the status, the head and the body."""
    import socket

    reply = b""
    lines = "".join(f"{h}\r\n" for h in headers)
    with socket.create_connection(address, timeout=5) as sock:
        sock.sendall(f"{method} {path} HTTP/1.1\r\n{lines}\r\n".encode())
        while chunk := sock.recv(65536):
            reply += chunk
    head, _, body = reply.partition(b"\r\n\r\n")
    return int(head.split(b"\r\n")[0].split(b" ")[1]), head, body


def test_a_request_line_that_cannot_be_split_gets_400(rich, capsys):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler([rich]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code, head, body = _raw(server.server_address, "http://[x/api/decisions?run=rich",
                                [f"Host: 127.0.0.1:{server.server_address[1]}"])
    finally:
        server.shutdown()
        server.server_close()
    assert code == 400 and b"frame-ancestors 'none'" in head
    assert body == b"the request path cannot be read"
    assert "Traceback" not in capsys.readouterr().err


def test_requests_addressed_to_another_host_get_403(rich):
    """DNS rebinding. A page on another site can point its own name at 127.0.0.1, and its requests then arrive with
    that name in the Host header, so anything but 127.0.0.1 or localhost on the serving port is refused everywhere."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler([rich]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address, port = server.server_address, server.server_address[1]
    paths = ("/", "/index.html", "/api/state", "/api/decisions?run=rich", "/api/decision?run=rich&id=1", "/nope",
             "http://[x/api/state")
    try:
        for host in (f"127.0.0.1:{port}", f"localhost:{port}", f"LOCALHOST:{port}", f" localhost:{port} "):
            assert _raw(address, "/api/decisions?run=rich", [f"Host: {host}"])[0] == 200, host
            assert _raw(address, "/", [f"Host: {host}"])[0] == 200, host
        refused = [[f"Host: attacker.example:{port}"], ["Host: attacker.example"], ["Host: 127.0.0.1"],
                   ["Host: localhost"], [f"Host: 127.0.0.1:{port + 1}"], [f"Host: localhost:{port}0"],
                   [f"Host: 127.0.0.1:{port}.attacker.example"], [f"Host: localhost.:{port}"],
                   [f"Host: 127.0.0.2:{port}"], [f"Host: [::1]:{port}"], [f"Host: user@127.0.0.1:{port}"],
                   ["Host: "], [], [f"Host: 127.0.0.1:{port}", "Host: attacker.example"],
                   [f"Host: 127.0.0.1:{port}", f"Host: 127.0.0.1:{port}"]]
        for headers in refused:
            for path in paths:
                code, head, body = _raw(address, path, headers)
                assert code == 403, (headers, path)
                assert b"frame-ancestors 'none'" in head and b"nosniff" in head and b"no-store" in head
                assert body.startswith(b"this dashboard only answers requests addressed to 127.0.0.1 or localhost")
        with pytest.raises(urllib.error.HTTPError) as err:  # what a rebound page's fetch would send
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/state",
                                                          headers={"Host": f"attacker.example:{port}"}))
        assert err.value.code == 403 and b"runs" not in err.value.read()
    finally:
        server.shutdown()
        server.server_close()


def test_every_method_meets_the_host_rule_and_every_answer_carries_the_headers(rich):
    """Only GET is served, but any other method is refused for a foreign Host too, and the standard library's own
    answers, 501 for a method and 431 for too many headers, carry the same security headers as the page."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler([rich]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address, port = server.server_address, server.server_address[1]
    secure = (b"frame-ancestors 'none'", b"nosniff", b"no-store", b"no-referrer")
    try:
        for method in ("HEAD", "POST", "OPTIONS", "PUT", "DELETE", "PATCH", "BREW"):
            code, head, body = _raw(address, "/api/state", [f"Host: attacker.example:{port}"], method)
            assert code == 403 and all(h in head for h in secure), method
            assert (body == b"") if method == "HEAD" else body.startswith(b"this dashboard only answers"), method
            code, head, body = _raw(address, "/api/state", [f"Host: 127.0.0.1:{port}"], method)
            assert code == 501 and all(h in head for h in secure) and b"runs" not in body, method
        padded = [f"Host: 127.0.0.1:{port}"] + [f"X-Pad-{i}: 1" for i in range(110)]
        code, head, body = _raw(address, "/api/state", padded)
        assert code == 431 and all(h in head for h in secure) and b"runs" not in body
        code, head, body = _raw(address, "/api/state", [f"Host: 127.0.0.1:{port}"])
        assert code == 200 and all(h in head for h in secure) and head.count(b"Cache-Control") == 1
    finally:
        server.shutdown()
        server.server_close()


def test_host_rule():
    from btc_survivor.dashboard import host_allowed

    assert host_allowed(["127.0.0.1:8050"], 8050) and host_allowed(["Localhost:8050"], 8050)
    assert not host_allowed(["127.0.0.1"], 8050) and not host_allowed(["localhost:8051"], 8050)
    assert host_allowed(["127.0.0.1"], 80) and host_allowed(["localhost:80"], 80)  # browsers leave out port 80
    assert not host_allowed([], 8050) and not host_allowed(["127.0.0.1:8050"] * 2, 8050)


def test_numbers_out_of_range_name_the_range(explorer):
    code, reply = explorer("/api/decision?run=rich&id=9223372036854775807")
    assert code == 400 and reply["error"] == "id must be a whole number from 1 to 4,611,686,018,427,387,904"
    code, reply = explorer("/api/decisions?run=rich&per=201")
    assert code == 400 and reply["error"] == "per must be a whole number from 1 to 200"
