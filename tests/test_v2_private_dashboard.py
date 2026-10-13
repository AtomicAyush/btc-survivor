"""The private stock dashboard of PLAN_V2 section 13.4 on seeded databases in a test folder, served on 127.0.0.1 at a
free port, never 8050 or 8060. It binds to 127.0.0.1 only and refuses port 8050, a foreign Host header gets 403, every
method that could write fails, every response carries the version 1 security headers, no JSON body holds a secret, and
the daily spend alert shows when a seeded heartbeat carries it. The decisions explorer is served through the same
guards. The page shows every time in UTC or the viewer's own zone, as version 1 does, and the explorer's from and to,
typed in either zone, reach the server as the UTC minute they mean."""

import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import threading
from http.server import ThreadingHTTPServer

import pytest

from btc_survivor.clock import parse_iso
from btc_survivor.dashboard import SECURITY_HEADERS
from btc_survivor.v2 import dashboard, decisions, evaluate, local_secrets, redact
from btc_survivor.v2.config import load_run_pair
from btc_survivor.v2.local_secrets import ANTHROPIC, CONTACT, FINNHUB
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.stocks import store
from test_v2_contact_absent import fingerprint, holds
from test_v2_store import contents, news_database, run_database, v1_database
import v2_runkit as kit

NOW = parse_iso(kit.NOW)
WRITES = ("POST", "PUT", "PATCH", "DELETE")
PATHS = ("/", "/api/state", "/api/decisions", "/api/decision?run=s1_prices&id=1")


class Server:
    """The dashboard handler on 127.0.0.1 at a free port in a thread, with its clock fixed at the seeded now."""

    def __init__(self, runs, now_ms=NOW):
        self.s1, self.s2 = load_run_pair(runs.s1, runs.s2)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(self.s1, self.s2, lambda: now_ms))
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def request(self, method, path, host="own"):
        """One request with the Host header given, our own by default and none at all for None."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            if host is None:
                conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
                conn.endheaders()
            else:
                conn.request(method, path, headers={"Host": f"127.0.0.1:{self.port}" if host == "own" else host})
            reply = conn.getresponse()
            return reply.status, {name.lower(): value for name, value in reply.getheaders()}, reply.read()
        finally:
            conn.close()

    def json(self, path):
        code, _, payload = self.request("GET", path)
        return code, json.loads(payload)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return kit.seeded(tmp_path_factory.mktemp("dashboard"))


@pytest.fixture(scope="module")
def server(runs):
    served = Server(runs)
    yield served
    served.close()


class Stub:
    """Stands in for ThreadingHTTPServer, recording the address and stopping at once as Ctrl+C would."""

    seen: list = []

    def __init__(self, address, handler):
        Stub.seen.append(address)
        self.server_address = address

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        pass


@pytest.fixture
def stubbed(monkeypatch):
    """The server class, logging setup and process hooks replaced, with the redactor's values isolated."""
    Stub.seen = []
    monkeypatch.setattr(dashboard, "ThreadingHTTPServer", Stub)
    monkeypatch.setattr(dashboard, "setup_logging", lambda level: None)
    monkeypatch.setattr(redact, "install", lambda: None)
    monkeypatch.setattr(redact, "_values", set())
    return Stub


def digests(*paths):
    return [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]


# Binding and the port


def test_the_server_binds_to_127_0_0_1_alone(runs):
    s1, s2 = load_run_pair(runs.s1, runs.s2)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    httpd = dashboard.make_server(s1, s2, port)
    try:
        assert httpd.server_address == ("127.0.0.1", port)
    finally:
        httpd.server_close()
    assert dashboard.HOST == "127.0.0.1" and dashboard.PORT == 8060


def test_the_command_binds_127_0_0_1_on_8060_and_registers_the_secrets(runs, stubbed, capsys):
    assert dashboard.main([str(runs.s1), str(runs.s2)]) == 0
    assert stubbed.seen == [("127.0.0.1", 8060)]
    assert capsys.readouterr().out == "private dashboard at http://127.0.0.1:8060, local only, Ctrl+C stops it\n"
    assert {redact.redact(value) for value in kit.FAKE.values()} == {redact.MASK}


def test_the_host_cannot_be_configured(runs, stubbed, capsys):
    for flag in ("--host", "--bind", "--address"):
        with pytest.raises(SystemExit) as stop:
            dashboard.main([str(runs.s1), str(runs.s2), flag, "0.0.0.0"])
        assert stop.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err
    assert stubbed.seen == []


def test_port_8050_and_privileged_ports_are_refused_before_binding(runs, stubbed, capsys):
    for port, message in (("8050", "port 8050 belongs to the version 1 dashboard, use 8060"),
                          ("80", "port 80 is outside 1024 to 65535"), ("70000", "port 70000 is outside")):
        assert dashboard.main([str(runs.s1), str(runs.s2), "--port", port]) == 2
        out, err = capsys.readouterr()
        assert out == "" and err.startswith("error: ") and message in err
    assert stubbed.seen == []
    with pytest.raises(dashboard.DashboardError):
        dashboard.make_server(*load_run_pair(runs.s1, runs.s2), 8050)
    assert dashboard.check_port(8061) == 8061


def test_a_missing_secrets_file_or_a_bad_run_pair_is_refused(tmp_path, runs, stubbed, capsys):
    s1, s2, secrets = kit.configs(tmp_path)
    secrets.unlink()
    assert dashboard.main([str(s1), str(s2)]) == 2
    assert "secrets file" in capsys.readouterr().err and stubbed.seen == []
    assert dashboard.main([str(runs.s2), str(runs.s1)]) == 2
    assert "the first config must be the prices run" in capsys.readouterr().err


# The Host header


@pytest.mark.parametrize("host", ["evil.example", "evil.example:{port}", "127.0.0.1", "localhost", "127.0.0.1:{other}",
                                  "localhost.:{port}", "127.0.0.1.nip.io:{port}", "[::1]:{port}", "0.0.0.0:{port}",
                                  "127.0.0.1:{port}.evil.example", None])
def test_a_foreign_or_missing_host_header_gets_403_on_every_path_and_method(server, host):
    if host is not None:
        host = host.format(port=server.port, other=server.port + 1)
    for path in PATHS:
        for method in ("GET", *WRITES):
            code, _, payload = server.request(method, path, host)
            assert (code, payload) == (403, b"forbidden"), (method, path)


@pytest.mark.parametrize("host", ["127.0.0.1:{port}", "localhost:{port}", "LOCALHOST:{port}"])
def test_this_machine_on_its_own_port_is_served(server, host):
    code, _, payload = server.request("GET", "/", host.format(port=server.port))
    assert code == 200 and b"<title>BTC Survivor stocks</title>" in payload


# Writes and read only databases


def test_every_method_that_could_write_fails_and_no_database_changes(server, runs):
    files = (runs.s1_db, runs.s2_db, runs.news_db)
    before = digests(*files)
    for path in (*PATHS, "/api/decisions?run=s1_prices", "/nope"):
        for method in WRITES:
            code, headers, payload = server.request(method, path)
            assert (code, payload, headers["allow"]) == (405, b"read only", "GET"), (method, path)
        server.request("GET", path)
    assert digests(*files) == before


def test_every_database_opens_read_only(runs):
    s1, s2 = load_run_pair(runs.s1, runs.s2)
    conn = dashboard.open_run(s1)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM decisions")
    finally:
        conn.close()
    news = news_db.open_reader(s2.news_db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            news.execute("DELETE FROM meta")
    finally:
        news.close()


# Security headers


def test_every_response_carries_the_version_1_security_headers(server):
    replies = [server.request("GET", "/"), server.request("GET", "/api/state"),
               server.request("GET", "/api/decisions?limit=1"), server.request("GET", "/api/decisions?symbol=XYZ"),
               server.request("GET", "/api/decision?run=s1_prices&id=99"), server.request("GET", "/nope"),
               server.request("GET", "/api/state?x=1"), server.request("GET", "/", "evil.example"),
               server.request("POST", "/api/state"), server.request("HEAD", "/")]
    assert [code for code, _, _ in replies] == [200, 200, 200, 400, 404, 404, 404, 403, 405, 501]
    assert "frame-ancestors 'none'" in SECURITY_HEADERS["Content-Security-Policy"]
    for code, headers, _ in replies:
        for name, value in SECURITY_HEADERS.items():
            assert headers[name.lower()] == value, (code, name)
        assert headers["cache-control"] == "no-store", code


# The view


def test_the_page_names_its_panels_and_the_footer(server):
    _, headers, payload = server.request("GET", "/")
    page = payload.decode()
    assert headers["content-type"] == "text/html; charset=utf-8"
    for text in ("/api/state", "/api/decisions", "/api/decision?run=", "Market session", "Finnhub feed",
                 "News process", "Prices", "Forecast skill", "Recent news", "Populations", "Decisions",
                 "next forced trade", "This is a private view of Finnhub data for personal non professional use. "
                 "It is not to be shared, tunnelled or published."):
        assert text in page, text


def test_the_state_holds_every_panel_from_the_seeded_rows(server):
    code, state = server.json("/api/state")
    assert code == 200 and state["now_ms"] == NOW and state["page_version"] == dashboard.page_version()
    assert state["explorer"] == {"runs": ["s1_prices", "s2_news"], "symbols": list(kit.SYMBOLS),
                                 "actions": list(decisions.ACTIONS), "kinds": ["learner", "twin"], "page_rows": 50}
    assert state["market"]["state"] == "closed" and state["market"]["next_open"]["et"] == "2026-10-23 09:30 EDT"
    feed = state["feed"]
    assert feed["runtime"]["feed_state"] == "healthy" and feed["runtime"]["age_s"] == 60.0
    assert [(s["symbol"], s["state"]) for s in feed["symbols"]][5:7] == [("META", "healthy"), ("TSLA", "stale")]
    assert feed["print_delay"]["detail"] == "median delay 1.2 s" and feed["events"][0]["kind"] == "print_delay"
    aapl = state["prices"]["symbols"][0]
    assert state["prices"]["session_day"] == "2026-10-22" and aapl["last"] == kit.CANARY_PRICE
    assert aapl["session_return"] == pytest.approx(kit.CANARY_PRICE / 186.0 - 1) and len(aapl["line"]) == 3
    s1, s2 = state["runs"]
    assert (s1["name"], s1["news"], s2["name"], s2["news"]) == ("s1_prices", False, "s2_news", True)
    assert [row["predictor"] for row in s1["skill"]] == ["population", "linear_prices"]
    assert [row["predictor"] for row in s2["skill"]] == ["population", "linear_prices", "news"]
    # Three window sessions of 12 evaluation minutes over 8 symbols, the last minute pending and session 1 left out.
    assert {(row["rows"], row["minutes"]) for row in s1["skill"] + s2["skill"]} == {(280, 35)}
    assert s1["skill"][0]["corr"] > 0.5 and s1["skill"][0]["mse_vs_zero"] < 1 and s2["skill"][2]["mse_vs_zero"] is None
    assert state["s2_against_s1"]["rows"] == 280
    assert [b["bot_id"] for b in s1["bots"]] == [1, 2, 3, 4] and len(s1["equity_curve"]) == 3
    assert s1["launch_ts"] == kit.LAUNCH and s1["chart_minutes_per_point"] == 1


def test_the_skill_panel_is_evaluate_s_window_rows_and_confirmatory_result_once_the_cutoff_is_reached(tmp_path):
    runs = kit.seeded(tmp_path)
    s1, s2 = load_run_pair(runs.s1, runs.s2)
    state = dashboard.build_state(s1, s2, NOW)
    assert state["window"]["final"] is False and {row["verdict"] for run in state["runs"] for row in run["skill"]} == {
        "descriptive"}
    conn = sqlite3.connect(runs.s2_db)
    with conn:
        values = json.loads(conn.execute("SELECT value FROM meta WHERE key = 'evaluation_json'").fetchone()[0])
        conn.execute("UPDATE meta SET value = ? WHERE key = 'evaluation_json'",
                     (json.dumps({**values, "final_healthy_session_minutes": 20}, sort_keys=True),))
    conn.close()
    state = dashboard.build_state(s1, s2, NOW)
    reader = store.open_reader(runs.s2_db)
    try:
        settings, window = dashboard.evaluation_window(reader)
        expected = evaluate.skill(evaluate.scored_rows(reader, window, 30)["news"], settings)
    finally:
        reader.close()
    # 12 run minutes in session 1, so the 20th is minute 240 of session 2, and only the 7 minutes whose labels
    # matured by then count, every later row past the cutoff left out
    assert state["window"] == {"start": window.start, "end": "2026-10-20T17:30:00.000Z", "final": True,
                               "run_minutes": 20}
    news = next(row for row in state["runs"][1]["skill"] if row["predictor"] == "news")
    assert (news["rows"], news["minutes"], news["corr"], news["t"]) == (56, 7, expected.corr, expected.t)
    assert news["verdict"] == ("pass" if expected.passed else "fail")
    assert {row["verdict"] for run in state["runs"] for row in run["skill"] if row is not news} == {"descriptive"}
    assert state["s2_against_s1"]["rows"] == 56


def test_the_news_panel_reads_the_heartbeat_spend_scores_and_what_s2_used(server):
    news = server.json("/api/state")[1]["news"]
    heartbeat = news["heartbeat"]
    assert (heartbeat["healthy"], heartbeat["haiku_state"], heartbeat["sonnet_state"]) == (True, "on", "on")
    assert (heartbeat["spent_day_usd"], heartbeat["spent_month_usd"], heartbeat["spend_alert"]) == (0.04, 7.25, False)
    assert heartbeat["channels_down"] == ["fed_speech"] and news["monthly_cap_usd"] == 20.0
    assert news["alerts"] == []
    assert news["items_today"] == {"finnhub": 1} and news["fallback_share"] == 0.0  # the filtered item is left out
    assert news["median_scoring_s"] == {"haiku": 1.5, "sonnet": None, "finbert": 0.8}
    [item] = news["recent"]
    assert item["tickers"] == ["AAPL"] and item["haiku"]["status"] == "scored" and item["sonnet"] is None
    assert item["haiku"]["impacts"][0]["strength"] == 0.6 and item["finbert"]["p_pos"] == 0.7
    assert item["s2_used"]["scorer"] == "haiku_v1" and item["s2_used"]["fallback"] == 0


def test_today_counts_items_fallbacks_and_latency_from_the_utc_day_alone(runs):
    _, s2 = load_run_pair(runs.s1, runs.s2)
    conn = news_db.open_reader(s2.news_db_path)
    try:
        panel = dashboard.news_panel(conn, None, parse_iso("2026-10-23T00:00:00.000Z"), 10**9)
    finally:
        conn.close()
    assert panel["items_today"] == {} and panel["fallback_share"] is None
    assert panel["median_scoring_s"] == {"haiku": None, "sonnet": None, "finbert": None}
    assert len(panel["recent"]) == 1 and panel["recent"][0]["s2_used"] is None


def test_the_daily_spend_alert_shows_when_a_seeded_heartbeat_carries_it(tmp_path):
    runs = kit.seeded(tmp_path, spend_alert=True)
    served = Server(runs)
    try:
        state = served.json("/api/state")[1]
        page = served.request("GET", "/")[2].decode()
    finally:
        served.close()
    assert state["news"]["heartbeat"]["spend_alert"] is True and state["news"]["heartbeat"]["spent_day_usd"] == 5.5
    assert state["news"]["alerts"] == [dashboard.SPEND_ALERT]
    assert "passed a quarter of the monthly cap" in dashboard.SPEND_ALERT
    assert "(n.alerts || []).map(a => banner(a, true))" in page and "daily spend alert" in page


def test_an_old_heartbeat_raises_its_own_alert(runs):
    _, s2 = load_run_pair(runs.s1, runs.s2)
    conn = news_db.open_reader(s2.news_db_path)
    try:
        beat = parse_iso("2026-10-22T20:09:30.000Z")
        fresh = dashboard.news_panel(conn, None, beat + 120_000, 120)
        old = dashboard.news_panel(conn, None, beat + 121_000, 120)
    finally:
        conn.close()
    assert fresh["alerts"] == [] and old["alerts"] == [dashboard.OLD_HEARTBEAT]


def test_the_state_is_shared_for_five_seconds(runs):
    ticks = iter(range(NOW, NOW + 10**6, 1000))
    served = Server(runs)
    served.httpd.RequestHandlerClass = dashboard.make_handler(served.s1, served.s2, lambda: next(ticks))
    try:
        first, second = served.json("/api/state")[1], served.json("/api/state")[1]
    finally:
        served.close()
    assert first["now_ms"] == second["now_ms"] == NOW


def test_ttl_cache_builds_once_per_window_and_keeps_the_newest_keys():
    now = [0.0]
    cache = dashboard.TtlCache(ttl_s=5.0, size=2, clock=lambda: now[0])
    built = []

    def build(key):
        return lambda: built.append(key) or f"{key} {len(built)}"

    assert [cache.get("a", build("a")), cache.get("a", build("a"))] == ["a 1", "a 1"]
    now[0] = 5.0
    assert cache.get("a", build("a")) == "a 2"
    cache.get("b", build("b"))
    cache.get("c", build("c"))
    assert list(cache.entries) == ["b", "c"]


# Missing and wrong databases


def test_a_missing_or_wrong_run_never_blanks_the_other_and_names_no_path(tmp_path):
    runs = kit.seeded(tmp_path)
    s1, s2 = load_run_pair(runs.s1, runs.s2)
    for suffix in ("", "-wal", "-shm"):
        runs.s1_db.with_name(runs.s1_db.name + suffix).unlink(missing_ok=True)
    state = json.loads(dashboard.body(dashboard.build_state(s1, s2, NOW)))
    assert state["runs"][0] == {"name": "s1_prices",
                                "error": "run s1_prices has no database yet, run init-db and start the stock process"}
    assert state["runs"][1]["bots"] and state["feed"]["runtime"]["feed_state"] == "healthy"
    assert state["s2_against_s1"] is None
    news_database(runs.s1_db)
    runs.news_db.unlink()
    for suffix in ("-wal", "-shm"):
        runs.news_db.with_name(runs.news_db.name + suffix).unlink(missing_ok=True)
    run_database(runs.news_db)
    text = dashboard.body(dashboard.build_state(s1, s2, NOW)).decode()
    state = json.loads(text)
    assert "is not a stock run database of this schema" in state["runs"][0]["error"]
    assert state["news"] == {"error": "news.db is not a news database of this schema, see the dashboard log"}
    assert str(tmp_path) not in text and str(tmp_path.resolve()) not in text


@pytest.mark.parametrize("make", [news_database, v1_database])
def test_a_news_or_version_1_database_at_a_run_path_is_refused_and_left_unchanged(tmp_path, make):
    s1, s2, _ = kit.configs(tmp_path)
    s1, _ = load_run_pair(s1, s2)
    s1.db_path.parent.mkdir(parents=True)
    make(s1.db_path)
    before = contents(s1.db_path)
    assert dashboard.open_run(s1) == ("the database of run s1_prices is not a stock run database of this schema, "
                                      "see the dashboard log")
    assert contents(s1.db_path) == before


def test_an_unreadable_run_database_is_named_without_a_path(tmp_path):
    runs = kit.seeded(tmp_path)
    s1, s2 = load_run_pair(runs.s1, runs.s2)
    os.chmod(runs.s1_db, 0)
    try:
        assert dashboard.open_run(s1) == "the database of run s1_prices cannot be opened, see the dashboard log"
    finally:
        os.chmod(runs.s1_db, 0o600)


def test_a_view_that_cannot_be_built_answers_500_with_the_headers(runs, monkeypatch):
    monkeypatch.setattr(dashboard, "build_state", lambda *a: 1 / 0)
    served = Server(runs)
    try:
        code, headers, payload = served.request("GET", "/api/state")
    finally:
        served.close()
    assert code == 500 and json.loads(payload) == {"error": dashboard.NO_VIEW}
    assert headers["content-security-policy"] == SECURITY_HEADERS["Content-Security-Policy"]


def test_a_panel_that_fails_shows_an_error_and_the_rest_still_render(runs, monkeypatch):
    s1, s2 = load_run_pair(runs.s1, runs.s2)
    monkeypatch.setattr(dashboard, "prices", lambda conn, symbols: 1 / 0)
    state = dashboard.build_state(s1, s2, NOW)
    assert state["prices"] == {"error": "the prices cannot be read, see the dashboard log"}
    assert state["feed"]["runtime"] and state["runs"][0]["bots"]


# Secrets


def test_no_json_body_holds_a_secret_even_when_a_row_does(tmp_path, monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    runs = kit.seeded(tmp_path)
    local_secrets.load_secrets(runs.secrets)
    raw = sqlite3.connect(runs.s1_db)
    raw.execute("INSERT INTO feed_events (ts, kind, symbol, detail) VALUES (?, 'disconnected', NULL, ?)",
                ("2026-10-22T20:00:00.000Z", f"close reason token={kit.FAKE[FINNHUB]}"))
    raw.execute("UPDATE decisions SET reason = ? WHERE id = 2", (f"oops {kit.FAKE[ANTHROPIC]}",))
    raw.commit()
    raw.close()
    writer = news_db.open_writer(runs.news_db)
    news_db.insert(writer, "news_items", {
        "source": "finnhub", "channel": "finnhub_company", "vendor_id": "3", "kind": "media",
        "received_at": "2026-10-22T15:00:00.000Z", "title": f"write to {kit.FAKE[CONTACT]} now",
        "tickers": "AAPL", "relevance": "title", "status": "new", "title_hash": "h3", "route": "llm"})
    writer.close()
    served = Server(runs)
    try:
        bodies = [served.request("GET", path)[2] for path in (
            "/api/state", "/api/decisions", "/api/decisions?reason=oops", "/api/decision?run=s1_prices&id=2")]
    finally:
        served.close()
    contact = fingerprint(kit.FAKE[CONTACT])
    for payload in bodies:
        assert not [name for name, value in kit.FAKE.items() if value.encode() in payload]
        assert not holds(*contact, payload)
        assert redact.MASK.encode() in payload
    state = json.loads(bodies[0])
    assert state["news"]["recent"][0]["title"] == f"write to {redact.MASK} now"
    assert json.loads(bodies[3])["decision"]["reason"] == f"oops {redact.MASK}"


def test_scrub_masks_keys_values_and_nested_lists():
    redact.register("fake-scrub-canary")
    try:
        nested = {"fake-scrub-canary": ("a fake-scrub-canary", [1.5, None, {"k": "fake-scrub-canary"}])}
        assert dashboard.scrub(nested) == {redact.MASK: [f"a {redact.MASK}", [1.5, None, {"k": redact.MASK}]]}
    finally:
        redact._values.discard("fake-scrub-canary")
    assert json.loads(dashboard.body({"a": float("nan"), "b": [float("inf")]})) == {"a": None, "b": [None]}


# The decisions explorer over HTTP


def test_the_explorer_pages_and_details_through_the_same_guards(server):
    code, first = server.json("/api/decisions?limit=2")
    assert code == 200 and [(r["run"], r["id"]) for r in first["rows"]] == [("s2_news", 8), ("s2_news", 7)]
    assert first["errors"] == [] and first["summary"]["decisions"] == 15
    code, second = server.json("/api/decisions?limit=2&before=" + first["next"])
    assert [(r["run"], r["id"]) for r in second["rows"]] == [("s1_prices", 7), ("s2_news", 6)]
    code, detail = server.json("/api/decision?run=s2_news&id=7")
    assert code == 200 and detail["decision"]["action"] == "flip" and detail["decision"]["position_after"] == "long"
    assert len(detail["decision"]["fills"]) == 2 and detail["errors"] == []


@pytest.mark.parametrize("path, code, message", [
    ("/api/decisions?symbol=XYZ", 400, "symbol must be one of"),
    ("/api/decisions?run=s1_prices&run=s2_news", 400, "run may be given only once"),
    ("/api/decisions?" + "&".join(f"slot={i}" for i in range(40)), 400, "the query is too long"),
    ("/api/decisions?before=2026-10-21,0,1", 400, "before must be a cursor"),
    ("/api/decision?run=s2_news&id=99", 404, "run s2_news has no decision 99"),
    ("/api/decision?run=s2_news", 400, "a decision is named by run and id"),
    ("/api/decision?run=s3&id=1", 400, "run must be one of s1_prices, s2_news"),
    ("/api/decision?run=s1_prices&id=x", 400, "id must be a whole number"),
])
def test_a_bad_explorer_query_answers_with_a_short_error(server, path, code, message):
    got, body = server.json(path)
    assert got == code and message in body["error"]


def test_the_explorer_lists_a_run_it_cannot_open_and_serves_the_other(tmp_path):
    runs = kit.seeded(tmp_path)
    for suffix in ("", "-wal", "-shm"):
        runs.s2_db.with_name(runs.s2_db.name + suffix).unlink(missing_ok=True)
    served = Server(runs)
    try:
        code, page = served.json("/api/decisions")
        missing = served.json("/api/decision?run=s2_news&id=1")
    finally:
        served.close()
    assert code == 200 and {r["run"] for r in page["rows"]} == {"s1_prices"} and len(page["rows"]) == 7
    assert page["errors"] == ["run s2_news has no database yet, run init-db and start the stock process"]
    assert missing[0] == 404 and missing[1]["errors"] == page["errors"]


def test_a_failing_explorer_query_answers_500_without_detail(runs, monkeypatch):
    s1, s2 = load_run_pair(runs.s1, runs.s2)
    monkeypatch.setattr(decisions, "page", lambda *a: 1 / 0)
    code, payload = dashboard.decisions_reply(s1, s2, "", dashboard.TtlCache())
    assert code == 500 and json.loads(payload) == {"error": dashboard.NOT_READABLE}


def test_a_stock_store_reader_is_what_the_explorer_opens(runs):
    s1, _ = load_run_pair(runs.s1, runs.s2)
    conn = dashboard.open_run(s1)
    try:
        assert store.read_meta(conn)["kind"] == "stocks"
    finally:
        conn.close()


# Times in UTC or the viewer's own zone


def page_script() -> str:
    page = dashboard.PAGE.read_text(encoding="utf-8")
    return page[page.index("<script>") + len("<script>"):page.index("</script>")]


def test_the_page_offers_a_time_zone_switch_as_version_1_does():
    """UTC or the viewer's own zone for every time on the page, remembered in the browser, while the server keeps
    UTC and the from and to filters of the explorer follow the switch."""
    page = dashboard.PAGE.read_text(encoding="utf-8")
    assert 'id="zone"' in page and 'data-zone="utc"' in page and 'data-zone="local"' in page
    assert 'store.get("v2tz")' in page and 'store.set("v2tz", zone)' in page
    assert "<b>Times.</b>" in page and "abbreviation" in page and "Times are UTC unless marked ET" not in page
    for fixed in ('"time UTC"', '"received UTC"', '"filled UTC"', '"opened UTC"', '"closed UTC"', '+ " UTC"',
                  ')} UTC', 'toISOString().slice(11', 'type="datetime-local"'):
        assert fixed not in page, fixed  # no label or time pinned to UTC
    assert "textTimes(e.detail)" in page  # a time quoted inside a feed event follows the switch too
    assert "exToServer(k, text)" in page and "exShow(k, " in page  # a local from or to reaches the server as UTC
    script = page_script()
    uses = [m.start() for m in re.finditer(r"localStorage", script)]
    wrapper = re.search(r"const store = \{.*?\n\};", script, re.S)
    assert len(uses) == 2 and all(wrapper.start() < u < wrapper.end() for u in uses)
    assert wrapper.group().count("try {") == 2  # the page still works where storage throws


def test_the_detail_shows_the_report_only_fill_columns_once_the_server_sends_them():
    """How long after the due time a fill's print traded and whether S2 saw news on the symbol just before, both report
    only, appear in the fills of a decision whenever its fill rows carry them."""
    page = dashboard.PAGE.read_text(encoding="utf-8")
    assert 'd.fills.some(f => "print_after_due_ms" in f)' in page
    assert '["print after due", 0,' in page and '["news window", 1,' in page and "yesNo(f.news_window)" in page


def test_the_minutes_a_local_from_and_to_send_are_what_the_server_takes(server):
    """A from and to typed in Chicago time reach the server as UTC minutes, and the to minute is included."""
    code, page = server.json("/api/decisions?from=2026-10-21T13:00&to=2026-10-23T04:59&limit=200")
    assert code == 200 and page["rows"]
    assert all("2026-10-21T13:00" <= r["ts"] < "2026-10-23T05:00" for r in page["rows"])
    code, whole = server.json("/api/decisions?limit=200")
    assert page["summary"]["decisions"] == sum("2026-10-21T13:00" <= r["ts"] < "2026-10-23T05:00"
                                               for r in whole["rows"])


NODE = shutil.which("node")
TIME_CHECK = r"""
zone = "local";
const year = Date.UTC(2026, 0, 1), out = {wrong: [], z: 0};
const minute = ms => new Date(ms).toISOString().slice(0, 16);
for (let t = year; t < year + 366 * 86400000; t += 900000) {
  for (const at of [t, t - 60000, t + 420000]) for (const k of EX_TIMES) {
    const text = exShow(k, minute(at));
    if (exInstant(k, text) !== at || exToServer(k, text) !== minute(at)) out.wrong.push([k, at, text]);
    if (text.endsWith("Z")) out.z++;
  }
}
out.pasted = [exToServer("from", "2026-10-12T07:30Z"), exToServer("to", "2026-10-12Z")];
out.typed = [exToServer("from", "2026-10-12 07:30"), exToServer("to", "2026-10-12"), exToServer("from", "2026-10-12")];
out.repeated = [exShow("from", "2026-11-01T06:30"), exShow("from", "2026-11-01T07:30")];
out.refused = ["2026-02-30", "2026-10-12 24:00", "2026-10-12 7:30", "12/10/2026", "2026-10-12 07:30:15",
  "0050-01-01", ""].map(v => exToServer("from", v));
out.detail = textTimes("restored the checkpoint from 2026-10-12T17:20:30.000Z <b>");
out.stamp = stamp("2026-10-12T17:20:30.000Z");
zone = "utc";
out.detailUtc = textTimes("restored the checkpoint from 2026-10-12T17:20:30.000Z <b>");
out.asTyped = [exToServer("from", "2026-10-12 07:30"), exToServer("to", "2026-10-12"), exToServer("to", "2026-10-12T07:30")];
out.shownUtc = [exShow("from", "2026-10-12T00:00"), exShow("to", "2026-10-12T23:59"), exShow("to", "2026-10-12T07:30")];
out.cells = [time("2026-10-12T17:20:30.000Z"), clock(Date.UTC(2026, 9, 12, 17, 20, 30))];
console.log(JSON.stringify(out));
"""


def page_times() -> str:
    """The page's own time code, the part that touches no element, for node to run."""
    script = page_script()
    esc = next(line for line in script.splitlines() if line.startswith("const esc = "))
    stored = script[script.index("const store = {"):script.index("current = store.get")]
    return "\n".join([esc, stored, script[script.index("// Times."):script.index("async function load()")]])


@pytest.mark.skipif(NODE is None, reason="node runs the page's own time code")
@pytest.mark.parametrize("tz", ["America/Chicago", "Europe/London", "Australia/Lord_Howe", "America/Santiago",
                                "Asia/Kolkata", "Pacific/Chatham", "UTC"])
def test_local_from_and_to_read_back_as_the_same_minute(tz):
    """Every from and to the page writes in Local reads back as the minute it came from and reaches the server as that
    UTC minute, all year, the hour that repeats when the clocks go back included, so leaving a field or switching zone
    never moves a filter. A time ending in Z is UTC in either zone, and anything the server would refuse is refused."""
    env = {**os.environ, "TZ": tz, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}
    run = subprocess.run([NODE, "-e", page_times() + TIME_CHECK], env=env, capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["wrong"] == []
    assert out["pasted"] == ["2026-10-12T07:30", "2026-10-12T23:59"]
    assert out["refused"] == [None] * 7
    assert out["asTyped"] == ["2026-10-12T07:30", "2026-10-12T23:59", "2026-10-12T07:30"]
    assert out["shownUtc"] == ["2026-10-12", "2026-10-12", "2026-10-12 07:30"]  # a whole day reads as its date
    assert out["detailUtc"] == "restored the checkpoint from 2026-10-12 17:20:30.000 UTC &lt;b&gt;"
    assert out["cells"][0].startswith("10-12 17:20:30") and out["cells"][1] == "17:20:30 UTC"
    if tz in ("UTC", "Asia/Kolkata"):
        assert out["z"] == 0  # no hour repeats, so every time is written in the zone itself
    if tz == "America/Chicago":
        assert out["typed"] == ["2026-10-12T12:30", "2026-10-13T04:59", "2026-10-12T05:00"]
        assert out["repeated"] == ["2026-11-01 01:30", "2026-11-01T07:30Z"]  # the second 01:30 is 01:30 CST
        assert out["z"] == 24  # the repeated hour's 4 quarter hours, at 3 offsets, for from and to
        assert out["detail"] == "restored the checkpoint from 2026-10-12 12:20:30.000 CDT &lt;b&gt;"
        assert out["stamp"] == "2026-10-12 12:20 CDT"
