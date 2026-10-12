"""The news-status report of PLAN_V2 section 7.7 on seeded news databases with fictional headlines. It prints the
status mix, bytes, backlog channel hours, outage minutes and the order and backlog verdicts as counts and verdicts
only, the verdicts fail on a planted out of order row and a planted backlog item outside any restart, a pause counts
over the whole freeze, and requests per channel per day, other rows and samples appear only under --detail."""

import re
from datetime import datetime, timedelta, timezone

import pytest

from btc_survivor.clock import MS_PER_DAY, MS_PER_MINUTE, MS_PER_SECOND, LiveClock, iso
from btc_survivor.v2 import news_status, output
from btc_survivor.v2.config import parse_news_config
from btc_survivor.v2.news import db
from helpers import ROOT

EDT = timezone(timedelta(hours=-4))
NEWS = ROOT / "configs" / "v2" / "news.toml"
TEXT = NEWS.read_text(encoding="utf-8")
CONFIG = parse_news_config(TEXT, NEWS.parent)
AAPL = "finnhub_company_AAPL"
SUMMARY_LINE = re.compile(r"[a-z][a-z_ ]*: (pass|fail|\d+)")


def at(h, mi, s=0, day=12):
    return int(datetime(2026, 10, day, h, mi, s, tzinfo=EDT).timestamp()) * MS_PER_SECOND


NOW = at(10, 40, 30)


@pytest.fixture
def conn(tmp_path):
    connection = db.open_writer(tmp_path / "news.db")
    yield connection
    connection.close()


def poll(conn, channel, started, status="ok", cadence=60, http_status=200):
    db.insert(conn, "news_polls", {"channel": channel, "started_at": iso(started), "finished_at": iso(started + 200),
                                   "cadence_s": cadence, "status": status, "http_status": http_status, "bytes": 10,
                                   "duration_ms": 200, "budget_wait_ms": 0})


def item(conn, channel, received, status="new", title=None, vendor_id=None, reason=None):
    return db.insert(conn, "news_items", {
        "source": "finnhub", "channel": channel, "vendor_id": vendor_id or f"{channel}-{received}", "kind": "media",
        "received_at": iso(received), "title": title or f"Fictional headline {received}", "tickers": "AAPL",
        "relevance": "title" if status != "filtered" else "none", "status": status, "filter_reason": reason,
        "title_hash": "0123456789abcdef", "route": "llm" if status == "new" else "none"})


def event(conn, kind, ts, detail=None):
    db.insert(conn, "collector_events", {"ts": iso(ts), "kind": kind, "detail": detail})


def every_minute(conn, channel, start, end):
    for started in range(start, end + 1, MS_PER_MINUTE):
        poll(conn, channel, started)


def seed_restart(conn):
    """A 60 s channel polled each minute from 10:00 to 10:10 and 10:30 to 10:40, with a restart at 10:29:30 and the
    backlog it explains at 10:30, plus seeds, new and filtered items and one item older than the window."""
    every_minute(conn, AAPL, at(10, 0), at(10, 10))
    event(conn, "started", at(10, 29, 30))
    every_minute(conn, AAPL, at(10, 30), at(10, 40))
    item(conn, AAPL, at(9, 0, day=1), "new")
    item(conn, AAPL, at(10, 0) + 5, "seed")
    item(conn, AAPL, at(10, 0) + 6, "seed")
    item(conn, AAPL, at(10, 5) + 5, "new")
    item(conn, AAPL, at(10, 6) + 5, "filtered", reason="junk")
    item(conn, AAPL, at(10, 30) + 5, "backlog")
    item(conn, AAPL, at(10, 30) + 6, "backlog")


def summary(conn, now=NOW, days=7):
    report, _ = news_status.build(conn, CONFIG, now, days)
    return {label: str(value) if not isinstance(value, bool) else value for label, value in report.summary}


def test_the_status_mix_backlog_hours_outage_and_verdicts(conn):
    seed_restart(conn)
    poll(conn, "bls_cpi", at(10, 0), cadence=1800)
    poll(conn, "bls_cpi", at(10, 30), "timeout", cadence=1800)
    found = summary(conn)
    assert found["items new"] == "1" and found["items seed"] == "2"
    assert found["items backlog"] == "2" and found["items filtered"] == "1"
    assert found["polls ok"] == "23" and found["polls timeout"] == "1" and found["polls clock_skew"] == "0"
    assert found["bytes received"] == "240"
    assert found["backlog channel hours"] == "1"
    assert found["outage minutes"] == "14" and found["longest outage minutes"] == "14"
    assert found["channels with outage"] == "1"
    assert found["order verdict"] is True and found["backlog verdict"] is True
    assert found["order violations"] == "0" and found["unexplained backlog items"] == "0"


def test_backlog_per_channel_per_hour_and_outage_rows(conn):
    seed_restart(conn)
    item(conn, "bls_cpi", at(11, 0) + 5, "backlog")
    poll(conn, "bls_cpi", at(11, 0), "rate_limited", cadence=1800, http_status=429)
    _, sections = news_status.build(conn, CONFIG, at(11, 5), 7)
    sent, backlog, outages = sections
    assert sent.title == "requests per channel per day"
    assert list(sent.rows) == [
        {"channel": "bls_cpi", "day": "2026-10-12", "requests": 1, "ok": 0, "not_modified": 0, "failed": 1,
         "http_403": 0, "http_429": 1, "bytes": 10, "mean_ms": 200, "max_ms": 200},
        {"channel": AAPL, "day": "2026-10-12", "requests": 22, "ok": 22, "not_modified": 0, "failed": 0,
         "http_403": 0, "http_429": 0, "bytes": 220, "mean_ms": 200, "max_ms": 200}]
    assert backlog.title == "backlog per channel per hour"
    assert list(backlog.rows) == [{"channel": "bls_cpi", "hour": "2026-10-12T15:00Z", "backlog": 1},
                                  {"channel": AAPL, "hour": "2026-10-12T14:00Z", "backlog": 2}]
    # 14 minutes from 10:16 to 10:29, and 19 from 10:46 to 11:04 after the last poll at 10:40
    assert list(outages.rows) == [{"channel": AAPL, "day": "2026-10-12", "minutes": 33}]


def test_the_order_verdict_fails_on_a_planted_out_of_order_row(conn):
    seed_restart(conn)
    assert summary(conn)["order verdict"] is True
    item(conn, AAPL, at(10, 20), "new")
    found = summary(conn)
    assert found["order verdict"] is False and found["order violations"] == "1"
    assert found["backlog verdict"] is True


def test_the_backlog_verdict_fails_on_a_planted_backlog_item_outside_any_restart(conn):
    seed_restart(conn)
    item(conn, AAPL, at(10, 40) + 10, "backlog")       # returned by the 10:40 poll, a minute after the last one
    found = summary(conn)
    assert found["backlog verdict"] is False and found["unexplained backlog items"] == "1"
    assert found["order verdict"] is True


@pytest.mark.parametrize("explain", ["restart", "pause", "failed poll", "outage"])
def test_a_backlog_item_after_a_restart_pause_failure_or_outage_is_explained(conn, explain):
    poll(conn, "bls_cpi", at(9, 0), cadence=1800)
    if explain == "restart":
        event(conn, "started", at(9, 30))
    elif explain == "pause":
        event(conn, "pause", at(9, 30))
    elif explain == "failed poll":
        poll(conn, "bls_cpi", at(9, 30), "http_error", cadence=1800)
    later = at(10, 45) if explain == "outage" else at(9, 45)    # 6,300 s is past three cadences of 1,800 s
    poll(conn, "bls_cpi", later, cadence=1800)
    item(conn, "bls_cpi", later + 5, "backlog")
    assert summary(conn, now=later + MS_PER_MINUTE)["backlog verdict"] is True


def test_a_pause_counts_over_the_whole_freeze_its_late_event_reports(conn):
    """The heartbeat writes the pause only after waking, and a channel whose wait ended first polls before it."""
    poll(conn, "bls_cpi", at(9, 0), cadence=1800)
    poll(conn, "bls_cpi", at(9, 45), cadence=1800)
    item(conn, "bls_cpi", at(9, 45) + 5, "backlog")
    event(conn, "pause", at(9, 46), "the process was paused for 960 s")      # frozen from 9:30 to 9:46
    assert summary(conn, now=at(9, 50))["backlog verdict"] is True


def test_a_pause_that_ended_before_the_previous_poll_explains_nothing(conn):
    event(conn, "pause", at(8, 59), "the process was paused for 960 s")
    poll(conn, "bls_cpi", at(9, 0), cadence=1800)
    poll(conn, "bls_cpi", at(9, 45), cadence=1800)
    item(conn, "bls_cpi", at(9, 45) + 5, "backlog")
    assert summary(conn, now=at(9, 50))["backlog verdict"] is False


def test_a_backlog_item_without_a_previous_successful_poll_is_unexplained(conn):
    poll(conn, "bls_cpi", at(9, 0), cadence=1800)
    item(conn, "bls_cpi", at(9, 0) + 5, "backlog")
    assert summary(conn)["backlog verdict"] is False


@pytest.mark.parametrize("explain", ["restart", "failed poll"])
def test_a_backlog_item_whose_previous_success_was_pruned_needs_a_restart_or_failure_before_it(conn, explain):
    if explain == "restart":
        event(conn, "started", at(8, 59))
    else:
        poll(conn, "bls_cpi", at(8, 30), "timeout", cadence=1800)
    poll(conn, "bls_cpi", at(9, 0), cadence=1800)
    item(conn, "bls_cpi", at(9, 0) + 5, "backlog")
    assert summary(conn)["backlog verdict"] is True


def test_the_window_counts_only_recent_rows(conn):
    seed_restart(conn)
    found = summary(conn, days=1)
    assert found["days"] == "1" and found["items new"] == "1"
    later = summary(conn, now=NOW + 3 * MS_PER_DAY, days=1)
    assert later["items new"] == "0" and later["items seed"] == "0" and later["polls ok"] == "0"


def test_the_dormant_fomc_channel_counts_no_outage_between_statement_days(conn):
    window = at(13, 58, day=28)
    for n in range(0, 22 * 4):
        poll(conn, "fed_press_monetary", window + n * 15 * MS_PER_SECOND, cadence=15)
    found = summary(conn, now=at(12, 0, day=30))
    assert found["outage minutes"] == "0"


def samples_conn(conn):
    seed_restart(conn)
    original = item(conn, AAPL, at(10, 35), "new", title="Fictional sample headline kept for the audit")
    item(conn, AAPL, at(10, 36), "filtered", title="Fictional sample headline filtered as junk", reason="junk")
    for vendor, jaccard in (("dup-81", 0.81), ("dup-95", 0.95), ("dup-79", 0.79)):
        db.insert(conn, "news_dups", {"item_seq": original, "channel": "finnhub_general", "vendor_id": vendor,
                                      "seen_at": iso(at(10, 37)), "method": "near", "jaccard": jaccard})


def test_samples_are_drawn_with_the_report_seed_and_hold_near_pairs_from_the_threshold_to_point_nine(conn):
    samples_conn(conn)
    since = iso(NOW - MS_PER_DAY)
    drawn = news_status.samples(conn, CONFIG, since, 50)
    assert set(drawn) == {"kept headlines", "filtered headlines", "near duplicate pairs"}
    assert [row["duplicate"] for row in drawn["near duplicate pairs"]] == ["dup-81"]
    assert {row["title"] for row in drawn["filtered headlines"]} >= {"Fictional sample headline filtered as junk"}
    two = news_status.samples(conn, CONFIG, since, 2)
    assert two == news_status.samples(conn, CONFIG, since, 2)
    assert len(two["kept headlines"]) == 2
    assert [r["seq"] for r in two["kept headlines"]] == sorted(r["seq"] for r in two["kept headlines"])


def status_config(tmp_path):
    """A copy of the committed news config whose data_dir is a test folder."""
    text, found = re.subn(r'(?m)^data_dir = "[^"]*"', 'data_dir = "data/news"', TEXT)
    assert found == 1
    path = tmp_path / "news.toml"
    path.write_text(text, encoding="utf-8")
    return path, tmp_path / "data" / "news" / "news.db"


def run(capsys, *argv):
    code = news_status.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def seeded_now(conn):
    """Rows relative to the real clock, which main reads, with a restart explaining two backlog items."""
    now = LiveClock().now_ms() - 30 * MS_PER_MINUTE
    poll(conn, "bls_cpi", now - 40 * MS_PER_MINUTE, cadence=1800)
    event(conn, "started", now - 5 * MS_PER_MINUTE)
    poll(conn, "bls_cpi", now, cadence=1800)
    item(conn, "bls_cpi", now - 40 * MS_PER_MINUTE + 5, "seed", title="Fictional seed headline")
    item(conn, "bls_cpi", now + 5, "backlog", title="Fictional backlog headline")
    item(conn, "bls_cpi", now + 6, "backlog", title="Fictional second backlog headline")
    return now


def test_main_prints_counts_and_verdicts_only_without_detail(tmp_path, capsys):
    config, path = status_config(tmp_path)
    conn = db.open_writer(path)
    seeded_now(conn)
    conn.close()
    code, out, err = run(capsys, "--config", str(config), "--samples", "5")
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == "news status" and "items backlog: 2" in lines and "backlog verdict: pass" in lines
    assert all(SUMMARY_LINE.fullmatch(line) for line in lines[1:-1]), lines
    assert re.fullmatch(r"\d+ rows hidden, pass --detail in your own terminal to see them", lines[-1])
    assert "Fictional" not in out and "bls_cpi" not in out


def test_detail_adds_rows_and_samples(tmp_path, capsys):
    config, path = status_config(tmp_path)
    conn = db.open_writer(path)
    seeded_now(conn)
    conn.close()
    code, out, _ = run(capsys, "--config", str(config), "--detail")
    assert code == 0 and "backlog per channel per hour" in out and "bls_cpi" in out
    assert "Fictional" not in out and "kept headlines" not in out
    code, out, _ = run(capsys, "--config", str(config), "--detail", "--samples", "5")
    assert code == 0
    assert "kept headlines" in out and "Fictional backlog headline" in out and "Fictional seed headline" in out


def test_main_exits_1_on_a_failed_verdict(tmp_path, capsys):
    config, path = status_config(tmp_path)
    conn = db.open_writer(path)
    now = seeded_now(conn)
    item(conn, "bls_cpi", now - MS_PER_MINUTE, "new", title="Fictional late headline")
    conn.close()
    code, out, _ = run(capsys, "--config", str(config))
    assert code == 1 and "order verdict: fail" in out and "order violations: 1" in out


def test_main_refuses_a_missing_database_and_a_window_past_poll_retention(tmp_path, capsys):
    config, path = status_config(tmp_path)
    code, out, err = run(capsys, "--config", str(config))
    assert code == 2 and out == "" and err.startswith("error:") and "does not exist" in err
    assert not path.exists()
    db.open_writer(path).close()
    code, _, err = run(capsys, "--config", str(config), "--days", str(CONFIG.news.keep_polls_days + 1))
    assert code == 2 and "keep_polls_days" in err
    code, _, _ = run(capsys, "--config", str(config), "--days", str(CONFIG.news.keep_polls_days))
    assert code == 0


def test_main_opens_the_database_read_only(tmp_path, capsys):
    config, path = status_config(tmp_path)
    conn = db.open_writer(path)
    seeded_now(conn)
    before = conn.execute("SELECT count(*) FROM news_items").fetchone()[0]
    run(capsys, "--config", str(config), "--detail", "--samples", "3")
    assert conn.execute("SELECT count(*) FROM news_items").fetchone()[0] == before
    conn.close()


def test_every_summary_value_is_a_count_or_a_verdict(conn):
    seed_restart(conn)
    report, _ = news_status.build(conn, CONFIG, NOW, 7, 3)
    assert all(isinstance(value, (bool, output.Count)) for _, value in report.summary)
