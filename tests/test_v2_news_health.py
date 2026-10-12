"""News health of PLAN_V2 section 7.8. The heartbeat flag drops for each condition alone, FinBERT not on, an item
routed to a scorer waiting longer than score_wait_seconds for its primary score, and an unclean clock check, while
Haiku and Sonnet never count. A channel in outage is listed and never makes news unhealthy, and the dormant FOMC
channel is never in outage between statement days. Instants are written with fixed UTC offsets."""

from datetime import datetime, timedelta, timezone

import pytest

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, iso
from btc_survivor.v2.config import load_news_config
from btc_survivor.v2.news import db, health
from btc_survivor.v2.news.health import CLOCK_SKEW, FINBERT_OFF, ON, SCORE_WAIT, Heartbeat, Scorers, Spend
from btc_survivor.v2.news.schedule import Schedule
from helpers import ROOT

EDT, EST = timezone(timedelta(hours=-4)), timezone(timedelta(hours=-5))
CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
SCHEDULE = Schedule(CONFIG)
HAIKU = CONFIG.llm.scorer_name
WAIT_MS = int(CONFIG.news.score_wait_seconds * MS_PER_SECOND)
FINBERT = Scorers(finbert=ON)


def at(y, mo, d, h, mi, tz, s=0):
    return int(datetime(y, mo, d, h, mi, s, tzinfo=tz).timestamp()) * MS_PER_SECOND


NOW = at(2026, 10, 12, 10, 0, EDT)                 # a Monday in regular hours


@pytest.fixture
def conn(tmp_path):
    connection = db.open_writer(tmp_path / "news.db")
    yield connection
    connection.close()


def item(conn, received_ms, route="llm", vendor_id=None):
    """A minimal news_items row, with a fictional headline."""
    return db.insert(conn, "news_items", {
        "source": "finnhub", "channel": "finnhub_general", "vendor_id": vendor_id or str(received_ms), "kind": "media",
        "received_at": iso(received_ms), "title": "Fictional headline", "tickers": "MACRO", "relevance": "macro",
        "status": "new" if route != "none" else "seed", "title_hash": "0123456789abcdef", "route": route})


def score(conn, item_seq, scorer, status="scored"):
    db.insert(conn, "item_scores", {"item_seq": item_seq, "scorer": scorer, "route": "llm", "status": status,
                                    "attempts": 1, "scored_at": iso(NOW)})


def write(conn, now=NOW, clock_clean=True, last_ok=None, since=None, scorers=FINBERT):
    return Heartbeat(CONFIG, conn, SCHEDULE).write(now, clock_clean, last_ok or {}, now if since is None else since,
                                                   scorers)


def test_healthy_with_finbert_on_no_long_wait_and_a_clean_clock(conn):
    row = write(conn)
    assert (row["healthy"], row["reason"], row["channels_down"]) == (1, None, "")
    stored = conn.execute("SELECT healthy, reason, waiting, oldest_wait_s FROM news_heartbeat").fetchone()
    assert tuple(stored) == (1, None, 0, None)


@pytest.mark.parametrize("finbert", ["off", "failed", "loading"])
def test_finbert_not_on_drops_the_flag_alone(conn, finbert):
    row = write(conn, scorers=Scorers(haiku=ON, sonnet=ON, finbert=finbert))
    assert (row["healthy"], row["reason"], row["finbert_state"]) == (0, FINBERT_OFF, finbert)


def test_an_unclean_clock_check_drops_the_flag_alone(conn):
    row = write(conn, clock_clean=False)
    assert (row["healthy"], row["reason"]) == (0, CLOCK_SKEW)


def test_a_routed_item_waiting_past_score_wait_seconds_drops_the_flag_at_the_edge(conn):
    seq = item(conn, NOW - WAIT_MS)
    row = write(conn)
    assert (row["healthy"], row["waiting"], row["oldest_wait_s"]) == (1, 1, CONFIG.news.score_wait_seconds)
    row = write(conn, now=NOW + 1)
    assert (row["healthy"], row["reason"], row["waiting"]) == (0, SCORE_WAIT, 1)
    score(conn, seq, HAIKU)
    assert write(conn, now=NOW + 1)["healthy"] == 1


def test_every_failed_condition_is_named(conn):
    item(conn, NOW - WAIT_MS - 1)
    row = write(conn, clock_clean=False, scorers=Scorers())
    assert row["healthy"] == 0
    assert row["reason"] == ", ".join((FINBERT_OFF, SCORE_WAIT, CLOCK_SKEW))


def test_haiku_and_sonnet_states_never_count(conn):
    for haiku, sonnet in (("off", "off"), ("paused", "capped"), ("capped", "on")):
        row = write(conn, scorers=Scorers(haiku=haiku, sonnet=sonnet, finbert=ON))
        assert (row["healthy"], row["haiku_state"], row["sonnet_state"]) == (1, haiku, sonnet)


def test_only_a_primary_score_ends_the_wait(conn):
    old = NOW - WAIT_MS - MS_PER_MINUTE
    cases = {
        "haiku scored": ("llm", [(HAIKU, "scored")], False),
        "haiku irrelevant": ("llm", [(HAIKU, "irrelevant")], False),
        "haiku failed, finbert fallback": ("llm", [(HAIKU, "failed"), (health.FINBERT_SCORER, "scored")], False),
        "haiku skipped, finbert untagged": ("llm", [(HAIKU, "skipped_off"), (health.FINBERT_SCORER, "untagged")],
                                            False),
        "haiku failed alone": ("llm", [(HAIKU, "failed")], True),
        "finbert alone on an llm item": ("llm", [(health.FINBERT_SCORER, "scored")], True),
        "sonnet alone": ("llm", [(CONFIG.llm.sonnet.scorer_name, "scored")], True),
        "nothing yet": ("llm", [], True),
        "finbert route scored": ("finbert", [(health.FINBERT_SCORER, "scored")], False),
        "finbert route waiting": ("finbert", [(HAIKU, "scored")], True),
        "never routed": ("none", [], False),
    }
    for name, (route, scores, waits) in cases.items():
        conn.execute("DELETE FROM item_scores")
        conn.execute("DELETE FROM news_items")
        seq = item(conn, old, route, vendor_id=name)
        for scorer, status in scores:
            score(conn, seq, scorer, status)
        waiting = Heartbeat(CONFIG, conn, SCHEDULE).waiting(NOW)
        assert waiting.count == int(waits), name
        assert (waiting.oldest_wait_s is not None) is waits, name


def test_the_wait_floor_never_hides_a_later_item(conn):
    heart = Heartbeat(CONFIG, conn, SCHEDULE)
    first = item(conn, NOW - 3 * MS_PER_MINUTE)
    second = item(conn, NOW - 2 * MS_PER_MINUTE)
    assert heart.waiting(NOW).count == 2
    score(conn, first, HAIKU)
    assert heart.waiting(NOW) == health.Waiting(1, 120.0)
    score(conn, second, HAIKU, "irrelevant")
    assert heart.waiting(NOW) == health.Waiting(0, None)
    third = item(conn, NOW - MS_PER_MINUTE)
    assert heart.waiting(NOW) == health.Waiting(1, 60.0)
    score(conn, third, HAIKU, "failed")
    score(conn, third, health.FINBERT_SCORER)
    assert heart.waiting(NOW).count == 0


def test_a_channel_in_outage_is_listed_and_news_stays_healthy(conn):
    last_ok = {"finnhub_company_AAPL": (NOW - 20 * MS_PER_MINUTE, 60), "bls_cpi": (NOW - MS_PER_MINUTE, 1800)}
    row = write(conn, last_ok=last_ok, since=NOW - MS_PER_MINUTE)
    assert row["healthy"] == 1 and row["reason"] is None
    assert row["channels_down"] == "finnhub_company_AAPL"
    stored = conn.execute("SELECT healthy, channels_down FROM news_heartbeat").fetchone()
    assert tuple(stored) == (1, "finnhub_company_AAPL")


def test_outage_starts_past_three_cadences_or_the_floor_whichever_is_longer():
    news = CONFIG.news
    cases = (
        ("finnhub_company_AAPL", NOW, 60, 300),                              # regular hours, the 300 s floor
        ("edgar_atom_8k", NOW, 30, 300),
        ("bls_cpi", NOW, 1800, 5400),                                        # 07:00 to 19:00 on a weekday
        ("fed_speeches", at(2026, 10, 17, 12, 0, EDT), 3600, 10800),         # a Saturday
        ("finnhub_general", at(2026, 10, 12, 1, 0, EDT), 600, 1800),         # overnight
    )
    for channel, now, cadence, limit_s in cases:
        assert SCHEDULE.cadence(channel, now) == cadence, channel
        last = now - limit_s * MS_PER_SECOND
        args = (news.outage_cadence_multiple, news.outage_floor_seconds)
        assert not health.in_outage(SCHEDULE, channel, last, cadence, now, *args), channel
        assert health.in_outage(SCHEDULE, channel, last - 1, cadence, now, *args), channel


def test_a_cadence_that_just_got_faster_uses_the_longer_one():
    news = CONFIG.news
    regular = at(2026, 10, 12, 9, 30, EDT)                    # Finnhub goes from 120 s to 60 s
    args = (news.outage_cadence_multiple, news.outage_floor_seconds)
    assert SCHEDULE.cadence("finnhub_company_AAPL", regular) == 60
    last = regular - 330 * MS_PER_SECOND
    assert not health.in_outage(SCHEDULE, "finnhub_company_AAPL", last, 120, regular, *args)
    assert health.in_outage(SCHEDULE, "finnhub_company_AAPL", last, 60, regular, *args)


def test_the_dormant_fomc_channel_is_never_in_outage_between_statement_days():
    news = CONFIG.news
    args = (news.outage_cadence_multiple, news.outage_floor_seconds)
    last = at(2026, 10, 28, 14, 19, EDT, 45)                  # the last fast poll of the October window
    channel = "fed_press_monetary"
    for now in (at(2026, 10, 28, 14, 30, EDT), at(2026, 11, 5, 12, 0, EST), at(2026, 12, 9, 13, 57, EST, 59)):
        assert SCHEDULE.cadence(channel, now) is None
        assert not health.in_outage(SCHEDULE, channel, last, 15, now, *args)
    window = at(2026, 12, 9, 13, 58, EST)
    assert health.dormant_ms(SCHEDULE, channel, last, 15) == window - last - 15 * MS_PER_SECOND
    edge = window + 285 * MS_PER_SECOND                       # 300 s after a poll one cadence before the window
    assert not health.in_outage(SCHEDULE, channel, last, 15, edge, *args)
    assert health.in_outage(SCHEDULE, channel, last, 15, edge + 1, *args)
    assert health.channels_down(SCHEDULE, {channel: (last, 15)}, at(2026, 11, 5, 12, 0, EST), *args,
                                since_ms=last) == tuple(c.name for c in SCHEDULE.channels if c.name != channel)


def test_a_channel_never_polled_counts_from_the_process_start():
    news = CONFIG.news
    args = (news.outage_cadence_multiple, news.outage_floor_seconds)
    since = NOW - 300 * MS_PER_SECOND
    down = health.channels_down(SCHEDULE, {}, NOW, *args, since_ms=since)
    assert down == ()
    down = health.channels_down(SCHEDULE, {}, NOW + 1, *args, since_ms=since)
    assert "finnhub_company_AAPL" in down and "bls_cpi" not in down and "fed_press_monetary" not in down


def test_outage_minutes_are_the_minute_boundaries_past_the_limit():
    news = CONFIG.news
    minutes = health.outage_minutes(SCHEDULE, "finnhub_company_AAPL", NOW, 60, NOW, NOW + 20 * MS_PER_MINUTE,
                                    news.outage_cadence_multiple, news.outage_floor_seconds)
    assert minutes == [NOW + m * MS_PER_MINUTE for m in range(6, 20)]


def test_the_heartbeat_row_holds_hashes_states_and_counts_only(conn):
    item(conn, NOW - MS_PER_MINUTE)
    row = Heartbeat(CONFIG, conn, SCHEDULE).write(NOW, True, {}, NOW, Scorers(), Spend(0.5, 2.0, True))
    stored = dict(conn.execute("SELECT * FROM news_heartbeat").fetchone())
    assert stored == {**row, "seq": 1}
    assert stored["config_hash"] == CONFIG.config_hash
    assert (stored["prompt_sha256"], stored["schema_sha256"]) == (CONFIG.llm.prompt_sha256, CONFIG.llm.schema_sha256)
    assert (stored["haiku_state"], stored["sonnet_state"], stored["finbert_state"]) == ("off", "off", "off")
    assert (stored["waiting"], stored["oldest_wait_s"]) == (1, 60.0)
    assert (stored["spent_day_usd"], stored["spent_month_usd"], stored["spend_alert"]) == (0.5, 2.0, 1)
    assert stored["ts"] == iso(NOW)
