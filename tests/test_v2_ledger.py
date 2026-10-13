"""The key ledger and the monthly cap of PLAN_V2 section 8.6, on throwaway ledgers in temporary folders. The cost
formula is checked by hand, the byte bound never falls below an exact count, 10,000 seeded sequences of Haiku and
Sonnet calls never pass the cap and never cost more than their reservations, two processes on one ledger never pass it
together, a running reservation blocks a call at the edge, the month rolls over at 00:00 UTC, nothing caps a day, and
the daily alert is raised once, logged, written and set on the heartbeat while calls go on."""

import dataclasses
import json
import logging
import math
import os
import random
import sqlite3
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone

import pytest

from btc_survivor.clock import MS_PER_DAY, MS_PER_MINUTE, iso
from btc_survivor.v2.config import HAIKU_MODEL, SONNET_MODEL, LlmCapsConfig, load_news_config
from btc_survivor.v2.news import db, ledger
from btc_survivor.v2.news.health import ON, Heartbeat, Scorers, Spend
from btc_survivor.v2.news.ledger import (
    DONE, LOST, RESERVED, SPEND_ALERT, STARTED, UNKNOWN, USAGE, Ledger, LedgerError, Prices, Rates, SpendWatch, Usage,
    byte_bound, cap_until, cost_usd, day_bounds, month_bounds, request_bytes, reserve_usd)
from btc_survivor.v2.news.schedule import Schedule
from helpers import ROOT

CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
PRICES = Prices.from_config(CONFIG.llm.prices)
CAP = CONFIG.llm.caps.monthly_usd
MAX_TOKENS = {HAIKU_MODEL: CONFIG.llm.max_tokens, SONNET_MODEL: CONFIG.llm.sonnet.max_tokens}
OVERHEAD = CONFIG.llm.caps.reserve_overhead_tokens


def utc(y, mo, d, h=0, mi=0, s=0):
    return int(datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp()) * 1000


NOON = utc(2026, 10, 14, 12)


def process(tmp_path, name="news", config_hash="a" * 64):
    return f"{config_hash} {tmp_path / name}"


def open_ledger(tmp_path, monthly_usd=CAP, name="news", config_hash="a" * 64):
    return Ledger(tmp_path / "llm_ledger.db", tmp_path / "llm_ledger.lock", monthly_usd,
                  process(tmp_path, name, config_hash))


@pytest.fixture
def key_ledger(tmp_path):
    with open_ledger(tmp_path) as opened:
        yield opened


def rows(tmp_path):
    conn = sqlite3.connect(tmp_path / "llm_ledger.db")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM reservations ORDER BY id")]
    finally:
        conn.close()


# The cost formula by hand


def test_prices_come_from_the_config_with_sonnet_input_at_one_rate():
    assert PRICES.haiku == Rates(0.10, 0.50, 1.25, 0.10)
    assert PRICES.sonnet == Rates(2.00, 10.00, 1.0, 1.0)
    assert (PRICES.haiku.input_bound, PRICES.sonnet.input_bound) == (1.25, 1.0)
    assert PRICES.long_prompt_threshold_tokens == 100_000
    assert json.loads(PRICES.table()) == {
        "haiku": {"in_rate": 0.10, "out_rate": 0.50, "write": 1.25, "read": 0.10},
        "sonnet": {"in_rate": 2.00, "out_rate": 10.00, "write": 1.0, "read": 1.0},
        "long_prompt_threshold_tokens": 100_000}


def test_the_cost_formula_by_hand():
    # Haiku, 1,000 plain input tokens, the 1,200 token system block written to the cache and 300 output tokens,
    # (1000 * 0.10 + 1200 * 0.10 * 1.25 + 300 * 0.50) / 1e6 = (100 + 150 + 150) / 1e6
    written = Usage(input_tokens=1000, output_tokens=300, cache_creation_input_tokens=1200)
    assert PRICES.cost(HAIKU_MODEL, written) == pytest.approx(0.000400, rel=1e-12)
    # the same call reading the block from the cache, (100 + 1200 * 0.10 * 0.10 + 150) / 1e6 = 262 / 1e6
    read = Usage(input_tokens=1000, output_tokens=300, cache_read_input_tokens=1200)
    assert PRICES.cost(HAIKU_MODEL, read) == pytest.approx(0.000262, rel=1e-12)
    # Sonnet on a short official item, (1900 * 2.00 + 150 * 10.00) / 1e6 = 5300 / 1e6
    assert PRICES.cost(SONNET_MODEL, Usage(1900, 150)) == pytest.approx(0.0053, rel=1e-12)
    assert cost_usd(Rates(1.0, 1.0), Usage(0, 0)) == 0.0


def test_the_reservation_by_hand_matches_the_plan():
    # A full Haiku batch counted at 4,000 tokens, (4000 * 0.10 * 1.25 + 4096 * 0.50) / 1e6, about 0.003 dollars.
    assert PRICES.reserve(HAIKU_MODEL, 4000, 4096) == pytest.approx(0.002548, rel=1e-12)
    # A short and a long form Sonnet call, (1900 * 2 + 1024 * 10) / 1e6 and (9000 * 2 + 1024 * 10) / 1e6.
    assert PRICES.reserve(SONNET_MODEL, 1900, 1024) == pytest.approx(0.01404, rel=1e-12)
    assert PRICES.reserve(SONNET_MODEL, 9000, 1024) == pytest.approx(0.02824, rel=1e-12)
    assert reserve_usd(PRICES.haiku, 0, 1) == pytest.approx(0.5e-6, rel=1e-12)


def test_a_call_that_uses_every_bound_costs_exactly_its_reservation():
    for counted, max_tokens in ((1, 1), (4000, 4096), (19_999, 4096), (12_345, 1)):
        reserved = PRICES.reserve(HAIKU_MODEL, counted, max_tokens)
        assert PRICES.cost(HAIKU_MODEL, Usage(0, max_tokens, cache_creation_input_tokens=counted)) == reserved
        reserved = PRICES.reserve(SONNET_MODEL, counted, max_tokens)
        assert PRICES.cost(SONNET_MODEL, Usage(counted, max_tokens)) == reserved


def test_thinking_tokens_are_booked_as_output_and_a_thinking_room_reservation_covers_them():
    # Sonnet with 1,024 tokens of answer room and 1,024 of thinking room on a short item, (1900 * 2 + 2048 * 10) / 1e6.
    reserved = PRICES.reserve(SONNET_MODEL, 1900, 1024 + 1024)
    assert reserved == pytest.approx(0.02428, rel=1e-12)
    # output_tokens already holds the thinking, so 900 thinking tokens inside 1,000 output cost as 1,000 output.
    thought = Usage(1900, 1000, thinking_tokens=900)
    assert thought.billed_output_tokens == 1000
    assert PRICES.cost(SONNET_MODEL, thought) == PRICES.cost(SONNET_MODEL, Usage(1900, 1000)) < reserved
    # A thinking count above output_tokens, which the API never reports, still books every thinking token as output.
    assert PRICES.cost(SONNET_MODEL, Usage(1900, 100, thinking_tokens=2048)) == reserved
    with pytest.raises(ValueError):
        cost_usd(PRICES.sonnet, Usage(0, 0, thinking_tokens=-1))


def test_bad_inputs_and_unknown_models_are_refused():
    with pytest.raises(LedgerError, match="no prices"):
        PRICES.rates("not-a-model")
    with pytest.raises(LedgerError, match="100000 token"):
        PRICES.reserve(HAIKU_MODEL, 100_001, 4096)
    assert PRICES.reserve(HAIKU_MODEL, 100_000, 4096) > 0
    with pytest.raises(ValueError):
        reserve_usd(PRICES.haiku, -1, 10)
    with pytest.raises(ValueError):
        reserve_usd(PRICES.haiku, 10, 0)
    with pytest.raises(ValueError):
        cost_usd(PRICES.haiku, Usage(-1, 0))


# The byte bound


ALPHABET = ("abc XYZ 0189 .,;:!?'\"\\/\n\t{}[]<>&%$#@" "éüßñ" "日本語の見出し" "Ωπ" "\U0001F4C8\U0001F680" "\u0007")


def random_text(rng, n):
    return "".join(rng.choice(ALPHABET) for _ in range(n))


def random_request(rng):
    """A request of the shape the scorer sends, with text of every width UTF 8 has, quotes, escapes and controls."""
    items = [{"id": f"i{k:02d}", "headline": random_text(rng, rng.randint(0, 300)),
              "text": random_text(rng, rng.randint(0, 700)), "source": "finnhub"} for k in range(rng.randint(1, 20))]
    schema = {"type": "object", "properties": {"items": {"type": "array", "description": random_text(rng, 40)}}}
    return {
        "model": rng.choice((HAIKU_MODEL, SONNET_MODEL)), "max_tokens": 4096,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        "system": [{"type": "text", "text": random_text(rng, rng.randint(1, 2000)),
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Score each item.\n" + "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) for item in items)}],
    }


def fake_exact_count(request):
    """A tokenizer at its worst, one token for every UTF 8 byte of the system prompt, the schema and the messages,
    plus 200 tokens of framing, below reserve_overhead_tokens."""
    system = sum(len(block["text"].encode("utf-8")) for block in request["system"])
    schema = len(json.dumps(request["output_config"]["format"]["schema"], ensure_ascii=False).encode("utf-8"))
    messages = sum(len(message["content"].encode("utf-8")) for message in request["messages"])
    return system + schema + messages + 200


def test_the_byte_bound_is_never_below_an_exact_count():
    rng = random.Random(20261012)
    for _ in range(2000):
        request = random_request(rng)
        assert byte_bound(request, OVERHEAD) >= fake_exact_count(request)
        assert byte_bound(request, OVERHEAD) == request_bytes(request) + OVERHEAD
    assert request_bytes({"b": "é", "a": 1}) == len('{"a":1,"b":"\\u00e9"}')


# The monthly cap


def month_start(index):
    """The first instant of the index-th UTC month after October 2026."""
    months = 9 + index
    return utc(2026 + months // 12, months % 12 + 1, 1)


def random_usage(rng, model, counted, max_tokens):
    """Usage inside the call's bounds, its counted input split between plain, cache written and cache read tokens,
    sometimes at the very bounds."""
    if rng.random() < 0.1:
        return Usage(0, max_tokens, cache_creation_input_tokens=counted) if model == HAIKU_MODEL \
            else Usage(counted, max_tokens)
    used = rng.randint(0, counted)
    written = rng.randint(0, used)
    read = rng.randint(0, used - written)
    return Usage(used - written - read, rng.randint(0, max_tokens), written, read)


def test_ten_thousand_seeded_sequences_never_pass_the_cap_or_cost_more_than_reserved(key_ledger):
    refusals = accepted = at_edge = 0
    for seed in range(10_000):
        rng = random.Random(seed)
        start = month_start(seed)
        spent_before = CAP - rng.uniform(0, 0.12)
        key_ledger.book(HAIKU_MODEL, "earlier calls", 0, spent_before, start)
        now = start + MS_PER_MINUTE
        for _ in range(rng.randint(1, 10)):
            model = rng.choice((HAIKU_MODEL, SONNET_MODEL))
            counted, max_tokens = rng.randint(1, 20_000), MAX_TOKENS[model]
            reserved = PRICES.reserve(model, counted, max_tokens)
            reservation = key_ledger.reserve(model, ledger.make_ref("scorer", [seed]), 1, reserved, now)
            if reservation is None:
                refusals += 1
                assert key_ledger.spend(now).month_usd + reserved > CAP - 1e-12, seed
                break
            accepted += 1
            assert key_ledger.spend(now).month_usd <= CAP, seed
            outcome = rng.random()
            if outcome < 0.7:
                cost = PRICES.cost(model, random_usage(rng, model, counted, max_tokens))
                assert cost <= reserved, seed
                at_edge += cost == reserved
                assert key_ledger.finish(reservation, cost) == cost
            elif outcome < 0.9:
                assert key_ledger.finish(reservation, None) == reserved
            now += rng.randint(1, 3 * MS_PER_MINUTE)
        assert key_ledger.spend(now).month_usd <= CAP, seed
    assert refusals > 2000 and accepted > 20_000 and at_edge > 1000


def test_a_running_reservation_blocks_a_call_at_the_edge(tmp_path):
    with open_ledger(tmp_path, monthly_usd=1.0) as small:
        running = small.reserve(SONNET_MODEL, "sonnet_v1:1", 1, 0.75, NOON)
        assert running is not None
        assert small.reserve(HAIKU_MODEL, "haiku_v1:2", 1, 0.25 + 2 ** -40, NOON) is None
        edge = small.reserve(HAIKU_MODEL, "haiku_v1:3", 1, 0.25, NOON)
        assert edge is not None and small.spend(NOON).month_usd == 1.0
        assert small.reserve(HAIKU_MODEL, "haiku_v1:4", 1, 2 ** -40, NOON) is None
        assert small.finish(running, 0.5) == 0.5
        assert small.reserve(HAIKU_MODEL, "haiku_v1:5", 1, 0.25, NOON) is not None
        assert small.reserve(HAIKU_MODEL, "haiku_v1:6", 1, 2 ** -40, NOON) is None
    assert [row["status"] for row in rows(tmp_path)] == [DONE, STARTED, STARTED]


def test_a_failed_or_lost_call_stays_booked_at_its_reservation(tmp_path, key_ledger):
    failed = key_ledger.reserve(HAIKU_MODEL, "haiku_v1:1,2", 2, 0.003, NOON)
    assert key_ledger.finish(failed, None) == 0.003
    with pytest.raises(LedgerError, match="not running"):
        key_ledger.finish(failed, 0.001)
    lost = key_ledger.reserve(SONNET_MODEL, ledger.make_ref("sonnet_v1", [7]), 1, 0.014, NOON)
    other = Ledger(tmp_path / "llm_ledger.db", tmp_path / "llm_ledger.lock", CAP, process(tmp_path, "drill"))
    elsewhere = other.reserve(HAIKU_MODEL, "haiku_v1:9", 1, 0.002, NOON)
    restarted = open_ledger(tmp_path, config_hash="b" * 64)   # the same data folder with a changed config
    try:
        assert restarted.recover() == [lost]
        assert restarted.recover() == []
        assert ledger.parse_ref(lost.ref) == ("sonnet_v1", (7,))
        assert other.finish(elsewhere, 0.001) == 0.001
    finally:
        restarted.close()
        other.close()
    assert [(row["status"], row["cost_usd"], row["cost_basis"]) for row in rows(tmp_path)] == [
        (UNKNOWN, 0.003, RESERVED), (LOST, 0.014, RESERVED), (DONE, 0.001, USAGE)]
    assert key_ledger.spend(NOON).month_usd == pytest.approx(0.018, rel=1e-12)


def test_the_month_rolls_over_at_midnight_utc(tmp_path):
    with open_ledger(tmp_path, monthly_usd=1.0) as small:
        last = utc(2026, 10, 31, 23, 59, 59) + 999
        assert small.reserve(HAIKU_MODEL, "haiku_v1:1", 1, 1.0, last) is not None
        assert small.reserve(HAIKU_MODEL, "haiku_v1:2", 1, 0.001, last) is None
        assert cap_until(last) == utc(2026, 11, 1)
        assert small.reserve(HAIKU_MODEL, "haiku_v1:3", 1, 1.0, utc(2026, 11, 1)) is not None
        assert small.spend(last) == Spend(day_usd=1.0, month_usd=1.0)
        assert small.spend(utc(2026, 11, 1)) == Spend(day_usd=1.0, month_usd=1.0)
        december = utc(2026, 12, 31, 23, 59)
        assert small.reserve(SONNET_MODEL, "sonnet_v1:4", 1, 1.0, december) is not None
        assert cap_until(december) == utc(2027, 1, 1)
        assert small.spend(utc(2027, 1, 1)).month_usd == 0.0
    assert month_bounds(utc(2026, 2, 14)) == (utc(2026, 2, 1), utc(2026, 3, 1))
    assert month_bounds(utc(2028, 2, 29, 23)) == (utc(2028, 2, 1), utc(2028, 3, 1))
    assert day_bounds(NOON) == (utc(2026, 10, 14), utc(2026, 10, 15))


def test_no_daily_cap_of_any_kind(tmp_path):
    caps = {field.name for field in dataclasses.fields(LlmCapsConfig)}
    assert caps == {"ledger_file", "ledger_lock_file", "monthly_usd", "daily_alert_share", "reserve_overhead_tokens"}
    with open_ledger(tmp_path) as key_ledger:
        key_ledger.book(HAIKU_MODEL, "earlier today", 0, 19.0, NOON)
        for k in range(4):
            assert key_ledger.reserve(HAIKU_MODEL, f"haiku_v1:{k}", 1, 0.25, NOON + k) is not None
        assert key_ledger.spend(NOON) == Spend(day_usd=20.0, month_usd=20.0)
        assert key_ledger.reserve(HAIKU_MODEL, "haiku_v1:5", 1, 2 ** -40, NOON) is None
    with open_ledger(tmp_path / "busy") as busy:
        for k in range(3000):             # no limit on calls or items in one day either
            assert busy.reserve(HAIKU_MODEL, f"haiku_v1:{k}", 20, 0.001, NOON + k) is not None


# Two processes on one ledger


WORKER = """
import json, random, sys, time
from pathlib import Path
from btc_survivor.v2.config import HAIKU_MODEL, SONNET_MODEL
from btc_survivor.v2.news.ledger import Ledger
folder, seed, cap, now = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
ledger = Ledger(folder / "llm_ledger.db", folder / "llm_ledger.lock", cap, f"hash{seed} {folder / str(seed)}")
(folder / f"ready{seed}").touch()
while not (folder / "go").exists():
    time.sleep(0.001)
accepted = refused = over = steps = 0
running = []
while refused < 100 and steps < 5000:
    steps += 1
    reservation = ledger.reserve(rng.choice((HAIKU_MODEL, SONNET_MODEL)), f"w{seed}", 1, rng.uniform(0.0005, 0.03), now)
    if reservation is None:
        refused += 1
    else:
        accepted += 1
        running.append(reservation)
    over += ledger.spend(now).month_usd > cap
    if running and rng.random() < 0.5:
        done = running.pop(rng.randrange(len(running)))
        ledger.finish(done, done.reserved_usd * rng.random())
print(json.dumps({"accepted": accepted, "refused": refused, "over": over}))
"""


def test_two_processes_on_one_ledger_never_pass_the_cap_together(tmp_path):
    cap = 2.0
    workers = [subprocess.Popen([sys.executable, "-c", WORKER, str(tmp_path), str(seed), str(cap), str(NOON)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for seed in (1, 2)]
    try:
        deadline = time.monotonic() + 60
        while not all((tmp_path / f"ready{seed}").exists() for seed in (1, 2)):
            assert time.monotonic() < deadline and all(worker.poll() is None for worker in workers)
            time.sleep(0.01)
        (tmp_path / "go").touch()
        results = []
        for worker in workers:
            out, err = worker.communicate(timeout=120)
            assert worker.returncode == 0, err
            results.append(json.loads(out))
    finally:
        for worker in workers:
            worker.kill()
    assert all(result["over"] == 0 and result["accepted"] > 10 and result["refused"] > 0 for result in results)
    stored = rows(tmp_path)
    assert len(stored) == sum(result["accepted"] for result in results)
    owners = [row["process"].rsplit("/", 1)[1] for row in stored]
    assert set(owners) == {"1", "2"}
    assert sum(a != b for a, b in zip(owners, owners[1:])) > 5     # the two really took turns
    assert math.fsum(row["reserved_usd"] for row in stored) > cap
    with Ledger(tmp_path / "llm_ledger.db", tmp_path / "llm_ledger.lock", cap, "check") as check:
        assert check.spend(NOON).month_usd <= cap


# The daily alert


def test_the_daily_alert_is_raised_once_logged_written_and_on_the_heartbeat_while_calls_go_on(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=ledger.__name__)
    conn = db.open_writer(tmp_path / "news" / "news.db")
    key_ledger = open_ledger(tmp_path)
    try:
        share = CONFIG.llm.caps.daily_alert_share
        watch = SpendWatch(key_ledger, conn, share)
        assert watch.threshold_usd == 5.0
        assert watch.check(NOON) == Spend(0.0, 0.0, False)
        key_ledger.book(HAIKU_MODEL, "earlier today", 0, 5.0, NOON)
        assert watch.check(NOON + 1) == Spend(5.0, 5.0, False)   # a day at exactly a quarter has not passed it
        running = key_ledger.reserve(SONNET_MODEL, "sonnet_v1:1", 1, 0.25, NOON + 2)
        first = watch.check(NOON + 3)
        assert first == Spend(5.25, 5.25, True)
        assert watch.check(NOON + 4).alert
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1 and "passed the daily alert of 5.00 dollars" in warnings[0].getMessage()
        events = conn.execute("SELECT ts, kind, detail FROM collector_events WHERE kind = ?", (SPEND_ALERT,)).fetchall()
        assert [(row["ts"], row["kind"]) for row in events] == [(iso(NOON + 3), SPEND_ALERT)]
        assert "5.2500 dollars" in events[0]["detail"]
        assert key_ledger.reserve(HAIKU_MODEL, "haiku_v1:2", 20, 0.003, NOON + 5) is not None   # nothing stops

        beat = Heartbeat(CONFIG, conn, Schedule(CONFIG)).write(NOON + 6, True, {}, NOON, Scorers(finbert=ON), first)
        assert (beat["spend_alert"], beat["spent_day_usd"], beat["spent_month_usd"]) == (1, 5.25, 5.25)
        stored = conn.execute("SELECT spend_alert, spent_day_usd FROM news_heartbeat").fetchone()
        assert tuple(stored) == (1, 5.25)

        key_ledger.finish(running, 0.0)                    # back under a quarter, and the alert holds all day
        later = watch.check(NOON + MS_PER_MINUTE)
        assert later.alert and later.day_usd == pytest.approx(5.003, rel=1e-12)
        restarted = SpendWatch(key_ledger, conn, share)    # a restart the same day writes no second event
        assert restarted.check(NOON + 2 * MS_PER_MINUTE).alert
        tomorrow = day_bounds(NOON)[1]
        assert watch.check(tomorrow) == Spend(0.0, later.month_usd, False)
        assert not restarted.check(tomorrow + MS_PER_DAY // 2).alert
        assert conn.execute("SELECT count(*) FROM collector_events WHERE kind = ?", (SPEND_ALERT,)).fetchone()[0] == 1
        assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 1
    finally:
        key_ledger.close()
        conn.close()


def test_the_alert_needs_a_share_between_zero_and_one(key_ledger, tmp_path):
    conn = db.open_writer(tmp_path / "news" / "news.db")
    try:
        for share in (0.0, 1.0, -0.1):
            with pytest.raises(ValueError):
                SpendWatch(key_ledger, conn, share)
    finally:
        conn.close()


# The ledger file


def test_the_ledger_is_for_the_owner_alone_and_refuses_another_database(tmp_path):
    with open_ledger(tmp_path / "keys"):
        pass
    for name in ("llm_ledger.db", "llm_ledger.lock"):
        assert stat.S_IMODE(os.stat(tmp_path / "keys" / name).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(tmp_path / "keys").st_mode) == 0o700
    news = tmp_path / "news" / "news.db"
    db.open_writer(news).close()
    with pytest.raises(LedgerError, match="is not a key ledger"):
        Ledger(news, tmp_path / "news" / "llm_ledger.lock", CAP, "x")
    conn = db.open_reader(news)
    try:
        assert db.check(conn) == []
    finally:
        conn.close()


def test_from_config_reads_llm_caps_and_names_the_process(tmp_path):
    caps = dataclasses.replace(CONFIG.llm.caps, ledger_file=str(tmp_path / "keys" / "llm_ledger.db"),
                               ledger_lock_file=str(tmp_path / "keys" / "llm_ledger.lock"), monthly_usd=0.01)
    config = dataclasses.replace(CONFIG, llm=dataclasses.replace(CONFIG.llm, caps=caps))
    with Ledger.from_config(config) as drill:
        assert (drill.path, drill.monthly_usd) == (tmp_path / "keys" / "llm_ledger.db", 0.01)
        assert drill.process == f"{config.config_hash} {config.data_dir}"
        assert drill.reserve(HAIKU_MODEL, "haiku_v1:1", 1, 0.003, NOON) is not None
        assert drill.reserve(HAIKU_MODEL, "haiku_v1:2", 1, 0.008, NOON) is None


def test_the_cap_drill_total_is_booked_as_one_row(key_ledger, tmp_path):
    row_id = key_ledger.book(HAIKU_MODEL, "cap drill", 9, 0.0093, NOON)
    assert rows(tmp_path) == [{"id": row_id, "ts": iso(NOON), "process": process(tmp_path), "model": HAIKU_MODEL,
                               "ref": "cap drill", "n_items": 9, "reserved_usd": 0.0093, "cost_usd": 0.0093,
                               "cost_basis": USAGE, "status": DONE}]
    with pytest.raises(LedgerError):
        key_ledger.reserve("not-a-model", "x", 1, 0.001, NOON)
    with pytest.raises(ValueError):
        key_ledger.reserve(HAIKU_MODEL, "x", 1, 0.0, NOON)
