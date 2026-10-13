"""The Haiku and Sonnet scorers of PLAN_V2 sections 8.3 to 8.8, on a temporary news.db, a throwaway key ledger, a fake
clock and a fake client that answers by markers in each headline, so no test reaches the Anthropic API.

One refused item among 20 is isolated in at most 11 calls, a cut off or unreadable answer splits the same way, a batch
never takes more than 24 calls, and an item left out of an answer is requeued once. Every other row of the failure
table gets its action, a 5xx, a timeout or a call lost when the process died among them, each booked at its
reservation with its items failed and never sent again. Every Finnhub item routed llm reaches Haiku and never Sonnet,
Sonnet sees only the official kinds, a Sonnet error never changes Haiku's state, and a refusal by the monthly cap
turns both models off until the next UTC month while FinBERT covers every item."""

import asyncio
import dataclasses
import hashlib
import json
import logging
import sqlite3
import zlib
from datetime import datetime, timezone

import pytest

from btc_survivor.clock import MS_PER_HOUR, MS_PER_MINUTE, MS_PER_SECOND, LiveClock, iso
from btc_survivor.v2.config import HAIKU_MODEL, OFFICIAL_KINDS, SONNET_MODEL, load_news_config
from btc_survivor.v2.news import db, llm_request
from btc_survivor.v2.news.finbert import Probs
from btc_survivor.v2.news.finbert_scorer import FinbertScorer
from btc_survivor.v2.news.health import FINBERT_SCORER, Heartbeat
from btc_survivor.v2.news.batcher import SINGLE, Batch, Pending
from btc_survivor.v2.news.ledger import Ledger, Prices, Usage, byte_bound, make_ref
from btc_survivor.v2.news.llm_client import (
    AUTH, BAD_REQUEST, CONNECTION, MALFORMED, MAX_TOKENS, MODEL_MISMATCH, OK, RATE_LIMITED, REFUSAL, SERVER,
    SPEND_LIMIT, TIMEOUT, CallResult, sdk_version)
from btc_survivor.v2.news.llm_request import RequestError
from btc_survivor.v2.news.llm_scorer import CAPPED, OFF, ON, PAUSED, decide
from btc_survivor.v2.news.schedule import Schedule
from btc_survivor.v2.news.scoring import Scoring
from helpers import ROOT

BASE = load_news_config(ROOT / "configs" / "v2" / "news.toml")
CONFIG = dataclasses.replace(BASE, llm=dataclasses.replace(
    BASE.llm, prompt_sha256=llm_request.PROMPT_SHA256, schema_sha256=llm_request.SCHEMA_SHA256,
    sonnet=dataclasses.replace(BASE.llm.sonnet, enabled=True)))    # both models on, whatever the committed config
PRICES = Prices.from_config(CONFIG.llm.prices)
MARGIN = CONFIG.llm.caps.reserve_overhead_tokens   # reserved on top of an exact count, section 8.6
HAIKU, SONNET = CONFIG.llm.scorer_name, CONFIG.llm.sonnet.scorer_name
SONNET_MAX = llm_request.sonnet_spec(CONFIG.llm).request_max_tokens   # answer and thinking room, section 8.1
SOURCE = {"media": "finnhub", "8k": "edgar", "8k_a": "edgar", "fed_press": "fed", "fed_monetary": "fed",
          "fed_speech": "fed", "bls_release": "bls"}
TICKERS = {"finnhub": "AAPL", "edgar": "MSFT", "fed": "MACRO", "bls": "MACRO"}
SECOND, MINUTE = MS_PER_SECOND, MS_PER_MINUTE


def utc(y, mo, d, h=0, mi=0):
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()) * MS_PER_SECOND


NOW = utc(2026, 10, 14, 4)            # before the 06:00 UTC probe, so no probe joins a test that is not about it
NEXT_MONTH = utc(2026, 11, 1)
USAGE = Usage(1800, 60)
IMPACT = {"ticker": "AAPL", "event_type": "earnings", "is_new": True, "direction": 1, "surprise": 1,
          "horizon": "days", "confidence": 4}


# The fake client


def result(status, model, text=None, *, http=200, stop="end_turn", usage=USAGE, response_model=None, **extra):
    """A call that came back with a response."""
    return CallResult(status, 900, http_status=http, stop_reason=stop, usage=usage,
                      response_model=response_model or model, request_id="req_fixture", text=text,
                      raw_json=json.dumps({"fixture": text}), **extra)


def failure(status, http=None, retry_after_s=None):
    """A call that ended in an error, with no usage."""
    return CallResult(status, 1200, http_status=http, error_class="FixtureError", error_detail="a fictional error",
                      retry_after_s=retry_after_s)


def impacts_for(item):
    if "NOTHING" in item["headline"]:
        return []
    if "INVALID" in item["headline"]:
        return [{**IMPACT, "direction": 2}]
    if "TWO" in item["headline"]:
        return [IMPACT, {**IMPACT, "ticker": "MACRO", "event_type": "macro_data", "direction": -1, "confidence": 5}]
    return [IMPACT]


def answer(model, items, **extra):
    entries = [{"id": item["id"], "impacts": impacts_for(item)} for item in items if "SILENT" not in item["headline"]]
    return result(OK, model, json.dumps({"items": entries}), **extra)


def default_rule(model, items):
    """Refuse a call holding REFUSE, cut one holding LONG, leave out every SILENT item and answer the rest."""
    headlines = " ".join(item["headline"] for item in items)
    if "REFUSE" in headlines:
        return result(REFUSAL, model, "", stop="refusal", refusal_category="fixture_category")
    if "LONG" in headlines:
        return result(MAX_TOKENS, model, '{"items": [{"id": "i01", "imp', stop="max_tokens")
    return answer(model, items)


def lines(request):
    return [json.loads(line) for line in request["messages"][0]["content"].split("\n")[1:]]


class FakeClient:
    """Answers each call by a rule on its model and items, recording every request, and never reaches the network."""

    def __init__(self, rule=default_rule, counted=1800):
        self.rule = rule
        self.counted = counted
        self.requests = []

    async def count_tokens(self, request):
        return self.counted(lines(request)) if callable(self.counted) else self.counted

    async def send(self, request):
        self.requests.append(request)
        return self.rule(request["model"], lines(request))

    def headlines(self, model=None):
        return [[item["headline"] for item in lines(r)] for r in self.requests if model in (None, r["model"])]


class FakeClock(LiveClock):
    def __init__(self, ms):
        super().__init__()
        self.ms = ms

    def now_ms(self):
        return self.ms


class FakeModel:
    def score(self, headline, text=None):
        return Probs(0.6, 0.1, 0.3, 1, 12)


# One news process on temporary files


def process_name(folder):
    return f"{'a' * 64} {folder / 'news'}"


class Env:
    """A news.db, a throwaway key ledger, a fake clock and client, and the scoring of one news process on them."""

    def __init__(self, folder, rule=default_rule, monthly_usd=20.0, config=CONFIG, counted=1800):
        self.folder = folder
        self.config = config
        self.monthly_usd = monthly_usd
        self.conn = db.open_writer(folder / "news.db")
        self.clock = FakeClock(NOW)
        self.client = FakeClient(rule, counted)
        self.ledger = self._ledger()
        self.scoring = Scoring(config, self.conn, self.clock, self.client, self.ledger)
        self.scoring.recover(NOW)

    def _ledger(self):
        return Ledger(self.folder / "llm_ledger.db", self.folder / "llm_ledger.lock", self.monthly_usd,
                      process_name(self.folder))

    def restart(self, rule=default_rule):
        """A new news process on the same files, as after a crash or a stop."""
        self.ledger.close()
        self.ledger = self._ledger()
        self.client = FakeClient(rule)
        self.scoring = Scoring(self.config, self.conn, self.clock, self.client, self.ledger)
        return self.scoring.recover(self.clock.stamp())

    def run(self, at=None, stop=None):
        """Step both scorers at one instant until neither sends anything more."""
        if at is not None:
            self.clock.ms = at

        async def go():
            event = stop or asyncio.Event()
            while any([await scorer.step(self.clock.ms, event) for scorer in self.scoring.scorers]):
                pass
        asyncio.run(go())

    def close(self):
        self.ledger.close()
        self.conn.close()


@pytest.fixture
def make_env(tmp_path):
    made = []

    def make(**kwargs):
        env = Env(tmp_path / f"process{len(made)}", **kwargs)
        made.append(env)
        return env
    yield make
    for env in made:
        env.close()


_vendor_ids = iter(range(1, 10**6))


def add(conn, headline="Apple results beat estimates", kind="media", route="llm", status="new", received=NOW,
        vendor_ts=None, body=None):
    source = SOURCE[kind]
    row = {"source": source, "channel": "fixture", "vendor_id": str(next(_vendor_ids)), "kind": kind,
           "received_at": iso(received), "detected_at": None, "vendor_ts": vendor_ts, "title": headline,
           "summary": "A fictional summary of the item.", "url": None, "publisher": None,
           "tickers": TICKERS[source], "relevance": "official" if source != "finnhub" else "title",
           "status": status, "filter_reason": None, "title_hash": "0" * 16, "items_8k": None, "route": route}
    with db.transaction(conn):
        seq = db.insert(conn, "news_items", row)
        if body is not None:
            db.insert(conn, "news_bodies", {"item_seq": seq, "text": body, "chars_total": len(body), "truncated": 0,
                                            "doc_urls": "[]"})
    return seq


def finals(conn, scorer=HAIKU):
    return {row["item_seq"]: dict(row) for row in conn.execute("SELECT * FROM item_scores WHERE scorer = ?",
                                                                 (scorer,))}


def calls(conn, scorer=HAIKU):
    return [dict(row) for row in conn.execute("SELECT * FROM llm_calls WHERE scorer = ? ORDER BY call_id",
                                              (scorer,))]


def call_items(conn, call_id):
    return [row[0] for row in conn.execute("SELECT item_seq FROM llm_call_items WHERE call_id = ? ORDER BY position",
                                           (call_id,))]


def state(conn, key):
    return db.get_value(conn, "scorer_state", key)


def events(conn, kind):
    return [row[0] for row in conn.execute("SELECT detail FROM collector_events WHERE kind = ? ORDER BY id", (kind,))]


def ledger_rows(env):
    conn = sqlite3.connect(env.folder / "llm_ledger.db")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM reservations ORDER BY id")]
    finally:
        conn.close()


# Splits, the call cap and the requeue


def test_one_refused_item_among_20_is_isolated_in_at_most_11_calls(make_env):
    worst = 0
    for position in range(20):
        env = make_env()
        seqs = [add(env.conn, f"Item {n} REFUSE" if n == position else f"Item {n} fine") for n in range(20)]
        env.run()
        sent = env.client.requests
        assert len(sent) <= 11, position
        worst = max(worst, len(sent))
        rows = finals(env.conn)
        refused = rows[seqs[position]]
        assert refused["status"] == "refused"
        assert all(rows[seq]["status"] == "scored" for seq in seqs if seq != seqs[position])
        made = calls(env.conn)
        assert [call["kind"] for call in made] == ["batch"] + ["split"] * (len(made) - 1)
        assert {call["root_call_id"] for call in made[1:]} == {made[0]["call_id"]} and made[0]["root_call_id"] is None
        last = next(call for call in made if call["call_id"] == refused["call_id"])
        assert (last["status"], last["n_items"], last["refusal_category"]) == ("refusal", 1, "fixture_category")
        assert call_items(env.conn, last["call_id"]) == [seqs[position]]
        assert refused["attempts"] == sum(seqs[position] in call_items(env.conn, c["call_id"]) for c in made)
        assert finals(env.conn, SONNET) == {}
    assert worst == 11


@pytest.mark.parametrize("cut", [
    lambda model: result(MAX_TOKENS, model, '{"items": [{"id"', stop="max_tokens"),
    lambda model: result(OK, model, "this is not json"),
    lambda model: result(OK, model, json.dumps({"items": [], "note": "an extra outer field"})),
    lambda model: result(MALFORMED, model, json.dumps({"items": []}), stop="pause_turn"),
], ids=["max_tokens", "unparsable", "envelope", "stop_reason"])
def test_a_cut_off_or_unreadable_answer_splits_and_a_single_such_item_ends_malformed(make_env, cut):
    env = make_env(rule=lambda model, items: cut(model) if any("LONG" in i["headline"] for i in items)
                   else answer(model, items))
    seqs = [add(env.conn, "Item LONG" if n == 3 else f"Item {n}") for n in range(5)]
    env.run(at=NOW + MINUTE)
    assert env.client.headlines() == [
        ["Item 0", "Item 1", "Item 2", "Item LONG", "Item 4"], ["Item 0", "Item 1"],
        ["Item 2", "Item LONG", "Item 4"], ["Item 2"], ["Item LONG", "Item 4"], ["Item LONG"], ["Item 4"]]
    rows = finals(env.conn)
    assert rows[seqs[3]]["status"] == "malformed" and rows[seqs[3]]["attempts"] == 4
    assert all(rows[seq]["status"] == "scored" for n, seq in enumerate(seqs) if n != 3)
    statuses = [call["status"] for call in calls(env.conn)]
    expected = "max_tokens" if statuses[0] == "max_tokens" else "malformed"
    assert statuses == [expected, "ok", expected, "ok", expected, expected, "ok"]


def test_a_batch_never_takes_more_than_24_calls_and_its_last_items_end_malformed(make_env, caplog):
    def cut_unless_alone(model, items):
        if len(items) > 1:
            return result(MAX_TOKENS, model, '{"items": [', stop="max_tokens")
        return answer(model, items)

    env = make_env(rule=cut_unless_alone)
    seqs = [add(env.conn, f"Item {n}") for n in range(20)]
    with caplog.at_level(logging.WARNING):
        env.run()
        env.run(at=NOW + 10 * MINUTE)
    assert len(env.client.requests) == CONFIG.llm.max_calls_per_batch == 24
    rows = finals(env.conn)
    assert set(rows) == set(seqs)
    alone = {call_items(env.conn, call["call_id"])[0] for call in calls(env.conn) if call["n_items"] == 1}
    assert {seq for seq in seqs if rows[seq]["status"] == "scored"} == alone
    assert {seq for seq in seqs if rows[seq]["status"] == "malformed"} == set(seqs) - alone
    assert alone == set(seqs[:11])
    assert all(rows[seq]["attempts"] >= 1 and rows[seq]["call_id"] is not None for seq in seqs)
    assert "24 calls on one batch" in caplog.text


def test_an_item_left_out_of_an_answer_is_requeued_once_then_marked_malformed(make_env):
    env = make_env()
    seqs = [add(env.conn, "Item one"), add(env.conn, "Item SILENT"), add(env.conn, "Item three")]
    env.run(at=NOW + MINUTE)
    env.run(at=NOW + 10 * MINUTE)
    assert env.client.headlines() == [["Item one", "Item SILENT", "Item three"], ["Item SILENT"]]
    rows = finals(env.conn)
    made = calls(env.conn)
    assert (rows[seqs[1]]["status"], rows[seqs[1]]["attempts"], rows[seqs[1]]["call_id"]) == (
        "malformed", 2, made[1]["call_id"])
    assert [rows[seqs[0]]["status"], rows[seqs[2]]["status"]] == ["scored", "scored"]
    assert [call["kind"] for call in made] == ["batch", "batch"] and made[1]["root_call_id"] is None


def test_a_requeued_item_answered_the_second_time_is_scored(make_env):
    seen = set()

    def once_silent(model, items):
        out = [{**item, "headline": item["headline"] if item["headline"] not in seen else "answered"}
               for item in items]
        seen.update(item["headline"] for item in items)
        return answer(model, out)

    env = make_env(rule=once_silent)
    seq = add(env.conn, "Item SILENT")
    env.run(at=NOW + MINUTE)
    row = finals(env.conn)[seq]
    assert (row["status"], row["attempts"], len(env.client.requests)) == ("scored", 2, 2)


def test_a_model_paused_between_the_calls_of_a_split_skips_the_rest_of_its_batch_at_once(make_env):
    refused = result(REFUSAL, HAIKU_MODEL, "", stop="refusal")
    outcomes = iter([refused, refused, failure(RATE_LIMITED, 429)])
    env = make_env(rule=lambda model, items: next(outcomes))
    seqs = [add(env.conn, f"Item {n}") for n in range(8)]
    env.run(at=NOW + MINUTE)
    assert [len(request) for request in env.client.headlines()] == [8, 4, 2]
    rows, made = finals(env.conn), calls(env.conn)
    assert [(rows[s]["status"], rows[s]["attempts"], rows[s]["call_id"]) for s in seqs] == [
        ("failed", 3, made[2]["call_id"])] * 2 + [("skipped_off", 2, made[1]["call_id"])] * 2 + [
        ("skipped_off", 1, made[0]["call_id"])] * 4


def test_an_invalid_item_ends_malformed_at_once_while_valid_items_are_kept(make_env):
    env = make_env()
    seqs = [add(env.conn, "Item fine"), add(env.conn, "Item INVALID"), add(env.conn, "Item NOTHING")]
    env.run(at=NOW + MINUTE)
    rows = finals(env.conn)
    assert [rows[seq]["status"] for seq in seqs] == ["scored", "malformed", "irrelevant"]
    assert len(env.client.requests) == 1 and calls(env.conn)[0]["status"] == "ok"


# The rest of the failure table


def test_a_valid_answer_gives_each_item_one_final_row_and_the_call_its_full_record(make_env):
    env = make_env()
    seqs = [add(env.conn, "Item TWO"), add(env.conn, "Item NOTHING")]
    env.run(at=NOW + MINUTE)
    rows = finals(env.conn)
    assert [(rows[seq]["status"], rows[seq]["attempts"], rows[seq]["route"]) for seq in seqs] == [
        ("scored", 1, "llm"), ("irrelevant", 1, "llm")]
    impacts = [dict(row) for row in env.conn.execute("SELECT * FROM score_impacts ORDER BY ticker")]
    # direction * confidence / 5 * 0.5 for days * (2 + 1) / 5, so 0.24 and -0.3
    assert [(i["item_seq"], i["scorer"], i["ticker"], i["strength"], i["direction"], i["confidence"])
            for i in impacts] == [(seqs[0], HAIKU, "AAPL", 0.24, 1, 4), (seqs[0], HAIKU, "MACRO", -0.3, -1, 5)]
    [call] = calls(env.conn)
    assert rows[seqs[0]]["call_id"] == call["call_id"] and call_items(env.conn, call["call_id"]) == seqs
    request = env.client.requests[0]
    body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    reserved = PRICES.reserve(HAIKU_MODEL, 1800 + MARGIN, 4096)
    assert {key: call[key] for key in ("kind", "status", "n_items", "model", "response_model", "request_id",
                                       "effort", "thinking", "max_tokens", "stop_reason", "http_status",
                                       "counted_tokens", "count_basis", "input_tokens", "output_tokens",
                                       "cost_basis", "sdk_version", "prompt_sha256", "schema_sha256")} == {
        "kind": "batch", "status": "ok", "n_items": 2, "model": HAIKU_MODEL, "response_model": HAIKU_MODEL,
        "request_id": "req_fixture", "effort": "low", "thinking": "disabled", "max_tokens": 4096,
        "stop_reason": "end_turn", "http_status": 200, "counted_tokens": 1800, "count_basis": "count_tokens",
        "input_tokens": 1800, "output_tokens": 60, "cost_basis": "usage", "sdk_version": sdk_version(),
        "prompt_sha256": llm_request.PROMPT_SHA256, "schema_sha256": llm_request.SCHEMA_SHA256}
    assert call["request_sha256"] == hashlib.sha256(body).hexdigest()
    assert call["reserved_usd"] == reserved and call["cost_usd"] == PRICES.cost(HAIKU_MODEL, USAGE) < reserved
    assert json.loads(call["price_table"]) == json.loads(PRICES.table())
    assert json.loads(zlib.decompress(call["response_z"])) == {"fixture": json.dumps(
        {"items": [{"id": "i01", "impacts": impacts_for({"headline": "Item TWO"})}, {"id": "i02", "impacts": []}]})}
    [booked] = ledger_rows(env)
    assert (booked["status"], booked["cost_usd"], booked["cost_basis"], booked["ref"], booked["n_items"]) == (
        "done", call["cost_usd"], "usage", make_ref(HAIKU, seqs), 2)
    assert call["ledger_id"] == booked["id"] and state(env.conn, "haiku_call") == ""
    assert state(env.conn, "haiku_response_model") == HAIKU_MODEL


@pytest.mark.parametrize("status, http", [(SERVER, 500), (SERVER, 529), (TIMEOUT, None), (CONNECTION, None)])
def test_a_5xx_timeout_or_lost_connection_is_booked_at_its_reservation_with_its_items_failed_and_never_resent(
        make_env, status, http):
    env = make_env(rule=lambda model, items: failure(status, http))
    seqs = [add(env.conn, "Item one"), add(env.conn, "Item two")]
    env.run(at=NOW + MINUTE)
    env.run(at=NOW + 2 * MINUTE)
    env.run(at=NOW + 20 * MINUTE)
    assert len(env.client.requests) == 1
    rows = finals(env.conn)
    [call] = calls(env.conn)
    assert [(rows[s]["status"], rows[s]["attempts"], rows[s]["call_id"]) for s in seqs] == [
        ("failed", 1, call["call_id"])] * 2
    assert (call["status"], call["http_status"], call["cost_basis"], call["error_class"]) == (
        status, http, "reserved", "FixtureError")
    assert call["cost_usd"] == call["reserved_usd"] == PRICES.reserve(HAIKU_MODEL, 1800 + MARGIN, 4096)
    assert json.loads(zlib.decompress(call["response_z"])) == {"error_detail": "a fictional error"}
    [booked] = ledger_rows(env)
    assert (booked["status"], booked["cost_usd"], booked["cost_basis"]) == ("unknown", call["reserved_usd"],
                                                                            "reserved")
    assert env.scoring.words() == (ON, ON)
    later = add(env.conn, "Item later", received=NOW + 20 * MINUTE)
    env.run(at=NOW + 21 * MINUTE)
    assert finals(env.conn)[later]["status"] == "failed" and len(env.client.requests) == 2


@pytest.mark.parametrize("retry_after_s, pause_ms", [(None, 15 * MINUTE), (60.0, 15 * MINUTE),
                                                     (1800.0, 30 * MINUTE)])
def test_a_429_pauses_that_model_for_the_longer_of_15_minutes_and_its_retry_after(make_env, retry_after_s, pause_ms):
    def rule(model, items):
        return failure(RATE_LIMITED, 429, retry_after_s) if model == HAIKU_MODEL else answer(model, items)

    env = make_env(rule=rule)
    first = add(env.conn, "Item first")
    env.run(at=NOW + MINUTE)
    until = NOW + MINUTE + pause_ms
    assert finals(env.conn)[first]["status"] == "failed"
    assert env.scoring.words(NOW + MINUTE) == (PAUSED, ON)
    assert (state(env.conn, "haiku_state"), state(env.conn, "haiku_until")) == (PAUSED, iso(until))
    assert f"{HAIKU} paused until {iso(until)} after rate_limited" in events(env.conn, "scorer_state")
    during = add(env.conn, "Item during", received=until - SECOND)
    official = add(env.conn, "An official item", kind="8k", received=until - SECOND)
    env.run(at=until - SECOND)
    assert finals(env.conn)[during]["status"] == finals(env.conn)[official]["status"] == "skipped_off"
    assert finals(env.conn, SONNET)[official]["status"] == "scored"
    after = add(env.conn, "Item after", received=until)
    env.client.rule = default_rule
    env.run(at=until + MINUTE)
    assert finals(env.conn)[after]["status"] == "scored" and env.scoring.words() == (ON, ON)
    assert state(env.conn, "haiku_state") == ON and state(env.conn, "haiku_until") == ""


def test_a_spend_limit_turns_both_models_off_until_the_next_utc_month(make_env):
    env = make_env(rule=lambda model, items: failure(SPEND_LIMIT, 400))
    official = add(env.conn, "An official item", kind="8k")
    env.run()
    assert finals(env.conn)[official]["status"] == "failed"
    assert finals(env.conn, SONNET)[official]["status"] == "skipped_cap"
    assert env.scoring.words() == (CAPPED, CAPPED) and state(env.conn, "cap_until") == iso(NEXT_MONTH)
    arriving = add(env.conn, "Item arriving", received=NEXT_MONTH - MINUTE)
    env.run(at=NEXT_MONTH - MINUTE)
    assert finals(env.conn)[arriving]["status"] == "skipped_cap" and len(env.client.requests) == 1
    assert env.scoring.words(NEXT_MONTH) == (ON, ON)


@pytest.mark.parametrize("status, http", [(BAD_REQUEST, 400), (AUTH, 401), (AUTH, 403), (BAD_REQUEST, 404)])
def test_any_other_4xx_turns_that_model_off_until_restart_with_an_error_line(make_env, caplog, status, http):
    env = make_env(rule=lambda model, items: failure(status, http) if model == HAIKU_MODEL else answer(model, items))
    seq = add(env.conn, "Item first")
    with caplog.at_level(logging.WARNING):
        env.run(at=NOW + MINUTE)
    assert finals(env.conn)[seq]["status"] == "failed" and env.scoring.words() == (OFF, ON)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and f"{HAIKU} turned off until restart by {status}, http {http}" in errors[0].getMessage()
    later = add(env.conn, "Item later", received=NOW + 2 * MINUTE)
    env.run(at=NOW + 3 * MINUTE)
    assert finals(env.conn)[later]["status"] == "skipped_off" and len(env.client.requests) == 1
    env.restart()
    assert env.scoring.words() == (ON, ON)
    again = add(env.conn, "Item after restart", received=NOW + 3 * MINUTE)
    env.run(at=NOW + 4 * MINUTE)
    assert finals(env.conn)[again]["status"] == "scored"


def test_an_answer_from_another_model_is_discarded_booked_at_its_reservation_and_turns_that_model_off(make_env):
    env = make_env(rule=lambda model, items: dataclasses.replace(
        answer(model, items, response_model="another-model"), status=MODEL_MISMATCH))
    seq = add(env.conn, "Item first")
    env.run(at=NOW + MINUTE)
    [call] = calls(env.conn)
    assert finals(env.conn)[seq]["status"] == "failed"
    assert env.conn.execute("SELECT count(*) FROM score_impacts").fetchone()[0] == 0
    assert (call["status"], call["response_model"], call["cost_basis"]) == (MODEL_MISMATCH, "another-model",
                                                                             "reserved")
    assert call["cost_usd"] == call["reserved_usd"] and env.scoring.words()[0] == OFF
    assert state(env.conn, "haiku_response_model") is None


def test_refusals_in_more_than_a_fifth_of_the_last_50_calls_pause_the_model_with_an_alert(make_env, caplog):
    env = make_env(rule=lambda model, items: result(REFUSAL, model, "", stop="refusal"))
    seqs = [add(env.conn, f"Official item {n}", kind="fed_speech") for n in range(12)]
    with caplog.at_level(logging.WARNING):
        env.run()
    rows = finals(env.conn)
    assert [rows[seq]["status"] for seq in seqs] == ["refused"] * 11 + ["skipped_off"]
    assert len(env.client.headlines(HAIKU_MODEL)) == 11 and env.scoring.words()[0] == PAUSED
    assert state(env.conn, "haiku_until") == iso(NOW + 15 * MINUTE)
    alerts = events(env.conn, "scorer_alert")
    assert any(alert.startswith(f"{HAIKU} refused more than 20% of its last 50 calls") for alert in alerts)
    assert f"{HAIKU} refused more than 20% of its last 50 calls" in caplog.text


def test_a_call_lost_when_the_process_died_is_booked_at_its_reservation_with_its_items_failed(make_env):
    env = make_env()

    class Hanging(FakeClient):
        async def send(self, request):
            self.requests.append(request)
            await asyncio.Event().wait()

    env.client = env.scoring.caller.client = Hanging()
    seq = add(env.conn, "An official item", kind="8k")

    async def die():
        task = asyncio.create_task(env.scoring.haiku.step(NOW, asyncio.Event()))
        while not env.client.requests:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(die())
    assert finals(env.conn) == {} and calls(env.conn) == []
    record = json.loads(state(env.conn, "haiku_call"))
    assert record["seqs"] == [seq] and record["row"]["kind"] == "single"
    env.clock.ms = NOW + 30 * SECOND
    assert env.restart() == 1
    [call] = calls(env.conn)
    reserved = PRICES.reserve(HAIKU_MODEL, 1800 + MARGIN, 4096)
    assert (call["status"], call["kind"], call["started_at"], call["finished_at"], call["cost_basis"]) == (
        "lost", "single", iso(NOW), iso(NOW + 30 * SECOND), "reserved")
    assert call["cost_usd"] == call["reserved_usd"] == reserved and len(call["request_sha256"]) == 64
    row = finals(env.conn)[seq]
    assert (row["status"], row["attempts"], row["call_id"]) == ("failed", 1, call["call_id"])
    [booked] = ledger_rows(env)
    assert (booked["status"], booked["cost_usd"], booked["cost_basis"]) == ("lost", reserved, "reserved")
    assert state(env.conn, "haiku_call") == ""
    env.run(at=NOW + MINUTE)
    assert env.client.headlines(HAIKU_MODEL) == [] and finals(env.conn, SONNET)[seq]["status"] == "scored"
    assert env.scoring.recover(NOW + 2 * MINUTE) == 0 and len(calls(env.conn)) == 1
    db.set_values(env.conn, "scorer_state", {"haiku_call": json.dumps(record)})   # a recovery cut short
    assert env.restart() == 0 and len(calls(env.conn)) == 1 and state(env.conn, "haiku_call") == ""


def test_a_reservation_lost_before_its_record_still_books_lost_and_fails_its_items(make_env):
    env = make_env()
    seqs = [add(env.conn, "Item one"), add(env.conn, "Item two")]
    env.ledger.reserve(HAIKU_MODEL, make_ref(HAIKU, seqs), 2, 0.003, NOW)
    env.clock.ms = NOW + MINUTE
    assert env.restart() == 1
    [call] = calls(env.conn)
    assert (call["status"], call["kind"], call["request_sha256"], call["counted_tokens"], call["started_at"]) == (
        "lost", "batch", "", None, iso(NOW + MINUTE))
    assert call_items(env.conn, call["call_id"]) == seqs
    assert [finals(env.conn)[seq]["status"] for seq in seqs] == ["failed", "failed"]
    assert ledger_rows(env)[0]["status"] == "lost"


def test_a_recovery_dying_before_it_writes_a_lost_call_leaves_it_for_the_next_start(make_env, monkeypatch):
    env = make_env()
    seqs = [add(env.conn, "Item one"), add(env.conn, "Item two")]
    env.ledger.reserve(HAIKU_MODEL, make_ref(HAIKU, seqs), 2, 0.003, NOW)

    def killed(self, record, now_ms):
        raise RuntimeError("killed during recovery")
    with monkeypatch.context() as patch:
        patch.setattr(Scoring, "_write_lost", killed)
        with pytest.raises(RuntimeError, match="killed"):
            env.restart()
    assert [row["status"] for row in ledger_rows(env)] == ["started"] and finals(env.conn) == {}
    assert env.restart() == 1
    assert [finals(env.conn)[seq]["status"] for seq in seqs] == ["failed", "failed"]
    assert [row["status"] for row in ledger_rows(env)] == ["lost"] and env.restart() == 0


def test_items_a_stop_left_between_the_calls_of_a_batch_fail_at_the_next_start(make_env):
    env = make_env()
    seqs = [add(env.conn, f"Item {n} REFUSE" if n == 0 else f"Item {n}") for n in range(4)]
    stop = asyncio.Event()

    def stop_after_first(model, items):
        stop.set()
        return default_rule(model, items)

    env.client.rule = stop_after_first
    env.run(at=NOW + MINUTE, stop=stop)
    assert len(env.client.requests) == 1 and finals(env.conn) == {}
    assert env.restart() == 0
    first = calls(env.conn)[0]["call_id"]
    assert [(finals(env.conn)[seq]["status"], finals(env.conn)[seq]["attempts"], finals(env.conn)[seq]["call_id"])
            for seq in seqs] == [("failed", 1, first)] * 4
    env.run(at=NOW + 2 * MINUTE)
    assert env.client.requests == []


def test_a_refusal_by_the_monthly_cap_turns_both_models_off_until_the_next_month_and_finbert_covers_every_item(
        make_env):
    env = make_env(monthly_usd=0.0024)
    early = [add(env.conn, "Item one"), add(env.conn, "Item two")]
    env.run(at=NOW + MINUTE)
    assert [finals(env.conn)[seq]["status"] for seq in early] == ["scored", "scored"]
    late = [add(env.conn, "Item three", received=NOW + MINUTE), add(env.conn, "Item four", received=NOW + MINUTE),
            add(env.conn, "An official item", kind="fed_monetary", received=NOW + MINUTE)]
    env.run(at=NOW + 2 * MINUTE)
    assert len(env.client.requests) == 1 and len(ledger_rows(env)) == 1
    haiku, sonnet = finals(env.conn), finals(env.conn, SONNET)
    assert [(haiku[seq]["status"], haiku[seq]["call_id"], haiku[seq]["attempts"]) for seq in late] == [
        ("skipped_cap", None, 0)] * 3
    assert sonnet[late[2]]["status"] == "skipped_cap" and set(sonnet) == {late[2]}
    assert env.scoring.words() == (CAPPED, CAPPED)
    assert (state(env.conn, "haiku_state"), state(env.conn, "sonnet_state"), state(env.conn, "cap_until")) == (
        CAPPED, CAPPED, iso(NEXT_MONTH))
    arriving = add(env.conn, "Item arriving", received=NEXT_MONTH - MINUTE)
    env.run(at=NEXT_MONTH - MINUTE)
    assert finals(env.conn)[arriving]["status"] == "skipped_cap" and len(env.client.requests) == 1

    finbert = FinbertScorer(CONFIG, env.conn, env.clock, FakeModel())

    async def cover():
        for item in finbert.pending():
            await finbert.score(item)
    asyncio.run(cover())
    every = [*early, *late, arriving]
    assert all(finals(env.conn, FINBERT_SCORER)[seq]["status"] == "scored" for seq in every)
    assert Heartbeat(CONFIG, env.conn, Schedule(CONFIG)).waiting(NEXT_MONTH - MINUTE).count == 0

    env.restart()
    assert env.scoring.words(NEXT_MONTH - 1) == (CAPPED, CAPPED) and env.scoring.words(NEXT_MONTH) == (ON, ON)
    fresh = add(env.conn, "Item next month", received=NEXT_MONTH)
    env.run(at=NEXT_MONTH + MINUTE)
    assert finals(env.conn)[fresh]["status"] == "scored" and state(env.conn, "haiku_state") == ON


# Routing between the two models


def test_every_finnhub_item_routed_llm_reaches_haiku_and_never_sonnet_which_sees_only_official_kinds(make_env):
    env = make_env()
    finnhub = [add(env.conn, f"Finnhub item {n}") for n in range(5)]
    official = [add(env.conn, f"Official {kind}", kind=kind) for kind in OFFICIAL_KINDS]
    amended = add(env.conn, "An amended filing", kind="8k_a", route="none")
    fallback = add(env.conn, "Finnhub item for FinBERT alone", route="finbert")
    seed = add(env.conn, "A seed item", route="none", status="seed")
    env.run(at=NOW + MINUTE)
    haiku, sonnet = finals(env.conn), finals(env.conn, SONNET)
    assert set(haiku) == {*finnhub, *official} and all(row["status"] == "scored" for row in haiku.values())
    assert set(sonnet) == set(official) and all(row["status"] == "scored" for row in sonnet.values())
    assert not {amended, fallback, seed} & (set(haiku) | set(sonnet))
    haiku_calls, sonnet_calls = calls(env.conn), calls(env.conn, SONNET)
    batched = [call for call in haiku_calls if call["kind"] == "batch"]
    assert len(batched) == 1 and call_items(env.conn, batched[0]["call_id"]) == finnhub
    assert sorted(call_items(env.conn, c["call_id"])[0] for c in haiku_calls if c["kind"] == "single") == official
    assert all(call["kind"] == "single" and call["n_items"] == 1 for call in sonnet_calls)
    sent_to_sonnet = {seq for call in sonnet_calls for seq in call_items(env.conn, call["call_id"])}
    assert sent_to_sonnet == set(official)
    kinds = dict(env.conn.execute("SELECT seq, kind FROM news_items").fetchall())
    assert {kinds[seq] for seq in sent_to_sonnet} == set(OFFICIAL_KINDS)
    assert all(not headline.startswith("Finnhub") for request in env.client.headlines(SONNET_MODEL)
               for headline in request)
    assert all(call["model"] == SONNET_MODEL and call["max_tokens"] == SONNET_MAX for call in sonnet_calls)
    assert all(call["thinking"] == CONFIG.llm.sonnet.thinking for call in sonnet_calls)


@pytest.mark.parametrize("status, http", [(BAD_REQUEST, 400), (AUTH, 401), (RATE_LIMITED, 429), (SERVER, 500),
                                          (TIMEOUT, None), (MODEL_MISMATCH, 200)])
def test_a_sonnet_error_never_changes_haiku_state(make_env, status, http):
    def rule(model, items):
        if model == SONNET_MODEL:
            if status == MODEL_MISMATCH:
                return dataclasses.replace(answer(model, items, response_model="another-model"), status=status)
            return failure(status, http)
        return answer(model, items)

    env = make_env(rule=rule)
    first = add(env.conn, "Official first", kind="bls_release")
    env.run()
    assert finals(env.conn)[first]["status"] == "scored" and finals(env.conn, SONNET)[first]["status"] == "failed"
    assert env.scoring.words()[0] == ON and state(env.conn, "haiku_state") == ON
    assert env.scoring.words()[1] == {SERVER: ON, TIMEOUT: ON, RATE_LIMITED: PAUSED}.get(status, OFF)
    second = add(env.conn, "Official second", kind="fed_press", received=NOW + MINUTE)
    env.run(at=NOW + MINUTE)
    assert finals(env.conn)[second]["status"] == "scored"
    assert state(env.conn, "haiku_state") == ON and state(env.conn, "cap_until") is None


def test_with_sonnet_disabled_every_official_item_gets_a_sonnet_skipped_off_row_and_no_sonnet_call(make_env):
    config = dataclasses.replace(CONFIG, llm=dataclasses.replace(
        CONFIG.llm, sonnet=dataclasses.replace(CONFIG.llm.sonnet, enabled=False)))
    env = make_env(config=config)
    official = add(env.conn, "Official item", kind="8k")
    env.run()
    assert finals(env.conn)[official]["status"] == "scored"
    assert finals(env.conn, SONNET)[official]["status"] == "skipped_off"
    assert env.client.headlines(SONNET_MODEL) == [] and env.scoring.words() == (ON, OFF)


# Admission, the size guard, timing and alerts


def test_items_past_their_queue_age_or_vendor_guard_get_their_final_rows_at_once(make_env):
    env = make_env()
    old = add(env.conn, "Item old", kind="8k", received=NOW - 300 * SECOND)
    young = add(env.conn, "Item young", kind="8k", received=NOW - 300 * SECOND + 1)
    stale = add(env.conn, "Item stale", kind="8k", vendor_ts=iso(NOW - 24 * MS_PER_HOUR - 1))
    env.run()
    for scorer in (HAIKU, SONNET):
        rows = finals(env.conn, scorer)
        assert [rows[old]["status"], rows[young]["status"], rows[stale]["status"]] == [
            "expired", "scored", "stale_vendor"]
        assert rows[old]["attempts"] == rows[stale]["attempts"] == 0
    assert len(env.client.requests) == 2


def test_a_batch_above_the_size_guard_splits_without_a_call_and_a_single_item_above_it_fails(make_env, caplog):
    env = make_env(counted=lambda items: 30_000 if "HUGE" in items[0]["headline"] else 1500 + 1000 * len(items))
    seqs = [add(env.conn, f"Item {n}") for n in range(20)]
    huge = add(env.conn, "Item HUGE", kind="8k")
    with caplog.at_level(logging.WARNING):
        env.run()
    assert [len(request) for request in env.client.headlines(HAIKU_MODEL)] == [10, 10]
    assert all(finals(env.conn)[seq]["status"] == "scored" for seq in seqs)
    huge_row = finals(env.conn)[huge]
    assert (huge_row["status"], huge_row["attempts"], huge_row["call_id"]) == ("failed", 0, None)
    assert "above the guard of 20000" in caplog.text and len(ledger_rows(env)) == 2


@pytest.mark.parametrize("scorer, kind, status, route", [
    (HAIKU, "media", "backlog", "llm"), (HAIKU, "media", "seed", "llm"), (HAIKU, "media", "new", "finbert"),
    (SONNET, "media", "new", "llm"), (SONNET, "8k", "backlog", "llm")])
def test_an_item_that_is_not_live_and_routed_to_the_model_is_refused_just_before_its_call(
        make_env, caplog, scorer, kind, status, route):
    env = make_env()
    seq = add(env.conn, "A refused item", kind=kind, status=status, route=route)
    row = env.conn.execute("SELECT * FROM news_items WHERE seq = ?", (seq,)).fetchone()
    model = env.scoring.haiku if scorer == HAIKU else env.scoring.sonnet
    with caplog.at_level(logging.ERROR):
        asyncio.run(model.process(Batch(SINGLE, (Pending.from_row(row),)), asyncio.Event()))
    assert env.client.requests == [] and ledger_rows(env) == [] and calls(env.conn, scorer) == []
    assert finals(env.conn, scorer)[seq]["status"] == "failed" and "not live items" in caplog.text


def test_without_an_exact_count_the_reservation_rests_on_the_byte_bound(make_env):
    env = make_env(counted=None)
    seq = add(env.conn, "An official item", kind="fed_speech")
    env.run()
    [call] = calls(env.conn)
    bound = byte_bound(env.client.requests[0], CONFIG.llm.caps.reserve_overhead_tokens)
    assert (call["counted_tokens"], call["count_basis"]) == (bound, "bytes")
    assert call["reserved_usd"] == PRICES.reserve(HAIKU_MODEL, bound, 4096)
    assert finals(env.conn)[seq]["status"] == "scored"


def test_a_count_below_the_billed_input_by_up_to_the_margin_never_passes_the_cap(make_env, caplog):
    def rule(model, items):
        if model == HAIKU_MODEL:      # the dearest Haiku answer writes every input token to the cache
            return answer(model, items, usage=Usage(0, 4096, 1800 + MARGIN, 0))
        return answer(model, items, usage=Usage(1800 + MARGIN, SONNET_MAX))

    haiku = PRICES.reserve(HAIKU_MODEL, 1800 + MARGIN, 4096)
    sonnet = PRICES.reserve(SONNET_MODEL, 1800 + MARGIN, SONNET_MAX)
    env = make_env(rule=rule, monthly_usd=haiku + sonnet)
    seq = add(env.conn, "An official item", kind="fed_monetary")
    with caplog.at_level(logging.WARNING):
        env.run(at=NOW + MINUTE)
    assert finals(env.conn)[seq]["status"] == finals(env.conn, SONNET)[seq]["status"] == "scored"
    booked = ledger_rows(env)
    assert sorted(row["cost_usd"] for row in booked) == sorted([haiku, sonnet])
    assert all(row["cost_usd"] == row["reserved_usd"] for row in booked)
    assert env.ledger.spend(NOW + MINUTE).month_usd <= env.monthly_usd and events(env.conn, "scorer_alert") == []


def test_scored_at_never_goes_backwards_so_seq_order_is_scored_at_order(make_env):
    env = make_env()
    for n in range(30):
        add(env.conn, f"Item {n} SILENT" if n % 7 == 0 else f"Item {n}", kind="8k" if n % 5 == 0 else "media")
    env.run(at=NOW + MINUTE)
    env.run(at=NOW + 2 * MINUTE)
    stamps = [row[0] for row in env.conn.execute("SELECT scored_at FROM item_scores ORDER BY seq")]
    assert len(stamps) == 30 + 6 and stamps == sorted(stamps)


def test_thinking_blocks_sonnet_cache_tokens_and_a_cost_above_its_reservation_raise_alerts(make_env, caplog):
    def rule(model, items):
        if model == SONNET_MODEL:
            return answer(model, items, usage=Usage(1800, 60, 0, 1200))
        return answer(model, items, thinking_blocks=1, usage=Usage(1800, 60_000))

    env = make_env(rule=rule)
    add(env.conn, "An official item", kind="8k")
    with caplog.at_level(logging.WARNING):
        env.run()
    alerts = events(env.conn, "scorer_alert")
    assert f"{HAIKU} answered with 1 thinking blocks and 0 thinking tokens while its thinking is disabled" in alerts
    assert f"{HAIKU} call cost more than its reservation" in alerts
    assert f"{SONNET} reported cache tokens, whose prices are not in the research" in alerts
    assert all(alert in caplog.text for alert in alerts) and env.scoring.words() == (ON, ON)


@pytest.mark.parametrize("word", ["between_tools", "adaptive"])
def test_sonnet_thinking_is_reserved_and_booked_as_output_and_its_blocks_raise_no_alert(make_env, word):
    config = dataclasses.replace(CONFIG, llm=dataclasses.replace(CONFIG.llm, sonnet=dataclasses.replace(
        CONFIG.llm.sonnet, thinking=word, thinking_tokens=4096)))
    usage = Usage(1800, 900, thinking_tokens=800)

    def rule(model, items):
        return answer(model, items, thinking_blocks=1, usage=usage) if model == SONNET_MODEL else answer(model, items)

    env = make_env(rule=rule, config=config)
    seq = add(env.conn, "An official item", kind="fed_speech")
    env.run()
    [request] = [r for r in env.client.requests if r["model"] == SONNET_MODEL]
    assert request["thinking"] == {"type": word} and request["max_tokens"] == 1024 + 4096
    [call] = calls(env.conn, SONNET)
    assert (call["thinking"], call["max_tokens"], call["thinking_blocks"], call["output_tokens"]) == (
        word, 1024 + 4096, 1, 900)
    assert call["reserved_usd"] == PRICES.reserve(SONNET_MODEL, 1800 + MARGIN, 1024 + 4096)
    assert call["cost_usd"] == PRICES.cost(SONNET_MODEL, Usage(1800, 900)) == (1800 * 2 + 900 * 10) / 1e6
    assert finals(env.conn, SONNET)[seq]["status"] == "scored" and events(env.conn, "scorer_alert") == []


def test_thinking_from_a_sonnet_whose_thinking_is_disabled_raises_an_alert(make_env):
    config = dataclasses.replace(CONFIG, llm=dataclasses.replace(CONFIG.llm, sonnet=dataclasses.replace(
        CONFIG.llm.sonnet, thinking="disabled", thinking_tokens=0)))

    def rule(model, items):
        if model == SONNET_MODEL:
            return answer(model, items, usage=Usage(1800, 300, thinking_tokens=200))
        return answer(model, items)

    env = make_env(rule=rule, config=config)
    add(env.conn, "An official item", kind="fed_speech")
    env.run()
    [request] = [r for r in env.client.requests if r["model"] == SONNET_MODEL]
    assert request["thinking"] == {"type": "disabled"} and request["max_tokens"] == 1024
    assert events(env.conn, "scorer_alert") == [
        f"{SONNET} answered with 0 thinking blocks and 200 thinking tokens while its thinking is disabled"]


def test_a_pause_survives_a_restart_and_off_until_restart_does_not(make_env):
    env = make_env(rule=lambda model, items: failure(RATE_LIMITED, 429) if model == HAIKU_MODEL
                   else failure(BAD_REQUEST, 400))
    add(env.conn, "Official item", kind="8k")
    env.run()
    assert env.scoring.words() == (PAUSED, OFF)
    env.restart()
    assert env.scoring.words() == (PAUSED, ON) and env.scoring.words(NOW + 15 * MINUTE) == (ON, ON)


# The daily probe


def test_the_daily_probe_goes_to_haiku_alone_once_a_day_at_06_00_utc_as_one_call(make_env):
    env = make_env()
    six = utc(2026, 10, 14, 6)
    env.run(at=six - 1)
    assert env.client.requests == []
    env.run(at=six)
    env.run(at=six + 5 * MS_PER_HOUR)
    [probe] = [call for call in calls(env.conn) if call["kind"] == "probe"]
    assert (probe["n_items"], probe["model"], probe["started_at"]) == (20, HAIKU_MODEL, iso(six))
    assert call_items(env.conn, probe["call_id"]) == [] and calls(env.conn, SONNET) == []
    rows = env.conn.execute("SELECT run_at, probe_id, call_id FROM llm_probes ORDER BY probe_id").fetchall()
    assert [tuple(row) for row in rows] == [(iso(six), f"p{n:02d}", probe["call_id"]) for n in range(1, 21)]
    env.run(at=six + 24 * MS_PER_HOUR)
    assert len([call for call in calls(env.conn) if call["kind"] == "probe"]) == 2


def test_a_failed_or_lost_probe_is_not_sent_again_the_same_day(make_env):
    env = make_env(rule=lambda model, items: failure(SERVER, 503))
    six = utc(2026, 10, 14, 6)
    env.run(at=six)
    env.run(at=six + MINUTE)
    assert len(env.client.requests) == 1 and env.conn.execute("SELECT count(*) FROM llm_probes").fetchone()[0] == 0
    env.ledger.reserve(HAIKU_MODEL, make_ref(HAIKU, ()), 20, 0.003, six + 2 * MINUTE)
    env.restart()
    lost = calls(env.conn)[-1]
    assert (lost["status"], lost["kind"], lost["n_items"]) == ("lost", "probe", 20)
    env.run(at=six + 3 * MINUTE)
    assert env.client.requests == []


# Words for the heartbeat and refusal to start


def test_decide_never_splits_one_item_and_maps_each_error_to_a_state():
    one, two = ("i01",), ("i01", "i02")
    assert decide(result(REFUSAL, HAIKU_MODEL, ""), one, MINUTE).finals == {0: ("refused", ())}
    assert decide(result(REFUSAL, HAIKU_MODEL, ""), two, MINUTE).split
    for status, word in ((MODEL_MISMATCH, OFF), (BAD_REQUEST, OFF), (AUTH, OFF), (RATE_LIMITED, PAUSED),
                         (SPEND_LIMIT, CAPPED), (SERVER, None), (TIMEOUT, None), (CONNECTION, None)):
        decision = decide(failure(status), two, MINUTE)
        assert decision.state == word and decision.finals == {0: ("failed", ()), 1: ("failed", ())}
        assert not decision.split


def test_scoring_refuses_to_start_unless_the_config_pins_the_prompt_and_schema(tmp_path):
    conn = db.open_writer(tmp_path / "news.db")
    with Ledger(tmp_path / "l.db", tmp_path / "l.lock", 20.0, process_name(tmp_path)) as ledger:
        with pytest.raises(RequestError, match="not pinned"):
            Scoring(dataclasses.replace(BASE, llm=dataclasses.replace(BASE.llm, prompt_sha256="")), conn,
                    FakeClock(NOW), FakeClient(), ledger)
    conn.close()
