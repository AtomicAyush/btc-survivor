"""The score store of PLAN_V2 sections 8.5 and 9.1 on a temporary news.db. A second final row for one item and scorer
is refused by the store's own read and again by the unique key, a refused write leaves nothing behind, and an
llm_calls row is written once, with its items, the finals it decided, any probe rows and scorer_state values, in one
transaction, because the update trigger refuses any later change."""

import sqlite3
import zlib

import pytest

from btc_survivor.clock import iso
from btc_survivor.v2.news import db
from btc_survivor.v2.news.llm_validate import Impact
from btc_survivor.v2.news.llm_validate import CONFIDENCES, DIRECTIONS, HORIZONS, SURPRISES
from btc_survivor.v2.news.score_store import (
    EXPIRED, FAILED, FINAL, HORIZON_WEIGHTS, IRRELEVANT, MALFORMED, SCORED, SKIPPED_CAP, SKIPPED_OFF, Final,
    ScoreStoreError, call_recorded, has_final, impact_row, strength, write_call, write_finals)

NOW = 1_792_000_000_000
HAIKU, SONNET = "haiku_v1", "sonnet_v1"
UP = Impact(ticker="AAPL", event_type="earnings", is_new=True, direction=1, surprise=1, horizon="days", confidence=4)
DOWN = Impact(ticker="MACRO", event_type="macro_data", is_new=True, direction=-1, surprise=2, horizon="hours",
              confidence=5)


@pytest.fixture
def conn(tmp_path):
    connection = db.open_writer(tmp_path / "news.db")
    yield connection
    connection.close()


_vendor_ids = iter(range(1, 10**6))


def add_item(conn):
    row = {"source": "finnhub", "channel": "finnhub_general", "vendor_id": str(next(_vendor_ids)), "kind": "media",
           "received_at": iso(NOW), "detected_at": None, "vendor_ts": None, "title": "A fictional headline",
           "summary": None, "url": None, "publisher": None, "tickers": "AAPL", "relevance": "title",
           "status": "new", "filter_reason": None, "title_hash": "0" * 16, "items_8k": None, "route": "llm"}
    return db.insert(conn, "news_items", row)


def call_row(scorer=HAIKU, ledger_id=1, **changes):
    row = {"root_call_id": None, "ledger_id": ledger_id, "kind": "batch", "scorer": scorer, "started_at": iso(NOW),
           "finished_at": iso(NOW + 900), "status": "ok", "n_items": 3, "model": "claude-haiku-5-5",
           "sdk_version": "1.8.0", "prompt_sha256": "a" * 64, "schema_sha256": "b" * 64, "request_sha256": "c" * 64,
           "effort": "low", "thinking": "disabled", "max_tokens": 4096, "reserved_usd": 0.003, "cost_usd": 0.0004,
           "cost_basis": "usage", "price_table": "{}", "response_z": zlib.compress(b"{}")}
    return {**row, **changes}


def rows(conn, table):
    return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]


def test_the_final_statuses_are_those_of_section_9_1():
    assert FINAL == ("scored", "irrelevant", "refused", "malformed", "failed", "skipped_cap", "skipped_off",
                     "expired", "stale_vendor", "untagged")


def test_a_second_final_row_for_one_item_and_scorer_is_refused(conn):
    seq = add_item(conn)
    write_finals(conn, HAIKU, [Final(seq, "llm", EXPIRED)], NOW)
    for second in (Final(seq, "llm", SKIPPED_OFF), Final(seq, "llm", SCORED, 1, impacts=(UP,))):
        with pytest.raises(ScoreStoreError, match="already has its final haiku_v1 row"):
            write_finals(conn, HAIKU, [second], NOW + 1)
    assert [(r["item_seq"], r["scorer"], r["status"]) for r in rows(conn, "item_scores")] == [(seq, HAIKU, EXPIRED)]
    assert rows(conn, "score_impacts") == []
    write_finals(conn, SONNET, [Final(seq, "llm", SKIPPED_OFF)], NOW + 2)
    assert has_final(conn, seq, HAIKU) and has_final(conn, seq, SONNET)


def test_the_unique_key_refuses_a_second_row_even_without_the_store(conn):
    seq = add_item(conn)
    write_finals(conn, HAIKU, [Final(seq, "llm", FAILED, 1)], NOW)
    with pytest.raises(sqlite3.IntegrityError):
        db.insert(conn, "item_scores", {"item_seq": seq, "scorer": HAIKU, "route": "llm", "status": SCORED,
                                        "call_id": None, "attempts": 1, "scored_at": iso(NOW)})


def test_a_call_deciding_an_item_that_already_has_its_final_row_leaves_nothing_behind(conn):
    first, second = add_item(conn), add_item(conn)
    write_finals(conn, HAIKU, [Final(first, "llm", EXPIRED)], NOW)
    finals = [Final(second, "llm", SCORED, 1, impacts=(UP,)), Final(first, "llm", SCORED, 1, impacts=(DOWN,))]
    with pytest.raises(ScoreStoreError):
        write_call(conn, call_row(), [second, first], finals, NOW, state={"haiku_call": ""})
    assert rows(conn, "llm_calls") == [] and rows(conn, "llm_call_items") == []
    assert [r["item_seq"] for r in rows(conn, "item_scores")] == [first]
    assert rows(conn, "score_impacts") == []
    assert db.get_value(conn, "scorer_state", "haiku_call") is None


def test_a_set_of_finals_holding_one_item_twice_is_refused_whole(conn):
    first, second = add_item(conn), add_item(conn)
    with pytest.raises(ScoreStoreError):
        write_finals(conn, HAIKU, [Final(first, "llm", SKIPPED_CAP), Final(second, "llm", SKIPPED_CAP),
                                   Final(first, "llm", SKIPPED_CAP)], NOW)
    assert rows(conn, "item_scores") == []


@pytest.mark.parametrize("final, reason", [
    (Final(1, "llm", SCORED, 1), "needs at least one impact"),
    (Final(1, "llm", IRRELEVANT, 1, impacts=(UP,)), "carries no impacts"),
    (Final(1, "llm", MALFORMED, 1, impacts=(UP,)), "carries no impacts"),
    (Final(1, "llm", "pending"), "not a final status"),
    (Final(1, "llm", FAILED, -1), "never negative"),
])
def test_rows_the_rules_forbid_are_refused(conn, final, reason):
    add_item(conn)
    with pytest.raises(ScoreStoreError, match=reason):
        write_finals(conn, HAIKU, [final], NOW)
    assert rows(conn, "item_scores") == []


def test_a_model_strength_folds_horizon_and_surprise_into_direction_times_confidence_over_five(conn):
    # direction * confidence / 5 * H[horizon] * (2 + surprise) / 5, with H = minutes 0.25, hours 1 and days 0.5.
    assert HORIZON_WEIGHTS == {"minutes": 0.25, "hours": 1.0, "days": 0.5} and set(HORIZON_WEIGHTS) == set(HORIZONS)
    assert strength(1, 5, "hours", 3) == 1.0 and strength(-1, 5, "hours", 3) == -1.0
    assert strength(1, 5, "hours", 0) == 0.4                       # a routine item at full confidence
    assert strength(1, 4, "days", 1) == 0.24 and strength(-1, 5, "hours", 2) == -0.8
    assert strength(1, 3, "minutes", 1) == 0.09 and strength(-1, 1, "minutes", 0) == -0.02
    assert strength(0, 5, "hours", 3) == 0.0
    assert impact_row(7, HAIKU, DOWN) == {"item_seq": 7, "scorer": HAIKU, "ticker": "MACRO", "strength": -0.8,
                                         "direction": -1, "surprise": 2, "event_type": "macro_data",
                                         "confidence": 5, "horizon": "hours", "is_new": 1}


def test_every_strength_lies_between_minus_one_and_one_with_the_sign_of_its_direction():
    for d in DIRECTIONS:
        for c in CONFIDENCES:
            for h in HORIZONS:
                for s in SURPRISES:
                    value = strength(d, c, h, s)
                    assert -1.0 <= value <= 1.0 and (value > 0) - (value < 0) == d
                    assert value == pytest.approx(d * (c / 5) * HORIZON_WEIGHTS[h] * (2 + s) / 5, abs=1e-15)


def test_a_call_is_written_once_with_its_items_finals_probe_rows_and_state_in_one_transaction(conn):
    seqs = [add_item(conn) for _ in range(3)]
    finals = [Final(seqs[0], "llm", SCORED, 1, impacts=(UP, DOWN)), Final(seqs[2], "llm", IRRELEVANT, 1)]
    probe = {"run_at": iso(NOW), "probe_id": "p01", "call_id": 0, "ticker": "AAPL", "event_type": "earnings",
             "is_new": 1, "direction": 1, "surprise": 1, "horizon": "days", "confidence": 4}
    call_id = write_call(conn, call_row(), seqs, finals, NOW + 900, [probe], state={"haiku_call": ""})
    [call] = rows(conn, "llm_calls")
    assert call["call_id"] == call_id and call["status"] == "ok" and call["thinking_blocks"] == 0
    assert [(r["item_seq"], r["position"]) for r in rows(conn, "llm_call_items")] == [
        (seqs[0], "i01"), (seqs[1], "i02"), (seqs[2], "i03")]
    scores = {r["item_seq"]: r for r in rows(conn, "item_scores")}
    assert set(scores) == {seqs[0], seqs[2]}
    assert {r["call_id"] for r in scores.values()} == {call_id}
    assert {r["scored_at"] for r in scores.values()} == {iso(NOW + 900)}
    assert sorted((r["ticker"], r["strength"]) for r in rows(conn, "score_impacts")) == [("AAPL", 0.24),
                                                                                          ("MACRO", -0.8)]
    assert [r["call_id"] for r in rows(conn, "llm_probes")] == [call_id]
    assert db.get_value(conn, "scorer_state", "haiku_call") == ""
    assert call_recorded(conn, 1) and not call_recorded(conn, 2)


def test_the_database_gives_the_call_id_and_the_trigger_refuses_any_change(conn):
    with pytest.raises(ScoreStoreError, match="given by the database"):
        write_call(conn, {**call_row(), "call_id": 5}, [], [], NOW)
    write_call(conn, call_row(), [], [], NOW)
    with pytest.raises(sqlite3.DatabaseError, match="never updated"):
        conn.execute("UPDATE llm_calls SET status = 'lost'")
    assert [r["status"] for r in rows(conn, "llm_calls")] == ["ok"]
