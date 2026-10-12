"""Validation of model answers against schema_v1.json, PLAN_V2 section 8.4, plus the probe rows and agreement of
section 8.8 that are built from validated answers. Every answer here is hand written and fictional."""

import json
import sqlite3

import pytest

from btc_survivor.v2.news import db, llm_validate, probes
from btc_survivor.v2.news.llm_request import load_schema, position_id
from btc_survivor.v2.news.llm_validate import (
    CONFIDENCES, DIRECTIONS, DUPLICATE_ID, DUPLICATE_TICKER, ENVELOPE, EVENT_TYPES, HORIZONS, INVALID, MAX_IMPACTS,
    MISSING, SURPRISES, TICKERS, TOO_MANY_IMPACTS, UNPARSABLE, validate,
)

IDS = ("i01", "i02", "i03")


def impact(**changes):
    base = {"ticker": "AAPL", "event_type": "earnings", "is_new": True, "direction": 1, "surprise": 1,
            "horizon": "days", "confidence": 4}
    base.update(changes)
    return base


def entry(id_, *impacts):
    return {"id": id_, "impacts": list(impacts)}


def answer(*entries, **extra):
    return json.dumps({"items": list(entries), **extra})


def good():
    return [entry("i01", impact()), entry("i02", impact(ticker="MACRO", event_type="macro_data", direction=-1)),
            entry("i03")]


def test_a_valid_answer_keeps_every_item_including_an_empty_one():
    result = validate(answer(*good()), IDS)
    assert list(result.scored) == list(IDS) and not result.failed and result.whole is None
    assert result.scored["i03"] == ()
    first = result.scored["i01"][0]
    assert (first.ticker, first.event_type, first.is_new, first.direction, first.surprise, first.horizon,
            first.confidence) == ("AAPL", "earnings", True, 1, 1, "days", 4)
    assert result.unknown_ids == () and result.unattributed == 0


def test_answer_order_does_not_matter_and_scored_follows_the_batch():
    result = validate(answer(*reversed(good())), IDS)
    assert list(result.scored) == list(IDS)


@pytest.mark.parametrize("field, bad", [
    ("ticker", "BTC"), ("ticker", "aapl"), ("ticker", "GOOG"), ("ticker", ""),
    ("event_type", "earning"), ("event_type", "Earnings"),
    ("is_new", 1), ("is_new", "true"), ("is_new", None),
    ("direction", 2), ("direction", True), ("direction", 1.0), ("direction", "1"),
    ("surprise", 4), ("surprise", -1), ("surprise", False),
    ("horizon", "months"), ("horizon", "Days"),
    ("confidence", 0), ("confidence", 6), ("confidence", True), ("confidence", 4.0),
])
def test_an_enum_violation_fails_only_its_item(field, bad):
    entries = good()
    entries[1] = entry("i02", impact(**{field: bad}))
    result = validate(answer(*entries), IDS)
    assert result.failed == {"i02": INVALID}
    assert list(result.scored) == ["i01", "i03"] and result.whole is None


@pytest.mark.parametrize("where", ["impact", "item"])
def test_an_extra_field_fails_only_its_item(where):
    entries = good()
    if where == "impact":
        entries[0] = entry("i01", {**impact(), "price_target": 250})
    else:
        entries[0] = {**entry("i01", impact()), "note": "extra"}
    result = validate(answer(*entries), IDS)
    assert result.failed == {"i01": INVALID} and list(result.scored) == ["i02", "i03"]


def test_an_extra_field_on_the_outer_object_fails_the_whole_answer():
    result = validate(answer(*good(), note="extra"), IDS)
    assert result.whole == ENVELOPE and not result.scored and result.failed == dict.fromkeys(IDS, ENVELOPE)


@pytest.mark.parametrize("field", ["ticker", "event_type", "is_new", "direction", "surprise", "horizon",
                                   "confidence"])
def test_a_missing_impact_field_fails_only_its_item(field):
    entries = good()
    broken = impact()
    del broken[field]
    entries[2] = entry("i03", broken)
    result = validate(answer(*entries), IDS)
    assert result.failed == {"i03": INVALID} and list(result.scored) == ["i01", "i02"]


def test_an_item_without_impacts_fails_and_one_without_an_id_is_unattributed():
    entries = good()
    entries[0] = {"id": "i01"}
    entries[1] = {"impacts": []}
    result = validate(answer(*entries), IDS)
    assert result.failed == {"i01": INVALID, "i02": MISSING}
    assert list(result.scored) == ["i03"] and result.unattributed == 1


def test_missing_ids_are_named_and_the_rest_kept():
    result = validate(answer(good()[0]), IDS)
    assert result.failed == {"i02": MISSING, "i03": MISSING} and list(result.scored) == ["i01"]
    assert validate(answer(), IDS).failed == dict.fromkeys(IDS, MISSING)


def test_a_duplicate_id_fails_every_copy_and_the_rest_are_kept():
    entries = [*good(), entry("i02", impact(ticker="MSFT"))]
    result = validate(answer(*entries), IDS)
    assert result.failed == {"i02": DUPLICATE_ID} and list(result.scored) == ["i01", "i03"]


def test_unknown_ids_are_dropped_and_named():
    entries = [*good(), entry("i04", impact()), entry("I01", impact()), entry("1234567", impact())]
    result = validate(answer(*entries), IDS)
    assert result.unknown_ids == ("1234567", "I01", "i04")
    assert list(result.scored) == list(IDS) and not result.failed


def test_non_string_ids_and_non_object_entries_are_unattributed():
    entries = [*good(), {"id": 1, "impacts": []}, "i01", None]
    result = validate(answer(*entries), IDS)
    assert result.unattributed == 3 and list(result.scored) == list(IDS)


def test_duplicate_tickers_fail_their_item():
    entries = good()
    entries[0] = entry("i01", impact(), impact(direction=-1))
    result = validate(answer(*entries), IDS)
    assert result.failed == {"i01": DUPLICATE_TICKER} and list(result.scored) == ["i02", "i03"]


def test_more_than_10_impacts_fail_their_item():
    entries = good()
    entries[0] = entry("i01", *[impact(ticker=TICKERS[i % len(TICKERS)]) for i in range(MAX_IMPACTS + 1)])
    result = validate(answer(*entries), IDS)
    assert result.failed == {"i01": TOO_MANY_IMPACTS} and list(result.scored) == ["i02", "i03"]
    every = entry("i01", *[impact(ticker=t) for t in TICKERS])
    assert validate(answer(every, *good()[1:]), IDS).scored["i01"][-1].ticker == "MACRO"


@pytest.mark.parametrize("text, reason", [
    ("", UNPARSABLE), ("{\"items\": [", UNPARSABLE), ("not json", UNPARSABLE),
    ("[]", ENVELOPE), ("{}", ENVELOPE), ("{\"items\": {}}", ENVELOPE), ("{\"items\": null}", ENVELOPE),
    ("\"items\"", ENVELOPE),
])
def test_an_unreadable_answer_fails_as_a_whole(text, reason):
    result = validate(text, IDS)
    assert result.whole == reason and not result.scored and result.failed == dict.fromkeys(IDS, reason)


def test_the_validator_enums_match_the_schema_file():
    impact_schema = load_schema()["properties"]["items"]["items"]["properties"]["impacts"]["items"]
    props = impact_schema["properties"]
    assert props["ticker"]["enum"] == list(TICKERS)
    assert props["event_type"]["enum"] == list(EVENT_TYPES)
    assert props["direction"] == {"type": "integer", "enum": list(DIRECTIONS)}
    assert props["surprise"] == {"type": "integer", "enum": list(SURPRISES)}
    assert props["horizon"]["enum"] == list(HORIZONS)
    assert props["confidence"] == {"type": "integer", "enum": list(CONFIDENCES)}
    assert props["is_new"] == {"type": "boolean"}
    assert impact_schema["required"] == list(props) == list(llm_validate.Impact.model_fields)
    item_schema = load_schema()["properties"]["items"]["items"]
    assert item_schema["required"] == list(llm_validate.Item.model_fields)
    assert "BTC" not in TICKERS


def test_reasons_never_repeat_a_value_from_the_answer():
    entries = good()
    entries[0] = entry("i01", impact(ticker="CANARY-VALUE"))
    result = validate(answer(*entries, extra="CANARY-VALUE"), IDS)
    assert "CANARY-VALUE" not in repr(result.failed) + repr(result.whole)


# Probe rows and agreement


PROBES = probes.load_probes()
IDS20 = tuple(position_id(i) for i in range(20))


def key_answer():
    """An answer giving each probe exactly its key, with fixed surprise, horizon and confidence."""
    return answer(*[entry(position_id(i), *[impact(ticker=k.ticker, event_type=k.event_type, is_new=k.is_new,
                                                   direction=k.direction) for k in p.key])
                    for i, p in enumerate(PROBES)])


def test_the_answer_key_validates_against_the_schema():
    result = validate(key_answer(), IDS20)
    assert len(result.scored) == 20 and not result.failed


def test_probe_rows_hold_one_row_per_impact_and_one_for_an_empty_answer():
    result = validate(key_answer(), IDS20)
    rows = probes.probe_rows("2026-10-13T06:00:00.123Z", 41, PROBES, result)
    assert len(rows) == sum(max(1, len(p.key)) for p in PROBES)
    empty = [r for r in rows if r["ticker"] == probes.NO_IMPACT]
    assert [r["probe_id"] for r in empty] == [p.probe_id for p in PROBES if not p.key]
    assert all(r[f] is None for r in empty for f in probes.FIELDS)
    assert {(r["run_at"], r["call_id"]) for r in rows} == {("2026-10-13T06:00:00.123Z", 41)}
    tariff = [r for r in rows if r["probe_id"] == "p16"]
    assert [(r["ticker"], r["direction"], r["is_new"]) for r in tariff] == [("MACRO", -1, 1), ("AAPL", -1, 1)]


def test_probe_rows_insert_into_the_llm_probes_table():
    conn = sqlite3.connect(":memory:")
    conn.executescript(db.TABLES_SQL)
    rows = probes.probe_rows("2026-10-13T06:00:00.123Z", 41, PROBES, validate(key_answer(), IDS20))
    columns = [name for _, name, *_ in conn.execute("PRAGMA table_info(llm_probes)")]
    assert all(list(r) == columns for r in rows)
    conn.executemany(f"INSERT INTO llm_probes VALUES ({', '.join('?' * len(columns))})",
                     [tuple(r.values()) for r in rows])
    assert conn.execute("SELECT count(*) FROM llm_probes").fetchone()[0] == len(rows)


def test_a_probe_without_a_valid_answer_gets_no_row():
    data = json.loads(key_answer())
    data["items"][0]["impacts"][0]["ticker"] = "BTC"
    del data["items"][1]
    rows = probes.probe_rows("2026-10-13T06:00:00.123Z", 41, PROBES, validate(json.dumps(data), IDS20))
    assert not {r["probe_id"] for r in rows} & {"p01", "p02"}


def test_agreement_with_the_key_is_full_for_the_key_answer():
    rows = probes.probe_rows("2026-10-13T06:00:00.123Z", 41, PROBES, validate(key_answer(), IDS20))
    result = probes.agreement(probes.key_rows(PROBES), rows, probes.KEY_FIELDS)
    assert result == {"probes": 20, "pairs": 21, "tickers": 1.0, "event_type": 1.0, "is_new": 1.0, "direction": 1.0}


def test_agreement_counts_changed_fields_and_pairs_only_one_run_named():
    day1 = probes.probe_rows("2026-10-13T06:00:00.123Z", 41, PROBES, validate(key_answer(), IDS20))
    data = json.loads(key_answer())
    data["items"][0]["impacts"][0]["direction"] = -1
    data["items"][2]["impacts"].append(impact(ticker="QQQ"))
    day2 = probes.probe_rows("2026-10-14T06:00:00.456Z", 57, PROBES, validate(json.dumps(data), IDS20))
    result = probes.agreement(day1, day2)
    assert result["probes"] == 20 and result["pairs"] == 22
    assert result["tickers"] == pytest.approx(21 / 22)
    assert result["direction"] == pytest.approx(20 / 22)
    assert result["horizon"] == pytest.approx(21 / 22)


def test_agreement_skips_probes_one_run_did_not_answer():
    day1 = probes.probe_rows("2026-10-13T06:00:00.123Z", 41, PROBES, validate(key_answer(), IDS20))
    day2 = [r for r in day1 if r["probe_id"] != "p05"]
    result = probes.agreement(day1, day2)
    assert result["probes"] == 19 and result["direction"] == 1.0
    assert probes.agreement(day1, []) == {"probes": 0, "pairs": 0, "tickers": None,
                                          **dict.fromkeys(probes.FIELDS)}
