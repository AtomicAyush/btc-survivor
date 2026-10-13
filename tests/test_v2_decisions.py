"""The decisions explorer of the private dashboard on seeded run databases, its filters, its paging with a cursor that
new decisions never shift, the counts of every matching decision and the full detail of one decision."""

import pytest

from btc_survivor.v2.decisions import (ACTIONS, MAX_PAGE_ROWS, MAX_TEXT, PAGE_ROWS, FilterError, Filters,
                                       parse_filters, position_after)
from btc_survivor.v2 import decisions
from btc_survivor.v2.config import load_run_pair
from btc_survivor.v2.stocks import records, store
from btc_survivor.v2.stocks.store import StockStore
import v2_runkit as kit

NAMES = ("s1_prices", "s2_news")
# Every seeded decision as (run, id), newest first by time, then the news run before the prices run, then id.
ALL = [("s2_news", 8), ("s2_news", 7), ("s1_prices", 7), ("s2_news", 6), ("s1_prices", 6), ("s2_news", 5),
       ("s1_prices", 5), ("s2_news", 4), ("s1_prices", 4), ("s2_news", 3), ("s2_news", 2), ("s1_prices", 3),
       ("s1_prices", 2), ("s2_news", 1), ("s1_prices", 1)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return kit.seeded(tmp_path_factory.mktemp("decisions"))


@pytest.fixture
def conns(runs):
    opened = [store.open_reader(runs.s1_db), store.open_reader(runs.s2_db)]
    yield opened
    for conn in opened:
        conn.close()


def query(**values):
    """A parsed query string, each value a one item list as parse_qs gives it."""
    return {name.rstrip("_"): [str(value)] for name, value in values.items()}


def filters(**values):
    return parse_filters(query(**values), NAMES, kit.SYMBOLS)


def keys(page):
    return [(row["run"], row["id"]) for row in page["rows"]]


def everything(conns, **values):
    """Every page of a query walked with its cursor, as (run, id) keys."""
    found, cursor = [], None
    while True:
        page = decisions.page(conns, NAMES, filters(**values, **({"before": cursor} if cursor else {})))
        found += keys(page)
        cursor = page["next"]
        if cursor is None:
            return found


# Filters


def test_no_filter_takes_both_runs_and_the_default_page():
    assert filters() == Filters(runs=(0, 1))
    assert parse_filters({"run": [""], "reason": ["  "]}, NAMES, kit.SYMBOLS) == Filters(runs=(0, 1))
    assert PAGE_ROWS == 50 and ACTIONS == ("none", "hold", "open_long", "open_short", "close", "flip")


def test_every_filter_is_parsed():
    f = filters(run="s2_news", symbol="TSLA", slot=6, bot=2, kind="twin", action="flip", forced="no", gate="yes",
                reason="clears", from_="2026-10-21", to="2026-10-22T15:06", limit=7)
    assert f == Filters(runs=(1,), symbol="TSLA", slot=6, bot_id=2, kind="twin", action="flip", forced=False,
                        gate=True, reason="clears", since="2026-10-21T00:00:00.000Z",
                        until="2026-10-22T15:07:00.000Z", limit=7)


def test_an_end_date_includes_its_whole_day_and_an_end_minute_that_minute():
    assert filters(to="2026-10-21").until == "2026-10-22T00:00:00.000Z"
    assert filters(to="2026-10-21T14:30").until == "2026-10-21T14:31:00.000Z"
    assert filters(from_="2026-10-21T14:30").since == "2026-10-21T14:30:00.000Z"


@pytest.mark.parametrize("bad, message", [
    ({"color": ["red"]}, "unknown filter color"),
    ({"run": ["s1_prices", "s2_news"]}, "run may be given only once"),
    ({"run": ["s3"]}, "run must be one of s1_prices, s2_news"),
    ({"symbol": ["aapl"]}, "symbol must be one of AAPL"),
    ({"slot": ["-1"]}, "slot must be a whole number"),
    ({"slot": ["10000"]}, "slot must be a whole number from 0 to 9999"),
    ({"bot": ["1.5"]}, "bot must be a whole number"),
    ({"kind": ["learners"]}, "kind must be one of learner, twin"),
    ({"action": ["buy"]}, "action must be one of none, hold"),
    ({"forced": ["1"]}, "forced must be one of yes, no"),
    ({"gate": ["true"]}, "gate must be one of yes, no"),
    ({"reason": ["x" * (MAX_TEXT + 1)]}, f"reason text is at most {MAX_TEXT} characters"),
    ({"from": ["21 Oct 2026"]}, "from must be a UTC date"),
    ({"to": ["2026-10-21T14:30:00Z"]}, "to must be a UTC date"),
    ({"to": ["2026-02-30"]}, "to is not a real date or time"),
    ({"from": ["2026-10-22"], "to": ["2026-10-21"]}, "from must come before to"),
    ({"limit": [str(MAX_PAGE_ROWS + 1)]}, f"limit must be a whole number from 0 to {MAX_PAGE_ROWS}"),
    ({"before": ["2026-10-21T14:30:00.000Z,2,5"]}, "before must be a cursor"),
    ({"before": ["2026-10-21T14:30:00Z,1,5"]}, "before must be a cursor"),
    ({"before": ["2026-10-21T14:30:00.000Z,1"]}, "before must be a cursor"),
    ({"before": ["nonsense"]}, "before must be a cursor"),
])
def test_a_filter_the_explorer_cannot_use_is_refused_by_name(bad, message):
    with pytest.raises(FilterError, match=message.replace("(", r"\(")):
        parse_filters(bad, NAMES, kit.SYMBOLS)


def test_a_cursor_from_a_page_parses_back():
    f = filters(before="2026-10-21T14:30:00.000Z,1,5")
    assert (f.before.ts, f.before.run, f.before.id) == ("2026-10-21T14:30:00.000Z", 1, 5)
    assert f.before.text() == "2026-10-21T14:30:00.000Z,1,5"


# Pages


def test_a_page_is_newest_first_across_both_runs_with_the_news_run_first_at_one_minute(conns):
    page = decisions.page(conns, NAMES, filters())
    assert keys(page) == ALL and page["next"] is None
    row = page["rows"][0]
    assert {k: row[k] for k in ("ts", "bot_id", "slot", "kind", "symbol", "position_before", "action", "forced",
                                "gate_passed", "prediction_bps", "required_edge_bps", "label_bps")} == {
        "ts": "2026-10-22T19:30:00.000Z", "bot_id": 1, "slot": 0, "kind": "learner", "symbol": "AAPL",
        "position_before": "flat", "action": "open_long", "forced": 1, "gate_passed": 0, "prediction_bps": 0.8,
        "required_edge_bps": 9.0, "label_bps": None}


@pytest.mark.parametrize("limit", [1, 2, 4, 14, 15])
def test_paging_with_the_cursor_visits_every_decision_once(conns, limit):
    assert everything(conns, limit=limit) == ALL


def test_the_last_full_page_has_no_next_cursor(conns):
    first = decisions.page(conns, NAMES, filters(limit=4))
    assert keys(first) == ALL[:4] and first["next"] == "2026-10-22T14:36:00.000Z,1,6"
    last = decisions.page(conns, NAMES, filters(limit=15))
    assert keys(last) == ALL and last["next"] is None


def test_new_decisions_never_shift_an_older_page(tmp_path):
    runs = kit.seeded(tmp_path)
    s1, _ = load_run_pair(runs.s1, runs.s2)
    readers = [store.open_reader(runs.s1_db), store.open_reader(runs.s2_db)]
    try:
        first = decisions.page(readers, NAMES, filters(limit=5))
        writer = StockStore.open(s1.db_path)
        writer.commit_now([kit.decision(1, "2026-10-22T19:30:00.000Z", "AAPL", "none"),
                           kit.decision(2, "2026-10-23T14:00:00.000Z", "TSLA", "none")])
        writer.close()
        second = decisions.page(readers, NAMES, filters(limit=5, before=first["next"]))
        assert keys(first) == ALL[:5] and keys(second) == ALL[5:10]
        assert keys(decisions.page(readers, NAMES, filters(limit=3))) == [
            ("s1_prices", 9), ("s2_news", 8), ("s1_prices", 8)]
        assert decisions.page(readers, NAMES, filters())["summary"]["decisions"] == 17
    finally:
        for conn in readers:
            conn.close()


@pytest.mark.parametrize("values, expected", [
    ({"run": "s1_prices"}, [key for key in ALL if key[0] == "s1_prices"]),
    ({"run": "s2_news", "limit": 2}, [key for key in ALL if key[0] == "s2_news"]),
    ({"symbol": "TSLA"}, [key for key in ALL if key[1] in (5, 6, 7)]),
    ({"slot": 0}, [key for key in ALL if key[1] in (1, 2, 3, 4, 8)]),
    ({"bot": 2}, [key for key in ALL if key[1] in (5, 6, 7)]),
    ({"kind": "twin"}, [("s2_news", 3), ("s1_prices", 3)]),
    ({"kind": "learner"}, [key for key in ALL if key[1] != 3]),
    ({"action": "open_short"}, [key for key in ALL if key[1] in (3, 5, 6)]),
    ({"forced": "yes"}, [key for key in ALL if key[1] in (2, 6, 8)]),
    ({"forced": "no", "gate": "no"}, [key for key in ALL if key[1] in (1, 3, 4)]),
    ({"gate": "yes"}, [key for key in ALL if key[1] in (5, 7)]),
    ({"reason": "WARMUP"}, [("s2_news", 1), ("s1_prices", 1)]),
    ({"reason": "_"}, []),
    ({"reason": "%"}, []),
    ({"reason": "below required edge 9.00"}, [key for key in ALL if key[1] == 4]),
    ({"from": "2026-10-22"}, ALL[:5]),
    ({"to": "2026-10-21"}, ALL[5:]),
    ({"to": "2026-10-21T14:30"}, ALL[9:]),
    ({"from": "2026-10-21T14:30", "to": "2026-10-21T15:00"}, ALL[7:13]),
    ({"run": "s1_prices", "slot": 0, "action": "close"}, [("s1_prices", 4)]),
])
def test_each_filter_keeps_only_its_decisions(conns, values, expected):
    values = {("from_" if name == "from" else name): value for name, value in values.items()}
    assert everything(conns, **values) == expected


def test_the_counts_cover_every_matching_decision_whatever_the_page(conns):
    page = decisions.page(conns, NAMES, filters(limit=2))
    assert page["summary"] == {
        "decisions": 15, "forced": 5, "gate_passed": 4, "labelled": 10, "directional": 10, "right": 8,
        "directional_label_bps": 38.0, "hit_rate": 0.8, "mean_directional_label_bps": 3.8,
        "actions": {"none": 2, "open_long": 3, "open_short": 6, "close": 2, "flip": 2}}
    assert page["by_run"]["s1_prices"]["decisions"] == 7 and page["by_run"]["s2_news"]["forced"] == 3
    assert page["by_run"]["s1_prices"]["hit_rate"] == 0.8
    twins = decisions.page(conns, NAMES, filters(kind="twin", run="s1_prices"))
    assert list(twins["by_run"]) == ["s1_prices"] and twins["summary"]["decisions"] == 1
    none = decisions.page(conns, NAMES, filters(action="hold"))
    assert none["rows"] == [] and none["next"] is None
    assert none["summary"]["decisions"] == 0 and none["summary"]["hit_rate"] is None


def test_a_run_whose_database_could_not_open_is_left_out(conns):
    page = decisions.page([None, conns[1]], NAMES, filters())
    assert [key for key in keys(page)] == [key for key in ALL if key[0] == "s2_news"]
    assert list(page["by_run"]) == ["s2_news"]


# Detail


def test_a_forced_open_shows_its_fill_costs_trade_position_and_matured_label(conns):
    d = decisions.detail(conns[0], "s1_prices", 2)
    assert {k: d[k] for k in ("run", "id", "ts", "symbol", "action", "forced", "gate_passed", "prediction_bps",
                              "required_edge_bps", "cost_estimate_bps", "margin_bps", "ref_price", "edge_gap_bps",
                              "position_before", "position_after")} == {
        "run": "s1_prices", "id": 2, "ts": "2026-10-21T14:30:00.000Z", "symbol": "AAPL", "action": "open_long",
        "forced": True, "gate_passed": False, "prediction_bps": 1.2, "required_edge_bps": 9.0,
        "cost_estimate_bps": 6.5, "margin_bps": 2.5, "ref_price": kit.CANARY_PRICE,
        "edge_gap_bps": pytest.approx(-7.8), "position_before": "flat", "position_after": "long"}
    assert d["reason"] == "forced by the exploration timer, prediction 1.20 bps"
    assert d["bot"] == {"bot_id": 1, "slot": 0, "kind": "learner", "home_symbol": "AAPL", "state": "active",
                        "born_at": kit.LAUNCH, "died_at": None}
    [fill] = d["fills"]
    assert {k: fill[k] for k in ("trade_id", "leg", "side", "status", "reason", "filled_at", "latency_ms",
                                 "print_price", "price", "allowance_bps", "slippage_bps", "fee", "sec_fee",
                                 "taf_fee")} == {
        "trade_id": 1, "leg": "open", "side": "buy", "status": "filled", "reason": "decision",
        "filled_at": "2026-10-21T14:30:01.200Z", "latency_ms": 1200, "print_price": kit.CANARY_PRICE,
        "price": kit.CANARY_FILL, "allowance_bps": 2.0, "slippage_bps": 0.42, "fee": 0.03, "sec_fee": 0.02,
        "taf_fee": 0.01}
    # The report only columns reach the decision detail, which the page shows once they hold values.
    assert {"print_after_due_ms", "news_window"} <= set(fill)
    [trade] = d["trades"]
    assert (trade["id"], trade["side"], trade["closed_at"], trade["pnl_net"], trade["exit_reason"], trade["forced"]) \
        == (1, "long", "2026-10-21T15:00:01.200Z", 0.92, "signal", 1)
    assert d["label"]["label_bps"] == 5.0 and d["label"]["matured"] and d["label"]["direction_right"] is True
    assert d["label"]["evaluation"]["status"] == "matured"  # 14:30 is also an evaluation minute


def test_a_rejected_open_leaves_the_position_and_opens_no_trade(conns):
    d = decisions.detail(conns[0], "s1_prices", 5)
    assert [(f["status"], f["reason"], f["price"], f["fee"]) for f in d["fills"]] == [
        ("rejected", "symbol_not_healthy", None, None)]
    assert d["trades"] == [] and d["position_after"] == "flat" and d["gate_passed"] is True
    assert d["label"]["direction_right"] is True and d["label"]["evaluation"] is None  # 15:06 is not one


def test_a_flip_shows_both_legs_both_trades_and_an_unmatured_label(conns):
    d = decisions.detail(conns[1], "s2_news", 7)
    assert [(f["leg"], f["side"], f["trade_id"]) for f in d["fills"]] == [("close", "buy", 3), ("open", "buy", 4)]
    assert [(t["id"], t["exit_reason"], t["closed_at"] is None) for t in d["trades"]] == [(3, "flip", False),
                                                                                         (4, None, True)]
    assert (d["position_before"], d["position_after"]) == ("short", "long")
    assert d["label"] == {"label_bps": None, "matured": False, "direction_right": None, "evaluation": None}


def test_a_wrong_way_label_a_close_a_pending_open_and_a_twin(conns):
    wrong = decisions.detail(conns[0], "s1_prices", 6)
    assert wrong["label"]["direction_right"] is False and wrong["position_after"] == "short"
    close = decisions.detail(conns[0], "s1_prices", 4)
    assert close["position_after"] == "flat" and close["trades"][0]["id"] == 1
    pending = decisions.detail(conns[1], "s2_news", 8)
    assert pending["fills"] == [] and pending["position_after"] is None
    twin = decisions.detail(conns[0], "s1_prices", 3)
    assert twin["bot"]["kind"] == "twin" and twin["position_after"] == "short" and twin["trades"][0]["id"] == 2
    warmup = decisions.detail(conns[0], "s1_prices", 1)
    assert warmup["fills"] == [] and warmup["position_after"] == "flat"
    assert decisions.detail(conns[0], "s1_prices", 8) is None


@pytest.mark.parametrize("before, action, fills, after", [
    ("flat", "none", [], "flat"),
    ("long", "hold", [], "long"),
    ("flat", "open_long", [], None),
    ("flat", "open_short", [{"status": "filled", "leg": "open", "side": "sell"}], "short"),
    ("long", "close", [{"status": "filled", "leg": "close", "side": "sell"}], "flat"),
    ("long", "flip", [{"status": "filled", "leg": "close", "side": "sell"},
                      {"status": "rejected", "leg": "open", "side": "sell"}], "flat"),
    ("short", "flip", [{"status": "rejected", "leg": "close", "side": "buy"}], "short"),
    ("long", "none", [{"status": "filled", "leg": "close", "side": "sell"}], "flat"),  # a stop at the same minute
])
def test_position_after_follows_the_filled_legs_in_order(before, action, fills, after):
    assert position_after(before, action, fills) == after


def test_the_explorer_reads_through_the_read_only_reader(runs, conns):
    with pytest.raises(Exception, match="readonly"):
        conns[0].execute(records.INSERT_FEED_EVENT, ("2026-10-22T20:00:00.000Z", "x", None, None))
