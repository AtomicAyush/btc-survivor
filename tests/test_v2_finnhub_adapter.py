"""The Finnhub adapter on hand written messages with synthetic prices, and a fake canary token only."""

import json
import logging

import pytest

from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, FeedAdapter, decode
from btc_survivor.types import FeedNotice
from btc_survivor.v2 import redact
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import finnhub
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter, Print, PrintBatch
from helpers import ROOT

CANARY = "CANARYfinnhubTOKEN0042"
SYMBOLS = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml").stock_feed.symbols


@pytest.fixture(autouse=True)
def canary(monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    redact.register(CANARY)


@pytest.fixture
def adapter():
    return FinnhubAdapter(SYMBOLS, CANARY)


def trade(*prints):
    return json.dumps({"type": "trade", "data": list(prints)})


def p(symbol, price, size, t, c=None):
    item = {"s": symbol, "p": price, "v": size, "t": t}
    if c is not None:
        item["c"] = c
    return item


def test_a_multi_symbol_trade_message_becomes_one_batch(adapter):
    raw = trade(p("AAPL", 101.25, 3, 1_000), p("MSFT", 202.5, 10, 1_001, ["1", "12"]), p("QQQ", 303.0, 0.5, 1_002, []))
    events = adapter.parse(raw, 5_000, 7)
    assert events == [PrintBatch(5_000, 7, (Print("AAPL", 101.25, 3.0, 1_000, ()),
                                            Print("MSFT", 202.5, 10.0, 1_001, ("1", "12")),
                                            Print("QQQ", 303.0, 0.5, 1_002, ())))]
    assert isinstance(events[0].prints[0].size, float) and isinstance(events[0].prints[0].vendor_ms, int)
    assert not adapter.dropped and not adapter.unknown


def test_prints_are_sorted_by_vendor_time_then_position(adapter):
    raw = trade(p("NVDA", 1.0, 1, 30), p("AAPL", 2.0, 1, 10), p("TSLA", 3.0, 1, 30), p("META", 4.0, 1, 10),
                p("AMZN", 5.0, 1, 20))
    [batch] = adapter.parse(raw, 99, 1)
    assert [(x.symbol, x.vendor_ms) for x in batch.prints] == [("AAPL", 10), ("META", 10), ("AMZN", 20), ("NVDA", 30),
                                                               ("TSLA", 30)]


def test_null_and_numeric_conditions(adapter):
    [batch] = adapter.parse(trade(p("AAPL", 1.0, 1, 5, [1, "24"]), {"s": "AAPL", "p": 1.0, "v": 1, "t": 6, "c": None}),
                            9, 1)
    assert [x.conditions for x in batch.prints] == [("1", "24"), ()]


def test_bad_prints_are_dropped_and_counted_by_reason(adapter):
    raw = ('{"type":"trade","data":['
           '{"s":"AAPL","p":10.5,"v":2,"t":1},'
           '{"s":"IBM","p":10.5,"v":2,"t":1},'
           '{"s":"AAPL","p":NaN,"v":2,"t":1},'
           '{"s":"AAPL","p":Infinity,"v":2,"t":1},'
           '{"s":"AAPL","p":0,"v":2,"t":1},'
           '{"s":"AAPL","p":-1.5,"v":2,"t":1},'
           '{"s":"AAPL","p":10.5,"v":0,"t":1},'
           '{"s":"AAPL","p":10.5,"v":-3,"t":1},'
           '{"s":"AAPL","p":10.5,"v":Infinity,"t":1},'
           '{"s":"AAPL","p":"10.5","v":2,"t":1},'
           '{"s":"AAPL","p":10.5,"v":2},'
           '{"s":"AAPL","p":true,"v":2,"t":1},'
           '{"s":"AAPL","p":10.5,"v":2,"t":1,"c":"1"},'
           '["AAPL",10.5,2,1]'
           ']}')
    [batch] = adapter.parse(raw, 50, 2)
    assert batch.prints == (Print("AAPL", 10.5, 2.0, 1, ()),)
    assert adapter.dropped == {"unsubscribed": 1, "price": 4, "size": 3, "malformed": 5}


def test_an_integer_too_large_for_a_float_is_malformed_and_never_raises(adapter):
    huge = 10 ** 400
    raw = trade(p("AAPL", huge, 1, 1), p("AAPL", 1.0, huge, 1), p("AAPL", 1.0, 1, huge), p("AAPL", -huge, 1, 1),
                p("AAPL", 2.0, 3, 4))
    [batch] = adapter.parse(raw, 9, 1)
    assert batch.prints == (Print("AAPL", 2.0, 3.0, 4, ()),)
    assert adapter.dropped == {"malformed": 4}


def test_a_trade_message_with_nothing_valid_gives_no_event(adapter):
    assert adapter.parse(trade(p("IBM", 1.0, 1, 1)), 1, 1) == []
    assert adapter.parse('{"type":"trade","data":null}', 1, 1) == []
    assert adapter.parse('{"type":"trade","data":[]}', 1, 1) == []
    assert adapter.dropped == {"unsubscribed": 1, "malformed": 1}


def test_ping_is_a_notice_with_no_reply(adapter):
    assert adapter.parse('{"type":"ping"}', 77, 3) == [FeedNotice(77, 3, "ping")]
    assert adapter.subscribe_messages() == [adapter.subscribe_frame(s) for s in SYMBOLS]


def test_error_is_a_redacted_notice(adapter):
    raw = json.dumps({"type": "error", "msg": f"Invalid token {CANARY} " + "x" * 400})
    [notice] = adapter.parse(raw, 12, 4)
    assert (notice.recv_ms, notice.conn_id, notice.kind) == (12, 4, "error")
    assert notice.detail.startswith(f"Invalid token {redact.MASK} ")
    assert CANARY not in notice.detail and len(notice.detail) == finnhub.ERROR_DETAIL_CHARS


def test_unknown_types_are_counted_and_logged_once_per_type_per_connection(adapter, caplog):
    caplog.set_level(logging.WARNING, logger=finnhub.__name__)
    for raw in ('{"type":"news","data":[]}', '{"type":"news"}', '{"type":"status"}', "not json", "[1, 2]",
                '{"data":[]}', '{"type":"Trade With Spaces 123"}', '{"type":{"nested":1}}'):
        assert adapter.parse(raw, 1, 1) == []
    adapter.on_open(2)
    adapter.parse('{"type":"news"}', 2, 2)
    assert adapter.unknown == {"news": 3, "status": 1, "unreadable": 5}
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["Finnhub message of type news on connection 1, counted from now on",
                        "Finnhub message of type status on connection 1, counted from now on",
                        "Finnhub message of type unreadable on connection 1, counted from now on",
                        "Finnhub message of type news on connection 2, counted from now on"]


def test_eight_subscribe_frames_in_config_order():
    adapter = FinnhubAdapter(SYMBOLS, CANARY)
    assert SYMBOLS == ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
    assert adapter.subscribe_messages() == [f'{{"type":"subscribe","symbol":"{s}"}}' for s in SYMBOLS]


def test_the_url_holds_the_token_but_the_repr_never_does(adapter):
    assert adapter.url == f"wss://ws.finnhub.io?token={CANARY}"
    assert FinnhubAdapter(SYMBOLS, CANARY, "ws://127.0.0.1:9").url == f"ws://127.0.0.1:9?token={CANARY}"
    for text in (repr(adapter), str(adapter), f"{adapter}", f"{adapter!r}", repr([adapter])):
        assert CANARY not in text
    assert repr(adapter) == "FinnhubAdapter(symbols=8, base_url='wss://ws.finnhub.io')"
    assert isinstance(adapter, FeedAdapter)
    assert adapter.max_message_bytes == 4 * 1024 * 1024


@pytest.mark.parametrize("token", ["", "has space", "a&b=c", "tok/en", "tök", "a?b", "x#y"])
def test_a_token_a_url_would_change_is_refused_without_showing_it(token):
    with pytest.raises(ValueError, match="may hold only") as info:
        FinnhubAdapter(SYMBOLS, token)
    assert not token or token not in str(info.value)


def test_no_symbols_is_refused():
    with pytest.raises(ValueError, match="at least one symbol"):
        FinnhubAdapter((), CANARY)


def test_parse_is_pure_and_decode_reuses_it(adapter):
    raw = trade(p("AAPL", 1.5, 2, 3), p("MSFT", 2.5, 4, 1))
    assert adapter.parse(raw, 10, 1) == adapter.parse(raw, 10, 1)
    other = FinnhubAdapter(SYMBOLS, CANARY)
    assert other.parse(raw, 10, 1) == adapter.parse(raw, 10, 1)
    assert decode(adapter, OPEN, "", 5, 6) == [FeedNotice(5, 6, "connected")]
    assert decode(adapter, MESSAGE, raw, 10, 6) == adapter.parse(raw, 10, 6)
    assert decode(adapter, CLOSE, "shutdown", 11, 6) == [FeedNotice(11, 6, "disconnected", detail="shutdown")]
