"""Adapter tests with message shapes copied from a live keyless probe of the Advanced Trade feed."""

import json

from btc_survivor.clock import iso
from btc_survivor.feed.base import CLOSE, OPEN, decode
from btc_survivor.feed.coinbase import CoinbaseAdapter
from btc_survivor.types import ASK, BID, BookDelta, BookSnapshot, FeedNotice, Trade

TS = "2026-10-11T03:00:12.113085165Z"


def msg(channel, seq, events, ts=TS):
    return json.dumps({"channel": channel, "timestamp": ts, "sequence_num": seq, "events": events})


SNAPSHOT = msg("l2_data", 0, [{"type": "snapshot", "product_id": "BTC-USD", "updates": [
    {"side": "bid", "event_time": "2026-10-11T03:00:11.71911Z", "price_level": "83012", "new_quantity": "0.10374745"},
    {"side": "bid", "event_time": "2026-10-11T03:00:11.71911Z", "price_level": "83011.98", "new_quantity": "0.00002053"},
    {"side": "offer", "event_time": "2026-10-11T03:00:11.71911Z", "price_level": "83012.01", "new_quantity": "0.5"},
    {"side": "offer", "event_time": "2026-10-11T03:00:11.71911Z", "price_level": "83012.5", "new_quantity": "1.25"},
]}])
UPDATE = msg("l2_data", 1, [{"type": "update", "product_id": "BTC-USD", "updates": [
    {"side": "bid", "event_time": "2026-10-11T03:00:12.1Z", "price_level": "83012", "new_quantity": "0"},
    {"side": "offer", "event_time": "2026-10-11T03:00:12.1Z", "price_level": "83013", "new_quantity": "0.2"},
]}])
TRADES_SNAPSHOT = msg("market_trades", 2, [{"type": "snapshot", "trades": [
    {"product_id": "BTC-USD", "trade_id": "1104632112", "price": "83012.01", "size": "0.00035778", "time": "2026-10-11T03:00:11.00225Z", "side": "SELL"},
    {"product_id": "BTC-USD", "trade_id": "1104632111", "price": "83012.01", "size": "0.00011926", "time": "2026-10-11T03:00:10.758894Z", "side": "SELL"},
]}])
HEARTBEAT = msg("heartbeats", 3, [{"current_time": "2026-10-11 03:00:12.584636631 +0000 UTC m=+359251.267558137", "heartbeat_counter": 359251}])


def trades_update(seq, ids):
    trades = [{"product_id": "BTC-USD", "trade_id": str(i), "price": "83012.01", "size": "0.001",
               "time": "2026-10-11T03:00:12.05Z", "side": "SELL"} for i in ids]
    return msg("market_trades", seq, [{"type": "update", "trades": trades}])


def fresh():
    adapter = CoinbaseAdapter("BTC-USD")
    adapter.on_open(1)
    return adapter


def test_subscribe_messages_are_public_channels_only():
    subs = [json.loads(m) for m in CoinbaseAdapter("BTC-USD").subscribe_messages()]
    assert [s["channel"] for s in subs] == ["level2", "market_trades", "heartbeats"]
    for s in subs:
        assert set(s) == {"type", "product_ids", "channel"} and s["product_ids"] == ["BTC-USD"]


def test_snapshot_parses_levels_and_envelope_time():
    [ev] = fresh().parse(SNAPSHOT, 1000, 1)
    assert isinstance(ev, BookSnapshot)
    assert ev.bids == [(83012.0, 0.10374745), (83011.98, 0.00002053)]
    assert ev.asks == [(83012.01, 0.5), (83012.5, 1.25)]
    assert iso(ev.exch_ms) == "2026-10-11T03:00:12.113Z"
    assert ev.recv_ms == 1000 and ev.conn_id == 1


def test_update_maps_sides_and_zero_quantity():
    adapter = fresh()
    adapter.parse(SNAPSHOT, 1000, 1)
    [ev] = adapter.parse(UPDATE, 1001, 1)
    assert isinstance(ev, BookDelta)
    assert ev.changes == [(BID, 83012.0, 0.0), (ASK, 83013.0, 0.2)]


def test_sequence_jump_reports_a_gap_before_the_message_events():
    adapter = fresh()
    adapter.parse(SNAPSHOT, 1000, 1)
    events = adapter.parse(UPDATE.replace('"sequence_num": 1', '"sequence_num": 5'), 1001, 1)
    assert isinstance(events[0], FeedNotice) and events[0].kind == "gap"
    assert events[0].detail == "expected 1, got 5"
    assert isinstance(events[1], BookDelta)
    assert adapter.parse(HEARTBEAT.replace('"sequence_num": 3', '"sequence_num": 6'), 1002, 1)[0].kind == "heartbeat"


def test_snapshot_may_arrive_before_the_subscription_ack():
    adapter = fresh()
    first = adapter.parse(SNAPSHOT, 1000, 1)
    ack = adapter.parse(msg("subscriptions", 1, [{"subscriptions": {"level2": ["BTC-USD"]}}]), 1001, 1)
    assert isinstance(first[0], BookSnapshot)
    assert ack[0].kind == "subscribed"


def test_new_connection_resets_sequence_expectation():
    adapter = fresh()
    adapter.parse(SNAPSHOT, 1000, 1)
    adapter.on_open(2)
    assert not any(isinstance(e, FeedNotice) and e.kind == "gap" for e in adapter.parse(SNAPSHOT, 2000, 2))


def test_trade_snapshot_only_sets_the_cursor():
    adapter = fresh()
    assert adapter.parse(TRADES_SNAPSHOT.replace('"sequence_num": 2', '"sequence_num": 0'), 1000, 1) == []
    events = adapter.parse(trades_update(1, [1104632114, 1104632113]), 1001, 1)  # newest first, like the feed
    assert [e.trade_id for e in events] == [1104632113, 1104632114]
    assert all(isinstance(e, Trade) and e.price == 83012.01 and e.size == 0.001 for e in events)


def test_duplicate_trades_are_skipped_and_missing_ids_reported():
    adapter = fresh()
    adapter.parse(TRADES_SNAPSHOT.replace('"sequence_num": 2', '"sequence_num": 0'), 1000, 1)
    events = adapter.parse(trades_update(1, [1104632112, 1104632115]), 1001, 1)
    assert [type(e).__name__ for e in events] == ["FeedNotice", "Trade"]
    assert events[0].kind == "trade_gap" and "1104632113 to 1104632114" in events[0].detail
    assert events[1].trade_id == 1104632115


def test_other_products_are_ignored():
    adapter = fresh()
    other = SNAPSHOT.replace("BTC-USD", "ETH-USD")
    assert adapter.parse(other, 1000, 1) == []


def test_error_messages_become_notices():
    [ev] = fresh().parse(json.dumps({"type": "error", "message": "failure to subscribe"}), 1000, 1)
    assert ev.kind == "error" and "failure" in ev.detail


def test_decode_turns_lifecycle_lines_into_notices():
    adapter = CoinbaseAdapter("BTC-USD")
    [opened] = decode(adapter, OPEN, "", 500, 3)
    [closed] = decode(adapter, CLOSE, "server closed", 900, 3)
    assert opened.kind == "connected" and opened.conn_id == 3
    assert closed.kind == "disconnected" and closed.detail == "server closed"


def test_trades_for_other_products_are_ignored():
    adapter = fresh()
    adapter.parse(TRADES_SNAPSHOT.replace('"sequence_num": 2', '"sequence_num": 0'), 1000, 1)
    other = trades_update(1, [1104632113]).replace("BTC-USD", "ETH-USD")
    assert adapter.parse(other, 1001, 1) == []
