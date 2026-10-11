import json

from btc_survivor.clock import MS_PER_MINUTE, next_minute
from btc_survivor.feed.base import decode
from btc_survivor.feed.coinbase import CoinbaseAdapter
from btc_survivor.feed.market import Market
from btc_survivor.synthetic import Fault, SyntheticFeed
from btc_survivor.types import BookDelta, BookSnapshot, FeedNotice, Trade

START = 1_791_000_000_000  # a minute boundary


def drive(lines, market=None):
    """Feed recorded lines through the market the way the engine will, closing minutes before later events."""
    adapter = CoinbaseAdapter("BTC-USD")
    market = market or Market(20, 5_000, 2_000)
    boundary = None
    minutes, resyncs = [], []
    for kind, raw, recv, conn in lines:
        boundary = boundary or next_minute(recv)
        while recv >= boundary:
            market.advance(boundary)
            minutes.append(market.close_minute(boundary))
            boundary += MS_PER_MINUTE
        market.advance(recv)
        for ev in decode(adapter, kind, raw, recv, conn):
            market.on_event(ev)
        reason = market.take_resync()
        if reason:
            resyncs.append(reason)
    return market, minutes, resyncs


def test_clean_feed_is_healthy_after_the_first_full_minute():
    market, minutes, resyncs = drive(SyntheticFeed(1, START).lines(600))
    flags = [healthy for _, healthy in minutes]
    assert flags[0] is False and all(flags[1:])
    assert resyncs == [] and market.health.epoch == 0
    assert all(c.mid is not None and c.trade_count > 0 for c, _ in minutes[1:])


def test_faults_request_one_resync_each_and_pause_the_right_minutes():
    faults = (Fault(150, "gap"), Fault(330, "crossed"), Fault(510, "silence", 20))
    market, minutes, resyncs = drive(SyntheticFeed(2, START).lines(800, faults))
    assert resyncs == ["gap", "crossed", "stale"]
    unhealthy = [c.ts for c, healthy in minutes if not healthy]
    for at in (150, 330, 510):
        assert next_minute(START + at * 1000) in unhealthy
    assert market.health.epoch == 3
    assert minutes[-1][1]


def test_short_silence_and_skew_do_not_force_a_reconnect():
    faults = (Fault(150, "silence", 3), Fault(300, "skew", 90, 6_000))
    market, minutes, resyncs = drive(SyntheticFeed(3, START).lines(600, faults))
    assert resyncs == []
    kinds = [k for _, k, _ in market.health.drain_events()]
    assert "clock_skew" in kinds and "disconnected" not in kinds
    assert not dict((c.ts, h) for c, h in minutes)[next_minute(START + 330_000)]


def test_updates_are_ignored_while_the_book_is_invalid():
    market = Market(20, 5_000, 2_000)
    market.on_event(BookDelta(1, 1, [("bid", 100.0, 1.0)]))
    assert market.book.level_count("bid") == 0


def test_trades_count_toward_the_minute_they_were_received_in():
    market = Market(20, 5_000, 2_000)
    boundary = START + MS_PER_MINUTE
    market.on_event(Trade(boundary - 10, 1, 1, 80_000.0, 0.5, exch_ms=boundary - 30))
    candle, _ = market.close_minute(boundary)  # the engine closes the minute before applying later events
    market.on_event(Trade(boundary + 5, 1, 2, 80_001.0, 2.0, exch_ms=boundary - 100))  # stamped earlier, arrived later
    nxt, _ = market.close_minute(boundary + MS_PER_MINUTE)
    assert candle.volume == 0.5 and candle.trade_count == 1
    assert nxt.volume == 2.0 and nxt.close == 80_001.0


def test_trade_snapshot_on_connect_never_reaches_a_candle():
    adapter = CoinbaseAdapter("BTC-USD")
    market = Market(20, 5_000, 2_000)
    lines = list(SyntheticFeed(5, START, trades_per_s=0.0).lines(30))
    snapshot_lines = [raw for kind, raw, _, _ in lines if kind == "message" and '"market_trades"' in raw]
    assert any('"snapshot"' in raw for raw in snapshot_lines)
    for kind, raw, recv, conn in lines:
        for ev in decode(adapter, kind, raw, recv, conn):
            market.on_event(ev)
    candle, _ = market.close_minute(START + MS_PER_MINUTE)
    assert candle.trade_count == 0


def test_only_one_resync_request_per_connection():
    market = Market(20, 5_000, 2_000)
    market.on_event(FeedNotice(0, 1, "connected"))
    market.on_event(FeedNotice(1, 1, "gap", detail="x"))
    market.on_event(FeedNotice(2, 1, "gap", detail="y"))
    market.advance(60_000)
    assert market.take_resync() == "gap"
    market.on_event(FeedNotice(61_000, 1, "gap", detail="z"))
    market.advance(65_000)  # stale too, still the same connection
    assert market.take_resync() is None
    market.on_event(FeedNotice(70_000, 2, "connected"))
    market.advance(80_000)
    assert market.take_resync() == "stale"
    assert json.dumps([k for _, k, _ in market.health.drain_events()])


def _healthy_market():
    """A market that has been Healthy for a minute, built from a short synthetic run."""
    market, minutes, _ = drive(SyntheticFeed(6, START).lines(150))
    assert minutes[-1][1] and market.health.is_healthy
    return market


def test_crossed_snapshot_requests_a_resync():
    market = Market(20, 5_000, 2_000)
    market.on_event(FeedNotice(0, 1, "connected"))
    market.on_event(BookSnapshot(1, 1, [(101.0, 1.0)], [(100.0, 1.0)]))
    assert market.take_resync() == "crossed" and not market.book.valid


def test_one_sided_book_is_lost_and_bumps_the_epoch():
    market = _healthy_market()
    epoch = market.health.epoch
    asks = [p for p, _ in market.book.top("ask", 1000)]
    t = START + 151_000
    market.on_event(BookDelta(t, 1, [("ask", p, 0.0) for p in asks]))
    assert not market.book.valid and market.health.epoch == epoch + 1
    assert market.take_resync() == "one_sided"


def test_one_sided_snapshot_is_not_usable():
    market = Market(20, 5_000, 2_000)
    market.on_event(FeedNotice(0, 1, "connected"))
    market.on_event(BookSnapshot(1, 1, [(99.0, 1.0)], []))
    assert not market.book.valid and not market.health.book_ok
    assert market.take_resync() == "one_sided"
