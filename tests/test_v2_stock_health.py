"""Price health of PLAN_V2 section 4.5 on synthetic prints, the reconnect and resubscribe rules of section 4.6 on fixed
times and against the recorder's inline copy, the delay rule, and the news health of one minute from section 7.8, alone
and through the shared market layer, where a silent minute after an unhealthy heartbeat stays unhealthy."""

import sys
from datetime import date

import numpy as np
import pytest

from btc_survivor.clock import MS_PER_HOUR, MS_PER_MINUTE, MS_PER_SECOND
from btc_survivor.types import FeedNotice
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import health
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter, Print, PrintBatch
from btc_survivor.v2.stocks.health import (CLOSED, HASH_MISMATCH, HEALTHY, NO_HEARTBEAT, OLD_HEARTBEAT, PRINT_DELAY,
                                           RECOVERING, SESSION_CLOSE, SESSION_OPEN, STALE, UNHEALTHY_HEARTBEAT,
                                           DelayRule, HealthEvent, Heartbeat, PriceHealth, delay_rule, news_minute,
                                           quiet_symbols, should_mark_healthy, silence_reason)
from btc_survivor.v2.stocks.market import NEWS_HEALTHY, NEWS_UNHEALTHY, MarketLayer
from btc_survivor.v2.stocks.news_inputs import HeartbeatLine
from btc_survivor.v2.stocks.prints import PriceRules
from btc_survivor.v2.stocks.runner import BackoffSettings, StockFeedClient
from helpers import ROOT
from v2_feedkit import CANARY, S1

sys.path.insert(0, str(ROOT / "scripts"))
import v2_record_stocks  # noqa: E402

SEC, MIN = MS_PER_SECOND, MS_PER_MINUTE
DAY = sessions.session_on(date(2026, 10, 13))
NEXT = sessions.session_on(date(2026, 10, 14))
OPEN, CLOSE = DAY.open_ms, DAY.close_ms
SYMBOLS = ("AAPL", "MSFT", "QQQ")
OTHERS = ("MSFT", "QQQ")
STALE_MS = 5 * SEC


def monitor():
    return PriceHealth(SYMBOLS, STALE_MS, 15 * SEC, 2 * SEC, PriceRules(late_ms=4 * SEC, excluded=frozenset({"Z"})))


def batch(recv, names=SYMBOLS, delay=100, conn=1, conditions=()):
    """One trade message received at recv with one synthetic print per symbol, delay milliseconds old."""
    return PrintBatch(recv, conn, tuple(Print(s, 100.0, 1.0, recv - delay, conditions) for s in names))


def stream(h, start, end, names=SYMBOLS, every=500, delay=100):
    for t in range(start, end, every):
        h.on_event(batch(t, names, delay))


def connected(at=OPEN - 30 * MIN):
    h = monitor()
    h.on_event(FeedNotice(at, 1, "connected"))
    return h


def healthy_after_first_minute():
    """A monitor connected before the open whose every symbol was healthy at 09:31, with its events drained."""
    h = connected()
    stream(h, OPEN, OPEN + MIN)
    assert h.judge_minute(OPEN + MIN) == dict.fromkeys(SYMBOLS, True)
    h.drain_events()
    return h


def states(h):
    return {symbol: s.state for symbol, s in h.symbols.items()}


def epochs(h):
    return {symbol: s.epoch for symbol, s in h.symbols.items()}


def test_settings_come_from_the_stock_feed_config():
    feed = load_stock_config(S1).stock_feed
    h = PriceHealth.from_config(feed)
    assert tuple(h.symbols) == feed.symbols
    assert (h.feed_stale_ms, h.symbol_stale_ms, h.max_delay_ms) == tuple(
        round(value * SEC) for value in (feed.feed_stale_after_seconds, feed.symbol_stale_after_seconds,
                                         feed.max_print_delay_seconds))
    assert h.rules == PriceRules.from_config(feed)
    assert set(states(h).values()) == {CLOSED} and h.drain_events() == []


def test_every_symbol_stays_closed_outside_the_session_with_only_whole_feed_events_and_no_reconnect():
    h = monitor()
    h.advance(CLOSE + MIN)
    h.on_event(FeedNotice(CLOSE + 2 * MIN, 1, "connected"))
    stream(h, CLOSE + 2 * MIN, CLOSE + 3 * MIN, delay=9 * SEC)  # post market prints far behind, and only some
    h.on_event(FeedNotice(CLOSE + 4 * MIN, 1, "error", detail="synthetic error"))
    h.on_event(FeedNotice(CLOSE + 5 * MIN, 1, "disconnected", detail="closed by server"))
    h.on_event(FeedNotice(CLOSE + 6 * MIN, 2, "connected"))
    assert h.judge_minute(CLOSE + 7 * MIN) == dict.fromkeys(SYMBOLS, False)
    pre_open = NEXT.open_ms - 10 * MIN
    checked = 0
    for t in range(CLOSE + 7 * MIN, pre_open, MIN):  # a whole silent night, up to the pre open check
        reason, checked = silence_reason(t, CLOSE + 6 * MIN, STALE_MS, checked)
        assert reason is None and checked == 0
        assert quiet_symbols(t, CLOSE + 6 * MIN, SYMBOLS, {"AAPL"}, {}) == []
    h.advance(pre_open)
    stream(h, NEXT.open_ms - 2 * MIN, NEXT.open_ms, names=("AAPL",), delay=9 * SEC)  # pre market
    assert h.drain_events() == [HealthEvent(CLOSE + 2 * MIN, "connected", None, ""),
                                HealthEvent(CLOSE + 4 * MIN, "error", None, "synthetic error"),
                                HealthEvent(CLOSE + 5 * MIN, "disconnected", None, "closed by server"),
                                HealthEvent(CLOSE + 6 * MIN, "connected", None, "")]
    assert set(states(h).values()) == {CLOSED} and set(epochs(h).values()) == {0}
    assert h.session is None and h.feed_ok


def test_mark_healthy_after_60_seconds_connected_outside_the_session_and_on_health_inside_it():
    after = round(load_stock_config(S1).stock_feed.mark_healthy_outside_session_seconds * SEC)
    assert after == 60 * SEC
    night = CLOSE + 3 * MS_PER_HOUR
    assert not should_mark_healthy(night + after - 1, night, after, False)
    assert should_mark_healthy(night + after, night, after, False)
    assert not should_mark_healthy(night + after, None, after, True)
    mid = OPEN + MS_PER_HOUR
    assert not should_mark_healthy(mid, OPEN - MS_PER_HOUR, after, False)  # open for hours, no healthy symbol
    assert should_mark_healthy(mid, mid - 1, after, True)
    client = StockFeedClient(FinnhubAdapter(SYMBOLS, CANARY), BackoffSettings(5.0, 60.0), lambda *row: None,
                             np.random.default_rng(3))
    for _ in range(4):
        client.backoff_seconds()
    assert client.attempt == 4
    if should_mark_healthy(night + after, night, after, False):
        client.mark_healthy()
    assert client.attempt == 0 and 2.5 <= client.backoff_seconds() <= 5


def test_a_connected_feed_is_healthy_for_the_minute_ending_0931():
    h = connected()
    stream(h, OPEN - 10 * MIN, OPEN, every=2 * SEC)  # pre market, still closed
    assert h.drain_events() == [HealthEvent(OPEN - 30 * MIN, "connected", None, "")]
    stream(h, OPEN, OPEN + MIN)
    assert h.judge_minute(OPEN + MIN) == dict.fromkeys(SYMBOLS, True)
    events = h.drain_events()
    assert events[0] == HealthEvent(OPEN, SESSION_OPEN, None, "")
    assert [(e.ms, e.kind, e.symbol) for e in events[1:]] == (
        [(OPEN, RECOVERING, s) for s in SYMBOLS] + [(OPEN + MIN, HEALTHY, s) for s in SYMBOLS])
    assert h.any_healthy and set(epochs(h).values()) == {0}

    cold = monitor()  # never connected, so stale from the open
    cold.advance(OPEN - MIN)
    cold.advance(OPEN + 30 * SEC)
    assert set(states(cold).values()) == {STALE}
    assert cold.judge_minute(OPEN + MIN) == dict.fromkeys(SYMBOLS, False)
    with pytest.raises(ValueError, match="once, before applying"):
        cold.judge_minute(OPEN + MIN)
    cold.advance(OPEN + 2 * MIN)
    with pytest.raises(ValueError, match="once, before applying"):
        cold.judge_minute(OPEN + 2 * MIN)


def test_the_silence_clocks_start_at_the_open_and_count_from_the_instant_the_limit_ran_out():
    h = connected()
    h.on_event(batch(OPEN - 10 * SEC))  # the last message, before the open
    h.advance(OPEN + STALE_MS - 1)
    assert set(states(h).values()) == {RECOVERING}
    h.advance(OPEN + STALE_MS + 700)  # a ticker that notices late
    stale = [(e.ms, e.symbol, e.detail) for e in h.drain_events() if e.kind == STALE]
    assert stale == [(OPEN + STALE_MS, s, "no message for 5 s") for s in SYMBOLS]
    assert silence_reason(OPEN + STALE_MS - 1, OPEN - 10 * SEC, STALE_MS, 0) == (None, 0)
    assert silence_reason(OPEN + STALE_MS, OPEN - 10 * SEC, STALE_MS, 0)[0] == "no message for 5 s in session"

    h = connected()
    for t in range(OPEN - 10 * SEC, OPEN + 20 * SEC, SEC):  # pings keep the socket alive, but nothing prints
        h.on_event(FeedNotice(t, 1, "ping"))
    stale = [(e.ms, e.symbol, e.detail) for e in h.drain_events() if e.kind == STALE]
    assert stale == [(OPEN + 15 * SEC, s, "no eligible print for 15 s") for s in SYMBOLS]
    assert h.feed_ok


def test_one_symbol_goes_stale_alone_after_15_seconds_without_a_reconnect():
    h = healthy_after_first_minute()
    b1, b2 = OPEN + MIN, OPEN + 2 * MIN
    stream(h, b1, b1 + 10 * SEC)  # AAPL's last print at b1 + 9.5 s
    stream(h, b1 + 10 * SEC, b1 + 24 * SEC, OTHERS)
    h.advance(b1 + 24_499)
    assert states(h) == dict.fromkeys(SYMBOLS, HEALTHY)
    stream(h, b1 + 24 * SEC + 500, b2, OTHERS)
    assert [(e.ms, e.kind, e.symbol) for e in h.drain_events()] == [(b1 + 24_500, STALE, "AAPL")]
    assert h.judge_minute(b2) == {"AAPL": False, "MSFT": True, "QQQ": True}
    assert epochs(h) == {"AAPL": 1, "MSFT": 0, "QQQ": 0}
    assert silence_reason(b2, h.last_msg_ms, STALE_MS, 0) == (None, 0)
    assert quiet_symbols(b2, h.connected_ms, SYMBOLS, h.printed, {}) == []


def test_five_seconds_of_total_silence_stales_every_symbol_and_asks_for_a_reconnect():
    h = healthy_after_first_minute()
    b1 = OPEN + MIN
    stream(h, b1, b1 + 10 * SEC)
    last = h.last_msg_ms
    h.advance(last + STALE_MS - 1)
    assert states(h) == dict.fromkeys(SYMBOLS, HEALTHY)
    assert silence_reason(last + STALE_MS - 1, last, STALE_MS, 0)[0] is None
    reason, _ = silence_reason(last + STALE_MS, last, STALE_MS, 0)
    assert reason == "no message for 5 s in session"
    h.advance(last + STALE_MS)
    assert states(h) == dict.fromkeys(SYMBOLS, STALE) and epochs(h) == dict.fromkeys(SYMBOLS, 1)
    h.on_event(FeedNotice(last + 6 * SEC, 1, "disconnected", detail=reason))
    h.on_event(FeedNotice(last + 7 * SEC, 2, "connected"))
    assert states(h) == dict.fromkeys(SYMBOLS, RECOVERING)
    h.on_event(FeedNotice(last + 8 * SEC, 2, "error", detail="synthetic error"))
    stream(h, last + 8 * SEC, b1 + MIN)
    assert h.judge_minute(b1 + MIN) == dict.fromkeys(SYMBOLS, False)
    stream(h, b1 + MIN, b1 + 2 * MIN)
    assert h.judge_minute(b1 + 2 * MIN) == dict.fromkeys(SYMBOLS, True)
    whole = [(e.ms, e.kind, e.detail) for e in h.drain_events() if e.symbol is None]
    assert whole == [(last + 6 * SEC, "disconnected", reason), (last + 7 * SEC, "connected", ""),
                     (last + 8 * SEC, "error", "synthetic error")]


def test_a_connection_without_a_close_before_it_starts_every_clean_stretch_again():
    h = healthy_after_first_minute()
    stream(h, OPEN + MIN, OPEN + MIN + 2 * SEC)
    h.on_event(FeedNotice(OPEN + MIN + 2 * SEC, 2, "connected"))  # a recording after a kill
    assert states(h) == dict.fromkeys(SYMBOLS, RECOVERING) and epochs(h) == dict.fromkeys(SYMBOLS, 1)
    whole = [(e.kind, e.detail) for e in h.drain_events() if e.symbol is None]
    assert whole == [("disconnected", "connected again without a close"), ("connected", "")]
    stream(h, OPEN + MIN + 2 * SEC, OPEN + 2 * MIN)
    assert h.judge_minute(OPEN + 2 * MIN) == dict.fromkeys(SYMBOLS, False)


def test_a_delayed_feed_is_held_out_until_its_median_delay_returns():
    h = connected()
    first = OPEN + 3 * SEC  # the first vendor time in the session
    stream(h, first, first + 1_500, delay=3 * SEC)  # 9 samples, eligible but 3 s behind
    assert h.delay_ok and states(h) == dict.fromkeys(SYMBOLS, RECOVERING)
    h.on_event(batch(first + 1_500, delay=3 * SEC))
    assert not h.delay_ok and states(h) == dict.fromkeys(SYMBOLS, STALE)
    assert h.drain_events()[-4:] == [HealthEvent(first + 1_500, PRINT_DELAY, None, "median 3000 ms"),
                                     *(HealthEvent(first + 1_500, STALE, s, PRINT_DELAY) for s in SYMBOLS)]
    stream(h, first + 2 * SEC, OPEN + MIN, delay=3 * SEC)
    assert h.judge_minute(OPEN + MIN) == dict.fromkeys(SYMBOLS, False)
    stream(h, OPEN + MIN, OPEN + 2 * MIN)  # the median of the last 50 returns at the ninth timely batch
    cleared = OPEN + MIN + 4 * SEC
    assert h.drain_events() == [HealthEvent(cleared, PRINT_DELAY, None, "cleared, median 100 ms"),
                                *(HealthEvent(cleared, RECOVERING, s, "print delay cleared") for s in SYMBOLS)]
    assert h.judge_minute(OPEN + 2 * MIN) == dict.fromkeys(SYMBOLS, False)
    stream(h, OPEN + 2 * MIN, OPEN + 3 * MIN)
    assert h.judge_minute(OPEN + 3 * MIN) == dict.fromkeys(SYMBOLS, True)

    early = connected()  # either way, and a new connection clears the samples
    stream(early, OPEN, OPEN + 2 * SEC, delay=-2_500)  # 12 samples from prints that seem early
    assert not early.delay_ok
    early.on_event(FeedNotice(OPEN + 3 * SEC, 2, "connected"))
    assert early.delay_ok
    kinds = [(e.kind, e.detail) for e in early.drain_events() if e.kind == PRINT_DELAY]
    assert kinds == [(PRINT_DELAY, "median -2500 ms"), (PRINT_DELAY, "cleared by a new connection")]


def test_the_delay_rule_at_its_floor_and_cap():
    assert delay_rule(100, 300) == DelayRule(2.0, 4.0, False)
    assert delay_rule(100, 1_300) == DelayRule(2.0, 4.0, False)  # ceil(1.95) is the floor
    assert delay_rule(100, 1_334) == DelayRule(3.0, 6.0, False)  # ceil(2.001)
    assert delay_rule(100, 2_000) == DelayRule(3.0, 6.0, False)  # the p95 sets it, not the median
    assert delay_rule(2_200, 2_200) == DelayRule(4.0, 8.0, False)  # ceil(3.3)
    assert delay_rule(100, 3_333) == DelayRule(5.0, 10.0, False)
    assert delay_rule(100, 3_334) == DelayRule(5.0, 10.0, False)  # ceil(5.001) is capped
    assert delay_rule(-3_000, -2_000) == DelayRule(2.0, 4.0, False)
    for settings in (delay_rule(100, p95) for p95 in range(-6_000, 12_000, 7)):  # always passes the config check
        assert 2 <= settings.max_print_delay_seconds <= 5
        assert settings.late_print_seconds == 2 * settings.max_print_delay_seconds
    assert not delay_rule(5_000, 9_000).stop
    assert delay_rule(5_001, 9_000) == DelayRule(5.0, 10.0, True)
    assert not delay_rule(-5_000, -4_000).stop  # either way, as live health objects to a clock that runs ahead
    assert delay_rule(-5_001, -4_000) == DelayRule(2.0, 4.0, True)


def test_epochs_bump_on_leaving_healthy_and_once_for_every_symbol_at_every_close():
    h = monitor()
    h.on_event(FeedNotice(CLOSE - 3 * MIN, 1, "connected"))  # a start inside the session joins it at once
    assert h.drain_events()[0] == HealthEvent(CLOSE - 3 * MIN, SESSION_OPEN, None, "joined in session")
    stream(h, CLOSE - 3 * MIN, CLOSE - 2 * MIN)
    assert h.judge_minute(CLOSE - 2 * MIN) == dict.fromkeys(SYMBOLS, True)
    stream(h, CLOSE - 2 * MIN, CLOSE - MIN, OTHERS)
    assert h.judge_minute(CLOSE - MIN) == {"AAPL": False, "MSFT": True, "QQQ": True}
    stream(h, CLOSE - MIN, CLOSE, ("AAPL", "MSFT"))
    assert h.judge_minute(CLOSE) == {"AAPL": True, "MSFT": True, "QQQ": False}
    assert epochs(h) == {"AAPL": 1, "MSFT": 0, "QQQ": 1}
    h.drain_events()
    h.advance(CLOSE + 10 * MIN)  # the feed went quiet at the close, which is no stale event
    assert epochs(h) == {"AAPL": 2, "MSFT": 1, "QQQ": 2}
    assert h.drain_events() == [HealthEvent(CLOSE, SESSION_CLOSE, None, ""),
                                *(HealthEvent(CLOSE, CLOSED, s, SESSION_CLOSE) for s in SYMBOLS)]
    h.advance(NEXT.open_ms + SEC)
    assert states(h) == dict.fromkeys(SYMBOLS, RECOVERING) and epochs(h) == {"AAPL": 2, "MSFT": 1, "QQQ": 2}


def test_the_pre_open_reconnect_runs_once_per_session():
    ten = OPEN - 10 * MIN
    assert silence_reason(ten - 1, ten - MS_PER_HOUR, STALE_MS, 0) == (None, 0)
    reason, checked = silence_reason(ten + 300, ten - 6 * MIN, STALE_MS, 0)
    assert reason == "silent for 5 minutes before the open" and checked == OPEN
    reopened = ten + 2 * SEC  # the new connection stays silent up to the open
    for t in range(reopened, OPEN, SEC):
        reason, checked = silence_reason(t, reopened, STALE_MS, checked)
        assert reason is None
    reason, checked = silence_reason(ten, ten - MIN, STALE_MS, 0)  # quiet but not silent at the one check
    assert (reason, checked) == (None, OPEN)
    assert silence_reason(ten + 5 * MIN, ten - MIN, STALE_MS, checked) == (None, OPEN)
    tomorrow = NEXT.open_ms - 10 * MIN
    assert silence_reason(tomorrow, tomorrow - MS_PER_HOUR, STALE_MS, checked)[0] is not None


def test_resubscribe_limits():
    h = connected(OPEN - MS_PER_HOUR)
    h.on_event(batch(OPEN + SEC, ("AAPL", "MSFT")))
    assert h.printed == {"AAPL", "MSFT"}
    early = h.connected_ms
    assert quiet_symbols(OPEN + 119_999, early, SYMBOLS, h.printed, {}) == []
    assert quiet_symbols(OPEN + 120_000, early, SYMBOLS, h.printed, {}) == ["QQQ"]
    now = OPEN + 30 * MIN
    assert quiet_symbols(now, early, SYMBOLS, h.printed, {"QQQ": now - 5 * MIN + 1}) == []
    assert quiet_symbols(now, early, SYMBOLS, h.printed, {"QQQ": now - 5 * MIN}) == ["QQQ"]
    late = OPEN + 10 * MIN  # 120 seconds count from a connection made inside the session too
    assert quiet_symbols(late + 119_999, late, SYMBOLS, h.printed, {}) == []
    assert quiet_symbols(late + 120_000, late, SYMBOLS, h.printed, {}) == ["QQQ"]
    assert quiet_symbols(now, early, SYMBOLS, set(), {}) == []  # nothing prints, so no symbol is singled out
    assert quiet_symbols(OPEN - 1, early, SYMBOLS, h.printed, {}) == []
    h.on_event(FeedNotice(OPEN + 2 * SEC, 2, "connected"))
    assert h.printed == set() and h.connected_ms == OPEN + 2 * SEC


def test_a_five_minute_halt_stops_one_symbol_and_never_reconnects():
    h = healthy_after_first_minute()
    checked = 0
    for n in range(1, 7):  # AAPL halts after 09:31 and prints again from 09:36
        start = OPEN + n * MIN
        names = SYMBOLS if n == 6 else OTHERS
        for t in range(start, start + MIN, 500):
            h.on_event(batch(t, names))
            if t % SEC == 0:
                reason, checked = silence_reason(t, h.last_msg_ms, STALE_MS, checked)
                assert reason is None
                assert quiet_symbols(t, h.connected_ms, SYMBOLS, h.printed, {}) == []
        assert h.judge_minute(start + MIN) == {"AAPL": n == 6, "MSFT": True, "QQQ": True}
    assert epochs(h) == {"AAPL": 1, "MSFT": 0, "QQQ": 0}
    assert [(e.kind, e.symbol) for e in h.drain_events()] == [(STALE, "AAPL"), (RECOVERING, "AAPL"),
                                                              (HEALTHY, "AAPL")]


def test_replayed_late_early_excluded_and_pre_market_prints_never_freshen_a_symbol():
    h = connected()
    odd = [batch(OPEN + 100, ("AAPL",) * 12, delay=9 * SEC),  # vendor time before the open, and no delay sample
           batch(OPEN + 6_250, ("AAPL",), delay=4_001),  # late
           batch(OPEN + 8_250, ("AAPL",), conditions=("1", "Z")),  # excluded
           batch(OPEN + 10_250, ("AAPL",), delay=-2_001)]  # early
    for ev in sorted(odd + [batch(t, OTHERS) for t in range(OPEN, OPEN + 15 * SEC, 500)], key=lambda e: e.recv_ms):
        h.on_event(ev)
    h.advance(OPEN + 15 * SEC)
    assert states(h) == {"AAPL": STALE, "MSFT": RECOVERING, "QQQ": RECOVERING} and h.delay_ok
    reconnect = OPEN + 20 * SEC
    h.on_event(FeedNotice(reconnect - SEC, 1, "disconnected"))
    h.on_event(FeedNotice(reconnect, 2, "connected"))
    h.on_event(batch(reconnect + 300, SYMBOLS * 4, delay=9 * SEC, conn=2))  # replayed by the new connection
    assert h.delay_ok and states(h) == {"AAPL": STALE, "MSFT": RECOVERING, "QQQ": RECOVERING}
    h.on_event(batch(reconnect + SEC, ("AAPL",), conn=2))
    assert states(h)["AAPL"] == RECOVERING


def eventful_morning():
    """Prints from before the open to 09:36 at seeded intervals, with a 7 second silence, a reconnect, a 20 second AAPL
    halt and 30 seconds 3 seconds behind."""
    rng = np.random.default_rng(5)
    events = [FeedNotice(OPEN - 2 * MIN, 1, "connected")]
    gone, back = OPEN + 3 * MIN + 100, OPEN + 3 * MIN + 2 * SEC
    t = OPEN - 2 * MIN
    while t < OPEN + 6 * MIN:
        t += int(rng.integers(50, 900))
        if OPEN + 70 * SEC <= t < OPEN + 77 * SEC or gone <= t <= back:
            continue
        names = tuple(s for s in SYMBOLS if not (s == "AAPL" and OPEN + 2 * MIN <= t < OPEN + 2 * MIN + 20 * SEC))
        delay = 3 * SEC if OPEN + 4 * MIN <= t < OPEN + 4 * MIN + 30 * SEC else int(rng.integers(20, 400))
        events.append(batch(t, names, delay, 2 if t > back else 1))
    events += [FeedNotice(gone, 1, "disconnected", detail="test"), FeedNotice(back, 2, "connected")]
    return sorted(events, key=lambda ev: ev.recv_ms)


def run_through(events, end, tick_ms=None):
    """Feed events in stamp order, judging every boundary before anything stamped at or after it, with an optional
    ticker that advances every tick_ms between them."""
    h = monitor()
    verdicts = {}
    boundary = OPEN + MIN
    ticks = range(OPEN - MIN + 337, end, tick_ms) if tick_ms else ()
    timeline = sorted([(ev.recv_ms, 1, ev) for ev in events] + [(t, 0, None) for t in ticks], key=lambda x: x[:2])
    for at, _, ev in timeline + [(end, 0, None)]:
        while boundary <= at:
            verdicts[boundary] = h.judge_minute(boundary)
            boundary += MIN
        if ev is None:
            h.advance(at)
        else:
            h.on_event(ev)
    return verdicts, h.drain_events(), epochs(h)


def test_states_and_events_do_not_depend_on_the_ticker():
    events = eventful_morning()
    end = OPEN + 7 * MIN
    replayed = run_through(events, end)
    verdicts, logged, _ = replayed
    assert len(verdicts) == 7 and any(all(v.values()) for v in verdicts.values())
    assert {e.kind for e in logged} >= {STALE, RECOVERING, HEALTHY, PRINT_DELAY, "connected", "disconnected"}
    assert run_through(events, end, tick_ms=SEC) == replayed
    assert run_through(events, end, tick_ms=97) == replayed


def test_the_reconnect_rules_match_the_recorder():
    for name in ("PRE_OPEN_CHECK_MS", "PRE_OPEN_SILENCE_MS", "RESUBSCRIBE_AFTER_MS", "RESUBSCRIBE_EVERY_MS"):
        assert getattr(health, name) == getattr(v2_record_stocks, name)
    rng = np.random.default_rng(11)
    watched = v2_record_stocks.Watched()
    checked = 0
    ages = [0, 4_999, 5_000, 6_000, 299_999, 300_000, MS_PER_HOUR]
    t = OPEN - 30 * MIN
    reasons = 0
    while t < NEXT.open_ms + 30 * MIN:
        t += int(rng.integers(200, 120 * SEC))
        last = t - ages[int(rng.integers(len(ages)))]
        reason, checked = silence_reason(t, last, STALE_MS, checked)
        assert reason == v2_record_stocks.silence_reason(t, last, STALE_MS, watched)
        assert checked == watched.pre_open_checked
        reasons += reason is not None
        connected_ms = t - int(rng.integers(0, 10 * MIN))
        printed = {s for s in SYMBOLS if rng.random() < 0.5}
        resubscribed = {s: t - int(rng.integers(0, 10 * MIN)) for s in SYMBOLS if rng.random() < 0.3}
        assert (quiet_symbols(t, connected_ms, SYMBOLS, printed, resubscribed)
                == v2_record_stocks.quiet_symbols(t, connected_ms, SYMBOLS, printed, resubscribed))
    assert reasons > 10 and checked == NEXT.open_ms


def test_news_health_of_a_minute_needs_a_fresh_healthy_heartbeat_with_the_launch_hashes():
    age = round(load_stock_config(S1).news_input.heartbeat_max_age_seconds * SEC)
    assert age == 120 * SEC
    b = OPEN + 10 * MIN
    start = b - MIN

    def good(t):
        return Heartbeat(t, True, True)

    assert news_minute(b, good(start - 10 * SEC), [good(start + 20 * SEC), good(start + 50 * SEC)], age) is None
    assert news_minute(b, good(b - age), [], age) is None
    assert news_minute(b, good(b - age - 1), [], age) == OLD_HEARTBEAT
    assert news_minute(b, good(start - 100 * SEC), [good(start + 20 * SEC)], age) is None
    assert news_minute(b, good(start - 100 * SEC), [good(start + 20 * SEC + 1)], age) == OLD_HEARTBEAT
    assert news_minute(b, None, [], age) == NO_HEARTBEAT
    assert news_minute(b, None, [good(start + 1)], age) == NO_HEARTBEAT
    assert news_minute(b, None, [good(start), good(start + 30 * SEC)], age) is None
    unhealthy = [good(start + 20 * SEC), Heartbeat(start + 50 * SEC, False, True)]
    assert news_minute(b, good(start - 10 * SEC), unhealthy, age) == UNHEALTHY_HEARTBEAT
    changed = [Heartbeat(start + 20 * SEC, False, False)]
    assert news_minute(b, good(start - 10 * SEC), changed, age) == HASH_MISMATCH
    # The line in force at the start of the minute is judged too, so a minute with no line after an unhealthy line or
    # one with other hashes stays out, as does a minute whose healthy line came after an unhealthy one.
    sick = Heartbeat(start - 10 * SEC, False, True)
    assert news_minute(b, sick, [], age) == UNHEALTHY_HEARTBEAT
    assert news_minute(b, Heartbeat(start - 10 * SEC, True, False), [], age) == HASH_MISMATCH
    assert news_minute(b, sick, [good(start + 20 * SEC)], age) == UNHEALTHY_HEARTBEAT
    assert news_minute(b + MIN, good(start + 20 * SEC), [good(b + 10 * SEC)], age) is None


def news_reasons(lines, boundaries):
    """Feed heartbeat lines in stamp order through one shared market layer, closing each boundary before any line
    stamped at or after it, and return the news reason of every minute with the news feed events written."""
    market = MarketLayer(load_stock_config(S1), LAUNCH_HASHES)
    pending = sorted(lines, key=lambda line: line.recv_ms)
    reasons = []
    for b in boundaries:
        while pending and pending[0].recv_ms < b:
            market.on_heartbeat(pending.pop(0))
        m = market.close_minute(b)
        assert m.news_ok == (m.news_reason is None)
        reasons.append(m.news_reason)
    news = [(ev.ms, ev.kind, ev.detail) for ev in market.drain_events() if ev.kind.startswith("news_")]
    return reasons, news


LAUNCH_HASHES = ("c" * 64, "p" * 64, "s" * 64)


def beat(seq, t, healthy=True, hashes=LAUNCH_HASHES):
    return HeartbeatLine(seq, t, healthy, None if healthy else "score_wait", *hashes, (), recv_ms=t)


def test_a_silent_minute_after_an_unhealthy_heartbeat_stays_unhealthy_in_the_market_layer():
    # Healthy lines every 30 seconds to 10:00:30, an unhealthy one at 10:00:59, then nothing until 10:02:05, a gap the
    # 120 second age allows, two more unhealthy lines and two healthy ones from 10:03:10.
    ten = OPEN + 30 * MIN
    stamps = [(ten - 2 * MIN + i * 30 * SEC, True) for i in range(6)] + [
        (ten + 59 * SEC, False), (ten + 2 * MIN + 5 * SEC, False), (ten + 2 * MIN + 35 * SEC, False),
        (ten + 3 * MIN + 10 * SEC, True), (ten + 3 * MIN + 40 * SEC, True)]
    lines = [beat(i, t, ok) for i, (t, ok) in enumerate(stamps, start=1)]
    boundaries = [ten + i * MIN for i in range(6)]
    reasons, news = news_reasons(lines, boundaries)
    assert reasons == [None] + [UNHEALTHY_HEARTBEAT] * 4 + [None]
    assert news == [(ten + MIN, NEWS_UNHEALTHY, UNHEALTHY_HEARTBEAT), (ten + 5 * MIN, NEWS_HEALTHY, "")]
    # A line with other hashes, then a silent minute, stays a hash mismatch until a line with the launch hashes is in
    # force for a whole minute.
    other = ("d" * 64, *LAUNCH_HASHES[1:])
    changed = [beat(i, ten - (3 - i) * 30 * SEC) for i in range(4)] + [
        beat(4, ten + 59 * SEC, hashes=other), beat(5, ten + 2 * MIN + 5 * SEC, hashes=other)]
    reasons, news = news_reasons(changed, boundaries[:4])
    assert reasons == [None] + [HASH_MISMATCH] * 3
    assert news == [(ten + MIN, NEWS_UNHEALTHY, HASH_MISMATCH)]
