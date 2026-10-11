from btc_survivor.feed.health import HEALTHY, RECOVERING, STALE, HealthMonitor

M = 1_791_000_000_000  # a minute boundary
MIN = 60_000
SEC = 1_000


def connected_with_snapshot(at=M + 2 * SEC):
    h = HealthMonitor(stale_after_ms=5 * SEC, max_skew_ms=2 * SEC)
    h.on_connected(at)
    h.on_snapshot(at + 100, valid=True)
    return h


def feed(h, start, end, every=500, skew=20):
    """Messages and good book updates every 500 ms with a given clock offset."""
    for t in range(start, end, every):
        h.on_message(t, t - skew)
        h.on_book_update(t)


def test_starts_stale_and_needs_a_full_clean_minute_after_the_snapshot():
    h = HealthMonitor(5 * SEC, 2 * SEC)
    assert h.state == STALE and not h.judge_minute(M)
    h = connected_with_snapshot(M + 2 * SEC)
    assert h.state == RECOVERING
    feed(h, M + 2 * SEC, M + MIN)
    assert not h.judge_minute(M + MIN)  # the snapshot arrived inside this minute
    feed(h, M + MIN, M + 2 * MIN)
    assert h.judge_minute(M + 2 * MIN) and h.state == HEALTHY
    feed(h, M + 2 * MIN, M + 3 * MIN)
    assert h.judge_minute(M + 3 * MIN)


def test_silence_goes_stale_and_bumps_the_epoch():
    h = connected_with_snapshot()
    feed(h, M + 2 * SEC, M + 2 * MIN)
    assert h.judge_minute(M + 2 * MIN)
    epoch = h.epoch
    feed(h, M + 2 * MIN, M + 2 * MIN + 10 * SEC)
    assert not h.check(M + 2 * MIN + 14 * SEC)
    assert h.check(M + 2 * MIN + 16 * SEC) and h.state == STALE
    assert h.epoch == epoch + 1
    assert not h.judge_minute(M + 3 * MIN)


def test_messages_without_book_updates_still_go_stale():
    h = connected_with_snapshot()
    for t in range(M + 2 * SEC, M + 20 * SEC, 500):
        h.on_message(t, t - 20)
    assert h.check(M + 20 * SEC) and h.state == STALE


def test_lost_book_needs_a_new_snapshot_and_another_full_minute():
    h = connected_with_snapshot()
    feed(h, M + 2 * SEC, M + 2 * MIN)
    assert h.judge_minute(M + 2 * MIN)
    h.on_book_lost(M + 2 * MIN + 30 * SEC, "gap", "expected 5, got 7")
    assert h.state == RECOVERING and h.epoch == 1
    h.on_disconnected(M + 2 * MIN + 30 * SEC)
    h.on_connected(M + 2 * MIN + 31 * SEC)
    h.on_snapshot(M + 2 * MIN + 32 * SEC, valid=True)
    feed(h, M + 2 * MIN + 32 * SEC, M + 4 * MIN)
    assert not h.judge_minute(M + 3 * MIN)
    assert h.judge_minute(M + 4 * MIN)
    kinds = [k for _, k, _ in h.drain_events()]
    assert "gap" in kinds and kinds.count("snapshot") == 2


def test_epoch_only_moves_when_leaving_healthy():
    h = connected_with_snapshot()
    h.on_book_lost(M + 3 * SEC, "crossed")
    h.on_snapshot(M + 4 * SEC, valid=True)
    assert h.epoch == 0  # never was healthy


def test_crossed_snapshot_is_a_lost_book():
    h = HealthMonitor(5 * SEC, 2 * SEC)
    h.on_connected(M)
    h.on_snapshot(M + 1, valid=False)
    assert not h.book_ok and h.good_since is None


def test_clock_skew_holds_the_feed_out_of_healthy_until_a_clean_minute():
    h = connected_with_snapshot()
    feed(h, M + 2 * SEC, M + 2 * MIN)
    assert h.judge_minute(M + 2 * MIN)
    feed(h, M + 2 * MIN, M + 3 * MIN, skew=4 * SEC)
    assert h.state == RECOVERING and not h.skew_ok
    assert not h.judge_minute(M + 3 * MIN)
    feed(h, M + 3 * MIN, M + 4 * MIN)  # skew clears partway through
    assert h.skew_ok and not h.judge_minute(M + 4 * MIN)
    feed(h, M + 4 * MIN, M + 5 * MIN)
    assert h.judge_minute(M + 5 * MIN)


def test_new_connection_gets_the_full_limit_to_deliver_a_snapshot():
    h = connected_with_snapshot()
    h.on_disconnected(M + 10 * SEC)
    assert h.check(M + 30 * SEC)  # long backoff, stale
    h.on_connected(M + 30 * SEC)
    assert not h.check(M + 33 * SEC)
    assert h.check(M + 36 * SEC)  # no snapshot within the limit


def test_events_from_one_message_count_as_one_skew_sample():
    h = connected_with_snapshot()
    feed(h, M + 2 * SEC, M + 2 * MIN)
    assert h.judge_minute(M + 2 * MIN)
    late = M + 2 * MIN + 10 * SEC
    for _ in range(30):  # one market_trades message carrying 30 trades, 3 s late
        h.on_message(late, late - 3 * SEC)
    feed(h, late + 500, M + 3 * MIN)
    assert h.skew_ok and h.judge_minute(M + 3 * MIN)
