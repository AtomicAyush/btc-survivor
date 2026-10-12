"""Session bounded labels: training records pushed only when their horizon ends by the close, maturing only within one
epoch, the evaluation minutes of full and early close sessions, the stock_labels rows, and the same rows recomputed
from candles alone."""

import math
from datetime import date

import numpy as np
import pytest

from btc_survivor.clock import MS_PER_MINUTE
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.labels import (GAP, MATURED, PENDING, EvaluationLabels, LabelRow, SessionLabelQueue,
                                           fits, is_evaluation_minute, label_bps, make_label_scale, push_k,
                                           recompute_labels, stamp)
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml")
H = CFG.time.horizon_minutes
DAY = sessions.session_on(date(2026, 10, 13))
NEXT = sessions.session_on(date(2026, 10, 14))
WED = sessions.session_on(date(2026, 11, 25))
EARLY = sessions.session_on(date(2026, 11, 27))
MONDAY = sessions.session_on(date(2026, 11, 30))


def z(v):
    return np.full(10, float(v))


def k_of(session, minute):
    return session.first_k + minute - 1


def test_the_label_queue_stamps_k_minutes():
    assert H == 30 and stamp(5) == 5 * MS_PER_MINUTE
    q = SessionLabelQueue(H)
    k = k_of(DAY, 40)
    assert q.push(k, 0, z(1), 100.0)
    assert q.queue.pending[0].minute_ms == k * 60_000


@pytest.mark.parametrize("session, last_push", [(DAY, 360), (EARLY, 180)])
def test_records_are_pushed_only_when_the_horizon_ends_by_the_close(session, last_push):
    q = SessionLabelQueue(H)
    pushed = [m for m in range(1, session.minutes + 1) if q.push(k_of(session, m), 0, z(m), 100.0)]
    assert pushed == list(range(1, last_push + 1))
    assert len(q) == last_push
    # The last record pushed ends exactly at the close, 15:30 plus 30 minutes or 12:30 plus 30 on an early close.
    assert sessions.boundary_of_k(k_of(session, last_push)) + H * MS_PER_MINUTE == session.close_ms
    assert not fits(session.last_k, H) and fits(k_of(session, last_push), H)


def test_a_record_matures_at_k_plus_30_within_its_epoch():
    q = SessionLabelQueue(H)
    k = k_of(DAY, 100)
    q.push(k, 3, z(1), 100.0)
    for j in range(k + 1, k + H):
        assert q.mature(j, 3, 101.0) is None
    sample = q.mature(k + H, 3, 102.0)
    assert push_k(sample) == k and sample.label_bps == label_bps(100.0, 102.0) == 1e4 * math.log(1.02)
    assert np.array_equal(sample.z, z(1)) and q.discarded == 0 and len(q) == 0


def test_an_epoch_change_discards_the_record():
    q = SessionLabelQueue(H)
    k = k_of(DAY, 100)
    q.push(k, 3, z(1), 100.0)
    q.push(k + 1, 3, z(2), 100.0)
    assert q.mature(k + H, 4, 101.0) is None
    q.drop_other_epochs(4)
    assert len(q) == 0 and q.discarded == 2


def test_a_minute_without_a_reference_price_discards_the_records_due():
    q = SessionLabelQueue(H)
    k = k_of(DAY, 100)
    q.push(k, 0, z(1), 100.0)
    q.push(k + 1, 0, z(2), 100.0)
    assert q.mature(k + H, 0, None) is None
    assert len(q) == 1 and q.discarded == 1
    assert push_k(q.mature(k + H + 1, 0, 99.0)) == k + 1


def test_labels_never_cross_a_close():
    q = SessionLabelQueue(H)
    for m in range(1, DAY.minutes + 1):
        q.push(k_of(DAY, m), 0, z(m), 100.0 + m)
    matured = [q.mature(k_of(DAY, m), 0, 100.0 + m) for m in range(H + 1, DAY.minutes + 1)]
    assert [push_k(s) - DAY.first_k + 1 for s in matured] == list(range(1, 361))
    assert len(q) == 0
    # Session minute indexes run on across the night, but no record is pending into the next session.
    assert NEXT.first_k == DAY.last_k + 1
    q.push(k_of(DAY, 361), 0, z(0), 100.0)
    assert q.mature(k_of(NEXT, 1), 1, 100.0) is None and len(q) == 0


def test_a_record_pushed_before_an_epoch_bump_never_matures_after_the_close():
    q = SessionLabelQueue(H)
    q.push(k_of(DAY, 360), 5, z(1), 100.0)
    assert q.mature(DAY.last_k, 6, 101.0) is None              # the close bumped every epoch first
    assert q.discarded == 1


@pytest.mark.parametrize("session, minutes", [(DAY, list(range(30, 361, 30))), (EARLY, [30, 60, 90, 120, 150, 180])])
def test_evaluation_minutes_on_full_and_early_close_sessions(session, minutes):
    found = [m for m in range(1, session.minutes + 1) if is_evaluation_minute(k_of(session, m), H)]
    assert found == minutes == list(session.evaluation_minutes(H))
    assert len(found) == (12 if session.minutes == 390 else 6)


def run_rows(session, symbol, prices, healthy, epochs):
    """Step one symbol's evaluation labels through a session and return every row."""
    labels = EvaluationLabels(symbol, H)
    rows = []
    for m in range(1, session.minutes + 1):
        k = k_of(session, m)
        rows += labels.step(k, prices.get(k), healthy.get(k, True), epochs.get(k, 0))
    return rows, labels


def test_evaluation_rows_mature_within_one_healthy_epoch_and_end_as_gap_otherwise():
    prices = {k_of(DAY, m): 100.0 + m / 10 for m in range(1, 391)}
    healthy = {k_of(DAY, m): False for m in (59, 60, 61)}       # unhealthy around the minute 60 entry
    epochs = {k_of(DAY, m): 1 for m in range(59, 391)}          # leaving healthy bumped the epoch
    epochs.update({k_of(DAY, m): 2 for m in range(200, 391)})   # and a second bump inside the 180 to 210 horizon
    rows, labels = run_rows(DAY, "AAPL", prices, healthy, epochs)
    pending = [r for r in rows if r.status == PENDING]
    final = {r.ts: r for r in rows if r.status != PENDING}
    assert [r.ts for r in pending] == [DAY.boundary(m) for m in range(30, 361, 30)]
    assert sorted(final) == [r.ts for r in pending] and labels.pending == 0
    for m in range(30, 361, 30):
        row = final[DAY.boundary(m)]
        assert row.entry_price == prices[k_of(DAY, m)]
        if m in (30, 60, 180):
            assert (row.status, row.exit_price, row.label_bps) == (GAP, None, None)
        else:
            assert row.status == MATURED and row.exit_price == prices[k_of(DAY, m + 30)]
            assert row.label_bps == 1e4 * math.log(row.exit_price / row.entry_price)


def test_a_final_row_comes_before_the_new_pending_row_and_never_after_the_close():
    prices = {k_of(EARLY, m): 50.0 for m in range(1, 211)}
    rows, _ = run_rows(EARLY, "QQQ", prices, {}, {})
    assert len(rows) == 12
    b30, b60 = EARLY.boundary(30), EARLY.boundary(60)
    assert [(r.ts, r.status) for r in rows[:3]] == [(b30, PENDING), (b30, MATURED), (b60, PENDING)]
    assert max(r.ts for r in rows) == EARLY.boundary(180)
    assert all(r.label_bps in (None, 0.0) for r in rows)


def test_no_reference_price_at_an_evaluation_minute_writes_no_row_and_none_at_the_exit_ends_as_gap():
    prices = {k_of(DAY, m): 100.0 for m in range(1, 391) if m not in (30, 90)}
    rows, _ = run_rows(DAY, "MSFT", prices, {}, {})
    final = {r.ts: r for r in rows if r.status != PENDING}
    assert DAY.boundary(30) not in final
    assert final[DAY.boundary(60)].status == GAP                # its exit at minute 90 had no price
    assert final[DAY.boundary(120)].status == MATURED


def test_a_row_whose_exit_minute_was_never_stepped_ends_as_gap():
    labels = EvaluationLabels("AAPL", H)
    k = k_of(DAY, 30)
    [row] = labels.step(k, 100.0, True, 0)
    assert row == LabelRow(DAY.boundary(30), "AAPL", 100.0, None, None, PENDING)
    [gap] = labels.step(k + H + 1, 101.0, True, 0)
    assert gap.status == GAP and gap.ts == row.ts


def candles_and_epochs(seed, session):
    """One symbol's candles over a session with random unhealthy stretches, and the epoch at each minute, bumped on
    each exit from healthy as price health does."""
    rng = np.random.default_rng(seed)
    candles, healthy, epochs = [], {}, {}
    epoch, was, ok, p = 0, False, True, 100.0
    for m in range(1, session.minutes + 1):
        k = k_of(session, m)
        p *= math.exp(rng.normal(0, 0.001))
        ok = rng.random() > (0.01 if ok else 0.3)
        ref = None if rng.random() < 0.01 else p
        # A healthy minute always has an eligible print, so a minute without a reference price is not healthy.
        if was and not (ok and ref is not None):
            epoch += 1
        was = ok and ref is not None
        candles.append(StockCandle("NVDA", session.boundary(m), k, ref, ref, ref, ref, ref, 1.0, 0.5, 0.5, 1, 1, 0,
                                   was))
        healthy[k], epochs[k] = was, epoch
    return candles, healthy, epochs


@pytest.mark.parametrize("seed, session", [(1, DAY), (2, EARLY), (3, MONDAY), (4, WED)])
def test_rows_recomputed_from_the_candles_match_the_live_rows(seed, session):
    candles, healthy, epochs = candles_and_epochs(seed, session)
    labels = EvaluationLabels("NVDA", H)
    live = []
    for c in candles:
        live += [r for r in labels.step(c.k, c.ref_price, healthy[c.k], epochs[c.k]) if r.status != PENDING]
    assert live == recompute_labels(candles, H)
    assert {r.status for r in live} == {MATURED, GAP}


def test_the_label_scale_shares_the_standardizer_half_life():
    scale = make_label_scale(CFG.features)
    assert scale.min_samples == 30 and not scale.ready
    assert scale.stats.alpha == pytest.approx(1 - 0.5 ** (1 / 10080))
