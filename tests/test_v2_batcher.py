"""The batching rule of PLAN_V2 section 8.3 on fixed instants. Finnhub items flush at 20 waiting or once the oldest has
waited 60 seconds, at any hour, every official item goes alone at once and never with a Finnhub item, the order is
fixed by receive time and seq, an item expires 300 seconds after its receipt, and no Finnhub item is batched while
send_to_llm is false."""

import dataclasses
import random
from datetime import datetime, timezone

import pytest

from btc_survivor.clock import MS_PER_HOUR, MS_PER_SECOND, iso
from btc_survivor.v2.config import OFFICIAL_KINDS, load_news_config
from btc_survivor.v2.news.batcher import (
    BATCH, EXPIRED, SINGLE, SKIPPED_OFF, Batch, Batcher, BatchSettings, Pending, Taken, split)
from btc_survivor.v2.news.collector import STALE_VENDOR
from btc_survivor.v2.news.prefilter import route
from helpers import ROOT

CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
HAIKU = BatchSettings.haiku(CONFIG)
SONNET = BatchSettings.sonnet(CONFIG)
SOURCE = {"8k": "edgar", "fed_press": "fed", "fed_monetary": "fed", "fed_speech": "fed", "bls_release": "bls"}
SECOND, MINUTE = MS_PER_SECOND, 60 * MS_PER_SECOND


def utc(y, mo, d, h=0, mi=0, s=0):
    return int(datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp()) * MS_PER_SECOND


T0 = utc(2026, 10, 14, 14)            # 10:00 ET on a Wednesday


def finnhub(seq, at, vendor=None, route_to="llm", status="new"):
    return Pending(seq, "finnhub", "media", route_to, status, iso(at), None if vendor is None else iso(vendor))


def official(seq, at, kind="8k"):
    return Pending(seq, SOURCE[kind], kind, "llm", "new", iso(at), iso(at - MINUTE))


def drain_due(batcher, now):
    """Every batch due at one instant, in the order take releases them."""
    batches = []
    while (taken := batcher.take(now)).batch is not None:
        assert taken.expired == ()
        batches.append(taken.batch)
    return batches


def test_settings_come_from_the_config():
    assert HAIKU == BatchSettings(kinds=None, alone_kinds=OFFICIAL_KINDS, send_to_llm=True, flush_max_items=20,
                                  flush_after_ms=60 * SECOND, max_queue_age_ms=300 * SECOND, max_vendor_age_hours=24)
    assert SONNET == dataclasses.replace(HAIKU, kinds=OFFICIAL_KINDS)
    narrow = dataclasses.replace(CONFIG.llm, max_items_per_call=12)
    assert BatchSettings.haiku(dataclasses.replace(CONFIG, llm=narrow)).flush_max_items == 12


# The flush at 20 items and 60 seconds


def test_a_batch_goes_the_moment_twenty_items_wait():
    batcher = Batcher(HAIKU)
    for seq in range(1, 20):
        assert batcher.admit(finnhub(seq, T0 + seq * SECOND), T0 + seq * SECOND) is None
    assert batcher.take(T0 + 20 * SECOND) == Taken((), None)
    assert batcher.admit(finnhub(20, T0 + 20 * SECOND), T0 + 20 * SECOND) is None
    taken = batcher.take(T0 + 20 * SECOND)
    assert taken.batch == Batch(BATCH, tuple(finnhub(seq, T0 + seq * SECOND) for seq in range(1, 21)))
    assert len(batcher) == 0


def test_forty_five_waiting_items_go_as_two_full_batches_and_the_rest_waits():
    batcher = Batcher(HAIKU)
    for seq in range(1, 46):
        batcher.admit(finnhub(seq, T0), T0)
    batches = drain_due(batcher, T0 + SECOND)
    assert [batch.seqs for batch in batches] == [tuple(range(1, 21)), tuple(range(21, 41))]
    assert len(batcher) == 5
    assert batcher.take(T0 + 60 * SECOND - 1).batch is None
    assert batcher.take(T0 + 60 * SECOND).batch.seqs == (41, 42, 43, 44, 45)


@pytest.mark.parametrize("start", [T0, utc(2026, 10, 18, 3), utc(2026, 11, 26, 21, 30)],
                         ids=["session", "sunday_night", "holiday_evening"])
def test_a_batch_goes_once_the_oldest_has_waited_sixty_seconds_at_any_hour(start):
    batcher = Batcher(HAIKU)
    for seq, offset in ((1, 0), (2, 10), (3, 59)):
        batcher.admit(finnhub(seq, start + offset * SECOND), start + offset * SECOND)
    assert batcher.take(start + 60 * SECOND - 1) == Taken((), None)
    assert batcher.take(start + 60 * SECOND).batch == Batch(BATCH, (
        finnhub(1, start), finnhub(2, start + 10 * SECOND), finnhub(3, start + 59 * SECOND)))
    assert batcher.take(start + 10 * MINUTE) == Taken((), None)


# Official items alone


@pytest.mark.parametrize("kind", OFFICIAL_KINDS)
def test_an_official_item_goes_alone_at_once_to_each_model(kind):
    for settings in (HAIKU, SONNET):
        batcher = Batcher(settings)
        item = official(7, T0, kind)
        assert batcher.admit(item, T0) is None
        assert batcher.take(T0) == Taken((), Batch(SINGLE, (item,)))


def test_official_items_never_share_a_call_with_finnhub_items():
    rng = random.Random(15)
    batcher = Batcher(HAIKU)
    seen, wanted_alone, batches, now = set(), set(), [], T0
    for seq in range(1, 2001):
        now += rng.randint(0, 4 * SECOND)
        item = official(seq, now, rng.choice(OFFICIAL_KINDS)) if rng.random() < 0.1 else finnhub(seq, now)
        if item.source != "finnhub":
            wanted_alone.add(seq)
        assert batcher.admit(item, now) is None
        if rng.random() < 0.3:
            batches += drain_due(batcher, now)
    batches += drain_due(batcher, now + 60 * SECOND)
    officials = set()
    for batch in batches:
        sources = {item.source for item in batch.items}
        if batch.kind == SINGLE:
            assert len(batch.items) == 1 and sources != {"finnhub"}
            officials.update(batch.seqs)
        else:
            assert batch.kind == BATCH and sources == {"finnhub"} and 1 <= len(batch.items) <= 20
        seen.update(batch.seqs)
    assert seen == set(range(1, 2001)) and len(batcher) == 0
    assert officials == wanted_alone and len(officials) > 150


def test_sonnet_takes_only_its_official_kinds_and_never_a_finnhub_item():
    batcher = Batcher(SONNET)
    assert not batcher.wants(finnhub(1, T0))
    with pytest.raises(ValueError, match="not routed to this model"):
        batcher.admit(finnhub(1, T0), T0)
    fewer = Batcher(dataclasses.replace(SONNET, kinds=("8k",)))
    assert fewer.wants(official(2, T0, "8k")) and not fewer.wants(official(3, T0, "fed_speech"))
    for seq, kind in enumerate(OFFICIAL_KINDS, 10):
        batcher.admit(official(seq, T0 + seq, kind), T0 + seq)
    batches = drain_due(batcher, T0 + MINUTE)
    assert [(batch.kind, batch.seqs) for batch in batches] == [(SINGLE, (seq,)) for seq in range(10, 15)]


def test_only_new_llm_items_are_admitted():
    batcher = Batcher(HAIKU)
    for item in (finnhub(1, T0, route_to="none", status="seed"), finnhub(2, T0, route_to="finbert"),
                 finnhub(3, T0, status="backlog"), dataclasses.replace(official(4, T0), route="none")):
        assert not batcher.wants(item)
        with pytest.raises(ValueError, match="not routed"):
            batcher.admit(item, T0)
    assert batcher.admit(finnhub(5, T0), T0) is None
    with pytest.raises(ValueError, match="already admitted"):
        batcher.admit(finnhub(5, T0), T0)


# Deterministic order


def test_the_order_is_receive_time_then_seq_whatever_the_admission_order():
    items = [finnhub(seq, T0 + (seq % 7) * SECOND) for seq in range(1, 41)]
    items += [official(seq, T0 + (seq % 7) * SECOND, OFFICIAL_KINDS[seq % 5]) for seq in range(41, 49)]
    runs = []
    for seed in range(5):
        shuffled = items[:]
        random.Random(seed).shuffle(shuffled)
        batcher = Batcher(HAIKU)
        for item in shuffled:
            batcher.admit(item, T0 + 7 * SECOND)
        runs.append([(batch.kind, batch.seqs) for batch in drain_due(batcher, T0 + 2 * MINUTE)])
    assert all(run == runs[0] for run in runs)
    first = runs[0]
    by_order = sorted(items, key=lambda item: (item.received_ms, item.seq))
    # Items 7, 14, 21, 28, 35 and the filing 42 came first, so the Finnhub batch holding 7 goes before the filing.
    assert first[0] == (BATCH, tuple(item.seq for item in by_order if item.source == "finnhub")[:20])
    finnhub_seqs = [seq for kind, seqs in first if kind == BATCH for seq in seqs]
    assert finnhub_seqs == [item.seq for item in by_order if item.source == "finnhub"]
    official_seqs = [seqs[0] for kind, seqs in first if kind == SINGLE]
    assert official_seqs == [item.seq for item in by_order if item.source != "finnhub"]


def test_two_due_batches_go_oldest_first():
    batcher = Batcher(HAIKU)
    batcher.admit(finnhub(1, T0), T0)
    batcher.admit(official(2, T0 + 30 * SECOND), T0 + 30 * SECOND)
    assert batcher.take(T0 + 30 * SECOND).batch.seqs == (2,)        # the Finnhub item is not due yet
    batcher.admit(official(3, T0 + 61 * SECOND), T0 + 61 * SECOND)
    assert [batch.seqs for batch in drain_due(batcher, T0 + 61 * SECOND)] == [(1,), (3,)]


# Expiry after 300 seconds and the vendor guard


def test_an_item_received_300_seconds_ago_or_more_expires_at_once():
    batcher = Batcher(HAIKU)
    assert batcher.admit(finnhub(1, T0), T0 + 300 * SECOND - 1) is None
    assert batcher.admit(finnhub(2, T0), T0 + 300 * SECOND) == EXPIRED
    assert batcher.admit(official(3, T0), T0 + 301 * SECOND) == EXPIRED
    assert len(batcher) == 1


def test_waiting_items_expire_300_seconds_after_their_receipt():
    batcher = Batcher(HAIKU)
    first, second, third, filing = finnhub(1, T0), finnhub(2, T0 + SECOND), finnhub(3, T0 + 2 * SECOND), official(4, T0)
    for item in (first, second, third, filing):
        batcher.admit(item, T0 + 2 * SECOND)
    assert batcher.take(T0 + 301 * SECOND) == Taken((first, filing, second), Batch(BATCH, (third,)))
    assert len(batcher) == 0


def test_a_vendor_time_past_24_hours_gives_stale_vendor_at_once():
    batcher = Batcher(HAIKU)
    assert batcher.admit(finnhub(1, T0, vendor=T0 - 24 * MS_PER_HOUR), T0) is None
    assert batcher.admit(finnhub(2, T0, vendor=T0 - 24 * MS_PER_HOUR - 1), T0) == STALE_VENDOR
    assert batcher.admit(finnhub(3, T0, vendor=None), T0) is None
    assert len(batcher) == 2


# send_to_llm false


def test_no_finnhub_item_is_batched_while_send_to_llm_is_false():
    config = dataclasses.replace(CONFIG, finnhub=dataclasses.replace(CONFIG.finnhub, send_to_llm=False))
    settings = BatchSettings.haiku(config)
    assert not settings.send_to_llm
    batcher = Batcher(settings)
    assert route("finnhub", "media", "new", "title", False) == "finbert"
    with pytest.raises(ValueError, match="not routed"):
        batcher.admit(finnhub(1, T0, route_to="finbert"), T0)
    batches = []
    for seq in range(2, 62):            # items still routed llm from before the switch go to FinBERT alone
        assert batcher.admit(finnhub(seq, T0 + seq * SECOND), T0 + seq * SECOND) == SKIPPED_OFF
        batches += drain_due(batcher, T0 + seq * SECOND)
    batcher.admit(official(70, T0 + 70 * SECOND, "fed_monetary"), T0 + 70 * SECOND)
    batches += drain_due(batcher, T0 + 70 * SECOND)
    batches += drain_due(batcher, T0 + 4 * MINUTE)
    assert [(batch.kind, batch.seqs) for batch in batches] == [(SINGLE, (70,))]


# Splits, requeues and draining


@pytest.mark.parametrize("n", range(2, 26))
def test_a_split_gives_the_first_half_floor_n_over_2_items(n):
    first, second = split(tuple(range(n)))
    assert (first, second) == (tuple(range(n // 2)), tuple(range(n // 2, n)))


def test_a_single_item_cannot_be_split():
    with pytest.raises(ValueError):
        split((finnhub(1, T0),))


def test_a_requeued_item_returns_in_receive_order_and_drain_empties_the_queue():
    batcher = Batcher(HAIKU)
    for seq in range(1, 21):
        batcher.admit(finnhub(seq, T0 + seq * SECOND), T0 + 20 * SECOND)
    batch = batcher.take(T0 + 20 * SECOND).batch
    batcher.admit(finnhub(21, T0 + 30 * SECOND), T0 + 30 * SECOND)
    batcher.requeue(batch.items[4])
    with pytest.raises(ValueError, match="never admitted or is still waiting"):
        batcher.requeue(batch.items[4])
    with pytest.raises(ValueError, match="never admitted or is still waiting"):
        batcher.requeue(finnhub(99, T0))
    assert batcher.take(T0 + 65 * SECOND).batch.seqs == (5, 21)
    batcher.admit(finnhub(22, T0 + 66 * SECOND), T0 + 66 * SECOND)
    batcher.admit(official(23, T0 + 66 * SECOND), T0 + 66 * SECOND)
    assert [item.seq for item in batcher.drain()] == [22, 23]
    assert len(batcher) == 0 and batcher.take(T0 + 10 * MINUTE) == Taken((), None)


def test_pending_reads_a_news_items_row():
    row = {"seq": 9, "source": "fed", "kind": "fed_monetary", "route": "llm", "status": "new",
           "received_at": "2026-10-28T18:00:01.250Z", "vendor_ts": None}
    item = Pending.from_row(row)
    assert item == Pending(9, "fed", "fed_monetary", "llm", "new", "2026-10-28T18:00:01.250Z")
    assert item.received_ms == utc(2026, 10, 28, 18, 0, 1) + 250
