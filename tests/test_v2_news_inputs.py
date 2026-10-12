"""News lines, the score streams and the decayed news inputs on hand written lines: decay in session minutes, a score
counting only from its receipt and never before scored_at, full weight to the open, the shrinkage, the index symbol's
aggregation, fallbacks and paired streams visible only at the later of two receipts, Sonnet never in the primary
stream, exactly one primary score per item, and inputs stored at each push minute, zero in S1."""

import itertools
import json
from datetime import date

import numpy as np
import pytest

from btc_survivor.clock import MS_PER_HOUR, MS_PER_MINUTE, iso
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_news_config, load_stock_config
from btc_survivor.v2.news import health
from btc_survivor.v2.stocks.news_inputs import (FINBERT_STREAM, FINBERT_MATCHED, HAIKU, HAIKU_OFFICIAL, NEWS, SONNET,
                                                STREAMS, EngineInputs, HeartbeatLine, Impact, NewsInputs, ScoreLine,
                                                ScoreStreams, Scorers, derive_streams, session_minutes)
from helpers import ROOT

S1 = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml")
S2 = load_stock_config(ROOT / "configs" / "v2" / "s2_news.toml")
DAY = sessions.session_on(date(2026, 10, 13))
NEXT = sessions.session_on(date(2026, 10, 14))
THIRD = sessions.session_on(date(2026, 10, 15))
SYMBOLS = ("AAPL", "MSFT", "QQQ")
HK, SN, FB = Scorers()
STATUSES = ("scored", "irrelevant", "refused", "malformed", "failed", "skipped_cap", "skipped_off", "expired",
            "stale_vendor", "untagged")


def line(seq, item, scorer, recv, scored=None, status="scored", route="llm", impacts=(("AAPL", 0.6, True),),
         kind="media", source="finnhub"):
    """A score line received at recv, scored one second earlier unless given."""
    scored = recv - 1000 if scored is None else scored
    tickers = tuple(sorted({t for t, _, _ in impacts}))
    return ScoreLine(seq, item, source, kind, route, scorer, status, scored, tickers,
                     tuple(Impact(t, s, n) for t, s, n in impacts), recv)


def inputs(symbols=SYMBOLS):
    return NewsInputs(symbols, "QQQ", 120.0, 5.0)


def b(session, minute):
    return session.boundary(minute)


def w(age):
    return 2.0 ** (-age / 120)


# Lines


def test_a_score_line_round_trips_without_headline_text():
    original = line(7, 3, HK, b(DAY, 50) + 250, impacts=(("AAPL", 0.6, True), ("MACRO", -0.4, False)))
    raw = original.encode()
    assert ScoreLine.decode(raw, original.recv_ms) == original
    keys = set(json.loads(raw))
    assert keys == {"seq", "item_seq", "source", "kind", "route", "scorer", "status", "scored_at", "tickers",
                    "impacts"}
    assert "title" not in raw and "summary" not in raw and "headline" not in raw
    finbert = line(8, 3, FB, b(DAY, 50) + 300, impacts=(("AAPL", -0.25, None),))
    assert ScoreLine.decode(finbert.encode(), finbert.recv_ms).impacts[0].is_new is None


def test_a_score_line_from_database_rows():
    score = {"seq": 11, "item_seq": 4, "scorer": "haiku_v1", "route": "llm", "status": "scored",
             "scored_at": iso(b(DAY, 20))}
    item = {"source": "edgar", "kind": "8k", "tickers": "AAPL"}
    impacts = [{"ticker": "AAPL", "strength": 0.8, "is_new": 1}, {"ticker": "QQQ", "strength": 0.2, "is_new": 0}]
    got = ScoreLine.from_rows(score, item, impacts)
    assert got == ScoreLine(11, 4, "edgar", "8k", "llm", "haiku_v1", "scored", b(DAY, 20), ("AAPL",),
                            (Impact("AAPL", 0.8, True), Impact("QQQ", 0.2, False)))
    assert [i.counted for i in got.impacts] == [True, False]


def test_a_heartbeat_line_round_trips_and_checks_the_launch_hashes():
    row = {"seq": 5, "ts": iso(b(DAY, 9)), "healthy": 1, "reason": None, "config_hash": "c", "prompt_sha256": "p",
           "schema_sha256": "s", "channels_down": "bls_cpi,fed_speeches"}
    hb = HeartbeatLine.from_row(row)
    assert hb.channels_down == ("bls_cpi", "fed_speeches") and hb.healthy
    got = HeartbeatLine.decode(hb.encode(), b(DAY, 9) + 40)
    assert got.recv_ms == b(DAY, 9) + 40 and got.seq == 5 and got.hashes == ("c", "p", "s")
    assert got.heartbeat(("c", "p", "s")) == (b(DAY, 9) + 40, True, True)
    assert got.heartbeat(("c", "other", "s")).hashes_match is False


# Decay


def test_session_minutes_run_on_the_k_scale_and_stand_still_outside_the_session():
    assert session_minutes(b(DAY, 1)) == DAY.first_k
    assert session_minutes(DAY.close_ms) == DAY.last_k
    assert session_minutes(DAY.close_ms + 4 * MS_PER_HOUR) == DAY.last_k
    assert session_minutes(NEXT.open_ms - 1) == DAY.last_k
    assert session_minutes(NEXT.open_ms) == NEXT.first_k - 1 == DAY.last_k
    assert session_minutes(b(DAY, 100) + 30_000) == DAY.first_k + 99.5


def test_weights_decay_in_session_minutes_across_the_night():
    n = inputs()
    n.add(line(1, 1, FB, b(DAY, 300) + 500, scored=b(DAY, 300), route="finbert", impacts=(("AAPL", 0.5, None),)))
    at = n.step(b(DAY, 360))[NEWS]["AAPL"]
    assert at.company == pytest.approx(w(60) * 0.5 / (w(60) + 1)) and at.company_items == 1
    at = n.step(b(NEXT, 30))[NEWS]["AAPL"]                     # 90 minutes to the close and 30 after the open
    assert at.company == pytest.approx(0.5 * 0.5 / 1.5)


def test_a_score_counts_only_from_its_receipt_and_never_before_scored_at():
    n = inputs()
    n.add(line(1, 1, FB, b(DAY, 40), route="finbert"))                              # received at the boundary
    n.add(line(2, 2, FB, b(DAY, 40) - 1, route="finbert", impacts=(("MSFT", 1.0, None),)))
    n.add(line(3, 3, FB, b(DAY, 39), scored=b(DAY, 41), route="finbert", impacts=(("QQQ", 1.0, None),)))
    at = n.step(b(DAY, 40))[NEWS]
    assert at["AAPL"].company_items == 0 and at["MSFT"].company_items == 1 and at["QQQ"].company_items == 1
    assert at["MSFT"].newest_ms == b(DAY, 40) - 1
    at = n.step(b(DAY, 41))[NEWS]
    assert at["AAPL"].company_items == 1 and at["QQQ"].company_items == 2           # QQQ also takes AAPL and MSFT
    at = n.step(b(DAY, 42))[NEWS]
    assert at["QQQ"].company_items == 3 and at["QQQ"].newest_ms == b(DAY, 41)


def test_a_score_after_the_close_keeps_full_weight_until_the_open():
    n = inputs()
    for i, scored in enumerate((DAY.close_ms, DAY.close_ms + 3 * MS_PER_HOUR, NEXT.open_ms - 20 * MS_PER_MINUTE)):
        n.add(line(i + 1, i + 1, FB, scored + 500, scored=scored, route="finbert", impacts=(("MACRO", 0.3, None),)))
    at = n.step(b(NEXT, 1))[NEWS]["AAPL"]
    assert at.macro_items == 3 and at.macro == pytest.approx(3 * w(1) * 0.3 / (3 * w(1) + 1))


def test_the_extra_one_shrinks_toward_zero_and_no_news_gives_exactly_zero():
    n = inputs()
    at = n.step(b(DAY, 10))
    assert all(v.company == 0.0 and v.macro == 0.0 and v.newest_ms is None for s in STREAMS for v in at[s].values())
    n.add(line(1, 1, FB, b(DAY, 100) + 500, scored=b(DAY, 100), route="finbert", impacts=(("AAPL", 0.8, None),)))
    n.add(line(2, 2, FB, b(DAY, 220) + 500, scored=b(DAY, 220), route="finbert", impacts=(("AAPL", -0.2, None),)))
    at = n.step(b(DAY, 221))[NEWS]["AAPL"]
    want = (w(121) * 0.8 + w(1) * -0.2) / (w(121) + w(1) + 1)
    assert at.company == pytest.approx(want, rel=1e-14) and at.macro == 0.0


def test_the_index_symbol_takes_its_own_and_every_company_impact_and_macro_reaches_every_symbol():
    n = inputs()
    n.add(line(1, 1, FB, b(DAY, 50), route="finbert", impacts=(("AAPL", 0.4, None), ("QQQ", 0.6, None))))
    n.add(line(2, 2, FB, b(DAY, 50) + 1, route="finbert", impacts=(("MSFT", -0.2, None),)))
    n.add(line(3, 3, FB, b(DAY, 50) + 2, route="finbert", impacts=(("MACRO", 0.5, None),)))
    n.add(line(4, 4, FB, b(DAY, 50) + 3, route="finbert", impacts=(("TSLA", 0.9, None),)))   # not in this universe
    at = n.step(b(DAY, 51))[NEWS]
    # Each line was scored one second before its receipt.
    w1, w2, w3 = (w(session_minutes(b(DAY, 51)) - session_minutes(b(DAY, 50) + i - 1000)) for i in range(3))
    assert at["AAPL"].company == pytest.approx(w1 * 0.4 / (w1 + 1)) and at["AAPL"].company_items == 1
    assert at["MSFT"].company == pytest.approx(w2 * -0.2 / (w2 + 1))
    assert at["QQQ"].company == pytest.approx((w1 * 0.4 + w1 * 0.6 + w2 * -0.2) / (2 * w1 + w2 + 1))
    assert at["QQQ"].company_items == 3
    assert all(v.macro == pytest.approx(w3 * 0.5 / (w3 + 1)) and v.macro_items == 1 for v in at.values())


def test_a_model_impact_counts_only_when_marked_new():
    n = inputs()
    n.add(line(1, 1, HK, b(DAY, 50), impacts=(("AAPL", 0.8, False), ("MSFT", 0.6, True))))
    at = n.step(b(DAY, 51))[NEWS]
    assert at["AAPL"].company_items == 0 and at["AAPL"].company == 0.0
    assert at["MSFT"].company_items == 1


def test_impacts_older_than_five_half_lives_are_dropped():
    n = inputs()
    n.add(line(1, 1, FB, b(DAY, 10) + 500, scored=b(DAY, 10), route="finbert"))
    assert n.step(b(NEXT, 220))[NEWS]["AAPL"].company == pytest.approx(w(600) * 0.6 / (w(600) + 1))
    assert len(n.kept_lines()) == 1
    assert n.step(b(NEXT, 221))[NEWS]["AAPL"].company_items == 0
    assert n.kept_lines() == [] and n.streams.items == {}


def test_steps_only_go_forward():
    n = inputs()
    n.step(b(DAY, 5))
    with pytest.raises(ValueError, match="not after"):
        n.step(b(DAY, 5))


# Streams


def test_a_fallback_is_visible_only_at_the_later_of_its_two_receipts():
    t1, t2 = b(DAY, 50) + 5_000, b(DAY, 52) + 5_000
    for first, second in ((line(1, 1, FB, t1, impacts=(("AAPL", 0.3, None),)), line(2, 1, HK, t2, status="failed")),
                          (line(1, 1, HK, t1, status="failed"), line(2, 1, FB, t2, impacts=(("AAPL", 0.3, None),)))):
        n = inputs()
        assert [s.stream for s in n.add(first)] in ([FINBERT_STREAM], [])
        [primary] = [s for s in n.add(second) if s.stream == NEWS]
        assert primary.fallback and primary.scorer == FB and primary.visible_ms == t2
        assert primary.scored_ms == (first if first.scorer == FB else second).scored_ms
        assert n.step(b(DAY, 52))[NEWS]["AAPL"].company_items == 0
        assert n.step(b(DAY, 53))[NEWS]["AAPL"].company_items == 1
        assert n.step(b(DAY, 54))[FINBERT_STREAM]["AAPL"].company_items == 1


def test_haiku_scored_or_irrelevant_is_the_primary_score_at_its_own_receipt():
    for status in ("scored", "irrelevant"):
        n = inputs()
        n.add(line(1, 1, FB, b(DAY, 50), impacts=(("AAPL", -0.9, None),)))
        impacts = (("AAPL", 0.6, True),) if status == "scored" else ()
        scores = n.add(line(2, 1, HK, b(DAY, 51), status=status, impacts=impacts))
        [primary] = [s for s in scores if s.stream == NEWS]
        assert primary.scorer == HK and not primary.fallback and primary.visible_ms == b(DAY, 51)
        assert len(primary.impacts) == len(impacts)


def paired(scores, stream):
    return [s for s in scores if s.stream == stream]


def test_finbert_matched_holds_only_items_haiku_scored_at_the_later_receipt():
    lines = [line(1, 1, HK, b(DAY, 50)), line(2, 1, FB, b(DAY, 52), impacts=(("AAPL", 0.1, None),)),
             line(3, 2, FB, b(DAY, 53)), line(4, 2, HK, b(DAY, 51), status="refused"),
             line(5, 3, FB, b(DAY, 54)), line(6, 3, HK, b(DAY, 55), status="irrelevant", impacts=()),
             line(7, 4, FB, b(DAY, 56))]
    out = derive_streams(lines)
    assert [(s.item_seq, s.visible_ms, s.scorer) for s in out[FINBERT_MATCHED]] == [
        (1, b(DAY, 52), FB), (3, b(DAY, 55), FB)]
    assert [s.item_seq for s in out[FINBERT_STREAM]] == [1, 2, 3, 4]
    assert [s.item_seq for s in out[HAIKU]] == [1, 3]


def test_sonnet_and_haiku_official_hold_only_items_both_models_scored_at_the_later_receipt():
    official = {"kind": "8k", "source": "edgar"}
    lines = [line(1, 1, SN, b(DAY, 50), **official, impacts=(("AAPL", -0.4, True),)),
             line(2, 1, HK, b(DAY, 53), **official, impacts=(("AAPL", 0.8, True),)),
             line(3, 2, HK, b(DAY, 54), **official), line(4, 2, SN, b(DAY, 55), status="failed", **official),
             line(5, 3, SN, b(DAY, 56), **official), line(6, 3, HK, b(DAY, 57), status="skipped_off", **official),
             line(7, 4, HK, b(DAY, 58), **official), line(8, 4, SN, b(DAY, 60), status="irrelevant", impacts=(),
                                                           **official)]
    out = derive_streams(lines)
    assert [(s.item_seq, s.scorer, s.visible_ms, s.scored_ms) for s in out[SONNET]] == [
        (1, SN, b(DAY, 53), b(DAY, 50) - 1000), (4, SN, b(DAY, 60), b(DAY, 60) - 1000)]
    assert [(s.item_seq, s.scorer, s.visible_ms, s.scored_ms) for s in out[HAIKU_OFFICIAL]] == [
        (1, HK, b(DAY, 53), b(DAY, 53) - 1000), (4, HK, b(DAY, 60), b(DAY, 58) - 1000)]
    assert out[SONNET][0].impacts == (Impact("AAPL", -0.4, True),)
    assert out[HAIKU_OFFICIAL][0].impacts == (Impact("AAPL", 0.8, True),)


def test_sonnet_never_enters_the_primary_stream():
    official = {"kind": "fed_monetary", "source": "fed"}
    n = inputs()
    assert paired(n.add(line(1, 1, SN, b(DAY, 50), impacts=(("MACRO", 1.0, True),), **official)), NEWS) == []
    assert paired(n.add(line(2, 1, HK, b(DAY, 51), status="failed", **official)), NEWS) == []
    [primary] = paired(n.add(line(3, 1, FB, b(DAY, 52), impacts=(("MACRO", -0.2, None),), **official)), NEWS)
    assert primary.scorer == FB and primary.fallback
    at = n.step(b(DAY, 53))
    assert at[NEWS]["AAPL"].macro < 0 and at[SONNET]["AAPL"].macro_items == 0
    # A Sonnet line on its own never makes a primary score, whatever its route.
    n = inputs()
    assert paired(n.add(line(1, 9, SN, b(DAY, 50), route="finbert")), NEWS) == []


def present(case):
    """The lines of one case, None for a scorer that never wrote a row."""
    return [ln for ln in case if ln is not None]


def test_exactly_one_primary_score_per_item_across_every_route_and_status():
    checked = 0
    for route in ("llm", "finbert", "none"):
        for h_status, f_status, s_status in itertools.product(STATUSES + (None,), ("scored", "untagged",
                                                                                   "stale_vendor", None),
                                                              ("scored", "failed", None)):
            rows = [None if h_status is None else (HK, h_status), None if f_status is None else (FB, f_status),
                    None if s_status is None else (SN, s_status)]
            for order in itertools.permutations(present(rows)):
                lines = [line(i + 1, 1, scorer, b(DAY, 50) + 10_000 * i, status=status, route=route)
                         for i, (scorer, status) in enumerate(order)]
                news = derive_streams(lines)[NEWS]
                receipts = {ln.scorer: ln.recv_ms for ln in lines}
                haiku_ok = h_status in ("scored", "irrelevant")
                if route == "finbert":
                    want = [(FB, receipts[FB], False)] if f_status else []
                elif route == "llm" and haiku_ok:
                    want = [(HK, receipts[HK], False)]
                elif route == "llm" and h_status and f_status:
                    want = [(FB, max(receipts[HK], receipts[FB]), True)]
                else:
                    want = []
                assert [(s.scorer, s.visible_ms, s.fallback) for s in news] == want
                checked += 1
    assert checked > 300


def test_a_second_row_for_one_item_and_scorer_an_unknown_scorer_and_an_unrouted_item_are_ignored():
    router = ScoreStreams()
    assert router.add(line(1, 1, HK, b(DAY, 50)))
    assert router.add(line(2, 1, HK, b(DAY, 51), impacts=(("AAPL", -1.0, True),))) == []
    assert router.add(line(3, 2, "haiku_v2", b(DAY, 52))) == []
    assert router.add(line(4, 3, FB, b(DAY, 53), route="none")) == []
    assert router.ignored == 3 and set(router.items) == {1}


def test_the_state_rebuilds_from_the_kept_lines():
    rng = np.random.default_rng(5)
    lines, seq = [], 0
    for item in range(1, 80):
        recv = b(DAY if item % 2 else NEXT, int(rng.integers(1, 391))) + int(rng.integers(0, MS_PER_MINUTE))
        for scorer, status in ((HK, str(rng.choice(["scored", "failed"]))), (FB, "scored")):
            seq += 1
            ticker = str(rng.choice(["AAPL", "MSFT", "QQQ", "MACRO"]))
            lines.append(line(seq, item, scorer, recv + seq, status=status,
                              impacts=((ticker, float(rng.uniform(-1, 1)), scorer == HK or None),)))
    lines.sort(key=lambda ln: ln.recv_ms)
    a = inputs()
    boundaries = [b(day, m) for day in (DAY, NEXT, THIRD) for m in range(1, 391)]
    pending = list(lines)
    resumed = None
    compared = 0
    for boundary in boundaries:
        while pending and pending[0].recv_ms < boundary:
            ln = pending.pop(0)
            a.add(ln)
            if resumed is not None:
                resumed.add(ln)
        if boundary == b(NEXT, 300):
            resumed = inputs()
            for ln in a.kept_lines():
                resumed.add(ln)
            assert 0 < len(a.kept_lines()) < len(lines) - len(pending)
        got = a.step(boundary)
        if resumed is not None:
            assert resumed.step(boundary) == got
            compared += any(v.company != 0.0 for v in got[NEWS].values())
    assert compared > 300


# Engine inputs


def test_s1_pins_both_inputs_at_zero_and_stores_zeros_at_each_push_minute():
    s1 = EngineInputs.from_config(S1)
    assert not s1.use_news and s1.symbols == S1.stock_feed.symbols
    assert s1.add(line(1, 1, FB, b(DAY, 50), route="finbert")) == []
    current = s1.step(b(DAY, 51))
    assert set(current) == set(S1.stock_feed.symbols) and all(not v.any() for v in current.values())
    k = DAY.first_k + 50
    for symbol in S1.stock_feed.symbols:
        assert s1.push(k, symbol).tolist() == [0.0, 0.0]
    assert s1.take(k, "AAPL").tolist() == [0.0, 0.0]


def test_s2_stores_its_inputs_per_symbol_at_each_push_minute():
    s2 = EngineInputs.from_config(S2)
    assert s2.use_news and s2.horizon == 30
    s2.add(line(1, 1, FB, b(DAY, 49), route="finbert", impacts=(("NVDA", 0.5, None), ("MACRO", -0.5, None))))
    current = s2.step(b(DAY, 50))
    k = DAY.first_k + 49
    stored = {s: s2.push(k, s) for s in ("NVDA", "AAPL", "QQQ")}
    assert stored["NVDA"][0] > 0 and stored["AAPL"][0] == 0 and stored["QQQ"][0] > 0
    assert {v[1] for v in stored.values()} == {current["AAPL"][1]} and current["AAPL"][1] < 0
    assert s2.streams[NEWS]["NVDA"].company == current["NVDA"][0]
    assert np.array_equal(s2.take(k, "NVDA"), stored["NVDA"])
    with pytest.raises(KeyError):
        s2.take(k, "NVDA")                                     # taken once


def test_a_score_received_after_the_push_minute_never_enters_that_record_but_moves_the_current_inputs():
    s2 = EngineInputs.from_config(S2)
    k = DAY.first_k + 99
    pushed = s2.step(b(DAY, 100))["AAPL"].copy()
    s2.push(k, "AAPL")
    s2.add(line(1, 1, FB, b(DAY, 100) + 10_000, route="finbert", impacts=(("AAPL", 0.9, None),)))
    for minute in range(101, 130):
        s2.step(b(DAY, minute))
    current = s2.step(b(DAY, 130))["AAPL"]
    assert current[0] > 0 and not pushed.any()
    # The training sample and the linear_news update of the record pushed at k both take what was stored then.
    assert np.array_equal(s2.take(k, "AAPL"), pushed)


def test_stored_inputs_are_forgotten_once_their_records_can_no_longer_mature():
    s2 = EngineInputs.from_config(S2)
    k = DAY.first_k + 9
    s2.step(b(DAY, 10))
    s2.push(k, "MSFT")
    s2.step(b(DAY, 40))
    assert s2.take(k, "MSFT").shape == (2,)                    # matures at k plus 30
    s2.push(k + 30, "MSFT")
    s2.step(b(DAY, 71))
    with pytest.raises(KeyError):
        s2.take(k + 30, "MSFT")


def test_the_default_scorer_names_are_the_ones_the_committed_news_config_writes():
    news = load_news_config(ROOT / "configs" / "v2" / "news.toml")
    assert Scorers() == (news.llm.scorer_name, news.llm.sonnet.scorer_name, health.FINBERT_SCORER)
