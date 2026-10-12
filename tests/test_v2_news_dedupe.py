"""Deduplication by vendor id, headline hash and Jaccard index, within one source and 24 hours, on fictional
headlines."""

import dataclasses

import pytest

from btc_survivor.v2.config import load_news_config
from btc_survivor.v2.news import dedupe
from btc_survivor.v2.news.dedupe import HASH, NEAR, SILENT, VENDOR_ID, Duplicate, find_duplicate, jaccard, repoll, seen
from btc_survivor.v2.news.text import title_hash, tokens
from helpers import ROOT

SETTINGS = load_news_config(ROOT / "configs" / "v2" / "news.toml").dedupe
HOUR = 3_600_000
T0 = 1_793_368_800_000        # 14:00 UTC on Friday 30 October 2026
FIN = "finnhub"


def terms(*numbers: int) -> str:
    """A fictional headline whose tokens are exactly the given numbered terms."""
    return " ".join(f"term{n:03d}" for n in numbers)


def test_the_committed_settings_match_the_plan():
    assert (SETTINGS.hash_window_hours, SETTINGS.near_dup_window_hours) == (24, 24)
    assert (SETTINGS.near_dup_jaccard, SETTINGS.near_dup_min_tokens) == (0.8, 4)


def test_a_repoll_by_the_same_channel_is_silent():
    assert repoll(set(), "finnhub_company_AAPL") is None
    assert repoll({"finnhub_company_AAPL"}, "finnhub_company_AAPL") == SILENT
    assert repoll({"finnhub_company_AAPL"}, "finnhub_company_MSFT") == VENDOR_ID
    # Once MSFT's news_dups row exists, MSFT's own repolls of the same id write nothing more.
    assert repoll({"finnhub_company_AAPL", "finnhub_company_MSFT"}, "finnhub_company_MSFT") == SILENT
    assert repoll(("fed_press_all",), "fed_press_monetary") == VENDOR_ID


def test_jaccard():
    a, b = frozenset({"aa", "bb", "cc", "dd"}), frozenset({"aa", "bb", "cc", "ee"})
    assert jaccard(a, b) == 3 / 5
    assert jaccard(a, a) == 1.0
    assert jaccard(a, frozenset()) == 0.0
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_seen_rebuilds_the_stored_hash_and_tokens():
    item = seen(7, FIN, T0, "Fictional Corp raises outlook - Reuters")
    assert item == dedupe.Seen(7, FIN, T0, title_hash("Fictional Corp raises outlook"),
                               frozenset({"fictional", "corp", "raises", "outlook"}))


def test_the_same_hash_from_the_same_source_is_a_duplicate():
    recent = [seen(1, FIN, T0, "Fictional Corp & Widget Co agree merger")]
    found = find_duplicate(FIN, T0 + HOUR, "FICTIONAL CORP AND WIDGET CO AGREE MERGER - Reuters", recent, SETTINGS)
    assert found == Duplicate(1, HASH, None)


def test_the_earliest_hash_match_wins_and_hash_comes_before_near():
    recent = [
        seen(5, FIN, T0 + 2, terms(1, 2, 3, 4, 5)),
        seen(3, FIN, T0 + 1, terms(1, 2, 3, 4, 5)),
        seen(2, FIN, T0, terms(1, 2, 3, 4, 5, 6)),
    ]
    assert find_duplicate(FIN, T0 + HOUR, terms(1, 2, 3, 4, 5), recent, SETTINGS) == Duplicate(3, HASH)


def test_deduplication_stays_within_one_source():
    title = "Fictional Corp & Widget Co agree merger"
    recent = [seen(1, "fed", T0, title), seen(2, "edgar", T0, terms(1, 2, 3, 4, 5))]
    assert find_duplicate(FIN, T0 + HOUR, title, recent, SETTINGS) is None
    assert find_duplicate(FIN, T0 + HOUR, terms(1, 2, 3, 4), recent, SETTINGS) is None
    assert find_duplicate("edgar", T0 + HOUR, terms(1, 2, 3, 4), recent, SETTINGS) == Duplicate(2, NEAR, 0.8)


def test_jaccard_at_0_8_is_a_near_duplicate():
    original = [seen(1, FIN, T0, terms(1, 2, 3, 4, 5))]
    found = find_duplicate(FIN, T0 + HOUR, terms(1, 2, 3, 4), original, SETTINGS)
    assert found == Duplicate(1, NEAR, 0.8) and found.jaccard == 0.8
    long = [seen(1, FIN, T0, terms(*range(10)))]
    assert find_duplicate(FIN, T0 + HOUR, terms(*range(8)), long, SETTINGS) == Duplicate(1, NEAR, 0.8)


def test_jaccard_at_0_79_is_not_a_near_duplicate():
    original = [seen(1, FIN, T0, terms(*range(100)))]
    candidate = terms(*range(79))
    assert jaccard(tokens(candidate), original[0].tokens) == 0.79
    assert find_duplicate(FIN, T0 + HOUR, candidate, original, SETTINGS) is None
    assert find_duplicate(FIN, T0 + HOUR, terms(*range(80)), original, SETTINGS) == Duplicate(1, NEAR, 0.8)


def test_a_real_looking_near_duplicate():
    recent = [seen(4, FIN, T0, "Fictional Corp raises full year outlook after strong quarter")]
    found = find_duplicate(FIN, T0 + HOUR, "Fictional Corp raises its full year outlook after a strong quarter",
                           recent, SETTINGS)
    assert found == Duplicate(4, NEAR, 1.0)


def test_both_token_sets_need_at_least_four_tokens():
    three = [seen(1, FIN, T0, "Fictional widget recall")]
    assert find_duplicate(FIN, T0 + HOUR, "The fictional widget recall", three, SETTINGS) is None
    four = [seen(1, FIN, T0, "Fictional widget recall widens")]
    assert find_duplicate(FIN, T0 + HOUR, "The fictional widget recall widens", four, SETTINGS) == \
        Duplicate(1, NEAR, 1.0)
    looser = dataclasses.replace(SETTINGS, near_dup_min_tokens=3)
    assert find_duplicate(FIN, T0 + HOUR, "The fictional widget recall", three, looser) == Duplicate(1, NEAR, 1.0)


def test_the_highest_jaccard_wins_with_ties_to_the_earliest():
    recent = [
        seen(9, FIN, T0, terms(1, 2, 3, 4, 5)),           # 0.8
        seen(8, FIN, T0, terms(1, 2, 3, 4, 6)),           # 0.8, earlier
        seen(7, FIN, T0, terms(1, 2, 3, 4, 7, 8)),        # 0.67, below the threshold
    ]
    assert find_duplicate(FIN, T0 + HOUR, terms(1, 2, 3, 4), recent, SETTINGS) == Duplicate(8, NEAR, 0.8)
    recent.append(seen(10, FIN, T0, terms(1, 2, 3, 4, 99) + " the"))
    recent.append(seen(11, FIN, T0, terms(1, 2, 3, 4) + " of"))  # 1.0 but a different hash
    assert find_duplicate(FIN, T0 + HOUR, terms(1, 2, 3, 4), recent, SETTINGS) == Duplicate(11, NEAR, 1.0)


@pytest.mark.parametrize("age, method", [
    (0, HASH), (24 * HOUR, HASH), (24 * HOUR + 1, None),
], ids=["same instant", "exactly 24 hours", "24 hours and 1 ms"])
def test_the_24_hour_hash_window(age, method):
    recent = [seen(1, FIN, T0, "Fictional Corp raises outlook")]
    found = find_duplicate(FIN, T0 + age, "Fictional Corp raises outlook", recent, SETTINGS)
    assert (found.method if found else None) == method


@pytest.mark.parametrize("age, method", [
    (0, NEAR), (24 * HOUR, NEAR), (24 * HOUR + 1, None),
], ids=["same instant", "exactly 24 hours", "24 hours and 1 ms"])
def test_the_24_hour_near_window(age, method):
    recent = [seen(1, FIN, T0, terms(1, 2, 3, 4, 5))]
    found = find_duplicate(FIN, T0 + age, terms(1, 2, 3, 4), recent, SETTINGS)
    assert (found.method if found else None) == method


def test_the_windows_follow_their_own_settings():
    """Past a one hour hash window the same headline is still a near duplicate until its two hour window ends."""
    recent = [seen(1, FIN, T0, terms(1, 2, 3, 4, 5))]
    short = dataclasses.replace(SETTINGS, hash_window_hours=1, near_dup_window_hours=2)
    assert find_duplicate(FIN, T0 + HOUR + 1, terms(1, 2, 3, 4, 5), recent, short) == Duplicate(1, NEAR, 1.0)
    assert find_duplicate(FIN, T0 + 2 * HOUR + 1, terms(1, 2, 3, 4, 5), recent, short) is None


def test_an_item_received_after_the_candidate_is_ignored():
    recent = [seen(1, FIN, T0 + 1, "Fictional Corp raises outlook")]
    assert find_duplicate(FIN, T0, "Fictional Corp raises outlook", recent, SETTINGS) is None


def test_new_tickers_are_the_ones_the_original_lacks():
    assert dedupe.new_tickers(("AAPL", "MSFT", "MACRO"), ("AAPL",)) == "MSFT,MACRO"
    assert dedupe.new_tickers(("AAPL",), ("AAPL", "MSFT")) == ""
    assert dedupe.new_tickers((), ()) == ""
