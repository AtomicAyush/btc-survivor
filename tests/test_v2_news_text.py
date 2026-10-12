"""The headline normalization, the 16 character headline hash and the near duplicate token sets, on fictional
headlines."""

import hashlib

import pytest

from btc_survivor.v2.news import text
from btc_survivor.v2.news.text import normalize, title_hash, tokens


def sha16(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@pytest.mark.parametrize("title, expected", [
    ("Fictional Corp Raises Outlook", "fictional corp raises outlook"),
    ("Ａｐｐｌｅ ﬁnds new supplier", "apple finds new supplier"),     # NFKC
    ("Widget Co beats   estimates", "widget co beats estimates"),
    ("AT&T and Widget&Co agree terms", "at and t and widget and co agree terms"),
    ("Coca-Cola style merger talk", "coca cola style merger talk"),
    ("Q3: profits up 12.5%!", "q3 profits up 12 5"),
    ("snake_case and dots...", "snake case and dots"),
    ("Société fictive annonce", "société fictive annonce"),
    ("", ""),
], ids=range(9))
def test_normalize(title, expected):
    assert normalize(title) == expected


@pytest.mark.parametrize("title", [
    "Fictional Corp raises outlook - Reuters",
    "Fictional Corp raises outlook – The Daily Ledger",
    "Fictional Corp raises outlook — The Fictional Daily Ledger",
    "Fictional Corp raises outlook | Ledger",
    "Fictional Corp raises outlook   -   Reuters  ",
    "fictional corp raises outlook",
], ids=range(6))
def test_a_trailing_publisher_suffix_of_up_to_four_words_is_dropped(title):
    assert normalize(title) == "fictional corp raises outlook"


def test_a_longer_suffix_and_an_unspaced_hyphen_are_kept():
    assert normalize("Widget sales rise - The Fictional Daily Ledger Today") == \
        "widget sales rise the fictional daily ledger today"
    assert normalize("Widget sales rise -Reuters") == "widget sales rise reuters"
    assert normalize("Widget sales rise-Reuters") == "widget sales rise reuters"


def test_only_the_last_separator_starts_a_suffix():
    assert normalize("Apple - Microsoft deal - Reuters") == "apple microsoft deal"
    assert normalize("Apple | Microsoft deal") == "apple"
    assert text.MAX_SUFFIX_WORDS == 4


def test_a_bare_suffix_is_not_dropped():
    assert normalize("- Reuters") == "reuters"


def test_title_hash_is_the_first_16_hex_digits_of_sha_256():
    digest = title_hash("Apple & Microsoft sign deal - Reuters")
    assert digest == sha16("apple and microsoft sign deal")
    assert len(digest) == 16 and int(digest, 16) >= 0 and digest == digest.lower()
    assert title_hash("") == sha16("")


def test_variants_of_one_headline_share_a_hash():
    variants = [
        "Fictional Corp & Widget Co agree merger",
        "FICTIONAL CORP AND WIDGET CO AGREE MERGER",
        "Fictional Corp and Widget Co agree merger!",
        "  Fictional  Corp, and Widget Co: agree merger - Reuters",
        "Fictional Corp & Widget Co agree merger | Fictional Wire",
    ]
    assert len({title_hash(v) for v in variants}) == 1


def test_different_words_give_different_hashes():
    assert title_hash("Fictional Corp raises outlook") != title_hash("Fictional Corp cuts outlook")
    assert title_hash("Fictional Corp raises outlook") != title_hash("Fictional Corp raises the outlook")


def test_tokens_drop_short_words_and_stop_words():
    assert tokens("The Fictional Corp's CEO is to step down at the end of a year - Reuters") == frozenset(
        {"fictional", "corp", "ceo", "step", "down", "end", "year"})
    assert tokens("A I") == frozenset()
    assert tokens("") == frozenset()


def test_the_stop_word_list():
    assert len(text.STOP_WORDS) == 20
    assert all(w == normalize(w) and len(w) >= 2 for w in text.STOP_WORDS)
    assert tokens(" ".join(sorted(text.STOP_WORDS))) == frozenset()
