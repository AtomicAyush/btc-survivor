"""Headline normalization, the headline hash and the token sets of PLAN_V2 section 7.5.

Both deduplication steps read the same normalized text, so a headline repeated with other case, punctuation, an
ampersand or a trailing publisher suffix hashes alike and gives the same tokens. Callers pass the title as stored,
already cut to its maximum length, so a hash recomputed from news.db matches the one written at insert.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

HASH_HEX_CHARS = 16
MIN_TOKEN_CHARS = 2
MAX_SUFFIX_WORDS = 4
STOP_WORDS = frozenset({
    "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "its", "of", "on", "or", "the", "to", "with",
})

# A publisher suffix is a spaced hyphen, en dash, em dash or bar followed by one to four words that end the
# headline, such as " - Reuters". A word is never a lone separator, so only the last separator can start a suffix.
_SEPARATOR = "[-–—|]"
_WORD = rf"(?!{_SEPARATOR}(?:\s|$))\S+"
_SUFFIX = re.compile(rf"\s+{_SEPARATOR}\s+{_WORD}(?:\s+{_WORD}){{0,{MAX_SUFFIX_WORDS - 1}}}\s*$")
_NON_ALNUM = re.compile(r"[\W_]+")


def normalize(title: str) -> str:
    """NFKC, lower case, an ampersand read as and, a trailing publisher suffix of at most four words dropped, and
    every run of characters that are not letters or digits turned into one space."""
    text = unicodedata.normalize("NFKC", title).lower().replace("&", " and ")
    text = _SUFFIX.sub("", text)
    return _NON_ALNUM.sub(" ", text).strip()


def title_hash(title: str) -> str:
    """The first 16 hex characters of the SHA 256 of the normalized headline, stored as news_items.title_hash."""
    return hashlib.sha256(normalize(title).encode("utf-8")).hexdigest()[:HASH_HEX_CHARS]


def tokens(title: str) -> frozenset[str]:
    """The near duplicate token set, normalized words of two or more characters without the stop words."""
    return frozenset(w for w in normalize(title).split() if len(w) >= MIN_TOKEN_CHARS and w not in STOP_WORDS)
