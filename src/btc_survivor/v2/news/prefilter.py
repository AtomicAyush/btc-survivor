"""The prefilter and the route of PLAN_V2 sections 7.5 and 7.6, fixed before launch and never compared with prices.

Aliases are case sensitive with word boundaries, while macro keywords and junk patterns ignore case. Exchange
notation such as (NASDAQ: AAPL) is stripped before the macro check, and an item that names any alias never also gets
MACRO, so one company's score never enters all eight macro inputs. Elon Musk is left out of the aliases, because much
of his coverage is about other companies.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from btc_survivor.v2.config import OFFICIAL_KINDS, NewsConfig

MACRO = "MACRO"

ALIASES = {
    "AAPL": ("Apple", "iPhone", "AAPL", "Tim Cook"),
    "MSFT": ("Microsoft", "Azure", "MSFT", "Satya Nadella"),
    "NVDA": ("Nvidia", "NVIDIA", "NVDA", "Jensen Huang"),
    "AMZN": ("Amazon", "AWS", "AMZN", "Andy Jassy"),
    "GOOGL": ("Alphabet", "Google", "YouTube", "Waymo", "GOOGL", "GOOG", "Sundar Pichai"),
    "META": ("Meta", "Meta Platforms", "Facebook", "Instagram", "WhatsApp", "META", "Mark Zuckerberg"),
    "TSLA": ("Tesla", "TSLA"),
    "QQQ": ("QQQ", "Nasdaq 100", "Nasdaq-100", "Invesco QQQ"),
}

MACRO_KEYWORDS = (
    "Fed", "Federal Reserve", "FOMC", "Powell", "interest rate", "rate cut", "rate hike", "inflation", "CPI", "PPI",
    "PCE", "payrolls", "jobs report", "unemployment", "jobless claims", "Treasury yields", "tariff", "recession",
    "GDP", "Wall Street", "stock market", "Nasdaq Composite", "S&P 500", "Dow Jones", "semiconductor", "chip stocks",
    "antitrust",
)

JUNK_PATTERNS = (
    r"\b(top|best) \d+ stocks?\b",
    r"\bstocks? to (buy|watch)\b",
    r"\bshould you buy\b",
    r"\bmillionaire\b",
    r"\bbuy and hold forever\b",
    r"\bmotley fool\b",
)

# Relevance tiers, filter reasons and routes, as news_items stores them.
TITLE, BODY, VENDOR, MACRO_TIER, OFFICIAL, NONE = "title", "body", "vendor", "macro", "official", "none"
JUNK, NO_MATCH, EMPTY, FORM = "junk", "no_match", "empty", "form"
LLM, FINBERT = "llm", "finbert"
SCORED_TIERS = (TITLE, BODY, MACRO_TIER)
FINNHUB, EDGAR, FED, BLS = "finnhub", "edgar", "fed", "bls"
SOURCES = (FINNHUB, EDGAR, FED, BLS)


def _phrases(words: tuple[str, ...]) -> str:
    """An alternation of literal phrases in which any run of white space matches the space between two words."""
    return "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in words)


# A keyword also matches its plural, such as tariffs or rate cuts, but never a longer word, such as FedEx or federal.
_MACRO = re.compile(rf"\b(?:{_phrases(MACRO_KEYWORDS)})s?\b", re.IGNORECASE)
_JUNK = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    """What the prefilter reads from one parsed item. related holds Finnhub's related symbols, general marks a
    Finnhub general news item, and filer and form belong to EDGAR items."""

    source: str
    kind: str
    title: str
    summary: str = ""
    publisher: str = ""
    related: tuple[str, ...] = ()
    general: bool = False
    filer: str = ""
    form: str = ""


@dataclass(frozen=True)
class Tags:
    """The relevance tier and tickers fixed at first receipt, with the filter reason of a filtered item."""

    relevance: str
    tickers: tuple[str, ...] = ()
    filter_reason: str | None = None

    @property
    def filtered(self) -> bool:
        return self.filter_reason is not None

    @property
    def tickers_text(self) -> str:
        """The tickers as news_items stores them, comma separated and possibly empty."""
        return ",".join(self.tickers)


def _clean(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


class Prefilter:
    """The aliases of the universe, the macro check after stripping exchange notation, and the junk patterns."""

    def __init__(self, config: NewsConfig):
        missing = [s for s in config.news.universe if s not in ALIASES]
        if missing:
            raise ValueError(f"no aliases are fixed for {', '.join(missing)}")
        self.universe = config.news.universe
        self.forms = config.edgar.forms
        self.alias_excludes_macro = config.prefilter.alias_excludes_macro
        self._strip = re.compile(config.prefilter.macro_strip_pattern)
        self._aliases = {s: re.compile(rf"\b(?:{_phrases(ALIASES[s])})\b") for s in self.universe}

    def symbols(self, text: str) -> tuple[str, ...]:
        """The universe symbols whose aliases appear in the text, in universe order."""
        text = _clean(text)
        return tuple(s for s in self.universe if self._aliases[s].search(text))

    def macro(self, text: str) -> bool:
        """Whether a macro keyword appears once exchange notation such as NASDAQ: AAPL is stripped."""
        return bool(_MACRO.search(self._strip.sub(" ", _clean(text))))

    def junk(self, text: str) -> bool:
        return bool(_JUNK.search(_clean(text)))

    def tag(self, item: Candidate) -> Tags:
        """The relevance tier and tickers of one item that survived deduplication, or the reason it is filtered."""
        if item.source not in SOURCES:
            raise ValueError(f"unknown news source {item.source!r}")
        if not _clean(item.title).strip():
            return Tags(NONE, (), EMPTY)
        if item.source == EDGAR:
            if item.form not in self.forms:
                return Tags(NONE, (), FORM)
            if item.filer not in self.universe:
                return Tags(NONE, (), NO_MATCH)
            return Tags(OFFICIAL, (item.filer,))
        if item.source in (FED, BLS):
            return Tags(OFFICIAL, (MACRO,))
        return self._media(item)

    def _media(self, item: Candidate) -> Tags:
        """The tiers of a Finnhub item, title before body before macro before vendor."""
        if any(self.junk(text) for text in (item.title, item.summary, item.publisher)):
            return Tags(NONE, (), JUNK)
        in_title, in_body = self.symbols(item.title), self.symbols(item.summary)
        named = tuple(s for s in self.universe if s in in_title or s in in_body)
        is_macro = item.general and (self.macro(item.title) or self.macro(item.summary))
        if named:
            extra = (MACRO,) if is_macro and not self.alias_excludes_macro else ()
            return Tags(TITLE if in_title else BODY, named + extra)
        if is_macro:
            return Tags(MACRO_TIER, (MACRO,))
        related = tuple(s for s in self.universe if s in item.related)
        if related:
            return Tags(VENDOR, related)
        return Tags(NONE, (), NO_MATCH)


def route(source: str, kind: str, status: str, relevance: str, send_to_llm: bool) -> str:
    """The route set once at insert. Finnhub items with status new in the title, body or macro tier go to llm, or to
    finbert only while send_to_llm is false, the five official kinds with status new go to llm, and everything else,
    8k_a included, goes nowhere."""
    if status != "new":
        return NONE
    if source == FINNHUB:
        if relevance in SCORED_TIERS:
            return LLM if send_to_llm else FINBERT
        return NONE
    if kind in OFFICIAL_KINDS and relevance == OFFICIAL:
        return LLM
    return NONE
