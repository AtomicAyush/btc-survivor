"""RSS 2.0 and Atom feeds, the Federal Reserve and BLS items of PLAN_V2 section 7.4, and RawItem, the parsed item
every source returns.

A Fed item's vendor id is its link and its kind comes from the link alone, a press release under the monetary path
being fed_monetary, so a release in two feeds is one item of one kind whichever feed arrives first. The Fed publishes
RSS with a byte order mark and BLS publishes Atom, and both are parsed from bytes so the declared encoding holds.
"""

from __future__ import annotations

import html
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from btc_survivor.clock import iso
from btc_survivor.v2.news import edgar_docs
from btc_survivor.v2.news.edgar_docs import Body

log = logging.getLogger(__name__)

FED_URL = "https://www.federalreserve.gov/feeds/{feed}.xml"
BLS_URL = "https://www.bls.gov/feed/{feed}.rss"
FED_FEEDS = ("press_all", "press_monetary", "speeches")
BLS_FEEDS = ("cpi", "empsit", "ppi", "jolts")
FED, BLS = "fed", "bls"
FED_PUBLISHER, BLS_PUBLISHER = "Federal Reserve Board", "Bureau of Labor Statistics"
SPEECH_PATHS = ("/newsevents/speech/", "/newsevents/testimony/")
# A release page keeps its text in the article element, inside the main content, after about a thousand words of menus.
FED_ARTICLE_IDS = ("article", "content")
TITLE_MAX_CHARS = 300
ATOM = "{http://www.w3.org/2005/Atom}"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class RawItem:
    """One item as its source parsed it, before deduplication, the prefilter and the insert. The names follow
    news_items and the prefilter's Candidate, with related holding Finnhub's tags, general marking Finnhub general
    news, filer and form belonging to EDGAR items, and an empty string wherever a source has no value."""

    source: str
    vendor_id: str
    kind: str
    title: str
    vendor_ts: str | None
    summary: str = ""
    url: str = ""
    publisher: str = ""
    related: tuple[str, ...] = ()
    general: bool = False
    filer: str = ""
    form: str = ""
    items_8k: str = ""


@dataclass(frozen=True)
class Entry:
    """One RSS item or Atom entry as raw text, guid holding an Atom id and published an Atom updated time when the
    entry has no published time, and category an Atom term when it has one."""

    title: str
    link: str
    guid: str
    summary: str
    published: str
    category: str


def _text(node: ET.Element, path: str) -> str:
    return (node.findtext(path) or "").strip()


def _rss_entry(item: ET.Element) -> Entry:
    return Entry(_text(item, "title"), _text(item, "link"), _text(item, "guid"), _text(item, "description"),
                 _text(item, "pubDate"), _text(item, "category"))


def _atom_link(entry: ET.Element) -> str:
    """The address of an entry's alternate link, where a link without rel counts as alternate."""
    for link in entry.findall(f"{ATOM}link"):
        if link.get("rel", "alternate") == "alternate":
            return (link.get("href") or "").strip()
    return ""


def _atom_entry(entry: ET.Element) -> Entry:
    category = entry.find(f"{ATOM}category")
    term = (category.get("term") or category.text or "").strip() if category is not None else ""
    return Entry(_text(entry, f"{ATOM}title"), _atom_link(entry), _text(entry, f"{ATOM}id"),
                 _text(entry, f"{ATOM}summary") or _text(entry, f"{ATOM}content"),
                 _text(entry, f"{ATOM}published") or _text(entry, f"{ATOM}updated"), term)


def parse_feed(body: bytes) -> list[Entry]:
    """The entries of an RSS 2.0 or Atom feed in document order. A body that is neither, or that declares a document
    type, which no feed here needs and which could expand entities, raises ValueError."""
    if b"<!DOCTYPE" in body:
        raise ValueError("the feed declares a document type, which is refused")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError(f"the feed is not well formed XML, {exc}") from None
    if root.tag == "rss":
        return [_rss_entry(item) for item in root.iter("item")]
    if root.tag == f"{ATOM}feed":
        return [_atom_entry(entry) for entry in root.iter(f"{ATOM}entry")]
    raise ValueError("the body is neither an RSS nor an Atom feed")


def vendor_time(text: str) -> str | None:
    """A vendor time as ISO 8601 UTC text with milliseconds, read as ISO 8601 or RFC 822 with an offset, or None when
    it cannot be read or names no offset, because it only measures delivery lag and is never guessed."""
    text = text.strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        try:
            moment = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
    if moment.tzinfo is None:
        return None
    try:
        return iso((moment - _EPOCH) // timedelta(milliseconds=1))
    except (OverflowError, OSError, ValueError):
        return None


def plain(text: str, max_chars: int | None = None) -> str:
    """A headline with entities unescaped and white space collapsed, cut to max_chars."""
    return " ".join(html.unescape(text).split())[:max_chars]


def summary_text(markup: str, max_chars: int) -> str:
    """A feed description with any markup stripped and entities unescaped, cut to max_chars."""
    return (edgar_docs.html_text(markup) or "")[:max_chars]


def _warn_dropped(name: str, dropped: int) -> None:
    if dropped:
        log.warning("%d %s entries dropped without a title or an id", dropped, name)


def fed_url(feed: str) -> str:
    if feed not in FED_FEEDS:
        raise ValueError(f"unknown Fed feed {feed!r}")
    return FED_URL.format(feed=feed)


def bls_url(feed: str) -> str:
    if feed not in BLS_FEEDS:
        raise ValueError(f"unknown BLS feed {feed!r}")
    return BLS_URL.format(feed=feed)


def fed_kind(link: str, monetary_pattern: str) -> str:
    """The kind of a Fed item from its link alone, fed_monetary under the monetary press release path, fed_speech for
    a speech or testimony and fed_press otherwise, never from the feed that saw it."""
    if monetary_pattern in link:
        return "fed_monetary"
    if any(path in link for path in SPEECH_PATHS):
        return "fed_speech"
    return "fed_press"


def fed_items(body: bytes, monetary_pattern: str, summary_max_chars: int) -> list[RawItem]:
    """The items of one Fed feed in feed order, each keyed by its link, dropping entries without a title or link."""
    entries = parse_feed(body)
    items = []
    for entry in entries:
        link, title = entry.link or entry.guid, plain(entry.title, TITLE_MAX_CHARS)
        if link and title:
            items.append(RawItem(FED, link, fed_kind(link, monetary_pattern), title, vendor_time(entry.published),
                                 summary=summary_text(entry.summary, summary_max_chars), url=link,
                                 publisher=FED_PUBLISHER))
    _warn_dropped("Fed", len(entries) - len(items))
    return items


def bls_items(body: bytes, summary_max_chars: int) -> list[RawItem]:
    """The releases of one BLS feed in feed order, each keyed by its id, or its link when it has none."""
    entries = parse_feed(body)
    items = []
    for entry in entries:
        vendor_id, title = entry.guid or entry.link, plain(entry.title, TITLE_MAX_CHARS)
        if vendor_id and title:
            items.append(RawItem(BLS, vendor_id, "bls_release", title, vendor_time(entry.published),
                                 summary=summary_text(entry.summary, summary_max_chars), url=entry.link,
                                 publisher=BLS_PUBLISHER))
    _warn_dropped("BLS", len(entries) - len(items))
    return items


def fed_page_body(page: bytes, url: str, max_chars: int) -> Body:
    """The stored text of a monetary policy release page, from its article, else its main content, else the whole
    page, cut to max_chars."""
    return edgar_docs.body([(url, edgar_docs.document(page, max_chars, FED_ARTICLE_IDS))])
