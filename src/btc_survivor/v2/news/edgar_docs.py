"""Fetched documents as the text news_bodies keeps, from an EDGAR filing index, an 8-K and its Exhibit 99.1, or a Fed
release page, per PLAN_V2 section 7.4.

Text is the page with tags stripped by html.parser, entities unescaped and white space collapsed, then cut to a fixed
length with the full length and the cut recorded. Markup that html.parser refuses, such as an unknown marked section
on Python 3.13, falls back to a plain pattern that strips tags, and a filing index keeps the rows read before the
failure, so one odd page never fails a poll. Everything here is a pure function of a response body, because
news/http.py does the fetching.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser

SEC_ORIGIN = "https://www.sec.gov"
VIEWER = "/ix?doc="
MAIN_TYPES = frozenset({"8-K", "8-K/A"})
EXHIBIT_99_1 = re.compile(r"EX-99\.0?1")
DOC_PATH = re.compile(r"/Archives/edgar/data/\d+/\d+/[\w.-]+")

# Elements whose text is never kept, among them the hidden inline XBRL header of a filing and a page's menus.
SKIP = frozenset({"script", "style", "title", "noscript", "template", "nav", "ix:header"})
# Elements that separate words, so adjacent cells or paragraphs never run together.
BLOCK = frozenset({
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "footer", "h1", "h2", "h3", "h4",
    "h5", "h6", "header", "hr", "li", "main", "ol", "p", "pre", "section", "table", "tbody", "td", "th", "thead",
    "tr", "ul",
})
_SPACE = re.compile(r"\s+")
_HIDDEN = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]*>")


@dataclass(frozen=True)
class Doc:
    """One document's kept text, its length before the cut and whether the cut removed anything."""

    text: str
    chars_total: int
    truncated: bool


@dataclass(frozen=True)
class Body:
    """A news_bodies row before its item has a seq, with doc_urls a JSON list of the documents the text came from."""

    text: str
    chars_total: int
    truncated: bool
    doc_urls: str


@dataclass(frozen=True)
class FilingDocs:
    """The direct addresses of a filing's main document and first Exhibit 99.1, None when the index lists none."""

    main: str | None
    exhibit: str | None


class _Text(HTMLParser):
    """Collects the text outside skipped elements, and only inside the first element with the wanted id if one is
    given, counting nested elements of that element's own tag to find where it ends."""

    def __init__(self, element_id: str | None) -> None:
        super().__init__(convert_charrefs=True)
        self.element_id = element_id
        self.parts: list[str] = []
        self.skipping = 0
        self.found = False
        self.target: str | None = None
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP:
            self.skipping += 1
        if self.target is not None:
            if tag == self.target:
                self.depth += 1
        elif self.element_id is not None and not self.found and ("id", self.element_id) in attrs:
            self.found, self.target, self.depth = True, tag, 1
        if tag in BLOCK:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP and self.skipping:
            self.skipping -= 1
        if tag == self.target:
            self.depth -= 1
            if self.depth == 0:
                self.target = None
        if tag in BLOCK:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.skipping and (self.element_id is None or self.target is not None):
            self.parts.append(data)


class _Rows(HTMLParser):
    """Collects the cell texts and the first link of every table row that has data cells."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[list[str], str]] = []
        self.cells: list[str] = []
        self.href = ""
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.end_row()
        elif tag == "td":
            self.end_cell()
            self.cell = []
        elif tag == "a" and not self.href:
            self.href = dict(attrs).get("href") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            self.end_cell()
        elif tag in ("tr", "table"):
            self.end_row()

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def end_cell(self) -> None:
        if self.cell is not None:
            self.cells.append(" ".join("".join(self.cell).split()))
            self.cell = None

    def end_row(self) -> None:
        self.end_cell()
        if self.cells:
            self.rows.append((self.cells, self.href))
        self.cells, self.href = [], ""


def decode(body: bytes) -> str:
    """A fetched page as text, UTF-8 without any byte order mark when it is valid, else Windows 1252, which older
    filings use."""
    try:
        return body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return body.decode("cp1252", errors="replace")


def stripped(markup: str) -> str:
    """The text of markup by a plain pattern, scripts, styles and tags removed and entities unescaped, for markup that
    html.parser refuses."""
    return _SPACE.sub(" ", html.unescape(_TAG.sub(" ", _HIDDEN.sub(" ", markup)))).strip()


def html_text(markup: str, element_id: str | None = None) -> str | None:
    """The visible text of markup with white space collapsed, taken only from inside the first element whose id is
    element_id when one is given, or None when no element carries that id. When html.parser refuses the markup, the
    whole page's text comes from the plain pattern, and no element is found."""
    parser = _Text(element_id)
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # html.parser raises AssertionError on some odd markup, which must never fail a poll
        return None if element_id is not None else stripped(markup)
    if element_id is not None and not parser.found:
        return None
    return _SPACE.sub(" ", "".join(parser.parts)).strip()


def cut(text: str, max_chars: int) -> Doc:
    """The first max_chars characters of text, with the full length and whether anything was cut."""
    return Doc(text[:max_chars], len(text), len(text) > max_chars)


def document(page: bytes, max_chars: int, element_ids: Sequence[str] = ()) -> Doc:
    """The kept text of a fetched page, from the first of element_ids the page holds, else from the whole page."""
    markup = decode(page)
    for element_id in element_ids:
        text = html_text(markup, element_id)
        if text is not None:
            return cut(text, max_chars)
    return cut(html_text(markup) or "", max_chars)


def _document_url(href: str) -> str | None:
    """The direct sec.gov address of a filing document link, unwrapping the inline XBRL viewer, or None for a link
    to anywhere else, so no address from a page leads off the SEC archive."""
    path = href.strip().removeprefix(SEC_ORIGIN).removeprefix(VIEWER)
    return SEC_ORIGIN + path if DOC_PATH.fullmatch(path) else None


def filing_documents(index_page: bytes) -> FilingDocs:
    """The main 8-K document and the first Exhibit 99.1 a filing index page lists, found by the Type column of its
    Seq, Description, Document, Type and Size table."""
    parser = _Rows()
    try:
        parser.feed(decode(index_page))
        parser.close()
    except Exception:  # html.parser can refuse odd markup, and the rows read before it still count
        pass
    parser.end_row()
    main = exhibit = None
    for cells, href in parser.rows:
        url = _document_url(href) if len(cells) >= 4 else None
        if url is None:
            continue
        doc_type = cells[3].upper()
        if main is None and doc_type in MAIN_TYPES:
            main = url
        elif exhibit is None and EXHIBIT_99_1.fullmatch(doc_type):
            exhibit = url
    return FilingDocs(main, exhibit)


def body(docs: Sequence[tuple[str, Doc]]) -> Body:
    """The news_bodies row of the fetched documents in order, their texts joined by a blank line, their full lengths
    summed and truncated when any one was cut."""
    return Body("\n\n".join(doc.text for _, doc in docs), sum(doc.chars_total for _, doc in docs),
                any(doc.truncated for _, doc in docs), json.dumps([url for url, _ in docs]))
