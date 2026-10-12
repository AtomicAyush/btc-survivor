"""SEC EDGAR filings of PLAN_V2 section 7.4, from the latest 8-K Atom feed and each company's submissions file.

Both paths keep only the forms in [edgar] forms filed by the seven CIKs and give one RawItem per filing, with the
accession number as vendor id and the filing index page as url, so a filing seen on both paths is one item. The index,
the main document and Exhibit 99.1 are then fetched and read through edgar_docs before the item is inserted.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from btc_survivor.v2.news.rss import TITLE_MAX_CHARS, RawItem, parse_feed, plain, vendor_time

log = logging.getLogger(__name__)

ATOM_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include"
            "&start=0&count=100&output=atom")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{accession}-index.htm"
EDGAR = "edgar"
PUBLISHER = "SEC EDGAR"
KINDS = {"8-K": "8k", "8-K/A": "8k_a"}
NEW_YORK = ZoneInfo("America/New_York")
_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
# An Atom entry title such as 8-K - Example Corp (0000123456) (Filer).
_ATOM_TITLE = re.compile(r"(?P<form>\S+) - (?P<company>.+?) \((?P<cik>\d{10})\)")
_ATOM_ITEM = re.compile(r"\bItem (\d{1,2}\.\d{2})")
_ITEM_NUMBER = re.compile(r"\d{1,2}\.\d{2}")


def submissions_url(cik: int) -> str:
    return SUBMISSIONS_URL.format(cik=cik)


def index_url(cik: int, accession: str) -> str:
    """The filing index page, which lists the main document and the exhibits."""
    return INDEX_URL.format(cik=cik, folder=accession.replace("-", ""), accession=accession)


def accepted_time(text: str) -> str | None:
    """The UTC time of a submissions acceptanceDateTime. EDGAR marks it Z but shifts it later by the New York offset,
    as three filings checked against their index pages on 12 October 2026 showed in both EDT and EST, so the New York
    wall time of the stated instant is the true UTC time. Unverified beyond those filings."""
    try:
        stated = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    if stated.tzinfo is None:
        return None
    return vendor_time(stated.astimezone(NEW_YORK).replace(tzinfo=timezone.utc).isoformat())


def _item(accession: str, cik: int, symbol: str, form: str, company: str, items: list[str],
          vendor_ts: str | None) -> RawItem:
    """One filing as a RawItem, titled by its company, form and 8-K item numbers."""
    numbers = list(dict.fromkeys(items))
    title = f"{company} {form}" + (f", items {', '.join(numbers)}" if numbers else "")
    return RawItem(EDGAR, accession, KINDS[form], plain(title, TITLE_MAX_CHARS), vendor_ts,
                   url=index_url(cik, accession), publisher=PUBLISHER, filer=symbol, form=form,
                   items_8k=",".join(numbers))


def parse_atom(body: bytes, ciks: Mapping[str, int], forms: Collection[str]) -> list[RawItem]:
    """The filings in the latest filings feed whose form is kept and whose filer is one of ciks, a map from symbol to
    CIK, in feed order. The form comes from the entry's category and the items from its summary."""
    symbols = {cik: symbol for symbol, cik in ciks.items()}
    entries = parse_feed(body)
    items, dropped = [], 0
    for entry in entries:
        title = _ATOM_TITLE.match(entry.title)
        accession = _ACCESSION.search(entry.guid) or _ACCESSION.search(entry.link)
        if title is None or accession is None:
            dropped += 1
            continue
        cik, form = int(title["cik"]), entry.category or title["form"]
        if cik in symbols and form in forms:
            items.append(_item(accession.group(), cik, symbols[cik], form, title["company"],
                               _ATOM_ITEM.findall(entry.summary), vendor_time(entry.published)))
    if dropped:
        log.warning("%d EDGAR feed entries dropped without a filer or an accession number", dropped)
    return items


def _cik(value: object) -> int | None:
    try:
        return int(str(value))
    except ValueError:
        return None


def parse_submissions(body: bytes, symbol: str, cik: int, forms: Collection[str]) -> list[RawItem]:
    """The kept forms among filings.recent of one company's submissions file, newest first as the file lists them. A
    body that is not such a file, or a file of another CIK, raises ValueError."""
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        raise ValueError("the submissions file is not JSON") from None
    filings = data.get("filings") if isinstance(data, dict) else None
    recent = filings.get("recent") if isinstance(filings, dict) else None
    if not isinstance(recent, dict):
        raise ValueError("the submissions file has no filings.recent")
    if _cik(data.get("cik")) != cik:
        raise ValueError(f"the submissions file is not the one of CIK {cik}")
    columns = [recent.get(key) for key in ("accessionNumber", "form", "acceptanceDateTime", "items")]
    if not all(isinstance(column, list) for column in columns) or len({len(column) for column in columns}) != 1:
        raise ValueError("the columns of filings.recent are missing or of unequal length")
    company = data.get("name") if isinstance(data.get("name"), str) else symbol
    items, dropped = [], 0
    for accession, form, accepted, numbers in zip(*columns):
        if not isinstance(form, str) or form not in forms:
            continue
        if not isinstance(accession, str) or not _ACCESSION.fullmatch(accession):
            dropped += 1
            continue
        items.append(_item(accession, cik, symbol, form, company,
                           _ITEM_NUMBER.findall(numbers) if isinstance(numbers, str) else [],
                           accepted_time(accepted) if isinstance(accepted, str) else None))
    if dropped:
        log.warning("%d EDGAR submissions rows dropped without an accession number", dropped)
    return items
