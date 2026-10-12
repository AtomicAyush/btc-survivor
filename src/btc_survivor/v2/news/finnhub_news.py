"""Finnhub company and general news of PLAN_V2 section 7.4, as request addresses and a pure parse of response bodies.

The token travels in the X-Finnhub-Token header that news/http.py adds, so no address built here ever holds it. The
fields id, datetime, headline, summary, source, url and related follow Finnhub's documented shape, and the test
fixtures are fictional. Whether minId works is unverified, so the parse itself drops every id at or below the cursor.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from btc_survivor.clock import MS_PER_SECOND, iso
from btc_survivor.v2.news.rss import RawItem, plain

log = logging.getLogger(__name__)

BASE_URL = "https://finnhub.io/api/v1"
FINNHUB = "finnhub"
MEDIA = "media"
MAX_EPOCH_SECONDS = 32_503_680_000          # 1 January 3000, later than any real vendor time
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.]{0,9}")
_CATEGORY = re.compile(r"[a-z]{1,32}")


def company_url(symbol: str, now_ms: int, lookback_days: int) -> str:
    """The company news address for one symbol over UTC dates from lookback_days before today to today, a range that
    covers whatever time zone the vendor's dates use."""
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError(f"not a symbol {symbol!r}")
    today = datetime.fromtimestamp(now_ms // MS_PER_SECOND, tz=timezone.utc).date()
    start = today - timedelta(days=lookback_days)
    return f"{BASE_URL}/company-news?symbol={symbol}&from={start.isoformat()}&to={today.isoformat()}"


def general_url(category: str, min_id: int) -> str:
    """The general news address asking only for ids above min_id, the cursor kept in meta."""
    if not _CATEGORY.fullmatch(category):
        raise ValueError(f"not a news category {category!r}")
    if isinstance(min_id, bool) or not isinstance(min_id, int) or min_id < 0:
        raise ValueError("minId must be a whole number of at least zero")
    return f"{BASE_URL}/news?category={category}&minId={min_id}"


def _id(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _vendor_ts(value: object) -> str | None:
    """Finnhub's datetime in Unix seconds as ISO 8601 UTC text, or None when it is missing or out of range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not 0 < value < MAX_EPOCH_SECONDS:
        return None
    return iso(round(value * MS_PER_SECOND))


def _related(value: object) -> tuple[str, ...]:
    """The symbols of Finnhub's comma separated related field, in order and without repeats."""
    tags = (tag.strip().upper() for tag in _text(value).split(","))
    return tuple(dict.fromkeys(tag for tag in tags if _SYMBOL.fullmatch(tag)))


def _web_url(value: object) -> str:
    url = _text(value).strip()
    return url if url.startswith(("https://", "http://")) else ""


def parse_news(body: bytes, headline_max_chars: int, summary_max_chars: int, min_id: int = 0,
               general: bool = False) -> list[RawItem]:
    """The entries of a company or general news body with an id above min_id, in body order, dropping entries
    without a positive whole number id or a headline. A body that is not a JSON list, such as an error object,
    raises ValueError."""
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        raise ValueError("the Finnhub news body is not JSON") from None
    if not isinstance(data, list):
        raise ValueError("the Finnhub news body is not a list")
    items, dropped = [], 0
    for entry in data:
        news_id = _id(entry.get("id")) if isinstance(entry, dict) else None
        title = plain(_text(entry.get("headline")), headline_max_chars) if news_id is not None else ""
        if not title:
            dropped += 1
        elif news_id > min_id:
            items.append(RawItem(FINNHUB, str(news_id), MEDIA, title, _vendor_ts(entry.get("datetime")),
                                 summary=plain(_text(entry.get("summary")), summary_max_chars),
                                 url=_web_url(entry.get("url")), publisher=plain(_text(entry.get("source"))),
                                 related=_related(entry.get("related")), general=general))
    if dropped:
        log.warning("%d Finnhub news entries dropped without an id or a headline", dropped)
    return items


def newest_id(items: Iterable[RawItem], current: int) -> int:
    """The general news cursor after a poll, the largest Finnhub id seen so far."""
    return max([current, *(int(item.vendor_id) for item in items)])
