"""The one HTTP client of the news process, PLAN_V2 section 7.2.

It sends GET requests with urllib.request in a worker thread, over https to the hosts of the four sources alone,
straight to the host and never through a proxy. The contact address goes in the User-Agent only for sec.gov,
data.sec.gov and bls.gov, which require it, and the Finnhub token only in the X-Finnhub-Token header of a finnhub.io
request, never in a URL. A body is read gzip decoded and refused past max_bytes, a 304 counts as success, a redirect
is followed only within the same host, and every failure comes back as a poll status with a redacted detail instead
of an exception. Nothing here logs, so headers are never logged.
"""

from __future__ import annotations

import asyncio
import email.utils
import re
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timezone
from email.message import Message
from http.client import HTTPException
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from btc_survivor.v2 import local_secrets
from btc_survivor.v2.config import NewsConfig
from btc_survivor.v2.local_secrets import Secrets
from btc_survivor.v2.news.budget import BLS, FED, FINNHUB, SEC
from btc_survivor.v2.redact import redact

USER_AGENT = "btc-survivor/2"
TOKEN_HEADER = "X-Finnhub-Token"
HOSTS = {
    "finnhub.io": FINNHUB,
    "sec.gov": SEC, "www.sec.gov": SEC, "data.sec.gov": SEC,
    "federalreserve.gov": FED, "www.federalreserve.gov": FED,
    "bls.gov": BLS, "www.bls.gov": BLS,
}
CONTACT_GROUPS = (SEC, BLS)

# Poll statuses this module decides. The collector adds parse_error and clock_skew.
OK, NOT_MODIFIED, HTTP_ERROR, TIMEOUT, RATE_LIMITED, BLOCKED = (
    "ok", "not_modified", "http_error", "timeout", "rate_limited", "blocked")
_BY_CODE = {304: NOT_MODIFIED, 403: BLOCKED, 429: RATE_LIMITED}

DETAIL_CHARS = 200
CHUNK_BYTES = 65536
_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
_CONTACT = re.compile(r"[^\s@<>()\",;]+@[^\s@<>()\",;]+\.[A-Za-z]{2,}")


class _TooLarge(Exception):
    """A body past max_bytes, before or after gzip decoding."""


class RequestDeadline(TimeoutError):
    """The whole request took longer than the timeout, which urllib applies only to each socket operation."""


@dataclass(frozen=True)
class Validators:
    """The ETag and Last-Modified of a channel's last good response, sent back as If-None-Match and
    If-Modified-Since. A 304 or a failure hands back the ones sent, and the collector keeps them per channel and
    replaces them only after a poll it parsed."""

    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class Response:
    """One GET as a poll status. body holds the decoded bytes of a 2xx answer and is empty otherwise, wire_bytes the
    bytes read, date_ms the response Date header for the clock check, and detail a redacted reason for a failure."""

    status: str
    http_status: int | None
    body: bytes
    wire_bytes: int
    duration_ms: int
    date_ms: int | None
    validators: Validators
    detail: str = ""

    @property
    def ok(self) -> bool:
        """A 2xx answer or a 304, both a successful poll."""
        return self.status in (OK, NOT_MODIFIED)


def host_group(url: str) -> str:
    """The budget group of an https URL on one of the source hosts, refusing any other URL."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        port = -1
    if parts.scheme != "https" or host not in HOSTS or parts.username or parts.password or port not in (None, 443):
        raise ValueError(redact(f"refusing a URL that is not https on a news source host, found host {host!r}"))
    return HOSTS[host]


class SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to https on the same host and port, so the contact address and the token never reach
    another host. Any other redirect comes back as its 3xx status."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Message,
                         newurl: str) -> urllib.request.Request | None:
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        if new.scheme != "https" or (new.hostname, new.port) != (old.hostname, old.port):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_opener() -> urllib.request.OpenerDirector:
    """An opener that ignores every proxy setting in the environment and follows only same host redirects."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), SameHostRedirect())


def _date_ms(headers: Message | None) -> int | None:
    """The server's time in UTC milliseconds, the Date header plus any Age a cache added for how long it kept the
    answer, or None when Date is missing or unreadable. A zone of -0000 is read as UTC, and an Age that is not a
    whole number of seconds is ignored."""
    value = headers.get("Date") if headers is not None else None
    if not value:
        return None
    try:
        date = email.utils.parsedate_to_datetime(value)
        ms = round(date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp() * 1000)
    except (TypeError, ValueError, OverflowError):
        return None
    age = (headers.get("Age") or "").strip()
    return ms + int(age) * 1000 if age.isascii() and age.isdigit() else ms


def _validators(headers: Message) -> Validators:
    """The ETag and Last-Modified of a 2xx answer, empty when it carries neither."""
    return Validators(headers.get("ETag"), headers.get("Last-Modified"))


def _gunzip(raw: bytes, max_bytes: int) -> bytes:
    """Decode one gzip stream, refusing output past max_bytes before it is all in memory."""
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    body = decoder.decompress(raw, max_bytes + 1)
    if len(body) > max_bytes:
        raise _TooLarge
    if not decoder.eof:
        raise zlib.error("the gzip stream is cut short")
    return body


def _detail(text: str) -> str:
    """A failure reason redacted before it is cut, so a cut never leaves part of a secret behind."""
    return redact(text)[:DETAIL_CHARS]


def _describe(exc: BaseException) -> str:
    """An exception as its type name and text, redacted and cut."""
    return _detail(f"{type(exc).__name__}: {exc}")


class HttpClient:
    """Sends the news process's GET requests. It holds the contact address and the token only to build headers, and
    its repr shows neither."""

    def __init__(self, contact: str | None = None, finnhub_token: str | None = None, timeout_s: float = 10.0,
                 max_bytes: int = 8_000_000, opener: Any = None,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        if contact is not None and not (contact.isascii() and _CONTACT.fullmatch(contact)):
            raise ValueError("the contact address must be one plain email address")
        if finnhub_token is not None and not _TOKEN.fullmatch(finnhub_token):
            raise ValueError("the Finnhub token may hold only letters, digits, underscores and hyphens")
        self._contact = contact
        self._token = finnhub_token
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self._opener = opener if opener is not None else build_opener()
        self._monotonic = monotonic

    @classmethod
    def from_config(cls, config: NewsConfig, secrets: Secrets) -> HttpClient:
        """A client with the contact address when EDGAR or BLS is enabled and the token when Finnhub is."""
        contact = secrets.get(local_secrets.CONTACT) if config.edgar.enabled or config.bls.enabled else None
        token = secrets.get(local_secrets.FINNHUB) if config.finnhub.enabled else None
        return cls(contact, token, config.news.http_timeout_seconds, config.news.max_response_bytes)

    def __repr__(self) -> str:
        return (f"HttpClient(contact={self._contact is not None}, finnhub_token={self._token is not None}, "
                f"timeout_s={self.timeout_s}, max_bytes={self.max_bytes})")

    def request(self, url: str, validators: Validators | None = None) -> urllib.request.Request:
        """The GET request for url with its headers, refusing a URL that holds a secret or any token parameter."""
        group = host_group(url)
        held = [value for value in (self._contact, self._token) if value]
        if redact(url) != url or any(value.lower() in url.lower() for value in held):
            raise ValueError("refusing a URL that holds a secret")
        if any("token" in key.lower() for key, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True)):
            raise ValueError(f"refusing a URL with a token parameter, the token goes only in {TOKEN_HEADER}")
        headers = {"Accept-Encoding": "gzip", "User-Agent": USER_AGENT}
        if group in CONTACT_GROUPS:
            if self._contact is None:
                raise ValueError(f"{group} requires the contact address, which this client was not given")
            headers["User-Agent"] = f"{USER_AGENT} {self._contact}"
        if group == FINNHUB:
            if self._token is None:
                raise ValueError("Finnhub needs the token, which this client was not given")
            headers[TOKEN_HEADER] = self._token
        if validators is not None and validators.etag:
            headers["If-None-Match"] = validators.etag
        if validators is not None and validators.last_modified:
            headers["If-Modified-Since"] = validators.last_modified
        return urllib.request.Request(url, headers=headers, method="GET")

    def get(self, url: str, validators: Validators | None = None) -> Response:
        """Send one GET and return its poll status. A network or HTTP failure never raises, only a URL this client
        must not call does."""
        validators = validators or Validators()
        request = self.request(url, validators)
        start = self._monotonic()
        try:
            with self._opener.open(request, timeout=self.timeout_s) as answer:
                return self._answer(answer, start, validators)
        except urllib.error.HTTPError as exc:
            exc.close()
            status = _BY_CODE.get(exc.code, HTTP_ERROR)
            detail = "" if status == NOT_MODIFIED else _describe(exc)
            return Response(status, exc.code, b"", 0, self._elapsed_ms(start), _date_ms(exc.headers), validators,
                            detail)
        except urllib.error.URLError as exc:
            status = TIMEOUT if isinstance(exc.reason, TimeoutError) else HTTP_ERROR
            return self._failure(status, None, start, validators, _describe(exc))
        except TimeoutError as exc:
            return self._failure(TIMEOUT, None, start, validators, _describe(exc))
        except (OSError, HTTPException, zlib.error) as exc:
            return self._failure(HTTP_ERROR, None, start, validators, _describe(exc))

    async def fetch(self, url: str, validators: Validators | None = None) -> Response:
        """get in a worker thread, so the event loop never waits on the network."""
        return await asyncio.to_thread(self.get, url, validators)

    def _answer(self, answer: Any, start: float, validators: Validators) -> Response:
        """Read and decode a 2xx answer, refusing a body past max_bytes or an encoding other than gzip."""
        headers = answer.headers
        length = headers.get("Content-Length", "")
        encoding = (headers.get("Content-Encoding") or "identity").strip().lower()
        if encoding not in ("identity", "gzip", "x-gzip"):
            return self._failure(HTTP_ERROR, answer.status, start, validators, f"unsupported encoding {encoding}")
        try:
            if length.isdigit() and int(length) > self.max_bytes:
                raise _TooLarge
            raw = self._read(answer, start)
            body = raw if encoding == "identity" else _gunzip(raw, self.max_bytes)
        except _TooLarge:
            return self._failure(HTTP_ERROR, answer.status, start, validators,
                                 f"the body is larger than {self.max_bytes} bytes")
        return Response(OK, answer.status, body, len(raw), self._elapsed_ms(start), _date_ms(headers),
                        _validators(headers))

    def _read(self, answer: Any, start: float) -> bytes:
        """The raw body, read in chunks until the end, past max_bytes or past the timeout for the whole request."""
        chunks: list[bytes] = []
        total = 0
        while True:
            if self._monotonic() - start > self.timeout_s:
                raise RequestDeadline(f"the request took longer than {self.timeout_s} s")
            chunk = answer.read(min(CHUNK_BYTES, self.max_bytes + 1 - total))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
            if total > self.max_bytes:
                raise _TooLarge

    def _failure(self, status: str, http_status: int | None, start: float, validators: Validators,
                 detail: str) -> Response:
        """A failed poll with no body, handing back the validators that were sent."""
        return Response(status, http_status, b"", 0, self._elapsed_ms(start), None, validators, _detail(detail))

    def _elapsed_ms(self, start: float) -> int:
        """Milliseconds since start on the monotonic clock."""
        return max(0, round((self._monotonic() - start) * 1000))
