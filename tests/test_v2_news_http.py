"""The one HTTP client of the news process, PLAN_V2 section 7.2, against a fake opener, so no test reaches the
network. The contact address goes only to sec.gov, data.sec.gov and bls.gov, the Finnhub token only in its header
and never in a URL, gzip is decoded, an oversized body is refused, a 304 is a success, no proxy is ever used and every
failure detail is redacted. The contact address and token are fake canaries."""

import asyncio
import email.message
import gzip
import io
import threading
import urllib.error
import urllib.request
import urllib.response
from datetime import datetime, timezone
from http.client import IncompleteRead

import pytest

from btc_survivor.v2 import redact
from btc_survivor.v2.config import parse_news_config
from btc_survivor.v2.local_secrets import ANTHROPIC, CONTACT, FINNHUB, SecretsError, load_secrets
from btc_survivor.v2.news import http
from btc_survivor.v2.news.http import HttpClient, Validators
from helpers import ROOT

FAKE_CONTACT = "canary.contact@example.invalid"
FAKE_TOKEN = "CANARYnewsTOKEN0077"
NEWS = ROOT / "configs" / "v2" / "news.toml"
SEC_ATOM = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0" \
           "&count=100&output=atom"
FINNHUB_NEWS = "https://finnhub.io/api/v1/news?category=general&minId=7"
FED_PRESS = "https://www.federalreserve.gov/feeds/press_all.xml"
CONTACT_URLS = (SEC_ATOM, "https://sec.gov/files/company_tickers.json",
                "https://data.sec.gov/submissions/CIK0000320193.json", "https://www.bls.gov/feed/cpi.rss",
                "https://bls.gov/feed/empsit.rss")
OTHER_URLS = (FED_PRESS, "https://federalreserve.gov/feeds/speeches.xml", FINNHUB_NEWS,
              "https://finnhub.io/api/v1/company-news?symbol=AAPL&from=2026-10-11&to=2026-10-12")


@pytest.fixture(autouse=True)
def canaries(monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    redact.register(FAKE_CONTACT)
    redact.register(FAKE_TOKEN)


def message(**values):
    """Response headers, with underscores in the names written as hyphens."""
    headers = email.message.Message()
    for name, value in values.items():
        headers[name.replace("_", "-")] = value
    return headers


def answer(body, code=200, **headers):
    return urllib.response.addinfourl(io.BytesIO(body), message(**headers), "https://fake.invalid", code)


def status(code, reason="reason", **headers):
    def respond(request):
        raise urllib.error.HTTPError(request.full_url, code, reason, message(**headers), None)
    return respond


def raising(exc):
    def respond(request):
        raise exc
    return respond


class FakeOpener:
    """Records each request, its timeout and its thread, and answers with what respond returns or raises."""

    def __init__(self, respond=None):
        self.calls = []
        self.respond = respond or (lambda request: answer(b"[]"))

    def open(self, request, timeout=None):
        self.calls.append((request, timeout, threading.current_thread()))
        return self.respond(request)


def client(respond=None, **settings):
    opener = FakeOpener(respond)
    return HttpClient(FAKE_CONTACT, FAKE_TOKEN, opener=opener, **settings), opener


def sent_headers(opener):
    return {name.lower(): value for name, value in opener.calls[-1][0].header_items()}


def test_the_contact_address_goes_only_to_sec_and_bls():
    c, opener = client()
    for url in CONTACT_URLS:
        assert c.get(url).ok
        assert sent_headers(opener)["user-agent"] == f"btc-survivor/2 {FAKE_CONTACT}"
    for url in OTHER_URLS:
        assert c.get(url).ok
        headers = sent_headers(opener)
        assert headers["user-agent"] == "btc-survivor/2"
        assert not any(FAKE_CONTACT in value for value in headers.values())
    assert not any(FAKE_CONTACT in request.full_url for request, _, _ in opener.calls)


def test_the_finnhub_token_travels_only_in_its_header():
    c, opener = client()
    c.get(FINNHUB_NEWS)
    request = opener.calls[-1][0]
    assert sent_headers(opener)["x-finnhub-token"] == FAKE_TOKEN
    assert FAKE_TOKEN not in request.full_url and "token" not in request.full_url.lower()
    for url in (*CONTACT_URLS, FED_PRESS):
        c.get(url)
        assert "x-finnhub-token" not in sent_headers(opener)
        assert not any(FAKE_TOKEN in value for value in sent_headers(opener).values())


@pytest.mark.parametrize("url", [
    f"https://finnhub.io/api/v1/news?category=general&token={FAKE_TOKEN}",
    f"https://finnhub.io/api/v1/{FAKE_TOKEN}/news",
    "https://finnhub.io/api/v1/news?category=general&TOKEN=fake",
    "https://finnhub.io/api/v1/news?api_token=fake",
    f"https://www.sec.gov/{FAKE_CONTACT.upper()}",
])
def test_a_url_holding_a_secret_or_a_token_parameter_is_refused_unsent(url):
    c, opener = client()
    with pytest.raises(ValueError) as info:
        c.get(url)
    assert FAKE_TOKEN not in str(info.value) and FAKE_CONTACT not in str(info.value)
    assert opener.calls == []


@pytest.mark.parametrize("url", [
    "http://www.sec.gov/cgi-bin/browse-edgar",
    "ftp://www.sec.gov/x",
    "file:///etc/hosts",
    "https://sec.gov.example.invalid/x",
    "https://notsec.gov/x",
    "https://www.bls.gov.example.invalid/feed/cpi.rss",
    "https://user:pw@www.sec.gov/x",
    "https://www.sec.gov:8443/x",
    "https://www.sec.gov:port/x",
    "https://api.anthropic.com/v1/messages",
    "https://127.0.0.1/feeds/press_all.xml",
])
def test_only_https_on_the_source_hosts_is_ever_called(url):
    c, opener = client()
    with pytest.raises(ValueError, match="refusing a URL"):
        c.get(url)
    assert opener.calls == []


def test_a_source_needs_its_credential_in_the_client():
    opener = FakeOpener()
    bare = HttpClient(opener=opener)
    for url in (SEC_ATOM, "https://www.bls.gov/feed/cpi.rss"):
        with pytest.raises(ValueError, match="requires the contact address"):
            bare.get(url)
    with pytest.raises(ValueError, match="Finnhub needs the token"):
        bare.get(FINNHUB_NEWS)
    assert bare.get(FED_PRESS).ok and len(opener.calls) == 1
    for contact in ("two@example.invalid three@example.invalid", "line@example.invalid\r\nX-Other: 1", "no-at-sign",
                    "café@example.invalid"):
        with pytest.raises(ValueError, match="one plain email address"):
            HttpClient(contact=contact, opener=opener)
    with pytest.raises(ValueError, match="only letters, digits"):
        HttpClient(finnhub_token="abc def", opener=opener)


def test_gzip_is_asked_for_and_decoded():
    data = b'[{"id": 7, "headline": "Fictional headline"}]'
    packed = gzip.compress(data)
    c, opener = client(lambda request: answer(packed, Content_Encoding="gzip"))
    response = c.get(FINNHUB_NEWS)
    assert sent_headers(opener)["accept-encoding"] == "gzip"
    assert (response.status, response.http_status, response.body, response.wire_bytes) == ("ok", 200, data,
                                                                                           len(packed))
    c, _ = client(lambda request: answer(data))
    assert c.get(FINNHUB_NEWS).body == data


class Unreadable(io.BytesIO):
    def read(self, size=-1):
        raise AssertionError("an oversized body must be refused before it is read")


def test_an_oversized_body_is_refused():
    def sized(body, **headers):
        c, _ = client(lambda request: answer(body, **headers), max_bytes=100)
        return c.get(FED_PRESS)

    assert sized(b"x" * 100).body == b"x" * 100
    for response in (sized(b"x" * 101), sized(gzip.compress(b"0" * 101), Content_Encoding="gzip")):
        assert (response.status, response.ok, response.body) == ("http_error", False, b"")
        assert response.detail == "the body is larger than 100 bytes"
    assert sized(gzip.compress(b"0" * 100), Content_Encoding="gzip").body == b"0" * 100
    huge = urllib.response.addinfourl(Unreadable(), message(Content_Length="9000000"), "https://fake.invalid", 200)
    c, _ = client(lambda request: huge, max_bytes=8_000_000)
    assert c.get(FED_PRESS).detail == "the body is larger than 8000000 bytes"


def test_a_cut_gzip_stream_or_another_encoding_fails():
    cut = gzip.compress(b"fictional text " * 50)[:-12]
    c, _ = client(lambda request: answer(cut, Content_Encoding="gzip"))
    response = c.get(FED_PRESS)
    assert response.status == "http_error" and "gzip stream is cut short" in response.detail
    c, _ = client(lambda request: answer(b"x", Content_Encoding="br"))
    assert c.get(FED_PRESS).detail == "unsupported encoding br"


def test_a_304_is_a_success_and_validators_go_back_to_the_host():
    old = Validators('"v1"', "Sun, 11 Oct 2026 10:00:00 GMT")
    c, opener = client(status(304, "Not Modified"))
    response = c.get(SEC_ATOM, old)
    headers = sent_headers(opener)
    assert (headers["if-none-match"], headers["if-modified-since"]) == ('"v1"', "Sun, 11 Oct 2026 10:00:00 GMT")
    assert (response.status, response.ok, response.http_status, response.body) == ("not_modified", True, 304, b"")
    assert response.validators == old and response.detail == ""
    c, opener = client(lambda request: answer(b"<feed/>", ETag='"v2"', Last_Modified="Mon, 12 Oct 2026 10:00:00 GMT"))
    response = c.get(SEC_ATOM)
    assert "if-none-match" not in sent_headers(opener) and "if-modified-since" not in sent_headers(opener)
    assert response.validators == Validators('"v2"', "Mon, 12 Oct 2026 10:00:00 GMT")
    c, _ = client(lambda request: answer(b"<feed/>"))
    assert c.get(SEC_ATOM, old).validators == Validators()
    c, _ = client(status(500))
    assert c.get(SEC_ATOM, old).validators == old


@pytest.mark.parametrize("code, expected", [(403, "blocked"), (429, "rate_limited"), (500, "http_error"),
                                            (503, "http_error"), (404, "http_error"), (302, "http_error")])
def test_http_errors_map_to_poll_statuses(code, expected):
    c, _ = client(status(code, Date="Mon, 12 Oct 2026 14:30:00 GMT"))
    response = c.get(SEC_ATOM)
    assert (response.status, response.ok, response.http_status, response.body) == (expected, False, code, b"")
    assert response.detail == f"HTTPError: HTTP Error {code}: reason"
    assert response.date_ms is not None


class Drip(io.BytesIO):
    """A body that arrives one byte per read while a fake clock moves 3 seconds."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def read(self, size=-1):
        self.clock[0] += 3.0
        return b"x"


def test_timeouts_and_lost_connections_are_poll_statuses():
    cases = {
        "timeout": [urllib.error.URLError(TimeoutError("timed out")), TimeoutError("The read operation timed out")],
        "http_error": [urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
                       ConnectionResetError(54, "Connection reset by peer"), IncompleteRead(b"x", 10)],
    }
    for expected, errors in cases.items():
        for exc in errors:
            c, _ = client(raising(exc))
            response = c.get(SEC_ATOM)
            assert (response.status, response.http_status, response.body) == (expected, None, b""), exc
            assert response.detail.startswith(type(exc).__name__)
    now = [0.0]
    c, _ = client(lambda request: urllib.response.addinfourl(Drip(now), message(), "https://fake.invalid", 200),
                  monotonic=lambda: now[0])
    response = c.get(FED_PRESS)
    assert (response.status, response.duration_ms) == ("timeout", 12_000)
    assert response.detail == "RequestDeadline: the request took longer than 10.0 s"


def test_every_failure_detail_is_redacted_before_it_is_cut():
    failures = [
        raising(urllib.error.URLError(f"failed near {FAKE_TOKEN} for {FAKE_CONTACT}")),
        raising(OSError(f"{'x' * 185}{FAKE_TOKEN}")),
        status(500, f"bad {FAKE_TOKEN}"),
        raising(TimeoutError(FAKE_CONTACT)),
    ]
    for respond in failures:
        c, _ = client(respond)
        detail = c.get(SEC_ATOM).detail
        assert redact.MASK[:5] in detail and len(detail) <= http.DETAIL_CHARS
        assert FAKE_TOKEN[:6] not in detail and FAKE_CONTACT[:6] not in detail


def test_the_date_header_is_read_for_the_clock_check():
    c, _ = client(lambda request: answer(b"[]", Date="Mon, 12 Oct 2026 14:30:00 GMT"))
    expected = datetime(2026, 10, 12, 14, 30, tzinfo=timezone.utc).timestamp() * 1000
    assert c.get(FED_PRESS).date_ms == expected
    c, _ = client(lambda request: answer(b"[]", Date="Mon, 12 Oct 2026 14:30:00 -0000"))
    assert c.get(FED_PRESS).date_ms == expected
    c, _ = client(lambda request: answer(b"[]", Date="not a date"))
    assert c.get(FED_PRESS).date_ms is None
    assert client()[0].get(FED_PRESS).date_ms is None


@pytest.mark.parametrize("age, added", [("30", 30_000), ("0", 0), ("-5", 0), ("soon", 0), ("1.5", 0)])
def test_a_cached_answer_adds_its_age_to_the_date_header(age, added):
    c, _ = client(lambda request: answer(b"[]", Date="Mon, 12 Oct 2026 14:30:00 GMT", Age=age))
    expected = datetime(2026, 10, 12, 14, 30, tzinfo=timezone.utc).timestamp() * 1000
    assert c.get(FED_PRESS).date_ms == expected + added


def secrets_file(folder, *names):
    values = {FINNHUB: FAKE_TOKEN, CONTACT: FAKE_CONTACT, ANTHROPIC: "fake-anthropic-value"}
    path = folder / "secrets.env"
    path.write_text("".join(f"{name}={values[name]}\n" for name in names), encoding="utf-8")
    path.chmod(0o600)
    return load_secrets(path)


def news_config(*disabled):
    text = NEWS.read_text(encoding="utf-8")
    for section in disabled:
        assert text.count(f"[{section}]\nenabled = true\n") == 1
        text = text.replace(f"[{section}]\nenabled = true\n", f"[{section}]\nenabled = false\n")
    return parse_news_config(text, NEWS.parent)


def test_from_config_takes_the_timeout_the_cap_and_only_the_secrets_its_sources_need(tmp_path):
    full = HttpClient.from_config(news_config(), secrets_file(tmp_path, FINNHUB, CONTACT, ANTHROPIC))
    assert (full.timeout_s, full.max_bytes) == (10, 8_000_000)
    assert FAKE_TOKEN not in repr(full) and FAKE_CONTACT not in repr(full)
    opener = FakeOpener()
    full._opener = opener
    full.get(SEC_ATOM)
    assert opener.calls[-1][1] == 10
    HttpClient.from_config(news_config("finnhub"), secrets_file(tmp_path, CONTACT))
    HttpClient.from_config(news_config("edgar", "bls"), secrets_file(tmp_path, FINNHUB))
    with pytest.raises(SecretsError, match=FINNHUB):
        HttpClient.from_config(news_config(), secrets_file(tmp_path, CONTACT))
    with pytest.raises(SecretsError, match=CONTACT):
        HttpClient.from_config(news_config("edgar"), secrets_file(tmp_path, FINNHUB))


def test_no_proxy_is_ever_used(monkeypatch):
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    stock = urllib.request.build_opener()
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies for h in stock.handlers)
    for opener in (http.build_opener(), HttpClient()._opener):
        assert not any(isinstance(h, urllib.request.ProxyHandler) for h in opener.handlers)
        assert not any(getattr(h, "proxies", None) for h in opener.handlers)
        redirects = [h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
        assert [type(h) for h in redirects] == [http.SameHostRedirect]


def test_a_redirect_is_followed_only_to_https_on_the_same_host():
    c, _ = client()
    request = c.request(SEC_ATOM)
    handler = http.SameHostRedirect()

    def follow(newurl):
        return handler.redirect_request(request, io.BytesIO(), 301, "Moved", message(), newurl)

    same = follow("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom")
    assert same is not None and same.get_header("User-agent") == f"btc-survivor/2 {FAKE_CONTACT}"
    for newurl in ("https://example.invalid/x", "http://www.sec.gov/x", "https://data.sec.gov/x",
                   "https://www.sec.gov:8443/x"):
        assert follow(newurl) is None, newurl


def test_fetch_runs_the_request_in_a_worker_thread():
    c, opener = client()
    assert asyncio.run(c.fetch(FED_PRESS)).ok
    assert opener.calls[-1][2] is not threading.main_thread()
