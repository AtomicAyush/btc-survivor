"""The news sources on hand written fixtures, the SEC, Fed and BLS ones in the shape of the public feeds with fictional
text and the Finnhub ones fictional, with no network and no secret."""

import ast
import json
import re

import pytest

from btc_survivor.v2.config import NEWS_KINDS, load_news_config
from btc_survivor.v2.news import edgar, edgar_docs, finnhub_news, rss
from btc_survivor.v2.news.edgar_docs import Doc, FilingDocs
from btc_survivor.v2.news.rss import RawItem
from helpers import ROOT

FIXTURES = ROOT / "tests" / "fixtures"
CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
FH, ED, FED, BLS = CONFIG.finnhub, CONFIG.edgar, CONFIG.fed, CONFIG.bls
NOW_MS = 1_791_812_700_000                      # Monday 12 October 2026 13:45 UTC
STATEMENT = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000101/"
SOURCES = (finnhub_news, edgar, edgar_docs, rss)
UTC_TEXT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


def fixture(name):
    return (FIXTURES / name).read_bytes()


def finnhub_body(*entries):
    return json.dumps(list(entries)).encode()


def entry(news_id, headline="Fixture headline", **fields):
    return {"category": "company", "datetime": 1_791_812_700, "headline": headline, "id": news_id, "image": "",
            "related": "AAPL", "source": "Fixture Wire", "summary": "", "url": "", **fields}


def fed_items(name):
    return rss.fed_items(fixture("news/" + name), FED.monetary_link_pattern, FED.summary_max_chars)


def all_fixture_items():
    items = finnhub_news.parse_news(fixture("finnhub/company_news_aapl.json"), FH.headline_max_chars,
                                    FH.summary_max_chars)
    items += finnhub_news.parse_news(fixture("finnhub/general_news.json"), FH.headline_max_chars,
                                     FH.summary_max_chars, general=True)
    items += edgar.parse_atom(fixture("news/edgar_atom_8k.xml"), ED.ciks, ED.forms)
    items += edgar.parse_submissions(fixture("news/edgar_submissions_aapl.json"), "AAPL", 320193, ED.forms)
    for name in ("fed_press_all.xml", "fed_press_monetary.xml", "fed_speeches.xml"):
        items += fed_items(name)
    return items + rss.bls_items(fixture("news/bls_cpi.rss"), BLS.summary_max_chars)


# Finnhub


def test_company_url_covers_yesterday_and_today_in_utc_dates_and_holds_no_token():
    for symbol in FH.company_symbols:
        url = finnhub_news.company_url(symbol, NOW_MS, FH.lookback_days)
        assert url == f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2026-10-11&to=2026-10-12"
        assert "token" not in url.lower()
    # 00:30 UTC on 12 October is still 11 October in New York, and the dates stay UTC dates.
    late = finnhub_news.company_url("AAPL", 1_791_765_000_000, FH.lookback_days)
    assert late.endswith("&from=2026-10-11&to=2026-10-12")


def test_general_url_carries_the_cursor_and_inputs_are_checked():
    url = finnhub_news.general_url(FH.general_category, 140000203)
    assert url == "https://finnhub.io/api/v1/news?category=general&minId=140000203"
    assert finnhub_news.general_url("general", 0).endswith("&minId=0")
    for bad_symbol in ("aapl", "AAPL&token=x", "", "TOOLONGSYMBOL"):
        with pytest.raises(ValueError):
            finnhub_news.company_url(bad_symbol, NOW_MS, 1)
    for category, min_id in (("General", 0), ("general&x=1", 0), ("general", -1), ("general", True),
                             ("general", 1.5)):
        with pytest.raises(ValueError):
            finnhub_news.general_url(category, min_id)


def test_company_news_parses_the_documented_fields(caplog):
    items = finnhub_news.parse_news(fixture("finnhub/company_news_aapl.json"), FH.headline_max_chars,
                                    FH.summary_max_chars)
    assert [i.vendor_id for i in items] == ["140000105", "140000104", "140000103", "140000100"]
    assert all(i.source == "finnhub" and i.kind == "media" and not i.general for i in items)
    first, second, third, last = items
    assert first == RawItem("finnhub", "140000105", "media", "Apple fixture headline about a supplier meeting",
                            "2026-10-12T13:45:00.000Z",
                            summary="A fictional summary written by hand for the parser tests.",
                            url="https://news.example.com/fixture/140000105", publisher="Fixture Wire",
                            related=("AAPL",))
    assert second.title == "Apple & Microsoft fixture headline on a cloud partnership"
    assert second.summary == "A second fictional summary, spread over a line break."
    assert second.related == ("AAPL", "MSFT")
    assert (third.summary, third.url) == ("", "")
    assert (last.vendor_ts, last.related) == (None, ())
    assert "3 Finnhub news entries dropped" in caplog.text


def test_general_news_with_min_id_returns_only_newer_ids():
    body = fixture("finnhub/general_news.json")
    every = finnhub_news.parse_news(body, FH.headline_max_chars, FH.summary_max_chars, general=True)
    assert [i.vendor_id for i in every] == ["140000205", "140000204", "140000203", "140000202", "140000201"]
    newer = finnhub_news.parse_news(body, FH.headline_max_chars, FH.summary_max_chars, min_id=140000203,
                                    general=True)
    assert [i.vendor_id for i in newer] == ["140000205", "140000204"]
    assert all(i.general for i in newer)
    assert finnhub_news.newest_id(newer, 140000203) == 140000205
    assert finnhub_news.newest_id([], 140000203) == 140000203
    assert finnhub_news.parse_news(body, FH.headline_max_chars, FH.summary_max_chars, min_id=140000205) == []


def test_company_news_with_min_id_returns_only_newer_ids():
    body = fixture("finnhub/company_news_aapl.json")
    newer = finnhub_news.parse_news(body, FH.headline_max_chars, FH.summary_max_chars, min_id=140000103)
    assert [i.vendor_id for i in newer] == ["140000105", "140000104"]
    assert finnhub_news.newest_id(newer, 140000103) == 140000105


def test_headline_and_summary_are_cut_to_the_configured_lengths():
    body = finnhub_body(entry(7, headline="word " * 100, summary="text " * 200))
    [item] = finnhub_news.parse_news(body, FH.headline_max_chars, FH.summary_max_chars)
    assert (len(item.title), len(item.summary)) == (FH.headline_max_chars, FH.summary_max_chars) == (300, 600)


def test_odd_finnhub_values_never_break_the_parse():
    body = finnhub_body(entry(1, datetime=True), entry(2, datetime=float("1e30")), entry(3, datetime=-5),
                        entry(4, related=None, source=None, summary=7), entry(True), entry(0), entry(-3),
                        entry(5, headline=None))
    items = finnhub_news.parse_news(body, FH.headline_max_chars, FH.summary_max_chars)
    assert [i.vendor_id for i in items] == ["1", "2", "3", "4"]
    assert [i.vendor_ts for i in items] == [None, None, None, "2026-10-12T13:45:00.000Z"]
    assert (items[3].related, items[3].publisher, items[3].summary) == ((), "", "")


def test_a_finnhub_body_that_is_not_a_list_raises():
    with pytest.raises(ValueError, match="not a list"):
        finnhub_news.parse_news(b'{"error": "fixture error text"}', 300, 600)
    for body in (b"not json", b"\xff\xfe\x00", b"[" * 100_000):
        with pytest.raises(ValueError, match="not JSON"):
            finnhub_news.parse_news(body, 300, 600)
    assert finnhub_news.parse_news(b"[]", 300, 600) == []


# SEC EDGAR


def test_the_seven_ciks_and_their_zero_padded_submissions_urls():
    assert ED.ciks == {"AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "AMZN": 1018724, "GOOGL": 1652044,
                       "META": 1326801, "TSLA": 1318605}
    for cik in ED.ciks.values():
        assert edgar.submissions_url(cik) == f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    assert edgar.submissions_url(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert edgar.index_url(320193, "0000320193-26-000101") == ARCHIVE + "0000320193-26-000101-index.htm"
    assert edgar.ATOM_URL == ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb="
                              "&owner=include&start=0&count=100&output=atom")
    assert {edgar.KINDS[form] for form in ED.forms} == {"8k", "8k_a"} <= set(NEWS_KINDS)


def test_atom_feed_keeps_the_two_forms_for_the_seven_ciks(caplog):
    items = edgar.parse_atom(fixture("news/edgar_atom_8k.xml"), ED.ciks, ED.forms)
    assert [(i.vendor_id, i.kind, i.filer, i.form, i.items_8k, i.vendor_ts) for i in items] == [
        ("0000320193-26-000101", "8k", "AAPL", "8-K", "2.02,9.01", "2026-10-08T20:30:12.000Z"),
        ("0001193125-26-400001", "8k_a", "MSFT", "8-K/A", "5.02", "2026-10-08T20:12:40.000Z"),
        ("0001628280-26-000555", "8k", "TSLA", "8-K", "7.01,9.01", "2026-10-08T10:05:00.000Z"),
    ]
    assert items[0].title == "Apple Inc. 8-K, items 2.02, 9.01"
    assert items[0].url == ARCHIVE + "0000320193-26-000101-index.htm"
    assert all(i.source == "edgar" and i.publisher == "SEC EDGAR" and not i.related for i in items)
    assert "1 EDGAR feed entries dropped" in caplog.text
    only_apple = edgar.parse_atom(fixture("news/edgar_atom_8k.xml"), {"AAPL": 320193}, ("8-K/A",))
    assert only_apple == []


def test_atom_feed_is_read_in_its_declared_latin_1_encoding():
    body = fixture("news/edgar_atom_8k.xml")
    with pytest.raises(UnicodeDecodeError):
        body.decode("utf-8")
    titles = [e.title for e in rss.parse_feed(body)]
    assert "8-K - SOCIÉTÉ EXEMPLE FIXTURE SA (0009999901) (Filer)" in titles


def test_submissions_keep_only_8k_and_8k_a_from_recent(caplog):
    items = edgar.parse_submissions(fixture("news/edgar_submissions_aapl.json"), "AAPL", 320193, ED.forms)
    assert [(i.vendor_id, i.kind, i.items_8k, i.vendor_ts) for i in items] == [
        ("0000320193-26-000101", "8k", "2.02,9.01", "2026-10-08T20:30:12.000Z"),
        ("0000320193-26-000099", "8k_a", "5.07", "2026-10-02T21:05:44.000Z"),
        ("0000320193-26-000090", "8k", "2.02,9.01", "2026-07-30T20:31:02.000Z"),
        ("0000320193-26-000020", "8k", "8.01", "2026-02-10T21:05:00.000Z"),
    ]
    assert all(i.filer == "AAPL" and i.url == edgar.index_url(320193, i.vendor_id) for i in items)
    assert "1 EDGAR submissions rows dropped" in caplog.text


def test_submissions_acceptance_times_lose_their_new_york_shift():
    # Accepted 17:30:15 EDT and 16:46:37 EST by their index pages, stated four and five hours later with a Z.
    assert edgar.accepted_time("2026-10-10T01:30:15.000Z") == "2026-10-09T21:30:15.000Z"
    assert edgar.accepted_time("2026-02-28T02:46:37.000Z") == "2026-02-27T21:46:37.000Z"
    for unreadable in ("", "2026-10-10T01:30:15", "late"):
        assert edgar.accepted_time(unreadable) is None


def test_a_filing_seen_on_both_edgar_paths_is_one_item():
    from_atom = edgar.parse_atom(fixture("news/edgar_atom_8k.xml"), ED.ciks, ED.forms)
    from_submissions = edgar.parse_submissions(fixture("news/edgar_submissions_aapl.json"), "AAPL", 320193,
                                               ED.forms)
    assert from_atom[0] == from_submissions[0]
    for first, second in ((from_atom, from_submissions), (from_submissions, from_atom)):
        store = {}
        for item in first + second:
            store.setdefault((item.source, item.vendor_id), item)
        assert len(store) == 6
        assert store[("edgar", "0000320193-26-000101")] == from_atom[0]


def test_a_submissions_file_of_another_cik_or_shape_raises():
    body = fixture("news/edgar_submissions_aapl.json")
    with pytest.raises(ValueError, match="CIK 789019"):
        edgar.parse_submissions(body, "MSFT", 789019, ED.forms)
    data = json.loads(body)
    data["filings"]["recent"]["form"].pop()
    for bad in (b"{}", b"[]", b"{", b'{"filings": []}', json.dumps(data).encode()):
        with pytest.raises(ValueError):
            edgar.parse_submissions(bad, "AAPL", 320193, ED.forms)


def test_index_page_gives_the_main_document_and_the_first_exhibit_99_1():
    docs = edgar_docs.filing_documents(fixture("news/edgar_index_8k.htm"))
    assert docs == FilingDocs(ARCHIVE + "aapl-20261008.htm", ARCHIVE + "a8-kex991q4fy26.htm")


def index_page(*rows):
    cells = "".join(f'<tr><td>{seq}</td><td>{kind}</td><td><a href="{href}">x</a></td><td>{kind}</td><td>1</td></tr>'
                    for seq, kind, href in rows)
    return f'<table class="tableFile"><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th>' \
           f'<th>Size</th></tr>{cells}</table>'.encode()


def test_exhibit_variants_a_missing_exhibit_and_links_off_the_archive():
    main = "/Archives/edgar/data/1318605/000162828026000555/tsla-8k.htm"
    exhibit = "/Archives/edgar/data/1318605/000162828026000555/tsla-ex9901.htm"
    sec = "https://www.sec.gov"
    page = index_page((1, "8-K", main), (2, "EX-99.01", exhibit))
    assert edgar_docs.filing_documents(page) == FilingDocs(sec + main, sec + exhibit)
    page = index_page((1, "8-K/A", sec + main), (2, "EX-99.2", exhibit), (3, "EX-99", exhibit))
    assert edgar_docs.filing_documents(page) == FilingDocs(sec + main, None)
    page = index_page((1, "8-K", "https://elsewhere.example.com" + main), (2, "EX-99.1", "/cgi-bin/x?y=1"),
                      (3, "EX-99.1", "javascript:alert(1)"))
    assert edgar_docs.filing_documents(page) == FilingDocs(None, None)
    assert edgar_docs.filing_documents(b"<html><body>no table</body></html>") == FilingDocs(None, None)


def test_document_text_strips_tags_hidden_parts_and_entities():
    doc = edgar_docs.document(fixture("news/edgar_8k_main.htm"), ED.main_doc_max_chars)
    assert doc.text.startswith("UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 8-K")
    assert "Item 2.02 Results of Operations and Financial Condition." in doc.text
    assert "registrant’s “results”" in doc.text
    assert "00-0000000 (State or other jurisdiction" in doc.text
    assert "99.1 Press release issued" in doc.text
    assert "HIDDEN" not in doc.text and "  " not in doc.text and "\xa0" not in doc.text
    assert (doc.chars_total, doc.truncated) == (len(doc.text), False)
    exhibit = edgar_docs.document(fixture("news/edgar_ex99_1.htm"), ED.exhibit_max_chars)
    assert exhibit.text.startswith("Exhibit 99.1 Fixture Registrant reports fictional fourth quarter results")
    assert "— This exhibit" in exhibit.text and "unescape & text" in exhibit.text
    assert "Fixture line one two Another fixture line" in exhibit.text and "HIDDEN" not in exhibit.text


def test_markup_html_parser_refuses_falls_back_to_a_plain_tag_strip(monkeypatch):
    markup = "<p>Fixture <b>release</b> text &amp; more</p><script>var hidden = 1;</script>"
    assert edgar_docs.html_text(markup) == "Fixture release text & more"

    def refuse(self, data):
        raise AssertionError("unknown status keyword 'foo' in marked section")

    monkeypatch.setattr(edgar_docs._Text, "feed", refuse)
    assert edgar_docs.html_text(markup) == "Fixture release text & more"
    assert edgar_docs.html_text(markup, "article") is None
    assert edgar_docs.document(markup.encode(), 100, ("article",)).text == "Fixture release text & more"
    assert rss.summary_text(markup, 10) == "Fixture re"


def test_odd_markup_never_fails_a_parse_on_any_python():
    """Python 3.13's html.parser raises on an unknown marked section, which 3.11 passes over."""
    page = fixture("news/edgar_index_8k.htm")
    assert edgar_docs.filing_documents(page + b"<![foo[ x ]]>") == edgar_docs.filing_documents(page)
    assert edgar_docs.html_text("<p>kept</p><![foo[ x ]]><p>after</p>") == "kept after"
    assert rss.summary_text("<p>a</p><![foo[ x ]]>b", 100) == "a b"


def test_truncation_is_recorded_at_the_configured_lengths():
    page = ("<html><body><p>" + "fixture word " * 2000 + "</p></body></html>").encode()
    full = len(("fixture word " * 2000).strip())
    for limit in (ED.main_doc_max_chars, ED.exhibit_max_chars, FED.monetary_body_max_chars):
        doc = edgar_docs.document(page, limit)
        assert (len(doc.text), doc.chars_total, doc.truncated) == (limit, full, True)
    assert (ED.main_doc_max_chars, ED.exhibit_max_chars) == (4000, 12000)
    short = edgar_docs.document(fixture("news/edgar_ex99_1.htm"), 100)
    whole = edgar_docs.document(fixture("news/edgar_ex99_1.htm"), 100_000)
    assert short == Doc(whole.text[:100], len(whole.text), True)
    assert edgar_docs.cut("abc", 3) == Doc("abc", 3, False)


def test_filing_body_joins_the_main_document_and_the_exhibit():
    main = edgar_docs.document(fixture("news/edgar_8k_main.htm"), ED.main_doc_max_chars)
    exhibit = edgar_docs.document(fixture("news/edgar_ex99_1.htm"), 100)
    body = edgar_docs.body([(ARCHIVE + "aapl-20261008.htm", main), (ARCHIVE + "a8-kex991q4fy26.htm", exhibit)])
    assert body.text == main.text + "\n\n" + exhibit.text
    assert body.chars_total == main.chars_total + exhibit.chars_total
    assert body.truncated is True
    assert json.loads(body.doc_urls) == [ARCHIVE + "aapl-20261008.htm", ARCHIVE + "a8-kex991q4fy26.htm"]
    alone = edgar_docs.body([(ARCHIVE + "aapl-20261008.htm", main)])
    assert (alone.text, alone.truncated) == (main.text, False)


def test_a_windows_1252_page_is_decoded():
    page = "<p>café “quoted”</p>".encode("cp1252")
    assert edgar_docs.document(page, 100).text == "café “quoted”"
    assert edgar_docs.document(b"\xef\xbb\xbf<p>x</p>", 100).text == "x"


# Federal Reserve


def test_fed_feeds_parse_despite_their_byte_order_mark():
    body = fixture("news/fed_press_all.xml")
    assert body.startswith(b"\xef\xbb\xbf")
    items = fed_items("fed_press_all.xml")
    assert [i.kind for i in items] == ["fed_press", "fed_press", "fed_monetary", "fed_press"]
    statement = items[2]
    assert statement == RawItem("fed", STATEMENT, "fed_monetary", "Federal Reserve issues FOMC statement",
                                "2026-09-16T18:00:00.000Z", summary="Federal Reserve issues FOMC statement",
                                url=STATEMENT, publisher="Federal Reserve Board")


def test_a_release_in_two_fed_feeds_is_one_item_of_kind_fed_monetary_in_both_orders():
    for order in (("fed_press_all.xml", "fed_press_monetary.xml"), ("fed_press_monetary.xml", "fed_press_all.xml")):
        store = {}
        for name in order:
            for item in fed_items(name):
                store.setdefault((item.source, item.vendor_id), item)
        assert len(store) == 5
        assert store[("fed", STATEMENT)].kind == "fed_monetary"
    [from_all] = [i for i in fed_items("fed_press_all.xml") if i.vendor_id == STATEMENT]
    [from_monetary] = [i for i in fed_items("fed_press_monetary.xml") if i.vendor_id == STATEMENT]
    assert from_all == from_monetary


def test_fed_kind_comes_from_the_link_alone():
    pattern = FED.monetary_link_pattern
    assert rss.fed_kind(STATEMENT, pattern) == "fed_monetary"
    assert rss.fed_kind("https://www.federalreserve.gov/newsevents/pressreleases/bcreg20260915a.htm",
                        pattern) == "fed_press"
    assert rss.fed_kind("https://www.federalreserve.gov/newsevents/speech/example20261008a.htm",
                        pattern) == "fed_speech"
    assert rss.fed_kind("https://www.federalreserve.gov/newsevents/testimony/example20261001a.htm",
                        pattern) == "fed_speech"


def test_speeches_unescape_entities_in_their_descriptions():
    items = fed_items("fed_speeches.xml")
    assert [i.kind for i in items] == ["fed_speech", "fed_speech"]
    assert items[0].summary == ("Speech At a fixture economic forum, Istanbul, Türkiye, sponsored by a "
                                "fictional organization")
    assert items[0].vendor_ts == "2026-10-08T08:30:00.000Z"


def rss_body(title, link, description):
    return (f'<?xml version="1.0" encoding="utf-8" ?><rss version="2.0"><channel><title>t</title><item>'
            f"<title>{title}</title><link>{link}</link><guid>{link}</guid><description><![CDATA[{description}]]>"
            f"</description><pubDate>Wed, 7 Oct 2026 18:00:00 GMT</pubDate></item></channel></rss>").encode()


def test_fed_text_is_cut_and_entries_without_a_title_or_link_are_dropped(caplog):
    [item] = rss.fed_items(rss_body("word " * 100, STATEMENT, "<b>text</b> " * 400), FED.monetary_link_pattern,
                           FED.summary_max_chars)
    assert (len(item.title), len(item.summary)) == (rss.TITLE_MAX_CHARS, FED.summary_max_chars) == (300, 1000)
    assert "<b>" not in item.summary
    assert rss.fed_items(rss_body("", STATEMENT, "x"), FED.monetary_link_pattern, 1000) == []
    assert rss.fed_items(rss_body("A title", "", "x"), FED.monetary_link_pattern, 1000) == []
    assert "1 Fed entries dropped" in caplog.text


def test_a_monetary_release_page_keeps_its_article_text():
    body = rss.fed_page_body(fixture("news/fed_monetary_page.htm"), STATEMENT, FED.monetary_body_max_chars)
    assert body.text.startswith("September 16, 2026 Federal Reserve issues FOMC statement For release at 2:00 p.m. "
                                "EDT This fixture stands in for the statement text")
    assert "A paragraph inside a nested division, still part of the release." in body.text
    assert body.text.endswith("Implementation Note issued September 16, 2026")
    for outside in ("MENUTEXT", "NAVTEXT", "OUTSIDEARTICLE", "FOOTERTEXT", "never kept", "Home"):
        assert outside not in body.text
    assert (body.chars_total, body.truncated, json.loads(body.doc_urls)) == (len(body.text), False, [STATEMENT])


def test_a_release_page_falls_back_to_its_main_content_then_to_the_whole_page():
    page = fixture("news/fed_monetary_page.htm")
    content = rss.fed_page_body(page.replace(b'id="article"', b'id="other"'), STATEMENT, 8000).text
    assert content.startswith("Home News & Events Press Releases September 16, 2026")
    assert "OUTSIDEARTICLE" in content and "MENUTEXT" not in content and "FOOTERTEXT" not in content
    whole = rss.fed_page_body(page.replace(b'id="article"', b"").replace(b'id="content"', b""), STATEMENT,
                              8000).text
    assert "MENUTEXT" in whole and "FOOTERTEXT" in whole
    assert "NAVTEXT" not in whole and "never kept" not in whole and "Federal Reserve Board -" not in whole


def test_a_long_release_page_is_cut_at_its_configured_length():
    page = ('<html><body><div id="menu">menu</div><div id="article"><p>' + "statement word " * 1000 +
            "</p></div></body></html>").encode()
    body = rss.fed_page_body(page, STATEMENT, FED.monetary_body_max_chars)
    assert (len(body.text), body.truncated) == (FED.monetary_body_max_chars, True) == (8000, True)
    assert body.chars_total == len(("statement word " * 1000).strip()) and "menu" not in body.text


# BLS


def test_bls_atom_feed_gives_one_release_per_entry():
    items = rss.bls_items(fixture("news/bls_cpi.rss"), BLS.summary_max_chars)
    assert [(i.vendor_id, i.vendor_ts) for i in items] == [
        ("cpi-2026_09_11__07_50_40", "2026-09-11T11:50:40.968Z"),
        ("cpi-2026_08_12__07_51_16", "2026-08-12T11:51:16.210Z"),
        ("https://www.bls.gov/news.release/archives/cpi_02112026.htm", "2026-02-11T12:50:02.000Z"),
    ]
    assert all(i.source == "bls" and i.kind == "bls_release" and i.publisher == "Bureau of Labor Statistics"
               for i in items)
    assert items[0].title == "Fixture CPI release headline for August 2026"
    assert items[0].url == "https://www.bls.gov/news.release/archives/cpi_09112026.htm"
    assert items[0].summary == ("A fictional summary of the August 2026 release, written by hand for the parser "
                                "tests, with extra white space.")


def test_bls_summary_is_cut_at_its_configured_length():
    body = ('<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            '<title>t</title><link href="https://www.bls.gov/x.htm"/><id>x</id><content>' + "figure " * 400 +
            "</content><published>2026-09-11T08:30:00-04:00</published></entry></feed>").encode()
    [item] = rss.bls_items(body, BLS.summary_max_chars)
    assert len(item.summary) == BLS.summary_max_chars == 1500


def test_fed_and_bls_feed_urls():
    assert [rss.fed_url(f) for f in FED.feeds] == [f"https://www.federalreserve.gov/feeds/{f}.xml"
                                                   for f in ("press_all", "press_monetary", "speeches")]
    assert [rss.bls_url(f) for f in BLS.feeds] == [f"https://www.bls.gov/feed/{f}.rss"
                                                   for f in ("cpi", "empsit", "ppi", "jolts")]
    for bad in (lambda: rss.fed_url("press_other"), lambda: rss.bls_url("cpi.rss")):
        with pytest.raises(ValueError):
            bad()


# Shared rules


def test_feeds_refuse_a_document_type_and_anything_but_rss_or_atom():
    bomb = b'<?xml version="1.0"?><!DOCTYPE rss [<!ENTITY a "aaaa">]><rss version="2.0"><channel/></rss>'
    with pytest.raises(ValueError, match="document type"):
        rss.parse_feed(bomb)
    with pytest.raises(ValueError, match="well formed"):
        rss.parse_feed(b"<rss><channel>")
    with pytest.raises(ValueError, match="neither"):
        rss.parse_feed(b"<html><body>maintenance</body></html>")
    with pytest.raises(ValueError):
        rss.fed_items(b"", FED.monetary_link_pattern, 1000)


def test_vendor_times_are_utc_text_and_never_guessed():
    assert rss.vendor_time("Wed, 16 Sep 2026 18:00:00 GMT") == "2026-09-16T18:00:00.000Z"
    assert rss.vendor_time(" 2026-09-11T07:50:40.968-04:00 ") == "2026-09-11T11:50:40.968Z"
    assert rss.vendor_time("2026-10-08T20:30:12.000Z") == "2026-10-08T20:30:12.000Z"
    for unreadable in ("", "2026-10-09T17:30:15", "Wed, 16 Sep 2026 18:00:00 -0000", "soon", "Wed, 99 Foo"):
        assert rss.vendor_time(unreadable) is None


def test_every_fixture_item_has_a_known_kind_and_utc_vendor_time():
    items = all_fixture_items()
    assert len(items) == 4 + 5 + 3 + 4 + 4 + 2 + 2 + 3
    assert all(isinstance(i, RawItem) for i in items)
    assert {i.kind for i in items} <= set(NEWS_KINDS)
    assert {i.source for i in items} == {"finnhub", "edgar", "fed", "bls"}
    assert all(i.vendor_ts is None or UTC_TEXT.fullmatch(i.vendor_ts) for i in items)
    assert all(i.title and len(i.title) <= 300 for i in items)


def imported_modules(source):
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_the_sources_never_touch_the_network():
    network = {"socket", "ssl", "http", "urllib", "asyncio", "websockets", "anthropic", "requests", "httpx",
               "aiohttp", "subprocess", "local_secrets", "redact"}
    for module in SOURCES:
        names = imported_modules(open(module.__file__, encoding="utf-8").read())
        parts = {part for name in names for part in name.split(".")}
        assert not parts & network, f"{module.__name__} imports {sorted(parts & network)}"
        assert "btc_survivor.v2.news.http" not in names
