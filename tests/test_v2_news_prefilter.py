"""The prefilter's aliases, macro keywords, junk patterns and relevance tiers, and the route set at insert, on
fictional items written by hand."""

import dataclasses
import itertools

import pytest

from btc_survivor.v2.config import NEWS_KINDS, OFFICIAL_KINDS, PrefilterConfig, load_news_config
from btc_survivor.v2.news import prefilter
from btc_survivor.v2.news.prefilter import (ALIASES, BODY, EMPTY, FINBERT, FORM, JUNK, LLM, MACRO, MACRO_KEYWORDS,
                                            MACRO_TIER, NO_MATCH, NONE, OFFICIAL, TITLE, VENDOR, Candidate, Prefilter,
                                            Tags, route)
from helpers import ROOT

CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
PRE = Prefilter(CONFIG)


def general(title, summary="", publisher=""):
    return Candidate("finnhub", "media", title, summary, publisher, general=True)


def company(title, summary="", related=(), publisher=""):
    return Candidate("finnhub", "media", title, summary, publisher, related=tuple(related))


def test_the_alias_table_covers_the_universe():
    assert tuple(ALIASES) == CONFIG.news.universe
    assert "Elon Musk" not in {a for aliases in ALIASES.values() for a in aliases}


@pytest.mark.parametrize("symbol, alias", [(s, a) for s, aliases in ALIASES.items() for a in aliases],
                         ids=lambda v: str(v))
def test_every_alias_tags_its_symbol_alone(symbol, alias):
    assert PRE.symbols(f"Fictional note on {alias} today") == (symbol,)
    assert PRE.symbols(f"{alias}'s fictional update") == (symbol,)


@pytest.mark.parametrize("text", [
    "apple pie sales rise", "APPLE PIE SALES RISE", "a meta analysis of fictional trials", "nvidia", "nVidia",
    "the iphone of fictional kitchens", "tesla coil maker", "google it", "aws", "msft", "Tim cook",
    "satya nadella", "nasdaq 100", "Qqq",
], ids=range(14))
def test_aliases_are_case_sensitive(text):
    assert PRE.symbols(text) == ()


@pytest.mark.parametrize("text", [
    "Pineapple growers merge", "Metaverse land sales", "Applebee's adds menu", "Teslas on the road",
    "AAPLX fund launch", "Amazonia rainforest report", "GOOGLE", "Microsoftish design", "QQQM fund flows",
], ids=range(9))
def test_aliases_need_word_boundaries(text):
    assert PRE.symbols(text) == ()


def test_alias_spaces_match_any_white_space():
    assert PRE.symbols("Tim  Cook speaks") == ("AAPL",)
    assert PRE.symbols("Tim Cook speaks") == ("AAPL",)
    assert PRE.symbols("Satya\nNadella speaks") == ("MSFT",)


def test_symbols_come_in_universe_order():
    assert PRE.symbols("Tesla, Meta, Apple and the Nasdaq-100") == ("AAPL", "META", "TSLA", "QQQ")


def test_musk_alone_tags_nothing():
    assert PRE.symbols("Elon Musk posts about a fictional rocket") == ()
    assert PRE.tag(general("Elon Musk posts about a fictional rocket")) == Tags(NONE, (), NO_MATCH)
    assert PRE.tag(company("Musk unveils fictional tunnel", related=["TSLA"])) == Tags(VENDOR, ("TSLA",))
    assert PRE.tag(general("Musk says Tesla will build a fictional car")) == Tags(TITLE, ("TSLA",))


@pytest.mark.parametrize("keyword", MACRO_KEYWORDS)
def test_every_macro_keyword_ignores_case(keyword):
    for variant in (keyword, keyword.lower(), keyword.upper()):
        assert PRE.macro(f"Fictional note on {variant} this week"), variant
        assert PRE.tag(general(f"Fictional note on {variant} this week")) == Tags(MACRO_TIER, (MACRO,))


@pytest.mark.parametrize("text", ["New tariffs hit fictional imports", "Two rate cuts priced in",
                                  "Interest rates hold", "Feds weigh fictional plan", "Semiconductors slide"],
                         ids=range(5))
def test_a_keyword_also_matches_its_plural(text):
    assert PRE.macro(text)


@pytest.mark.parametrize("text", ["FedEx delivers fictional parcels", "A federal judge rules on a fictional case",
                                  "Inflationary fiction", "Tariffing is not a word", "GDPR fine for fictional app"],
                         ids=range(5))
def test_a_keyword_never_matches_a_longer_word(text):
    assert not PRE.macro(text)


@pytest.mark.parametrize("notation", ["(NASDAQ: AAPL)", "(NASDAQ:CPI)", "(NYSE: GDP)", "(NYSEARCA: PCE)",
                                      "(NYSE : PPI.B)"], ids=range(5))
def test_exchange_notation_never_tags_macro(notation):
    assert not PRE.macro(f"Fictional Labs {notation} names a new chief")
    tags = PRE.tag(general(f"Fictional Labs {notation} names a new chief"))
    assert MACRO not in tags.tickers


def test_nasdaq_aapl_tags_apple_and_never_macro():
    tags = PRE.tag(general("Fictional Labs (NASDAQ: AAPL) wins supplier award as inflation cools"))
    assert tags == Tags(TITLE, ("AAPL",))


def test_an_aliased_item_never_gets_macro():
    assert PRE.tag(general("Apple slides as tariffs and inflation bite")) == Tags(TITLE, ("AAPL",))
    assert PRE.tag(general("Inflation report lifts the stock market", "Microsoft and Nvidia lead")) == \
        Tags(BODY, ("MSFT", "NVDA"))
    assert PRE.tag(general("Fed holds rates", "Analysts expect Invesco QQQ flows")) == Tags(BODY, ("QQQ",))


def test_macro_joins_aliases_only_when_the_setting_allows_it():
    loose = Prefilter(dataclasses.replace(CONFIG, prefilter=PrefilterConfig(
        CONFIG.prefilter.macro_strip_pattern, alias_excludes_macro=False)))
    assert loose.tag(general("Apple slides as tariffs bite")) == Tags(TITLE, ("AAPL", MACRO))
    assert loose.tag(company("Apple slides as tariffs bite", related=["AAPL"])) == Tags(TITLE, ("AAPL",))


def test_the_title_tier_carries_summary_aliases_too():
    tags = PRE.tag(company("Apple widens fictional deal", "Microsoft and Amazon also sign", related=["AAPL"]))
    assert tags == Tags(TITLE, ("AAPL", "MSFT", "AMZN")) and tags.tickers_text == "AAPL,MSFT,AMZN"


def test_the_body_tier():
    assert PRE.tag(company("Fictional supplier wins big order", "The deal with Nvidia runs to 2028",
                           related=["NVDA"])) == Tags(BODY, ("NVDA",))


def test_the_vendor_tier_needs_a_universe_symbol():
    assert PRE.tag(company("Fictional supplier wins contract", related=["NVDA", "XYZ"])) == Tags(VENDOR, ("NVDA",))
    assert PRE.tag(company("Fictional supplier wins contract", related=["XYZ"])) == Tags(NONE, (), NO_MATCH)
    assert PRE.tag(company("Fictional supplier wins contract")) == Tags(NONE, (), NO_MATCH)


def test_the_macro_tier_is_for_general_items_only():
    assert PRE.tag(general("Inflation cools in fictional survey")) == Tags(MACRO_TIER, (MACRO,))
    assert PRE.tag(general("Fictional survey out", "Analysts expect a rate cut")) == Tags(MACRO_TIER, (MACRO,))
    assert PRE.tag(company("Inflation cools in fictional survey", related=["AMZN"])) == Tags(VENDOR, ("AMZN",))
    assert PRE.tag(general("Fictional bakery opens downtown")) == Tags(NONE, (), NO_MATCH)


@pytest.mark.parametrize("source, kind", [("fed", "fed_press"), ("fed", "fed_monetary"), ("fed", "fed_speech"),
                                          ("bls", "bls_release")])
def test_fed_and_bls_items_are_official_and_macro(source, kind):
    assert PRE.tag(Candidate(source, kind, "Fictional release on fictional figures")) == Tags(OFFICIAL, (MACRO,))
    assert PRE.tag(Candidate(source, kind, "Top 10 stocks to buy, says a fictional speaker")) == \
        Tags(OFFICIAL, (MACRO,))
    assert PRE.tag(Candidate(source, kind, "  ")) == Tags(NONE, (), EMPTY)


def test_edgar_items_carry_their_filer():
    assert PRE.tag(Candidate("edgar", "8k", "8-K fictional filing", filer="AAPL", form="8-K")) == \
        Tags(OFFICIAL, ("AAPL",))
    assert PRE.tag(Candidate("edgar", "8k_a", "8-K/A fictional filing", filer="TSLA", form="8-K/A")) == \
        Tags(OFFICIAL, ("TSLA",))
    assert PRE.tag(Candidate("edgar", "8k", "10-Q fictional filing", filer="AAPL", form="10-Q")) == \
        Tags(NONE, (), FORM)
    assert PRE.tag(Candidate("edgar", "8k", "8-K fictional filing", filer="XYZ", form="8-K")) == \
        Tags(NONE, (), NO_MATCH)
    assert PRE.tag(Candidate("edgar", "8k", "", filer="AAPL", form="8-K")) == Tags(NONE, (), EMPTY)


@pytest.mark.parametrize("title", ["", "   ", " \n"], ids=range(3))
def test_an_empty_headline_is_filtered(title):
    tags = PRE.tag(general(title, "Apple and inflation in the summary"))
    assert tags == Tags(NONE, (), EMPTY) and tags.filtered and tags.tickers_text == ""


@pytest.mark.parametrize("title", [
    "Top 10 Stocks for a fictional decade", "best 3 stock picks", "2 Stocks to Buy in a Fictional Dip",
    "The stock to watch this fictional week", "Should You Buy Apple Before Its Fictional Event?",
    "This Nvidia bet could make you a millionaire", "Buy and Hold Forever: fictional picks",
    "What the Motley Fool says about Tesla",
], ids=range(8))
def test_every_junk_pattern(title):
    tags = PRE.tag(general(title))
    assert tags == Tags(NONE, (), JUNK) and tags.filtered


def test_junk_in_the_summary_or_publisher():
    assert PRE.tag(company("Apple widget sales rise", "One of the best 5 stocks", ["AAPL"])) == Tags(NONE, (), JUNK)
    assert PRE.tag(company("Apple widget sales rise", publisher="The Motley Fool")) == Tags(NONE, (), JUNK)


@pytest.mark.parametrize("title", ["Top stocks rally on a fictional day", "Apple buyback grows",
                                   "Ten stocks watched closely", "Fictional fool's errand"], ids=range(4))
def test_near_misses_are_not_junk(title):
    assert not PRE.junk(title)


def test_a_universe_symbol_without_aliases_is_refused():
    news = dataclasses.replace(CONFIG.news, universe=("AAPL", "XYZ"))
    with pytest.raises(ValueError, match="XYZ"):
        Prefilter(dataclasses.replace(CONFIG, news=news))


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="source"):
        PRE.tag(Candidate("wire", "media", "Fictional headline"))


# Routing.

STATUSES = ("new", "seed", "backlog", "filtered")
TIERS = (TITLE, BODY, VENDOR, MACRO_TIER, OFFICIAL, NONE)
SOURCE_KINDS = [("finnhub", "media"), ("edgar", "8k"), ("edgar", "8k_a"), ("fed", "fed_press"),
                ("fed", "fed_monetary"), ("fed", "fed_speech"), ("bls", "bls_release")]


def test_source_kinds_cover_every_kind():
    assert {kind for _, kind in SOURCE_KINDS} == set(NEWS_KINDS)


@pytest.mark.parametrize("relevance", [TITLE, BODY, MACRO_TIER])
def test_finnhub_items_route_to_llm_or_to_finbert_only_while_send_to_llm_is_false(relevance):
    assert route("finnhub", "media", "new", relevance, send_to_llm=True) == LLM
    assert route("finnhub", "media", "new", relevance, send_to_llm=False) == FINBERT


@pytest.mark.parametrize("kind", OFFICIAL_KINDS)
def test_official_kinds_route_to_llm_whatever_send_to_llm_says(kind):
    source = {"8k": "edgar", "bls_release": "bls"}.get(kind, "fed")
    for send in (True, False):
        assert route(source, kind, "new", OFFICIAL, send) == LLM


def test_8k_a_vendor_and_filtered_items_route_nowhere():
    for send in (True, False):
        assert route("edgar", "8k_a", "new", OFFICIAL, send) == NONE
        assert route("finnhub", "media", "new", VENDOR, send) == NONE
        assert route("finnhub", "media", "filtered", NONE, send) == NONE
        assert route("edgar", "8k", "filtered", NONE, send) == NONE


@pytest.mark.parametrize("source, kind", SOURCE_KINDS)
def test_the_route_for_every_source_kind_status_and_tier(source, kind):
    for status, relevance, send in itertools.product(STATUSES, TIERS, (True, False)):
        got = route(source, kind, status, relevance, send)
        if status != "new":
            expected = NONE
        elif source == "finnhub":
            expected = (LLM if send else FINBERT) if relevance in (TITLE, BODY, MACRO_TIER) else NONE
        else:
            expected = LLM if kind != "8k_a" and relevance == OFFICIAL else NONE
        assert got == expected, (status, relevance, send)
        assert got != FINBERT or (source == "finnhub" and not send)


def test_tag_then_route_on_fictional_items():
    send = CONFIG.finnhub.send_to_llm
    items = [
        (general("Apple unveils a fictional widget"), LLM),
        (general("Inflation cools in a fictional survey"), LLM),
        (company("Fictional supplier wins contract", related=["NVDA"]), NONE),
        (general("Top 10 stocks for a fictional decade"), NONE),
        (Candidate("edgar", "8k", "8-K fictional filing", filer="META", form="8-K"), LLM),
        (Candidate("edgar", "8k_a", "8-K/A fictional filing", filer="META", form="8-K/A"), NONE),
        (Candidate("fed", "fed_monetary", "Fictional FOMC statement"), LLM),
    ]
    assert send is True
    for item, expected in items:
        tags = PRE.tag(item)
        assert route(item.source, item.kind, "filtered" if tags.filtered else "new", tags.relevance, send) == expected
        if expected == LLM and item.source == "finnhub":
            assert route(item.source, item.kind, "new", tags.relevance, False) == FINBERT
    assert prefilter.SCORED_TIERS == (TITLE, BODY, MACRO_TIER)
