"""The model request of PLAN_V2 sections 8.1 to 8.3 and the probe file of section 8.8, on fictional items and the
committed news config. Nothing here reaches the network. The SDK checks run the pinned anthropic package against an
in process transport that answers with a canned body, and every secret is a fake value."""

import codecs
import dataclasses
import hashlib
import inspect
import json
import math
import re
import typing

import pytest

from btc_survivor.v2 import redact
from btc_survivor.v2.config import HAIKU_MODEL, OFFICIAL_KINDS, SONNET_MODEL, load_news_config
from btc_survivor.v2.local_secrets import ANTHROPIC, CONTACT, FINNHUB, load_secrets
from btc_survivor.v2.news import llm_request, probes
from btc_survivor.v2.news.llm_request import (
    HERE, INSTRUCTION, PROMPT_FILE, PROMPT_SHA256, SCHEMA_FILE, SCHEMA_SHA256, RequestError, RequestItem,
    RequestTooLarge, build_request, canonical, check_pins, check_size, clean, haiku_spec, input_tokens, load_prompt,
    load_schema, request_item, sonnet_spec,
)
from btc_survivor.v2.news.llm_validate import EVENT_TYPES, TICKERS
from helpers import ROOT

LLM = load_news_config(ROOT / "configs" / "v2" / "news.toml").llm
HAIKU = haiku_spec(LLM)
SONNET = sonnet_spec(LLM)
SENT_KEYS = {"model", "max_tokens", "thinking", "output_config", "system", "messages"}
NEVER_SENT = {"temperature", "top_p", "top_k", "tools", "tool_choice", "metadata", "stop_sequences", "fallbacks",
              "betas", "service_tier", "inference_geo", "container", "stream", "speed", "mcp_servers",
              "context_management", "user_profile_id", "workspace_id"}
SENTINEL = "probe.sentinel.contact@example.invalid"
FAKE_SECRETS = {FINNHUB: "fakefinnhub0001token", ANTHROPIC: "fake-anthropic-0002-value", CONTACT: SENTINEL}
DATE_PATTERNS = (
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|"
               r"Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|"
               r"Sunday)\b"),
)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def row(**changes):
    """A full news_items row of a fictional Finnhub item, every column filled, as the scorer reads it."""
    base = {"seq": 7, "source": "finnhub", "channel": "finnhub_company_AAPL", "vendor_id": "1234567", "kind": "media",
            "received_at": "2026-10-13T14:30:01.123Z", "detected_at": None, "vendor_ts": "2026-10-13T14:29:30.000Z",
            "title": "Fictional Widget Corp raises its outlook", "summary": "Fictional Widget Corp said demand rose.",
            "url": "https://example.invalid/story", "publisher": "Example Wire", "tickers": "AAPL",
            "relevance": "title", "status": "new", "filter_reason": None, "title_hash": "0123456789abcdef",
            "items_8k": None, "route": "llm"}
    base.update(changes)
    return base


def official_row(kind, **changes):
    """A fictional official item of one kind, with the source and tickers the collector gives it."""
    source = {"8k": "edgar", "fed_press": "fed", "fed_monetary": "fed", "fed_speech": "fed", "bls_release": "bls"}[kind]
    fields = {"source": source, "kind": kind, "tickers": "MSFT" if source == "edgar" else "MACRO",
              "relevance": "official", "channel": f"{source}_channel", "title": f"Fictional {kind} release",
              "summary": f"Fictional summary of a {kind} release."}
    return row(**{**fields, **changes})


def items(*rows, body=None):
    return [request_item(r, body, LLM) for r in rows]


def build(spec, request_items=None):
    return build_request(spec, request_items or items(row()))


def all_keys(value):
    """Every mapping key anywhere inside a JSON value."""
    if isinstance(value, dict):
        return set(value) | set().union(*(all_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(all_keys(v) for v in value))
    return set()


def count_key(value, key):
    """How many mappings inside a JSON value hold key."""
    if isinstance(value, dict):
        return (key in value) + sum(count_key(v, key) for v in value.values())
    if isinstance(value, list):
        return sum(count_key(v, key) for v in value)
    return 0


def lines(request):
    return request.params()["messages"][0]["content"].split("\n")


# Identical bytes


def test_identical_items_give_identical_bytes():
    first = build(HAIKU, items(row(), row(title="Second fictional headline")))
    llm_request._file.cache_clear()
    again = build(HAIKU, items(row(seq=99, channel="finnhub_general", vendor_id="99", url="https://example.invalid/b",
                                   publisher="Other Wire", received_at="2026-10-14T09:00:00.000Z",
                                   vendor_ts="2026-10-14T08:59:00.000Z", relevance="body"),
                               row(title="Second fictional headline", seq=100)))
    assert again.body == first.body and again.sha256 == first.sha256
    assert again.ids == first.ids == ("i01", "i02")
    changed = build(HAIKU, items(row(), row(title="A different fictional headline")))
    assert changed.body != first.body and changed.sha256 != first.sha256


def test_params_are_a_fresh_copy_of_the_bytes():
    request = build(HAIKU)
    params = request.params()
    params["model"] = "changed"
    params["output_config"]["format"]["schema"]["type"] = "changed"
    assert request.params()["model"] == HAIKU_MODEL
    assert request.params()["output_config"]["format"]["schema"] == load_schema()
    assert json.loads(request.body) == request.params()


# Pinned prompt and schema


def test_the_prompt_and_schema_files_hash_to_their_pins():
    prompt = (HERE / PROMPT_FILE).read_bytes()
    assert hashlib.sha256(prompt).hexdigest() == PROMPT_SHA256 == llm_request.prompt_sha256()
    schema = json.loads((HERE / SCHEMA_FILE).read_bytes())
    canonical_bytes = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical_bytes).hexdigest() == SCHEMA_SHA256 == llm_request.schema_sha256()


def test_the_prompt_is_utf8_with_lf_endings_and_sent_unchanged():
    prompt = (HERE / PROMPT_FILE).read_bytes()
    assert b"\r" not in prompt and not prompt.startswith(codecs.BOM_UTF8) and prompt.endswith(b"\n")
    assert prompt.decode("utf-8") == load_prompt()
    assert "src/btc_survivor/v2/news/prompt_v1.txt text eol=lf" in (ROOT / ".gitattributes").read_text().splitlines()
    for spec in (HAIKU, SONNET):
        assert build(spec).params()["system"][0]["text"].encode("utf-8") == prompt


def test_the_schema_hash_ignores_layout_and_key_order_but_not_content():
    schema = load_schema()
    reordered = json.loads(json.dumps(schema, sort_keys=True, indent=7))
    assert canonical(reordered) == canonical(schema)
    assert hashlib.sha256(canonical(schema)).hexdigest() == SCHEMA_SHA256
    schema["properties"]["items"]["items"]["properties"]["id"]["type"] = "integer"
    assert hashlib.sha256(canonical(schema)).hexdigest() != SCHEMA_SHA256


def test_check_pins_needs_both_pins_to_match_the_files():
    pinned = dataclasses.replace(LLM, prompt_sha256=PROMPT_SHA256, schema_sha256=SCHEMA_SHA256)
    check_pins(pinned)
    for bad, match in ((dataclasses.replace(pinned, prompt_sha256=""), "not pinned"),
                       (dataclasses.replace(pinned, schema_sha256=""), "not pinned"),
                       (dataclasses.replace(pinned, prompt_sha256="0" * 64), "disagree"),
                       (dataclasses.replace(pinned, schema_sha256="f" * 64), "disagree")):
        with pytest.raises(RequestError, match=match):
            check_pins(bad)


def test_the_committed_news_config_pins_the_files():
    check_pins(LLM)
    assert (LLM.prompt_sha256, LLM.schema_sha256) == (PROMPT_SHA256, SCHEMA_SHA256)


# What a request holds


@pytest.mark.parametrize("spec", [HAIKU, SONNET], ids=["haiku", "sonnet"])
def test_no_sampling_parameters_tools_metadata_or_fallbacks(spec):
    params = build(spec).params()
    assert set(params) == SENT_KEYS
    assert not all_keys(params) & NEVER_SENT
    assert set(build(spec).count_params()) == SENT_KEYS - {"max_tokens"}


@pytest.mark.parametrize("spec", [HAIKU, SONNET], ids=["haiku", "sonnet"])
def test_thinking_disabled_and_effort_low(spec):
    params = build(spec).params()
    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": load_schema()}}


def test_each_scorer_sends_its_exact_model_id():
    assert HAIKU.model == "claude-haiku-5-5" and SONNET.model == "claude-sonnet-5-5"
    assert build(HAIKU).params()["model"] == HAIKU_MODEL
    assert build(SONNET).params()["model"] == SONNET_MODEL
    assert HAIKU.name == "haiku_v1" and SONNET.name == "sonnet_v1"


@pytest.mark.parametrize("change", [{"model": "model-x"}, {"effort": "medium"}, {"thinking": "adaptive"},
                                   {"max_tokens": 0}], ids=["model", "effort", "thinking", "max_tokens"])
def test_a_setting_outside_the_plan_is_refused(change):
    with pytest.raises(RequestError):
        build(dataclasses.replace(HAIKU, **change))


@pytest.mark.parametrize("kind", OFFICIAL_KINDS)
def test_haiku_and_sonnet_send_the_same_prompt_schema_and_user_message_for_an_official_item(kind):
    body = "Fictional filing text. " * 40 if kind in LLM.long_form_kinds else None
    official = items(official_row(kind), body=body)
    haiku, sonnet = build(HAIKU, official).params(), build(SONNET, official).params()
    assert haiku["system"][0]["text"] == sonnet["system"][0]["text"] == load_prompt()
    assert haiku["output_config"] == sonnet["output_config"]
    assert haiku["messages"] == sonnet["messages"]
    del haiku["system"][0]["cache_control"]
    assert {**haiku, "model": SONNET_MODEL, "max_tokens": SONNET.max_tokens} == sonnet


def test_max_tokens_is_1024_for_sonnet_and_4096_for_haiku():
    assert build(SONNET).params()["max_tokens"] == 1024
    assert build(HAIKU).params()["max_tokens"] == 4096


def test_no_btc_in_the_ticker_enum_prompt_or_probes():
    schema = load_schema()
    impact = schema["properties"]["items"]["items"]["properties"]["impacts"]["items"]
    assert impact["properties"]["ticker"]["enum"] == list(TICKERS)
    assert len(TICKERS) == 9 and "MACRO" in TICKERS and "QQQ" in TICKERS
    for text in ((HERE / SCHEMA_FILE).read_text(), load_prompt(), (HERE / probes.PROBE_FILE).read_text()):
        assert not re.search(r"btc|bitcoin|xbt", text, re.IGNORECASE)


def test_cache_control_on_haikus_single_system_block_and_none_for_sonnet():
    haiku, sonnet = build(HAIKU).params(), build(SONNET).params()
    assert haiku["system"] == [{"type": "text", "text": load_prompt(), "cache_control": {"type": "ephemeral"}}]
    assert sonnet["system"] == [{"type": "text", "text": load_prompt()}]
    assert count_key(haiku, "cache_control") == 1 and count_key(sonnet, "cache_control") == 0
    assert SONNET.cache_control is False and LLM.sonnet.cache_control is False


def test_no_dates_reach_a_request():
    request = build(HAIKU, items(row(), row(received_at="2027-01-04T15:00:00.000Z", detected_at="2027-01-04T14:59:00Z",
                                            vendor_ts="2027-01-04T14:58:00.000Z", seq=8)))
    for text in (request.body.decode("utf-8"), (HERE / probes.PROBE_FILE).read_text()):
        for pattern in DATE_PATTERNS:
            assert not pattern.search(text), pattern.pattern
    assert "2027" not in request.body.decode("utf-8")
    # The only year anywhere is the knowledge limit the plan writes into the ground rules.
    assert YEAR.findall(request.body.decode("utf-8")) == ["2026"]
    assert "Your knowledge ends around mid 2026." in load_prompt()
    for text in ((HERE / SCHEMA_FILE).read_text(), (HERE / probes.PROBE_FILE).read_text()):
        assert not YEAR.search(text)


def test_a_sentinel_contact_address_never_reaches_a_request(tmp_path, monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    secrets_file = tmp_path / "secrets.env"
    secrets_file.write_text("".join(f"{name}={value}\n" for name, value in FAKE_SECRETS.items()), encoding="utf-8")
    secrets_file.chmod(0o600)
    assert load_secrets(secrets_file).get(CONTACT) == SENTINEL
    monkeypatch.setenv(CONTACT, SENTINEL)
    planted = {"url": f"https://example.invalid/{SENTINEL}", "publisher": SENTINEL, "vendor_id": SENTINEL,
               "channel": SENTINEL, "filter_reason": SENTINEL, "items_8k": SENTINEL}
    requests = [build(HAIKU, items(row(**planted))), build(SONNET, items(official_row("8k", **planted),
                                                                         body="Fictional filing text.")),
                probes.probe_request(LLM, probes.load_probes())]
    captured = []
    for request in requests:
        captured += send_through_the_sdk(request)
        for data in (request.body, canonical(request.count_params())):
            assert SENTINEL.encode("utf-8") not in data.lower()
    for data in captured:
        assert SENTINEL.encode("utf-8") not in data.lower()


# The size guard


def test_the_size_guard_on_the_exact_count():
    request = build(HAIKU)
    assert check_size(request, 20000, counted=20000) == (20000, "count_tokens")
    with pytest.raises(RequestTooLarge, match="20001 input tokens by count_tokens"):
        check_size(request, 20000, counted=20001)
    by_bytes, _ = input_tokens(request, None)
    assert check_size(request, by_bytes - 1, counted=by_bytes - 1) == (by_bytes - 1, "count_tokens")


def test_the_size_guard_on_bytes_over_3_when_no_count_exists():
    filing = items(official_row("8k"), body="Fictional filing word " * 2000)
    request = build(HAIKU, filing)
    tokens, basis = check_size(request, LLM.max_input_tokens)
    assert basis == "bytes" and tokens == math.ceil(len(request.body) / 3)
    assert check_size(request, tokens) == (tokens, "bytes")
    with pytest.raises(RequestTooLarge, match="by bytes"):
        check_size(request, tokens - 1)


def test_the_largest_planned_requests_pass_the_guard_by_bytes():
    full = items(*[row(title="h" * 400, summary="s" * 1000, seq=i) for i in range(LLM.max_items_per_call)])
    filing = items(official_row("8k"), body="f" * 30000)
    for request in (build(HAIKU, full), build(HAIKU, filing), build(SONNET, filing)):
        check_size(request, LLM.max_input_tokens)


# The user message


def test_the_user_message_is_an_instruction_then_one_sorted_compact_line_per_item():
    request = build(HAIKU, items(row(), row(title="Second fictional headline"), official_row("8k")))
    first, *rest = lines(request)
    assert first == INSTRUCTION
    assert [json.loads(line)["id"] for line in rest] == ["i01", "i02", "i03"] == list(request.ids)
    for line in rest:
        assert line == canonical(json.loads(line)).decode("utf-8")
        assert list(json.loads(line)) == ["filer", "headline", "id", "source", "text"]


def test_item_text_is_nfc_plain_collapsed_and_cut():
    item = request_item(row(title="Café <b>owner</b>  &amp;\n\tpartner", summary="x" * 1000), None, LLM)
    assert item.headline == "Café owner & partner"
    assert item.text == "x" * LLM.text_max_chars
    assert request_item(row(title="h" * 400), None, LLM).headline == "h" * LLM.headline_max_chars
    assert clean(None, 10) == "" and clean("  a  b  ", 10) == "a b" and clean("abcdef  gh", 7) == "abcdef"


def test_long_form_kinds_keep_24000_characters_and_others_700():
    long_body = "y" * 30000
    for kind in OFFICIAL_KINDS:
        item = request_item(official_row(kind), long_body, LLM)
        expected = LLM.long_form_max_chars if kind in LLM.long_form_kinds else LLM.text_max_chars
        assert len(item.text) == expected, kind
    assert set(LLM.long_form_kinds) == {"8k", "fed_monetary"} and LLM.long_form_max_chars == 24000


def test_the_body_is_preferred_and_the_filer_is_only_for_edgar():
    filing = request_item(official_row("8k", summary="ignored summary"), "Filing body text.", LLM)
    assert (filing.source, filing.filer, filing.text) == ("edgar", "MSFT", "Filing body text.")
    assert request_item(official_row("8k", summary=None), "", LLM).text == ""
    assert request_item(row(tickers="AAPL,MSFT"), None, LLM).filer == ""
    assert request_item(official_row("fed_press"), None, LLM).filer == ""


def test_a_batch_holds_1_to_25_items():
    with pytest.raises(RequestError):
        build_request(HAIKU, [])
    with pytest.raises(RequestError):
        build_request(HAIKU, items(*[row(seq=i) for i in range(26)]))
    assert build_request(HAIKU, items(*[row(seq=i) for i in range(25)])).ids[-1] == "i25"


# The pinned SDK


def send_through_the_sdk(request):
    """Send a request's create and count_tokens arguments through the pinned SDK into an in process transport, and
    return the raw bytes of each HTTP request it built, headers included."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen = []

    def answer(http_request):
        seen.append(http_request)
        if http_request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1})
        return httpx2.Response(200, json={
            "id": "msg_fake", "type": "message", "role": "assistant", "model": request.spec.model,
            "content": [{"type": "text", "text": "{\"items\":[]}"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = anthropic.Anthropic(api_key=FAKE_SECRETS[ANTHROPIC], base_url="https://api.anthropic.com", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(answer)))
    client.messages.create(**request.params())
    client.messages.count_tokens(**request.count_params())
    assert [r.url.path for r in seen] == ["/v1/messages", "/v1/messages/count_tokens"]
    assert json.loads(seen[0].content) == request.params()
    assert json.loads(seen[1].content) == request.count_params()
    assert not any("anthropic-beta" in r.headers for r in seen)
    return [r.content + "".join(f"{k}: {v}\n" for k, v in r.headers.items()).encode("utf-8") for r in seen]


@pytest.mark.parametrize("spec", [HAIKU, SONNET], ids=["haiku", "sonnet"])
def test_the_pinned_sdk_sends_exactly_these_parameters(spec, monkeypatch):
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_LOG", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    send_through_the_sdk(build(spec))


def test_the_request_fits_the_types_of_the_pinned_sdk():
    anthropic = pytest.importorskip("anthropic")
    from anthropic.resources.messages import AsyncMessages
    from anthropic.types import (CacheControlEphemeralParam, JSONOutputFormatParam, OutputConfigParam,
                                 TextBlockParam, ThinkingConfigDisabledParam)
    assert anthropic.__version__ == "1.8.0"
    create = inspect.signature(AsyncMessages.create).parameters
    count = inspect.signature(AsyncMessages.count_tokens).parameters
    assert not {"temperature", "top_p", "top_k", "fallbacks", "betas"} & set(create)
    effort = next(a for a in typing.get_args(typing.get_type_hints(OutputConfigParam)["effort"]) if a is not type(None))
    for spec in (HAIKU, SONNET):
        params = build(spec).params()
        assert set(params) <= set(create) and set(build(spec).count_params()) <= set(count)
        hints = typing.get_type_hints(ThinkingConfigDisabledParam)
        assert typing.get_args(hints["type"]) == (params["thinking"]["type"],)
        assert set(params["output_config"]) == set(OutputConfigParam.__annotations__)
        assert params["output_config"]["effort"] in typing.get_args(effort)
        assert set(params["output_config"]["format"]) == set(JSONOutputFormatParam.__annotations__)
        for block in params["system"]:
            assert set(block) <= set(TextBlockParam.__annotations__)
            assert set(block.get("cache_control", {})) <= set(CacheControlEphemeralParam.__annotations__)


# The probe file


PROBES = probes.load_probes()


def test_the_probe_file_holds_20_fictional_items_with_a_key():
    assert [p.probe_id for p in PROBES] == [f"p{i:02d}" for i in range(1, 21)]
    keyed = {k.ticker for p in PROBES for k in p.key}
    assert keyed == set(TICKERS)
    assert {k.event_type for p in PROBES for k in p.key} == set(EVENT_TYPES)
    assert any(not p.key for p in PROBES) and any(len(p.key) > 1 for p in PROBES)
    assert {p.kind for p in PROBES} == {"media", "8k", "fed_monetary", "fed_speech", "bls_release"}
    assert all(p.why for p in PROBES)


def test_every_probe_item_is_clean_and_within_its_limits():
    for p in PROBES:
        limit = LLM.long_form_max_chars if p.kind in LLM.long_form_kinds else LLM.text_max_chars
        assert clean(p.item.headline, LLM.headline_max_chars) == p.item.headline, p.probe_id
        assert clean(p.item.text, limit) == p.item.text, p.probe_id
        assert (p.item.filer != "") == (p.item.source == "edgar"), p.probe_id


def test_the_probe_request_is_one_haiku_call_of_20_items():
    request = probes.probe_request(LLM, PROBES)
    params = request.params()
    assert params["model"] == HAIKU_MODEL and params["max_tokens"] == LLM.max_tokens
    assert request.ids == tuple(f"i{i:02d}" for i in range(1, 21))
    assert [json.loads(line)["headline"] for line in lines(request)[1:]] == [p.item.headline for p in PROBES]
    check_size(request, LLM.max_input_tokens)


def test_a_broken_probe_file_is_refused(tmp_path, monkeypatch):
    data = json.loads((HERE / probes.PROBE_FILE).read_text())
    monkeypatch.setattr(llm_request, "HERE", tmp_path)
    for broken in (lambda d: d["probes"].pop(),
                   lambda d: d["probes"][0]["key"].append({"ticker": "BTC", "event_type": "other", "is_new": True,
                                                           "direction": 0}),
                   lambda d: d["probes"][0]["key"][0].update(direction=True),
                   lambda d: d["probes"][15]["key"].append(dict(d["probes"][15]["key"][0])),
                   lambda d: d["probes"][1].update(kind="tweet"),
                   lambda d: d["probes"][2].update(probe_id="p99")):
        copy = json.loads(json.dumps(data))
        broken(copy)
        (tmp_path / "broken.json").write_text(json.dumps(copy))
        with pytest.raises(ValueError):
            probes.load_probes("broken.json")


@pytest.mark.parametrize("now, last, expected", [
    ("2026-10-13T05:59:59.999Z", None, False),
    ("2026-10-13T06:00:00.000Z", None, True),
    ("2026-10-13T06:00:00.000Z", "2026-10-12T06:00:01.000Z", True),
    ("2026-10-13T06:00:05.000Z", "2026-10-13T06:00:00.000Z", False),
    ("2026-10-13T23:30:00.000Z", "2026-10-12T06:00:00.000Z", True),
    ("2026-10-14T00:10:00.000Z", "2026-10-13T06:00:00.000Z", False),
])
def test_the_probe_is_due_once_a_day_after_0600_utc(now, last, expected):
    from btc_survivor.clock import parse_iso
    assert probes.due(parse_iso(now), last, LLM.probe_time_utc) is expected
