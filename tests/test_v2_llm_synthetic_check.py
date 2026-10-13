"""scripts/llm_synthetic_check.py of PLAN_V2 sections 16 and 17, only ever on fake clients and temporary configs and
ledgers, so no test reaches the Anthropic API, reads a secret or touches data/v2.

The 40 fictional items and their answer key stay inside the schema, the dry run counts every request and sends none,
the live check passes on good answers and fails on a bad first call, a cold cache, a key match under 90 percent or a
disabled Sonnet's absence handled, every call is booked in the key ledger, and the default output holds only verdicts,
counts, hashes and words. The cap drill writes the config the plan describes, refuses a folder left by an earlier
drill, passes against the real scorers running as its news process, and books its spend into the key ledger even when
it fails."""

import asyncio
import dataclasses
import importlib.util
import json
import re
import sqlite3
import sys
import threading
import tomllib

import pytest

from btc_survivor.clock import LiveClock
from btc_survivor.v2.config import HAIKU_MODEL, OFFICIAL_KINDS, SONNET_MODEL, load_news_config
from btc_survivor.v2.news import db, llm_request, probes
from btc_survivor.v2.news.collector import wait_or_stop
from btc_survivor.v2.news.finbert import Probs
from btc_survivor.v2.news.finbert_scorer import FinbertScorer
from btc_survivor.v2.news.ledger import Ledger, Usage
from btc_survivor.v2.news.llm_client import BAD_REQUEST, OK, RATE_LIMITED, SPEND_LIMIT, CallResult
from btc_survivor.v2.news.llm_validate import DIRECTIONS, TICKERS
from btc_survivor.v2.news.runner import record_start
from btc_survivor.v2.news.scoring import Scoring
from helpers import ROOT


def _load_script():
    spec = importlib.util.spec_from_file_location("llm_synthetic_check", ROOT / "scripts" / "llm_synthetic_check.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load_script()
COMMITTED = ROOT / "configs" / "v2" / "news.toml"
DATE = re.compile(r"\b(19|20)\d\d-\d\d-\d\d\b|\b(January|February|March|April|May|June|July|August|September|"
                  r"October|November|December|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b")
SUMMARY_VALUE = re.compile(r"pass|fail|\d+|[0-9a-f]{64}|[a-z][a-z_ ]*")


def write_config(folder, sonnet=True, thinking=None):
    """The committed news config with the files pinned and every path inside folder, written with the script's own
    TOML writer. thinking, when given, is Sonnet's thinking setting and thinking room."""
    raw = tomllib.loads(COMMITTED.read_text(encoding="utf-8"))
    raw["news"].update(data_dir=str(folder / "data" / "news"), secrets_file=str(folder / "secrets.env"),
                       rest_lock_file=str(folder / "finnhub_rest.lock"))
    raw["llm"].update(prompt_sha256=llm_request.PROMPT_SHA256, schema_sha256=llm_request.SCHEMA_SHA256)
    raw["llm"]["sonnet"]["enabled"] = sonnet
    if thinking is not None:
        raw["llm"]["sonnet"].update(thinking=thinking[0], thinking_tokens=thinking[1])
    raw["llm"]["caps"].update(ledger_file=str(folder / "key" / "llm_ledger.db"),
                              ledger_lock_file=str(folder / "key" / "llm_ledger.lock"))
    path = folder / "configs" / "news.toml"
    path.parent.mkdir(parents=True)
    path.write_text(check.toml_text(raw), encoding="utf-8")
    return path


def key_ledger_rows(folder):
    path = folder / "key" / "llm_ledger.db"
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM reservations ORDER BY id")]
    finally:
        conn.close()


# A fake client that answers from the answer keys


def lines(request):
    return [json.loads(line) for line in request["messages"][0]["content"].split("\n")[1:]]


class KeyClient:
    """Answers every synthetic item and probe with its written key, Haiku reading the cached prompt after its first
    call, and records each request. A test may change one answer by id or replace a model's result."""

    def __init__(self, llm, flips=(), sonnet_result=None, cache_read=1400, counted=2000):
        self.keys = {}
        for item in check.SYNTHETIC:
            sent = check.as_request_item(item, llm)
            self.keys[(sent.headline, sent.text)] = list(item.key)
        for probe in probes.load_probes(llm.probe_file):
            self.keys[(probe.item.headline, probe.item.text)] = [(k.ticker, k.direction) for k in probe.key]
        self.flips = {(item.headline, item.text) for item in check.SYNTHETIC if item.synthetic_id in flips}
        self.sonnet_result = sonnet_result
        self.cache_read = cache_read
        self.counted = counted
        self.requests = []
        self.closed = False

    async def count_tokens(self, request):
        size = len(request["messages"][0]["content"]) // 4
        return None if self.counted is None else size + (self.counted if "system" in request else 0)

    async def send(self, request):
        self.requests.append(request)
        model = request["model"]
        if model == SONNET_MODEL and self.sonnet_result is not None:
            return self.sonnet_result
        entries = []
        for item in lines(request):
            key = self.keys[(item["headline"], item["text"])]
            if (item["headline"], item["text"]) in self.flips:
                key = [(ticker, -direction or 1) for ticker, direction in key]
            entries.append({"id": item["id"], "impacts": [
                {"ticker": t, "event_type": "other", "is_new": True, "direction": d, "surprise": 1,
                 "horizon": "days", "confidence": 3} for t, d in key]})
        first = sum(r["model"] == model for r in self.requests) == 1
        read = 0 if model == SONNET_MODEL or first else self.cache_read
        write = 1400 if model == HAIKU_MODEL and first else 0
        text = json.dumps({"items": entries})
        return CallResult(OK, 700, http_status=200, stop_reason="end_turn", usage=Usage(2000, 150, write, read),
                          response_model=model, request_id="req_fixture", text=text, raw_json=text)

    async def close(self):
        self.closed = True

    def models(self):
        return [request["model"] for request in self.requests]


@pytest.fixture(autouse=True)
def quiet_redact(monkeypatch):
    """main installs redaction hooks for the whole process, which a test run must not keep."""
    monkeypatch.setattr(check.redact, "install", lambda: None)


def run_main(capsys, argv, client=None, start=None):
    code = check.main(argv, connector=(lambda config: client) if client is not None else check.connect,
                      start=start or check.start_news)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def summary(out):
    return dict(line.split(": ", 1) for line in out.splitlines()[1:] if ": " in line and "rows hidden" not in line)


# The synthetic items


def test_the_synthetic_set_is_40_fictional_undated_items_with_a_key_inside_the_schema():
    items = check.SYNTHETIC
    assert [item.synthetic_id for item in items] == [f"s{n:02d}" for n in range(1, 41)]
    assert sum(item.source == "finnhub" for item in items) == 30
    officials = [item for item in items if item.source != "finnhub"]
    assert sorted(item.kind for item in officials) == sorted(OFFICIAL_KINDS * 2)
    assert all(item.kind == "media" for item in items if item.source == "finnhub")
    assert all(bool(item.filer) == (item.source == "edgar") for item in items)
    assert len({(item.headline, item.text) for item in items}) == 40
    for item in items:
        assert all(ticker in TICKERS and direction in DIRECTIONS for ticker, direction in item.key)
        assert len({ticker for ticker, _ in item.key}) == len(item.key)
        assert not DATE.search(item.headline + " " + item.text), item.synthetic_id
        assert not re.search(r"btc|bitcoin|xbt", item.headline + item.text, re.IGNORECASE)
    assert sum(item.clear for item in items) == 29
    assert not {item.synthetic_id for item in items if not item.clear} & {"s01", "s02", "s03", "s35", "s40"}
    assert {"s09", "s17"} <= {item.synthetic_id for item in items if not item.clear}
    assert any("Ignore all previous instructions" in item.text and item.key == () for item in items)


def test_drill_items_are_full_length_fictional_long_form_statements_drawn_the_same_every_time():
    llm = load_news_config(COMMITTED).llm
    drawn = [check.drill_item(n) for n in range(check.DRILL_ITEMS)]
    assert drawn == [check.drill_item(n) for n in range(check.DRILL_ITEMS)]
    assert len({item.text for item in drawn}) == check.DRILL_ITEMS
    for item in drawn:
        assert item.kind in llm.long_form_kinds and len(item.text) == llm.long_form_max_chars
        assert not DATE.search(item.text) and item.text.startswith("Fictional statement")
        request = llm_request.build_request(llm_request.haiku_spec(llm), [check.as_request_item(item, llm)])
        llm_request.check_size(request, llm.max_input_tokens)


def test_the_plan_sends_finnhub_items_in_full_batches_and_each_official_item_alone_to_each_model():
    llm = load_news_config(COMMITTED).llm
    planned = check.plan(llm, True)
    assert [len(p.items) for p in planned if p.name.startswith("haiku batch")] == [20, 10]
    singles = [p for p in planned if len(p.items) == 1]
    assert [p.request.spec.model for p in singles] == [HAIKU_MODEL] * 10 + [SONNET_MODEL] * 10
    assert all(p.items[0].source != "finnhub" for p in singles)
    assert [p.name for p in check.plan(llm, False)] == [p.name for p in planned if p.name.startswith("haiku")]


def test_an_answer_matches_its_key_ticker_by_ticker_and_an_empty_key_wants_no_impact():
    impact = check.Impact(ticker="AAPL", event_type="earnings", is_new=True, direction=1, surprise=1,
                          horizon="days", confidence=4)
    assert check.matches((("AAPL", 1),), (impact,))
    assert check.matches((("AAPL", 1),), (impact, impact.model_copy(update={"ticker": "QQQ"})))
    assert not check.matches((("AAPL", -1),), (impact,))
    assert not check.matches((("AAPL", 1), ("MACRO", -1)), (impact,))
    assert check.matches((), ()) and not check.matches((), (impact,))
    assert not check.matches((("AAPL", 1),), None)


# The dry run


def test_the_dry_run_counts_every_request_records_its_hash_and_sends_none(tmp_path, capsys):
    config = write_config(tmp_path)
    llm = load_news_config(config).llm
    client = KeyClient(llm)
    code, out, _ = run_main(capsys, [str(config), "--dry-run"], client)
    assert code == 0 and client.requests == [] and client.closed
    found = summary(out)
    planned = check.plan(llm, True)
    probe_request, _ = check.probe_plan(llm)
    assert found["verdict"] == "pass" and found["requests"] == found["requests counted"] == str(len(planned) + 1)
    assert found["byte bound covers every count"] == found["size guard passed"] == "pass"
    assert (found["prompt tokens"], found["prompt above cache minimum"]) == ("2000", "pass")
    assert found["first haiku request"] == planned[0].request.sha256
    assert found["first sonnet request"] == next(p.request.sha256 for p in planned if p.name.startswith("sonnet"))
    assert found["probe request"] == probe_request.sha256
    assert "rows hidden" in out and "usd" not in out
    _, detail, _ = run_main(capsys, [str(config), "--dry-run", "--detail"], KeyClient(llm))
    assert "reserve usd" in detail and planned[0].request.sha256 in detail


def test_a_dry_run_without_counts_or_with_a_count_above_the_byte_bound_fails(tmp_path, capsys):
    config = write_config(tmp_path)
    llm = load_news_config(config).llm
    code, out, _ = run_main(capsys, [str(config), "--dry-run"], KeyClient(llm, counted=None))
    assert code == 1 and summary(out)["requests counted"] == "0" and "prompt tokens" not in summary(out)
    code, out, _ = run_main(capsys, [str(config), "--dry-run"], KeyClient(llm, counted=10**6))
    assert code == 1 and summary(out)["byte bound covers every count"] == "fail"
    assert summary(out)["size guard passed"] == "fail"


# The live check


def test_the_live_check_passes_on_good_answers_and_books_every_call_in_the_key_ledger(tmp_path, capsys):
    config = write_config(tmp_path)
    news = load_news_config(config)
    client = KeyClient(news.llm)
    code, out, _ = run_main(capsys, [str(config)], client)
    found = summary(out)
    assert code == 0 and found["verdict"] == "pass", out
    assert client.models() == [HAIKU_MODEL] * 12 + [SONNET_MODEL] * 10 + [HAIKU_MODEL] * 2
    assert found["calls"] == "24" and found["clear cases"] == found["clear cases matched"] == "29"
    assert found["haiku cache read tokens"] == "1400" and found["probe direction agreement"] == "ok"
    assert found["haiku first status"] == found["sonnet first status"] == "ok"
    assert (found["sonnet thinking"], found["sonnet first call thinking tokens"]) == (news.llm.sonnet.thinking, "0")
    rows = key_ledger_rows(tmp_path)
    assert len(rows) == 24 and {row["status"] for row in rows} == {"done"}
    assert {row["process"] for row in rows} == {f"{news.config_hash} {check.CHECK_PROCESS}"}
    assert all(row["cost_usd"] <= row["reserved_usd"] for row in rows)


def test_a_failed_first_sonnet_call_stops_sonnet_and_fails_the_check(tmp_path, capsys):
    config = write_config(tmp_path)
    llm = load_news_config(config).llm
    refused = CallResult(BAD_REQUEST, 300, http_status=400, error_class="BadRequestError",
                         error_detail="a fictional message about the thinking setting")
    client = KeyClient(llm, sonnet_result=refused)
    code, out, _ = run_main(capsys, [str(config), "--detail"], client)
    found = summary(out)
    assert code == 1 and found["verdict"] == "fail"
    assert client.models().count(SONNET_MODEL) == 1
    assert (found["sonnet first call"], found["sonnet first status"]) == ("fail", "bad_request")
    assert found["haiku first call"] == found["haiku matches the key"] == "pass"
    assert "a fictional message about the thinking setting" in out
    assert sum(row["model"] == SONNET_MODEL for row in key_ledger_rows(tmp_path)) == 1


class ThinkingSonnet(KeyClient):
    """Answers from the key, Sonnet with one thinking block and 120 thinking tokens inside its output."""

    async def send(self, request):
        result = await super().send(request)
        if request["model"] != SONNET_MODEL:
            return result
        usage = dataclasses.replace(result.usage, output_tokens=result.usage.output_tokens + 120, thinking_tokens=120)
        return dataclasses.replace(result, thinking_blocks=1, usage=usage)


@pytest.mark.parametrize("thinking, verdict", [(("between_tools", 1024), "pass"), (("adaptive", 4096), "pass"),
                                               (("disabled", 0), "fail")])
def test_sonnet_thinking_passes_the_first_call_unless_its_thinking_is_disabled(tmp_path, capsys, thinking, verdict):
    config = write_config(tmp_path, thinking=thinking)
    client = ThinkingSonnet(load_news_config(config).llm)
    code, out, _ = run_main(capsys, [str(config)], client)
    found = summary(out)
    assert found["sonnet first call"] == found["verdict"] == verdict and code == (0 if verdict == "pass" else 1)
    assert (found["sonnet thinking"], found["sonnet first call thinking tokens"]) == (thinking[0], "120")
    sent = [r for r in client.requests if r["model"] == SONNET_MODEL]
    assert {r["max_tokens"] for r in sent} == {1024 + thinking[1]}
    assert {json.dumps(r["thinking"]) for r in sent} == {json.dumps({"type": thinking[0]})}
    assert all(row["cost_usd"] <= row["reserved_usd"] for row in key_ledger_rows(tmp_path))


class FailsFrom(KeyClient):
    """Answers from the key until one model's nth call, then returns one failed result for every call of that
    model."""

    def __init__(self, llm, model, nth, failed):
        super().__init__(llm)
        self.model, self.nth, self.failed = model, nth, failed

    async def send(self, request):
        if request["model"] == self.model and sum(r["model"] == self.model for r in self.requests) >= self.nth - 1:
            self.requests.append(request)
            return self.failed
        return await super().send(request)


def test_a_later_spend_limit_stops_both_models_and_a_later_rate_limit_stops_its_own(tmp_path, capsys):
    config = write_config(tmp_path / "spend")
    spend = CallResult(SPEND_LIMIT, 300, http_status=400, error_class="BadRequestError", error_detail="a fiction")
    client = FailsFrom(load_news_config(config).llm, HAIKU_MODEL, 3, spend)
    code, out, _ = run_main(capsys, [str(config)], client)
    assert code == 1 and client.models() == [HAIKU_MODEL] * 3 and len(key_ledger_rows(tmp_path / "spend")) == 3
    config = write_config(tmp_path / "rate")
    limited = CallResult(RATE_LIMITED, 300, http_status=429, error_class="RateLimitError", error_detail="a fiction")
    client = FailsFrom(load_news_config(config).llm, HAIKU_MODEL, 3, limited)
    code, out, _ = run_main(capsys, [str(config)], client)
    assert code == 1 and client.models() == [HAIKU_MODEL] * 3 + [SONNET_MODEL] * 10


def test_no_cache_read_on_the_second_haiku_call_fails_the_check(tmp_path, capsys):
    config = write_config(tmp_path)
    code, out, _ = run_main(capsys, [str(config)], KeyClient(load_news_config(config).llm, cache_read=0))
    assert code == 1 and summary(out)["haiku cache read on the second call"] == "fail"


@pytest.mark.parametrize("wrong, verdict", [(("s01", "s02"), "pass"), (("s01", "s02", "s03"), "fail")])
def test_the_key_must_match_on_at_least_90_percent_of_the_clear_cases(tmp_path, capsys, wrong, verdict):
    config = write_config(tmp_path)
    code, out, _ = run_main(capsys, [str(config)], KeyClient(load_news_config(config).llm, flips=wrong))
    found = summary(out)
    assert found["clear cases matched"] == str(29 - len(wrong)) and found["haiku matches the key"] == verdict
    assert code == (0 if verdict == "pass" else 1)


@pytest.mark.parametrize("thinking, room", [("between_tools", 1024), ("adaptive", 4096), ("disabled", 0)])
def test_sonnet_thinking_on_the_command_line_turns_sonnet_on_for_one_run_and_leaves_the_file(tmp_path, capsys,
                                                                                               thinking, room):
    config = write_config(tmp_path, sonnet=False)
    before = config.read_bytes()
    client = ThinkingSonnet(load_news_config(config).llm)
    code, out, _ = run_main(capsys, [str(config), "--sonnet-thinking", thinking], client)
    found = summary(out)
    sent = [r for r in client.requests if r["model"] == SONNET_MODEL]
    # A thinking answer fails the first call while thinking is disabled, and a failed first call sends nothing more.
    assert len(sent) == (1 if room == 0 else 10) and {r["max_tokens"] for r in sent} == {1024 + room}
    assert {json.dumps(r["thinking"]) for r in sent} == {json.dumps({"type": thinking})}
    assert found["sonnet thinking"] == thinking and found["sonnet first call"] == ("fail" if room == 0 else "pass")
    assert config.read_bytes() == before and code == (1 if room == 0 else 0)
    assert all(row["cost_usd"] <= row["reserved_usd"] for row in key_ledger_rows(tmp_path))


def test_the_cap_drill_refuses_a_sonnet_thinking_setting(tmp_path, capsys):
    config = write_config(tmp_path, sonnet=False)
    code, _, err = run_main(capsys, [str(config), "--cap-drill", "--sonnet-thinking", "between_tools"])
    assert code == 2 and "takes no --sonnet-thinking" in err


def test_with_sonnet_disabled_the_check_sends_no_sonnet_call(tmp_path, capsys):
    config = write_config(tmp_path, sonnet=False)
    client = KeyClient(load_news_config(config).llm)
    code, out, _ = run_main(capsys, [str(config)], client)
    assert code == 0 and SONNET_MODEL not in client.models()
    assert summary(out)["sonnet"] == "skipped while disabled in the config" and "sonnet first call" not in out


def test_the_check_books_its_own_calls_lost_by_an_earlier_run_before_it_starts(tmp_path, capsys):
    config = write_config(tmp_path)
    news = load_news_config(config)
    with Ledger.from_config(news, process=f"{news.config_hash} {check.CHECK_PROCESS}") as ledger:
        ledger.reserve(HAIKU_MODEL, "check_haiku_v1:", 1, 0.003, LiveClock().now_ms())
    with Ledger.from_config(news, process=f"{news.config_hash} {tmp_path / 'news'}") as other:
        other.reserve(HAIKU_MODEL, "haiku_v1:1", 1, 0.003, LiveClock().now_ms())
    run_main(capsys, [str(config)], KeyClient(news.llm))
    rows = key_ledger_rows(tmp_path)
    assert rows[0]["status"] == "lost" and rows[1]["status"] == "started"


def test_default_output_holds_only_verdicts_counts_hashes_and_words(tmp_path, capsys):
    config = write_config(tmp_path)
    code, out, _ = run_main(capsys, [str(config)], KeyClient(load_news_config(config).llm))
    assert out.splitlines()[0] == "llm synthetic check"
    for label, value in summary(out).items():
        assert SUMMARY_VALUE.fullmatch(value), (label, value)
    for item in check.SYNTHETIC:
        assert item.headline not in out
    assert "usd" not in out and re.search(r"\d\.\d", out) is None


def test_an_unpinned_config_or_a_proxy_variable_cannot_run(tmp_path, capsys, monkeypatch):
    raw = tomllib.loads(COMMITTED.read_text(encoding="utf-8"))
    raw["llm"]["prompt_sha256"] = ""
    unpinned = tmp_path / "unpinned" / "configs" / "unpinned.toml"
    unpinned.parent.mkdir(parents=True)
    unpinned.write_text(check.toml_text(raw), encoding="utf-8")
    code, _, err = run_main(capsys, [str(unpinned), "--dry-run"], KeyClient(load_news_config(COMMITTED).llm))
    assert code == 2 and "not pinned" in err
    config = write_config(tmp_path)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:1")
    code, _, err = run_main(capsys, [str(config), "--dry-run"])
    assert code == 2 and "HTTPS_PROXY" in err


# The cap drill


def test_the_toml_writer_round_trips_the_committed_news_config():
    raw = tomllib.loads(COMMITTED.read_text(encoding="utf-8"))
    assert tomllib.loads(check.toml_text(raw)) == raw


def test_the_drill_config_changes_only_what_the_plan_lists(tmp_path):
    source = write_config(tmp_path)
    drill_dir = tmp_path / "rehearsal" / "cap_drill"
    path = tmp_path / "rehearsal" / "cap_drill_news.toml"
    path.parent.mkdir()
    path.write_text(check.drill_toml(source, drill_dir), encoding="utf-8")
    before, after = load_news_config(source), load_news_config(path)
    assert after.data_dir == drill_dir
    assert [after.finnhub.enabled, after.edgar.enabled, after.fed.enabled, after.bls.enabled] == [False] * 4
    assert after.llm.sonnet.enabled is False and after.llm.caps.monthly_usd == 0.01
    assert after.path(after.llm.caps.ledger_file) == drill_dir / "llm_ledger.db"
    assert after.path(after.llm.caps.ledger_lock_file) == drill_dir / "llm_ledger.lock"
    assert after.llm.probe_time_utc == "23:59"
    for key in ("model_file", "tokenizer_file", "manifest_file"):
        assert after.path(getattr(after.finbert, key)) == before.path(getattr(before.finbert, key))
    assert after.path(after.news.secrets_file) == before.path(before.news.secrets_file)
    changed = {("news", "data_dir"), ("news", "secrets_file"), ("news", "rest_lock_file"), ("finnhub", "enabled"),
               ("edgar", "enabled"), ("fed", "enabled"), ("bls", "enabled"), ("llm", "probe_time_utc"),
               ("llm", "sonnet"), ("llm", "caps"), ("finbert", "model_file"), ("finbert", "tokenizer_file"),
               ("finbert", "manifest_file")}
    old, new = before.sections(), after.sections()
    for section in old:
        for key in old[section]:
            if (section, key) not in changed:
                assert old[section][key] == new[section][key], (section, key)
    assert {k: v for k, v in new["llm"]["sonnet"].items() if k != "enabled"} == {
        k: v for k, v in old["llm"]["sonnet"].items() if k != "enabled"}
    assert new["llm"]["caps"]["daily_alert_share"] == old["llm"]["caps"]["daily_alert_share"]


class FakeModel:
    def score(self, headline, text=None):
        return Probs(0.2, 0.5, 0.3, 1, 40)


class DrillClient:
    """A long form Haiku call counted at 7,000 tokens and answering with 150, so each books about 0.0008 dollars
    against a reservation of about 0.003, as the plan expects of a long form call."""

    def __init__(self):
        self.requests = []

    async def count_tokens(self, request):
        return 7000

    async def send(self, request):
        self.requests.append(request)
        text = json.dumps({"items": [{"id": item["id"], "impacts": []} for item in lines(request)]})
        return CallResult(OK, 500, http_status=200, stop_reason="end_turn", usage=Usage(7000, 150),
                          response_model=request["model"], text=text, raw_json=text)


class FakeNews:
    """The drill's news process in a thread, running the real Scoring and FinbertScorer on the drill config with a
    fake client and model, as the integrated news process does, and stopped the way SIGTERM stops it."""

    def __init__(self, config_path, log_dir, client=None):
        self.config = load_news_config(config_path)
        self.client = client or DrillClient()
        self.ready = threading.Event()
        self.stopped = False
        self.thread = threading.Thread(target=lambda: asyncio.run(self.main()), daemon=True)
        self.thread.start()
        assert self.ready.wait(10)

    async def main(self):
        self.loop = asyncio.get_running_loop()
        self.event = asyncio.Event()
        conn = db.open_writer(self.config.db_path)
        ledger = Ledger.from_config(self.config)
        clock = LiveClock()
        try:
            record_start(conn, self.config, clock.stamp(), 0)
            scoring = Scoring(self.config, conn, clock, self.client, ledger)
            finbert = FinbertScorer(self.config, conn, clock, FakeModel())
            self.ready.set()

            async def wait(stop, seconds):
                return await wait_or_stop(stop, 0.01)
            await asyncio.gather(scoring.run(self.event, wait), finbert.run(self.event, wait))
        finally:
            ledger.close()
            conn.close()

    def alive(self):
        return self.thread.is_alive()

    def stop(self):
        self.stopped = True
        self.loop.call_soon_threadsafe(self.event.set)
        self.thread.join(10)


FAST = check.DrillTiming(start_s=10.0, item_s=10.0, poll_s=0.01)


def test_the_drill_passes_against_the_real_scorers_and_books_its_spend_into_the_key_ledger(tmp_path, capsys):
    source = write_config(tmp_path)
    started = []

    def start(path, log_dir):
        started.append(FakeNews(path, log_dir))
        return started[-1]

    report = check.cap_drill(source, start, tmp_path / "rehearsal", FAST)
    found = dict(report.summary)
    assert found["verdict"] is True, report
    [news] = started
    assert news.stopped and not news.alive()
    items = found["items inserted"].value
    assert items == found["haiku calls"].value + 2 <= check.DRILL_ITEMS
    assert len(news.client.requests) == found["haiku calls"].value
    assert [row["haiku"] for row in report.rows[:-1]] == ["irrelevant"] * (items - 2) + ["skipped_cap"] * 2
    assert {row["finbert"] for row in report.rows[:-1]} == {"scored"}
    drill = load_news_config(tmp_path / "rehearsal" / "cap_drill_news.toml")
    conn = db.open_reader(drill.db_path)
    try:
        assert db.get_value(conn, "scorer_state", "haiku_state") == "capped"
        assert conn.execute("SELECT count(*) FROM item_scores WHERE scorer = 'sonnet_v1' AND status != "
                            "'skipped_off'").fetchone()[0] == 0
    finally:
        conn.close()
    [booked] = key_ledger_rows(tmp_path)
    spent = float(report.rows[-1]["drill spend usd"])
    assert (booked["ref"], booked["model"], booked["n_items"], booked["status"], booked["reserved_usd"]) == (
        check.DRILL_PROCESS, HAIKU_MODEL, check.DRILL_ITEMS, "done", check.DRILL_MONTHLY_USD)
    assert abs(booked["cost_usd"] - spent) < 1e-6 and 0 < booked["cost_usd"] <= check.DRILL_MONTHLY_USD
    assert booked["process"] == f"{load_news_config(source).config_hash} {check.DRILL_PROCESS}"


class SpendLimitClient(DrillClient):
    """Answers like DrillClient for two calls, then returns the API's spend limit, which caps Haiku without the
    code's own cap refusing anything."""

    async def send(self, request):
        if len(self.requests) >= 2:
            self.requests.append(request)
            return CallResult(SPEND_LIMIT, 300, http_status=400, error_class="BadRequestError",
                              error_detail="a fictional spend limit")
        return await super().send(request)


def test_a_drill_capped_by_the_api_instead_of_the_code_fails(tmp_path):
    source = write_config(tmp_path)
    started = []

    def start(path, log_dir):
        started.append(FakeNews(path, log_dir, SpendLimitClient()))
        return started[-1]

    report = check.cap_drill(source, start, tmp_path / "rehearsal", FAST)
    found = dict(report.summary)
    assert found["verdict"] is False and found["a call refused by the cap"] is False
    assert found["haiku capped until the next utc month"] is True and found["drill spend within one cent"] is True
    assert [row["haiku"] for row in report.rows[:-1]] == ["irrelevant"] * 2 + ["failed"] + ["skipped_cap"] * 2


def test_the_drill_refuses_to_start_without_a_cent_left_on_the_key_ledger(tmp_path):
    source = write_config(tmp_path)
    config = load_news_config(source)
    now = LiveClock().now_ms()
    with Ledger.from_config(config, process="other") as key_ledger:
        key_ledger.book(HAIKU_MODEL, "other", 1, config.llm.caps.monthly_usd - 0.005, now)
    with pytest.raises(check.DrillError, match="less than the drill's one cent"):
        check.cap_drill(source, lambda path, log_dir: pytest.fail("started"), tmp_path / "rehearsal", FAST)
    assert not (tmp_path / "rehearsal").exists() and len(key_ledger_rows(tmp_path)) == 1


def test_a_drill_an_earlier_run_left_running_is_booked_lost_at_its_cent(tmp_path, capsys):
    source = write_config(tmp_path)
    config = load_news_config(source)
    with check._key_ledger(config) as key_ledger:
        key_ledger.reserve(HAIKU_MODEL, check.DRILL_PROCESS, check.DRILL_ITEMS, check.DRILL_MONTHLY_USD,
                           LiveClock().now_ms())
    report = check.cap_drill(source, lambda path, log_dir: DeadNews(), tmp_path / "rehearsal", FAST)
    assert dict(report.summary)["verdict"] is False
    earlier, this = key_ledger_rows(tmp_path)
    assert (earlier["status"], earlier["cost_usd"], this["status"], this["cost_usd"]) == (
        "lost", check.DRILL_MONTHLY_USD, "done", 0.0)


def test_the_drill_refuses_a_folder_left_by_an_earlier_drill(tmp_path):
    source = write_config(tmp_path)
    folder = tmp_path / "rehearsal" / "cap_drill"
    folder.mkdir(parents=True)
    (folder / "llm_ledger.db").write_bytes(b"")
    with pytest.raises(check.DrillError, match="earlier drill"):
        check.cap_drill(source, lambda path, log_dir: pytest.fail("started"), tmp_path / "rehearsal", FAST)
    assert key_ledger_rows(tmp_path) == []


class DeadNews:
    stopped = False

    def alive(self):
        return False

    def stop(self):
        self.stopped = True


def test_a_drill_whose_news_process_never_starts_fails_and_still_books_its_spend(tmp_path, capsys):
    source = write_config(tmp_path)
    dead = DeadNews()
    code, out, _ = run_main(capsys, [str(source), "--cap-drill", "--drill-root", str(tmp_path / "rehearsal")],
                            start=lambda path, log_dir: dead)
    found = summary(out)
    assert code == 1 and found["verdict"] == "fail" and found["items inserted"] == "0" and dead.stopped
    [booked] = key_ledger_rows(tmp_path)
    assert (booked["ref"], booked["cost_usd"], booked["n_items"], booked["status"]) == (
        check.DRILL_PROCESS, 0.0, check.DRILL_ITEMS, "done")

