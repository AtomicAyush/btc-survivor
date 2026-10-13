"""The model client of PLAN_V2 section 8.1, only ever on a fake SDK client, so no test reaches the Anthropic API. One
request goes per call and never a second, every SDK error becomes a status, the answer's text, usage, model, stop
reason and refusal category come back, and the key reaches the SDK only through local_secrets on a plain dict, never
the process environment. Error objects are the SDK's own, built on a request that is never sent."""

import asyncio
import json

import anthropic
import httpx2
import pytest
from anthropic.types import Message, MessageTokensCount

from btc_survivor.v2 import local_secrets, redact
from btc_survivor.v2.config import HAIKU_MODEL, SONNET_MODEL
from btc_survivor.v2.local_secrets import ANTHROPIC, CONTACT, FINNHUB, SecretsError, load_secrets
from btc_survivor.v2.news import llm_client
from btc_survivor.v2.news.ledger import Usage
from btc_survivor.v2.news.llm_client import (
    AUTH, BAD_REQUEST, CONNECTION, MALFORMED, MAX_TOKENS, MODEL_MISMATCH, OK, RATE_LIMITED, REFUSAL, SERVER,
    SPEND_LIMIT, TIMEOUT, CallResult, LlmClient, check_request)

FAKE = {FINNHUB: "fakefinnhub0001token", ANTHROPIC: "fake-anthropic-0002-value", CONTACT: "fake.owner@example.invalid"}
URL = "https://api.anthropic.invalid/v1/messages"   # only labels the error objects, nothing is sent
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": {"type": "object"}}}}
REQUEST = {
    "model": HAIKU_MODEL, "max_tokens": 4096, "thinking": {"type": "disabled"},
    "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    "system": [{"type": "text", "text": "Fictional fixed prompt.", "cache_control": {"type": "ephemeral"}}],
    "messages": [{"role": "user", "content": "Score each item.\n{\"headline\":\"Fictional headline\",\"id\":\"i01\"}"}],
}


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(redact, "_values", set())


def message(text='{"items":[]}', stop_reason="end_turn", model=HAIKU_MODEL, usage=None, content=None,
            stop_details=None):
    """A response as the SDK parses it, built locally."""
    return Message.model_validate({
        "id": "msg_fixture", "type": "message", "role": "assistant", "model": model,
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
        "usage": usage or {"input_tokens": 120, "output_tokens": 40}})


def status_error(cls, code, text, headers=None, error_type="invalid_request_error"):
    response = httpx2.Response(code, headers=headers or {}, request=httpx2.Request("POST", URL))
    body = {"type": "error", "error": {"type": error_type, "message": text}}
    return cls(f"Error code: {code} - {body}", response=response, body=body)


class FakeMessages:
    """Records every request and answers with one fixed outcome, a response or an exception."""

    def __init__(self, outcome=None, counted=1500):
        self.outcome = outcome if outcome is not None else message()
        self.counted = counted
        self.created, self.counts = [], []

    async def create(self, **request):
        self.created.append(request)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    async def count_tokens(self, **request):
        self.counts.append(request)
        if isinstance(self.counted, BaseException):
            raise self.counted
        return MessageTokensCount(input_tokens=self.counted)


class FakeSdk:
    def __init__(self, outcome=None, counted=1500):
        self.messages = FakeMessages(outcome, counted)
        self.closed = False

    async def close(self):
        self.closed = True


def send(outcome, request=REQUEST):
    sdk = FakeSdk(outcome)
    result = asyncio.run(LlmClient(sdk).send(request))
    return result, sdk


# Answers


def test_a_valid_answer_returns_its_text_usage_model_and_raw_json_after_one_request():
    response = message(text='{"items":', usage={"input_tokens": 300, "output_tokens": 55,
                                                 "cache_creation_input_tokens": None, "cache_read_input_tokens": 1200},
                       content=[{"type": "text", "text": '{"items":'}, {"type": "text", "text": "[]}"}])
    result, sdk = send(response)
    assert sdk.messages.created == [REQUEST]
    assert (result.status, result.http_status, result.stop_reason, result.refusal_category) == (OK, 200, "end_turn",
                                                                                              None)
    assert result.text == '{"items":[]}' and json.loads(result.text) == {"items": []}
    assert result.usage == Usage(input_tokens=300, output_tokens=55, cache_creation_input_tokens=0,
                                 cache_read_input_tokens=1200)
    assert (result.response_model, result.request_id, result.thinking_blocks) == (HAIKU_MODEL, None, 0)
    assert json.loads(result.raw_json)["id"] == "msg_fixture"
    assert result.error_class is None and result.latency_ms >= 0


def test_a_refusal_returns_its_category():
    result, _ = send(message(stop_reason="refusal", text="",
                             stop_details={"type": "refusal", "category": "cyber", "explanation": None}))
    assert (result.status, result.refusal_category) == (REFUSAL, "cyber")
    result, _ = send(message(stop_reason="refusal", text=""))
    assert (result.status, result.refusal_category) == (REFUSAL, None)


@pytest.mark.parametrize("stop_reason, status", [("max_tokens", MAX_TOKENS), ("stop_sequence", MALFORMED),
                                                 ("pause_turn", MALFORMED), ("model_context_window_exceeded",
                                                                             MALFORMED)])
def test_other_stop_reasons(stop_reason, status):
    result, _ = send(message(stop_reason=stop_reason))
    assert (result.status, result.stop_reason) == (status, stop_reason)
    assert result.usage is not None


def test_an_answer_from_another_model_is_a_model_mismatch():
    result, _ = send(message(model=SONNET_MODEL))
    assert (result.status, result.response_model) == (MODEL_MISMATCH, SONNET_MODEL)
    result, _ = send(message(model=HAIKU_MODEL), {**REQUEST, "model": SONNET_MODEL, "max_tokens": 1024})
    assert result.status == MODEL_MISMATCH


def test_thinking_blocks_are_counted_and_never_read_as_the_answer():
    content = [{"type": "thinking", "thinking": "", "signature": "fixture"}, {"type": "redacted_thinking", "data": "x"},
               {"type": "text", "text": '{"items":[]}'}]
    result, _ = send(message(content=content))
    assert (result.status, result.thinking_blocks, result.text) == (OK, 2, '{"items":[]}')


def test_thinking_tokens_reported_in_the_usage_details_are_kept_as_part_of_the_output():
    content = [{"type": "thinking", "thinking": "", "signature": "fixture"}, {"type": "text", "text": '{"items":[]}'}]
    usage = {"input_tokens": 1900, "output_tokens": 700, "output_tokens_details": {"thinking_tokens": 620}}
    request = {**REQUEST, "model": SONNET_MODEL, "max_tokens": 2048, "thinking": {"type": "adaptive"}}
    result, sdk = send(message(model=SONNET_MODEL, content=content, usage=usage), request)
    assert sdk.messages.created == [request]
    assert (result.status, result.thinking_blocks, result.text) == (OK, 1, '{"items":[]}')
    assert result.usage == Usage(1900, 700, 0, 0, thinking_tokens=620) and result.usage.billed_output_tokens == 700
    result, _ = send(message(), REQUEST)
    assert result.usage.thinking_tokens == 0


def test_sonnet_between_tools_is_sent_as_given():
    request = {**REQUEST, "model": SONNET_MODEL, "max_tokens": 2048, "thinking": {"type": "between_tools"}}
    result, sdk = send(message(model=SONNET_MODEL), request)
    assert sdk.messages.created == [request] and result.status == OK


# Errors, each one request and no retry


ERRORS = [
    (status_error(anthropic.RateLimitError, 429, "Number of requests has exceeded your rate limit",
                  {"retry-after": "30", "request-id": "req_fixture"}, "rate_limit_error"), RATE_LIMITED, 429),
    (status_error(anthropic.RateLimitError, 429, "Your credit balance is too low", error_type="rate_limit_error"),
     SPEND_LIMIT, 429),
    (status_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the API"),
     SPEND_LIMIT, 400),
    (status_error(anthropic.BadRequestError, 400, "You have reached your specified API usage limits"),
     SPEND_LIMIT, 400),
    (status_error(anthropic.BadRequestError, 400, "The workspace spend limit was reached"), SPEND_LIMIT, 400),
    (status_error(anthropic.APIStatusError, 402, "Payment required", error_type="billing_error"), SPEND_LIMIT, 402),
    (status_error(anthropic.BadRequestError, 400, "thinking.type: unexpected value"), BAD_REQUEST, 400),
    (status_error(anthropic.NotFoundError, 404, "model not found", error_type="not_found_error"), BAD_REQUEST, 404),
    (status_error(anthropic.RequestTooLargeError, 413, "request too large", error_type="request_too_large"),
     BAD_REQUEST, 413),
    (status_error(anthropic.AuthenticationError, 401, "invalid x-api-key", error_type="authentication_error"),
     AUTH, 401),
    (status_error(anthropic.PermissionDeniedError, 403, "not allowed", error_type="permission_error"), AUTH, 403),
    (status_error(anthropic.APIStatusError, 408, "request timeout", error_type="timeout_error"), TIMEOUT, 408),
    (status_error(anthropic.InternalServerError, 500, "internal error", error_type="api_error"), SERVER, 500),
    (status_error(anthropic.OverloadedError, 529, "overloaded", error_type="overloaded_error"), SERVER, 529),
    (anthropic.APITimeoutError(request=httpx2.Request("POST", URL)), TIMEOUT, None),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", URL)), CONNECTION, None),
    (anthropic.CredentialsError("no credentials"), AUTH, None),
    (anthropic.AnthropicError("something else"), SERVER, None),
]


@pytest.mark.parametrize("error, status, code", ERRORS)
def test_every_sdk_error_becomes_a_status_after_one_request(error, status, code):
    result, sdk = send(error)
    assert len(sdk.messages.created) == 1
    assert (result.status, result.http_status, result.error_class) == (status, code, type(error).__name__)
    assert result.usage is None and result.text is None and result.raw_json is None


def test_the_spend_limit_is_read_from_the_api_message_alone_and_never_from_a_bare_word():
    response = httpx2.Response(400, request=httpx2.Request("POST", URL))
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "thinking.type: unexpected value"}}
    echoed = anthropic.BadRequestError(f"Error code: 400 - {body} with credit lines and a spend limit echoed",
                                       response=response, body=body)
    assert send(echoed)[0].status == BAD_REQUEST
    assert send(status_error(anthropic.BadRequestError, 400, "the item mentions credit lines"))[0].status == BAD_REQUEST
    plain = anthropic.BadRequestError("Your credit balance is too low", response=response, body=None)
    assert send(plain)[0].status == SPEND_LIMIT


def test_a_rate_limit_carries_its_retry_after_time_and_request_id():
    result, _ = send(ERRORS[0][0])
    assert (result.retry_after_s, result.request_id) == (30.0, "req_fixture")
    result, _ = send(status_error(anthropic.RateLimitError, 429, "slow down", {"retry-after": "soon"}))
    assert result.retry_after_s is None


def test_an_error_detail_is_redacted_and_short():
    redact.register("fake-canary-0003")
    result, _ = send(status_error(anthropic.BadRequestError, 400, "echo fake-canary-0003 " + "x" * 600))
    assert "fake-canary-0003" not in result.error_detail and redact.MASK in result.error_detail
    assert len(result.error_detail) == llm_client.DETAIL_CHARS


@pytest.mark.parametrize("error", [asyncio.CancelledError(), RuntimeError("a bug")], ids=["cancelled", "bug"])
def test_a_cancellation_or_a_bug_passes_through_so_the_reservation_stays_running(error):
    with pytest.raises(type(error)):
        send(error)


# The request


@pytest.mark.parametrize("key", ["temperature", "top_p", "top_k", "tools", "tool_choice", "metadata", "stop_sequences",
                                 "fallbacks", "betas", "service_tier", "inference_geo", "stream", "cache_control"])
def test_a_request_with_any_other_key_is_refused_before_sending(key):
    sdk = FakeSdk()
    with pytest.raises(ValueError, match=key):
        asyncio.run(LlmClient(sdk).send({**REQUEST, key: 1}))
    with pytest.raises(ValueError, match=key):
        asyncio.run(LlmClient(sdk).count_tokens({**REQUEST, key: 1}))
    assert sdk.messages.created == [] and sdk.messages.counts == []


@pytest.mark.parametrize("model", [HAIKU_MODEL + "-20261007", SONNET_MODEL + "-fast", "", None])
def test_only_the_two_chosen_models_are_sent(model):
    sdk = FakeSdk()
    with pytest.raises(ValueError, match="model must be one of"):
        asyncio.run(LlmClient(sdk).send({**REQUEST, "model": model}))
    assert sdk.messages.created == []
    assert check_request(REQUEST) == HAIKU_MODEL
    assert check_request({**REQUEST, "model": SONNET_MODEL}) == SONNET_MODEL


def test_count_tokens_sends_the_full_request_without_max_tokens():
    sdk = FakeSdk(counted=1873)
    assert asyncio.run(LlmClient(sdk).count_tokens(REQUEST)) == 1873
    assert sdk.messages.counts == [{key: value for key, value in REQUEST.items() if key != "max_tokens"}]
    assert sdk.messages.created == []


@pytest.mark.parametrize("counted", [
    status_error(anthropic.BadRequestError, 400, "output_config is not supported here"),
    anthropic.APIConnectionError(request=httpx2.Request("POST", URL)), 0], ids=["refused", "connection", "zero"])
def test_a_failed_count_returns_none_for_the_byte_bound(counted):
    sdk = FakeSdk(counted=counted)
    assert asyncio.run(LlmClient(sdk).count_tokens(REQUEST)) is None
    assert len(sdk.messages.counts) == 1


def test_close_closes_the_sdk_client():
    sdk = FakeSdk()
    asyncio.run(LlmClient(sdk).close())
    assert sdk.closed


# The key and the SDK client


def secrets_file(tmp_path):
    path = tmp_path / "secrets.env"
    path.write_text("".join(f"{name}={value}\n" for name, value in FAKE.items()), encoding="utf-8")
    path.chmod(0o600)
    return load_secrets(path)


def test_connect_puts_the_key_in_a_plain_dict_through_local_secrets_alone(tmp_path):
    secrets = secrets_file(tmp_path)
    environ = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9", "ANTHROPIC_AUTH_TOKEN": "fake-token", "PATH": "/usr/bin"}
    built = []

    def factory(timeout_s):
        built.append((timeout_s, dict(environ)))
        return FakeSdk()

    client = llm_client.connect(secrets, 60.0, environ, factory)
    assert isinstance(client, LlmClient)
    assert [timeout for timeout, _ in built] == [60.0]
    assert sorted(built[0][1]) == [ANTHROPIC, "PATH"]
    assert environ[ANTHROPIC] == FAKE[ANTHROPIC]


def test_connect_refuses_the_sdk_log_before_building_anything(tmp_path):
    secrets = secrets_file(tmp_path)
    environ = {"ANTHROPIC_LOG": "debug"}
    built = []
    with pytest.raises(SecretsError, match="ANTHROPIC_LOG"):
        llm_client.connect(secrets, 60.0, environ, lambda timeout_s: built.append(timeout_s))
    assert built == [] and environ == {"ANTHROPIC_LOG": "debug"}


def test_a_real_client_is_built_only_on_the_process_environment(tmp_path, monkeypatch):
    secrets = secrets_file(tmp_path)
    environ = {"PATH": "/usr/bin"}
    monkeypatch.setattr(local_secrets, "prepare_anthropic_env", lambda *args: pytest.fail("the key moved"))
    with pytest.raises(ValueError, match="process environment"):
        llm_client.connect(secrets, 60.0, environ)
    assert environ == {"PATH": "/usr/bin"}


def test_the_sdk_client_never_retries(monkeypatch):
    built = []
    monkeypatch.setattr(llm_client.anthropic, "AsyncAnthropic", lambda **kwargs: built.append(kwargs) or "client")
    assert llm_client.sdk_client(60.0) == "client"
    assert built == [{"max_retries": 0, "timeout": 60.0}]
    assert llm_client.sdk_version() == anthropic.__version__


def test_a_call_result_without_a_response_has_no_usage():
    assert CallResult(TIMEOUT, 12).usage is None
