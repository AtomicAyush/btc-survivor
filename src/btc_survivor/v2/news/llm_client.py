"""The one module that imports anthropic, PLAN_V2 section 8.1.

It sends one request and returns what came back, the status, stop reason, refusal category, usage, response model,
request id, raw JSON and any error class, and never retries. The SDK client is built with max_retries=0, so the SDK
never sends a request twice either, and it finds the key only where local_secrets.prepare_anthropic_env put it. An
error from the SDK becomes a status, never an exception, while a cancellation passes through and leaves the call's
reservation running, so the next start books it lost. It also counts a request's tokens, returning None when counting
fails so the caller reserves on the byte bound. Thinking blocks are counted and never read as the answer, and the
thinking tokens a response reports are kept with its usage.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import anthropic

from btc_survivor.v2 import local_secrets, redact
from btc_survivor.v2.config import HAIKU_MODEL, SONNET_MODEL
from btc_survivor.v2.local_secrets import Secrets
from btc_survivor.v2.news.ledger import Usage

MODELS = (HAIKU_MODEL, SONNET_MODEL)
# The only keys a request may carry, so no sampling parameter, tool, metadata, stop sequence or fallback list is sent.
REQUEST_KEYS = frozenset({"model", "max_tokens", "thinking", "output_config", "system", "messages"})

# The llm_calls statuses a call can end in here. The scorer adds malformed for an answer that fails validation.
OK, REFUSAL, MAX_TOKENS, MALFORMED = "ok", "refusal", "max_tokens", "malformed"
RATE_LIMITED, SPEND_LIMIT, BAD_REQUEST, AUTH = "rate_limited", "spend_limit", "bad_request", "auth"
SERVER, TIMEOUT, CONNECTION, MODEL_MISMATCH = "server", "timeout", "connection", "model_mismatch"
_STOP = {"end_turn": OK, "refusal": REFUSAL, "max_tokens": MAX_TOKENS}
# A 400 or 429 whose own error message names a spend limit, the credit balance, billing or the workspace's usage
# limit, whose wording is unverified. Bare words such as credit are left out, because they occur in news text.
_SPEND = re.compile(r"spend limit|credit balance|billing|usage limit", re.IGNORECASE)
DETAIL_CHARS = 300


@dataclass(frozen=True)
class CallResult:
    """What one call returned. usage is None when no response came back, and the call is then booked at its
    reservation. text is the answer's text blocks joined, the JSON the scorer validates."""

    status: str
    latency_ms: int
    http_status: int | None = None
    stop_reason: str | None = None
    refusal_category: str | None = None
    usage: Usage | None = None
    response_model: str | None = None
    request_id: str | None = None
    text: str | None = None
    raw_json: str | None = None
    thinking_blocks: int = 0
    error_class: str | None = None
    error_detail: str | None = None
    retry_after_s: float | None = None


def check_request(request: Mapping[str, Any]) -> str:
    """The model of a request the client may send, refusing any key outside REQUEST_KEYS and any model but the two
    chosen."""
    extra = sorted(set(request) - REQUEST_KEYS)
    if extra:
        raise ValueError(f"a request may not carry {', '.join(extra)}")
    model = request.get("model")
    if model not in MODELS:
        raise ValueError(f"the model must be one of {', '.join(MODELS)}")
    return model


def status_for(code: int, error_type: str | None, message: str) -> str:
    """The status of an HTTP error. A 402, a billing error, or a 400 or 429 whose message names a spend limit, the
    credit balance, billing or a usage limit is the spend limit, any other 429 a rate limit, 401 and 403 auth, 408 a
    timeout, 500 and above, 529 included, a server error, and any other 4xx a bad request."""
    if code == 402 or error_type == "billing_error" or (code in (400, 429) and _SPEND.search(message)):
        return SPEND_LIMIT
    if code == 429:
        return RATE_LIMITED
    if code in (401, 403):
        return AUTH
    if code == 408:
        return TIMEOUT
    if code >= 500:
        return SERVER
    return BAD_REQUEST


def error_message(exc: anthropic.APIStatusError) -> str:
    """The API's own error message from the body, never the SDK's text, which holds the whole body and so could echo
    request content."""
    body = exc.body
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message if isinstance(message, str) else exc.message


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The retry-after header in seconds, None when it is missing, unreadable or negative."""
    try:
        seconds = float(headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def failure(exc: anthropic.AnthropicError, latency_ms: int) -> CallResult:
    """The result of a call the SDK ended with an error, which never carries usage."""
    detail = redact.redact(str(getattr(exc, "message", exc)))[:DETAIL_CHARS]
    common = {"latency_ms": latency_ms, "error_class": type(exc).__name__, "error_detail": detail}
    if isinstance(exc, anthropic.APITimeoutError):
        return CallResult(TIMEOUT, **common)
    if isinstance(exc, anthropic.APIConnectionError):
        return CallResult(CONNECTION, **common)
    if isinstance(exc, anthropic.APIStatusError):
        return CallResult(status_for(exc.status_code, exc.type, error_message(exc)), http_status=exc.status_code,
                          request_id=exc.request_id, retry_after_s=_retry_after(exc.response.headers), **common)
    if isinstance(exc, anthropic.CredentialsError):
        return CallResult(AUTH, **common)
    return CallResult(SERVER, **common)


def thinking_tokens(usage: Any) -> int:
    """The output tokens a response spent on thinking, from usage.output_tokens_details, 0 when it reports none."""
    details = getattr(usage, "output_tokens_details", None)
    return getattr(details, "thinking_tokens", None) or 0


def answer(response: Any, model: str, latency_ms: int) -> CallResult:
    """The result of a response. A response from another model than the one requested is model_mismatch whatever it
    says, and a stop reason other than end_turn, refusal or max_tokens is malformed."""
    blocks = list(response.content or [])
    usage = response.usage
    stop_details = getattr(response, "stop_details", None)
    status = MODEL_MISMATCH if response.model != model else _STOP.get(response.stop_reason, MALFORMED)
    return CallResult(
        status, latency_ms, http_status=200, stop_reason=response.stop_reason,
        refusal_category=getattr(stop_details, "category", None) if response.stop_reason == "refusal" else None,
        usage=Usage(usage.input_tokens or 0, usage.output_tokens or 0, usage.cache_creation_input_tokens or 0,
                    usage.cache_read_input_tokens or 0, thinking_tokens(usage)),
        response_model=response.model, request_id=getattr(response, "_request_id", None),
        text="".join(block.text for block in blocks if block.type == "text"),
        raw_json=response.to_json(indent=None),
        thinking_blocks=sum(block.type in ("thinking", "redacted_thinking") for block in blocks))


class LlmClient:
    """Sends requests through one SDK client, one request per call and never a second."""

    def __init__(self, sdk: Any) -> None:
        self._sdk = sdk

    async def count_tokens(self, request: Mapping[str, Any]) -> int | None:
        """The exact input token count of a request from the count_tokens endpoint, or None when counting fails or
        does not take the full request."""
        check_request(request)
        fields = {key: value for key, value in request.items() if key != "max_tokens"}
        try:
            counted = await self._sdk.messages.count_tokens(**fields)
        except anthropic.AnthropicError:
            return None
        tokens = getattr(counted, "input_tokens", None)
        return tokens if isinstance(tokens, int) and tokens > 0 else None

    async def send(self, request: Mapping[str, Any]) -> CallResult:
        """Send one request and return what came back, mapping every SDK error to a status."""
        model = check_request(request)
        started = time.monotonic()
        try:
            response = await self._sdk.messages.create(**request)
        except anthropic.AnthropicError as exc:
            return failure(exc, _elapsed_ms(started))
        return answer(response, model, _elapsed_ms(started))

    async def close(self) -> None:
        await self._sdk.close()


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since a monotonic start."""
    return round((time.monotonic() - started) * 1000)


def sdk_version() -> str:
    """The SDK version every llm_calls row stores."""
    return anthropic.__version__


def sdk_client(timeout_s: float) -> anthropic.AsyncAnthropic:
    """The real SDK client, with no retries, which reads the key from the process environment."""
    return anthropic.AsyncAnthropic(max_retries=0, timeout=timeout_s)


def connect(secrets: Secrets, timeout_s: float, environ: MutableMapping[str, str] = os.environ,
            factory: Callable[[float], Any] | None = None) -> LlmClient:
    """Put the key where the SDK reads it through local_secrets alone, then build the client. The real SDK reads
    only the process environment, so without a factory environ must be os.environ, and tests pass a plain dict with
    a fake factory."""
    if factory is None:
        if environ is not os.environ:
            raise ValueError("the SDK reads its key from the process environment, so a real client needs os.environ")
        factory = sdk_client
    local_secrets.prepare_anthropic_env(secrets, environ)
    return LlmClient(factory(timeout_s))
