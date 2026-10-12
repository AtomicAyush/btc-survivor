"""Validation of a model's answer against schema_v1.json, PLAN_V2 section 8.4.

The answer is checked against strict Pydantic models that forbid extra fields, then each id must belong to the batch
once, tickers must be unique within an item, and an item may hold at most 10 impacts. Each item is checked on its
own, so the valid items are kept when others fail. Only an answer that is not JSON, or whose outer object breaks the
schema, fails as a whole. Reasons name the rule broken and never repeat a value from the answer.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ", "MACRO")
EVENT_TYPES = ("earnings", "guidance", "product", "legal_regulatory", "mergers_acquisitions", "management",
               "capital_return", "financing", "analyst_rating", "partners_supply", "macro_policy", "macro_data",
               "market_commentary", "other")
DIRECTIONS = (-1, 0, 1)
SURPRISES = (0, 1, 2, 3)
HORIZONS = ("hours", "days", "weeks")
CONFIDENCES = (1, 2, 3, 4, 5)
MAX_IMPACTS = 10

# Why a batch id has no valid result.
MISSING = "missing"
DUPLICATE_ID = "duplicate_id"
INVALID = "invalid"
DUPLICATE_TICKER = "duplicate_ticker"
TOO_MANY_IMPACTS = "too_many_impacts"
# Why a whole answer failed.
UNPARSABLE = "unparsable"
ENVELOPE = "envelope"


def _integer(value: Any) -> Any:
    """Refuse anything but a plain int before the enum check, which alone would take true for 1 or 1.0 for 1."""
    if type(value) is not int:
        raise ValueError("must be an integer")
    return value


_STRICT = ConfigDict(extra="forbid", strict=True, frozen=True)


class Impact(BaseModel):
    """One impact of one item, every field a required enum."""

    model_config = _STRICT
    ticker: Literal[TICKERS]
    event_type: Literal[EVENT_TYPES]
    is_new: bool
    direction: Annotated[Literal[DIRECTIONS], BeforeValidator(_integer)]
    surprise: Annotated[Literal[SURPRISES], BeforeValidator(_integer)]
    horizon: Literal[HORIZONS]
    confidence: Annotated[Literal[CONFIDENCES], BeforeValidator(_integer)]


class Item(BaseModel):
    """One answered item, its batch id and its impacts, which may be empty when it bears on no instrument."""

    model_config = _STRICT
    id: str
    impacts: list[Impact]


class _Envelope(BaseModel):
    """The outer object, whose entries are checked one at a time so one bad item cannot sink the rest."""

    model_config = _STRICT
    items: list[Any]


@dataclass(frozen=True)
class Validation:
    """The outcome for one call. scored maps each batch id with a valid result to its impacts, in batch order, and
    failed maps every other batch id to its reason. unknown_ids are answered ids outside the batch, unattributed
    counts entries without a string id, and whole names why the entire answer failed, None when it did not."""

    scored: Mapping[str, tuple[Impact, ...]]
    failed: Mapping[str, str]
    unknown_ids: tuple[str, ...]
    unattributed: int
    whole: str | None


def _whole(ids: Sequence[str], reason: str) -> Validation:
    """Every batch id failed, because the answer as a whole could not be read."""
    return Validation({}, {i: reason for i in ids}, (), 0, reason)


def _check(entry: Any) -> Item | str:
    """The checked item, or the reason it fails, for one entry already known to carry a batch id."""
    try:
        item = Item.model_validate(entry)
    except ValidationError:
        return INVALID
    if len(item.impacts) > MAX_IMPACTS:
        return TOO_MANY_IMPACTS
    if len({impact.ticker for impact in item.impacts}) < len(item.impacts):
        return DUPLICATE_TICKER
    return item


def validate(text: str, ids: Sequence[str]) -> Validation:
    """Check an answer's text against the batch ids it was asked about, keeping every valid item."""
    try:
        data = json.loads(text)
    except ValueError:
        return _whole(ids, UNPARSABLE)
    try:
        envelope = _Envelope.model_validate(data)
    except ValidationError:
        return _whole(ids, ENVELOPE)
    batch = set(ids)
    keyed = [(e["id"], e) for e in envelope.items if isinstance(e, dict) and isinstance(e.get("id"), str)]
    counts = Counter(key for key, _ in keyed)
    entries = {key: entry for key, entry in keyed if key in batch}
    scored: dict[str, tuple[Impact, ...]] = {}
    failed: dict[str, str] = {}
    for key in ids:
        if key not in entries:
            failed[key] = MISSING
        elif counts[key] > 1:
            failed[key] = DUPLICATE_ID
        else:
            checked = _check(entries[key])
            if isinstance(checked, str):
                failed[key] = checked
            else:
                scored[key] = tuple(checked.impacts)
    unknown = tuple(sorted({key for key, _ in keyed if key not in batch}))
    return Validation(scored, failed, unknown, len(envelope.items) - len(keyed), None)
