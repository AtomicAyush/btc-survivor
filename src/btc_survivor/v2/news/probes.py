"""The daily probe of PLAN_V2 section 8.8, the 20 fictional items of probes_v2.json sent to Haiku as one call.

The file also holds a written answer key, the tickers, event types, novelty, directions, surprise and horizons a careful
reader would give each item under prompt_v2, every field of a strength but the confidence, for judging the first live
calls by hand. Sonnet runs no probe. Results go only to llm_probes, one row per impact plus an empty ticker row for a
valid answer that names no instrument, so a probe answered with no impacts differs from one that failed. Agreement
between two runs, or between a run and the key, is counted field by field over the probes both answered, and changes
nothing automatically.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from btc_survivor.clock import MS_PER_DAY, MS_PER_HOUR, MS_PER_MINUTE, parse_iso
from btc_survivor.v2.config import NEWS_KINDS, LlmConfig
from btc_survivor.v2.news import llm_request
from btc_survivor.v2.news.llm_request import Request, RequestItem
from btc_survivor.v2.news.llm_validate import DIRECTIONS, EVENT_TYPES, HORIZONS, SURPRISES, TICKERS, Validation

PROBE_FILE = "probes_v2.json"
PROBE_COUNT = 20
NO_IMPACT = ""
FIELDS = ("event_type", "is_new", "direction", "surprise", "horizon", "confidence")
KEY_FIELDS = ("event_type", "is_new", "direction", "surprise", "horizon")


@dataclass(frozen=True)
class KeyImpact:
    """One impact the answer key expects."""

    ticker: str
    event_type: str
    is_new: bool
    direction: int
    surprise: int
    horizon: str


@dataclass(frozen=True)
class Probe:
    """One fictional probe item, its kind, the item as sent and its written key."""

    probe_id: str
    kind: str
    item: RequestItem
    key: tuple[KeyImpact, ...]
    why: str


def _key_impact(probe_id: str, raw: Mapping[str, Any]) -> KeyImpact:
    """One key impact, refused unless every field is present and a level the schema allows."""
    impact = KeyImpact(*(raw.get(name) for name in ("ticker", "event_type", "is_new", "direction", "surprise",
                                                     "horizon")))
    if (impact.ticker not in TICKERS or impact.event_type not in EVENT_TYPES or type(impact.is_new) is not bool
            or type(impact.direction) is not int or impact.direction not in DIRECTIONS
            or type(impact.surprise) is not int or impact.surprise not in SURPRISES
            or impact.horizon not in HORIZONS):
        raise ValueError(f"probe {probe_id} has a key impact outside the schema")
    return impact


def load_probes(name: str = PROBE_FILE) -> tuple[Probe, ...]:
    """The probes in file order, refused unless there are 20 with ids p01 to p20, known kinds and valid keys."""
    raw = json.loads((llm_request.HERE / name).read_bytes())["probes"]
    probes = []
    for entry in raw:
        probe_id = entry["probe_id"]
        key = tuple(_key_impact(probe_id, k) for k in entry["key"])
        if entry["kind"] not in NEWS_KINDS:
            raise ValueError(f"probe {probe_id} has an unknown kind")
        if len({k.ticker for k in key}) < len(key):
            raise ValueError(f"probe {probe_id} names a ticker twice in its key")
        item = RequestItem(entry["source"], entry["filer"], entry["headline"], entry["text"])
        probes.append(Probe(probe_id, entry["kind"], item, key, entry["why"]))
    if [p.probe_id for p in probes] != [f"p{i:02d}" for i in range(1, PROBE_COUNT + 1)]:
        raise ValueError(f"{name} must hold {PROBE_COUNT} probes with ids p01 to p{PROBE_COUNT:02d} in order")
    return tuple(probes)


def probe_request(llm: LlmConfig, probes: Sequence[Probe]) -> Request:
    """The one Haiku request of a probe run, the items in file order."""
    return llm_request.build_request(llm_request.haiku_spec(llm), [p.item for p in probes])


def due(now_ms: int, last_run_at: str | None, probe_time_utc: str) -> bool:
    """Whether a probe run is due, once now has passed today's probe time in UTC and no run started since then."""
    hours, minutes = (int(part) for part in probe_time_utc.split(":"))
    moment = now_ms - now_ms % MS_PER_DAY + hours * MS_PER_HOUR + minutes * MS_PER_MINUTE
    return now_ms >= moment and (last_run_at is None or parse_iso(last_run_at) < moment)


def probe_rows(run_at: str, call_id: int, probes: Sequence[Probe], validation: Validation) -> list[dict[str, Any]]:
    """The llm_probes rows of one run, one per impact of each validly answered probe and an empty ticker row for a
    valid answer with no impacts. A probe without a valid answer gets no row."""
    rows = []
    for index, probe in enumerate(probes):
        impacts = validation.scored.get(llm_request.position_id(index))
        if impacts is None:
            continue
        base = {"run_at": run_at, "probe_id": probe.probe_id, "call_id": call_id}
        if not impacts:
            rows.append({**base, "ticker": NO_IMPACT, **dict.fromkeys(FIELDS)})
        for impact in impacts:
            rows.append({**base, "ticker": impact.ticker, "event_type": impact.event_type,
                         "is_new": int(impact.is_new), "direction": impact.direction, "surprise": impact.surprise,
                         "horizon": impact.horizon, "confidence": impact.confidence})
    return rows


def key_rows(probes: Sequence[Probe]) -> list[dict[str, Any]]:
    """The answer key in the shape of llm_probes rows, for agreement over KEY_FIELDS."""
    rows = []
    for probe in probes:
        if not probe.key:
            rows.append({"probe_id": probe.probe_id, "ticker": NO_IMPACT, **dict.fromkeys(KEY_FIELDS)})
        rows.extend({"probe_id": probe.probe_id, "ticker": k.ticker, "event_type": k.event_type,
                     "is_new": int(k.is_new), "direction": k.direction, "surprise": k.surprise,
                     "horizon": k.horizon} for k in probe.key)
    return rows


def agreement(first: Iterable[Mapping[str, Any]], second: Iterable[Mapping[str, Any]],
              fields: Sequence[str] = FIELDS) -> dict[str, float | int | None]:
    """Per field agreement of two sets of probe rows over the probes both answered. Each probe and ticker pair either
    names counts once, and a pair only one names disagrees on every field. tickers is the share of pairs both name.
    Shares are None when no probe was answered by both."""
    a = {(row["probe_id"], row["ticker"]): row for row in first}
    b = {(row["probe_id"], row["ticker"]): row for row in second}
    common = {probe for probe, _ in a} & {probe for probe, _ in b}
    pairs = [pair for pair in sorted(a.keys() | b.keys()) if pair[0] in common]
    both = [pair for pair in pairs if pair in a and pair in b]
    result: dict[str, float | int | None] = {"probes": len(common), "pairs": len(pairs)}
    for name in ("tickers", *fields):
        agreed = sum(1 for pair in both if name == "tickers" or a[pair][name] == b[pair][name])
        result[name] = agreed / len(pairs) if pairs else None
    return result
