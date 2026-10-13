"""News lines, the score streams and the two decayed news inputs of PLAN_V2 sections 6.3, 8.7 and 10.4.

The news puller turns each item_scores row, with its impacts and item fields, into one recorded line of kind news, and
each news_heartbeat row into one of kind news_heartbeat, never with headline text. A line's recorded stamp is when this
process received it, and a score counts from that receipt and never before its scored_at.

ScoreStreams applies the one rule of section 8.7 to lines in receipt order. The primary stream news takes an llm item's
Haiku score when its Haiku row says scored or irrelevant, visible at that row's receipt, and otherwise its FinBERT score
as a flagged fallback, visible at the later receipt of the two rows, and a finbert item's FinBERT score at its receipt,
so every routed item gets exactly one primary score and Sonnet never enters it. The streams haiku and finbert hold
their scorer's rows, and finbert_matched, sonnet and haiku_official only the items both of their scorers scored, each
visible at the later receipt of the two rows.

NewsInputs decays each stream into a company and a macro input per symbol. A weight halves every 120 session minutes
from scored_at, so a score after the close keeps full weight until the open, an impact older than 600 session minutes is
dropped, and one neutral prior item in each denominator makes a minute with no news exactly zero. A model impact counts
only when marked new, and the index symbol's company input takes the impacts of the index and of every company.
EngineInputs keeps one engine's two network inputs, zero in S1, and the inputs it stored at each push minute, so a
matured sample sees only news received before its record was pushed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from btc_survivor.clock import MS_PER_MINUTE, iso, parse_iso
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.news.health import FINBERT_SCORER
from btc_survivor.v2.news.prefilter import FINBERT, LLM, MACRO
from btc_survivor.v2.stocks.health import Heartbeat

# Recorded line kinds the group dispatches before the version 1 decode.
NEWS_LINE, HEARTBEAT_LINE = "news", "news_heartbeat"
# The scorer names of [llm] and [llm.sonnet] scorer_name in configs/v2/news.toml.
HAIKU_SCORER, SONNET_SCORER = "haiku_v1", "sonnet_v1"
SCORED_STATUSES = ("scored", "irrelevant")
# The streams of section 8.7, the primary one first.
NEWS, HAIKU, FINBERT_STREAM, FINBERT_MATCHED, SONNET, HAIKU_OFFICIAL = (
    "news", "haiku", "finbert", "finbert_matched", "sonnet", "haiku_official")
STREAMS = (NEWS, HAIKU, FINBERT_STREAM, FINBERT_MATCHED, SONNET, HAIKU_OFFICIAL)


class Scorers(NamedTuple):
    """The scorer names whose rows each stream reads."""

    haiku: str = HAIKU_SCORER
    sonnet: str = SONNET_SCORER
    finbert: str = FINBERT_SCORER


# News lines


@dataclass(frozen=True)
class Impact:
    """One score_impacts row as a line carries it, is_new None for FinBERT."""

    ticker: str
    strength: float
    is_new: bool | None = None

    @property
    def counted(self) -> bool:
        """A model impact counts only when marked new, and a FinBERT impact always."""
        return self.is_new is not False


@dataclass(frozen=True)
class ScoreLine:
    """One item_scores row with its item's source, kind and tickers and its impacts, received at recv_ms."""

    seq: int
    item_seq: int
    source: str
    kind: str
    route: str
    scorer: str
    status: str
    scored_ms: int
    tickers: tuple[str, ...]
    impacts: tuple[Impact, ...]
    recv_ms: int = 0

    @property
    def visible_ms(self) -> int:
        """When the line counts from, its receipt and never before scoring finished."""
        return max(self.recv_ms, self.scored_ms)

    @property
    def scored(self) -> bool:
        return self.status in SCORED_STATUSES

    @classmethod
    def from_rows(cls, score: Mapping[str, Any], item: Mapping[str, Any],
                  impacts: Sequence[Mapping[str, Any]]) -> ScoreLine:
        """The line for one item_scores row, its news_items row and its score_impacts rows, read by column name."""
        return cls(int(score["seq"]), int(score["item_seq"]), item["source"], item["kind"], score["route"],
                   score["scorer"], score["status"], parse_iso(score["scored_at"]),
                   tuple(t for t in item["tickers"].split(",") if t),
                   tuple(Impact(i["ticker"], float(i["strength"]), None if i["is_new"] is None else bool(i["is_new"]))
                         for i in impacts))

    def encode(self) -> str:
        """The recorded text of the line, without its receive stamp, which the recording keeps."""
        return json.dumps({
            "seq": self.seq, "item_seq": self.item_seq, "source": self.source, "kind": self.kind,
            "route": self.route, "scorer": self.scorer, "status": self.status, "scored_at": iso(self.scored_ms),
            "tickers": list(self.tickers),
            "impacts": [{"ticker": i.ticker, "strength": i.strength, "is_new": i.is_new} for i in self.impacts],
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def decode(cls, raw: str, recv_ms: int) -> ScoreLine:
        """The line from its recorded text and the stamp it was recorded with."""
        d = json.loads(raw)
        return cls(int(d["seq"]), int(d["item_seq"]), d["source"], d["kind"], d["route"], d["scorer"], d["status"],
                   parse_iso(d["scored_at"]), tuple(d["tickers"]),
                   tuple(Impact(i["ticker"], float(i["strength"]), i["is_new"]) for i in d["impacts"]), recv_ms)


@dataclass(frozen=True)
class HeartbeatLine:
    """One news_heartbeat row as the stock side reads it, received at recv_ms."""

    seq: int
    ts_ms: int
    healthy: bool
    reason: str | None
    config_hash: str
    prompt_sha256: str
    schema_sha256: str
    channels_down: tuple[str, ...]
    recv_ms: int = 0

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> HeartbeatLine:
        """The line for one news_heartbeat row, read by column name."""
        return cls(int(row["seq"]), parse_iso(row["ts"]), bool(row["healthy"]), row["reason"], row["config_hash"],
                   row["prompt_sha256"], row["schema_sha256"], tuple(c for c in row["channels_down"].split(",") if c))

    @property
    def hashes(self) -> tuple[str, str, str]:
        return self.config_hash, self.prompt_sha256, self.schema_sha256

    def encode(self) -> str:
        """The recorded text of the line, without its receive stamp."""
        return json.dumps({
            "seq": self.seq, "ts": iso(self.ts_ms), "healthy": self.healthy, "reason": self.reason,
            "config_hash": self.config_hash, "prompt_sha256": self.prompt_sha256,
            "schema_sha256": self.schema_sha256, "channels_down": list(self.channels_down),
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def decode(cls, raw: str, recv_ms: int) -> HeartbeatLine:
        """The line from its recorded text and the stamp it was recorded with."""
        d = json.loads(raw)
        return cls(int(d["seq"]), parse_iso(d["ts"]), bool(d["healthy"]), d["reason"], d["config_hash"],
                   d["prompt_sha256"], d["schema_sha256"], tuple(d["channels_down"]), recv_ms)

    def heartbeat(self, launch_hashes: tuple[str, str, str]) -> Heartbeat:
        """The line as price health's news_minute reads it, against the hashes the run recorded at launch."""
        return Heartbeat(self.recv_ms, self.healthy, self.hashes == tuple(launch_hashes))


# The streams


@dataclass(frozen=True)
class Score:
    """One score of one stream, the counted impacts of the line it uses, decaying from that line's scored_at and
    counting from visible_ms."""

    stream: str
    item_seq: int
    score_seq: int
    scorer: str
    scored_ms: int
    visible_ms: int
    impacts: tuple[Impact, ...]
    fallback: bool = False


def _score(stream: str, used: ScoreLine, *receipts: ScoreLine, fallback: bool = False) -> Score:
    """The score of used, visible at the later of the receipts."""
    return Score(stream, used.item_seq, used.seq, used.scorer, used.scored_ms,
                 max(line.visible_ms for line in receipts), tuple(i for i in used.impacts if i.counted), fallback)


class ScoreStreams:
    """Every stream's scores from the lines of each routed item, fed in receipt order. A line of an unrouted item, of
    an unknown scorer, or a second line for one item and scorer is ignored and counted, as the news database never
    holds one."""

    def __init__(self, scorers: Scorers = Scorers()) -> None:
        self.scorers = scorers
        self.items: dict[int, dict[str, ScoreLine]] = {}
        self.ignored = 0

    def add(self, line: ScoreLine) -> list[Score]:
        """The scores that this line completes, in stream order."""
        rows = self.items.get(line.item_seq, {})
        if line.scorer in rows or line.scorer not in self.scorers or line.route not in (LLM, FINBERT):
            self.ignored += 1
            return []
        rows[line.scorer] = line
        self.items[line.item_seq] = rows
        h, s, f = (rows.get(name) for name in self.scorers)
        haiku_ok = h is not None and h.scored
        out = self._primary(line, h, f)
        if line is h and haiku_ok:
            out.append(_score(HAIKU, h, h))
        if line is f:
            out.append(_score(FINBERT_STREAM, f, f))
        if haiku_ok and f is not None and line in (h, f):
            out.append(_score(FINBERT_MATCHED, f, h, f))
        if haiku_ok and s is not None and s.scored and line in (h, s):
            out.extend((_score(SONNET, s, h, s), _score(HAIKU_OFFICIAL, h, h, s)))
        return out

    @staticmethod
    def _primary(line: ScoreLine, h: ScoreLine | None, f: ScoreLine | None) -> list[Score]:
        """The primary score this line completes, if any. Sonnet never enters it."""
        if line.route == FINBERT:
            return [_score(NEWS, f, f)] if line is f else []
        if line is h and h.scored:
            return [_score(NEWS, h, h)]
        if h is not None and not h.scored and f is not None and line in (h, f):
            return [_score(NEWS, f, h, f, fallback=True)]
        return []


def derive_streams(lines: Sequence[ScoreLine], scorers: Scorers = Scorers()) -> dict[str, list[Score]]:
    """Every stream's scores from lines in receipt order, the one pure rule of section 8.7."""
    router = ScoreStreams(scorers)
    out: dict[str, list[Score]] = {stream: [] for stream in STREAMS}
    for line in lines:
        for score in router.add(line):
            out[score.stream].append(score)
    return out


# Decayed inputs


def session_minutes(ms: int) -> float:
    """Regular session minutes elapsed by the instant ms on the k scale, equal to k at each boundary and still from a
    close to the next open."""
    session = sessions.next_session(ms)
    return session.first_k - 1 + max(0, ms - session.open_ms) / MS_PER_MINUTE


class NewsInput(NamedTuple):
    """One symbol's two inputs from one stream at one boundary, the impacts behind each and the newest visible stamp
    among them."""

    company: float
    macro: float
    company_items: int
    macro_items: int
    newest_ms: int | None

    @property
    def score(self) -> float:
        """Company plus macro, the score predictor of the stream."""
        return self.company + self.macro


class NewsInputs:
    """S2's scores of every stream and the decayed inputs they give at each boundary, stepped in increasing order."""

    def __init__(self, symbols: Sequence[str], index_symbol: str, half_life_minutes: float = 120.0,
                 drop_after_half_lives: float = 5.0, scorers: Scorers = Scorers()) -> None:
        self.symbols = tuple(symbols)
        self.half_life = half_life_minutes
        self.drop_after = drop_after_half_lives * half_life_minutes
        self.streams = ScoreStreams(scorers)
        companies = tuple(s for s in self.symbols if s != index_symbol)
        # The company inputs each ticker feeds, its own and the index symbol's.
        self._targets = {s: (s,) for s in self.symbols}
        self._targets.update({c: (c, index_symbol) for c in companies})
        self._scores: dict[str, list[tuple[Score, float]]] = {stream: [] for stream in STREAMS}
        self._kept: dict[int, ScoreLine] = {}
        self._newest: dict[int, float] = {}
        self._last_ms: int | None = None

    @classmethod
    def from_config(cls, cfg: StockConfig, scorers: Scorers = Scorers()) -> NewsInputs:
        news = cfg.news_input
        return cls(cfg.stock_feed.symbols, cfg.stock_feed.index_symbol, news.half_life_session_minutes,
                   news.drop_after_half_lives, scorers)

    def add(self, line: ScoreLine) -> list[Score]:
        """Take one line in receipt order and return the scores it completes."""
        scores = self.streams.add(line)
        if self.streams.items.get(line.item_seq, {}).get(line.scorer) is line:
            self._kept[line.seq] = line
            at = session_minutes(line.scored_ms)
            self._newest[line.item_seq] = max(at, self._newest.get(line.item_seq, at))
        for score in scores:
            self._scores[score.stream].append((score, session_minutes(score.scored_ms)))
        return scores

    def kept_lines(self) -> list[ScoreLine]:
        """The lines that can still matter, in the order they were added, from which a resumed engine rebuilds this
        state exactly by adding them again."""
        return list(self._kept.values())

    def bears_on(self, symbol: str, tickers: Sequence[str]) -> bool:
        """Whether an item tagged with tickers reaches symbol's inputs, through its own ticker, MACRO, which reaches
        every symbol, or a company ticker, which also reaches the index symbol."""
        return MACRO in tickers or any(symbol in self._targets.get(ticker, ()) for ticker in tickers)

    def step(self, boundary_ms: int) -> dict[str, dict[str, NewsInput]]:
        """Every stream's inputs per symbol at boundary_ms, from scores visible before it, after dropping every score
        and item older than the drop age."""
        if self._last_ms is not None and boundary_ms <= self._last_ms:
            raise ValueError(f"news inputs step to {iso(boundary_ms)}, not after {iso(self._last_ms)}")
        self._last_ms = boundary_ms
        now = session_minutes(boundary_ms)
        self._prune(now)
        return {stream: self._inputs(self._scores[stream], boundary_ms, now) for stream in STREAMS}

    def _prune(self, now: float) -> None:
        """Forget scores, and items with every line, whose age from scored_at passed the drop age."""
        for stream, scores in self._scores.items():
            self._scores[stream] = [(score, at) for score, at in scores if now - at <= self.drop_after]
        old = [item for item, at in self._newest.items() if now - at > self.drop_after]
        for item in old:
            del self._newest[item]
            for line in self.streams.items.pop(item).values():
                self._kept.pop(line.seq, None)

    def _inputs(self, scores: list[tuple[Score, float]], boundary_ms: int, now: float) -> dict[str, NewsInput]:
        """The shrunk weighted means of strength per symbol, company and macro, over the visible scores."""
        company = {s: [0.0, 0.0, 0, None] for s in self.symbols}
        macro = [0.0, 0.0, 0, None]
        for score, at in scores:
            if score.visible_ms >= boundary_ms:
                continue
            weight = 2.0 ** (-(now - at) / self.half_life)
            for impact in score.impacts:
                targets = (macro,) if impact.ticker == MACRO else \
                    tuple(company[s] for s in self._targets.get(impact.ticker, ()))
                for acc in targets:
                    acc[0] += weight * impact.strength
                    acc[1] += weight
                    acc[2] += 1
                    acc[3] = score.visible_ms if acc[3] is None else max(acc[3], score.visible_ms)
        m = macro[0] / (macro[1] + 1.0)
        return {s: NewsInput(c[0] / (c[1] + 1.0), m, c[2], macro[2], _newest(c[3], macro[3]))
                for s, c in company.items()}


def _newest(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else max(a, b)


class EngineInputs:
    """One engine's two network inputs per symbol, computed by S2 from the primary stream and pinned at zero in S1,
    and the inputs stored at each push minute, which a matured sample joins instead of the current ones."""

    def __init__(self, symbols: Sequence[str], index_symbol: str, use_news: bool, horizon_minutes: int,
                 half_life_minutes: float = 120.0, drop_after_half_lives: float = 5.0,
                 scorers: Scorers = Scorers()) -> None:
        self.symbols = tuple(symbols)
        self.horizon = horizon_minutes
        self.news = NewsInputs(symbols, index_symbol, half_life_minutes, drop_after_half_lives, scorers) \
            if use_news else None
        self.streams: dict[str, dict[str, NewsInput]] = {}
        self.current = {s: np.zeros(2) for s in self.symbols}
        self._pushed: dict[tuple[int, str], np.ndarray] = {}

    @classmethod
    def from_config(cls, cfg: StockConfig, scorers: Scorers = Scorers()) -> EngineInputs:
        news = cfg.news_input
        return cls(cfg.stock_feed.symbols, cfg.stock_feed.index_symbol, news.use_news, cfg.time.horizon_minutes,
                   news.half_life_session_minutes, news.drop_after_half_lives, scorers)

    @property
    def use_news(self) -> bool:
        return self.news is not None

    def add(self, line: ScoreLine) -> list[Score]:
        """Take one news line in S2. S1 reads no news and keeps nothing."""
        return [] if self.news is None else self.news.add(line)

    def step(self, boundary_ms: int) -> dict[str, np.ndarray]:
        """The two inputs per symbol at boundary_ms, company then macro of the primary stream in S2 and zeros in S1,
        forgetting stored inputs whose records can no longer mature."""
        if self.news is not None:
            self.streams = self.news.step(boundary_ms)
            self.current = {s: np.array([i.company, i.macro]) for s, i in self.streams[NEWS].items()}
        k = sessions.k_at(boundary_ms)
        if k is not None:
            self._pushed = {key: v for key, v in self._pushed.items() if key[0] >= k - self.horizon}
        return self.current

    def push(self, k: int, symbol: str) -> np.ndarray:
        """Store the symbol's current inputs under the minute k of a label record pushed now, and return them."""
        stored = self.current[symbol].copy()
        self._pushed[(k, symbol)] = stored
        return stored

    def take(self, k: int, symbol: str) -> np.ndarray:
        """The inputs stored when the record pushed at k matured, for its training sample and linear_news update."""
        return self._pushed.pop((k, symbol))
