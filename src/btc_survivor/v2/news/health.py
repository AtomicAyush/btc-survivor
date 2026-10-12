"""News health, channel outages and the news_heartbeat row of PLAN_V2 section 7.8.

News is healthy when FinBERT is on, no item routed to a scorer has waited more than score_wait_seconds for its
primary score, and the clock check is clean. The states of Haiku and Sonnet never count. A channel whose last
successful poll is older than outage_cadence_multiple times its cadence, or outage_floor_seconds when that is longer,
is in outage, which every heartbeat lists and which never touches news health. The cadence is the longer of the one
in force now and the one at that poll, as in the backlog rule, and a stretch in which a channel is dormant, such as
the FOMC channel between statement days, is never counted against it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, iso, parse_iso
from btc_survivor.v2.config import NewsConfig
from btc_survivor.v2.news import db
from btc_survivor.v2.news.schedule import Schedule

ON, OFF = "on", "off"
FINBERT_SCORER = "finbert_v1"      # the name the Phase 15 FinBERT scorer writes, not in the config
FINBERT_OFF = "finbert is not on"
SCORE_WAIT = "an item waited too long for its primary score"
CLOCK_SKEW = "clock skew"

# Items routed to a scorer that have no primary score yet. An llm item has one once its final Haiku row says scored
# or irrelevant, or once both a final Haiku row and a FinBERT row exist, and a finbert item once its FinBERT row
# exists, as the primary stream of section 8.7 reads them. Items at or below the floor all had one already.
_WAITING = """
SELECT count(*), min(i.seq), min(i.received_at) FROM news_items i
WHERE i.seq > :floor AND i.route != 'none'
  AND NOT EXISTS (SELECT 1 FROM item_scores h WHERE h.item_seq = i.seq AND h.scorer = :haiku AND i.route = 'llm'
                  AND h.status IN ('scored', 'irrelevant'))
  AND NOT (EXISTS (SELECT 1 FROM item_scores f WHERE f.item_seq = i.seq AND f.scorer = :finbert)
           AND (i.route = 'finbert'
                OR EXISTS (SELECT 1 FROM item_scores h WHERE h.item_seq = i.seq AND h.scorer = :haiku)))
"""


@dataclass(frozen=True)
class Scorers:
    """The state words a heartbeat carries for the three scorers, all off until Phase 15 builds them."""

    haiku: str = OFF
    sonnet: str = OFF
    finbert: str = OFF


@dataclass(frozen=True)
class Spend:
    """The key ledger's spend for the UTC day and month and the daily alert, all zero until Phase 15."""

    day_usd: float = 0.0
    month_usd: float = 0.0
    alert: bool = False


@dataclass(frozen=True)
class Waiting:
    """How many routed items wait for their primary score, and for how long the oldest has waited."""

    count: int
    oldest_wait_s: float | None


def assess(finbert: str, oldest_wait_s: float | None, score_wait_s: float, clock_clean: bool) -> str | None:
    """Why news is unhealthy, each failed condition named once, or None when it is healthy. A wait of exactly
    score_wait_s still passes."""
    reasons = []
    if finbert != ON:
        reasons.append(FINBERT_OFF)
    if oldest_wait_s is not None and oldest_wait_s > score_wait_s:
        reasons.append(SCORE_WAIT)
    if not clock_clean:
        reasons.append(CLOCK_SKEW)
    return ", ".join(reasons) or None


def dormant_ms(schedule: Schedule, channel: str, last_ms: int, last_cadence_s: int) -> int:
    """How long a channel was dormant after a poll at last_ms before it next came due, 0 for a channel that never
    sleeps. The FOMC channel polled last in one statement window is dormant until the next."""
    expected = schedule.next_due(channel, last_ms, last_ms)
    if expected is None or schedule.cadence(channel, expected - 1) is not None:
        return 0
    return max(0, expected - last_ms - last_cadence_s * MS_PER_SECOND)


def active_gap_ms(schedule: Schedule, channel: str, last_ms: int, last_cadence_s: int, now_ms: int) -> int:
    """The time from a poll at last_ms to now_ms, less the dormant stretch after it, which the backlog and outage rules
    both measure."""
    return now_ms - last_ms - dormant_ms(schedule, channel, last_ms, last_cadence_s)


def _limit_ms(cadence_s: int, last_cadence_s: int, multiple: float, floor_s: float) -> float:
    return max(multiple * max(cadence_s, last_cadence_s), floor_s) * MS_PER_SECOND


def in_outage(schedule: Schedule, channel: str, last_ms: int, last_cadence_s: int, now_ms: int, multiple: float,
              floor_s: float) -> bool:
    """Whether a channel whose last successful poll began at last_ms is in outage at now_ms. A dormant channel never
    is."""
    cadence = schedule.cadence(channel, now_ms)
    if cadence is None:
        return False
    gap = active_gap_ms(schedule, channel, last_ms, last_cadence_s, now_ms)
    return gap > _limit_ms(cadence, last_cadence_s, multiple, floor_s)


def outage_minutes(schedule: Schedule, channel: str, last_ms: int, last_cadence_s: int, start_ms: int, end_ms: int,
                   multiple: float, floor_s: float) -> list[int]:
    """The minute boundaries in [start_ms, end_ms) at which the channel is in outage, with no successful poll after
    the one at last_ms. It applies in_outage with the dormant stretch worked out once."""
    dormant = dormant_ms(schedule, channel, last_ms, last_cadence_s)
    first = -(-start_ms // MS_PER_MINUTE) * MS_PER_MINUTE
    minutes = []
    for at in range(first, end_ms, MS_PER_MINUTE):
        cadence = schedule.cadence(channel, at)
        if cadence is not None and at - last_ms - dormant > _limit_ms(cadence, last_cadence_s, multiple, floor_s):
            minutes.append(at)
    return minutes


def channels_down(schedule: Schedule, last_ok: Mapping[str, tuple[int, int] | None], now_ms: int, multiple: float,
                  floor_s: float, since_ms: int) -> tuple[str, ...]:
    """The channels in outage at now_ms, in schedule order. last_ok maps a channel to the start and cadence of its
    last successful poll, and a channel without one counts from since_ms, the start of the process."""
    down = []
    for channel in schedule.channels:
        ok = last_ok.get(channel.name)
        last_ms, cadence = ok if ok is not None else (since_ms, schedule.cadence(channel.name, now_ms) or 0)
        if in_outage(schedule, channel.name, last_ms, cadence, now_ms, multiple, floor_s):
            down.append(channel.name)
    return tuple(down)


class Heartbeat:
    """Writes the news_heartbeat rows of one news process. It remembers the seq at or below which every routed item
    has its primary score, so each heartbeat reads only the newer items."""

    def __init__(self, config: NewsConfig, conn: sqlite3.Connection, schedule: Schedule) -> None:
        self.config = config
        self.conn = conn
        self.schedule = schedule
        self._floor = 0

    def waiting(self, now_ms: int) -> Waiting:
        """The routed items still waiting for a primary score at now_ms."""
        count, first_seq, first_at = self.conn.execute(_WAITING, {
            "floor": self._floor, "haiku": self.config.llm.scorer_name, "finbert": FINBERT_SCORER}).fetchone()
        if count == 0:
            self._floor = self.conn.execute("SELECT coalesce(max(seq), 0) FROM news_items").fetchone()[0]
            return Waiting(0, None)
        self._floor = first_seq - 1
        return Waiting(count, max(0, now_ms - parse_iso(first_at)) / MS_PER_SECOND)

    def write(self, now_ms: int, clock_clean: bool, last_ok: Mapping[str, tuple[int, int] | None], since_ms: int,
              scorers: Scorers = Scorers(), spend: Spend = Spend()) -> dict:
        """Insert one heartbeat row for now_ms and return it. Channels in outage are listed and never make news
        unhealthy."""
        news, llm = self.config.news, self.config.llm
        waiting = self.waiting(now_ms)
        reason = assess(scorers.finbert, waiting.oldest_wait_s, news.score_wait_seconds, clock_clean)
        down = channels_down(self.schedule, last_ok, now_ms, news.outage_cadence_multiple, news.outage_floor_seconds,
                             since_ms)
        row = {
            "ts": iso(now_ms), "healthy": int(reason is None), "reason": reason,
            "config_hash": self.config.config_hash, "prompt_sha256": llm.prompt_sha256,
            "schema_sha256": llm.schema_sha256, "haiku_state": scorers.haiku, "sonnet_state": scorers.sonnet,
            "finbert_state": scorers.finbert, "waiting": waiting.count, "oldest_wait_s": waiting.oldest_wait_s,
            "channels_down": ",".join(down), "spent_day_usd": spend.day_usd, "spent_month_usd": spend.month_usd,
            "spend_alert": int(spend.alert),
        }
        db.insert(self.conn, "news_heartbeat", row)
        return row
