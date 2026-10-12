"""Pure rules for one Finnhub print, from PLAN_V2 sections 4.2 and 4.3, shared by the candles, price health and the
tape check.

Classification reads the vendor time t through the session table, because it arrives in the message, while timing
never reads it. A print is eligible for price when it arrived at most late_print_seconds after its vendor time, at
most 2 seconds before it, and carries no excluded condition. A print whose vendor time is more than 1 second before
the stamp of the latest connected notice may be a replay by the new connection, so it counts for neither price nor
volume.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockFeedConfig
from btc_survivor.v2.sessions import Session

# Classes of a print by vendor time.
PRE, REGULAR, POST, OTHER = "pre", "regular", "post", "other"
# Price eligibility of a print.
ELIGIBLE, LATE, EARLY, EXCLUDED = "eligible", "late", "early", "excluded"

# Pre market runs from 04:00 ET to the 09:30 open and post market from the close to 20:00 ET, or to 17:00 after a
# 13:00 early close. New York changes its clocks at 02:00 on a Sunday, so these spans never cross a change.
PRE_MARKET_MS = 330 * MS_PER_MINUTE
POST_MARKET_MS = 240 * MS_PER_MINUTE
EARLY_LIMIT_MS = 2 * MS_PER_SECOND
REPLAY_LIMIT_MS = 1 * MS_PER_SECOND


@dataclass(frozen=True)
class PriceRules:
    """The two settings that judge price eligibility, the late limit in milliseconds and the excluded conditions."""

    late_ms: int
    excluded: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, feed: StockFeedConfig) -> PriceRules:
        return cls(round(feed.late_print_seconds * MS_PER_SECOND), frozenset(feed.price_excluded_conditions))


def phase_in(session: Session, ms: int) -> str:
    """The class of an instant on the New York date of one session, pre, regular, post or other."""
    if session.open_ms <= ms < session.close_ms:
        return REGULAR
    if session.open_ms - PRE_MARKET_MS <= ms < session.open_ms:
        return PRE
    if session.close_ms <= ms < session.close_ms + POST_MARKET_MS:
        return POST
    return OTHER


def classify(vendor_ms: int) -> str:
    """The class of a print by its vendor time. A weekend, a full closure or a time outside the session table is
    other, so a malformed vendor time can never stop the caller."""
    try:
        session = sessions.session_on(sessions.ny_date(vendor_ms))
    except (ValueError, OverflowError, OSError):
        return OTHER
    return OTHER if session is None else phase_in(session, vendor_ms)


def in_session(session: Session, recv_ms: int, vendor_ms: int) -> bool:
    """Whether a print belongs to a candle of this session, which needs its receive stamp and its vendor time both in
    the regular hours."""
    return session.open_ms <= recv_ms < session.close_ms and session.open_ms <= vendor_ms < session.close_ms


def age_ms(recv_ms: int, vendor_ms: int) -> int:
    """The age of a print on arrival, its receive stamp minus its vendor time, negative when it seems early."""
    return recv_ms - vendor_ms


def eligibility(recv_ms: int, vendor_ms: int, conditions: Iterable[str], rules: PriceRules) -> str:
    """Eligible, or the first reason it is not, checking late, then early, then an excluded condition."""
    age = age_ms(recv_ms, vendor_ms)
    if age > rules.late_ms:
        return LATE
    if age < -EARLY_LIMIT_MS:
        return EARLY
    if rules.excluded and any(code in rules.excluded for code in conditions):
        return EXCLUDED
    return ELIGIBLE


def is_replayed(vendor_ms: int, connected_ms: int | None) -> bool:
    """Whether the vendor time is more than 1 second before the stamp of the latest connected notice."""
    return connected_ms is not None and connected_ms - vendor_ms > REPLAY_LIMIT_MS
