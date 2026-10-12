"""The key ledger, the cost formula and the one monthly cap of PLAN_V2 section 8.6, with the daily spend alert.

Every process that may call either model with the key reserves against one SQLite ledger per key, written only under
an exclusive flock on the lock file beside it, so two processes can never pass the cap together. A call goes only when
the UTC month's spend plus its reservation stays at or under monthly_usd. Spend counts the cost of every call that
returned usage and the reservation of every other call, whether running, failed after sending or lost. There is no
daily cap of any kind, only an alert once a UTC day's spend passes daily_alert_share of the monthly cap, which stops
nothing. Dollars are worked out exactly and rounded once, and spend is an exactly rounded sum, so no rounding step lets
a cost pass its reservation or the spend pass the cap.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import logging
import math
import os
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from btc_survivor.clock import MS_PER_DAY, MS_PER_SECOND, iso
from btc_survivor.v2.config import HAIKU_MODEL, SONNET_MODEL, LlmPricesConfig, NewsConfig
from btc_survivor.v2.news import db
from btc_survivor.v2.news.health import Spend

log = logging.getLogger(__name__)

MILLION = 1_000_000
MODELS = (HAIKU_MODEL, SONNET_MODEL)
STARTED, DONE, UNKNOWN, LOST = "started", "done", "unknown", "lost"
USAGE, RESERVED = "usage", "reserved"
SPEND_ALERT = "spend_alert"

SCHEMA = """
CREATE TABLE reservations (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL,
  process TEXT NOT NULL,
  model TEXT NOT NULL, ref TEXT NOT NULL, n_items INTEGER NOT NULL,
  reserved_usd REAL NOT NULL, cost_usd REAL, cost_basis TEXT,
  status TEXT NOT NULL);
CREATE INDEX reservations_ts ON reservations (ts);
"""
# The spend of one row, its cost once it returned usage or failed, and its reservation while it runs.
_SPENT = "SELECT coalesce(cost_usd, reserved_usd) FROM reservations WHERE ts >= ? AND ts < ?"


class LedgerError(RuntimeError):
    """Raised for a file that is not a key ledger, a model other than the two chosen, or a reservation that is not
    running when it finishes."""


@dataclass(frozen=True)
class Rates:
    """One model's prices in dollars per million tokens, with the multipliers of cache writes and cache reads."""

    in_rate: float
    out_rate: float
    write: float = 1.0
    read: float = 1.0

    @property
    def input_bound(self) -> float:
        """The highest multiplier any input token can meet, which the reservation prices every input token at."""
        return max(1.0, self.write, self.read)


@dataclass(frozen=True)
class Usage:
    """The token counts a response reports."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _dollars(terms: Iterable[tuple[int, float, float]]) -> float:
    """The sum of tokens times rate times multiplier over a million, exact until one rounding to the nearest float,
    which never changes the order of two amounts."""
    total = sum((Fraction(n) * Fraction(rate) * Fraction(mult) for n, rate, mult in terms), Fraction(0))
    return float(total / MILLION)


def reserve_usd(rates: Rates, counted: int, max_tokens: int) -> float:
    """What a call can cost at most, every counted input token at the highest input rate it can meet and all of
    max_tokens at the output rate."""
    if counted < 0 or max_tokens < 1:
        raise ValueError("a reservation needs a count of at least 0 and max_tokens of at least 1")
    return _dollars([(counted, rates.in_rate, rates.input_bound), (max_tokens, rates.out_rate, 1.0)])


def cost_usd(rates: Rates, usage: Usage) -> float:
    """What a call cost from the usage it reported, cache writes and reads at their multipliers."""
    counts = (usage.input_tokens, usage.output_tokens, usage.cache_creation_input_tokens, usage.cache_read_input_tokens)
    if min(counts) < 0:
        raise ValueError("token counts are never negative")
    return _dollars([(usage.input_tokens, rates.in_rate, 1.0),
                     (usage.cache_creation_input_tokens, rates.in_rate, rates.write),
                     (usage.cache_read_input_tokens, rates.in_rate, rates.read),
                     (usage.output_tokens, rates.out_rate, 1.0)])


def request_bytes(request: Mapping[str, Any]) -> int:
    """The length of the request as compact ASCII JSON, at least the UTF 8 bytes of its system prompt, schema and
    messages, since an escape is never shorter than the bytes it stands for."""
    return len(json.dumps(request, sort_keys=True, separators=(",", ":")))


def byte_bound(request: Mapping[str, Any], overhead_tokens: int) -> int:
    """The count to reserve when count_tokens fails or does not take the full request, which holds because every
    token covers at least one byte of text."""
    return request_bytes(request) + overhead_tokens


@dataclass(frozen=True)
class Prices:
    """The rates of [llm.prices]. Sonnet sends no cache control and its cache prices are not in the research, so
    every Sonnet input token prices at its input rate."""

    haiku: Rates
    sonnet: Rates
    long_prompt_threshold_tokens: int

    @classmethod
    def from_config(cls, prices: LlmPricesConfig) -> Prices:
        haiku = Rates(prices.haiku_input_per_mtok, prices.haiku_output_per_mtok, prices.haiku_cache_write_multiplier,
                      prices.haiku_cache_read_multiplier)
        sonnet = Rates(prices.sonnet_input_per_mtok, prices.sonnet_output_per_mtok)
        return cls(haiku, sonnet, prices.long_prompt_threshold_tokens)

    def rates(self, model: str) -> Rates:
        """The rates of one of the two chosen models."""
        if model == HAIKU_MODEL:
            return self.haiku
        if model == SONNET_MODEL:
            return self.sonnet
        raise LedgerError(f"no prices for {model!r}, only {', '.join(MODELS)} are priced")

    def reserve(self, model: str, counted: int, max_tokens: int) -> float:
        """The reservation of one call, refused above the token count where the prices of the table stop applying."""
        if counted > self.long_prompt_threshold_tokens:
            raise LedgerError(f"a count of {counted} tokens passes the {self.long_prompt_threshold_tokens} token "
                              f"line where prices rise, so the call is refused")
        return reserve_usd(self.rates(model), counted, max_tokens)

    def cost(self, model: str, usage: Usage) -> float:
        """The cost of one call from the usage it reported."""
        return cost_usd(self.rates(model), usage)

    def table(self) -> str:
        """The rates as canonical JSON, which every llm_calls row stores in price_table."""
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))


def _ms(moment: datetime) -> int:
    return int(moment.timestamp()) * MS_PER_SECOND


def month_bounds(now_ms: int) -> tuple[int, int]:
    """The first instant of the UTC month holding now_ms and of the month after it."""
    now = datetime.fromtimestamp(now_ms // MS_PER_SECOND, tz=timezone.utc)
    start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    end = datetime(now.year + now.month // 12, now.month % 12 + 1, 1, tzinfo=timezone.utc)
    return _ms(start), _ms(end)


def day_bounds(now_ms: int) -> tuple[int, int]:
    """The first instant of the UTC day holding now_ms and of the day after it."""
    start = now_ms - now_ms % MS_PER_DAY
    return start, start + MS_PER_DAY


def cap_until(now_ms: int) -> int:
    """00:00 UTC on the first of the next month, until which both models stay off once the cap refuses a call."""
    return month_bounds(now_ms)[1]


def process_name(config: NewsConfig) -> str:
    """The process column of a news process, its config hash and data folder. A restart recovers the rows of its data
    folder, whatever hash wrote them."""
    return f"{config.config_hash} {config.data_dir}"


def _folder(process: str) -> str:
    return process.partition(" ")[2] or process


def make_ref(scorer: str, item_seqs: Iterable[int]) -> str:
    """The ref of a call, its scorer and item seqs, which recover hands back so a lost call's items are marked
    failed."""
    return f"{scorer}:{','.join(str(seq) for seq in item_seqs)}"


def parse_ref(ref: str) -> tuple[str, tuple[int, ...]]:
    """The scorer and item seqs of a ref that make_ref wrote."""
    scorer, _, seqs = ref.partition(":")
    return scorer, tuple(int(seq) for seq in seqs.split(",") if seq)


@dataclass(frozen=True)
class Reservation:
    """One running call's row in the ledger, as reserve booked it."""

    id: int
    model: str
    ref: str
    n_items: int
    reserved_usd: float


def _row(row: sqlite3.Row) -> Reservation:
    return Reservation(row["id"], row["model"], row["ref"], row["n_items"], row["reserved_usd"])


def _create_file(path: Path) -> None:
    """Create path for the owner alone in a folder for the owner alone, leaving an existing file as it is."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        pass


class Ledger:
    """One key ledger. Every read and write runs in one transaction under the exclusive lock, so the cap check and
    the reservation it admits are one step for every process using the key."""

    def __init__(self, path: Path, lock_path: Path, monthly_usd: float, process: str) -> None:
        if not 0 < monthly_usd < math.inf:
            raise ValueError("monthly_usd must be above zero")
        self.path, self.lock_path = Path(path), Path(lock_path)
        self.monthly_usd = monthly_usd
        self.process = process
        _create_file(self.lock_path)
        _create_file(self.path)
        self._lock: int | None = os.open(self.lock_path, os.O_RDWR)
        self._conn: sqlite3.Connection | None = None
        try:
            self._conn = sqlite3.connect(self.path, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._open()
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_config(cls, config: NewsConfig, process: str | None = None) -> Ledger:
        """The key ledger of [llm.caps], for the news process by default."""
        caps = config.llm.caps
        return cls(config.path(caps.ledger_file), config.path(caps.ledger_lock_file), caps.monthly_usd,
                   process or process_name(config))

    def _open(self) -> None:
        """Create the schema in an empty file, refuse any other database, and use WAL so a commit is one append, synced
        in full so a reservation survives a power loss as well as a crash."""
        with self._flock(), db.transaction(self._conn) as conn:
            objects = _objects(conn)
            if not objects:
                for statement in filter(str.strip, SCHEMA.split(";")):
                    conn.execute(statement)
            elif objects != _expected():
                raise LedgerError(f"{self.path} is not a key ledger, it holds "
                                  f"{', '.join(f'{kind} {name}' for kind, name in sorted(objects))}")
        with self._flock():
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _flock(self) -> Iterator[None]:
        """Hold the exclusive lock on the lock file, waiting for any other process that holds it."""
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lock, fcntl.LOCK_UN)

    @contextmanager
    def _locked(self) -> Iterator[sqlite3.Connection]:
        """One immediate transaction under the exclusive lock, rolled back on any error."""
        with self._flock(), db.transaction(self._conn) as conn:
            yield conn

    @staticmethod
    def _spent(conn: sqlite3.Connection, bounds: tuple[int, int]) -> list[float]:
        return [row[0] for row in conn.execute(_SPENT, (iso(bounds[0]), iso(bounds[1])))]

    def reserve(self, model: str, ref: str, n_items: int, reserved_usd: float, now_ms: int) -> Reservation | None:
        """Book a running call at its reservation, or return None and book nothing when the month's spend plus the
        reservation would pass the cap. A reservation that reaches the cap exactly is admitted."""
        if model not in MODELS:
            raise LedgerError(f"{model!r} is not one of {', '.join(MODELS)}")
        if not 0 < reserved_usd < math.inf:
            raise ValueError("a reservation must be above zero")
        with self._locked() as conn:
            if math.fsum([*self._spent(conn, month_bounds(now_ms)), reserved_usd]) > self.monthly_usd:
                return None
            row_id = conn.execute(
                "INSERT INTO reservations (ts, process, model, ref, n_items, reserved_usd, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (iso(now_ms), self.process, model, ref, n_items, reserved_usd, STARTED)).lastrowid
        return Reservation(row_id, model, ref, n_items, reserved_usd)

    def finish(self, reservation: Reservation, cost: float | None) -> float:
        """Book a running call at the cost of the usage it returned, or at its reservation when it returned none, as
        after a timeout, a lost connection or an error, and return what was booked."""
        if cost is not None and not 0 <= cost < math.inf:
            raise ValueError("a cost must be zero or more")
        status, booked, basis = (UNKNOWN, reservation.reserved_usd, RESERVED) if cost is None else (DONE, cost, USAGE)
        with self._locked() as conn:
            changed = conn.execute(
                "UPDATE reservations SET status = ?, cost_usd = ?, cost_basis = ? WHERE id = ? AND status = ?",
                (status, booked, basis, reservation.id, STARTED)).rowcount
        if changed != 1:
            raise LedgerError(f"reservation {reservation.id} is not running")
        return booked

    def _running(self, conn: sqlite3.Connection) -> list[Reservation]:
        rows = conn.execute("SELECT * FROM reservations WHERE status = ? ORDER BY id", (STARTED,)).fetchall()
        return [_row(row) for row in rows if _folder(row["process"]) == _folder(self.process)]

    def running(self) -> list[Reservation]:
        """The running rows of this process's data folder, listed without changing them, so a restart can write their
        lost calls before recover marks them."""
        with self._locked() as conn:
            return self._running(conn)

    def recover(self) -> list[Reservation]:
        """At restart, mark the running rows of this process's data folder lost, booked at their reservation, and
        return them so their items are marked failed and never sent again."""
        with self._locked() as conn:
            mine = self._running(conn)
            conn.executemany(
                "UPDATE reservations SET status = ?, cost_usd = reserved_usd, cost_basis = ? WHERE id = ?",
                [(LOST, RESERVED, reservation.id) for reservation in mine])
        return mine

    def book(self, model: str, ref: str, n_items: int, spent_usd: float, now_ms: int) -> int:
        """Book money already spent elsewhere as one finished row, such as the cap drill's total from its throwaway
        ledger, so this ledger still holds every cent the key spent."""
        if model not in MODELS:
            raise LedgerError(f"{model!r} is not one of {', '.join(MODELS)}")
        if not 0 <= spent_usd < math.inf:
            raise ValueError("spent_usd must be zero or more")
        with self._locked() as conn:
            return conn.execute(
                "INSERT INTO reservations (ts, process, model, ref, n_items, reserved_usd, cost_usd, cost_basis, "
                "status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (iso(now_ms), self.process, model, ref, n_items, spent_usd, spent_usd, USAGE, DONE)).lastrowid

    def rows(self) -> list[Reservation]:
        """Every row of the ledger in id order, for the cap drill's check of its throwaway ledger."""
        with self._locked() as conn:
            return [_row(row) for row in conn.execute("SELECT * FROM reservations ORDER BY id")]

    def spend(self, now_ms: int) -> Spend:
        """The key's spend for the UTC day and month holding now_ms, counted as the cap counts it."""
        with self._locked() as conn:
            day = math.fsum(self._spent(conn, day_bounds(now_ms)))
            month = math.fsum(self._spent(conn, month_bounds(now_ms)))
        return Spend(day_usd=day, month_usd=month)


def _objects(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Every table and index by type and name, with its SQL in single spaces."""
    rows = conn.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL")
    return {(row[0], row[1]): " ".join(row[2].split()) for row in rows}


def _expected() -> dict[tuple[str, str], str]:
    """The objects SCHEMA creates, built in memory."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA)
        return _objects(conn)
    finally:
        conn.close()


class SpendWatch:
    """The daily spend alert of one news process. Once the key's spend for a UTC day passes daily_alert_share of the
    monthly cap it logs one WARNING line and writes one spend_alert event to collector_events, and every heartbeat for
    the rest of that UTC day carries the alert. It stops nothing, and a restart on the same day writes no second
    event."""

    def __init__(self, ledger: Ledger, conn: sqlite3.Connection, daily_alert_share: float) -> None:
        if not 0 < daily_alert_share < 1:
            raise ValueError("daily_alert_share must lie between zero and one")
        self.ledger = ledger
        self.conn = conn
        self.threshold_usd = daily_alert_share * ledger.monthly_usd
        self._checked_day: int | None = None
        self._alert_day: int | None = None

    def check(self, now_ms: int) -> Spend:
        """The spend for the heartbeat at now_ms with its alert flag, raising the alert first when the day has just
        passed the threshold."""
        spend = self.ledger.spend(now_ms)
        day = day_bounds(now_ms)
        if self._checked_day != day[0]:
            self._checked_day = day[0]
            found = self.conn.execute("SELECT 1 FROM collector_events WHERE kind = ? AND ts >= ? AND ts < ? LIMIT 1",
                                      (SPEND_ALERT, iso(day[0]), iso(day[1]))).fetchone()
            if found is not None:
                self._alert_day = day[0]
        if self._alert_day != day[0] and spend.day_usd > self.threshold_usd:
            detail = (f"the key's spend for the UTC day, {spend.day_usd:.4f} dollars, passed the daily alert of "
                      f"{self.threshold_usd:.2f} dollars")
            db.insert(self.conn, "collector_events", {"ts": iso(now_ms), "kind": SPEND_ALERT, "detail": detail})
            log.warning("%s, which stops nothing", detail)
            self._alert_day = day[0]
        return dataclasses.replace(spend, alert=self._alert_day == day[0])
