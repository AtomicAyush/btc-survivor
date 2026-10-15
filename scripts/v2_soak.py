"""Soak of the stock process by PLAN_V2 section 16 phase 19, long replays through the real stock process code that
watch memory, CPU per print, invariants and determinism.

Replay results are never used to choose, tune, or rank strategies. The soak only checks the plumbing. By default it
runs the rehearsal pair, configs/v2/rehearsal_s1.toml and rehearsal_s2.toml, whose 30 minute warmup and forced timer
exist only so that trading starts within a replayed session. That pair is never a strategy setting.

The input is every recording in the raw stocks folder joined end to end, followed by a seeded synthetic run of several
days with many reconnects, halts, late, early and replayed prints, news bursts and news outages, its news score and
heartbeat lines merged in stamp order as the live puller records them. The newest recording is copied up to its size
at the start and read only to its last complete line, so a recorder or stock process still writing it is never
disturbed. Where two recordings do not continue one connection, a close marker ends the open connection and the next
one gets new connection ids, as a live reconnect does. A recording of the recorder alone holds no news lines, so a
healthy heartbeat line every 30 s with the launch hashes is merged into it, and the engines trade on the real tape as
beside a healthy news process. The launch hashes are those of the first heartbeat line any recording holds, or the
soak's own without one. The synthetic run continues the news sequences of the recordings, carries the launch hashes,
and starts each symbol at its last reference price.

This process reads the recordings, generates the synthetic run and pipes every line to two child processes. Each
holds only the stock process, both runs' engines on one shared market layer, the recorder and the one second ticker,
on a clock that follows the recorded stamps, with the local only check passing at once. Feed lines pass the redacting
sink and news lines go straight to the process, as live. The first child is measured. At the end of every replayed
hour it stops reading while this process reads its memory from outside, because macOS only exposes peak memory to the
process itself. The second child replays the same lines into its own folder, and every table of both runs must equal
the first child's apart from runtime_status, whose rows carry the process's own memory, which is the determinism check.

The plan's rule is that memory at the end is no more than 5 MB above memory at the end of the first replayed hour. On
Linux, the system of the Pi, resident memory read from /proc decides it, strictly. On a Mac, which keeps freed pages
resident and returns them in bursts, the memory the allocators report in use decides it, with resident memory reported
beside it, and without an in use reading the memory result is inconclusive. Memory in use also grows while history
windows fill in the first sessions and while news is kept over a weekend, since decay runs in session minutes, so by
the owner's decision of 12 October 2026 a Mac passes as well when memory in use grew less than 1 MB over the final
replayed week, from one reading to another at the same point of the weekly cycle, even when it grew more than 5 MB in
all. A replay shorter than a week and an hour holds no final week, so there only the first hour rule decides, and the
default synthetic run of two weeks always gives the week figure. The invariants are
the engines' own checks at every equity snapshot, the books of every bot checked from memory every replayed hour, the
ledger identities on both finished databases, fills never before their decision and each on a print at or after its
due time, opens only inside the open window, no position kept past a close except by a halt, forecasts never using news
received after their minute, news lines never received before they were scored, S1 and S2 sharing every candle, label,
session, spread check, price z score and linear_prices row, and a recording holding every line once in stamp order. Everything is written into a new folder under data/v2/soak. By
default it prints only verdicts, counts, hashes and timings, and --detail adds the hourly readings and every check. The
exit code is 0 for passed, 1 for failed, 2 when the soak cannot run and 3 for inconclusive.

    python scripts/v2_soak.py [--s1 CONFIG --s2 CONFIG] [--raw data/v2/raw_stocks] [--skip-real] [--synthetic-days 14]
                              [--out data/v2/soak/NAME] [--detail]
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import heapq
import json
import logging
import os
import platform
import re
import resource
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from btc_survivor.account import INVARIANT_TOLERANCE
from btc_survivor.clock import MS_PER_DAY, MS_PER_HOUR, MS_PER_MINUTE, MS_PER_SECOND, LiveClock, iso, parse_iso
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN
from btc_survivor.recorder import read_recording, recording_files
from btc_survivor.replay import NEVER_FOR_SELECTION, ReplayError
from btc_survivor.store import StoreError
from btc_survivor.v2 import output, redact, sessions
from btc_survivor.v2.config import HAIKU_MODEL, SONNET_MODEL, ConfigError, StockConfig, load_run_pair
from btc_survivor.v2.localcheck import LocalCheck
from btc_survivor.v2.news.health import FINBERT_SCORER, SCORE_WAIT
from btc_survivor.v2.news.prefilter import FINBERT, LLM, MACRO
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks import process
from btc_survivor.v2.stocks import replay as stock_replay
from btc_survivor.v2.stocks.bot import OPENS, SESSION_FLATTEN, opens_allowed
from btc_survivor.v2.stocks.engine import DEATH, INVARIANT_VIOLATION
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter
from btc_survivor.v2.stocks.group import MESSAGES, NEWS_LINES, PRINTS, StockGroup
from btc_survivor.v2.stocks.news_inputs import (HAIKU_SCORER, HEARTBEAT_LINE, NEWS_LINE, SONNET_SCORER, HeartbeatLine,
                                                Impact, ScoreLine)
from btc_survivor.v2.stocks.runner import StockRecorder
from btc_survivor.v2.stocks.store import TABLES
from btc_survivor.v2.stocks.synthetic import (DISCONNECT, EARLY_PRINTS, HALT, LATE_PRINTS, PAUSE, PROCESS_KILL,
                                              PROCESS_START, REPLAYED_PRINTS, SILENCE, Fault, SyntheticTape)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs" / "v2"
DEFAULT_S1, DEFAULT_S2 = CONFIGS / "rehearsal_s1.toml", CONFIGS / "rehearsal_s2.toml"
SOAK_ROOT = ROOT / "data" / "v2" / "soak"
PLUMBING_LABEL = "plumbing configuration, never a strategy setting"
SOAK_SENTENCE = ("The soak runs the rehearsal pair, a plumbing configuration that is never a strategy setting, "
                 "and checks only memory, CPU, invariants and determinism.")
MAX_GROWTH_MB = 5.0
MAX_WEEK_GROWTH_MB = 1.0  # on a Mac, growth in use over the final replayed week below this passes whatever the total
SLOPE_HOURS = 24.0  # the memory slope is fitted over the last day of replayed time
WEEK_HOURS = 168.0  # the week over week growth compares the end with the reading this long before it
SOAK_TAG = 19  # the seed tag of the synthetic run, apart from the version 1 tags in seeds.py
SYNTHETIC_START = "2026-10-29T04:00:00.000Z"  # midnight in New York before Thursday 29 October, used without recordings
JOIN_GAP_MS = 2 * MS_PER_SECOND  # the synthetic run starts this long after the recordings end, like a quick reconnect
JOIN_REASON = "soak join, the next recording starts"
SOAK_HASHES = tuple(hashlib.sha256(f"soak {name}".encode()).hexdigest() for name in ("config", "prompt", "schema"))
COMPARED = tuple(t for t in TABLES if t != "runtime_status")  # runtime_status rows carry the process's own memory
SHARED_TABLES = ("stock_candles", "stock_labels", "sessions", "spread_checks", "price_z")
EXIT_CODES = {"passed": 0, "failed": 1, "inconclusive": 3}
CANNOT_RUN = 2
CHUNK = 1 << 18
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# Synthetic news, every time relative to its session's open.
HEARTBEAT_MS = 30 * MS_PER_SECOND
OUTAGE_AT, UNHEALTHY_AT, BURST_AT = 150, 330, 270  # 12:00, 15:00 and 14:00 in New York
UNHEALTHY_MINUTES = 3
OFFICIAL = (("8k", "edgar"), ("fed_monetary", "fed"), ("bls_release", "bls"))
# Synthetic feed faults, also relative to the open, each on a symbol in turn.
LONG_HALT_AT, SHORT_HALT_AT, SHORT_HALT_MINUTES = 90, 240, 5
LATE_AT, LATE_MINUTES, LATE_SHIFT_MS = 60, 3, 6 * MS_PER_SECOND
EARLY_AT, EARLY_MINUTES, EARLY_SHIFT_MS = 315, 2, 3 * MS_PER_SECOND

Line = tuple[str, str, int, int]
NEWS_KINDS = (NEWS_LINE, HEARTBEAT_LINE)


class SoakError(RuntimeError):
    """Raised when the soak cannot run, such as for a folder that is not empty."""


@dataclass(frozen=True)
class Synthetic:
    """The shape of the synthetic run, its length, print rate and how often each fault comes."""

    days: float = 14.0                      # two weeks, so the week over week growth is always reported
    prints_per_second: float = 1.0          # per symbol in the session
    reconnect_minutes: float = 120.0        # one connection fault about this often, each staling every symbol
    halt_minutes: float = 20.0              # the long halt of each session, past max_resume_gap_minutes
    news_minutes: float = 5.0               # one news item this often, day and night
    burst_items: int = 300                  # the news burst of each session, burst_items of the news config
    outage_minutes: float = 15.0            # the heartbeat outage of each session


# Choosing and joining the recordings


@dataclass(frozen=True)
class Recording:
    """One recording the soak replays, where it reads it from and why."""

    source: Path
    path: Path  # what the soak reads, the source itself or a frozen copy of its first bytes
    status: str


def copy_prefix(source: Path, target: Path) -> int:
    """Copy the bytes source holds now to target, reading it once and never writing it, and return how many."""
    size = left = source.stat().st_size
    with source.open("rb") as src, target.open("wb") as dst:
        while left > 0:
            chunk = src.read(min(CHUNK, left))
            if not chunk:
                break
            dst.write(chunk)
            left -= len(chunk)
    return size - left


def select_recordings(raw: Path, input_dir: Path) -> list[Recording]:
    """Every recording in raw in time order, the newest one copied first, since a writer may still be appending."""
    files = recording_files([raw]) if raw.is_dir() else []
    chosen = [Recording(path, path, "read in place") for path in files[:-1]]
    if files:
        input_dir.mkdir(parents=True, exist_ok=True)
        copy = input_dir / files[-1].name
        size = copy_prefix(files[-1], copy)
        chosen.append(Recording(files[-1], copy, f"newest, its first {size} bytes copied and read to the last "
                                                 "complete line"))
    return chosen


def recording_news(recordings: Sequence[Recording]) -> list[tuple[str, str, str] | None]:
    """Each recording's news hashes from its first heartbeat line, or None for a recording of the recorder alone,
    which holds no news lines."""
    out = []
    for recording in recordings:
        hashes = stock_replay.recorded_hashes(read_recording([recording.path]))
        out.append(None if hashes == stock_replay.NO_HASHES else hashes)
    return out


def launch_hashes(recordings: Sequence[Recording], found: Sequence[tuple[str, str, str] | None] | None = None
                  ) -> tuple[tuple[str, str, str], str]:
    """The news hashes of the first heartbeat line any recording holds, or the soak's own, and where they came
    from."""
    first = next((h for h in (recording_news(recordings) if found is None else found) if h is not None), None)
    return (first, "recorded") if first is not None else (SOAK_HASHES, "soak")


@dataclass
class JoinState:
    """Where the join of the recordings and the synthetic run stands, with the connection ids and news sequences it
    has seen."""

    segment: str = ""
    lines: list[int] = field(default_factory=list)  # lines taken from each source
    markers: int = 0  # close and open markers added at joins
    clamped: int = 0  # receive times raised to keep time from going backwards
    last_ms: int | None = None
    open_conn: int | None = None  # the joined id of a connection still open
    top_conn: int = 0
    offset: int = 0
    score_seq: int = 0  # the newest news sequences seen, which the synthetic run continues
    item_seq: int = 0
    heartbeat_seq: int = 0
    heartbeats_added: int = 0  # healthy heartbeat lines merged into recordings without news lines


Source = tuple[str, Callable[[JoinState], Iterable[Line]]]


def _markers(state: JoinState, kind: str, recv_ms: int, conn: int) -> Iterator[Line]:
    """At a source's first feed line, close the open connection unless the line continues it, renumber connection
    ids above every id used so far, and open a connection for a line that starts inside one."""
    if kind != OPEN and state.open_conn is not None and conn + state.offset == state.open_conn:
        return
    if state.open_conn is not None:
        state.markers += 1
        yield CLOSE, JOIN_REASON, state.last_ms, state.open_conn
        state.open_conn = None
    state.offset = state.top_conn
    if kind != OPEN:
        state.markers += 1
        state.open_conn = conn + state.offset
        yield OPEN, "", recv_ms, state.open_conn


def _note_news(state: JoinState, kind: str, raw: str) -> None:
    """Keep the newest score, item and heartbeat sequence, skipping a line that cannot be read."""
    try:
        d = json.loads(raw)
        if kind == NEWS_LINE:
            state.score_seq, state.item_seq = max(state.score_seq, int(d["seq"])), max(state.item_seq,
                                                                                        int(d["item_seq"]))
        else:
            state.heartbeat_seq = max(state.heartbeat_seq, int(d["seq"]))
    except (ValueError, KeyError, TypeError):
        pass


def join(sources: Sequence[Source], state: JoinState) -> Iterator[Line]:
    """Recordings end to end with news lines passed through as they are. A file that continues the open connection,
    like a daily rotation, follows directly, and every other one gets the markers of a reconnect."""
    for segment, make in sources:
        state.segment = segment
        state.lines.append(0)
        first = True
        for kind, raw, recv_ms, conn in make(state):
            if state.last_ms is not None and recv_ms < state.last_ms:
                recv_ms = state.last_ms
                state.clamped += 1
            if kind in NEWS_KINDS:
                _note_news(state, kind, raw)
            else:
                if first:
                    first = False
                    yield from _markers(state, kind, recv_ms, conn)
                conn += state.offset
                if kind == OPEN:
                    state.open_conn = conn
                elif kind == CLOSE and conn == state.open_conn:
                    state.open_conn = None
                state.top_conn = max(state.top_conn, conn)
            state.last_ms = recv_ms
            state.lines[-1] += 1
            yield kind, raw, recv_ms, conn


def with_heartbeats(lines: Iterable[Line], hashes: tuple[str, str, str], state: JoinState) -> Iterator[Line]:
    """A recording of the recorder with a healthy heartbeat line carrying hashes every 30 s from its first line to its
    last, each numbered after every heartbeat the join has seen, so news health holds on the real tape."""
    next_beat = None
    for line in lines:
        t = line[2]
        if next_beat is None:
            next_beat = t - t % HEARTBEAT_MS + HEARTBEAT_MS
        while next_beat <= t:
            state.heartbeats_added += 1
            beat = HeartbeatLine(state.heartbeat_seq + 1, next_beat - 400, True, None, *hashes, ())
            yield HEARTBEAT_LINE, beat.encode(), next_beat, process.NEWS_CONN
            next_beat += HEARTBEAT_MS
        yield line


# The synthetic run


@dataclass
class Seqs:
    """The last news sequences used, so every synthetic line comes after the recordings' lines."""

    score: int = 0
    item: int = 0
    heartbeat: int = 0


def session_days(start_ms: int, end_ms: int) -> list[Session]:
    """The sessions that overlap start_ms to end_ms."""
    day, last, out = sessions.ny_date(start_ms), sessions.ny_date(end_ms), []
    while day <= last:
        session = sessions.session_on(day)
        if session is not None and session.close_ms > start_ms and session.open_ms < end_ms:
            out.append(session)
        day += timedelta(days=1)
    return out


def _after_open(session: Session, minutes: float) -> int:
    """The UTC milliseconds a number of minutes after a session's open."""
    return session.open_ms + round(minutes * MS_PER_MINUTE)


def feed_faults(start_ms: int, end_ms: int, symbols: Sequence[str], plan: Synthetic,
                rng: np.random.Generator) -> tuple[Fault, ...]:
    """A connection fault about every reconnect_minutes, cycling through a dropped connection, a silence of 8 s, a
    pause of 12 s, a drop with two refused handshakes and a drop whose next connection replays 30 s of prints, and in
    each session a long and a short halt, late prints and early prints, each on a symbol in turn."""
    every = round(plan.reconnect_minutes * MS_PER_MINUTE)
    faults: list[Fault] = []
    for n, base in enumerate(range(start_ms + every, end_ms, every)):
        at = base + int(rng.uniform(-0.25, 0.25) * every)
        faults += [(Fault(DISCONNECT, at),), (Fault(SILENCE, at, 8 * MS_PER_SECOND),),
                   (Fault(PAUSE, at, 12 * MS_PER_SECOND),), (Fault(DISCONNECT, at, refused=2),),
                   (Fault(REPLAYED_PRINTS, at, 30 * MS_PER_SECOND), Fault(DISCONNECT, at))][n % 5]
    n = len(symbols)
    for i, session in enumerate(session_days(start_ms, end_ms)):
        faults += [Fault(HALT, _after_open(session, LONG_HALT_AT), round(plan.halt_minutes * MS_PER_MINUTE),
                         symbol=symbols[i % n]),
                   Fault(HALT, _after_open(session, SHORT_HALT_AT), SHORT_HALT_MINUTES * MS_PER_MINUTE,
                         symbol=symbols[(i + 3) % n]),
                   Fault(LATE_PRINTS, _after_open(session, LATE_AT), LATE_MINUTES * MS_PER_MINUTE,
                         symbol=symbols[(i + 1) % n], shift_ms=LATE_SHIFT_MS),
                   Fault(EARLY_PRINTS, _after_open(session, EARLY_AT), EARLY_MINUTES * MS_PER_MINUTE,
                         symbol=symbols[(i + 2) % n], shift_ms=EARLY_SHIFT_MS)]
    return tuple(sorted((f for f in faults if start_ms <= f.at_ms < end_ms), key=lambda f: f.at_ms))


def _lines(item: int, kind: str, source: str, ticker: str, receipts: Sequence[tuple[str, int, str]],
           rng: np.random.Generator) -> list[tuple[int, ScoreLine]]:
    """One item's score lines with their receipts, each receipt (scorer, receive time, status), routed llm when a
    model scores it and finbert otherwise."""
    route = LLM if len(receipts) > 1 else FINBERT
    out = []
    for scorer, recv, status in receipts:
        strength = round(float(rng.uniform(-1.0, 1.0)), 3)
        impacts = (Impact(ticker, strength, None if scorer == FINBERT_SCORER else True),) if status == "scored" else ()
        out.append((recv, ScoreLine(0, item, source, kind, route, scorer, status, recv - 500, (ticker,), impacts)))
    return out


def _item(item: int, t: int, symbols: Sequence[str], rng: np.random.Generator) -> list[tuple[int, ScoreLine]]:
    """The lines of an item found at t, FinBERT's, then Haiku's and for an official item Sonnet's. Every fifth Haiku
    row failed, so FinBERT is the fallback, and every eleventh Finnhub item goes to FinBERT alone."""
    tickers = (*symbols, MACRO)
    kind, source = OFFICIAL[item % len(OFFICIAL)] if item % 7 == 0 else ("media", "finnhub")
    receipts = [(FINBERT_SCORER, t + 2 * MS_PER_SECOND, "scored")]
    if kind != "media" or item % 11:
        receipts.append((HAIKU_SCORER, t + 8 * MS_PER_SECOND, "failed" if item % 5 == 0 else "scored"))
        if kind != "media":
            receipts.append((SONNET_SCORER, t + 15 * MS_PER_SECOND, "scored"))
    return _lines(item, kind, source, tickers[item % len(tickers)], receipts, rng)


def _burst_item(item: int, i: int, burst_ms: int, symbols: Sequence[str],
                rng: np.random.Generator) -> list[tuple[int, ScoreLine]]:
    """Item i of a burst, FinBERT scoring one every 150 ms and Haiku taking calls of 20 every 6 s after a minute,
    with every tenth Haiku row failed."""
    receipts = [(FINBERT_SCORER, burst_ms + i * 150, "scored"),
                (HAIKU_SCORER, burst_ms + 60 * MS_PER_SECOND + (i // 20) * 6 * MS_PER_SECOND,
                 "failed" if i % 10 == 9 else "scored")]
    return _lines(item, "media", "finnhub", symbols[i % len(symbols)], receipts, rng)


def news_lines(start_ms: int, end_ms: int, symbols: Sequence[str], hashes: tuple[str, str, str], seqs: Seqs,
               plan: Synthetic, rng: np.random.Generator) -> list[Line]:
    """Heartbeat lines every 30 s, none during each session's outage and unhealthy for 3 minutes each afternoon, and
    the score lines of an item every news_minutes plus each session's burst, with sequences in receipt order."""
    days = session_days(start_ms, end_ms)
    outages = [(_after_open(s, OUTAGE_AT), _after_open(s, OUTAGE_AT + plan.outage_minutes)) for s in days]
    unhealthy = [(_after_open(s, UNHEALTHY_AT), _after_open(s, UNHEALTHY_AT + UNHEALTHY_MINUTES)) for s in days]
    beats = []
    for t in range(start_ms - start_ms % HEARTBEAT_MS + HEARTBEAT_MS, end_ms, HEARTBEAT_MS):
        if any(a <= t < b for a, b in outages):
            continue
        bad = any(a <= t < b for a, b in unhealthy)
        seqs.heartbeat += 1
        beat = HeartbeatLine(seqs.heartbeat, t - 400, not bad, SCORE_WAIT if bad else None, *hashes,
                             ("bls",) if bad else ())
        beats.append((HEARTBEAT_LINE, beat.encode(), t, process.NEWS_CONN))
    scored: list[tuple[int, ScoreLine]] = []
    every = round(plan.news_minutes * MS_PER_MINUTE)
    for t in range(start_ms + every, end_ms, every):
        seqs.item += 1
        scored += _item(seqs.item, t, symbols, rng)
    for session in days:
        burst = _after_open(session, BURST_AT)
        for i in range(plan.burst_items if start_ms <= burst < end_ms else 0):
            seqs.item += 1
            scored += _burst_item(seqs.item, i, burst, symbols, rng)
    scored = [(recv, line) for recv, line in sorted(scored, key=lambda pair: pair[0]) if recv < end_ms]
    lines = []
    for recv, line in scored:
        seqs.score += 1
        lines.append((NEWS_LINE, replace(line, seq=seqs.score).encode(), recv, process.NEWS_CONN))
    return list(heapq.merge(beats, lines, key=lambda line: line[2]))


def synthetic_lines(master_seed: int, symbols: Sequence[str], start_ms: int, plan: Synthetic,
                    hashes: tuple[str, str, str], seqs: Seqs, prices: dict[str, float | None]) -> Iterator[Line]:
    """The synthetic run from start_ms, the feed of a seeded market and recorder through its faults merged in stamp
    order with its news lines, without the recorder's control lines, each symbol starting at its given price."""
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, SOAK_TAG]))
    end_ms = start_ms + round(plan.days * MS_PER_DAY)
    tape = SyntheticTape(int(rng.integers(2**62)), symbols, regular_rate=plan.prints_per_second)
    tape.prices.update({s: float(p) for s, p in prices.items() if p is not None and s in tape.prices})
    faults = feed_faults(start_ms, end_ms, symbols, plan, rng)
    feed = (line for line in tape.lines(start_ms, end_ms, faults) if line[0] not in (PROCESS_START, PROCESS_KILL))
    return heapq.merge(feed, news_lines(start_ms, end_ms, symbols, hashes, seqs, plan, rng),
                       key=lambda line: line[2])


# Memory, read from outside the replay process and from its allocators


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime")]


def process_memory(pid: int) -> dict:
    """Current resident memory of a child process in MB, its physical footprint on macOS, and where they came
    from."""
    try:
        with open(f"/proc/{pid}/statm", encoding="ascii") as fh:
            rss = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        return {"rss_mb": rss, "footprint_mb": None, "source": "proc"}
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
            info = _RUsageInfoV0()
            if lib.proc_pid_rusage(pid, 0, ctypes.byref(info)) == 0:
                return {"rss_mb": info.ri_resident_size / 2**20, "footprint_mb": info.ri_phys_footprint / 2**20,
                        "source": "proc_pid_rusage"}
        except OSError:
            pass
    found = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return {"rss_mb": int(found) / 1024 if found else None, "footprint_mb": None, "source": "ps"}


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB, which the system reports in bytes on macOS and in KB on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


class _MallocStatistics(ctypes.Structure):  # macOS malloc_statistics_t
    _fields_ = [("blocks_in_use", ctypes.c_uint), ("size_in_use", ctypes.c_size_t),
                ("max_size_in_use", ctypes.c_size_t), ("size_allocated", ctypes.c_size_t)]


class _MallInfo2(ctypes.Structure):  # glibc struct mallinfo2
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def malloc_in_use_mb() -> float | None:
    """Bytes the C allocator has handed out and not taken back, in MB, or None where it cannot be read. Freed memory
    the allocator still holds is left out, so this separates memory in use from memory retained."""
    try:
        libc = ctypes.CDLL(None)
        if sys.platform == "darwin":
            stats = _MallocStatistics()
            fn = libc.malloc_zone_statistics
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.POINTER(_MallocStatistics)], None
            fn(None, ctypes.byref(stats))
            return stats.size_in_use / 2**20
        fn = libc.mallinfo2
        fn.argtypes, fn.restype = [], _MallInfo2
        info = fn()
        return (info.uordblks + info.hblkhd) / 2**20
    except (OSError, AttributeError):
        return None


def pymalloc_stats() -> dict:
    """CPython's small object arenas in use and the bytes in their allocated blocks. The arenas are mapped outside
    the C allocator, so their blocks add to malloc_in_use_mb without counting twice."""
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as tmp:
            sys.stderr.flush()
            saved = os.dup(2)
            os.dup2(tmp.fileno(), 2)
            try:
                sys._debugmallocstats()
            finally:
                os.dup2(saved, 2)
                os.close(saved)
            tmp.seek(0)
            text = tmp.read()
    except (AttributeError, OSError):
        return {"arenas": None, "pymalloc_mb": None}

    def number(label: str) -> int | None:
        found = re.search(rf"# {label}\s*=\s*([\d,]+)", text)
        return int(found.group(1).replace(",", "")) if found else None

    blocks = number("bytes in allocated blocks")
    return {"arenas": number("arenas allocated current"), "pymalloc_mb": None if blocks is None else blocks / 2**20}


def _in_use(sample: dict) -> float | None:
    """Memory in use, the C allocator's and CPython's small objects together, or None without both readings."""
    if sample.get("malloc_mb") is None or sample.get("pymalloc_mb") is None:
        return None
    return sample["malloc_mb"] + sample["pymalloc_mb"]


def _slope(points: list[tuple[float, float]]) -> float | None:
    """Least squares slope of (hours, MB) points in MB per hour, or None with fewer than three points."""
    if len(points) < 3:
        return None
    x, y = np.array(points, dtype=float).T
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _week_before(name: str, samples: list[dict], end: dict) -> float | None:
    """The reading of one measure a week of replayed time before the end, the latest at or before that point, or
    None when the replay is shorter than a week and an hour."""
    found = [s[name] for s in samples if s.get(name) is not None and s["hours"] <= end["hours"] - WEEK_HOURS]
    return found[-1] if found else None


def _growth(name: str, samples: list[dict], end: dict, limit_mb: float) -> dict | None:
    """The plan's comparison for one memory measure, plus the worst later reading, the recent slope and the growth
    over the last week, which compares two readings at the same point of the weekly cycle."""
    first, last = samples[0].get(name) if samples else None, end.get(name)
    if first is None or last is None:
        return None
    later = [(s["hours"], s[name]) for s in samples[1:] if s.get(name) is not None] + [(end["hours"], last)]
    recent = [(h, v) for h, v in [(s["hours"], s.get(name)) for s in samples] + [(end["hours"], last)]
              if v is not None and h >= end["hours"] - SLOPE_HOURS]
    slope = _slope(recent)
    week = _week_before(name, samples, end)
    return {"first_hour_mb": round(first, 3), "end_mb": round(last, 3), "growth_mb": round(last - first, 3),
            "rule_held": last - first <= limit_mb, "largest_growth_mb": round(max(v for _, v in later) - first, 3),
            "later_readings": len(later), "readings_over_limit": sum(v - first > limit_mb for _, v in later),
            "slope_mb_per_hour_last_day": None if slope is None else round(slope, 4),
            "week_over_week_mb": None if week is None else round(last - week, 3)}


def memory_verdict(samples: list[dict], end: dict | None, limit_mb: float = MAX_GROWTH_MB,
                   week_limit_mb: float = MAX_WEEK_GROWTH_MB, final_week_on_linux: bool = False) -> dict:
    """The plan's rule, memory at the end at most limit_mb above it at the end of the first replayed hour, decided by
    resident memory read from /proc on Linux alone and by memory in use on a Mac, where growth in use below
    week_limit_mb over the final replayed week also passes, and inconclusive without either reading. With
    final_week_on_linux, the owner's decision of 15 October 2026 for the home PC, resident memory gets the same final
    week rule, while a Raspberry Pi keeps the strict rule."""
    end = dict(end or {})
    samples = [{**s, "in_use_mb": _in_use(s)} for s in samples]
    end["in_use_mb"] = _in_use(end)
    rss = _growth("rss_mb", samples, end, limit_mb)
    if rss is None:
        return {"outcome": "failed", "judged_on": None, "decided_by": None, "limit_mb": limit_mb,
                "week_limit_mb": week_limit_mb,
                "reason": "no memory reading at the end of a first replayed hour or at the end"}
    source = end.get("source")
    in_use = _growth("in_use_mb", samples, end, limit_mb)
    judged_on = "resident" if source == "proc" else "in_use" if in_use is not None else None
    result = {"source": source, "judged_on": judged_on, "decided_by": "first hour", "limit_mb": limit_mb,
              "week_limit_mb": week_limit_mb, **rss,
              "footprint": _growth("footprint_mb", samples, end, limit_mb), "in_use": in_use}
    if judged_on == "resident" and final_week_on_linux:
        result.update(_final_week_verdict(rss, "resident memory", limit_mb, week_limit_mb))
    elif judged_on == "resident":
        result["outcome"] = "passed" if rss["rule_held"] else "failed"
        result["reason"] = f"resident memory grew {rss['growth_mb']} MB after the first hour against {limit_mb} MB"
    elif judged_on == "in_use":
        result.update(_in_use_verdict(in_use, limit_mb, week_limit_mb))
        result["reason"] += f", while resident memory read with {source} grew {rss['growth_mb']} MB"
    else:
        result["outcome"], result["decided_by"] = "inconclusive", None
        result["reason"] = (f"resident memory read with {source} grew {rss['growth_mb']} MB, and only a Linux reading "
                            "or memory in use decides the rule")
    return result


def _in_use_verdict(in_use: dict, limit_mb: float, week_limit_mb: float) -> dict:
    """The Mac's verdict on memory in use, passing when it grew at most limit_mb after the first hour or, failing that,
    less than week_limit_mb over the final replayed week."""
    return _final_week_verdict(in_use, "memory in use", limit_mb, week_limit_mb)


def _final_week_verdict(measure: dict, label: str, limit_mb: float, week_limit_mb: float) -> dict:
    """Passing when the measure grew at most limit_mb after the first hour or, failing that, less than week_limit_mb
    over the final replayed week."""
    growth, week = measure["growth_mb"], measure["week_over_week_mb"]
    first = f"{label} grew {growth} MB after the first hour against {limit_mb} MB"
    if measure["rule_held"]:
        return {"outcome": "passed", "decided_by": "first hour", "reason": first}
    if week is None:
        return {"outcome": "failed", "decided_by": "first hour",
                "reason": f"{first}, and the replay holds no final week to compare"}
    return {"outcome": "passed" if week < week_limit_mb else "failed", "decided_by": "final week",
            "reason": f"{first} and {week} MB over the final replayed week against {week_limit_mb} MB"}


# The frames this process pipes to the replay processes


FRAME = struct.Struct("<BBqqI")  # frame type, segment, receive time, connection id, payload bytes
LINE_CODES = {OPEN: 1, MESSAGE: 2, CLOSE: 3, NEWS_LINE: 4, HEARTBEAT_LINE: 5}
LINE_KINDS = {code: kind for kind, code in LINE_CODES.items()}
HOUR, PRICES, END = 10, 11, 12  # requests that a replay process answers with one JSON line
SEGMENTS = ("real", "synthetic")
PASSES = ("run", "check")


class ChildGone(RuntimeError):
    """Raised when a replay process stops reading or answering."""


class Child:
    """This process's end of one pipe. Lines go in as frames, and each request is answered once everything before it
    has been replayed, so the child is idle, blocked on its next read, whenever its memory is read."""

    def __init__(self, spec_path: Path, stderr_path: Path) -> None:
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
        with stderr_path.open("wb") as err:
            self.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env)

    def send(self, segment: int, line: Line) -> None:
        kind, raw, recv_ms, conn = line
        data = raw.encode("utf-8")
        try:
            self.proc.stdin.write(FRAME.pack(LINE_CODES[kind], segment, recv_ms, conn, len(data)) + data)
        except (BrokenPipeError, OSError) as exc:
            raise ChildGone(f"a replay process stopped reading, {exc}") from exc

    def ask(self, request: int, ts_ms: int = 0, segment: int = 0) -> dict:
        try:
            self.proc.stdin.write(FRAME.pack(request, segment, ts_ms, 0, 0))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ChildGone(f"a replay process stopped reading, {exc}") from exc
        text = self.proc.stdout.readline()
        if not text:
            raise ChildGone("a replay process ended without answering")
        return json.loads(text)

    def close(self) -> int:
        """Close the pipe, which lets the replay process close its databases and exit, and return its exit code."""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        return self.proc.wait()


# The replay process


class ReplayClock(LiveClock):
    """The live clock's stamp rule on the replayed time the worker sets, so the ticker follows the recorded stamps."""

    def __init__(self, t: int) -> None:
        super().__init__()
        self.t = t

    def now_ms(self) -> int:
        return self.t


class Inline(Executor):
    """Runs the local only check at once, so the next tick writes its result, as the worker thread would."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


PASSED = LocalCheck((), ())


class _LogCounter(logging.Handler):
    """Counts warnings and errors, and the invariant violation lines a healthy run never shows."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        self.counts[record.levelname.lower()] += 1
        if record.getMessage().startswith("invariant violation"):
            self.counts["invariant violation"] += 1


def soak_configs(cfgs: Sequence[StockConfig], folder: Path) -> tuple[StockConfig, ...]:
    """Each run writing into its own new folder under folder, named relative to it as replay does, so every pass
    writes the same rows and checkpoints, and the recording into folder/raw_stocks."""
    out = []
    for cfg in cfgs:
        run_cfg = stock_replay.replay_config(cfg, folder / cfg.run.name)
        out.append(replace(run_cfg, stock_feed=replace(run_cfg.stock_feed, raw_dir="../raw_stocks")))
    return tuple(out)


def news_meta(hashes: Sequence[str]) -> dict[str, str]:
    """The news.db meta a first launch copies, with the hashes the news lines carry."""
    return {"config_hash": hashes[0], "prompt_sha256": hashes[1], "schema_sha256": hashes[2],
            "haiku_model": HAIKU_MODEL, "sonnet_model": SONNET_MODEL}


def check_books(group: StockGroup) -> tuple[int, list[str]]:
    """The account identities of every bot in both runs, read from memory between two lines, marking a position at
    its symbol's last healthy reference price and skipping one with no price yet."""
    checked, problems = 0, []
    for engine in group.engines:
        for bot in engine.all_bots():
            mark = bot.mark(float(group.market.last_ref[bot.home]))
            if bot.account.is_open and mark is None:
                continue
            checked += 1
            problems += [f"{engine.cfg.run.name} bot {bot.bot_id}: {e}" for e in bot.account.invariant_errors(mark)]
    return checked, problems


class Soaked:
    """The stock process of a replay process, launched fresh at the first line, with its one second ticker and the
    CPU it spends by part and by segment."""

    def __init__(self, cfgs: Sequence[StockConfig], meta: dict[str, str], commit: str, start_ms: int) -> None:
        self.start_ms = start_ms
        self.clock = ReplayClock(start_ms)
        adapter = FinnhubAdapter(cfgs[0].stock_feed.symbols, stock_replay.REPLAY_TOKEN)
        group, _ = process.open_runs(cfgs, adapter, start_ms, commit, news_meta=lambda _: (meta, 0))
        recorder = StockRecorder(cfgs[0].raw_dir) if cfgs[0].stock_feed.record_raw_messages else None
        self.process = process.StockProcess(group, None, self.clock, recorder, check=lambda: PASSED,
                                            executor=Inline())
        self.feed = redact.redacting_sink(self.process.sink)
        self.next_tick = (start_ms // MS_PER_SECOND + 1) * MS_PER_SECOND
        self.parts: Counter[str] = Counter()
        self.segments = {name: Counter() for name in SEGMENTS}
        group._boundary = self._timed(group._boundary, "minute_steps")
        for engine in group.engines:
            engine.persistence_point = self._timed(engine.persistence_point, "persistence")
        self.process.heartbeat = self._timed(self.process.heartbeat, "heartbeats")

    def _timed(self, inner: Callable, name: str) -> Callable:
        """Wrap one method on this instance only, adding its CPU time to the part name."""
        def wrapper(*args, **kwargs):
            started = time.process_time()
            try:
                return inner(*args, **kwargs)
            finally:
                self.parts[name] += time.process_time() - started
        return wrapper

    def _spent(self, segment: str, work: Callable[[], object]) -> None:
        """Run one piece of the process's work, counting its CPU and prints against the segment."""
        counts, prints = self.segments[segment], self.process.group.counts[PRINTS]
        started = time.process_time()
        work()
        counts["cpu_s"] += time.process_time() - started
        counts["prints"] += self.process.group.counts[PRINTS] - prints

    def ticks(self, until_ms: int, segment: str) -> None:
        """Every tick of the one second ticker up to until_ms, each on the clock set to its second."""
        while self.next_tick <= until_ms:
            self.clock.t = self.next_tick
            self._spent(segment, self.process.tick)
            self.next_tick += MS_PER_SECOND

    def line(self, kind: str, raw: str, recv_ms: int, conn: int, segment: str) -> None:
        """One line after the ticks before it, a feed line through the redacting sink and a news line as the puller
        hands it over."""
        self.ticks(recv_ms, segment)
        sink = self.process.sink if kind in NEWS_KINDS else self.feed
        self._spent(segment, lambda: sink(kind, raw, recv_ms, conn))
        self.segments[segment]["lines"] += 1
        self.segments[segment]["messages"] += kind == MESSAGE

    def finish(self, segment: str) -> None:
        """The shutdown of the live process, a final checkpoint of both runs and the stopped state."""
        self._spent(segment, self.process.finish)

    def prices(self) -> dict[str, float | None]:
        """Each symbol's last healthy reference price, where the synthetic run continues from."""
        market = self.process.group.market
        return {s: None if np.isnan(p) else float(p) for s, p in zip(market.symbols, market.last_ref.tolist())}

    def snapshot(self, kind: str, ts_ms: int, segment: str) -> dict:
        """Counts, CPU and memory at the end of a replayed hour or of the replay, with the books checked."""
        group = self.process.group
        checked, problems = check_books(group)
        return {"type": kind, "hours": (ts_ms - self.start_ms) / MS_PER_HOUR, "start_ts": iso(self.start_ms),
                "ts": iso(ts_ms), "segment": segment,
                "lines": sum(s["lines"] for s in self.segments.values()), "prints": group.counts[PRINTS],
                "messages": group.counts[MESSAGES], "news": group.counts[NEWS_LINES],
                "cpu_s": sum(s["cpu_s"] for s in self.segments.values()), "worker_cpu_s": time.process_time(),
                "peak_rss_mb": peak_rss_mb(), "malloc_mb": malloc_in_use_mb(), **pymalloc_stats(),
                "live_blocks": sys.getallocatedblocks(),
                "living": [len(e.population.living_learners()) for e in group.engines], "checked": checked,
                "problems": problems[:20], "problem_count": len(problems)}

    def close(self) -> None:
        """Release the recording and both run databases."""
        self.process.close()
        for engine in self.process.group.engines:
            engine.store.close()


def _read_exact(stream, size: int) -> bytes:
    """Exactly size bytes from the pipe, or an exit when the soak closed it early."""
    data = stream.read(size)
    if len(data) != size:
        raise SystemExit("the soak closed the pipe early")
    return data


def worker(spec_path: Path) -> int:
    """Replay the lines piped in by the soak, answering its requests at every replayed hour and at the end."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    folder = Path(spec["folder"])
    leak = int(spec.get("leak_kb_per_message", 0) * 1024)  # a test hook that proves a leak fails the soak
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr  # nothing else may write into the answer channel
    inp = sys.stdin.buffer

    def answer(message: dict) -> None:
        proto.write(json.dumps(message) + "\n")
        proto.flush()

    cfgs = soak_configs(load_run_pair(spec["s1"], spec["s2"]), folder)
    handler = logging.FileHandler(folder / "soak.log", encoding="utf-8")
    handler.setFormatter(redact.RedactingFormatter(logging.Formatter(LOG_FORMAT)))
    counter = _LogCounter()
    logging.getLogger().handlers[:] = [handler, counter]
    logging.getLogger().setLevel(cfgs[0].run.log_level)
    soaked: Soaked | None = None
    leaked: list[bytes] = []
    try:
        while True:
            code, segment, recv_ms, conn, size = FRAME.unpack(_read_exact(inp, FRAME.size))
            name = SEGMENTS[segment]
            if code in LINE_KINDS:
                raw = _read_exact(inp, size).decode("utf-8") if size else ""
                if soaked is None:
                    soaked = Soaked(cfgs, news_meta(spec["hashes"]), spec["commit"], recv_ms)
                soaked.line(LINE_KINDS[code], raw, recv_ms, conn, name)
                if leak and code == LINE_CODES[MESSAGE]:
                    leaked.append(b"\x01" * leak)
            elif soaked is None:
                raise SystemExit("the soak has no lines to replay")
            elif code == PRICES:
                answer({"type": "prices", "prices": soaked.prices()})
            elif code == HOUR:
                soaked.ticks(recv_ms, name)
                answer(soaked.snapshot("hour", recv_ms, name))
            elif code == END:
                soaked.ticks(recv_ms, name)
                soaked.finish(name)
                end = soaked.snapshot("end", soaked.process.group.now_ms, name)
                end.update(segments=soaked.segments, parts=soaked.parts, log_counts=counter.counts)
                answer(end)
                inp.read()  # stay idle while the soak reads memory, until it closes the pipe
                return 0
            else:
                raise SystemExit(f"unknown frame type {code}")
    finally:
        if soaked is not None:
            soaked.close()


# Checks on the finished databases


LEDGER_CHECKS = {
    "equity snapshots, cash equals start plus realized minus fees, borrow and rent": (
        "SELECT count(*), coalesce(sum(abs(cash - (:start + realized_gross - fees_total - funding_total - rent_total))"
        " > :tol), 0) FROM equity_snapshots"),
    "equity snapshots, equity equals cash plus unrealized": (
        "SELECT count(*), coalesce(sum(abs(equity - cash - unrealized) > :tol), 0) FROM equity_snapshots"),
    "deaths, the five parts and the writeoff explain the final equity": (
        "SELECT count(*), coalesce(sum(abs(final_equity - (:start - fees_total - funding_total - rent_total"
        " - stop_loss_total - other_losses_total + writeoff)) > :tol), 0) FROM deaths"),
    "deaths, final equity is never negative": (
        "SELECT count(*), coalesce(sum(final_equity < -:tol), 0) FROM deaths"),
    "closed trades, net equals gross minus both fees and borrow": (
        "SELECT count(*), coalesce(sum(abs(pnl_net - (pnl_gross - fee_open - fee_close - funding)) > :tol), 0)"
        " FROM trades WHERE closed_at IS NOT NULL"),
    "dead bots, each has exactly one death row": (
        "SELECT count(*), coalesce(sum(d.bot_id IS NULL), 0) FROM bots b LEFT JOIN deaths d USING (bot_id)"
        " WHERE b.state = 'dead'"),
    "fills, never before their decision": (
        "SELECT count(*), coalesce(sum(filled_at < decided_at), 0) FROM fills WHERE status = 'filled'"),
    "fills, each on a print at or after its due time": (
        "SELECT count(*), coalesce(sum(print_after_due_ms IS NULL OR print_after_due_ms < 0), 0) FROM fills"
        " WHERE status = 'filled'"),
    "forecasts, no news input received after the minute": (
        "SELECT count(*), coalesce(sum(inputs_max_recv_at > ts), 0) FROM stock_forecasts"),
    "news lines, never received before they were scored": (
        "SELECT count(*), coalesce(sum(recv_at < scored_at), 0) FROM news_seen"),
    "engine checks at equity snapshots, no invariant violation": (
        "SELECT (SELECT count(*) FROM equity_snapshots), count(*) FROM feed_events"
        f" WHERE kind = '{INVARIANT_VIOLATION}'"),
}
OPEN_ACTIONS = (*OPENS, "flip")
WINDOW_CHECK = "decisions, opens only inside the open window"
CARRY_CHECK = "trades, no position kept past a close except by a halt"
CARRY_EXITS = (SESSION_FLATTEN, DEATH)


def _session_checks(conn: sqlite3.Connection, cfg: StockConfig) -> dict[str, dict]:
    """Opens only from session minute 30 to 35 minutes before the close, and every position closed by its session's
    close unless it was carried and closed later with session_flatten or a death close."""
    opens = conn.execute(f"SELECT ts FROM decisions WHERE action IN ({','.join('?' * len(OPEN_ACTIONS))})",
                         OPEN_ACTIONS).fetchall()
    outside = 0
    for (ts,) in opens:
        ms = parse_iso(ts)
        session = sessions.session_of_boundary(ms)
        outside += session is None or not opens_allowed(session, session.session_minute(ms), cfg.session)
    trades = conn.execute("SELECT opened_at, closed_at, exit_reason FROM trades WHERE closed_at IS NOT NULL").fetchall()
    kept = 0
    for opened, closed, reason in trades:
        session = sessions.session_at(parse_iso(opened))
        kept += session is None or (parse_iso(closed) > session.close_ms and reason not in CARRY_EXITS)
    return {WINDOW_CHECK: {"rows": len(opens), "failures": outside}, CARRY_CHECK: {"rows": len(trades),
                                                                                    "failures": kept}}


def database_report(db: Path, cfg: StockConfig) -> dict:
    """The ledger and session checks of one finished run database, and its counts of activity."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        params = {"start": cfg.trading.start_equity, "tol": INVARIANT_TOLERANCE}
        checks = {name: dict(zip(("rows", "failures"), conn.execute(sql, params).fetchone()))
                  for name, sql in LEDGER_CHECKS.items()}
        checks.update(_session_checks(conn, cfg))

        def scalar(sql: str) -> int:
            return conn.execute(sql).fetchone()[0]

        return {
            "checks": checks,
            "session_minutes": scalar("SELECT count(DISTINCT ts) FROM stock_candles"),
            "healthy_candles": scalar("SELECT count(*) FROM stock_candles WHERE valid = 1"),
            "decisions": scalar("SELECT count(*) FROM decisions"),
            "forced_decisions": scalar("SELECT count(*) FROM decisions WHERE forced = 1"),
            "trades": scalar("SELECT count(*) FROM trades"),
            "closed_trades": scalar("SELECT count(*) FROM trades WHERE closed_at IS NOT NULL"),
            "carried_open_at_end": scalar("SELECT count(*) FROM trades WHERE closed_at IS NULL"),
            "fills": dict(conn.execute("SELECT status, count(*) FROM fills GROUP BY status ORDER BY status")),
            "exit_reasons": dict(conn.execute("SELECT coalesce(exit_reason, 'open'), count(*) FROM trades"
                                              " GROUP BY 1 ORDER BY 1")),
            "deaths": scalar("SELECT count(*) FROM deaths"),
            "news_seen": scalar("SELECT count(*) FROM news_seen"),
            "news_minutes": dict(conn.execute("SELECT healthy, count(*) FROM news_minutes GROUP BY healthy")),
            "feed_events": dict(conn.execute("SELECT kind, count(*) FROM feed_events GROUP BY kind ORDER BY kind")),
            "checkpoints": scalar("SELECT count(*) FROM checkpoints"),
        }
    finally:
        conn.close()


LINEAR_PRICES = "SELECT * FROM stock_forecasts WHERE predictor = 'linear_prices' ORDER BY ts, symbol"


def shared_rows_check(s1: Path, s2: Path) -> dict:
    """S1 and S2 share every candle, label, session, spread check, price z score row and linear_prices row, which
    differ only by news."""
    a, b = stock_replay.table_digests(s1), stock_replay.table_digests(s2)
    rows, failures = 0, 0
    for table in SHARED_TABLES:
        rows += a[table][0]
        failures += a[table] != b[table]
    conns = [sqlite3.connect(f"file:{path}?mode=ro", uri=True) for path in (s1, s2)]
    try:
        linear = [conn.execute(LINEAR_PRICES).fetchall() for conn in conns]
    finally:
        for conn in conns:
            conn.close()
    return {"rows": rows + len(linear[0]), "failures": failures + (linear[0] != linear[1])}


def recording_digest(folder: Path) -> dict:
    """How many lines the process recorded, how often a stamp went backwards, and the digest of every line."""
    digest, lines, backwards, last = hashlib.sha256(), 0, 0, None
    for line in read_recording([folder]) if folder.is_dir() else ():
        digest.update(json.dumps(line, separators=(",", ":")).encode("utf-8") + b"\n")
        lines += 1
        backwards += last is not None and line[2] < last
        last = line[2]
    return {"lines": lines, "backwards": backwards, "digest": digest.hexdigest()}


def determinism(first: Path, second: Path, names: Sequence[str]) -> dict:
    """Every table but runtime_status of each run database of the second pass against the first, and the two
    recordings, with one digest over the first pass's tables."""
    differing, rows, whole = [], 0, hashlib.sha256()
    for name in names:
        a = stock_replay.table_digests(first / name / "stocks.db")
        b = stock_replay.table_digests(second / name / "stocks.db")
        for table in COMPARED:
            rows += a[table][0]
            whole.update(f"{name}.{table}:{a[table][1]}\n".encode("utf-8"))
            if a[table] != b[table]:
                differing.append(f"{name}.{table}")
    same_recording = recording_digest(first / "raw_stocks") == recording_digest(second / "raw_stocks")
    if not same_recording:
        differing.append("recording")
    return {"passed": not differing, "tables": len(names) * len(COMPARED), "rows": rows, "differing": differing,
            "digest": whole.hexdigest()}


# The soak


def run_soak(args: argparse.Namespace) -> dict:
    """Pipe the joined recordings and the synthetic run to both replay processes and build the report."""
    cfgs = load_run_pair(args.s1, args.s2)
    out = (args.out or SOAK_ROOT / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")).resolve()
    raw = (args.raw or cfgs[0].raw_dir).resolve()
    if out == raw or raw in out.parents:
        raise SoakError(f"{out} is inside the recordings folder, the soak never writes there")
    if out.exists() and any(out.iterdir()):
        raise SoakError(f"{out} is not empty, the soak only writes into a new or empty folder")
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "soak.log", encoding="utf-8")  # such as a recording that ends early
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    try:
        return _soak(args, cfgs, out, raw)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _soak(args: argparse.Namespace, cfgs: tuple[StockConfig, StockConfig], out: Path, raw: Path) -> dict:
    """The soak into out, after its folder was checked."""
    recordings = [] if args.skip_real else select_recordings(raw, out / "input")
    found = recording_news(recordings)
    hashes, hashes_from = launch_hashes(recordings, found)
    plan = Synthetic(args.synthetic_days, args.prints_per_second, args.reconnect_minutes, args.halt_minutes,
                     args.news_minutes, args.burst_items, args.outage_minutes)
    commit = process.code_commit()
    children = []
    for name in PASSES:
        (out / name).mkdir()
        spec = {"folder": str(out / name), "s1": str(Path(args.s1).resolve()), "s2": str(Path(args.s2).resolve()),
                "hashes": list(hashes), "commit": commit,
                "leak_kb_per_message": args.leak_kb_per_message if name == "run" else 0}
        (out / name / "spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        children.append(Child(out / name / "spec.json", out / name / "stderr.log"))
    measured = children[0]
    state = JoinState()

    def synthetic(st: JoinState) -> Iterable[Line]:
        start, prices = parse_iso(args.synthetic_start), {}
        if st.last_ms is not None:
            start, prices = st.last_ms + JOIN_GAP_MS, measured.ask(PRICES)["prices"]
        return synthetic_lines(cfgs[0].run.master_seed, cfgs[0].stock_feed.symbols, start, plan, hashes,
                               Seqs(st.score_seq, st.item_seq, st.heartbeat_seq), prices)

    sources: list[Source] = [
        ("real", (lambda st, p=r.path: read_recording([p])) if news is not None else
         (lambda st, p=r.path: with_heartbeats(read_recording([p]), hashes, st))) for r, news in zip(recordings, found)]
    if plan.days > 0:
        sources.append(("synthetic", synthetic))
    started, cpu_started = time.monotonic(), time.process_time()
    samples: list[dict] = []
    ends: list[dict | None] = [None, None]
    failure = None
    sent = 0
    try:
        next_hour = None
        for line in join(sources, state):
            segment = SEGMENTS.index(state.segment)
            if next_hour is None:
                next_hour = line[2] + MS_PER_HOUR
            while line[2] >= next_hour:
                sample = {**measured.ask(HOUR, next_hour, segment), "hour": len(samples) + 1,
                          **process_memory(measured.proc.pid)}
                samples.append(sample)
                if args.detail:
                    print(_progress(sample), flush=True)
                next_hour += MS_PER_HOUR
            for child in children:
                child.send(segment, line)
            sent += 1
        if next_hour is None:
            raise ChildGone("the soak has no lines to replay")
        segment = SEGMENTS.index(state.segment)
        ends[0] = {**measured.ask(END, state.last_ms, segment), **process_memory(measured.proc.pid)}
        ends[1] = children[1].ask(END, state.last_ms, segment)
    except ChildGone as exc:
        failure = str(exc)
    codes = [child.close() for child in children]
    run = Run(cfgs, (Path(args.s1), Path(args.s2)), out, recordings, hashes_from, plan, state, sent, samples, ends,
              codes, failure,
              time.monotonic() - started, time.process_time() - cpu_started)
    report = build_report(run, args.max_growth_mb, args.max_week_growth_mb, args.final_week_on_linux)
    write_outputs(out, report, samples)
    return report


@dataclass
class Run:
    """Everything the soak gathered, for the report."""

    cfgs: tuple[StockConfig, ...]
    config_paths: tuple[Path, Path]
    out: Path
    recordings: list[Recording]
    hashes_from: str
    plan: Synthetic
    state: JoinState
    sent: int
    samples: list[dict]
    ends: list[dict | None]
    codes: list[int]
    failure: str | None
    wall_s: float
    parent_cpu_s: float


def _progress(m: dict) -> str:
    """One hourly reading for --detail, memory and counts only."""
    in_use = "" if _in_use(m) is None else f" in use {_in_use(m):.1f} MB"
    return (f"  hour {m['hour']:>3} {m['ts'][:16]} {m['segment']:<9} rss {m['rss_mb']:.1f} MB{in_use} "
            f"lines {m['lines']} prints {m['prints']}")


def _cpu(end: dict, session_minutes: int, replayed_s: float) -> dict:
    """CPU of the stock process per print, per session minute and against live speed, whole and by segment."""
    parts = end["parts"]
    total, prints = end["cpu_s"], max(end["prints"], 1)
    handling = total - parts.get("minute_steps", 0.0) - parts.get("persistence", 0.0) - parts.get("heartbeats", 0.0)
    segments = {name: {"lines": s["lines"], "messages": s["messages"], "prints": s["prints"],
                       "cpu_s": round(s["cpu_s"], 3),
                       "us_per_print": round(s["cpu_s"] / s["prints"] * 1e6, 2) if s["prints"] else None}
                for name, s in end["segments"].items() if s.get("lines")}
    return {"process_cpu_s": round(total, 3), "us_per_print": round(total / prints * 1e6, 2),
            "handling_us_per_print": round(handling / prints * 1e6, 2),
            "minute_steps_cpu_s": round(parts.get("minute_steps", 0.0), 3),
            "ms_per_session_minute": round(parts.get("minute_steps", 0.0) / session_minutes * 1e3, 3)
            if session_minutes else None,
            "persistence_cpu_s": round(parts.get("persistence", 0.0), 3),
            "heartbeats_cpu_s": round(parts.get("heartbeats", 0.0), 3),
            "share_of_one_core_at_live_speed": round(total / replayed_s, 6) if replayed_s else None,
            "worker_cpu_s": round(end["worker_cpu_s"], 3), "segments": segments}


def build_report(run: Run, limit_mb: float = MAX_GROWTH_MB, week_limit_mb: float = MAX_WEEK_GROWTH_MB,
                 final_week_on_linux: bool = False) -> dict:
    """The outcome with memory, CPU, invariants, determinism and activity, from what the soak gathered."""
    cfgs, out, end = run.cfgs, run.out, run.ends[0]
    names = [cfg.run.name for cfg in cfgs]
    report = {
        "label": PLUMBING_LABEL, "never_for_selection": NEVER_FOR_SELECTION, "out": str(out),
        "configs": [str(path) for path in run.config_paths], "news_hashes": run.hashes_from,
        "worker_exit_codes": run.codes, "wall_seconds": round(run.wall_s, 3),
        "parent_cpu_s": round(run.parent_cpu_s, 3),
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "recordings": [{"source": str(r.source), "read": str(r.path), "status": r.status, "lines": n}
                       for r, n in zip(run.recordings, run.state.lines)],
        "synthetic": {**vars(run.plan), "lines": run.state.lines[-1] if run.plan.days > 0 and
                      len(run.state.lines) > len(run.recordings) else 0},
        "hourly": [{"hour": s["hour"], "ts": s["ts"], "segment": s["segment"], "rss_mb": s["rss_mb"],
                    "footprint_mb": s["footprint_mb"], "in_use_mb": _in_use(s), "lines": s["lines"],
                    "prints": s["prints"], "cpu_s": s["cpu_s"]} for s in run.samples],
    }
    if end is None or run.ends[1] is None or any(run.codes):
        why = run.failure or f"a replay process failed with exit codes {run.codes}"
        report.update(outcome="failed", passed=False, verdict=f"failed, {why}, see the logs in {out}")
        return report
    replayed_s = end["hours"] * 3600
    dbs = {name: database_report(out / "run" / name / "stocks.db", cfg)
           for name, cfg in zip(names, cfgs)}
    checks = [{"check": check, "run": name, **result} for name, db in dbs.items()
              for check, result in db["checks"].items()]
    checks.append({"check": "S1 and S2, the same candles, labels, sessions, spread checks, price z scores and "
                            "linear_prices rows",
                   "run": "both", **shared_rows_check(*(out / "run" / name / "stocks.db" for name in names))})
    recording = recording_digest(out / "run" / "raw_stocks")
    recorded = cfgs[0].stock_feed.record_raw_messages
    checks.append({"check": "recording, every line once in stamp order", "run": "both", "rows": recording["lines"],
                   "failures": int(recording["backwards"] > 0 or (recorded and recording["lines"] != run.sent))})
    hourly = sum(s["problem_count"] for s in run.samples) + end["problem_count"]
    checks.append({"check": "books of every bot from memory every replayed hour", "run": "both",
                   "rows": sum(s["checked"] for s in run.samples) + end["checked"], "failures": hourly})
    checks.append({"check": "invariant violation log lines", "run": "both", "rows": end["lines"],
                   "failures": end["log_counts"].get("invariant violation", 0)})
    failing = [c for c in checks if c["failures"]]
    memory = memory_verdict(run.samples, end, limit_mb, week_limit_mb, final_week_on_linux)
    memory["peak_rss_mb"] = round(end["peak_rss_mb"], 3)
    if run.samples:  # live blocks flat while arenas rise means fragmentation, not a leak
        memory["live_blocks"] = {"first_hour": run.samples[0]["live_blocks"], "end": end["live_blocks"]}
        memory["arenas"] = {"first_hour": run.samples[0]["arenas"], "end": end["arenas"]}
    same = determinism(out / "run", out / "check", names)
    first = dbs[names[0]]
    report.update(
        replay={"first_ts": end["start_ts"], "last_ts": end["ts"],
                "hours": round(end["hours"], 3), "lines": run.sent, "prints": end["prints"],
                "messages": end["messages"], "news_lines": end["news"], "join_markers": run.state.markers,
                "heartbeats_added": run.state.heartbeats_added,
                "times_clamped": run.state.clamped, "session_minutes": first["session_minutes"],
                "log_counts": end["log_counts"]},
        memory=memory, cpu=_cpu(end, first["session_minutes"], replayed_s),
        checks={"passed": not failing, "count": len(checks), "failing": len(failing), "items": checks,
                "hourly_problem_examples": [p for s in run.samples for p in s["problems"]][:10]},
        determinism=same, activity={name: {k: v for k, v in db.items() if k != "checks"} for name, db in dbs.items()},
    )
    reasons = [] if memory["outcome"] == "passed" else [memory["reason"]]
    if failing:
        reasons.append(f"{len(failing)} invariant checks failed")
    if not same["passed"]:
        reasons.append(f"the second pass differed in {', '.join(same['differing'])}")
    outcome = "failed" if failing or not same["passed"] or memory["outcome"] == "failed" else memory["outcome"]
    report.update(outcome=outcome, passed=outcome == "passed",
                  verdict=outcome if not reasons else f"{outcome}, " + ", ".join(reasons))
    return report


def write_outputs(out: Path, report: dict, samples: list[dict]) -> None:
    """The whole report as JSON and the hourly readings as CSV in the soak's folder."""
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["hour", "hours", "ts", "segment", "rss_mb", "footprint_mb", "malloc_mb", "pymalloc_mb", "arenas",
              "live_blocks", "peak_rss_mb", "lines", "prints", "cpu_s", "worker_cpu_s"]
    with (out / "memory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for s in samples:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})


# What the soak prints


def _kib(mb: float | None) -> output.Count:
    """Megabytes of this process's memory as a count of whole KiB, the form the default output allows."""
    return output.Count(round((mb or 0.0) * 1024))


def summary(report: dict) -> list[output.Report]:
    """The default summary of verdicts, counts, hashes and timings, with the hourly readings and every check as rows
    for --detail."""
    hourly = tuple({"hour": s["hour"], "ts": s["ts"][:16], "segment": s["segment"], "rss_mb": round(s["rss_mb"], 1),
                    "in_use_mb": None if s["in_use_mb"] is None else round(s["in_use_mb"], 1), "lines": s["lines"],
                    "prints": s["prints"], "cpu_s": round(s["cpu_s"], 2)} for s in report["hourly"])
    if "memory" not in report:
        return [output.Report("stock soak", (("verdict", output.Word(report["outcome"])),
                                             ("run time", output.Seconds(report["wall_seconds"]))), hourly)]
    r, m, c, k, d = report["replay"], report["memory"], report["cpu"], report["checks"], report["determinism"]
    judged = (m["in_use"] if m["judged_on"] == "in_use" else m) or {}
    slope, week = judged.get("slope_mb_per_hour_last_day"), judged.get("week_over_week_mb")
    main = output.Report("stock soak", (
        ("verdict", output.Word(report["outcome"])),
        ("replayed minutes", output.Count(round(r["hours"] * 60))),
        ("lines", output.Count(r["lines"])),
        ("prints", output.Count(r["prints"])),
        ("news lines", output.Count(r["news_lines"])),
        ("session minutes", output.Count(r["session_minutes"])),
        ("memory rule", m["outcome"] == "passed"),
        ("memory judged on", output.Word((m["judged_on"] or "nothing decisive").replace("_", " "))),
        ("memory decided by", output.Word(m.get("decided_by") or "nothing decisive")),
        ("memory first hour kib", _kib(judged.get("first_hour_mb"))),
        ("memory end kib", _kib(judged.get("end_mb"))),
        ("memory growth kib", _kib(judged.get("growth_mb"))),
        ("memory limit kib", _kib(m["limit_mb"])),
        *((("memory week limit kib", _kib(m["week_limit_mb"])),) if m["judged_on"] == "in_use" else ()),
        ("resident growth kib", _kib(m.get("growth_mb"))),
        ("readings over limit", output.Count(judged.get("readings_over_limit", 0))),
        *((("memory slope kib per day", _kib(slope * 24)),) if slope is not None else ()),
        *((("memory week over week kib", _kib(week)),) if week is not None else ()),
        ("cpu per million prints", output.Seconds(c["us_per_print"])),
        ("handling cpu per million prints", output.Seconds(c["handling_us_per_print"])),
        ("cpu of the stock process", output.Seconds(c["process_cpu_s"])),
        ("cpu of minute steps", output.Seconds(c["minute_steps_cpu_s"])),
        ("cpu of persistence", output.Seconds(c["persistence_cpu_s"])),
        ("cpu per replayed hour", output.Seconds(c["process_cpu_s"] / r["hours"] if r["hours"] else 0.0)),
        ("invariants", k["passed"]),
        ("invariant checks", output.Count(k["count"])),
        ("failing checks", output.Count(k["failing"])),
        ("determinism", d["passed"]),
        ("tables compared", output.Count(d["tables"])),
        ("database digest", output.Hash(d["digest"])),
        ("run time", output.Seconds(report["wall_seconds"])),
    ), hourly)
    checks = output.Report("soak checks", (("checks", output.Count(k["count"])),
                                           ("failing checks", output.Count(k["failing"]))),
                           tuple({"run": i["run"], "rows": i["rows"], "failures": i["failures"], "check": i["check"]}
                                 for i in k["items"]))
    return [main, checks]


def build_parser() -> argparse.ArgumentParser:
    """The soak's arguments, with the worker and leak flags hidden."""
    p = argparse.ArgumentParser(description="Soak of the stock process. Replays every recording and a synthetic run "
                                            "of several days through the real stock process code with the rehearsal "
                                            "pair, a plumbing configuration that is never a strategy setting, and "
                                            "reports memory, CPU per print, invariants and determinism. "
                                            f"{NEVER_FOR_SELECTION} Exit code 0 passed, 1 failed, 2 cannot run, "
                                            "3 inconclusive.")
    p.add_argument("--s1", type=Path, default=DEFAULT_S1, help="the prices run config, by default rehearsal_s1.toml")
    p.add_argument("--s2", type=Path, default=DEFAULT_S2, help="the news run config, by default rehearsal_s2.toml")
    p.add_argument("--raw", type=Path, default=None, help="folder of recordings, only ever read, by default raw_dir")
    p.add_argument("--skip-real", action="store_true", help="replay only the synthetic run")
    p.add_argument("--synthetic-days", type=float, default=Synthetic.days)
    p.add_argument("--synthetic-start", default=SYNTHETIC_START, help="UTC start of the synthetic run when there are "
                                                                      "no recordings")
    p.add_argument("--prints-per-second", type=float, default=Synthetic.prints_per_second,
                   help="synthetic prints a second per symbol in the session")
    p.add_argument("--reconnect-minutes", type=float, default=Synthetic.reconnect_minutes,
                   help="about how often a synthetic connection fault comes")
    p.add_argument("--halt-minutes", type=float, default=Synthetic.halt_minutes,
                   help="length of the long halt in each synthetic session")
    p.add_argument("--news-minutes", type=float, default=Synthetic.news_minutes,
                   help="how often a synthetic news item comes, day and night")
    p.add_argument("--burst-items", type=int, default=Synthetic.burst_items,
                   help="items in the news burst of each synthetic session")
    p.add_argument("--outage-minutes", type=float, default=Synthetic.outage_minutes,
                   help="length of the news heartbeat outage in each synthetic session")
    p.add_argument("--max-growth-mb", type=float, default=MAX_GROWTH_MB)
    p.add_argument("--max-week-growth-mb", type=float, default=MAX_WEEK_GROWTH_MB,
                   help="on a Mac, growth in use over the final replayed week below this also passes")
    p.add_argument("--final-week-on-linux", action="store_true",
                   help="apply the final week rule to resident memory on Linux too, the owner's decision of 15 October "
                        "2026 for the home PC, never on a Raspberry Pi")
    p.add_argument("--out", type=Path, default=None, help="a new or empty folder, by default under data/v2/soak")
    output.add_detail_flag(p)
    p.add_argument("--leak-kb-per-message", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


EXPECTED = (SoakError, ConfigError, ReplayError, StoreError, sessions.CalendarError, ValueError, OSError)


def main(argv: list[str] | None = None) -> int:
    """Run the soak and print its summary, or run as a replay process with --worker."""
    args = build_parser().parse_args(argv)
    os.umask(0o077)  # databases, recordings and reports are for the owner alone
    if args.worker is not None:
        return worker(args.worker)
    try:
        report = run_soak(args)
    except EXPECTED as exc:
        print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
        return CANNOT_RUN
    for part in summary(report):
        output.emit(part, args.detail)
    if args.detail:
        print(f"report written to {Path(report['out']) / 'report.json'}")
    print(SOAK_SENTENCE)
    print(NEVER_FOR_SELECTION)
    return EXIT_CODES[report["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
