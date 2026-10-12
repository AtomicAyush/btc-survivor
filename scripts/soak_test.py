"""Soak test: long replays through the real engine, watching memory, CPU per message and the books.

Replay results are never used to choose, tune, or rank strategies. The soak only checks the plumbing. It runs the
plumbing configuration of PLAN section 3, configs/soak_plumbing.toml, whose small start equity, short parent
requirements and short forced timer exist only so that deaths, mutated births and new generations happen within a
few replayed days. That file is never a strategy setting.

The input is every recording in data/raw joined end to end, followed by a synthetic recording of several days with
a full size book snapshot at every resync and one outage longer than max_resume_gap_minutes, so the gap flatten
closes real positions. A file the recorder is still writing is copied up to its size at the start and read only to
its last complete line, so the recorder is never disturbed. Where two recordings do not continue one connection, a
close marker ends the open connection and the next one gets new connection ids, so the adapter resets exactly as it
does on a live reconnect.

This process reads the recordings, generates the synthetic one and pipes every line to a child process that holds
only the engine and the adapter, so the child's memory is what a live run holds. At the end of every replayed hour
the child stops reading while this process reads its current resident memory from outside, because macOS only
exposes peak memory to the process itself. The plan's rule is that memory at the end is no more than 5 MB above
memory at the end of the first replayed hour. Only a Linux reading from /proc, the number the live heartbeat shows,
decides it. macOS keeps freed pages resident for a while and returns them in bursts, so on a Mac the rule is
reported as inconclusive, unless the memory the allocators report in use grew past the limit too, which fails on
any system. The run also fails if any balance check fails or the child does not exit cleanly. Everything is written
into a new folder under data/soak. The exit code is 0 for passed, 1 for failed and 3 for inconclusive.

    python scripts/soak_test.py [--synthetic-hours 120] [--resync-minutes 120] [--out data/soak/NAME]
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import itertools
import json
import logging
import math
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
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from btc_survivor.account import INVARIANT_TOLERANCE
from btc_survivor.clock import MS_PER_HOUR, iso
from btc_survivor.config import load_config
from btc_survivor.engine import Engine
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, load_adapter
from btc_survivor.logsetup import DATE_FORMAT, FORMAT
from btc_survivor.recorder import read_recording, recording_files
from btc_survivor.replay import NEVER_FOR_SELECTION, replay_config
from btc_survivor.store import Store
from btc_survivor.synthetic import Fault, SyntheticFeed

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "soak_plumbing.toml"
DEFAULT_RAW = ROOT / "data" / "raw"
SOAK_ROOT = ROOT / "data" / "soak"
PLUMBING_LABEL = "plumbing configuration, never a strategy setting"
MAX_GROWTH_MB = 5.0
SLOPE_HOURS = 24.0  # the memory slope is fitted over the last day of replayed time
# The synthetic soak recording's seed tag. seeds.py holds the bot, population and reconnect tags, and it is a
# version 1 module the live runs depend on, so this tag stays here and tests/test_soak.py checks it never collides.
SYNTHETIC_TAG = 4
SYNTHETIC_START_MS = 1_791_000_000_000  # used only when there are no real recordings to continue from
JOIN_GAP_MS = 2_000  # the synthetic recording starts this long after the real ones end, like a quick reconnect
FULL_BIDS, FULL_ASKS = 21_600, 19_300  # level counts of a real Coinbase BTC-USD snapshot in October 2026
DEEP_GAP = 0.05  # deep synthetic levels start 5 percent from the touch, far beyond any move between resyncs
DEEP_SPAN = 0.45  # and spread over 45 percent of the price
FAULT_KINDS = ("disconnect", "gap", "crossed", "silence")  # synthetic.py caps a silence at stale_s + 1 s
BIG_MESSAGE_BYTES = 1_000_000
CHUNK = 1 << 18
EXIT_CODES = {"passed": 0, "failed": 1, "inconclusive": 3}

Line = tuple[str, str, int, int]
log = logging.getLogger("soak")


# Choosing and freezing the recordings


@dataclass
class Recording:
    source: Path
    path: Path  # what the replay reads, the source itself or a frozen copy of its first bytes
    status: str


def gzip_is_complete(path: Path, size: int, copy_to: Path | None = None) -> bool:
    """Whether the first size bytes of path are whole gzip members. With copy_to, those exact bytes are copied there.

    The file is opened read only and read once, so a recorder still appending to it is not disturbed.
    """
    decomp = zlib.decompressobj(wbits=31)
    ok, left = True, size
    sink = None if copy_to is None else copy_to.open("wb")
    try:
        with path.open("rb") as fh:
            while left > 0:
                chunk = fh.read(min(CHUNK, left))
                if not chunk:
                    break
                left -= len(chunk)
                if sink is not None:
                    sink.write(chunk)
                while chunk and ok:
                    try:
                        decomp.decompress(chunk)
                    except zlib.error:
                        ok = False
                        break
                    chunk = decomp.unused_data if decomp.eof else b""
                    if chunk:
                        decomp = zlib.decompressobj(wbits=31)  # the next gzip member
    finally:
        if sink is not None:
            sink.close()
    return ok and left == 0 and decomp.eof


def select_recordings(raw: Path, input_dir: Path) -> list[Recording]:
    """Every recording in raw in time order. The newest one, unless already complete, is frozen as a copy first."""
    files = recording_files([raw]) if raw.is_dir() else []
    chosen = []
    for i, path in enumerate(files):
        size = path.stat().st_size
        copy = None
        if i == len(files) - 1:
            input_dir.mkdir(parents=True, exist_ok=True)
            copy = input_dir / path.name
        if gzip_is_complete(path, size, copy):
            if copy is not None:
                copy.unlink()
            chosen.append(Recording(path, path, "complete"))
        elif copy is not None:
            chosen.append(Recording(path, copy, f"still being written, its first {size} bytes were copied and are "
                                                 "read to the last complete line"))
        else:
            chosen.append(Recording(path, path, "cut short, read to the last complete line"))
    return chosen


# Joining recordings end to end


@dataclass
class JoinState:
    segment: str = ""
    lines: list[int] = field(default_factory=list)  # lines taken from each source
    markers: int = 0  # close and open markers added at joins
    clamped: int = 0  # receive times raised to keep time from going backwards
    last_ms: int | None = None
    open_conn: int | None = None  # the joined id of a connection still open
    top_conn: int = 0
    offset: int = 0


Source = tuple[str, Callable[[JoinState], Iterable[Line]]]


def join(sources: list[Source], state: JoinState) -> Iterator[Line]:
    """Recordings end to end. A file that continues the open connection, like a daily rotation, follows directly.

    Otherwise the open connection is closed with a marker, a file that starts inside a connection gets an open
    marker, and connection ids are renumbered above every id used so far. Connection ids start at 1.
    """
    for segment, make in sources:
        state.segment = segment
        state.lines.append(0)
        first = True
        for kind, raw, recv_ms, conn in make(state):
            if state.last_ms is not None and recv_ms < state.last_ms:
                recv_ms = state.last_ms
                state.clamped += 1
            if first:
                first = False
                continues = kind != OPEN and state.open_conn is not None and conn + state.offset == state.open_conn
                if not continues and state.last_ms is not None:
                    if state.open_conn is not None:
                        state.markers += 1
                        yield CLOSE, "soak join, the next recording starts", state.last_ms, state.open_conn
                        state.open_conn = None
                    state.offset = state.top_conn
                if not continues and kind != OPEN:
                    state.markers += 1
                    state.open_conn = conn + state.offset
                    yield OPEN, "", recv_ms, state.open_conn
            joined = conn + state.offset
            if kind == OPEN:
                state.open_conn = joined
            elif kind == CLOSE and joined == state.open_conn:
                state.open_conn = None
            state.top_conn = max(state.top_conn, joined)
            state.last_ms = recv_ms
            state.lines[-1] += 1
            yield kind, raw, recv_ms, joined


# The synthetic recording


def full_size_snapshot(raw: str, bids: int, asks: int, rng: np.random.Generator) -> str:
    """Pad a synthetic level2 snapshot with deep levels until it has bids and asks levels, the size of a real one.

    The deep levels sit far from the touch, so the synthetic updates near the mid never cross or reach them.
    """
    msg = json.loads(raw)
    event = msg["events"][0]
    updates = event["updates"]
    stamp = updates[0]["event_time"]
    near = {"bid": [], "offer": []}
    for u in updates:
        near[u["side"]].append(float(u["price_level"]))
    extra = []
    for side, count, edge, sign in (("bid", bids - len(near["bid"]), min(near["bid"]), -1),
                                    ("offer", asks - len(near["offer"]), max(near["offer"]), 1)):
        if count <= 0:
            continue
        prices = edge * (1 + sign * DEEP_GAP) + sign * edge * DEEP_SPAN / count * np.arange(count)
        sizes = rng.uniform(0.0001, 2.0, count)
        extra += [{"side": side, "event_time": stamp, "price_level": f"{p:.2f}", "new_quantity": f"{s:.8f}"}
                  for p, s in zip(prices.tolist(), sizes.tolist())]
    event["updates"] = updates + extra
    return json.dumps(msg, separators=(",", ":"))


def resync_faults(hours: float, every_minutes: float, rng: np.random.Generator) -> tuple[Fault, ...]:
    """One fault about every every_minutes, cycling through the kinds that make the client resync."""
    seconds, every = hours * 3600, every_minutes * 60
    faults = []
    for k in itertools.count(1):
        at = k * every + float(rng.uniform(-0.25, 0.25)) * every
        if at >= seconds:
            break
        kind = FAULT_KINDS[(k - 1) % len(FAULT_KINDS)]
        faults.append(Fault(at, kind, duration_s=30.0 if kind == "silence" else 0.0))
    return tuple(faults)


def synthetic_lines(seed: int, start_ms: int, hours: float, every_minutes: float, mid: float, bids: int, asks: int,
                    product: str, outage: tuple[float, float] | None = None) -> Iterator[Line]:
    """A synthetic recording of hours from synthetic.py, with every book snapshot padded to full size.

    outage is (start hour, minutes): the connection closes there and reconnects only after that many minutes.
    """
    rng = np.random.default_rng(seed)
    faults = resync_faults(hours, every_minutes, rng)
    if outage is not None and outage[0] < hours:
        start, length = outage[0] * 3600, outage[1] * 60  # other faults stay clear of it, so it cannot be skipped
        faults = tuple(f for f in faults if not start - 300 <= f.at_s <= start + length + 300)
        faults += (Fault(start, "disconnect", duration_s=length),)
    feed = SyntheticFeed(int(rng.integers(2**62)), start_ms, product=product, mid=mid)
    for kind, raw, recv_ms, conn in feed.lines(hours * 3600, faults):
        if kind == MESSAGE and raw.startswith('{"channel":"l2_data"') and '"type":"snapshot"' in raw[:300]:
            raw = full_size_snapshot(raw, bids, asks, rng)
        yield kind, raw, recv_ms, conn


def synthetic_seed(master_seed: int) -> int:
    return int(np.random.SeedSequence([master_seed, SYNTHETIC_TAG]).generate_state(1)[0])


# Memory, read from outside the replay process and from its allocators


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime")]


def process_memory(pid: int) -> dict:
    """Current resident memory of another process in MB, its physical footprint on macOS, and where they came from."""
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
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return {"rss_mb": int(out) / 1024 if out else None, "footprint_mb": None, "source": "ps"}


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


class _MallocStatistics(ctypes.Structure):  # macOS malloc_statistics_t
    _fields_ = [("blocks_in_use", ctypes.c_uint), ("size_in_use", ctypes.c_size_t),
                ("max_size_in_use", ctypes.c_size_t), ("size_allocated", ctypes.c_size_t)]


class _MallInfo2(ctypes.Structure):  # glibc struct mallinfo2
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def malloc_in_use_mb() -> float | None:
    """Bytes the C allocator has handed out and not taken back, in MB, or None where it cannot be read.

    Freed memory the allocator still holds is left out, so this separates memory in use from memory retained.
    """
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
    """CPython's small object arenas in use and the bytes in their allocated blocks, from its allocator statistics.

    The arenas are mapped outside the C allocator, so their blocks add to malloc_in_use_mb without counting twice.
    """
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


# The frames the parent pipes to the replay process


FRAME = struct.Struct("<BBqqI")  # frame type, segment, receive time, connection id, payload bytes
LINE_CODES = {OPEN: 1, MESSAGE: 2, CLOSE: 3}
LINE_KINDS = {code: kind for kind, code in LINE_CODES.items()}
HOUR, MID, END = 10, 11, 12  # requests that the replay process answers with one JSON line
SEGMENTS = ("real", "synthetic")


class ChildGone(RuntimeError):
    """Raised when the replay process stops reading or answering."""


class ReplayProcess:
    """The parent's end of the pipe: lines go in as frames, and each request is answered once everything before it
    has been replayed, so the child is idle, blocked on its next read, whenever its memory is read."""

    def __init__(self, spec_path: Path) -> None:
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
        self.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def send(self, segment: int, line: Line) -> None:
        kind, raw, recv_ms, conn = line
        data = raw.encode("utf-8")
        try:
            self.proc.stdin.write(FRAME.pack(LINE_CODES[kind], segment, recv_ms, conn, len(data)))
            self.proc.stdin.write(data)
        except (BrokenPipeError, OSError) as exc:
            raise ChildGone(f"the replay process stopped reading, {exc}") from exc

    def ask(self, request: int, ts_ms: int = 0, segment: int = 0) -> dict:
        try:
            self.proc.stdin.write(FRAME.pack(request, segment, ts_ms, 0, 0))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ChildGone(f"the replay process stopped reading, {exc}") from exc
        text = self.proc.stdout.readline()
        if not text:
            raise ChildGone("the replay process ended without answering")
        return json.loads(text)

    def close(self) -> int:
        """Close the pipe, which lets the replay process close its store and exit, and return its exit code."""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        return self.proc.wait()


# The replay process


class _LogCounter(logging.Handler):
    """Counts the log lines a healthy run never shows."""

    WATCH = ("invariant violation", "slow minute step", "slow database write")

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.counts = {w: 0 for w in self.WATCH}

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        for w in self.WATCH:
            if text.startswith(w):
                self.counts[w] += 1


def _last_mid(engine: Engine | None) -> float | None:
    if engine is None:
        return None
    book = engine.market.book
    if book.valid and book.mid is not None:
        return book.mid
    series = engine.series
    if series.last_index is not None:
        for back in range(series.capacity):
            value = float(series.mid[(series.last_index - back) % series.capacity])
            if not math.isnan(value):
                return value
    return None


def check_books(engine: Engine) -> tuple[int, list[str]]:
    """The account identities for every bot with books open, read from memory between two messages."""
    book = engine.market.book
    usable = book.valid and book.mid is not None
    checked, problems = 0, []
    for bot in engine.all_bots():
        if bot.account.is_open and not usable:
            continue  # no price to mark an open position at, so the engine's next snapshot checks it
        checked += 1
        problems += [f"bot {bot.bot_id}: {e}" for e in bot.account.invariant_errors(bot.mark_price(book))]
    return checked, problems


def _timed(engine: Engine, name: str, totals: dict) -> None:
    """Wrap one engine method on this instance only, adding its CPU time to totals[name]."""
    inner = getattr(engine, name)

    def wrapper(*args):
        started = time.process_time()
        try:
            return inner(*args)
        finally:
            totals[name] += time.process_time() - started

    setattr(engine, name, wrapper)


def _read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise SystemExit("the soak parent closed the pipe early")
    return data


def worker(spec_path: Path) -> int:
    """Replay the lines piped in by the parent, answering its requests at every replayed hour and at the end."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out = Path(spec["out"])
    leak = int(spec.get("leak_kb_per_message", 0) * 1024)  # a test hook that proves a leak fails the soak
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr  # nothing else may write into the answer channel
    inp = sys.stdin.buffer

    def answer(message: dict) -> None:
        proto.write(json.dumps(message) + "\n")
        proto.flush()

    cfg = load_config(spec["config"])
    handler = logging.FileHandler(out / "soak.log", encoding="utf-8")
    formatter = logging.Formatter(FORMAT, DATE_FORMAT)
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    counter = _LogCounter()
    root = logging.getLogger()
    root.handlers[:] = [handler, counter]
    root.setLevel(cfg.run.log_level)

    run_cfg = replay_config(cfg, out / "db")
    seg = {name: {"messages": 0, "markers": 0, "engine_cpu_s": 0.0, "handling_cpu_s": 0.0, "big_messages": 0,
                  "big_handling_cpu_s": 0.0, "biggest_bytes": 0, "first_ms": None, "last_ms": None}
           for name in SEGMENTS}
    parts = {"_minute": 0.0, "_persistence_point": 0.0}
    leaked: list[bytes] = []
    engine: Engine | None = None
    engine_cpu, messages, checked_total, problems_total = 0.0, 0, 0, []
    cpu_start, wall_start, first_ms = time.process_time(), time.monotonic(), None
    store = Store.open(run_cfg.db_path)
    try:
        def snapshot(kind: str, ts_ms: int, segment: str) -> dict:
            nonlocal checked_total
            checked, problems = check_books(engine)
            checked_total += checked
            problems_total.extend(problems[:20])
            living = [b for b in engine.population.learners if b.state != "dead"]
            return {"type": kind, "hours": (ts_ms - first_ms) / MS_PER_HOUR, "ts": iso(ts_ms), "segment": segment,
                    "messages": messages, "engine_cpu_s": engine_cpu,
                    "worker_cpu_s": time.process_time() - cpu_start, "wall_s": time.monotonic() - wall_start,
                    "peak_rss_mb": peak_rss_mb(), "malloc_mb": malloc_in_use_mb(), **pymalloc_stats(),
                    "live_blocks": sys.getallocatedblocks(), "living": len(living), "deaths": engine.deaths,
                    "generation": max(b.generation for b in engine.slot_bots()), "checked": checked,
                    "problems": problems}

        while True:
            code, segment, recv_ms, conn, size = FRAME.unpack(_read_exact(inp, FRAME.size))
            if code in LINE_KINDS:
                kind = LINE_KINDS[code]
                raw = _read_exact(inp, size).decode("utf-8") if size else ""
                if engine is None:
                    engine = Engine.fresh(run_cfg, store, load_adapter(run_cfg.feed), recv_ms)
                    first_ms = recv_ms
                    for name in parts:
                        _timed(engine, name, parts)
                started, steps = time.process_time(), sum(parts.values())
                engine.on_line(kind, raw, recv_ms, conn)
                spent = time.process_time() - started
                handling = spent - (sum(parts.values()) - steps)  # leaves out minute steps and persistence it set off
                engine_cpu += spent
                s = seg[SEGMENTS[segment]]
                s["engine_cpu_s"] += spent
                s["handling_cpu_s"] += handling
                if s["first_ms"] is None:
                    s["first_ms"] = recv_ms
                s["last_ms"] = recv_ms
                if kind == MESSAGE:
                    messages += 1
                    s["messages"] += 1
                    if len(raw) >= BIG_MESSAGE_BYTES:
                        s["big_messages"] += 1
                        s["big_handling_cpu_s"] += handling
                        s["biggest_bytes"] = max(s["biggest_bytes"], len(raw))
                    if leak:
                        leaked.append(b"\x01" * leak)
                else:
                    s["markers"] += 1
            elif code == MID:
                answer({"type": "mid", "mid": _last_mid(engine)})
            elif engine is None:
                raise SystemExit("the soak has no lines to replay")
            elif code == HOUR:
                answer(snapshot("hour", recv_ms, SEGMENTS[segment]))
            elif code == END:
                started = time.process_time()
                engine.finish()
                engine_cpu += time.process_time() - started
                end = snapshot("end", engine.now_ms, SEGMENTS[segment])
                end.update(first_ts=iso(first_ms), last_ts=iso(engine.now_ms), segments=seg,
                           minute_step_cpu_s=parts["_minute"], persistence_cpu_s=parts["_persistence_point"],
                           log_counts=counter.counts, checked_total=checked_total, problems_total=problems_total)
                answer(end)
                inp.read()  # stay idle while the parent reads memory, until it closes the pipe
                return 0
            else:
                raise SystemExit(f"unknown frame type {code}")
    finally:
        store.close()


# Checks on the finished database


LEDGER_CHECKS = {
    "equity snapshots, cash equals start plus realized minus fees, funding and rent": (
        "SELECT count(*), coalesce(sum(abs(cash - (:start + realized_gross - fees_total - funding_total - rent_total))"
        " > :tol), 0) FROM equity_snapshots"),
    "equity snapshots, equity equals cash plus unrealized": (
        "SELECT count(*), coalesce(sum(abs(equity - cash - unrealized) > :tol), 0) FROM equity_snapshots"),
    "deaths, the five parts and the writeoff explain the final equity": (
        "SELECT count(*), coalesce(sum(abs(final_equity - (:start - fees_total - funding_total - rent_total"
        " - stop_loss_total - other_losses_total + writeoff)) > :tol), 0) FROM deaths"),
    "deaths, final equity is never negative": (
        "SELECT count(*), coalesce(sum(final_equity < -:tol), 0) FROM deaths"),
    "closed trades, net equals gross minus both fees and funding": (
        "SELECT count(*), coalesce(sum(abs(pnl_net - (pnl_gross - fee_open - fee_close - funding)) > :tol), 0)"
        " FROM trades WHERE closed_at IS NOT NULL"),
    "dead bots, each has exactly one death row": (
        "SELECT count(*), coalesce(sum(d.bot_id IS NULL), 0) FROM bots b LEFT JOIN deaths d USING (bot_id)"
        " WHERE b.state = 'dead'"),
    "mutated births, each has a lineage row and is one generation below its parent": (
        "SELECT count(*), coalesce(sum(l.bot_id IS NULL OR p.bot_id IS NULL OR b.generation != p.generation + 1"
        " OR l.parent_id != b.parent_id), 0) FROM bots b LEFT JOIN lineage l USING (bot_id)"
        " LEFT JOIN bots p ON p.bot_id = b.parent_id WHERE b.origin = 'mutated'"),
}


def database_report(db: Path, start_equity: float) -> dict:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        params = {"start": start_equity, "tol": INVARIANT_TOLERANCE}
        ledger = {name: dict(zip(("rows", "failures"), conn.execute(sql, params).fetchone()))
                  for name, sql in LEDGER_CHECKS.items()}

        def scalar(sql: str) -> int:
            return conn.execute(sql).fetchone()[0]

        return {
            "ledger": ledger,
            "engine_snapshot_checks": scalar("SELECT count(*) FROM equity_snapshots"),
            "engine_violations": scalar("SELECT count(*) FROM feed_events WHERE kind = 'invariant_violation'"),
            "births": dict(conn.execute("SELECT origin, count(*) FROM bots GROUP BY origin ORDER BY origin")),
            "mutated_births": scalar("SELECT count(*) FROM bots WHERE origin = 'mutated'"),
            "lineage_rows": scalar("SELECT count(*) FROM lineage"),
            "highest_generation": scalar("SELECT coalesce(max(generation), 0) FROM bots"),
            "deaths": scalar("SELECT count(*) FROM deaths"),
            "learner_deaths": scalar("SELECT count(*) FROM deaths JOIN bots USING (bot_id) WHERE origin != 'control'"),
            "causes": dict(conn.execute("SELECT cause, count(*) FROM deaths GROUP BY cause ORDER BY cause")),
            "trades": scalar("SELECT count(*) FROM trades"),
            "closed_trades": scalar("SELECT count(*) FROM trades WHERE closed_at IS NOT NULL"),
            "forced_trades": scalar("SELECT count(*) FROM trades WHERE forced = 1"),
            "exit_reasons": dict(conn.execute("SELECT coalesce(exit_reason, 'open'), count(*) FROM trades"
                                              " GROUP BY 1 ORDER BY 1")),
            "decisions": scalar("SELECT count(*) FROM decisions"),
            "feed_events": dict(conn.execute("SELECT kind, count(*) FROM feed_events GROUP BY kind ORDER BY kind")),
            "flatten_events": [row[0] for row in conn.execute(
                "SELECT ts || ' ' || detail FROM feed_events WHERE kind = 'resume' ORDER BY ts")],
            "healthy_minutes": scalar("SELECT count(*) FROM candles WHERE valid = 1"),
        }
    finally:
        conn.close()


def _slope(points: list[tuple[float, float]]) -> float | None:
    """Least squares slope of (hours, MB) points in MB per hour, or None with fewer than three points."""
    if len(points) < 3:
        return None
    x, y = np.array(points, dtype=float).T
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _growth(name: str, samples: list[dict], end: dict, limit_mb: float) -> dict | None:
    """The plan's comparison for one memory measure, plus the worst later reading and the recent slope."""
    first, last = samples[0].get(name) if samples else None, end.get(name)
    if first is None or last is None:
        return None
    later = [(s["hours"], s[name]) for s in samples[1:] if s.get(name) is not None] + [(end["hours"], last)]
    recent = [(h, v) for h, v in [(s["hours"], s[name]) for s in samples] + [(end["hours"], last)]
              if v is not None and h >= end["hours"] - SLOPE_HOURS]
    slope = _slope(recent)
    return {"first_hour_mb": round(first, 2), "end_mb": round(last, 2), "growth_mb": round(last - first, 2),
            "rule_held": last - first <= limit_mb, "largest_growth_mb": round(max(v for _, v in later) - first, 2),
            "later_readings": len(later), "readings_over_limit": sum(v - first > limit_mb for _, v in later),
            "slope_mb_per_hour_last_day": None if slope is None else round(slope, 3)}


def memory_verdict(samples: list[dict], end: dict | None, limit_mb: float = MAX_GROWTH_MB) -> dict:
    """The plan's rule: resident memory at the end at most limit_mb above it at the end of the first replayed hour.

    A Linux reading from /proc decides it. Elsewhere the rule only fails when the memory the allocators report in
    use grew past the limit as well, and otherwise the result is inconclusive.
    """
    end = end or {}
    rss = _growth("rss_mb", samples, end, limit_mb)
    if rss is None:
        return {"outcome": "failed", "limit_mb": limit_mb,
                "reason": "no memory reading at the end of a first replayed hour or at the end"}
    source = end.get("source")
    for s in samples:
        s["in_use_mb"] = _in_use(s)
    end["in_use_mb"] = _in_use(end)
    result = {"source": source, "decisive": source == "proc", "limit_mb": limit_mb, **rss,
              "footprint": _growth("footprint_mb", samples, end, limit_mb),
              "in_use": _growth("in_use_mb", samples, end, limit_mb)}
    in_use = result["in_use"]
    if result["decisive"]:
        outcome = "passed" if rss["rule_held"] else "failed"
    elif not rss["rule_held"] and in_use is not None and not in_use["rule_held"]:
        outcome = "failed"
    else:
        outcome = "inconclusive"
    result["outcome"] = outcome
    grew = f"memory grew {rss['growth_mb']} MB after the first hour against the {limit_mb} MB limit"
    if outcome == "passed":
        result["reason"] = f"{grew}, read from /proc"
    elif outcome == "failed" and result["decisive"]:
        result["reason"] = f"{grew}, read from /proc"
    elif outcome == "failed":
        result["reason"] = f"{grew}, and memory in use by the allocators grew {in_use['growth_mb']} MB"
    else:
        others = "".join(f", {label} grew {g['growth_mb']} MB" for label, g in (
            ("the physical footprint", result["footprint"]), ("memory in use", in_use)) if g is not None)
        result["reason"] = (f"{grew}, read with {source}, which keeps freed pages resident{others}, so only a "
                            "Linux reading decides the rule")
    return result


def _in_use(sample: dict) -> float | None:
    if sample.get("malloc_mb") is None or sample.get("pymalloc_mb") is None:
        return None
    return sample["malloc_mb"] + sample["pymalloc_mb"]


# The parent


def run_soak(args: argparse.Namespace) -> dict:
    out = args.out or SOAK_ROOT / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = out.resolve()
    raw = args.raw.resolve()
    if out == raw or raw in out.parents:
        raise SystemExit(f"{out} is inside the recordings folder, the soak never writes there")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty, the soak only writes into a new or empty folder")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    outage_minutes = (cfg.persistence.max_resume_gap_minutes + 5 if args.outage_minutes is None
                      else args.outage_minutes)
    outage = None if args.outage_at_hours < 0 else (args.outage_at_hours, outage_minutes)
    recordings = [] if args.skip_real else select_recordings(raw, out / "input")
    spec = {
        "out": str(out), "config": str(args.config.resolve()), "recordings": [str(r.path) for r in recordings],
        "synthetic": {"hours": args.synthetic_hours, "resync_minutes": args.resync_minutes,
                      "bids": args.snapshot_bids, "asks": args.snapshot_asks, "outage": outage},
        "leak_kb_per_message": args.leak_kb_per_message,
    }
    (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"soak into {out}, {PLUMBING_LABEL}. {NEVER_FOR_SELECTION}", flush=True)
    for r in recordings:
        print(f"  recording {r.source.name}: {r.status}", flush=True)

    started, cpu_started = time.monotonic(), time.process_time()
    child = ReplayProcess(out / "spec.json")
    state = JoinState()

    def synthetic_source(st: JoinState) -> Iterable[Line]:
        start, mid = SYNTHETIC_START_MS, None
        if st.last_ms is not None:
            start, mid = st.last_ms + JOIN_GAP_MS, child.ask(MID)["mid"]
        mid = mid or 80_000.0
        print(f"  synthetic recording starts at {iso(start)} from mid {mid:.2f}", flush=True)
        return synthetic_lines(synthetic_seed(cfg.run.master_seed), start, args.synthetic_hours, args.resync_minutes,
                               mid, args.snapshot_bids, args.snapshot_asks, cfg.feed.product, outage)

    sources: list[Source] = [("real", lambda st, p=r.path: read_recording([p])) for r in recordings]
    if args.synthetic_hours > 0:
        sources.append(("synthetic", synthetic_source))
    samples, end, failure = [], None, None
    try:
        next_hour, hour = None, 0
        for line in join(sources, state):
            segment = SEGMENTS.index(state.segment)
            if next_hour is None:
                next_hour = line[2] + MS_PER_HOUR
            while line[2] >= next_hour:
                hour += 1
                sample = {**child.ask(HOUR, next_hour, segment), "hour": hour, **process_memory(child.proc.pid)}
                samples.append(sample)
                print(_progress(sample), flush=True)
                next_hour += MS_PER_HOUR
            child.send(segment, line)
        if next_hour is None:
            raise ChildGone("the soak has no lines to replay")
        end = {**child.ask(END, 0, SEGMENTS.index(state.segment)), **process_memory(child.proc.pid)}
    except ChildGone as exc:
        failure = str(exc)
    code = child.close()
    wall, parent_cpu = time.monotonic() - started, time.process_time() - cpu_started
    report = build_report(cfg, args, out, recordings, state, samples, end, code, wall, parent_cpu, failure)
    write_outputs(out, report, samples)
    print(summary(report), flush=True)
    return report


def _progress(m: dict) -> str:
    foot = "" if m.get("footprint_mb") is None else f" footprint {m['footprint_mb']:.1f} MB"
    in_use = "" if _in_use(m) is None else f" in use {_in_use(m):.1f} MB"
    return (f"  hour {m['hour']:>3} {m['ts'][:16]} {m['segment']:<9} rss {m['rss_mb']:.1f} MB{foot}{in_use} "
            f"arenas {m['arenas']} messages {m['messages']:,} living {m['living']} deaths {m['deaths']} "
            f"generation {m['generation']}")


def build_report(cfg, args: argparse.Namespace, out: Path, recordings: list[Recording], state: JoinState,
                 samples: list[dict], end: dict | None, code: int, wall: float, parent_cpu: float,
                 failure: str | None) -> dict:
    report = {
        "label": PLUMBING_LABEL, "never_for_selection": NEVER_FOR_SELECTION, "config": str(args.config),
        "out": str(out), "worker_exit_code": code, "wall_seconds": round(wall, 1),
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "recordings": [{"source": str(r.source), "read": str(r.path), "status": r.status, "lines": n}
                       for r, n in zip(recordings, state.lines)],
        "synthetic": {"hours": args.synthetic_hours, "resync_minutes": args.resync_minutes,
                      "snapshot_bids": args.snapshot_bids, "snapshot_asks": args.snapshot_asks,
                      "lines": state.lines[-1] if args.synthetic_hours > 0 and len(state.lines) > len(recordings)
                      else 0},
    }
    if args.leak_kb_per_message:
        report["leak_kb_per_message"] = args.leak_kb_per_message
    if end is None or code != 0:
        why = failure or f"the replay process failed with exit code {code}"
        report.update(outcome="failed", passed=False, verdict=f"failed, {why}, see soak.log")
        return report
    db = database_report(out / "db" / "btc_survivor.db", cfg.trading.start_equity)
    memory = memory_verdict(samples, end, args.max_growth_mb)
    memory["peak_rss_mb"] = round(end["peak_rss_mb"], 2)
    memory["max_sampled_mb"] = round(max([s["rss_mb"] for s in samples] + [end["rss_mb"]]), 2)
    if samples:  # live blocks flat while arenas rise means fragmentation, not a leak
        memory["live_blocks"] = {"first_hour": samples[0]["live_blocks"], "end": end["live_blocks"]}
        memory["arenas"] = {"first_hour": samples[0]["arenas"], "end": end["arenas"],
                            "most": max([s["arenas"] or 0 for s in samples] + [end["arenas"] or 0])}
    replayed_s = end["hours"] * 3600
    segments = {}
    for name, s in end["segments"].items():
        if s["messages"] or s["markers"]:
            span = (s["last_ms"] - s["first_ms"]) / 1000
            small = s["messages"] - s["big_messages"]
            segments[name] = {
                "messages": s["messages"], "markers": s["markers"], "hours": round(span / 3600, 2),
                "messages_per_second": round(s["messages"] / span, 2) if span else None,
                "engine_cpu_s": round(s["engine_cpu_s"], 2),
                "engine_us_per_message": round(s["engine_cpu_s"] / s["messages"] * 1e6, 1) if s["messages"] else None,
                "full_size_snapshots": s["big_messages"], "biggest_message_bytes": s["biggest_bytes"],
                "full_size_snapshot_ms_each": round(s["big_handling_cpu_s"] / s["big_messages"] * 1e3, 1)
                if s["big_messages"] else None,
                "other_message_us_each": round((s["handling_cpu_s"] - s["big_handling_cpu_s"]) / small * 1e6, 1)
                if small else None}
    step, persist = end["minute_step_cpu_s"], end["persistence_cpu_s"]
    handling = end["engine_cpu_s"] - step - persist
    big = sum(s["big_handling_cpu_s"] for s in end["segments"].values())
    bigs = sum(s["big_messages"] for s in end["segments"].values())
    cpu = {
        "engine_share_of_one_core_at_live_speed": round(end["engine_cpu_s"] / replayed_s, 5),
        "engine_cpu_s": round(end["engine_cpu_s"], 2),
        "engine_us_per_message": round(end["engine_cpu_s"] / end["messages"] * 1e6, 1),
        "minute_step_cpu_s": round(step, 2),
        "minute_step_ms_per_healthy_minute": round(step / db["healthy_minutes"] * 1e3, 2)
        if db["healthy_minutes"] else None,
        "persistence_cpu_s": round(persist, 2),
        "message_handling_cpu_s": round(handling, 2),
        "message_handling_us_per_message": round(handling / end["messages"] * 1e6, 1),
        "full_size_snapshot_cpu_s": round(big, 2),
        "full_size_snapshot_ms_each": round(big / bigs * 1e3, 1) if bigs else None,
        "other_message_us_each": round((handling - big) / (end["messages"] - bigs) * 1e6, 1),
        "replay_process_cpu_s": round(end["worker_cpu_s"], 2),
        "parent_cpu_s": round(parent_cpu, 2),  # reading the recordings, generating the synthetic one, piping
        "segments": segments,
    }
    hourly_problems = sum(len(s["problems"]) for s in samples) + len(end["problems"])
    ledger_failures = sum(c["failures"] for c in db["ledger"].values())
    checks = {
        "engine_checks_at_equity_snapshots": db["engine_snapshot_checks"],
        "engine_violations": db["engine_violations"],
        "violation_log_lines": end["log_counts"]["invariant violation"],
        "hourly_bot_checks": end["checked_total"], "hourly_problems": hourly_problems,
        "hourly_problem_examples": end["problems_total"][:10],
        "ledger": db["ledger"], "ledger_failures": ledger_failures,
    }
    checks["passed"] = (checks["engine_violations"] == 0 and checks["violation_log_lines"] == 0
                        and hourly_problems == 0 and ledger_failures == 0)
    report.update(
        replay={"first_ts": end["first_ts"], "last_ts": end["last_ts"], "hours": round(end["hours"], 2),
                "messages": end["messages"], "join_markers": state.markers, "times_clamped": state.clamped,
                "healthy_minutes": db["healthy_minutes"], "feed_events": db["feed_events"],
                "resume_events": db["flatten_events"],
                "slow_minute_steps": end["log_counts"]["slow minute step"],
                "slow_database_writes": end["log_counts"]["slow database write"]},
        memory=memory, cpu=cpu, checks=checks,
        evolution={k: db[k] for k in ("births", "mutated_births", "lineage_rows", "highest_generation", "deaths",
                                      "learner_deaths", "causes", "trades", "closed_trades", "forced_trades",
                                      "exit_reasons", "decisions")},
    )
    reasons = []
    if memory["outcome"] != "passed":
        reasons.append(memory["reason"])
    if not checks["passed"]:
        reasons.append("a balance check failed")
    if not checks["passed"] or memory["outcome"] == "failed":
        outcome = "failed"
    else:
        outcome = memory["outcome"]
    report.update(outcome=outcome, passed=outcome == "passed")
    report["verdict"] = outcome if not reasons else f"{outcome}, " + ", ".join(reasons)
    return report


def write_outputs(out: Path, report: dict, samples: list[dict]) -> None:
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["hour", "hours", "ts", "segment", "rss_mb", "footprint_mb", "in_use_mb", "malloc_mb", "pymalloc_mb",
              "arenas", "live_blocks", "peak_rss_mb", "messages", "engine_cpu_s", "worker_cpu_s", "living", "deaths",
              "generation"]
    with (out / "memory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for s in samples:
            writer.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})


def _mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def summary(report: dict) -> str:
    if "memory" not in report:
        return f"soak {report['verdict']}"
    m, c, k, e, r = report["memory"], report["cpu"], report["checks"], report["evolution"], report["replay"]
    ledger = k["ledger"]
    lines = [
        f"soak {report['verdict']} ({PLUMBING_LABEL})",
        f"  replayed {r['hours']} hours from {r['first_ts']} to {r['last_ts']}, {r['messages']:,} messages, "
        f"{r['join_markers']} join markers, {r['healthy_minutes']} healthy minutes",
    ]
    if "first_hour_mb" in m:
        lines.append(f"  memory read with {m['source']}, {m['first_hour_mb']} MB after the first hour, {m['end_mb']} MB "
                     f"at the end, growth {m['growth_mb']} MB against a {m['limit_mb']} MB limit, largest growth "
                     f"{m['largest_growth_mb']} MB, {m['readings_over_limit']} of {m['later_readings']} later readings "
                     f"over the limit, slope over the last day {_mb(m['slope_mb_per_hour_last_day'])} MB an hour, "
                     f"peak {m['peak_rss_mb']} MB")
        for name in ("footprint", "in_use"):
            g = m.get(name)
            if g is not None:
                lines.append(f"    {name.replace('_', ' ')} {g['first_hour_mb']} MB then {g['end_mb']} MB, growth "
                             f"{g['growth_mb']} MB, largest {g['largest_growth_mb']} MB, slope over the last day "
                             f"{_mb(g['slope_mb_per_hour_last_day'])} MB an hour")
        lines.append(f"    small object arenas {m['arenas']['first_hour']} after the first hour, {m['arenas']['end']} "
                     f"at the end, {m['arenas']['most']} at most")
    lines.append(f"  cpu {c['engine_share_of_one_core_at_live_speed'] * 100:.3f} percent of one core at live speed for "
                 f"the engine, {c['engine_us_per_message']} us of engine CPU per message with minute steps and "
                 f"database writes included, {c['minute_step_ms_per_healthy_minute']} ms per healthy minute step, "
                 f"{c['full_size_snapshot_ms_each']} ms per full size snapshot, {c['other_message_us_each']} us per "
                 "other message")
    for name, s in c["segments"].items():
        lines.append(f"    {name}: {s['messages']:,} messages over {s['hours']} hours, {s['engine_us_per_message']} us "
                     f"per message, {s['full_size_snapshots']} full size snapshots at {s['full_size_snapshot_ms_each']} "
                     f"ms each, {s['other_message_us_each']} us per other message")
    first = next(iter(ledger.values()))
    lines.append(f"  checks {k['engine_checks_at_equity_snapshots']:,} engine checks at equity snapshots with "
                 f"{k['engine_violations']} violations, {k['hourly_bot_checks']:,} hourly checks with "
                 f"{k['hourly_problems']} problems, {len(ledger)} ledger checks over {first['rows']:,} equity "
                 f"snapshots, {e['deaths']} deaths and {e['closed_trades']:,} closed trades with "
                 f"{k['ledger_failures']} failures")
    lines.append(f"  evolution {e['deaths']} deaths ({e['learner_deaths']} learners) by cause {e['causes']}, "
                 f"{e['mutated_births']} mutated births, highest generation {e['highest_generation']}, exits "
                 f"{e['exit_reasons']}")
    for event in r["resume_events"]:
        lines.append(f"    resume {event}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Soak test. Replays every recording in data/raw and a synthetic "
                                            "recording of several days through the real engine with the plumbing "
                                            f"configuration, and reports memory, CPU and balance checks. "
                                            f"{NEVER_FOR_SELECTION} Exit code 0 passed, 1 failed, 3 inconclusive.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="plumbing config, never a strategy setting")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="folder of recordings, only ever read")
    p.add_argument("--skip-real", action="store_true", help="replay only the synthetic recording")
    p.add_argument("--synthetic-hours", type=float, default=120.0)
    p.add_argument("--resync-minutes", type=float, default=120.0, help="about how often the synthetic feed resyncs")
    p.add_argument("--snapshot-bids", type=int, default=FULL_BIDS)
    p.add_argument("--snapshot-asks", type=int, default=FULL_ASKS)
    p.add_argument("--outage-at-hours", type=float, default=24.0,
                   help="synthetic hour at which one long outage starts, negative for none")
    p.add_argument("--outage-minutes", type=float, default=None,
                   help="length of that outage, by default max_resume_gap_minutes plus 5")
    p.add_argument("--max-growth-mb", type=float, default=MAX_GROWTH_MB)
    p.add_argument("--out", type=Path, default=None, help="a new or empty folder, by default under data/soak")
    p.add_argument("--leak-kb-per-message", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker is not None:
        return worker(args.worker)
    return EXIT_CODES[run_soak(args)["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
