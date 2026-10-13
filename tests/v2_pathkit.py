"""The machinery of the whole path acceptance test in test_v2_acceptance, every piece offline and in a test folder.

The news side is the real news process, its collector, the Haiku scorer with its daily probe and the FinBERT scorer
on a key ledger in the test folder, run through serve on virtual time into a temporary news.db, with a fake Finnhub
news client, a fake Anthropic client and a fake FinBERT model, so no request is sent and no model is loaded. The stock
side is the real stock process run as a child on the rehearsal pair with eight learners per run, against a fake
Finnhub websocket on 127.0.0.1 that plays a seeded synthetic tape. The child's wall clock is a shared file the driver
moves in lock step, two seconds a step while prints flow and eight seconds otherwise, waiting at every step until the
child has taken each message and ticked, so it never sees a pause and three sessions take about a minute. The child
holds right after S1 commits the checkpoint at HOLD and before S2 does, where the driver kills it with SIGKILL, so the
restart must roll S1 back to the checkpoint both runs hold. Nothing here opens data/v2, the real secrets file, a real
recording or the network beyond 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import itertools
import json
import mmap
import os
import re
import signal
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as clock_time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, LiveClock, iso
from btc_survivor.feed.base import MESSAGE
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import NewsConfig, load_news_config
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.news import llm_request, probes, runner
from btc_survivor.v2.news.budget import Budget, rules_from_config
from btc_survivor.v2.news.finbert import Probs
from btc_survivor.v2.news.http import OK as HTTP_OK
from btc_survivor.v2.news.http import Response, Validators
from btc_survivor.v2.news.ledger import Ledger, Usage
from btc_survivor.v2.news.llm_client import OK, CallResult
from btc_survivor.v2.news.llm_validate import HORIZONS
from btc_survivor.v2.stocks import synthetic, tapecheck
from helpers import ROOT
from v2_feedkit import script_env

CONFIGS = ROOT / "configs" / "v2"
SYMBOLS = synthetic.SYMBOLS
COMPANIES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet",
             "META": "Meta Platforms", "TSLA": "Tesla"}
MACRO = "MACRO"
CANARY = "CANARYacceptanceTOKEN0055"
COMMIT = "acceptance-commit"
GATE, TUE, WED, THU = (sessions.session_on(date(2026, 10, day)) for day in (19, 20, 21, 22))
DAYS = (TUE, WED, THU)
FEED_STEP_MS = 2 * MS_PER_SECOND            # the tape's tick while prints flow
IDLE_STEP_MS = 8 * MS_PER_SECOND            # below the stock process's 10 second pause rule
PING_EVERY_MS = 16 * MS_PER_SECOND          # pings between sessions, so the pre open silence rule never fires
NEWS_EVERY_MS = 10 * MS_PER_SECOND          # how often the news side's virtual time catches up
TICK_SLACK_MS = 4 * MS_PER_SECOND           # in a feed window the child must have ticked this recently
STEP_TIMEOUT_S = 60.0
# Every path of the rehearsal pair moved into the test folder, eight learners and a fast reconnect.
STOCK_SETTINGS = {"secrets_file": '"secrets.env"', "raw_dir": '"data/raw_stocks"', "lock_file": '"lock/ws.lock"',
                  "news_db": '"data/news/news.db"', "size": "8", "reconnect_backoff_start_seconds": "0.05",
                  "reconnect_backoff_max_seconds": "0.2"}
SLOT = struct.Struct("<q")
CLOCK, TICK, TAKEN, HELD = 0, 8, 16, 24     # offsets in the shared file
SHARED_BYTES = 32

# The stock process as the live command runs it, with the wall clock read from the shared file, a one millisecond
# ticker, the local only check passing at once and a fixed commit. It reports each tick and each message it took, and
# right after S1 commits the checkpoint at ACCEPTANCE_HOLD_AT it says so and waits to be killed.
CHILD = """
import mmap, os, struct, sys, time
from btc_survivor import clock
from btc_survivor.feed.base import MESSAGE
from btc_survivor.v2 import localcheck
from btc_survivor.v2.stocks import engine, process

fd = os.open(os.environ["ACCEPTANCE_CLOCK"], os.O_RDWR)
shared = mmap.mmap(fd, 32)
slot = struct.Struct("<q")
hold_at = int(os.environ["ACCEPTANCE_HOLD_AT"])

def read(offset):
    while True:
        first, second = shared[offset:offset + 8], shared[offset:offset + 8]
        if first == second:
            return slot.unpack(first)[0]

def write(offset, value):
    shared[offset:offset + 8] = slot.pack(value)

clock.LiveClock.now_ms = lambda self: read(0)
process.TICK_SECONDS = 0.001
localcheck.run_check = lambda: localcheck.LocalCheck((), ())
process.code_commit = lambda: os.environ["ACCEPTANCE_COMMIT"]
tick, sink, persist = process.StockProcess.tick, process.StockProcess.sink, engine.StockEngine.persist
taken = [0]

def ticked(self):
    out = tick(self)
    write(8, self._last_tick)
    return out

def sunk(self, kind, raw, recv_ms, conn_id):
    sink(self, kind, raw, recv_ms, conn_id)
    if kind == MESSAGE:
        taken[0] += 1
        write(16, taken[0])

def persisted(self, at):
    persist(self, at)
    if at == hold_at and not self.inputs.use_news:
        write(24, 1)
        time.sleep(3600)

process.StockProcess.tick, process.StockProcess.sink, engine.StockEngine.persist = ticked, sunk, persisted
sys.exit(process.main(sys.argv[1:]))
"""


def at(session: sessions.Session, clock: str) -> int:
    """The UTC milliseconds of a New York clock time on a session's date."""
    parts = [int(p) for p in clock.split(":")]
    return int(datetime.combine(session.day, clock_time(*parts), tzinfo=sessions.new_york()).timestamp()) * 1000


START = at(TUE, "09:14")                    # the launch, before the local only check ten minutes ahead of the open
FIRST_NEWS = at(WED, "11:31")               # after the first decisions and fills of session 2
HOLD = at(WED, "13:00:30")                  # a checkpoint mid session, where the child is killed between the runs
RESTART = at(WED, "13:15")                  # more than max_resume_gap_minutes later, so a restart flatten is due
STOP = at(THU, "16:10")


# Configs


def edit(text: str, section: str, key: str, value: str) -> str:
    """Set one key in one section of a TOML text, refusing to write if it is not there."""
    start = re.search(rf"(?m)^\[{re.escape(section)}\][ \t]*(#.*)?$", text).start()
    tail, found = re.subn(rf"(?m)^{re.escape(key)} = [^#\n]*", f"{key} = {value} ", text[start:], count=1)
    assert found == 1, (section, key)
    return text[:start] + tail


def stock_configs(folder: Path) -> tuple[Path, Path]:
    """The rehearsal pair with every path in folder and a secrets file holding only a fake token."""
    secrets = folder / "secrets.env"
    secrets.write_text(f"FINNHUB_API_KEY={CANARY}\n", encoding="utf-8")
    secrets.chmod(0o600)
    paths = []
    for name in ("rehearsal_s1", "rehearsal_s2"):
        text = (CONFIGS / f"{name}.toml").read_text(encoding="utf-8")
        for key, value in {**STOCK_SETTINGS, "data_dir": f'"data/{name}"'}.items():
            text, found = re.subn(rf"(?m)^{key} = [^#\n]*", f"{key} = {value} ", text)
            assert found == 1, key
        assert "~" not in text and "../" not in text
        path = folder / f"{name}.toml"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1]


def news_config(folder: Path) -> NewsConfig:
    """The committed news config with every path in folder, Finnhub news as the only source, and the prompt, schema
    and probe files the code pins."""
    text = (CONFIGS / "news.toml").read_text(encoding="utf-8")
    for section, key, value in (
            ("news", "data_dir", '"data/news"'), ("news", "secrets_file", '"secrets.env"'),
            ("news", "rest_lock_file", '"lock/rest.lock"'), ("edgar", "enabled", "false"),
            ("fed", "enabled", "false"), ("bls", "enabled", "false"),
            ("llm", "prompt_file", f'"{llm_request.PROMPT_FILE}"'),
            ("llm", "prompt_sha256", f'"{llm_request.PROMPT_SHA256}"'),
            ("llm", "schema_file", f'"{llm_request.SCHEMA_FILE}"'),
            ("llm", "schema_sha256", f'"{llm_request.SCHEMA_SHA256}"'), ("llm", "probe_file", f'"{probes.PROBE_FILE}"'),
            ("llm.caps", "ledger_file", '"ledger/llm_ledger.db"'),
            ("llm.caps", "ledger_lock_file", '"ledger/llm_ledger.lock"')):
        text = edit(text, section, key, value)
    assert "~" not in text
    path = folder / "news.toml"
    path.write_text(text, encoding="utf-8")
    return load_news_config(path)


# The planned news


SILENT = "silent"  # an item the fake model leaves out of its answer, so FinBERT's score becomes the fallback


@dataclass(frozen=True)
class Planned:
    """One fictional Finnhub item, published at ms, with the impact the fake model gives it."""

    news_id: int
    ms: int
    symbol: str
    headline: str
    impact: dict | str


def planned_news(seed: int = 5) -> list[Planned]:
    """Two old items per channel that the first polls seed, then an item every 11 minutes from FIRST_NEWS through the
    second session, a few overnight and every 11 minutes through the third session, rotating over the companies and
    the general macro channel, every ninth new one left out by the model."""
    rng = np.random.default_rng(seed)
    channels = (*COMPANIES, MACRO)
    times = [at(GATE, "12:00")] * (2 * len(channels))
    times += list(range(FIRST_NEWS, at(WED, "15:50"), 11 * MS_PER_MINUTE))
    times += [at(WED, "17:30"), at(WED, "20:15"), at(THU, "06:40"), at(THU, "08:50")]
    times += list(range(at(THU, "09:41"), at(THU, "15:40"), 11 * MS_PER_MINUTE))
    verbs = ("raises", "trims", "confirms", "delays", "expands", "reviews")
    nouns = ("guidance", "outlook", "plans", "buyback", "hiring", "pricing")
    out = []
    for n, ms in enumerate(times):
        symbol = channels[n % len(channels)]
        name = "Fed officials" if symbol == MACRO else COMPANIES[symbol]
        impact: dict | str = {"ticker": symbol, "event_type": "macro_data" if symbol == MACRO else "guidance",
                              "is_new": bool(rng.random() < 0.85), "direction": int(rng.choice((-1, 1))),
                              "surprise": int(rng.integers(0, 4)), "horizon": str(rng.choice(HORIZONS)),
                              "confidence": int(rng.integers(1, 6))}
        out.append(Planned(10_000 + n, ms, symbol, f"{name} {verbs[n % 6]} {nouns[n // 6 % 6]} in fixture report "
                                                   f"{n:03d}", SILENT if ms > START and n % 9 == 4 else impact))
    return out


def entry(item: Planned) -> dict:
    """A Finnhub news entry in its documented fields."""
    return {"category": "general" if item.symbol == MACRO else "company", "datetime": item.ms // MS_PER_SECOND,
            "headline": item.headline, "id": item.news_id, "image": "",
            "related": "" if item.symbol == MACRO else item.symbol, "source": "Fixture Wire",
            "summary": f"A fictional summary for fixture item {item.news_id}.",
            "url": f"https://news.example.com/fixture/{item.news_id}"}


# The news side


class NewsClock(LiveClock):
    """The news process's wall clock, which only the virtual time moves."""

    def __init__(self, ms: int) -> None:
        super().__init__()
        self.ms = ms

    def now_ms(self) -> int:
        return self.ms


class FakeNewsRest:
    """Finnhub's company and general news answered from the planned items published by the news clock, with a Date
    header on that clock, and HTTP 404 for any other address. It never reaches the network."""

    def __init__(self, clock: NewsClock, items: list[Planned]) -> None:
        self.clock = clock
        self.items = items
        self.calls = 0

    async def fetch(self, url: str, validators: Validators | None = None) -> Response:
        self.calls += 1
        now = self.clock.ms
        query = parse_qs(urlsplit(url).query)
        if "/company-news" in url:
            chosen = [i for i in self.items if i.symbol == query["symbol"][0] and i.ms <= now]
        elif "/news" in url and query.get("category") == ["general"]:
            low = int(query.get("minId", ["0"])[0])
            chosen = [i for i in self.items if i.symbol == MACRO and i.ms <= now and i.news_id > low]
        else:
            return Response("http_error", 404, b"", 0, 1, now, validators or Validators(), "HTTP 404")
        body = json.dumps([entry(i) for i in reversed(chosen)]).encode()
        return Response(HTTP_OK, 200, body, len(body), 1, now, Validators())


class FakeAnthropic:
    """Answers every call with the planned impact of each headline, an empty list for a probe item, and leaves out
    each silent item. It never reaches the network."""

    def __init__(self, items: list[Planned]) -> None:
        self.impacts = {item.headline: item.impact for item in items}
        self.calls = 0
        self.closed = False

    async def count_tokens(self, request) -> int:
        return 1800

    async def send(self, request) -> CallResult:
        self.calls += 1
        entries = []
        for line in request["messages"][0]["content"].split("\n")[1:]:
            item = json.loads(line)
            impact = self.impacts.get(item["headline"])
            if impact != SILENT:
                entries.append({"id": item["id"], "impacts": [] if impact is None else [impact]})
        text = json.dumps({"items": entries})
        return CallResult(OK, 900, http_status=200, stop_reason="end_turn", usage=Usage(1800, 60),
                          response_model=request["model"], request_id=f"req_fixture_{self.calls}", text=text,
                          raw_json=json.dumps({"fixture": text}))

    async def close(self) -> None:
        self.closed = True


class FakeFinbert:
    """Probabilities from a hash of the headline, so every item gets a nonzero strength without a model."""

    def score(self, headline: str, text: str | None = None) -> Probs:
        digest = hashlib.sha256(headline.encode()).digest()
        pos = 0.05 + 0.9 * digest[0] / 255
        neg = (1 - pos) * digest[1] / 255
        return Probs(pos, neg, 1 - pos - neg, 1, 12)


class InlinePool(ThreadPoolExecutor):
    """Runs each call at once in the event loop's thread, so a FinBERT score is written before virtual time moves."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class NewsTime:
    """Virtual time for the news process. Every wait ends when the driver moves the clock to it, in time order, and
    each woken task runs until it waits again before the clock moves on."""

    def __init__(self, ms: int) -> None:
        self.clock = NewsClock(ms)
        self.sleepers: list[tuple[int, int, asyncio.Future]] = []
        self.order = itertools.count()

    async def wait(self, stop: asyncio.Event | None, seconds: float) -> bool:
        if stop is not None and stop.is_set():
            return True
        future = asyncio.get_running_loop().create_future()
        wake = self.clock.ms + max(0, round(seconds * MS_PER_SECOND))
        heapq.heappush(self.sleepers, (wake, next(self.order), future))
        await future
        return stop is not None and stop.is_set()

    async def sleep(self, seconds: float) -> None:
        await self.wait(None, seconds)

    async def settle(self) -> None:
        """Let every runnable task run until none is left, read from the loop's ready queue where it has one."""
        loop = asyncio.get_running_loop()
        for _ in range(10_000):
            await asyncio.sleep(0)
            if not getattr(loop, "_ready", True):
                return

    def _wake(self) -> None:
        """End the earliest wait, moving the clock to it."""
        wake, _, future = heapq.heappop(self.sleepers)
        self.clock.ms = max(self.clock.ms, wake)
        if not future.done():
            future.set_result(None)

    async def advance(self, until: int) -> None:
        """End every wait due by until in time order."""
        while True:
            await self.settle()
            if not self.sleepers or self.sleepers[0][0] > until:
                return
            self._wake()

    async def drain(self, stop: asyncio.Event, task: asyncio.Task) -> None:
        """Set stop as SIGTERM would and end every wait in time order until serve returns."""
        stop.set()
        while not task.done():
            await self.settle()
            if self.sleepers and not task.done():
                self._wake()


class NewsSide:
    """The real news process serving on virtual time in its own thread, with its writer connection and key ledger
    opened there."""

    def __init__(self, config: NewsConfig, items: list[Planned], start_ms: int) -> None:
        self.config = config
        self.time = NewsTime(start_ms)
        self.rest = FakeNewsRest(self.time.clock, items)
        self.llm = FakeAnthropic(items)
        self.counts = None
        self.error: BaseException | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._main, name="news", daemon=True)

    def start(self) -> None:
        """Start serve in its thread, failing with its error if it could not start."""
        self._thread.start()
        assert self._ready.wait(30), "the news process did not start"
        self._raise()

    def _main(self) -> None:
        """The thread, which runs serve until it returns and then cancels whatever is left on its loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_default_executor(InlinePool(max_workers=1))
        conn = None
        try:
            conn = news_db.open_writer(self.config.db_path)
            with Ledger.from_config(self.config) as ledger:
                self.loop, self.stop = loop, asyncio.Event()
                budget = Budget(rules_from_config(self.config), lambda: self.time.clock.ms, self.time.sleep)
                self.task = loop.create_task(runner.serve(
                    self.config, conn, self.rest, self.stop, self.time.clock, budget, self.time.wait,
                    model=FakeFinbert(), llm=self.llm, ledger=ledger))
                self._ready.set()
                self.counts = loop.run_until_complete(self.task)
        except BaseException as exc:
            self.error = exc
        finally:
            self._ready.set()
            left = asyncio.all_tasks(loop)
            for task in left:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*left, return_exceptions=True))
            if conn is not None:
                conn.close()
            loop.close()

    def _raise(self) -> None:
        """Fail with where serve raised, once it has."""
        if self.error is not None:
            where = "".join(f", raised at {Path(f.filename).name} line {f.lineno}"
                            for f in traceback.extract_tb(self.error.__traceback__)[-1:])
            raise AssertionError(f"the news process failed with {type(self.error).__name__}: {self.error}{where}")

    def _run(self, coroutine) -> None:
        """Run a coroutine on the news loop and wait for it, or for the loop to end with serve."""
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        except RuntimeError:  # the loop already closed
            coroutine.close()
            return
        deadline = time.monotonic() + STEP_TIMEOUT_S
        while True:
            try:
                future.result(0.1)
                return
            except CancelledError:  # serve ended and the loop cancelled what was left
                break
            except TimeoutError:
                if not self._thread.is_alive():
                    break
                assert time.monotonic() < deadline, "the news process did not answer"
        self._thread.join(STEP_TIMEOUT_S)

    def advance(self, until: int) -> None:
        """Run the news process up to until in virtual time."""
        self._raise()
        self._run(self.time.advance(until))
        self._raise()
        assert self._thread.is_alive(), "the news process stopped by itself"

    def finish(self) -> None:
        """Stop the news process and wait for serve to return its counts. The loop ends with serve, which cancels the
        drain still waiting on it."""
        if self._thread.is_alive() and hasattr(self, "task"):
            self._run(self.time.drain(self.stop, self.task))
        self._thread.join(STEP_TIMEOUT_S)
        assert not self._thread.is_alive(), "the news process did not stop"
        self._raise()


# The stock side


class SharedClock:
    """The wall clock the child reads, and the last tick, the messages taken and the hold it reports, in one file."""

    def __init__(self, path: Path, ms: int) -> None:
        path.write_bytes(bytes(SHARED_BYTES))
        self.path = path
        self._fd = os.open(path, os.O_RDWR)
        self._map = mmap.mmap(self._fd, SHARED_BYTES)
        self.set(ms)

    def set(self, ms: int) -> None:
        self._map[CLOCK:CLOCK + 8] = SLOT.pack(ms)

    def read(self, offset: int) -> int:
        """One slot, read twice until both reads agree, so a write in progress is never taken."""
        while True:
            first, second = self._map[offset:offset + 8], self._map[offset:offset + 8]
            if first == second:
                return SLOT.unpack(first)[0]

    def reset(self) -> None:
        """Forget what the last child reported before the next one starts."""
        self._map[TICK:SHARED_BYTES] = bytes(SHARED_BYTES - TICK)

    def close(self) -> None:
        self._map.close()
        os.close(self._fd)


class FeedServer:
    """A fake Finnhub websocket on 127.0.0.1 in its own thread, which takes each connection's subscribe frames and
    then sends whatever the driver hands it. The paths hold the token, so tests compare them and never print them."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.frames: list[list[str]] = []
        self.ws = None
        self.port = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), name="feed", daemon=True)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def connected(self) -> bool:
        return self.ws is not None

    def __enter__(self) -> FeedServer:
        self._thread.start()
        assert self._ready.wait(10)
        return self

    def __exit__(self, *exc) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(10)

    async def _main(self) -> None:
        self._loop, self._stop = asyncio.get_running_loop(), asyncio.Event()
        async with ws_serve(self._handler, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    async def _handler(self, ws) -> None:
        self.paths.append(ws.request.path)
        frames: list[str] = []
        self.frames.append(frames)
        try:
            for _ in SYMBOLS:
                frames.append(await asyncio.wait_for(ws.recv(), 30))
            self.ws = ws
            async for frame in ws:  # resubscribe frames, kept and never answered
                frames.append(frame)
        except ConnectionClosed:
            pass
        finally:
            if self.ws is ws:
                self.ws = None

    def send(self, texts: list[str]) -> int:
        """Send texts on the open connection and return how many went out."""
        return asyncio.run_coroutine_threadsafe(self._send(texts), self._loop).result(STEP_TIMEOUT_S)

    async def _send(self, texts: list[str]) -> int:
        ws, sent = self.ws, 0
        if ws is None:
            return 0
        try:
            for text in texts:
                await ws.send(text)
                sent += 1
        except ConnectionClosed:
            pass
        return sent


class Feed:
    """The messages the fake Finnhub plays, a seeded tape from 09:25 to 16:05 New York time in each session and a ping
    every 16 seconds between them, each due at its planned receive time."""

    def __init__(self, seed: int = 11) -> None:
        tape = synthetic.SyntheticTape(seed, SYMBOLS, tick_ms=FEED_STEP_MS, regular_rate=1.0)
        self.windows = [(at(day, "09:25"), at(day, "16:05")) for day in DAYS]
        lines = itertools.chain.from_iterable(tape.lines(start, end) for start, end in self.windows)
        self._lines = ((recv_ms, raw) for kind, raw, recv_ms, _ in lines if kind == MESSAGE)
        self._next = next(self._lines, None)
        self._pinged = 0

    def inside(self, t: int) -> bool:
        return any(start <= t < end for start, end in self.windows)

    def step(self, t: int) -> int:
        """The next step after t, every tape tick inside a window and at most IDLE_STEP_MS outside, landing on each
        window's start."""
        if self.inside(t):
            return t + FEED_STEP_MS
        return min([t + IDLE_STEP_MS, *(start for start, _ in self.windows if start > t)])

    def take(self, t: int) -> list[str]:
        """Every message due by t."""
        out = []
        while self._next is not None and self._next[0] <= t:
            out.append(self._next[1])
            self._next = next(self._lines, None)
        if not self.inside(t) and t - self._pinged >= PING_EVERY_MS:
            out.append(synthetic.PING)
            self._pinged = t
        return out


class Driver:
    """Moves the shared clock in lock step with the child, the fake Finnhub and the news process."""

    def __init__(self, folder: Path, s1: Path, s2: Path, news: NewsSide, server: FeedServer) -> None:
        self.folder, self.s1, self.s2, self.news, self.server = folder, s1, s2, news, server
        self.clock = SharedClock(folder / "clock.bin", START)
        self.feed = Feed()
        self.child: subprocess.Popen | None = None
        self.children = 0
        self.sent = 0
        self.t = START
        self.news_at = START

    def start_child(self, hold_at: int = 0) -> None:
        """Start the stock process and hold the clock until it ticked and subscribed."""
        self.clock.reset()
        self.children += 1
        self.sent = 0
        connections = len(self.server.frames)
        with (self.folder / f"stocks_{self.children}.err").open("w", encoding="utf-8") as err:
            self.child = subprocess.Popen(
                [sys.executable, "-c", CHILD, str(self.s1), str(self.s2), "--test-url", self.server.url,
                 "--log-dir", str(self.folder / "logs")], stdout=err, stderr=err, cwd=self.folder,
                env=script_env(self.folder, ACCEPTANCE_CLOCK=str(self.clock.path), ACCEPTANCE_COMMIT=COMMIT,
                               ACCEPTANCE_HOLD_AT=str(hold_at)))
        self.wait(lambda: self.clock.read(TICK) >= self.t and len(self.server.frames) > connections
                  and self.server.connected, "the stock process did not start")

    def held(self) -> bool:
        """Whether the child holds between S1's and S2's checkpoint commits, waiting to be killed."""
        return self.child is not None and self.clock.read(HELD) == 1

    def kill_child(self) -> None:
        """End the stock process with SIGKILL, as a crash or a power cut would."""
        os.kill(self.child.pid, signal.SIGKILL)
        assert self.child.wait(30) == -signal.SIGKILL
        self.child = None
        self.wait(lambda: not self.server.connected, "the fake Finnhub did not see the connection drop")

    def stop_child(self) -> int:
        """End the stock process with SIGTERM and return its exit code."""
        self.child.send_signal(signal.SIGTERM)
        code = self.child.wait(STEP_TIMEOUT_S)
        self.child = None
        return code

    def wait(self, done, what: str) -> None:
        """Spin until done, failing with the child's own error output when it exits or STEP_TIMEOUT_S passes."""
        deadline = time.monotonic() + STEP_TIMEOUT_S
        spins = 0
        while not done():
            spins += 1
            if spins % 64 == 0:
                if self.child is not None and self.child.poll() is not None:
                    raise AssertionError(f"the stock process exited with {self.child.returncode}\n" + self.errors())
                if time.monotonic() > deadline:
                    raise AssertionError(f"{what} at {iso(self.t)}\n" + self.errors())
                time.sleep(0.0001)

    def errors(self) -> str:
        """The end of each child's own error output."""
        return "".join(path.read_text(encoding="utf-8")[-3000:] for path in sorted(self.folder.glob("stocks_*.err")))

    def run_until(self, end: int) -> None:
        """Step the clock to end, or until the child holds, catching the news process up every NEWS_EVERY_MS,
        sending each due message and waiting for the child to take it and to tick."""
        while self.t < end and not self.held():
            self.t = self.feed.step(self.t)
            self.clock.set(self.t)
            if self.t >= self.news_at:
                self.news.advance(self.t)
                self.news_at = self.t + NEWS_EVERY_MS
            texts = self.feed.take(self.t)
            if self.child is None:
                continue
            if texts:
                self.sent += self.server.send(texts)
            tick = self.t - TICK_SLACK_MS if self.feed.inside(self.t) else self.t
            self.wait(lambda: self.held() or (self.clock.read(TAKEN) >= self.sent and self.clock.read(TICK) >= tick),
                      "the stock process fell behind")

    def close(self) -> None:
        """Kill a child still running, as after a failed step, and release the shared file."""
        if self.child is not None:
            self.child.kill()
            self.child.wait(30)
        self.clock.close()


def run_path(folder: Path, s1: Path, s2: Path, config: NewsConfig, items: list[Planned]) -> dict:
    """Launch at START, kill the child between the runs' checkpoint commits at HOLD, start it again at RESTART, stop
    it with SIGTERM at STOP and stop the news process, returning what the tests read besides the files."""
    news = NewsSide(config, items, START - MS_PER_MINUTE)
    with FeedServer() as server:
        news.start()
        try:
            news.advance(START)
            driver = Driver(folder, s1, s2, news, server)
            try:
                driver.start_child(hold_at=HOLD)
                driver.run_until(RESTART)
                assert driver.held(), "the stock process never reached the checkpoint at HOLD"
                held_at = driver.t
                driver.kill_child()
                driver.run_until(RESTART)
                driver.start_child()
                driver.run_until(STOP)
                stopped = driver.stop_child()
            finally:
                driver.close()
        finally:
            news.finish()
    # The paths hold the fake token, so only their count and how many carry it leave this function.
    return {"stopped": stopped, "held_at": held_at, "connections": len(server.paths),
            "token_paths": sum(CANARY in path for path in server.paths), "frames": server.frames,
            "calls": news.llm.calls, "closed": news.llm.closed, "counts": news.counts, "errors": driver.errors()}


# The gate session and the commands


def cli(folder: Path, *args: str) -> subprocess.CompletedProcess:
    """One btc_survivor_v2 command in a subprocess, as the owner runs it."""
    return subprocess.run([sys.executable, "-m", "btc_survivor.v2", *args], capture_output=True, text=True,
                          timeout=300, cwd=folder, env=script_env(folder))


def gate_record(folder: Path, s1: Path) -> subprocess.CompletedProcess:
    """A synthetic recording of the gate session the day before launch and its official figures at 25 times the
    tape's volume, run through tape-check, which writes the coverage record beside the figures."""
    start, end = tapecheck.window(GATE)
    tape = synthetic.SyntheticTape(31, SYMBOLS, regular_rate=0.05, extended_rate=0.005)
    synthetic.write_recording(folder / "gate_raw", tape.lines(start - 30 * MS_PER_MINUTE, end + 30 * MS_PER_MINUTE))
    checks = folder / "checks"
    checks.mkdir()
    official = checks / f"official_{GATE.day:%Y_%m_%d}.toml"
    official.write_text("consolidated_label = true\n" + "".join(
        f"\n[symbols.{s}]\nvolume = {tape.truth[(GATE.day, s)] * 25}\nclose = 123.25\n" for s in SYMBOLS),
        encoding="utf-8")
    return cli(folder, "tape-check", str(folder / "gate_raw"), str(official), "--config", str(s1))


def query(path: Path, sql: str, params=()) -> list[tuple]:
    """Rows from a database opened read only."""
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def recording_text(folder: Path) -> str:
    """Every recording file's text, a killed process's file read up to where it was cut."""
    return "".join(zlib.decompressobj(31).decompress(path.read_bytes()).decode("utf-8", "replace")
                   for path in sorted(folder.glob("*.jsonl.gz")))
