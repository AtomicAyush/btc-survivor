"""scripts/v2_record_stocks.py run as a process against a fake Finnhub server on 127.0.0.1, with a throwaway config,
a fake canary token and synthetic prices, its rules run in process on a clock set to a chosen time of day, and the
rules alone on fixed times."""

import argparse
import asyncio
import gzip
import json
import logging
import re
import signal
import stat
import sys
import threading
import time
from collections import Counter
from datetime import date

import pytest

from btc_survivor.clock import MS_PER_HOUR, MS_PER_MINUTE, MS_PER_SECOND, LiveClock
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, decode
from btc_survivor.recorder import read_recording, recording_files
from btc_survivor.v2 import redact, sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.local_secrets import ANTHROPIC
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter, PrintBatch
from btc_survivor.v2.stocks.runner import RecordingError, StockFeedClient, StockRecorder, websocket_lock
from helpers import ROOT
from v2_feedkit import CANARY, CLOSED_URL, PING, FakeFinnhub, receive, recorder_config, run_script, start_script

sys.path.insert(0, str(ROOT / "scripts"))
import v2_record_stocks  # noqa: E402
from v2_record_stocks import Watched, heartbeat_line, local_url, quiet_symbols, silence_reason  # noqa: E402

SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
FRAMES = [json.dumps({"type": "subscribe", "symbol": s}, separators=(",", ":")) for s in SYMBOLS]
# Synthetic prices and sizes with digits no count, stamp or timing in a log line could hold.
PRICES = ("187.31415", "211.71828", "99.14142", "305.27182")
SIZES = ("27.18281", "31.41592")


def trade(*prints):
    return json.dumps({"type": "trade", "data": [{"s": s, "p": float(p), "v": float(v), "t": t, "c": ["1"]}
                                                 for s, p, v, t in prints]})


FIRST = [
    trade(("AAPL", PRICES[0], SIZES[0], 1_791_072_000_001), ("MSFT", PRICES[1], SIZES[1], 1_791_072_000_002)),
    PING,
    trade(("NVDA", PRICES[2], SIZES[0], 1_791_072_000_003), ("IBM", PRICES[3], SIZES[1], 1_791_072_000_004)),
    '{"type":"news","data":[]}',
    trade(("QQQ", PRICES[3], SIZES[1], 1_791_072_000_005)),
]
SECOND = [
    trade(("TSLA", PRICES[0], SIZES[1], 1_791_072_001_001), ("META", PRICES[2], SIZES[0], 1_791_072_001_002),
          ("GOOGL", PRICES[1], SIZES[0], 1_791_072_001_003)),
    trade(("AMZN", PRICES[3], SIZES[1], 1_791_072_001_004)),
]
HEARTBEAT = re.compile(r"heartbeat feed=(connected|disconnected) connection=\d+ messages=\d+ bytes=\d+ prints=\d+ "
                       r"pings=\d+ errors=\d+ decode_errors=\d+ pauses=\d+ connects=\d+ disconnects=\d+ "
                       r"symbols_printing=\d+ dropped=\d+ unknown=\d+ "
                       r"file=(none|\d{4}-\d{2}-\d{2}_\d{6}\.jsonl\.gz) file_bytes=\d+")


def pinging(ws, interval=0.2):
    """Keep a connection busy with pings, so no silence rule fires whatever the time of day."""
    async def loop():
        while True:
            await ws.send(PING)
            await asyncio.sleep(interval)
    return loop()


def assert_clean(text):
    """No token, no synthetic price or size and no symbol in text that left the recorder."""
    assert CANARY not in text
    assert not [figure for figure in PRICES + SIZES if figure in text]
    assert not re.findall(r"\b(" + "|".join(SYMBOLS) + r")\b", text)


def test_the_recorder_writes_every_message_with_markers_and_stops_cleanly_on_sigterm(tmp_path):
    config = recorder_config(tmp_path)
    second_sent = threading.Event()

    async def first(ws, frames):
        await receive(ws, frames, len(SYMBOLS))
        for message in FIRST:
            await ws.send(message)
        await asyncio.sleep(0.3)
        ws.transport.abort()  # a dropped connection, with no close frame

    async def second(ws, frames):
        await receive(ws, frames, len(SYMBOLS))
        for message in SECOND:
            await ws.send(message)
        second_sent.set()
        await pinging(ws)

    with FakeFinnhub(first, second) as server:
        before = int(time.time() * 1000)
        process = start_script(config, "--test-url", server.url, "--heartbeat-seconds", "0.5",
                               "--log-dir", str(tmp_path / "logs"))
        try:
            assert second_sent.wait(30)
            time.sleep(1.5)
            process.send_signal(signal.SIGTERM)
            out, err = process.communicate(timeout=30)
        finally:
            process.kill()
        after = int(time.time() * 1000)
        token_sent = [path == f"/?token={CANARY}" for path in server.paths]
        frames = server.frames

    assert process.returncode == 0, err[-2000:]
    assert token_sent == [True, True]
    assert frames == [FRAMES, FRAMES]

    raw_dir = tmp_path / "data" / "raw_stocks"
    files = recording_files([raw_dir])
    assert files and all(CANARY not in f.name for f in files)
    owner_only = [raw_dir, tmp_path / "logs"]
    assert {stat.S_IMODE(path.stat().st_mode) for path in owner_only} == {0o700}
    assert {stat.S_IMODE(path.stat().st_mode) for path in [*files, tmp_path / "logs" / "btc_survivor.log"]} == {0o600}
    whole = b"".join(gzip.open(f).read() for f in files)  # a full read needs every gzip trailer
    assert CANARY.encode() not in whole

    rows = list(read_recording([raw_dir]))
    kinds = [kind for kind, _, _, _ in rows]
    ends = [i for i, kind in enumerate(kinds) if kind == CLOSE]
    assert kinds[0] == OPEN and len(ends) == 2 and ends[1] == len(rows) - 1
    one, two = rows[:ends[0] + 1], rows[ends[0] + 1:]
    assert [r[3] for r in one] == [1] * len(one) and [r[3] for r in two] == [2] * len(two)
    assert [(k, d) for k, d, _, _ in one[:-1]] == [(OPEN, "")] + [(MESSAGE, m) for m in FIRST]
    assert one[-1][1].startswith("ConnectionClosedError")
    assert [(k, d) for k, d, _, _ in two[:len(SECOND) + 1]] == [(OPEN, "")] + [(MESSAGE, m) for m in SECOND]
    assert {d for k, d, _, _ in two[len(SECOND) + 1:-1]} == {PING}
    assert two[-1][:2] == (CLOSE, "shutdown")
    stamps = [r[2] for r in rows]
    assert stamps == sorted(stamps) and before <= stamps[0] and stamps[-1] <= after
    assert two[0][2] > one[-1][2]

    adapter = FinnhubAdapter(SYMBOLS, CANARY)
    events = [e for kind, raw, ms, conn in rows for e in decode(adapter, kind, raw, ms, conn)]
    batches = [e for e in events if isinstance(e, PrintBatch)]
    assert sum(len(b.prints) for b in batches) == 8 and {p.symbol for b in batches for p in b.prints} == set(SYMBOLS)
    notices = Counter(e.kind for e in events if not isinstance(e, PrintBatch))
    assert notices["connected"] == 2 and notices["disconnected"] == 2 and notices["ping"] >= 2
    assert adapter.dropped == {"unsubscribed": 1} and adapter.unknown == {"news": 1}

    log = (tmp_path / "logs" / "btc_survivor.log").read_text(encoding="utf-8")
    for text in (out, err, log):
        assert_clean(text)
    assert out == ""
    beats = [line.split(" INFO v2_record_stocks ", 1)[1] for line in log.splitlines() if " heartbeat " in line]
    assert beats and all(HEARTBEAT.fullmatch(beat) for beat in beats)
    stopped = next(line for line in log.splitlines() if " stopped, " in line)
    assert "prints=8 " in stopped and "connects=2 disconnects=2 " in stopped and "dropped=1 unknown=1," in stopped
    assert "Finnhub message of type news on connection 1" in log
    assert err.count("\n") == log.count("\n")
    with websocket_lock(tmp_path / "lock" / "finnhub_ws.lock"):
        pass


def test_a_killed_recorder_leaves_a_readable_file_missing_at_most_its_last_second(tmp_path):
    config = recorder_config(tmp_path)
    sent = threading.Event()

    async def play(ws, frames):
        await receive(ws, frames, len(SYMBOLS))
        for message in FIRST:
            await ws.send(message)
        sent.set()
        await pinging(ws)

    with FakeFinnhub(play) as server:
        process = start_script(config, "--test-url", server.url)
        try:
            assert sent.wait(30)
            time.sleep(2.5)
            process.kill()
            process.communicate(timeout=30)
        finally:
            process.kill()

    raw_dir = tmp_path / "data" / "raw_stocks"
    [path] = recording_files([raw_dir])
    with pytest.raises(EOFError):
        gzip.open(path).read()  # no trailer after a kill
    rows = list(read_recording([raw_dir]))
    assert [(k, d) for k, d, _, _ in rows[:len(FIRST) + 1]] == [(OPEN, "")] + [(MESSAGE, m) for m in FIRST]
    assert {d for _, d, _, _ in rows[len(FIRST) + 1:]} <= {PING}
    assert CLOSE not in [k for k, _, _, _ in rows]
    log = tmp_path / "logs" / "v2" / "record_stocks" / "btc_survivor.log"  # the default, two folders above raw_dir
    assert "recording 8 symbols" in log.read_text(encoding="utf-8")


def test_dry_run_prints_a_verdict_and_counts_without_connecting(tmp_path):
    config = recorder_config(tmp_path)

    async def play(ws, frames):
        await receive(ws, frames, len(SYMBOLS))

    with FakeFinnhub(play) as server:
        result = run_script(config, "--dry-run", "--test-url", server.url)
        connections = len(server.paths)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["record stocks dry run", "verdict: pass", "symbols: 8", "subscribe frames: 8",
                                          "lock: taken and released"]
    assert result.stderr == "" and connections == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lock", "s1_prices.toml", "secrets.env"]
    assert_clean(result.stdout)
    with websocket_lock(tmp_path / "lock" / "finnhub_ws.lock"):
        pass


def test_dry_run_names_a_missing_key_without_any_value(tmp_path):
    config = recorder_config(tmp_path, secrets=f"{ANTHROPIC}=fake-other-{CANARY}\n")
    result = run_script(config, "--dry-run", "--test-url", CLOSED_URL)
    assert result.returncode == 2 and result.stdout == ""
    assert result.stderr.startswith("error: FINNHUB") and "is missing from" in result.stderr
    assert CANARY not in result.stderr
    assert not (tmp_path / "lock").exists()


def test_dry_run_refuses_a_proxy_or_certificate_variable_by_name_only(tmp_path):
    config = recorder_config(tmp_path)
    result = run_script(config, "--dry-run", "--test-url", CLOSED_URL, https_proxy="http://127.0.0.1:9",
                        SSL_CERT_FILE=str(tmp_path / "ca.pem"))
    assert result.returncode == 2 and result.stdout == ""
    assert result.stderr.startswith("error: unset SSL_CERT_FILE, https_proxy first")
    assert "127.0.0.1:9" not in result.stderr and "ca.pem" not in result.stderr
    assert not (tmp_path / "lock").exists()


@pytest.mark.parametrize("url", ["wss://ws.finnhub.io", "ws://example.com:80", "ws://127.0.0.1:80/x",
                                 "ws://localhost:8000", "ws://127.0.0.10:80", "ws://127.0.0.1:80?token=x"])
def test_the_test_url_accepts_only_a_plain_loopback_address(url):
    with pytest.raises(argparse.ArgumentTypeError, match="only ws://127.0.0.1:PORT"):
        local_url(url)
    assert local_url("ws://127.0.0.1:8765") == "ws://127.0.0.1:8765"


def test_the_script_exits_with_code_2_on_a_test_url_elsewhere(tmp_path):
    result = run_script(recorder_config(tmp_path), "--dry-run", "--test-url", "wss://ws.finnhub.io")
    assert result.returncode == 2 and "only ws://127.0.0.1:PORT" in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s1_prices.toml", "secrets.env"]


OPEN_MS = sessions.session_on(date(2026, 10, 12)).open_ms  # 13:30 UTC, 09:30 in New York
STALE_MS = 5 * MS_PER_SECOND


def test_silence_in_session_counts_from_the_later_of_the_last_message_and_the_open():
    mid = OPEN_MS + 2 * MS_PER_HOUR
    assert silence_reason(mid, mid - STALE_MS + 1, STALE_MS, Watched()) is None
    assert silence_reason(mid, mid - STALE_MS, STALE_MS, Watched()) == "no message for 5 s in session"
    assert silence_reason(OPEN_MS + STALE_MS - 1, OPEN_MS - MS_PER_HOUR, STALE_MS, Watched()) is None
    assert silence_reason(OPEN_MS + STALE_MS, OPEN_MS - MS_PER_HOUR, STALE_MS, Watched()) is not None


def test_silence_before_the_open_matters_only_in_its_last_ten_minutes():
    ten = OPEN_MS - 10 * MS_PER_MINUTE
    assert silence_reason(ten, ten - 5 * MS_PER_MINUTE, STALE_MS, Watched()) == "silent for 5 minutes before the open"
    assert silence_reason(ten, ten - 5 * MS_PER_MINUTE + 1, STALE_MS, Watched()) is None
    assert silence_reason(ten - 1, ten - MS_PER_HOUR, STALE_MS, Watched()) is None
    night = OPEN_MS - 7 * MS_PER_HOUR
    assert silence_reason(night, night - 3 * MS_PER_HOUR, STALE_MS, Watched()) is None
    saturday = OPEN_MS - 2 * 24 * MS_PER_HOUR
    assert silence_reason(saturday, saturday - MS_PER_HOUR, STALE_MS, Watched()) is None


def test_the_pre_open_check_runs_once_per_session_so_it_reconnects_at_most_once():
    ten = OPEN_MS - 10 * MS_PER_MINUTE
    watched = Watched()
    assert silence_reason(ten + 300, ten - 6 * MS_PER_MINUTE, STALE_MS, watched) is not None
    reopened = ten + 2 * MS_PER_SECOND  # the new connection stays silent up to the open
    assert not [t for t in range(reopened, OPEN_MS, MS_PER_SECOND)
                if silence_reason(t, reopened, STALE_MS, watched) is not None]
    quiet = Watched()
    assert silence_reason(ten, ten - MS_PER_MINUTE, STALE_MS, quiet) is None
    assert silence_reason(ten + 5 * MS_PER_MINUTE, ten - MS_PER_MINUTE, STALE_MS, quiet) is None
    tomorrow = sessions.session_on(date(2026, 10, 13)).open_ms - 10 * MS_PER_MINUTE
    assert silence_reason(tomorrow, tomorrow - MS_PER_HOUR, STALE_MS, quiet) is not None


def test_quiet_symbols_get_one_subscribe_frame_every_5_minutes_after_120_seconds():
    printed = {"AAPL", "MSFT"}
    early = OPEN_MS - MS_PER_HOUR
    assert quiet_symbols(OPEN_MS + 119_999, early, SYMBOLS, printed, {}) == []
    assert quiet_symbols(OPEN_MS + 120_000, early, SYMBOLS, printed, {}) == [s for s in SYMBOLS if s not in printed]
    now = OPEN_MS + 30 * MS_PER_MINUTE
    assert "NVDA" not in quiet_symbols(now, early, SYMBOLS, printed, {"NVDA": now - 5 * MS_PER_MINUTE + 1})
    assert "NVDA" in quiet_symbols(now, early, SYMBOLS, printed, {"NVDA": now - 5 * MS_PER_MINUTE})
    late = OPEN_MS + 10 * MS_PER_MINUTE
    assert quiet_symbols(late + 119_999, late, SYMBOLS, printed, {}) == []
    assert len(quiet_symbols(late + 120_000, late, SYMBOLS, printed, {})) == 6
    assert quiet_symbols(now, early, SYMBOLS, set(), {}) == []
    assert quiet_symbols(OPEN_MS - 1, early, SYMBOLS, printed, {}) == []


def test_the_heartbeat_holds_counts_states_and_the_file_size_only(tmp_path):
    recorder = StockRecorder(tmp_path)
    adapter = FinnhubAdapter(SYMBOLS, CANARY)
    adapter.parse(FIRST[2], 1, 1)
    assert HEARTBEAT.fullmatch("heartbeat " + heartbeat_line(Counter(), 0, "disconnected", 0, adapter, recorder))
    recorder.write(MESSAGE, FIRST[0], 1_791_072_000_000, 1)
    recorder.flush()
    line = heartbeat_line(Counter(messages=3, prints=7), 5, "connected", 2, adapter, recorder)
    size = recorder.path.stat().st_size
    recorder.close()
    assert HEARTBEAT.fullmatch("heartbeat " + line)
    assert "messages=3 " in line and "prints=7 " in line and "symbols_printing=5 " in line and "dropped=1 " in line
    assert size > 0 and line.endswith(f" file={recorder.path.name} file_bytes={size}")
    assert_clean(line)


SATURDAY_MS = OPEN_MS - 2 * 24 * MS_PER_HOUR  # no session rule applies
IN_SESSION_MS = OPEN_MS + MS_PER_HOUR
BOOM = '{"type":"boom"}'


class ShiftedClock(LiveClock):
    """The live clock moved to a chosen instant, which a test may move on, so the session rules see a fixed time of
    day while real time passes."""

    def __init__(self, at_ms):
        super().__init__()
        self.shift = at_ms - super().now_ms()

    def now_ms(self):
        return super().now_ms() + self.shift


class FullDisk:
    """A file whose every write fails as on a full disk."""

    def write(self, text):
        raise OSError(28, "No space left on device")

    def flush(self):
        pass

    def close(self):
        pass


class SpyRecorder(StockRecorder):
    """A recorder that also keeps every line in memory, for tests that watch the recording while it runs, and that
    can fail from the first line of one kind on."""

    def __init__(self, folder, fail_on=None):
        super().__init__(folder)
        self.lines = []
        self.fail_on = fail_on

    def write(self, kind, raw, recv_ms, conn_id):
        if kind == self.fail_on and not isinstance(self._file, FullDisk):
            self._file.close()
            self._file = FullDisk()
        super().write(kind, raw, recv_ms, conn_id)
        self.lines.append((kind, raw, recv_ms, conn_id))


class Fragile(FinnhubAdapter):
    """An adapter whose parse fails on one message, standing in for a decode bug."""

    def parse(self, raw, recv_ms, conn_id):
        if raw == BOOM:
            raise RuntimeError(f"synthetic decode failure {CANARY}")
        return super().parse(raw, recv_ms, conn_id)


@pytest.fixture
def in_process(tmp_path, monkeypatch):
    """Run record() in this process against a fake server, with the canary registered, a tick of 50 ms and every
    StockFeedClient kept, until a condition on the recorded lines holds."""
    monkeypatch.setattr(redact, "_values", set())
    redact.register(CANARY)
    monkeypatch.setattr(v2_record_stocks, "TICK_SECONDS", 0.05)
    clients = []

    class Kept(StockFeedClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

    monkeypatch.setattr(v2_record_stocks, "StockFeedClient", Kept)

    def run(server, at_ms, until, adapter_type=FinnhubAdapter, fail_on=None, **settings):
        cfg = load_stock_config(recorder_config(tmp_path, **settings))
        recorder = SpyRecorder(cfg.raw_dir, fail_on)
        clock = ShiftedClock(at_ms)

        async def main():
            stop = asyncio.Event()
            adapter = adapter_type(SYMBOLS, CANARY, server.url)
            task = asyncio.create_task(v2_record_stocks.record(cfg, adapter, recorder, MS_PER_HOUR, clock, stop))
            for _ in range(1000):
                if task.done() or until(recorder.lines, clock, clients):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("the condition never held")
            stop.set()
            return await asyncio.wait_for(task, 10)

        try:
            return recorder.lines, asyncio.run(main())
        finally:
            recorder.close()

    return run


def opens(lines):
    return [line for line in lines if line[0] == OPEN]


def closes(lines):
    return [line[1] for line in lines if line[0] == CLOSE]


async def silent(ws, frames):
    await receive(ws, frames, len(SYMBOLS))
    await ws.wait_closed()


async def busy(ws, frames):
    await receive(ws, frames, len(SYMBOLS))
    await pinging(ws, 0.1)


def test_a_finnhub_error_closes_the_connection_once_and_reconnects(in_process):
    async def erring(ws, frames):
        await receive(ws, frames, len(SYMBOLS))
        await ws.send(json.dumps({"type": "error", "msg": f"Invalid token {CANARY}"}))
        await ws.send(json.dumps({"type": "error", "msg": "again"}))
        await ws.wait_closed()

    with FakeFinnhub(erring, busy) as server:
        lines, total = in_process(server, SATURDAY_MS, lambda lines, clock, clients: len(opens(lines)) >= 2)
    assert [kind for kind, _, _, _ in lines[:4]] == [OPEN, MESSAGE, CLOSE, OPEN]
    assert closes(lines) == ["finnhub error", "shutdown"]
    assert redact.MASK in lines[1][1] and not [line for line in lines if CANARY in line[1]]
    assert total["errors"] == 1 and total["connects"] == 2


def test_silence_in_session_replaces_the_connection(in_process):
    with FakeFinnhub(silent, busy) as server:
        lines, total = in_process(server, IN_SESSION_MS, lambda lines, clock, clients: len(opens(lines)) >= 2,
                                  feed_stale_after_seconds="0.5")
    assert closes(lines) == ["no message for 0.5 s in session", "shutdown"]
    assert total["connects"] == 2


def test_a_symbol_that_has_not_printed_gets_one_more_subscribe_frame_on_the_wire(in_process, monkeypatch):
    monkeypatch.setattr(v2_record_stocks, "RESUBSCRIBE_AFTER_MS", 300)

    async def two_print(ws, frames):
        await receive(ws, frames, len(SYMBOLS))
        await ws.send(trade(("AAPL", PRICES[0], SIZES[0], 1), ("MSFT", PRICES[1], SIZES[1], 2)))
        pings = asyncio.ensure_future(pinging(ws, 0.1))
        try:
            async for frame in ws:
                frames.append(frame)
        finally:
            pings.cancel()

    with FakeFinnhub(two_print) as server:
        def resubscribed(lines, clock, clients):
            return server.frames and len(server.frames[0]) >= len(SYMBOLS) + 6

        lines, total = in_process(server, IN_SESSION_MS, resubscribed)
        frames = server.frames
    assert frames == [FRAMES + [frame for frame, s in zip(FRAMES, SYMBOLS) if s not in ("AAPL", "MSFT")]]
    assert closes(lines) == ["shutdown"]


def test_the_backoff_resets_once_a_connection_has_lasted_the_healthy_interval(in_process):
    async def closing(ws, frames):
        await receive(ws, frames, len(SYMBOLS))
        await ws.close()

    seen = []

    def reset_after_backoff(lines, clock, clients):
        if len(opens(lines)) >= 2 and clients:
            seen.append(clients[0].attempt)
        return 0 in seen

    with FakeFinnhub(closing, busy) as server:
        lines, total = in_process(server, SATURDAY_MS, reset_after_backoff, mark_healthy_outside_session_seconds="0.3")
    assert seen[0] == 1 and seen[-1] == 0
    assert closes(lines) == ["closed by server", "shutdown"]


def test_a_paused_process_replaces_its_connection_when_it_runs_again(in_process):
    jumped = []

    def jump_then_reopen(lines, clock, clients):
        if not jumped and MESSAGE in [line[0] for line in lines]:
            jumped.append(True)
            clock.shift += 60 * MS_PER_SECOND
        return len(opens(lines)) >= 2

    with FakeFinnhub(busy) as server:
        lines, total = in_process(server, SATURDAY_MS, jump_then_reopen)
    assert re.fullmatch(r"process paused for 6\d s", closes(lines)[0]) and closes(lines)[1] == "shutdown"
    assert total["pauses"] == 1


def test_a_decode_failure_is_counted_and_logged_by_type_while_recording_goes_on(in_process, caplog):
    caplog.set_level(logging.INFO)

    async def breaking(ws, frames):
        await receive(ws, frames, len(SYMBOLS))
        for message in (BOOM, BOOM, FIRST[0]):
            await ws.send(message)
        await pinging(ws, 0.1)

    with FakeFinnhub(breaking) as server:
        lines, total = in_process(server, SATURDAY_MS,
                                  lambda lines, clock, clients: FIRST[0] in [line[1] for line in lines],
                                  adapter_type=Fragile)
    assert [line[1] for line in lines[1:4]] == [BOOM, BOOM, FIRST[0]]
    assert total["decode_errors"] == 2 and total["prints"] == 2 and closes(lines) == ["shutdown"]
    failures = [r.getMessage() for r in caplog.records if "could not decode" in r.getMessage()]
    assert failures == ["could not decode a line on connection 1 (RuntimeError), counted from now on"]
    assert not [r for r in caplog.records if CANARY in r.getMessage()]


def test_a_failed_write_stops_the_recorder_instead_of_reconnecting(in_process):
    with FakeFinnhub(busy) as server:
        with pytest.raises(RecordingError, match="writing the recording failed with OSError"):
            in_process(server, SATURDAY_MS, lambda lines, clock, clients: False, fail_on=MESSAGE)
        connections = len(server.paths)
    assert connections == 1
