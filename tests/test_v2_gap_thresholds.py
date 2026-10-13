"""scripts/v2_gap_thresholds.py, the stale threshold rule of 12 October 2026: the rounding at its steps, a hand built
session whose planted silences give exact gaps, disconnected and paused time left out, eligibility as live health has
it, a synthetic day measured independently with a halt past the bound, silence closes, the refusals, and a default
output of thresholds, verdicts, counts and timings only."""

import json
import re
import sys
from datetime import date

import pytest

from btc_survivor.clock import iso
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN
from btc_survivor.v2 import sessions
from btc_survivor.v2.stocks import synthetic
from btc_survivor.v2.stocks.synthetic import (DROPPED, HALT, PROCESS_START, SHUTDOWN, SILENCE, SILENT, SYMBOLS, Fault,
                                              SyntheticTape, recorded, write_recording)
from helpers import ROOT
from test_v2_tapecheck import PINNED, pin_config

sys.path.insert(0, str(ROOT / "scripts"))
import v2_gap_thresholds as gaps  # noqa: E402
from v2_gap_thresholds import feed_threshold, symbol_threshold  # noqa: E402

SEC, MIN = 1_000, 60_000
DAY = sessions.session_on(date(2026, 10, 13))
O, C = DAY.open_ms, DAY.close_ms
ALLOWED = re.compile(r"[a-z_ ]+: (pass|fail|\d+|\d+\.\d{3} s|[a-z ]+)")


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    return pin_config(tmp_path_factory.mktemp("config"), **PINNED)


def run(capsys, config, recording, day="2026-10-13", *extra):
    code = gaps.main([str(recording), day, "--config", str(config), *extra])
    out, err = capsys.readouterr()
    return code, out, err


def summary(text):
    return dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)


@pytest.mark.parametrize("gap_ms, seconds", [
    (1, 30), (12_000, 30), (12_001, 60), (24_000, 60), (24_001, 90), (28_000, 90), (70_000, 180), (72_000, 180),
    (72_001, 210), (120_000, 300), (120_001, 330),
])
def test_the_symbol_rule_is_two_and_a_half_times_the_gap_rounded_up_to_30_s(gap_ms, seconds):
    assert symbol_threshold(gap_ms) == seconds


@pytest.mark.parametrize("gap_ms, seconds", [
    (1, 5), (1_000, 5), (1_001, 10), (5_000, 25), (5_400, 30), (6_000, 30), (6_001, 35), (12_000, 60), (12_001, 65),
])
def test_the_feed_rule_is_five_times_the_gap_rounded_up_to_5_s(gap_ms, seconds):
    assert feed_threshold(gap_ms) == seconds


# A hand built session.

AAPL_QUIET = (O + 1000 * SEC, O + 1070 * SEC)
DROP, REOPEN = O + 3005 * SEC, O + 3125 * SEC
FEED_QUIET = (O + 5000 * SEC, O + 5007_300)
PAUSE_CLOSE, PAUSED_FROM, AFTER_PAUSE = O + 8032 * SEC, O + 8002 * SEC, O + 8040 * SEC


def trade(recv, symbols, late=()):
    items = [{"p": 10.0, "s": s, "t": recv - (6 * SEC if s in late else 100), "v": 100} for s in symbols]
    return synthetic.trade_message(items)


def session_lines():
    """One recorder run from ten minutes before the open, a trade for every symbol each 10 s and pings between them,
    with planted silences, AAPL quiet for 70 s apart from one late print, a whole feed silence of 7.3 s, a drop with
    MSFT quiet on both sides of it, and a 30 s pause closed with its reason."""
    events = [(t, MESSAGE, synthetic.PING) for t in range(O - 10 * MIN, O, 2 * SEC)]
    for k in range(2340):
        at = O + k * 10 * SEC
        events += [(at + offset * SEC, MESSAGE, synthetic.PING) for offset in (2, 4, 6, 8)]
        if k:
            symbols = [s for s in SYMBOLS
                       if not (s == "AAPL" and AAPL_QUIET[0] < at < AAPL_QUIET[1] and at != O + 1030 * SEC)
                       and not (s == "MSFT" and (O + 2950 * SEC <= at <= DROP or REOPEN < at < O + 3180 * SEC))]
            events.append((at, MESSAGE, trade(at, symbols, late={"AAPL"} if at == O + 1030 * SEC else ())))
    events = [e for e in events if not (DROP < e[0] < REOPEN or O + 8000 * SEC < e[0] <= AFTER_PAUSE
                                        or FEED_QUIET[0] < e[0] < FEED_QUIET[1])]
    events += [(FEED_QUIET[1], MESSAGE, synthetic.PING), (DROP, CLOSE, DROPPED), (REOPEN, OPEN, ""),
               (PAUSE_CLOSE, CLOSE, "process paused for 30 s"), (AFTER_PAUSE, OPEN, ""), (C + 5 * MIN, CLOSE, SHUTDOWN)]
    lines, conn = [(PROCESS_START, "", O - 10 * MIN, 0), (OPEN, "", O - 10 * MIN, 1)], 1
    for at, kind, raw in sorted(events, key=lambda e: e[0]):
        conn += kind == OPEN
        lines.append((kind, raw, at, conn))
    return lines


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    folder = tmp_path_factory.mktemp("raw_stocks")
    write_recording(folder, session_lines())
    return folder


def test_planted_silences_give_exact_gaps_and_thresholds(built, config, capsys):
    code, out, err = run(capsys, config, built)
    fields = summary(out)
    assert (code, err) == (0, "") and out.splitlines()[0] == "gap thresholds"
    # MSFT's 240 s across the drop less the 120 s dropped, longer than AAPL's 70 s inside one connection.
    assert fields["symbol_stale_after_seconds"] == "300.000 s" and fields["feed_stale_after_seconds"] == "40.000 s"
    assert fields["longest symbol gap"] == "120.000 s" and fields["longest whole feed gap"] == "7.300 s"
    assert fields["longest symbol gap inside one connection"] == "70.000 s"
    assert fields["disconnects inside the longest symbol gap"] == "1"
    assert fields["symbol threshold within the bound"] == fields["feed threshold within the bound"] == "pass"
    assert fields["symbol threshold at least the feed threshold"] == "pass" and fields["full session"] == "pass"
    assert fields["disconnected time left out"] == "158.000 s"   # 120 s dropped and 38 s from the pause on
    assert fields["connections"] == "3" and fields["silence closes"] == "0"
    assert fields["symbols without an eligible print"] == "0"


def test_symbol_gaps_run_across_reconnects_less_the_time_lost_and_feed_gaps_stay_inside_one_connection(built, config,
                                                                                                        capsys):
    _, out, _ = run(capsys, config, built, "2026-10-13", "--detail")
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith(SYMBOLS)}
    assert rows["AAPL"][2:] == ["70.000", iso(AAPL_QUIET[0]), iso(AAPL_QUIET[1]), "180", "0"]
    assert int(rows["AAPL"][1]) == int(rows["NVDA"][1]) - 6    # five quiet trades and one late print renew nothing
    # Live health keeps a symbol's clock across a reconnect, so the drop never splits MSFT's silence into 65 s and
    # 55 s, and the 120 s it was disconnected, when no print could be seen, are left out.
    assert rows["MSFT"][2:] == ["120.000", iso(O + 2940 * SEC), iso(O + 3180 * SEC), "300", "1"]
    assert rows["NVDA"][2:] == ["12.000", iso(O + 8000 * SEC), iso(O + 8050 * SEC), "30", "1"]   # 50 s less 38 paused
    feed = next(line.split() for line in out.splitlines() if line.startswith("whole feed"))
    assert feed[3:] == ["7.300", iso(FEED_QUIET[0]), iso(FEED_QUIET[1]), "40", "0"]


def test_connected_time_is_the_time_inside_the_stretches():
    upto = gaps.connected_time([(10, 20), (30, 45)])
    assert [upto(t) for t in (0, 10, 15, 20, 25, 30, 40, 45, 99)] == [0, 0, 5, 10, 10, 10, 20, 25, 25]
    assert gaps.connected_time([])(5) == 0


def test_default_output_holds_only_thresholds_verdicts_counts_and_timings(built, config, capsys):
    _, out, _ = run(capsys, config, built)
    lines = out.splitlines()
    assert lines[-1] == "9 rows hidden, pass --detail in your own terminal to see them"
    assert all(ALLOWED.fullmatch(line) for line in lines[1:-1]), lines
    assert "2026" not in out and "AAPL" not in out


# A synthetic day measured independently.

def independent(lines, late_ms=4_000):
    """The longest whole feed gap and per symbol gap of one connection open through the session, by a plain walk."""
    feed, stamps = [O], {s: [O] for s in SYMBOLS}
    for kind, raw, recv, _ in recorded(lines):
        if kind != MESSAGE or not O <= recv < C:
            continue
        feed.append(recv)
        if '"trade"' in raw:
            for item in json.loads(raw)["data"]:
                if O <= item["t"] < C and -2_000 <= recv - item["t"] <= late_ms and stamps[item["s"]][-1] != recv:
                    stamps[item["s"]].append(recv)
    widest = lambda times: max(b - a for a, b in zip(times, [*times[1:], C]))  # noqa: E731
    return widest(feed), max(widest(times) for times in stamps.values())


def test_a_halt_sets_the_symbol_threshold_past_its_bound(tmp_path, config, capsys):
    tape = SyntheticTape(11, regular_rate=0.3)
    lines = list(tape.lines(O - 30 * MIN, C + 15 * MIN, [Fault(HALT, O + 60 * MIN, 200 * SEC, "TSLA")]))
    write_recording(tmp_path / "raw", lines)
    feed_ms, symbol_ms = independent(lines)
    assert symbol_ms > 200 * SEC
    code, out, _ = run(capsys, config, tmp_path / "raw")
    fields = summary(out)
    assert code == 1 and fields["symbol threshold within the bound"] == "fail"
    assert fields["symbol_stale_after_seconds"] == f"{symbol_threshold(symbol_ms)}.000 s"
    assert fields["feed_stale_after_seconds"] == f"{feed_threshold(feed_ms)}.000 s"
    assert fields["longest symbol gap"] == f"{symbol_ms / SEC:.3f} s"
    assert fields["longest whole feed gap"] == f"{feed_ms / SEC:.3f} s"
    assert fields["disconnected time left out"] == "0.000 s" and fields["connections"] == "1"


def test_silence_closes_are_counted_and_cut_the_whole_feed_gap(tmp_path, config, capsys):
    tape = SyntheticTape(12, regular_rate=0.3)
    lines = list(tape.lines(O - 30 * MIN, C + 15 * MIN, [Fault(SILENCE, O + 30 * MIN, 20 * SEC)]))
    write_recording(tmp_path / "raw", lines)
    closes = sum(1 for kind, raw, recv, _ in lines if kind == CLOSE and raw == SILENT and O <= recv < C)
    assert closes >= 2
    code, out, _ = run(capsys, config, tmp_path / "raw")
    fields = summary(out)
    assert code == 0 and fields["silence closes"] == str(closes)
    assert fields["longest whole feed gap"] == "5.000 s" and fields["feed_stale_after_seconds"] == "25.000 s"


# Refusals.

def test_what_cannot_be_measured_exits_with_code_2_and_one_error_line(tmp_path, config, capsys):
    code, out, err = run(capsys, config, tmp_path / "missing")
    assert (code, out) == (2, "") and err.startswith("error: no recording files")
    write_recording(tmp_path / "day", [(PROCESS_START, "", O - MIN, 0), (OPEN, "", O - MIN, 1),
                                       (MESSAGE, synthetic.PING, C - MIN, 1)])
    assert run(capsys, config, tmp_path / "day")[2] == f"error: {gaps.INCOMPLETE}\n"
    assert "2026-11-26 has no regular session" in run(capsys, config, tmp_path / "day", "2026-11-26")[2]
    write_recording(tmp_path / "away", [(PROCESS_START, "", O - 10 * MIN, 0), (OPEN, "", O - 10 * MIN, 1),
                                        (CLOSE, DROPPED, O - 5 * MIN, 1), (OPEN, "", C + 11 * MIN, 2)])
    code, out, err = run(capsys, config, tmp_path / "away")
    assert (code, out, err) == (2, "", f"error: {gaps.NO_CONNECTED_TIME}\n")
    with pytest.raises(SystemExit) as caught:
        gaps.main([str(tmp_path / "day"), "2026-13-01"])
    assert caught.value.code == 2
    capsys.readouterr()
