"""The tape coverage report of PLAN_V2 section 4.9 on seeded synthetic raw recordings with no database: the synthetic
recordings against the recorder's format and faults, every connection rule, a pause close read past the margin, an
incomplete recording refused, the ratios and their median on a partial tape, lost time left out of the covered ratio,
outages, the clean flag, the coverage record and its reader, the factor of criterion 1.1, the delay rule either way,
the official figures, which git must ignore, and a default output of verdicts, counts and timings only."""

import gzip
import json
import re
import statistics
import subprocess
import zlib
from datetime import date

import pytest

from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN
from btc_survivor.recorder import read_recording
from btc_survivor.v2 import output, sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import synthetic, tapecheck
from btc_survivor.v2.stocks.prints import OTHER, POST, PRE, REGULAR, PriceRules, phase_in
from btc_survivor.v2.stocks.synthetic import (DISCONNECT, DROPPED, EARLY_PRINTS, HALT, KILL, LATE_PRINTS, PAUSE,
                                              PROCESS_KILL, PROCESS_START, REPLAYED_PRINTS, SHUTDOWN, SILENCE,
                                              SILENT, Fault, SyntheticTape, recorded, write_recording)
from btc_survivor.v2.stocks.tapecheck import (CLEAN_REASON, DISCONNECTED_REASON, EARLY_CLOSE_REASON, OUTAGE,
                                              OUTAGE_REASON, Connections, CoverageRecord, Gap, Official,
                                              TapeCheckError, check, clean_reason, connected_stretches, load_official,
                                              measure, parse_official, percentile, read_records, record_of, report,
                                              session_files, silences, window, within_factor, write_record)
from helpers import ROOT

S1 = ROOT / "configs" / "v2" / "s1_prices.toml"
EXAMPLE = ROOT / "configs" / "v2" / "official_example.toml"
SYMBOLS = load_stock_config(S1).stock_feed.symbols
RULES = PriceRules(late_ms=4_000)
SEC, MIN = 1_000, 60_000
STALE = 30 * SEC
DAY = sessions.session_on(date(2026, 10, 13))       # EDT, so 20:00 New York time is UTC midnight
EST_DAY = sessions.session_on(date(2026, 11, 3))    # EST, so UTC midnight falls at 19:00 in post market
EARLY_CLOSE = sessions.session_on(date(2026, 11, 27))
# Official closes no synthetic price, count or timing could print by chance.
CANARY_CLOSES = {symbol: 187.4321 + i for i, symbol in enumerate(SYMBOLS)}
# A partial tape, each symbol carrying its own share of the official volume, with a median of 0.0375.
SHARES = {symbol: 0.02 + 0.005 * i for i, symbol in enumerate(SYMBOLS)}
ALLOWED = re.compile(r"[a-z_ ]+: (pass|fail|\d+|\d+\.\d{3} s|[a-z ]+)")
RECORD_KEYS = {"kind", "session", "clean", "reason", "early_close", "median_ratio", "median_ratio_covered", "ratios",
               "ratios_covered", "disconnected_seconds", "outage_seconds", "covered_share"}


def full_day(session, seed=11, faults=(), regular_rate=0.3, extended_rate=0.02):
    """A recording of one recorder run from half an hour before 04:00 to half an hour after 20:00 New York time."""
    tape = SyntheticTape(seed, regular_rate=regular_rate, extended_rate=extended_rate)
    start, end = window(session)
    return tape, list(tape.lines(start - 30 * MIN, end + 30 * MIN, faults))


def official_toml(volumes, closes):
    text = "consolidated_label = true\n"
    for symbol in volumes:
        text += f"\n[symbols.{symbol}]\nvolume = {volumes[symbol]!r}\nclose = {closes[symbol]!r}\n"
    return text


def write_official(folder, session, tape, shares=None):
    """official_YYYY_MM_DD.toml whose volumes give each symbol the share given, all of the volume by default, on a
    recording with no loss."""
    shares = shares or dict.fromkeys(SYMBOLS, 1.0)
    volumes = {s: tape.truth[(session.day, s)] / shares[s] for s in SYMBOLS}
    path = folder / f"official_{session.day:%Y_%m_%d}.toml"
    path.write_text(official_toml(volumes, CANARY_CLOSES), encoding="utf-8")
    return path


def pin_config(folder, source=S1, **values):
    """A copy of a committed run config with the given [stock_feed] settings, so later edits to the committed file
    change no outcome of these tests."""
    text = source.read_text(encoding="utf-8")
    for key, value in values.items():
        text, found = re.subn(rf"^{key} = .*$", f"{key} = {value}", text, flags=re.M)
        assert found == 1, key
    path = folder / source.name
    path.write_text(text, encoding="utf-8")
    return path


PINNED = {"feed_stale_after_seconds": 5, "symbol_stale_after_seconds": 15, "max_print_delay_seconds": 2.0,
          "late_print_seconds": 4, "price_excluded_conditions": "[]"}


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    return pin_config(tmp_path_factory.mktemp("config"), **PINNED)


def run_main(capsys, config, recording, official, *extra):
    code = tapecheck.main([str(recording), str(official), "--config", str(config), *extra])
    out, err = capsys.readouterr()
    return code, out, err


def summary(text):
    return dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)


def trade_prints(lines):
    """(recv_ms, conn_id, print dict) for every print in the trade messages of lines."""
    found = []
    for kind, raw, recv_ms, conn_id in lines:
        if kind == MESSAGE and '"trade"' in raw:
            found += [(recv_ms, conn_id, item) for item in json.loads(raw)["data"]]
    return found


def flat(volume=1.0):
    return Official(DAY.day, dict.fromkeys(SYMBOLS, volume), dict.fromkeys(SYMBOLS, 1.0))


def record_file(folder, day=DAY.day):
    return json.loads((folder / f"coverage_{day:%Y_%m_%d}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clean_day(tmp_path_factory):
    """A clean synthetic day in its own folder, with the generator holding its truth."""
    folder = tmp_path_factory.mktemp("raw_stocks")
    tape, lines = full_day(DAY)
    return tape, lines, write_recording(folder, lines), folder


# The synthetic recordings.

def test_the_synthetic_tape_is_seeded_and_faults_leave_the_market_alone():
    start, end = DAY.open_ms - 10 * MIN, DAY.open_ms + 20 * MIN
    faults = [Fault(DISCONNECT, DAY.open_ms + 5 * MIN, refused=1), Fault(PAUSE, DAY.open_ms + 8 * MIN, 30 * SEC),
              Fault(SILENCE, DAY.open_ms + 12 * MIN, 20 * SEC), Fault(KILL, DAY.open_ms + 15 * MIN, 40 * SEC)]
    one, two, other, faulty = SyntheticTape(3), SyntheticTape(3), SyntheticTape(4), SyntheticTape(3)
    lines = list(one.lines(start, end))
    assert lines == list(two.lines(start, end))
    assert lines != list(other.lines(start, end))
    assert list(faulty.lines(start, end, faults)) != lines
    assert faulty.truth == one.truth and faulty.closes == one.closes


def test_recordings_use_the_recorders_line_format_and_rotate_at_utc_midnight(clean_day):
    _, lines, files, _ = clean_day
    assert [f.name[:13] for f in files] == ["2026-10-13_07", "2026-10-14_00"]
    for path in files:
        for text in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines():
            row = json.loads(text)
            assert list(row) == ["t", "k", "c", "d"]
            assert text == json.dumps(row, separators=(",", ":"))
    rows = list(read_recording(files))
    assert rows == recorded(lines)
    assert [r[0] for r in rows].count(OPEN) == 1 and rows[0][0] == OPEN and rows[-1][:2] == (CLOSE, SHUTDOWN)
    second = list(read_recording(files[1:]))
    assert second[0][0] == MESSAGE and second[0][3] == rows[0][3]     # the midnight file continues the connection
    assert lines[0][0] == PROCESS_START and PROCESS_START not in {r[0] for r in rows}


def test_a_kill_leaves_a_file_without_close_or_trailer_and_ids_restart(tmp_path):
    kill = DAY.open_ms + 2 * MIN
    tape = SyntheticTape(5)
    lines = list(tape.lines(DAY.open_ms - 5 * MIN, DAY.open_ms + 10 * MIN, [Fault(KILL, kill, 30 * SEC, refused=2)]))
    files = write_recording(tmp_path, lines)
    assert len(files) == 2
    with pytest.raises(EOFError):
        gzip.decompress(files[0].read_bytes())
    text = zlib.decompressobj(wbits=31).decompress(files[0].read_bytes()).decode("utf-8")
    whole, partial = text.rsplit("\n", 1)
    assert partial and whole.splitlines()[-1].startswith(partial)
    killed = [i for i, line in enumerate(lines) if line[0] == PROCESS_KILL]
    first, second = list(read_recording(files[:1])), list(read_recording(files[1:]))
    assert first == recorded(lines[:killed[0]])
    assert CLOSE not in {r[0] for r in first} and first[0][:2] == (OPEN, "") and first[0][3] == 1
    assert second[0][0] == OPEN and second[0][3] == 3                  # two refused handshakes used ids 1 and 2
    assert second[0][2] >= kill + 30 * SEC


def test_the_faults_show_in_the_recording():
    o = DAY.open_ms
    faults = [Fault(LATE_PRINTS, o + 2 * MIN, MIN, "AAPL", shift_ms=6 * SEC),
              Fault(EARLY_PRINTS, o + 4 * MIN, MIN, "MSFT", shift_ms=3 * SEC),
              Fault(HALT, o + 6 * MIN, 3 * MIN, "TSLA"),
              Fault(DISCONNECT, o + 10 * MIN, refused=1), Fault(REPLAYED_PRINTS, o + 10 * MIN, MIN),
              Fault(SILENCE, o + 15 * MIN, 20 * SEC), Fault(PAUSE, o + 20 * MIN, 30 * SEC)]
    lines = recorded(SyntheticTape(9, regular_rate=2.0).lines(o - 10 * MIN, o + 30 * MIN, faults))
    found = trade_prints(lines)

    def ages(symbol, at, span):
        return [recv - i["t"] for recv, _, i in found if i["s"] == symbol and at <= recv < at + span]
    assert min(ages("AAPL", o + 2 * MIN, MIN)) > 6 * SEC and max(ages("AAPL", o + 3 * MIN, MIN)) < 4 * SEC
    assert set(ages("MSFT", o + 4 * MIN, MIN)) == {-3 * SEC}
    assert ages("TSLA", o + 6 * MIN, 3 * MIN) == [] and ages("TSLA", o + 9 * MIN, MIN)
    sizes = [i["v"] for _, _, i in found]
    assert any(s < 100 for s in sizes) and any(s % 100 == 0 for s in sizes)
    assert any("c" not in i for _, _, i in found) and {"1", "12"} <= {c for _, _, i in found for c in i.get("c", [])}

    closes = [(raw, ms) for kind, raw, ms, _ in lines if kind == CLOSE]
    opens = [(ms, conn) for kind, _, ms, conn in lines if kind == OPEN]
    assert closes[0] == (DROPPED, o + 10 * MIN)
    assert opens[1][1] == opens[0][1] + 2                               # one refused handshake skipped an id
    replay = next(raw for kind, raw, ms, _ in lines if kind == MESSAGE and ms == opens[1][0])
    assert all(opens[1][0] - 1_000 > item["t"] >= opens[1][0] - MIN for item in json.loads(replay)["data"])
    assert (SILENT, o + 15 * MIN + 4 * SEC) in closes                  # 5 s after the last message
    assert ("process paused for 30 s", o + 20 * MIN + 30 * SEC) in closes
    assert closes[-1][0] == SHUTDOWN


def test_a_full_day_holds_pre_and_post_market_prints(clean_day):
    tape, lines, _, _ = clean_day
    phases = {phase_in(DAY, item["t"]) for _, _, item in trade_prints(lines)}
    assert {PRE, REGULAR, POST} <= phases
    assert all(tape.truth[(DAY.day, s)] > 0 for s in SYMBOLS)


# Connections and disconnected time.

W0, W1 = window(DAY)
O, C = DAY.open_ms, DAY.close_ms


def gaps(lines, start=W0, end=W1):
    links = Connections()
    for kind, raw, ms in lines:
        links.line(kind, raw, ms)
    links.finish()
    return [(g.start_ms, g.end_ms) for g in links.disconnected(start, end)]


def test_one_connection_across_the_window_has_no_disconnected_time():
    assert gaps([(OPEN, "", W0 - MIN), (MESSAGE, "m", W0 + SEC), (MESSAGE, "m", W1 + SEC)]) == []


def test_time_from_a_close_to_the_next_open_is_disconnected():
    lines = [(OPEN, "", W0 - MIN), (MESSAGE, "m", W0 + 10 * SEC), (CLOSE, DROPPED, W0 + 20 * SEC),
             (OPEN, "", W0 + 27 * SEC), (MESSAGE, "m", W1 + SEC)]
    assert gaps(lines) == [(W0 + 20 * SEC, W0 + 27 * SEC)]


def test_a_killed_connection_ends_at_its_last_line_before_the_next_open():
    lines = [(OPEN, "", W0 - MIN), (MESSAGE, "m", W0 + 10 * SEC), (MESSAGE, "m", W0 + 11 * SEC),
             (OPEN, "", W0 + 40 * SEC), (MESSAGE, "m", W1 + SEC)]
    assert gaps(lines) == [(W0 + 11 * SEC, W0 + 40 * SEC)]


def test_a_paused_close_adds_its_seconds_before_the_close():
    lines = [(OPEN, "", W0 - MIN), (MESSAGE, "m", W0 + 10 * SEC), (CLOSE, "process paused for 12 s", W0 + 30 * SEC),
             (OPEN, "", W0 + 35 * SEC), (MESSAGE, "m", W1 + SEC)]
    assert gaps(lines) == [(W0 + 18 * SEC, W0 + 35 * SEC)]


def test_a_stream_starting_or_ending_inside_the_window_is_disconnected_outside_its_lines():
    started_inside = [(MESSAGE, "m", W0 + 10 * SEC), (MESSAGE, "m", W1 + SEC)]
    assert gaps(started_inside) == [(W0, W0 + 10 * SEC)]
    ended_inside = [(MESSAGE, "m", W0 - SEC), (MESSAGE, "m", W1 - 5 * SEC)]
    assert gaps(ended_inside) == [(W1 - 5 * SEC, W1)]
    closed_inside = [(OPEN, "", W0 - SEC), (CLOSE, SHUTDOWN, W1 - 5 * SEC)]
    assert gaps(closed_inside) == [(W1 - 5 * SEC, W1)]
    assert gaps([]) == [(W0, W1)]


def test_gaps_outside_the_window_are_ignored_and_overlaps_merge():
    lines = [(OPEN, "", W0 - 2 * MIN), (CLOSE, DROPPED, W0 - MIN), (OPEN, "", W0 - 30 * SEC),
             (CLOSE, "process paused for 20 s", W0 + 10 * SEC), (OPEN, "", W0 + 15 * SEC),
             (CLOSE, DROPPED, W1 + SEC), (OPEN, "", W1 + MIN)]
    assert gaps(lines) == [(W0, W0 + 15 * SEC)]


def test_connection_ids_are_never_used(clean_day):
    _, lines, _, _ = clean_day
    renumbered = [(kind, raw, ms, 7 if i % 2 else 2) for i, (kind, raw, ms, _) in enumerate(recorded(lines))]
    same, other = measure(recorded(lines), DAY, SYMBOLS, RULES), measure(renumbered, DAY, SYMBOLS, RULES)
    assert (other.gaps, other.connections, other.symbols) == (same.gaps, same.connections, same.symbols)
    assert other.messages == same.messages


def test_reading_stops_at_the_first_line_ten_minutes_after_the_window_which_still_extends_the_connection():
    def lines():
        yield OPEN, "", W0 - MIN, 1
        yield MESSAGE, synthetic.PING, W1 - 5 * SEC, 1
        yield MESSAGE, synthetic.PING, W1 + 10 * MIN, 1
        yield MESSAGE, synthetic.PING, W1 + 10 * MIN + tapecheck.TRAIL_MS, 1
        raise AssertionError("read past the margin")
    tape = measure(lines(), DAY, SYMBOLS, RULES)
    assert tape.lines == 2 and tape.gaps == [] and tape.complete


def test_a_pause_close_written_just_after_waking_past_the_margin_still_counts():
    """A freeze from 19:00 until the next morning, where the recorder first writes what arrived during the freeze and
    only then closes with the pause it measured."""
    woke = W1 + 12 * 60 * MIN
    lines = [(OPEN, "", W0 - MIN, 1), (MESSAGE, synthetic.PING, W1 - 60 * MIN, 1),
             (MESSAGE, synthetic.PING, woke, 1), (MESSAGE, synthetic.PING, woke + 5, 1),
             (CLOSE, f"process paused for {(woke + 2 * SEC - (W1 - 60 * MIN)) // SEC} s", woke + 2 * SEC, 1),
             (OPEN, "", woke + 7 * SEC, 2)]
    tape = measure(lines, DAY, SYMBOLS, RULES)
    assert [(g.start_ms, g.end_ms) for g in tape.gaps] == [(W1 - 60 * MIN, W1)]
    assert clean_reason(tape) == DISCONNECTED_REASON and tape.complete
    late = lines[:3] + [(MESSAGE, synthetic.PING, woke + tapecheck.TRAIL_MS, 1)] + lines[4:]
    assert measure(late, DAY, SYMBOLS, RULES).gaps == []       # a close later than the trail is not read


def test_a_recording_that_ends_inside_its_connection_before_the_margin_is_incomplete():
    trade = synthetic.trade_message([{"p": 10.0, "s": "AAPL", "t": W1 - 2 * MIN - 150, "v": 100}])
    quiet = [(OPEN, "", W0 - MIN, 1), (MESSAGE, trade, W1 - 2 * MIN, 1)]   # a post market print and no ping after it
    assert not measure(quiet, DAY, SYMBOLS, RULES).complete
    assert not measure(quiet + [(MESSAGE, synthetic.PING, W1 + 5 * MIN, 1)], DAY, SYMBOLS, RULES).complete
    stopped = quiet + [(CLOSE, SHUTDOWN, W1 + 5 * MIN, 1)]
    assert measure(stopped, DAY, SYMBOLS, RULES).complete
    assert measure(quiet + [(CLOSE, DROPPED, W1 - MIN, 1)], DAY, SYMBOLS, RULES).complete   # a real disconnect
    assert measure([], DAY, SYMBOLS, RULES).complete


def test_an_incomplete_recording_exits_with_code_2_and_writes_no_record(tmp_path, config, capsys):
    trade = synthetic.trade_message([{"p": 10.0, "s": "AAPL", "t": W1 - 2 * MIN - 150, "v": 100}])
    lines = [(PROCESS_START, "", W0 - MIN, 0), (OPEN, "", W0 - MIN, 1), (MESSAGE, trade, W1 - 2 * MIN, 1)]
    write_recording(tmp_path / "raw", lines)
    official = tmp_path / "official_2026_10_13.toml"
    official.write_text(official_toml(dict.fromkeys(SYMBOLS, 1000), CANARY_CLOSES), encoding="utf-8")
    code, out, err = run_main(capsys, config, tmp_path / "raw", official)
    assert (code, out) == (2, "") and err == f"error: {tapecheck.INCOMPLETE}\n"
    assert not list(tmp_path.glob("coverage_*.json"))
    write_recording(tmp_path / "stopped", lines + [(CLOSE, SHUTDOWN, W1 + 5 * MIN, 1)])
    code, out, _ = run_main(capsys, config, tmp_path / "stopped", official)
    fields = summary(out)                                     # connected all day, but silent through the session
    assert code == 1 and fields["clean session"] == "fail" and fields["reason"] == OUTAGE_REASON
    assert fields["disconnected time"] == "0.000 s" and fields["outage time"] == "23400.000 s"
    assert [p.name for p in tmp_path.glob("coverage_*.json")] == ["coverage_2026_10_13.json"]


def test_an_official_file_that_git_would_track_is_refused(clean_day, tmp_path, config, capsys):
    tape, _, _, folder = clean_day
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    official = write_official(repo, DAY, tape)
    code, out, err = run_main(capsys, config, folder, official)
    assert (code, out) == (2, "") and err == f"error: {tapecheck.NOT_IGNORED}\n"
    assert not list(repo.glob("coverage_*.json"))
    (repo / ".gitignore").write_text("official_*.toml\ncoverage_*.json\n", encoding="utf-8")
    code, _, _ = run_main(capsys, config, folder, official)
    assert code == 0 and (repo / "coverage_2026_10_13.json").stat().st_mode & 0o777 == 0o600


def test_lines_of_other_kinds_belong_to_no_connection():
    lines = [(OPEN, "", W0 - MIN, 1), (CLOSE, DROPPED, W0 + MIN, 1), ("news", '{"seq":1}', W0 + 2 * MIN, 0),
             (OPEN, "", W0 + 3 * MIN, 2), (MESSAGE, synthetic.PING, W1 + SEC, 2)]
    tape = measure(lines, DAY, SYMBOLS, RULES)
    assert [(g.start_ms, g.end_ms) for g in tape.gaps] == [(W0 + MIN, W0 + 3 * MIN)] and tape.lines == 4
    assert not tape.unknown


def test_session_files_start_from_the_last_file_begun_before_04_00(tmp_path):
    names = ["2026-10-12_000000.jsonl.gz", "2026-10-13_000000.jsonl.gz", "2026-10-13_073000.jsonl.gz",
             "2026-10-13_081500.jsonl.gz", "2026-10-14_000000.jsonl.gz"]
    files = [tmp_path / name for name in names]
    assert [f.name for f in session_files(files, W0)] == names[2:]       # 04:00 EDT is 08:00 UTC
    assert session_files(files, W0 - 8 * 60 * MIN) == files[1:]
    assert session_files(files[3:], W0) == files[3:]


# Silences and outages.

def test_connected_stretches_and_silences_never_span_disconnected_time():
    lost = [Gap(O + 10 * SEC, O + 20 * SEC, DROPPED), Gap(O + 50 * SEC, O + 60 * SEC, DROPPED)]
    stretches = connected_stretches(lost, O, O + MIN)
    assert stretches == [(O, O + 10 * SEC), (O + 20 * SEC, O + 50 * SEC)]
    times = [O + 4 * SEC, O + 4 * SEC, O + 30 * SEC, O + 55 * SEC, O + MIN]
    assert list(silences(stretches, times)) == [(O, O + 4 * SEC), (O + 4 * SEC, O + 10 * SEC),
                                                (O + 20 * SEC, O + 30 * SEC), (O + 30 * SEC, O + 50 * SEC)]
    assert list(silences(stretches, [])) == stretches
    assert connected_stretches([Gap(O - SEC, O + MIN + SEC, DROPPED)], O, O + MIN) == []
    assert tapecheck.longest([(0, 5), (10, 15), (20, 24)]) == (0, 5) and tapecheck.longest([]) is None


def test_an_outage_counts_in_full_on_an_open_connection_in_session_only():
    """Pings every 10 s across the window, a 40 s hole in the session and a 2 minute hole in pre market."""
    hole = (O + 60 * MIN, O + 60 * MIN + 40 * SEC)
    early = (W0 + 30 * MIN, W0 + 32 * MIN)
    times = [t for t in range(W0, W1 + 11 * MIN, 10 * SEC)
             if not (hole[0] < t < hole[1] or early[0] < t < early[1])]
    lines = [(OPEN, "", W0 - MIN, 1)] + [(MESSAGE, synthetic.PING, t, 1) for t in times]
    result = check(lines, flat(), SYMBOLS, RULES, STALE)
    assert result.coverage.outages == (Gap(*hole, OUTAGE),)
    assert not result.clean and result.reason == OUTAGE_REASON and result.tape.disconnected_ms == 0
    assert result.coverage.shares == {PRE: 1.0, REGULAR: (23_400 - 40) / 23_400, POST: 1.0}
    assert record_of(result)["outage_seconds"] == 40.0
    longer = check(lines, flat(), SYMBOLS, RULES, 41 * SEC)
    assert longer.coverage.outages == () and longer.clean and longer.reason is None


def test_an_outage_ends_at_a_silence_close_and_starts_again_at_the_next_open():
    lines = [(OPEN, "", W0 - MIN, 1), *((MESSAGE, synthetic.PING, t, 1) for t in range(W0, O + 10 * MIN + 1, 5 * SEC)),
             (CLOSE, "no message for 30 s in session", O + 10 * MIN + 35 * SEC, 1),
             (OPEN, "", O + 10 * MIN + 40 * SEC, 2),
             *((MESSAGE, synthetic.PING, t, 2) for t in range(O + 11 * MIN + 20 * SEC, W1 + 11 * MIN, 5 * SEC))]
    result = check(lines, flat(), SYMBOLS, RULES, STALE)
    assert result.coverage.outages == (Gap(O + 10 * MIN, O + 10 * MIN + 35 * SEC, OUTAGE),
                                       Gap(O + 10 * MIN + 40 * SEC, O + 11 * MIN + 20 * SEC, OUTAGE))
    assert [(g.start_ms, g.end_ms) for g in result.tape.gaps] == [(O + 10 * MIN + 35 * SEC, O + 10 * MIN + 40 * SEC)]
    assert result.tape.silence_closes == 1 and result.coverage.lost_ms[REGULAR] == 80 * SEC
    assert result.reason == DISCONNECTED_REASON


# Coverage on whole synthetic days.

def test_a_clean_day_reports_its_partial_coverage_from_the_raw_recording_alone(clean_day, tmp_path, config, capsys):
    tape, _, _, folder = clean_day
    official = write_official(tmp_path, DAY, tape, SHARES)
    code, out, err = run_main(capsys, config, folder, official)
    assert (code, err) == (0, "")
    fields = summary(out)
    assert out.splitlines()[0] == "tape coverage"
    assert fields["clean session"] == "pass" and fields["reason"] == CLEAN_REASON
    assert fields["symbols"] == fields["symbols with prints"] == "8" and fields["covered ratio computed"] == "pass"
    assert fields["disconnected time"] == fields["outage time"] == "0.000 s"
    assert fields["connections"] == "1" and fields["files"] == "2" and fields["delay verdict"] == "pass"
    assert fields["max_print_delay_seconds"] == "2.000 s" and fields["late_print_seconds"] == "4.000 s"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["coverage_2026_10_13.json", official.name]
    assert sorted(p.suffix for p in folder.iterdir()) == [".gz", ".gz"]   # no database anywhere
    record = record_file(tmp_path)
    assert set(record) == RECORD_KEYS and record["kind"] == "coverage" and record["session"] == "2026-10-13"
    assert record["clean"] is True and record["early_close"] is False and record["reason"] == CLEAN_REASON
    assert record["ratios"] == pytest.approx(SHARES) and record["median_ratio"] == pytest.approx(0.0375)
    assert record["ratios_covered"] == record["ratios"] and record["median_ratio_covered"] == record["median_ratio"]
    assert record["median_ratio"] == statistics.median(record["ratios"].values())
    assert record["covered_share"] == {PRE: 1.0, REGULAR: 1.0, POST: 1.0}
    assert (record["disconnected_seconds"], record["outage_seconds"]) == (0.0, 0.0)
    assert (tmp_path / "coverage_2026_10_13.json").stat().st_mode & 0o777 == 0o600
    assert read_records(tmp_path) == [CoverageRecord(DAY.day, True, False, record["median_ratio"],
                                                     record["median_ratio"], record["ratios"], record["ratios"],
                                                     0.0, 0.0)]


def test_a_rerun_replaces_its_record_whole(clean_day, tmp_path, config, capsys):
    tape, _, _, folder = clean_day
    official = write_official(tmp_path, DAY, tape, SHARES)
    run_main(capsys, config, folder, official)
    write_official(tmp_path, DAY, tape)
    assert run_main(capsys, config, folder, official)[0] == 0
    assert record_file(tmp_path)["ratios"] == pytest.approx(dict.fromkeys(SYMBOLS, 1.0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["coverage_2026_10_13.json", official.name]


def test_default_output_holds_only_verdicts_counts_and_timings(clean_day, tmp_path, config, capsys):
    tape, _, _, folder = clean_day
    official = write_official(tmp_path, DAY, tape, SHARES)
    figures = [f"{v:.4f}" for v in CANARY_CLOSES.values()] + [f"{tape.truth[(DAY.day, s)]}" for s in SYMBOLS]
    figures += [f"{tape.closes[(DAY.day, s)]:.4f}" for s in SYMBOLS]
    _, out, _ = run_main(capsys, config, folder, official)
    lines = out.splitlines()
    assert lines[0] == "tape coverage" and lines[-1].startswith("9 rows hidden")
    assert all(ALLOWED.fullmatch(line) for line in lines[1:-1]), lines
    ratios = [f"{r:.6f}" for r in record_file(tmp_path)["ratios"].values()] + ["0.037500"]
    assert [f for f in figures + ratios if f in out] == []
    _, detail, _ = run_main(capsys, config, folder, official, "--detail")
    assert [f for f in figures + ratios if f not in detail] == []
    for title in ("tape prints", "covered time", "condition histogram", "disconnected intervals", "outage intervals",
                  "parse counts"):
        assert title in detail.splitlines()
    assert "hidden" not in detail


def test_a_file_begun_at_utc_midnight_in_post_market_continues_the_connection(tmp_path, config, capsys):
    tape, lines = full_day(EST_DAY, seed=21, regular_rate=0.1, extended_rate=0.01)
    files = write_recording(tmp_path / "raw", lines)
    assert [f.name[:13] for f in files] == ["2026-11-03_08", "2026-11-04_00"]
    assert list(read_recording(files[1:]))[0][0] == MESSAGE
    code, out, _ = run_main(capsys, config, tmp_path / "raw", write_official(tmp_path, EST_DAY, tape))
    assert code == 0 and summary(out)["clean session"] == "pass"
    assert record_file(tmp_path, EST_DAY.day)["median_ratio"] == pytest.approx(1.0)


def disconnected_after(lines, kind, raw):
    """Seconds from the close with reason raw, or the last line before the kill, to the next open."""
    for i, (k, r, ms, _) in enumerate(lines):
        if k == kind and (raw is None or r == raw):
            end = ms if kind == CLOSE else lines[i - 1][2]
            return (next(m for k2, _, m, _ in lines[i:] if k2 == OPEN) - end) / SEC
    raise AssertionError(kind)


def test_disconnected_time_by_every_rule_makes_the_session_not_clean(tmp_path, config, capsys):
    faults = [Fault(DISCONNECT, DAY.open_ms - 3 * 60 * MIN, refused=1),
              Fault(KILL, DAY.open_ms + 60 * MIN, 45 * SEC, refused=1),
              Fault(PAUSE, DAY.close_ms + 60 * MIN, 25 * SEC)]
    tape, lines = full_day(DAY, seed=12, faults=faults, regular_rate=0.05, extended_rate=0.005)
    files = write_recording(tmp_path / "raw", lines)
    assert len(files) == 3
    kept = [line for line in lines if line[0] != PROCESS_START]
    expected = (disconnected_after(kept, CLOSE, DROPPED) + disconnected_after(kept, PROCESS_KILL, None)
                + 25 + disconnected_after(kept, CLOSE, "process paused for 25 s"))
    code, out, _ = run_main(capsys, config, tmp_path / "raw", write_official(tmp_path, DAY, tape))
    fields = summary(out)
    assert code == 1 and fields["clean session"] == "fail" and fields["reason"] == DISCONNECTED_REASON
    assert fields["disconnected time"] == f"{expected:.3f} s" and fields["connections"] == "4"
    record = record_file(tmp_path)
    assert record["clean"] is False and record["reason"] == DISCONNECTED_REASON
    assert record["disconnected_seconds"] == pytest.approx(expected)


def test_lost_time_is_left_out_of_the_covered_ratio_on_an_even_tape(tmp_path, config, capsys):
    """Half an hour killed in session on a tape with an even rate in each phase, against the true volumes, so the plain
    median falls by about the lost share while the covered median stays near 1."""
    tape, lines = full_day(DAY, seed=14, faults=[Fault(KILL, DAY.open_ms + 120 * MIN, 30 * MIN)])
    write_recording(tmp_path / "raw", lines)
    code, _, _ = run_main(capsys, config, tmp_path / "raw", write_official(tmp_path, DAY, tape))
    record = record_file(tmp_path)
    lost = 1 - record["covered_share"][REGULAR]
    assert code == 1 and 0.07 < lost < 0.08 and record["covered_share"][PRE] == record["covered_share"][POST] == 1.0
    assert record["median_ratio"] < 0.95 and record["median_ratio_covered"] == pytest.approx(1.0, abs=0.02)


def test_the_covered_ratio_scales_each_phase_by_its_length_over_its_covered_length():
    """A recording that starts halfway through pre market, with one AAPL print of 100 shares in pre market, 1000 in
    the session and 50 in post market, against an official volume of 10000."""
    half = W0 + (O - W0) // 2

    def trade(vendor, size):
        return MESSAGE, synthetic.trade_message([{"p": 10.0, "s": "AAPL", "t": vendor, "v": size}]), vendor + 100, 1
    pings = [(MESSAGE, synthetic.PING, t, 1) for t in range(half, W1 + 11 * MIN, 10 * SEC)]
    prints = [trade(half + 10 * MIN, 100), trade(O + 60 * MIN, 1000), trade(C + 60 * MIN, 50)]
    lines = sorted([(OPEN, "", half, 1), *pings, *prints], key=lambda line: (line[2], line[0] != OPEN))
    result = check(lines, flat(10_000.0), SYMBOLS, RULES, STALE)
    cov = result.coverage
    assert cov.shares == {PRE: 0.5, REGULAR: 1.0, POST: 1.0} and cov.outages == ()
    assert cov.ratios["AAPL"] == 1150 / 10_000 and cov.covered["AAPL"] == 0.125
    assert cov.ratios["MSFT"] == cov.covered["MSFT"] == 0.0 and cov.median == cov.median_covered == 0.0
    assert result.reason == DISCONNECTED_REASON
    unknown = check([line for line in lines if line[2] >= O], flat(10_000.0), SYMBOLS, RULES, STALE)
    assert unknown.coverage.shares[PRE] == 0.0 and unknown.coverage.covered is None
    record = record_of(unknown)
    assert record["ratios_covered"] is None and record["median_ratio_covered"] is None
    assert summary(output.render(report(unknown, 1)))["covered ratio computed"] == "fail"


def test_replayed_prints_are_left_out_of_the_volume():
    o = DAY.open_ms
    tape = SyntheticTape(13, regular_rate=1.0)
    lines = recorded(tape.lines(o - 5 * MIN, o + 15 * MIN, [Fault(DISCONNECT, o + 5 * MIN),
                                                             Fault(REPLAYED_PRINTS, o + 5 * MIN, 2 * MIN)]))
    reopened = [ms for kind, _, ms, _ in lines if kind == OPEN][1]
    replay = [i for recv, _, i in trade_prints(lines) if recv == reopened and i["t"] < reopened - 1_000]
    delivered = [i for recv, _, i in trade_prints(lines) if not (recv == reopened and i["t"] < reopened - 1_000)]
    measured = measure(lines, DAY, SYMBOLS, RULES)
    assert replay and sum(t.replayed for t in measured.symbols.values()) == len(replay)
    for symbol in SYMBOLS:
        own = [i for i in delivered if i["s"] == symbol and phase_in(DAY, i["t"]) != OTHER]
        assert measured.symbols[symbol].full == sum(i["v"] for i in own)
        assert measured.symbols[symbol].prints == len(own)


def test_eligible_stamps_follow_live_health():
    """Only an eligible print received in the session with its vendor time there, and not replayed, renews a symbol,
    once per message."""
    at = O + MIN
    items = [{"p": 10.0, "s": "AAPL", "t": at - 100, "v": 100}, {"p": 10.0, "s": "AAPL", "t": at - 50, "v": 100},
             {"p": 10.0, "s": "MSFT", "t": at - 6 * SEC, "v": 100},          # late
             {"p": 10.0, "s": "NVDA", "t": at + 3 * SEC, "v": 100},          # early
             {"p": 10.0, "s": "META", "t": at - 100, "v": 100, "c": ["12"]},  # an excluded condition
             {"p": 10.0, "s": "TSLA", "t": at - 100, "v": 100}]
    replayed = [{"p": 10.0, "s": "QQQ", "t": at - 3 * SEC, "v": 100}]
    before_open = [{"p": 10.0, "s": "AMZN", "t": O - 50, "v": 100}]      # received after the open, traded before
    lines = [(OPEN, "", W0, 1), (MESSAGE, synthetic.trade_message(before_open), O + 50, 1),
             (CLOSE, DROPPED, at - 2 * SEC, 1), (OPEN, "", at, 2),
             (MESSAGE, synthetic.trade_message(replayed), at, 2), (MESSAGE, synthetic.trade_message(items), at, 2)]
    tape = measure(lines, DAY, SYMBOLS, PriceRules(4_000, frozenset({"12"})))
    renewed = {s: list(t.eligible_ms) for s, t in tape.symbols.items() if t.eligible_ms}
    assert renewed == {"AAPL": [at], "TSLA": [at]}
    assert list(tape.messages) == [O + 50, at, at] and tape.symbols["QQQ"].replayed == 1
    assert tape.symbols["AMZN"].prints == 1 and tape.symbols["AMZN"].late == 0


# The clean flag, the record and criterion 1.1.

def test_an_early_close_or_a_missing_recording_is_never_clean():
    early = measure([], EARLY_CLOSE, SYMBOLS, RULES)
    assert clean_reason(early) == EARLY_CLOSE_REASON
    missing = measure([], DAY, SYMBOLS, RULES)
    assert clean_reason(missing) == DISCONNECTED_REASON
    assert missing.disconnected_ms == W1 - W0 and missing.connections == 0
    assert clean_reason(measure([(OPEN, "", W0 - MIN, 1), (CLOSE, SHUTDOWN, W1 + MIN, 1)], DAY, SYMBOLS, RULES),
                        [Gap(O, O + STALE, OUTAGE)]) == OUTAGE_REASON


@pytest.mark.parametrize("baseline, median, expected", [
    (0.04, 0.02, True), (0.04, 0.08, True), (0.04, 0.04, True), (0.04, 0.0199, False), (0.04, 0.0801, False),
    (0.04, None, False), (None, 0.04, False), (0.0, 0.0, False), (0.04, 0.0, False), (0.04, float("nan"), False),
    (float("inf"), 0.04, False),
])
def test_criterion_1_1_compares_within_a_factor_of_2_either_way(baseline, median, expected):
    assert within_factor(baseline, median) is expected


def test_records_are_read_in_session_order_and_the_replaced_verdicts_are_ignored(tmp_path):
    def record(day, clean=True, covered=0.04):
        return {"kind": "coverage", "session": day, "clean": clean, "reason": CLEAN_REASON, "early_close": False,
                "median_ratio": 0.03, "median_ratio_covered": covered, "ratios": {"AAPL": 0.03},
                "ratios_covered": None if covered is None else {"AAPL": covered}, "disconnected_seconds": 0,
                "outage_seconds": 0.0, "covered_share": {PRE: 1.0, REGULAR: 1.0, POST: 1.0}}
    assert read_records(tmp_path) == []
    write_record(tmp_path, record("2026-10-20", covered=None))
    write_record(tmp_path, record("2026-10-12", clean=False))
    (tmp_path / "tape_2026_10_13.json").write_text('{"verdict": "consolidated"}', encoding="utf-8")
    (tmp_path / "coverage_notes.json").write_text("{}", encoding="utf-8")
    found = read_records(tmp_path)
    assert [(r.session, r.clean, r.median_ratio_covered, r.ratios_covered) for r in found] == [
        (date(2026, 10, 12), False, 0.04, {"AAPL": 0.04}), (date(2026, 10, 20), True, None, None)]
    assert not list(tmp_path.glob(".*"))                      # the temporary file was replaced into place


@pytest.mark.parametrize("change", [
    {"kind": "tape"}, {"session": "2026-10-14"}, {"clean": 1}, {"median_ratio": True}, {"median_ratio": None},
    {"ratios": {}}, {"ratios": {"AAPL": "0.03"}}, {"median_ratio_covered": None}, {"ratios_covered": None},
    {"disconnected_seconds": float("inf")}, {"outage_seconds": None}, {"early_close": None},
])
def test_a_malformed_record_is_refused(tmp_path, change):
    record = {"kind": "coverage", "session": "2026-10-13", "clean": True, "reason": CLEAN_REASON,
              "early_close": False, "median_ratio": 0.03, "median_ratio_covered": 0.03, "ratios": {"AAPL": 0.03},
              "ratios_covered": {"AAPL": 0.03}, "disconnected_seconds": 0.0, "outage_seconds": 0.0}
    record.update(change)
    (tmp_path / "coverage_2026_10_13.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(TapeCheckError, match="coverage_2026_10_13.json is not a coverage record"):
        read_records(tmp_path)
    (tmp_path / "coverage_2026_10_13.json").write_text("{", encoding="utf-8")
    with pytest.raises(TapeCheckError, match="is not a coverage record"):
        read_records(tmp_path)


# The delay rule.

def test_percentiles_are_nearest_rank():
    counts = {value: 1 for value in range(1, 21)}
    assert [percentile(counts, p) for p in (50, 95, 100)] == [10, 19, 20]
    assert percentile({5: 3}, 95) == 5 and percentile({}, 50) is None


@pytest.mark.parametrize("ages, verdict, setting", [
    ([5_000] * 3, "pass", "5.000 s"), ([5_000, 5_001, 5_001], "fail", "5.000 s"), ([-6_000] * 3, "fail", "2.000 s"),
    ([-2_000] * 3, "pass", "2.000 s"), ([-2_001] * 3, "fail", "2.000 s"),
    ([100, 1_333, 1_334], "pass", "3.000 s"), ([100, 2_666, 2_666], "pass", "4.000 s"),
])
def test_the_delay_verdict_and_settings_come_from_the_regular_session_delays(ages, verdict, setting):
    at = DAY.open_ms + MIN
    message = synthetic.trade_message([{"p": 10.0, "s": "AAPL", "t": at - age, "v": 100} for age in ages])
    pre = synthetic.trade_message([{"p": 10.0, "s": "AAPL", "t": W0 + MIN - 60_000, "v": 100}])
    lines = [(OPEN, "", W0 - MIN, 1), (MESSAGE, pre, W0 + MIN, 1), (MESSAGE, message, at, 1)]
    fields = summary(output.render(report(check(lines, flat(), SYMBOLS, RULES, STALE), 0)))
    assert (fields["delay verdict"], fields["max_print_delay_seconds"]) == (verdict, setting)


def test_a_delayed_feed_fails_the_delay_verdict_and_caps_the_settings():
    o = DAY.open_ms
    tape = SyntheticTape(17, regular_rate=1.0)
    lines = recorded(tape.lines(o, o + 10 * MIN, [Fault(LATE_PRINTS, o, 10 * MIN, shift_ms=6 * SEC)]))
    fields = summary(output.render(report(check(lines, flat(), SYMBOLS, RULES, STALE), 0)))
    assert fields["delay verdict"] == "fail"
    assert (fields["max_print_delay_seconds"], fields["late_print_seconds"]) == ("5.000 s", "10.000 s")
    empty = summary(output.render(report(check([], flat(), SYMBOLS, RULES, STALE), 0)))
    assert empty["delay verdict"] == "fail" and empty["max_print_delay_seconds"] == "not computed"


# The official figures.

def filled_example():
    return EXAMPLE.read_text(encoding="utf-8").replace("volume = 0", "volume = 123").replace("close = 0.0",
                                                                                              "close = 4.5")


def test_the_example_is_a_placeholder_for_the_config_symbols(tmp_path):
    copy = tmp_path / "official_2026_10_13.toml"
    copy.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(TapeCheckError, match="AAPL volume must be a number above zero"):
        load_official(copy)
    copy.write_text(filled_example(), encoding="utf-8")
    official = load_official(copy)
    assert official.day == DAY.day and tuple(official.volumes) == SYMBOLS
    assert set(official.volumes.values()) == {123.0} and set(official.closes.values()) == {4.5}


@pytest.mark.parametrize("name", ["official.toml", "official_2026_13_01.toml", "official_2026-10-13.toml"])
def test_the_official_file_is_named_by_its_session_date(tmp_path, name):
    path = tmp_path / name
    path.write_text(filled_example(), encoding="utf-8")
    with pytest.raises(TapeCheckError, match="official_YYYY_MM_DD.toml"):
        load_official(path)


@pytest.mark.parametrize("data, message", [
    ({"symbols": {"AAPL": {"volume": 9, "close": 2}}}, "consolidated_label"),
    ({"consolidated_label": True, "symbols": {}}, "symbols"),
    ({"consolidated_label": True, "extra": 1, "symbols": {"AAPL": {"volume": 9, "close": 2}}}, "unknown keys extra"),
    ({"consolidated_label": True, "symbols": {"AAPL": {"volume": 9, "close": 2, "open": 3}}}, "unknown keys open"),
    ({"consolidated_label": True, "symbols": {"AAPL": {"volume": "9", "close": 2}}}, "AAPL volume"),
    ({"consolidated_label": True, "symbols": {"AAPL": {"volume": True, "close": 2}}}, "AAPL volume"),
    ({"consolidated_label": True, "symbols": {"AAPL": {"volume": 9, "close": float("nan")}}}, "AAPL close"),
    ({"consolidated_label": False, "symbols": {"AAPL": {"volume": 9, "close": 2}}}, "AAPL second_volume"),
    ({"consolidated_label": False, "symbols": {"AAPL": {"volume": 1000, "close": 2, "second_volume": 1051}}},
     "differ by more than 5 percent"),
])
def test_official_figures_are_refused_without_naming_a_figure(data, message):
    with pytest.raises(TapeCheckError, match=message) as caught:
        parse_official(data, DAY.day)
    assert not re.search(r"\d{3}", str(caught.value))


def test_a_second_display_within_5_percent_is_accepted():
    data = {"consolidated_label": False, "symbols": {"AAPL": {"volume": 1000, "close": 2, "second_volume": 1050}}}
    assert parse_official(data, DAY.day).volumes == {"AAPL": 1000.0}


def test_bad_inputs_exit_with_code_2_and_one_error_line(clean_day, tmp_path, config, capsys):
    tape, _, _, folder = clean_day
    wrong = tmp_path / "official_2026_10_13.toml"
    wrong.write_text(official_toml({"AAPL": 5}, {"AAPL": 6}), encoding="utf-8")
    code, out, err = run_main(capsys, config, folder, wrong)
    assert (code, out) == (2, "") and err.startswith("error: the official figures must cover exactly")
    holiday = write_official(tmp_path, DAY, tape).rename(tmp_path / "official_2026_11_26.toml")
    code, _, err = run_main(capsys, config, folder, holiday)
    assert code == 2 and "2026-11-26 has no regular session" in err
    code, _, err = run_main(capsys, config, tmp_path / "missing", write_official(tmp_path, DAY, tape))
    assert code == 2 and err.startswith("error: no recording files")
    assert not list(tmp_path.glob("coverage_*.json"))
