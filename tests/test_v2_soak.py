"""The soak of PLAN_V2 section 16 phase 19 and its rehearsal pair. The rehearsal configs load as a pair, pass every
cross check and differ from the run configs only in their own name and folder and the two plumbing values, labelled
never a strategy setting. The soak freezes the newest recording without touching it, joins recordings with news lines
passing through, generates a synthetic run with reconnects, halts, late, early and replayed prints, news bursts and
outages, and a short soak over recordings and a synthetic run through the real stock process code reports memory, CPU
per print, invariants and determinism, with a week over week memory figure that tells windows and weekend news from a
leak and on a Mac passes a soak whose memory in use grew less than 1 MB over the final replayed week, and default
output of verdicts, counts, hashes and timings only. A leak, a planted bad row in any check, a
differing second pass and a busy or unsafe folder each make it fail or refuse. Every recording, database and news line
is synthetic and lives in a test folder."""

from __future__ import annotations

import contextlib
import heapq
import io
import json
import re
import shutil
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from btc_survivor import seeds
from btc_survivor.clock import MS_PER_DAY, MS_PER_HOUR, MS_PER_MINUTE, iso, parse_iso
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN
from btc_survivor.recorder import read_recording
from btc_survivor.v2 import output, sessions
from btc_survivor.v2.config import check_run_pair, load_run_pair
from btc_survivor.v2.news.health import FINBERT_SCORER
from btc_survivor.v2.stocks.news_inputs import (HAIKU_SCORER, HEARTBEAT_LINE, NEWS_LINE, SONNET_SCORER, HeartbeatLine,
                                                ScoreLine)
from btc_survivor.v2.stocks.synthetic import (DISCONNECT, DROPPED, EARLY_PRINTS, HALT, KILL, LATE_PRINTS, PAUSE, PING,
                                              PROCESS_KILL, PROCESS_START, REPLAYED_PRINTS, SILENCE, SILENT, Fault,
                                              SyntheticTape, write_recording)
from helpers import ROOT
from test_v2_group import HASHES, SYMBOLS, THU, at, config_text, heartbeat_lines, score_lines

sys.path.insert(0, str(ROOT / "scripts"))
import v2_soak as soak  # noqa: E402

CONFIGS = ROOT / "configs" / "v2"
RUNS = (CONFIGS / "s1_prices.toml", CONFIGS / "s2_news.toml")
REHEARSALS = (CONFIGS / "rehearsal_s1.toml", CONFIGS / "rehearsal_s2.toml")
PLUMBING = {("time", "warmup_hours"): 0.5, ("trading", "force_trade_after_hours"): 0.5}
REAL_START, REAL_END = at(THU, "09:00"), at(THU, "10:30")
# Four bots per run and label scales after 10 labels, so a short soak trades on its second session.
FAST = {"size": "4", "min_samples": "10"}
SUMMARY_LINE = re.compile(r"[a-z][a-z_ ]*: (pass|fail|-?\d+|[0-9a-f]{64}|\d+\.\d{3} s|[a-z][a-z ]*)")


def differences(a, b) -> dict:
    """Every setting of b that differs from a, by (section, key), a whole section that only one has by (section,
    None)."""
    out = {}
    for section, x in a.sections().items():
        y = b.sections()[section]
        if x is None or y is None:
            if x != y:
                out[(section, None)] = y
            continue
        out.update({(section, key): y[key] for key in x if x[key] != y[key]})
    return out


# The rehearsal pair


def test_the_rehearsal_configs_differ_from_the_run_configs_only_where_the_plan_says():
    runs, rehearsals = load_run_pair(*RUNS), load_run_pair(*REHEARSALS)
    for run, rehearsal, name, folder in zip(runs, rehearsals, ("rehearsal_s1", "rehearsal_s2"),
                                            ("s1_prices", "s2_news")):
        assert differences(run, rehearsal) == {("run", "name"): name,
                                               ("run", "data_dir"): f"../../data/v2/rehearsal/{folder}", **PLUMBING}
        assert rehearsal.data_dir.resolve() == (ROOT / "data" / "v2" / "rehearsal" / folder).resolve()
        assert rehearsal.news_db_path.resolve() == run.news_db_path.resolve() == \
            (ROOT / "data" / "v2" / "news" / "news.db").resolve()
        assert rehearsal.raw_dir.resolve() == run.raw_dir.resolve()
        assert rehearsal.path(rehearsal.stock_feed.lock_file) == run.path(run.stock_feed.lock_file)
        assert rehearsal.experiment_hash != run.experiment_hash
    # The two rehearsal configs differ from each other exactly as S1 and S2 do, in news and evaluation.
    apart = {key: value for key, value in differences(*rehearsals).items() if key[0] != "run"}
    assert apart == {key: value for key, value in differences(*runs).items() if key[0] != "run"}
    assert set(apart) == {("news_input", "use_news"), ("evaluation", None)}
    check_run_pair(*rehearsals)


def test_the_rehearsal_configs_are_labelled_never_a_strategy_setting():
    for path in REHEARSALS:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert "PLUMBING ONLY, never a strategy setting" in lines[0]
        assert "phase 19" in " ".join(lines[:8]) and "never uses this file" in " ".join(lines[:8])
        for key, minutes in (("warmup_hours", "training"), ("force_trade_after_hours", "healthy")):
            line = next(line for line in lines if line.startswith(f"{key} = "))
            assert line.split("#")[1].strip() == f"plumbing only, 30 {minutes} minutes, never a strategy setting"
    assert "data/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_the_soak_runs_the_rehearsal_pair_into_data_v2_soak_by_default():
    args = soak.build_parser().parse_args([])
    assert (args.s1, args.s2) == (soak.DEFAULT_S1, soak.DEFAULT_S2) == REHEARSALS
    assert soak.SOAK_ROOT == ROOT / "data" / "v2" / "soak" and args.out is None and args.raw is None
    assert soak.MAX_GROWTH_MB == 5.0 and args.synthetic_days == 14.0 and soak.WEEK_HOURS == 168.0
    assert soak.MAX_WEEK_GROWTH_MB == 1.0 == args.max_week_growth_mb
    help_text = " ".join(soak.build_parser().format_help().split())
    assert soak.NEVER_FOR_SELECTION in help_text and "never a strategy setting" in help_text


def test_the_synthetic_seed_tag_never_collides_with_a_version_1_tag():
    tags = {v for k, v in vars(seeds).items() if k.isupper() and isinstance(v, int)}
    assert tags and soak.SOAK_TAG not in tags | {4}  # 4 is the version 1 soak's tag


# Recordings


def alive(lines):
    """Lines as a recorder process writes them, dropping news lines while the process is down after a kill."""
    up = True
    for line in lines:
        if line[0] in (PROCESS_START, PROCESS_KILL):
            up = line[0] == PROCESS_START
        elif not up:
            continue
        yield line


def real_tape(start: int = REAL_START, end: int = REAL_END) -> list:
    """A recorder killed and restarted once and killed for good just before the end, so its newest file is cut as
    one still being written, with heartbeat and score lines carrying HASHES."""
    faults = (Fault(KILL, start + 40 * MS_PER_MINUTE, 2 * MS_PER_MINUTE), Fault(KILL, end - MS_PER_MINUTE,
                                                                               60 * MS_PER_MINUTE))
    feed = SyntheticTape(7, SYMBOLS).lines(start, end, faults)
    news = heapq.merge(heartbeat_lines(start, end), score_lines(start, end), key=lambda line: line[2])
    return list(alive(heapq.merge(feed, news, key=lambda line: line[2])))


def test_the_newest_recording_is_copied_up_to_its_size_and_never_touched(tmp_path):
    files = write_recording(tmp_path / "raw", real_tape())
    assert len(files) == 2
    before = files[-1].read_bytes()
    chosen = soak.select_recordings(tmp_path / "raw", tmp_path / "input")
    assert [r.source for r in chosen] == files
    assert chosen[0].path == files[0] and chosen[0].status == "read in place"
    assert chosen[1].path == tmp_path / "input" / files[1].name and chosen[1].status.startswith("newest")
    assert chosen[1].path.read_bytes() == before == files[1].read_bytes()
    with files[1].open("ab") as fh:  # the writer keeps going, and the frozen copy does not move
        fh.write(b"more")
    assert chosen[1].path.read_bytes() == before
    tape = real_tape()
    restart = [i for i, line in enumerate(tape) if line[0] == PROCESS_START][1]
    assert list(read_recording([chosen[1].path])) == [line for line in tape[restart:]
                                                      if line[0] not in (PROCESS_START, PROCESS_KILL)]
    assert soak.select_recordings(tmp_path / "missing", tmp_path / "unused") == []
    assert soak.launch_hashes(chosen) == (HASHES, "recorded")
    assert soak.launch_hashes([]) == (soak.SOAK_HASHES, "soak")


def test_joins_close_the_open_connection_renumber_ids_and_pass_news_lines_through():
    score = ScoreLine(4, 2, "finnhub", "media", "llm", HAIKU_SCORER, "scored", 1, ("AAPL",), ()).encode()
    later = ScoreLine(7, 5, "finnhub", "media", "llm", HAIKU_SCORER, "scored", 1, ("AAPL",), ()).encode()
    beat = HeartbeatLine(9, 1, True, None, *HASHES, ()).encode()
    a = [(OPEN, "", 1, 1), (MESSAGE, "a", 2, 1), (NEWS_LINE, score, 2, 0), (CLOSE, "x", 3, 1), (OPEN, "", 4, 2),
         (MESSAGE, "b", 5, 2)]
    rotated = [(HEARTBEAT_LINE, beat, 6, 0), (MESSAGE, "c", 6, 2), (CLOSE, "x", 7, 2), (OPEN, "", 8, 3),
               (MESSAGE, "d", 9, 3)]
    restarted = [(OPEN, "", 9, 1), (MESSAGE, "e", 10, 1)]
    midway = [(NEWS_LINE, later, 8, 0), (MESSAGE, "f", 8, 7)]
    state = soak.JoinState()
    sources = [(f"s{i}", lambda st, lines=lines: lines) for i, lines in enumerate((a, rotated, restarted, midway))]
    out = list(soak.join(sources, state))
    assert out == a + rotated + [
        (CLOSE, soak.JOIN_REASON, 9, 3), (OPEN, "", 9, 4), (MESSAGE, "e", 10, 4),
        (NEWS_LINE, later, 10, 0), (CLOSE, soak.JOIN_REASON, 10, 4), (OPEN, "", 10, 11), (MESSAGE, "f", 10, 11)]
    assert state.markers == 3 and state.clamped == 2 and state.lines == [6, 5, 2, 2]
    assert (state.score_seq, state.item_seq, state.heartbeat_seq) == (7, 5, 9)
    times = [t for _, _, t, _ in out]
    assert times == sorted(times)


def test_a_recording_without_news_lines_gets_a_healthy_heartbeat_every_30_s_with_the_launch_hashes(tmp_path):
    bare = [line for line in real_tape() if line[0] not in (NEWS_LINE, HEARTBEAT_LINE)]
    recorder = write_recording(tmp_path / "recorder", bare[:len(bare) // 2])
    rehearsal = write_recording(tmp_path / "rehearsal", real_tape(REAL_START + 3 * MS_PER_HOUR,
                                                                  REAL_START + 4 * MS_PER_HOUR))
    chosen = [soak.Recording(p, p, "read in place") for p in [*recorder, *rehearsal]]
    found = soak.recording_news(chosen)
    assert found[:len(recorder)] == [None] * len(recorder) and set(found[len(recorder):]) == {HASHES}
    assert soak.launch_hashes(chosen, found) == soak.launch_hashes(chosen) == (HASHES, "recorded")
    state = soak.JoinState(heartbeat_seq=4)
    lines = list(soak.join([("real", lambda st: soak.with_heartbeats(read_recording(recorder), HASHES, st))], state))
    beats = [(HeartbeatLine.decode(raw, t), t) for kind, raw, t, _ in lines if kind == HEARTBEAT_LINE]
    recorded = list(read_recording(recorder))
    first, last = recorded[0][2], recorded[-1][2]
    assert [t for _, t in beats] == list(range(first - first % 30_000 + 30_000, last + 1, 30_000))
    assert [b.seq for b, _ in beats] == list(range(5, 5 + len(beats))) and state.heartbeats_added == len(beats)
    assert all(b.healthy and (b.config_hash, b.prompt_sha256, b.schema_sha256) == HASHES for b, _ in beats)
    assert [line for line in lines if line[0] != HEARTBEAT_LINE] == recorded
    assert [t for _, _, t, _ in lines] == sorted(t for _, _, t, _ in lines)


# The synthetic run


def news_of(lines):
    scores = [(ScoreLine.decode(raw, t), t) for kind, raw, t, _ in lines if kind == NEWS_LINE]
    beats = [HeartbeatLine.decode(raw, t) for kind, raw, t, _ in lines if kind == HEARTBEAT_LINE]
    return scores, beats


def test_the_synthetic_run_brings_reconnects_halts_bursts_and_outages():
    plan = soak.Synthetic(days=1.0, prints_per_second=0.2, reconnect_minutes=20)
    start = parse_iso(soak.SYNTHETIC_START)
    seqs = soak.Seqs(100, 50, 7)
    lines = list(soak.synthetic_lines(12345, SYMBOLS, start, plan, HASHES, seqs, {"AAPL": 150.0, "MSFT": None}))
    again = list(soak.synthetic_lines(12345, SYMBOLS, start, plan, HASHES, soak.Seqs(100, 50, 7), {"AAPL": 150.0}))
    assert lines == again
    assert [t for _, _, t, _ in lines] == sorted(t for _, _, t, _ in lines)
    assert lines[0][2] >= start and lines[-1][2] <= start + MS_PER_DAY
    assert {kind for kind, _, _, _ in lines} == {OPEN, MESSAGE, CLOSE, NEWS_LINE, HEARTBEAT_LINE}
    session = sessions.session_on(sessions.ny_date(start))
    # Connection faults of every kind, and each symbol's price continuing from the one given.
    faults = soak.feed_faults(start, start + MS_PER_DAY, SYMBOLS, plan, soak.np.random.default_rng(0))
    kinds = {f.kind for f in faults}
    assert kinds == {DISCONNECT, SILENCE, PAUSE, REPLAYED_PRINTS, HALT, LATE_PRINTS, EARLY_PRINTS}
    assert any(f.refused == 2 for f in faults) and len([f for f in faults if f.kind != HALT]) >= 70
    reasons = {raw for kind, raw, _, _ in lines if kind == CLOSE}
    assert {DROPPED, SILENT} <= reasons and any(r.startswith("process paused for") for r in reasons)
    conns = [c for kind, _, _, c in lines if kind == OPEN]
    assert len(conns) > 40 and any(b - a > 1 for a, b in zip(conns, conns[1:]))  # refused handshakes skip ids
    prints = [p for kind, raw, _, _ in lines if kind == MESSAGE and raw != PING
              for p in json.loads(raw)["data"]]
    assert abs(next(p["p"] for p in prints if p["s"] == "AAPL") / 150.0 - 1) < 0.01
    halt = (session.open_ms + soak.LONG_HALT_AT * MS_PER_MINUTE, session.open_ms + (soak.LONG_HALT_AT + 20) *
            MS_PER_MINUTE)
    in_halt = [p for kind, raw, t, _ in lines if kind == MESSAGE and halt[0] + 1000 < t < halt[1] and raw != PING
               for p in json.loads(raw)["data"]]
    assert in_halt and not any(p["s"] == SYMBOLS[0] for p in in_halt)
    # News continuing the given sequences in receipt order, a burst, an outage and unhealthy heartbeats.
    scores, beats = news_of(lines)
    assert [line.seq for line, _ in scores] == list(range(101, 101 + len(scores))) and seqs.score == scores[-1][0].seq
    assert min(line.item_seq for line, _ in scores) == 51 and seqs.item == max(line.item_seq for line, _ in scores)
    assert all(t >= line.scored_ms for line, t in scores)
    assert {line.scorer for line, _ in scores} == {FINBERT_SCORER, HAIKU_SCORER, SONNET_SCORER}
    assert {line.status for line, _ in scores} == {"scored", "failed"}
    assert {line.route for line, _ in scores} == {"llm", "finbert"}
    assert all(line.kind != "media" for line, _ in scores if line.scorer == SONNET_SCORER)
    burst = session.open_ms + soak.BURST_AT * MS_PER_MINUTE
    assert sum(burst <= t < burst + 2 * MS_PER_MINUTE for _, t in scores) >= plan.burst_items + 100
    assert [b.seq for b in beats] == list(range(8, 8 + len(beats))) and {b.hashes for b in beats} == {HASHES}
    outage = session.open_ms + soak.OUTAGE_AT * MS_PER_MINUTE
    stamps = [b.recv_ms for b in beats]
    assert max(b - a for a, b in zip(stamps, stamps[1:])) >= plan.outage_minutes * MS_PER_MINUTE
    assert not any(outage <= t < outage + plan.outage_minutes * MS_PER_MINUTE for t in stamps)
    assert [b.healthy for b in beats].count(False) == soak.UNHEALTHY_MINUTES * 2


# Memory


def reading(hours, rss, source="proc", malloc=None, pymalloc=None, footprint=None):
    return {"hours": hours, "rss_mb": rss, "source": source, "malloc_mb": malloc, "pymalloc_mb": pymalloc,
            "footprint_mb": footprint}


def test_on_linux_resident_memory_may_grow_by_exactly_the_limit_and_decides_alone():
    held = soak.memory_verdict([reading(1, 100.0)], reading(30, 105.0))
    assert held["outcome"] == "passed" and held["judged_on"] == "resident" and held["growth_mb"] == 5.0
    assert held["decided_by"] == "first hour"
    grew = soak.memory_verdict([reading(1, 100.0)], reading(30, 105.01))
    assert grew["outcome"] == "failed" and "resident memory grew 5.01 MB" in grew["reason"]
    retained = soak.memory_verdict([reading(1, 100.0, malloc=30.0, pymalloc=20.0)],
                                   reading(30, 112.0, malloc=30.5, pymalloc=20.0))
    assert retained["outcome"] == "failed" and retained["in_use"]["rule_held"]
    missing = soak.memory_verdict([], reading(30, 90.0))
    assert missing["outcome"] == "failed" and "no memory reading" in missing["reason"]


def test_on_a_mac_memory_in_use_decides_and_resident_memory_is_only_reported():
    mac = "proc_pid_rusage"
    held = soak.memory_verdict([reading(1, 100.0, mac, 30.0, 20.0)], reading(30, 112.0, mac, 31.0, 20.5))
    assert held["outcome"] == "passed" and held["judged_on"] == "in_use" and not held["rule_held"]
    assert held["decided_by"] == "first hour"
    assert held["in_use"]["growth_mb"] == 1.5
    assert "resident memory read with proc_pid_rusage grew 12.0" in held["reason"]
    edge = soak.memory_verdict([reading(1, 100.0, mac, 30.0, 20.0)], reading(30, 100.0, mac, 35.0, 20.0))
    assert edge["outcome"] == "passed" and edge["in_use"]["growth_mb"] == 5.0
    leaked = soak.memory_verdict([reading(1, 100.0, mac, 30.0, 20.0)], reading(30, 100.0, mac, 35.0, 20.01))
    assert leaked["outcome"] == "failed" and "memory in use grew 5.01 MB" in leaked["reason"]
    assert leaked["decided_by"] == "first hour" and "holds no final week" in leaked["reason"]  # 29 hours hold no week
    unknown = soak.memory_verdict([reading(1, 100.0, mac)], reading(30, 112.0, mac))
    assert unknown["outcome"] == "inconclusive" and unknown["in_use"] is None and unknown["judged_on"] is None
    assert unknown["decided_by"] is None


def test_the_verdict_reports_the_worst_reading_and_the_recent_slope():
    samples = [reading(h, 100.0 + (12.0 if 3 <= h <= 20 else 0.0) + (0.5 * (h - 30) if h > 30 else 0.0))
               for h in range(1, 60)]
    result = soak.memory_verdict(samples, reading(60, 115.0))
    assert result["largest_growth_mb"] == 15.0 and result["readings_over_limit"] == 18 + 19 + 1
    assert result["later_readings"] == 59 and result["outcome"] == "failed"
    assert result["slope_mb_per_hour_last_day"] == pytest.approx(0.5, abs=0.01)
    assert result["week_over_week_mb"] is None  # 59 replayed hours hold no week


def test_the_week_over_week_growth_tells_windows_and_weekend_news_from_a_leak():
    # Windows filling over the first two sessions and news kept over each weekend, the shape a two week soak of the
    # stock process shows, fail the first hour rule while a week of replayed time moves memory by nothing, so on a Mac
    # the final week decides, passing below 1 MB and failing at or above it.
    def in_use(h, leak=0.0):
        return 30.0 + 6.0 * min(h, 48) / 48 + (2.5 if 72 <= h % 168 < 120 else 0.0) + leak * h

    mac = "proc_pid_rusage"
    for leak, week, outcome in ((0.0, 0.0, "passed"), (0.005, 0.84, "passed"), (0.01, 1.68, "failed")):
        samples = [reading(h, 90.0, mac, in_use(h, leak), 5.0) for h in range(1, 360)]
        result = soak.memory_verdict(samples, reading(360, 90.0, mac, in_use(360, leak), 5.0))
        assert result["outcome"] == outcome and result["in_use"]["growth_mb"] > soak.MAX_GROWTH_MB
        assert result["decided_by"] == "final week" and not result["in_use"]["rule_held"]
        assert result["in_use"]["week_over_week_mb"] == pytest.approx(week)
        assert f"{result['in_use']['week_over_week_mb']} MB over the final replayed week against 1.0 MB" in \
            result["reason"]
        assert result["week_over_week_mb"] == 0.0  # resident memory held flat
    # Exactly 1 MB over the final week is not under the limit, and a tighter limit can be given.
    samples = [reading(h, 90.0, mac, 30.0 + (8.0 if h > 1 else 0.0) + (1.0 if h >= 300 else 0.0), 5.0)
               for h in range(1, 360)]
    edge = soak.memory_verdict(samples, reading(360, 90.0, mac, 39.0, 5.0))
    assert edge["in_use"]["week_over_week_mb"] == 1.0 and edge["outcome"] == "failed"
    assert soak.memory_verdict(samples, reading(360, 90.0, mac, 38.99, 5.0))["outcome"] == "passed"
    assert soak.memory_verdict(samples, reading(360, 90.0, mac, 38.99, 5.0), week_limit_mb=0.5)["outcome"] == "failed"
    # On Linux resident memory keeps the strict first hour rule, whatever the final week shows.
    linux = [reading(h, in_use(h), "proc", in_use(h), 5.0) for h in range(1, 360)]
    strict = soak.memory_verdict(linux, reading(360, in_use(360), "proc", in_use(360), 5.0))
    assert strict["judged_on"] == "resident" and strict["decided_by"] == "first hour"
    assert strict["outcome"] == "failed" and strict["week_over_week_mb"] == 0.0
    # The figure needs a reading a whole week before the end.
    samples = [reading(h, 90.0, mac, in_use(h), 5.0) for h in range(1, 169)]
    assert soak.memory_verdict(samples, reading(168.5, 90.0, mac, 36.0, 5.0))["in_use"]["week_over_week_mb"] is None
    assert soak.memory_verdict(samples, reading(169, 90.0, mac, 36.0, 5.0))["in_use"]["week_over_week_mb"] == \
        pytest.approx(36.0 - in_use(1), abs=0.001)


# Short soaks through the real stock process code


def fast_pair(folder):
    """The rehearsal pair with four bots per run and every path in folder."""
    paths = []
    for name in ("rehearsal_s1", "rehearsal_s2"):
        path = folder / f"{name}.toml"
        path.write_text(config_text(name, **FAST), encoding="utf-8")
        paths.append(path)
    return paths


def run_main(argv):
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = soak.main(argv)
    return code, printed.getvalue()


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    """About 31 replayed hours, a recording of Thursday morning cut by a kill with a newest file cut as one still
    being written, then a synthetic run past Friday's close, so the bots trade in their second session."""
    folder = tmp_path_factory.mktemp("clean")
    s1, s2 = fast_pair(folder)
    files = write_recording(folder / "data" / "raw_stocks", real_tape())
    before = files[-1].read_bytes()
    out = folder / "out"
    code, printed = run_main(["--s1", str(s1), "--s2", str(s2), "--out", str(out), "--synthetic-days", "1.25",
                              "--reconnect-minutes", "90", "--burst-items", "100"])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return SimpleNamespace(code=code, printed=printed, report=report, out=out, files=files, before=before,
                           configs=load_run_pair(s1, s2))


def test_a_short_soak_reports_memory_cpu_invariants_and_determinism(clean):
    report, out = clean.report, clean.out
    # The memory rule is a measurement of the stock process itself, so the outcome follows it once every invariant
    # and the determinism check hold, and a leak fails it below.
    memory = report["memory"]
    assert report["worker_exit_codes"] == [0, 0] and report["checks"]["passed"] and report["determinism"]["passed"]
    assert report["outcome"] == memory["outcome"] and clean.code == soak.EXIT_CODES[memory["outcome"]]
    assert report["verdict"] == (memory["outcome"] if memory["outcome"] == "passed"
                                 else f"{memory['outcome']}, {memory['reason']}")
    assert report["label"] == soak.PLUMBING_LABEL and report["news_hashes"] == "recorded"
    assert [r["source"] for r in report["recordings"]] == [str(f) for f in clean.files]
    assert clean.files[-1].read_bytes() == clean.before == (out / "input" / clean.files[-1].name).read_bytes()
    replay = report["replay"]
    assert replay["first_ts"] == iso(REAL_START) and 31 < replay["hours"] < 32
    assert replay["join_markers"] >= 2 and replay["prints"] > 300_000 and replay["news_lines"] > 1000
    assert replay["session_minutes"] == 2 * 390

    assert memory["source"] == ("proc" if sys.platform.startswith("linux") else "proc_pid_rusage")
    assert memory["judged_on"] == ("resident" if sys.platform.startswith("linux") else "in_use")
    if sys.platform == "darwin":
        assert 0 < memory["in_use"]["first_hour_mb"] < memory["first_hour_mb"]
        assert memory["in_use"]["rule_held"] is (memory["outcome"] == "passed")
    hours = int(replay["hours"])
    assert memory["later_readings"] == len(report["hourly"]) == hours and memory["live_blocks"]["end"] > 0
    rows = (out / "memory.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("hour,hours,ts,segment,rss_mb") and len(rows) == hours + 1

    cpu = report["cpu"]
    assert cpu["us_per_print"] > cpu["handling_us_per_print"] > 0
    assert cpu["minute_steps_cpu_s"] > 0 and cpu["persistence_cpu_s"] > 0 and cpu["heartbeats_cpu_s"] > 0
    assert set(cpu["segments"]) == {"real", "synthetic"} and all(s["prints"] > 0 for s in cpu["segments"].values())
    assert 0 < cpu["share_of_one_core_at_live_speed"] < 0.05

    checks = report["checks"]
    assert checks["passed"] and checks["failing"] == 0 and checks["count"] == 2 * (len(soak.LEDGER_CHECKS) + 2) + 4
    by_name = {(c["run"], c["check"]): c for c in checks["items"]}
    for run in ("rehearsal_s1", "rehearsal_s2"):
        for name in ("equity snapshots, cash equals start plus realized minus fees, borrow and rent",
                     "closed trades, net equals gross minus both fees and borrow", "fills, never before their decision",
                     "fills, each on a print at or after its due time", soak.WINDOW_CHECK, soak.CARRY_CHECK):
            assert by_name[(run, name)]["rows"] > 0, (run, name)
    for name in ("forecasts, no news input received after the minute",
                 "news lines, never received before they were scored"):
        assert by_name[("rehearsal_s2", name)]["rows"] > 0
    assert by_name[("both", "recording, every line once in stamp order")]["rows"] == replay["lines"]
    assert by_name[("both", "books of every bot from memory every replayed hour")]["rows"] > 0

    same = report["determinism"]
    assert same["passed"] and same["tables"] == 2 * (len(soak.TABLES) - 1) and same["rows"] > 10_000
    activity = report["activity"]
    for run in ("rehearsal_s1", "rehearsal_s2"):
        events = activity[run]["feed_events"]
        assert activity[run]["trades"] > 0 and activity[run]["decisions"] > 0
        assert {"connected", "disconnected", "stale", "healthy", "local_only", "lock", "code_commit", "resume",
                "news_unhealthy", "news_healthy", "session_close"} <= set(events)
        assert events["connected"] >= 20
    assert activity["rehearsal_s2"]["news_seen"] > 300 and activity["rehearsal_s1"]["news_seen"] == 0
    assert activity["rehearsal_s2"]["news_minutes"]["0"] >= 15


def test_the_default_output_holds_only_verdicts_counts_hashes_and_timings(clean):
    lines = clean.printed.splitlines()
    assert lines[:2] == ["stock soak", f"verdict: {clean.report['outcome']}"]
    summary = [line for line in lines if ": " in line and not line.startswith("report written")]
    assert summary and all(SUMMARY_LINE.fullmatch(line) for line in summary), summary
    assert any(line.startswith("memory growth kib: ") for line in lines)
    assert f"memory decided by: {clean.report['memory']['decided_by']}" in lines
    assert any(line.startswith("memory slope kib per day: ") for line in lines)
    assert any(line.startswith("database digest: ") for line in lines)
    assert sum("rows hidden, pass --detail" in line for line in lines) == 2
    assert lines[-2:] == [soak.SOAK_SENTENCE, soak.NEVER_FOR_SELECTION]
    assert not any(str(clean.out) in line for line in lines)
    assert not any(line.startswith("memory week over week kib: ") for line in lines)  # 31 replayed hours
    weekly = json.loads(json.dumps(clean.report))
    judged = weekly["memory"]["in_use"] if weekly["memory"]["judged_on"] == "in_use" else weekly["memory"]
    judged["week_over_week_mb"] = -0.5
    assert "memory week over week kib: -512" in output.render(soak.summary(weekly)[0]).splitlines()
    # --detail adds the hourly readings and every check as rows.
    main, checks = soak.summary(clean.report)
    detail = output.render(main, detail=True).splitlines()
    assert detail[len(main.summary) + 1].split() == ["hour", "ts", "segment", "rss_mb", "in_use_mb", "lines", "prints",
                                                     "cpu_s"]
    assert len(detail) == len(main.summary) + 2 + len(clean.report["hourly"])
    rows = output.render(checks, detail=True).splitlines()
    assert rows[3].split()[:3] == ["run", "rows", "failures"] and len(rows) == 4 + clean.report["checks"]["count"]


PLANTED = {
    "equity snapshots, cash equals start plus realized minus fees, borrow and rent":
        "UPDATE equity_snapshots SET realized_gross = realized_gross + 1 WHERE rowid = (SELECT min(rowid) FROM "
        "equity_snapshots)",
    "equity snapshots, equity equals cash plus unrealized":
        "UPDATE equity_snapshots SET unrealized = unrealized + 1 WHERE rowid = (SELECT max(rowid) FROM "
        "equity_snapshots)",
    "deaths, the five parts and the writeoff explain the final equity":
        "INSERT INTO deaths VALUES (9001, '2026-10-30T15:00:00.000Z', 0, 'rent', 0, 0, 0, 0, 0, 0, 0, 10001.0, 0, 1.0)",
    "deaths, final equity is never negative":
        "INSERT INTO deaths VALUES (9002, '2026-10-30T15:00:00.000Z', 0, 'fees', 10001.0, 0, 0, 0, 0, 0, 0, -1.0, 0, "
        "1.0)",
    "closed trades, net equals gross minus both fees and borrow":
        "UPDATE trades SET pnl_gross = pnl_gross + 1 WHERE id = (SELECT min(id) FROM trades WHERE closed_at IS NOT "
        "NULL)",
    "dead bots, each has exactly one death row":
        "UPDATE bots SET state = 'dead' WHERE bot_id = (SELECT min(bot_id) FROM bots)",
    "fills, never before their decision":
        "UPDATE fills SET filled_at = '2026-01-01T00:00:00.000Z' WHERE id = (SELECT min(id) FROM fills WHERE status = "
        "'filled')",
    "fills, each on a print at or after its due time":
        "UPDATE fills SET print_after_due_ms = -1 WHERE id = (SELECT max(id) FROM fills WHERE status = 'filled')",
    "forecasts, no news input received after the minute":
        "UPDATE stock_forecasts SET inputs_max_recv_at = '2027-01-01T00:00:00.000Z' WHERE rowid = (SELECT min(rowid) "
        "FROM stock_forecasts WHERE inputs_max_recv_at IS NOT NULL)",
    "news lines, never received before they were scored":
        "UPDATE news_seen SET recv_at = '2026-01-01T00:00:00.000Z' WHERE score_seq = (SELECT min(score_seq) FROM "
        "news_seen)",
    "engine checks at equity snapshots, no invariant violation":
        "INSERT INTO feed_events (ts, kind, detail, symbol) VALUES ('2026-10-30T15:00:00.000Z', 'invariant_violation', "
        "'planted', 'AAPL')",
    soak.WINDOW_CHECK:
        "UPDATE decisions SET ts = '2026-10-30T13:35:00.000Z' WHERE id = (SELECT min(id) FROM decisions WHERE "
        "action IN ('open_long', 'open_short', 'flip'))",
    soak.CARRY_CHECK:
        "UPDATE trades SET closed_at = '2026-10-30T20:30:00.000Z', exit_reason = 'signal' WHERE id = (SELECT max(id) "
        "FROM trades WHERE closed_at IS NOT NULL AND exit_reason NOT IN ('session_flatten', 'death') AND opened_at < "
        "'2026-10-30T20:00:00.000Z')",
}


def copy_db(source, target):
    """A writable copy of a finished run database, taken through the backup API."""
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    src.backup(dst)
    src.close()
    return dst


def test_each_database_check_catches_one_planted_row(clean, tmp_path):
    assert set(PLANTED) == set(soak.LEDGER_CHECKS) | {soak.WINDOW_CHECK, soak.CARRY_CHECK}
    s2 = clean.configs[1]
    checks = soak.database_report(clean.out / "run" / s2.run.name / "stocks.db", s2)["checks"]
    assert set(checks) == set(PLANTED) and not any(c["failures"] for c in checks.values())
    target = copy_db(clean.out / "run" / s2.run.name / "stocks.db", tmp_path / "copy.db")
    for sql in PLANTED.values():
        assert target.execute(sql).rowcount == 1, sql
    target.commit()
    target.close()
    checks = soak.database_report(tmp_path / "copy.db", s2)["checks"]
    assert {name: c["failures"] for name, c in checks.items()} == {name: 1 for name in PLANTED}


def test_s1_and_s2_must_share_every_price_row_and_a_second_pass_must_match_the_first(clean, tmp_path):
    names = [cfg.run.name for cfg in clean.configs]
    s1, s2 = (clean.out / "run" / name / "stocks.db" for name in names)
    assert soak.shared_rows_check(s1, s2)["failures"] == 0
    changed = copy_db(s2, tmp_path / "s2.db")
    changed.execute("UPDATE stock_candles SET volume = volume + 1 WHERE rowid = 1")
    changed.execute("UPDATE stock_forecasts SET prediction = prediction + 1 WHERE rowid = (SELECT min(rowid) FROM "
                    "stock_forecasts WHERE predictor = 'linear_prices')")
    changed.commit()
    changed.close()
    assert soak.shared_rows_check(s1, tmp_path / "s2.db")["failures"] == 2
    # A copy of both passes, then one decision changed in the second.
    for name in soak.PASSES:
        shutil.copytree(clean.out / name, tmp_path / "passes" / name)
    first, second = (tmp_path / "passes" / name for name in soak.PASSES)
    assert soak.determinism(first, second, names)["passed"]
    conn = sqlite3.connect(second / names[1] / "stocks.db")
    conn.execute("UPDATE decisions SET reason = 'planted' WHERE id = 1")
    conn.commit()
    conn.close()
    same = soak.determinism(first, second, names)
    assert not same["passed"] and same["differing"] == [f"{names[1]}.decisions"]
    assert same["digest"] == clean.report["determinism"]["digest"]


def test_a_leak_fails_the_soak(tmp_path):
    s1, s2 = fast_pair(tmp_path)
    out = tmp_path / "out"
    code, printed = run_main(["--s1", str(s1), "--s2", str(s2), "--out", str(out), "--skip-real",
                              "--synthetic-start", iso(at(THU, "06:00")), "--synthetic-days", "0.2",
                              "--prints-per-second", "0.2", "--leak-kb-per-message", "16"])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "failed" and code == 1 and not report["passed"]
    assert report["checks"]["passed"] and report["determinism"]["passed"]
    assert "grew" in report["verdict"] and "verdict: failed" in printed.splitlines()
    if sys.platform == "darwin":
        assert report["memory"]["in_use"]["growth_mb"] > soak.MAX_GROWTH_MB


def test_the_soak_refuses_a_busy_folder_a_folder_inside_the_recordings_and_a_bad_pair(tmp_path, capsys):
    s1, s2 = fast_pair(tmp_path)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file").write_text("x", encoding="utf-8")
    assert soak.main(["--s1", str(s1), "--s2", str(s2), "--out", str(busy), "--skip-real"]) == soak.CANNOT_RUN
    assert "not empty" in capsys.readouterr().err
    inside = tmp_path / "data" / "raw_stocks" / "soak"
    assert soak.main(["--s1", str(s1), "--s2", str(s2), "--out", str(inside)]) == soak.CANNOT_RUN
    assert "inside the recordings folder" in capsys.readouterr().err and not inside.exists()
    assert soak.main(["--s1", str(s2), "--s2", str(s1), "--out", str(tmp_path / "swapped")]) == soak.CANNOT_RUN
    err = capsys.readouterr().err
    assert err.startswith("error: invalid configuration") and not (tmp_path / "swapped").exists()


def test_the_home_pc_judges_resident_memory_on_the_final_week_when_asked():
    # The owner's decision of 15 October 2026: on the home PC, resident memory that fills its windows and then holds
    # still passes on the final week, as memory in use does on a Mac, while the default stays strict for a Pi.
    def rss(h, leak=0.0):
        return 60.0 + 17.0 * min(h, 48) / 48 + leak * h

    for leak, outcome in ((0.0, "passed"), (0.005, "passed"), (0.01, "failed")):
        samples = [reading(h, rss(h, leak)) for h in range(1, 360)]
        strict = soak.memory_verdict(samples, reading(360, rss(360, leak)))
        assert strict["outcome"] == "failed" and strict["decided_by"] == "first hour"
        result = soak.memory_verdict(samples, reading(360, rss(360, leak)), final_week_on_linux=True)
        assert result["outcome"] == outcome and result["judged_on"] == "resident"
        assert result["decided_by"] == "final week"
        assert "resident memory grew" in result["reason"] and "over the final replayed week against 1.0 MB" in \
            result["reason"]
    held = soak.memory_verdict([reading(1, 100.0)], reading(30, 104.0), final_week_on_linux=True)
    assert held["outcome"] == "passed" and held["decided_by"] == "first hour"
    short = soak.memory_verdict([reading(1, 100.0)], reading(30, 112.0), final_week_on_linux=True)
    assert short["outcome"] == "failed" and "holds no final week" in short["reason"]
