"""The soak script: recordings are joined safely, a file still being written is read without being touched, a short
soak reports memory over time, CPU per message, balance checks, the gap flatten, mutated births and the highest
generation, and a leak or a broken ledger makes it fail."""

import contextlib
import dataclasses
import io
import json
import sqlite3
import sys
import zlib

import pytest

from btc_survivor import seeds
from btc_survivor.clock import MS_PER_MINUTE
from btc_survivor.config import load_config
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, decode, load_adapter
from btc_survivor.feed.book import LocalBook
from btc_survivor.recorder import Recorder, _file_name, read_recording
from btc_survivor.synthetic import SyntheticFeed
from enginekit import START
from helpers import EXAMPLE_CONFIG, ROOT, dump_toml, example_dict

sys.path.insert(0, str(ROOT / "scripts"))
import soak_test as soak  # noqa: E402

PLUMBING_KEYS = {
    "run.data_dir": "../data/soak",
    "trading.start_equity": 4.0,
    "trading.notional": 4.0,
    "trading.force_trade_after_hours": 1.0,
    "population.parent_min_age_hours": 2.0,
    "population.parent_min_trades": 2,
}
LINUX = sys.platform.startswith("linux")


def differences(a, b):
    out = {}
    for section in ("run", "feed", "time", "features", "model", "trading", "costs", "genes", "population",
                    "controls", "persistence"):
        x, y = dataclasses.asdict(getattr(a, section)), dataclasses.asdict(getattr(b, section))
        out.update({f"{section}.{k}": y[k] for k in x if x[k] != y[k]})
    return out


def recorded(folder, lines):
    recorder = Recorder(folder)
    for line in lines:
        recorder.write(*line)
    recorder.close()
    return recorder.path


def unfinished(folder, lines, flush_every=200, cut=120):
    """A gzip recording as the recorder leaves it mid write: synced every few lines, no trailer, a cut tail."""
    comp = zlib.compressobj(wbits=31)
    data = bytearray()
    for i, (kind, raw, recv_ms, conn) in enumerate(lines, 1):
        text = json.dumps({"t": recv_ms, "k": kind, "c": conn, "d": raw}, separators=(",", ":")) + "\n"
        data += comp.compress(text.encode())
        if i % flush_every == 0:
            data += comp.flush(zlib.Z_SYNC_FLUSH)
    data += comp.flush(zlib.Z_SYNC_FLUSH)
    path = folder / _file_name(lines[0][2])
    path.write_bytes(bytes(data[:-cut]))
    return path


def fast_soak_config(folder):
    """Features within 16 minutes, a 2 minute horizon and a 9 percent fee, so deaths and births come within the hour."""
    data = example_dict()
    data["run"]["data_dir"] = "data"
    data["time"].update(horizon_minutes=2, history_minutes_required=16, warmup_hours_random=0.05,
                        warmup_hours_inherited=0.02)
    data["features"].update(return_windows_minutes=[1, 2, 3, 4], vol_windows_minutes=[2, 3],
                            volume_baseline_minutes=15, min_samples=5, imbalance_levels=5)
    data["model"].update(skill_window_samples=30, replay_window_minutes=60)
    data["trading"].update(start_equity=100.0, notional=100.0, force_trade_after_hours=0.05)
    data["costs"].update(fee_profile="test", taker_fee_bps_test=900.0)
    data["population"].update(size=4, parent_min_age_hours=0.05, parent_min_trades=1)
    data["persistence"].update(checkpoint_every_seconds=120, equity_snapshot_every_seconds=60, keep_candles_days=1,
                               max_resume_gap_minutes=5)
    path = folder / "fast.toml"
    path.write_text(dump_toml(data), encoding="utf-8")
    return path


def short_soak(folder, *extra):
    """About 1.8 replayed hours: a finished recording, one still being written, then 75 synthetic minutes with full
    size snapshots and a 10 minute outage after 36 of them, while bots hold positions."""
    raw = folder / "raw"
    done = recorded(raw, list(SyntheticFeed(5, START).lines(20 * 60)))
    live = unfinished(raw, list(SyntheticFeed(6, START + 21 * MS_PER_MINUTE).lines(10 * 60)), flush_every=500)
    before = live.read_bytes()
    out = folder / "soak"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = soak.main(["--config", str(fast_soak_config(folder)), "--raw", str(raw), "--out", str(out),
                          "--synthetic-hours", "1.25", "--resync-minutes", "10", "--snapshot-bids", "6000",
                          "--snapshot-asks", "5000", "--outage-at-hours", "0.6", *extra])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return {"code": code, "report": report, "out": out, "done": done, "live": live, "before": before,
            "printed": printed.getvalue()}


@pytest.fixture(scope="module")
def clean_soak(tmp_path_factory):
    return short_soak(tmp_path_factory.mktemp("clean"))


def test_the_plumbing_config_changes_only_plumbing_keys_and_says_it_is_never_a_strategy_setting():
    path = ROOT / "configs" / "soak_plumbing.toml"
    cfg = load_config(path)
    assert differences(load_config(EXAMPLE_CONFIG), cfg) == PLUMBING_KEYS
    assert "never a strategy setting" in path.read_text(encoding="utf-8").splitlines()[0]
    assert cfg.data_dir.resolve() == (ROOT / "data" / "soak").resolve()
    assert "data/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert soak.DEFAULT_CONFIG == path and soak.SOAK_ROOT == ROOT / "data" / "soak"


def test_the_synthetic_soak_seed_tag_never_collides_with_a_tag_in_seeds():
    tags = {v for k, v in vars(seeds).items() if k.isupper() and isinstance(v, int)}
    assert tags and soak.SYNTHETIC_TAG not in tags


def test_a_file_still_being_written_is_copied_once_and_read_to_its_last_complete_line(tmp_path):
    raw = tmp_path / "raw"
    early = list(SyntheticFeed(1, START).lines(120))
    late = list(SyntheticFeed(2, START + 10 * MS_PER_MINUTE).lines(300))
    done = recorded(raw, early)
    live = unfinished(raw, late)
    before = live.read_bytes()
    chosen = soak.select_recordings(raw, tmp_path / "input")
    assert [r.source for r in chosen] == [done, live]
    assert chosen[0].path == done and chosen[0].status == "complete"
    assert chosen[1].path == tmp_path / "input" / live.name and "still being written" in chosen[1].status
    assert chosen[1].path.read_bytes() == before == live.read_bytes()
    with live.open("ab") as fh:  # the recorder keeps writing, the frozen copy does not move
        fh.write(b"more")
    assert chosen[1].path.read_bytes() == before
    got = list(read_recording([chosen[1].path]))
    assert 200 <= len(got) < len(late) and got == late[:len(got)]
    assert soak.gzip_is_complete(done, done.stat().st_size)
    assert not soak.gzip_is_complete(live, len(before))


def test_joins_close_the_open_connection_unless_the_next_file_continues_it():
    a = [(OPEN, "", 1, 1), (MESSAGE, "a", 2, 1), (CLOSE, "x", 3, 1), (OPEN, "", 4, 2), (MESSAGE, "b", 5, 2)]
    rotated = [(MESSAGE, "c", 6, 2), (CLOSE, "x", 7, 2), (OPEN, "", 8, 3), (MESSAGE, "d", 9, 3)]
    restarted = [(OPEN, "", 9, 1), (MESSAGE, "e", 10, 1)]
    midway = [(MESSAGE, "f", 8, 7)]
    state = soak.JoinState()
    sources = [(f"s{i}", lambda st, lines=lines: lines) for i, lines in enumerate((a, rotated, restarted, midway))]
    out = list(soak.join(sources, state))
    assert out == a + rotated + [
        (CLOSE, "soak join, the next recording starts", 9, 3), (OPEN, "", 9, 4), (MESSAGE, "e", 10, 4),
        (CLOSE, "soak join, the next recording starts", 10, 4), (OPEN, "", 10, 11), (MESSAGE, "f", 10, 11)]
    assert state.markers == 3 and state.clamped == 1 and state.lines == [5, 4, 2, 1]
    times = [t for _, _, t, _ in out]
    assert times == sorted(times)


def test_a_full_size_snapshot_keeps_the_touch_and_the_book_valid():
    cfg = load_config(EXAMPLE_CONFIG)
    lines = list(SyntheticFeed(3, START).lines(2))
    raw = next(r for k, r, _, _ in lines if k == MESSAGE and '"type":"snapshot"' in r[:300])
    padded = soak.full_size_snapshot(raw, 2000, 1500, soak.np.random.default_rng(0))
    small, big = (decode(load_adapter(cfg.feed), MESSAGE, text, START, 1)[0] for text in (raw, padded))
    assert (len(big.bids), len(big.asks)) == (2000, 1500) and len(padded) > 2000 * 100
    book = LocalBook(cfg.feed.book_depth_levels)
    book.apply_snapshot(big.bids, big.asks)
    assert book.valid and book.best_bid == max(p for p, _ in small.bids)
    assert book.best_ask == min(p for p, _ in small.asks)
    assert len({p for p, _ in big.bids}) == 2000 and min(p for p, _ in big.bids) > 0


def test_the_long_outage_is_never_skipped_and_lasts_its_full_length():
    lines = list(soak.synthetic_lines(7, START, 1.0, 10, 80_000.0, 30, 30, "BTC-USD", outage=(0.5, 15)))
    closes = [t for k, _, t, _ in lines if k == CLOSE]
    gaps = [b - a for (_, _, a, _), (_, _, b, _) in zip(lines, lines[1:])]
    assert max(gaps) >= 15 * MS_PER_MINUTE
    outage_at = START + 30 * MS_PER_MINUTE
    assert any(abs(t - outage_at) < 1000 for t in closes)


def reading(hours, rss, source="proc", malloc=None, pymalloc=None, footprint=None):
    return {"hours": hours, "rss_mb": rss, "source": source, "malloc_mb": malloc, "pymalloc_mb": pymalloc,
            "footprint_mb": footprint}


def test_on_linux_memory_may_grow_by_exactly_the_limit_and_no_more():
    held = soak.memory_verdict([reading(1, 100.0)], reading(30, 105.0))
    assert held["outcome"] == "passed" and held["decisive"] and held["growth_mb"] == 5.0
    grew = soak.memory_verdict([reading(1, 100.0)], reading(30, 105.01))
    assert grew["outcome"] == "failed" and "memory grew 5.01 MB" in grew["reason"]
    missing = soak.memory_verdict([], reading(30, 90.0))
    assert missing["outcome"] == "failed" and "no memory reading" in missing["reason"]


def test_on_macos_the_rule_is_inconclusive_unless_memory_in_use_grew_too():
    mac = "proc_pid_rusage"
    held = soak.memory_verdict([reading(1, 100.0, mac, 30.0, 20.0)], reading(30, 101.0, mac, 30.5, 20.0))
    assert held["outcome"] == "inconclusive" and not held["decisive"] and held["rule_held"]
    retained = soak.memory_verdict([reading(1, 100.0, mac, 30.0, 20.0)], reading(30, 112.0, mac, 31.0, 20.5))
    assert retained["outcome"] == "inconclusive" and not retained["rule_held"]
    assert retained["in_use"]["growth_mb"] == 1.5 and retained["in_use"]["rule_held"]
    leaked = soak.memory_verdict([reading(1, 100.0, mac, 30.0, 20.0)], reading(30, 112.0, mac, 40.0, 21.0))
    assert leaked["outcome"] == "failed" and "in use by the allocators grew 11.0 MB" in leaked["reason"]
    unknown = soak.memory_verdict([reading(1, 100.0, mac)], reading(30, 112.0, mac))
    assert unknown["outcome"] == "inconclusive" and unknown["in_use"] is None


def test_the_verdict_reports_the_worst_reading_and_the_recent_slope():
    samples = [reading(h, 100.0 + (12.0 if 3 <= h <= 20 else 0.0) + (0.5 * (h - 30) if h > 30 else 0.0))
               for h in range(1, 60)]
    result = soak.memory_verdict(samples, reading(60, 115.0))
    assert result["largest_growth_mb"] == 15.0 and result["readings_over_limit"] == 18 + 19 + 1
    assert result["later_readings"] == 59 and result["outcome"] == "failed"
    assert result["slope_mb_per_hour_last_day"] == pytest.approx(0.5, abs=0.01)


def test_a_short_soak_reports_memory_cpu_checks_the_gap_flatten_and_evolution(clean_soak):
    report, out, code = clean_soak["report"], clean_soak["out"], clean_soak["code"]
    expected = "passed" if LINUX else "inconclusive"
    assert report["outcome"] == expected and code == soak.EXIT_CODES[expected] and report["worker_exit_code"] == 0
    assert report["passed"] is LINUX
    assert report["label"] == soak.PLUMBING_LABEL and "never used to choose" in report["never_for_selection"]
    assert [r["source"] for r in report["recordings"]] == [str(clean_soak["done"]), str(clean_soak["live"])]
    live, before = clean_soak["live"], clean_soak["before"]
    assert live.read_bytes() == before and (out / "input" / live.name).read_bytes() == before
    assert report["replay"]["hours"] > 1.0 and report["replay"]["join_markers"] == 2

    memory = report["memory"]
    assert memory["rule_held"] and memory["growth_mb"] <= soak.MAX_GROWTH_MB
    assert memory["source"] == ("proc" if LINUX else "proc_pid_rusage" if sys.platform == "darwin" else "ps")
    if sys.platform == "darwin" or LINUX:
        assert memory["in_use"]["rule_held"] and memory["in_use"]["first_hour_mb"] > 0
    assert memory["live_blocks"]["end"] > 0 and memory["arenas"]["end"] > 0
    rows = (out / "memory.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("hour,hours,ts,segment,rss_mb") and len(rows) >= 2

    cpu = report["cpu"]
    assert cpu["engine_us_per_message"] > cpu["message_handling_us_per_message"] > 0
    assert cpu["minute_step_cpu_s"] > 0 and cpu["persistence_cpu_s"] > 0 and cpu["parent_cpu_s"] > 0
    assert cpu["full_size_snapshot_ms_each"] > 0 and cpu["other_message_us_each"] > 0
    assert set(cpu["segments"]) == {"real", "synthetic"}
    assert cpu["segments"]["synthetic"]["full_size_snapshots"] >= 3
    assert cpu["segments"]["real"]["full_size_snapshots"] == 0

    checks = report["checks"]
    assert checks["passed"] and checks["engine_checks_at_equity_snapshots"] > 0 and checks["hourly_bot_checks"] > 0
    assert all(c["failures"] == 0 for c in checks["ledger"].values())
    assert checks["ledger"]["deaths, the five parts and the writeoff explain the final equity"]["rows"] > 0

    evolution = report["evolution"]
    assert evolution["deaths"] > 0 and evolution["mutated_births"] > 0 and evolution["highest_generation"] >= 1
    assert evolution["lineage_rows"] == evolution["mutated_births"]
    assert evolution["exit_reasons"].get("gap_flatten", 0) > 0
    flattens = [e for e in report["replay"]["resume_events"] if "gap_flatten" in e]
    assert flattens and not flattens[0].endswith("sent 0 closing orders")
    printed = clean_soak["printed"]
    assert "hour   1" in printed and "mutated births" in printed and f"soak {report['verdict']}" in printed


def test_a_leak_fails_the_soak(tmp_path):
    run = short_soak(tmp_path, "--leak-kb-per-message", "1")
    report = run["report"]
    assert report["outcome"] == "failed" and report["passed"] is False and run["code"] == 1
    assert "memory grew" in report["verdict"] and report["checks"]["passed"]
    assert report["memory"]["growth_mb"] > soak.MAX_GROWTH_MB
    if sys.platform == "darwin" or LINUX:
        assert report["memory"]["in_use"]["growth_mb"] > soak.MAX_GROWTH_MB


CORRUPTIONS = {
    "equity snapshots, cash equals start plus realized minus fees, funding and rent":
        "UPDATE equity_snapshots SET realized_gross = realized_gross + 1"
        " WHERE rowid = (SELECT min(rowid) FROM equity_snapshots)",
    "equity snapshots, equity equals cash plus unrealized":
        "UPDATE equity_snapshots SET unrealized = unrealized + 1 WHERE rowid = (SELECT max(rowid) FROM equity_snapshots)",
    "deaths, the five parts and the writeoff explain the final equity":
        "UPDATE deaths SET writeoff = writeoff + 1 WHERE bot_id = (SELECT min(bot_id) FROM deaths)",
    "deaths, final equity is never negative":
        "UPDATE deaths SET final_equity = -1, writeoff = writeoff - final_equity - 1"
        " WHERE bot_id = (SELECT max(bot_id) FROM deaths)",
    "closed trades, net equals gross minus both fees and funding":
        "UPDATE trades SET pnl_gross = pnl_gross + 1 WHERE id = (SELECT min(id) FROM trades WHERE closed_at IS NOT NULL)",
    "dead bots, each has exactly one death row":
        "UPDATE bots SET state = 'dead' WHERE bot_id = (SELECT min(bot_id) FROM bots WHERE state != 'dead')",
    "mutated births, each has a lineage row and is one generation below its parent":
        "DELETE FROM lineage WHERE bot_id = (SELECT min(bot_id) FROM lineage)",
}


def test_each_ledger_check_catches_one_corrupted_row(clean_soak, tmp_path):
    assert set(CORRUPTIONS) == set(soak.LEDGER_CHECKS)
    source = sqlite3.connect(f"file:{clean_soak['out'] / 'db' / 'btc_survivor.db'}?mode=ro", uri=True)
    copy = tmp_path / "copy.db"
    target = sqlite3.connect(copy)
    source.backup(target)
    source.close()
    assert target.execute("SELECT count(*) FROM deaths").fetchone()[0] >= 2
    for sql in CORRUPTIONS.values():
        assert target.execute(sql).rowcount == 1, sql
    target.commit()
    target.close()
    start = load_config(fast_soak_config(tmp_path)).trading.start_equity
    ledger = soak.database_report(copy, start)["ledger"]
    assert {name: c["failures"] for name, c in ledger.items()} == {name: 1 for name in CORRUPTIONS}


def test_the_soak_refuses_a_folder_that_is_not_empty(tmp_path):
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        soak.main(["--skip-real", "--out", str(busy)])
