"""Checkpoints of the stock process by PLAN_V2 sections 4.4, 9.2 and 14.2, on both runs launched through open_runs and
driven over a seeded synthetic tape with news lines until learners hold positions: each run's checkpoint round trips
bit for bit, a group restored from the checkpoints equals the in memory copy it was saved from in every attribute
and continues to write exactly the same rows, and resume refuses a changed experiment hash, a changed session index
or copies of the shared layer that differ. Also a run without a checkpoint, runs without a common checkpoint, and an
unreadable newest checkpoint that falls back to the one both runs hold and rolls the later rows back."""

import copy
import dataclasses
import re
import sqlite3
from collections import deque
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from btc_survivor.clock import iso
from btc_survivor.feed.base import OPEN
from btc_survivor.model import MLP
from btc_survivor.v2 import sessions
from btc_survivor.v2.stocks import checkpoint
from btc_survivor.v2.stocks.checkpoint import ResumeError
from btc_survivor.v2.stocks.group import StockGroup
from btc_survivor.v2.stocks.replay import table_digests
from btc_survivor.v2.stocks.store import StockStore
from test_v2_group import FAST, FRI, HASHES, THU, adapter, at, close, launch, pair, query, run_lines, tape

START, CUT, RESTART, END = at(THU, "09:00"), at(FRI, "11:05:30"), at(FRI, "11:08"), at(FRI, "11:40")
# Attributes a checkpoint never carries: the store and checkpointer a process attaches, the config it was given, and
# S2's decayed score list, which adding the kept lines again rebuilds and the next step prunes to the same inputs.
SKIP = {"store", "checkpointer", "cfg", "_scores"}


def backup(source: Path, target: Path) -> None:
    """A consistent copy of a database through the SQLite backup API, put in WAL mode again by the copy's own
    connection, so its side files exist for a read only open as they do beside a database its writer closed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    src, dst = sqlite3.connect(source), sqlite3.connect(target)
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=WAL")
    finally:
        src.close()
        dst.close()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """Both runs at the 11:05:30 persistence point on Friday, as databases and as an in memory copy of the shared
    layer and the engines, with the lines received after it."""
    folder = tmp_path_factory.mktemp("checkpoint")
    lines = tape(START, END)
    cfgs, group = launch(folder, START)
    used = run_lines(group, lines, CUT)
    stores = [engine.store for engine in group.engines]
    for engine in group.engines:
        engine.store = engine.checkpointer = None
    state = copy.deepcopy((group.market, group.engines))
    for engine, store in zip(group.engines, stores):
        engine.store = store
    snapshot = folder / "snapshot"
    for cfg in cfgs:
        backup(cfg.db_path, snapshot / cfg.run.name / "stocks.db")
    close(group)
    return SimpleNamespace(cfgs=cfgs, state=state, snapshot=snapshot, rest=lines[used:], engines=group.engines)


def fresh(saved, folder: Path, **values: str):
    """Configs in folder and copies of the saved databases where they look for them."""
    cfgs = pair(folder, **{**FAST, **values})
    for cfg in cfgs:
        backup(saved.snapshot / cfg.run.name / "stocks.db", cfg.db_path)
    return cfgs


def newest(path: Path) -> tuple:
    return query(path, "SELECT id, ts, meta, arrays FROM checkpoints ORDER BY id DESC LIMIT 1")[0]


def retie(engines) -> None:
    """A deep copy turns each network's weight views into separate arrays, which a training step would leave stale,
    so build every copied network on its parameters again, as a checkpoint load does."""
    for engine in engines:
        for bot in engine.all_bots():
            bot.model = MLP(bot.model.n_in, bot.model.hidden, params=bot.model.params)


def differences(a, b, path: str = "", seen: set | None = None) -> list[str]:
    """Every attribute path where two object graphs differ, comparing arrays bit for bit and generators by state."""
    seen = set() if seen is None else seen
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        same = isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and \
            a.shape == b.shape and a.tobytes() == b.tobytes()
        return [] if same else [path]
    if isinstance(a, np.random.Generator):
        return [] if a.bit_generator.state == b.bit_generator.state else [path]
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return [] if a == b or (a != a and b != b) else [path]
    if type(a) is not type(b):
        return [f"{path} is a {type(a).__name__} and a {type(b).__name__}"]
    if isinstance(a, (str, int, bool, bytes, type(None), Path, date, np.generic, frozenset, set)):
        return [] if a == b else [path]
    if isinstance(a, dict):
        if list(a) != list(b):
            return [f"{path} keys"]
        return [d for key in a for d in differences(a[key], b[key], f"{path}[{key!r}]", seen)]
    if isinstance(a, (list, tuple, deque)):
        if len(a) != len(b):
            return [f"{path} length"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]", seen)]
    if (id(a), id(b)) in seen:
        return []
    seen.add((id(a), id(b)))
    if dataclasses.is_dataclass(a):
        names = [f.name for f in dataclasses.fields(a)]
    elif hasattr(a, "__dict__"):
        names = list(vars(a))
    else:
        names = [name for cls in type(a).__mro__ for name in getattr(cls, "__slots__", ())]
    return [d for name in names if name not in SKIP
            for d in differences(getattr(a, name), getattr(b, name), f"{path}.{name}", seen)]


def test_each_run_checkpoint_round_trips_bit_for_bit(saved, tmp_path):
    cfgs = fresh(saved, tmp_path)
    s1, s2 = saved.engines
    assert any(b.account.is_open for b in s1.all_bots()) and any(b.trained_minutes for b in s1.all_bots())
    assert s2.inputs.news.kept_lines() and s2.window.count
    for cfg in cfgs:
        _, ts, text, blob = newest(cfg.db_path)
        assert ts == iso(CUT)
        meta, arrays = checkpoint.decode(text, blob)
        store = StockStore.open(cfg.db_path)
        market = checkpoint.load_market(cfg, HASHES, meta[checkpoint.MARKET], arrays)
        engine = checkpoint.load_engine(cfg, store, meta[checkpoint.ENGINE], arrays, ts)
        assert checkpoint.encode(market, engine, meta["at"]) == (text, blob)
        assert meta["session_k"] == FRI.k(at(FRI, "11:05")) and meta["experiment_hash"] == cfg.experiment_hash
        store.close()
    s1_meta, s2_meta = (checkpoint.decode(*newest(c.db_path)[2:])[0] for c in cfgs)
    assert s1_meta[checkpoint.MARKET] == s2_meta[checkpoint.MARKET]
    assert s2_meta[checkpoint.ENGINE]["news"]["lines"] and s1_meta[checkpoint.ENGINE]["news"] is None


def test_a_restored_group_equals_its_saved_copy_and_continues_with_exactly_the_same_rows(saved, tmp_path):
    copied = fresh(saved, tmp_path / "copied")
    market, engines = copy.deepcopy(saved.state)
    retie(engines)
    for engine, cfg in zip(engines, copied):
        engine.store = StockStore.open(cfg.db_path)
    from_copy = StockGroup(market, engines, adapter(), RESTART)
    restored_cfgs = fresh(saved, tmp_path / "restored")
    stores = [StockStore.open(cfg.db_path) for cfg in restored_cfgs]
    from_checkpoint = StockGroup(*checkpoint.load(restored_cfgs, stores), adapter(), RESTART)
    # Before the resume only what a resume rebuilds differs, the candles, price health apart from its epochs, the
    # heartbeat clock and the Roll accumulators.
    rebuilt = {"candles", "health", "rolls", "restarted", "_beats", "_beat_ms", "_channels"}
    found = differences(from_copy.market, from_checkpoint.market)
    assert found and {re.match(r"\.(\w+)", d).group(1) for d in found} <= rebuilt
    assert [h.epoch for h in from_copy.market.health.symbols.values()] == \
        [h.epoch for h in from_checkpoint.market.health.symbols.values()]
    for one, other in zip(from_copy.engines, from_checkpoint.engines):
        assert differences(one, other) == []
    for group in (from_copy, from_checkpoint):
        group.resume(RESTART)
    assert differences(from_copy.market, from_checkpoint.market) == []
    assert all(differences(a, b) == [] for a, b in zip(from_copy.engines, from_checkpoint.engines))
    # The restarted process opens a new connection before its first line.
    later = [(OPEN, "", RESTART, 9)] + [line for line in saved.rest if line[2] >= RESTART]
    for group in (from_copy, from_checkpoint):
        run_lines(group, later, END)
        group.finish()
        close(group)
    for one, other in zip(copied, restored_cfgs):
        digests = table_digests(one.db_path)
        assert digests == table_digests(other.db_path)
        assert query(one.db_path, "SELECT count(*) FROM fills WHERE decided_at > ?", (iso(RESTART),))[0][0] > 0
        assert query(one.db_path, "SELECT count(*) FROM stock_candles WHERE valid = 1 AND ts > ?",
                     (iso(RESTART),))[0][0] > 0
        assert query(one.db_path, "SELECT detail FROM feed_events WHERE kind = 'resume' AND symbol IS NULL") == \
            [("resumed, 0 orders in flight rejected",)]


def test_resume_refuses_a_changed_experiment_hash_and_changes_nothing(saved, tmp_path):
    cfgs = fresh(saved, tmp_path, learning_rate="0.0003")
    before = [table_digests(cfg.db_path) for cfg in cfgs]
    stores = [StockStore.open(cfg.db_path) for cfg in cfgs]
    with pytest.raises(ResumeError, match="s1_prices: the config changed since the checkpoint"):
        checkpoint.load(cfgs, stores)
    for store in stores:
        store.close()
    assert [table_digests(cfg.db_path) for cfg in cfgs] == before


@pytest.fixture
def earlier_closure(monkeypatch):
    """The session table with one more closure in January 2026, which shifts every later session minute index."""
    monkeypatch.setattr(sessions, "CLOSED", sessions.CLOSED | {date(2026, 1, 5)})
    sessions._table.cache_clear()
    yield
    monkeypatch.undo()
    sessions._table.cache_clear()


def test_resume_refuses_a_changed_session_index(saved, tmp_path, earlier_closure):
    cfgs = fresh(saved, tmp_path)
    stores = [StockStore.open(cfg.db_path) for cfg in cfgs]
    with pytest.raises(ResumeError, match="the session table changed") as info:
        checkpoint.load(cfgs, stores)
    k = FRI.k(at(FRI, "11:05"))  # from the table as it was when the module loaded
    for name in ("s1_prices", "s2_news"):
        assert f"{name}: the session table changed, the checkpoint at {iso(CUT)} was at session minute index {k} " \
               f"and the table now gives {k - 390}" in str(info.value)
    for store in stores:
        store.close()


def rewrite(path: Path, change) -> None:
    """Rewrite a run's newest checkpoint after change(meta, arrays) altered it."""
    row = newest(path)
    meta, arrays = checkpoint.decode(row[2], row[3])
    change(meta, arrays)
    blob = checkpoint.v1.Arrays()
    for key, value in arrays.items():
        blob.put(key, value)
    text, data = checkpoint.v1.encode(meta, blob)
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("UPDATE checkpoints SET meta = ?, arrays = ? WHERE id = ?", (text, data, row[0]))
    conn.close()


@pytest.mark.parametrize("change", [
    lambda meta, arrays: arrays["market.3.standardizer.mean"].__setitem__(0, 1e-12 + arrays[
        "market.3.standardizer.mean"][0]),
    lambda meta, arrays: meta["market"].__setitem__("run_minutes", meta["market"]["run_minutes"] + 1),
], ids=["array", "value"])
def test_resume_refuses_copies_of_the_shared_layer_that_differ(saved, tmp_path, change):
    cfgs = fresh(saved, tmp_path)
    rewrite(cfgs[1].db_path, change)
    stores = [StockStore.open(cfg.db_path) for cfg in cfgs]
    with pytest.raises(ResumeError, match="copies of the shared market layer"):
        checkpoint.load(cfgs, stores)
    for store in stores:
        store.close()


def test_resume_refuses_runs_without_a_common_checkpoint_and_bots_without_any(saved, tmp_path):
    cfgs = fresh(saved, tmp_path)
    conn = sqlite3.connect(cfgs[1].db_path)
    with conn:
        conn.execute("DELETE FROM checkpoints")
    stores = [StockStore.open(cfg.db_path) for cfg in cfgs]
    with pytest.raises(ResumeError, match="share no readable checkpoint"):
        checkpoint.load(cfgs, stores)
    stores[0].commit_now([("DELETE FROM checkpoints", ())])
    with pytest.raises(ResumeError, match="holds bots but no readable checkpoint"):
        checkpoint.load(cfgs, stores)
    for store in stores:
        store.close()
    empty = pair(tmp_path / "empty", **FAST)
    stores = [StockStore.open(cfg.db_path) for cfg in empty]
    assert checkpoint.load(empty, stores) is None
    for store in stores:
        store.close()


def test_an_unreadable_newest_checkpoint_resumes_from_the_one_both_runs_hold_and_rolls_back(saved, tmp_path, caplog):
    cfgs = fresh(saved, tmp_path)
    conn = sqlite3.connect(cfgs[0].db_path)
    with conn:
        conn.execute("UPDATE checkpoints SET arrays = x'00' WHERE id = (SELECT max(id) FROM checkpoints)")
    conn.close()
    earlier = iso(CUT - 5 * 60_000)
    stores = [StockStore.open(cfg.db_path) for cfg in cfgs]
    with caplog.at_level("WARNING"):
        market, engines = checkpoint.load(cfgs, stores)
    assert [e.last_checkpoint_ts for e in engines] == [earlier, earlier]
    assert "rolled back" in caplog.text
    for cfg, store in zip(cfgs, stores):
        assert store.query("SELECT max(ts) FROM checkpoints")[0][0] == earlier
        for table, column in (("stock_candles", "ts"), ("decisions", "ts"), ("fills", "decided_at")):
            assert store.query(f"SELECT count(*) FROM {table} WHERE {column} > ?", (earlier,))[0][0] == 0
        store.close()
    assert market.session_day == FRI.day.isoformat()
    assert market.healthy_minutes == FRI.k(at(FRI, "11:00")) - FRI.first_k + 1
    # the rollback deleted the open session's row, and the load wrote it again in both runs as the checkpoint left it
    rows = [query(cfg.db_path, "SELECT * FROM sessions WHERE session_date = ?", (FRI.day.isoformat(),))
            for cfg in cfgs]
    assert rows[0] == rows[1] and len(rows[0]) == 1
    assert tuple(rows[0][0])[-2:] == (market.healthy_minutes, market.run_minutes)


def test_of_two_checkpoints_at_one_ms_resume_takes_the_newer(saved, tmp_path):
    cfgs = fresh(saved, tmp_path)
    for cfg in cfgs:
        _, ts, text, blob = newest(cfg.db_path)
        meta, arrays = checkpoint.decode(text, blob)
        meta[checkpoint.ENGINE]["news_seq"] = 99  # a shutdown checkpoint written in the same ms as a point
        copy_blob = checkpoint.v1.Arrays()
        for key, value in arrays.items():
            copy_blob.put(key, value)
        conn = sqlite3.connect(cfg.db_path)
        with conn:
            conn.execute("INSERT INTO checkpoints (ts, schema_version, meta, arrays) SELECT ts, schema_version, ?, ? "
                         "FROM checkpoints WHERE id = (SELECT max(id) FROM checkpoints)",
                         checkpoint.v1.encode(meta, copy_blob))
        conn.close()
        assert ts == iso(CUT)
    stores = [StockStore.open(cfg.db_path) for cfg in cfgs]
    _, engines = checkpoint.load(cfgs, stores)
    assert [e.news_seq for e in engines] == [99, 99] and [e.last_checkpoint_ts for e in engines] == [iso(CUT)] * 2
    for store in stores:
        store.close()
