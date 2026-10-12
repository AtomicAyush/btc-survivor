"""The run configs differ from the example only where PLAN section 21 says they do."""

import dataclasses
import sqlite3
import tomllib

import pytest

from btc_survivor.bot import decide
from btc_survivor.broker import estimate_round_trip
from btc_survivor.config import load_config
from btc_survivor.feed.book import LocalBook
from btc_survivor.types import FLAT, LONG, SHORT
from enginekit import fast_config, feed, new_engine, synthetic
from helpers import EXAMPLE_CONFIG, ROOT

EXAMPLE = load_config(EXAMPLE_CONFIG)
RUN_NAMES = ["run_a", "run_b", "run_c", "run_d", "run_e"]


def differences(cfg):
    out = {}
    for section in ("run", "feed", "time", "features", "model", "trading", "costs", "genes", "population",
                    "controls", "persistence"):
        a, b = dataclasses.asdict(getattr(EXAMPLE, section)), dataclasses.asdict(getattr(cfg, section))
        out.update({f"{section}.{k}": b[k] for k in a if a[k] != b[k]})
    return out


def flat_toml(name: str) -> dict:
    """Every setting of a run config as dotted keys, read from the file itself."""
    raw = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text(encoding="utf-8"))
    return {f"{section}.{key}": value for section, table in raw.items() for key, value in table.items()}


@pytest.mark.parametrize("name, changes", [
    ("run_a", {}),
    ("run_b", {"costs.fee_profile": "low_fee_tier"}),
    ("run_c", {"trading.start_equity": 500.0}),
    ("run_d", {"costs.fee_profile": "low_fee_tier", "time.horizon_minutes": 240, "genes.lr_init": 0.00003,
               "genes.lr_min": 0.000003, "genes.lr_max": 0.0003}),
    ("run_e", {"costs.fee_profile": "hypothetical_zero",
               "costs.taker_fees_bps": {"retail_default": 90.0, "active_retail": 30.0, "low_fee_tier": 5.0,
                                        "hypothetical_zero": 0.0},
               "costs.extra_slippage_bps": 0.0, "costs.short_funding_bps_per_hour": 0.0,
               "costs.short_open_fee_bps": 0.0, "genes.margin_bps_init": 0.0, "genes.margin_bps_min": 0.0}),
])
def test_each_run_changes_only_its_own_settings(name, changes):
    cfg = load_config(ROOT / "configs" / f"{name}.toml")
    assert differences(cfg) == {"run.data_dir": f"../data/{name}", **changes}
    assert cfg.data_dir.resolve() == (ROOT / "data" / name).resolve()


def test_every_run_config_is_listed():
    assert sorted(p.stem for p in (ROOT / "configs").glob("run_*.toml")) == RUN_NAMES


def test_run_e_differs_from_run_b_only_in_the_intended_keys():
    """Run E is run B made frictionless, so the seed, population, twins, horizon, timers, features, learning and
    rent all stay as in run B, and its founders start from the same networks."""
    b, e = flat_toml("run_b"), flat_toml("run_e")
    changed = {key: (b.get(key), e.get(key)) for key in sorted(b.keys() | e.keys()) if b.get(key) != e.get(key)}
    assert changed == {
        "run.data_dir": ("../data/run_b", "../data/run_e"),
        "costs.fee_profile": ("low_fee_tier", "hypothetical_zero"),
        "costs.taker_fee_bps_hypothetical_zero": (None, 0.0),
        "costs.extra_slippage_bps": (1.0, 0.0),
        "costs.short_funding_bps_per_hour": (0.5, 0.0),
        "costs.short_open_fee_bps": (2.0, 0.0),
        "genes.margin_bps_init": (8.0, 0.0),
        "genes.margin_bps_min": (2.0, 0.0),
    }
    assert e["run.master_seed"] == b["run.master_seed"] and e["costs.daily_running_cost"] == 1.16


def test_run_e_is_labelled_hypothetical_in_its_header():
    text = (ROOT / "configs" / "run_e.toml").read_text(encoding="utf-8")
    header = text.split("[run]")[0]
    assert all(line.startswith("#") or not line.strip() for line in header.splitlines())
    prose = " ".join(line.lstrip("#").strip() for line in header.splitlines())
    prose = " ".join(prose.split())
    assert "HYPOTHETICAL" in prose and "never a realistic cost setting" in prose
    assert "12 October 2026 by the owner's decision" in prose and "not a confirmatory test" in prose


def test_run_e_leaves_only_the_book_in_the_required_edge():
    """With no fee, slippage, short charges or margin, the required edge is the spread plus the walk on both legs,
    so a prediction far below anything runs A to D could act on passes the gate there and nowhere else."""
    book = LocalBook(20)
    book.apply_snapshot(bids=[(99_999.99, 0.004), (99_999.0, 1.0)], asks=[(100_000.01, 0.004), (100_001.0, 1.0)])
    e = load_config(ROOT / "configs" / "run_e.toml")
    qty = e.trading.notional / book.mid
    for side in (LONG, SHORT):
        est = estimate_round_trip(book, e.trading.notional, e.costs, side, e.time.horizon_minutes)
        walk = book.walk("buy" if side == LONG else "sell", qty).avg_price
        impact = abs(walk - (book.best_ask if side == LONG else book.best_bid)) / book.mid * 1e4
        assert est.fee_bps == est.slippage_bps == est.short_bps == 0.0
        assert est.total_bps == pytest.approx(book.spread_bps + 2 * impact)
        required = est.total_bps + e.genes.margin_bps_init
        pred = side * (required + 0.01)
        assert decide(FLAT, pred, required, True, False)[:2] == ("open_long" if side == LONG else "open_short", True)
    for name in ("run_a", "run_b", "run_c", "run_d"):
        cfg = load_config(ROOT / "configs" / f"{name}.toml")
        est = estimate_round_trip(book, cfg.trading.notional, cfg.costs, LONG, cfg.time.horizon_minutes)
        required = est.total_bps + cfg.genes.margin_bps_init
        assert required >= 20.0
        assert decide(FLAT, 8.0, required, True, False)[1] is False


def test_run_e_bots_trade_on_their_own_predictions(tmp_path):
    """Run E's costs and genes on the fast engine, with the forced timer out of reach. Every learner open passes the
    gate, none is forced, no fee is charged, the twins copy each move and no bot dies."""
    e = load_config(ROOT / "configs" / "run_e.toml")
    cfg = dataclasses.replace(fast_config(tmp_path, trading={"force_trade_after_hours": 1000.0}),
                              costs=e.costs, genes=e.genes)
    engine = new_engine(cfg)
    feed(engine, synthetic(3, 2400))
    engine.finish()
    engine.store.close()
    db = sqlite3.connect(cfg.db_path)
    try:
        moves = db.execute("SELECT b.origin, d.gate_passed, d.forced, count(*) FROM decisions d JOIN bots b "
                           "USING (bot_id) WHERE d.action IN ('open_long', 'open_short', 'flip') "
                           "GROUP BY 1, 2, 3").fetchall()
        fees = db.execute("SELECT count(*), coalesce(sum(fee), 0) FROM fills WHERE status = 'filled'").fetchone()
        deaths = db.execute("SELECT count(*) FROM deaths").fetchone()[0]
    finally:
        db.close()
    learner = [(gate, forced, n) for origin, gate, forced, n in moves if origin != "control"]
    assert learner and all(gate == 1 and forced == 0 for gate, forced, _ in learner)
    twin_moves = sum(n for origin, _, _, n in moves if origin == "control")
    assert 0 < twin_moves <= sum(n for _, _, n in learner)
    assert fees[0] > 0 and fees[1] == 0.0
    assert deaths == 0
