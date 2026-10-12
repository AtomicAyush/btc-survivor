"""scripts/v2_learner_check.py, the synthetic learning rate check of section 6.4: its calendar, its generator by hand
and in aggregate, the market layer it builds, the bots' own learning code it runs, its figures, the rule, and the rate
the run configs carry."""

import ast
import dataclasses
import gzip
import math
import sqlite3
import sys
from datetime import date

import numpy as np
import pytest

from btc_survivor.bot import WARMUP
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks.broker import StockBroker
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES
from btc_survivor.v2.stocks.labels import fits
from helpers import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import v2_learner_check as check  # noqa: E402
from v2_learner_check import Generator, Scenario, Settings  # noqa: E402

CONFIGS = ROOT / "configs" / "v2"
CFG = load_stock_config(CONFIGS / "s1_prices.toml")
SYMBOLS, INDEX = CFG.stock_feed.symbols, CFG.stock_feed.index_symbol
HORIZON = CFG.time.horizon_minutes
GEN = Generator()
CHOSEN = 0.0001  # what the full default check chose on 12 October 2026
# Session 2 trains from minute 60, once 30 labels per symbol have matured, through 390, 331 training minutes, so
# warmup's 360 ends at minute 59 of session 3 and labels exist through minute 360.
ACTIVE_MINUTES = 360 - 59 + 1


def synthetic(signal, sessions=3, seed=0):
    days = check.run_sessions(check.LAUNCH, sessions)
    return check.synthesize(np.random.SeedSequence([7, seed]), SYMBOLS, INDEX, days, signal, HORIZON, GEN)


def scenario(rate, spread, ratio, signal=0.0):
    return Scenario(signal, rate, spread, ratio, 0.0, 0.0, 1, 1)


def test_sessions_run_from_the_launch_on_the_real_calendar():
    days = check.run_sessions(check.LAUNCH, 40)
    assert (days[0].day, days[-1].day, len(days)) == (date(2026, 10, 19), date(2026, 12, 14), 40)
    assert date(2026, 11, 26) not in {d.day for d in days}
    assert [d.day for d in days if d.early_close] == [date(2026, 11, 27)]
    assert all(b.first_k == a.last_k + 1 for a, b in zip(days, days[1:]))
    with pytest.raises(ValueError, match="holds only"):
        check.run_sessions(date(2028, 12, 1), 40)


def test_labels_are_the_next_horizon_of_moves_and_never_cross_the_close():
    data = synthetic(0.1, sessions=30)
    ks = np.arange(data.first_k, data.days[-1].last_k + 1)
    assert (np.isfinite(data.labels) == np.array([fits(int(k), HORIZON) for k in ks])[:, None]).all()
    inside = np.flatnonzero(np.isfinite(data.labels[:, 0]))
    expected = 1e4 * np.log(data.prices[inside + HORIZON] / data.prices[inside])
    np.testing.assert_allclose(data.labels[inside], expected, atol=1e-6)
    early = next(d for d in data.days if d.early_close)
    assert np.isfinite(data.labels[early.last_k - HORIZON - data.first_k]).all()
    assert np.isnan(data.labels[early.last_k - HORIZON + 1 - data.first_k]).all()


def test_one_seed_gives_the_same_noise_at_every_signal():
    none, some = synthetic(0.0), synthetic(0.1)
    np.testing.assert_array_equal(none.raw, some.raw)
    assert not none.truth.any() and some.truth.all()
    beta = np.array([check.signal_beta(0.1, GEN.signal_persistence, HORIZON, check.SD_BPS[s] / math.sqrt(HORIZON))
                     for s in SYMBOLS])
    planted = np.cumsum(beta * some.raw[:, :, 0], axis=0)
    np.testing.assert_allclose(1e4 * np.log(some.prices / none.prices), planted, atol=1e-6)
    np.testing.assert_allclose(some.truth, beta * check.carried(GEN.signal_persistence, HORIZON) * some.raw[:, :, 0])


def test_signal_beta_gives_the_correlation_by_hand_and_refuses_one_out_of_reach():
    phi, sd = 0.97, 6.0
    c = sum(phi ** j for j in range(1, 31))
    v = 30 + 2 * sum((30 - d) * phi ** d for d in range(1, 30))
    beta = check.signal_beta(0.1, phi, 30, sd)
    assert beta * c / math.sqrt(beta ** 2 * v + 30 * sd ** 2) == pytest.approx(0.1)
    assert check.carried(phi, 30) == pytest.approx(c)
    assert check.signal_beta(0.0, phi, 30, sd) == 0.0
    with pytest.raises(ValueError, match="not reachable"):
        check.signal_beta(0.75, phi, 30, sd)


@pytest.mark.parametrize("signal", [0.05, 0.1])
def test_the_planted_conditional_mean_has_the_requested_correlation(signal):
    data = synthetic(signal, sessions=200)
    ok = np.isfinite(data.labels)
    corr = [np.corrcoef(data.truth[ok[:, s], s], data.labels[ok[:, s], s])[0, 1] for s in range(len(SYMBOLS))]
    assert np.mean(corr) == pytest.approx(signal, abs=0.01)


def test_the_generator_follows_the_planning_values():
    data = synthetic(0.0, sessions=100)
    np.testing.assert_allclose(np.nanstd(data.labels, axis=0), [check.SD_BPS[s] for s in SYMBOLS], rtol=0.05)
    stocks, q = [i for i, s in enumerate(SYMBOLS) if s != INDEX], SYMBOLS.index(INDEX)

    def shares(series):
        c = np.corrcoef(series.T)
        return np.mean([c[a, b] for a in stocks for b in stocks if a < b]), np.mean([c[a, q] for a in stocks])

    together = math.sqrt(GEN.stock_market_share * GEN.index_market_share)
    assert shares(np.diff(np.log(data.prices), axis=0)) == pytest.approx((GEN.stock_market_share, together), abs=0.03)
    assert shares(data.raw[:, :, 5]) == pytest.approx((GEN.stock_market_share, together), abs=0.05)
    assert shares(data.raw[:, :, 0]) == pytest.approx((0.0, 0.0), abs=0.05)
    signal = data.raw[:, :, 0]
    lag = np.mean([np.corrcoef(signal[1:, s], signal[:-1, s])[0, 1] for s in range(len(SYMBOLS))])
    assert lag == pytest.approx(GEN.signal_persistence, abs=0.005)


def test_the_market_layer_starts_in_session_two_and_matures_every_record_at_its_horizon():
    data = synthetic(0.0)
    minutes = list(check.market_minutes(CFG, data))
    second = data.days[1]
    assert minutes[0][1].k == second.first_k and len(minutes) == sum(d.minutes for d in data.days[1:])
    for i, ctx in minutes:
        minute, first_session = ctx.session_minute, ctx.session == second
        assert i == ctx.k - data.first_k
        assert len(ctx.matured) == (len(SYMBOLS) if minute > HORIZON else 0)
        for s, push, label in ctx.matured:
            assert push == ctx.k - HORIZON and label == pytest.approx(data.labels[push - data.first_k, s])
        assert (ctx.window is None) == (first_session and minute < 60)
        assert np.isnan(ctx.label_sd).all() == (first_session and minute < 60)
        assert ctx.rows[:, :N_PRICE_FEATURES].any() == (not first_session or minute > CFG.features.min_samples)
        assert not ctx.rows[:, N_PRICE_FEATURES:].any() and ctx.news_ok and ctx.valid.all()


def test_learn_and_predict_trains_and_predicts_as_the_bot_step_does():
    data = synthetic(0.1, sessions=2)
    born = data.days[0].open_ms
    stepped = check.newborns(CFG, (CHOSEN,), 1, 1, born)[CHOSEN][0]
    learned = check.newborns(CFG, (CHOSEN,), 1, 1, born)[CHOSEN][0]
    broker = StockBroker.from_config(CFG)
    for _, ctx in check.market_minutes(CFG, data):
        assert not stepped.step(ctx, broker).orders
        check.learn_and_predict(learned, ctx)
        np.testing.assert_array_equal(stepped.preds, learned.preds)
    assert stepped.trained_minutes == learned.trained_minutes == 331
    assert stepped.state == learned.state == WARMUP
    for name in ("W1", "b1", "W2", "b2"):
        np.testing.assert_array_equal(getattr(stepped.model, name), getattr(learned.model, name))


def test_learners_of_every_rate_are_born_alike():
    bots = check.newborns(CFG, check.RATES, 2, 4, 0)
    for j in range(2):
        first = bots[check.RATES[0]][j]
        for rate in check.RATES:
            other = bots[rate][j]
            assert other.cfg.model.learning_rate == rate and other.bot_id == 4 + j and other.slot == j
            assert other.cfg == dataclasses.replace(CFG, model=dataclasses.replace(CFG.model, learning_rate=rate))
            for name in ("W1", "b1", "W2", "b2"):
                np.testing.assert_array_equal(getattr(other.model, name), getattr(first.model, name))
            assert other.rng.bit_generator.state == first.rng.bit_generator.state
    assert not np.array_equal(bots[check.RATES[0]][0].model.W1, bots[check.RATES[0]][1].model.W1)


def test_a_run_trains_in_session_two_and_judges_the_active_minutes_with_labels():
    data = synthetic(0.0)
    outcomes, samples = check.run(CFG, data, (0.0003, CHOSEN), 2, 1)
    assert samples == len(SYMBOLS) * 360 * 2
    assert {o.rows for o in outcomes.values()} == {len(SYMBOLS) * ACTIVE_MINUTES}
    assert outcomes[0.0003] != outcomes[CHOSEN]


def test_judge_by_hand():
    labels = np.array([[1.0, 2.0], [np.nan, -1.0], [3.0, -2.0]])
    sd = np.array([[1.0, 2.0], [1.0, 2.0], [np.nan, 2.0]])
    data = check.Synthetic((), np.zeros((3, 2, N_PRICE_FEATURES)), np.ones((3, 2)), labels, np.zeros((3, 2)))
    out = np.array([[[0.1, np.nan], [0.2, 0.3], [0.1, -0.2]],
                    [[0.3, 0.1], [0.0, -0.1], [0.2, 0.4]]])
    result = check.judge(out, data, sd)  # rows (0, 0), (1, 1) and (2, 1), labels 1, -0.5 and -1
    assert result.rows == 3
    assert result.spread == pytest.approx((math.sqrt(0.14 / 3) + math.sqrt(0.26 / 3)) / 2)
    assert result.error_ratio == pytest.approx((2.09 + 2.61) / 2 / 2.25)
    assert result.corr == pytest.approx(np.corrcoef([0.2, 0.1, 0.1], [1.0, -0.5, -1.0])[0, 1])
    assert result.ideal == 0.0
    nothing = check.judge(np.full((1, 3, 2), np.nan), data, sd)
    assert nothing.rows == 0 and math.isnan(nothing.spread) and math.isnan(nothing.error_ratio)
    assert not check.qualifies(Scenario(0.0, CHOSEN, nothing.spread, nothing.error_ratio, 0.0, 0.0, 0, 0))


def test_the_rule_takes_the_largest_rate_meeting_both_limits_with_no_signal():
    r = check.RATES
    assert r == (0.0003, 0.0001, 0.00003, 0.00001)
    at_the_edges = [scenario(r[0], 0.1001, 1.0), scenario(r[1], 0.10, 1.02), scenario(r[2], 0.05, 1.0),
                    scenario(r[3], 0.02, 1.0)]
    assert check.choose_rate(at_the_edges) == 0.0001
    error_over = [scenario(r[0], 0.05, 1.03), scenario(r[1], 0.05, 1.0201), scenario(r[2], 0.05, 1.0),
                  scenario(r[3], 0.05, 1.0)]
    assert check.choose_rate(error_over) == 0.00003
    only_signal = [scenario(r[0], 0.01, 0.9, signal=0.1), *(scenario(rate, 0.2, 1.0) for rate in r)]
    assert check.choose_rate(only_signal) == check.FALLBACK_RATE == 0.00001
    assert check.choose_rate([scenario(rate, math.nan, math.nan) for rate in r]) == 0.00001
    assert check.choose_rate([scenario(r[1], 0.05, 1.0), scenario(r[2], 0.2, 1.0)]) == 0.0001


def test_the_learner_finds_a_strong_signal_and_stays_small_on_none():
    strong = check.run(CFG, synthetic(0.3, sessions=6), (0.0003,), 1, 1)[0][0.0003]
    assert strong.corr > 0.15 and strong.error_ratio < 0.98
    none = check.run(CFG, synthetic(0.0, sessions=6), (CHOSEN,), 1, 1)[0][CHOSEN]
    assert none.spread < 0.1 and none.error_ratio < 1.02


def test_the_same_seed_gives_the_same_scenarios_and_nothing_is_opened(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the learner check opened a database or a recording")

    monkeypatch.setattr(sqlite3, "connect", refuse)
    monkeypatch.setattr(gzip, "open", refuse)
    settings = Settings(sessions=3, seeds=1, learners=1, rates=(0.0003, 0.00001), signals=(0.0, 0.1))
    first = check.simulate(CFG, settings)
    assert first == check.simulate(CFG, settings)
    assert first != check.simulate(CFG, dataclasses.replace(settings, seed=1))
    assert [(s.signal, s.rate) for s in first] == [(0.0, 0.0003), (0.0, 0.00001), (0.1, 0.0003), (0.1, 0.00001)]
    assert {(s.rows, s.samples) for s in first} == {(len(SYMBOLS) * ACTIVE_MINUTES, len(SYMBOLS) * 360 * 2)}


def test_main_prints_counts_a_verdict_a_timing_the_synthetic_rows_and_the_chosen_rate(capsys):
    assert check.main(["--sessions", "3", "--seeds", "1", "--learners", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:7] == ["synthetic learner check", "sessions: 3", "seeds: 1", "learners per rate: 1", "symbols: 8",
                         "scenarios: 12", f"matured samples per scenario: {len(SYMBOLS) * 360 * 2}"]
    assert lines[7].split(": ")[0] == "a rate meets the rule" and lines[7].split(": ")[1] in ("pass", "fail")
    assert lines[8].startswith("run time: ") and lines[8].endswith(" s")
    assert lines[9].split()[:3] == ["signal", "learning", "rate"]
    rows = [line.split() for line in lines[10:22]]
    assert [(row[0], float(row[1])) for row in rows] == [(f"{s:.2f}", r) for s in check.SIGNALS for r in check.RATES]
    assert all(row[6] == str(len(SYMBOLS) * ACTIVE_MINUTES) and row[7] in ("pass", "fail") for row in rows[:4])
    assert lines[22:24] == ["chosen learning rate", "learning rate"] and float(lines[24]) in check.RATES
    assert len(lines) == 25 and "hidden" not in "\n".join(lines)


@pytest.mark.parametrize("args", [["--sessions", "2"], ["--seeds", "0"], ["--learners", "0"]])
def test_main_refuses_settings_outside_their_range(args):
    with pytest.raises(SystemExit):
        check.main(args)


def test_the_script_opens_nothing_itself_and_imports_no_feed_news_or_store_module():
    source = (ROOT / "scripts" / "v2_learner_check.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            raise AssertionError("the learner check opens a file")
    assert imported == {"__future__", "argparse", "dataclasses", "math", "time", "collections.abc", "datetime",
                        "pathlib", "numpy", "btc_survivor.bot", "btc_survivor.v2", "btc_survivor.v2.config",
                        "btc_survivor.v2.sessions", "btc_survivor.v2.stocks.bot", "btc_survivor.v2.stocks.features",
                        "btc_survivor.v2.stocks.labels", "btc_survivor.v2.stocks.learning"}


def test_every_run_config_carries_the_rate_the_check_chose():
    paths = [CONFIGS / "s1_prices.toml", CONFIGS / "s2_news.toml", *sorted(CONFIGS.glob("rehearsal_s*.toml"))]
    assert {load_stock_config(path).model.learning_rate for path in paths} == {CHOSEN}
    assert CHOSEN in check.RATES
