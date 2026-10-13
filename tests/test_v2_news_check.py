"""scripts/v2_news_check.py, the planted news check of section 6.4: its synthetic news by hand and in aggregate, the
stock process's own NewsInputs it runs that news through, the planted correlation, the paired market layer feeding S1
and S2, learners born alike that part only through news, its figures by hand, determinism and its output."""

import ast
import dataclasses
import gzip
import math
import sqlite3
import sys
from datetime import date, datetime, timezone

import numpy as np
import pytest

from btc_survivor.v2 import sessions, stats
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.news import score_store
from btc_survivor.v2.news.prefilter import LLM, MACRO
from btc_survivor.v2.stocks.bot import StockBot
from btc_survivor.v2.stocks.broker import StockBroker
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES
from btc_survivor.v2.stocks.labels import fits
from btc_survivor.v2.stocks.news_inputs import HAIKU_SCORER, Impact, ScoreLine
from helpers import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import v2_learner_check as lc  # noqa: E402
import v2_news_check as check  # noqa: E402
import v2_power  # noqa: E402
from v2_news_check import FAST, MATCHED, NONE, Figures, Generator, Settings  # noqa: E402

CFG = load_stock_config(ROOT / "configs" / "v2" / "s2_news.toml")
SYMBOLS, INDEX = CFG.stock_feed.symbols, CFG.stock_feed.index_symbol
HORIZON = CFG.time.horizon_minutes
HALF_LIFE, DROPS = CFG.news_input.half_life_session_minutes, CFG.news_input.drop_after_half_lives
GEN = Generator()
TAGS = (*SYMBOLS, MACRO)
MINUTE = 60_000


def days(count):
    return lc.run_sessions(lc.LAUNCH, count)


def line(seq, ms, tag, strength):
    return ScoreLine(seq, seq, "synthetic", "news", LLM, HAIKU_SCORER, "scored", ms, (tag,),
                     (Impact(tag, strength, True),), ms)


def settings(**changes):
    return dataclasses.replace(Settings(sessions=3, seeds=1, learners=1, calibration_seeds=1), **changes)


def planted(r, absorption=MATCHED, count=3, seed=0, gen=GEN):
    """A planted run whose beta comes from its own seed's news, enough for the layer and learner tests."""
    s = settings(sessions=count, generator=gen, signals=(0.0, r))
    span = days(count)
    beta = check.betas_from(CFG, s, [check.calibration_part(CFG, s, span, seed)])[absorption, r] if r else \
        np.zeros(len(SYMBOLS))
    return check.plant(check.make_draw(CFG, s, span, seed), absorption if r else NONE, beta, HORIZON)


def test_the_planning_values_are_those_of_the_power_script_and_the_run_config():
    power = v2_power.Generator()
    assert (GEN.impacts_per_session, GEN.session_share, GEN.neutral_share) == (
        power.impacts_per_session, power.session_share, power.neutral_share)
    assert (HALF_LIFE, HALF_LIFE * DROPS) == (power.half_life_minutes, power.drop_after_minutes)
    assert GEN.strength_scale == 1.0 and GEN.fast_half_life == 15.0 and GEN.prices == lc.Generator()
    full = Settings()
    assert (full.sessions, full.seeds, full.learners, full.calibration_seeds) == (40, 24, CFG.population.size, 8)
    assert full.signals == (0.0, 0.05, 0.1) and full.absorptions == (MATCHED, FAST) and full.launch == lc.LAUNCH
    assert check.scenarios(full) == [(NONE, 0.0), (MATCHED, 0.05), (MATCHED, 0.1), (FAST, 0.05), (FAST, 0.1)]


def test_overnight_items_start_at_the_previous_close():
    friday = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)  # 16:00 ET in daylight time
    assert check.previous_close(days(1)[0]) == int(friday.timestamp() * 1000)


def test_news_lines_arrive_at_the_planning_rates_inside_the_session_and_overnight():
    span = days(200)
    lines = check.news_lines(np.random.default_rng(1), span, TAGS, GEN)
    assert [x.seq for x in lines] == list(range(1, len(lines) + 1))
    assert [x.recv_ms for x in lines] == sorted(x.recv_ms for x in lines)
    for x in lines:
        assert (x.route, x.scorer, x.status, x.recv_ms) == (LLM, HAIKU_SCORER, "scored", x.scored_ms)
        assert x.item_seq == x.seq
        assert len(x.impacts) == 1 and x.impacts[0].counted and x.tickers == (x.impacts[0].ticker,)
    inside = sum(any(d.open_ms <= x.scored_ms < d.close_ms for d in span) for x in lines)
    assert inside / (len(span) * len(TAGS)) == pytest.approx(GEN.impacts_per_session * GEN.session_share, abs=0.1)
    outside = len(lines) - inside
    assert outside / (len(span) * len(TAGS)) == pytest.approx(GEN.impacts_per_session * (1 - GEN.session_share),
                                                               abs=0.1)
    assert lines[0].scored_ms >= check.previous_close(span[0]) and lines[-1].scored_ms < span[-1].close_ms
    strengths = np.array([x.impacts[0].strength for x in lines])
    assert np.mean(strengths == 0) == pytest.approx(GEN.neutral_share, abs=0.02)
    # prompt_v2's strengths, every one score_store.strength of a confidence, horizon and surprise.
    possible = {round(abs(score_store.strength(1, c, h, s)), 12) for c in range(1, 6)
                for h in score_store.HORIZON_WEIGHTS for s in range(4)}
    assert set(np.round(np.abs(strengths[strengths != 0]), 12)) <= possible
    assert {x.impacts[0].ticker for x in lines} == set(TAGS)
    halved = check.news_lines(np.random.default_rng(1), span, TAGS, dataclasses.replace(GEN, strength_scale=0.5))
    assert [x.impacts[0].strength for x in halved] == pytest.approx(list(strengths * 0.5))


def test_decayed_inputs_follow_section_6_3_by_hand():
    span = days(3)
    first, second = span[0], span[1]
    at = first.open_ms + 9 * MINUTE + 30_000  # session minute 9.5 of session 1
    night = second.open_ms - 3 * 60 * MINUTE  # three hours before the second open
    lines = [line(1, at, "AAPL", 0.8), line(2, night, MACRO, -0.4)]
    out = check.decayed(lines, span, SYMBOLS, INDEX, HALF_LIFE, DROPS)
    assert out.shape == (span[-1].last_k - first.first_k + 1, len(SYMBOLS), 2)
    a, q, m = SYMBOLS.index("AAPL"), SYMBOLS.index(INDEX), SYMBOLS.index("MSFT")
    assert (out[:9, a, 0] == 0).all()  # received after the boundary of minute 9
    for minute in (10, 100, 390):
        w = 2.0 ** (-(minute - 9.5) / HALF_LIFE)
        assert out[minute - 1, a, 0] == pytest.approx(0.8 * w / (w + 1))
    assert (out[:, q, 0] == out[:, a, 0]).all() and not out[:, m, 0].any()  # QQQ takes every company's items
    assert not out[:first.minutes, :, 1].any()  # the overnight item is not yet received
    for minute in (1, 200):  # it keeps full weight to the open, then decays in session minutes
        w = 2.0 ** (-minute / HALF_LIFE)
        assert out[first.minutes + minute - 1, :, 1] == pytest.approx(np.full(len(SYMBOLS), -0.4 * w / (w + 1)))
    dropped = first.minutes + int(600 - (first.minutes - 9.5)) + 1  # older than 600 session minutes
    assert out[dropped - 2, a, 0] > 0 and out[dropped - 1, a, 0] == 0
    fast = check.decayed(lines, span, SYMBOLS, INDEX, 15.0, 40.0)
    w = 2.0 ** (-(100 - 9.5) / 15.0)
    assert fast[99, a, 0] == pytest.approx(0.8 * w / (w + 1))


def test_one_seed_draws_the_same_prices_and_news_in_every_scenario():
    s, span = settings(), days(3)
    one, two = check.make_draw(CFG, s, span, 0), check.make_draw(CFG, s, span, 0)
    for name in ("raw", "noise", "inputs"):
        np.testing.assert_array_equal(getattr(one, name), getattr(two, name))
    assert not np.array_equal(one.inputs, check.make_draw(CFG, s, span, 1).inputs)
    np.testing.assert_array_equal(one.planted[MATCHED], one.inputs.sum(axis=-1))
    assert one.planted[FAST].shape == one.planted[MATCHED].shape
    assert not np.array_equal(one.planted[FAST], one.planted[MATCHED])
    assert one.inputs.any() and np.abs(one.inputs).max() <= 1 and one.items > 0
    lines = check.news_lines(np.random.default_rng(np.random.SeedSequence([s.seed, check.NEWS_STREAM, 0])), span,
                             TAGS, GEN)
    np.testing.assert_array_equal(one.planted[FAST],
                                  check.decayed(lines, span, SYMBOLS, INDEX, 15.0, 40.0).sum(axis=-1))
    calibration = check.calibration_part(CFG, s, span, 0)
    assert not np.array_equal(calibration[MATCHED][0], one.planted[MATCHED][check.evaluation_rows(span, HORIZON)])


def test_evaluation_rows_and_window_thirds():
    span = days(3)
    rows = check.evaluation_rows(span, HORIZON)
    offset = span[1].first_k - span[0].first_k
    assert rows.tolist() == [offset + m - 1 for m in range(30, 361, 30)] + [offset + 390 + m - 1
                                                                            for m in range(30, 361, 30)]
    thirds = check.window_thirds(span)
    assert np.concatenate(thirds).tolist() == list(range(offset, offset + 780))
    assert [len(t) for t in thirds] == [260, 260, 260]


def test_horizon_sums_and_the_planted_labels_by_hand():
    data_none = planted(0.0)
    data = planted(0.1)
    span = data.days
    ks = np.arange(span[0].first_k, span[-1].last_k + 1)
    assert (np.isfinite(data.labels) == np.array([fits(int(k), HORIZON) for k in ks])[:, None]).all()
    inside = np.flatnonzero(np.isfinite(data.labels[:, 0]))
    np.testing.assert_allclose(data.labels[inside], 1e4 * np.log(data.prices[inside + HORIZON] / data.prices[inside]),
                               atol=1e-6)
    sums = check.horizon_sums(data.planted, span, HORIZON)
    t = inside[5]
    assert sums[t, 0] == pytest.approx(data.planted[t:t + HORIZON, 0].sum())
    beta = (data.labels - data_none.labels)[inside] / sums[inside]
    assert np.allclose(beta, beta[0], rtol=1e-6) and (beta[0] > 0).all()  # the drift is beta times the sums alone
    np.testing.assert_array_equal(data_none.planted, 0.0)
    np.testing.assert_array_equal(data.raw, data_none.raw)


def test_signal_beta_gives_the_population_correlation_and_refuses_one_out_of_reach():
    rng = np.random.default_rng(3)
    p = rng.normal(size=5000)
    d = 20 * p + rng.normal(size=5000) * 4
    beta = check.signal_beta(0.1, p, d, 30.0)
    cov = np.mean((p - p.mean()) * (d - d.mean()))
    assert beta * cov / (p.std() * math.sqrt(beta ** 2 * d.var() + 30.0 ** 2)) == pytest.approx(0.1)
    assert check.signal_beta(0.0, p, d, 30.0) == 0.0
    with pytest.raises(ValueError, match="out of reach"):
        check.signal_beta(0.99, p, d, 30.0)


def test_betas_are_per_symbol_and_plant_the_correlation_over_seeds():
    s = settings(sessions=20, calibration_seeds=2, signals=(0.0, 0.1))
    span = days(20)
    betas = check.betas_from(CFG, s, [check.calibration_part(CFG, s, span, j) for j in range(2)])
    assert set(betas) == {(NONE, 0.0), (MATCHED, 0.1), (FAST, 0.1)} and not betas[NONE, 0.0].any()
    assert (betas[FAST, 0.1] > betas[MATCHED, 0.1]).all()  # a fast reaction needs a larger drift for one r
    rows = check.evaluation_rows(span, HORIZON)
    ideal = {a: [] for a in (MATCHED, FAST)}
    for seed in range(4):
        draw = check.make_draw(CFG, s, span, seed)
        for a in ideal:
            data = check.plant(draw, a, betas[a, 0.1], HORIZON)
            ideal[a].append(check.skill(rows, data.planted, data.labels).corr)
    assert np.mean(ideal[MATCHED]) == pytest.approx(0.1, abs=0.035)
    assert np.mean(ideal[FAST]) == pytest.approx(0.1, abs=0.035)


def test_the_paired_market_layer_feeds_s1_zeros_and_s2_the_inputs_of_each_push_minute():
    data = planted(0.1)
    seen = []
    for i, one, two in check.paired_minutes(CFG, data):  # the windows change as the minutes go, so check each now
        seen.append((i, one.k, one.window is not None and one.window is two.window))
        assert (one.k, one.boundary_ms, one.matured) == (two.k, two.boundary_ms, two.matured)
        np.testing.assert_array_equal(one.label_sd, two.label_sd)
        np.testing.assert_array_equal(one.rows[:, :N_PRICE_FEATURES], two.rows[:, :N_PRICE_FEATURES])
        assert not one.rows[:, N_PRICE_FEATURES:].any()
        np.testing.assert_array_equal(two.rows[:, N_PRICE_FEATURES:], data.inputs[i])
        if two.window is not None and two.matured:
            newest = two.window.x[two.window._slot(two.window.count - 1)]
            s, push, _ = two.matured[-1]
            np.testing.assert_array_equal(newest[N_PRICE_FEATURES:], data.inputs[push - data.first_k, s])
            assert not one.window.x[:, N_PRICE_FEATURES:].any()
    assert seen[0][1] == data.days[1].first_k and len(seen) == sum(d.minutes for d in data.days[1:])
    assert [i for i, _, _ in seen] == list(range(data.days[1].first_k - data.first_k, len(data.prices)))
    assert not any(shared for _, _, shared in seen)


def test_learning_with_news_inputs_matches_the_bot_step():
    data = planted(0.1, count=2)
    born = data.days[0].open_ms
    stepped, learned = (StockBot.newborn(CFG, 1, 0, born) for _ in range(2))
    broker = StockBroker.from_config(CFG)
    for _, _, ctx in check.paired_minutes(CFG, data):
        assert not stepped.step(ctx, broker).orders
        lc.learn_and_predict(learned, ctx)
        np.testing.assert_array_equal(stepped.preds, learned.preds)
    assert stepped.trained_minutes == learned.trained_minutes == 331
    np.testing.assert_array_equal(stepped.model.W1, learned.model.W1)
    assert learned.model.W1[:, N_PRICE_FEATURES:].any()


def test_s1_and_s2_stay_identical_without_news_and_part_only_through_it():
    quiet = planted(0.0, gen=dataclasses.replace(GEN, neutral_share=1.0))
    assert not quiet.inputs.any()
    alike = check.run_pair(CFG, quiet, 2, 1)
    np.testing.assert_array_equal(alike.s1, alike.s2)
    assert alike.s1_news_zero and alike.news_weight == (0.0, 0.0, 0.0)
    data = planted(0.1)
    parted = check.run_pair(CFG, data, 2, 1)
    assert parted.s1_news_zero and parted.news_weight[-1] > 0 and len(parted.news_weight) == 3
    first = np.flatnonzero(np.isfinite(parted.s1[:, 0]))[0]
    assert first == data.days[1].first_k - data.first_k + 59  # label scales ready at session minute 60
    assert not np.array_equal(parted.s1[first:], parted.s2[first:])
    span = data.days[1].first_k - data.first_k
    assert np.isnan(parted.s1[:span]).all() and np.isnan(parted.label_sd[:span]).all()


def figures(**changes):
    base = Figures(0.1, 0.1, 2.0, 1.0, 0.5, 1.5, 2.5, (0.5, 0.6, 0.7), (0.01, 0.02, 0.03), 0.2, 100, True)
    return dataclasses.replace(base, **changes)


def test_judge_uses_the_pooled_statistics_of_evaluate_by_hand():
    data = planted(0.1)
    rng = np.random.default_rng(5)
    shape = data.prices.shape
    label_sd = np.full(shape, 20.0)
    s1 = rng.normal(size=shape)
    s2 = s1 + 3.0 * data.planted * label_sd
    s1[: data.days[1].first_k - data.first_k + 59] = np.nan
    s2[np.isnan(s1)] = np.nan
    rows = check.evaluation_rows(data.days, HORIZON)
    s1[rows[-1], 2] = np.nan  # one missing S1 row stays in S2's skill and leaves the paired test
    learned = check.Learned(s1, s2, label_sd, (0.1, 0.2, 0.3), 0.25, True)
    found = check.judge(data, learned, HORIZON)
    assert found.extraction == pytest.approx((1.0, 1.0, 1.0))
    keep = [(r, s) for r in rows for s in range(len(SYMBOLS)) if np.isfinite(s1[r, s]) and np.isfinite(s2[r, s])]
    minute, symbol = [r for r, _ in keep], [str(s) for _, s in keep]
    by_hand = stats.pooled_dm(minute, symbol, [s2[r, s] for r, s in keep], [s1[r, s] for r, s in keep],
                              [data.labels[r, s] for r, s in keep])
    assert found.dm_t == by_hand.t and found.rows == by_hand.rows == len(keep)
    every = [(r, s) for r in rows for s in range(len(SYMBOLS))]
    ideal = stats.pooled_skill([r for r, _ in every], [str(s) for _, s in every],
                               [data.planted[r, s] for r, s in every], [data.labels[r, s] for r, s in every])
    assert found.ideal == ideal.corr and found.news_corr == ideal.corr  # matched, the planted predictor is news
    assert found.s1_t == check.skill(rows, s1, data.labels).t and math.isfinite(found.part_t)
    assert (found.news_weight, found.price_weight, found.s1_news_zero) == ((0.1, 0.2, 0.3), 0.25, True)
    quiet = check.judge(dataclasses.replace(data, planted=np.zeros(shape)), learned, HORIZON)
    assert math.isnan(quiet.ideal) and all(math.isnan(x) for x in quiet.extraction)


def test_aggregate_averages_seeds_and_counts_passes_above_2_33():
    found = [figures(news_t=2.34, dm_t=2.33), figures(news_t=math.nan, dm_t=3.0, ideal=math.nan),
             figures(news_t=1.0, dm_t=math.nan, extraction=(math.nan, 0.2, 0.3), s1_news_zero=False)]
    s = check.aggregate((MATCHED, 0.1), found)
    assert (s.absorption, s.signal, s.seeds, s.rows) == (MATCHED, 0.1, 3, 300)
    assert s.news_passed == pytest.approx(1 / 3) and s.s2_beat_s1 == pytest.approx(1 / 3)
    assert s.news_t == pytest.approx((2.34 + 1.0) / 2) and s.dm_t == pytest.approx((2.33 + 3.0) / 2)
    assert s.ideal == pytest.approx(0.1) and s.extraction == pytest.approx((0.5, 1.4 / 3, 1.7 / 3))
    assert s.news_weight == pytest.approx((0.01, 0.02, 0.03)) and not s.s1_news_zero
    assert math.isnan(check.mean_of([math.nan])) and check.passed([math.nan, 2.4]) == 0.5


def test_the_learner_extracts_a_strong_planted_signal():
    data = planted(0.3, count=6)
    learned = check.run_pair(CFG, data, 2, 1)
    found = check.judge(data, learned, HORIZON)
    assert found.extraction[-1] > 0.5 and found.news_weight[-1] > found.news_weight[0]
    assert found.part_t > 0


def same(a, b):
    np.testing.assert_equal([dataclasses.astuple(x) for x in a], [dataclasses.astuple(x) for x in b])


def test_the_same_seed_gives_the_same_curve_whatever_the_workers_and_nothing_is_opened(monkeypatch):
    s = settings(seeds=2, signals=(0.0, 0.1))
    first = check.simulate(CFG, s, workers=2)

    def refuse(*args, **kwargs):
        raise AssertionError("the news check opened a database or a recording")

    monkeypatch.setattr(sqlite3, "connect", refuse)
    monkeypatch.setattr(gzip, "open", refuse)
    same(first, check.simulate(CFG, s))
    other = check.simulate(CFG, dataclasses.replace(s, seed=1))
    assert [x.dm_t for x in other] != [x.dm_t for x in first]
    assert [(x.absorption, x.signal, x.seeds) for x in first] == [(NONE, 0.0, 2), (MATCHED, 0.1, 2), (FAST, 0.1, 2)]
    assert all(x.s1_news_zero for x in first) and first[0].rows == first[1].rows > 0


def test_main_prints_counts_a_verdict_and_a_timing_and_the_rows_only_with_detail(capsys):
    assert check.main(["--sessions", "3", "--seeds", "1", "--learners", "1", "--calibration-seeds", "1",
                       "--detail"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:8] == ["synthetic planted news check", "sessions: 3", "seeds: 1", "calibration seeds: 1",
                         "learners per run: 1", "symbols: 8", "scenarios: 5",
                         f"evaluation rows per seed: {24 * len(SYMBOLS)}"]
    assert lines[8] == "price only learners keep zero news weights: pass"
    assert lines[9].startswith("run time: ") and lines[9].endswith(" s")
    assert lines[10].split()[:4] == ["absorption", "planted", "ideal", "news"]
    assert [row.split()[:2] for row in lines[11:]] == [["none", "0.00"], ["matched", "0.05"], ["matched", "0.10"],
                                                       ["fast", "0.05"], ["fast", "0.10"]]
    assert check.main(["--sessions", "3", "--seeds", "1", "--learners", "1", "--calibration-seeds", "1"]) == 0
    short = capsys.readouterr().out.splitlines()
    assert short[:9] == lines[:9] and short[9].startswith("run time: ")
    assert short[10:] == ["5 rows hidden, pass --detail in your own terminal to see them"]


@pytest.mark.parametrize("args", [["--sessions", "2"], ["--seeds", "0"], ["--learners", "0"],
                                  ["--calibration-seeds", "0"], ["--workers", "0"], ["--strength-scale", "0"],
                                  ["--strength-scale", "1.5"]])
def test_main_refuses_settings_outside_their_range(args):
    with pytest.raises(SystemExit):
        check.main(args)


def test_the_script_opens_nothing_itself_and_imports_no_feed_store_or_database_module():
    source = (ROOT / "scripts" / "v2_news_check.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            raise AssertionError("the news check opens a file")
    assert imported == {"__future__", "argparse", "math", "multiprocessing", "time", "collections.abc",
                        "concurrent.futures", "dataclasses", "datetime", "pathlib", "numpy", "v2_learner_check",
                        "v2_power",
                        "btc_survivor.v2", "btc_survivor.v2.config", "btc_survivor.v2.news.prefilter",
                        "btc_survivor.v2.sessions", "btc_survivor.v2.stocks.bot", "btc_survivor.v2.stocks.features",
                        "btc_survivor.v2.stocks.labels", "btc_survivor.v2.stocks.learning",
                        "btc_survivor.v2.stocks.news_inputs"}
    assert sessions.all_sessions()[-1].day >= date(2027, 1, 1)  # room for the full 40 sessions from the launch
