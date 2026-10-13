"""scripts/v2_power.py, the synthetic power of sections 11.4 and 11.5: its generator by hand and in aggregate, the
detectable correlations of the plan's table reproduced with fixed fast settings, and the incremental test H2 holding
its level, keeping most of H1's power and refusing news that only repeats a price score."""

import ast
import math
import sys

import numpy as np
import pytest

from btc_survivor.v2.news import score_store
from helpers import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import v2_power  # noqa: E402
from v2_power import Generator, Settings  # noqa: E402

GEN = Generator()
FAST = Settings(runs=300, sessions=(40,), cs=(0.2, 1.0))


@pytest.fixture(scope="module")
def fast():
    return {s.c: s for s in v2_power.simulate(FAST)}


def test_detectable_correlations_match_the_plan_near_0_076_and_0_123(fast):
    assert fast[0.2].detectable == pytest.approx(0.076, abs=0.004)
    assert fast[1.0].detectable == pytest.approx(0.123, abs=0.004)


def test_the_fast_run_holds_the_null_level_and_power_rises_with_the_true_correlation(fast):
    for scenario in fast.values():
        assert scenario.null_rejection <= v2_power.null_bound(FAST.runs)
        low, high = scenario.power
        assert low < high and scenario.power_90 >= 0.07
    assert fast[1.0].power_90 > fast[0.2].power_90  # a shared macro input costs power
    assert fast[0.2].power[0] > fast[1.0].power[0]


def one(block, offset, strength, blocks=30):
    """The input at every block start from hand placed impacts on one input of one run."""
    cell = np.array(block)
    return v2_power.inputs_at(cell, np.array(offset, float), np.array(strength, float), (1, 1, blocks), GEN)[0, 0]


def test_h2_holds_its_level_and_the_sequence_keeps_most_of_h1_s_power(fast):
    bound = v2_power.null_bound(FAST.runs)
    for scenario in fast.values():
        assert scenario.h2_null_rejection <= bound
        assert all(seq <= h1 for seq, h1 in zip(scenario.h2_power, scenario.power))
    assert fast[0.2].h2_power[1] >= 0.9 and fast[0.2].h2_power[0] >= fast[0.2].power[0] - 0.1


def test_news_that_only_repeats_momentum_passes_h1_often_and_the_sequence_within_the_level(fast):
    bound = v2_power.null_bound(FAST.runs)
    (mild_h1, mild_seq), (strong_h1, strong_seq) = fast[0.2].confounded
    assert strong_h1 > 0.5 and mild_h1 > 2 * bound  # the raw test mistakes momentum for news
    assert all(seq <= bound for s in fast.values() for _, seq in s.confounded)


def test_the_price_scores_never_change_h1():
    plain = v2_power.simulate(Settings(runs=8, sessions=(3,), cs=(0.5,), confounds=(), chunk=4))
    full = v2_power.simulate(Settings(runs=8, sessions=(3,), cs=(0.5,), chunk=4))
    for a, b in zip(plain, full):
        assert (a.null_rejection, a.detectable, a.power_90, a.power) == (b.null_rejection, b.detectable, b.power_90,
                                                                         b.power)
    assert plain[0].confounded == () and len(full[0].confounded) == 2


def test_price_scores_have_unit_variance_their_persistence_and_a_common_share():
    scores = v2_power.noise_scores(np.random.default_rng(9), 40, 40, GEN)
    assert scores.shape == (40, 480, 8, 9)
    assert scores.var(axis=(0, 1, 2)) == pytest.approx(np.ones(9), abs=0.08)
    lag1 = [np.mean(scores[:, 1:, :, f] * scores[:, :-1, :, f]) for f in range(9)]
    assert lag1 == pytest.approx(np.linspace(0, 0.9, 9), abs=0.05)
    assert np.mean(scores[..., 0, 0] * scores[..., 1, 0]) == pytest.approx(0.45, abs=0.05)
    pred = np.random.default_rng(10).normal(size=(5000, 8))
    score = v2_power.momentum(pred, np.random.default_rng(11).normal(size=pred.shape), 0.5, 1.0)
    assert np.corrcoef(score.ravel(), pred.ravel())[0, 1] == pytest.approx(0.5, abs=0.02)
    assert v2_power.with_momentum(scores[0], scores[0, ..., 0]).shape == (480, 8, 10)


def test_an_impact_halves_every_120_minutes_and_drops_after_600():
    values = one([0], [0.0], [1.0])
    w = 2.0 ** (-30 * np.arange(30) / 120)
    assert values[0] == 0.0  # not yet received at its own block's start
    assert values[4] == pytest.approx(1 / 3)  # 120 minutes old, weight one half
    assert values[1:21] == pytest.approx(w[1:21] / (w[1:21] + 1))
    assert values[20] > 0 and (values[21:] == 0).all()  # 600 minutes old is kept, older is dropped


def test_an_impact_inside_a_block_counts_from_its_own_minute_and_neutral_ones_dilute():
    w = 2.0 ** (-15 / 120)  # arrived 15 minutes into block 0, so 15 minutes old at block 1
    assert one([0], [15.0], [-0.6])[1] == pytest.approx(-0.6 * w / (w + 1))
    assert one([0, 0], [15.0, 15.0], [1.0, 0.0])[1] == pytest.approx(w / (2 * w + 1))
    assert one([0, 0], [0.0, 0.0], [1.0, -1.0])[1] == 0.0


def test_an_impact_from_outside_the_session_keeps_full_weight_until_the_open():
    values = one([13], [0.0], [1.0])  # placed at the open of session 2, offset 0
    assert (values[:14] == 0).all()
    assert values[v2_power.eval_blocks(2)[12]] == pytest.approx(2 ** -0.25 / (2 ** -0.25 + 1))  # minute 30
    shape = (2, 3, 4 * v2_power.BLOCKS_PER_SESSION)
    cell, offset, strength = v2_power.impacts(np.random.default_rng(1), shape, Generator(impacts_per_session=50))
    at_open = offset == 0
    assert (cell[at_open] % v2_power.BLOCKS_PER_SESSION == 0).all()
    assert at_open.mean() == pytest.approx(0.6, abs=0.05)
    # prompt_v2's strengths, every one score_store.strength of a confidence, horizon and surprise.
    possible = {round(abs(score_store.strength(1, c, h, s)), 12) for c in range(1, 6)
                for h in score_store.HORIZON_WEIGHTS for s in range(4)}
    assert set(np.round(np.abs(strength[strength != 0]), 12)) <= possible and np.abs(strength).max() <= 1


def test_the_strength_draw_is_score_store_strength_at_the_planning_shares():
    assert v2_power.HORIZON_WEIGHTS == score_store.HORIZON_WEIGHTS
    assert sum(v2_power.HORIZON_SHARES.values()) == pytest.approx(1) and sum(v2_power.SURPRISE_SHARES) == 1
    direction = np.repeat([-1.0, 0.0, 1.0], 20_000)
    drawn = v2_power.draw_strengths(np.random.default_rng(3), direction)
    rng = np.random.default_rng(3)
    n = len(direction)
    confidence = rng.integers(1, 6, n)
    horizon = np.array(list(v2_power.HORIZON_SHARES))[rng.choice(3, n, p=list(v2_power.HORIZON_SHARES.values()))]
    surprise = rng.choice(4, n, p=v2_power.SURPRISE_SHARES)
    expected = [score_store.strength(int(d), int(c), str(h), int(s))
                for d, c, h, s in zip(direction, confidence, horizon, surprise)]
    assert drawn.tolist() == pytest.approx(expected, abs=1e-15)
    assert (drawn[direction == 0] == 0).all() and (np.sign(drawn) == direction).all()
    # An hours item at surprise 1 and confidence 5 keeps 0.6 of version 1's strength, and the mean factor is below it.
    assert score_store.strength(1, 5, "hours", 1) == pytest.approx(0.6)
    assert 0.3 < np.mean(np.abs(drawn[direction != 0])) / 0.6 < 0.6


def test_evaluation_minutes_are_30_to_360_of_each_session():
    assert v2_power.eval_blocks(2).tolist() == [*range(1, 13), *range(14, 26)]


def test_inputs_are_bounded_and_zero_inflated_and_predictions_correlate_at_c():
    rng = np.random.default_rng(2)
    inputs = v2_power.input_values(rng, 40, 9, 20, GEN)
    assert inputs.shape == (40, 9, 240) and np.abs(inputs).max() < 1
    assert (inputs == 0).mean() > 0.02
    pred = v2_power.predictions(inputs, 0.5).reshape(-1, 8)
    corr = np.corrcoef(pred.T)
    assert corr[~np.eye(8, dtype=bool)].mean() == pytest.approx(0.5, abs=0.04)
    same = v2_power.predictions(inputs, 1.0)
    assert (same == same[..., :1]).all()


def test_labels_have_the_planning_spread_and_market_factor():
    rng = np.random.default_rng(3)
    noise = v2_power.label_noise(rng, 50, 100, GEN)
    labels = v2_power.planted(np.zeros_like(noise), noise, 0.0, 1.0).reshape(-1, 8)
    assert labels.std(axis=0) == pytest.approx(v2_power.SD_BPS, rel=0.05)
    corr = np.corrcoef(labels.T)
    assert corr[0, 1] == pytest.approx(0.45, abs=0.04) and corr[0, 7] == pytest.approx(math.sqrt(0.405), abs=0.04)
    z = noise[..., 0]
    returns = np.mean([np.corrcoef(row[1:], row[:-1])[0, 1] for row in z])
    assert abs(returns) < 0.02  # no predictable return


def test_shocks_have_unit_variance_fat_tails_and_clustered_volatility():
    def lag1_of_size(gen):
        z = v2_power.shocks(np.random.default_rng(5), (50, 1200, 1), gen)[..., 0]
        return z, np.mean([np.corrcoef(np.abs(row[1:]), np.abs(row[:-1]))[0, 1] for row in z])

    z, clustered = lag1_of_size(GEN)
    _, flat = lag1_of_size(Generator(vol_log_sd=0.0))
    assert (z ** 2).mean() == pytest.approx(1.0, abs=0.05)
    assert ((z - z.mean()) ** 4).mean() / z.var() ** 2 > 6  # a normal shock gives 3
    assert clustered > 0.08 and abs(flat) < 0.02


def test_a_planted_correlation_is_recovered_on_average():
    rng = np.random.default_rng(4)
    inputs = v2_power.input_values(rng, 30, 9, 40, GEN)
    noise = v2_power.label_noise(rng, 30, 40, GEN)
    pred = v2_power.predictions(inputs, 0.5)
    scale = float(inputs.std())
    corr = [v2_power.matrix_skill(p, v2_power.planted(p, e, 0.2, scale)).corr for p, e in zip(pred, noise)]
    assert np.mean(corr) == pytest.approx(0.2, abs=0.02)


def test_the_same_seed_gives_the_same_scenarios():
    settings = Settings(runs=12, sessions=(3, 5), cs=(0.5,), chunk=5)
    assert v2_power.simulate(settings) == v2_power.simulate(settings)


def test_main_prints_counts_verdicts_a_timing_and_synthetic_rows(capsys):
    assert v2_power.main(["--runs", "6", "--sessions", "3", "4", "--c", "0.5"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == ["synthetic power", "runs: 6", "symbols: 8", "scenarios: 2"]
    assert [line.split(": ")[0] for line in lines[4:7]] == [
        "null rejection within the level", "incremental null rejection within the level",
        "confounded sequence within the level"]
    assert all(line.split(": ")[1] in ("pass", "fail") for line in lines[4:7])
    assert lines[7].startswith("run time: ") and lines[7].endswith(" s")
    assert lines[8].split()[:4] == ["sessions", "rows", "per", "symbol"] and "h2 null rejection" in lines[8]
    assert [line.split()[:3] for line in lines[9:11]] == [["3", "24", "0.50"], ["4", "36", "0.50"]]
    assert lines[11:13] == ["synthetic confounds", "confounds: 2"] and lines[13].split()[:4] == [
        "sessions", "c", "b", "rho"]
    assert [line.split()[:4] for line in lines[14:]] == [["3", "0.50", "0.05", "0.50"], ["3", "0.50", "0.10", "0.70"],
                                                         ["4", "0.50", "0.05", "0.50"], ["4", "0.50", "0.10", "0.70"]]
    assert "hidden" not in "\n".join(lines)


@pytest.mark.parametrize("args", [["--sessions", "1"], ["--c", "1.5"], ["--runs", "0"]])
def test_main_refuses_settings_outside_their_range(args):
    with pytest.raises(SystemExit):
        v2_power.main(args)


def test_the_script_reads_nothing_and_imports_no_feed_news_or_store_module():
    source = (ROOT / "scripts" / "v2_power.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            raise AssertionError("the power script opens a file")
    assert imported == {"__future__", "argparse", "math", "time", "dataclasses", "numpy", "btc_survivor.cli.stats",
                        "btc_survivor.v2", "btc_survivor.v2.stats"}
