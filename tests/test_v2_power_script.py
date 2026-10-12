"""scripts/v2_power.py, the synthetic power of section 11.4: its generator by hand and in aggregate, and the
detectable correlations of the plan's table reproduced with fixed fast settings."""

import ast
import math
import sys

import numpy as np
import pytest

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
    assert set(np.round(np.abs(strength) * 5).astype(int)) == {0, 1, 2, 3, 4, 5}


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


def test_main_prints_counts_a_verdict_a_timing_and_synthetic_rows(capsys):
    assert v2_power.main(["--runs", "6", "--sessions", "3", "4", "--c", "0.5"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == ["synthetic power", "runs: 6", "symbols: 8", "scenarios: 2"]
    assert lines[4].split(": ")[0] == "null rejection within the level" and lines[4].split(": ")[1] in ("pass", "fail")
    assert lines[5].startswith("run time: ") and lines[5].endswith(" s")
    assert lines[6].split()[:4] == ["sessions", "rows", "per", "symbol"]
    assert [line.split()[:3] for line in lines[7:]] == [["3", "24", "0.50"], ["4", "36", "0.50"]]
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
