"""predict_rows on seeded networks: each row against MLP.predict, one pass over the 8 symbols, the news part skipped
while every news input is zero so S1 and S2 outputs are bitwise equal, and networks born at 10 inputs with two zero
news columns appended."""

import numpy as np
import pytest

from btc_survivor.model import MLP
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks.learning import N_NEWS_INPUTS, network_inputs, predict_rows, with_news_inputs
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s2_news.toml")
HIDDEN = CFG.model.hidden_units


def born(seed):
    """A learner as both runs build it, drawn at 10 inputs with the news columns appended."""
    return with_news_inputs(MLP.random(10, HIDDEN, np.random.default_rng(seed), CFG.model.init_output_scale))


def rows(seed, news=True):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(size=(8, 10)), rng.uniform(-1, 1, (8, 2)) if news else np.zeros((8, 2))], axis=1)


def test_a_learner_has_12_inputs_and_225_parameters():
    model = born(1)
    assert (model.n_in, model.hidden, model.size) == (CFG.model.inputs, 16, 225) == (12, 16, 225)
    assert N_NEWS_INPUTS == 2 and not model.W1[:, 10:].any()


def test_born_networks_predict_as_their_10_input_parents():
    parent = MLP.random(10, HIDDEN, np.random.default_rng(2), 0.1)
    child = with_news_inputs(parent)
    assert np.array_equal(child.W1[:, :10], parent.W1) and np.array_equal(child.b1, parent.b1)
    assert np.array_equal(child.W2, parent.W2) and np.array_equal(child.b2, parent.b2)
    X = rows(3, news=False)
    assert np.allclose(predict_rows(child, X), [parent.predict(r[:10]) for r in X], rtol=1e-13, atol=1e-15)
    # The parent's parameters are copied, never shared.
    child.W1[0, 0] += 1.0
    assert child.W1[0, 0] != parent.W1[0, 0]


@pytest.mark.parametrize("news", [False, True])
def test_each_row_matches_mlp_predict(news):
    model = born(4)
    model.W1[:, 10:] = np.random.default_rng(5).normal(size=(HIDDEN, 2))
    X = rows(6, news)
    out = predict_rows(model, X)
    assert out.shape == (8,)
    assert np.allclose(out, [model.predict(r) for r in X], rtol=1e-13, atol=1e-15)


def test_s1_and_s2_are_bitwise_equal_while_news_inputs_are_zero():
    s1, s2 = born(7), born(7)
    assert np.array_equal(s1.params, s2.params)
    s2.W1[:, 10:] = np.random.default_rng(8).normal(size=(HIDDEN, 2))   # S2 has learned news weights
    X = rows(9, news=False)
    assert np.array_equal(predict_rows(s1, X), predict_rows(s2, X))
    # One nonzero news input on any row brings the news part in, for that pass only.
    X[3, 11] = 0.5
    assert not np.array_equal(predict_rows(s1, X), predict_rows(s2, X))
    # S1's news weights are zero, so even a nonzero news input would leave its outputs unchanged.
    assert np.array_equal(predict_rows(s1, X), predict_rows(s1, rows(9, news=False)))


def test_zero_news_inputs_give_the_news_weights_zero_gradient():
    model = born(10)
    X, y = rows(11, news=False), np.random.default_rng(12).normal(size=8)
    _, grad = model.loss_and_grad(X, y, CFG.model.huber_delta)
    W1_grad = grad[:model.n_in * HIDDEN].reshape(HIDDEN, model.n_in)
    assert not W1_grad[:, 10:].any() and W1_grad[:, :10].any()


def test_network_inputs_put_the_news_inputs_after_the_price_z_scores():
    z = np.arange(10.0)
    row = network_inputs(z, np.array([0.25, -0.5]))
    assert row.tolist() == list(range(10)) + [0.25, -0.5]
