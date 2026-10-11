import numpy as np
import pytest

from btc_survivor.labels import Matured, MaturedWindow
from btc_survivor.model import MLP, AdamW
from btc_survivor.standardizer import LabelScale


def test_parameter_count_matches_the_brief():
    assert MLP(11, 16).size == 209


def test_views_share_the_flat_parameter_array():
    m = MLP(3, 2)
    m.W1[0, 1] = 5.0
    m.b2[0] = 7.0
    assert m.params[1] == 5.0 and m.params[-1] == 7.0


def test_random_init_is_small_varied_and_seeded():
    a = MLP.random(11, 16, np.random.default_rng(1), 0.1)
    b = MLP.random(11, 16, np.random.default_rng(1), 0.1)
    c = MLP.random(11, 16, np.random.default_rng(2), 0.1)
    assert np.array_equal(a.params, b.params) and not np.array_equal(a.params, c.params)
    assert np.all(a.b1 == 0) and a.b2[0] == 0
    assert np.abs(a.W1).max() <= np.sqrt(6 / 27) and np.abs(a.W2).max() <= 0.1 * np.sqrt(6 / 17)
    preds = [a.predict(x) for x in np.random.default_rng(3).normal(size=(50, 11))]
    assert 0 < np.std(preds) < 0.5 and len(set(np.round(preds, 12))) == 50


@pytest.mark.parametrize("delta", [0.3, 1.0, 100.0])
def test_gradient_matches_finite_differences(delta):
    rng = np.random.default_rng(42)
    model = MLP.random(11, 16, rng, 1.0)
    model.b1[:] = rng.normal(size=16) * 0.1
    model.b2[0] = 0.2
    X, y = rng.normal(size=(8, 11)), rng.normal(size=8) * 2
    _, grad = model.loss_and_grad(X, y, delta)
    numeric = np.zeros_like(grad)
    h = 1e-6
    for i in range(model.size):
        original = model.params[i]
        model.params[i] = original + h
        up, _ = model.loss_and_grad(X, y, delta)
        model.params[i] = original - h
        down, _ = model.loss_and_grad(X, y, delta)
        model.params[i] = original
        numeric[i] = (up - down) / (2 * h)
    assert np.allclose(grad, numeric, rtol=1e-6, atol=1e-9)


def test_huber_loss_values():
    model = MLP(1, 1)  # all zero, so it predicts 0
    loss, _ = model.loss_and_grad(np.zeros((2, 1)), np.array([0.5, 3.0]), 1.0)
    assert loss == pytest.approx((0.5 * 0.25 + 1.0 * (3.0 - 0.5)) / 2)


def test_one_adamw_step_hand_computed():
    start = np.array([0.5, 0.3, -0.4, 0.2])  # W1, b1, W2, b2, with nonzero biases
    model = MLP(1, 1, params=start)
    opt = AdamW(model.size, 0.9, 0.999)
    grad = np.array([0.2, -0.1, 0.3, 0.0])
    opt.step(model, grad, lr=0.01, weight_decay=0.1, grad_clip=10.0)
    adam = 0.01 * grad / (np.abs(grad) + 1e-8)  # the first bias corrected step is lr times the sign
    expected = start - adam - 0.01 * 0.1 * start * np.array([1.0, 0.0, 1.0, 0.0])  # biases are not decayed
    assert np.allclose(model.params, expected)


def test_gradient_is_clipped_to_the_norm_limit():
    model = MLP(1, 1, params=np.zeros(4))
    opt = AdamW(4, 0.9, 0.999)
    opt.step(model, np.array([30.0, 40.0, 0.0, 0.0]), lr=0.1, weight_decay=0.0, grad_clip=5.0)
    assert np.allclose(opt.m[:2], 0.1 * np.array([3.0, 4.0]))


def _online_run(pattern, seed, steps=4000, test=2000):
    """Train online the way bots do, with a shared window and label scale, then score held out samples."""
    rng = np.random.default_rng(seed)
    model = MLP.random(11, 16, rng, 0.1)
    opt = AdamW(model.size, 0.9, 0.999)
    window = MaturedWindow(240, 11)
    scale = LabelScale(168 * 60, 30)

    def sample():
        x = rng.normal(size=11)
        noise = rng.normal() * 10.0
        return x, (pattern(x) + noise if pattern else noise)

    for minute in range(steps):
        x, label = sample()
        window.add(Matured(minute, x, label))
        scale.update(label)
        if not scale.ready:
            continue
        X, y = window.batch(rng, 8)
        _, grad = model.loss_and_grad(X, y / scale.scale, 1.0)
        opt.step(model, grad, 0.003, 1e-4, 5.0)
    xs, ys = zip(*(sample() for _ in range(test)))
    preds = np.array([model.predict(x) * scale.scale for x in xs])
    ys = np.array(ys)
    return float(np.mean((preds - ys) ** 2)), float(np.mean(ys ** 2)), float(np.corrcoef(preds, ys)[0, 1])


def test_online_learner_finds_a_planted_pattern():
    mse, baseline, corr = _online_run(lambda x: 12.0 * x[0] - 10.0 * np.tanh(x[1]), seed=7)
    assert mse < 0.6 * baseline
    assert corr > 0.6


def test_online_learner_finds_nothing_in_pure_noise_beyond_chance():
    for seed in (1, 2, 3):
        mse, baseline, corr = _online_run(None, seed=seed)
        assert abs(corr) < 3 / np.sqrt(2000)
        assert mse < 1.2 * baseline  # fitting noise would push held out error well above predicting zero
