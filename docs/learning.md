# How the bots learn

Every bot is a small neural network that predicts the Bitcoin return over the next horizon, 30 minutes in runs A, B, C and E and 4 hours in run D. It learns online, one minute at a time, from labels that have already happened, and every prediction is scored before its own label is allowed to train anything. This page covers the inputs, the labels, the network and its optimizer, how the learning rate was chosen, and how skill is measured. The full design is in [section 8 of the plan](../PLAN.md#8-learning), and the trading side is in [the execution model](execution_model.md).

## The 11 features

`features.py` computes 11 numbers once per healthy minute, at the UTC minute boundary, from the shared minute series and the order book as it stands at that boundary. The same vector is used by every bot.

| Number | Feature | How it is computed |
| --- | --- | --- |
| 1 to 4 | Returns over 5, 15, 30 and 60 minutes | `1e4 * ln(mid_t / mid_{t-k})` in bps, from the book mid at each boundary |
| 5 and 6 | Realized volatility over 30 and 120 minutes | the standard deviation of one minute log returns in bps, using only pairs of consecutive valid minutes |
| 7 | Volume ratio | `ln((V15 + eps) / (V360 / 24 + eps))`, clipped to plus or minus 3, a fixed constant |
| 8 | Imbalance | `(bids - asks) / (bids + asks)` over the sizes at the top 10 levels |
| 9 | Spread | the spread in bps |
| 10 and 11 | Time of day | sine and cosine of the UTC minute of the day |

A vector is valid only once the series holds `history_minutes_required = 360` minutes, six hours, of healthy data. After a later gap a coverage rule applies instead of a fresh six hour wait. A return needs a valid mid at exactly its start minute, a volatility needs at least 80 percent of its window's one minute returns, and the volume terms need at least 80 percent of their window, set by `min_window_coverage = 0.8`. A minute without a valid vector is not used for prediction or training. Any value that is not finite also makes the minute invalid.

Two features are weaker than they look. On Coinbase the spread is usually one tick, one cent or about 0.0012 bps, so the spread feature mostly tracks slow drift in the price level. Imbalance counts only the top 10 price levels, which on a one cent book cover a very small slice of price, and a depth based version is listed as later work in [evolution.md](evolution.md#later-additions-that-are-not-built). The code is in [features.py](../src/btc_survivor/features.py).

## The standardizer

The network never sees raw features. `standardizer.py` turns each one into a z score using an exponentially weighted mean and variance with a half life of `standardizer_half_life_hours = 168`, one week of healthy minutes. Each new sample gets the weight `max(alpha, 1/n)`, where `n` is the sample count. Early on that is a plain running mean and variance, and only after about 14,500 samples, roughly ten days, does the exponential weighting take over. Without this rule the first weeks would be badly biased toward the zero starting values.

A feature outputs zero until it has `min_samples = 30` samples, and after that a z score clipped to plus or minus `clip_z = 5`. The standard deviation is floored at a tiny constant, so a feature that never changes gives a z score of zero rather than a division by zero. The vector for a minute is standardized with the statistics from before that minute, and only then are the statistics updated, so a sample never shapes its own z score. The standardizer is shared by every bot and computed once per minute. The code is in [standardizer.py](../src/btc_survivor/standardizer.py).

## Labels and the no lookahead rule

The label for minute `t` is the log return of the mid from `t` to `t` plus the horizon, in bps, written `1e4 * ln(mid_{t+h} / mid_t)`. At minute `t` the bot only predicts it. The z vector and the entry mid go into a shared pending queue together with the feed epoch, and the record matures exactly one horizon later, but only if the epoch has not changed. Every exit from a healthy feed raises the epoch, so a label never spans a gap, and every restart counts as a gap and discards the whole pending queue. After each gap no sample matures for one horizon, so no bot trains for that long.

The no lookahead rule is enforced in several places, most of them with their own test.

* Candles are built by receive time, not exchange time, so a trade that arrived after a boundary belongs to the next minute because at the boundary it was not yet known. A test in [tests/test_market.py](../tests/test_market.py) feeds a trade whose exchange time is before a boundary but which arrived after it, and requires it to count toward the next minute.
* A feature at `t` reads only data up to `t`. A test in [tests/test_features.py](../tests/test_features.py) computes features on a full series and on the same series cut at each `t`, including series with gaps inside windows, and requires identical vectors.
* A z score uses statistics from before its own sample, as [tests/test_standardizer.py](../tests/test_standardizer.py) checks.
* A label for `t` is used only after `t` plus the horizon has passed. The label scale, the shared matured window that bots draw training batches from, and the linear baseline are all updated only with matured labels. [tests/test_labels.py](../tests/test_labels.py) checks maturity, the epoch discards and the matured window, and [tests/test_engine.py](../tests/test_engine.py) checks that the linear baseline is updated exactly as often as the label scale, each time with the z vector of the minute whose label matured.
* Each prediction is scored against its label before that label trains anything, as [tests/test_bot.py](../tests/test_bot.py) checks.
* Learner settings were fixed before launch on synthetic data only, never on market data. This is a rule of process that no test can enforce. The check that fixed them, [scripts/learner_check.py](../scripts/learner_check.py), builds its own synthetic series and never reads market data, and [tests/test_learner_check.py](../tests/test_learner_check.py) exercises it.

The label scale is the running standard deviation of matured labels, with the same weighting, floor and half life as the standardizer. Training starts once it holds 30 labels. On a fresh launch that means features after about 6 hours, and training after about 7 hours in runs A, B, C and E and after about 10.5 hours in run D, whose first label matures 4 hours after the features become valid. The code is in [labels.py](../src/btc_survivor/labels.py).

## Overlapping labels and few independent outcomes

A new sample matures every healthy minute, but at a 30 minute horizon two consecutive labels share 29 of their 30 one minute moves. The samples are therefore far less independent than their count suggests. As a rule of thumb, one horizon of minutes holds about one independent outcome.

| Window | Samples | Independent outcomes at 30 minutes | Independent outcomes at 4 hours |
| --- | --- | --- | --- |
| Matured window that training batches are drawn from | 240 | about 8 | about 1 |
| Rolling skill window of each bot | 360 | about 12 | about 1.5 |
| One day of healthy data | 1,440 | 48 | 6 |
| 28 days of healthy data | 40,320 | about 1,340 | about 168 |

This shapes everything that follows. A bot's rolling skill statistics rest on about 12 outcomes, so they are noisy by nature, and in run D they mean almost nothing. Among 20 bots chance alone produces one whose numbers look impressive, so the best single bot is never evidence. The confirmatory test in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) therefore uses one minute per horizon window, so its labels never overlap.

## The model

`model.py` holds a network with one hidden layer, 11 inputs, 16 tanh hidden units and one linear output. That is 209 float64 parameters in one flat array, with views for the two weight matrices and two bias vectors. The forward and backward passes are written by hand in numpy, and a central difference gradient check in [tests/test_model.py](../tests/test_model.py) must agree to one part in a million.

Weights start with Xavier uniform initialization drawn from the bot's own seeded generator, biases start at zero, and the output weights are multiplied by `init_output_scale = 0.1`, so a fresh bot predicts only a few bps. The network learns the label divided by the label scale, and a prediction in bps is the output times the label scale. The loss is Huber with `huber_delta = 1.0`, so an error larger than one label standard deviation counts linearly instead of quadratically, and the gradient norm is clipped to `grad_clip = 5.0`.

A bot takes one training step only in a minute in which a new sample matured. The batch is that newest sample plus 7 draws with replacement from the shared matured window of the last 240 samples, chosen with the bot's own generator, for `batch_size = 8`. A random bot stays in Warmup for 6 hours of training minutes, and a mutated child for 1 hour, so an outage cannot end a warmup in which nothing was learned. The code is in [model.py](../src/btc_survivor/model.py) and the bot loop is in [bot.py](../src/btc_survivor/bot.py).

## Learning does not depend on trading

A learner trains whether or not it trades. In every healthy minute, any learner that is not dead and has a valid feature vector, a ready label scale and a newly matured sample takes its training step, whether it is in Warmup or Active, flat or holding a position, and whether or not an order is in flight. The label is the market's log return of the mid over the horizon, the same for every bot, and never the bot's own trade profit. The cost estimate reads the live book and never the bot's own fills. A trade therefore adds no training data, and the forced trades exist so that costs, the twin comparison, deaths and parent eligibility can be measured, not so that the bots learn.

The one link from trading to learning is the generator. Every order a learner sends, whether a decision, a stop or a death close, draws its latency jitter from the learner's own seeded generator, which also draws its training batches, so each order shifts which past samples later batches draw. The external review ran two identical run A learners over the same 3,000 minutes, one forced to trade every 12 hours, which made 5 opens and paid 45 dollars of fees, and one that never traded. Their weights differed by at most 0.0024, and with the jitter moved to a generator of its own they were identical bit for bit. Beyond those draws, trading reaches the weights only through selection in run C, where a bot's trades decide when it dies and which survivor's weights a child copies, as [evolution.md](evolution.md#why-fees-decide-survival-at-retail-costs) explains. The training step is `_learn` in [bot.py](../src/btc_survivor/bot.py), and the jitter is drawn in [broker.py](../src/btc_survivor/broker.py).

## AdamW

The optimizer is Adam with `adam_beta1 = 0.9`, `adam_beta2 = 0.999`, an epsilon of `1e-8` and the usual bias correction, plus decoupled weight decay in the AdamW form. Decay applies to the two weight matrices and not to the biases. Each bot has its own optimizer state, and a mutated child starts with fresh state.

```
w -= lr * (m_hat / (sqrt(v_hat) + 1e-8)) + lr * weight_decay * w     # weight matrices
b -= lr * (m_hat / (sqrt(v_hat) + 1e-8))                              # biases
```

With `weight_decay = 0.0001` and the default learning rate, decay shrinks an unused weight by only about 0.01 percent a week, so it is a weak regulariser. The decay rate is a config value, while the learning rate is a gene that mutation can change, as [evolution.md](evolution.md#genes) explains. The methods are described in [Adam](https://arxiv.org/abs/1412.6980) and [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).

## The learning rate of 0.0001

The default learning rate is 0.0001, ten times lower than the brief's 0.001, with gene bounds moved down by the same factor to 0.00001 and 0.001 so that mutation can still move both ways. The value was fixed on synthetic data only, before launch, and never on market data.

[scripts/learner_check.py](../scripts/learner_check.py) is the check that fixed it. It builds synthetic z features as autoregressive random series, plants a linear signal of known correlation in one of them, builds overlapping labels from one minute moves of 3.5 bps the same way the live labels overlap, and then runs the bots' own learning code minute by minute exactly as the engine does. It never reads market data.

| Setting | Spread of one bot's predictions with nothing to predict | Squared error against predicting zero |
| --- | --- | --- |
| learning rate 0.001, the brief's value | about 9 bps | about 16 percent worse |
| learning rate 0.0001, the default | about 3 bps | within about 1 percent |

Over 28 synthetic days a population of three bots at 0.0001 recovered about two thirds of a planted correlation of 0.1 and about half of one of 0.05. The lower rate has two costs. Adaptation is slower, so a signal that flips sign every week is followed about as well as before, but one that flips every two days is mostly missed. And predictions stay a few bps, so even run B's required edge of about 20 bps will almost never be met, which means nearly every trade in runs A and B is a forced one. These commands reproduce the comparison.

```bash
.venv/bin/python scripts/learner_check.py --config configs/run_a.toml --bots 3
.venv/bin/python scripts/learner_check.py --config configs/run_a.toml --bots 3 --lr 0.001
.venv/bin/python scripts/learner_check.py --config configs/run_a.toml --bots 3 --flip-days 2
```

The reasoning is recorded in [section 1](../PLAN.md#1-key-decisions) and [section 8](../PLAN.md#8-learning) of the plan.

## Run D and its learning rate of 0.00003

Run D is run B with a 4 hour horizon. It was added on 11 October 2026, about an hour after runs A, B and C started, to see trades that the gate lets through. At a 5 bps fee the required edge is about 20 bps, and over 4 hours a prediction has more room to clear it. Its settings were fixed before its own launch with the same synthetic check, run at the 4 hour horizon.

| Setting at a 4 hour horizon | Spread with nothing to predict | Squared error against predicting zero |
| --- | --- | --- |
| learning rate 0.0001 | about 17 bps | about 9 percent worse |
| learning rate 0.00003, run D | about 6 bps | within about 2 percent |

At 0.0001 the predictions were wide enough to pass the 20 bps gate on noise alone, so run D uses `lr_init = 0.00003` with gene bounds of 0.000003 and 0.0003. A longer replay window did not help. Run D is exploratory and not part of the success criteria. Each bot decides once every 4 hours, all 20 decide within the first 20 minutes of each 4 hour block, and with about 6 independent windows a day its skill test at day 28 can only detect a correlation of about 0.28, so its numbers are descriptive.

```bash
.venv/bin/python scripts/learner_check.py --config configs/run_d.toml --bots 3
.venv/bin/python scripts/learner_check.py --config configs/run_d.toml --bots 3 --lr 0.0001
```

Its config is [configs/run_d.toml](../configs/run_d.toml), and the reasoning is in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria).

## Prequential skill

Skill is measured prequentially, which means each prediction is scored against an outcome the model had not seen when it made the prediction. Every bot keeps its prediction in bps for each pending minute. When that minute's label matures, `skill.py` records the original prediction against the realized label first, and only then does the bot train on the sample. The score is therefore out of sample by construction, with no held out set and no lookahead. A bot born after a record was made has no prediction for it and skips that pair.

Each bot keeps ring buffers of its last `skill_window_samples = 360` pairs and reports three numbers, the correlation between prediction and label once it has at least 30 pairs, the directional accuracy over pairs where neither side is zero, and the mean absolute error. They are written into every equity snapshot, and the rolling correlation is shown by `status` and `explain`. With `gate_on_skill = true` a bot would also need a full window and a correlation above `skill_gate_min_corr` before the gate passes, but every run leaves that switch off. As the table above shows, 360 pairs at a 30 minute horizon hold only about 12 independent outcomes. The code is in [skill.py](../src/btc_survivor/skill.py).

## The forecasts table

The headline result is not any single bot but the `forecasts` table. Every healthy minute with a valid feature vector and a ready label scale writes one row per predictor, and the row receives its label when that label matures.

| Predictor | What it predicts |
| --- | --- |
| `population` | the mean prediction in bps of the living learners, with the coin flip twins left out because they do not predict |
| `linear` | an online ridge regression on the same z scores, fitted only to matured labels with the standardizer's weighting and half life, and a prior worth 1,000 samples of zero so early fits stay near zero |

Predicting zero needs no row, because its error is just the label. The linear baseline shows whether 209 parameters add anything over a one line rule. A later predictor, for example one that reads the news, adds rows under its own name and is scored by exactly the same rules.

```sql
CREATE TABLE forecasts (
  ts TEXT NOT NULL,                -- the minute the forecast is for
  predictor TEXT NOT NULL,         -- population, linear, or a later predictor
  prediction_bps REAL NOT NULL,
  label_bps REAL,                  -- filled when the label matures
  PRIMARY KEY (ts, predictor)
);
```

The `evaluate` command scores these rows for the success criteria. It keeps one minute per horizon window, the minutes whose index is a multiple of the horizon, so labels never overlap, and it skips the first day after the first forecast row so the startup noise of a fresh model is never counted. Criterion 1 passes when the correlation between forecast and label is above zero at the one sided 1 percent level, with a Newey West standard error over 12 windows, about six hours in runs A, B, C and E and about 48 hours in run D. The same rows report the squared error against predicting zero and against the linear baseline, each with a Diebold Mariano test. At a 30 minute horizon, with about 1,280 scored windows by day 28, the test can detect a correlation of about 0.1 with 90 percent power if the windows are independent. `evaluate` prints the realized window count, the Newey West standard error, a one sided 95 percent upper bound on the correlation, which is the correlation plus 1.645 of those errors, and two detectable correlations, 3.61 over the square root of the window count and 3.61 times the realized Newey West error, which is the honest figure and usually the larger one when volatility clusters in both forecast and label. The pass rule stays a Newey West t above 2.33. A test that fails rules out only correlations above the upper bound, and it is never read as evidence that the correlation is below about 0.1, as [honesty_and_limits.md](honesty_and_limits.md#statistical-limits) explains. With no skill the squared error is expected to come out a few percent worse than predicting zero. The dashboard shows the same statistics on the same rows as they build up, marked descriptive until the final day. The table grows by about 2,900 rows a day. The code is in [baselines.py](../src/btc_survivor/baselines.py) and the criteria are in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria).

## Replay is for plumbing only

Replay results are never used to choose, tune, or rank strategies.

The replay harness runs recorded feed messages through the same parsing code and the same engine as live, at full speed, so that health, fills, persistence and determinism can be tested. It always starts from a fresh population, never loads a checkpoint, and writes only into a new or empty folder that is not the configured `data_dir`. Every setting that shapes learning was fixed from reasoning or from the synthetic check above, never from a replay. The code is in [replay.py](../src/btc_survivor/replay.py) and the harness is described in [section 15 of the plan](../PLAN.md#15-replay-harness-and-determinism).
