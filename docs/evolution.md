# How evolution works

Each run keeps 20 learner slots full. A bot leaves only by dying, and a dead learner is replaced in the same minute, either by a random newcomer or by a mutated copy of a fit survivor. Beside every learner sits a coin flip twin that copies its timing with a random direction, so luck can be told apart from skill. At retail costs the fees decide who survives long before skill can, and this page shows why with the numbers. Within their windows only run C sees deaths, and there selection is mostly noise from the costs of forced trades, so version 1 checks that this machinery works and does not test whether evolution helps. The design is in [section 11 of the plan](../PLAN.md#11-evolution), the learning side is in [learning.md](learning.md), and the costs are worked through in [the execution model](execution_model.md).

## Death and replacement

A bot dies when its equity falls below the cost of one more round trip at the current estimate plus one day of rent. At retail fees that threshold is about 10.26 dollars, a 9.10 dollar round trip plus 1.16 dollars of rent, as the [death threshold](execution_model.md#death-threshold) section of the execution model works out. The dead bot leaves its slot at once, any open position is closed through the broker, and the cause of death is the largest of five parts, fees, funding, rent, stop loss losses and other trading losses, all of which are stored.

The replacement takes the same slot in the same minute. The population generator first draws whether it is a newcomer, with probability `newcomer_fraction = 0.10`. Otherwise a parent is chosen by tournament among the eligible bots. A bot is eligible when it is alive, has lived at least `parent_min_age_hours = 24` hours of healthy minutes, and has closed at least `parent_min_trades = 20` trades. If no bot is eligible, the replacement is a newcomer. A quiet bot closes about 2 trades a day, so it needs about 10 days to become eligible. Every replacement starts with a fresh account at full start equity, so after deaths the mean equity of the living learners is topped up with new money, and the self financing value described in [honesty_and_limits.md](honesty_and_limits.md#population-averages-and-the-self-financing-value) is the figure to set against buy and hold. The code is in [population.py](../src/btc_survivor/population.py), and the death rule is in [section 10 of the plan](../PLAN.md#10-execution-and-accounting).

## Fitness

Each bot records its equity every 60 healthy minutes. Fitness looks at the hourly changes over the last `fitness_window_hours = 168` samples, one week, or over the bot's whole life if that is shorter. Let `m` be the mean hourly change and `s` the standard deviation of the hourly changes, floored at `fitness_std_floor_usd = 1.0` dollar.

```
fitness = m / s            when m >= 0
fitness = m * s / floor    when m < 0
```

The brief's literal formula, the net change divided by `s`, has two flaws. Net change grows with age, so among losing bots it favours the youngest eligible ones. And dividing a loss by a larger spread makes it look smaller, so among losing bots it ranks the most volatile highest. Using the mean keeps bots of different ages comparable, and multiplying by the spread when the mean is negative means volatility always counts against a bot. Take two bots that both lose 0.80 dollars an hour, one with an hourly spread of 2 dollars and one with 4. Dividing gives scores of 0.4 and 0.2 below zero, so the volatile bot wins. This formula gives 1.6 and 3.2 below zero, so the steady bot wins. The idea follows the modified [Sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio) of Israelsen, and the code is in [population.py](../src/btc_survivor/population.py).

## Tournament

To choose a parent, the population generator samples the smaller of `tournament_size = 3` and the number of eligible bots, without replacement, from the eligible bots in slot order, and the fittest of the sample wins. Ties break on rolling skill correlation, although fitness is a continuous number, so exact ties will be rare. A single eligible bot is the parent. A small tournament keeps selection pressure moderate, so the fittest bot wins only the tournaments it is drawn into and does not parent every child. The method is standard [tournament selection](https://en.wikipedia.org/wiki/Tournament_selection), and every draw comes from the seeded population generator, as [section 15 of the plan](../PLAN.md#15-replay-harness-and-determinism) describes.

## Mutation

A mutated child gets its own generator, seeded from the master seed and its new bot id. That generator adds Gaussian noise with `mutation_sigma_weights = 0.05` to every one of the parent's 209 parameters, then multiplies each gene by `exp(N(0, mutation_sigma_genes))` with `mutation_sigma_genes = 0.20` and clamps it to its bounds. One standard deviation of that factor multiplies or divides a gene by about 1.22.

The child starts with fresh optimizer state, a new bot id, a fresh account at the run's start equity, the parent's generation plus one, and a warmup of `warmup_hours_inherited = 1` hour of training minutes instead of the 6 hours a random bot needs. Its forced trade timer starts at a random delay, as a newborn's does, so children do not trade in step. A noise level of 0.05 is small next to the first layer's starting weights, which lie within about plus or minus 0.47, but it is larger than the range the output weights start in, about plus or minus 0.06, so a child's first predictions can be wider than its parent's.

Every mutated birth writes a `lineage` row with the parent, generation, birth time, both noise levels, both gene factors, the parent's genes, the parent's fitness and the parent's skill correlation. A newcomer gets new random weights, the initial gene values and generation 0. The command `lineage BOT_ID` prints the chain of ancestors with their lifespans, equity and causes of death. The code is in [population.py](../src/btc_survivor/population.py) and [genes.py](../src/btc_survivor/genes.py).

## Genes

Each learner carries two genes. Everything else, such as the horizon, the weight decay and the network size, is fixed config shared by every bot.

| Gene | Meaning | Initial value | Bounds |
| --- | --- | --- | --- |
| `margin_bps` | the extra edge in bps a prediction must clear beyond the estimated round trip cost | 8, or 0 in run E | 2 to 60, or 0 to 60 in run E |
| `lr` | the AdamW learning rate | 0.0001, or 0.00003 in run D | 0.00001 to 0.001, or 0.000003 to 0.0003 in run D |

The required edge at a decision is the estimated round trip cost plus the margin gene. At retail fees the cost is about 182 bps, so the required edge is about 190 bps and the margin barely matters. At the 5 bps fee the cost is about 12 bps and the required edge about 20 bps, so the margin is a large part of it. In run E, which is hypothetical and has no fee, slippage or short charges, the cost is only the spread and the walk through the book, about 0.001 bps, and the margin starts at its floor of 0, where mutation keeps it because it multiplies the gene. A test in [tests/test_genes.py](../tests/test_genes.py) runs 100,000 mutations with an extreme noise level and checks that both genes always stay within their bounds. How the learning rate was chosen is in [learning.md](learning.md#the-learning-rate-of-00001).

## Coin flip twins

Every learner slot also holds a coin flip twin with its own account and no network. At launch the learners have ids 1 to 20 and their twins ids 21 to 40. Whenever the learner opens, closes or flips, the twin does the same at the same moment, choosing long or short with a fair coin from its own seeded generator. When the learner is closed out by its stop loss or by death, the twin closes too. A twin that still holds a position while its learner is flat closes it at the next healthy minute, and a twin closes any leftover position before copying the next open, so the pair never drifts out of step.

A twin pays the same fees, slippage, latency, funding and rent, and has its own stop loss and death check. It does nothing when its learner holds or does nothing. Twins are never parents and never count toward fitness, and a dead twin is replaced by a fresh twin in the same slot. Because a learner and its twin trade at the same moments, the only difference between them is the direction, so a gap between their results measures prediction skill rather than timing luck.

Criterion 2 of the success criteria pairs each closed learner trade with its twin's trade from the same decision and compares gross profit before fees, grouped into 12 hour blocks. The coin adds noise as well as removing timing luck, so at a correlation of about 0.1 this test has only about a third of the power of the skill test on the forecasts table, and it is supporting evidence. The command `status --twins` shows the twins, and `coin_flip_twins = false` runs a population without them. The code is in [controls.py](../src/btc_survivor/controls.py), and the criteria are in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria).

## Why fees decide survival at retail costs

Every trade is 500 dollars. On Coinbase the spread is about one cent and walking the book for 500 dollars costs about 0.0006 bps, so one leg costs almost exactly the taker fee plus the 1 bp of extra slippage. At the retail fee of 90 bps that is 4.55 dollars a leg and 9.10 dollars a round trip, and at the 5 bps fee it is 0.30 dollars a leg and 0.60 dollars a round trip. The fee profiles and the cost estimate behind these figures are in [the execution model](execution_model.md#the-cost-estimate-and-the-required-edge).

At retail costs the required edge is about 190 bps. A 30 minute Bitcoin move has a standard deviation of about 32 to 34 bps and an average absolute size of about 21.5 bps, so even perfect foresight would earn only about 21.5 bps per trade on average, while the predictions themselves are a few bps. The gate therefore never passes, and every retail trade is a forced exploration trade. Each bot opens once every 12 hours when its forced timer runs out and closes at its next decision, which makes it a quiet bot with two round trips a day. That costs 2 times 9.10 dollars plus 1.16 dollars of rent, about 19.36 dollars a day, the same for every bot.

| Fee profile | Quiet bot | Active bot, 30 minute horizon | Churning bot, 30 minute horizon |
| --- | --- | --- | --- |
| retail, 90 bps | 516 days | 45 days | 23 days |
| active retail, 30 bps | 1,358 days | 132 days | 67 days |
| low fee, 5 bps | 4,237 days | 643 days | 334 days |

The table gives days until death from costs alone at a start equity of 10,000 dollars. A quiet bot only trades when forced, an active bot opens or closes at every decision, and a churning bot flips at every decision. Market gains and losses barely move these numbers, because a typical 30 minute move on 500 dollars is about 1.65 dollars either way.

The consequences are plain. At 10,000 dollars the first death is more than a year away, so runs A, B, D and E see no deaths, no replacements, no tournaments and no mutation within their windows, and every learner in them stays a founder. Run C starts at 500 dollars precisely so that deaths happen. There a quiet retail bot dies in about 25 days, with the first wave of deaths expected between days 24 and 28 and the second between days 48 and 56, when generation 2 appears. But every bot in run C pays the same 19.36 dollars a day, so which bot dies first, and which survivor wins a tournament, is decided by a few dollars of market noise. Parent choice in run C mostly reflects that noise, because costs swamp any plausible skill a bot could have. The seeded Monte Carlo in `scripts/run_c_analysis.py`, with forced opens 12 healthy hours apart as the engine spaces them, puts a run C bot with no skill at a mean death day of 25.83 with a standard deviation of 0.60 days, and a forecast correlation of 0.1 moves that mean by only 0.33 days and outlives a bot with no skill 65 percent of the time. Selection acts only through the profit and loss of forced trades, while every learner trains on every matured label whether or not it trades, and the margin gene changes nothing while the gate never passes. Run C checks that death, replacement, mutation and lineage work under live conditions, and it is not evidence that evolution finds skill. There is no control run without selection, so version 1 does not test whether evolution helps, and after an external review in October 2026 that was recorded as a limit rather than tested, as [honesty_and_limits.md](honesty_and_limits.md#selection-and-why-evolution-is-not-tested) explains. Run C also reaches its deaths on leverage no spot account allows, nearly 50 times its equity at a quiet bot's last open, as [the execution model](execution_model.md#no-principal-is-reserved) shows. If every bot dies to fees, that is a real finding and the logs show it plainly, and [honesty_and_limits.md](honesty_and_limits.md#outcomes-fixed-by-arithmetic) lists these outcomes as fixed by arithmetic before launch. The arithmetic is in [section 3 of the plan](../PLAN.md#3-what-to-expect-from-the-default-experiment), and the fee table is the public [Coinbase Advanced fee tiers](https://www.coinbase.com/advanced-vip) page.

## Later additions that are not built

These ideas came up during the design. None of them is built in version 1, and any of them would be fixed before a run starts, never tuned on data already collected.

* **Culling.** Retiring the weakest living bots on a schedule instead of waiting for death would let selection run at 10,000 dollars too. In version 1 bots leave only by dying, as the brief describes, and at retail costs a fitness ranking would mostly rank noise.
* **Crossover.** Mixing the weights of two parents. The hidden units of two separately trained networks do not line up, so a plain mix of weights tends to break both parents, and the units would have to be matched first.
* **A hall of fame.** Keeping copies of the fittest bots ever seen, as parents or as a fixed comparison, so that a good lineage is not lost to one run of bad luck.
* **A tiered fee profile.** Charging each bot by its own trailing 30 day volume from the public fee table. A quiet bot trades about 2,000 dollars a day, so a real account would reach 75 bps after about 5 days and 30 bps after about 25 days. The fixed 90 bps is the rate of a new account and harsher than reality after the first weeks.
* **Depth based imbalance.** Measuring order book imbalance within 5, 10 and 25 bps of the mid instead of over the top 10 levels, which on a one cent book cover a very small slice of price.
* **Volatility gating.** Trading only in volatile hours, when a 30 minute move is large relative to costs.

The list follows [section 11](../PLAN.md#11-evolution) and [section 21](../PLAN.md#21-controls-parallel-runs-and-success-criteria) of the plan, and the fee tiers behind the tiered profile are in [section 2.2 of the plan](../PLAN.md#22-do-the-fee-placeholders-match-and-which-profile-should-be-the-default-question-2).

## Road map

The version 1 road map has four steps, each adding one source of information that must show skill on the forecast records before it is trusted. They are longer horizons, more market data, current affairs as a predictor, and variable position sizing only after the skill test passes. The approved [version 2 plan](../PLAN_V2.md) turns these into stages, each switched on only by evidence from the one before.

1. **The bench.** Version 1, runs A to E on Bitcoin, checks the machinery of learning, trading, feed health and evaluation.
2. **The skill stage.** Version 2 adds large US stocks and news scored by language models, and asks only whether a predictor forecasts the next return, through a skill test fixed before launch. Its details are in the [version 2 plan](../PLAN_V2.md#117-road-map-after-version-2).
3. **The profit stage.** Only after a predictor passes the skill test, each bot predicts several horizons at once, for example 30 minutes, 2 hours, the rest of the session and the next session. It enters where the expected gain after costs is largest, holds while the expected remaining gain beats the cost of exiting, and exits when it does not, so the holding period comes from the predictions rather than a timer. Position sizes grow with confidence. Its rules are written before its own launch and tested forward against a fixed horizon version of the same bots launched beside it, never fitted to data already collected.

Low costs are what make the profit stage plausible. A stock round trip in version 2 costs about 3.6 to 6.6 bps, against about 190 bps for the version 1 retail Bitcoin round trip, so a modest forecast has a chance to clear its costs there. The stages are set out in [section 11.7 of the version 2 plan](../PLAN_V2.md#117-road-map-after-version-2), and the version 1 road map is in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria).
