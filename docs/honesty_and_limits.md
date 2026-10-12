# Honesty and limits

This page says what the four live runs can show and what they cannot, and it was written before any of them reached a result, so the reading of the results cannot be bent to fit them. Most of what happens to the bots under the default settings is decided by fees, notional and the forced timer before launch, and the page says which parts are arithmetic and which parts are open questions. The design behind every number here is in [section 3](../PLAN.md#3-what-to-expect-from-the-default-experiment), [section 20](../PLAN.md#20-honest-limits-and-risks) and [section 21](../PLAN.md#21-controls-parallel-runs-and-success-criteria) of the plan.

## What the runs can show

The runs answer three narrow questions honestly.

* **Prediction skill.** Whether the population's mean forecast of the next 30 minute move in BTC USD correlates with what then happened, scored against predicting zero and against a plain linear model on the same inputs. This is the headline result, and it comes from run A at day 28 of healthy data.
* **Skill against luck.** Whether learners make more gross profit than coin flip twins that trade at exactly the same moments with a random direction.
* **The machinery.** That the books balance at every snapshot, that costs kill bots for the reasons the ledger says, and that death, replacement, mutation and lineage work under live conditions, which run C checks.

The success criteria and their thresholds are in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria), and the tables for writing them up are in [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md).

## What the runs cannot show

* **That a strategy makes money.** Paper fills are kinder than real ones, as the section on paper liquidity below explains, so even a profitable paper bot would not prove a profitable real one.
* **That the market holds no edge.** A failed skill test only says the population's forecasts had a correlation below about 0.1 over those weeks. A better model, more data or a longer horizon could still find something.
* **Anything about other markets or other times.** Each run covers one product on one venue for 28 or 56 days of one market regime. BTC volatility was about 42 percent a year in 2025 and 46 percent so far in 2026, and the 30 days before launch were calmer than both.
* **That the best bot is good.** Among 20 bots chance alone produces one that looks impressive, so the best single bot is never evidence on its own.
* **Anything from replay.** Replay exists to test the plumbing, and replay results are never used to choose, tune or rank strategies.

The limits are listed in full in [section 20 of the plan](../PLAN.md#20-honest-limits-and-risks).

## Outcomes fixed by arithmetic

At a notional of 500 dollars the spread is one cent and walking the book costs about 0.0006 bps, so one leg of a trade costs the taker fee plus the 1 bp of extra slippage and almost nothing else. That is 4.55 dollars a leg at 90 bps and 0.30 dollars at 5 bps, so a round trip costs 9.10 dollars at retail fees and 0.60 dollars at the low fee. A bot that only trades when forced makes one round trip every 12 healthy hours, two a day, and pays 1.16 dollars a day of rent. Its burn is therefore 19.36 dollars a day at retail fees and 2.36 dollars a day at the low fee, plus small short charges of 2 bps to open and 0.5 bps an hour, which come to about 0.11 dollars for a 30 minute short.

These numbers fix the following outcomes before launch. They check the accounting, they are judged only as matched or not matched, and none of them is a finding about trading.

* **Run A.** Every open is forced and no bot dies by day 28. A learner at 10,000 dollars would take about 516 days to burn down to the death threshold. The mean equity of the founding learners should sit within about 20 dollars of 10,000 minus their rent and 9.10 dollars for each round trip they made, which at day 28 is about 55 round trips and puts them near 9,470 dollars.
* **Run B.** Almost every open is forced, and no bot dies, because even a bot that flips side at every decision would take about 334 days to die at the 5 bps fee.
* **Run C.** Every bot dies of fees. At 500 dollars the death threshold is about 10 dollars, one round trip plus a day of rent, so a quiet bot has about 25 days to live. The first wave of deaths falls between days 24 and 28 and the second between days 48 and 56, and generation 2 appears in the second wave. Parent choice in run C reflects market noise, because costs swamp any skill a bot could have.
* **Run D.** Run D is run B with a 4 hour horizon. It burns about 2.36 dollars a day plus 4 hours of short funding per short, so no bot dies. It is exploratory and not part of the success criteria.

The `evaluate` command checks each of these against the database. The cost model is in [execution_model.md](execution_model.md), and the full death table is in [section 3 of the plan](../PLAN.md#3-what-to-expect-from-the-default-experiment).

## Why the gate almost never passes at retail fees

A bot opens a trade on its own only when its predicted move clears the required edge, which is the estimated round trip cost plus its margin gene. At retail fees that is two fees of 90 bps, two legs of 1 bp extra slippage, a spread and book impact that round to zero, and the starting margin of 8 bps, about 190 bps in all. A typical 30 minute move in BTC has a standard deviation of about 32 to 34 bps and an average absolute size of about 21.5 bps. Even perfect foresight would earn about 21.5 bps per 30 minute trade on average, so a 190 bps move is about six standard deviations away.

The predictions are much smaller than the moves. The output layer starts at a tenth of its normal scale, and with the learning rate of 0.0001 a learner with no signal to find keeps its predictions within a few bps, about 3 bps of spread in the synthetic check. A learner with realistic skill keeps them that small too, because an honest forecast of a noisy return stays close to zero. So the gate will not pass in practice in runs A and C, and every retail trade will be forced. At the 5 bps fee the required edge is still about 20 bps, several times the size of the predictions, so run B will be almost entirely forced as well.

This is not a bug. It is what the fees say, a retail account pays more to cross the spread twice than a 30 minute move is usually worth. The decision rule is in [the bot section of the plan](../PLAN.md#9-bots-and-trading-logic) and the learning rate check is in [learning.md](learning.md).

## Forced trades

Without forced trades a bot that never passes the gate would never trade, and nothing about its trading would ever be measured. So a flat bot that has gone `force_trade_after_hours = 12` healthy hours since its last open opens at its next decision in the direction of its current prediction, even though the gate failed. Each bot's timer starts at a random delay of up to 12 hours drawn from its own generator, so the 20 bots do not all trade in the same few minutes, and the first forced trades fall about 13 to 25 hours after launch.

Forced trades shape what the runs mean.

* Their timing comes from the timer, not from the model. Only their direction comes from the prediction.
* At retail fees they are what kills bots, at a steady 18.20 dollars a day, 18.00 of it in fees and 0.20 in extra slippage, which moves the fill price and so shows up in gross profit rather than in fees.
* Their profit before fees is a test of direction at arbitrary moments, which is exactly what the coin flip twins are built to compare.

Every decision row carries a `forced` flag next to the real gate result, and every trade row carries the `forced` flag too, so forced and unforced trades can always be told apart. The forced timer is described in [section 9 of the plan](../PLAN.md#9-bots-and-trading-logic).

## Paper liquidity and other kindnesses

Paper trading is kinder than real trading in ways that cannot be fully modelled.

* **Orders never consume liquidity.** A learner and its twin trade at the same moment and both fill against the same price levels, no paper order ever moves the price another bot sees, and nobody else in the market reacts to them.
* **There is no queue and no rejection by the exchange.** An order fills against the book as it stands after the simulated latency of 1,000 milliseconds plus up to 300 milliseconds of jitter. It is rejected when the top 20 levels cannot fill it, when the feed leaves Healthy first or when the process restarts.
* **Fees never change during a run.** A fixed 90 bps is harsher than reality after the first weeks, because a real account trading about 2,000 dollars a day would reach the 75 bps tier in about 5 days and 30 bps in about 25 days.
* **The short leg is a composite.** It pays the Coinbase spot taker fee plus Kraken style margin charges. Kraken says Coinbase US does not offer spot margin shorting, and that could not be checked with Coinbase.
* **The low fee profile is out of reach for a US retail account.** The 5 bps tier needs 20 million dollars of volume a month, so a good result in run B says the learning loop works, not that a retail account could trade it. The Intro taker fee is 50 bps in the EU, the UK and Canada and 10 bps in the rest of the world, so the account's region matters.

The extra slippage of 1 bp per leg is larger than anything the book walk produced at 500 dollars, which pushes the other way. The fill model and its sources are in [execution_model.md](execution_model.md).

## The coin flip twins and their power

Every learner slot also holds a twin with its own account. Whenever the learner opens, closes or flips, the twin does the same at the same moment, choosing long or short with a fair coin. The twin pays the same fees, slippage, latency, funding and rent, has its own stop loss and death check, is never a parent and never counts toward fitness. Because the pair trade at the same moments, the only difference between them is the direction, so a gap between them measures direction skill rather than timing luck.

Criterion 2 pairs each closed learner trade with its twin's trade from the same decision and compares gross profit before fees, funding and short charges, since those differ only through the direction mix. The pairs are grouped into fixed 12 hour blocks of the UTC clock, which keeps the test valid when trades of different bots overlap in time. A 28 day run has about 56 blocks, a few more when gaps stretch its 28 healthy days over more calendar days. The test passes when the mean of the block means is positive and its t statistic is above the one sided 1 percent quantile of Student's t, with one degree of freedom fewer than the number of blocks.

The twin's coin adds noise as well as removing timing luck. At a forecast correlation of about 0.1 this test has only about a third of the power of the skill test, so it is supporting evidence, and a failure of this test alone does not contradict a pass of the skill test. In run C the twins die of fees on about the same schedule as the learners, and a dead twin is replaced by a fresh twin in the same slot. The twins are described in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) and their code is in [controls.py](../src/btc_survivor/controls.py).

## Statistical limits

* **One confirmatory test.** Only the skill test on run A at day 28 counts. It uses one forecast per 30 minute window so labels never overlap, skips the first day of forecasts, and uses a Newey West standard error over 12 windows, about six hours, with a one sided 1 percent level. Forecasts begin about 7 hours after launch, once the label scale is ready, so day 28 brings about 1,280 scored windows, fewer after gaps. At that count it can detect a correlation of about 0.1 with 90 percent power, and `evaluate` prints the realized window count and the correlation it can detect.
* **Overlapping labels.** Consecutive one minute labels share almost all of their 30 minutes. The rolling skill of a single bot over 360 samples rests on only about 12 independent outcomes, so the skill column in `status` is noisy by nature.
* **Squared error.** With no skill the population's squared error is expected to come out a few percent worse than predicting zero, by about the square of the forecast's spread over the label's. A small loss against zero is therefore not evidence of harmful predictions.
* **Shared labels.** Runs B and C learn from the same market as run A, so they are reported beside it and are not separate tests.
* **Run D.** With about 6 windows a day, its skill test at day 28 can only detect a correlation of about 0.28. All 20 of its bots decide within the first 20 minutes of each 4 hour block, so its trades cluster in time. Its numbers are descriptive.
* **Early days.** Results at days 7 and 14 are descriptive, only the final day counts, and no setting changes during a run.
* **Healthy days, not calendar days.** The criteria count days of healthy data. Feed gaps, `clock_skew` pauses and a sleeping Mac all stop the count, so day 28 arrives later than 28 calendar days after launch.

The tests are defined in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) and their code is in [stats.py](../src/btc_survivor/cli/stats.py).

## A null result is a real finding

If the skill test fails, the runs still say something concrete. The population's forecasts of 30 minute BTC moves had a correlation below about 0.1 over those weeks, which fits the near zero autocorrelation of 15 and 30 minute returns found while planning and the gross edges of about 0.5 to 5 bps per trade that older studies suggest. If every bot in run C dies of fees, that is what the costs predicted, and the `deaths` table shows which of fees, funding, rent, stop losses and other trading losses killed each one.

Every run is reported whatever it shows, and no default was tuned to make the bots look better. A null result is the honest answer to the question version 1 asks, and it is the baseline any later predictor has to beat. Version 2 runs as a separate system that paper trades seven big tech stocks and the QQQ fund, and tests whether news scored by language models forecasts their next 30 minute return, under one confirmatory skill test fixed before launch that adapts these rules to stocks, as set out in [PLAN_V2.md](../PLAN_V2.md), its [confirmatory test](../PLAN_V2.md#115-the-confirmatory-test) and its [road map](../PLAN_V2.md#117-road-map-after-version-2).
