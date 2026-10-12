# Evaluation

This guide explains how version 2 is judged, what `btc_survivor_v2 evaluate` computes, the success criteria and outcomes fixed before launch, how much the confirmatory test can detect, and how the learning rate was fixed. Every rule here was written before launch and never changes after it, and before the cutoff every number is descriptive. The design is [section 11](../PLAN_V2.md#11-evaluation-and-success-criteria-fixed-before-launch) and [section 12 of the plan](../PLAN_V2.md#12-outcomes-fixed-by-arithmetic), and the overview is [v2_overview.md](v2_overview.md).

## The evaluate command

```bash
.venv-v2/bin/btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml
.venv-v2/bin/btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml --detail
```

`v2/evaluate.py` opens both run databases, `news.db`, the key ledger and the tape check records, all read only. It takes every setting from the runs' `meta` at launch, the `[evaluation]` section from S2's `evaluation_json` and every cost, timer and margin from `config_toml`, so editing a config file later changes no result, and the report says when a config file no longer matches its launch. It computes every number from rows up to the cutoff, so a later run never changes a result.

* By default it prints each criterion's verdict, each outcome as matched or not, and counts, never a price, a forecast or a statistic.
* `--detail` adds the problems behind each verdict, every predictor's pooled skill over all rows and over event rows, the squared error and paired comparisons, calibration, the twin tests with each learner's share of long opens, deaths, the outcomes, data quality, spend and the code commits.
* `--checks DIR` names the folder of tape check records, by default `data/v2/checks` beside the run folders, and `--ledger FILE` the key ledger, by default `~/.config/btc_survivor/llm_ledger.db`.
* It exits with code 0 when the run is valid so far, 1 when it is not, and 2 when it cannot run, such as before launch, for runs not launched together or for a database of the wrong kind.

The command is specified in [section 11.1 of the plan](../PLAN_V2.md#111-the-evaluate-command).

## Run minutes, the cutoff and the window

* **A run minute** is a session minute in which news health held and at least one symbol was healthy, read from S2's `news_minutes` and its valid candles. The rule is stored in `meta` as `run_minute_rule`, and `evaluate` refuses any other.
* **The cutoff** is the minute S2 reaches 15,600 run minutes counted from launch, the length of 40 full sessions. From a launch on 19 October 2026 with no minute lost, it falls on 15 December 2026, in session 41, because the early close of 27 November is shorter.
* **The window** runs from the open of session 2 to the cutoff. Session 1 is burn in, because features need a session of history and the puller starts at the newest score at launch.
* **A row counts** only when its label matured by the end of the window, so a full window holds about 467 rows per symbol and predictor.

Minutes lost to a halt of every symbol or to a news outage push the cutoff later, and `evaluate` reports them by cause. Before the cutoff, at the descriptive looks of sessions 10 and 20, it computes the same numbers on the run so far. The rules are in [section 11.1 of the plan](../PLAN_V2.md#111-the-evaluate-command).

## Predictors and labels

At every evaluation minute, session minutes 30, 60 and so on while the minute plus 30 is still in the session, `10:00` to `15:30` ET, each usable symbol gets one row per predictor in `stock_forecasts`. That is 12 minutes a full session and 6 on an early close.

| Predictor | Run | Units | Meaning |
| --- | --- | --- | --- |
| `population` | S1 and S2 | bps | mean prediction of the living learners for the symbol |
| `linear_prices` | S1 and S2 | bps | the version 1 online ridge on the price z scores, pooled over the symbols |
| `linear_news` | S2 | bps | the same ridge with the two news inputs |
| `news` | S2 | score | company plus macro input from the primary stream, the confirmatory predictor |
| `news_company`, `news_macro` | S2 | score | each input alone |
| `haiku`, `finbert` | S2 | score | the same formula on Haiku scores only, and on FinBERT scores of every routed item |
| `finbert_matched` | S2 | score | FinBERT on exactly the items Haiku scored |
| `sonnet`, `haiku_official` | S2 | score | Sonnet and Haiku on exactly the official items both scored |

Score predictors have no fitted scale, so they are judged by correlation only. The label is `1e4 * ln(exit / entry)` on the reference prices 30 session minutes apart, and it matures only when both minutes were healthy in one epoch, otherwise it ends as a gap. Every row stores `inputs_max_recv_at`, the newest receipt behind it. The predictors are in [section 10.4 of the plan](../PLAN_V2.md#104-evaluation-minutes-forecasts-and-labels).

## Pooled skill

`v2/stats.py` judges one predictor's rows in the window.

1. Prediction and label are turned into z scores within each symbol, and a symbol whose prediction never varies is left out and named.
2. Each row's product of the two z scores is averaged over the symbols present at its minute.
3. The correlation is the mean of those minute values, with the version 1 Newey West error at 12 lags, one session, as its standard error, the Driscoll Kraay estimator in its simplest form. It allows for the market moving every symbol together at a minute.
4. A predictor passes when the correlation over its standard error exceeds 2.33, one sided at 1 percent. Beside it come the detectable correlation, 3.61 standard errors, and the one sided 95 percent upper bound, the correlation plus 1.645 standard errors.

Pooling without clustering would overstate the evidence. On 1,000 seeded null runs at c of 0.5 in `test_v2_stats`, the clustered test rejected 1.0 percent and an unclustered one 9.0 percent. `pooled_dm` compares two bps forecasts by their squared errors over each symbol's label variance with the same 12 lag error, and `session_block_test` compares learner and twin trades with one block per session. The statistics are in [section 11.2 of the plan](../PLAN_V2.md#112-pooled-skill).

## The confirmatory test

* **The test.** The predictor `news` in S2 over the window, passing when t exceeds 2.33. It is the only test with a pass label, and everything else is printed without one.
* **What it states.** Whether this forecast correlates with the next 30 minute return after this system received the news, for these megacaps at these delays. It says nothing about news in general, news after the close or smaller companies.
* **Why this predictor.** It is the new information version 2 adds, with no fitted parameters and no warmup. S2's learners mix price skill with news weights that start at zero and move slowly, so S2 against S1 is descriptive.
* **Looks.** Sessions 10 and 20 are descriptive looks. There is no early stop for success, and the run stops early only when criterion 1 fails.
* **A failure** means the test did not detect a correlation. It is reported with the one sided 95 percent upper bound, the largest correlation the data leave plausible.
* **The expected result,** written before launch, is not passed. Megacaps absorb headlines within minutes, the pipeline adds Finnhub's delay and up to a minute or two of batching, and most earnings come after the close.

The test is fixed in [section 11.5 of the plan](../PLAN_V2.md#115-the-confirmatory-test).

## Success criteria

1. **The run is valid.** `evaluate` prints one verdict for each check, and the run is invalid when any fails.
   1. `tape checks`, a consolidated tape check before launch, and a consolidated verdict on the first clean full session at or after sessions 10, 20 and 40, passing over at most one inconclusive session. The records come from `tape-check` runs in `data/v2/checks`, so each of those checks must actually be run.
   2. `lookahead audit`, no forecast row with an input received after its minute and no score line received before its `scored_at`.
   3. `labels recomputed`, every stored label equal to the label recomputed from `stock_candles`, which are kept for the life of the run for this.
   4. `runs identical`, S1 and S2 with identical run minutes, labels and `linear_prices` rows.
   5. `invariants`, no `invariant_violation` event in either run.
   6. `one config prompt and model`, the config hash, prompt hash, schema hash and one model id per scorer recorded at launch covering every model call, every heartbeat `news.db` still holds and every S2 minute. Heartbeats are pruned after 14 days, so S2's `hash_mismatch` minutes carry this check over the whole window.
   7. `spend within cap`, the key ledger's spend of both models together, each call at its cost or else its reservation, at or under 20 dollars in every UTC month of the run.
   8. `local only checks`, no failed `local_only` check and a passed one in each run between the previous close and the close of every session of the window. A session in which the stock process never ran therefore fails this check.
   9. `code commits listed`, a commit recorded at launch and at every later start. A commit marked dirty is listed but cannot be reproduced, so restarts come only from a committed tree.

   A fallback share above 25 percent of the items routed to Haiku in session is a caveat, not a failure, and so is a day that raised the spend alert.
2. **The confirmatory news skill,** as above.
3. **Descriptive results.** Every predictor's skill over all rows and over event rows, S2 against S1, `linear_news` against `linear_prices`, `haiku` against `finbert_matched`, `sonnet` against `haiku_official`, `news_company` and `news_macro`, calibration slopes, the twin test per run, any deaths and data quality, meaning the healthy fraction, run minutes lost to news health by cause, channel outage minutes, receipt to scoring lag per scorer, items per source and the fallback share.
4. **The outcomes fixed by arithmetic** matched, as the next section sets out.
5. **Version 1 undisturbed.** Runs A to D kept running and `test_v1_untouched` passes. `evaluate` does not check this, so the write up records it by hand.
6. **Cost.** About 1 to 5 dollars a month for both models together and never above the cap, reported per model.

The project succeeds when criterion 1 holds and the criterion 2 result is reported, whatever it shows. The criteria are in [section 11.6 of the plan](../PLAN_V2.md#116-success-criteria).

## Outcomes fixed by arithmetic

These check the accounting, not skill, apply to both runs and use the planning volatilities of the plan, 20 percent a year for QQQ up to 60 for TSLA, with three quarters of daily variance inside the session.

| Symbol | Round trip bps | Round trip dollars at 500 | 30 minute sd bps | Required edge bps at margin 0.25 |
| --- | --- | --- | --- | --- |
| QQQ | 3.6 | 0.18 | 30 | 11.2 |
| MSFT | 4.6 | 0.23 | 33 | 12.9 |
| AAPL | 4.6 | 0.23 | 38 | 14.1 |
| GOOGL | 4.6 | 0.23 | 45 | 16.0 |
| AMZN | 4.6 | 0.23 | 45 | 16.0 |
| META | 6.6 | 0.33 | 53 | 19.9 |
| NVDA | 4.6 | 0.23 | 68 | 21.6 |
| TSLA | 6.6 | 0.33 | 91 | 29.3 |

`evaluate` judges each outcome as matched, not matched or not reached yet, with these thresholds, which the owner confirms before launch.

* **First trades** in session 3, at session minutes 60 to 270, which is `10:30` to `13:30` ET plus up to 29 minutes of waiting for a phase minute.
* **Rhythm** of 1.5 to 2.5 round trips per full session for each learner, measured from session 4.
* **Burn** within 25 percent of plan, which is 1.68 dollars of rent and about two round trips a full session, 2.18 dollars on average, about 86 dollars by the cutoff.
* **Founder equity.** The learners' mean equity equals 10,000 dollars less rent of 0.0043081 dollars per usable minute of the home symbol and the round trips paid, within twice the square root of their trades times 50 bps of the 500 dollar notional, about 43 dollars at 75 trades, with a floor of one cent.
* **No deaths.** The death threshold is about 2 dollars and the burn about 2.2 a session, so a death would take about 4,500 sessions.
* **Flat at every close,** except where a close was rejected because its symbol was not healthy. A position carried over a close is excused only when that symbol's candle at the close is invalid, and a missing candle row, meaning the process was down, is not excused.
* **The gate.** At least 95 percent of learner opens forced, while the realized prediction spread stays at or under 0.10 label standard deviations.
* **Pairing.** S1 and S2 predictions identical until S2's first training step with a nonzero news input, expected in session 2 or later.

The planned rows and the arithmetic behind them are in [section 12 of the plan](../PLAN_V2.md#12-outcomes-fixed-by-arithmetic), and `test_v2_arithmetic` works them by hand.

## Power

`scripts/v2_power.py` simulates the news predictor of S2 for the eight symbols on synthetic data only, with zero inflated scores decaying over 120 session minutes, a macro part shared by every symbol, and fat tailed labels with clustered volatility, using planning values never fitted to recordings. The key unknown is c, the correlation between two symbols' predictions at one minute, which rises with the share of macro news. Each run is judged by the same estimator `evaluate` uses. The full run of 2,000 runs per scenario gave these figures on 12 October 2026.

| Sessions | c | Null rejection | Detectable | Power 90 | Power at 0.075 | Power at 0.11 |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.2 | 0.0150 | 0.105 | 0.113 | 0.62 | 0.88 |
| 20 | 0.5 | 0.0105 | 0.133 | 0.146 | 0.41 | 0.72 |
| 20 | 0.8 | 0.0075 | 0.156 | 0.179 | 0.27 | 0.54 |
| 20 | 1.0 | 0.0110 | 0.170 | 0.220 | 0.21 | 0.42 |
| 40 | 0.2 | 0.0085 | 0.075 | 0.079 | 0.88 | 0.99 |
| 40 | 0.5 | 0.0100 | 0.095 | 0.100 | 0.70 | 0.94 |
| 40 | 0.8 | 0.0100 | 0.112 | 0.120 | 0.54 | 0.84 |
| 40 | 1.0 | 0.0115 | 0.122 | 0.139 | 0.44 | 0.76 |
| 60 | 0.2 | 0.0105 | 0.061 | 0.063 | 0.96 | 1.00 |
| 60 | 0.5 | 0.0065 | 0.078 | 0.081 | 0.86 | 0.99 |
| 60 | 0.8 | 0.0060 | 0.092 | 0.096 | 0.72 | 0.96 |
| 60 | 1.0 | 0.0055 | 0.100 | 0.108 | 0.61 | 0.91 |

* **Detectable** is 3.61 times the root mean square standard error under the null, the figure `evaluate` prints from the realized error. At 40 sessions it runs from 0.075 to 0.122, which confirms the plan's 0.076 to 0.123.
* **Power 90** is the true correlation the test finds in 90 percent of runs, 0.079 to 0.139 at 40 sessions.
* **Null rejection** stays near 1 percent, so the test holds its level.
* Power at a true 0.075 runs from 0.88 at c of 0.2 to 0.44 at c of 1.0. The plan's lower 35 to 40 percent came from a sparser design, about one impact a session. The simulation's four impacts a session is a planning value, and power falls when news is sparser, while the detectable figure does not move.
* The simulation keeps every pair of symbols at c, while S2 pools all seven companies into QQQ's company input, which would raise the low end to about 0.083 by an analytic estimate made beside the script.

Trading every decision in the forecast's direction breaks even only at a correlation of about 0.08 to 0.17 across these symbols, so a just detectable predictor would still not cover costs on most of them. Version 2 measures skill and does not promise profit. Power is discussed in [section 11.4 of the plan](../PLAN_V2.md#114-power).

## The learning rate

`scripts/v2_learner_check.py` fixed the learning rate before launch on synthetic data only. It runs the stock process's own learning code, the per symbol standardizer and label scale, the session bounded label queue, the pooled window and the bot's own training step, on 40 synthetic sessions of the eight symbols with a common market factor and a planted correlation of 0, 0.05 or 0.1. The rule takes the largest rate whose prediction spread with no signal is at most 0.10 label standard deviations and whose squared error with no signal is within 2 percent of predicting zero.

| Rate | Spread with no signal | Squared error against zero | Rule |
| --- | --- | --- | --- |
| 0.0003 | 0.134 | 1.0201 | fails both limits |
| 0.0001 | 0.072 | 1.0074 | passes, chosen |
| 0.00003 | 0.050 | 1.0037 | passes |
| 0.00001 | 0.056 | 1.0039 | passes |

The check ran twice with 3 seeds and 3 learners per rate and gave identical figures, and 0.0001 is in both run configs. With a planted correlation of 0.1 the learners reached a correlation of about 0.06 against an ideal of 0.11, so they learn slowly by design. The rule is in [section 6.4 of the plan](../PLAN_V2.md#64-learning).

## What a result can and cannot show

* A pass shows that the scored news forecast the next 30 minutes on these eight symbols during this window, at this system's delays. A failure shows only that the test did not detect it, bounded by the upper confidence bound.
* Haiku against FinBERT and Sonnet against Haiku are descriptive. Sonnet sees only about 5 to 20 official items a day, so only a large difference could show.
* Every result rests on one set of scores from each model, and a fresh rerun would differ by about what the probes measure.
* News health selects minutes, so a scoring backlog around FOMC, CPI or earnings could remove minutes near the biggest news, which the report of lost minutes by cause makes visible.
* Minutes lost to news health are reported by cause, but `evaluate` does not report their overlap with scheduled releases, because the release calendar lives only in the news config, which it does not read.
* Event rows use the item tickers recorded in `news_seen`, received in the 30 minutes before the minute, and QQQ's event rows count only items tagged QQQ.

The honest limits are listed in [section 18 of the plan](../PLAN_V2.md#18-risks-and-honest-limits), and the results are written up in [RESULTS_V2_TEMPLATE.md](RESULTS_V2_TEMPLATE.md).
