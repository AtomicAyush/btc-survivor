# Evaluation

This guide explains how version 2 is judged, what `btc_survivor_v2 evaluate` computes, the success criteria and outcomes fixed before launch, how much the confirmatory test can detect, and how the learning rate was fixed. Every rule here was written before launch and never changes after it, and before the cutoff every number is descriptive. The design is [section 11](../PLAN_V2.md#11-evaluation-and-success-criteria-fixed-before-launch) and [section 12 of the plan](../PLAN_V2.md#12-outcomes-fixed-by-arithmetic), and the overview is [v2_overview.md](v2_overview.md).

## The evaluate command

```bash
.venv-v2/bin/btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml
.venv-v2/bin/btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml --detail
```

`v2/evaluate.py` opens both run databases, `news.db`, the key ledger and the coverage records of the tape check, all read only. It takes every setting from the runs' `meta` at launch, the `[evaluation]` section from S2's `evaluation_json` and every cost, timer and margin from `config_toml`, so editing a config file later changes no result, and the report says when a config file no longer matches its launch. It computes every number from rows up to the cutoff, so a later run never changes a result, apart from criterion 1.1, whose record at or after a due session may come only after the cutoff.

* By default it prints each criterion's verdict, each outcome as matched or not, and counts, never a price, a forecast or a statistic.
* `--detail` adds the problems behind each verdict, every predictor's pooled skill over all rows and over event rows, the test beyond price, the squared error and paired comparisons, calibration, the twin tests with each learner's share of long opens, the own trades, the round trips per session forced and own, the largest notional over equity, the learners' mean equity with and without the dead, deaths, the outcomes, data quality, spend and the code commits.
* `--checks DIR` names the folder of coverage records, by default `data/v2/checks` beside the run folders, and `--ledger FILE` the key ledger, by default `~/.config/btc_survivor/llm_ledger.db`.
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

* **The claim.** A fixed sequence of two tests at one familywise level of 1 percent, decided by the owner on 12 October 2026. H1 is the predictor `news` in S2 over the window, passing when t exceeds 2.33. Only when H1 passes is H2 run, the partial test of `news` against the label with both residualised on the 10 price z scores S2 stores in `price_z` at each evaluation minute, within each symbol, judged with the same clustered error and the same one sided 1 percent level. H2 never runs without a pass of H1, so the chance of any false pass stays at 1 percent. These are the only tests with a pass label, and everything else is printed without one.
* **What H1 states.** Whether the decayed news score correlates with each 30 minute return over the following session hours after this system received the news, for these megacaps at these delays. Half of a score's weight falls in the first 2 session hours, and the first evaluation minute comes 0 to 30 minutes after receipt, so the market's first reaction is never part of it. It says nothing about news in general, news after the close or smaller companies.
* **What H2 adds.** Whether that correlation holds beyond what the price features already say. A synthetic check made for the external review of 12 October 2026, 2,000 runs of 40 sessions at c of 0.5, found that price momentum alone, with headlines that track recent moves, passed H1 in 9 percent of runs with a mild confound and in 63 percent with a strong one, against about 1 percent for H2. Where news is nearly independent of price, H1 then H2 costs 2 to 4 points of H1's power, 0.64 against 0.67 at a true 0.075 and 0.91 against 0.93 at 0.11 at c of 0.5 in the full run of `scripts/v2_power.py`, and H1's own power is unchanged. `v2/stats.py` corrects the clustered error of the residuals for their small sample size, a variant of its own that the owner signs off. With exogenous price scores it held H2 at H1's level on synthetic null runs at 468 rows per symbol, while a review with scores built from the label's own past returns, as three of the ten are, found H2 alone rejecting 1.4 to 1.6 percent, so the familywise level rests on the sequence, which stayed at 0.8 to 0.9 percent under the null in that review.
* **Why this predictor.** It is the new information version 2 adds, with no fitted parameters and no warmup. S2's learners mix price skill with news weights that start at zero and move slowly, so S2 against S1 is descriptive.
* **The prompt matches the test.** Each model is asked about the price move still to come after the market's first reaction to an item, with its horizon in minutes, hours or days and its surprise, and the score's strength folds both in, as [v2_news.md](v2_news.md#the-score-streams) sets out.
* **Looks.** Sessions 10 and 20 are descriptive looks. There is no early stop for success, and the run stops early only when criterion 1 fails.
* **A failure** of H1 means the test did not detect a correlation. It is reported with the one sided 95 percent upper bound, the largest correlation the data leave plausible.
* **The expected result,** written before launch, is that H1 does not pass. Megacaps absorb headlines within minutes, the pipeline adds Finnhub's delay and up to a minute or two of batching, and most earnings come after the close.

The test is fixed in [section 11.5 of the plan](../PLAN_V2.md#115-the-confirmatory-test).

## Success criteria

1. **The run is valid.** `evaluate` prints one verdict for each check, and the run is invalid when any fails.
   1. `coverage stability`. The free feed is a partial tape, which the owner kept, so the check asks that its coverage stays stable. The baseline is the median ratio of recorded to official volume in the gate session, 12 October 2026, the earliest full session with a coverage record before launch, whether or not its recording was clean, and the first clean full session at or after sessions 10, 20 and 40 must keep its median ratio within a factor of 2 of it, either way. The records come from `tape-check` runs in `data/v2/checks`, so each of those reports must actually be run.
   2. `lookahead audit`, no forecast row with an input received after its minute and no score line received before its `scored_at`.
   3. `labels recomputed`, every stored label equal to the label recomputed from `stock_candles`, which are kept for the life of the run for this.
   4. `runs identical`, S1 and S2 with identical run minutes, labels and `linear_prices` rows.
   5. `invariants`, no `invariant_violation` event in either run.
   6. `one config prompt and model`, the config hash, prompt hash, schema hash and one model id per scorer recorded at launch covering every model call, every heartbeat `news.db` still holds and every S2 minute. Heartbeats are pruned after 14 days, so S2's `hash_mismatch` minutes carry this check over the whole window.
   7. `spend within cap`, the key ledger's spend of both models together, each call at its cost or else its reservation, at or under 20 dollars in every UTC month of the run.
   8. `local only checks`, no failed `local_only` check and a passed one in each run between the previous close and the close of every session of the window. A session in which the stock process never ran therefore fails this check.
   9. `code commits listed`, a commit recorded at launch and at every later start. A commit marked dirty is listed but cannot be reproduced, so restarts come only from a committed tree.

   A fallback share above 25 percent of the items routed to Haiku in session is a caveat, not a failure, and so is a day that raised the spend alert.
2. **The confirmatory news skill,** H1 and then H2, as above.
3. **Descriptive results.** Every predictor's skill over all rows and over event rows, S2 against S1, `linear_news` against `linear_prices`, `haiku` against `finbert_matched`, `sonnet` against `haiku_official`, `news_company` and `news_macro`, calibration slopes, the twin test per run, any deaths and data quality, meaning the healthy fraction, run minutes lost to news health by cause, channel outage minutes, receipt to scoring lag per scorer, items per source and the fallback share.
4. **The outcomes fixed by arithmetic** matched, as the next section sets out.
5. **Version 1 undisturbed.** Runs A to D kept running and `test_v1_untouched` passes. `evaluate` does not check this, so the write up records it by hand.
6. **Cost.** About 1 to 5 dollars a month for both models together and never above the cap, reported per model.

The project succeeds when criterion 1 holds and the criterion 2 result is reported, whatever it shows. The criteria are in [section 11.6 of the plan](../PLAN_V2.md#116-success-criteria).

## Outcomes fixed by arithmetic

These check the accounting, not skill, apply to both runs and use the planning volatilities of the plan, 20 percent a year for QQQ up to 60 for TSLA, with three quarters of daily variance inside the session. The trading gate has no margin, `margin_sigma = 0` by the owner's decision of 12 October 2026, so the required edge is the round trip cost itself, and a learner opens on its own whenever its prediction clears it. Under the holding rule the owner approved on 13 October 2026, a learner holding a position keeps it while its prediction points its way by more than the holding cost, nothing for a long and about 0.006 bps of borrow for a short over 30 minutes, and flips only when the prediction points the other way by more than the full required edge of the new side, so positions last longer and round trips are fewer.

| Symbol | Round trip bps | Round trip dollars at 500 | 30 minute sd bps | Required edge in label sd |
| --- | --- | --- | --- | --- |
| QQQ | 3.6 | 0.18 | 30 | 0.12 |
| MSFT | 4.6 | 0.23 | 33 | 0.14 |
| AAPL | 4.6 | 0.23 | 38 | 0.12 |
| GOOGL | 4.6 | 0.23 | 45 | 0.10 |
| AMZN | 4.6 | 0.23 | 45 | 0.10 |
| META | 6.6 | 0.33 | 53 | 0.12 |
| NVDA | 4.6 | 0.23 | 68 | 0.07 |
| TSLA | 6.6 | 0.33 | 91 | 0.07 |

The learner check measured one learner's predictions at a root mean square of about 0.07 label standard deviations with no signal. If they are roughly normal, a decision clears the cost about 5 percent of the time on MSFT, 8 to 9 percent on QQQ, AAPL and META, 14 percent on GOOGL and AMZN and 30 to 33 percent on NVDA and TSLA, out of about 11 decisions a session inside the open window while flat, so most learners make one or two own opens a session and the NVDA and TSLA learners more. Every own open restarts the forced timer, so own trades both add round trips and crowd out forced ones. Under the holding rule a learner keeps a position at about a third of its decisions, so it is flat, and can open on its own, less often.

`evaluate` judges each outcome as matched, not matched or not reached yet, with these thresholds, which the owner confirms before launch.

* **First trades** in session 3, at session minutes 60 to 270, which is `10:30` to `13:30` ET plus up to 29 minutes of waiting for a phase minute.
* **Round trips per session,** reported only. The forced timer alone gives at most about two round trips per full session from session 4, fewer when a position is kept past the timer, and with no margin own trades come beside it and restart it. In a one off synthetic check of 13 October 2026, a scratch script that is not in the repository, built on the learner check's generator with no signal and the committed settings, a no skill learner whose predictions spread about 0.07 label standard deviations made about 2.3 round trips a session under the holding rule, 1.4 forced and 0.9 its own, held about 80 minutes on average, against about 2.6 held about 36 minutes under the version 1 rule. `evaluate` reports each run's mean round trips per full session, all, forced and own, with the mean minutes they were held, in its round trips table and judges none of them, as it does the own trade share.
* **Burn** within 25 percent of plan, where the plan is the rent and the cost of the round trips each learner actually made, 1.68 dollars of rent plus about 0.18 to 0.33 per round trip, 2.18 dollars a session at two round trips, so own trades and longer holds change the dollars but not the check. Keeping a position longer costs nothing more apart from a short's borrow, about 0.006 bps per 30 minutes.
* **Founder equity.** The learners' mean equity equals 10,000 dollars less rent of 0.0043081 dollars per usable minute of the home symbol and the round trips paid, within twice the square root of the horizons their positions were held times 50 bps of the 500 dollar notional, each closed trade counting its session time held over 30 minutes, so a position carried overnight counts the session minutes it was held and not the night, since under the holding rule one round trip can last several horizons. That is about 43 dollars at 75 trades of one horizon each and about 76 at the 230 horizons a no skill learner held by the cutoff in the synthetic check, with a floor of one cent. Counting horizons rather than round trips follows from the holding rule, and question 15 of the plan asks the owner to confirm it before launch.
* **No deaths.** The death threshold is about 2 dollars and the burn about 2.2 a session, so a death would take about 4,500 sessions.
* **Flat at every close,** except where a close was rejected because its symbol was not healthy. A position carried over a close is excused only when that symbol's candle at the close is invalid, and a missing candle row, meaning the process was down, is not excused.
* **Own trades,** descriptive only. With no margin the plan's outcome of at least 95 percent of learner opens forced no longer applies, and `evaluate` instead reports the learner opens, how many the timer forced and how many a learner made on its own, the own trade share, the share of decisions that cleared the required edge and the prediction spread in label standard deviations, without a matched word. A flip counts as an open and a decision that kept a position does not, and the share that cleared the edge counts every decision against the full edge an open needs, kept positions included. The holding rule leaves a learner flat less often, so its own share fell to about 0.38 in the synthetic check from about 0.45 under the version 1 rule.
* **Pairing.** S1 and S2 predictions identical until S2's first training step with a nonzero news input, expected in session 2 or later.

The planned rows and the arithmetic behind them are in [section 12 of the plan](../PLAN_V2.md#12-outcomes-fixed-by-arithmetic), and `test_v2_arithmetic` works them by hand.

## Power

`scripts/v2_power.py` simulates the news predictor of S2 for the eight symbols on synthetic data only, with zero inflated scores decaying over 120 session minutes, a macro part shared by every symbol, and fat tailed labels with clustered volatility, using planning values never fitted to recordings. The key unknown is c, the correlation between two symbols' predictions at one minute, which rises with the share of macro news. Each run is judged by the same estimator `evaluate` uses. The full run of 2,000 runs per scenario gave these figures on 12 October 2026, with prompt_v2's strengths drawn as `score_store.strength` gives them, a uniform confidence and the horizons and surprise levels of the probe answer key. H1 then H2 is the share of runs where both pass, with no news value beyond the price scores planted against them.

| Sessions | c | Null rejection | Detectable | Power 90 | Power at 0.075 | Power at 0.11 | H1 then H2 at 0.075 | H1 then H2 at 0.11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.2 | 0.0135 | 0.106 | 0.118 | 0.60 | 0.86 | 0.53 | 0.83 |
| 20 | 0.5 | 0.0110 | 0.134 | 0.152 | 0.39 | 0.70 | 0.33 | 0.65 |
| 20 | 0.8 | 0.0105 | 0.157 | 0.201 | 0.25 | 0.49 | 0.20 | 0.44 |
| 20 | 1.0 | 0.0120 | 0.172 | 0.274 | 0.19 | 0.36 | 0.14 | 0.31 |
| 40 | 0.2 | 0.0105 | 0.075 | 0.081 | 0.87 | 0.99 | 0.84 | 0.98 |
| 40 | 0.5 | 0.0085 | 0.095 | 0.104 | 0.67 | 0.93 | 0.64 | 0.91 |
| 40 | 0.8 | 0.0115 | 0.112 | 0.126 | 0.49 | 0.81 | 0.46 | 0.80 |
| 40 | 1.0 | 0.0125 | 0.122 | 0.151 | 0.39 | 0.69 | 0.36 | 0.67 |
| 60 | 0.2 | 0.0145 | 0.062 | 0.065 | 0.96 | 1.00 | 0.95 | 1.00 |
| 60 | 0.5 | 0.0120 | 0.078 | 0.082 | 0.83 | 0.99 | 0.81 | 0.98 |
| 60 | 0.8 | 0.0080 | 0.092 | 0.101 | 0.67 | 0.94 | 0.65 | 0.93 |
| 60 | 1.0 | 0.0110 | 0.100 | 0.116 | 0.58 | 0.87 | 0.55 | 0.86 |

* **Detectable** is 3.61 times the root mean square standard error under the null, the figure `evaluate` prints from the realized error. At 40 sessions it runs from 0.075 to 0.122, the plan's figures.
* **Power 90** is the true correlation the test finds in 90 percent of runs, 0.081 to 0.151 at 40 sessions.
* **Null rejection** stays between 0.8 and 1.5 percent over 2,000 runs, within sampling noise of 1 percent, so the test holds its level. H2 alone rejected 1.0 to 1.3 percent of the null runs at 40 sessions with these exogenous price scores.
* **Confounds.** With news that only repeats a momentum score with skill of its own, H1 alone passed in 6 to 15 percent of runs at a skill of 0.05 and a correlation with news of 0.5 and in 35 to 81 percent at 0.10 and 0.7, at 40 sessions, while H1 then H2 passed in 0.4 to 1.5 percent.
* Power at a true 0.075 runs from 0.87 at c of 0.2 to 0.39 at c of 1.0, a little below the figures with version 1's strengths, 0.88 to 0.44, because prompt_v2's horizon and surprise weights spread the strengths more. The plan's first estimate of 35 to 40 percent came from a sparser design, about one impact a session. The simulation's four impacts a session is a planning value, and power falls when news is sparser, while the detectable figure does not move.
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

* A pass of H1 shows that the scored news moved with the 30 minute returns that followed it on these eight symbols during this window, at this system's delays. Only a pass of H2 as well shows that the news carried information beyond the 10 price features. A failure of H1 shows only that the test did not detect a correlation, bounded by the upper confidence bound, and then H2 is not run.
* S2 against S1 cannot tell news without information from a learner too slow to use it. `scripts/v2_news_check.py` plants synthetic news of known strength in the learner check and prints, before launch, how often S2 would beat S1, the power curve that frames a null result. Its full run of 12 October 2026, 24 seeds of 40 sessions with prompt_v2's strengths, found the learners do pick up planted news, extracting about 0.86 of it by the last third at a matched true correlation near 0.1, while S2 beat S1 at the 1 percent level in 10 of 24 seeds there, in none at 0.05, and in none when the price absorbs news within about 15 minutes. So a null S2 against S1 says nothing about news correlations of 0.05 or less, as [section 6.4 of the plan](../PLAN_V2.md#64-learning) sets out with the whole curve.
* Fills come only from prints traded at or after an order's due time, and the confirmatory tests read forecasts and labels alone, never fills, so the fill rule cannot change them.
* Haiku against FinBERT and Sonnet against Haiku are descriptive. Sonnet sees only about 5 to 20 official items a day, so only a large difference could show.
* Every result rests on one set of scores from each model, and a fresh rerun would differ by about what the probes measure.
* News health selects minutes, so a scoring backlog around FOMC, CPI or earnings could remove minutes near the biggest news, which the report of lost minutes by cause makes visible.
* Minutes lost to news health are reported by cause, but `evaluate` does not report their overlap with scheduled releases, because the release calendar lives only in the news config, which it does not read.
* Event rows use the item tickers recorded in `news_seen`, received in the 30 minutes before the minute, and QQQ's event rows count only items tagged QQQ.

The honest limits are listed in [section 18 of the plan](../PLAN_V2.md#18-risks-and-honest-limits), and the results are written up in [RESULTS_V2_TEMPLATE.md](RESULTS_V2_TEMPLATE.md).
