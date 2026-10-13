# Results template for version 2

This file holds the empty tables for writing up runs S1 and S2. Results are written up at sessions 10 and 20, which are descriptive looks, and at the cutoff of 15,600 run minutes, about session 41, which is the only result that counts. Every run is reported whatever it shows, no setting changes during the window, and only the confirmatory test carries a pass label. The criteria were fixed before launch in [section 11 of the plan](../PLAN_V2.md#11-evaluation-and-success-criteria-fixed-before-launch), and what they can and cannot show is in [v2_evaluation.md](v2_evaluation.md).

A filled copy holds figures derived from the stock feed, so it never enters the repository. Copy this file to the gitignored `data/v2/results` folder, such as `data/v2/results/session_10.md`, and fill the copy there. Most numbers come from `evaluate`, which freezes every number at the cutoff, so running it again later gives the same rows.

```bash
mkdir -p data/v2/results && cp docs/RESULTS_V2_TEMPLATE.md data/v2/results/session_10.md
.venv-v2/bin/btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml
.venv-v2/bin/btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml --detail
.venv-v2/bin/btc_survivor_v2 status configs/v2/s1_prices.toml configs/v2/s2_news.toml --detail
.venv-v2/bin/btc_survivor_v2 news-status --days 14 --detail
```

Write timestamps and negative numbers in inline code, such as `2026-10-19T13:30Z` or `-0.012`, because the documentation test reads a bare colon or minus sign in prose as a style error. The commands are described in [v2_operations.md](v2_operations.md).

## Run header

The settings below are fixed for the life of the window and come from each run's `meta`. Add a hardware row whenever the runs move to another machine.

| Setting | S1 prices | S2 news |
| --- | --- | --- |
| Config | `configs/v2/s1_prices.toml` | `configs/v2/s2_news.toml` |
| Launch, `launch_ts` in `meta` |  |  |
| First session date |  |  |
| Cutoff minute, or the newest run minute at a look |  |  |
| Master seed | 12345 | 12345 |
| Experiment hash |  |  |
| News config hash |  |  |
| Prompt and schema hashes |  |  |
| Haiku model | `claude-haiku-5-5` | `claude-haiku-5-5` |
| Sonnet model and thinking setting, or disabled with the owner's reason |  |  |
| Prompt and schema files | `prompt_v2.txt` and `schema_v2.json` | `prompt_v2.txt` and `schema_v2.json` |
| News start sequence |  |  |
| Learning rate | 0.0001 | 0.0001 |
| Margin, `margin_sigma` | 0 | 0 |
| `feed_stale_after_seconds` and `symbol_stale_after_seconds`, set from the gate session |  |  |
| Population | 24 learners and 24 coin flip twins | 24 learners and 24 coin flip twins |
| FinBERT revision and model hash |  |  |
| Code commit at launch |  |  |
| Hardware, Python, numpy, anthropic and onnxruntime at launch |  |  |
| Later hardware, with the date of the move |  |  |

Every setting is explained in [section 15 of the plan](../PLAN_V2.md#15-configuration).

## Interruptions

Record every stretch longer than a few minutes in which a session minute was lost or a process was down, such as a sleeping Mac, a feed outage, a halt of every symbol, a news outage, a restart or a move. The `feed_events`, `news_minutes` and `collector_events` tables hold the exact times.

| Date, UTC | Process | What happened | Run minutes lost, by cause | Positions flattened |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

Price health and news health are described in [section 4.5](../PLAN_V2.md#45-price-health-without-an-order-book) and [section 7.8 of the plan](../PLAN_V2.md#78-news-health).

## Tape coverage

The free feed is a partial tape, so each row records how much of it the feed carried. One row for the gate session, whose median ratio of recorded to official volume is the baseline, and one for the first clean full session at or after sessions 10, 20 and 40. The per symbol ratios stay in `data/v2/checks`.

| Check | Session date | Clean | Median ratio against the gate session | Within a factor of 2 |
| --- | --- | --- | --- | --- |
| Gate session, the baseline |  |  | 1 | baseline |
| At or after session 10 |  |  |  |  |
| At or after session 20 |  |  |  |  |
| At or after session 40 |  |  |  |  |

The gate session is the baseline whether or not its recording was clean. The report is specified in [section 4.9 of the plan](../PLAN_V2.md#49-tape-coverage-check).

## Criterion 1, the run is valid

Every check must pass, otherwise the run is reported invalid with the reason. Copy the verdict and any problem `evaluate --detail` prints.

| Check | Verdict | Problems |
| --- | --- | --- |
| 1.1 `coverage stability` |  |  |
| 1.2 `lookahead audit` |  |  |
| 1.3 `labels recomputed` |  |  |
| 1.4 `runs identical` |  |  |
| 1.5 `invariants` |  |  |
| 1.6 `one config prompt and model` |  |  |
| 1.7 `spend within cap` |  |  |
| 1.8 `local only checks` |  |  |
| 1.9 `code commits listed`, with every commit listed |  |  |

Caveats, never failures, are a fallback share above 25 percent of the items routed to Haiku in session and any day that raised the spend alert. The checks are listed in [section 11.6 of the plan](../PLAN_V2.md#116-success-criteria).

## Criterion 2, the confirmatory test

A fixed sequence at one familywise level of 1 percent. H1 is the predictor `news` in S2 over the window, passing when t exceeds 2.33. Only if H1 passes at the cutoff is H2 run, the partial test with `news` and the label each residualised on the 10 stored price z scores within each symbol, passing when its t exceeds 2.33. Before the cutoff every row is descriptive.

| Test | Look | Rows | Minutes | Symbols, and any left out | Correlation | Standard error | t | Passed | Detectable correlation | Upper bound, one sided 95 percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | Session 10, descriptive |  |  |  |  |  |  | descriptive |  |  |
| H1 | Session 20, descriptive |  |  |  |  |  |  | descriptive |  |  |
| H1 | Cutoff |  |  |  |  |  |  |  |  |  |
| H2 | Cutoff, only after H1 passed |  |  |  |  |  |  |  |  |  |

A failure is reported with its upper bound, the largest correlation the data leave plausible. A pass of H1 alone says the scored news moved with the returns that followed, and a pass of H2 as well says it carried information beyond the price features. The tests are fixed in [section 11.5 of the plan](../PLAN_V2.md#115-the-confirmatory-test).

## Criterion 3, descriptive results

Everything in this section is descriptive and carries no pass label. The definitions are in [section 11.3 of the plan](../PLAN_V2.md#113-squared-error-twins-events-and-calibration).

### Skill of every predictor

| Predictor | Run | Rows | Correlation | t | Detectable | Event rows | Event correlation | Event t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `population` | S1 |  |  |  |  |  |  |  |
| `population` | S2 |  |  |  |  |  |  |  |
| `linear_prices` | S1 |  |  |  |  |  |  |  |
| `linear_prices` | S2 |  |  |  |  |  |  |  |
| `linear_news` | S2 |  |  |  |  |  |  |  |
| `news` | S2 |  |  |  |  |  |  |  |
| `news_company` | S2 |  |  |  |  |  |  |  |
| `news_macro` | S2 |  |  |  |  |  |  |  |
| `haiku` | S2 |  |  |  |  |  |  |  |
| `finbert` | S2 |  |  |  |  |  |  |  |
| `finbert_matched` | S2 |  |  |  |  |  |  |  |
| `sonnet` | S2 |  |  |  |  |  |  |  |
| `haiku_official` | S2 |  |  |  |  |  |  |  |

Pooled skill is defined in [section 11.2 of the plan](../PLAN_V2.md#112-pooled-skill).

### Squared error comparisons

A ratio below 1 and a positive t favour the first forecast.

| First | Second | Rows | Squared error ratio | t |
| --- | --- | --- | --- | --- |
| `population` in S1 | zero |  |  |  |
| `population` in S2 | zero |  |  |  |
| `population` in S1 | `linear_prices` in S1 |  |  |  |
| `population` in S2 | `population` in S1, paired rows |  |  |  |
| `linear_news` | `linear_prices` |  |  |  |

The comparisons are defined in [section 11.3 of the plan](../PLAN_V2.md#113-squared-error-twins-events-and-calibration).

### Model comparisons

Each pair uses exactly its paired rows, so both sides see the same items at the same times.

| First | Second | Rows | First correlation | Second correlation | Event rows | First event correlation | Second event correlation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `haiku` | `finbert_matched` |  |  |  |  |  |  |
| `sonnet` | `haiku_official` |  |  |  |  |  |  |

The score streams are defined in [section 8.7 of the plan](../PLAN_V2.md#87-finbert-and-the-score-streams).

### Twins

Learner and twin trades from one decision, paired by slot, symbol and decision time, compared on gross profit with one block per session.

| Run | Pairs | Sessions | Mean difference per session | t | Threshold | Learners' share of long opens |
| --- | --- | --- | --- | --- | --- | --- |
| S1 |  |  |  |  |  |  |
| S2 |  |  |  |  |  |  |

Stocks drift up a little over the window, so the share of long opens stands beside the result, as [section 11.3 of the plan](../PLAN_V2.md#113-squared-error-twins-events-and-calibration) explains.

### Own trades

With no margin a learner opens on its own whenever its prediction clears the round trip cost, so this is reported and never judged.

| Run | Learner decisions | Learner opens | Forced opens | Own opens | Own share | Share of decisions that cleared the required edge | Prediction spread, label sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 |  |  |  |  |  |  |  |
| S2 |  |  |  |  |  |  |  |

The gate is described in [section 12 of the plan](../PLAN_V2.md#12-outcomes-fixed-by-arithmetic).

### Calibration

The slope of label on prediction within each symbol, for the bps predictors.

| Predictor | Run | AAPL | MSFT | NVDA | AMZN | GOOGL | META | TSLA | QQQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `population` | S1 |  |  |  |  |  |  |  |  |
| `population` | S2 |  |  |  |  |  |  |  |  |
| `linear_prices` | S2 |  |  |  |  |  |  |  |  |
| `linear_news` | S2 |  |  |  |  |  |  |  |  |

Calibration is described in [section 11.3 of the plan](../PLAN_V2.md#113-squared-error-twins-events-and-calibration).

### Data quality

| Measure | Value |
| --- | --- |
| Session minutes since launch |  |
| Run minutes |  |
| Healthy fraction of symbol minutes |  |
| Run minutes lost to `no_heartbeat` |  |
| Run minutes lost to `heartbeat_old` |  |
| Run minutes lost to `heartbeat_unhealthy` |  |
| Run minutes lost to `hash_mismatch` |  |
| Lost minutes overlapping a scheduled release, counted by hand from the calendar |  |
| Channel outage minutes, the largest three channels |  |
| Receipt to scoring lag, median and 90th percentile, per scorer |  |
| Items per source |  |
| Fallback share of the items routed to Haiku in session |  |
| Fill print after the due time, median and largest, from `print_after_due_ms` in `fills` |  |
| S2 fills inside a news event window, the first line of an item bearing on the symbol, tagged with it, with MACRO or, for QQQ, with any company, in the 30 minutes before the decision, from `news_window` in `fills` |  |
| Deaths, with bot, time and cause |  |

`evaluate` does not read the release calendar, so the overlap with scheduled releases is counted by hand. Data quality is listed in [section 11.6 of the plan](../PLAN_V2.md#116-success-criteria).

## Criterion 4, outcomes fixed by arithmetic

| Outcome | Expected | S1 | S2 | Matched |
| --- | --- | --- | --- | --- |
| First trades | session 3, session minutes 60 to 270 |  |  |  |
| Round trips per full session from session 4, forced and own, from the round trips table | about 2 from the timer alone, own trades beside them, reported only |  |  | reported |
| Burn | the rent and the round trips each learner actually made, within 25 percent, 2.18 dollars a session at two round trips |  |  |  |
| Founder equity | within about 43 dollars of plan at 75 trades |  |  |  |
| No deaths | 0 |  |  |  |
| Flat at every close | none carried while healthy |  |  |  |
| Pairing, both runs together | first difference in session 2 or later |  |  |  |

The arithmetic is in [section 12 of the plan](../PLAN_V2.md#12-outcomes-fixed-by-arithmetic).

## Criterion 5, version 1 undisturbed

`evaluate` does not check this, so record it by hand.

| Check | Result |
| --- | --- |
| Runs A to D kept their heartbeats through the window |  |
| `test_v1_untouched` passes at the cutoff commit |  |
| Version 1 dashboard stayed on `127.0.0.1:8050` |  |

The isolation rules are in [section 3 of the plan](../PLAN_V2.md#3-scope-isolation-from-version-1-and-module-layout).

## Criterion 6, cost

Spend per UTC month from the key ledger, each call at its cost or else its reservation.

| Month | Haiku, dollars | Sonnet, dollars | Together | Cap | Days that raised the spend alert | Charged to the included credits |
| --- | --- | --- | --- | --- | --- | --- |
| October 2026 |  |  |  | 20.00 |  |  |
| November 2026 |  |  |  | 20.00 |  |  |
| December 2026 |  |  |  | 20.00 |  |  |

The expected cost is about 1 to 5 dollars a month for both models, as [section 8.10 of the plan](../PLAN_V2.md#810-expected-cost) estimates.

## Probes

Per field agreement of each look's daily Haiku probe run with the first run and with the previous day's.

| Date | Against | Ticker | Event type | New | Direction | Surprise | Horizon | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | first run |  |  |  |  |  |  |  |
|  | previous day |  |  |  |  |  |  |  |

Direction agreement below 80 percent is flagged and changes nothing, as [section 8.8 of the plan](../PLAN_V2.md#88-lookahead-non-determinism-and-probes) sets out.

## Summary

Write a few sentences on whether the run was valid, the confirmatory result with its upper bound, and anything the descriptive results or the interruptions suggest, without choosing or tuning anything from them. The honest limits that frame every result are in [section 18 of the plan](../PLAN_V2.md#18-risks-and-honest-limits).
