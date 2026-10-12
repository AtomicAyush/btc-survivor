# Version 2 overview

Version 2 of BTC Survivor paper trades seven large US technology stocks, AAPL, MSFT, NVDA, AMZN, GOOGL, META and TSLA, plus the QQQ fund, and asks one question, whether news scored by a language model forecasts the next 30 minutes of returns. It places no real order anywhere and holds no code that could place one. It runs beside version 1, whose four Bitcoin runs A to D keep running untouched, and nothing here changes how they behave. The approved design is [PLAN_V2.md](../PLAN_V2.md). This page and the four guides beside it describe the code as built in the week before the launch on Monday 19 October 2026, and say where the code reads the plan more precisely or differs from it.

## The rules carried over from version 1

* There are no real orders and no order placement code. The order scan in `tests/test_no_orders.py` looks for broker names and order fields in every file.
* There is no lookahead. A price counts from the moment this system received its print, and a scored headline from the moment the stock process received its score, never from a vendor or article timestamp.
* No backtest or replay chooses, tunes or ranks anything. Every setting is fixed before launch, by reasoning or by synthetic checks that never read market data.
* Nothing trades or learns on stale prices or while the news pipeline is down.
* The books always balance, and a restart is not a death.
* Everything inside is UTC, everything random is seeded from one master seed, 12345, and each module stays small.

These rules open [the plan](../PLAN_V2.md#1-key-decisions), and the version 1 plan they come from is [PLAN.md](../PLAN.md).

## What runs

Three version 2 processes share the Mac with runs A to D, and the stock process replaces the raw recorder at launch.

| Process | Command | Network | Writes |
| --- | --- | --- | --- |
| news | `btc_survivor_v2 news configs/v2/news.toml` | SEC EDGAR, the Federal Reserve, BLS, Finnhub REST and the Anthropic API | `data/v2/news/news.db` and the key ledger `~/.config/btc_survivor/llm_ledger.db` |
| stocks | `btc_survivor_v2 stocks configs/v2/s1_prices.toml configs/v2/s2_news.toml` | the one Finnhub websocket the key allows | `data/v2/s1_prices/stocks.db`, `data/v2/s2_news/stocks.db` and `data/v2/raw_stocks` |
| dashboard | `btc_survivor_v2 dashboard configs/v2/s1_prices.toml configs/v2/s2_news.toml` | none, it listens on `127.0.0.1:8060` only | nothing |
| raw recorder, until launch | `python scripts/v2_record_stocks.py configs/v2/s1_prices.toml` | the one Finnhub websocket | `data/v2/raw_stocks` |

The news process and the stock process never talk to each other directly. The news process writes `news.db`, and the stock process reads it read only once a second.

```text
SEC EDGAR, Federal Reserve, BLS, Finnhub REST
        |
        v
news process: collector -> news_items, then Haiku, Sonnet and FinBERT -> item_scores and score_impacts
        |
        |  news.db, opened read only once a second by the puller
        v
stock process: Finnhub websocket prints -> one shared market layer -> S1 engine and S2 engine
        |                                 \-> raw recording of every price line and news line
        v
s1_prices/stocks.db and s2_news/stocks.db -> evaluate, status, export and the dashboard, all read only
```

The processes and their files are set out in [section 14 of the plan](../PLAN_V2.md#14-runtime-and-processes), and starting and stopping them in [v2_operations.md](v2_operations.md).

## The two runs

Both runs live in the one stock process and share one market layer, so they see exactly the same candles, prices, health and labels.

* **S1 prices** feeds each network the 10 price features of a symbol and two news inputs pinned at zero.
* **S2 news** feeds the same 10 price features and the two news inputs computed from the scored news, one for company news and one for macro news.

Each run has 24 learners and 24 coin flip twins. Every learner runs one network of 12 inputs and 16 hidden units, 225 parameters, predicts all eight symbols every usable minute and learns from all eight, but trades only its home symbol, the one at position `slot % 8` in the config, so three learners trade each symbol. The learner in each slot of either run is drawn with the same generator calls, with two zero columns appended for the news inputs, so S1 and S2 make bitwise identical predictions until S2 first trains on a nonzero news input. Every later difference between them comes from news. There is no evolution and no replacement, and a learner that dies leaves its slot empty, although none is expected to die.

Each learner trades a fixed 500 dollars on a start equity of 10,000 dollars, with a fixed 30 session minute horizon, a margin of 0.25 label standard deviations of its symbol, a stop at 3 label standard deviations and a forced trade after 3 healthy session hours without one. It decides at the session minutes where `k % 30 == slot % 30`, which gives it 13 decision minutes a full session. The learning rate is 0.0001, fixed on synthetic data as [v2_evaluation.md](v2_evaluation.md#the-learning-rate) explains. The runs and predictors are defined in [section 10 of the plan](../PLAN_V2.md#10-runs-and-predictors).

## Market hours

Only the regular session counts, `09:30` to `16:00` New York time, or to `13:00` on an early close. Version 2 keeps its own holiday table for 2026 to 2028 in `v2/sessions.py`, so nights, weekends and holidays never look stale, and every series, label and window counts session minutes on one index k that skips them. Opens, flips and forced opens are allowed from `10:00` to `15:25` ET, or `12:25` on an early close. From `15:55` ET every open position gets a closing order with exit reason `session_close`. A position that survives the close, because its symbol was halted into it or the process was down, closes at that symbol's first healthy minute of the next session with exit reason `session_flatten` and an `overnight_carry` event. No label ever spans a close. The calendar and its rules are in [section 4.4 of the plan](../PLAN_V2.md#44-market-calendar-and-the-session-clock).

## Prices without an order book

Finnhub's free websocket sends trade prints and nothing else, no quotes and no depth. Each print joins the one minute candle of the minute in which this system received it, and only prints that arrived at most `late_print_seconds` after their vendor time count toward prices. The reference price at each minute boundary is the median of the last 5 eligible prints, and it drives features, labels, marks and stops.

Each symbol has its own price health, closed, stale, recovering or healthy. A whole feed problem, such as 5 seconds without any message in session or a feed running late, makes every symbol stale and asks for a reconnect. A symbol without an eligible print for 15 seconds goes stale alone, as a halt does. An order fills at the first eligible print of its symbol received at or after the decision plus 1,000 ms and a seeded jitter of up to 300 ms, moved against the bot by a fixed half spread allowance plus 1 bp. Prints, candles and health are in [section 4 of the plan](../PLAN_V2.md#4-data-feed-and-market-hours) and fills in [section 5.1](../PLAN_V2.md#51-fill-model).

## A minute in the stock process

At each minute boundary inside a session the stock process does these steps once for both runs, in this order.

1. The shared market layer judges news health for the minute, then for each symbol closes the candle, takes the reference price, judges price health, computes and standardizes the features, matures the label pushed 30 minutes earlier when the symbol's epoch did not change, and pushes a new label record when the minute is usable.
2. S2 computes its two news inputs per symbol from the score lines it received before the boundary, and S1 stores zeros.
3. Gap flattens and session flattens go out per symbol.
4. Each learner pays rent and borrow, marks its position, checks death and its stop, trains on the samples that matured, predicts the eight symbols and decides at its phase minute. Each twin follows its learner.
5. From `15:55` ET the session close orders go out.
6. Candles, forecasts, labels, decisions, the benchmark and snapshots are written to each run's own database.

A symbol minute is usable when the symbol is healthy and news health held. Rent, training, decisions, new label records and forecast rows need a usable minute, while marks, stops, the session close and the maturing of pending labels need only price health. A news outage therefore pauses learning and trading in both runs but never discards a label, bumps an epoch or flattens a position. The step is set out in [section 4.8 of the plan](../PLAN_V2.md#48-the-stock-minute-step).

## Costs

A round trip costs 3.6 bps for QQQ, 4.6 bps for AAPL, MSFT, NVDA, AMZN and GOOGL and 6.6 bps for META and TSLA, 0.18 to 0.33 dollars at 500 dollars, against about 190 bps for the version 1 retail Bitcoin round trip. That is two legs of half spread allowance plus 1 bp each, the SEC Section 31 fee and the FINRA trading activity fee on every sale, each rounded up to the cent, and short borrow at 1 percent a year. Rent stays the cost of capital of version 1, 1.16 dollars a calendar day spread over usable session minutes, about 1.68 dollars a full session. Shorts open whole shares only, so their notional runs from about 400 to 700 dollars. The limits a real account would meet on shorts, margin and day trading are in [v2_data_and_licensing.md](v2_data_and_licensing.md#shorts-margin-and-day-trading), and the cost model in [section 5 of the plan](../PLAN_V2.md#5-trading-costs-and-fills).

## News in brief

The news process polls SEC EDGAR for the `8-K` filings of the seven companies, the Federal Reserve's press releases and speeches, four BLS release feeds and Finnhub's company and general news. Each item is stamped when it arrives, deduplicated, tagged with the symbols it names or with MACRO, and routed. `claude-haiku-5-5` scores every routed item, `claude-sonnet-5-5` also scores the official items alone as a comparison, and FinBERT, a local model, scores every routed item as the fallback and the baseline. One cap in code of 20 dollars a month covers both models. The stock process reads finished scores, never calls a model, and records every score line it reads, so a replay needs only the recording. The whole pipeline is in [v2_news.md](v2_news.md).

## What version 2 measures

There is one confirmatory test. The predictor `news` in S2, the company plus macro input from the primary score stream, is judged by its pooled correlation with the next 30 minute return over the eight symbols, from session 2 to the cutoff of 15,600 run minutes, about 40 sessions, one sided at 1 percent with errors clustered by minute. Everything else is descriptive, including S2 against S1, Haiku against FinBERT and Sonnet against Haiku. The expected result, written before launch, is that the test does not pass, because megacaps absorb headlines within minutes and the pipeline adds delay. A synthetic power check puts the smallest correlation the test can detect at 40 sessions between about 0.075 and 0.122, depending on how much of the news is shared macro news. The run is valid only if every check of criterion 1 holds. The criteria, the power figures and what a failure means are in [v2_evaluation.md](v2_evaluation.md).

## Isolation from version 1

* Version 2 lives in its own git worktree, `~/code/btc_survivor_v2` on branch `feat/v2-stocks`, with its own environment `.venv-v2`, the package `btc_survivor.v2` and the console script `btc_survivor_v2`. Runs A to D import the main tree, so nothing built here can reach them.
* `test_v1_untouched` asserts that no version 1 module or run config changed since the fork, and `test_v1_isolation` that no version 1 module imports version 2 and the version 1 loader refuses every version 2 config.
* Every version 2 database carries `meta.kind`, stocks or news, and every reader refuses a database of another kind.
* The version 1 dashboard keeps `127.0.0.1:8050` and the version 2 dashboard takes `127.0.0.1:8060`, which refuses port 8050.
* Version 2 reuses version 1 modules unchanged, such as the clock, the label queue, the standardizer, the network and its optimizer, the account, the decision rule and the statistics helpers.

The isolation rules are in [section 3 of the plan](../PLAN_V2.md#3-scope-isolation-from-version-1-and-module-layout).

## Open items before launch

These were open on 12 October 2026, and each needs a step or an owner decision before the launch.

* **The tape verdict.** Whether Finnhub's free websocket carries the full consolidated tape is unverified until the tape check of the first clean full session, and every result depends on it.
* **Sonnet's thinking setting.** Anthropic's documentation read on 12 October 2026 says Sonnet 5.5 rejects `thinking` set to disabled at every effort level, so the request the plan specifies would be refused with a 400 and turn Sonnet off until restart. Until the owner chooses another setting, which needs a change to `config.py` and `llm_request.py`, or no Sonnet stream, `[llm.sonnet] enabled` is false in the committed config, so nothing is sent to Sonnet.
* **Pins.** The prompt and schema hashes, and the three FinBERT pins written by the export the owner must approve, go into `configs/v2/news.toml` before the news process restarts onto the scoring code. Pinning changes the news config hash, which is fine only before launch.
* **The calendar.** BLS listed no JOLTS release after 1 December 2026 when the calendar was copied, so those dates are added before launch if BLS lists them, and can never be added after.
* **Settings from the first sessions.** The delay rule sets `max_print_delay_seconds` and `late_print_seconds`, and the meanings of Finnhub's condition codes fill `price_excluded_conditions`.
* **The soak memory rule.** The two week synthetic soak ended 5.95 MB above its first hour against a 5 MB bound, from windows filling and weekend news kept in memory rather than a leak, and the owner decides how to read it, as [v2_operations.md](v2_operations.md#the-soak) explains.

The full list of decisions waiting for the owner is in [section 19 of the plan](../PLAN_V2.md#19-open-questions-for-the-owner).

## Honest limits

Version 2 measures skill, not profit. Even a predictor that passes the test would not cover costs on most symbols at the correlations the window can detect. The free feed may not be the full tape, there are no quotes, fills rest on a fixed allowance, shorts are kinder than a small real account, only the regular session is traded, so the overnight reaction to earnings is invisible, and every result rests on one set of scores from each model. Licensing allows no public page and no sharing of stock data. The limits are listed in [section 18 of the plan](../PLAN_V2.md#18-risks-and-honest-limits) and the licensing in [v2_data_and_licensing.md](v2_data_and_licensing.md).

## The version 2 guides

* [v2_data_and_licensing.md](v2_data_and_licensing.md) covers the sources, their terms, the owner's decisions on Finnhub headlines and on stock data in the assistant session, secrets, and the limits on shorts, margin and day trading.
* [v2_news.md](v2_news.md) covers collection, routing, scoring with Haiku, Sonnet and FinBERT, the spend cap, the score streams and news health.
* [v2_evaluation.md](v2_evaluation.md) covers the `evaluate` command, the criteria, the outcomes fixed by arithmetic, power and the learning rate.
* [v2_operations.md](v2_operations.md) covers running everything on the Mac, the checks before and during the window, launchd and later systemd.
* [RESULTS_V2_TEMPLATE.md](RESULTS_V2_TEMPLATE.md) holds the empty tables for writing up the window.
