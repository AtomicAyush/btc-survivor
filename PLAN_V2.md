# BTC Survivor version 2 plan

This is the design for version 2 of BTC Survivor. It adds seven large US technology stocks, AAPL, MSFT, NVDA, AMZN, GOOGL, META and TSLA, plus the QQQ fund, and it adds news scored by language models. Version 1, the approved design in [PLAN.md](PLAN.md), keeps running untouched. Its four Bitcoin runs A, B, C and D have been live since 11 October 2026, and nothing here may change how they behave, even after one of them restarts.

Every hard rule of version 1 carries over.

* There are no real orders and no order placement code anywhere.
* There is no lookahead. A feature or a scored headline counts from the moment this system received it or finished scoring it, never from a vendor or article timestamp.
* No backtest or replay chooses, tunes or ranks anything, so every setting is fixed before launch, from reasoning or from synthetic checks.
* Nothing trades or learns on stale data.
* The books always balance, and a restart is not a death.
* Everything inside is UTC, everything random is seeded from one master seed, and the implementation stays the simplest that meets the need, in small modules.

The owner has already decided the following.

* The universe is the seven stocks plus QQQ, with scored news.
* Trade prints and company news come from the Finnhub free plan on a data only key. Official news comes from SEC EDGAR, the Federal Reserve and BLS, which need no key, only a contact address in the request header.
* Scoring uses `claude-haiku-5-5` through the Anthropic API on every routed item, Finnhub and official alike. Local FinBERT also scores every item, as the fallback when the model is capped or fails and as a baseline predictor, so the run measures whether the model adds anything over FinBERT on the same items.
* Finnhub headlines and summaries go to the Anthropic API for that scoring. Finnhub's terms bar sharing its data with third parties without written approval, sending headlines to the Anthropic API arguably counts, and the realistic consequence is that Finnhub could suspend the free key, which also carries the price feed. The owner accepts that risk, and a written request to Finnhub would remove the doubt, as section 13.1 and question 3 set out.
* `claude-sonnet-5-5` also scores the official items and nothing else, SEC `8-K` filings with their exhibits, Federal Reserve releases and speeches and BLS releases, about 5 to 20 a day, with the same prompt and schema, as a separate stream and predictor that measures whether a stronger model predicts better than Haiku on those items. Opus is not used.
* Spend has one cap, 20 dollars a month in code on one ledger shared by both models, matching the 20 dollar monthly limit the owner set on the dedicated Anthropic Console workspace, and it is paid from the included API credits of the owner's Max plan. There is no daily cap. A day whose spend passes a quarter of the monthly cap raises a warning on the log and the private dashboard and stops nothing.
* Stock prices and anything derived from them may enter the assistant session that builds and checks this system, by the owner's decision of 12 October 2026, which lifted an earlier rule. The risk the owner accepts is the one of the headline decision, that Finnhub could suspend the free key, as section 13.3 sets out.
* The Finnhub key, the Anthropic key and the contact address are in `~/.config/btc_survivor/secrets.env`, outside the repository, as `FINNHUB_API_KEY`, `ANTHROPIC_API_KEY` and `CONTACT_EMAIL`. They are never printed or committed, and the contact address never appears in the repository or on any page.
* Every dashboard stays local only, for version 1 and version 2 alike. The version 1 dashboard on `127.0.0.1:8050` and the version 2 dashboard on `127.0.0.1:8060` are never tunnelled, shared or published, so no public page exists and no stock data, news or anything derived from them leaves this machine through a page.
* The end goal is a bot with broad market knowledge and current affairs that sizes its trades by confidence. Version 2 is the next honest step, the skill stage of the road map in section 11.7, and keeps a fixed 30 minute horizon and a fixed 500 dollar notional. Variable sizing and holding periods chosen from predictions belong to the later profit stage, which switches on only once a predictor passes the skill test.

Vendor facts, prices, limits and licensing come from the vendor research of Sunday 11 October 2026. Anything that research could not confirm, or that lies outside it, is marked unverified. Volatilities, spreads, print rates and similar market numbers are planning estimates, measured in the first live sessions. Version 1 has finished Phase 8 and has Phases 9 and 10 still to do.

## 1. Key decisions

| Decision | Version 1 | Version 2 choice | Why |
| --- | --- | --- | --- |
| Where the code lives | one tree on `feat/initial-bot` | a git worktree at `~/code/btc_survivor_v2` on `feat/v2-stocks`, with its own virtual environment, the package `btc_survivor.v2` and the console script `btc_survivor_v2` | runs A to D import the main tree through an editable install, so nothing built for version 2 can reach them |
| Version 1 files changed | not applicable | no module, only `pyproject.toml`, `tests/test_no_orders.py`, `tests/test_docs_style.py` and a new `.gitattributes` | `test_v1_untouched` and the unchanged version 1 suite prove version 1 did not move |
| Universe | BTC USD | AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA and QQQ | all Nasdaq listed under one licensing plan, QQQ as the market reference |
| Price data | keyless Coinbase book and trades | Finnhub free websocket trade prints, one minute candles by receive time | the only free live source found that claims consolidated prints on a key that cannot trade |
| Full tape | not applicable | a tape check from the raw recording against Nasdaq's consolidated daily volume gates Phase 13 and repeats at sessions 10, 20 and 40 | whether the free tier carries the full tape is unverified |
| Market hours | 24 hours | the regular session only, on a session minute clock with a local holiday table for 2026 to 2028 | nights, weekends and holidays never look stale or count as gaps |
| Overnight | not applicable | intraday only, opens from `10:00` to `15:25` ET, flat at `15:55` ET | no stop fires overnight, and no overnight gap, overnight borrow or dividend risk is modelled beyond the rare carry of section 4.7 |
| Fills | book walk after latency | the first eligible print at or after decision plus 1,000 ms plus seeded jitter, moved against the bot by a half spread allowance plus 1 bp | no quotes or depth, so a fixed allowance replaces the book |
| Half spread allowance | 1 bp extra slippage | 0.5 bp QQQ, 1.0 bp AAPL, MSFT, NVDA, AMZN and GOOGL, 2.0 bp META and TSLA, each plus 1 bp | at least three and a half times the implied half spread |
| Fees | Coinbase taker fee | no commission, SEC Section 31 at 20.60 dollars per million and FINRA TAF at 0.000195 dollars a share on every sale, each rounded up to the cent | rates read on 11 October 2026 outside the research, rechecked before launch, with the TAF charged through its pause |
| Shorts | 2 bps to open plus 0.5 bp an hour | whole shares with a one share minimum, borrow at 1 percent a year, the Rule 201 restriction | brokers do not short fractional shares, and megacaps lend near the general collateral rate |
| Size and rent | 500 dollars on 10,000, 1.16 dollars a day | unchanged size on one home symbol, rent 0.0043081 dollars per usable session minute, 1.68 per full session | the version 1 account and the same yearly cost of capital |
| Horizon | 30 minutes | 30 session minutes, the only horizon, with several horizons left to the profit stage of section 11.7 | 12 independent windows per session, and power is the binding limit |
| Gate margin and stop | 8 bps gene, 1 percent stop | a fixed margin of 0.25 and a stop of 3 label standard deviations of the symbol | 8 bps passes noise on 11 percent of TSLA decisions, and 1 percent is one TSLA standard deviation |
| Features | 11 from mids and the book | 10 per symbol from prints | no quotes or depth for spread and imbalance |
| Learners | 20 per run, evolving | 24 pooled learners per run, each predicting all 8 symbols and trading its home symbol, no replacement or mutation | eight times the samples, and no death is expected |
| Runs | A to D | S1 on prices and S2 on prices plus two news inputs, one 12 input network in both with S1's news inputs pinned at zero | identical code and parameters, so later differences come from news |
| News and scoring | none | EDGAR, Fed and BLS plus Finnhub news, Haiku 5.5 on every routed item, Finnhub and official, in a separate news process, FinBERT on every item as fallback and baseline | no model call in a trading minute step, and FinBERT on the same items measures what the model adds |
| Finnhub headlines | not applicable | sent to the Anthropic API for scoring, by the owner's decision | Finnhub's terms bar sharing with third parties without written approval, so Finnhub could suspend the free key, a risk the owner accepts and a written request to Finnhub would remove |
| Model comparison | not applicable | Sonnet 5.5 also scores the official items, meaning `8-K` filings, Fed releases and speeches and BLS releases, about 5 to 20 a day, as its own stream and predictor, and Opus is never used | measures whether a stronger model predicts better than Haiku on the same items, for about half a dollar to 3 dollars a month |
| Model spend | not applicable | one cap in code of 20 dollars a month on one ledger per key shared by both models, matching a 20 dollar monthly workspace limit, no daily cap, a warning when one day passes 5 dollars, each call reserved from an exact token count, and no failed call sent again | the cap in code cannot be passed, and the Console limit backs it |
| News timing | not applicable | visible when the stock process receives a score, decaying from when scoring finished | no vendor or article time places anything in time |
| Health | Coinbase feed health | price health per symbol for prices, epochs and labels, and news health, meaning the news pipeline is alive, gating rent, learning, decisions and new labels in both runs | never trade or learn on stale data, and S1 and S2 share their minutes |
| Confirmatory test | run A skill at day 28 | the pooled correlation of the news score with the next 30 minute return over 8 symbols, sessions 2 to 40, one sided at 1 percent, clustered by minute, with a detectable correlation from about 0.075 to about 0.12 as the macro share rises, and a failure reported with its one sided 95 percent upper bound | the one question version 2 adds |
| Dashboards and assistants | a local page | both dashboards local only on `127.0.0.1` with no tunnel, every command printing aggregates unless `--detail` is passed, and stock data allowed in the assistant session by the owner's decision of 12 October 2026 | the dashboards stay private, and the owner accepts the licensing risk of section 13.3 |
| Secrets | none | one reader of a mode 600 file outside the repository, redaction on every log handler and recorded line | the Finnhub token travels in the websocket URL |
| Variable sizing and exits | road map step 4 | still out, with a fixed 500 dollar notional and a fixed 30 minute horizon, until the profit stage of section 11.7 | no predictor has passed a skill test |

## 2. Where the area drafts disagreed

Five area drafts fed this plan. Where they disagreed, this plan picks one answer.

1. **Code.** A separate worktree with its own console script, never the main tree or the version 1 entry point.
2. **Finnhub calls.** The stock process owns the websocket and makes no REST call, and the news process makes every REST call, at most 30 in any rolling minute.
3. **Locks** live under `~/.config/btc_survivor`, so the main tree and the worktree see the same ones.
4. **One news process** with one SQLite writer runs collection and scoring.
5. **News reaches the engines** as recorded lines pulled once a second, so replay needs only the recording, and a score counts from its receipt and decays from when scoring finished.
6. **Thresholds** are 5 seconds for the whole feed and 15 per symbol, with a backoff from 5 seconds.
7. **Prices** are the median of the last 5 eligible prints, eligibility uses age on arrival plus excluded codes, and fills use one real print.
8. **One 30 minute horizon,** opens from `10:00` ET and a flatten at `15:55` ET. The 120 minute and one day horizons cannot confirm anything in 40 sessions.
9. **Bots** trade one home symbol at 500 dollars, with margin and stop in label standard deviations and per symbol costs instead of a flat 3 bps.
10. **Inputs** are 10 price features, no Bitcoin input, and two bounded news inputs with one half life of 120 session minutes, because 40 sessions cannot support the 13 proposed news features.
11. **One confirmatory test,** the news score, with Haiku against FinBERT and Sonnet against Haiku, each on the same items, descriptive.
12. **Smaller choices.** Batches flush at 20 items or 60 seconds, text timestamps, a `collector_events` table, no releases or event study tables, a forced timer from the first Active minute, the union of both learning rate candidate sets, 30 days of raw recordings, and one secrets module.

## 3. Scope, isolation from version 1 and module layout

**Order of work.** Version 1 Phases 9, docs and deploy, and 10, the soak, finish first on `feat/initial-bot`. Version 2 then runs as Phases 11 to 19, dated in section 16.

**A separate worktree.** `git worktree add ~/code/btc_survivor_v2 -b feat/v2-stocks feat/initial-bot`, with its own `.venv-v2`. Runs A to D import `~/code/btc_survivor/src` through the editable install in `.venv311`, and the version 1 dashboard rereads `dashboard.html` from that tree on every request, so any checkout there would change running processes. Version 2 data lives under the worktree's gitignored `data/v2/`, and the branch is pushed to the existing private repository. A merge into the main tree waits until the experiment ends, and runs A to D restart onto merged code only with the owner's approval.

**Version 1 files.** No version 1 module is edited.

* `pyproject.toml` gains the console script `btc_survivor_v2 = "btc_survivor.v2.__main__:main"` and an extra `v2news` with `anthropic`, `pydantic`, `onnxruntime` and `tokenizers`, each pinned exactly to a release at least two weeks old. A version 1 install and the Pi are unaffected.
* `.gitattributes` forces LF line endings on the model prompt file, so its bytes never change.
* `tests/test_no_orders.py` adds the broker patterns `alpaca`, `trade-api`, `paper-api`, `time_in_force`, `order_type`, `tradier` and `ibkr`, and allows the existing `api_?key` pattern in exactly one file, `src/btc_survivor/v2/local_secrets.py`.
* `tests/test_docs_style.py` adds `PLAN_V2.md` to the prose rules only, because a plan is exempt from the section link rule as `PLAN.md` is, and the version 2 docs to both rules.

**Configs that cannot be confused.** Every version 2 run config has `[run] kind = "stocks"` and sections version 1 does not know, so the version 1 `load_config` and dashboard reject it. Every version 2 database carries `meta.kind`, checked by every reader. The Finnhub adapter lives in `v2/stocks/finnhub.py`, because the version 1 validator accepts any module in `btc_survivor/feed/` as an exchange.

**Proof that version 1 did not move.** `test_v1_untouched` asserts that `git diff` from the fork commit, the merge base with `feat/initial-bot`, is empty for `src/btc_survivor` outside `v2` and for `configs/run_*.toml`. The unchanged version 1 suite runs in every phase.

**Imported unchanged.** `clock`, `LabelQueue`, `RunningStats`, `Standardizer`, `LabelScale`, the `MLP` parameters and `loss_and_grad`, `AdamW`, `RollingSkill`, `Account`, `decide`, `LinearBaseline`, `Recorder`, `FeedAdapter`, `decode`, `FeedClient`, the checkpoint `Arrays`, `encode` and `decode`, and `newey_west_se`, `diebold_mariano` and `block_test` from `cli/stats.py`. `LabelQueue` takes the session minute index times 60,000 as its minute stamp.

**New where version 1 does not fit.**

* `PooledWindow` keeps a symbol column, returns the newest n samples plus draws, and divides each label by its own symbol's scale, which `MaturedWindow` cannot.
* `predict_rows(model, X)` runs one forward pass over 8 rows, where `MLP.predict` takes one vector.
* `StockSeries` wraps `MinuteSeries` for reference price and volume, adds buy and sell volume rings, takes session starts from `sessions.py`, and holds at least 420 minutes.
* `StockStore` overrides `create_schema`, `check_schema` and `rollback_after` for the version 2 tables.
* `BackoffSettings` gives `FeedClient` its two backoff fields in place of a version 1 `FeedConfig`.
* `StockTwin` replaces the twin upkeep that walks a version 1 `LocalBook`.
* `genes.py` is not used, because without evolution the margin and the learning rate are two fixed floats.

**Commands.** By default every subcommand prints only counts, pass or fail verdicts, hashes and timings of this system. `--detail` adds rows and per symbol figures for the owner's own terminal, as section 13.3 requires.

| Subcommand | What it does | Tested by |
| --- | --- | --- |
| `init-db CONFIG` | creates a run or news database with its `meta.kind` | `test_store`, `test_news_db` |
| `stocks S1 S2` | the stock process for both runs | `test_stock_engine`, `test_group` |
| `news CONFIG` | the news process | `test_news_collector`, `test_llm_scorer` |
| `news-status` | week one and data quality report | `test_news_status` |
| `tape-check RECORDING OFFICIAL` | the tape verdict from a raw recording alone | `test_tapecheck` |
| `evaluate S1 S2` | the criteria of section 11 | `test_evaluate` |
| `status S1 S2` and `export CONFIG` | state, and CSV under `data/v2/exports` | `test_status`, `test_export` |
| `replay CONFIG FILES --out DIR` | recordings into a new folder, with the version 1 help sentence that replay never chooses, tunes or ranks | `test_stock_replay` |
| `dashboard S1 S2` | the private page on `127.0.0.1:8060` | `test_private_dashboard` |
| `check-secrets` | names the keys present, never a value | `test_local_secrets` |
| `check-local` | the local only check of section 13.2, printing its verdict and counts | `test_localcheck` |

**Module layout.**

```
btc_survivor_v2/                       the worktree
  PLAN_V2.md
  configs/v2/  s1_prices.toml  s2_news.toml  news.toml  rehearsal_s1.toml  rehearsal_s2.toml
  deploy/macos/   btc_survivor_v2_news.plist  btc_survivor_v2_stocks.plist   installed only with approval
  deploy/systemd/ btc_survivor_v2_news.service  btc_survivor_v2_stocks.service
  docs/  v2_overview.md  v2_data_and_licensing.md  v2_news.md  v2_evaluation.md  v2_operations.md  RESULTS_V2_TEMPLATE.md
  src/btc_survivor/v2/
    __main__.py  output.py         subcommands, aggregate printing by default, rows only with --detail
    local_secrets.py  redact.py    the only reader of secrets.env, redacting formatter and exception hooks
    localcheck.py                  loopback only listeners on 8050 and 8060 and no tunnel process, for criterion 1.8
    sessions.py                    closures and early closes 2026 to 2028, UTC sessions, session minute index
    config.py                      loaders with frozen dataclasses, experiment and news config hashes
    stats.py  evaluate.py          pooled skill and squared error tests, criteria, audits
    status.py  export.py  news_status.py  dashboard.py  dashboard.html
    stocks/
      finnhub.py  prints.py        adapter and pure parse, classification, eligibility, replayed prints
      candles.py  series.py        receive time candles, median reference price, tick rule, StockSeries
      health.py                    whole feed and per symbol states, epochs, delay guard, news health per minute
      market.py  features.py       the shared market layer, the 10 price features and coverage
      labels.py  learning.py       session bounded label queues, PooledWindow, predict_rows
      news_inputs.py               news lines, primary score per item, decayed company and macro inputs
      costs.py  broker.py          allowances, fees, borrow, edge, Roll diagnostic, print fills, quantities
      bot.py  twins.py  population.py  engine.py
      group.py                     dispatches news kinds, decodes once, one market layer for both engines
      runner.py                    StockFeedClient, ticker, news puller, heartbeat, lock, shutdown
      store.py  records.py  checkpoint.py  replay.py  synthetic.py  tapecheck.py
    news/
      db.py  http.py  budget.py  schedule.py  text.py  dedupe.py  prefilter.py
      finnhub_news.py  edgar.py  edgar_docs.py  rss.py  collector.py
      llm_request.py  llm_client.py  llm_validate.py  ledger.py  batcher.py  llm_scorer.py  probes.py
      finbert.py  finbert_scorer.py  health.py  runner.py
      prompt_v1.txt  schema_v1.json  probes_v1.json
  scripts/  v2_record_stocks.py  v2_learner_check.py  v2_power.py  v2_secret_scan.py  v2_soak.py
            export_finbert.py  llm_synthetic_check.py
  tests/    one test file per module, fixtures/news/ and fixtures/finnhub/
```

Only `v2/news/llm_client.py` imports `anthropic` and only `v2/news/finbert.py` imports `onnxruntime`, no trading module imports either, and `torch` never enters the project environment.

## 4. Data feed and market hours

### 4.1 One process, one connection

`btc_survivor_v2 stocks configs/v2/s1_prices.toml configs/v2/s2_news.toml` runs both stock runs in one asyncio process that owns the single Finnhub websocket the key allows and makes no REST calls. It refuses to start unless the two configs are identical apart from `[run]` `name` and `data_dir`, `[news_input]` `use_news` and `[evaluation]`. It takes an exclusive, non blocking `fcntl.flock` on `~/.config/btc_survivor/finnhub_ws.lock`, which `scripts/v2_record_stocks.py` also takes, so a stray recorder can never knock out the live connection. The Mac and the Pi cannot both stream on one key. Runs A to D share nothing with this process.

### 4.2 The Finnhub adapter

`v2/stocks/finnhub.py` subclasses the version 1 `FeedAdapter`. The URL is `wss://ws.finnhub.io?token=` plus the token, built in memory and never logged, and `max_message_bytes` is 4 MiB. `subscribe_messages()` returns one `{"type":"subscribe","symbol":"AAPL"}` frame per symbol in config order, sent after each connect without waiting for an acknowledgement. `parse` is a pure function of the raw text, the receive stamp and per connection state.

* **Trade.** A `trade` message becomes one `PrintBatch(recv_ms, conn_id, prints)`, each print `(symbol, price, size, vendor_ms, conditions)` from `s`, `p`, `v`, `t` and `c`, sorted by vendor time and then position. Unsubscribed symbols, prices that are not finite and prices or sizes at or below zero are dropped and counted. A missing `c` means no conditions.
* **Ping** becomes `FeedNotice(kind="ping")`, proves the connection is alive and gets no reply, because none is documented.
* **Error** becomes `FeedNotice(kind="error")`, redacted, and asks for a reconnect.
* **Anything else** is counted and logged once per type per connection.

These field names, the ping type and the absence of a trade id or sequence number come from Finnhub's docs outside the research and are unverified until the Phase 13 recording. Without a sequence number a lost message cannot be detected, so the tape check measures loss in aggregate. A new connection might replay recent prints, so any print whose vendor time is more than 1 second before the stamp of the latest `connected` notice is ignored for price and volume.

### 4.3 Prints to one minute candles

**Classification** by vendor time `t` through the calendar gives pre market from `04:00` ET to the open, the regular session, post market to `20:00` ET (`17:00` on early close days), or other. Classification may read `t` because it arrives in the message. Timing never reads it.

**Candles.** A print enters a session candle only if its receive stamp and its vendor time both fall in the regular session. Candles are assigned by receive time, so the candle at boundary B covers receive stamps in `[B - 60 s, B)`. Pre and post market prints live only in the raw recording.

**Price eligibility.** A print is eligible when it arrived at most `late_print_seconds` after its vendor time, at most 2 seconds before it, and none of its conditions is in `price_excluded_conditions`. Only eligible prints set open, high, low and close, freshness, the reference price, the tick rule and fills. Age on arrival is the one honest filter that does not depend on the unverified condition table. `price_excluded_conditions` ships empty and is filled before launch from the meanings in Finnhub's published table, using the Phase 13 histogram only to see which codes occur.

**Reference price** at B is the median of the last 5 eligible prints received before B in the current session, which damps bid and ask bounce and a stray print. The buffer resets at each open. It replaces the version 1 minute mid for features, labels, marks, stops and forecast rows.

**Trade signs.** Each eligible print is signed plus one above the previous eligible print of its symbol in the session, minus one below, and the previous sign when equal. Signs reset at each open, and each candle stores buy and sell volume.

**Empty minutes** repeat the previous close with zero volume and are never healthy.

### 4.4 Market calendar and the session clock

`v2/sessions.py` holds NYSE full closures and `13:00` ET early closes, hand copied from the NYSE hours page on 11 October 2026 with the source in a comment. Nasdaq is understood to follow the same dates, which is unverified because its page timed out, and Phase 12 checks the table against Nasdaq's list.

| Year | Closed | Early close at `13:00` ET |
| --- | --- | --- |
| 2026 | 1 Jan, 19 Jan, 16 Feb, 3 Apr, 25 May, 19 Jun, 3 Jul, 7 Sep, 26 Nov, 25 Dec | 27 Nov, 24 Dec |
| 2027 | 1 Jan, 18 Jan, 15 Feb, 26 Mar, 31 May, 18 Jun, 5 Jul, 6 Sep, 25 Nov, 24 Dec | 26 Nov |
| 2028 | 17 Jan, 21 Feb, 14 Apr, 29 May, 19 Jun, 4 Jul, 4 Sep, 23 Nov, 25 Dec, with 1 Jan a Saturday and not observed | 3 Jul, 24 Nov |

**Session times.** `09:30` to `16:00` America/New_York on weekdays not in the table, or to `13:00` on early closes, converted to UTC through `zoneinfo`. The open moves from `13:30` to `14:30` UTC on 2 November 2026 and back after 14 March 2027. At start the process checks four known opens, 30 October and 2 November 2026 and 12 and 15 March 2027, and refuses to run if any differs.

**Why a local table.** Live and replay see the same sessions with no REST call. The calendar refuses a date past its last year and writes a daily `calendar_warning` event from `calendar_warn_days`, 60, before the end. It is append only, and resume refuses to continue if recomputing the checkpoint's session index gives a different number. A closure missing from the table, or a market wide halt, shows up as silence, so nothing trades, learns or pays rent.

**The session minute index** k counts regular session minutes only, with k = 1 at the `09:31` boundary of Friday 2 January 2026, and the first minute of a session follows the last minute of the previous one directly. Series, label queues, windows, decision phases, history and gap measures run on k, and every row also keeps its UTC time, k and the New York session date. A full session has 390 minutes and an early close 210, both multiples of 30. A night, weekend or holiday adds no invalid minutes, so `max_resume_gap_minutes`, 10 session minutes per symbol, never flattens on an ordinary night but does after a restart or outage inside a session.

**Extended hours** are not used for trading, learning, labels or health, because prints there are thin and bursty, a fixed allowance would understate their spreads, and brokers take only limit orders then.

### 4.5 Price health without an order book

Each symbol has four states, closed, stale, recovering and healthy, and its own feed epoch.

**Closed.** Outside the regular session every symbol is closed, which is not stale. No stale events or reconnect requests are written, and nothing trades, learns, pays rent or creates labels. `StockFeedClient` calls `mark_healthy` once a connection has stayed open 60 seconds outside the session, so the reused backoff does not keep growing overnight.

**Whole feed conditions,** for every symbol in session.

1. The socket is connected.
2. Some message arrived within `feed_stale_after_seconds`, 5, counted from the later of the last message and the open.
3. Over the last 50 non replayed session prints, the median of receive stamp minus vendor time is within `max_print_delay_seconds` either way, with no objection until 10 samples exist. Samples clear at every open and connect. This replaces the version 1 skew check and catches a delayed feed.

**The delay rule, written now.** After the first recorded session in Phase 13, `max_print_delay_seconds` becomes the larger of 2 and `ceil(1.5 * p95)` of that session's delays in seconds, capped at 5, and `late_print_seconds` becomes twice that, from the delay distribution only, never from prices. A median above 5 seconds stops the build for the owner. Until then the config holds 2.0 and 4.

**Per symbol condition.** An eligible print arrived within `symbol_stale_after_seconds`, 15, counted from the later of the last eligible print and the open. On the full tape each name prints many times a second. At one print a second, a thin single venue's rate, gaps over 15 seconds arrive at about 3.1 in 10 million per second, about 0.007 per session, about 1.8 false stale events per symbol a year and 14 across the eight. These rates are estimates the first session measures.

**Scope and epochs.** A whole feed problem makes every symbol stale and asks for a reconnect. A symbol problem, such as a halt or a limit up limit down pause, stops only that symbol's bots and forecast rows. As in version 1, recovering becomes healthy at the first boundary closing a full minute in which every condition held, every exit from healthy bumps the symbol's epoch, every close bumps every epoch, and each change goes to `feed_events`. At the open a connected symbol with no delay objection moves to recovering with its clean stretch starting at the open, so the minute ending `09:31` is the first healthy one.

News health is separate and never touches price health, epochs or label maturity, as section 7.8 sets out.

### 4.6 Reconnects

The reused `FeedClient` grows the backoff from `reconnect_backoff_start_seconds`, 5, to 60 seconds with seeded jitter between half and the full delay, resets it when the caller reports health, and pings every 20 seconds with a 20 second timeout. `StockFeedClient` in `v2/stocks/runner.py` adds a `send(text)` method for resubscribe frames, so `feed/base.py` stays untouched. Reconnects happen only when no message arrived for 5 seconds in session, Finnhub sent an error, the connection has been silent for 5 minutes at ten minutes before the open, or the library closed the socket. A single quiet symbol never causes a reconnect. A symbol with no print on the current connection 120 seconds into the session, while others print, gets its subscribe frame again, at most once every 5 minutes, and a symbol that printed and went quiet is treated as halted. A handshake refused with HTTP 401 backs off and logs one ERROR line an hour.

### 4.7 Labels, opens and closes around the session

**Labels never span the close.** A record is pushed at B only when B plus 30 minutes is at or before the close, so the minutes ending `09:31` to `15:30` create records, and the close bumps every epoch.

**Opens,** including flips and forced opens, are allowed from session minute `first_open_session_minute`, 30, to the close minus `last_open_minutes_before_close`, 35, which is `10:00` to `15:25` ET, or `12:25` on an early close. Outside it an opening decision becomes action none with reason `session_window`, a flip becomes a close, and forced exploration waits for the next allowed phase minute.

**Closing before the bell.** At the close minus `flatten_minutes_before_close`, 5, which is `15:55` or `12:55` ET, every open position, twins included, gets a closing order with exit reason `session_close`. It fills on a continuous trading print, never the closing auction or a post market print. A rejected close is resent each healthy minute, and at the close unsettled orders are rejected with reason `session_closed`.

**A position that survives a close,** because its symbol was halted into it or the process was down, closes at that symbol's first healthy minute of the next session with exit reason `session_flatten` and an `overnight_carry` event, booking the overnight move in full.

### 4.8 The stock minute step

The one second ticker and every event call `advance_to`. At each boundary B inside a session the group does this.

1. **The shared market layer,** once for both runs. It judges news health for the minute, then per symbol in config order closes the candle, takes the reference price, judges price health, appends to the `StockSeries` at k, computes and standardizes the features, matures the label for k minus 30 within the same epoch, updates the label scale and `linear_prices`, and pushes a new label record when the minute is usable.
2. **News inputs.** S2 computes its two inputs per symbol from the news lines received before B and stores them under the minute k of any label record the shared layer pushed at B. S1 stores zeros. When a record matures, each engine joins it to the inputs it stored at the push minute, never to the current ones, so a training sample and `linear_news` see only news received before the push.
3. **Gap and session flattens** per symbol and engine.
4. **Bots in slot order.** Each learner charges rent and borrow, marks its position, checks death and its stop, trains on the samples that matured, predicts all 8 symbols, and decides at its phase minute on its home symbol. Each twin follows its learner.
5. **Session close orders** at the close minus 5 minutes.
6. **Rows** for candles, forecasts, labels, decisions, the benchmark and due snapshots are buffered for each run database.

A symbol minute is usable when the symbol is healthy and news health held. Rent, training, decisions, new label records and forecast rows need a usable minute, while prices, marks, stops, the session close, epochs and the maturity of pending labels run on price health alone. Prediction and training need only the symbols involved, so a learner keeps learning while its home symbol is halted, but rent, marking, death, the stop and decisions need the home symbol healthy. At the close every symbol moves to closed and unsettled orders are rejected. Outside sessions boundaries only persist.

### 4.9 Consolidated tape check

One session gates the build in Phase 13, the first full regular session, not an early close, whose raw recording shows no disconnected time from `04:00` to `20:00` ET.

**The official figures.** The next morning the owner, or the assistant session viewing the page in a browser on the owner's behalf, opens Nasdaq's public historical quote page for each symbol in an ordinary browser, confirms it labels its volume as consolidated, and copies the day's volume and official close into the gitignored `data/v2/checks/official_YYYY_MM_DD.toml`. No script fetches the page, which is for personal viewing. Without a consolidated label the owner uses a second consolidated display they may view, and the two must agree within 5 percent. Whether the page's volume is consolidated, and its terms, are unverified, because it timed out on 11 October 2026.

**The computation.** `tapecheck.py` reads only the raw recording, replaying `finnhub.parse` and the print classification, so it needs no run database. Per symbol it computes full day volume from `04:00` to `20:00` ET, pre, regular and post volume, the ratio r of full day volume to the official figure, odd lot and late shares, replayed prints, the condition histogram, delay percentiles, the last eligible print before `16:00` against the official close, and disconnected time. By default it prints the verdict and how many symbols fall in each band.

**The verdict.**

* **Consolidated** when the median r is between 0.85 and 1.15 and no symbol is below 0.70.
* **Not consolidated** when the median r is below 0.50, or after two consecutive inconclusive clean sessions. The build stops and the choice goes to the owner.
* **Inconclusive** otherwise, and the check repeats on the next clean session.
* A median above 1.15 means the reference was a single venue figure.

The bands are fixed by reasoning. Vendors differ by a few percent over late reports, odd lots and prints after `20:00`. A single venue carries at most about a fifth of a Nasdaq listed stock's volume, an unverified estimate, and IEX about 2.5 to 3.8 percent, but a partial feed that includes a trade reporting facility can carry much more, which is why repeated inconclusive verdicts go to the owner. A close within 0.2 percent is supporting evidence only. The runs repeat the check on the first clean full session at or after sessions 10, 20 and 40, and a failure makes the run invalid under criterion 1. If the gate fails, the fallback is the owner's choice in section 19, between Alpaca, whose credential the research marks as trading capable unless an unverified data only OAuth route works, and Databento Standard at 199 dollars a month.

### 4.10 Load estimates

If the feed is the full tape, the eight symbols print around 5 to 10 million times a day, a few hundred a second on average and several thousand a second after the open. With one event per message, a few microseconds per print and one shared market layer, the stock process should stay well under 10 percent of a core on the Mac. Raw recordings would run about 0.4 to 0.8 GB a day before compression, perhaps 50 to 120 MB compressed. These are estimates the first sessions measure, and the Pi needs its own.

## 5. Trading costs and fills

### 5.1 Fill model

A decision at B takes the home symbol's reference price at B. The order is due at B plus `latency_ms`, 1,000, plus a uniform jitter up to `latency_jitter_ms`, 300, from the bot's own generator, and fills at the first eligible print of that symbol received at or after the due time with vendor time at or after B, moved against the bot.

```text
buy_price  = print * (1 + (half_spread_bps[symbol] + extra_slippage_bps) / 1e4)
sell_price = print * (1 - (half_spread_bps[symbol] + extra_slippage_bps) / 1e4)
```

A trade print sits at the bid, at the ask or between, so on average near the midpoint, and a market order pays about half the spread more. An allowance at least as large as the real half spread therefore charges at least the real cost on average. It is fixed before launch and never tuned.

| Symbol | Typical quoted spread | Planning price | Implied half spread bps | `half_spread_bps` | Per leg with 1 bp extra |
| --- | --- | --- | --- | --- | --- |
| QQQ | one cent | 600 | 0.08 | 0.5 | 1.5 |
| MSFT | one cent | 500 | 0.10 | 1.0 | 2.0 |
| AAPL | one cent | 250 | 0.20 | 1.0 | 2.0 |
| GOOGL | one cent | 250 | 0.20 | 1.0 | 2.0 |
| AMZN | one cent | 220 | 0.23 | 1.0 | 2.0 |
| NVDA | one cent | 180 | 0.28 | 1.0 | 2.0 |
| META | about four cents | 700 | 0.29 | 2.0 | 3.0 |
| TSLA | about four cents | 400 | 0.50 | 2.0 | 3.0 |

The allowance is at least three and a half times the implied half spread, rounded up to a half bp, and the version 1 extra slippage of 1 bp covers latency drift and print selection. Spreads and prices are estimates from mid 2025 market structure, and if any price halved the allowance would still be at least about 1.8 times the implied half spread, the tightest case being NVDA. No opens come in the first 30 minutes, when spreads are widest.

**Rejections.** An order is rejected if its symbol leaves healthy, the connection drops, or the session ends first, and stops and flatten closes retry at the next healthy minute. A flip fills both legs at one print, bots on one symbol in one second share a print, and paper orders never move the market.

**Quantity.** A long opens `notional / reference price` shares, rounded down to 6 decimals, because the main US brokers sell fractional shares in session. A short opens the nearest whole number of shares with a one share minimum, because they do not short fractional shares, so a short's notional runs from about 400 to 700 dollars at planning prices and is recorded. Fill rows keep the version 1 columns plus `symbol`, `print_price`, `allowance_bps`, `notional`, `sec_fee` and `taf_fee`, with `levels_consumed` null.

**Roll diagnostic, report only.** Each day `costs.py` estimates each symbol's effective half spread as `sqrt(-cov(dp_t, dp_t_minus_1))` in bps when the covariance is negative and writes it to `spread_checks`, flagging sessions above the allowance. It never changes `half_spread_bps` before or during the runs, and any later allowance is set by reasoning before that run's own launch. Its accuracy on a mixed tape with odd lots is unverified.

### 5.2 Fees

There is no commission, kept as `commission_per_order = 0.0`. Regulatory fees fall only on sales, the closing leg of a long and the opening leg of a short.

* **SEC Section 31 fee.** 20.60 dollars per million of proceeds, from the fiscal 2026 advisory, in force from 4 April 2026 until 60 days after the fiscal 2027 appropriation. The 87.00 dollar fiscal 2027 figure is the Section 6(b) registration fee, a different fee.
* **FINRA trading activity fee.** 0.000195 dollars per share sold, capped at 9.79 dollars per trade. Filing `SR-FINRA-2026-021` pauses it from 1 October to 31 December 2026, and the model charges it throughout, the harsher reading.
* **Rounding.** Each fee rounds up to the next cent per sale, minimum one cent, as Robinhood and Revolut describe their pass through. Broker waivers are not modelled.
* **Left out.** The CAT fee of 0.000001 dollars per share, about 0.0001 bps here.

These figures are outside the research, read on 11 October 2026 and rechecked against the SEC and FINRA pages before launch. A FINRA filing said to defer the increases planned for 2027 to 2029, the CAT fee and other brokers' rounding are unverified.

At 500 dollars a sale pays 0.02 dollars of SEC fee and 0.01 of trading activity fee, 0.6 bps together. The round trip estimate for the gate and the death threshold is

```text
cost_bps = 2 * (half_spread_bps[symbol] + extra_slippage_bps) + sale_fees_bps + short_borrow_bps_over_horizon
```

which gives 3.6 bps for QQQ, 4.6 for AAPL, MSFT, NVDA, AMZN and GOOGL, and 6.6 for META and TSLA, or 0.18, 0.23 and 0.33 dollars at 500, against about 190 bps for the version 1 retail Bitcoin round trip.

### 5.3 Shorts

**Borrow.** `short_borrow_pct_per_year = 1.0` accrues on entry notional over calendar time through the version 1 funding accrual, 0.0114 bps an hour or about 0.006 bps per 30 minute short, with no opening charge. Megacaps lend near the general collateral rate, so 1 percent errs harsh. That rate, other brokers' practice and an Alpaca page saying easy to borrow names cost nothing on its API are outside the research and unverified.

**Short sale restriction.** When the reference price falls 10 percent or more below the previous session's last healthy reference price, new short opens in that symbol are refused for the rest of the session and the next, as SEC Rule 201 does. Without a healthy previous price it cannot trigger.

**Kinder than reality.** Intraday shorting needs a margin account, and FINRA's pattern day trader rule limits a margin account under 25,000 dollars to three day trades in five business days, while each learner day trades about ten times a week. Whether proposed changes to that rule are in force in October 2026 is unverified and is checked against FINRA's current rule before launch. Section 18 and `docs/v2_data_and_licensing.md` list this. The confirmatory test reads only forecasts and labels, so it does not depend on it.

**Splits.** An overnight move above `max_overnight_gap_pct`, 25, keeps that symbol's features invalid until 391 fresh session minutes exist and writes a `split_guard` event.

**Dividends.** Intraday only means no dividend handling is needed.

### 5.4 Rent, marking, stops and death

**Rent** stays the cost of capital, 1.16 dollars a calendar day from the 4.23 percent three month Treasury yield of 8 October 2026, spread over session minutes as `1.16 * 365 / (252 * 390) = 0.0043081` dollars per usable minute, 1.68 per full session, 0.90 per early close and about 423 a year. Start equity stays 10,000 and notional 500, so the version 1 `Account` is reused unchanged.

**Marking** is at the symbol's reference price moved to the exit side by the per leg allowance.

**Stop.** `stop_loss_sigma = 3.0` label standard deviations of the symbol, fixed at entry and checked every healthy minute against the mark. A fixed 1 percent would be one standard deviation for TSLA and three for QQQ.

**Death.** A bot dies when its equity falls below the round trip cost of the position it holds, or of a long on its home symbol if flat, plus one session of rent, computed on the actual notional, about 1.86 to 2.14 dollars at planning prices. None is expected. A death leaves the slot empty and `evaluate` reports it.

## 6. Features and labels

### 6.1 Price features

Every feature uses eligible prints, is computed per symbol at boundary t, and counts session minutes.

| Number | Feature | Definition |
| --- | --- | --- |
| 1 | 30 minute return without the overnight move | `1e4 * ln(p_t / p_t_minus_30)`, minus the overnight gap when the window spans today's open |
| 2 | 120 minute return without the overnight move | the same over 120 session minutes |
| 3 | One session return | `1e4 * ln(p_t / p_t_minus_390)`, including the overnight move |
| 4 | Realized volatility over 60 minutes | standard deviation in bps of one minute log returns within the same session |
| 5 | Volume ratio | `ln(V30 / mean 30 minute volume over the trailing 390 session minutes)`, clipped to plus or minus 3 |
| 6 | Trade sign imbalance | `(buy_volume - sell_volume) / (buy_volume + sell_volume)` over 30 minutes |
| 7 | Time since the open | session minutes since the open over 390 |
| 8 | Time to the close | session minutes to the close over 390, different from feature 7 only on early closes |
| 9 | Overnight gap | `1e4 * ln(first healthy reference price today / last healthy reference price of the previous session)`, invalid if either is missing |
| 10 | Return relative to QQQ | feature 3 of the symbol minus feature 3 of QQQ, zero for QQQ |

The gap is its own input and far larger than a 30 minute move, so features 1, 2 and 4 leave it out. Features 7 and 8 carry the time of day pattern in volume.

**Coverage.** With `min_window_coverage = 0.8`, a return needs healthy reference prices at both ends, volatility, the volume ratio and the imbalance need 80 percent of their minutes healthy, and feature 10 needs QQQ healthy at both ends. An invalid input makes the symbol's vector invalid for the minute, so there is no prediction, sample or decision tick for it, while marking and stops continue. With 391 minutes of history needed, vectors first become valid at `09:31` of the second session.

### 6.2 Standardization, label scale and labels

Each symbol has its own version 1 `Standardizer` over the 10 features, with transform before update, warmup weighting, `min_samples = 30`, `clip_z = 5` and `standardizer_half_life_hours = 168` healthy session hours, about 10,080 samples or 26 sessions, and its own `LabelScale` with the same half life, all in the shared market layer. The target is the label over its symbol's scale, and a prediction in bps is the output times that scale.

The label at t is `1e4 * ln(p(t + 30) / p(t))` on reference prices. Records are pushed every usable minute with a valid vector when the end fits before the close, mature only if the symbol's epoch is unchanged, and are discarded otherwise. Evaluation labels go to `stock_labels`, from section 10.4.

### 6.3 The two news inputs

Both runs carry two inputs per symbol, bounded between minus one and one and never standardized, so a minute with no news gives exactly zero. S1 pins them at zero, and S2 computes them from the primary stream of section 8.7.

```text
w_j     = 2 ** (-a_j / 120)          a_j = regular session minutes from scored_at_j to t
company = sum(w_j * strength_j) / (sum(w_j) + 1)     over primary impacts tagged with the symbol
macro   = sum(w_j * strength_j) / (sum(w_j) + 1)     over primary impacts tagged MACRO
```

* An impact counts at t only if the stock process received its line before t, never earlier than scoring finished.
* Its weight decays from `scored_at` in session minutes, so a score after the close keeps full weight until the open and a backlog read after a restart carries its true age.
* The extra 1 in each denominator acts as one neutral prior item, and impacts older than 5 half lives, 600 session minutes, are dropped.
* Strength is `direction * confidence / 5` for a model score, counted only when marked new, and `p_pos - p_neg` for FinBERT.
* QQQ's company input uses impacts tagged QQQ or any of the seven companies.

A half life of 120 session minutes keeps 84 percent of a headline's weight across one horizon and little by the next afternoon.

### 6.4 Learning

**One network in both runs.** Each learner runs one 12 input network with 16 hidden units, 225 parameters, over all eight symbols every usable minute through `predict_rows`. The hidden pre activation is the price part, the first 10 inputs times their weights, plus the news part, skipped while every row's news inputs are zero. S1 and S2 therefore share one code path and parameter count, and S1's news weights get zero gradient and stay zero.

**Training.** A learner trains on the samples of every symbol that matured this minute, up to 8, plus draws with replacement from the `PooledWindow` to a batch of 16. The window holds the last 240 session minutes of samples from every symbol, up to 1,920. A sample's news inputs are the ones its engine stored at the record's push minute, as section 4.8 sets out. The Huber loss, gradient clipping, AdamW and the output scale of 0.1 at birth follow version 1.

**Learning rate.** `scripts/v2_learner_check.py` fixes it before launch on synthetic data only, running the bots' own learning code on pooled sessions of 8 symbols with planted correlations of 0, 0.05 and 0.1, a common market factor, labels that respect the close and the batch rule above. Its generator parameters are the planning values of section 12, never fitted to recordings. The rule takes the largest of 0.0003, 0.0001, 0.00003 and 0.00001 whose prediction spread with no signal is at most 0.10 label standard deviations and whose squared error with no signal is within 2 percent of predicting zero, or 0.00001 if none qualifies.

**Warmup** is 6 hours of training minutes, which are the 360 minutes of a full session in which samples mature.

**Timeline from a cold start** with a Monday launch and clean sessions. Session 1 records history. In session 2 vectors are valid from `09:31`, the first labels mature at `10:01`, the label scales reach 30 labels by about `10:30`, and training starts. In session 3 warmup completes at about `10:30`, each learner's forced timer starts at minus a seeded delay of up to 3 hours, and the first forced trades fall between about `10:30` and `13:30` ET.

## 7. News pipeline

### 7.1 The news process

`btc_survivor_v2 news configs/v2/news.toml` is one long running asyncio process. It polls SEC EDGAR, the Federal Reserve, BLS and Finnhub over REST, scores items with Haiku, Sonnet and FinBERT, and writes everything to `data/v2/news/news.db` through one writer connection on the event loop thread. Network calls run with `urllib.request` in worker threads, and the collector needs only the standard library. No version 1 module imports the news package, and it imports nothing from the stock modules. While `[finnhub] enabled` is true it holds an exclusive lock on `~/.config/btc_survivor/finnhub_rest.lock`, as the only Finnhub REST client.

The stock process opens `news.db` with `mode=ro` only, never `immutable`, so it always sees later commits. It does not reuse the version 1 helper in `cli/data.py`, which adds `immutable=1` whenever the WAL file is absent, as after a clean stop of the news process. The puller reopens the file on any error and whenever no new heartbeat has arrived for 60 seconds, and never reads during a minute step.

### 7.2 Channels, cadences and the REST budget

A channel is one URL pattern on its own schedule, which reads America/New_York through `zoneinfo` only to pick a cadence. Every weekday counts as a session and filing day for polling. When a cadence gets faster, the channel polls at once.

| Channel | Host | Cadence | Weekday requests |
| --- | --- | --- | --- |
| `finnhub_company_SYM`, one per stock | Finnhub | 60 s from `09:30` to `16:00` ET, 120 s from `04:00` to `09:30` and `16:00` to `20:00`, 600 s otherwise | about 5,060 |
| `finnhub_general` | Finnhub | the same | about 720 |
| `edgar_atom_8k` | sec.gov | 30 s from `06:00` to `22:00` ET on weekdays, 900 s otherwise | about 1,950 |
| `edgar_sub_SYM`, one per CIK | data.sec.gov | 60 s in the same window, 900 s otherwise | about 6,940 |
| EDGAR documents | sec.gov | on demand, about 3 per `8-K` | about 10 |
| `fed_press_all` | federalreserve.gov | 60 s from `07:00` to `19:00` ET on weekdays, 900 s otherwise | about 770 |
| `fed_press_monetary` | federalreserve.gov | 15 s from `13:58` to `14:20` ET on FOMC statement days only | 88 on those days |
| `fed_speeches` | federalreserve.gov | 600 s from `07:00` to `21:00` ET on weekdays, 3,600 s otherwise | about 95 |
| `bls_cpi`, `bls_empsit`, `bls_ppi`, `bls_jolts` | bls.gov | 1,800 s from `07:00` to `19:00` ET on weekdays, 7,200 s otherwise, 15 s from one minute before to 15 minutes after a scheduled release | about 120, plus 64 on a release day |

QQQ has no company channel, and its company input aggregates the seven companies.

**Budget.** `v2/news/budget.py` keeps one token bucket per host with an injected clock.

* **Finnhub.** At most 30 calls in any rolling 60 seconds, at least 1.1 seconds apart, against the key's 60 a minute and 30 a second. The planned peak is 8 a minute. A 429 pauses every Finnhub channel for 60 seconds. Whether the websocket handshake counts against the 60 is unverified.
* **SEC.** At most 2 requests a second and 60 a minute across both hosts, against the SEC's 10 a second. A 403 or 429 pauses every SEC channel for 900 seconds.
* **Fed and BLS.** At most 10 a minute each. A 403 or 429 pauses that host for 900 seconds.

A failed poll is next due after the cadence times 2 to the power of its consecutive failures, capped at 900 seconds, with no jitter. A poll the budget delays runs late and logs `budget_wait`. There are no automatic retries.

**Requests.** `v2/news/http.py` sends `Accept-Encoding: gzip` and decodes it, caps a body at 8 MB, times out after 10 seconds, and sends `If-None-Match` or `If-Modified-Since` when it has an `ETag` or `Last-Modified`, counting a 304 as success. The `User-Agent` holds the contact address only for sec.gov, data.sec.gov and bls.gov, which require it, and Fed and Finnhub get `btc-survivor/2`. The Finnhub token travels in the `X-Finnhub-Token` header, never a URL, which is unverified until the first call. Headers are never logged.

**Clock check.** While the median difference between response `Date` headers and the local clock over the last 20 responses exceeds 5 seconds, every poll is recorded as `clock_skew` and counts as failed.

### 7.3 The receive timestamp rule

1. `received_at` is the live clock's stamp, taken in the event loop right after the body is read, in the first poll from any channel that returns the item. Parsing, deduplication, the prefilter, the insert and the commit then run with no `await` between them.
2. One clock that never goes backwards stamps every row, and each insert commits before the next stamp, so `seq` order equals `received_at` order. The scorers stamp and commit `scored_at` the same way.
3. A row is written only once its fields are known. An `8-K` seen in the Atom feed is inserted only after its index page, main document and Exhibit 99.1 are fetched, with the earlier sighting kept as `detected_at`. A document that fails twice leaves the row flagged without that text, and an accession being fetched is skipped by other channels.
4. Vendor times, Finnhub `datetime`, EDGAR `acceptanceDateTime`, RSS `pubDate` and Atom `updated`, go to `vendor_ts` only to measure delivery lag and for one guard. An item whose vendor time is more than 24 hours before its receive time gets status `stale_vendor` from every scorer.
5. Rows are never updated after insert, only pruned, which a test checks by grepping for `UPDATE`.

A score then becomes visible to features only once the stock process has received its line.

### 7.4 Sources and parsing

* **Finnhub company news.** `GET /api/v1/company-news?symbol=SYM&from=YESTERDAY&to=TODAY` in UTC dates, a two day range because the vendor's time zone is unverified. Repolls of known ids are skipped silently. `vendor_id` is the Finnhub `id`, `publisher` its `source`, and `related` its tags.
* **Finnhub general news.** `GET /api/v1/news?category=general&minId=N`, with the largest id seen kept in `meta`, a behaviour that is unverified.
* **EDGAR latest filings.** The Atom feed `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=100&output=atom`, keeping entries filed by one of the seven CIKs, checked against fixtures and the first live filing.
* **EDGAR submissions.** `https://data.sec.gov/submissions/CIK##########.json` per CIK, zero padded to ten digits, keeping only `8-K` and `8-K/A` from `filings.recent`. The CIKs are AAPL 320193, MSFT 789019, NVDA 1045810, AMZN 1018724, GOOGL 1652044, META 1326801 and TSLA 1318605, checked once against `https://www.sec.gov/files/company_tickers.json`.
* **Filing text.** The main `8-K` document is cut to 4,000 characters and the first Exhibit 99.1 to 12,000, about 3,000 tokens, after stripping tags with `html.parser`, unescaping entities and collapsing whitespace. Other exhibits are not fetched, and `chars_total` and `truncated` record the cut.
* **Fed.** `press_all.xml`, `press_monetary.xml` and `speeches.xml` under `https://www.federalreserve.gov/feeds/`. `vendor_id` is the link, so a release in two feeds is one item, and the kind comes from the link, where a press release path under `monetary` means `fed_monetary`, never from which channel saw it first. Monetary policy releases also get their page fetched, cut to 8,000 characters.
* **BLS.** The `cpi`, `empsit`, `ppi` and `jolts` feeds at `https://www.bls.gov/feed/`, file names confirmed on the first poll, `vendor_id` the guid or link, description kept to 1,500 characters. Whether it carries the headline figures is unverified.
* **Kinds.** `media`, `8k`, `8k_a`, `fed_press`, `fed_monetary`, `fed_speech` and `bls_release`.

The text lengths of 4,000, 12,000, 8,000, 1,500, 700 and 24,000 characters change what the models score, so they are fixed now. Most of the seven usually report third quarter results in the second half of October and NVIDIA in late November, and filings after `17:30` ET are disseminated the next business day.

### 7.5 Seed, backlog, deduplication and the prefilter

Each raw item passes these steps in order, inside its poll.

1. **Vendor id.** `(source, vendor_id)` is unique. A known id from the same channel writes nothing, and from another channel writes one `news_dups` row with method `vendor_id`.
2. **Seed.** Every item from a channel's very first successful poll gets status `seed`, kept for deduplication and never scored.
3. **Backlog.** New items get status `backlog` when the time since the channel's previous successful poll exceeds the longer of the cadences in force at that poll and this one, plus `backlog_gap_seconds`, 600, plus any logged budget wait. Ordinary polls at 600, 1,800, 3,600 or 7,200 seconds give status `new`, while a restart or outage about ten minutes longer than the cadence gives `backlog`. Kind `8k` is exempt, because a filing is rare and material.
4. **Headline hash.** NFKC, lowercase, `&` read as and, a trailing publisher suffix of at most four words dropped, non alphanumerics turned to spaces and collapsed, then the first 16 hex characters of SHA 256. The same hash from the same source within 24 hours is a duplicate with method `hash`.
5. **Near duplicate.** Token sets of normalized words of two or more characters, minus about 20 stop words. Two sets of at least 4 tokens with a Jaccard index of at least 0.8 against an item of the same source from the last 24 hours make a duplicate with method `near`.
6. **Prefilter** on the survivors.

A duplicate never changes its original, and its extra tickers go to `news_dups.new_tickers` for measurement only, so tickers are fixed at first receipt. Deduplication stays within one source.

**Aliases,** as regular expressions with word boundaries, case sensitive for names.

| Symbol | Aliases |
| --- | --- |
| AAPL | Apple, iPhone, AAPL, Tim Cook |
| MSFT | Microsoft, Azure, MSFT, Satya Nadella |
| NVDA | Nvidia, NVIDIA, NVDA, Jensen Huang |
| AMZN | Amazon, AWS, AMZN, Andy Jassy |
| GOOGL | Alphabet, Google, YouTube, Waymo, GOOGL, GOOG, Sundar Pichai |
| META | Meta, Meta Platforms, Facebook, Instagram, WhatsApp, META, Mark Zuckerberg |
| TSLA | Tesla, TSLA |
| QQQ | QQQ, Nasdaq 100, `Nasdaq-100`, Invesco QQQ |

Elon Musk is left out, because much of his coverage is about other companies.

**Macro keywords,** case insensitive. Fed, Federal Reserve, FOMC, Powell, interest rate, rate cut, rate hike, inflation, CPI, PPI, PCE, payrolls, jobs report, unemployment, jobless claims, Treasury yields, tariff, recession, GDP, Wall Street, stock market, Nasdaq Composite, S&P 500, Dow Jones, semiconductor, chip stocks, antitrust. Exchange notation matching `\b(NASDAQ|NYSE|NYSEARCA)\s*:\s*[A-Z.]+` is stripped first, and an item that matches any alias never also gets MACRO, so one company's score never enters all eight macro inputs.

**Junk patterns,** case insensitive. `\b(top|best) \d+ stocks?\b`, `\bstocks? to (buy|watch)\b`, `\bshould you buy\b`, `\bmillionaire\b`, `\bbuy and hold forever\b`, `\bmotley fool\b`.

**Relevance tiers.** `title` when an alias is in the headline, `body` when only in the summary, `vendor` when only Finnhub's `related` names a symbol, which is never scored, `macro` for a general item with a macro keyword and no alias, and `official` for EDGAR, Fed and BLS items that passed their rules. Anything else is `filtered` with reason `junk`, `no_match`, `empty` or `form`. Fed and BLS items carry MACRO, and EDGAR items their company's symbol. The lists are fixed now, and the week one audit may fix only outright regular expression bugs, before launch.

### 7.6 Routing and the text kept

The route is set once at insert. `llm` covers Finnhub items with status `new` and relevance `title`, `body` or `macro`, and the official kinds `8k`, `fed_press`, `fed_monetary`, `fed_speech` and `bls_release` with status `new`. Haiku scores every `llm` item, FinBERT also scores every `llm` item as fallback and baseline, and Sonnet also scores the `llm` items of the five official kinds and nothing else. `none` covers everything else, including `8k_a`. The route `finbert` exists only for the case where `[finnhub] send_to_llm` is false, when it takes the Finnhub items that `llm` would have taken.

Finnhub headlines and summaries go to Haiku by the owner's decision, with the risk section 13.1 states. `finnhub.send_to_llm` is true, and setting it false sends Finnhub items to FinBERT alone, a config change that ends the window under criterion 1.6 if made after launch. Sonnet never sees a Finnhub item, and the scorer checks the kind again before every Sonnet call.

Only headlines up to 300 characters, summaries up to 600 for Finnhub, 1,000 for the Fed and 1,500 for BLS, and the truncated filing and FOMC texts are kept, never full articles, images or raw responses. All fetched text is untrusted, and the scorer presents it as quoted data whose instructions are ignored. Finnhub fixtures are hand written in the documented shape, while SEC, Fed and BLS fixtures may be trimmed public domain files.

### 7.7 Week one

The collector starts as soon as Phase 14 passes, in parallel with the Phase 13 tape gate and at least four trading days before launch. Week one may change only cadences, budgets and retention. Routing, aliases, junk patterns, text lengths, the half life and every threshold that touches a score are fixed now and never compared with prices or labels. The news process writes its config hash, prompt and schema hashes and both model ids to `meta` on every start, and a stock run records them at launch. If they ever differ, news health fails and a feed event says why.

`btc_survivor_v2 news-status --days 7` reports these as counts and verdicts.

1. Requests per channel per day, the status mix, bytes, durations, the 304 share, and every 403 and 429.
2. Items per channel per day by stage, status, tier, filter reason and symbol, and backlog items per channel per hour, which must be zero outside restarts and outages.
3. Delivery lag, `received_at` minus `vendor_ts`, as median, 90th percentile and maximum per channel.
4. For each `8-K`, which EDGAR path saw it first and by how much.
5. Duplicates that brought new tickers, and the spread of near duplicate Jaccard values.
6. Whether a Finnhub company call returns the same round count every time, a sign of a hidden cap.
7. Outage minutes per channel per day, and the longest outage.
8. The delay from each scheduled release to `received_at`.
9. Scoring lag per scorer, model cost per item for each model, and the days whose spend raised the daily alert of section 8.6.
10. Database growth per day.
11. An order verdict, passed when `received_at` never decreases and matches `seq` order, and a backlog verdict, passed when no backlog item falls outside a logged restart or outage.

Only the owner runs `news-status --detail --samples 50`, which prints a sample drawn with `report_seed` of kept and filtered headlines and of near duplicate pairs between 0.8 and 0.9.

Wednesday 11 November 2026, Veterans Day, falls inside the window. Stocks trade, but EDGAR accepts no filings and BLS publishes nothing.

### 7.8 News health

News health means the news pipeline is alive. The news process writes a `news_heartbeat` row every 30 seconds, healthy when FinBERT is on, no item routed to a scorer has waited more than `score_wait_seconds` for its primary score, and the clock check is clean. Neither model's state counts, because a paused, capped or off Haiku hands every waiting item to FinBERT at once, and Sonnet feeds no primary score. `score_wait_seconds` is 600, or twice FinBERT's time on the 300 item synthetic burst measured in Phase 15 if longer, so an FOMC, CPI or earnings burst does not break it.

A session minute ending at B is news healthy on the stock side when the newest heartbeat line received was never more than 120 seconds old during the minute, every heartbeat line in it said healthy, and each carried the hashes recorded at launch.

**What it gates.** Without news health, both runs charge no rent, train nothing, make no decision, push no label record and write no forecast row. Prices, marks, stops, the session close, epochs and the maturity of pending labels go on. A news outage therefore bumps no epoch, discards no label, flattens nothing and leaves price features such as the one session return valid the next day.

**Channel outages.** A channel whose last successful poll is older than three times its cadence or 300 seconds, whichever is longer, is in outage, which never touches news health. Each heartbeat lists those channels, S2 records them per minute in `news_minutes`, and `news-status` and `evaluate` count outage minutes per channel as data quality, the same way vendor delay is part of the measured pipeline. Minutes lost to news health are reported by cause, with their overlap with scheduled releases, after the first session and at sessions 10, 20 and 40.

### 7.9 Volume estimates

These are unverified estimates from the research, replaced by week one numbers.

| Channel | Requests per weekday | Items kept per day |
| --- | --- | --- |
| Finnhub | about 5,800, peak 8 a minute | a few hundred raw, about 100 to 300 after deduplication and the prefilter |
| EDGAR | about 8,900, peak about 9 a minute plus documents | well under one `8-K` a day across the seven, more in earnings weeks |
| Fed | about 860 | 1 to 5 |
| BLS | about 120 to 190 | about 4 releases a month |

A Finnhub company call returns a whole day, perhaps 50 to 150 KB for a busy name, and a submissions file 30 to 60 KB compressed, so without 304 responses bandwidth reaches several hundred MB a day. Haiku and FinBERT each see about 100 to 300 Finnhub items a day plus a handful of official items. Sonnet sees the official items alone, planned at about 5 to 20 a day, though the table above suggests fewer outside earnings weeks and FOMC days.

## 8. Language model scoring with spend controls

### 8.1 Placement and the call

A Haiku task, a Sonnet task and a FinBERT task run inside the news process and share its writer. FinBERT runs in `asyncio.to_thread`, so polling never stalls. The stock process never calls a model. It reads finished scores, stamps them on arrival and records them, so replay reproduces every news input without `news.db` or any model, and nothing scored reaches runs A to D.

`v2/news/llm_client.py` is the only module that imports `anthropic`. It sends one request, returns the status, stop reason, refusal category, usage, `response.model`, request id, raw JSON and any error class, and never retries.

```python
client = anthropic.AsyncAnthropic(max_retries=0, timeout=60.0)
response = await client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=4096,
    thinking={"type": "disabled"},
    output_config={"effort": "low",
                   "format": {"type": "json_schema", "schema": SCHEMA_V1}},
    system=[{"type": "text", "text": SYSTEM_PROMPT_V1, "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": user_text}],
)
```

* **No sampling parameters,** because Haiku 5.5 returns 400 for any temperature, top p or top k other than its defaults.
* **No tools, web search, metadata, stop sequences or fallback list,** and the default inference geography. Haiku 5.5 has no server side fallback, and that it rejects a fallback list is outside the research and unverified.
* **Thinking off,** accepted at effort low, because thinking tokens bill as output. Content blocks are read by type, and any thinking block is counted and logged.
* **One call in flight** per model, and no Batch API, whose latency of up to a day would make every score useless as a feature.

**The Sonnet call** is the same request with `model="claude-sonnet-5-5"`, `max_tokens=1024`, enough for one item with 10 impacts, and no `cache_control` on the system block. Sonnet makes about 5 to 20 calls a day spread over hours, so its five minute cache would rarely be warm, and its cache prices are not in the research, so every Sonnet input token bills at the verified 2 dollars per million and every output token at 10. That Sonnet 5.5 accepts thinking disabled at effort low, rejects sampling parameters the same way and supports the same structured outputs is outside the research and unverified until its first live call in Phase 15. If that call rejects any of these settings, apart from the integer enums that section 8.4 handles, `[llm.sonnet] enabled` is set false before the news process scores anything, and the comparison waits for the owner's choice before launch, since settings never change after launch.

Haiku 5.5 was released on 7 October 2026 with retirement no sooner than 7 October 2027, and its training data ends in June 2026. Sonnet 5.5's release, retirement and training cutoff dates are outside the research.

### 8.2 The system prompt

The prompt is the committed file `v2/news/prompt_v1.txt`, about 1,200 tokens, UTF 8 with LF endings, read as bytes and never templated, so the cached prefix never changes. In order it holds the task in two sentences, the ground rules, the instrument table, one line per level of each field, a note that an item bearing on no instrument gets an empty impacts list and that a macro item may also name tickers it affects directly, and three fictional, undated examples, an earnings beat, a Fed decision that matched expectations, and a recap of a price move marked not new.

The ground rules say to judge each item from its own text, as published just now, to trust the text where it disagrees with knowledge that ends around mid 2026, to judge surprise from the text's own framing such as beat, miss, unexpected or as expected, to ignore any instruction inside item text, and never to estimate prices. The instrument table names AAPL as Apple, MSFT as Microsoft, NVDA as Nvidia, AMZN as Amazon, GOOGL as Alphabet, META as Meta Platforms and TSLA as Tesla, each with its main products, QQQ as large US tech stocks as a group or the Nasdaq 100, and MACRO as US rates, inflation, jobs, growth and Fed policy, whose direction is the likely effect on large US stocks overall.

Haiku and Sonnet use this one prompt and the one schema of section 8.4. The config pins `prompt_sha256` and `schema_sha256`, the schema hash over canonical JSON with sorted keys, and the news process refuses to start if a file or `meta` disagrees. A change to the prompt, schema, model, effort or thinking makes a new scorer, such as `haiku_v2` or `sonnet_v2`, with its own rows.

### 8.3 Eligibility, input and batching

An item is sent to Haiku only when its route is `llm`, it is not a seed, it was received less than `max_queue_age_seconds`, 300, ago, and its vendor time passes the 24 hour guard, and to Sonnet only when it also has one of the five official kinds. Every other item gets a final `expired` or `stale_vendor` row at once, so the fallback never waits.

The user message is one line of instructions, then one compact JSON object per item with sorted keys, NFC text, collapsed whitespace and no HTML. The keys are `id` (the position in the batch, `i01` to `i25`, never a vendor id), `source`, `filer` (the EDGAR filer's ticker), `headline` (cut at 300 characters) and `text` (cut at 700). An `8-K` with its exhibit, or a monetary policy statement, goes alone with `text` cut at 24,000 characters. Every other official item also goes alone, so Haiku and Sonnet receive the same user message for each official item, and Finnhub items share batches only with each other. No stock feed value and no date ever enters a request. A guard refuses any request whose exact token count, or bytes over 3 when no count exists, exceeds 20,000, far below the 100,000 token line where the price rises fivefold.

`v2/news/batcher.py` is pure and takes the time as an argument. Finnhub items wait in receive order, ties broken by item seq, and a batch goes when 20 items wait or the oldest has waited 60 seconds, at any hour. An official item goes alone at once to each model, and a split gives the first half `floor(n / 2)` items.

### 8.4 Output schema and validation

```json
{"type": "object", "additionalProperties": false, "required": ["items"],
 "properties": {"items": {"type": "array", "items": {
   "type": "object", "additionalProperties": false, "required": ["id", "impacts"],
   "properties": {
     "id": {"type": "string"},
     "impacts": {"type": "array", "items": {
       "type": "object", "additionalProperties": false,
       "required": ["ticker", "event_type", "is_new", "direction", "surprise", "horizon", "confidence"],
       "properties": {
         "ticker": {"type": "string", "enum": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ", "MACRO"]},
         "event_type": {"type": "string", "enum": ["earnings", "guidance", "product", "legal_regulatory", "mergers_acquisitions", "management", "capital_return", "financing", "analyst_rating", "partners_supply", "macro_policy", "macro_data", "market_commentary", "other"]},
         "is_new": {"type": "boolean"},
         "direction": {"type": "integer", "enum": [-1, 0, 1]},
         "surprise": {"type": "integer", "enum": [0, 1, 2, 3]},
         "horizon": {"type": "string", "enum": ["hours", "days", "weeks"]},
         "confidence": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}}}}}}}}
```

The model states what happened and whether it is new before a direction and a confidence. The API supports enums but not numeric bounds, so every field is a required enum. Whether integer enums are accepted is unverified, and if the first live call of either model rejects them the schema switches to string enums for both before launch. `is_new` is true when the item itself first reports an event, figure or decision, and false for recaps, previews, opinion and price moves. There is no Bitcoin ticker, which could return only in a new scorer version.

`v2/news/llm_validate.py` checks the response against strict Pydantic models that forbid extra fields, then that each id belongs to the batch once, tickers are unique within an item and an item has at most 10 impacts. Valid items are kept when others fail.

### 8.5 Failure handling

| What happened | What the scorer does |
| --- | --- |
| A valid answer with stop reason `end_turn` | one final row per item, scored or irrelevant |
| Some ids missing or invalid | keeps the valid items, requeues each missing item once, then marks it malformed |
| A refusal | splits the batch in halves recursively, so a single refused item becomes final with its category |
| `max_tokens`, unparsable JSON, or a whole answer failing the schema | splits the same way, and a single item ends malformed |
| More than 24 calls for one batch | marks the remaining items malformed |
| A 429 or 400 naming a spend, credit or billing limit | no retry, both models off until `00:00` UTC on the first of the next month |
| Any other 429 | no retry, that model paused for 15 minutes or the retry after time, whichever is longer |
| Any other 400, 401, 403 or 404 | no retry, that model off until restart with an ERROR line |
| A 5xx, 529, timeout or lost connection | booked at its full reservation, and the items are marked failed with no second call |
| `response.model` is not the model requested | discards the answer, that model off until restart |
| Refusals in more than 20 percent of a model's last 50 calls | pauses that model and logs an alert |
| The process dies during a call | at restart the call becomes lost, its reservation stays booked, and its items are marked failed |
| The monthly cap would be passed | no call, both models off until `00:00` UTC on the first of the next month, waiting items marked `skipped_cap` |

While Haiku is paused, capped or off, every waiting and arriving item routed to it gets a final `skipped_off` or `skipped_cap` row at once, so FinBERT takes over immediately. Sonnet's items get the same rows and need no fallback, since Sonnet feeds no primary score, and a Sonnet error never changes Haiku's state. Each state lives in `scorer_state` and survives a restart. The SDK retries 429s twice by default, which would hammer a hard cap, hence `max_retries=0`, and because the spend limit error's wording is unverified, every 429 is final.

There are no automatic retries. A call that ends in an error, a timeout, a lost connection or a dead process is never sent again, by the SDK or by the scorer, and FinBERT covers its items. Only a split and the one requeue put an item into a second call, and each follows an answer that came back refused, cut off, unparsable or without a valid result for that item, as a new reserved call within the limit of 24 calls per batch.

### 8.6 Cost ledger and the monthly cap

Every process that may call either model with this key, meaning the news process, its probes and `scripts/llm_synthetic_check.py`, reserves against one ledger per key, `~/.config/btc_survivor/llm_ledger.db`, under an exclusive `fcntl.flock` on `llm_ledger.lock` beside it. Haiku and Sonnet share this ledger and its one cap. The rehearsal cap drill alone uses a throwaway ledger of its own and books its total into the key ledger when it ends, as section 16 describes. Before each call the caller counts the exact request, system prompt, schema and messages, with the `count_tokens` endpoint, which Anthropic's docs describe as free, outside the research and confirmed in Phase 15. If counting fails or does not accept the full request, a byte bound is used, which holds because every token covers at least one byte of text.

```text
rates in dollars per million tokens, from [llm.prices], read 11 Oct 2026
  claude-haiku-5-5     in_rate 0.10   out_rate 0.50    write 1.25, the cached system block
  claude-sonnet-5-5    in_rate 2.00   out_rate 10.00   write 1.00, no cache control
counted = count_tokens(system, schema and messages)
          or, when that fails, request_bytes + schema_bytes + reserve_overhead_tokens
reserve = (counted * in_rate * write + max_tokens * out_rate) / 1e6
cost    = (input_tokens * in_rate + cache_creation_input_tokens * in_rate * 1.25
           + cache_read_input_tokens * in_rate * 0.10 + output_tokens * out_rate) / 1e6
```

The reservation prices every input token at the highest rate the call can meet and all of `max_tokens` at the output rate, so it bounds what the call can cost, about 0.003 dollars for a full Haiku batch, about 0.014 for a short Sonnet call and about 0.028 for a long form Sonnet call. A Sonnet answer that reports any cache token raises an alert, because Sonnet's cache prices are not in the research. Spend for the UTC month is `cost_usd` for calls that returned usage plus `reserved_usd` for every other call, whether running, timed out, disconnected, answered with a 5xx after sending, or lost, each with `cost_basis = reserved`.

* **The cap.** A call goes only when the month's spend plus its reservation stays at or under `monthly_usd = 20.00`, the same as the Console workspace limit, so the code stops first. There is no daily cap and no daily limit on items or calls.
* **At the cap** both models are off until `00:00` UTC on the first of the next month, and FinBERT covers every item until then.
* **The daily alert.** When the spend for one UTC day, counted the same way, passes `daily_alert_share` of the monthly cap, a quarter of 20 dollars or 5 dollars, the news process logs one WARNING line, writes a `spend_alert` event to `collector_events` and sets `spend_alert` on every heartbeat for the rest of that UTC day, and the private dashboard shows it. The alert stops nothing. The expected daily spend of section 8.10 is about a thirtieth to a hundred and fiftieth of it, so an alert means something ran away, such as a loop, a flood of items or a change in Anthropic's prices.

Each item is scored once per scorer, deduplication runs before routing, and no failed call is ever sent again, so a runaway is limited by what arrives. On the first live day every call logs its reservation beside its actual cost, and actual must never exceed reserved.

**The Console backstop.** The key belongs to a dedicated workspace that is not the default one, which cannot have its own limits, with a 20 dollar monthly spend limit set by the owner, an organisation limit below the tier cap and auto reload off. The Max plan includes monthly API credits, 100 dollars on Max 5x and 200 on Max 20x, which need a linked Console organisation and 7 days on the plan and expire each billing cycle. Whether the workspace limit applies to those credits is unverified, one reason the cap in code exists.

### 8.7 FinBERT and the score streams

**Export.** `scripts/export_finbert.py` runs once, with the owner's approval, in a throwaway virtual environment with torch and transformers pinned, never in the project environment. It downloads `ProsusAI/finbert` at a pinned commit, about 440 MB by an estimate outside the research, exports ONNX at opset 17 with dynamic axes, saves the fast tokenizer as `tokenizer.json`, checks that ONNX and PyTorch logits agree within 0.0001 on 50 fictional sentences, and writes `manifest.json` with the revision, label order, sizes and SHA 256 hashes under the gitignored `data/models/finbert`. The code is Apache 2.0, the weights have no stated licence, and the training data is CC BY NC SA 3.0, which suits this personal non commercial project.

**Runtime.** `v2/news/finbert.py` opens an `onnxruntime` CPU session with one intra op thread, one inter op thread and sequential execution, and tokenizes with `tokenizers`, so scores are bitwise repeatable on one machine. It scores the headline and text joined by a period, in windows of up to 510 tokens and at most 16 windows, averaging probabilities weighted by token count. A hash mismatch stops the news process. FinBERT cannot tell which company a text is about, so its impacts use the prefilter's tags, the EDGAR filer, and MACRO for Fed and BLS, and an untagged item ends `untagged`. It scores every item routed to `llm` or `finbert` as it arrives, with no cost or cap.

**The streams,** derived from `item_scores` by one pure function, tested to give exactly one primary score per item.

* **`news`, the primary stream.** An `llm` item uses its Haiku score when its final Haiku row says scored or irrelevant, visible when that row is received, and otherwise the FinBERT score as a flagged fallback, visible at the later receipt of the final Haiku row and the FinBERT row. A `finbert` item, which exists only while `send_to_llm` is false, uses its FinBERT score, visible when that row is received. The stream decays from the `scored_at` of the score used, and feeds S2's inputs and the confirmatory predictor.
* **`haiku`,** Haiku rows only, and **`finbert`,** FinBERT rows for every routed item, the baseline.
* **`finbert_matched`,** FinBERT rows on exactly the items whose final Haiku row says scored or irrelevant, each visible at the later receipt of the two rows, so `haiku` against `finbert_matched` measures what the model adds over FinBERT on the same items.
* **`sonnet` and `haiku_official`,** Sonnet rows and Haiku rows on exactly the official items both models scored, each visible at the later receipt of the two model rows, so the pair compares the two models on the same items at the same times.

Sonnet never enters the primary stream, so the confirmatory test does not depend on it, and the primary stream mixes in FinBERT strengths only for fallback items, whose share is reported.

### 8.8 Lookahead, non determinism and probes

Only live items are sent, received after launch, all after Haiku's June 2026 training cutoff and after Sonnet's, whose date is outside the research but falls before the model became available, which is before launch, so forward scoring is free of lookahead by construction. No historical news is ever scored, and a hard check in the scorer refuses any backfill marked contaminated. Company names distract most for megacaps, but stay unmasked because linking items to tickers needs them, and a masked variant could later run as its own predictor.

Sampling cannot be fixed on Haiku 5.5, and Sonnet 5.5 is treated the same way, so each item is scored once per scorer and frozen as data, the uniqueness constraint on item and scorer blocks a second score, and replay, `evaluate` and `export` read stored rows. Batching and splitting are deterministic.

Daily at `06:00` UTC, `v2/news/probes.py` sends the 20 fictional items of `probes_v1.json`, written before launch, to Haiku as one reserved call of kind probe. Sonnet runs no probe, because its stream is descriptive and never feeds the confirmatory test. Results go only to `llm_probes`, and `news-status` reports per field agreement with the launch day run and the previous day, flagging direction agreement below 80 percent and changing nothing automatically. A fresh rerun of the scores would differ by about what the probes measure.

### 8.9 Credentials and redaction

`v2/local_secrets.py` is the only reader of `secrets.env`. It parses plain `KEY=value` lines, accepts only the three names, refuses the file unless the user owns it and group and others have no access, names a missing key without any value, shows only which keys exist in its `repr`, and registers every value with `v2/redact.py`. In the news process only, it puts the Anthropic key into the process environment for the SDK, removes `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_PROFILE` and `ANTHROPIC_BASE_URL`, and refuses to start while `ANTHROPIC_LOG` is set, so the SDK cannot fall back to a profile or endpoint the owner did not choose.

`v2/redact.py` installs a redacting `Formatter` on every handler, so traceback text added by `formatException` is redacted too, and routes `sys.excepthook` and the asyncio exception handler through logging, so nothing reaches stderr or a launchd log unredacted. It also redacts recorded close reasons, which embed the exception text and, for a websockets `InvalidURI`, the URL with the token, and every stored detail.

The contact address goes only into the request header for sec.gov, data.sec.gov and bls.gov. A test puts a sentinel in its place and checks it never reaches a model request, log line, database row or page.

### 8.10 Expected cost

**Haiku.** At about 150 input and 50 to 100 output tokens per Finnhub item, a full warm batch of 20 costs about 0.0008 to 0.0013 dollars, in line with the research's 0.19 dollars a month at 200 items a day, which works out to about 0.0006 a batch. At 100 to 300 Finnhub items a day most batches flush after 60 seconds with one to a few items, each call paying for the roughly 1,200 token prompt as a cache read of about 0.00001 dollars or, after five quiet minutes, a cache write of about 0.00015. An item then costs about 0.00005 to 0.0002 dollars, and Finnhub items come to about 0.15 to 2 dollars a month. The official items, each alone at about 0.0001 dollars for a short one and 0.001 for a long form one, add under 0.15, and the probes about 0.04.

**Sonnet.** At about 1,900 input tokens for a short official item and about 9,000 for a long form one, plus 100 to 150 output tokens, a short item costs about 0.005 dollars and a long form one about 0.02. At 5 to 20 official items a day on about 22 weekdays a month, with a few long form filings in earnings weeks, Sonnet comes to about half a dollar to 3 dollars a month. The research prices Sonnet at 20 times Haiku for the same work, which is why it sees only the official items.

**Together** the expected cost is about 1 to 5 dollars a month, so the 20 dollar cap is about 4 to 20 times the expected spend and the 5 dollar daily alert about 30 to 150 times the expected daily spend. Haiku 5.5's tokenizer counts about 30 percent more tokens than Haiku 4.5's, and Sonnet 5.5's tokenizer is outside the research, so real counts come from `count_tokens`. These are estimates until week one, and the Max plan's included credits are expected to cover them, confirmed in Phase 15.

## 9. Storage

### 9.1 The news database

`data/v2/news/news.db` runs in WAL mode with `synchronous=NORMAL`. The collector commits once per insert, and the scorers commit each final row with its impacts in one transaction. Times are ISO 8601 UTC text with milliseconds and a trailing `Z`.

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  -- kind = news, schema_version, config_hash, prompt_sha256, schema_sha256, haiku_model, sonnet_model,
  -- first_poll.<channel>, cursor.finnhub_general
CREATE TABLE news_items (
  seq INTEGER PRIMARY KEY,                                   -- commit order, equal to received_at order
  source TEXT NOT NULL, channel TEXT NOT NULL,               -- finnhub, edgar, fed, bls, and the first channel
  vendor_id TEXT NOT NULL,                                   -- Finnhub id, accession number, RSS guid or link
  kind TEXT NOT NULL,                                        -- media, 8k, 8k_a, fed_press, fed_monetary, fed_speech, bls_release
  received_at TEXT NOT NULL, detected_at TEXT, vendor_ts TEXT,
  title TEXT NOT NULL, summary TEXT, url TEXT, publisher TEXT,
  tickers TEXT NOT NULL,                                     -- comma separated symbols and MACRO, may be empty
  relevance TEXT NOT NULL,                                   -- title, body, vendor, macro, official, none
  status TEXT NOT NULL, filter_reason TEXT,                  -- new, seed, backlog, filtered
  title_hash TEXT NOT NULL, items_8k TEXT,                   -- items such as 2.02,9.01
  route TEXT NOT NULL,                                       -- llm, none, or finbert only while send_to_llm is false
  UNIQUE (source, vendor_id));
CREATE INDEX news_items_received ON news_items (received_at);
CREATE INDEX news_items_hash ON news_items (source, title_hash);
CREATE TABLE news_bodies (item_seq INTEGER PRIMARY KEY, text TEXT NOT NULL, chars_total INTEGER NOT NULL,
  truncated INTEGER NOT NULL, doc_urls TEXT NOT NULL);
CREATE TABLE news_dups (id INTEGER PRIMARY KEY, item_seq INTEGER NOT NULL, channel TEXT NOT NULL,
  vendor_id TEXT NOT NULL, seen_at TEXT NOT NULL, method TEXT NOT NULL, jaccard REAL, new_tickers TEXT);
CREATE TABLE news_polls (
  seq INTEGER PRIMARY KEY, channel TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
  cadence_s INTEGER NOT NULL,                                -- the cadence in force, for the backlog rule
  status TEXT NOT NULL,      -- ok, not_modified, http_error, timeout, parse_error, rate_limited, blocked, clock_skew
  http_status INTEGER, bytes INTEGER, duration_ms INTEGER, budget_wait_ms INTEGER,
  n_raw INTEGER, n_new INTEGER, n_backlog INTEGER, n_dup INTEGER, n_filtered INTEGER,
  detail TEXT);                                              -- passed through the redactor
CREATE INDEX news_polls_channel ON news_polls (channel, finished_at);
CREATE TABLE collector_events (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT);
  -- started, stopped, config_changed, pause, budget_wait, clock_skew, spend_alert
CREATE TABLE item_scores (                                   -- insert only, one final row per item and scorer
  seq INTEGER PRIMARY KEY, item_seq INTEGER NOT NULL,        -- seq is the order consumers read in
  scorer TEXT NOT NULL, route TEXT NOT NULL,                 -- haiku_v1, sonnet_v1 or finbert_v1, route copied from the item
  status TEXT NOT NULL,      -- scored, irrelevant, refused, malformed, failed, skipped_cap, skipped_off,
                             -- expired, stale_vendor, untagged
  call_id INTEGER, attempts INTEGER NOT NULL,
  scored_at TEXT NOT NULL,                                   -- when the result existed or the status was decided
  p_pos REAL, p_neg REAL, p_neu REAL,
  UNIQUE (item_seq, scorer));
CREATE TABLE score_impacts (item_seq INTEGER NOT NULL, scorer TEXT NOT NULL, ticker TEXT NOT NULL,
  strength REAL NOT NULL,                                    -- model direction * confidence / 5, FinBERT p_pos - p_neg
  direction INTEGER, surprise INTEGER, event_type TEXT, confidence INTEGER, horizon TEXT, is_new INTEGER,
  PRIMARY KEY (item_seq, scorer, ticker));
CREATE TABLE llm_calls (
  call_id INTEGER PRIMARY KEY, root_call_id INTEGER, ledger_id INTEGER NOT NULL,
  kind TEXT NOT NULL,                                        -- batch, single (one official item), split, probe
  scorer TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
  status TEXT NOT NULL,      -- started, ok, refusal, max_tokens, malformed, rate_limited, spend_limit,
                             -- bad_request, auth, server, timeout, connection, model_mismatch, lost
  n_items INTEGER NOT NULL, model TEXT NOT NULL, response_model TEXT, request_id TEXT, sdk_version TEXT NOT NULL,
  prompt_sha256 TEXT NOT NULL, schema_sha256 TEXT NOT NULL, request_sha256 TEXT NOT NULL,
  effort TEXT NOT NULL, thinking TEXT NOT NULL, max_tokens INTEGER NOT NULL,
  stop_reason TEXT, refusal_category TEXT, http_status INTEGER, error_class TEXT,
  thinking_blocks INTEGER NOT NULL DEFAULT 0, counted_tokens INTEGER, count_basis TEXT,   -- count_tokens or bytes
  input_tokens INTEGER, cache_creation_input_tokens INTEGER, cache_read_input_tokens INTEGER, output_tokens INTEGER,
  reserved_usd REAL NOT NULL, cost_usd REAL, cost_basis TEXT,                             -- usage or reserved
  price_table TEXT NOT NULL, latency_ms INTEGER, response_z BLOB);
CREATE TABLE llm_call_items (call_id INTEGER NOT NULL, item_seq INTEGER NOT NULL, position TEXT NOT NULL,
  PRIMARY KEY (call_id, item_seq));
CREATE TABLE llm_probes (run_at TEXT NOT NULL, probe_id TEXT NOT NULL, call_id INTEGER NOT NULL,
  ticker TEXT NOT NULL, event_type TEXT, is_new INTEGER, direction INTEGER, surprise INTEGER, horizon TEXT,
  confidence INTEGER, PRIMARY KEY (run_at, probe_id, ticker));
CREATE TABLE scorer_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  -- haiku_state, sonnet_state, the until time and first response model of each, cap_until
CREATE TABLE news_heartbeat (
  seq INTEGER PRIMARY KEY, ts TEXT NOT NULL, healthy INTEGER NOT NULL, reason TEXT,
  config_hash TEXT NOT NULL, prompt_sha256 TEXT NOT NULL, schema_sha256 TEXT NOT NULL,
  haiku_state TEXT NOT NULL, sonnet_state TEXT NOT NULL, finbert_state TEXT NOT NULL,
  waiting INTEGER NOT NULL, oldest_wait_s REAL,
  channels_down TEXT NOT NULL,                               -- channels in outage, may be empty
  spent_day_usd REAL NOT NULL, spent_month_usd REAL NOT NULL, spend_alert INTEGER NOT NULL);
```

Items, bodies, scores, impacts and calls are kept forever, except `filtered` items, kept 30 days. `news_polls`, `news_dups` and `news_heartbeat` are kept 14 days. Growth is about 1 MB a day for items, 2.5 MB for polls and up to 10 KB per model call, with a few hundred calls a day for the Finnhub items, roughly 1 to 2 GB a year, an estimate.

```sql
-- ~/.config/btc_survivor/llm_ledger.db, the key ledger, written only under llm_ledger.lock
CREATE TABLE reservations (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL,
  process TEXT NOT NULL,                                     -- config hash and data folder of the caller
  model TEXT NOT NULL, ref TEXT NOT NULL, n_items INTEGER NOT NULL,
  reserved_usd REAL NOT NULL, cost_usd REAL, cost_basis TEXT,   -- usage or reserved
  status TEXT NOT NULL);                                     -- started, done, unknown, lost
```

At restart a process marks its own started rows lost, booked at their reservation.

### 9.2 A stock run database

Each run has its own database, so S1 and S2 replay and resume independently. Both receive identical candles, labels and `linear_prices` rows from the shared market layer, and both checkpoints carry identical copies of the layer's state, which resume compares. `equity_snapshots`, `deaths` and `checkpoints` keep their version 1 shape, and `lineage` is not created. `bots` gains `home_symbol`, `decisions` and `trades` gain `symbol`, `fills` gains `symbol`, `print_price`, `allowance_bps`, `notional`, `sec_fee` and `taf_fee`, and `trades.exit_reason` gains `session_close` and `session_flatten`. `feed_events` gains `symbol`, null for whole feed events, and the kinds `session_open`, `session_close`, `print_delay`, `resubscribe`, `overnight_carry`, `calendar_warning`, `split_guard`, `news_unhealthy`, `news_healthy`, `lock` and `local_only`.

```sql
-- meta keys: kind = stocks, schema_version, run_name, master_seed, launch_ts, config_toml, experiment_hash,
-- news_config_hash, prompt_sha256, schema_sha256, haiku_model, sonnet_model, news_start_seq, evaluation_json,
-- run_minute_rule, code_commit, symbols_json
CREATE TABLE stock_candles (symbol TEXT NOT NULL, ts TEXT NOT NULL, session_date TEXT NOT NULL,
  session_k INTEGER NOT NULL, open REAL, high REAL, low REAL, close REAL, ref_price REAL,
  volume REAL NOT NULL, buy_volume REAL NOT NULL, sell_volume REAL NOT NULL,
  prints INTEGER NOT NULL, price_prints INTEGER NOT NULL, late_prints INTEGER NOT NULL, valid INTEGER NOT NULL,
  PRIMARY KEY (symbol, ts));
CREATE TABLE sessions (session_date TEXT PRIMARY KEY, open_ts TEXT NOT NULL, close_ts TEXT NOT NULL,
  early_close INTEGER NOT NULL, healthy_minutes INTEGER NOT NULL, run_minutes INTEGER NOT NULL);
CREATE TABLE stock_labels (ts TEXT NOT NULL, symbol TEXT NOT NULL, entry_price REAL NOT NULL,
  exit_price REAL, label_bps REAL, status TEXT NOT NULL,     -- pending, matured, gap
  PRIMARY KEY (ts, symbol));
CREATE TABLE stock_forecasts (ts TEXT NOT NULL, symbol TEXT NOT NULL, predictor TEXT NOT NULL,
  prediction REAL NOT NULL,                                  -- bps for population and linear, a score for news predictors
  items INTEGER, inputs_max_recv_at TEXT,                    -- impacts behind a score, newest receive time behind the row
  PRIMARY KEY (ts, symbol, predictor));
CREATE TABLE news_seen (                                     -- S2 only, every score line S2 read
  score_seq INTEGER PRIMARY KEY, item_seq INTEGER NOT NULL, recv_at TEXT NOT NULL, scored_at TEXT NOT NULL,
  source TEXT NOT NULL, route TEXT NOT NULL, scorer TEXT NOT NULL, status TEXT NOT NULL,
  tickers TEXT NOT NULL, primary_use INTEGER NOT NULL, fallback INTEGER NOT NULL);
CREATE TABLE news_minutes (ts TEXT PRIMARY KEY, healthy INTEGER NOT NULL, reason TEXT,   -- S2 only
  channels_down TEXT NOT NULL);
CREATE TABLE spread_checks (session_date TEXT NOT NULL, symbol TEXT NOT NULL, prints INTEGER NOT NULL,
  roll_half_spread_bps REAL, allowance_bps REAL NOT NULL, flagged INTEGER NOT NULL,
  PRIMARY KEY (session_date, symbol));
CREATE TABLE benchmark (ts TEXT PRIMARY KEY,
  ew_buy_hold REAL NOT NULL,                                 -- 10,000 split equally over the 8 at launch
  qqq_buy_hold REAL NOT NULL, matched_equity REAL NOT NULL,  -- matched is 500 in QQQ from launch, rest cash
  living_mean_equity REAL NOT NULL, living_median_equity REAL NOT NULL, living INTEGER NOT NULL,
  twin_mean_equity REAL, cash_equity REAL NOT NULL);
CREATE TABLE runtime_status (id INTEGER PRIMARY KEY CHECK (id = 1), ts TEXT NOT NULL, feed_state TEXT NOT NULL,
  session_state TEXT NOT NULL, prints_per_s REAL, msgs_per_s REAL, news_age_s REAL, living INTEGER NOT NULL,
  last_checkpoint_ts TEXT, rss_mb REAL, rss_peak_mb REAL);
CREATE TABLE symbol_status (symbol TEXT PRIMARY KEY, ts TEXT NOT NULL, state TEXT NOT NULL,
  ref_price REAL, last_print_age_s REAL);
```

`stock_candles` are kept for the life of the run, about 125,000 rows per run over 40 sessions, because `evaluate` recomputes every label from them. Raw recordings in `data/v2/raw_stocks` are pruned after `keep_raw_days`, 30, which still covers each repeat tape check. Everything else is kept forever. Per full session and run, decisions grow by about 400 rows, forecasts by about 1,060 in S2 and 190 in S1, labels by 96, candles by 3,120 and `news_minutes` by 390 in S2.

## 10. Runs and predictors

### 10.1 Two runs

Both runs cover the eight symbols in the one stock process on one shared market layer, with master seed 12345.

| Run | Inputs | Learners | Twins | Data folder |
| --- | --- | --- | --- | --- |
| S1 prices | the 10 price features, news inputs pinned at zero | 24, three per home symbol | 24 | `data/v2/s1_prices` |
| S2 news | the 10 price features plus the two news inputs | 24, three per home symbol | 24 | `data/v2/s2_news` |

### 10.2 Pooled learners with a home symbol

Every learner predicts all eight symbols every usable minute and learns from all eight, but trades only its home symbol, the one at position `slot % 8` in config order, so slots 0, 8 and 16 trade AAPL and slots 7, 15 and 23 trade QQQ. Each network gets eight times the version 1 samples, and each symbol's population forecast is the mean of 24 learners. A learner decides at session minutes where `k % 30 == slot % 30`, which gives it 13 decision minutes per full session, about 11 inside the open window. Coin flip twins keep their version 1 rules on the home symbol, with the session close added to the exits they copy. A dead learner or twin leaves its slot empty.

### 10.3 S1 and S2 born identical

The learner in each slot of either run is drawn with the same generator calls, `MLP.random` at 10 inputs, and then two zero columns are appended for the news inputs, so generator state, latency jitter, forced timer delays, replay batches and twin coins all match. While news inputs are zero the news part of the hidden layer is skipped and the news weights get zero gradient, so the two runs make identical predictions until S2's first training step whose batch holds a nonzero news input, expected in session 2. Every later difference is caused by news, and `test_zero_init` checks this exactly.

### 10.4 Evaluation minutes, forecasts and labels

Evaluation minutes are session minutes 30, 60 and so on while the minute plus 30 is within the session, so `10:00` to `15:30` ET, 12 per full session and 6 on an early close. At every evaluation minute that is usable for a symbol, it gets one row per predictor in `stock_forecasts`.

| Predictor | Written by | Units | Meaning |
| --- | --- | --- | --- |
| `population` | S1 and S2 | bps | mean prediction of the living learners for the symbol |
| `linear_prices` | S1 and S2 | bps | the version 1 online ridge on the price z scores, pooled, fitting the label over the symbol's scale |
| `linear_news` | S2 | bps | the same ridge with the two news inputs |
| `news` | S2 | score | company plus macro input from the primary stream, as S2's network sees them |
| `news_company`, `news_macro` | S2 | score | each input alone, descriptive |
| `haiku`, `finbert` | S2 | score | the same formula on Haiku scores only, and on FinBERT scores of every routed item, the baseline |
| `finbert_matched` | S2 | score | FinBERT on exactly the items Haiku scored, for Haiku against FinBERT on the same items |
| `sonnet`, `haiku_official` | S2 | score | Sonnet and Haiku on exactly the official items both scored, for Sonnet against Haiku on the same items |
| `zero` | none | bps | implicit, as in version 1 |

Score predictors have no fitted scale, so they are judged by correlation only, and squared error tests apply only to bps predictors.

**Labels.** `stock_labels` holds one row per evaluation minute and symbol, `1e4 * ln(exit / entry)` from the reference prices at the two boundaries. It matures only if both symbol minutes were healthy in one epoch, and otherwise ends as gap. `evaluate` recomputes every label from `stock_candles` and requires an exact match.

**The news audit.** `news_seen` records every score line S2 read, when it counts from, its route, scorer and primary use. Every score row stores `inputs_max_recv_at`, which `evaluate` checks is never later than the row's minute.

### 10.5 What changes from version 1

Unchanged are start equity, notional, shorts allowed, 16 hidden units, the output scale of 0.1 at birth, Huber loss, gradient clipping, AdamW with weight decay 0.0001, warmup of 6 hours, persistence intervals, the twin rules and the death rule's meaning. Changed, by reasoning before launch, are the horizon in session minutes, rent per usable session minute, `force_trade_after_hours = 3.0` healthy session hours from the first Active minute, which keeps about two forced trades per learner per session as version 1's 12 hours did in a 24 hour market, a fixed `margin_sigma = 0.25` with `required_edge_bps = cost_bps + margin_sigma * label_scale_bps[symbol]`, the stop at 3 label standard deviations, one fixed learning rate from section 6.4, and a death threshold of one session of rent with no replacement.

**Why the margin is in standard deviations.** Pure noise with spread s0 label standard deviations passes the gate with probability `2 * Phi(-(cost_bps / sigma + margin) / s0)`. With s0 at 0.10, about where the version 1 learner sits, and the planning volatilities of section 12, an 8 bps margin would pass about 11 percent of TSLA decisions, 6 percent of NVDA and 2 percent of all, while 0.25 standard deviations passes under 0.2 percent for every symbol and about 0.05 percent on average. A calibrated forecaster with correlation 0.10 makes predictions as large as a no skill learner with spread 0.10, so the gate cannot tell skill from noise and does not try. Almost every trade will be forced, the twins and the skill test measure direction, and a live pass share well above 1 percent with no passing skill test is reported, never tuned.

## 11. Evaluation and success criteria fixed before launch

### 11.1 The evaluate command

`btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml` opens both run databases and `news.db` read only, takes its settings from each run's `meta` at launch, and computes every number from rows up to the cutoff, so a later run never changes a result. Before the cutoff every number is descriptive. By default it prints each criterion's verdict and row counts, and the owner reads the numbers with `--detail`.

**The cutoff.** A run minute is a session minute in which news health held and at least one symbol was healthy, a rule stored as `run_minute_rule` in `meta`. The cutoff is the minute S2 reaches `final_healthy_session_minutes`, 15,600 run minutes counted from launch, the length of 40 full sessions with early closes at their real length.

**The window** runs from the first minute of session 2 to the cutoff, about 39 sessions and 468 rows per symbol. Session 1 is burn in, because features need a session of history and the puller starts at the newest score at launch.

### 11.2 Pooled skill

`v2/stats.py pooled_skill` takes one predictor's rows in the window.

1. Z score prediction and label within each symbol. A symbol whose prediction never varies is left out and named.
2. For each row, u is the product of the two z scores, and each evaluation minute's value is the mean of u over the symbols present.
3. The correlation is the mean of the per minute values, with the version 1 `newey_west_se` at 12 lags, one session, as its standard error, the Driscoll Kraay estimator in its simplest form. It allows for the common market move at a minute and slow drift in the forecasts.
4. The predictor passes when the correlation over its standard error exceeds 2.33. The detectable correlation, `(2.33 + 1.28)` standard errors, and the one sided 95 percent upper confidence bound, the correlation plus `1.645` standard errors, are printed beside it.

Pooling without clustering would overstate the evidence. In a design time simulation with one market factor explaining 45 percent of each stock's variance and 90 percent of QQQ's, the clustered test rejected 1.1 percent of 2,000 null runs at the 1 percent level and an unclustered test 7.8 percent.

### 11.3 Squared error, twins, events and calibration

`pooled_dm` divides each row's difference of squared errors by the symbol's label variance, averages per minute and uses the same 12 lag error, a positive t favouring the first forecast. It compares the population against zero and against `linear_prices`, S2's population against S1's on paired rows, and `linear_news` against `linear_prices`.

Learner and twin trades from one decision are paired by slot, symbol and decision time and compared on gross profit through the version 1 `block_test`, one block per session. Stocks drift up roughly 1 bp per 2 hours, an estimate, so the report shows each learner's share of long opens beside the result.

The score predictors are also reported over event rows, where an item tagged with the symbol was received in the 30 minutes before the minute, with `haiku` and `finbert_matched` on the same event rows, and `sonnet` and `haiku_official` on theirs. Sonnet against Haiku rests on the official items alone, about 5 to 20 a day, so only a large difference could show. The bps predictors get a slope of label on prediction within each symbol. All of this is descriptive.

### 11.4 Power

The key unknown is c, the correlation between symbols' predictions at one minute. The macro input is identical for all eight symbols and is fed by every Fed and BLS item and every general headline with a macro keyword, so the news predictor's c rises with the macro share. A synthetic rerun of the plan's own estimator at 468 rows per symbol, one sided 1 percent and 90 percent power gives these detectable correlations at 40 sessions.

| c | 0.2 | 0.5 | 0.8 | 1.0 |
| --- | --- | --- | --- | --- |
| Detectable correlation | 0.076 | 0.096 | 0.113 | 0.123 |

So the news predictor's detectable correlation runs from about 0.075 if company news dominates to about 0.12 if macro news dominates, and learned predictors sit near c of 0.5. At 20 sessions the figures are about 0.10 at c of 0.2 and 0.13 at 0.5, and at 60 sessions about 0.061 and 0.079. With sparse, decayed, heavy tailed scores at c of 0.8, power is about 35 to 40 percent at a true 0.075 and about 61 percent at 0.11. Null rejection stayed at or below 1.1 percent, falling to 0.35 percent with heavy tails, so the test is valid but conservative.

Before launch `scripts/v2_power.py` simulates zero inflated scores with the 120 minute decay, a macro component shared across symbols at c of 0.2, 0.5, 0.8 and 1.0, and fat tailed, volatility clustered labels, with generator parameters taken from the planning values of section 12 and never fitted to recordings. Its printed range replaces these figures here, in section 1 and in section 18 before the owner settles the window length. `evaluate` also prints the detectable correlation from the realized error. Trading every decision in the forecast's direction breaks even at about `cost_bps / (0.8 * sigma)`, 0.08 to 0.17 across these symbols, so a just detectable predictor would still not cover costs on most of them. Version 2 measures skill. It does not promise profit.

### 11.5 The confirmatory test

* **The test.** Predictor `news` in S2 over the window, with the pooled skill statistic, passing when t exceeds 2.33. It is the only test, and everything else is printed without a pass label.
* **What it states.** It concerns the forecast's correlation with the next 30 minute return after receipt, for these megacaps at these delays, not news in general.
* **Why this predictor.** It is the new information version 2 adds, with no fitted parameters and no warmup. S2's population mixes price skill with news whose weights start at zero and move slowly, so S2 against S1 is descriptive.
* **Looks.** Sessions 10 and 20 are descriptive looks. There is no early stop for success, and the run stops early only when invalid under criterion 1.
* **A failure** means the test did not detect the correlation. That sets no bound by itself, so a failure is reported with the one sided 95 percent upper confidence bound, the largest correlation the data leave plausible. It says nothing about news after the close or about smaller companies.
* **Expected result, written before launch.** Not passed. Megacaps absorb headlines within minutes, the pipeline adds Finnhub's delay and up to a minute or two of batching, and most earnings come after the close.
* **Routing.** `news` is Haiku on every routed item, Finnhub and official, with FinBERT only where Haiku failed, was off or was capped. Sonnet does not enter it. Routing is fixed for the window, and `evaluate` checks that one config hash, one prompt hash and one model id per scorer cover it.

### 11.6 Success criteria

1. **The run is valid.** All of these must hold, otherwise the run is reported invalid with the reason.
   1. The tape check passed in Phase 13 and on the first clean full session at or after sessions 10, 20 and 40.
   2. There are zero lookahead audit violations.
   3. Every label recomputed from the candles matches.
   4. S1 and S2 have identical run minutes, labels and `linear_prices` rows.
   5. There are zero invariant violations.
   6. One config hash, one prompt hash and one model id per scorer cover the window.
   7. Key ledger spend of both models together, with every call that returned no usage booked at its reservation, stayed within 20 dollars in every UTC month.
   8. No tunnel ran, and both dashboards stayed bound to `127.0.0.1`, shown by a passed `local_only` check before every session of the window, as section 13.2 describes.
   9. Every code commit active during the window is listed.

   A fallback share above 25 percent of the items routed to Haiku in session is a caveat, not a failure, and so is a day that raised the spend alert.
2. **Confirmatory news skill,** as in section 11.5.
3. **Descriptive results.** Every predictor's skill, S2 against S1, `linear_news` against `linear_prices`, `haiku` against `finbert_matched` and `sonnet` against `haiku_official` over all rows and over event rows, `news_company` and `news_macro`, calibration, the twin test per run, any deaths, and data quality, meaning the healthy fraction, run minutes lost to news health by cause, channel outage minutes, receipt to scoring latency, items per source and the fallback share.
4. **The outcomes fixed by arithmetic** in section 12 matched.
5. **Version 1 undisturbed.** Runs A to D kept running and `test_v1_untouched` passes.
6. **Cost.** About 1 to 5 dollars a month for both models together and never above the 20 dollar cap, reported per model, expected to come from the Max plan's included credits as confirmed in Phase 15.

The project succeeds when criterion 1 holds and the criterion 2 result is reported, whatever it shows. A later version may size trades or choose holding periods only by a predictor that passed criterion 2, in a new run whose settings are fixed before its own launch, as the profit stage of section 11.7 sets out.

### 11.7 Road map after version 2

The road map of the version 1 plan, longer horizons, more market data, current affairs as a predictor and variable sizing, now runs in stages, each switched on only by evidence from the one before.

1. **The bench.** Version 1, runs A to D on Bitcoin, checks the machinery of learning, trading, health and evaluation.
2. **The skill stage.** Version 2, and any later run built the same way, asks only whether a predictor forecasts the next return, through a skill test fixed before launch, here the confirmatory test of section 11.5. It trades a fixed 500 dollar notional on a fixed 30 minute horizon, so its trades check the accounting and the direction, never a sizing or exit rule.
3. **The profit stage,** which the owner wants prioritized once the skill stage has done its job. Each bot predicts several horizons at once, for example 30 minutes, 2 hours, the rest of the session and the next session. It enters where the expected gain after costs is largest, holds while the expected remaining gain beats the cost of exiting, and exits when it does not, so the holding period comes from the predictions rather than a timer. It combines this with position sizes that grow with confidence, the variable sizing of step 4 of the version 1 road map.

Two conditions bind the profit stage.

* It switches on only once a predictor passes the skill test, and only predictors that passed may drive it.
* Its rules for choosing horizons, exits and sizes are written before its own launch and tested forward against a fixed horizon version of the same bots, launched beside it on the same data. They are never fitted to data already collected, version 2's recordings included.

The low stock costs are what make this plausible. A round trip here costs 3.6 to 6.6 bps against about 190 bps for the version 1 retail Bitcoin round trip, so a modest forecast held over a horizon the predictions choose has a chance to clear its costs, which a forecast of the same skill on Bitcoin at retail costs would not. A horizon past the close also needs overnight holding, which version 2 leaves out, as question 13 of section 19 notes. Version 2 itself keeps the fixed 30 minute horizon and fixed 500 dollar positions.

## 12. Outcomes fixed by arithmetic

These follow from costs, session rules and timers, are judged as matched or not matched, check the accounting rather than skill, and apply to both runs. They use planning volatilities, estimates of 20 percent a year for QQQ, 22 for MSFT, 25 for AAPL, 30 for GOOGL and AMZN, 35 for META, 45 for NVDA and 60 for TSLA, with three quarters of daily variance inside the session.

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

Whole share shorts scale these dollar costs with their actual notional of about 400 to 700 dollars at planning prices.

* **First trades** follow the timeline of section 6.4, with the first forced trades between about `10:30` and `13:30` ET in session 3.
* **Rhythm.** Then each learner makes about two forced round trips per full session, from a timer of 180 healthy minutes plus the wait for an allowed phase minute.
* **Burn.** A quiet learner pays 1.68 dollars of rent and about two round trips per full session, about 2.04 dollars for QQQ, 2.14 for the five middle names and 2.34 for META and TSLA, 2.18 on average, or by the cutoff 67.21 dollars of rent plus about 19 of round trips, about 86 dollars.
* **Founder equity.** The learners' mean equity equals 10,000 minus 0.0043081 dollars per usable minute of their home symbol minus the round trip costs they paid, within about 45 dollars at the cutoff, a two standard deviation bound if every learner's trades move together, taking a 30 minute move of about 50 bps on 500 dollars and about 75 trades each.
* **No deaths.** The threshold is about 2 dollars and the burn about 2.2 a session, so death would take about 4,500 sessions. There are no births, and every slot stays filled.
* **Flat at every close,** except where a close was rejected because its symbol went stale, which `evaluate` lists.
* **Pairing.** S1 and S2 predictions are identical until S2's first training step with a nonzero news input, expected in session 2.
* **The gate.** At least 95 percent of learner opens are forced while the realized prediction spread stays at or under 0.10 label standard deviations, which `evaluate` prints.

## 13. Dashboards, licensing and data handling

### 13.1 Why stock data stays private, and the Finnhub decision

* Finnhub's terms bar sharing data or derived results with any third party without written approval.
* The NYSE subscriber agreement counts all information that derives from market data as market data.
* From 1 April 2027 the CT Plan replaces the CTA and UTP plans, and its section 4.10 says a non professional's outputs may not be retransmitted, published or furnished to any other person. Sections 4.7 and 4.10 appear to permit a non professional's own automated use only when all outputs go solely to that user, and how Finnhub will implement this is unknown.
* Delaying prices does not help. Until April 2027, delayed redistribution of Nasdaq listed names needs a UTP licence at 250 dollars a month plus 250 a year.

**The owner's decision on Finnhub headlines.** Finnhub headlines and summaries are sent to the Anthropic API for scoring by Haiku, by the owner's decision. Finnhub's terms bar sharing its data with third parties without written approval, and sending headlines to the Anthropic API arguably counts. The realistic consequence is that Finnhub could suspend the free key, which also carries the price feed, so prices and company news would stop together. The owner accepts this risk. A written request to Finnhub describing this personal, non commercial scoring use would remove the doubt, and question 3 of section 19 recommends sending one. Only headline and summary text within the limits of section 7.6 is sent, never a value from the stock feed or anything derived from one, through the dedicated workspace key alone. If Finnhub objects, `[finnhub] send_to_llm` goes false at once, which ends the window under criterion 1.6 if it has begun.

So prices, candles, simulated fills, profit and loss, forecasts, labels and everything else derived from the stock feed stay on this machine. Recordings, databases, logs, exports, filled results and the official figures stay under the gitignored `data` folder. Only `docs/RESULTS_V2_TEMPLATE.md` with empty tables goes in the repository.

### 13.2 Local only, with no public page

Every dashboard stays local only, for version 1 and version 2 alike. The version 1 dashboard serves `127.0.0.1:8050` and the version 2 dashboard `127.0.0.1:8060`, neither is tunnelled, shared or published, no remote view is built, and no tunnel of any kind runs on this machine for either version. No public page exists to carry stock data or news.

**The recorded check.** Ten minutes before each open, and at start when the stock process starts inside a session, it runs `v2/localcheck.py` in a worker thread, so no minute step waits for it. The check lists the listening sockets through `lsof` and the running processes through `ps`, and fails when anything listens on port 8050 or 8060 at an address other than `127.0.0.1`, or when a tunnel or remote forward runs, such as `cloudflared`, `ngrok`, `tailscale funnel`, `tailscale serve` or `ssh -R`. It writes the verdict and the count of offending sockets and processes to `feed_events` as kind `local_only` in both run databases. `evaluate` fails criterion 1.8 of section 11.6 on any failed check or any session in the window without one.

### 13.3 Stock data and assistant sessions

Any AI coding or chat session that builds, runs or checks this system sends its context to a third party. The approved plan first kept stock prices and anything derived from them out of such sessions, on exchange data licensing grounds. On 12 October 2026 the owner lifted that rule, so the assistant session may read prices, volumes, per symbol figures, recordings, databases and the private dashboard, run commands with `--detail`, and send dashboard screenshots to the owner. Finnhub's terms restrict sharing its data with third parties, so the realistic risk is that Finnhub suspends the free key, which also carries the price feed, the same risk the owner accepted for headlines in section 13.1. The rest of the data handling stays.

* Every version 2 command prints by default only counts, pass or fail verdicts, hashes, timings of this system and synthetic data, and `--detail` adds rows, per symbol figures and samples.
* Apart from the assistant session, no stock data leaves this machine, whether through a page, a tunnel or a push, as section 13.2 sets out.
* Logs hold only states, counts and timings, never a price, volume or fill.
* Tests use synthetic data and hand written Finnhub fixtures only.
* Docs pushed to GitHub carry no feed quality numbers, which stay under `data/v2/checks`.

### 13.4 The private stock dashboard

`btc_survivor_v2 dashboard configs/v2/s1_prices.toml configs/v2/s2_news.toml` serves `127.0.0.1:8060`. The host cannot be configured, port 8050 is refused, any Host header other than `127.0.0.1` or `localhost` on its port gets 403, which blocks DNS rebinding, every database opens read only, and one five second cache and the version 1 security headers apply. Its panels are these.

1. **Market session.** State, next open and close in ET and UTC, holiday and early close flags.
2. **Finnhub feed.** State, prints and messages per second, last print age per symbol, print delay, feed events.
3. **News process.** Heartbeat, channels in outage, items per source today, key ledger spend for the day and the month against the 20 dollar monthly cap, the daily spend alert of section 8.6 when raised, the state of Haiku and Sonnet, fallback share, median receipt to scoring time.
4. **Prices.** Last reference price, session return and the session's one minute line per symbol.
5. **Forecast skill.** Per predictor rows, pooled correlation, t, detectable correlation, upper bound and squared error ratio against zero, from the `evaluate` functions, descriptive until the cutoff, with S2 against S1.
6. **Recent news.** Source, symbols, receipt and scoring times, Haiku fields, Sonnet fields for official items, FinBERT score and the score S2 used.
7. **Populations.** Every learner and twin with home symbol, state, equity, trades, position and next forced trade, and equity curves against an equal weight buy and hold of the eight, QQQ buy and hold, the matched exposure benchmark and cash.
8. **A footer** saying this is a private view of Finnhub data for personal non professional use, not to be shared, tunnelled or published.

## 14. Runtime and processes

### 14.1 On the Mac

| Process | Command | Network | Listens | Writes |
| --- | --- | --- | --- | --- |
| runs A to D | unchanged | Coinbase, keyless | nothing | `data/run_a` to `data/run_d` in the main tree |
| version 1 dashboard | unchanged, never tunnelled | nothing | `127.0.0.1:8050` | nothing |
| news | `btc_survivor_v2 news configs/v2/news.toml` | EDGAR, Fed, BLS, Finnhub REST, Anthropic | nothing | `data/v2/news/news.db`, the key ledger |
| stocks | `btc_survivor_v2 stocks configs/v2/s1_prices.toml configs/v2/s2_news.toml` | the one Finnhub websocket | nothing | `data/v2/s1_prices`, `data/v2/s2_news`, `data/v2/raw_stocks` |
| version 2 dashboard | `btc_survivor_v2 dashboard configs/v2/s1_prices.toml configs/v2/s2_news.toml` | nothing | `127.0.0.1:8060` | nothing |

All of them share one machine, because the stock process reads the news process's SQLite file.

### 14.2 The stock process

`v2/stocks/runner.py` runs one asyncio loop.

* **Feed.** One `StockFeedClient` whose sink redacts secrets before a line reaches the recorder or the group, with the `websockets` logger at WARNING.
* **Ticker.** Every second it calls `advance_to` on the group and runs the news puller.
* **News puller.** It reads new `item_scores` rows with their impacts and item fields, and new `news_heartbeat` rows, and emits recorded lines of kind `news` and `news_heartbeat` with sequence numbers, item and source fields, route, scorer, status, scored time, impacts and channels in outage, never headline text. It stamps them with the same live clock and engine floor as `FeedClient`, so recorded order equals stamp order. Each engine skips sequence numbers at or below its checkpointed value, and at first launch the start sequence is the newest in `news.db`, stored in `meta`.
* **Heartbeat.** Once a minute it writes `runtime_status` and `symbol_status` to both databases and logs feed state, prints per second, news age, living bots, memory and ticker lag.
* **Shutdown.** SIGINT or SIGTERM lets pending fills settle while the home symbol is healthy, checkpoints both engines and the shared layer, and exits with code 0.

`v2/stocks/group.py` dispatches `news` and `news_heartbeat` lines itself, before the version 1 `decode`, which would hand any other kind to `adapter.parse`. It decodes each price line once, runs the shared market layer and hands the same frozen results to both engines in order. Each engine has its own database, checkpoint, persistence points 30 seconds after a boundary, and resume, and resume refuses if the two checkpoints' copies of the shared layer differ. A restart in session is a gap that discards pending labels, and after more than 10 session minutes the next healthy minute flattens. A restart outside the session flattens nothing, and code updates deploy only then. The version 1 `Recorder` writes every line to `data/v2/raw_stocks`, apart from the version 1 `data/raw`, since there is no free history and recordings feed replay and the tape check.

### 14.3 The news process

`v2/news/runner.py` runs a task per channel, the Haiku and Sonnet scorers, the FinBERT scorer in a worker thread, the daily probe, the heartbeat every 30 seconds and daily pruning. SIGTERM finishes the current poll and call, commits and exits with code 0 within 30 seconds. It needs about 40 MB for collection and an estimated 300 to 500 MB with FinBERT, unverified until measured.

### 14.4 Locks, secrets and supervision

* `finnhub_ws.lock` is held by whatever owns the websocket, `finnhub_rest.lock` by the news process while Finnhub is enabled, and `llm_ledger.lock` around every ledger write, all under `~/.config/btc_survivor`.
* `secrets.env` must be mode 600, and `check-secrets` confirms the three names without reading values back.
* `scripts/v2_secret_scan.py` takes the values from `local_secrets.py` without naming the keys and scans every tracked file, the index, `data/v2`, the launchd log paths, and every blob, commit message and author and committer field from the remote branch to `HEAD`. It prints only counts and paths, before every push.
* On the Mac two launchd agents with `KeepAlive` and a `ThrottleInterval` of 10 seconds, plus `caffeinate -s` on power, are recommended, installed only with approval, with nohup and manual restarts as the alternative. A sleeping laptop is an outage, never a death.

### 14.5 Later on the Pi, CPU and memory

Two systemd units, `btc_survivor_v2_news.service` and `btc_survivor_v2_stocks.service`, with `Restart=always`, `RestartSec=10`, `After=network-online.target time-sync.target` and `systemd-time-wait-sync`, move together on a weekend after the window, because they share `news.db`. The move sends SIGTERM to both processes, copies `data/v2` and the key ledger, runs `PRAGMA integrity_check` on every copy, starts the services and repeats the first live session checks. Beforehand the soak is replayed on the Pi with no network to measure CPU per print, and FinBERT's 300 to 500 MB may not fit a small Pi.

The 48 learners and 48 twins cost an estimated 3 ms a minute on the Mac. Parsing prints dominates, at several thousand a second in the opening burst on the full tape, and the shared layer does it once. The stock process should need about 60 to 100 MB. The first session measures all of this, with ticker lag under 1 second at the open.

## 15. Configuration

Each loader rejects unknown sections and keys, wrong types and out of range values, naming the key, the bad value and the allowed range. Relative paths resolve from the config file's folder, and the experiment hash covers every section except `log_level`.

**Cross checks.** Symbols are unique, upper case and at most 50, the per key websocket limit, and include the index symbol. The first open minute plus the horizon plus the flatten margin fits in an early close of 210 minutes. `symbol_stale_after_seconds` is at least `feed_stale_after_seconds`, `max_print_delay_seconds` is between 2 and 5, and `late_print_seconds` is twice it. S1 may not set `use_news`. Every `half_spread_bps` symbol is in the universe, and the confirmatory predictor is known. The two run configs are identical apart from `[run]` `name` and `data_dir`, `[news_input]` `use_news` and `[evaluation]`. `keep_candles_days` is 0, kept for the life of the run, or at least 120 days, twice the calendar length of 40 sessions. In the news config `monthly_usd` is above zero and at most 20, the Console workspace limit, `daily_alert_share` is above zero and below one, `[llm.sonnet] kinds` holds only official kinds, and no model id other than the two named is accepted.

`configs/v2/s1_prices.toml`.

```toml
[run]
kind = "stocks"
name = "s1_prices"
master_seed = 12345
data_dir = "../../data/v2/s1_prices"
log_level = "INFO"
secrets_file = "~/.config/btc_survivor/secrets.env"

[stock_feed]                              # identical in both runs of one stock process
provider = "finnhub"
symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ"]
index_symbol = "QQQ"
feed_stale_after_seconds = 5
symbol_stale_after_seconds = 15
max_print_delay_seconds = 2.0             # set after the first recorded session by the rule in section 4.5
late_print_seconds = 4                    # twice max_print_delay_seconds
price_excluded_conditions = []            # filled before launch from the meanings in Finnhub's published table
reconnect_backoff_start_seconds = 5
reconnect_backoff_max_seconds = 60
mark_healthy_outside_session_seconds = 60
record_raw_messages = true
raw_dir = "../../data/v2/raw_stocks"
keep_raw_days = 30
lock_file = "~/.config/btc_survivor/finnhub_ws.lock"

[session]
first_open_session_minute = 30            # 10:00 ET
last_open_minutes_before_close = 35       # 15:25 ET, 12:25 on early close days
flatten_minutes_before_close = 5          # 15:55 ET
calendar_warn_days = 60
max_overnight_gap_pct = 25.0

[news_input]
use_news = false                          # true in S2
news_db = "../../data/v2/news/news.db"
poll_seconds = 1
reopen_after_silent_seconds = 60
heartbeat_max_age_seconds = 120
half_life_session_minutes = 120
drop_after_half_lives = 5

[time]
horizon_minutes = 30                      # session minutes, the only horizon
history_session_minutes = 391
warmup_hours = 6                          # training minutes

[features]
return_windows_minutes = [30, 120]
day_return_minutes = 390
vol_window_minutes = 60
volume_window_minutes = 30
volume_baseline_minutes = 390
volume_ratio_clip = 3.0
tick_window_minutes = 30
standardizer_half_life_hours = 168        # healthy session hours, also the label scale
clip_z = 5.0
min_samples = 30
min_window_coverage = 0.8
series_capacity_minutes = 420

[model]
inputs = 12                               # 10 price features plus two news inputs, pinned at zero in S1
hidden_units = 16
init_output_scale = 0.1
batch_size = 16                           # newly matured samples plus draws from the window
replay_window_minutes = 240               # session minutes, all symbols
huber_delta = 1.0
grad_clip = 5.0
adam_beta1 = 0.9
adam_beta2 = 0.999
learning_rate = 0.0001                    # placeholder, set by scripts/v2_learner_check.py before launch
weight_decay = 0.0001
skill_window_samples = 360

[trading]
start_equity = 10000.0
notional = 500.0
qty_decimals = 6                          # longs
short_whole_shares = true                 # nearest whole share, at least one
margin_sigma = 0.25
stop_loss_sigma = 3.0
allow_short = true
force_trade_after_hours = 3.0             # healthy session hours, from the first Active minute

[stock_costs]
commission_per_order = 0.0
half_spread_bps = { AAPL = 1.0, MSFT = 1.0, NVDA = 1.0, AMZN = 1.0, GOOGL = 1.0, META = 2.0, TSLA = 2.0, QQQ = 0.5 }
extra_slippage_bps = 1.0
sec_fee_per_million = 20.60               # SEC fee rate advisory, from 4 Apr 2026
finra_taf_per_share = 0.000195            # charged through the pause from 1 Oct to 31 Dec 2026
finra_taf_max_per_trade = 9.79
fee_round_up_to_cent = true
short_borrow_pct_per_year = 1.0
short_sale_restriction_drop_pct = 10.0
latency_ms = 1000
latency_jitter_ms = 300
daily_running_cost = 1.16                 # 4.23 percent T bill yield on 10,000, 8 Oct 2026
trading_days_per_year = 252

[population]
size = 24                                 # no replacement, a death leaves the slot empty

[controls]
coin_flip_twins = true

[persistence]
checkpoint_every_seconds = 300
equity_snapshot_every_seconds = 300
max_resume_gap_minutes = 10               # session minutes, per symbol
keep_candles_days = 0                     # kept for the life of the run
keep_checkpoints = 3
```

`configs/v2/s2_news.toml` differs only in these keys.

```toml
[run]
name = "s2_news"
data_dir = "../../data/v2/s2_news"

[news_input]
use_news = true

[evaluation]                              # S2 only, copied into meta at launch and read from there
confirmatory_predictor = "news"
final_healthy_session_minutes = 15600     # run minutes from launch
run_minute_rule = "news_healthy_and_any_symbol_healthy"
burn_in_sessions = 1
newey_west_lags = 12
one_sided_z = 2.33
upper_bound_z = 1.645
interim_sessions = [10, 20]
tape_check_sessions = [10, 20, 40]        # first clean full session at or after each
```

`configs/v2/news.toml`.

```toml
[news]
kind = "news"
data_dir = "../../data/v2/news"
secrets_file = "~/.config/btc_survivor/secrets.env"
master_seed = 12345                       # report samples
universe = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ"]
schedule_tz = "America/New_York"          # only to pick cadences
http_timeout_seconds = 10
max_response_bytes = 8000000
backlog_gap_seconds = 600                 # added to the longer cadence of the two polls, plus budget waits
backlog_exempt_kinds = ["8k"]
max_vendor_age_hours = 24                 # stale_vendor on both routes
clock_skew_max_seconds = 5
clock_skew_samples = 20
failure_backoff_cap_seconds = 900
heartbeat_seconds = 30
outage_cadence_multiple = 3
outage_floor_seconds = 300
score_wait_seconds = 600                  # or twice the measured burst time if longer
burst_items = 300
keep_filtered_days = 30
keep_polls_days = 14
keep_dups_days = 14
keep_heartbeat_days = 14
rest_lock_file = "~/.config/btc_survivor/finnhub_rest.lock"

[finnhub]
enabled = true
company_symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
general_category = "general"
max_calls_per_minute = 30                 # key limit 60
min_call_spacing_seconds = 1.1            # key limit 30 a second
regular_window_et = ["09:30", "16:00"]
extended_window_et = ["04:00", "20:00"]
regular_poll_seconds = 60
extended_poll_seconds = 120
overnight_poll_seconds = 600
lookback_days = 1
headline_max_chars = 300
summary_max_chars = 600
rate_limit_pause_seconds = 60
send_to_llm = true                        # the owner's decision of section 13.1, false sends Finnhub items to FinBERT alone

[edgar]
enabled = true
ciks = { AAPL = 320193, MSFT = 789019, NVDA = 1045810, AMZN = 1018724, GOOGL = 1652044, META = 1326801, TSLA = 1318605 }
forms = ["8-K", "8-K/A"]
filing_window_et = ["06:00", "22:00"]
atom_poll_seconds = 30
submissions_poll_seconds = 60
off_window_poll_seconds = 900
max_requests_per_second = 2
max_requests_per_minute = 60
block_pause_seconds = 900
main_doc_max_chars = 4000
exhibit_max_chars = 12000

[fed]
enabled = true
feeds = ["press_all", "press_monetary", "speeches"]
monetary_link_pattern = "/pressreleases/monetary"
press_window_et = ["07:00", "19:00"]
press_poll_seconds = 60
speeches_window_et = ["07:00", "21:00"]
speeches_poll_seconds = 600
speeches_off_hours_poll_seconds = 3600
off_hours_poll_seconds = 900
fomc_fast_window_et = ["13:58", "14:20"]
fomc_fast_poll_seconds = 15
summary_max_chars = 1000
monetary_body_max_chars = 8000
max_requests_per_minute = 10
block_pause_seconds = 900

[bls]
enabled = true
feeds = ["cpi", "empsit", "ppi", "jolts"]
window_et = ["07:00", "19:00"]
poll_seconds = 1800
off_hours_poll_seconds = 7200
release_fast_poll_seconds = 15
release_fast_before_minutes = 1
release_fast_after_minutes = 15
summary_max_chars = 1500
max_requests_per_minute = 10
block_pause_seconds = 900

[dedupe]
hash_window_hours = 24
near_dup_window_hours = 24
near_dup_jaccard = 0.8
near_dup_min_tokens = 4

[prefilter]
macro_strip_pattern = '\b(NASDAQ|NYSE|NYSEARCA)\s*:\s*[A-Z.]+'
alias_excludes_macro = true

[calendar]                                # ET times copied from federalreserve.gov and bls.gov before launch, polling only
                                          # every date through 31 Mar 2027, never changed after launch
fomc_statement = ["2026-10-28T14:00", "2026-12-09T14:00"]   # unverified until copied, the 2027 dates added before launch
cpi = []                                  # filled before launch
empsit = []
ppi = []
jolts = []

[llm]                                     # Haiku, on every routed item
enabled = true
model = "claude-haiku-5-5"                # the only accepted value
effort = "low"                            # the only accepted value
thinking = "disabled"                     # the only accepted value
scorer_name = "haiku_v1"
prompt_file = "prompt_v1.txt"
prompt_sha256 = ""                        # pinned at build
schema_file = "schema_v1.json"
schema_sha256 = ""                        # pinned at build
request_timeout_seconds = 60
max_tokens = 4096
flush_max_items = 20
flush_after_seconds = 60
max_items_per_call = 25
long_form_kinds = ["8k", "fed_monetary"]
alone_kinds = ["8k", "fed_press", "fed_monetary", "fed_speech", "bls_release"]   # one item per call
max_queue_age_seconds = 300
headline_max_chars = 300
text_max_chars = 700
long_form_max_chars = 24000
max_input_tokens = 20000                  # exact count, or bytes over 3 without one
max_calls_per_batch = 24
max_requeues = 1                          # a missing item only, never a failed call
rate_limit_pause_minutes = 15
refusal_pause_share = 0.2
refusal_window_calls = 50
probe_file = "probes_v1.json"
probe_time_utc = "06:00"
probe_alert_direction_agreement = 0.8     # probes run on Haiku only

[llm.sonnet]                              # the comparison scorer, official items only
enabled = true
model = "claude-sonnet-5-5"               # the only accepted value, and Opus is never accepted
effort = "low"                            # the only accepted value
thinking = "disabled"                     # the only accepted value
scorer_name = "sonnet_v1"
kinds = ["8k", "fed_press", "fed_monetary", "fed_speech", "bls_release"]
max_tokens = 1024                         # one item with up to 10 impacts
cache_control = false                     # Sonnet cache prices are not in the research

[llm.caps]                                # one ledger and one cap for both models
ledger_file = "~/.config/btc_survivor/llm_ledger.db"
ledger_lock_file = "~/.config/btc_survivor/llm_ledger.lock"
monthly_usd = 20.00                       # equal to the Console workspace limit, and no daily cap
daily_alert_share = 0.25                  # a warning above 5 dollars in one UTC day, never a stop
reserve_overhead_tokens = 1000            # byte bound only, checked against real overhead in Phase 15

[llm.prices]                              # Anthropic pricing page, read 11 Oct 2026
haiku_input_per_mtok = 0.10
haiku_output_per_mtok = 0.50
haiku_cache_write_multiplier = 1.25
haiku_cache_read_multiplier = 0.10
sonnet_input_per_mtok = 2.00
sonnet_output_per_mtok = 10.00
long_prompt_threshold_tokens = 100000     # Haiku, never reached under the 20,000 token guard

[finbert]
model_file = "../../data/models/finbert/model.onnx"
model_sha256 = ""                         # pinned at export
tokenizer_file = "../../data/models/finbert/tokenizer.json"
tokenizer_sha256 = ""
manifest_file = "../../data/models/finbert/manifest.json"
source_revision = ""                      # the pinned commit of ProsusAI/finbert
window_tokens = 510
max_windows = 16
threads = 1
```

## 16. Build phases with tests

Each phase ends with the full suite green on Python 3.11 and 3.13, one single line commit carrying only the owner's name, a push of `feat/v2-stocks` to the private repository, and a short summary. Before existing code changes, the modules and tests that depend on it are listed. No test needs the network, randomized tests use seeded generators, no test puts a secret value inside an assert expression, and the suite never runs with `showlocals`. Version 2 test files carry the prefix `test_v2_` so they never collide with a version 1 file, so the `test_store` of Phase 16 lives in `tests/test_v2_store.py`, while `test_v1_untouched`, `test_v1_isolation` and the amended version 1 tests keep their names.

**Critical path.** On 12 October 2026 the owner asked for the earliest safe launch, which moved the launch from 2 November to 19 October. Phase 14 is built, and the collector started, in parallel with the Phase 13 live gate, because collection does not depend on the tape verdict. Phases 15 and 16 are built in parallel with the verdict, because the engine reads candles rather than the vendor, and only their live use waits for a consolidated verdict. Phases 17 and 18 follow launch, apart from `scripts/v2_power.py`, because evaluation, status, export, the private dashboard, docs and units only read what the runs record and every criterion is already fixed in sections 11 and 12. The rehearsal is one session, and launch may fall on any session open.

| Gate | Phase | Latest date |
| --- | --- | --- |
| Plan approved, version 1 Phases 9 and 10 finished | 11 | done, Monday 12 October 2026 |
| Guard rails green | 12 | done, Sunday 11 October |
| Gate session recorded, the first clean full session | 13 | Monday 12 October |
| Tape verdict consolidated, after at most one repeat session | 13 | Tuesday 13 October, or Wednesday 14 after a repeat |
| Collector live, four trading days before launch | 14 | Tuesday 13 October before `04:00` ET |
| First live Haiku and Sonnet calls | 15 | Wednesday 14 October |
| Learner check fixes the learning rate, and `scripts/v2_power.py` prints its range | 16 and 17 | Wednesday 14 October |
| Rehearsal session | 19 | Thursday 15 or Friday 16 October |
| Launch | 19 | Monday 19 October |
| Evaluate, status, export, the private dashboard, docs and units | 17 and 18 | the first week of the window |

If any gate slips, the launch moves to the first session open after a passed rehearsal with the collector at least four trading days old.

**The calendar before launch.** From 19 October the window of 15,600 run minutes ends on 15 December 2026, the 41st session, if no minute is lost, and there are 52 sessions to 31 December, so the window ends within the year unless more than about 4,300 minutes are lost. Whatever the expected end date, `[calendar]` therefore holds every FOMC statement and every CPI, employment situation, PPI and JOLTS release from launch through 31 March 2027 before launch, which also covers a 60 session window. `[calendar]` cannot change after launch, because any change alters the news config hash, which fails news health for the rest of the window under section 7.7 and breaks criterion 1.6.

**Before Phase 11.** Version 1 Phases 9 and 10 finish on `feat/initial-bot`.

11. **This plan and its approval.** The owner approves this plan, including the fixed criteria in sections 11 and 12, before any version 2 code is written. The plan was drafted in the main tree and becomes the first commit on `feat/v2-stocks`.

12. **Guard rails, no network.** The worktree, `.venv-v2`, the `pyproject.toml` script and extra, `local_secrets`, `redact`, `output`, `localcheck`, the stricter order scan, the config loaders and `sessions.py`.
    * `test_v1_untouched` checks the empty `git diff` from the fork commit of section 3.
    * `test_v1_isolation` checks that no version 1 module imports `btc_survivor.v2`, the version 1 loader rejects every version 2 config, the four version 1 run configs still load, and trading modules never import `anthropic`, `onnxruntime` or `torch`.
    * `test_no_orders`, amended, finds the new broker patterns, allows `api_?key` only in `local_secrets.py`, and checks that `local_secrets.py` imports no network module, only `llm_client.py` imports `anthropic`, and no model id other than `claude-haiku-5-5` and `claude-sonnet-5-5` appears in `src`, so no Opus id can.
    * `test_local_secrets` covers parsing, a group or world readable file, a missing key named without a value, unknown names, `repr`, the three scrubbed Anthropic variables and `ANTHROPIC_LOG`.
    * `test_redact` finds a canary redacted in a log record, a formatted traceback, `sys.excepthook`, the asyncio handler, a close reason, a Finnhub error, a feed event detail, a recorder line and a websockets `InvalidURI` raised during connect.
    * `test_contact_absent` collects tracked files and seeded page JSON whose candidate substrings hash to the contact address's SHA 256 and asserts the list is empty.
    * `test_output` finds no canary price, volume or fill in any command's default output on seeded databases.
    * `test_localcheck` feeds fixed socket and process listings to the check, passing listeners on `127.0.0.1` only and failing a listener on 8060 at `0.0.0.0`, a `cloudflared` process and an `ssh -R` forward.
    * `test_v2_config` checks every cross check in section 15.
    * `test_sessions` checks every closure and early close for 2026 to 2028, early closes at `18:00` UTC on 27 November and 24 December 2026 and 26 November 2027 and `17:00` UTC on 3 July 2028, opens on 12 October 2026 and 31 December 2027, opens at `13:30` UTC on 30 October 2026, `14:30` UTC on 2 November 2026 and 12 March 2027 and `13:30` UTC on 15 March 2027, k = 1 at `09:31` on 2 January 2026, k at `09:31` on a Tuesday after a holiday Monday equal to k at `16:00` Friday plus one, sessions of 390 and 210 minutes with 12 and 6 evaluation minutes, dates past the table raising, the self test refusing a broken time zone database, and the warning 60 days before the end.
    * `test_docs_style` covers this plan with the prose rules only.

13. **Stock data and the tape gate.** The adapter, prints, candles, `StockSeries`, health, `StockFeedClient` with `BackoffSettings`, the lock, synthetic recordings with faults, `scripts/v2_record_stocks.py` and `tapecheck.py`. Then, with the owner's approval for the first use of the key, the recorder runs live and the tape check runs on the first clean full session, stopping the build for the owner unless consolidated. The delay rule sets `max_print_delay_seconds` and `late_print_seconds`, and the condition histogram, read against the published meanings, fixes `price_excluded_conditions`.
    * `test_finnhub_adapter` covers multi symbol trades, ordering, drops with counts, ping, error, unknown types, the eight subscribe frames and the token kept out of `repr`.
    * `test_stock_candles` puts a print received at B minus 1 ms in candle B and one at B in the next, classifies by vendor time, counts a late print for volume only, refuses a print more than 2 seconds early, ignores replayed prints, takes the median of up to 5, resets at the open, carries an empty minute and applies the tick rule.
    * `test_stock_series` checks the capacity of 420, the buy and sell rings and session starts.
    * `test_stock_health` checks closed with no events outside the session, `mark_healthy` after 60 seconds connected outside it, a healthy `09:31`, the silence clock starting at the open, one symbol stale alone after 15 seconds, total silence for 5 seconds staling every symbol with a reconnect, a delayed feed held out, the delay rule at its floor and cap, epochs bumping at the close, the pre open reconnect, resubscribe limits, a 5 minute halt never reconnecting, and `BackoffSettings` giving the reused `FeedClient` a backoff from 5 to 60 seconds with seeded jitter, reset by `mark_healthy`, with no version 1 `FeedConfig` built.
    * `test_lock` makes a second process that takes the lock exit with a clear error.
    * `test_tapecheck` runs on a synthetic raw recording with no database, maps ratios to verdicts at the band edges, turns two inconclusive sessions into not consolidated, refuses disconnected time, and prints only the verdict without `--detail`.

14. **News collection,** in parallel with the Phase 13 live gate. `db`, `http`, `budget`, `schedule`, `text`, `dedupe`, `prefilter`, the sources, the collector, `health` and `news-status`. Then, with the owner's approval, the collector runs live from here on.
    * `test_news_db` checks that `init-db` on the news config creates `news.db` in WAL mode with the tables of section 9.1 and `meta.kind` set to news, and that the news process refuses a database of another kind.
    * `test_news_http` puts the contact header only on the three hosts that need it, keeps the token out of URLs, decodes gzip, refuses an oversized body and maps 304.
    * `test_news_budget` holds every host to its limits, pauses and capped backoff on a fake clock.
    * `test_news_schedule` checks cadences on both sides of every window in EDT and EST, across 1 November 2026, on weekends and in the fast windows.
    * `test_news_sources` parses the fixtures, keeps the two forms for the seven CIKs, finds Exhibit 99.1, records truncation, stores one item for a release in two Fed feeds, and gives an FOMC statement kind `fed_monetary` in both orders of arrival.
    * `test_news_text`, `test_news_dedupe` and `test_news_prefilter` check the hash, Jaccard at 0.79 and 0.8, the same source and 24 hour limits, a silent repoll, case sensitive aliases, every tier and junk pattern, Musk alone tagging nothing, `(NASDAQ: AAPL)` never tagging MACRO, an aliased item never tagged MACRO, and routing per kind and source, with Finnhub items routed `llm` and routed `finbert` only while `send_to_llm` is false.
    * `test_news_collector` checks that `received_at` never decreases and matches `seq` order across interleaved awaits, a first poll is all seed, BLS at 1,800 and 7,200 seconds, Fed speeches at 3,600, Finnhub at 600 and a cadence slowing from 15 to 1,800 seconds all give status new, a 60 second channel after a 20 minute restart gives backlog except for filings, a stale vendor time gives `stale_vendor` from every scorer, a filing waits for its documents, and clock skew marks polls.
    * `test_news_immutability` finds no `UPDATE` on news tables and checks pruning against retention.
    * `test_news_health` flips the heartbeat flag for each condition in section 7.8 and lists a channel outage without making news unhealthy.
    * `test_news_status` checks the status mix, backlog per channel per hour, outage minutes, and the order and backlog verdicts failing on a planted out of order row and a planted backlog item outside any restart, with samples only under `--detail`.

15. **Scoring.** `export_finbert.py` once with the owner's approval, then the FinBERT scorer, model request, client, validation, key ledger, batcher, the Haiku and Sonnet scorers, probes and `scripts/llm_synthetic_check.py`. FinBERT's time on the 300 item synthetic burst sets `score_wait_seconds`. Then, with the owner's approval, the first live Haiku and Sonnet calls.
    * `test_llm_request` checks identical bytes for identical items, the pinned hashes, no sampling parameters, tools, metadata or fallbacks, thinking disabled, effort low, the exact model id of each scorer, the same system prompt, schema and user message for Haiku and Sonnet on an official item, `max_tokens` of 1,024 for Sonnet, no BTC in the enum, cache control on Haiku's one system block and none for Sonnet, no dates, a sentinel contact address absent, and the size guard on the count and on bytes.
    * `test_llm_validate` catches enum violations, extra fields, missing, duplicate and unknown ids, duplicate tickers and more than 10 impacts, keeping the valid items.
    * `test_llm_scorer` uses a fake client to isolate one refused item among 20 in at most 11 calls, split on `max_tokens`, cap calls per batch, requeue a missing item once, and give every other row of the failure table its action, including a 5xx, a timeout or a call lost at restart booked at its reservation with its items marked failed and no second call. It also checks that every Finnhub item routed `llm` reaches Haiku and never Sonnet, Sonnet sees only the official kinds, a Sonnet error never changes Haiku's state, and a refusal by the monthly cap turns both models off until the next UTC month with FinBERT covering every item.
    * `test_ledger` checks the cost formula by hand, the byte bound never below a fake exact count, 10,000 seeded usage sequences mixing Haiku and Sonnet calls at their own rates that never pass the monthly cap and never cost more than their reservations, two processes on one key ledger that never pass it together, a reservation blocking a call at the edge, UTC month rollover, no daily cap of any kind, and the daily alert raised once when a day passes a quarter of the monthly cap, logged, written as `spend_alert` and set on the heartbeat, while calls continue.
    * `test_batcher` checks the flush at 20 items and 60 seconds, every official item alone and never batched with a Finnhub item, deterministic order, expiry after 300 seconds, and no Finnhub item batched while `send_to_llm` is false.
    * `test_score_store` refuses a second final row for one item and scorer.
    * `test_finbert` checks windowing, weighting, label order, strength and a hash mismatch on a fake session, and a `needs_model` test checks bitwise identical scores on two runs.

16. **Stock engine and replay.** The shared market layer, features, news inputs, labels, `PooledWindow`, `predict_rows`, costs, broker, bot, twins, population, engine, group, runner, `StockStore`, records, checkpoint and replay. Then `scripts/v2_learner_check.py` fixes the learning rate by the rule in section 6.4.
    * `test_stock_features` requires identical vectors from a full synthetic series and the same series cut at each t, across nights, weekends, early closes and gaps, and hand checks every feature, including feature 9 invalid without healthy reference prices.
    * `test_news_inputs` checks decay in session minutes, a score counting only from its receive stamp and never before `scored_at`, full weight to the open, the shrinkage, QQQ's aggregation, a fallback visible only at the later of its two receipts, the `finbert_matched`, `sonnet` and `haiku_official` streams holding only their paired items and each visible only at the later of its two receipts, Sonnet never entering the primary stream, exactly one primary score per item across every route and status, inputs stored per symbol at each push minute with zeros in S1, and a score received after a record's push minute never entering that record's training sample or its `linear_news` update, though it moves the current inputs.
    * `test_stock_labels` checks that labels never cross a close or an epoch change, with the right evaluation minutes on early closes.
    * `test_pooled_window` and `test_predict_rows` check the symbol column, newest n plus draws and per symbol scaling, each row against `MLP.predict`, and bitwise equal S1 and S2 outputs with zero news inputs.
    * `test_stock_costs` checks SEC fee rounding of 0.02 at 500 and 0.03 at 1,000, the trading activity fee minimum and cap, fees only on sales including short opens, round trips of 3.6, 4.6 and 6.6 bps, borrow across a weekend, and the required edge.
    * `test_stock_broker` fills at the first eligible print at or after due, including exactly at due, refuses a print with vendor time before the decision, rejects on stale symbol, disconnect and session end, fills a flip at one print, rounds longs to 6 decimals and shorts to whole shares with a one share minimum and recorded notional, and applies the short sale restriction, never triggered without a healthy previous price, and the split guard.
    * `test_stock_twins` checks that `StockTwin` does its upkeep on its learner's home symbol only, marking at the reference price, keeps the version 1 coin flip rules, takes the `session_close` and `session_flatten` exits, and never builds or reads a `LocalBook`.
    * `test_stock_bot` checks 8 predictions from one pass, trading only the home symbol, the open window, the close at `15:55`, the forced timer from the first Active minute, and the stop at 3 label standard deviations.
    * `test_zero_init` checks equal S1 and S2 weights and generator states after birth, bitwise identical predictions while every batch has zero news inputs, S1's news weights staying zero, and divergence after S2's first training step with a nonzero news input.
    * `test_stock_account` runs 5,000 seeded sequences of opens, closes, flips, stops, session closes, rent, borrow, cent rounded fees and deaths, checking the three version 1 invariant lines after every step.
    * `test_stock_engine` checks a synthetic day with 390 candle rows per symbol, no flatten after a night, weekend or holiday, a process killed at `15:54` with a position closing it next morning with `session_flatten` and `overnight_carry`, a 15 minute feed outage flattening, a 20 minute halt flattening only that symbol, a 15 minute news outage pausing rent, learning and decisions in both runs while bumping no epoch, flattening nothing and leaving the next day's feature 3 valid, a death leaving its slot empty, rent only in usable minutes, the DST Monday, and the invariants at every snapshot.
    * `test_stock_replay` replays one recording of prints and news lines twice with identical table digests, and never sends a frame or opens `news.db`.
    * `test_group` checks one decode feeding both engines, news kinds dispatched before the version 1 `decode`, recorded order equal to stamp order, a puller opened before the writer exists seeing later rows, a reopen after 60 seconds without a heartbeat, each engine resuming from its own news sequence, and a `local_only` event written to both databases before each open and at a start inside a session.
    * `test_store` checks that `init-db` creates a run database with `meta.kind` set to stocks and a news database with `meta.kind` set to news, the `StockStore` schema, its check and `rollback_after` on the version 2 tables, and that the stock process, the puller, `evaluate`, `status`, `export` and the dashboard each refuse a database of the wrong kind or a version 1 database.
    * `test_stock_checkpoint` checks a bit for bit round trip and resume refusing a changed experiment hash, session index or shared layer copy.

17. **Evaluation, status, export and the private dashboard,** with `scripts/v2_power.py`.
    * `test_stats` checks the pooled correlation against the mean within symbol correlation, a hand computed Driscoll Kraay error, eight copies of one symbol giving its t, a fast null simulation rejecting near 1 percent, the upper bound, and hand examples for `pooled_dm` and session blocks.
    * `test_evaluate` passes a planted correlation of 0.2 and fails zero, computes the cutoff from launch with partial halts and news outages, puts the window at sessions 2 to 40, recomputes labels from candles older than 30 days, reports deaths, reads settings from `meta`, and fails on a planted lookahead row, a label mismatch, a second prompt hash, a second model id for one scorer, a month whose ledger spend passed 20 dollars, differing S1 and S2 labels, or a failed or missing `local_only` check. It also checks that `sonnet` against `haiku_official` and `haiku` against `finbert_matched` each use exactly their paired rows and paired event rows, and that both comparisons are printed without a pass label.
    * `test_arithmetic` checks burn, founder equity and the death horizon by hand.
    * `test_private_dashboard` checks binding to 127.0.0.1 only, refusing port 8050, 403 for a foreign Host header, failing writes, the security headers, JSON free of secrets, and the daily spend alert shown when a seeded heartbeat carries it.
    * `test_status` and `test_export` check output against a seeded database.
    * `test_power_script` reproduces detectable correlations near 0.076 at c of 0.2 and 0.123 at c of 1.0 with fixed fast settings.

18. **Docs and deploy.** `docs/v2_overview.md`, `v2_data_and_licensing.md` with the short sale, margin and pattern day trader limits, `v2_news.md`, `v2_evaluation.md`, `v2_operations.md` and `docs/RESULTS_V2_TEMPLATE.md`, each section ending with links and covered by both docs rules, plus the launchd plists and systemd units, installed only with approval.

19. **Soak, rehearsal and launch.**
    * `scripts/v2_soak.py` replays the recorded sessions plus a synthetic multi day run with many reconnects, halts and news bursts, reporting memory against the 5 MB bound after the first hour, CPU per print, invariants and determinism.
    * A live rehearsal of one session uses `rehearsal_s1.toml` and `rehearsal_s2.toml`, separate data folders and the production `news.db` read only, with a plumbing warmup of 30 training minutes and a forced timer of 30 healthy minutes, labelled never a strategy setting. It starts during the session before its main day, so features have their history, and trading runs live on the main day. The drills kill the stock process mid session, stop and restart the production news process, which briefly interrupts collection, SIGSTOP the stock process past the stale limit, and run the cap drill below with its tiny monthly cap.
    * The cap drill runs on one rehearsal day. `scripts/llm_synthetic_check.py --cap-drill` writes the drill config to the gitignored `data/v2/rehearsal/cap_drill_news.toml`, a copy of `configs/v2/news.toml` with `[finnhub] enabled = false`, `[edgar] enabled = false`, `[fed] enabled = false`, `[bls] enabled = false`, `[llm.sonnet] enabled = false`, `data_dir` set to `data/v2/rehearsal/cap_drill`, a tiny `monthly_usd = 0.01`, and `ledger_file` and `ledger_lock_file` pointing to a throwaway ledger in that folder, so it is never a committed run config and its tiny cap never meets the month's real spend on the key ledger. With every channel off the drill polls nothing, so no real item reaches Haiku and only the items the drill inserts are scored. Sonnet stays off because a single Sonnet reservation of about 0.014 dollars or more would pass the tiny cap at once. It starts a separate news process on that config and then inserts fictional long form official items from its own synthetic set into the drill database, each as a fresh item with status `new` and route `llm`, and the next only after the previous one has its final Haiku row, so no item is a seed and each goes alone to a reserved call or to the cap check. It inserts at most 20. A long form Haiku call books about 0.001 dollars and reserves about 0.003, and the throwaway ledger starts empty, so the cap refuses a call within about ten items. The drill passes when a call is refused by the cap, the waiting items get `skipped_cap`, Haiku stays off until the next UTC month, FinBERT scores every item, and the drill's booked spend stays at or under 1 cent. When it ends the script books the drill's total into the key ledger as one row, so the key ledger still holds every cent the key spent. The drill database and its ledger stay under `data/v2/rehearsal` and are never read by a run.
    * S1 and S2 then launch with fresh databases at a session open, the news process already running, on the date the critical path gives.

## 17. First live session checks

Checks marked **owner** print prices, volumes, headlines, fills or per symbol figures with `--detail`, and since 12 October 2026 the assistant session may run them on the owner's behalf. Every other check prints verdicts and counts only.

**Before the first connection.** The owner approves the first use of the Finnhub key and confirms the Mac owns the one connection. `check-secrets` reports the three names in a mode 600 file, the secret scan finds zero copies of any value, the version 1 suite and `test_v1_untouched` pass, runs A to D show unchanged heartbeats, and `check-local` passes, with no tunnel process and both dashboard ports on loopback only.

**Feed and calendar, Phase 13.**

* The connection opens with no error, every symbol prints within 15 seconds of `09:30`, and the open is at `13:30` UTC under EDT and `14:30` UTC from 2 November.
* The delay verdict passes and the delay rule's setting is computed. **Owner** reads the per symbol p50, p95 and maximum delay and enters the setting, and a median above 5 seconds stops the build.
* Pings are counted, the connection survives an overnight idle with no reply sent, and disconnects over 24 hours are counted by reason.
* Messages and prints per second with peaks from `09:30` to `09:35` and `15:50` to `16:00`, bytes per second, compressed recording size, CPU, memory, and ticker lag at the open under 1 second.
* 390 candle rows per symbol, no stale events at the open or close, epochs bumping at the close, and a healthy share at or above 99 percent on a clean day.
* The tape verdict. **Owner** reads the per symbol ratios, first and last print times, condition histogram, odd lot, zero size, late and replayed shares, the last print against the official close, and each stale event against Nasdaq's public halt list.
* **Owner** compares the Roll half spread with the allowance, report only, and a few public quotes with the last print.

**News, Phase 14.**

* sec.gov, data.sec.gov and bls.gov answer 200 with the contact header and gzip, and 304 support is recorded. BLS answers 200 for all four feeds with matching file names.
* The Fed feeds parse and store a two feed item once.
* Finnhub accepts the token header, company news over two days returns items, `minId` returns only newer ids, and bytes per call are logged.
* No Finnhub 429 with the websocket and collector together, with REST at or under 10 calls a minute.
* The Atom feed parses, the CIKs match `company_tickers.json`, and every first poll is all seed.
* `news-status` prints by default a passed order verdict, `received_at` never decreasing and equal to `seq` order, and a passed backlog verdict, no backlog outside restarts and outages.
* **Owner** checks the first filing for any of the seven against sec.gov and reads the headline samples.
* If a scheduled release falls in the week, its fast window works and its delay from schedule to receipt is recorded.

**Scoring, Phase 15.**

* Before the first real call the owner confirms the dedicated workspace with the 20 dollar monthly limit, auto reload off, and included credits linked with 7 days on the plan, and the news config holds the same 20 dollars as `monthly_usd`.
* `scripts/llm_synthetic_check.py --dry-run` records the body hash and counted tokens. One live synthetic call returns HTTP 200, `response.model` equal to `claude-haiku-5-5`, no thinking blocks, stop reason `end_turn`, and JSON passing the schema with integer enums. One live synthetic Sonnet call on a fictional official item does the same with `claude-sonnet-5-5`, which settles whether Sonnet accepts thinking disabled at effort low, the request without sampling parameters and the integer enums, and its usage shows no cache tokens. If Sonnet rejects any setting other than the integer enums, it stays disabled as section 8.1 sets out.
* `count_tokens` is free, accepts the full request, confirms the prefix exceeds the cache minimum of 512 tokens, which is unverified for Haiku 5.5, and confirms the byte bound with `reserve_overhead_tokens` is never below the real count. Real bytes per token are recorded.
* A second call within five minutes shows cache reads near the prompt size and no cache writes.
* The 40 item synthetic check passes the schema on every Haiku answer and matches the answer key, written before any call, on at least 90 percent of clear cases, its official items also pass the schema on every Sonnet answer, and two back to back Haiku probe runs record the launch baseline.
* From the first 50 real items, tokens per item, cost per call and reservation against cost are recorded for each model, with cost never above reservation. The next day the key ledger total for both models matches the Console usage page, and the charge comes from the included credits.
* Refusals, malformed results, thinking blocks and 429s are zero, receipt to scoring lag is recorded per model, every Finnhub item routed `llm` has a final Haiku row, every official item has a final Sonnet row, and no Sonnet row exists for a Finnhub item.
* FinBERT's hash matches, its time per headline, long item and 300 item burst and the news process memory are recorded, and two runs over the 50 fictional sentences are bitwise identical.

**The runs, from launch.**

* S1 and S2 have identical run minutes, labels and `linear_prices` rows, and identical predictions until S2's first training step with a nonzero news input.
* `inputs_max_recv_at` is never after its row's minute, every label recomputes from the candles, every news line's receive stamp is at least its `scored_at`, and news inputs change only at the first boundary after receipt.
* Vectors, training and forced trades start as section 6.4 predicts.
* No opens before `10:00` or after `15:25` ET, no open position after `15:55`, and zero invariant violations. **Owner** recomputes a handful of fills by hand, print plus allowance, 0.02 and 0.01 dollars on a sale of about 500, nothing on buys, a whole share short and its borrow.
* After a mid session restart of the news process the puller reads new heartbeats within 60 seconds, and lost minutes are reported by cause.
* Runs A to D keep their heartbeats, CPU and memory, both dashboards answer only on `127.0.0.1`, a passed `local_only` event precedes every session, the version 2 dashboard gives 403 to a foreign Host header, `pmset` shows no sleep in session, and after a day the secret scan finds no token, Anthropic key or contact address in the repository, `data/v2`, the logs or the push range.

## 18. Risks and honest limits

* **The free feed may not be the full tape.** Finnhub's docs title the stream US Full SIP but name no tier, and the research could not confirm the free plan carries it. Every result depends on the tape check, and Finnhub could narrow the tier later.
* **Lost messages are invisible** without sequence numbers, so loss is estimated only in aggregate.
* **No quotes, no depth.** Fills rest on a fixed allowance, paper orders never move the market, and the allowance may be kind at the open or close for some names.
* **Shorts are kinder than a small account.** Intraday shorting needs a margin account, and the pattern day trader rule limits an account under 25,000 dollars to three day trades in five business days against about ten a week per learner. Whole share shorts also move notional away from 500 dollars. The confirmatory test does not depend on either.
* **Regular session only,** so the overnight reaction to earnings, where headline value is most likely, is invisible.
* **Delays.** Received time measures this system, not the market, and vendor delays, channel outages and filings after `17:30` ET all make news arrive late.
* **News health selects minutes.** Scoring backlogs around FOMC, CPI or earnings could remove minutes near the biggest news, which the burst sizing and the report of lost minutes by cause make visible.
* **Weak macro surprise.** Without consensus forecasts, a BLS release's surprise is judged from its text alone.
* **Word lists** miss stories naming only a product and catch a few false matches, and tickers are fixed at first receipt.
* **Mixed scales.** The primary stream uses FinBERT strengths, on a different scale from Haiku's, for fallback items, whose share is reported.
* **Non determinism and drift.** Every result rests on one set of scores from each model, Haiku only days old, and a rerun would differ by about what the probes measure for Haiku, while Sonnet runs no probe.
* **Lookahead in a model's head.** Live scoring is clean by construction, but megacap names can distract either model.
* **Power.** The detectable news correlation runs from about 0.075 to about 0.12 as the shared macro input dominates, and power at a true 0.075 may be only 35 to 40 percent.
* **Comparisons with little power.** Sonnet against Haiku rests on about 5 to 20 official items a day, so only a large difference could show, and both model comparisons stay descriptive.
* **No daily cap.** A runaway day could spend most of the 20 dollar month before anyone looks, after which FinBERT covers every item until the month resets. The daily alert at 5 dollars makes such a day visible on the log and the private dashboard, and scoring each item once with deduplication limits what a loop can send.
* **No deaths, no evolution.** Version 2 exercises neither, and runs A to D already do.
* **One machine and one connection.** The processes share a SQLite file, and one Finnhub connection rules out a second streamer.
* **Licensing.** Every affordable stock source forbids public display, and Finnhub forbids sharing its data or derived results with third parties without written approval. Sending Finnhub headlines to the Anthropic API arguably counts, and Finnhub could suspend the free key, which would stop prices and company news together and end the window. The owner accepts this risk, and a written request to Finnhub would remove the doubt. The CT Plan changes on 1 April 2027, so continuing past then needs the terms checked again.
* **Thin holiday sessions** in late December include an early close.
* **Estimates.** Print rates, CPU, memory, bandwidth, news volumes, model costs, volatilities, spreads and the power assumptions are estimates until measured.
* **The Mac.** A sleeping or rebooting laptop is an outage that keep alive supervision reduces but cannot remove.

## 19. Open questions for the owner

Each question carries a recommendation only, and silence never approves anything. Questions 1, 5, 6, 7, 9 and 11 need an explicit answer from the owner before the step each one gates, named in the question. The others follow their recommendation once the owner approves the plan under question 1, unless the owner answers otherwise, and question 2 stays open to a different answer until launch, after `scripts/v2_power.py` prints its range.

1. Is this plan approved, including the worktree, the stricter order scan, the sigma scaled margin, intraday only trading, and the fixed criteria and outcomes in sections 11 and 12? Recommended, approve. This gates Phase 11, and no version 2 code is written before an explicit yes.
2. Should the window stay at 40 sessions from Monday 19 October 2026, ending in mid December, or run 60 sessions to late January, once `scripts/v2_power.py` prints the power range? Recommended, 40 sessions, with the upper confidence bound reported, settled before launch because `[evaluation]` is fixed then.
3. Will the owner send Finnhub a written request describing the scoring of its headlines through the Anthropic API for this personal, non commercial use? Recommended, yes, before launch, since an approval would remove the doubt the owner's decision of section 13.1 accepts. Sending headlines does not wait for an answer and nothing is gated on it. An objection turns `send_to_llm` off at once, which ends the window if it has begun.
4. Will the owner copy the eight daily volumes and closes from nasdaq.com by hand after the gate session and each repeat, and which other consolidated display could replace it? Recommended, nasdaq.com, with a second display named only if needed.
5. If the tape check fails, which fallback is preferred? Recommended, Databento Standard at 199 dollars a month, whose key is data only and whose licence covers automated use, with Alpaca only if its data scope OAuth route is proven. This gates any spend on a fallback, and after a failed tape check the build stays stopped until the owner answers.
6. Before Phase 15, is the key in a dedicated workspace with the 20 dollar monthly limit and auto reload off, with the Max plan's included credits linked and at least 7 days on the plan? Recommended, the owner confirms each point. This gates the first live Haiku and Sonnet calls in Phase 15.
7. Are the FinBERT export, which downloads about 440 MB from Hugging Face into a throwaway environment, and the daily probe of about 0.04 dollars a month approved? Recommended, approve both. This gates the export and the first probe in Phase 15.
8. Should the launch go without company name masking? Recommended, unmasked, with a masked variant possible later as its own predictor.
9. May launchd agents with keep alive, plus caffeinate, be installed for the two version 2 processes? Recommended, yes. This gates the install in Phase 18, and nothing is installed without an explicit yes.
10. May aggregate feed quality numbers go into docs pushed to GitHub? Recommended, no, they stay under `data/v2/checks`.
11. Are the non professional, age and US residency attestations made at Finnhub signup accurate? Only the owner can answer. This gates the first use of the key in Phase 13.
12. Which Raspberry Pi model and memory will host version 2 later, and should runs A to D move with it? Recommended, everything stays on the Mac through the window.
13. May GDELT, company newsroom feeds, Form 4 and Form 144 filings, an earnings calendar, a Bitcoin input and overnight holding wait for a later version? Recommended, yes, with overnight holding taken up again by the profit stage of section 11.7, whose next session horizon needs it.

## 20. References

* Finnhub [pricing](https://finnhub.io/pricing), [websocket trades](https://finnhub.io/docs/api/websocket-trades), [rate limits](https://finnhub.io/docs/api/rate-limit), [company news](https://finnhub.io/docs/api/company-news) and [terms of service](https://finnhub.io/terms-of-service)
* [NYSE holidays and trading hours](https://www.nyse.com/markets/hours-calendars)
* Alpaca [market data](https://docs.alpaca.markets/docs/about-market-data-api), [paper trading](https://docs.alpaca.markets/docs/paper-trading), [OAuth and trading API](https://docs.alpaca.markets/docs/using-oauth2-and-trading-api), [historical stock data](https://docs.alpaca.markets/docs/historical-stock-data-1), [terms and conditions](https://files.alpaca.markets/disclosures/library/TermsAndConditions.pdf) and the [NYSE market data display services agreement](https://files.alpaca.markets/disclosures/library/NYSE+Market+Data+Display+Services+Agreement.pdf)
* Databento [pricing](https://databento.com/pricing) and [US equities catalog](https://databento.com/catalog/us-equities), and the [IEX filing on market share and fees](https://www.sec.gov/files/rules/sro/iex/2026/34-105989.pdf)
* CT Plan [policies](https://consolidatedtape.com/policies/policies-docs.html) and [fees](https://consolidatedtape.com/fees), the [CTA delayed market data policy](https://ctaplan.com/publicdocs/ctaplan/Policy_Delayed_Market_Data.pdf) and [UTP SIP fees](https://datashop.cboe.com/sip-fees)
* SEC [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [developer resources](https://www.sec.gov/about/developer-resources), [webmaster FAQ on fair access](https://www.sec.gov/os/webmaster-faq), [privacy and reuse](https://www.sec.gov/privacy), [filing status and dissemination times](https://www.sec.gov/page/edgar-how-do-i-determine-status-my-filing), [NVIDIA submissions JSON](https://data.sec.gov/submissions/CIK0001045810.json) and [Alphabet third quarter 2025 Exhibit 99.1](https://www.sec.gov/Archives/edgar/data/1652044/000165204425000087/googexhibit991q32025.htm)
* Federal Reserve [feeds](https://www.federalreserve.gov/feeds/feeds.htm) and [disclaimer](https://www.federalreserve.gov/disclaimer.htm), BLS [feeds](https://www.bls.gov/feed/) and [linking and reuse](https://www.bls.gov/bls/linksite.htm), and [GDELT](https://www.gdeltproject.org/about.html)
* Anthropic [pricing](https://platform.claude.com/docs/en/about-claude/pricing), [models overview](https://platform.claude.com/docs/en/models/overview), [Haiku 5.5 overview](https://platform.claude.com/docs/en/models/haiku-5-5/overview), [Haiku 5.5 migration guide](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide), [extended thinking](https://platform.claude.com/docs/en/build-with-claude/thinking), [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [rate limits and spend limits](https://platform.claude.com/docs/en/api/rate-limits), [API credit expiry](https://support.claude.com/en/articles/8977456) and [Max plan API credits](https://support.claude.com/en/articles/17154008)
* [Look ahead bias in GPT sentiment](https://arxiv.org/abs/2309.17322), [FinBERT](https://github.com/ProsusAI/finBERT) and the [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
* [Version 1 plan](PLAN.md)
