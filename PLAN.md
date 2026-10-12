# BTC Survivor plan

This is the Phase 0 deliverable. The plan was approved on 10 October 2026 for a US account, so every recommendation in section 1 stands. It covers the module layout, the data flow, the configuration, the database schema, the test plan, and the answers to the five open questions. Phases 1 to 10 follow in order, and each ends with the full test suite, a commit and a short phase summary.

Everything here was checked against live sources on 10 and 11 October 2026 UTC. That includes keyless probes of the Coinbase and Kraken public websockets from the development Mac, the official Coinbase fee page, and local benchmarks of the hot paths on Python 3.12.11. Measured numbers are marked as measured. Everything about the Raspberry Pi is an estimate until it runs on real hardware.

## 1. Key decisions

These choices change behaviour or results. Each has a recommendation, and section 2 gives the evidence. Section 4 lists the smaller readings of the brief that the build follows.

| Decision | Brief says | Recommended | Why in one line |
| --- | --- | --- | --- |
| Market data source | Coinbase Exchange websocket | Coinbase Advanced Trade websocket, channels `level2`, `market_trades`, `heartbeats` | The Exchange `level2` channel now requires keys, and its keyless `level2_batch` has no sequence numbers, so gaps cannot be detected |
| Coinbase region | not stated | the US (confirmed) | The Intro taker fee is 90 bps in the US, 50 bps in the EU, UK and Canada, and 10 bps in the rest of the world |
| Retail taker fee | 30 bps placeholder | 90 bps for a US account | Coinbase cut its US Intro tier from 120 to 90 bps on 16 September 2026, and 30 bps is the rate for 50,000 dollars a month of volume |
| Low fee taker fee | 5 bps placeholder | keep 5 bps | It equals Coinbase VIP3 exactly, which needs 20 million dollars a month |
| Default fee profile | not set | `retail_default` | It answers the honest question, and the low fee profile is one config line away |
| Optional middle profile | not in brief | add `active_retail` at 30 bps | It keeps the brief's original 30 bps as a named, sourced scenario |
| Short opening fee | not in brief | add `short_open_fee_bps = 2.0` | Kraken charges 1 to 2 bps to open a margin short, so hourly funding alone would be kinder than the venue it models |
| Daily running cost | 5 dollars | 1.16 dollars | That is the 4.23 percent three month Treasury yield on 10,000 dollars, so a bot must beat cash |
| Horizon | 15 or 30 minutes | 30 minutes | The same fee is paid against a move about 1.4 times larger |
| Fitness formula | net change over the standard deviation of hourly changes | mean hourly change over that standard deviation, multiplied rather than divided by it when negative | The literal formula favours the youngest eligible bots, and among losing bots it ranks the most volatile highest |
| Start equity versus notional | 10,000 and 500 | keep as written for the main run, and add a run at 500 | At 10,000 the first death is likely more than a year away, so a run at 500 is what exercises evolution |
| Coin flip controls | not in brief | one coin flip twin per learner | A twin trades at the same moments as its learner with a random direction, so a gap between them measures skill rather than luck |
| Parallel runs | one run | three runs from the same master seed | Run A at retail fees and 10,000 gives the skill result, run B at the low fee and 10,000 shows the same forced schedule at a 5 bps fee, and run C at retail fees and 500 checks the death and lineage machinery |
| Success criteria | not in brief | fixed before launch, in section 21 | With 20 bots and overlapping labels, chance alone produces a bot that looks impressive |
| Headline result | not in brief | the population's prediction skill against predicting zero and a plain linear model | At retail fees the costs decide survival before launch, so skill is the one open question the runs can answer |
| Learning rate | 0.001, bounds 0.0001 to 0.01 | 0.0001, bounds 0.00001 to 0.001 | On synthetic data with a planted signal, 0.001 made each bot's predictions mostly noise and about 16 percent worse than predicting zero |
| Forced timer start | counts from birth | starts at a random delay of up to 12 hours drawn from each bot's own generator | Otherwise all bots leave warmup together and make their forced trades in the same few minutes twice a day, so 20 bots act like one bet |
| Web dashboard | none in version 1 | a read only local page, `btc_survivor dashboard`, served on 127.0.0.1 only | The owner asked to watch the bots train and trade, and a read only page on the same queries as `status` cannot change a run |
| Python floor | 3.11 or newer | keep 3.11, with numpy pinned per Python version and the suite run on 3.11 and 3.13 | numpy 2.5 needs Python 3.12, and the Raspberry Pi OS Legacy image still ships 3.11 |

## 2. Answers to the five questions

### 2.1 Which keyless feed channels exist right now (question 1)

Each channel was subscribed on its own connection with no credentials, at about three in the morning UTC on 11 October 2026, and the current official documentation was read.

**Coinbase Exchange feed** at `wss://ws-feed.exchange.coinbase.com`. The channels `level2`, `level3` and `full` are rejected without keys. The server replies that these channels "now require authentication". The keyless channels are `heartbeat`, `status`, `auctionfeed`, `matches`, `ticker`, `ticker_batch` and `level2_batch`. The `level2_batch` channel sends a full depth snapshot and then updates batched every 50 milliseconds, but neither carries a sequence number. A dropped update would silently corrupt the local book. That fails the brief's requirement to detect gaps.

**Coinbase Advanced Trade feed** at `wss://advanced-trade-ws.coinbase.com`. The channels `level2`, `market_trades`, `heartbeats`, `ticker`, `ticker_batch`, `candles` and `status` need no key. Only `user` and `futures_balance_summary` need one. All three channels the project needs run on one connection. Every message on a connection carries `sequence_num`, a single counter shared by all channels on that connection. In the probe it ran from 0 to 1118 with no gaps across 1,119 messages. Any jump means a message was lost, so the book is thrown away and rebuilt. One Coinbase overview page describes the counter as per product, but the official AsyncAPI specification and the live probe both show it is per connection. Treating any jump on the connection as a gap is safe under either reading.

What the probe measured on BTC USD, in a quiet weekend hour.

* The `level2` snapshot is the full book, about 40,800 levels and 4.5 MB of JSON in one message. That is four times the websockets library's default message size limit of 1 MiB, so the client must raise it.
* Parsing that snapshot peaked at about 20 MB of Python objects and raised peak memory by 56 MB in a fresh process on macOS. The book itself then takes about 3.3 MB.
* Steady state was 17 messages and about 35 price level changes per second, at about 7 KB per second.
* Trades arrived at 1.45 per second. Trade identifiers were contiguous. `market_trades` batches every 250 milliseconds, newest first, and its `side` field is the maker side.
* About one delete per second arrived for a price level that was not in the book. These are harmless and must not trigger a resync.
* The snapshot can arrive before the subscription acknowledgement, so the adapter must not wait for the acknowledgement.
* The spread was one cent, which is 0.0012 bps. Walking the book for 500 dollars cost about 0.0006 bps. At this size the cost of a trade is almost entirely the fee.

**Recommendation.** Use Advanced Trade `level2`, `market_trades` and `heartbeats` on one keyless connection. Detect gaps with `sequence_num`. Resync by reconnecting, so there is one recovery path. Keep the full book internally, because a cancel at the top pulls deeper levels into the top 20, and expose only the top 20.

**Fallback, not built now.** Kraken WebSocket v2 at `wss://ws.kraken.com/v2` offers a keyless book at depth 25 with a CRC32 checksum on every message. The probe verified the checksum on 300 of 300 messages. Its snapshot is under 2 KB. A `feed/kraken.py` adapter would keep its own small depth 25 book of exact price strings to verify checksums and trim levels Kraken pushes out of depth without a delete, and would emit ordinary snapshot and delta events. No other module would change.

Coinbase notes that keyless connections may be less reliable than authenticated ones. The reconnect and resync path is built and tested for exactly that reason. The project never asks for or creates keys.

### 2.2 Do the fee placeholders match, and which profile should be the default (question 2)

Coinbase changed its Advanced fee schedule on 16 September 2026. Its blog post gives Intro taker rates by region. The rate is 90 bps in the US, 50 bps in the EU, the UK and Canada, and 10 bps in the rest of the world. The public [Coinbase Advanced fee tiers](https://www.coinbase.com/advanced-vip) page, read on 10 October 2026 without signing in, gives these US spot rates.

| Tier | 30 day volume in dollars | Maker bps | Taker bps |
| --- | --- | --- | --- |
| Intro | 0 | 50 | 90 |
| Advanced 1 | 10,000 | 35 | 75 |
| Advanced 2 | 50,000 | 15 | 30 |
| Advanced 3 | 250,000 | 10 | 15 |
| VIP1 | 1 million | 6 | 10 |
| VIP2 | 5 million | 4 | 6 |
| VIP3 | 20 million | 1 | 5 |

So the 30 bps placeholder is the US Advanced 2 rate, not what a new retail account pays. A new US account pays 90 bps, and paid 120 bps before 16 September. The 5 bps placeholder is exactly VIP3. Coinbase One Premium returns 25 percent of Advanced fees, which would make 90 into about 67.5, but that is a paid subscription and is not modelled.

A real account would also climb tiers on the bots' own volume. A quiet bot trades about 2,000 dollars a day, so it would reach 75 bps in about 5 days and 30 bps in about 25 days. A fixed 90 bps is therefore the rate of a new account and harsher than reality after the first weeks. That is the safe direction, so fixed profiles are recommended now. A tiered profile driven by each bot's trailing 30 day volume could be added later from the public table.

**Recommendation.** For an account in the US, set `taker_fee_bps_retail_default = 90.0` and make `retail_default` the default profile. Elsewhere use that region's Intro rate. Keep `low_fee_tier` at 5.0. Optionally add `active_retail` at 30.0. Every trade row stores gross profit and fees separately, so a run's own trades can be repriced under another fee as a rough sensitivity check. That is not what a run under that fee would do, because the fee sets the required edge and the death threshold, and so it changes every decision, death and parent choice. Each profile needs its own run.

**Short funding.** Kraken's short selling page says Coinbase US does not offer spot margin shorting. That is a competitor's statement which could not be verified with Coinbase, so it is treated as likely rather than certain. A realistic US retail short means Kraken margin or a perpetual style future. Kraken charges 0.01 to 0.02 percent per 4 hours to roll a BTC margin position, which is 0.25 to 0.5 bps per hour, plus 1 to 2 bps to open. On perpetuals in 2025 and 2026 shorts were usually paid, on average about 0.05 to 0.08 bps per hour. The 0.5 bps per hour placeholder is the top of Kraken's rollover range, so it is conservative for the hourly part. On its own it misses the opening charge, which for a 30 minute short is larger than the rollover. The recommendation is to keep 0.5 and add `short_open_fee_bps = 2.0`, charged once at each short open and included in the cost estimate for a short. The modelled short leg is a composite of the active profile's spot taker fee and Kraken's margin charges, and `docs/execution_model.md` will say so with sources and dates.

**Other cost inputs.** The measured median one way feed latency was 17 milliseconds on the `level2` channel used for fills and about 95 milliseconds on the batched `market_trades` channel. Order round trip latency was not measured. The configured 1,000 milliseconds plus up to 300 milliseconds of jitter is deliberately slower than the feed, which is the right direction. The extra slippage of 1 bp per leg is larger than anything the book walk produced at 500 dollars, which is also the right direction.

### 2.3 What daily running cost is sensible (question 3)

Rent should answer one question. Does the bot beat leaving its capital in cash? The honest anchor is the cost of capital. The three month Treasury yield was 4.23 percent on 8 October 2026, according to the Federal Reserve's DGS3MO series. On 10,000 dollars that is 1.159 dollars a day. **The recommendation is 1.16 dollars a day**, rounded up so the bots are not flattered.

This is a trade off, not a free choice. For a quiet bot, one that only trades when forced, rent at 5 dollars is 22 percent of its daily burn under the 90 bps profile, 45 percent under 30 bps and 81 percent under 5 bps. At 1.16 dollars those shares fall to 6, 16 and 49 percent, and a quiet bot lives 1.2, 1.5 and 2.6 times longer. Under the low fee profile, rent is the main thing that kills a quiet bot. A higher rent therefore speeds up deaths and evolution, and a lower rent keeps the comparison anchored to cash. Neither value fixes the timescale problem in section 3. Only start equity relative to notional does that.

### 2.4 How many bots fit on a Raspberry Pi 4 and on this laptop (question 4)

Measured on the development Mac, an M4 Pro, with Python 3.12.11, numpy 2.5.4 and websockets 17.2.

| Item | Measured on the development Mac | Pi 4 estimate |
| --- | --- | --- |
| One bot for one minute (predict, batch of 8, backward pass, Adam) | 31 microseconds | about 0.45 milliseconds |
| Shared features and standardizer per minute | 19 microseconds | about 0.3 milliseconds |
| Parse and apply one `level2` message | 30 microseconds | about 0.4 milliseconds |
| Whole cost of one message, with websocket, TLS and event loop wakeup | 0.13 to 0.21 milliseconds when paced at 100 a second | about 0.6 milliseconds central, 2.4 pessimistic |
| Memory after imports | 41 MB | not measured |
| Memory with the full book and shared state | 51 MB | not measured |
| Extra memory per bot | 0.03 MB | not measured |
| Extra peak memory while parsing a 4.5 MB snapshot | 20 to 56 MB | not measured |

The bot figure is the simple implementation in section 8, measured in Phase 4. Preallocated output buffers measured 19 microseconds in the planning benchmark, an optimization kept in reserve because 20 bots do not need it. The Pi estimates use published PyPerformance results for a 1.8 GHz Pi 400, which has the same processor as a recent Pi 4. They put it at about 7 to 9 times slower than the development Mac for pure Python, with a range of 6 to 13 times across benchmarks. Older Pi 4 boards run at 1.5 GHz and are about 10 times slower. Small numpy calls and JSON parsing were assumed to be 10 to 15 times slower. Memory on the Pi was not measured, and Linux may hold freed memory differently from macOS.

**Pi 4 CPU.** The bots themselves cost about 9 milliseconds a minute in total. Feed handling dominates, and most of its cost is the event loop waking for each message and the websocket and TLS work around the parse, which is why the whole per message cost is larger than the parse alone. At the measured quiet rate of 17 messages a second, the estimate is about 1 to 4 percent of one core. At an assumed busy rate of 100 messages a second, it is about 6 percent in the central case and about 24 percent in the pessimistic case, which is essentially the 25 percent target. Busy hour message rates were not measured. Phase 3 will measure one on a weekday afternoon UTC. The target "under 25 percent average CPU" is read as 25 percent of one core, the stricter of the two readings.

**Pi 4 memory.** Twenty bots need an estimated 110 MB at peak. About half of that is the baseline and about half is the brief spike while a snapshot is parsed after a reconnect. Each extra bot costs about 0.03 MB, plus about 0.08 MB while a checkpoint is written. A rough ceiling under 150 MB is therefore a few hundred bots, limited by memory, not CPU. This is an estimate to confirm on the target Raspberry Pi.

**This laptop.** Neither memory nor CPU limits it at hundreds of bots. CPU stays under 3 percent of one core. Checkpoint size, database growth and how readable the output is become the practical limits first.

**Recommendation.** Use 20 on the Pi and 20 on the laptop for the first runs, so the two runs can be compared. Raise the laptop to 100 later for more diversity. Phase 10 runs the soak test on the development Mac. The Pi numbers must be confirmed on the target Raspberry Pi, and `docs/raspberry_pi.md` will explain how.

### 2.5 Fifteen or thirty minute horizon (question 5)

Coinbase's daily closes give an annualized BTC volatility of 42 percent in 2025 and 46 percent so far in 2026. Coinbase's minute data gives a standard deviation for a 15 minute log return of about 22 to 24 bps, and for a 30 minute return about 32 to 34 bps. The average absolute move was about 15 bps over 15 minutes and about 21.5 bps over 30 minutes. Square root of time scaling held well from 1 to 30 minutes. The last 30 days were calmer, with a 30 minute standard deviation of about 26 bps.

The required edge is the cost of a round trip plus the margin gene. It does not grow with the horizon, but the move does.

| Profile | Required edge in bps | Multiple of the 15 minute standard deviation | Multiple of the 30 minute standard deviation |
| --- | --- | --- | --- |
| retail 90 bps | about 190 | about 8 | about 6 |
| active retail 30 bps | about 70 | about 3 | about 2.1 |
| low fee 5 bps | about 20 | about 0.85 | about 0.6 |

The 30 minute horizon has real costs. It produces half as many independent outcomes per day, so learning and the skill statistics are noisier. And every exit from a healthy feed discards up to one horizon of pending labels, so a 30 minute horizon loses twice as many labels per gap as a 15 minute one. The keyless gap rate is unmeasured, and Phase 3 will measure it. The project's own checks found 15 and 30 minute return autocorrelation close to zero. Older studies and studies of other venues suggest gross predictable edges of about 0.5 to 5 bps per trade at these horizons, which is an inference, not a measurement on Coinbase. Even perfect foresight would earn only about 21.5 bps per 30 minute trade on average.

**Recommendation.** Start with 30 minutes. Neither horizon is likely to be profitable after retail fees, but 30 minutes is strictly less hopeless for the same cost. If Phase 3 shows frequent gaps, its phase summary will say so and the horizon can be revisited. The code supports both, and changing the horizon is a config change.

## 3. What to expect from the default experiment

At 500 dollars a trade, one leg costs about the fee plus 1 bp, because spread and book impact round to zero. That is 4.55 dollars at 90 bps, 1.55 dollars at 30 bps and 0.30 dollars at 5 bps. A bot dies when its equity falls below one round trip plus one day of rent, which is between about 2 and 10 dollars. So a bot must lose almost all of its 10,000 dollars before it dies.

The table shows days until death from costs alone. Market gains and losses barely move it, because a typical 30 minute trade on 500 dollars moves about 1.65 dollars either way. "Quiet" means a bot that only trades when forced every 12 hours and closes at its next decision, which is 4 legs a day. "Active" means it opens or closes at every decision. "Churning" means it flips side at every decision. Rent is the recommended 1.16 dollars.

| Fee profile | Quiet | Active, 30 minute horizon | Churning, 30 minute horizon | Active, 15 minute horizon |
| --- | --- | --- | --- | --- |
| retail 90 bps | 516 days | 45 days | 23 days | 23 days |
| active retail 30 bps | 1,358 days | 132 days | 67 days | 67 days |
| low fee 5 bps | 4,237 days | 643 days | 334 days | 334 days |

With 5 dollars of rent the quiet column becomes 430, 892 and 1,612 days. The retail row barely changes elsewhere, but under the low fee profile an active bot falls from 643 to 515 days and a churning one from 334 to 296. Short opening charges and funding are left out of the table. If half of all opens were short, they would shorten the low fee row by about 5 to 8 percent and the other rows by under 2 percent.

What this means.

* **The first trade.** On a fresh launch, features need 6 hours of healthy history. The label scale then needs about 30 matured labels, so training starts after about 7 hours. Six hours of warmup follow, so bots become Active after about 13 hours. Each bot's forced timer starts at a random delay of up to 12 hours, so the first forced trades are spread over about 13 to 25 hours after launch, and then each bot makes one forced round trip every 12 hours.
* **The gate.** The model starts with predictions of a few bps, because the output layer starts at a tenth of its normal scale, and a learner with realistic skill keeps them that small. Under the retail profile the required edge is about 190 bps, many times a typical 30 minute move of 17 to 34 bps, so the gate will not pass in practice. Every retail trade will be forced, so every retail bot will be quiet. At the 5 bps fee the required edge is still about 20 bps, several times the size of the predictions, so run B will be almost entirely forced too. Section 21 writes down the outcomes this fixes before launch.
* **The first death.** Under these defaults the first death is likely more than a year away, and because bots are only replaced when they die, evolution will not start in any practical run. Parents would also need 20 closed trades, which a quiet bot takes about 10 days to make.
* **The lever.** This is a real finding, not a bug, and the logs will show it plainly. What changes the timescale is start equity relative to notional. With 500 dollars of start equity, a quiet retail bot dies in about 25 days and a churning one in about a day, which is why run C in section 21 starts at 500. That is the lowest start equity the configuration allows, because the notional may not exceed it.
* **What will be visible.** `status` shows each bot's burn rate over the last 24 hours and the days left at that rate, so the timescale is visible while the system runs.

The main run keeps 10,000 and 500 as the brief wrote them, because changing them would be tuning and the brief names 10,000 dollars repeatedly. Run C at 500 sits beside it rather than replacing it. The death, replacement, mutation and lineage machinery will still be exercised by unit tests and by the replay soak. The soak uses a separate plumbing configuration in its own data folder with small start equity, short parent requirements and a short forced timer, and it is labelled as never a strategy setting.

## 4. Changes to the brief and chosen readings

**Renames and additions.**

1. **`engine.py` is new.** It holds the deterministic core that turns feed events and time into minute steps, fills, persistence points and bot updates. `runner.py` keeps only the asyncio tasks, signals and database writes. `replay.py` drives the same engine from a file. This is how live and replay share exactly one pipeline.
2. **`skill.py` is new.** It holds the rolling prequential statistics, so `model.py` stays a pure network and optimizer.
3. **`benchmark.py` and `baselines.py` are new.** The first is a few lines that compute the buy and hold curves. The second holds the linear baseline that the forecast records score next to the population (section 21).
4. **`logsetup.py` is new.** It configures standard library logging once.
5. **`synthetic.py` is new.** It generates recordings in the Advanced Trade message format, with a random walk market and injected faults such as gaps, crossed books and silences. Tests and the soak script use it. The live path never imports it.
6. **`scripts/record_feed.py` is new.** It is a small development tool that runs the adapter, the book and the health monitor with the live reconnect rules, plus the recorder, without the bots or the database. Real recordings can therefore start in Phase 3 and replay exactly like recordings made by `run`. It logs each exit from Healthy. It is not a command line command. `scripts/learner_check.py` is new too. It checks the learner on synthetic data with a planted signal and never reads market data (section 8).
7. **`book.py`, `candles.py`, `health.py` and the new `market.py` live inside `feed/`**, next to `base.py` and `coinbase.py`. `market.py` applies feed events to the book, the candles and the health monitor and decides when to resync, so the engine and the development recorder share one copy of that logic. The indentation in the brief's layout was lost, and they are feed components that know nothing about any one exchange.
8. **One ordered event stream.** `feed/base.py` defines four event types. `BookSnapshot` and `BookDelta` form the order book output, `Trade` is the trade output, and `FeedNotice` carries a kind, a connection id and the exchange timestamp for connects, disconnects and gaps. They arrive on one ordered stream instead of separate streams, because arrival order between a trade and a book update matters and replay must reproduce it exactly.
9. **Resync means reconnect.** Every gap, crossed book or stale socket closes the connection and opens a new one, which brings a fresh snapshot. There is one recovery path. Clock skew is the exception, because reconnecting cannot fix a wrong local clock.
10. **New tables.** `fills` records every fill and rejection. `lineage` records mutation details. `meta` records the launch price and the schema version. `runtime_status` is one row that `status` reads for the live price. `forecasts` records each predictor's forecast for every healthy minute. Several existing tables gain columns, all listed in section 12.
11. **New config keys.** Each one is in section 16 and marked new. They are the clock skew limit, the feature coverage rule, the minimum sample count, the skill window, the skill gate threshold, the short opening fee, the optional active retail fee, the tournament size, the fitness floor, how many checkpoints to keep, and the coin flip twins switch.
12. **Clock skew guard.** A Pi 4 has no hardware clock and boots with a stale time until NTP syncs. The health monitor compares the exchange's message timestamps with local time and holds the feed out of Healthy, without reconnecting, while they disagree by more than 2 seconds.

**Readings of the brief.** Each is explained in the section named.

13. The local book keeps full depth and exposes the top 20 (section 7).
14. Candles are assigned by receive time, not exchange time (section 6.1).
15. A fill is priced right after the first book update received at or after decision time plus latency (section 6.4).
16. Warmup counts minutes in which the bot actually took a training step. Parent age counts healthy minutes since birth. Lifespan and the age column in `status` use wall clock time (section 9).
17. The forced exploration timer starts at minus a seeded random delay at birth and then counts healthy minutes since birth or since the last open, and forced exploration applies only to a flat bot (section 9).
18. Decision phase uses the bot's slot, and a replacement takes the slot of the bot that died (section 9).
19. Funding accrues through stale minutes and downtime, while rent does not (section 10).
20. Every restart discards pending labels, not only long ones, because any restart is a feed gap (section 13).
21. The flatten rule for gaps longer than `max_resume_gap_minutes` also applies to long outages while the process stays up (section 13).
22. Weight decay is the decoupled AdamW form on the weight matrices only (section 8).
23. Fitness uses the mean hourly change and penalises volatility when the mean is negative, following the approved row in section 1 (section 11).
24. After a later gap a feature window counts as valid once 80 percent of it is healthy, instead of a fresh six hour wait (section 8).
25. The standardizer and the label scale give each new sample the larger of the exponential weight and 1 over the sample count, so early statistics are not biased (section 8).
26. A bot trains only in minutes in which a new sample matured, so each gap pauses learning for one horizon (section 8).
27. A phase minute without a valid feature vector is not a decision tick and writes no row (section 6.3).
28. Every learner slot also holds a coin flip twin that copies the learner's trade timing with a random direction (section 21).
29. The baseline that never trades is computed as start equity minus rent rather than simulated (section 21).
30. Each learner decision row later records the label that matured for it, and a new `evaluate` command checks the success criteria (section 21).

## 5. Module layout

```
btc_survivor/
  pyproject.toml
  requirements-plot.txt        matplotlib only, never installed on the Pi
  README.md  LICENSE  CONTRIBUTING.md  PLAN.md  .gitignore  config.example.toml
  deploy/btc_survivor.service
  docs/  architecture.md  learning.md  execution_model.md  evolution.md
         operations.md  raspberry_pi.md  honesty_and_limits.md  RESULTS_TEMPLATE.md
  src/btc_survivor/
    __init__.py        version
    __main__.py        argparse entry point and command dispatch
    config.py          load TOML, validate, frozen dataclasses
    clock.py           LiveClock, ReplayClock, UTC millisecond helpers
    types.py           small dataclasses for events, orders, fills
    logsetup.py        logging configuration
    feed/
      base.py          FeedAdapter interface and the four event types
      coinbase.py      Advanced Trade adapter: connect, subscribe, parse, sequence check
      book.py          LocalBook: full depth dicts plus sorted price lists, top 20, walk
      candles.py       one minute candles and the shared minute series
      health.py        HealthMonitor state machine and feed epoch
      market.py        applies events to book, candles and health, asks for resyncs
    features.py        the 11 features from the minute series and the book
    standardizer.py    z scores with warmup weighting and a variance floor
    labels.py          pending label queue, matured window, label scale
    model.py           numpy MLP, Huber loss, backward pass, AdamW
    skill.py           rolling prediction versus label statistics
    broker.py          cost estimate, pending orders, latency, book walk fills
    account.py         one bot's books, invariant, death threshold
    bot.py             Warmup, Active, Dead state machine, per bot minute step, decision rule
    controls.py        coin flip twins that copy their learner's trade timing
    genes.py           margin and learning rate genes, mutation, bounds
    population.py      slots, fitness, eligibility, tournament, replacement
    benchmark.py       buy and hold curves
    baselines.py       online ridge baseline scored next to the population
    engine.py          deterministic core shared by live and replay
    store.py           SQLite wrapper, schema, batched writes, rollback to a checkpoint
    records.py         row builders for every table the engine writes
    checkpoint.py      atomic state snapshot and restore
    runner.py          live asyncio tasks, signals, database writes
    recorder.py        raw message recording, gzip JSON lines, daily rotation
    replay.py          drive the engine from recordings at full speed
    synthetic.py       synthetic recordings for tests and soak
    dashboard.py       read only local page and its JSON view, with dashboard.html
    cli/
      data.py  stats.py  status.py  explain.py  lineage.py  export.py  evaluate.py
  tests/               one test file per module, plus integration tests
    fixtures/          small hand written message files
  scripts/
    record_feed.py     development recorder, adapter, book, health and recorder
    learner_check.py   synthetic learner capability check, never reads market data
    soak_test.py       long replay with memory, CPU and invariant report
    plot_results.py    laptop only charts from exported CSV
```

Only `feed/coinbase.py` parses Coinbase messages, and only `synthetic.py` and the test fixtures emit them, for tests. `config.py` loads the adapter by name from the `feed` package, so a new exchange is one new file.

## 6. Time model and data flow

### 6.1 One timeline for live and replay

Every event carries a receive time in integer milliseconds since the UTC epoch. Live, that is the wall clock at the moment the adapter reads the message, raised if needed so it is never earlier than the previous stamp or the engine's current time. The stamp is recorded after that adjustment, so replay sees exactly what live saw. In replay, the receive time comes from the recording. The engine never reads a clock itself. It only learns that time has advanced, which is what makes replay deterministic.

The monotonic clock measures durations. That covers how long a minute step or a database write took, how far message handling lags behind arrival, and the reconnect backoff.

Candles use receive time, not exchange time. A trade that the exchange stamped at 12 minutes 59.9 seconds but that arrived after the 13 minute boundary belongs to the next candle, because at the boundary it was not yet known. This is the no lookahead rule applied to the data itself.

### 6.2 Event handling order

The engine exposes two calls. `advance_to(t)` processes every scheduled item due at or before `t` in time order. Scheduled items are minute boundaries and persistence points. `on_event(event)` first calls `advance_to` with the event's receive time, then applies the event to the book, the candle builder and the health monitor, and then settles any pending fill that is due, if the event was a book update.

Live, the feed task calls `on_event` for every message. A ticker task wakes every second and calls `advance_to` with the current time, so boundaries and persistence points fire even when no messages arrive. In replay, only `on_event` is called. Both give the same result, because the book at a boundary is always the book after every message received before that boundary.

### 6.3 The minute step

At each UTC minute boundary M the engine does the following, in this order.

1. **Close the candle** for the minute ending at M. Take the book mid at M as the minute mid.
2. **Judge the minute.** It is healthy if the feed was Healthy for the whole minute, or if it is the full fresh minute that completes a recovery, and a valid book exists at M. Every minute, healthy or not, is appended to the shared series and buffered as a candle row with its validity flag, so windows always line up with the clock. An unhealthy minute stops here. Nothing trades, nothing learns, no rent accrues, and no labels are created.
3. **Shared work, done once.** Compute the 11 features if the windows have enough valid history. Standardize them using statistics from before this minute, then update the statistics with them. Mature the label for minute M minus the horizon, if that record is still in the current feed epoch. Update the label scale and the linear baseline with the matured sample.
4. **Gap flatten.** If the time since the previous healthy minute exceeds `max_resume_gap_minutes`, send a closing order for every open position, with exit reason `restart_flatten` after a restart or `gap_flatten` after a live outage, and write a `resume` event.
5. **Dying bots.** Resend the closing order of any bot that died with a position whose close has not filled yet.

A bot has at most one pending order. Steps 4, 5, 6.3 and 6.4 send nothing for a bot that already has a close pending, step 4 skips bots on the dying list, and a bot that dies while a flatten or stop close is pending keeps that order as its death close.
6. **For each bot, in slot order.**
   1. Charge rent for one minute. Accrue funding on any open short.
   2. Mark the position to the exit side of the book.
   3. Check death. A dying bot moves to the dying list, sends its closing order if it has a position, and leaves the loop.
   4. Check the stop loss. A triggered stop sends a closing order.
   5. If a new sample matured this minute, record the prequential result for it and take one training step.
   6. Predict for minute M and queue the record.
   7. At the bot's phase minute, decide. A phase minute without a valid feature vector is not a decision tick. The bot keeps any position, its stop check still runs, no row is written, and it decides again at its next phase minute. Otherwise a Warmup bot writes a decision row with action none and reason warmup. A bot whose stop fired this minute writes action none and reason stopped. A bot with an order still pending, including a flatten close, writes action none and reason pending. An Active bot runs the decision rule and may send orders.

   Substeps 5 and 6 need a valid feature vector. Rent, marking, the death check and the stop check run in every healthy minute.
7. **Replace dead bots.** Each replacement takes the slot of the bot that died, so the population is full again before the minute ends.
8. **Record the minute.** Buffer the candle, benchmark, decision, forecast and, when due, equity snapshot rows in memory.

### 6.4 Order fills

A decision at M creates a pending order due at D, which is M plus `latency_ms` plus a jitter drawn uniformly from 0 to `latency_jitter_ms` with the bot's own seeded generator. The order is settled right after the engine applies the first book update whose receive time is at or after D, priced from the book as it stands then. That follows the brief's "first book snapshot at or after t plus latency" literally. A flip is one order with a closing leg and an opening leg, filled in that order against the same book. If the feed leaves Healthy before such an update arrives, the order is rejected and logged. Stops retry at the next healthy minute through the normal stop check, death closes retry through the dying list, and a rejected flatten is sent again at the next healthy minute whatever that gap. A bot that dies while an open is still in flight joins the dying list too, and is closed if the open fills. Opening legs are sized as `notional / reference mid`, rounded to 8 decimals. Closing legs always use the held quantity. Paper orders never consume liquidity, so twenty bots buying in the same second all see the same levels, and `docs/honesty_and_limits.md` will say this plainly.

### 6.5 Persistence points

Equity snapshots are taken every `equity_snapshot_every_seconds` of engine time, at minute boundaries. Every `checkpoint_every_seconds` of engine time the engine schedules a persistence point 30 seconds after a boundary, well away from the boundary and from fills, which all fall due 1.0 to 1.3 seconds after it. At that point the engine itself writes the buffered rows and a checkpoint, at the same engine time live and in replay. At the same 30 second offset in every other minute it commits only the new feed events. Checkpoint and snapshot timestamps are engine time, so two replays write identical rows.

## 7. Feed

**Adapter.** `feed/coinbase.py` opens one connection to `wss://advanced-trade-ws.coinbase.com` with a 16 MiB message size limit and compression on. It sends three subscribe messages for `level2`, `market_trades` and `heartbeats`, and does not wait for the acknowledgements. It parses messages into the four event types from section 4. It checks that `sequence_num` rises by exactly one on the connection, and any jump is a gap. The `market_trades` snapshot of the last 100 trades, sent on every connect, only sets the trade identifier cursor. Those trades happened before the connection, so they are never counted in a candle. Trade batches are sorted by trade identifier, because they arrive newest first. A jump in trade identifiers while `sequence_num` is contiguous is logged as a trade gap, because that minute's volume may be undercounted, but it does not resync the book. Parsing is a pure function of raw text and receive time, so replay runs exactly this code.

The network side reconnects with exponential backoff, starting at `reconnect_backoff_start_seconds`, doubling to `reconnect_backoff_max_seconds`, with jitter between half and the full delay drawn from a generator seeded from `master_seed`, so a reconnect is never instant. The backoff resets only when the feed reaches Healthy. A snapshot that arrives crossed, or any failure before Healthy, counts as a failed attempt and lets the backoff keep growing. The websockets library's built in backoff is not used, because it resets after any successful handshake.

**Book.** `feed/book.py` keeps a dict from price to size for each side, plus a sorted list of prices per side kept with `bisect`. Measured on the development Mac, a level change costs 0.2 to 0.6 microseconds, reading the top 20 costs 1.6 microseconds, and the crossed check costs almost nothing. Scanning the dict instead would cost 200 microseconds per check. Prices parse from strings to floats, and two strings for the same price give the same float even when trailing zeros differ. Deleting a level that is not present is ignored. A crossed book, where the best bid is at or above the best ask, invalidates the book and requests a resync. The book exposes best bid, best ask, mid, spread in bps, the top 20 levels, and `walk(side, qty)`. The walk returns the average fill price, the levels consumed, and the quantity left unfilled within the top 20. A Coinbase auction can legitimately cross the book, and the bots then stay paused until it clears.

**Candles.** `feed/candles.py` builds open, high, low, close, volume and trade count for each UTC minute by receive time. It records the book mid at the boundary. A minute with no trades copies the previous close into all four prices and has zero volume and zero trades. The shared minute series holds about 400 minutes of mid, volume and validity in numpy ring arrays. On startup it reloads from the `candles` table, which is committed in the same transaction as each checkpoint, so it is the single source for the series.

**Health.** `feed/health.py` has three states, Healthy, Recovering and Stale, and starts in Stale. The feed becomes **Stale** whenever time since the last message, or since the last good book, exceeds `stale_after_seconds`. Any gap, crossed book or disconnect throws the book away and reconnects, which puts the feed in **Recovering**. Clock skew also moves the feed to Recovering, but keeps the book and the connection. Leaving Stale also passes through Recovering. Recovering becomes **Healthy** at the first minute boundary that closes a full minute in which the book was valid and fresh and the clock agreed with the exchange. After a reconnect that means a full minute after the new snapshot, with no relapse. That full fresh minute is processed as the first healthy minute, so a resync usually costs the one minute in which it happened, or two if the snapshot arrives late. Every exit from Healthy increments a feed epoch, which labels use to refuse to cross a gap. Every state change, gap, resync, reconnect, clock skew and resume is written to `feed_events`.

**Clock skew.** Skew is measured from the envelope `timestamp` of each live message, never from trade times, since the trade snapshot holds trades that are minutes old. The guard trips when the median difference over the last 50 messages exceeds `max_clock_skew_seconds`.

**Recorder.** With `record_raw_messages = true`, `recorder.py` writes one JSON object per line to `data/raw/YYYY-MM-DD_HHMMSS.jsonl.gz`, opening a new file at every start so a power cut can only truncate the tail of one file. Each object holds the receive time, a kind (open, message or close) and the raw text. Files rotate at UTC midnight. The reader tolerates a truncated final gzip member left by a power cut.

## 8. Learning

**Features.** `features.py` computes the 11 numbers once per minute from the shared minute series and the book at the boundary.

* Returns over 5, 15, 30 and 60 minutes, as `1e4 * ln(mid_t / mid_{t-k})`.
* Realized volatility over 30 and 120 minutes, as the standard deviation of one minute log returns in bps.
* The volume ratio, `ln((V15 + eps) / (V360 / 24 + eps))`, clipped to plus or minus 3, a fixed documented constant.
* Imbalance over the top 10 levels.
* Spread in bps.
* Sine and cosine of the UTC minute of the day.

A feature vector is valid only once the series holds `history_minutes_required` minutes of healthy data, as the brief requires. After a later gap a coverage rule applies instead of another six hour wait, governed by the new `min_window_coverage = 0.8`. A return needs a valid mid at exactly its start minute. Volatility uses only one minute returns between consecutive valid minutes, and needs at least 80 percent of the window's returns. The volume terms use the mean volume per valid minute times 15, and need at least 80 percent of the window valid. Without this rule every short reconnect would freeze all bots for six hours. After an outage longer than about 72 minutes, the volume baseline stays invalid for up to about 5 hours. All of this uses only data up to t, and the truncation test in section 17 checks that, including gaps inside windows.

On Coinbase the spread is usually one tick, so the spread feature mostly tracks slow drift in the price level. `docs/learning.md` will say so. The brief defines the feature, so it stays.

**Standardizer.** Each feature has an exponentially weighted mean and variance with the half life from `standardizer_half_life_hours`. The weight of each new sample is the larger of the exponential weight and 1 divided by the sample count. The statistics are therefore a plain running mean and variance until about 14,500 samples, roughly ten days, and only then become exponentially weighted. Without this the first weeks would be badly biased. The variance update is the non negative weighted form, and the standard deviation is floored at a tiny constant, so a constant feature gives a z score of zero. A feature outputs zero until it has `min_samples` samples, then a z score clipped to plus or minus `clip_z`. The vector is standardized with the statistics from before the current sample, so a sample never shapes its own z score. Any value that is not finite makes that minute's vector invalid.

**Labels.** `labels.py` keeps three things. One is a pending queue of records holding minute, epoch, z vector and entry mid. Another is a shared matured window of the last `replay_window_minutes` matured samples. The third is the label scale, which uses the same weighting, floor and half life as the standardizer. A record matures at t plus the horizon only if the epoch is unchanged. Its label is `1e4 * ln(mid_{t+h} / mid_t)`. Features and labels are the same for every bot, so they are stored once. Each bot stores only its own prediction for each pending minute. Training starts once the label scale has `min_samples` labels.

**Model.** `model.py` holds 209 float64 parameters in one flat array, with views for W1, b1, W2 and b2. Weights use Xavier uniform initialization from the bot's seed, biases start at zero, and the output weights are multiplied by `init_output_scale`. The output targets label divided by label scale. A bot takes a training step only in a minute in which a new sample matured. The batch is that newest sample plus `batch_size - 1` draws with replacement from the matured window, using the bot's generator. A batch size of 1 turns replay off. The loss is Huber with `huber_delta`. The gradient norm is clipped to `grad_clip`. The update is AdamW with the bot's learning rate gene `lr`, so each step is `w -= lr * (m_hat / (sqrt(v_hat) + 1e-8)) + lr * weight_decay * w` for weight matrices and the same without the decay term for biases. At the default values this decay shrinks unused weights by only about 0.01 percent a week, so it is a weak regulariser. A finite difference gradient check is a test.

**Learning rate.** The default is 0.0001 rather than the brief's 0.001. `scripts/learner_check.py` plants a signal of known strength in synthetic features with overlapping 30 minute labels and runs the bots' own learning code minute by minute. Over 28 synthetic days with 0.001, each bot's predictions had a spread of about 9 bps when there was nothing to predict, and their squared error was about 16 percent worse than predicting zero. With 0.0001 the spread fell to about 3 bps and the squared error came within about 1 percent of predicting zero. A population of three recovered about two thirds of a planted correlation of 0.1 and about half of one of 0.05. The costs are two. Adaptation is slower, so a signal that flips sign every week is followed about as well as before, but one that flips every two days is mostly missed. And predictions stay a few bps, so even run B's required edge of about 20 bps will almost never be met. `python scripts/learner_check.py --bots 3`, with `--lr 0.001` or `--flip-days 2` for the comparisons, reproduces these numbers. The value was fixed on synthetic data only, before launch, and never on market data. The gene bounds move down by the same factor, to 0.00001 and 0.001, so mutation can still move both ways.

After every gap no sample matures for one horizon, so no training happens in that time and warmup takes longer. That fits the reason warmup counts training minutes.

**Skill.** Before a matured sample trains anything, `skill.py` records the bot's original prediction in bps against the realized label in bps. Each bot keeps ring buffers of the last `skill_window_samples = 360` pairs. From them it reports correlation, directional accuracy and mean absolute error, which are logged in equity snapshots. A bot born after a record was made has no prediction for it and skips that pair. With `gate_on_skill = true`, the gate also requires a full window and a correlation above `skill_gate_min_corr`. The default leaves it off. `docs/learning.md` explains that consecutive labels overlap almost completely. Over 360 samples at a 30 minute horizon there are only about 12 independent outcomes, so these statistics are noisy by nature.

## 9. Bots and trading logic

**States.** A bot starts in **Warmup**, where it predicts and trains every healthy minute but cannot open positions. Warmup ends after `warmup_hours_random` hours, or `warmup_hours_inherited` for a mutated child, counted in minutes in which the bot actually took a training step. That way an outage cannot end a warmup in which nothing was learned. The bot is then **Active** until it becomes **Dead**. Rent accrues from birth in every healthy minute, including Warmup and the hours before features become valid. Parent age counts healthy minutes since birth. Lifespan in the `deaths` table and the age column in `status` use wall clock time from birth.

**Per bot minute step.** `bot.py` holds the whole per bot part of section 6.3, from rent to decision, as one method that takes the shared minute context. That keeps Warmup, Dead, stop and forced timer behaviour testable in Phase 5. The engine only adds the shared stale gate and the orchestration.

**Schedule.** Bots live in numbered slots from 0 to size minus 1. A replacement takes the slot of the bot that died. A bot decides at minute boundaries where the UTC minute index modulo the horizon equals its slot modulo the horizon.

**Cost estimate.** At a decision the round trip cost in bps is two times the taker fee, plus the spread, plus two times the impact of walking the book for the notional beyond the best price, plus two times `extra_slippage_bps`. If the prediction points short, it also adds `short_open_fee_bps` and `short_funding_bps_per_hour` times the horizon in hours. The required edge is that cost plus the margin gene, which is clamped to its bounds.

**Decision rule.** `bot.decide()` is a pure function, so every branch is tested directly. The predicted move is the model output times the label scale.

| Signal | Position | Action |
| --- | --- | --- |
| passes, same side | long or short | hold, pay nothing |
| passes, opposite side | long or short | flip, both legs pay |
| passes | flat | open |
| fails | long or short | close |
| fails | flat | none |

A negative prediction with shorts disabled counts as a failing signal. Every decision, including none and hold, writes a `decisions` row. The row holds the prediction, required edge, cost estimate, margin gene, the real gate result, the forced flag, the position before the decision, the reference mid and a reason.

**Forced exploration.** The timer counts healthy minutes since birth or since the last open, flips included. At birth it starts below zero, at minus a random delay of up to `force_trade_after_hours` drawn from the bot's own generator, so newborn bots do not all reach their first forced trade together. When it reaches `force_trade_after_hours` and the bot is flat, its next decision opens in the direction of its current prediction even if the gate failed, and the row is marked forced with the real gate result kept. A bot that holds a position follows the normal rule, because it is not inactive. If shorts are disabled and the prediction is negative, the bot stays flat and the timer keeps running. A prediction of exactly zero does nothing.

**Stop loss.** Every healthy minute, an open position is marked by walking the exit side of the book for its quantity. If the adverse move from the entry fill is at least `stop_loss_pct`, the bot sends a market close through the broker, the trade is marked as a stop, and the bot does not decide again until its next phase minute.

**Never trade when stale, in Warmup or Dead.** This is enforced in code, not left to convention.

## 10. Execution and accounting

**Broker.** `broker.py` sizes opening legs from the notional and closing legs from the held quantity, as in section 6.4. A buy walks the asks and a sell walks the bids. If the top 20 levels cannot fill the order, it is rejected and logged. The average book price then moves against the bot by `extra_slippage_bps`. The fee is `taker_fee_bps / 1e4` times the filled notional, using the active profile, and it comes out of cash at fill time. Opening a short also charges `short_open_fee_bps` of its notional, booked as funding. Every fill row records the reference mid, the average book price, the final fill price, the slippage in bps against the reference mid, the fee, the latency used and the levels consumed. `broker.py` contains no exchange client of any kind.

**Account.** Cash here is settled cash, meaning start equity plus realized gross profit, minus costs. Opening a position moves only the fee and, for a short, the opening charge. The account tracks cash, side, quantity, entry price, realized gross profit, fees, funding, rent, stop loss losses, other trading losses, writeoff, trades and wins. A win is a closed trade whose net profit is above zero. Cash is kept as its own ledger, updated at every event, and is not derived from the totals, so the checks below are real checks. They are checked at every equity snapshot and after every step of the randomized tests, within an absolute tolerance of one millionth of a dollar.

```
cash   == start_equity + realized_gross_pnl - fees - funding - rent + writeoff
equity == cash + unrealized_pnl      (marked to the exit side of the book)
start_equity - equity == fees + funding + rent + stop_loss_losses + other_trading_losses - writeoff - unrealized_pnl
```

Here `stop_loss_losses` is the negated gross profit of trades closed by a stop, and `other_trading_losses` is the negated gross profit of every other closed trade. The last line therefore splits a bot's losses into five parts that add up. A live violation is logged at ERROR, written as an `invariant_violation` feed event and shown by `status` and the soak report. The run continues so the evidence is kept.

**Funding.** Funding accrues on the entry notional for the time elapsed since the last accrual. It is added at each healthy minute and at close. A short therefore also pays for stale minutes and for downtime before a flatten, because in the real world the short would still be open. Rent does not accrue during those times, as the brief requires. Charging funding only on healthy minutes would be kinder than reality.

**Death.** A bot dies when equity falls below the cost of one round trip at the current estimate, for the side it holds or for a long if it is flat, plus one day of rent. It leaves its slot at once and joins the dying list, and its replacement takes the slot in the same minute. If the bot holds a position, a closing order goes through the broker and is resent each healthy minute until it fills. When it fills, costs apply, any negative cash becomes a recorded writeoff with cash clamped to zero, the bot is marked Dead, and the death row is written. A bot with no position finishes at once. The cause is whichever of fees, funding, rent, stop loss losses and other trading losses is largest. All five are stored, together with the threshold and the final equity. The dying list is part of the checkpoint.

## 11. Evolution

`population.py` keeps every slot filled. When a bot dies it draws, with the population generator, whether the replacement is a newcomer, with probability `newcomer_fraction`. Otherwise it picks a parent by tournament among eligible bots. A bot is eligible if it is alive, has at least `parent_min_age_hours` of healthy minutes since birth, and has at least `parent_min_trades` closed trades. If no bot is eligible, the replacement is a newcomer.

**Fitness.** Each bot records its equity every 60 healthy minutes. Over the last `fitness_window_hours` samples, or the bot's whole life if shorter, let `m` be the mean hourly change and `s` the standard deviation of hourly changes floored at the new `fitness_std_floor_usd`. Under the approved row in section 1, fitness is `m / s` when `m` is zero or positive and `m * s / floor` when it is negative, so volatility always counts against a bot. This follows the modified Sharpe ratio idea of Israelsen. Under the brief's literal formula, fitness would be the net change divided by `s`.

**Tournament.** The population generator samples the smaller of `tournament_size` and the number of eligible bots, without replacement, from the eligible bots in slot order, and the fittest wins. Ties break on rolling skill correlation. Fitness is a continuous number, so exact ties will be rare and the tie breaker will seldom matter. A single eligible bot is the parent, as the brief implies.

**Mutation.** The child's own seeded generator adds Gaussian noise with `mutation_sigma_weights` to every parameter. It then multiplies the margin and learning rate genes by `exp(N(0, mutation_sigma_genes))` and clamps both to their bounds. Adam state starts fresh. The child gets a new bot id, a fresh 10,000 dollar account, the parent's generation plus one, and inherited warmup. A `lineage` row stores the parent, generation, birth time, both sigmas, both gene factors, the parent's genes, the parent's fitness and the parent's skill correlation. A newcomer gets new random weights, the initial gene values and generation 0.

Coin flip twins from section 21 are never parents and never count toward fitness. Bots leave only by dying. `docs/evolution.md` will list possible later additions, such as culling, crossover, a hall of fame, a tiered fee profile and the road map in section 21, without building them.

## 12. Storage schema

`store.py` opens SQLite with `journal_mode=WAL`, `synchronous=NORMAL` and `journal_size_limit`. One connection on the event loop thread does all writes, at the persistence points of section 6.5. Every write is timed on the monotonic clock, and slow ones are logged. `feed_events` are also committed once a minute at the same 30 second offset, live and in replay, and live also commits `runtime_status` once a minute. Neither feeds back into bot state.

```sql
CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL          -- schema_version, master_seed, launch_ts, launch_mid, launch_ask, config_toml
);
CREATE TABLE bots (
  bot_id INTEGER PRIMARY KEY,
  slot INTEGER NOT NULL,
  parent_id INTEGER,
  generation INTEGER NOT NULL,
  born_at TEXT NOT NULL,
  died_at TEXT,
  state TEXT NOT NULL,         -- warmup, active, dying, dead
  margin_bps REAL NOT NULL,
  lr REAL NOT NULL,
  seed INTEGER NOT NULL,
  origin TEXT NOT NULL         -- random, mutated, control
);
CREATE TABLE lineage (
  bot_id INTEGER PRIMARY KEY,
  parent_id INTEGER NOT NULL,
  generation INTEGER NOT NULL,
  born_at TEXT NOT NULL,
  sigma_weights REAL NOT NULL,
  sigma_genes REAL NOT NULL,
  margin_factor REAL NOT NULL,
  lr_factor REAL NOT NULL,
  parent_margin_bps REAL NOT NULL,
  parent_lr REAL NOT NULL,
  parent_fitness REAL,
  parent_skill_corr REAL
);
CREATE TABLE decisions (
  id INTEGER PRIMARY KEY,
  bot_id INTEGER NOT NULL,
  ts TEXT NOT NULL,
  position_before TEXT NOT NULL,   -- long, short, flat
  ref_mid REAL NOT NULL,
  prediction_bps REAL NOT NULL,
  required_edge_bps REAL NOT NULL,
  cost_estimate_bps REAL NOT NULL,
  margin_bps REAL NOT NULL,
  gate_passed INTEGER NOT NULL,
  forced INTEGER NOT NULL,
  action TEXT NOT NULL,            -- open_long, open_short, hold, close, flip, none
  reason TEXT NOT NULL,
  label_bps REAL                   -- filled when the label for this minute matures, for the success criteria
);
CREATE INDEX decisions_bot_ts ON decisions (bot_id, ts);
CREATE TABLE trades (
  id INTEGER PRIMARY KEY,
  bot_id INTEGER NOT NULL,
  side TEXT NOT NULL,              -- long, short
  opened_at TEXT NOT NULL,
  closed_at TEXT,
  qty REAL NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL,
  fee_open REAL NOT NULL,
  fee_close REAL,
  funding REAL NOT NULL DEFAULT 0,
  slippage_bps_open REAL NOT NULL,
  slippage_bps_close REAL,
  pnl_gross REAL,
  pnl_net REAL,
  forced INTEGER NOT NULL DEFAULT 0,
  exit_reason TEXT                 -- signal, flip, stop, death, restart_flatten, gap_flatten, copy_exit
);
CREATE INDEX trades_bot ON trades (bot_id);
CREATE TABLE fills (
  id INTEGER PRIMARY KEY,
  bot_id INTEGER NOT NULL,
  trade_id INTEGER,
  decided_at TEXT NOT NULL,
  filled_at TEXT,
  leg TEXT NOT NULL,               -- open, close
  side TEXT NOT NULL,              -- buy, sell
  qty REAL NOT NULL,
  ref_mid REAL NOT NULL,
  book_avg_price REAL,
  price REAL,
  levels_consumed INTEGER,
  slippage_bps REAL,
  fee REAL,
  latency_ms INTEGER NOT NULL,
  status TEXT NOT NULL,            -- filled, rejected
  reason TEXT NOT NULL             -- decision, stop, death, restart_flatten, gap_flatten, or the rejection cause
);
CREATE INDEX fills_bot ON fills (bot_id);
CREATE TABLE equity_snapshots (
  bot_id INTEGER NOT NULL,
  ts TEXT NOT NULL,
  cash REAL NOT NULL,
  equity REAL NOT NULL,
  unrealized REAL NOT NULL,
  realized_gross REAL NOT NULL,
  fees_total REAL NOT NULL,
  funding_total REAL NOT NULL,
  rent_total REAL NOT NULL,
  stop_loss_total REAL NOT NULL,
  position TEXT NOT NULL,
  skill_corr REAL,
  skill_dir_acc REAL,
  skill_mae REAL,
  skill_n INTEGER NOT NULL,
  PRIMARY KEY (bot_id, ts)
);
CREATE TABLE deaths (
  bot_id INTEGER PRIMARY KEY,
  died_at TEXT NOT NULL,
  generation INTEGER NOT NULL,
  cause TEXT NOT NULL,             -- fees, funding, rent, stop_losses, trading_losses
  fees_total REAL NOT NULL,
  funding_total REAL NOT NULL,
  rent_total REAL NOT NULL,
  stop_loss_total REAL NOT NULL,
  other_losses_total REAL NOT NULL,
  writeoff REAL NOT NULL,
  threshold REAL NOT NULL,
  final_equity REAL NOT NULL,
  trades INTEGER NOT NULL,
  lifespan_hours REAL NOT NULL
);
CREATE TABLE feed_events (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,   -- connected, disconnected, stale, recovering, healthy, snapshot, gap, trade_gap, crossed, reconnect, clock_skew, resume, invariant_violation
  detail TEXT           -- for snapshot events, the size in bytes and the level counts
);
CREATE TABLE checkpoints (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  schema_version INTEGER NOT NULL,
  meta TEXT NOT NULL,              -- JSON scalars, generator states, accounts, counters
  arrays BLOB NOT NULL             -- zlib compressed float64 arrays, laid out by the meta JSON
);
CREATE TABLE candles (
  ts TEXT PRIMARY KEY,
  open REAL, high REAL, low REAL, close REAL,
  volume REAL NOT NULL,
  trade_count INTEGER NOT NULL,
  mid REAL,
  valid INTEGER NOT NULL
);
CREATE TABLE benchmark (
  ts TEXT PRIMARY KEY,
  btc_mid REAL NOT NULL,
  buy_hold_equity REAL NOT NULL,   -- all of start equity bought at launch, one taker fee, marked at the bid
  matched_equity REAL NOT NULL,    -- notional bought at launch, rest held as cash
  living_mean_equity REAL NOT NULL,
  living_median_equity REAL NOT NULL,
  living INTEGER NOT NULL,
  twin_mean_equity REAL,           -- coin flip twins, when enabled
  cash_equity REAL NOT NULL        -- start equity minus rent, the bot that never trades
);
CREATE TABLE forecasts (
  ts TEXT NOT NULL,                -- the minute the forecast is for
  predictor TEXT NOT NULL,         -- population, linear, or a later predictor
  prediction_bps REAL NOT NULL,
  label_bps REAL,                  -- filled when the label matures
  PRIMARY KEY (ts, predictor)
);
CREATE TABLE runtime_status (      -- live only, never written by replay
  id INTEGER PRIMARY KEY CHECK (id = 1),
  ts TEXT NOT NULL,
  feed_state TEXT NOT NULL,
  best_bid REAL, best_ask REAL, mid REAL,
  living INTEGER NOT NULL,
  last_checkpoint_ts TEXT,
  rss_mb REAL,                     -- current resident memory
  rss_peak_mb REAL
);
```

All timestamps are ISO 8601 UTC strings with millisecond precision and a trailing `Z`, so text order is time order. Candles older than `keep_candles_days` are pruned in the checkpoint transaction. Only the last `keep_checkpoints` checkpoints are kept. Decisions, trades, fills, deaths, lineage, forecasts and bots are kept forever. At 20 learners, their 20 twins and a 30 minute horizon, decisions grow by about 1,000 rows a day, forecasts by about 2,900 and equity snapshots by about 11,500, roughly 700 MB a year in total.

## 13. Checkpoints, resume and shutdown

**What a checkpoint holds.** For every bot it holds parameters, Adam moments and step count, genes, account, open position and pending order, pending predictions, skill buffers, hourly equity samples, counters, the slot, and the bot's generator state. It also holds the dying list. Shared state includes the standardizer, the label scale, the matured window, the health epoch, the last healthy minute, id counters, the population generator state, the benchmark anchor and the linear baseline's statistics. Scalars and generator states go into JSON, and arrays go into one zlib compressed blob of raw float64 values whose layout is listed in the JSON. An `npz` file was dropped because zip archives embed the current time, which would make two identical replays write different bytes. The pending label queue is not stored, because every restart is a gap that discards it. There is no pickle. The benchmark measured JSON with zlib at 1.2 milliseconds per bot, which would stall the Pi's event loop for about 0.3 seconds at 20 bots. Raw arrays take under a millisecond in total. Before a checkpoint is committed, it is decoded once in memory to prove it reads back.

**Atomicity.** The buffered rows and the checkpoint are written in one transaction. After a power cut every bot and ledger row therefore matches a checkpoint. Only `feed_events` and `runtime_status` can be newer, because they are committed each minute, and they never feed back into bot state. Up to `checkpoint_every_seconds` of history can be lost, but the books and the bot state always agree. With WAL and `synchronous=NORMAL`, SQLite guarantees consistency but not durability of the very last commits. That is the trade the brief asks for on an SD card.

**Resume.** On start the engine loads the newest checkpoint that decodes. If it ever has to fall back to an older one, it first deletes or reopens every bot and ledger row written after that checkpoint, in one transaction, clears labels that matured after it, keeps `feed_events`, records why in a `resume` event and commits a fresh checkpoint at once. It refuses to resume when the master seed, the population size or the twin setting differs from the run in the database. If no checkpoint decodes but the database already has bots, it stops with a clear error rather than seeding a new population over old history. With no checkpoint at all, it creates a fresh population from the master seed. Every restart is a feed gap, so the epoch increments and every pending label is discarded. Restored pending orders are rejected with reason restart, while stops and death closes retry. Rent is never charged for downtime, because rent accrues only on healthy minutes. The gap is measured at the first healthy minute after the restart, from the last healthy minute stored in the checkpoint. If it is longer than `max_resume_gap_minutes`, step 4 of the minute step flattens every open position with exit reason `restart_flatten`. The same rule covers a long outage while the process stays up, with exit reason `gap_flatten`, so the same market risk is always treated the same way.

**Shutdown.** SIGINT and SIGTERM set a flag. The runner lets the current minute step finish, waits at most `latency_ms + latency_jitter_ms` for pending fills to settle while the feed is Healthy, writes a checkpoint with all buffered rows, closes the database and exits with code 0. The systemd unit allows 60 seconds for this.

## 14. Runtime and command line

`runner.py` runs one asyncio program with three tasks. The **feed task** reads the socket and hands each message to the engine. The **ticker task** wakes every second and calls `engine.advance_to`, which runs any minute boundary or persistence point that is due. Once a minute it also writes `runtime_status` and logs one heartbeat line, including during stale periods, with feed state, population size, equity mean, minimum and maximum, deaths so far, current and peak memory, and message lag, measured on the monotonic clock as the worst lateness of the one second tick. The **shutdown task** waits for a signal, as described in section 13. Current memory comes from `/proc/self/statm` on Linux. macOS only exposes peak memory to the process, so the soak script samples current memory from outside. The minute step is synchronous, so two steps can never overlap. Its duration is measured on the monotonic clock, and anything over 10 seconds is logged as slow.

Command line, through the `btc_survivor` entry point or `python -m btc_survivor`.

* `run` starts the live system.
* `status` prints one row per living learner, and with `--twins` the coin flip twins too. With `--watch` it redraws every minute. Columns are id, generation, age, state, equity, trades, win rate, fees, rent, rolling skill correlation, position, burn rate per day and days left at that rate. It then prints the live BTC price and its age from `runtime_status`, both buy and hold curves since launch, and a one line summary. The summary gives the living mean and median, the number of deaths so far, and an all bots figure that adds living equity to the final equity of every dead bot and compares it with the capital given to every bot ever born. That keeps the losses of dead bots visible after their replacements arrive. It warns when `runtime_status` is more than two minutes old, meaning the system is not running, or when the feed is not Healthy, and says that bot rows can be up to `checkpoint_every_seconds` old.
* `explain BOT_ID` prints the life story. That covers birth, origin, parent and generation, genes, key statistics, the five part cost breakdown with a check that it sums, the five worst trades, decision counts by action including forced ones, and for a dead bot the cause of death with the numbers behind it.
* `lineage BOT_ID` prints the ancestry chain back to a random ancestor, with each ancestor's lifespan, final or current equity and cause of death.
* `export --out DIR` writes CSV files of bots, trades, fills, decisions, equity snapshots, deaths, lineage, the benchmark and the forecasts.
* `evaluate` prints the success criteria and the outcomes fixed by arithmetic from section 21 for this run, computed from the database. `--final-day` names the day that counts, 28 for runs A and B and 56 for run C, and every result before it is marked descriptive.
* `dashboard [CONFIG ...] [--port 8050]` serves a read only page on 127.0.0.1 for one or more runs. It refreshes every 10 seconds, pauses while its tab is hidden, and reloads itself when a new version of the page is installed. It shows the feed, the launch stage, the timing (one minute candles, the 30 minute horizon and decision rhythm, the forced trade interval and how many minutes each chart point covers), every bot with its warmup progress, position and next forced trade, the equity curve against the benchmarks, the skill so far on the same rows as criterion 1, and recent decisions, fills and feed events. A panel explains how to read the page, every card and column has a hover hint, a guide under the equity chart says what each line is and which comparison matters, and moving the pointer over the chart shows the exact values at that minute. It opens each database read only and needs nothing beyond the project's own dependencies.
* `init-db` creates an empty database, or checks an existing one against the expected schema version and tables.
* `replay --out DIR FILE [FILE ...]` runs recordings through the pipeline into a new or empty folder given by `--out`, never into `data_dir`. Its help text states that replay results are never used to choose, tune or rank strategies.

## 15. Replay harness and determinism

`replay.py` reads recordings in order. It feeds each raw message through the same adapter parsing code and then into `engine.on_event`. Recorded connection open and close markers reset the adapter exactly as a live reconnect does. The replay clock is the recorded receive time, so health, staleness, latency fills, minute boundaries and persistence points all behave as they did live, only faster. Replay always starts from a fresh population and never loads a checkpoint. It writes only to the folder given by `--out`, and refuses one that is not empty. The soak script passes its own folder the same way. It never writes `runtime_status`.

Every random draw comes from generators seeded through `numpy.random.SeedSequence` from `master_seed` and fixed tags. Each bot's generator uses the master seed and the bot id. The population generator, the reconnect jitter and the synthetic data generator have their own tags. Given the same seed and the same files, two replays produce identical database contents. A test compares a digest of every table.

The sentence "Replay results are never used to choose, tune, or rank strategies" appears in `replay.py`, in the command line help and in `docs/learning.md`.

## 16. Configuration

`config.py` loads the TOML file with `tomllib` into frozen dataclasses. It rejects unknown sections and keys, wrong types and out of range values, with a message that names the key, the bad value and the allowed range. It also checks relations between keys. There must be exactly 4 return windows and 2 volatility windows, so the model has 11 inputs. `imbalance_levels` must not exceed `book_depth_levels`. `history_minutes_required` must cover the longest window and the volume baseline. `latency_ms + latency_jitter_ms` must stay well under one minute. `fee_profile` must name a profile that has a fee key. Placeholders are flagged in comments and in `docs/execution_model.md`, with source and date. This is the proposed `config.example.toml`. Values that differ from the brief, and new keys, are marked.

```toml
[run]
master_seed = 12345
data_dir = "data"
log_level = "INFO"

[feed]
exchange = "coinbase"
product = "BTC-USD"
book_depth_levels = 20
stale_after_seconds = 5
reconnect_backoff_start_seconds = 1
reconnect_backoff_max_seconds = 60
record_raw_messages = false
max_clock_skew_seconds = 2.0          # new

[time]
horizon_minutes = 30                  # brief 15, recommended 30
history_minutes_required = 360
warmup_hours_random = 6
warmup_hours_inherited = 1

[features]
return_windows_minutes = [5, 15, 30, 60]
vol_windows_minutes = [30, 120]
volume_baseline_minutes = 360
imbalance_levels = 10
standardizer_half_life_hours = 168    # also used for the label scale
clip_z = 5.0
min_samples = 30                      # new, was a constant in the brief
min_window_coverage = 0.8             # new

[model]
hidden_units = 16
init_output_scale = 0.1
batch_size = 8
replay_window_minutes = 240
huber_delta = 1.0
grad_clip = 5.0
adam_beta1 = 0.9
adam_beta2 = 0.999
gate_on_skill = false
skill_window_samples = 360            # new, was a constant in the brief
skill_gate_min_corr = 0.0             # new, used only when gate_on_skill is true

[trading]
start_equity = 10000.0
notional = 500.0
stop_loss_pct = 1.0
allow_short = true
force_trade_after_hours = 12

[costs]
fee_profile = "retail_default"        # retail_default, active_retail, low_fee_tier
taker_fee_bps_retail_default = 90.0   # brief 30, Coinbase US Intro since 16 Sep 2026
taker_fee_bps_active_retail = 30.0    # new, Coinbase US Advanced 2, 50k USD per 30 days
taker_fee_bps_low_fee_tier = 5.0      # Coinbase US VIP3, 20m USD per 30 days
latency_ms = 1000
latency_jitter_ms = 300               # uniform 0 to 300 added to latency_ms
extra_slippage_bps = 1.0
short_funding_bps_per_hour = 0.5      # top of Kraken BTC margin rollover range
short_open_fee_bps = 2.0              # new, top of Kraken BTC margin opening fee range
daily_running_cost = 1.16             # brief 5.0, 4.23 percent T bill yield on 10,000

[genes]
margin_bps_init = 8.0
margin_bps_min = 2.0
margin_bps_max = 60.0
lr_init = 0.0001                      # brief 0.001, see section 8
lr_min = 0.00001                      # brief 0.0001
lr_max = 0.001                        # brief 0.01
weight_decay = 0.0001

[population]
size = 20
newcomer_fraction = 0.10
parent_min_age_hours = 24
parent_min_trades = 20
mutation_sigma_weights = 0.05
mutation_sigma_genes = 0.20
fitness_window_hours = 168
fitness_std_floor_usd = 1.0           # new
tournament_size = 3                   # new

[controls]
coin_flip_twins = true                # new, one twin per learner slot

[persistence]
checkpoint_every_seconds = 300
equity_snapshot_every_seconds = 300
max_resume_gap_minutes = 10
keep_candles_days = 14
keep_checkpoints = 3                  # new
```

To return to the brief's original numbers, the edits are the horizon, the retail fee, the rent, the three learning rate values, and leaving out the short opening fee. The forced timer delay at birth is in code. Everything else is unchanged or additive.

## 17. Test plan

Tests run with `pytest`, need no network, and are deterministic. Randomized property tests use seeded numpy generators, with no extra libraries. Each module has its own test file, and integration tests drive the engine from synthetic recordings.

| Requirement from the brief | Test |
| --- | --- |
| Book walk averages across levels, rejects orders beyond visible depth, handles buys and sells | `test_book.py` with hand built books |
| Fee, slippage, latency and funding math | `test_broker.py` against hand computed examples, including a book update exactly at the due instant and one just after it, and the short opening charge |
| Balance invariant across thousands of random trade sequences | `test_account.py` runs 5,000 seeded sequences of opens, closes, flips, stops, rent, funding and deaths, and checks all three invariant lines after every step within the stated tolerance |
| Numerical gradient check | `test_model.py` compares the backward pass with central differences on random inputs, agreeing to one part in a million, plus one hand computed AdamW step |
| Learner reduces held out error on a known pattern, and not on noise beyond chance | `test_model.py` uses a synthetic series whose label depends on two features. Held out error must fall well below the predict zero baseline. On pure noise, held out correlation must stay inside three standard errors of zero |
| Labels never used before the horizon, never across a gap | `test_labels.py` asserts maturity time against creation time, and that an epoch change discards pending records |
| Features never use future data | `test_features.py` computes features on a full series and on the same series truncated at each t, including series with gaps inside windows, and requires identical vectors |
| Every branch of the decision rule, flips, holds, forced exploration, stops, warmup | `test_bot.py` tests the pure `decide()` table and the per bot minute step, including the forced timer from birth, forced exploration only when flat, a held position past the timer, a stop suppressing that minute's decision, Warmup rows, and an Active bot holding a position at a phase minute without valid features |
| Gene bounds after mutation | `test_genes.py` runs 100,000 mutations with an extreme sigma |
| Death rules, cause attribution, immediate replacement | `test_account.py` and `test_population.py` check the threshold, the writeoff clamp, the five parts summing, the slot refilled in the same minute, and a death close rejected once that settles at the next healthy minute |
| Stale data pauses trading, learning and rent, and recovery needs a fresh book plus a full minute | `test_health.py` and `test_engine.py` with synthetic silences, gaps, crossed books, a crossed snapshot and clock skew |
| Checkpoint round trip gives identical state, and a long gap flattens with no rent | `test_checkpoint.py` compares every array bit for bit and every generator state. `test_engine.py` checks short and long resume, a long live outage, flatten coinciding with a phase minute, and a stop triggering in the flatten minute without a second close |
| Same seed and same recording give identical results | `test_replay.py` replays twice and compares a digest of every table |
| No order placement code exists | `test_no_orders.py` scans `src/` for forbidden names and endpoints such as `place_order`, `cancel_order`, `create_order`, `/orders`, `api_secret`, `passphrase`, `CB-ACCESS-SIGN`, `hmac` and `jwt` |

Further tests cover the coin flip twins copying their learner's timing with a seeded coin and never becoming parents, the `evaluate` command's criteria on a seeded database, a scripted live clock that ticks just after each boundary and steps back below it, checking that stamps are clamped to engine time and that replaying them gives the same rows, config validation messages and the cross key checks, the store schema, batched transaction atomicity and rollback to an older checkpoint, adapter parsing of real message shapes (snapshot before the acknowledgement, newest first trade batches, the trade snapshot ignored for candles, unknown deletes, sequence jumps), candle alignment and empty minutes, recorder rotation and truncated files, the standardizer with a constant feature and a one tick jump, warmup weighting giving unit variance after 30 standard normal samples, skill statistics, fitness for bots of different ages with identical hourly behaviour, tournament sampling with fewer eligible bots than the tournament size, the benchmark arithmetic, signal handling that ends with a checkpoint and exit code 0, and every command line command against a seeded database. A last test checks the documentation rules in the README, CONTRIBUTING, the docs folder and this plan. Outside code blocks, inline code and links, it rejects colons, semicolons, em dashes, hyphenated words and emojis, and it requires every section of the README and the docs to end with at least one link.

## 18. Build phases

Each phase ends with the full suite green, one single line commit on `feat/initial-bot`, and a short phase summary. Before existing code changes, the modules and tests that depend on it are listed. Commits carry only the owner's name, and every commit message is a single line with no co author lines. Phase commits are pushed to the project's private GitHub repository on `feat/initial-bot`, and no `main` branch exists.

1. **Scaffolding.** First, a project repository. The folder sits inside a git repository rooted at the owner's home directory, so the project gets its own git repository, created with `git init -b feat/initial-bot` inside `btc_survivor` itself and verified with `git rev-parse --show-toplevel`, which must print this folder. The repository's author is set to Ayush Kansal with kansalayush2008@gmail.com, and only explicit paths are ever staged. The first commit holds `.gitignore` and `PLAN.md`. Then the repo files, MIT `LICENSE`, `CONTRIBUTING.md`, `pyproject.toml` with pins, config loading and validation, clock, types, logging, store and schema, and `init-db`, with tests for config and store. Python 3.11 is installed with pyenv, and the suite runs on 3.11 and 3.13.
2. **Money.** `account.py` and `broker.py`, with the book walk inside a minimal `feed/book.py`, fees, slippage, latency, funding, the short opening charge, the death threshold and cause, and the invariant tests.
3. **Data.** Feed interface, Coinbase adapter, the full book, candles, health, recorder and `synthetic.py`. Tests with synthetic and hand written messages covering gaps, crossed books, silences and skew. `scripts/record_feed.py` then runs in the background on the development Mac, keyless and public only, writing to `data/raw`. It gives real data for replay and soak, and it measures a weekday busy hour message rate and how often the feed leaves Healthy. Both go into the phase summary, and the recorder stops after Phase 10.
4. **Learning.** Features, standardizer, labels, model with the gradient check, AdamW, skill statistics, and the synthetic pattern and noise tests.
5. **Bot.** State machine, the per bot minute step, the decision rule, cost estimate and threshold, forced exploration, stop loss and warmup, with every branch tested at the bot level.
6. **Evolution and controls.** Genes, population, fitness, tournament, mutation, lineage and the coin flip twins, with tests.
7. **Runtime.** Engine, runner, benchmark with the cash baseline and twin averages, labels written back to decision rows, persistence points, checkpoints, resume, the dying list, signals and the replay harness. Tests for restart, short and long gaps, long live outages, and determinism. A short live smoke run on the development Mac checks the feed, health and persistence. Trading paths are exercised by replay, because the first live trade comes about 13 hours after launch. Suite on 3.11 and 3.13.

**Phase 7b, skill refocus.** The learning rate change, the forced timer delay, the `forecasts` table with the population and linear baseline rows, and `scripts/learner_check.py`, with tests. Added after critics showed that at retail fees the costs, not skill, decide survival.

8. **Command line and dashboard.** `status`, `explain`, `lineage`, `export`, `evaluate` and the read only `dashboard`, with the criterion 1 and 2 statistics of section 21 and the outcomes fixed by arithmetic, tested against a seeded database, plus `scripts/plot_results.py` and `requirements-plot.txt`.
9. **Docs and deploy.** Every document in the layout, each section ending with links, a systemd template unit that runs one service per config file, the configs for runs A, B and C, and the Raspberry Pi guide, plus the documentation test. The last step fills `docs/RESULTS_TEMPLATE.md` with a run header (start date, seed, fee profile, horizon, rent, hardware), the success criteria from section 21 with empty result rows for days 7, 14 and 28 and, for run C, day 56, and an empty table per generation with date, generation, born, alive, dead, mean equity, twin mean equity, top cause of death, mean skill correlation, and population against buy and hold and the cash baseline.
10. **Soak.** `scripts/soak_test.py` replays long recordings with the plumbing configuration from section 3. It uses real recordings joined end to end with reconnect markers, plus a synthetic one spanning several days with many full size snapshot resyncs. It reports current memory over time and passes only if memory at the end is no more than 5 MB above memory at the end of the first hour, CPU per message, invariant checks, mutated births and the highest generation reached. If Docker is available it also runs once in a Linux arm64 container. The live system also runs on the development Mac for as long as time allows, as runs A, B and C side by side, with two drills. One kills the process with `kill -9` and restarts it. The other pauses it with SIGSTOP for longer than `stale_after_seconds` to force the stale and reconnect path. A true network cut would affect the whole development Mac, so it needs the owner's approval first. The drills check the resume event, that no rent was charged for downtime, and that the invariants hold. The suite runs on 3.11 and 3.13. The final summary lists every done criterion from the brief, how it was checked, and which ones only the target Raspberry Pi can confirm, starting with the 24 hour run. On 12 October 2026 the owner kept the 150 MB budget and the resident memory rule for the Pi only. The Mac host, with 24 GB, has no tight budget, so a soak on the Mac is judged on memory in use, the measure a leak moves.

## 19. Packaging and platform

* `requires-python = ">=3.11"`. `numpy==2.4.6` on Python 3.11, because numpy 2.5 dropped 3.11, and `numpy==2.5.3` on 3.12 and newer, selected with environment markers. The benchmarks used 2.5.4, but it was released on the day of this plan, so the previous patch release is pinned and the Phase 10 soak measures with the pinned version. Also `websockets==17.2`, and `pytest==9.1.1` in a test extra. `matplotlib==3.10.7` goes only in `requirements-plot.txt`.
* The Pi target is Raspberry Pi OS Lite 64 bit, the current Trixie release with Python 3.13, in a virtual environment. The Bookworm based Legacy image with Python 3.11 is also supported.
* The unit is a template, `btc_survivor@.service`, so `btc_survivor@run_a`, `btc_survivor@run_b` and `btc_survivor@run_c` run `configs/run_a.toml`, `configs/run_b.toml` and `configs/run_c.toml` as separate services. Recommended settings are `Restart=always`, `RestartSec=5`, `KillSignal=SIGTERM`, `TimeoutStopSec=60`, `After=network-online.target time-sync.target`, `Wants=network-online.target`, and `Environment=PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1`. `WorkingDirectory` and `User` are placeholders. The guide explains enabling `systemd-time-wait-sync.service`. Without it, `time-sync.target` does not wait for a real clock.
* The Pi guide also covers SD card wear, booting from a USB SSD, how to check write durations on the SD card, the Pi kernel's default of no memory cgroup (which makes `MemoryMax` ineffective), checking the board's clock speed, and cooling. A throttled Pi 4 can run noticeably slower. It ends with a checklist for the 24 hour run on the target Raspberry Pi covering the same 5 MB memory bound, a pulled network cable, and a kill and restart.

## 20. Honest limits and risks

* **Paper is kinder than real trading.** Orders never move the book, there is no queue, the exchange never rejects an order, and fees never change mid run.
* **The short leg is a composite.** It pays the active profile's Coinbase spot taker fee plus Kraken style margin charges. Kraken says Coinbase US does not offer spot shorting, which could not be verified with Coinbase.
* **The low fee profile is out of reach for a US Coinbase retail account** through its own volume. A good result under it says the learning loop works, not that the strategy would work for a retail account. Outside the US, retail fees are lower, so the account's region matters.
* **Fixed fees are harsher than tiered reality** after the first weeks, because a real account would move down the tiers on this volume.
* **Keyless feeds may be dropped more often.** Coinbase recommends keys "for connection reliability". Reconnects and resyncs are expected, logged and tested, and each one discards up to a horizon of pending labels.
* **Snapshot memory on the Pi.** The 4.5 MB snapshot is the largest memory event. Linux memory behaviour after many resyncs was not measured. If the Pi soak shows peaks too close to 150 MB, the fixes are the Kraken adapter with its 2 KB snapshot or a streaming parse. Either change needs the owner's approval first.
* **CPU in busy markets.** The pessimistic estimate at an assumed 100 messages a second is about at the 25 percent target. Phase 3 measures a real busy hour.
* **Few deaths under the defaults.** Section 3 explains why. At 10,000 dollars of start equity evolution is unlikely to show anything live within a year, which is why run C starts at 500.
* **Three runs share one machine.** Each process stays within the per process targets, but together they use about three times the memory and feed CPU of one run.
* **All Pi performance numbers are estimates** from published benchmarks and measurements on the development Mac, until they run on the target Raspberry Pi.
* **Market statistics are from a short period.** The probes ran during a quiet weekend hour. Busy markets will show more messages, wider spreads and more impact. The cost model reads the live book every time, so it adapts.

## 21. Controls, parallel runs and success criteria

**Coin flip twins.** Every learner slot also holds a twin with its own account and no network. Whenever the learner's decision opens, closes or flips, the twin does the same at the same moment, choosing long or short with a fair coin from its own seeded generator. When the learner is closed out by its stop loss or by death, the twin closes at that moment too. A twin that still holds a position while its learner is flat with nothing pending closes it at the next healthy minute, and a twin holding a stale position closes it before copying the next open, so the pair never drifts out of step. A twin pays the same fees, slippage, latency, funding and rent, has its own stop loss and death check, and does nothing when its learner holds or does nothing. Twins are never parents and never count toward fitness, and a dead twin is replaced by a fresh twin in the same slot. Because a learner and its twin trade at the same moments, the only difference between them is the direction, so a gap between their results measures prediction skill rather than timing luck. Setting `coin_flip_twins = false` runs the population without them.

**The cash baseline.** The bot that never trades is not simulated. Its equity is start equity minus rent for the healthy minutes since launch, recorded in the benchmark table next to buy and hold.

**Forecast records.** Every healthy minute with a valid feature vector and a ready label scale writes one row per predictor to the `forecasts` table, and the row receives its label when that label matures. The predictors are the population, meaning the mean prediction of the living learners, and the linear baseline, an online ridge regression on the same z scores fitted to the same matured labels with the standardizer's half life, which shows whether 209 parameters add anything over a one line rule. Predicting zero needs no row. Any later predictor, for example one that reads the news, adds rows under its own name and is scored by exactly the same rules.

**Three runs.** Each run is a separate process with its own config file and data folder. All three use the same master seed, so they start from identical bots.

| Run | Fee profile | Start equity | What it shows |
| --- | --- | --- | --- |
| A main | `retail_default`, 90 bps | 10,000 | the skill result, and what retail costs do to a population that cannot pass the gate |
| B low fee | `low_fee_tier`, 5 bps | 10,000 | the same forced schedule at a 5 bps fee, with its survival and cost split |
| C machinery | `retail_default`, 90 bps | 500 | that death, replacement, mutation and lineage work under live conditions |

**Run D, added after launch.** Run D is run B with a 4 hour horizon. It was added on 11 October 2026, after runs A, B and C had started, to see trades that the gate lets through. At a 5 bps fee the required edge is about 20 bps, and over 4 hours a prediction has more room to clear it. Its settings were fixed before its own launch. The same synthetic check set its learning rate to 0.00003, with gene bounds 0.000003 and 0.0003, because at a 4 hour horizon 0.0001 made predictions about 9 percent worse than predicting zero and wide enough, at about 17 bps, to pass the gate on noise. At 0.00003 the spread with no signal was about 6 bps and the squared error within about 2 percent of predicting zero. A longer replay window did not help. Run D is exploratory and not part of the success criteria below. Each bot decides once every 4 hours, and all 20 decide within the first 20 minutes of each 4 hour block, so its trades cluster in time. With about 6 windows a day its skill test at day 28 can only detect a correlation of about 0.28, so its numbers are descriptive. Fees and rent cost a quiet run D bot about 2.36 dollars a day plus 4 hours of short funding per short, so no bot dies.

**Outcomes fixed by arithmetic.** These follow from fees, notional, start equity and the forced timer, so they are written down before launch and judged as matched or not matched. They check the accounting and are not findings about trading.

* Run A, every open is forced and no bot dies by day 28. Each learner pays 1.16 dollars of rent per healthy day and 9.10 dollars per round trip, two forced round trips a day once its first forced trade comes 13 to 25 hours after launch, about 19.36 dollars a day in all. The mean equity of the founding learners sits within about 20 dollars of 10,000 minus that rent and 9.10 dollars for each round trip they actually made, because market noise mostly cancels across 20 learners.
* Run B, almost every open is forced, because the required edge of about 20 bps is several times the size of the predictions, and no bot dies, because death is more than a year away even for a bot that trades at every decision.
* Run C, every bot dies of fees. The first wave of deaths falls between days 24 and 28 and the second between days 48 and 56, and generation 2 appears in the second wave. Parent choice in run C reflects market noise, because costs swamp any skill a bot could have.

**Success criteria, fixed before launch.** Runs A and B last at least 28 days of healthy data and run C at least 56. Results are written up at days 7, 14 and 28, and at day 56 for run C, and no setting changes during a run. Days 7 and 14 are descriptive, and only the final day counts. Every run is reported whatever it shows. The best single bot is never evidence on its own, because among 20 bots chance alone produces one that looks impressive.

1. **Skill, the headline.** One confirmatory test, computed from run A at day 28. It uses the population forecast rows at one minute per 30 minute window, the minutes whose index is a multiple of the horizon, so labels never overlap. Every predictor is scored only from one day after the first forecast row, so the startup noise of a fresh model is never counted. It passes when the correlation between forecast and label is above zero at the one sided 1 percent level, with a Newey West standard error over 12 windows, about six hours, because forecasts drift slowly. The same rows also report the squared error against predicting zero and against the linear baseline, each with a Diebold Mariano test using the same standard error. With about 1,340 windows in 28 days, fewer after gaps, the test can detect a forecast correlation of about 0.1 with 90 percent power, and `evaluate` prints the realized window count and the correlation it can detect. A failure therefore means the forecast's correlation is below about 0.1, not that the market holds no edge. With no skill the squared error is expected to come out a little worse than predicting zero, by about the square of the forecast's spread over the label's, a few percent, and it only breaks even near a correlation of half that ratio. Runs B and C learn from the same labels, so they are reported beside it and are not separate tests.
2. **Beats coin flips.** Each closed learner trade is paired with its twin's trade from the same decision, and pairs where either side has no trade are left out. The comparison uses gross profit before fees, funding and short charges, because those differ only through the direction mix. Pairs are grouped into 12 hour blocks, and it passes when the mean of the block means is positive and above the one sided 1 percent quantile of Student's t, with one degree of freedom fewer than the number of blocks. Blocks keep the test valid when trades of different bots overlap in time and their directions agree. The twin's coin adds noise as well as removing timing luck, so at a correlation of about 0.1 this test has only about a third of criterion 1's power. It is supporting evidence, and failing it alone does not contradict criterion 1.
3. **Survival.** Deaths, causes and lifespans of learners and twins are reported for each run, without a pass threshold, next to the outcomes fixed by arithmetic.
4. **Machinery works.** In run C, by day 56, at least one mutated child has been born and the lineage reaches generation 2. This checks the code path, not the strategy.
5. **Context.** Buy and hold, the exposure matched benchmark and the cash baseline are reported next to every result as context, not as pass criteria.

The `evaluate` command computes criteria 1, 2 and 4 and the outcomes fixed by arithmetic from the database, counting only learner rows wherever twins also write rows, and `docs/RESULTS_TEMPLATE.md` holds the tables for writing them up.

**Road map.** The owner's end goal is a bot that draws on broad market knowledge and current affairs and sizes its trades by confidence. Version 1 is the test bench for that goal. Each later step adds one source of information and must show skill on the forecast records before it is trusted.

1. Longer horizons, where moves are large relative to costs.
2. More market data, such as funding rates, open interest, other assets, and imbalance measured within 5, 10 and 25 bps of the mid.
3. Current affairs as a predictor, for example news headlines scored by a language model, which version 1 forbids inside the loop.
4. Variable position sizing from calibrated forecasts, only after criterion 1 passes for that predictor.

`docs/evolution.md` records these with trading only in volatile hours as later work.

## 22. References

* [Coinbase Advanced Trade WebSocket channels](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-channels)
* [Coinbase Advanced Trade WebSocket rate limits](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-rate-limits)
* [Coinbase Exchange WebSocket channels](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels)
* [Coinbase Exchange WebSocket authentication](https://docs.cdp.coinbase.com/exchange/websocket-feed/authentication)
* [Coinbase Advanced fee tiers](https://www.coinbase.com/advanced-vip)
* [Kraken WebSocket v2 book channel](https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/book)
* [Kraken book checksum guide](https://docs.kraken.com/exchange/guides/websockets/book-checksum-v2)
* [Kraken fee schedule including margin](https://www.kraken.com/features/fee-schedule)
* [Kraken US margin fees](https://support.kraken.com/articles/us-margin-fees)
* [Kraken short selling](https://www.kraken.com/pro/short-selling)
* [Three month Treasury yield, FRED DGS3MO](https://fred.stlouisfed.org/series/DGS3MO)
* [websockets client reference](https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html)
* [numpy release notes](https://numpy.org/doc/stable/release.html)
* [SQLite write ahead logging](https://www.sqlite.org/wal.html)
* [Raspberry Pi OS documentation](https://www.raspberrypi.com/documentation/computers/os.html)
* [OpenBenchmarking PyPerformance on Raspberry Pi 400](https://openbenchmarking.org/result/2412235-NE-RPI40020231)
