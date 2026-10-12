# Architecture

BTC Survivor is one Python process per run. A keyless websocket delivers market data as one ordered stream of events, a deterministic engine turns that stream and the passing of time into minute steps, paper fills and database rows, and a few read only tools report on the database. This page maps the modules and follows the data from the socket to the disk. The reasons behind each choice are in [PLAN.md](../PLAN.md), and what a trade costs is in [the execution model](execution_model.md).

## Module map

The package lives in `src/btc_survivor`, and each module has one job. The package matches [section 5 of the plan](../PLAN.md#5-module-layout), plus `seeds.py`, which holds the seed tags. The tables group the modules by layer.

**Entry point and shared pieces**

| Module | Job |
| --- | --- |
| `__main__.py` | the `btc_survivor` command, with one argparse subcommand and one function per command |
| `config.py` | loads the TOML file with `tomllib` into frozen dataclasses, rejects unknown keys, wrong types and out of range values, and checks relations between keys |
| `clock.py` | UTC millisecond helpers, the live clock that stamps messages, and the monotonic clock used only for durations |
| `types.py` | small dataclasses for feed events, order legs, orders, fills and closed trades |
| `seeds.py` | every random generator, derived from `master_seed` with a fixed tag per purpose |
| `logsetup.py` | standard library logging with UTC timestamps |

**Feed, inside `feed/`**

| Module | Job |
| --- | --- |
| `feed/base.py` | the adapter interface, the `decode` step shared by live and replay, and the websocket client with its reconnect backoff |
| `feed/coinbase.py` | the Coinbase Advanced Trade adapter, the only module that reads Coinbase messages, and the sequence check |
| `feed/book.py` | the local book at full depth, its top 20 levels and the book walk |
| `feed/candles.py` | one minute candles by receive time, and the shared minute series that the features read |
| `feed/health.py` | the Healthy, Recovering and Stale states, the clock skew guard and the feed epoch |
| `feed/market.py` | applies each event to the book, the candles and the health monitor, and asks for a resync when the book is lost |

**Learning, computed once and shared by every bot**

| Module | Job |
| --- | --- |
| `features.py` | the 11 features for each healthy minute |
| `standardizer.py` | running z scores for the features and the running scale of the labels |
| `labels.py` | the pending label queue, the matured window that training batches draw from, and the rule that a label never spans a feed gap |
| `model.py` | the 209 parameter network in numpy, with a hand written backward pass, Huber loss and AdamW |
| `skill.py` | rolling statistics of each prediction against its label, recorded before that label trains anything |
| `baselines.py` | an online ridge regression on the same inputs, scored next to the population in the `forecasts` table |

**Bots and money**

| Module | Job |
| --- | --- |
| `bot.py` | the Warmup, Active and Dead states, the per bot minute step and the pure decision rule |
| `controls.py` | the coin flip twins, which copy their learner's trade timing with a random direction |
| `genes.py` | the margin and learning rate genes, their mutation and their bounds |
| `population.py` | slots, fitness, tournament selection, mutation and replacement |
| `broker.py` | the cost estimate, orders with seeded latency, and fills by walking the book |
| `account.py` | one bot's ledger, the balance checks and the death threshold |
| `benchmark.py` | the buy and hold, exposure matched and cash curves |

**Runtime and storage**

| Module | Job |
| --- | --- |
| `engine.py` | the deterministic core that live and replay share |
| `records.py` | every SQL statement the engine writes, and the builders that turn bots, fills and trades into rows |
| `store.py` | the SQLite connection, the schema, batched writes and rollback to an older checkpoint |
| `checkpoint.py` | encoding and decoding of the whole engine state |
| `runner.py` | the live asyncio program, signal handling, the heartbeat line and `runtime_status` |
| `recorder.py` | raw message recording as gzip JSON lines, and the reader that replay uses |
| `replay.py` | drives a fresh engine from recordings at full speed |
| `synthetic.py` | synthetic recordings with injected faults, for tests and soak runs, never imported by the live path |

**Reporting, all read only**

| Module | Job |
| --- | --- |
| `cli/data.py` | the read only queries that the commands and the dashboard share |
| `cli/status.py`, `cli/explain.py`, `cli/lineage.py`, `cli/export.py`, `cli/evaluate.py` | one module per reporting command |
| `cli/stats.py` | the statistics behind the success criteria, written out in numpy so each step can be checked |
| `dashboard.py` with `dashboard.html` | the local page, its JSON view of each run and the two JSON views behind the decisions explorer, served on 127.0.0.1 only and only to requests addressed to `127.0.0.1` or `localhost` |

Three development scripts sit beside the package. `scripts/record_feed.py` runs the adapter, `feed/market.py` and the recorder with the live reconnect rules but without bots or a database, `scripts/learner_check.py` checks the learner on synthetic data with a planted signal, and `scripts/plot_results.py` draws charts from exported CSV files on a laptop. The source is in [the package folder](../src/btc_survivor), and [section 4 of the plan](../PLAN.md#4-changes-to-the-brief-and-chosen-readings) explains the main additions to the brief's layout, such as `engine.py`, `feed/market.py` and `synthetic.py`.

## Data flow at a glance

```
Coinbase Advanced Trade websocket  (level2, market_trades, heartbeats on one keyless connection)
        |
FeedClient                      live only, stamps each message with its receive time
        |   \
        |    Recorder           optional, data_dir/raw/*.jsonl.gz
        v
Engine.on_line  ->  decode  ->  BookSnapshot | BookDelta | Trade | FeedNotice
        |
Market                          LocalBook, CandleBuilder, HealthMonitor
        |
Engine.advance_to(t)            minute boundaries and persistence points, in time order
        |
minute step  ->  bots  ->  Broker orders  ->  fills at the first book update after latency
        |
Store queue  ->  one transaction per checkpoint  ->  data_dir/btc_survivor.db
        |
status, explain, lineage, export, evaluate, dashboard      read only
```

Replay enters at `Engine.on_line` with recorded lines instead of live ones, and everything below that point is the same code. The full timeline is in [section 6 of the plan](../PLAN.md#6-time-model-and-data-flow).

## One ordered event stream

The adapter turns each raw message into events of four types from `types.py`. A `BookSnapshot` replaces the whole book, a `BookDelta` carries absolute size changes, a `Trade` carries one trade, and a `FeedNotice` carries connects, disconnects, heartbeats, sequence gaps, trade gaps and errors. The brief asked for two outputs, trades and book updates. They travel on one stream rather than two, because the order between a trade and a book update matters and replay must reproduce it exactly.

Every event carries `recv_ms`, its receive time in integer milliseconds since the UTC epoch. Live, `FeedClient` stamps each message the moment it is read, through `LiveClock.stamp`, which never goes backwards and is never earlier than the engine's current time. In replay the stamp comes from the recording, which stores it after that adjustment, so replay sees exactly what live saw. The exchange's own timestamp travels alongside as `exch_ms` and is used only by the clock skew guard. Candles group trades by receive time, so a trade that the exchange stamped just before a boundary but that arrived after it belongs to the next minute, because at the boundary it was not yet known.

Parsing is a pure function of the raw text, the receive time and a little per connection state that an `open` line resets. `decode` turns recorded or live `open` and `close` lines into `connected` and `disconnected` notices, so a reconnect replays exactly as it happened. Only `feed/coinbase.py` knows the Coinbase wire format. `config.py` checks that the `exchange` name names a module in `feed/`, and `load_adapter` in `feed/base.py` loads that module for the runner and for replay, so another exchange means one new file in `feed/` and no change anywhere else.

`feed/market.py` applies each event in order. Book events update the book, trades update the open candle, and every message tells the health monitor that the socket is alive. A sequence gap or a crossed or one sided book throws the book away at once. Those faults, an unreadable message and a stale socket all ask for a resync, at most once per connection. A resync is always a reconnect, which clears the book and brings a fresh snapshot, so there is one recovery path.

`feed/health.py` starts in Stale. The feed becomes Stale when no message, or no good book update, has arrived for more than `stale_after_seconds`, which is 5. From Healthy, any reconnect, lost book or clock skew moves it to Recovering, and a Stale feed passes through Recovering once fresh data arrives. It becomes Healthy at the first minute boundary that closes a full minute in which the book was valid and fresh and the clock agreed with the exchange. Every exit from Healthy increments the feed epoch, and a label refuses to mature across an epoch change. The clock skew guard takes the median of receive time minus the message envelope timestamp over the last 50 messages, once it has at least 10, and trips above `max_clock_skew_seconds`, which is 2. It holds the feed in Recovering without reconnecting, because a reconnect cannot fix a wrong local clock. A large median can also mean that Coinbase is delivering messages late, so a `clock_skew` event says the two clocks disagree, not which one is wrong. Every state change and every gap, resync, reconnect and skew is a row in `feed_events`. See [section 6.1](../PLAN.md#61-one-timeline-for-live-and-replay) and [section 7 of the plan](../PLAN.md#7-feed), [`feed/base.py`](../src/btc_survivor/feed/base.py), [`feed/market.py`](../src/btc_survivor/feed/market.py) and [`feed/health.py`](../src/btc_survivor/feed/health.py).

## How the engine sees time

The engine never reads a clock. It has two entry points. `advance_to(t)` runs every scheduled item due at or before `t` in time order, and the scheduled items are minute boundaries and persistence points. `on_event(event)` first calls `advance_to` with the event's receive time, then applies the event to the market, and then, if the event was a book update and orders are waiting, settles every order that has fallen due. If the feed is no longer Healthy, every waiting order is rejected instead.

Live, the feed task hands every message to `on_line`, which decodes it and calls `on_event`, and a ticker calls `advance_to` with the current time once a second, so boundaries and persistence points fire even when no message arrives. In replay only `on_line` is called. Both give the same result, because the book at a boundary is always the book after every message received before that boundary. See [section 6.2 of the plan](../PLAN.md#62-event-handling-order) and [`engine.py`](../src/btc_survivor/engine.py).

## The minute step

At each UTC minute boundary M the engine does the following, in this order.

1. Close the candle for the minute that ends at M, and take the book mid at M as the minute mid. The candle row is queued with its validity flag whether the minute was healthy or not, so the minute series always lines up with the clock.
2. Judge the minute. It counts if the feed was Healthy for all of it, or if it is the full clean minute that completes a recovery, and a usable book exists at M. An unhealthy minute stops here. Nothing trades, nothing learns, no rent accrues, no label is created, and any order still waiting is rejected.
3. Do the shared work once for every bot. Compute the 11 features if the windows hold enough valid history, standardize them with the statistics from before this minute and then update those statistics. Mature the label for M minus the horizon if it is still in the current feed epoch, and add it to the matured window, the label scale and the linear baseline. Queue a new record for M.
4. Flatten after a gap. If more than `max_resume_gap_minutes`, which is 10, have passed since the last healthy minute, send a closing order for every open position, with exit reason `restart_flatten` after a restart or `gap_flatten` after an outage while the process stayed up, and write a `resume` event.
5. Resend the closing order of any dead bot whose position has not closed yet.
6. Step each slot in order, the learner first and then its twin. A learner pays one minute of rent, accrues funding on a short, marks its position to the exit side of the book, checks death, checks its stop loss, takes one training step if a new sample matured, predicts for M, and decides at its phase minute, where the UTC minute index modulo the horizon equals its slot modulo the horizon. A twin does the same upkeep, copies its learner's open, close or flip with a seeded coin choosing the side, and closes when its learner is stopped out or dies.
7. Once the features are valid and the label scale is ready, write one forecast row for the population, the mean prediction of the living learners, and one for the linear baseline.
8. Replace each bot that died. The replacement takes the slot of the bot that died, so the population is full again before the minute ends.
9. When due, every `equity_snapshot_every_seconds`, which is 300, write an equity snapshot for every living bot and run the balance checks on every bot whose books are still open. Then write one benchmark row.

Every row goes into the store's in memory queue, and nothing reaches the disk until the next checkpoint, at most `checkpoint_every_seconds` later. The minute step is synchronous, so two steps can never overlap, and a step slower than 10 seconds is logged as slow. A bot has at most one order in flight, and the engine keeps a map from bot to waiting order so that each book update settles the due ones, earliest due first. How those orders are priced is in [the execution model](execution_model.md), and the learning half of step 6 is in [the learning notes](learning.md). See [section 6.3 of the plan](../PLAN.md#63-the-minute-step), [`engine.py`](../src/btc_survivor/engine.py) and [`bot.py`](../src/btc_survivor/bot.py).

## Persistence points

A persistence point sits 30 seconds after each minute boundary, well away from the boundaries and from fills, which all fall due 1.0 to 1.3 seconds after a boundary. Every `checkpoint_every_seconds` of engine time, which is 300, the persistence point writes every buffered row, every changed bot state and a checkpoint in one transaction. In the other four minutes it commits only the new feed events. Live, the runner also writes the single `runtime_status` row once a minute, with the feed state, best bid, best ask, mid, living count, last checkpoint time and memory use. Neither `feed_events` nor `runtime_status` ever feeds back into bot state.

All timestamps are engine time, written as ISO 8601 UTC strings with milliseconds and a trailing `Z`, so text order is time order and two replays write identical rows. A power cut loses at most `checkpoint_every_seconds` of history, but every bot row and ledger row on disk matches a checkpoint. Only `feed_events` and `runtime_status` can be newer. WAL with `synchronous=NORMAL` keeps the database consistent but may lose the very last commit, so a power cut can cost slightly more than one checkpoint interval. See [section 6.5 of the plan](../PLAN.md#65-persistence-points) and [`engine.py`](../src/btc_survivor/engine.py).

## The store

`store.py` owns one SQLite connection on the event loop thread, opened with `journal_mode=WAL`, `synchronous=NORMAL` and a `journal_size_limit` of 64 MiB. Writes are queued in memory and committed by `flush` in one `BEGIN IMMEDIATE` transaction. If the transaction fails nothing is committed and the queue is kept, so the same writes can be retried. Every write is timed on the monotonic clock, and anything slower than one second is logged. `records.py` holds every statement and builds every row, so the engine only decides what to write and when.

Each run keeps its database at `btc_survivor.db` inside its `data_dir`, next to `logs` and, when recording is on, `raw`. On open, an empty file receives the schema, and any other file must match the expected tables, columns, indexes and `schema_version`, or the command stops and lists the differences. The schema has 14 tables, which are `meta`, `bots`, `lineage`, `decisions`, `trades`, `fills`, `equity_snapshots`, `deaths`, `feed_events`, `checkpoints`, `candles`, `benchmark`, `forecasts` and `runtime_status`. Candles older than `keep_candles_days`, which is 14, are pruned in the checkpoint transaction, and only the newest `keep_checkpoints`, which is 3, are kept. Decisions, trades, fills, deaths, lineage, forecasts and bots are kept forever. At 20 learners, their 20 twins and a 30 minute horizon the plan estimates roughly 700 MB a year.

Readers never write. `cli/data.py` opens each database through a SQLite URI with `mode=ro`, adding `immutable=1` only when no `-wal` file exists. Bot details come from the newest checkpoint, so they can be up to 300 seconds old, and the live price comes from `runtime_status`. The dashboard uses the same queries, caches the result for five seconds and binds to 127.0.0.1 only. It also answers 403 to any request, whatever its method, whose Host header is not `127.0.0.1` or `localhost` with its own port, so a web page elsewhere cannot reach it through DNS rebinding, and every answer it gives, its error pages included, carries the same security headers. Its decisions explorer runs its own fixed queries with every filter passed as a parameter, reads each answer in one transaction so its counts and rows agree, stops any query after three seconds and keeps each answer for five seconds. See [section 12 of the plan](../PLAN.md#12-storage-schema), [`store.py`](../src/btc_survivor/store.py), [`records.py`](../src/btc_survivor/records.py), [`cli/data.py`](../src/btc_survivor/cli/data.py), [`dashboard.py`](../src/btc_survivor/dashboard.py) and the [SQLite write ahead log notes](https://www.sqlite.org/wal.html).

## Checkpoints and resume

For every bot a checkpoint holds the network parameters, the Adam moments and step count, the genes, the account, the open position and any order in flight, the pending predictions, the skill buffers, the hourly equity samples, the counters, the slot and the state of the bot's own generator. It also holds the dying list and the shared state, which is the standardizer, the label scale, the matured window, the feed epoch, the last healthy minute, the id counters, the population generator, the benchmark anchor and the linear baseline's statistics. Scalars and generator states go into JSON with sorted keys. Arrays go into one zlib compressed blob of raw float64 values, whose layout is listed in the JSON. There is no pickle, and there is no `npz` file, because zip archives embed the current time and two identical replays would then write different bytes. Before a checkpoint is committed it is decoded once in memory to prove it reads back.

Two things are left out on purpose. The pending label queue is not stored, because every restart is a feed gap that discards it anyway. The minute series is not stored either, because it is rebuilt from the `candles` table, which is committed in the same transaction as each checkpoint.

On start, `Engine.start` loads the newest checkpoint that decodes. If it has to fall back to an older one, it first removes or reopens every bot and ledger row written after that checkpoint in one transaction, clears labels that matured after it, keeps `feed_events`, records why in a `resume` event and commits a fresh checkpoint at once. It refuses to resume when the master seed, the population size or the twin setting differs from the run in the database, and it refuses to seed a new population over a database that holds bots but no readable checkpoint. An empty database gets a fresh population from the master seed.

A restart is treated as a feed gap. The epoch increments, so every pending label is discarded. Orders that were in flight are rejected with reason `restart`, while stops and death closes retry through their normal paths. Rent is never charged for the downtime, because rent accrues only in healthy minutes. At the first healthy minute after a restart, the gap since the last healthy minute stored in the checkpoint decides whether step 4 of the minute step flattens every position with exit reason `restart_flatten`.

SIGINT and SIGTERM set a flag. The runner lets the current minute step finish, waits at most `latency_ms` plus `latency_jitter_ms`, which is 1.3 seconds, for orders in flight to settle while the feed is Healthy, writes a final checkpoint with every buffered row, closes the database and exits with code 0. See [section 13 of the plan](../PLAN.md#13-checkpoints-resume-and-shutdown), [`checkpoint.py`](../src/btc_survivor/checkpoint.py) and [the operations guide](operations.md).

## Live and replay share one engine

`btc_survivor run` calls `run_live` in `runner.py`. It opens the store, builds the engine with `Engine.start` and runs one asyncio loop with three tasks. The feed task keeps the websocket open through `FeedClient`, hands each message to the recorder when recording is on and then to `Engine.on_line`, passes any resync request back to the socket, and resets the reconnect backoff once the feed is Healthy. The backoff starts at `reconnect_backoff_start_seconds`, doubles up to `reconnect_backoff_max_seconds`, and draws its jitter between half and the full delay from a generator seeded by `master_seed`. The ticker task calls `advance_to` once a second and, once a minute, logs a heartbeat line and writes `runtime_status`. The shutdown task waits for a signal. Each run is its own process with its own config file, data folder and connection, which is how runs A to D share one machine.

`btc_survivor replay --out DIR FILE` calls `replay` in `replay.py`. It reads the recordings in order, builds a fresh engine born at the first recorded line, calls `on_line` for every line and finishes with a final checkpoint. Recorded `open` and `close` lines reset the adapter exactly as a live reconnect does, and the replay clock is the recorded receive time, so health, staleness, latency fills, boundaries and persistence points all behave as they did live, only faster. Replay never loads a checkpoint, never writes `runtime_status`, and writes only into a new or empty folder that is not the configured `data_dir`. Replay results are never used to choose, tune, or rank strategies, and that sentence appears in `replay.py` and in the command line help. See [section 14](../PLAN.md#14-runtime-and-command-line) and [section 15 of the plan](../PLAN.md#15-replay-harness-and-determinism), [`runner.py`](../src/btc_survivor/runner.py) and [`replay.py`](../src/btc_survivor/replay.py).

## Determinism

Given the same master seed and the same recorded messages, two replays write identical databases. Four rules make that true.

* The engine never reads the wall clock. It learns about time only through receive stamps and `advance_to`, and the monotonic clock only measures durations for the logs.
* Every random draw comes from a generator seeded through `numpy.random.SeedSequence` from `master_seed` and a fixed tag in `seeds.py`. Each bot's generator uses the bot tag and the bot id, and it drives the initial weights or, for a mutated child, its mutation noise, the forced timer delay, the training batches, the latency jitter and a twin's coin. The population generator draws newcomers and tournament picks, and the reconnect generator draws the backoff jitter.
* Every bot generator and the population generator are saved in every checkpoint, so a resumed run continues the same random streams for its decisions. The reconnect generator is not saved. It restarts from the master seed at each start, which affects only the backoff timing.
* Iteration order is fixed. Slots run in order, each learner before its twin, waiting orders settle by due time and then by the order they were sent, and checkpoint bytes depend only on the state, never on the time they were written.

`tests/test_replay.py` replays a synthetic recording with a sequence gap, a crossed book, a silence and a disconnect twice and compares a digest of every table, and a different synthetic recording changes the decisions. `tests/test_engine.py` drives the engine with one second ticks and clamped live stamps and checks that it writes the same rows as replay.

Some honest limits apply.

* Replay always starts from a fresh population, so it reproduces a live run only when the run never restarted and the recording covers it from its first message. After a restart, live writes `restart` rejections and `restart_flatten` closes where a replay of the recording sees only a silence, `feed_not_healthy` rejections and `gap_flatten` closes. A crash also loses up to 300 seconds of rows that the recording still holds.
* Runs A to D set `record_raw_messages = false`, so they keep no recordings and none of them can be replayed.
* Live, the ticker can notice a silent socket between messages, so a live `stale` event can carry an earlier timestamp than the same event in a replay of its recording.
* Live, `Engine.start` gives the founding population the clock time just before the socket opens, while replay gives it the receive time of the first recorded line. So `born_at` of the founding learners and twins differs, and so does the `lifespan_hours` of their death rows, which is computed from it. Every other bot and ledger value is the same either way.

See [section 15 of the plan](../PLAN.md#15-replay-harness-and-determinism), [`seeds.py`](../src/btc_survivor/seeds.py) and [`tests/test_replay.py`](../tests/test_replay.py).

## What the code cannot do

No module can place, cancel or sign an order on any exchange. `broker.py` contains no exchange client, the feed uses only keyless public channels, and nothing in the project asks for or stores an API key. `tests/test_no_orders.py` scans the source for forbidden names and endpoints such as `place_order`, `cancel_order`, `create_order`, `/orders`, `api_key`, `api_secret`, `passphrase`, `cb-access`, `hmac` and `jwt` in `src` and `scripts`, and fails if any appears. The dashboard and every reporting command open the database read only, so watching a run cannot change it. See [section 17 of the plan](../PLAN.md#17-test-plan), [`tests/test_no_orders.py`](../tests/test_no_orders.py) and [the honesty and limits page](honesty_and_limits.md).
