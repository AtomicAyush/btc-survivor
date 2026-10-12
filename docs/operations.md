# Operations

This guide covers running BTC Survivor day to day on the MacBook that hosts the four live runs, and reading what the runs report. Commands assume the project folder as the working directory and the `.venv311` environment. On a Raspberry Pi each run is a systemd service instead, as [raspberry_pi.md](raspberry_pi.md) describes, and the runtime design is in [section 14 of the plan](../PLAN.md#14-runtime-and-command-line).

## Where everything lives

* `configs/run_a.toml` to `configs/run_d.toml` hold the settings of each run. A relative `data_dir` is resolved from the config's own folder, so `../data/run_a` means `data/run_a` in the project.
* `data/run_a/btc_survivor.db` is the run's database, with its `-wal` and `-shm` files beside it. The three belong together, so never delete or move one of them alone.
* `data/run_a/logs/btc_survivor.log` is the log file, rotated at UTC midnight with 14 days kept.
* `data/run_a/stdout.log` is everything the process prints. It holds the same lines as the log file and is never rotated.
* `data/dashboard.log` is the dashboard's output.
* `data/raw` holds raw feed recordings, and `data/record_feed.log` holds the recorder's minute lines.

Nothing under `data/` is tracked by git. A consistent copy of a live database can be taken without stopping the run.

```bash
sqlite3 -readonly data/run_a/btc_survivor.db ".backup $HOME/run_a_copy.db"
```

The tables are described in [section 12 of the plan](../PLAN.md#12-storage-schema).

## Starting runs

The four runs and the dashboard were started on 11 October 2026, from the project folder, with commands equivalent to these. Runs A, B and C started at `20:41` UTC, and run D and the dashboard at `21:39` UTC. The shell opens each `stdout.log` before Python starts, so on a fresh clone each data folder must exist first, which `init-db` takes care of.

```bash
nohup .venv311/bin/python -m btc_survivor --config configs/run_a.toml run > data/run_a/stdout.log 2>&1 &
nohup .venv311/bin/python -m btc_survivor --config configs/run_b.toml run > data/run_b/stdout.log 2>&1 &
nohup .venv311/bin/python -m btc_survivor --config configs/run_c.toml run > data/run_c/stdout.log 2>&1 &
nohup .venv311/bin/python -m btc_survivor --config configs/run_d.toml run > data/run_d/stdout.log 2>&1 &
nohup .venv311/bin/python -m btc_survivor dashboard configs/run_a.toml configs/run_b.toml configs/run_c.toml configs/run_d.toml --port 8050 > data/dashboard.log 2>&1 &
```

`nohup` keeps each process alive after the terminal closes, and `2>&1` sends the log lines into `stdout.log`. A database without a checkpoint starts a fresh population from the master seed, and a database with one resumes. A clean start looks like this in run A's log, with its heartbeat lines left out and the data path shortened.

```
2026-10-11T20:41:43.261Z INFO btc_survivor.runner running coinbase BTC-USD with 20 learners, data in .../data/run_a
2026-10-11T20:41:43.527Z INFO btc_survivor.engine feed event 2026-10-11T20:41:43.527Z connected
2026-10-11T20:41:43.961Z INFO btc_survivor.engine feed event 2026-10-11T20:41:43.914Z snapshot 21626 bids, 19345 asks
2026-10-11T20:41:43.961Z INFO btc_survivor.engine feed event 2026-10-11T20:41:43.914Z recovering fresh data
2026-10-11T20:43:00.078Z INFO btc_survivor.engine feed event 2026-10-11T20:43:00.000Z healthy a full clean minute
```

A few rules keep restarts safe.

* Restart a run with `>>` in place of `>`, because `>` empties `stdout.log`. The rotated log file keeps its own 14 days either way.
* Never start a second process on the same config. Nothing in the code prevents it, and two writers would mix two histories in one database. Check with `pgrep -fl btc_survivor` first.
* Never edit a run's config, even between restarts. A restart refuses a changed master seed, population size or twin setting, but any other edit would pass unnoticed and break the rule that settings are fixed before launch.
* Restart a run in the environment it started with, `.venv311` for all four, so its Python and numpy stay the same.
* Both environments on the Mac are editable installs, so a restarted run loads the source as it stands at that moment. Restart only from a commit that leaves version 1 behaviour unchanged, and note the commit in the results.
* Never check out another branch in `~/code/btc_survivor` while the runs are live. The dashboard reads `dashboard.html` from that tree on every request and reloads the open page when the file changes, so a checkout changes the running dashboard at once, with no restart. Other work happens in a separate worktree, as [section 3 of the version 2 plan](../PLAN_V2.md#3-scope-isolation-from-version-1-and-module-layout) arranges.
* Put `--config` before the command. After the command it is rejected.

The timeline that follows a launch is in [section 3 of the plan](../PLAN.md#3-what-to-expect-from-the-default-experiment).

## Stopping runs

```bash
pgrep -fl btc_survivor
pkill -TERM -f 'configs/run_a.toml run'
tail -n 2 data/run_a/stdout.log
```

SIGTERM, or SIGINT from Ctrl+C in a foreground terminal, asks for a clean stop. The run finishes the current minute step, waits up to 1.3 seconds, which is `latency_ms` plus `latency_jitter_ms`, for pending fills while the feed is healthy, writes a checkpoint with every buffered row, closes the database and exits with code 0. The log then ends with `stopped cleanly, last checkpoint` and the checkpoint time, and `status` shows the feed as stopped. Never stop a run with `kill -9`. It skips the final checkpoint and loses up to five minutes of rows, although the run still resumes safely. The dashboard only reads, so it can be stopped at any time. Shutdown is specified in [section 13 of the plan](../PLAN.md#13-checkpoints-resume-and-shutdown).

## The heartbeat line

Once a minute, about 30 seconds past the minute, each run logs one heartbeat line and writes the same values to the `runtime_status` row that `status` and the dashboard read. It keeps coming during stale periods, so a missing heartbeat means the process is frozen or gone. This is a real line from run A.

```
2026-10-12T02:33:30.784Z INFO btc_survivor.runner heartbeat feed=healthy living=20 equity mean=9999.80 min=9999.80 max=9999.80 deaths=0 rss=n/a peak=90.9 MB lag=2 ms last_message_age=0.1 s
```

* `feed` is the feed state, `healthy`, `recovering` or `stale`. Bots act only in healthy minutes.
* `living` counts the living learners and leaves the twins out.
* `equity mean`, `min` and `max` are the living learners' equity in dollars, with open positions marked at the price they would close at.
* `deaths` counts every death since launch, twins included.
* `rss` is current resident memory. macOS does not expose it to the process, so it reads `n/a` on the Mac and shows a number on Linux.
* `peak` is peak resident memory. Each run started near 46 MB and peaked between 89 and 91 MB. Runs A, B and C reached that within their first second, while parsing the first full book snapshot of about 41,000 price levels, and it did not rise again during the first night.
* `lag` is the worst lateness of the one second tick during the minute. A few milliseconds is normal, and the worst on the first night was about 150 ms.
* `last_message_age` is the time since the last feed message, in seconds.

Three other lines are worth a search. `slow minute step` appears when a minute step takes over 10 seconds, `slow database write` when a write takes over a second, and the ERROR line `invariant violation` when a ledger check fails, which also writes an `invariant_violation` feed event. A healthy run shows none of them. The heartbeat is specified in [section 14 of the plan](../PLAN.md#14-runtime-and-command-line).

## The dashboard and status

```bash
.venv311/bin/python -m btc_survivor --config configs/run_a.toml status
.venv311/bin/python -m btc_survivor --config configs/run_a.toml status --watch --twins
```

`status` starts with a header that holds the feed state, the live BTC price and its age, the launch time and the healthy days so far. A stage line follows, which moves from collecting history to waiting for outcomes to mature, then to training while learners finish warmup, and finally to trading once every learner is active. The table has one row per living learner, and `--twins` adds the twins. Below it come buy and hold, the exposure matched benchmark, the cash baseline, the twin mean and one summary line. That line compares the total equity of every learner ever born, dead ones included, with the capital they were given, so losses stay visible after replacements arrive. `--watch` redraws the whole view every minute.

Two warnings matter. One says the system is not running, when `runtime_status` is missing or more than two minutes old. The other says the feed is not healthy, so bots are paused. Bot rows come from the latest checkpoint and can be up to five minutes old, while the price and the feed state are at most a minute old.

The dashboard at [http://127.0.0.1:8050](http://127.0.0.1:8050) shows the same numbers from the same read only queries for all four runs, with the equity chart, the skill so far and the latest decisions, fills and feed events. A run whose panel shows an error could not be read, and `data/dashboard.log` holds the reason. Days in `status`, the dashboard and `evaluate` are days of healthy data, 1,440 healthy minutes each, not calendar days. The page is described in the [README](../README.md#the-local-dashboard).

## Reading feed events

Every change in feed health is written to the `feed_events` table and to the log.

```bash
grep 'feed event' data/run_a/logs/btc_survivor.log | tail -n 20
sqlite3 -readonly data/run_a/btc_survivor.db "SELECT ts, kind, detail FROM feed_events ORDER BY id DESC LIMIT 20"
```

The feed is always in one of three states.

* **Healthy.** Messages arrive, the local book is valid, and local time agrees with the exchange's timestamps. Bots predict, learn, pay rent and trade in healthy minutes and in no others.
* **Recovering.** Something broke and is being repaired, after a reconnect, a lost book or clock skew. The feed turns healthy at the first minute boundary that closes a full minute in which everything held, logged as `healthy a full clean minute`, so a short break usually costs one or two minutes.
* **Stale.** No message, or no good book update, arrived for more than `stale_after_seconds`, which is 5. The detail says `no messages` or `no good book`, and the connection is closed and opened again.

| Kind | Meaning |
| --- | --- |
| `connected`, `disconnected` | a websocket opened or closed, with the reason in the detail |
| `snapshot` | a full book arrived, with its level counts, lately about 21,600 bids and 19,300 asks |
| `recovering`, `healthy`, `stale` | the feed changed state, with the reason |
| `gap` | the connection's `sequence_num` jumped, so a message was lost, the book is dropped and the feed reconnects |
| `crossed`, `one_sided`, `error` | the book became unusable or Coinbase sent an error, and the feed reconnects |
| `trade_gap` | trade ids skipped while the book stayed intact, so that minute's volume may be undercounted, with no reconnect |
| `clock_skew` | receive times drifted from Coinbase's timestamps, as the next section explains |
| `resume` | a restart restored a checkpoint, or the first healthy minute after a gap longer than 10 minutes flattened open positions |
| `invariant_violation` | a ledger check failed, which should never happen |

Reconnects are log lines rather than events, such as `feed connection 16 ended (stale), reconnecting in 58.4 s`. The wait doubles from 1 second up to 60, each wait is drawn between half and all of that, and it resets only once the feed is healthy again. A `gaierror` in that line means a DNS lookup failed, which is what a lost network looks like. The feed is specified in [section 7 of the plan](../PLAN.md#7-feed).

## What clock skew really measures

Every live message carries Coinbase's own timestamp. The health monitor keeps the difference between receive time and that timestamp for the last 50 messages, and once it holds at least 10 it compares their median with `max_clock_skew_seconds`, which is 2. Above that it logs `clock_skew` with the median and moves the feed to recovering without reconnecting, because a reconnect cannot fix a clock. When the median falls back it logs `clock_skew cleared`, and the usual full clean minute follows.

The name says clock, but the number is the clock error plus the delivery delay. A wrong local clock raises it, and so do a sleeping or waking Mac, slow delivery from Wi Fi, the network path or Coinbase itself, and a busy machine. Every trip on 11 October came after the lid had closed, with medians of 2.0 to 3.0 seconds. The power log shows a clamshell sleep at `23:26:19` UTC, and 27 seconds later all four runs and the recorder logged `clock_skew` within the same second, as messages that had queued while the Mac slept arrived during its short dark wakes. More trips followed over the next half hour while it went on sleeping and waking. The recorder's earlier trip, at `19:38` UTC before the runs started, came as the Mac woke from a clamshell sleep that began at `19:35:36` UTC. The closed lid caused both the trips and the outage that followed, and neither caused the other. The only trip so far while the Mac was awake came at `02:07` UTC on 12 October, when run D logged three `clock_skew` events with medians of 2.0 to 2.6 seconds and run C went stale with no messages in the same minute, while runs A and B stayed healthy. The Mac's clock was checked against `time.apple.com` that night and agreed to within about 25 ms, the uncertainty of the measurement, so none of these events was a wrong clock. This command prints the offset and changes nothing.

```bash
sntp time.apple.com
```

A trip out of healthy is a real gap with the same costs as any other, listed in the next section, so in run D each one pauses learning for 4 hours. The limit is fixed for the length of each run, like every other setting. The guard is described in [section 7 of the plan](../PLAN.md#7-feed).

## Sleep, restarts and outages

When the Mac sleeps, every process freezes. On wake the one second ticker catches up, every missed minute is judged unhealthy, the silent socket goes stale and closes, and the feed reconnects through recovering to healthy. A restart, a lost network and a clock skew trip all follow the same rules.

* **Rent stops.** Rent accrues only in healthy minutes, so no bot pays for downtime. By `02:33` UTC on 12 October each learner in run A had paid about 0.20 dollars of rent, which matches the roughly 4.2 healthy hours since launch and not the 5.9 hours on the clock.
* **Shorts keep paying funding.** Funding accrues through stale minutes and downtime, because a real short would still be open.
* **Long gaps flatten.** At the first healthy minute after a gap longer than `max_resume_gap_minutes`, which is 10, every open position is closed through the normal broker. The exit reason is `restart_flatten` after a restart and `gap_flatten` after an outage while the process stayed up, and a `resume` event records the length of the gap and how many closing orders went out. A rejected flatten is sent again at the next healthy minute.
* **Pending outcomes are dropped.** Every exit from healthy starts a new feed epoch, and no label crosses an epoch, so every pending prediction is discarded. No outcome matures for one horizon afterwards, so learning pauses for 30 minutes in runs A, B and C and for 4 hours in run D, and warmup, which counts training minutes, takes that much longer.
* **Features need clean windows.** A return needs a healthy minute exactly at its start, and each volatility and volume window needs 80 percent of its minutes healthy. A gap of 1 to 3 minutes costs four times its length, the later minutes whose return windows start inside it. A gap of 4 or 5 minutes costs about 25 minutes, a gap of 6 to 23 minutes about 35 to 55 minutes, one longer than 23 minutes about 1.6 hours, and one longer than 72 minutes about 4.8 hours of healthy data, while the 30 minute volatility, the 120 minute volatility and the 360 minute volume baseline refill. In that time bots neither predict nor learn nor decide, while rent, marking and stop checks go on.
* **A restart is not a death.** Bots keep their weights, genes, accounts and positions, as the next section explains.

The first night shows the whole sequence. Each run's last healthy minute before the outage was at `23:26` UTC. From `23:27` UTC, seconds after the lid closed at `23:26` UTC, run A's log shows stale connections and reconnects whose waits grew from 0.9 seconds at `23:27:03` to 58.6 seconds at `23:33:59`, and the first DNS failure, a `gaierror`, came at `23:37:39`. At `01:05` UTC every run logged `resume 99.0 minutes since the last healthy minute, gap_flatten sent 0 closing orders`. No bot was trading yet, so nothing needed closing.

For a run that has finished collecting history and is learning, a sleep that long costs the gap itself, about 4.8 healthy hours of invalid features and one more horizon before learning resumes. The first night cost less, because the runs were still collecting their first 360 healthy minutes of history. No label had matured and no learning had begun, so no learning was lost. In run A the sleep pushed the first valid feature vector from about `02:40` to about `05:50` UTC, which is the 99 minute gap plus about 1.5 healthy hours, because the 287 minute refill after the gap overlapped the 196 healthy minutes of history that were still to come. These rules are specified in [section 13 of the plan](../PLAN.md#13-checkpoints-resume-and-shutdown).

## Checkpoints and resume

Every 5 minutes, 30 seconds after a minute boundary, a run writes every buffered row and one checkpoint in a single transaction. The checkpoint holds each bot's weights, optimizer state, genes, account, open position, pending order, predictions, skill buffers and random generator state, plus the shared standardizer, label scale, matured outcomes, feed epoch and counters. It is decoded once in memory before it is committed, the newest 3 are kept, and candles older than 14 days are pruned in the same transaction. Feed events and `runtime_status` are committed every minute and never feed back into bot state.

On start, `run` loads the newest checkpoint that decodes and logs a `resume` event that says `restored the checkpoint from` and its time. Orders that were in flight are rejected with reason `restart`, while stop and death closes are tried again. If the gap turns out longer than 10 minutes, the first healthy minute flattens with `restart_flatten`. If only an older checkpoint decodes, every row written after it is rolled back first and the reason is recorded. A run refuses to resume when the master seed, the population size or the twin setting differs from its database, and it refuses to seed a new population over a database that holds bots but no readable checkpoint. A `kill -9` or a power cut therefore loses at most five minutes of history, and the books and the bot state always agree. Checkpoints are specified in [section 13 of the plan](../PLAN.md#13-checkpoints-resume-and-shutdown).

## The recorder

Runs A to D record no raw messages, because their configs set `record_raw_messages = false`. A separate development recorder has run since `16:58` UTC on 11 October, reading `config.toml` and writing into `data/raw`.

```bash
nohup .venv311/bin/python scripts/record_feed.py --config config.toml > data/record_feed.log 2>&1 &
```

It runs the same adapter, book, health monitor and reconnect rules as a run, with no bots and no database, and logs one line a minute with the message count, bytes, book level changes, trades, the mid and how often the feed has left healthy. Each start opens a new file named after its UTC start time, such as `2026-10-11_165829.jsonl.gz`, and files rotate at UTC midnight, so a power cut can only cut short the end of one file, which the reader tolerates. SIGTERM or Ctrl+C stops it, and the plan stops it after Phase 10.

Recordings feed the replay harness, which writes only into a new or empty folder.

```bash
.venv311/bin/python -m btc_survivor --config configs/run_a.toml replay --out $HOME/replays/first data/raw/2026-10-11_165829.jsonl.gz
```

Replay always starts a fresh population, never loads a checkpoint, and gives identical results for the same seed and the same files. Replay results are never used to choose, tune, or rank strategies. The harness is specified in [section 15 of the plan](../PLAN.md#15-replay-harness-and-determinism).

## Disk growth

* **Raw recordings** are the largest item. The feed delivered about 15 messages and 20 KB of JSON a second, and the first recording file holds 38.6 MB for about 7 hours, so compressed recordings grow by about 5.5 MB an hour, or 130 MB a day.
* **Run databases** grow slowly. With 20 learners, 20 twins and a 30 minute horizon, decisions grow by about 1,000 rows a day, forecasts by about 2,900 and equity snapshots by about 11,500, roughly 700 MB a year per run by the plan's estimate. Run D decides once every 4 hours, so its decisions grow more slowly. Each run folder held about 4.8 MB after its first 5 to 6 hours, most of it the write ahead log, which SQLite reuses rather than letting it grow, and `journal_size_limit` trims it back to 64 MB if a burst ever makes it larger. Candles are pruned after 14 days and only 3 checkpoints are kept, while decisions, trades, fills, deaths, lineage, forecasts and bots are kept for good.
* **Logs** stay small. The rotated log file keeps 14 days, and each `stdout.log` grows by about 0.3 MB a day and is never rotated.

```bash
du -sh data/*
df -h .
```

The schema and its retention rules are in [section 12 of the plan](../PLAN.md#12-storage-schema).

## Kill and pause drills

Phase 10 runs two drills on the live system, to show that a hard crash and a frozen process both recover. Each adds a gap to the run it is done on, with the costs listed above, so each is done once, on purpose, and noted in the results. Set `RUN` to the chosen run first, run C in this example.

```bash
RUN=run_c

# kill drill, then restart after a chosen pause
kill -9 "$(pgrep -f "configs/$RUN.toml run")"
nohup .venv311/bin/python -m btc_survivor --config configs/$RUN.toml run >> data/$RUN/stdout.log 2>&1 &

# pause drill, longer than stale_after_seconds
PID=$(pgrep -f "configs/$RUN.toml run")
kill -STOP "$PID"
sleep 30
kill -CONT "$PID"
```

After the kill drill the log shows a `resume` event with `restored the checkpoint from`, and when the process was down for more than 10 minutes a second `resume` event at the first healthy minute with `restart_flatten`. After the pause drill the run finds either a silent socket, which goes stale and reconnects, or a backlog of old messages, whose timestamps trip `clock_skew`. Either way the feed passes through recovering and turns healthy after a full clean minute, and a pause longer than 10 minutes also flattens with `gap_flatten`. Both drills check the same three things.

* The `resume` events are present and say what was restored or flattened.
* The rent column of `status` grows by 1.16 dollars a day only for healthy minutes, so the downtime adds nothing.
* `grep -c 'invariant violation' data/$RUN/stdout.log` prints 0.

A true network cut affects the whole Mac and every run at once, so it is done only with explicit approval. The first night's sleep already took all four runs through the outage path. The drills are listed in [section 18 of the plan](../PLAN.md#18-build-phases).

## Python and TLS

* `.venv311` is built on pyenv Python 3.11.16, and all four live runs use it. It connects to Coinbase with no extra step.
* `.venv` is built on the python.org Python 3.13.1 installer. That build carries its own OpenSSL without any trusted certificates, so the websocket fails with `CERTIFICATE_VERIFY_FAILED` until the installer's certificate script has run once.
* The tests need no network, so both environments run the whole suite, with `.venv/bin/python -m pytest -q` and `.venv311/bin/python -m pytest -q`.
* numpy is pinned per Python version, 2.4.6 on 3.11 and 2.5.3 on 3.12 and newer, because numpy 2.5 dropped Python 3.11. That is one more reason to keep each run on the environment it started with.

The certificate script for the python.org build runs like this.

```bash
open "/Applications/Python 3.13/Install Certificates.command"
```

The pins are in [section 19 of the plan](../PLAN.md#19-packaging-and-platform), and the Pi uses its system Python as [raspberry_pi.md](raspberry_pi.md) explains.

## Keeping a Mac awake

A sleeping Mac freezes every run, and a long sleep costs hours of valid features, as shown above. On the first night the MacBook slept with its lid closed on battery even though Amphetamine was running, and every run lost 99 minutes, which pushed run A's first valid features from about `02:40` to about `05:50` UTC.

* Keep the Mac on power. With the lid closed on battery it sleeps even with Amphetamine running.
* The simplest safe setup is on power with the lid open.
* On power with the lid open, `caffeinate -s` holds a sleep assertion, and `-w` ties it to a process, so the assertion ends when the run does.
* On a MacBook, System Settings, Battery, Options has a switch that prevents automatic sleeping on power adapter when the display is off.
* If stale periods and DNS failures appear while the Mac is awake, the network is the next suspect.

```bash
caffeinate -is -w "$(pgrep -f 'configs/run_a.toml run')" &
pmset -g assertions
pmset -g log | grep -E ' (Sleep|Wake)  ' | tail -n 20
```

The first `pmset` command shows what is holding the Mac awake, and the second shows when it last slept and woke. The two spaces in its pattern keep only the sleep and wake rows, and its times are local time, not UTC like the run logs. For a machine that never sleeps at all, the runs can move to a Raspberry Pi as [raspberry_pi.md](raspberry_pi.md) describes.
