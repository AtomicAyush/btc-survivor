# Operations

This guide covers running BTC Survivor day to day on the machine that hosts the six live runs, the MacBook until 12 October 2026 and the owner's home PC since, and reading what the runs report. Commands assume the project folder as the working directory and the `.venv311` environment. On a Raspberry Pi each run is a systemd service instead, as [raspberry_pi.md](raspberry_pi.md) describes, and the runtime design is in [section 14 of the plan](../PLAN.md#14-runtime-and-command-line).

## Where everything lives

* `configs/run_a.toml` to `configs/run_e.toml` and `configs/run_d2.toml` hold the settings of each run. Run E's config is hypothetical and never a realistic cost setting, as its header says. A relative `data_dir` is resolved from the config's own folder, so `../data/run_a` means `data/run_a` in the project.
* `data/run_a/btc_survivor.db` is the run's database, with its `-wal` and `-shm` files beside it. The three belong together, so never delete or move one of them alone.
* `data/run_a/logs/btc_survivor.log` is the log file, rotated at UTC midnight with 14 days kept.
* `data/run_a/stdout.log` is everything the process prints. It holds the same lines as the log file and is never rotated.
* `data/dashboard.log` is the dashboard's output.
* `data/raw` holds raw feed recordings, and `data/record_feed.log` holds the recorder's minute lines.
* `data/soak` holds one folder per soak run, as the soak section below explains.
* `data/drill` holds the separate drill run, with its own config, database and `stdout.log`.

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

Run E was added on 12 October 2026 by the owner's decision and starts the same way. It is run B with no fee, slippage, short charges or margin, so it is hypothetical, and its bots trade on their own predictions, as [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) explains. `init-db` creates `data/run_e` and its empty database first. The dashboard reads its list of runs only when it starts, so it is stopped and started again with all five configs, which costs the runs nothing because it only reads. Stop it by the process number that `pgrep` prints, which was `11451` on 12 October 2026, rather than with `pkill -f`, because a pattern can also match another shell whose command line holds the same words, and on Linux `pkill` does not even skip the shell that runs it. Wait until `pgrep -fl 'btc_survivor dashboard'` lists nothing before starting it again, so the port is free.

```bash
.venv311/bin/python -m btc_survivor --config configs/run_e.toml init-db
nohup .venv311/bin/python -m btc_survivor --config configs/run_e.toml run > data/run_e/stdout.log 2>&1 &
pgrep -fl 'btc_survivor dashboard'
kill -TERM 11451
pgrep -fl 'btc_survivor dashboard'
nohup .venv311/bin/python -m btc_survivor dashboard configs/run_a.toml configs/run_b.toml configs/run_c.toml configs/run_d.toml configs/run_e.toml --port 8050 >> data/dashboard.log 2>&1 &
```

Run E follows the same timeline as runs A to C. Features need 6 hours of healthy history and the label scale about another hour, then 6 hours of warmup follow, so its learners turn active about 13 hours after launch. Its gate then passes almost every prediction, so its first trades on its own calls come within half an hour of that, each copied by its coin flip twin, and forced trades become rare.

The dashboard's help panel was written for runs A to D. It names neither run E nor run D2, and its notes that most trades are forced every 12 hours and that trading will not beat cash at retail fees do not describe run E's tab. The file is left unchanged for now, because the running dashboard reads `dashboard.html` on every request, so any edit would change the open page at once, as the rules below explain.

Run D2 was added on 13 October 2026 by the owner's decision. It is run D with `stale_after_seconds` at 30 instead of 5 and `max_clock_skew_seconds` at 5 instead of 2, so delay trips and brief pauses in Coinbase's stream no longer end its feed epochs, within the limits [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) explains. On 12 October 2026 the five runs moved to the owner's always on home PC, an Ubuntu machine on Ethernet with the same Python 3.11.16 and numpy 2.4.6, one run at a time, each with a gap of about 2 minutes. There every run is a user systemd unit in `~/.config/systemd/user`, made from a template adapted from `deploy/btc_survivor@.service` and kept outside the repository. Unlike the repository unit, its instance name is the run's letter and it runs `configs/run_%i.toml`, so `btc_survivor@d2` runs `configs/run_d2.toml` with its data in `data/run_d2`. It asserts that the run's database exists, so `init-db` comes first. The dashboard unit lists its configs in its own `ExecStart` line, so `configs/run_d2.toml` is added there before the reload and the dashboard restarted, which costs the runs nothing because it only reads.

```bash
.venv311/bin/python -m btc_survivor --config configs/run_d2.toml init-db
systemctl --user enable --now btc_survivor@d2.service
sed -i 's#configs/run_d.toml configs/run_e.toml#configs/run_d.toml configs/run_d2.toml configs/run_e.toml#' ~/.config/systemd/user/btc_survivor_dashboard.service
systemctl --user daemon-reload
systemctl --user restart btc_survivor_dashboard.service
```

Run D2 follows run D's timeline from its own launch, about 10.5 hours of healthy data before training and about 16.5 before its learners turn active, later whenever a disconnect, a resync or a long stall ends its epoch.

A few rules keep restarts safe.

* Restart a run with `>>` in place of `>`, because `>` empties `stdout.log`. The rotated log file keeps its own 14 days either way.
* Never start a second process on the same config. Nothing in the code prevents it, and two writers would mix two histories in one database. Check with `pgrep -fl btc_survivor` first.
* Never edit a run's config, even between restarts. A restart refuses a changed master seed, population size or twin setting, but any other edit would pass unnoticed and break the rule that settings are fixed before launch.
* Before any restart run `.venv311/bin/python scripts/check_config_integrity.py configs/run_*.toml` and restart only on match, same settings or logging only. A setting that differs means a config file no longer holds the settings its run launched with, so put the file back first, and record the check in the results. A newly launched run's database holds no stored config until its first checkpoint, within about five minutes of its start, so until then the glob reports that run as missing and the script exits 2. The same happens on a machine without that run's data folder, such as the Mac for run D2. In either case name the other configs instead.
* Restart a run in the environment it started with, `.venv311` for all six, so its Python and numpy stay the same.
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

`status` starts with a header that holds the feed state, the live BTC price and its age, the launch time and the healthy days so far. A stage line follows, which moves from collecting history to waiting for outcomes to mature, then to training while learners finish warmup, and finally to trading once every learner is active. The table has one row per living learner, and `--twins` adds the twins. Below it come buy and hold, the exposure matched benchmark, the cash baseline, the twin mean and one summary line. That line compares the total equity of every learner ever born, dead ones included, with the capital they were given, so losses stay visible after replacements arrive. It leaves out writeoffs, because a dead learner's final equity is recorded after its negative cash is cleared, so the self financing value from `scripts/population_value.py` is the figure to set against buy and hold. `--watch` redraws the whole view every minute.

Two warnings matter. One says the system is not running, when `runtime_status` is missing or more than two minutes old. The other says the feed is not healthy, so bots are paused. Bot rows come from the latest checkpoint and can be up to five minutes old, while the price and the feed state are at most a minute old.

The dashboard at [http://127.0.0.1:8050](http://127.0.0.1:8050) shows the same numbers from the same read only queries for all six runs, with the equity chart, the skill so far and the latest decisions, fills and feed events. A run whose panel shows an error could not be read, and `data/dashboard.log` holds the reason. Days in `status`, the dashboard and `evaluate` are days of healthy data, 1,440 healthy minutes each, not calendar days. The page is described in the [README](../README.md#the-local-dashboard).

The Bots header names the checkpoint its rows come from, such as `as of the checkpoint at 17:10:30 UTC, 3 min ago`. A run commits its decisions, fills, trades, deaths and births only with each checkpoint, in one transaction, so no committed row is newer than the newest checkpoint and the table cannot be fresher than it. Only the price, the feed state and the living count in `runtime_status`, and the feed events, are newer. The UTC and Local switch at the top right shows every time on the page, times quoted inside feed events included, in UTC or in this Mac's own time zone with its abbreviation, and the browser remembers it. Run logs, `status` and the database stay in UTC whatever the switch says, so compare them with the page in UTC. In Local the explorer's from and to fields take local times, and the page sends the server the same moment in UTC, so a date alone in to means the end of that local day. A time ending in Z is read as UTC in either zone, so a time copied from a log or from `status` can be pasted as it is. A time in the hour that repeats when the clocks go back is shown in UTC with a Z, so it is never read as the other one. The persistence rules are in [section 6.5 of the plan](../PLAN.md#65-persistence-points).

Open the page as `http://127.0.0.1:8050` or `http://localhost:8050`. The server answers 403 to any request whose Host header names something else, such as this Mac's network name or another port, which stops a web page elsewhere from reaching it through DNS rebinding. That holds for every method, and every answer, its error pages included, carries the same security headers. A change to `dashboard.html` reaches open pages within seconds, while a change to `dashboard.py` takes effect only after the dashboard restarts, which never touches a run. The check is written in [`dashboard.py`](../src/btc_survivor/dashboard.py).

The Decisions section at the bottom of the page shows how a run is going between `status` checks. Choose learners and the last day to see what share of the matured predictions pointed the right way, where a coin gets about half, then choose twins to see the coin itself, scored by the side it chose rather than the prediction a twin copies. Filter on forced to see only the trades the timer made, which twins never show even when they copy one. Choosing a row opens the order, the fills and the trade behind that decision, and its twin's copy is one click away. A label reads waiting for one horizon after its decision, and never when a feed gap or a restart stopped it maturing, and the skill score leaves those minutes out too. The explorer only reads, like the rest of the page, and its filters are listed in the [README](../README.md#the-local-dashboard).

Choose skipped in the action filter, or its chip, to see every turn a learner skipped. A skipped turn is a phase minute, the one minute per horizon when a learner may decide, that a living learner passed without a decision row, because it decides only in a healthy minute where its prediction exists. Each row gives the most specific cause the database shows, and the reason filter searches it. The run was not running means no minute was recorded then, so the run was stopped, or it was killed and lost the minutes after its last checkpoint, and the reason names the restart that ended it and any resume event it logged, such as `resume 99.0 minutes since the last healthy minute`. The feed was not healthy means the candle of that minute is marked invalid, and the reason names the feed event that took the feed out of health, its time and how long until it was healthy again. The features were incomplete means the minute was healthy but no forecast row was written, and the reason names each feature window that failed. This is the subtle one, since a single unhealthy minute spoils every return window that starts in it, so a clock skew trip at `19:44:20` in run D made the 15 minute return at `20:00` impossible and cost bot 1 a whole 4 hour turn. Anything else reads no decision recorded, cause not in the database, which should not happen and is worth reporting. Once candles are pruned after 14 days, the cause comes from benchmark rows, which still name every failed window, and from feed events, and it says the minute records were pruned. Twins never skip a turn, because a twin only copies its learner's opens, closes and flips. Minutes after the newest checkpoint are not judged yet, since their rows are still buffered in the run. Choosing a skipped turn shows the feed events nearest it, the position the learner held, what its twin held and the next decision. The learner keeps that position until its next decision unless a stop loss, a death close or the flatten after a long gap or a restart closes it first, so after a stop like the one above the restart flatten closes it, and the panel shows when and why. The reason filter searches the times inside a reason as stored, in UTC. Skipped turns count toward no matured, right or waiting figure, and the summary line counts them apart. Many marked not healthy point at Coinbase stalls or clock skew trips, which run D2's wider limits are meant to ride out, as [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) explains, and the rule is written in [`dashboard.py`](../src/btc_survivor/dashboard.py).

## Reading feed events

Every change in feed health is written to the `feed_events` table and to the log.

```bash
grep 'feed event' data/run_a/logs/btc_survivor.log | tail -n 20
sqlite3 -readonly data/run_a/btc_survivor.db "SELECT ts, kind, detail FROM feed_events ORDER BY id DESC LIMIT 20"
```

The feed is always in one of three states.

* **Healthy.** Messages arrive, the local book is valid, and local time agrees with the exchange's timestamps. Bots predict, learn, pay rent and trade in healthy minutes and in no others.
* **Recovering.** Something broke and is being repaired, after a reconnect, a lost book or clock skew. The feed turns healthy at the first minute boundary that closes a full minute in which everything held, logged as `healthy a full clean minute`, so a short break usually costs one or two minutes.
* **Stale.** No message, or no good book update, arrived for more than `stale_after_seconds`, which is 5, or 30 in run D2. The detail says `no messages` or `no good book`, and the connection is closed and opened again.

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

Every live message carries Coinbase's own timestamp. The health monitor keeps the difference between receive time and that timestamp for the last 50 messages, and once it holds at least 10 it compares their median with `max_clock_skew_seconds`, which is 2 in every run but run D2, where it is 5. Above that it logs `clock_skew` with the median and moves the feed to recovering without reconnecting, because a reconnect cannot fix a clock. When the median falls back it logs `clock_skew cleared`, and the usual full clean minute follows.

The name says clock, but the number is the clock error plus the delivery delay. A wrong local clock raises it, and so do a sleeping or waking Mac, slow delivery from Wi Fi, the network path or Coinbase itself, and a busy machine. Every trip on 11 October came after the lid had closed, with medians of 2.0 to 3.0 seconds. The power log shows a clamshell sleep at `23:26:19` UTC, and 27 seconds later all four runs and the recorder logged `clock_skew` within the same second, as messages that had queued while the Mac slept arrived during its short dark wakes. More trips followed over the next half hour while it went on sleeping and waking. The recorder's earlier trip, at `19:38` UTC before the runs started, came as the Mac woke from a clamshell sleep that began at `19:35:36` UTC. The closed lid caused both the trips and the outage that followed, and neither caused the other. The only trip so far while the Mac was awake came at `02:07` UTC on 12 October, when run D logged three `clock_skew` events with medians of 2.0 to 2.6 seconds and run C went stale with no messages in the same minute, while runs A and B stayed healthy. The Mac's clock was checked against `time.apple.com` that night and agreed to within about 25 ms, the uncertainty of the measurement, so none of these events was a wrong clock. On the home PC, whose clock chrony held within a millisecond of NTP, three trips came between `00:57` and `05:46` UTC on 13 October with medians of 2.06 to 2.34 seconds, so they too were Coinbase's delivery and not the clock. This command prints the offset on the Mac and changes nothing.

```bash
sntp time.apple.com
```

A trip out of healthy is a real gap with the same costs as any other, listed in the next section, so in run D each one pauses learning for 4 hours. Run D2 allows a median of 5 seconds, so trips like these do not end its epochs as long as the median stays under 5. The limit is fixed for the length of each run, like every other setting. The guard is described in [section 7 of the plan](../PLAN.md#7-feed).

## Sleep, restarts and outages

When the Mac sleeps, every process freezes. On wake the one second ticker catches up, every missed minute is judged unhealthy, the silent socket goes stale and closes, and the feed reconnects through recovering to healthy. A restart, a lost network and a clock skew trip all follow the same rules.

* **Rent stops.** Rent accrues only in healthy minutes, so no bot pays for downtime. By `02:33` UTC on 12 October each learner in run A had paid about 0.20 dollars of rent, which matches the roughly 4.2 healthy hours since launch and not the 5.9 hours on the clock.
* **Shorts keep paying funding.** Funding accrues through stale minutes and downtime, because a real short would still be open.
* **Long gaps flatten.** At the first healthy minute after a gap longer than `max_resume_gap_minutes`, which is 10, every open position is closed through the normal broker. The exit reason is `restart_flatten` after a restart and `gap_flatten` after an outage while the process stayed up, and a `resume` event records the length of the gap and how many closing orders went out. A rejected flatten is sent again at the next healthy minute.
* **Pending outcomes are dropped.** Every exit from healthy starts a new feed epoch, and no label crosses an epoch, so every pending prediction is discarded. No outcome matures for one horizon afterwards, so learning pauses for 30 minutes in runs A, B, C and E and for 4 hours in runs D and D2, and warmup, which counts training minutes, takes that much longer.
* **Features need clean windows.** A return needs a healthy minute exactly at its start, and each volatility and volume window needs 80 percent of its minutes healthy. A gap of 1 to 3 minutes costs four times its length, the later minutes whose return windows start inside it. A gap of 4 or 5 minutes costs about 25 minutes, a gap of 6 to 23 minutes about 35 to 55 minutes, one longer than 23 minutes about 1.6 hours, and one longer than 72 minutes about 4.8 hours of healthy data, while the 30 minute volatility, the 120 minute volatility and the 360 minute volume baseline refill. In that time bots neither predict nor learn nor decide, while rent, marking and stop checks go on.
* **Run D's forced trades slip a whole window.** The forced timer counts healthy minutes since a bot's last open, and a run D bot decides only in its own minute of each 4 hour block, so 12 healthy hours is exactly three blocks. One unhealthy minute in between leaves the timer short at the third block, and the forced trade waits for the fourth, 16 hours after the last open. With trips every few hours that is the usual case, so run D's bots trade about every 16 hours rather than 12. That lowers a quiet run D bot's burn from about 2.36 to about 2.06 dollars a day and changes no outcome, because none dies either way. Run D2 has the same timer, and fewer trips may bring it nearer 12, though the home PC also saw 2 disconnects in its first 16 hours, and each ends run D2's epoch too.
* **A restart is not a death.** Bots keep their weights, genes, accounts and positions, as the next section explains.

The first night shows the whole sequence. Each run's last healthy minute before the outage was at `23:26` UTC. From `23:27` UTC, seconds after the lid closed at `23:26` UTC, run A's log shows stale connections and reconnects whose waits grew from 0.9 seconds at `23:27:03` to 58.6 seconds at `23:33:59`, and the first DNS failure, a `gaierror`, came at `23:37:39`. At `01:05` UTC every run logged `resume 99.0 minutes since the last healthy minute, gap_flatten sent 0 closing orders`. No bot was trading yet, so nothing needed closing.

For a run that has finished collecting history and is learning, a sleep that long costs the gap itself, about 4.8 healthy hours of invalid features and one more horizon before learning resumes. The first night cost less, because the runs were still collecting their first 360 healthy minutes of history. No label had matured and no learning had begun, so no learning was lost. In run A the sleep pushed the first valid feature vector from about `02:40` to about `05:50` UTC, which is the 99 minute gap plus about 1.5 healthy hours, because the 287 minute refill after the gap overlapped the 196 healthy minutes of history that were still to come. These rules are specified in [section 13 of the plan](../PLAN.md#13-checkpoints-resume-and-shutdown).

## Checkpoints and resume

Every 5 minutes, 30 seconds after a minute boundary, a run writes every buffered row and one checkpoint in a single transaction. The checkpoint holds each bot's weights, optimizer state, genes, account, open position, pending order, predictions, skill buffers and random generator state, plus the shared standardizer, label scale, matured outcomes, feed epoch and counters. It is decoded once in memory before it is committed, the newest 3 are kept, and candles older than 14 days are pruned in the same transaction. Feed events and `runtime_status` are committed every minute and never feed back into bot state.

On start, `run` loads the newest checkpoint that decodes and logs a `resume` event that says `restored the checkpoint from` and its time. Orders that were in flight are rejected with reason `restart`, while stop and death closes are tried again. If the gap turns out longer than 10 minutes, the first healthy minute flattens with `restart_flatten`. If only an older checkpoint decodes, every row written after it is rolled back first and the reason is recorded. A run refuses to resume when the master seed, the population size or the twin setting differs from its database, and it refuses to seed a new population over a database that holds bots but no readable checkpoint. A `kill -9` or a power cut therefore loses at most five minutes of history, and the books and the bot state always agree. Checkpoints are specified in [section 13 of the plan](../PLAN.md#13-checkpoints-resume-and-shutdown).

## The recorder

Runs A to E and D2 record no raw messages, because their configs set `record_raw_messages = false`. A separate development recorder has run since `16:58` UTC on 11 October, reading `config.toml` and writing into `data/raw`.

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

## The soak test

`scripts/soak_test.py` replays every recording in `data/raw`, joined end to end, and then 120 hours of synthetic data through the real engine. It uses `configs/soak_plumbing.toml`, which is plumbing only and never a strategy setting. That file differs from the example only in its 4 dollar start equity and notional, its 1 hour forced timer, its parent requirements of 2 hours and 2 trades and its data folder, so deaths, mutated births and new generations happen within a few replayed days. The synthetic part resyncs about every two hours with a full size book snapshot of about 4.5 MB, and after 24 synthetic hours it drops the connection for 15 minutes, longer than `max_resume_gap_minutes`, so the gap flatten closes real positions. Replay results are never used to choose, tune, or rank strategies.

```bash
nice -n 19 .venv311/bin/python scripts/soak_test.py
nice -n 19 .venv/bin/python scripts/soak_test.py --out data/soak/python313_run
```

* The soak only reads `data/raw`. The newest recording, which the recorder may still be writing, is first copied up to its current size into the output folder and read to its last complete line.
* Each run writes into a new folder under `data/soak`, named after its UTC start time unless `--out` names another new or empty folder. `report.json` holds the verdict and every number, `memory.csv` has one row per replayed hour, `soak.log` is the engine's log, and `db` is the replay database, which `explain` and `lineage` can read through a copy of the plumbing config whose `data_dir` points there.
* The script reads the recordings and generates the synthetic data itself and pipes every line to a child process that holds only the engine, so the child's memory is what a live run holds. At the end of every replayed hour the child waits while the script reads its resident memory from outside.
* The exit code is 0 when the soak passed, 1 when it failed and 3 when it was inconclusive.

The plan's memory rule is that memory at the end is no more than 5 MB above memory at the end of the first replayed hour. On Linux, the system of the Raspberry Pi, resident memory read from `/proc` decides it, because that is the number the live heartbeat shows there and the Pi's memory is tight. The Mac has 24 GB and the owner set no tight memory budget for it on 12 October 2026, so on a Mac the rule is judged on memory in use, the measure a leak moves, and resident memory is reported beside it. macOS keeps freed pages resident and gives them back in bursts, so resident memory on the Mac swings by tens of megabytes from hour to hour. Memory in use is what the C allocator has handed out plus CPython's small object blocks, read inside the child, and without that reading the verdict is `inconclusive`. `report.json` also gives the largest growth among all later readings, how many of them were over the limit, the slope over the last replayed day, and on macOS the physical footprint. Memory in use that stays flat while resident memory moves means the allocators are holding freed memory, not that anything leaks.

For CPU, the figure to compare with the 25 percent target is `engine_share_of_one_core_at_live_speed`, the engine's CPU time over the replayed time. The report also splits engine time into minute steps, database writes, full size snapshots and all other messages. A synthetic book update changes about 98 price levels where a real one changes about 10, so a synthetic message costs about three times a real one. None of these figures include the websocket and TLS work of a live process, which `ps` shows at about 1.3 to 1.8 percent of one core for each run on the Mac.

The first full runs on 12 October replayed 132.2 hours, 12.2 of them real, with 3.4 million messages and 86 full size snapshots, once under Python 3.11.16 with numpy 2.4.6 and once under Python 3.13.1 with numpy 2.5.3.

| Measure | Python 3.11 | Python 3.13 |
| --- | --- | --- |
| resident memory after the first hour and at the end | 90.1 and 106.7 MB | 94.2 and 99.6 MB |
| physical footprint after the first hour and at the end | 76.4 and 69.8 MB | 78.6 and 62.5 MB |
| memory in use after the first hour and at the end | 32.3 and 33.2 MB | 33.9 and 36.5 MB |
| largest resident growth over the first hour | 37.1 MB | 36.2 MB |
| peak resident memory | 127.2 MB | 130.4 MB |
| engine share of one core at live speed | 0.043 percent | 0.035 percent |
| one healthy minute step | 1.29 ms | 1.22 ms |
| one full size snapshot | 22.5 ms | 18.8 ms |

Both runs were first reported `inconclusive` under the earlier rule, and both pass the Mac rule, with memory in use 0.9 and 2.6 MB above the first hour. Memory in use stayed within 2.6 MB of the first hour with a slope near zero over the last replayed day, and CPython's small object arenas rose from 28 or 29 to at most 41 during the bursts of reconnects in the real night and never rose again through 61 synthetic resyncs. So nothing grows without bound, but freed memory stays mapped after a burst. CPython maps its arenas the same way on Linux, so the roughly 11 MB they added after the first hour is likely to count there too, and the Linux run may fail the rule even though memory in use is flat. That run decides it. Resident memory peaked during those real bursts. That peak is about 50 MB lower than in a first soak whose replay process also read the recordings, but still above the 89 to 91 MB peaks of the live runs, which is why section 20 of the plan names the Kraken adapter or a streaming parse as the fixes if the Pi shows too little headroom under 150 MB. The books held throughout, with 61,680 engine checks at equity snapshots, 5,320 hourly checks and 7 ledger checks on the database, all without a failure. The replay saw 160 deaths, 103 from fees and 57 from rent, 73 mutated births with a lineage row each, generation 4 at most, and one synthetic outage after which `gap_flatten` closed 20 open positions.

The Linux reading matters only for a move to the Pi, so it is taken there, or in a Linux arm64 container before a move. Docker Desktop is installed on the Mac but not running, and starting it beside the live runs needs the owner's approval. Once it runs, this command repeats the soak under Python 3.13 with the pinned numpy, with the project mounted read only and only `data/soak` writable.

```bash
docker run --rm --platform linux/arm64 --cpus 2 --memory 2g \
  -v "$PWD":/repo:ro -v "$PWD/data/soak":/repo/data/soak \
  -e PYTHONPATH=/repo/src -e PYTHONDONTWRITEBYTECODE=1 -w /repo \
  python:3.13-slim-trixie \
  sh -c 'pip install --quiet --root-user-action=ignore numpy==2.5.3 websockets==17.2 && python scripts/soak_test.py'
```

Its memory reading comes from `/proc`, so its verdict is `passed` or `failed`. The 24 hour run on the Pi, with the checklist in [raspberry_pi.md](raspberry_pi.md#checklist-for-the-first-24-hours-on-the-pi), has the final word. The soak is specified in [section 18 of the plan](../PLAN.md#18-build-phases).

## Kill and pause drills

Phase 10 ran its drills on a separate drill run, never on runs A to D. A drill run has its own process, config and data folder, so killing or pausing it costs the live runs nothing. Its config is a copy of a run's config with only `data_dir` changed, and that path must be absolute, because a relative `data_dir` is resolved from the config's own folder, which here is `data/drill`. The `sh` wrapper writes the exit code into `stdout.log` when the process ends, so even a detached run records how it stopped.

```bash
mkdir -p data/drill
sed "s|^data_dir = .*|data_dir = \"$PWD/data/drill/db\"|" configs/run_a.toml > data/drill/drill.toml
.venv311/bin/python -m btc_survivor --config data/drill/drill.toml init-db
nohup sh -c 'nice -n 19 .venv311/bin/python -m btc_survivor --config data/drill/drill.toml run; echo "exit=$? at $(date -u +%FT%TZ)"' >> data/drill/stdout.log 2>&1 &
PID=$(pgrep -f '^.venv311/bin/python -m btc_survivor --config data/drill/drill.toml run')

# kill drill, then start it again with the same nohup line after a short or a long pause
kill -9 "$PID"

# pause drill, longer than stale_after_seconds
kill -STOP "$PID"; sleep 35; kill -CONT "$PID"

# clean stop, after which the wrapper logs exit=0
kill -TERM "$PID"
```

Each drill checks three things. The `resume` events are present and say what was restored or flattened. Rent grows by 1.16 dollars a day only for healthy minutes, which the checkpoints and the benchmark rows show, because each healthy minute adds one benchmark row and its cash baseline falls by one minute of rent. And `grep -c 'invariant violation' data/drill/stdout.log` prints 0.

The drills on 12 October used a drill run with run A's settings and a fresh database.

* A `kill -9` with 61 seconds down. The restart logged `resume restored the checkpoint from` with the checkpoint of `04:00:30`, the feed was healthy again at `04:04`, and the 4 minute gap flattened nothing. The healthy minute at `04:01` was rolled back with the rows that had not been committed, so 2 healthy minutes were charged rent over 5 minutes on the clock.
* A 35 second SIGSTOP. The feed logged `stale no messages`, reconnected, took a new snapshot and was healthy again at `04:10`, and the stale and recovering minutes were charged no rent.
* A restart after 45 minutes stopped, and a `kill -9` that kept the run down for 16 minutes, apart from one start that was killed again before it turned healthy. Each logged a second `resume` event at its first healthy minute, 48.0 and 19.0 minutes after the last one, with `restart_flatten sent 0 closing orders`. Between the checkpoints of `05:05:30` and `05:25:30` only the healthy minutes `05:24` and `05:25` were charged rent.
* A SIGTERM. The log ended with `stopped cleanly, last checkpoint`, the write ahead log was left empty, and the wrapper logged `exit=0`, where each `kill -9` had logged `exit=137`.
* Every checkpoint and equity snapshot balanced, and no `invariant_violation` event appeared. Runs A to D logged a healthy heartbeat every minute throughout.

No drill bot held a position, so no restart has yet flattened one live. The soak's synthetic outage and the tests cover that path. Once the live bots trade, the kill drill is repeated on a drill run resumed from a read only copy of a live database and started more than 10 minutes after the copy's last checkpoint, so `restart_flatten` closes real positions while the live run itself is never touched.

```bash
mkdir -p data/drill_c/db
sqlite3 -readonly data/run_c/btc_survivor.db ".backup data/drill_c/db/btc_survivor.db"
sed "s|^data_dir = .*|data_dir = \"$PWD/data/drill_c/db\"|" configs/run_c.toml > data/drill_c/drill.toml
```

A true network cut affects the whole Mac and every run at once, so it is done only with explicit approval, and on the Pi the checklist pulls the cable instead. The first night's sleep already took all four runs through the outage path. The drills are listed in [section 18 of the plan](../PLAN.md#18-build-phases).

## Python and TLS

* `.venv311` is built on pyenv Python 3.11.16, and all six live runs use it. It connects to Coinbase with no extra step.
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
