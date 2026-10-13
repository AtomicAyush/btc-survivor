# Operations

This guide covers running version 2 day to day on the MacBook that also hosts version 1's runs A to D, from the checks before anything starts to the launch, the window and the later move to a Raspberry Pi. Commands assume the worktree `~/code/btc_survivor_v2` as the working directory and its `.venv-v2` environment. Version 1 lives in `~/code/btc_survivor` and is never touched from here, and neither is its dashboard on port 8050. The runtime design is in [section 14 of the plan](../PLAN_V2.md#14-runtime-and-processes), and the overview is [v2_overview.md](v2_overview.md).

## Where everything lives

* `configs/v2/news.toml`, `s1_prices.toml` and `s2_news.toml` hold the settings, fixed before launch, and `rehearsal_s1.toml` and `rehearsal_s2.toml` the plumbing settings of the rehearsal. A relative path resolves from the config's own folder.
* `data/v2/news/news.db` is the news database, `data/v2/s1_prices/stocks.db` and `data/v2/s2_news/stocks.db` the two run databases. Each has its `-wal` and `-shm` files beside it, which belong together, so never move or delete one alone.
* `data/v2/raw_stocks` holds the raw recordings, gzip JSON lines of every price line and news line with their receive stamps. Files rotate at each start and at UTC midnight, and the stock process deletes those older than 30 days once a UTC day.
* `data/v2/checks` holds the hand filled official figures and the coverage records of the tape check, `data/v2/exports` the CSV exports, and `data/v2/replays`, `data/v2/soak` and `data/v2/rehearsal` the outputs of replays, soaks and the rehearsal.
* `data/models/finbert` holds the exported FinBERT model, tokenizer and manifest.
* `data/logs/v2/news`, `data/logs/v2/stocks` and `data/logs/v2/record_stocks` hold each process's `btc_survivor.log`, rotated at UTC midnight with 14 days kept, and `data/logs/v2/launchd` the output of the launchd agents once installed. Logs hold states, counts and timings, never a price.
* `~/.config/btc_survivor` holds `secrets.env`, the key ledger `llm_ledger.db`, and the lock files `finnhub_ws.lock`, `finnhub_rest.lock` and `llm_ledger.lock`, which the main tree and the worktree share. A lock is released when its process exits, so never delete a lock file.

Nothing under `data/` is tracked by git, and every version 2 process that writes sets umask 077, so its files and folders are the owner's alone. A consistent copy of a live database can be taken without stopping anything.

```bash
sqlite3 -readonly data/v2/s2_news/stocks.db ".backup $HOME/s2_copy.db"
```

The tables are described in [section 9 of the plan](../PLAN_V2.md#9-storage).

## Environments

* `.venv-v2` is built on Python 3.11 with the `test` and `v2news` extras, which pin `anthropic`, `pydantic`, `onnxruntime` and `tokenizers`. Every live version 2 process runs from it.
* `.venv-v2-313` is built on Python 3.13 with the same extras, for running the suite a second time.
* Both are editable installs of the worktree, so a process loads the source as it stands when it starts. Restart a process only from a committed tree, because the stock process records the commit at every start, and a commit marked dirty cannot be reproduced.
* `torch` never enters either environment. The FinBERT export runs in a throwaway environment of its own.

```bash
.venv-v2/bin/python -m pytest -q
.venv-v2-313/bin/python -m pytest -q
```

No test needs the network, and the version 1 suite runs in both. The pins are in [section 3 of the plan](../PLAN_V2.md#3-scope-isolation-from-version-1-and-module-layout).

## Checks before anything starts

```bash
.venv-v2/bin/btc_survivor_v2 check-secrets
.venv-v2/bin/btc_survivor_v2 check-local
.venv-v2/bin/python scripts/v2_secret_scan.py --logs data/logs/v2
```

* `check-secrets` names the keys present and missing in `~/.config/btc_survivor/secrets.env` and never a value. It fails when the file is readable by anyone but its owner or a key is missing. The file holds `FINNHUB_API_KEY`, `ANTHROPIC_API_KEY` and `CONTACT_EMAIL`, one `KEY=value` line each, and is created by the owner with mode 600.
* `check-local` runs the local only check described below and exits with code 1 when it fails.
* The secret scan is described below.

Before the first use of a key the version 1 suite and `test_v1_untouched` pass, and runs A to D show unchanged heartbeats. The checks are listed in [section 17 of the plan](../PLAN_V2.md#17-first-live-session-checks), and the secrets in [v2_data_and_licensing.md](v2_data_and_licensing.md#secrets).

## The raw recorder

Until launch, `scripts/v2_record_stocks.py` keeps the one Finnhub websocket open around the clock and writes every raw message to `data/v2/raw_stocks` for the coverage report, the health thresholds, the soak and replay. It never trades, never writes a database and never logs a price. It takes `~/.config/btc_survivor/finnhub_ws.lock`, which the stock process also takes, so the two can never run together, and a second one exits with an error naming the holder's process id.

```bash
.venv-v2/bin/python scripts/v2_record_stocks.py configs/v2/s1_prices.toml --dry-run
(umask 077 && mkdir -p data/logs/v2/record_stocks)
nohup .venv-v2/bin/python scripts/v2_record_stocks.py configs/v2/s1_prices.toml >> data/logs/v2/record_stocks/stdout.log 2>&1 &
pkill -TERM -f 'scripts/v2_record_stocks.py'
```

The dry run loads the config and the secrets, takes and releases the lock and prints a verdict without connecting. A heartbeat line each minute holds counts, the connection state and the file size. SIGTERM stops it cleanly with code 0, writing a close marker. A process frozen for more than 10 seconds, by sleep or SIGSTOP, closes its connection with the reason `process paused for N s` as soon as it runs again, so the coverage report sees the pause as disconnected time. The recorder is described in [section 4.9 of the plan](../PLAN_V2.md#49-tape-coverage-check).

## The tape coverage report

The gate session is Monday 12 October 2026, the first full regular session the recorder covered, whether or not its recording turns out clean, because the recorder closed a silent connection after its own short feed limit and so may have left disconnected time in it. It showed that Finnhub's free websocket carries a partial tape, a small share of consolidated volume with long silences between one symbol's prints, and on 12 October 2026 the owner decided to keep it. So `tape-check` no longer judges whether the feed is consolidated. It reports how much of the tape the feed carries, and the run asks only that this coverage stays stable. The next morning the owner, or the assistant session on the owner's behalf, opens Nasdaq's public historical quote page for each symbol in an ordinary browser and copies the day's consolidated volume and official close into a copy of `configs/v2/official_example.toml` named by the New York session date. No script fetches that page.

```bash
cp configs/v2/official_example.toml data/v2/checks/official_2026_10_12.toml
.venv-v2/bin/btc_survivor_v2 tape-check data/v2/raw_stocks data/v2/checks/official_2026_10_12.toml
.venv-v2/bin/btc_survivor_v2 tape-check data/v2/raw_stocks data/v2/checks/official_2026_10_12.toml --detail
```

* It reads only the recording and the official file, needs no database, and refuses an official file that git would track.
* Per symbol the ratio r is the recorded volume over the official volume, and the session's coverage is the median r over the eight symbols. Each run writes `coverage_YYYY_MM_DD.json` beside the official file, readable by the owner alone.
* It refuses a recording that does not yet reach 10 minutes past `20:00` ET, so run it after that or after the recorder stopped. It exits with code 0 for a clean session, 1 for any other session and 2 when it writes no record.
* By default it prints the clean flag and its reason, counts, lost time, the delay verdict and the two delay settings, and `--detail` adds the ratios and their medians, volumes, covered time, delays, condition codes and the disconnected and outage intervals.

**Coverage stability, criterion 1.1.** The median r of the gate session is the baseline, clean or not, and `evaluate` takes it from the earliest full session with a coverage record before launch, so run `tape-check` on 12 October before any other session. The first clean full session at or after sessions 10, 20 and 40 of the window must keep its median r within a factor of 2 of the baseline, either way. `evaluate` reads the coverage records, so the check must also run on those sessions, or criterion 1.1 fails at the cutoff. A clean session at or after session 40 may come only after the cutoff, so keep the stock process recording and keep running the check after the cutoff until that record exists. The check counts a whole feed silence of at least `feed_stale_after_seconds` as outage time, so run it on 12 October only after the measured thresholds are in the configs, and a rerun replaces the record whole.

**The health thresholds.** The plan's 15 second limit for a symbol and 5 second limit for the whole feed assumed the full tape, where each name prints many times a second. On the partial tape a symbol can stay silent in session far longer than 15 seconds, and the whole feed longer than 5, while trading goes on normally, so on 12 October 2026 the owner wrote a rule, and the assistant session applies it to the full recording of the gate session that evening.

* `symbol_stale_after_seconds` is 2.5 times the longest gap in session between one symbol's eligible prints, the longest over the eight symbols, rounded up to a multiple of 30 seconds. A symbol gap runs across reconnects, as live health counts it, with the time the recording was disconnected left out, so a recorder silence close never splits a symbol's silence in two.
* `feed_stale_after_seconds` is 5 times the longest gap in session between any two messages of the whole feed, rounded up to a multiple of 5 seconds. A feed gap stays inside one connection, as live health restarts the feed clock at each connect.

```bash
.venv-v2/bin/python scripts/v2_gap_thresholds.py data/v2/raw_stocks 2026-10-12
.venv-v2/bin/python scripts/v2_gap_thresholds.py data/v2/raw_stocks 2026-10-12 --detail
```

The script prints both thresholds, the longest gaps behind them, the longest symbol gap inside one connection, the disconnects inside the longest symbol gap and whether each threshold fits its sanity bound, 300 seconds for a symbol and 60 for the feed, past which the owner decides, and never a price. `--detail` gives each longest gap's times, so a halt behind the symbol threshold can be read against Nasdaq's halt list first. Both values go into all four stock configs, `s1_prices.toml`, `s2_news.toml` and the rehearsal pair, which must agree in these settings, before the rehearsal, and they never change after launch. The cost of keeping the free feed is that a halted symbol counts as healthy, on its last prices, for longer before it goes stale. The recorder reads `feed_stale_after_seconds` from `s1_prices.toml` when it starts and closes a connection silent for that long, so restart it outside the session once the values are written, which also gives the later sessions before launch a chance to be clean.

After the first recorded session the delay rule sets `max_print_delay_seconds` to the larger of 2 and `ceil(1.5 * p95)` of the delays in seconds, capped at 5, and `late_print_seconds` to twice that, and a median delay above 5 seconds stops the build for the owner. Feed quality numbers stay in `data/v2/checks` and never go into the repository. The check is specified in [section 4.9 of the plan](../PLAN_V2.md#49-tape-coverage-check) and the health rules in [section 4.5](../PLAN_V2.md#45-price-health-without-an-order-book).

## The news process

```bash
.venv-v2/bin/btc_survivor_v2 init-db configs/v2/news.toml
.venv-v2/bin/btc_survivor_v2 news configs/v2/news.toml --dry-run
(umask 077 && mkdir -p data/logs/v2/news)
nohup .venv-v2/bin/btc_survivor_v2 news configs/v2/news.toml >> data/logs/v2/news/stdout.log 2>&1 &
pgrep -fl 'btc_survivor_v2 news'
pkill -TERM -f 'btc_survivor_v2 news configs'
```

* `init-db` creates `news.db` with `meta.kind` set to news, or checks an existing one, and prints only the verdict.
* The dry run loads the config and the secrets, takes and releases the Finnhub REST lock and prints a verdict without touching the network. Once the FinBERT pins are written it also checks the Anthropic key, the prompt and schema pins and the FinBERT files against their pins, and its last line says whether scoring is pinned.
* The process logs to `data/logs/v2/news/btc_survivor.log` and to stderr, so `stdout.log` holds the same lines plus any error printed before logging starts. Use `>>`, because `>` empties it.
* SIGTERM lets polls in flight finish for up to 20 seconds, writes a `stopped` event and exits with code 0. A model call still in flight is cancelled and booked lost at the next start, which is safe because its items are failed and FinBERT covers them.
* A held lock, a bad config, a missing Anthropic key or a FinBERT file that does not match its pins exits with code 2 and one `error` line, and a crash with code 1.

Once a minute the process logs a heartbeat line with `healthy`, the polls since start and how many failed, the counts of new, seed, backlog, filtered and duplicate items, the items waiting for a score and the channels in outage, plus a `reason` when news is unhealthy. Every 30 seconds it also writes a `news_heartbeat` row, which the stock process reads.

**Restarts.** A restart interrupts collection, and items from a channel whose gap exceeded its cadence by more than 600 seconds come back as `backlog`, which is never scored, except `8-K` filings. A restart changes nothing for the runs as long as `configs/v2/news.toml`, the prompt, the schema and the models stay the same. Any change to them after launch changes a hash the runs recorded, which fails news health for the rest of the window and fails criterion 1.6, so after launch the news config is never edited.

**Before the first scoring start.** The prompt and schema pins and the three FinBERT pins go into `configs/v2/news.toml`, the Sonnet setting is settled, and `scripts/llm_synthetic_check.py` runs its dry run and, with the owner's approval, its live check. The first scoring start gives every item collected so far an `expired` Haiku row while FinBERT works through the whole backlog, so news health reports a long wait until it catches up. That start must come well before the rehearsal. Until any FinBERT pin is written the process collects only and logs that scoring waits for the export, so a restart before the export never stops collection, and the first start after the pins are written begins scoring.

```bash
.venv-v2/bin/python scripts/llm_synthetic_check.py configs/v2/news.toml --dry-run
.venv-v2/bin/python scripts/llm_synthetic_check.py configs/v2/news.toml
.venv-v2/bin/python scripts/llm_synthetic_check.py configs/v2/news.toml --sonnet-thinking disabled
.venv-v2/bin/python scripts/llm_synthetic_check.py configs/v2/news.toml --sonnet-thinking between_tools
```

The plain live check runs Haiku alone while `[llm.sonnet] enabled` is false. `--sonnet-thinking` turns Sonnet on at one thinking setting with its thinking room, 0 for `disabled`, 1,024 for `between_tools` and 4,096 for `adaptive`, for that run alone, and never writes the config. Its summary gives `sonnet first status` and `sonnet first call thinking tokens`. The first setting whose first Sonnet call passes goes into `[llm.sonnet]` with its `thinking_tokens`, and `enabled` becomes true, and if `between_tools` is refused as well, `adaptive` is tried next.

**The week one report.** `news-status` prints the counts and the order and backlog verdicts, as [v2_news.md](v2_news.md#the-news-report) describes.

```bash
.venv-v2/bin/btc_survivor_v2 news-status --days 7
.venv-v2/bin/btc_survivor_v2 news-status --days 7 --detail --samples 50
```

The process is specified in [section 14.3 of the plan](../PLAN_V2.md#143-the-news-process).

## The stock process

```bash
.venv-v2/bin/btc_survivor_v2 init-db configs/v2/s1_prices.toml
.venv-v2/bin/btc_survivor_v2 init-db configs/v2/s2_news.toml
.venv-v2/bin/btc_survivor_v2 stocks configs/v2/s1_prices.toml configs/v2/s2_news.toml --dry-run
(umask 077 && mkdir -p data/logs/v2/stocks)
nohup .venv-v2/bin/btc_survivor_v2 stocks configs/v2/s1_prices.toml configs/v2/s2_news.toml >> data/logs/v2/stocks/stdout.log 2>&1 &
pkill -TERM -f 'btc_survivor_v2 stocks configs'
```

* It refuses to start unless the two configs are identical apart from the run name and data folder, `use_news` and `[evaluation]`, and it takes the Finnhub websocket lock, so the recorder must stop first.
* The first start launches both runs fresh. It reads `news.db` for the news hashes and the newest score, so the news process must be running, and writes the launch meta, a `lock` event and a `code_commit` event to both databases. Every later start resumes both runs from their checkpoints and writes a new `code_commit` event.
* It records every price line and news line to `data/v2/raw_stocks` itself, so the raw recorder is no longer needed after launch.
* SIGTERM lets pending fills settle for at most 1.3 seconds while their symbol is healthy, checkpoints both runs with the shared layer and exits with code 0. Never stop it with `kill -9` outside a drill, because that skips the final checkpoint and loses up to five minutes of rows, although resume stays safe.
* An expected failure, such as a held lock or a refused resume, exits with code 2, and a crash with code 1.

Once a minute it writes `runtime_status` and `symbol_status` to both databases and logs a heartbeat line with the feed state, whether the session is open, how many symbols are healthy, prints and messages per second, the age of the newest news heartbeat, the living learners of each run, memory and the worst lateness of the one second ticker. `rss` reads `n/a` on the Mac, where the process cannot see its own resident memory.

**Checkpoints and resume.** Each run writes a checkpoint every 5 minutes, 30 seconds after a minute boundary, and keeps the newest 3. Resume takes the newest checkpoint both runs hold and can read, rolls back anything written after it, and refuses a changed experiment hash, a changed session index or two copies of the shared layer that differ. Orders in flight at a restart are rejected with reason `restart`.

**Restarts and code updates.** A restart outside the session flattens nothing, so code updates deploy only then. A restart inside the session is a gap that discards pending labels, and after more than 10 session minutes the next healthy minute closes every position with `restart_flatten`. A sleeping Mac or a feed outage inside the session works the same way with `gap_flatten`. Never edit a run config after launch, because resume refuses a changed experiment hash. The process is specified in [section 14.2 of the plan](../PLAN_V2.md#142-the-stock-process).

## Launch day

The launch is Monday 19 October 2026, at the session open, with the news process already running on the final config and the collector at least five trading days old.

1. Before the open, run the checks before anything starts, including `check-local` and the secret scan, and confirm the version 1 runs keep their heartbeats.
2. Run `init-db` on both run configs, which must not hold a database yet.
3. Confirm the four stock configs hold the health thresholds and the delay settings measured before the rehearsal, then stop the raw recorder with SIGTERM and confirm it logged its stop.
4. Start the stock process, by hand or by loading its launchd agent with the owner's approval, by about `09:15` ET, so its local only check runs at `09:20` ET, ten minutes before the open.
5. Confirm both databases hold the `lock` and `code_commit` events and a passed `local_only` event, and that the heartbeat shows the feed connected and the news age under 60 seconds.
6. Through the first session, run the checks of the plan's first live session list, such as no opens before `10:00` ET, no position after `15:55` ET and zero invariant violations.

Session 1 only records history. Vectors become valid at `09:31` in session 2, training starts there, and the first trades come in session 3 once warmup ends, forced by the timer or made by a learner whose prediction clears the round trip cost. The checks are in [section 17 of the plan](../PLAN_V2.md#17-first-live-session-checks).

## Status, export, replay and the dashboard

```bash
.venv-v2/bin/btc_survivor_v2 status configs/v2/s1_prices.toml configs/v2/s2_news.toml
.venv-v2/bin/btc_survivor_v2 status configs/v2/s1_prices.toml configs/v2/s2_news.toml --detail
.venv-v2/bin/btc_survivor_v2 export configs/v2/s2_news.toml
.venv-v2/bin/btc_survivor_v2 replay configs/v2/s2_news.toml data/v2/raw_stocks --out data/v2/replays/check
nohup .venv-v2/bin/btc_survivor_v2 dashboard configs/v2/s1_prices.toml configs/v2/s2_news.toml >> data/logs/v2/dashboard.log 2>&1 &
```

* `status` prints state words, counts, timings and the experiment hash for both runs, and `--detail` adds per symbol rows with the last reference price, per bot rows and the runtime figures.
* `export` writes every table of one run except the checkpoints as CSV into `data/v2/exports/<run name>`, readable by the owner alone, from one read snapshot, and prints the row counts and one SHA 256.
* `replay` runs recordings through the same code into a new or empty folder, never opens `news.db` and never sends a frame. Replay results are never used to choose, tune or rank anything, and replay exists only to test the plumbing.
* `dashboard` serves the private page at [http://127.0.0.1:8060](http://127.0.0.1:8060), with the market session, the Finnhub feed, the news process with spend and the daily spend alert, prices, descriptive forecast skill, recent news, both populations with their equity curves, and the decisions explorer of every decision of both runs. It reads the secrets file at start only to register the values with the redactor, opens every database read only and answers 403 to any Host header other than `127.0.0.1` or `localhost` on its port.
* A UTC and Local switch at the top of the dashboard shows every time in UTC, the clock the runs keep, or in this computer's own zone with its abbreviation, as the version 1 dashboard does. Times marked ET stay New York time in both. The from and to filters of the decisions explorer read and show times in the chosen zone and send the server the UTC minute they mean, a time ending in Z is UTC in either zone, and the choice is remembered in the browser.

Every reader refuses a database of the wrong kind or a version 1 database. The dashboard is described in [section 13.4 of the plan](../PLAN_V2.md#134-the-private-stock-dashboard).

## The local only check

`btc_survivor_v2 check-local` lists listening sockets on ports 8050 and 8060 with `lsof` and `netstat`, running processes with `ps` and, when Tailscale is installed, its serve settings. It passes only when both ports listen on `127.0.0.1` alone and no tunnel or remote forward runs, such as `cloudflared`, `ngrok`, `tailscale funnel`, `tailscale serve` or `ssh -R`. By default it prints the verdict and counts, and `--detail` adds the process id and program behind each problem, never a full command line, because a tunnel's arguments can carry its token. The stock process runs the same check ten minutes before each open and at a start inside that window or inside a session, and writes the verdict to both runs as a `local_only` event. `evaluate` fails the run for any failed check and for any session of the window without a passed one, so a session in which the stock process never ran is a failure of criterion 1.8. The rule is in [section 13.2 of the plan](../PLAN_V2.md#132-local-only-with-no-public-page).

## The secret scan

```bash
.venv-v2/bin/python scripts/v2_secret_scan.py --logs data/logs/v2
```

It takes every value in `secrets.env` without naming a key and looks for each one, ignoring case, in every file the repository tracks or would add, every staged version in the index, every file under `data/v2`, every file under the log folders given and every git object from the remote branch to `HEAD`, which covers each blob, tree, commit message and author and committer field. Gzip recordings are read as a stream, so a file still being written is read up to its last flush. `--logs data/logs/v2` covers the logs of the news process, the stock process, the recorder and the launchd agents. It prints only counts and the paths of hits, never a value, and exits with code 0 when nothing holds a value, 1 when something does and 2 when it cannot run. It runs before every push and after the first day of the window. The scan is described in [section 14.4 of the plan](../PLAN_V2.md#144-locks-secrets-and-supervision).

## Keeping the Mac awake

A sleeping Mac freezes every process. The news process writes a `pause` event when it wakes, the stock process judges every minute of the sleep inside a session unhealthy and flattens open positions with `gap_flatten` after more than 10 session minutes, and a sleep across a whole session leaves that session without a `local_only` check, which fails criterion 1.8. A sleep is an outage, never a death.

* Keep the Mac on power. With the lid closed on battery it sleeps whatever holds it awake, as the first night of version 1 showed.
* `caffeinate -s` holds a sleep assertion while the Mac is on power, and `-w` ties it to a process, so the assertion ends when the process does. The launchd agents below start it for their process.
* On a MacBook, System Settings, Battery, Options has a switch that prevents automatic sleeping on power adapter when the display is off.

```bash
caffeinate -s -w "$(pgrep -f 'btc_survivor_v2 stocks configs')" &
caffeinate -s -w "$(pgrep -f 'btc_survivor_v2 news configs')" &
pmset -g assertions
pmset -g log | grep -E ' (Sleep|Wake)  ' | tail -n 20
```

The first `pmset` command shows what holds the Mac awake, and the second when it last slept and woke, in local time. After each session day `pmset` should show no sleep inside the session, as the plan's first live session checks ask. Supervision is described in [section 14.4 of the plan](../PLAN_V2.md#144-locks-secrets-and-supervision).

## launchd agents

`deploy/macos/btc_survivor_v2_news.plist` and `deploy/macos/btc_survivor_v2_stocks.plist` run the two processes as launchd agents of the logged in user. They are installed only with the owner's approval, question 9 of the plan, and nothing installs them otherwise.

* Each agent runs `/bin/sh`, which starts `caffeinate -s -w` on its own process id and then execs the process, so launchd signals the process itself and the sleep assertion lasts exactly as long as it runs.
* `RunAtLoad` and `KeepAlive` start the process when the agent loads and restart it after any exit, at most once every 10 seconds, the `ThrottleInterval`. An exit with code 2, such as a held lock, is retried every 10 seconds too, which is why the stock agent is loaded only at launch.
* `ExitTimeOut` gives a stopping process 60 seconds before launchd sends SIGKILL, room for the news process's 20 second grace and the stock process's final checkpoint.
* `Umask` 63 is 077, so every file is the owner's alone. `ProcessType` Interactive spares the processes the CPU and I/O throttling launchd gives agents by default.
* `PATH` lists the system folders that hold `git`, `ps`, `lsof`, `netstat` and `caffeinate`, and the Homebrew folders, so the local only check finds a Homebrew `tailscale`.
* Standard output and error go to `data/logs/v2/launchd/btc_survivor_v2_news.log` and `data/logs/v2/launchd/btc_survivor_v2_stocks.log`, which are never rotated, grow by about the size of the process log, and are covered by the secret scan.

To install the news agent, stop the news process started by hand first, because it holds the Finnhub REST lock. The `sed` line writes the current user into each path, which assumes the worktree sits at `~/code/btc_survivor_v2`.

```bash
mkdir -p data/logs/v2/launchd && chmod 700 data/logs/v2/launchd
sed "s|CHANGE_ME|$(id -un)|g" deploy/macos/btc_survivor_v2_news.plist > ~/Library/LaunchAgents/btc_survivor_v2_news.plist
plutil -lint ~/Library/LaunchAgents/btc_survivor_v2_news.plist
pkill -TERM -f 'btc_survivor_v2 news configs'
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/btc_survivor_v2_news.plist
launchctl print gui/$(id -u)/btc_survivor_v2_news | grep -E 'state|pid|last exit'
```

The stock agent is installed the same way at launch, after the recorder stopped, because its first start launches both runs.

```bash
sed "s|CHANGE_ME|$(id -un)|g" deploy/macos/btc_survivor_v2_stocks.plist > ~/Library/LaunchAgents/btc_survivor_v2_stocks.plist
plutil -lint ~/Library/LaunchAgents/btc_survivor_v2_stocks.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/btc_survivor_v2_stocks.plist
```

A clean restart sends SIGTERM and lets `KeepAlive` start the process again, and a stop unloads the agent, which sends SIGTERM first. An agent stays unloaded until it is bootstrapped again, and every agent in `~/Library/LaunchAgents` loads again at the next login.

```bash
launchctl kill SIGTERM gui/$(id -u)/btc_survivor_v2_news
launchctl bootout gui/$(id -u)/btc_survivor_v2_news
launchctl bootout gui/$(id -u)/btc_survivor_v2_stocks
```

The alternative without agents is the `nohup` lines above with `caffeinate -w` and restarts by hand. Supervision on the Mac is described in [section 14.4 of the plan](../PLAN_V2.md#144-locks-secrets-and-supervision).

## Later on a Raspberry Pi

`deploy/systemd/btc_survivor_v2_news.service` and `deploy/systemd/btc_survivor_v2_stocks.service` move the two processes to a Raspberry Pi on a weekend after the window, together, because they share `news.db`, and only with the owner's approval. Both restart always after 10 seconds, wait for the network and a synchronized clock, stop with SIGTERM and allow 60 seconds before SIGKILL, and the stock unit starts after the news unit. They assume the project at `/home/CHANGE_ME/btc_survivor_v2` with its own `.venv-v2`.

1. Replay the soak on the Pi first, with no network, to measure CPU per print and memory. FinBERT needs an estimated 300 to 500 MB, which may not fit a small Pi.
2. Stop both processes on the Mac with SIGTERM, because the Mac and the Pi cannot both stream on one key.
3. Copy `data/v2` and the key ledger, and run `PRAGMA integrity_check` on every copied database. The owner copies `secrets.env` by hand and sets it to mode 600.
4. Install and start the units.
5. Repeat the first live session checks.

```bash
sudo cp deploy/systemd/btc_survivor_v2_news.service deploy/systemd/btc_survivor_v2_stocks.service /etc/systemd/system/
sudo sed -i "s|CHANGE_ME|$(id -un)|g" /etc/systemd/system/btc_survivor_v2_news.service /etc/systemd/system/btc_survivor_v2_stocks.service
sudo systemctl enable systemd-time-wait-sync.service
sudo systemctl daemon-reload
sudo systemctl enable --now btc_survivor_v2_news.service
sudo systemctl enable --now btc_survivor_v2_stocks.service
journalctl -u btc_survivor_v2_stocks.service -n 20
```

On Linux the local only check lists sockets with `ss`, and `rss` in the stock heartbeat shows resident memory. The move is described in [section 14.5 of the plan](../PLAN_V2.md#145-later-on-the-pi-cpu-and-memory).

## Rehearsal and drills

Two rehearsal sessions run the whole stock process live before launch, on Thursday 15 and Friday 16 October, with `configs/v2/rehearsal_s1.toml` and `rehearsal_s2.toml`. They differ from the run configs only in their run names, `rehearsal_s1` and `rehearsal_s2`, their data folders under `data/v2/rehearsal`, and a warmup of 30 training minutes and a forced timer of 30 healthy minutes, so trading runs live on the second session. Those two values are plumbing only and never a strategy setting. The rehearsal runs the launch code, frozen on the Wednesday before, and the news process must already score on that code. It reads the production `news.db` read only and records into the production raw folder under the same websocket lock, so the recorder stops first, and `--log-dir` keeps its log beside the others.

```bash
pkill -TERM -f 'scripts/v2_record_stocks.py'
nohup .venv-v2/bin/btc_survivor_v2 stocks configs/v2/rehearsal_s1.toml configs/v2/rehearsal_s2.toml --log-dir data/logs/v2/rehearsal >> data/logs/v2/rehearsal.log 2>&1 &
```

* **Kill drill.** `kill -9` the stock process mid session and start it again, once after under 10 minutes and once after more, to see resume, the rejected orders and `restart_flatten`.
* **News restart drill.** Stop and restart the production news process, which briefly interrupts collection, and check that the puller reads new heartbeats within 60 seconds and the lost minutes are reported by cause.
* **Pause drill.** `kill -STOP` the stock process for at least twice `feed_stale_after_seconds`, well past the limit for a silent feed, then `kill -CONT`, and check the pause, the reconnect and the stale and recovering minutes.
* **Cap drill.** On one rehearsal day, `scripts/llm_synthetic_check.py configs/v2/news.toml --cap-drill` runs a separate news process on a copy of the news config with every channel and Sonnet off and a cap of one cent, with a throwaway ledger under `data/v2/rehearsal`. It reserves its cent on the key ledger before it starts, so it never runs when the month has less than a cent left, and books what it spent on that reservation, so a drill that is killed stays booked at the full cent. It passes when the code's own cap refuses a call, shown by the capped event and a ledger reservation for every call, waiting items get `skipped_cap`, Haiku stays capped until the next month, FinBERT scores every item and the drill spends at most one cent. The copy also moves the drill's daily probe to `23:59` UTC, a change beyond the plan's list that needs the owner's agreement.

**The pass rule.** The rehearsal passes only when all four of these hold. Otherwise the slip rule applies, and the launch moves to the first session open after a passed rehearsal with the collector at least five trading days old.

1. **FinBERT is on.** Every news heartbeat written during each rehearsal session reports FinBERT on, so news health could hold.
2. **S2's news inputs are nonzero.** S2 wrote `news` forecast rows with a nonzero value, so scored news reached its inputs.
3. **The readers run cleanly each day.** After each rehearsal session's close, `evaluate`, `status`, `export` and the dashboard run on the rehearsal databases. Each command exits with code 0 or 1, never 2, and prints no traceback, and the dashboard answers its state and the decisions explorer with no error entry. After two sessions some checks of `evaluate` are not due yet, and every check it fails is read and explained before launch.
4. **A replay digest matches.** Once the rehearsal's stock process has stopped, the files it recorded, named by the UTC time each started, from its first start on Thursday to its stop on Friday, are copied to `data/v2/rehearsal/recording`, so the set cannot change. Two replays of that folder into two new folders print the same database digest.

```bash
sqlite3 -readonly data/v2/news/news.db "SELECT substr(ts, 1, 10), finbert_state, count(*) FROM news_heartbeat WHERE ts >= '2026-10-15' AND substr(ts, 12, 5) BETWEEN '13:30' AND '19:59' GROUP BY 1, 2"
sqlite3 -readonly data/v2/rehearsal/s2_news/stocks.db "SELECT count(*) FROM stock_forecasts WHERE predictor = 'news' AND prediction != 0"
.venv-v2/bin/btc_survivor_v2 evaluate configs/v2/rehearsal_s1.toml configs/v2/rehearsal_s2.toml --detail; echo "exit $?"
.venv-v2/bin/btc_survivor_v2 status configs/v2/rehearsal_s1.toml configs/v2/rehearsal_s2.toml; echo "exit $?"
.venv-v2/bin/btc_survivor_v2 export configs/v2/rehearsal_s1.toml; echo "exit $?"
.venv-v2/bin/btc_survivor_v2 export configs/v2/rehearsal_s2.toml; echo "exit $?"
nohup .venv-v2/bin/btc_survivor_v2 dashboard configs/v2/rehearsal_s1.toml configs/v2/rehearsal_s2.toml >> data/logs/v2/rehearsal_dashboard.log 2>&1 &
curl -s http://127.0.0.1:8060/api/state | grep -o '"error"' | wc -l
curl -s http://127.0.0.1:8060/api/decisions | grep -o '"error"' | wc -l
pkill -TERM -f 'btc_survivor_v2 dashboard configs/v2/rehearsal'
.venv-v2/bin/btc_survivor_v2 replay configs/v2/rehearsal_s2.toml data/v2/rehearsal/recording --out data/v2/replays/rehearsal_a
.venv-v2/bin/btc_survivor_v2 replay configs/v2/rehearsal_s2.toml data/v2/rehearsal/recording --out data/v2/replays/rehearsal_b
```

The first query must show only `on` in session on both days, the second a count above zero, both `curl` lines 0, and the two replays one `database digest`. After the rehearsal the recorder can start again until launch. The rehearsal and drills are set out in [section 16 of the plan](../PLAN_V2.md#16-build-phases-with-tests).

## The soak

`scripts/v2_soak.py` replays every recording in `data/v2/raw_stocks`, joined end to end, and then a seeded synthetic run of 14 days with many reconnects, halts, news bursts and news outages, through the real stock process code in a child process, on the rehearsal pair by default. It checks the books of every bot, the ledger identities, fills, each of which must come from a print traded at or after its order's due time, opens, closes, forecasts, the news receipt rule, S1 and S2 sharing every price row and price z score row, and that a second child replaying the same lines writes identical tables. It copies the newest recording before reading it, so a process still writing it is never disturbed, and writes everything into a new folder under `data/v2/soak`.

```bash
nice -n 10 .venv-v2/bin/python scripts/v2_soak.py
nice -n 10 .venv-v2/bin/python scripts/v2_soak.py --skip-real --out data/v2/soak/synthetic_only
```

**The memory rule.** Memory at the end may be at most 5 MB above memory at the end of the first replayed hour.

* On Linux, as on the Pi, resident memory decides it, strictly.
* On a Mac the memory the allocators report in use decides it, because macOS keeps freed pages resident. By the owner's decision of 12 October 2026 a Mac soak also passes when memory in use grew less than 1 MB over the final replayed week, compared at the same point of the weekly cycle, even when it grew more than 5 MB in all.
* A soak shorter than a week and an hour has no final week, so only the 5 MB rule decides it.

The summary says which rule decided, `first hour` or `final week`, and `--max-week-growth-mb` sets the week limit. The exit code is 0 for passed, 1 for failed, 2 when it cannot run and 3 for inconclusive.

The synthetic run of 12 October 2026, without recordings, replayed 336 hours with about two million prints and 14,323 news lines in 247 seconds. Every invariant and the determinism check passed, at about 0.02 percent of one core at live speed, but memory in use ended 5.95 MB above the first hour, which failed the 5 MB rule. The growth is bounded rather than a leak. History windows fill during the first sessions, and S2 keeps each score for 600 session minutes, which do not pass at night or over a weekend, so a weekend's news stays in memory until Tuesday. Memory in use moved only 0.084 MB from one week to the next, so under the rule decided that day its memory verdict passes on the final week. An 8 day run at the plan's news volume, `--news-minutes 10 --burst-items 150`, passed the 5 MB rule with 3.92 MB. The synthetic news, one item every 5 minutes around the clock plus a 300 item burst each session, is about twice the plan's highest estimate. The soak runs again on the launch code before the rehearsal. The soak is described in [section 16 of the plan](../PLAN_V2.md#16-build-phases-with-tests).

## Disk growth

* **Raw recordings** are the largest item. The plan's estimate for the full tape, 5 to 10 million prints a day for the eight symbols and about 50 to 120 MB a day compressed, is far above what the partial free feed carries, and recordings are kept 30 days.
* **Run databases** grow per full session by about 400 decision rows, 3,120 candle rows, 96 label rows, 190 forecast rows in S1 and 1,060 in S2, and 390 `news_minutes` rows in S2. Candles are kept for the life of the run, about 125,000 rows a run over 40 sessions, because `evaluate` recomputes every label from them.
* **`news.db`** grows by roughly 1 to 2 GB a year, an estimate, mostly polls and model calls.
* **Logs** rotate at UTC midnight with 14 days kept, while `stdout.log` and the launchd logs are never rotated.

```bash
du -sh data/v2/* data/logs/v2
df -h .
```

The estimates are in [section 9 of the plan](../PLAN_V2.md#9-storage) and [section 4.10](../PLAN_V2.md#410-load-estimates), and the first sessions replace them with measured figures.
