# BTC Survivor

BTC Survivor is a population of 20 tiny neural networks that paper trade Bitcoin on live public market data from Coinbase, around the clock, on a laptop or a Raspberry Pi. Every bot starts with random weights and paper money, pays realistic fees, slippage and a daily running cost, and dies when its money can no longer cover one more round trip. A dead bot is replaced at once by a mutated copy of a survivor that won a small fitness tournament, or now and then by a random newcomer. At these costs only run C, which starts with 500 dollars, sees deaths within its window, and there costs and market noise rather than any plausible skill mostly decide who dies and who becomes a parent, so version 1 checks that this machinery works and does not test whether evolution helps. The bots learn forward in time only, one minute at a time, from data as it arrives. The design and the evidence behind every setting are in [PLAN.md](PLAN.md).

## What it is and what it is not

It is an experiment that asks whether small online learners can predict 30 minute Bitcoin moves well enough to pay real retail costs. It keeps books that always balance and logs that say why every bot lived or died. A result in which every bot dies to fees is a valid finding, and no default was tuned to make the bots look good.

* **Paper only.** No real money is involved. The code holds no function that places, cancels or signs an order on any exchange, and [tests/test_no_orders.py](tests/test_no_orders.py) scans the source for order and credential code.
* **Keyless data.** The bots read the public Coinbase Advanced Trade websocket channels `level2`, `market_trades` and `heartbeats`, which need no account and no API key. The project never asks for keys and never creates them.
* **No backtests.** Nothing is chosen, tuned or ranked on past market data. The replay harness exists only to test plumbing.
* **No advice and no strategy.** Paper fills are always kinder than real trading, and nothing here claims that any bot would make money.

Every minute the system turns the live order book and trades into 11 shared features. Each bot is a numpy network with 11 inputs, 16 tanh hidden units and one output, 209 parameters in all, that predicts the move of the mid price over the next 30 minutes. It takes one learning step whenever the outcome of an earlier prediction becomes known, and scores that prediction out of sample first. Once per horizon, at its own minute, it compares the predicted move with the cost of a round trip plus its margin gene, and opens, holds, flips or closes a fixed 500 dollar position. A flat bot that has opened nothing for 12 healthy hours must trade on the sign of its prediction. These forced trades are not how the bots learn. Every learner trains on every matured label whether or not it trades, so a trade adds no training data, and forced trades exist so that costs, the twin comparison, deaths and parent eligibility can be measured. Each learner also has a coin flip twin that trades at the same moments in a random direction, so the gap between the two measures skill rather than timing luck. The details are in [docs/architecture.md](docs/architecture.md), [docs/learning.md](docs/learning.md) and [docs/evolution.md](docs/evolution.md).

## Current status

Phases 1 to 9 of the build are complete, together with Phase 7b, which refocused the runs on prediction skill. That covers the feed and the order book, learning, the bots, evolution and the twins, the engine with checkpoints and resume, replay, the command line, the read only dashboard, this documentation and the systemd unit. The test suite runs on Python 3.11 and 3.13 and needs no network. Phase 10, the soak test with the kill and pause drills, has passed on the Mac. The books balanced throughout and memory in use stayed flat. The tighter Raspberry Pi rules, 150 MB and resident memory read on Linux, are checked only if the runs move to a Pi, as [docs/operations.md](docs/operations.md#the-soak-test) explains. The phases are listed in [section 18 of the plan](PLAN.md#18-build-phases).

Four live runs started on 11 October 2026 on a MacBook, under Python 3.11 in `.venv311`. All four use master seed 12345, so they started from identical networks, with run D's learning rate set lower, and each has its own config and data folder. A fifth, run E, was added on 12 October 2026 by the owner's decision. It is hypothetical, with no fee, slippage, short charges or margin, so its bots trade on their own predictions, and it uses the same seed and the same environment. A sixth, run D2, was added on 13 October 2026 by the owner's decision. It is run D with tolerance for short stalls in Coinbase's stream, so they stop discarding its 4 hour outcomes. On 12 October 2026 the five runs moved to the owner's always on home PC, an Ubuntu machine on Ethernet with the same Python 3.11.16 and numpy 2.4.6, one run at a time, each with a gap of about 2 minutes. No setting changes during a run, and every run is reported whatever it shows. The first night taught one lesson already. With its lid closed on battery the Mac slept, every run froze, and each logged a `resume` event 99 minutes after its last healthy minute. The runs recovered by themselves, no rent was charged for the gap, and no bot held a position that needed closing. That lesson and the others are in [docs/operations.md](docs/operations.md).

## The six runs

| Run | Config | Fee profile | Start equity | Horizon | Started, UTC | What it shows |
| --- | --- | --- | --- | --- | --- | --- |
| A, main | [`run_a.toml`](configs/run_a.toml) | `retail_default`, 90 bps | 10,000 dollars | 30 minutes | 11 October, `20:41` | the headline skill result, and what retail costs do to bots that cannot pass the gate |
| B, low fee | [`run_b.toml`](configs/run_b.toml) | `low_fee_tier`, 5 bps | 10,000 dollars | 30 minutes | 11 October, `20:41` | the same forced schedule at a 5 bps fee, with its survival and cost split |
| C, machinery | [`run_c.toml`](configs/run_c.toml) | `retail_default`, 90 bps | 500 dollars | 30 minutes | 11 October, `20:41` | that death, replacement, mutation and lineage work under live conditions |
| D, long horizon | [`run_d.toml`](configs/run_d.toml) | `low_fee_tier`, 5 bps | 10,000 dollars | 4 hours | 11 October, `21:39` | exploratory, whether a longer horizon lets real predictions clear the gate |
| D2, long horizon with stall tolerance | [`run_d2.toml`](configs/run_d2.toml) | `low_fee_tier`, 5 bps | 10,000 dollars | 4 hours | 13 October, `15:48` | exploratory, run D allowing 30 seconds of silence and 5 seconds of delay, what the strict stall rule costs a long horizon run |
| E, hypothetical frictionless | [`run_e.toml`](configs/run_e.toml) | `hypothetical_zero`, 0 bps, with no slippage, short charges or margin | 10,000 dollars | 30 minutes | 12 October | exploratory and hypothetical, whether the bots' own calls make money before costs against their twins and cash, never whether they could trade for real |

At 500 dollars a trade one leg costs about the fee plus 1 bp, because the spread and the walk through the book round to almost nothing. The 90 bps fee is the Coinbase US Intro taker rate since 16 September 2026, and 5 bps is the VIP3 rate, which needs 20 million dollars of volume a month. Under the retail fee a bot needs a predicted move of about 190 bps, many times a typical 30 minute move of 17 to 34 bps. Under the 5 bps fee it still needs about 20 bps, several times the size of a careful learner's predictions. These outcomes follow from that arithmetic and were written down before launch.

* **Run A.** Every open is forced and no bot dies by day 28. A learner pays 1.16 dollars of rent per healthy day and 9.10 dollars per round trip, about 19.36 dollars a day once its two forced round trips a day begin.
* **Run B.** Almost every open is forced, and no bot dies, because death is more than a year away even for a bot that trades at every decision.
* **Run C.** Every bot dies of fees. The first wave of deaths falls between days 24 and 28 and the second between days 48 and 56, when generation 2 appears. Parent choice there mostly reflects market noise, because costs swamp any plausible skill. Its notional stays 500 dollars while equity falls and no money is set aside for it, so by its last open a bot that trades only when forced carries nearly 50 times its equity, which no spot account allows.
* **Run D.** It was added after the other three had started, with its settings fixed before its own launch. Its learning rate is 0.00003, set on synthetic data only, because at a 4 hour horizon 0.0001 made predictions wide enough to pass the gate on noise. All 20 bots decide within the first 20 minutes of each 4 hour block, so its trades cluster in time, and with about 6 independent windows a day its skill numbers are descriptive only. A quiet run D bot spends about 2.36 dollars a day, so none dies.
* **Run D2.** It was added on 13 October 2026 by the owner's decision, with its settings fixed before its own launch. Coinbase's stream pauses for a few seconds, or delivers messages late, with delay medians logged at 2.0 to 3.25 seconds as each trip began, every few hours, which trips the 5 second stale rule or the 2 second clock skew rule. Every such trip ends the feed epoch and discards every pending label, and a 4 hour horizon nearly always contains one, so by 13 October only about 20 percent of run D's decisions had a matured label against about 87 percent in run A, and each trip paused its learning for 4 hours. Run D2 is run D allowing 30 seconds of silence and a 5 second median delay, so delay trips and brief pauses no longer end its epochs, while a pause in which Coinbase holds back more than about 6 seconds of stream, a disconnect, a resync or a restart still does. It shows whether its labels and learning keep going and how it compares with run D on the same market. Its bots may decide and learn through those short stalls, a real loosening that is small next to 4 hours. Like run D it is exploratory and none of its bots dies.
* **Run E.** It was added on 12 October 2026 by the owner's decision, because no prediction in runs A to D had passed the gate. The largest was about 8 bps against a required edge of about 190 bps in runs A and C and 20 to 24 bps in runs B and D. Run E is run B with the taker fee, extra slippage, short opening fee, short funding and margin gene set to zero, which no venue offers, so it is hypothetical and never a realistic cost setting. Rent stays 1.16 dollars a day. Only the spread and the walk through the book remain in its gate, about 0.001 bps, so about 97 percent of the decisions runs A to D recorded would have passed it, and its bots trade on their own calls. It shows whether those calls make money before costs and how they compare with their coin flip twins and with holding cash. It cannot show anything about real tradability, and its trades are not a confirmatory test. With no fees a bot's death line is about one day of rent, so none dies.

So runs A, B, D, D2 and E see no deaths, replacements or parent choice within their windows, and selection in run C is mostly noise. Version 1 has no control run without selection, so it does not test whether evolution helps. An external review in October 2026 confirmed this, and it was recorded as a limit rather than tested, as [docs/honesty_and_limits.md](docs/honesty_and_limits.md#selection-and-why-evolution-is-not-tested) explains, together with run C's leverage.

The one confirmatory test is computed from run A at day 28 of healthy data. It passes when the population's forecast correlates with the realized 30 minute move above zero at the one sided 1 percent level, using one forecast per 30 minute window so outcomes never overlap, with a Newey West standard error over 12 windows. With about 1,280 scored windows it can detect a correlation of about 0.1 with 90 percent power if the windows are independent. `evaluate` prints that figure beside a second detectable correlation taken from the realized Newey West error, which is usually larger when volatility clusters, and beside a one sided 95 percent upper bound on the correlation. A test that fails rules out only correlations above that bound, and it is never read as evidence that the correlation is below about 0.1. Beating the coin flip twins, survival, the run C machinery check, and the buy and hold, exposure matched and cash benchmarks are reported beside it. Days 7 and 14 are descriptive, and the best single bot is never evidence on its own. The criteria are in [section 21 of the plan](PLAN.md#21-controls-parallel-runs-and-success-criteria).

After a launch, features need 6 hours of healthy history and the label scale needs about 30 matured outcomes, so training starts after about 7 hours. Six hours of warmup follow, so learners turn active after about 13 hours of healthy data and make their first forced trades between about 13 and 25 hours after launch. Run D waits a full 4 hour horizon for its first outcomes, so each of these steps comes about 3.5 hours later there, and run D2 follows the same timeline from its own launch, later still whenever a disconnect or a long stall ends its epoch. Run E keeps the 30 minute timeline, and since its gate passes almost every prediction, its first trades on its own calls come about 13 hours after launch, within half an hour of warmup ending. Every gap in the feed pushes the timeline back, as [docs/operations.md](docs/operations.md#sleep-restarts-and-outages) explains.

## Quick start on a Mac

Live runs on the Mac use Python 3.11 from pyenv. The python.org Python 3.13 build cannot reach Coinbase over TLS until its bundled certificates are installed, so it serves only for tests. Run these steps from the project folder of a fresh clone. On the Mac that hosts the live runs they are already done and the runs are live, so check `pgrep -fl btc_survivor` before any `run`, because nothing in the code stops a second process from writing into the same database.

```bash
pyenv install 3.11.16
~/.pyenv/versions/3.11.16/bin/python -m venv .venv311
.venv311/bin/pip install -e ".[test]"
.venv311/bin/python -m pytest -q
.venv311/bin/python -m btc_survivor --config configs/run_a.toml init-db
nohup .venv311/bin/python -m btc_survivor --config configs/run_a.toml run >> data/run_a/stdout.log 2>&1 &
.venv311/bin/python -m btc_survivor --config configs/run_a.toml status
```

`init-db` creates `data/run_a`, and the other runs start the same way with their own config. For a separate experiment, copy [config.example.toml](config.example.toml) to `config.toml`, which every command reads when `--config` is left out. Restarting a run has its own rules, listed in [docs/operations.md](docs/operations.md#starting-runs). Keep the Mac on power with the lid open, as [docs/operations.md](docs/operations.md#keeping-a-mac-awake) describes.

## Quick start on a Raspberry Pi

The target is a Raspberry Pi 4 running Raspberry Pi OS Lite 64 bit. The Bookworm based Legacy image ships Python 3.11, so runs that move from the Mac resume on the same Python and numpy, and the Trixie image with Python 3.13 suits fresh runs. Each run is one systemd service made from the template unit [deploy/btc_survivor@.service](deploy/btc_survivor@.service), so `btc_survivor@run_a` runs `configs/run_a.toml`. The example assumes a Pi named `btcpi.local` with a user named `btc`.

```bash
# on the Mac, copy the project without environments, data or config.toml
rsync -a --exclude '.venv*' --exclude data --exclude config.toml ~/code/btc_survivor/ btc@btcpi.local:btc_survivor/

# on the Pi
sudo apt install -y python3-venv sqlite3
cd ~/btc_survivor
python3 -m venv .venv
.venv/bin/pip install ".[test]"
.venv/bin/pytest -q
.venv/bin/btc_survivor --config configs/run_a.toml init-db
sed 's/CHANGE_ME/btc/g' deploy/btc_survivor@.service | sudo tee /etc/systemd/system/btc_survivor@.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable systemd-time-wait-sync.service
sudo systemctl enable --now btc_survivor@run_a
journalctl -u btc_survivor@run_a -f
```

The unit restarts a run five seconds after a crash and gives it a minute to stop cleanly. The time sync service matters because a Pi 4 has no clock of its own and boots with a stale time until NTP answers. Moving a live run from the Mac, the USB SSD, memory and the checklist for the first 24 hours are in [docs/raspberry_pi.md](docs/raspberry_pi.md).

## Commands

Every command runs through the `btc_survivor` entry point or `python -m btc_survivor`. The `--config` option comes before the command and defaults to `config.toml`, so a typical call is `btc_survivor --config configs/run_a.toml status`. The report commands open the database read only, so they can never change a run.

| Command | What it does |
| --- | --- |
| `init-db` | creates an empty database in the run's data folder, or checks an existing one against the schema version |
| `run` | runs the live system until SIGINT or SIGTERM, then writes a final checkpoint and exits with code 0 |
| `status` | prints each living learner with age, state, equity, trades, win rate, fees, rent, skill, position, burn rate and days left, then the live price, the benchmarks and a summary |
| `status --watch` | redraws the status every minute until Ctrl+C |
| `status --twins` | adds the coin flip twins below the learners |
| `explain BOT_ID` | prints one bot's life story, with its genes, the five part cost breakdown, its worst trades and, for a dead bot, the cause of death with the numbers behind it |
| `lineage BOT_ID` | prints the chain of parents back to a random ancestor, with each one's lifespan, equity and cause of death |
| `export --out DIR` | writes CSV files of bots, trades, fills, decisions, equity snapshots, deaths, lineage, the benchmark and the forecasts |
| `evaluate` | prints the success criteria and the outcomes fixed by arithmetic for this run, computed from its database |
| `evaluate --final-day 56` | names the day of healthy data that counts, 28 by default for runs A and B and 56 for run C, and marks every earlier result descriptive |
| `dashboard [CONFIG ...] [--port 8050]` | serves the read only page for one or more runs on `127.0.0.1` until Ctrl+C |
| `replay --out DIR FILE [FILE ...]` | runs recorded feed files through the same pipeline into a new or empty folder, never the run's own data folder |

Scripts sit beside the commands. `scripts/record_feed.py` records the raw keyless feed with no bots and no database, `scripts/learner_check.py` checks the learner on synthetic data with a planted signal and never reads market data, and `scripts/plot_results.py` draws charts from an export folder on a laptop, with matplotlib from [`requirements-plot.txt`](requirements-plot.txt). Three read only reports were added after the external review. `scripts/population_value.py` prints the self financing value of every learner ever born beside buy and hold and cash, `scripts/run_c_analysis.py` prints notional over equity at every open, death days with and without a margin cap, the spread of death days against the effect of real skill, and parents' skill against the learners that could have been parents with them, and `scripts/check_config_integrity.py` compares each config file with the config text its run stored at launch. Each takes one or more run configs, opens every database read only, and with `--copies DIR` reads `DIR/run_x/btc_survivor.db` instead of the live run. Replay results are never used to choose, tune, or rank strategies. Each command is specified in [section 14 of the plan](PLAN.md#14-runtime-and-command-line).

## The local dashboard

The dashboard shows all six runs on one page. It runs from the project folder with a command equivalent to this one, and lives at [http://127.0.0.1:8050](http://127.0.0.1:8050).

```bash
nohup .venv311/bin/python -m btc_survivor dashboard configs/run_a.toml configs/run_b.toml configs/run_c.toml configs/run_d.toml configs/run_d2.toml configs/run_e.toml --port 8050 >> data/dashboard.log 2>&1 &
```

For each run the page shows the feed state, the launch stage, the timing of candles, horizon and forced trades, every bot with its warmup progress, position and next forced trade, the mean equity of the living learners against buy and hold, the exposure matched benchmark and the cash baseline, a self financing card that values every learner ever born and the outside money their replacements needed, the skill so far on the same rows as the headline test with its upper bound and detectable correlation, and the latest decisions, fills and feed events. It refreshes every 10 seconds and pauses while its tab is hidden. Every card and column has a hover hint, and the pointer over the equity chart shows the exact values at that minute. The Bots header gives the time of the checkpoint its rows come from, because bot state is saved only every 5 minutes. A UTC and Local switch in the page header shows every time on the page in UTC, the clock the runs keep, or in the viewer's own time zone with its abbreviation, such as CDT, and the browser remembers the choice. That covers the header's update time, the chart axis and the pointer readout, the bots, the recent decisions, fills and feed events with any time quoted inside an event, and the decisions explorer. The server binds to `127.0.0.1` only and opens each database read only, so it cannot change a run. It also refuses with status 403 any request whose Host header is not `127.0.0.1` or `localhost` with the serving port, such as `localhost:8050`, so a web page elsewhere cannot reach it by pointing a name of its own at this machine, which is called DNS rebinding. It stays local by choice, and nothing tunnels it to the internet. How to read it alongside `status` is in [docs/operations.md](docs/operations.md#the-dashboard-and-status).

Below the feed events sits the decisions explorer, which lists every decision of the open run, newest first, 50 to a page. It filters by bot, learners or twins, action, forced trades, the gate, the label, a recent period, a time range and words in the reason. The from and to fields read and show times in the zone the switch has chosen, and the page always sends the server the matching UTC time. A time ending in Z is read as UTC in either zone. Chips count each action under the other filters, and a summary line says what share of the matured calls pointed the right way. A learner's call is its prediction. A twin copies that prediction, so its call is the side its coin chose, and a twin's close has none. Choosing a row opens that decision in full, with every stored column, the bot's generation, origin and state, the fills of its order with price, fee, slippage and latency, the trade it opened or closed with its result once known, and the label. The filters stay in the browser, so they survive the refresh and a switch of tab, and the open decision is kept for each run. The page reads two JSON views, `/api/decisions?run=run_a` for one page of decisions and `/api/decision?run=run_a&id=N` for one decision. Both pass every filter to SQLite as a parameter, refuse a bad one with status 400 and a one sentence reason, and stop any query that runs past three seconds. They are written in [`dashboard.py`](src/btc_survivor/dashboard.py).

## Documentation

* [docs/architecture.md](docs/architecture.md) covers the modules, the data flow and the minute step that live runs and replay share.
* [docs/learning.md](docs/learning.md) covers features, standardization, labels, the model, skill tracking and why overlapping labels hold fewer lessons than they seem to.
* [docs/execution_model.md](docs/execution_model.md) covers fills, fees, slippage, latency and short funding, with sources and dates.
* [docs/evolution.md](docs/evolution.md) covers death, replacement, fitness, tournaments, mutation, lineage and later ideas.
* [docs/operations.md](docs/operations.md) covers starting and stopping runs, logs, feed events, sleep, restarts, checkpoints, the recorder and disk use.
* [docs/raspberry_pi.md](docs/raspberry_pi.md) covers the Pi setup, the systemd unit, SD card wear and time sync.
* [docs/honesty_and_limits.md](docs/honesty_and_limits.md) covers what paper results cannot show, including why version 1 does not test evolution, run C's leverage and how to read a failed skill test.
* [docs/RESULTS_TEMPLATE.md](docs/RESULTS_TEMPLATE.md) holds the empty tables for writing up each run.
* [PLAN.md](PLAN.md) is the approved version 1 design, and [CONTRIBUTING.md](CONTRIBUTING.md) holds the rules for changes.
* [PLAN_V2.md](PLAN_V2.md) is the approved version 2 plan, for seven large US technology stocks and the QQQ fund with news scored by language models. It runs beside version 1 and changes no version 1 module, config or run, and its [road map](PLAN_V2.md#117-road-map-after-version-2) sets out the stages after it.

## Safety rules

1. There is no real money and no order placement. The code holds no trading client of any kind, and a test fails if order or signing code appears.
2. Only public market data that needs no credentials is used, and no API key is ever created or requested.
3. There is no lookahead. A feature at minute t uses only data received by t, and a label is used only after its horizon has passed.
4. No backtest or replay chooses, tunes or ranks anything. Settings are fixed before launch, from reasoning or synthetic checks.
5. Nothing trades or learns on stale data. When the feed is unhealthy everything pauses, and no rent accrues.
6. The books always balance. The ledger invariants are checked at every equity snapshot, and a violation is logged and recorded.
7. A restart or a power cut is not a death, and every run resumes from its last checkpoint.
8. All times are UTC inside the system, and every random draw comes from the one master seed.
9. The dashboard binds to `127.0.0.1`, and every report opens the database read only.

The ground rules for changes are in [CONTRIBUTING.md](CONTRIBUTING.md), and what paper results can never show is in [docs/honesty_and_limits.md](docs/honesty_and_limits.md).

## License

BTC Survivor is released under the [MIT License](LICENSE).
