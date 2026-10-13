# Results template

This file holds the tables for writing up the six live runs. Results are written up at days 7, 14 and 28 of healthy data, and at day 56 for run C. Days 7 and 14 are descriptive and only the final day counts, which is day 28 for runs A and B and day 56 for run C. Runs D, D2 and E are exploratory and every row for them is descriptive. Run E is also hypothetical, with no fee, slippage, short charges or margin, so none of its rows says anything about real tradability. Every run is reported whatever it shows, no setting changes during a run, and the best single bot is never evidence on its own. The criteria were fixed before launch in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria), and what they can and cannot show is in [honesty_and_limits.md](honesty_and_limits.md).

Most numbers come straight from the `evaluate` command. With `--final-day` it freezes every number at the moment the run reached that many days of healthy data, so running it again later gives the same rows. Use the virtual environment the run itself uses, `.venv311` on the Mac or `.venv` on the Pi.

```bash
.venv311/bin/btc_survivor --config configs/run_a.toml evaluate --final-day 7     # the day 7 rows, descriptive
.venv311/bin/btc_survivor --config configs/run_a.toml evaluate --final-day 28    # run A at its final day
.venv311/bin/btc_survivor --config configs/run_c.toml evaluate --final-day 56    # run C at its final day
.venv311/bin/btc_survivor --config configs/run_a.toml status --twins            # living bots and the all bots figure, now only
.venv311/bin/btc_survivor --config configs/run_a.toml export --out ~/btc_exports/run_a  # every table as CSV, outside the repository
.venv311/bin/python scripts/check_config_integrity.py configs/run_a.toml configs/run_b.toml configs/run_c.toml configs/run_d.toml configs/run_d2.toml configs/run_e.toml
.venv311/bin/python scripts/population_value.py configs/run_a.toml configs/run_b.toml configs/run_c.toml configs/run_d.toml configs/run_d2.toml configs/run_e.toml
.venv311/bin/python scripts/run_c_analysis.py configs/run_c.toml                # add --every-open for one line per open
```

The three scripts are read only, and each takes `--copies DIR` to read `DIR/run_x/btc_survivor.db` instead of the live run.

Write timestamps and negative numbers in inline code, such as `2026-10-11T20:41Z` or `-0.012`, because the documentation test reads a bare colon or minus sign in prose as a style error. The commands are described in [operations.md](operations.md) and the documentation rules in [CONTRIBUTING.md](../CONTRIBUTING.md#documentation-style).

## Run header

The settings below are fixed for the life of each run. Add a hardware entry whenever a run moves to another machine. The runs do not record a commit themselves, so the launch commits are read from the commit times. Runs A to C launched at `20:41` UTC, about a minute after `174718f` was committed at `20:40:38` UTC. Run D launched at `21:39` UTC from `ef648f5` with `configs/run_d.toml` not yet committed, and that file was committed a minute later in `351efc2`. Run E was added on 12 October 2026 by the owner's decision and launched at `21:29` UTC from `622cf49` with `configs/run_e.toml` not yet committed, and that file was committed a few minutes later in `e132242`. Run D2 was added on 13 October 2026 by the owner's decision and launched on the home PC at `15:48` UTC from `95bafef`, the commit that added `configs/run_d2.toml`, after `check_config_integrity.py` matched runs A to E there. The config text each run actually started with is stored in its `meta` table under `config_toml`.

| Setting | Run A | Run B | Run C | Run D | Run D2 | Run E |
| --- | --- | --- | --- | --- | --- | --- |
| Purpose | main run, the skill result | low fee, the same forced schedule at 5 bps | machinery, death and lineage | long horizon, exploratory | long horizon with stall tolerance, exploratory, what the strict stall rule costs run D | hypothetical frictionless, exploratory, the bots' own calls before costs |
| Config | `configs/run_a.toml` | `configs/run_b.toml` | `configs/run_c.toml` | `configs/run_d.toml` | `configs/run_d2.toml` | `configs/run_e.toml` |
| Start, UTC | `2026-10-11T20:41Z` | `2026-10-11T20:41Z` | `2026-10-11T20:41Z` | `2026-10-11T21:39Z` | `2026-10-13T15:48Z` | `2026-10-12T21:29Z` |
| First healthy minute, `launch_ts` in `meta` |  |  |  |  |  |  |
| Master seed | 12345 | 12345 | 12345 | 12345 | 12345 | 12345 |
| Fee profile | `retail_default`, 90 bps taker | `low_fee_tier`, 5 bps taker | `retail_default`, 90 bps taker | `low_fee_tier`, 5 bps taker | `low_fee_tier`, 5 bps taker | `hypothetical_zero`, 0 bps taker, no slippage, short charges or margin |
| Horizon | 30 minutes | 30 minutes | 30 minutes | 4 hours | 4 hours | 30 minutes |
| Feed stall limits, silence and clock skew | 5 and 2 seconds | 5 and 2 seconds | 5 and 2 seconds | 5 and 2 seconds | 30 and 5 seconds | 5 and 2 seconds |
| Rent | 1.16 dollars a day | 1.16 dollars a day | 1.16 dollars a day | 1.16 dollars a day | 1.16 dollars a day | 1.16 dollars a day |
| Start equity and notional | 10,000 and 500 dollars | 10,000 and 500 dollars | 500 and 500 dollars | 10,000 and 500 dollars | 10,000 and 500 dollars | 10,000 and 500 dollars |
| Learning rate and gene bounds | 0.0001, 0.00001 to 0.001 | 0.0001, 0.00001 to 0.001 | 0.0001, 0.00001 to 0.001 | 0.00003, 0.000003 to 0.0003 | 0.00003, 0.000003 to 0.0003 | 0.0001, 0.00001 to 0.001 |
| Population | 20 learners and 20 coin flip twins | 20 learners and 20 coin flip twins | 20 learners and 20 coin flip twins | 20 learners and 20 coin flip twins | 20 learners and 20 coin flip twins | 20 learners and 20 coin flip twins |
| Final day | 28 | 28 | 56 | 28, descriptive | 28, descriptive | 28, descriptive |
| Hardware at launch | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 | home PC, Ubuntu 26.04, Python 3.11.16 in `.venv311`, numpy 2.4.6 | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 |
| Later hardware, with the date of the move | home PC, Ubuntu 26.04, Python 3.11.16 in `.venv311`, numpy 2.4.6, stopped on the Mac at `2026-10-12T23:18Z`, healthy on the PC at `23:20Z` | the same, `23:21Z` and `23:23Z` | the same, `23:24Z` and `23:26Z` | the same, `23:27Z` and `23:29Z` |  | the same, `23:30Z` and `23:32Z` |
| Code commit at launch | `174718f` | `174718f` | `174718f` | `ef648f5`, config committed in `351efc2` | `95bafef` | `622cf49`, config committed in `e132242` |

Every setting is explained in [config.example.toml](../config.example.toml) and the runs in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria).

## Interruptions

The criteria count healthy days, so every pause pushes the final day later. Record each stretch the runs were not Healthy for longer than a few minutes, such as a sleeping Mac, a `clock_skew` pause, a feed outage, a restart or a move to the Pi. The `feed_events` table holds the exact times.

| Date, UTC | Runs | What happened | Healthy minutes lost | Positions flattened |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

Feed health and resume are described in [section 7](../PLAN.md#7-feed) and [section 13](../PLAN.md#13-checkpoints-resume-and-shutdown) of the plan, and the lessons from the Mac are in [operations.md](operations.md).

## Config integrity

No setting changes during a run, but a resume checks only the master seed, the population size and the twin setting. Before any restart, and at each write up, compare every config file with the text its run stored at launch in `meta` under `config_toml`, using `scripts/check_config_integrity.py` as in the commands above. It prints match when the file's text is the stored text, same settings when only comments or layout differ, logging only when only `run.log_level` differs, and otherwise every setting that differs with its launch value and the file's value. The exit status is 0 when every run matches, holds the same settings or differs only in `run.log_level`, 1 when any other setting differs, and 2 when a database or its stored config is missing. Record every check, and with any verdict other than match say what was done about it.

| Date, UTC | Reason | Run | Verdict | File sha256, first 16 | Stored sha256, first 16 | Settings that differ |
| --- | --- | --- | --- | --- | --- | --- |
| `2026-10-12` | after the external review, on copies of the databases | A | match | `49d2b8c10613b6d6` | `49d2b8c10613b6d6` | none |
| `2026-10-12` | after the external review, on copies of the databases | B | match | `0c6649fa030c2f3c` | `0c6649fa030c2f3c` | none |
| `2026-10-12` | after the external review, on copies of the databases | C | match | `0bcb21d5a7fb706e` | `0bcb21d5a7fb706e` | none |
| `2026-10-12` | after the external review, on copies of the databases | D | match | `1e15c38808d63549` | `1e15c38808d63549` | none |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

Why the check exists is in [honesty_and_limits.md](honesty_and_limits.md#settings-across-restarts), and the rules for restarting a run are in [operations.md](operations.md#starting-runs).

## Success criteria

The five criteria below were written down before launch. Criteria 1, 2 and 4 have pass rules, criterion 3 is reported without a threshold, and criterion 5 is context. The `evaluate` command counts only learner rows wherever twins also write rows. The definitions are in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria).

### 1. Skill, the headline

One confirmatory test, on run A at day 28. It scores the population forecast at one minute per horizon window, from one day after the first forecast, and passes when the correlation is above zero at the one sided 1 percent level, a Newey West t above 2.33. Runs B and C learn from the same market and are reported beside run A, not as separate tests. A squared error ratio below 1 favours the population, and a positive Diebold Mariano t does too.

| Run | Day | Date | Windows | Correlation | Newey West t | Squared error against zero | Diebold Mariano t against zero | Squared error against linear | Diebold Mariano t against linear | Linear correlation | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 |  |  |  |  |  |  |  |  |  | descriptive |
| A | 14 |  |  |  |  |  |  |  |  |  | descriptive |
| A | 28 |  |  |  |  |  |  |  |  |  |  |
| B | 7 |  |  |  |  |  |  |  |  |  | descriptive |
| B | 14 |  |  |  |  |  |  |  |  |  | descriptive |
| B | 28 |  |  |  |  |  |  |  |  |  | beside run A |
| C | 7 |  |  |  |  |  |  |  |  |  | descriptive |
| C | 14 |  |  |  |  |  |  |  |  |  | descriptive |
| C | 28 |  |  |  |  |  |  |  |  |  | beside run A |
| C | 56 |  |  |  |  |  |  |  |  |  | beside run A |
| D | 7 |  |  |  |  |  |  |  |  |  | exploratory |
| D | 14 |  |  |  |  |  |  |  |  |  | exploratory |
| D | 28 |  |  |  |  |  |  |  |  |  | exploratory |
| E | 7 |  |  |  |  |  |  |  |  |  | exploratory |
| E | 14 |  |  |  |  |  |  |  |  |  | exploratory |
| E | 28 |  |  |  |  |  |  |  |  |  | exploratory |

A failure at day 28 rules out, at 95 percent confidence, only correlations above the upper bound printed for that row, which goes in the table below. It is never read as evidence that the correlation is below about 0.1, nor below a detectable correlation, which describes the test's power and not this run's correlation, and it does not say that the market holds no edge, as [honesty_and_limits.md](honesty_and_limits.md#statistical-limits) explains.

#### Upper bound and detectable correlation

`evaluate` prints these under the Newey West t. The upper bound is one sided at 95 percent, the correlation plus 1.645 Newey West errors. The first detectable correlation is 3.61 times the realized Newey West error and the second assumes independent windows, 3.61 over the square root of the window count, and both are correlations the test would find with 90 percent power. Record whether the upper bound is above 0.1, because a failure whose bound is above 0.1 does not rule out a correlation of 0.1. The pass rule is unchanged.

| Run | Day | Date | Windows | Correlation | Newey West error | One sided 95 percent upper bound | Detectable from the Newey West error | Detectable if windows were independent | Upper bound above 0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 |  |  |  |  |  |  |  |  |
| A | 14 |  |  |  |  |  |  |  |  |
| A | 28 |  |  |  |  |  |  |  |  |
| B | 7 |  |  |  |  |  |  |  |  |
| B | 14 |  |  |  |  |  |  |  |  |
| B | 28 |  |  |  |  |  |  |  |  |
| C | 7 |  |  |  |  |  |  |  |  |
| C | 14 |  |  |  |  |  |  |  |  |
| C | 28 |  |  |  |  |  |  |  |  |
| C | 56 |  |  |  |  |  |  |  |  |
| D | 7 |  |  |  |  |  |  |  |  |
| D | 14 |  |  |  |  |  |  |  |  |
| D | 28 |  |  |  |  |  |  |  |  |
| E | 7 |  |  |  |  |  |  |  |  |
| E | 14 |  |  |  |  |  |  |  |  |
| E | 28 |  |  |  |  |  |  |  |  |

In the external review's simulation at a true correlation of 0.1, 87 percent of the failures under shared volatility clustering still had an upper bound above 0.1, as [honesty_and_limits.md](honesty_and_limits.md#statistical-limits) explains.

### 2. Beats coin flips

Each closed learner trade is paired with its twin's trade from the same decision, and the gross profit difference is averaged within 12 hour blocks. It passes when the mean of the block means is positive and its t is above the one sided 1 percent quantile of Student's t. It has about a third of criterion 1's power, so it is supporting evidence.

| Run | Day | Date | Pairs | Blocks | Mean of block means in dollars | t | Threshold | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 |  |  |  |  |  |  | descriptive |
| A | 14 |  |  |  |  |  |  | descriptive |
| A | 28 |  |  |  |  |  |  |  |
| B | 7 |  |  |  |  |  |  | descriptive |
| B | 14 |  |  |  |  |  |  | descriptive |
| B | 28 |  |  |  |  |  |  |  |
| C | 7 |  |  |  |  |  |  | descriptive |
| C | 14 |  |  |  |  |  |  | descriptive |
| C | 28 |  |  |  |  |  |  | descriptive |
| C | 56 |  |  |  |  |  |  |  |
| D | 28 |  |  |  |  |  |  | exploratory |
| E | 28 |  |  |  |  |  |  | exploratory |

The twins are described in [honesty_and_limits.md](honesty_and_limits.md#the-coin-flip-twins-and-their-power).

### 3. Survival

Deaths, causes and lifespans of learners and twins, with no pass threshold. Death days count days of healthy data since launch, while lifespans are wall clock time from birth.

| Run | Day | Date | Learner deaths | Learner causes | Learner mean lifespan in days | Learner death days, first, median and last | Twin deaths | Twin causes | Twin mean lifespan in days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 |  |  |  |  |  |  |  |  |
| A | 14 |  |  |  |  |  |  |  |  |
| A | 28 |  |  |  |  |  |  |  |  |
| B | 7 |  |  |  |  |  |  |  |  |
| B | 14 |  |  |  |  |  |  |  |  |
| B | 28 |  |  |  |  |  |  |  |  |
| C | 7 |  |  |  |  |  |  |  |  |
| C | 14 |  |  |  |  |  |  |  |  |
| C | 28 |  |  |  |  |  |  |  |  |
| C | 56 |  |  |  |  |  |  |  |  |
| D | 7 |  |  |  |  |  |  |  |  |
| D | 14 |  |  |  |  |  |  |  |  |
| D | 28 |  |  |  |  |  |  |  |  |
| E | 7 |  |  |  |  |  |  |  |  |
| E | 14 |  |  |  |  |  |  |  |  |
| E | 28 |  |  |  |  |  |  |  |  |

The five causes of death are explained in [evolution.md](evolution.md) and the ledger behind them in [execution_model.md](execution_model.md).

### 4. Machinery works

Run C only. It passes when, by day 56, at least one mutated child has been born and the lineage reaches generation 2. This checks the code path, not the strategy.

| Day | Date | Mutated births | Highest generation | Generation 1 from day | Generation 2 from day | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 7 |  |  |  |  |  | descriptive |
| 14 |  |  |  |  |  | descriptive |
| 28 |  |  |  |  |  | descriptive |
| 56 |  |  |  |  |  |  |

The `lineage BOT_ID` command shows any one chain back to its random ancestor, as described in [evolution.md](evolution.md).

### 5. Context

Buy and hold, the exposure matched benchmark and the cash baseline, read from the `benchmark` table at the same moment as the other rows. The living mean and median count only the learners alive at that moment, and every birth refills a slot with full start equity, so after deaths they are topped up with new money. The all bots figure adds living equity to the final equity of every dead learner and compares it with the capital given to every learner ever born, but that capital grows with every birth, and a dead learner's final equity is recorded after any negative cash is written off, so writeoffs are left out. Neither is the population's return, and the self financing value in its own section below is the figure to set against buy and hold and cash. None of these is a pass criterion.

`status` prints the all bots figure only for the present moment, so the query below gives the whole row at a past cutoff. Replace the timestamp with the cutoff `evaluate` printed for that day, set the start equity to 10,000 dollars for runs A, B, D and E or 500 for run C, and run it against a read only connection. Like `status` it counts learners only, and a learner born after the last equity snapshot counts at its start equity.

```sql
-- sqlite3 -readonly -header -column data/run_a/btc_survivor.db < context.sql
-- start_equity is 10000 for runs A, B, D and E and 500 for run C
WITH at AS (SELECT '2026-11-08T20:41:00.000Z' AS ts, 10000 AS start_equity),
snap AS (SELECT max(e.ts) AS ts FROM equity_snapshots e, at WHERE e.ts <= at.ts),
bench AS (SELECT * FROM benchmark WHERE ts = (SELECT max(b.ts) FROM benchmark b, at WHERE b.ts <= at.ts)),
learner AS (SELECT b.bot_id FROM bots b, at WHERE b.origin != 'control' AND b.born_at <= at.ts),
dead AS (SELECT d.bot_id, d.final_equity FROM deaths d JOIN learner USING (bot_id), at WHERE d.died_at <= at.ts),
living AS (SELECT coalesce(e.equity, at.start_equity) AS equity FROM learner l CROSS JOIN at
           LEFT JOIN equity_snapshots e ON e.bot_id = l.bot_id AND e.ts = (SELECT ts FROM snap)
           WHERE l.bot_id NOT IN (SELECT bot_id FROM dead))
SELECT bench.ts, round(btc_mid, 2) AS btc_mid, round(living_mean_equity, 2) AS living_mean,
  round(living_median_equity, 2) AS living_median, round(twin_mean_equity, 2) AS twin_mean,
  round(buy_hold_equity, 2) AS buy_hold, round(matched_equity, 2) AS matched, round(cash_equity, 2) AS cash,
  round((SELECT coalesce(sum(equity), 0) FROM living) + (SELECT coalesce(sum(final_equity), 0) FROM dead), 2)
    AS all_bots_equity,
  (SELECT count(*) FROM learner) * at.start_equity AS all_bots_capital
FROM bench, at;
```

| Run | Day | Date | BTC mid | Living mean equity | Living median equity | Twin mean equity | Buy and hold | Matched exposure | Cash baseline | All bots figure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 |  |  |  |  |  |  |  |  |  |
| A | 14 |  |  |  |  |  |  |  |  |  |
| A | 28 |  |  |  |  |  |  |  |  |  |
| B | 7 |  |  |  |  |  |  |  |  |  |
| B | 14 |  |  |  |  |  |  |  |  |  |
| B | 28 |  |  |  |  |  |  |  |  |  |
| C | 7 |  |  |  |  |  |  |  |  |  |
| C | 14 |  |  |  |  |  |  |  |  |  |
| C | 28 |  |  |  |  |  |  |  |  |  |
| C | 56 |  |  |  |  |  |  |  |  |  |
| D | 7 |  |  |  |  |  |  |  |  |  |
| D | 14 |  |  |  |  |  |  |  |  |  |
| D | 28 |  |  |  |  |  |  |  |  |  |
| E | 7 |  |  |  |  |  |  |  |  |  |
| E | 14 |  |  |  |  |  |  |  |  |  |
| E | 28 |  |  |  |  |  |  |  |  |  |

The benchmark columns are defined in [section 12 of the plan](../PLAN.md#12-storage-schema).

## Self financing value

This is the population's return, the figure to compare with buy and hold and cash. `scripts/population_value.py` values every learner ever born, a living one at its newest equity snapshot and a dead one at its final equity less any writeoff, and charges every replacement birth its start equity. It prints a summary at the newest snapshot and, once a whole healthy day has passed, a table with one row for the end of every healthy day and a last row for the newest snapshot. The row for day N is taken at the same benchmark minute that `evaluate --final-day N` uses as its cutoff. The value per slot is the value over the 20 founding slots, its return is against start equity, and buy and hold and cash are per slot from the `benchmark` row at that moment. The outside money is the most that the dead learners' residuals, less the cost of the births, ever fell below zero, the money the births needed from elsewhere. Twins are left out. Before the first death the value per slot equals the living mean, as it did in all four runs on 12 October 2026, so in runs A, B and D the two are expected to match throughout.

| Run | Day | Date | Learners ever born | Deaths | Writeoffs | Value per slot | Self financing return | Outside money | Buy and hold | Buy and hold return | Cash baseline | Cash return | Living mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 |  |  |  |  |  |  |  |  |  |  |  |  |
| A | 14 |  |  |  |  |  |  |  |  |  |  |  |  |
| A | 28 |  |  |  |  |  |  |  |  |  |  |  |  |
| B | 7 |  |  |  |  |  |  |  |  |  |  |  |  |
| B | 14 |  |  |  |  |  |  |  |  |  |  |  |  |
| B | 28 |  |  |  |  |  |  |  |  |  |  |  |  |
| C | 7 |  |  |  |  |  |  |  |  |  |  |  |  |
| C | 14 |  |  |  |  |  |  |  |  |  |  |  |  |
| C | 28 |  |  |  |  |  |  |  |  |  |  |  |  |
| C | 56 |  |  |  |  |  |  |  |  |  |  |  |  |
| D | 7 |  |  |  |  |  |  |  |  |  |  |  |  |
| D | 14 |  |  |  |  |  |  |  |  |  |  |  |  |
| D | 28 |  |  |  |  |  |  |  |  |  |  |  |  |
| E | 7 |  |  |  |  |  |  |  |  |  |  |  |  |
| E | 14 |  |  |  |  |  |  |  |  |  |  |  |  |
| E | 28 |  |  |  |  |  |  |  |  |  |  |  |  |

The definition is in [`self_financing.py`](../src/btc_survivor/cli/self_financing.py), and why the living mean is not enough is in [honesty_and_limits.md](honesty_and_limits.md#population-averages-and-the-self-financing-value).

## Outcomes fixed by arithmetic

These follow from fees, notional, start equity and the forced timer, so they were written down before launch. They check the accounting, they are judged as matched or not matched, and they are not findings about trading. A round trip costs 9.10 dollars at 90 bps and 0.60 dollars at 5 bps, and a quiet bot burns 19.36 or 2.36 dollars a day. The day 56 column applies to run C only.

| Run | Expected before launch | Day 7 | Day 14 | Day 28 | Day 56 | Matched |
| --- | --- | --- | --- | --- | --- | --- |
| A | Every learner open is forced |  |  |  |  |  |
| A | No learner dies by day 28 |  |  |  |  |  |
| A | Founders' mean equity within about 20 dollars of 10,000 minus rent and 9.10 dollars per round trip made |  |  |  |  |  |
| B | Almost every learner open is forced |  |  |  |  |  |
| B | No learner dies |  |  |  |  |  |
| C | Every learner dies of fees |  |  |  |  |  |
| C | The first wave of deaths falls between days 24 and 28 |  |  |  |  |  |
| C | The second wave falls between days 48 and 56, and generation 2 appears in it |  |  |  |  |  |
| C | Parent choice mostly reflects market noise, because costs swamp any plausible skill |  |  |  |  |  |
| D | No learner dies, at a quiet burn of about 2.36 dollars a day plus short funding |  |  |  |  |  |
| E | No learner dies, its death line is about one day of rent |  |  |  |  |  |

How many of run E's learner opens are its own rather than forced is not fixed by arithmetic, because it depends on the predictions its bots make. It is expected from the predictions runs A to C recorded, about 97 percent of which would pass run E's gate, so it is recorded here as a descriptive count and never judged as matched.

| Run | Expected from recorded predictions | Day 7 | Day 14 | Day 28 |
| --- | --- | --- | --- | --- |
| E | Almost every learner open is its own rather than forced, recorded as own opens and forced opens |  |  |  |

The arithmetic is worked through in [honesty_and_limits.md](honesty_and_limits.md#outcomes-fixed-by-arithmetic) and in [section 3 of the plan](../PLAN.md#3-what-to-expect-from-the-default-experiment).

## Run C analyses

These come from `scripts/run_c_analysis.py`, which reads the ledger read only, and none of them is a success criterion. They were added on 12 October 2026 after the external review, before any death. The script has no cutoff and reads everything recorded so far, so run it at the write up moment, or on copies taken then with `--copies DIR`, and copy the healthy days from its first line into the tables. The reasons behind each analysis are in [honesty_and_limits.md](honesty_and_limits.md#run-c-trades-on-unreserved-leverage).

### Notional over equity at each open

Equity before an open is the bot's newest snapshot, at most five minutes old, less the fees of any fill since, and after the open it also loses the open's fee. From the config alone, a bot that trades only when forced and never gains or loses on the market, with its forced opens 12 healthy hours apart as the engine spaces them, passes 2 times its equity after the open on day 13.8, 5 times on day 21.3 and 10 times on day 23.8, and its last open, on day 25.8, is 33.2 times before the fee and 47.5 times after. A bot still alive carries at most 48.7 times, at the death threshold of 10.26 dollars. The first 13 learner opens and 13 twin opens of run C, by 12 October 2026, were at most 1.00 times before the fee and 1.01 times after.

| Run | Day | Date | Healthy days printed | Learner opens | Largest learner leverage before the fee | Largest learner leverage after the fee | Learner opens above 5 times | Twin opens | Largest twin leverage after the fee |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 7 |  |  |  |  |  |  |  |  |
| C | 14 |  |  |  |  |  |  |  |  |
| C | 28 |  |  |  |  |  |  |  |  |
| C | 56 |  |  |  |  |  |  |  |  |
| A | 28 |  |  |  |  |  |  |  |  |
| B | 28 |  |  |  |  |  |  |  |  |
| D | 28 |  |  |  |  |  |  |  |  |
| E | 28 |  |  |  |  |  |  |  |  |

Runs A, B, D and E are expected to stay near 0.05 times, as they were on 12 October 2026, and the mechanism is in [execution_model.md](execution_model.md#no-principal-is-reserved).

### Death days with and without a margin cap

Under the illustrative cap a bot stops trading at its first open above 5 times its equity and lives on rent until it crosses its death threshold. From the config alone a quiet run C bot dies on day 25.8 without a cap and on day 92.7 with it. For runs A, B and D the same arithmetic puts a quiet bot's death on day 516.3, 4,236.8 and 4,237.0, which agrees with the 516 and 4,237 days in evolution.md, and for run E, which pays almost nothing but rent, on day 8,619.8, so none of them can see a death within its window. The external review's verifier spaced forced opens 12.25 hours apart, half a horizon more than the engine does, which moves every one of these days later, and `--period-hours 12.25` reproduces his figures.

| Day | Date | Learner deaths | Recorded death day, mean and standard deviation | Death day with a 5 times cap, mean and standard deviation | Deaths the cap changes | Deaths whose last open was above 5 times | Living learners past 5 times | Twin deaths | Twin recorded and capped death days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 |  |  |  |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |  |  |  |
| 28 |  |  |  |  |  |  |  |  |  |
| 56 |  |  |  |  |  |  |  |  |  |

The cap is a sensitivity check and was never part of the runs, as [execution_model.md](execution_model.md#no-principal-is-reserved) explains.

### Spread of death days against skill

The Monte Carlo is seeded with 7 and depends only on the config, so its rows are filled in already. It follows 20,000 quiet bots whose direction has the given correlation with the next 30 minute move, with a spread of 32 bps, run C's costs, its stop and its death rule, and forced opens 12 healthy hours apart as the engine spaces them. The verifier's 12.25 hours gave means of 26.32, 26.66 and 27.36 with nearly the same spread. The founders' recorded death days go in the second table.

| Forecast correlation | Mean death day | Standard deviation in days | Shift against no skill in days | Outlives a bot with no skill |
| --- | --- | --- | --- | --- |
| 0 | 25.83 | 0.60 | none | 50 percent of the time |
| 0.1 | 26.16 | 0.61 | 0.33 | 65 percent of the time |
| 0.3 | 26.85 | 0.62 | 1.02 | 88 percent of the time |

| Day | Date | Founder learner deaths | Founder death day, mean | Founder death day, standard deviation |
| --- | --- | --- | --- | --- |
| 28 |  |  |  |  |
| 56 |  |  |  |  |

A founder death spread near 0.6 days is what costs and noise alone produce, as [honesty_and_limits.md](honesty_and_limits.md#selection-and-why-evolution-is-not-tested) explains.

### Parents' skill against the eligible pool

At each mutated birth the parent's rolling skill correlation, stored in the `lineage` row, is set beside the skill of the other learners that could have been parents then, read from their newest snapshots. That pool is what the tournament drew from, learners at least 24 healthy hours old with at least 20 closed trades and not dying in that minute, less the parent. Parents are always old and trained, so a comparison with every living learner would credit selection with age and training, and the mean over every other living learner is printed beside it only for reference. The t treats births as independent, which they are not, so it is descriptive, and a null result is expected because parent choice in run C mostly reflects market noise.

| Day | Date | Mutated births | Births compared | Parent skill mean | Eligible pool skill mean | Difference | t, births treated as independent | Parent above the eligible median | Every other living learner, mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28 |  |  |  |  |  |  |  |  |  |
| 56 |  |  |  |  |  |  |  |  |  |

Parent choice and the lineage table are described in [evolution.md](evolution.md#tournament).

## Generations

One table per generation, with a row per run at each write up. Generation 0 holds the founders and every newcomer. A newcomer is a fresh random bot that replaces a dead learner with probability 0.10, or whenever no living learner has the 24 healthy hours and 20 closed trades a parent needs, as in [section 11 of the plan](../PLAN.md#11-evolution). In run C the founders die within a few days of each other, while their children need about 10 days to close 20 trades, so late in the first wave many replacements are likely to be newcomers rather than generation 1. Runs A, B, D and E are not expected to have any deaths, so their later generation tables stay empty unless a death happens. Born, alive and dead count learners only. Mean equity and mean skill correlation are over that generation's living learners, and twin mean equity is over the twins paired with them. The last two columns are the living mean equity of the whole population minus buy and hold and minus the cash baseline, in dollars. They count living learners only and are topped up at every birth, so they are not the population's return, which is in the self financing table above.

This query prints every generation's row at one moment. Replace the timestamp with the cutoff `evaluate` printed for that day, set the start equity as for the context query above, and run it against a read only connection. A learner counts as alive when it was born by the cutoff and had not died by then, and one born after the last equity snapshot counts at its start equity.

```sql
-- sqlite3 -readonly -header -column data/run_c/btc_survivor.db < generations.sql
-- start_equity is 10000 for runs A, B, D and E and 500 for run C
WITH at AS (SELECT '2026-11-08T20:41:00.000Z' AS ts, 500 AS start_equity),
snap AS (SELECT max(e.ts) AS ts FROM equity_snapshots e, at WHERE e.ts <= at.ts),
bench AS (SELECT * FROM benchmark WHERE ts = (SELECT max(b.ts) FROM benchmark b, at WHERE b.ts <= at.ts)),
learner AS (SELECT bots.* FROM bots, at WHERE origin != 'control' AND born_at <= at.ts),
dead AS (SELECT l.generation, d.bot_id, d.cause FROM deaths d JOIN learner l USING (bot_id), at WHERE d.died_at <= at.ts),
living AS (SELECT l.generation, l.slot, coalesce(e.equity, at.start_equity) AS equity, e.skill_corr
           FROM learner l CROSS JOIN at
           LEFT JOIN equity_snapshots e ON e.bot_id = l.bot_id AND e.ts = (SELECT ts FROM snap)
           WHERE l.bot_id NOT IN (SELECT bot_id FROM dead)),
twin AS (SELECT v.generation, e.equity FROM living v
         JOIN bots t ON t.origin = 'control' AND t.slot = v.slot
         JOIN equity_snapshots e ON e.bot_id = t.bot_id JOIN snap ON e.ts = snap.ts)
SELECT g.generation,
  (SELECT count(*) FROM learner WHERE generation = g.generation) AS born,
  (SELECT count(*) FROM living WHERE generation = g.generation) AS alive,
  (SELECT count(*) FROM dead WHERE generation = g.generation) AS dead,
  (SELECT round(avg(equity), 2) FROM living WHERE generation = g.generation) AS mean_equity,
  (SELECT round(avg(equity), 2) FROM twin WHERE generation = g.generation) AS twin_mean_equity,
  (SELECT cause FROM dead WHERE generation = g.generation GROUP BY cause ORDER BY count(*) DESC LIMIT 1) AS top_cause,
  (SELECT round(avg(skill_corr), 4) FROM living WHERE generation = g.generation) AS mean_skill_corr,
  (SELECT round(living_mean_equity - buy_hold_equity, 2) FROM bench) AS living_mean_against_buy_hold,
  (SELECT round(living_mean_equity - cash_equity, 2) FROM bench) AS living_mean_against_cash
FROM (SELECT DISTINCT generation FROM learner) g ORDER BY g.generation;
```

The tables behind the query are listed in [section 12 of the plan](../PLAN.md#12-storage-schema).

### Generation 0

| Run | Day | Date | Generation | Born | Alive | Dead | Mean equity | Twin mean equity | Top cause of death | Mean skill correlation | Living mean against buy and hold | Living mean against cash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 |  | 0 |  |  |  |  |  |  |  |  |  |
| A | 14 |  | 0 |  |  |  |  |  |  |  |  |  |
| A | 28 |  | 0 |  |  |  |  |  |  |  |  |  |
| B | 7 |  | 0 |  |  |  |  |  |  |  |  |  |
| B | 14 |  | 0 |  |  |  |  |  |  |  |  |  |
| B | 28 |  | 0 |  |  |  |  |  |  |  |  |  |
| C | 7 |  | 0 |  |  |  |  |  |  |  |  |  |
| C | 14 |  | 0 |  |  |  |  |  |  |  |  |  |
| C | 28 |  | 0 |  |  |  |  |  |  |  |  |  |
| C | 56 |  | 0 |  |  |  |  |  |  |  |  |  |
| D | 7 |  | 0 |  |  |  |  |  |  |  |  |  |
| D | 14 |  | 0 |  |  |  |  |  |  |  |  |  |
| D | 28 |  | 0 |  |  |  |  |  |  |  |  |  |
| E | 7 |  | 0 |  |  |  |  |  |  |  |  |  |
| E | 14 |  | 0 |  |  |  |  |  |  |  |  |  |
| E | 28 |  | 0 |  |  |  |  |  |  |  |  |  |

Founders and newcomers are described in [evolution.md](evolution.md).

### Generation 1

| Run | Day | Date | Generation | Born | Alive | Dead | Mean equity | Twin mean equity | Top cause of death | Mean skill correlation | Living mean against buy and hold | Living mean against cash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 7 |  | 1 |  |  |  |  |  |  |  |  |  |
| C | 14 |  | 1 |  |  |  |  |  |  |  |  |  |
| C | 28 |  | 1 |  |  |  |  |  |  |  |  |  |
| C | 56 |  | 1 |  |  |  |  |  |  |  |  |  |

Generation 1 is expected in run C from the first wave of deaths, between days 24 and 28, as [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) predicts.

### Generation 2

| Run | Day | Date | Generation | Born | Alive | Dead | Mean equity | Twin mean equity | Top cause of death | Mean skill correlation | Living mean against buy and hold | Living mean against cash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 28 |  | 2 |  |  |  |  |  |  |  |  |  |
| C | 56 |  | 2 |  |  |  |  |  |  |  |  |  |

Generation 2 is expected in run C in the second wave, between days 48 and 56. Copy this table for any later generation. Mutation and parent choice are described in [evolution.md](evolution.md).

## Summary

Write a few plain sentences per write up. Say whether criterion 1 passed, its one sided 95 percent upper bound and both correlations it could detect, the self financing value against buy and hold and cash, whether the outcomes fixed by arithmetic matched, whether every config still matched its stored text, and anything that went wrong with the runs. A failed test is reported with its upper bound and never as a correlation below about 0.1. Version 1 does not test whether evolution helps, so run C's deaths and lineage are reported as a check of the machinery. A null result is written up as plainly as a pass, as [honesty_and_limits.md](honesty_and_limits.md#a-null-result-is-a-real-finding) explains.
