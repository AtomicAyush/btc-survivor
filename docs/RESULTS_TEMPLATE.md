# Results template

This file holds the tables for writing up the four live runs. Results are written up at days 7, 14 and 28 of healthy data, and at day 56 for run C. Days 7 and 14 are descriptive and only the final day counts, which is day 28 for runs A and B and day 56 for run C. Run D is exploratory and every row for it is descriptive. Every run is reported whatever it shows, no setting changes during a run, and the best single bot is never evidence on its own. The criteria were fixed before launch in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria), and what they can and cannot show is in [honesty_and_limits.md](honesty_and_limits.md).

Most numbers come straight from the `evaluate` command. With `--final-day` it freezes every number at the moment the run reached that many days of healthy data, so running it again later gives the same rows. Use the virtual environment the run itself uses, `.venv311` on the Mac or `.venv` on the Pi.

```bash
.venv311/bin/btc_survivor --config configs/run_a.toml evaluate --final-day 7     # the day 7 rows, descriptive
.venv311/bin/btc_survivor --config configs/run_a.toml evaluate --final-day 28    # run A at its final day
.venv311/bin/btc_survivor --config configs/run_c.toml evaluate --final-day 56    # run C at its final day
.venv311/bin/btc_survivor --config configs/run_a.toml status --twins            # living bots and the all bots figure, now only
.venv311/bin/btc_survivor --config configs/run_a.toml export --out ~/btc_exports/run_a  # every table as CSV, outside the repository
```

Write timestamps and negative numbers in inline code, such as `2026-10-11T20:41Z` or `-0.012`, because the documentation test reads a bare colon or minus sign in prose as a style error. The commands are described in [operations.md](operations.md) and the documentation rules in [CONTRIBUTING.md](../CONTRIBUTING.md#documentation-style).

## Run header

The settings below are fixed for the life of each run. Add a hardware entry whenever a run moves to another machine. The runs do not record a commit themselves, so the launch commits are read from the commit times. Runs A to C launched at `20:41` UTC, about a minute after `174718f` was committed at `20:40:38` UTC. Run D launched at `21:39` UTC from `ef648f5` with `configs/run_d.toml` not yet committed, and that file was committed a minute later in `351efc2`. The config text each run actually started with is stored in its `meta` table under `config_toml`.

| Setting | Run A | Run B | Run C | Run D |
| --- | --- | --- | --- | --- |
| Purpose | main run, the skill result | low fee, the same forced schedule at 5 bps | machinery, death and lineage | long horizon, exploratory |
| Config | `configs/run_a.toml` | `configs/run_b.toml` | `configs/run_c.toml` | `configs/run_d.toml` |
| Start, UTC | `2026-10-11T20:41Z` | `2026-10-11T20:41Z` | `2026-10-11T20:41Z` | `2026-10-11T21:39Z` |
| First healthy minute, `launch_ts` in `meta` |  |  |  |  |
| Master seed | 12345 | 12345 | 12345 | 12345 |
| Fee profile | `retail_default`, 90 bps taker | `low_fee_tier`, 5 bps taker | `retail_default`, 90 bps taker | `low_fee_tier`, 5 bps taker |
| Horizon | 30 minutes | 30 minutes | 30 minutes | 4 hours |
| Rent | 1.16 dollars a day | 1.16 dollars a day | 1.16 dollars a day | 1.16 dollars a day |
| Start equity and notional | 10,000 and 500 dollars | 10,000 and 500 dollars | 500 and 500 dollars | 10,000 and 500 dollars |
| Learning rate and gene bounds | 0.0001, 0.00001 to 0.001 | 0.0001, 0.00001 to 0.001 | 0.0001, 0.00001 to 0.001 | 0.00003, 0.000003 to 0.0003 |
| Population | 20 learners and 20 coin flip twins | 20 learners and 20 coin flip twins | 20 learners and 20 coin flip twins | 20 learners and 20 coin flip twins |
| Final day | 28 | 28 | 56 | 28, descriptive |
| Hardware at launch | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 | MacBook, macOS, Python 3.11 in `.venv311`, numpy 2.4.6 |
| Later hardware, with the date of the move |  |  |  |  |
| Code commit at launch | `174718f` | `174718f` | `174718f` | `ef648f5`, config committed in `351efc2` |

Every setting is explained in [config.example.toml](../config.example.toml) and the runs in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria).

## Interruptions

The criteria count healthy days, so every pause pushes the final day later. Record each stretch the runs were not Healthy for longer than a few minutes, such as a sleeping Mac, a `clock_skew` pause, a feed outage, a restart or a move to the Pi. The `feed_events` table holds the exact times.

| Date, UTC | Runs | What happened | Healthy minutes lost | Positions flattened |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

Feed health and resume are described in [section 7](../PLAN.md#7-feed) and [section 13](../PLAN.md#13-checkpoints-resume-and-shutdown) of the plan, and the lessons from the Mac are in [operations.md](operations.md).

## Success criteria

The five criteria below were written down before launch. Criteria 1, 2 and 4 have pass rules, criterion 3 is reported without a threshold, and criterion 5 is context. The `evaluate` command counts only learner rows wherever twins also write rows. The definitions are in [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria).

### 1. Skill, the headline

One confirmatory test, on run A at day 28. It scores the population forecast at one minute per horizon window, from one day after the first forecast, and passes when the correlation is above zero at the one sided 1 percent level, a Newey West t above 2.33. Runs B and C learn from the same market and are reported beside run A, not as separate tests. A squared error ratio below 1 favours the population, and a positive Diebold Mariano t does too.

| Run | Day | Date | Windows | Correlation | Newey West t | Detectable correlation | Squared error against zero | Diebold Mariano t against zero | Squared error against linear | Diebold Mariano t against linear | Linear correlation | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 |  |  |  |  |  |  |  |  |  |  | descriptive |
| A | 14 |  |  |  |  |  |  |  |  |  |  | descriptive |
| A | 28 |  |  |  |  |  |  |  |  |  |  |  |
| B | 7 |  |  |  |  |  |  |  |  |  |  | descriptive |
| B | 14 |  |  |  |  |  |  |  |  |  |  | descriptive |
| B | 28 |  |  |  |  |  |  |  |  |  |  | beside run A |
| C | 7 |  |  |  |  |  |  |  |  |  |  | descriptive |
| C | 14 |  |  |  |  |  |  |  |  |  |  | descriptive |
| C | 28 |  |  |  |  |  |  |  |  |  |  | beside run A |
| C | 56 |  |  |  |  |  |  |  |  |  |  | beside run A |
| D | 7 |  |  |  |  |  |  |  |  |  |  | exploratory |
| D | 14 |  |  |  |  |  |  |  |  |  |  | exploratory |
| D | 28 |  |  |  |  |  |  |  |  |  |  | exploratory |

A failure at day 28 means the forecast's correlation was below about the detectable correlation printed for that row, not that the market holds no edge, as [honesty_and_limits.md](honesty_and_limits.md#statistical-limits) explains.

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

Buy and hold, the exposure matched benchmark and the cash baseline, read from the `benchmark` table at the same moment as the other rows. The all bots figure adds living equity to the final equity of every dead learner and compares it with the capital given to every learner ever born. None of these is a pass criterion.

`status` prints the all bots figure only for the present moment, so the query below gives the whole row at a past cutoff. Replace the timestamp with the cutoff `evaluate` printed for that day, set the start equity to 10,000 dollars for runs A, B and D or 500 for run C, and run it against a read only connection. Like `status` it counts learners only, and a learner born after the last equity snapshot counts at its start equity.

```sql
-- sqlite3 -readonly -header -column data/run_a/btc_survivor.db < context.sql
-- start_equity is 10000 for runs A, B and D and 500 for run C
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

The benchmark columns are defined in [section 12 of the plan](../PLAN.md#12-storage-schema).

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
| C | Parent choice reflects market noise, because costs swamp any skill |  |  |  |  |  |
| D | No learner dies, at a quiet burn of about 2.36 dollars a day plus short funding |  |  |  |  |  |

The arithmetic is worked through in [honesty_and_limits.md](honesty_and_limits.md#outcomes-fixed-by-arithmetic) and in [section 3 of the plan](../PLAN.md#3-what-to-expect-from-the-default-experiment).

## Generations

One table per generation, with a row per run at each write up. Generation 0 holds the founders and every newcomer. A newcomer is a fresh random bot that replaces a dead learner with probability 0.10, or whenever no living learner has the 24 healthy hours and 20 closed trades a parent needs, as in [section 11 of the plan](../PLAN.md#11-evolution). In run C the founders die within a few days of each other, while their children need about 10 days to close 20 trades, so late in the first wave many replacements are likely to be newcomers rather than generation 1. Runs A, B and D are not expected to have any deaths, so their later generation tables stay empty unless a death happens. Born, alive and dead count learners only. Mean equity and mean skill correlation are over that generation's living learners, and twin mean equity is over the twins paired with them. The last two columns are the whole population's living mean equity minus buy and hold and minus the cash baseline, in dollars, so every generation row also shows where the population stands.

This query prints every generation's row at one moment. Replace the timestamp with the cutoff `evaluate` printed for that day, set the start equity as for the context query above, and run it against a read only connection. A learner counts as alive when it was born by the cutoff and had not died by then, and one born after the last equity snapshot counts at its start equity.

```sql
-- sqlite3 -readonly -header -column data/run_c/btc_survivor.db < generations.sql
-- start_equity is 10000 for runs A, B and D and 500 for run C
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
  (SELECT round(living_mean_equity - buy_hold_equity, 2) FROM bench) AS against_buy_hold,
  (SELECT round(living_mean_equity - cash_equity, 2) FROM bench) AS against_cash
FROM (SELECT DISTINCT generation FROM learner) g ORDER BY g.generation;
```

The tables behind the query are listed in [section 12 of the plan](../PLAN.md#12-storage-schema).

### Generation 0

| Run | Day | Date | Generation | Born | Alive | Dead | Mean equity | Twin mean equity | Top cause of death | Mean skill correlation | Population against buy and hold | Population against cash |
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

Founders and newcomers are described in [evolution.md](evolution.md).

### Generation 1

| Run | Day | Date | Generation | Born | Alive | Dead | Mean equity | Twin mean equity | Top cause of death | Mean skill correlation | Population against buy and hold | Population against cash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 7 |  | 1 |  |  |  |  |  |  |  |  |  |
| C | 14 |  | 1 |  |  |  |  |  |  |  |  |  |
| C | 28 |  | 1 |  |  |  |  |  |  |  |  |  |
| C | 56 |  | 1 |  |  |  |  |  |  |  |  |  |

Generation 1 is expected in run C from the first wave of deaths, between days 24 and 28, as [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) predicts.

### Generation 2

| Run | Day | Date | Generation | Born | Alive | Dead | Mean equity | Twin mean equity | Top cause of death | Mean skill correlation | Population against buy and hold | Population against cash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 28 |  | 2 |  |  |  |  |  |  |  |  |  |
| C | 56 |  | 2 |  |  |  |  |  |  |  |  |  |

Generation 2 is expected in run C in the second wave, between days 48 and 56. Copy this table for any later generation. Mutation and parent choice are described in [evolution.md](evolution.md).

## Summary

Write a few plain sentences per write up. Say whether criterion 1 passed, the correlation it could detect, whether the outcomes fixed by arithmetic matched, and anything that went wrong with the runs. A null result is written up as plainly as a pass, as [honesty_and_limits.md](honesty_and_limits.md#a-null-result-is-a-real-finding) explains.
