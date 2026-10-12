# Execution model

This page describes how a paper order is priced and what every bot pays. A bot trades a fixed 500 dollar notional against the live Coinbase book, after a delay, and pays a taker fee, extra slippage, a short opening charge and funding when short, and rent in every healthy minute. The rule from the brief is that the simulation must never be kinder than reality. Where it still is, this page says so plainly. The code is in [`broker.py`](../src/btc_survivor/broker.py) and [`account.py`](../src/btc_survivor/account.py), and the decisions behind it are in [section 10 of the plan](../PLAN.md#10-execution-and-accounting).

## Market data source

The feed is the Coinbase Advanced Trade websocket at `wss://advanced-trade-ws.coinbase.com`, with the `level2`, `market_trades` and `heartbeats` channels on one connection and no credentials. The brief named the Coinbase Exchange feed instead. When both were probed without keys at about three in the morning UTC on 11 October 2026, the Exchange feed rejected `level2`, `level3` and `full` with a reply that these channels now require authentication. Its keyless `level2_batch` channel sends a full snapshot and then updates batched every 50 milliseconds, but carries no sequence number, so a dropped update would silently corrupt the local book. That fails the brief's requirement to detect gaps.

On Advanced Trade the three channels the project needs are keyless, and every message on a connection carries `sequence_num`, one counter shared by all channels on that connection. In the probe it ran from 0 to 1118 with no gaps across 1,119 messages. Any jump means a message was lost, so the book is thrown away and the client reconnects for a fresh snapshot. A jump in trade ids while `sequence_num` stays contiguous is logged as a `trade_gap`, because that minute's volume may be undercounted, but it does not resync the book. Coinbase notes that keyless connections may be less reliable than authenticated ones, so reconnects are expected, logged and tested, and each one discards up to one horizon of pending labels. The project never asks for or creates keys. A Kraken adapter with a checksummed depth 25 book is the documented fallback and is not built.

What the probe measured on BTC USD in a quiet weekend hour.

* The `level2` snapshot is the full book, about 40,800 levels and 4.5 MB of JSON in one message, so the client raises the message size limit to 16 MiB.
* Steady state was 17 messages and about 35 price level changes per second, and trades arrived at 1.45 per second.
* The spread was one cent, which is 0.0012 bps, and walking the book for 500 dollars cost about 0.0006 bps beyond the best price.
* The median one way feed latency was 17 milliseconds on `level2` and about 95 milliseconds on the batched `market_trades` channel.

See [section 2.1 of the plan](../PLAN.md#21-which-keyless-feed-channels-exist-right-now-question-1) and the [Coinbase Advanced Trade WebSocket channels](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-channels) and [Coinbase Exchange WebSocket channels](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels) documentation.

## The local book

`feed/book.py` keeps the whole book, not only the top 20 levels, because a cancel at the top pulls deeper levels into the top 20. Each side is a dict from price to size plus a sorted list of prices kept with `bisect`, and asks are stored negated so that both best prices sit at the tail of their list, where inserts and deletes are cheap. Measured on the development Mac, a level change costs 0.2 to 0.6 microseconds and reading the top 20 costs 1.6 microseconds. A snapshot replaces the book, and each update sets a level to an absolute size, where size zero removes it. Removing a level that is not in the book is ignored, because Coinbase sends about one such delete a second and they are harmless.

The book exposes the best bid, the best ask, the mid, the spread in bps, the top `book_depth_levels` levels, which is 20, and the walk described below. A book is usable only when both sides exist and the best bid is below the best ask. A crossed or one sided book is thrown away and the client reconnects, and until a fresh snapshot arrives and a full clean minute passes, nothing trades. A Coinbase auction can legitimately cross the book, and the bots then stay paused until it clears.

One book serves every use. The minute mid at each boundary feeds the features and the labels, the spread and the top 10 levels feed two features, and the walk prices the cost estimate, the marking of open positions, the stop loss and every fill. See [section 7 of the plan](../PLAN.md#7-feed) and [`feed/book.py`](../src/btc_survivor/feed/book.py).

## From decision to fill

A decision at minute boundary M creates an order with the book mid at M as its reference mid. An opening leg buys or sells `notional / reference mid` BTC, rounded to 8 decimals, and a closing leg always uses the held quantity. A flip is one order with a closing leg and then an opening leg. The order falls due at D, which is M plus `latency_ms`, 1000, plus a jitter drawn uniformly from the whole milliseconds 0 to `latency_jitter_ms`, 300, with the bot's own seeded generator. Every order therefore falls due 1.0 to 1.3 seconds after its decision, and fills at the first book update after that. The measured feed delay is far shorter, and order round trip time was not measured, so this is deliberately slower than the data.

The order is settled right after the engine applies the first book update, a snapshot or a delta, whose receive time is at or after D, priced from the book as it stands then. That follows the brief's rule, the first book snapshot at or after t plus latency, to the letter. When several orders are due at once they settle earliest due first, and ties settle in the order they were sent. If the feed leaves Healthy before such an update arrives, every waiting order is rejected with reason `feed_not_healthy`. Other rejections are `insufficient_depth` when the top 20 levels cannot fill the leg, `book_invalid`, `no_position`, `already_open`, `zero_quantity`, and `restart` for an order restored from a checkpoint. Every rejection is a row in the `fills` table with status `rejected`.

A rejected decision is not retried, and the bot decides again at its next phase minute. Closes that must happen are retried. A stop is checked again at the next healthy minute, a death close is resent every healthy minute through the dying list until it fills, and a rejected gap flatten is sent again at the next healthy minute whatever that gap.

Open positions are marked every healthy minute by walking the exit side of the book for the held quantity, which is the price the bot would actually receive if it closed. If the adverse move from the entry fill reaches `stop_loss_pct`, which is 1 percent, the bot sends a market close through the same broker, the trade is marked as a stop, and the bot does not decide again until its next phase minute. See [section 6.4 of the plan](../PLAN.md#64-order-fills), [`engine.py`](../src/btc_survivor/engine.py) and [`tests/test_engine.py`](../tests/test_engine.py).

## The book walk

`walk(side, qty)` fills a quantity against the visible top 20 levels. A buy consumes asks from the best price upward and a sell consumes bids from the best price downward. It returns the average price across the consumed levels, the number of levels consumed and any quantity left unfilled. If anything is left unfilled the order is rejected and logged, because the brief allows no fill beyond the visible depth. Rounding residue below one trillionth of a coin counts as filled.

Legs of one order that trade on the same side share the depth. A flip from long to short sells twice into the bids, so the opening leg pays the deeper levels the closing leg left behind. Separate orders share nothing, which is the subject of the last section.

```
asks   100.00 x 0.3   100.10 x 0.5   100.20 x 1.0
buy 0.6   ->   0.3 at 100.00 and 0.3 at 100.10   ->   average 100.05 over 2 levels
buy 2.0   ->   only 1.8 visible                  ->   rejected, insufficient_depth
```

The cost estimate, the marking of positions and the stop loss all use the same walk, so a thin book raises all of them together. See [`feed/book.py`](../src/btc_survivor/feed/book.py) and [`tests/test_book.py`](../tests/test_book.py).

## Slippage and the fill price

The average book price then moves against the bot by `extra_slippage_bps`, which is 1 bp per leg. That is far more than the book walk produced at 500 dollars, about 0.0006 bps, so it errs on the harsh side. The taker fee is then charged on the filled notional at the final price.

```
sign             = +1 for a buy, -1 for a sell
fill_price       = book_avg_price * (1 + sign * extra_slippage_bps / 1e4)
fee              = taker_fee_bps / 1e4 * qty * fill_price
slippage_bps     = sign * (fill_price - ref_mid) / ref_mid * 1e4      positive means against the bot
```

The recorded slippage is measured against the reference mid at the decision, so it includes half the spread, the book walk, the extra 1 bp and whatever the price did during the delay. That last part is a random move and can help a fill as well as hurt it. In the toy book above, with a best bid of 99.90 and so a reference mid of 99.95, buying 0.6 fills at 100.060005 and records 11.0 bps of slippage, about 5 bps of half spread, 5 bps of walk and 1 bp extra. Each fill row stores the reference mid, the average book price, the final price, the slippage in bps, the fee, the latency used, the levels consumed, the status and the reason. See [`broker.py`](../src/btc_survivor/broker.py) and [`tests/test_broker.py`](../tests/test_broker.py).

## Fee profiles

Every fill pays the taker fee of the active profile, chosen by `fee_profile`. Each profile is a `taker_fee_bps_` key in the `[costs]` section, so a new scenario is one new key. All three values are marked as placeholders in [config.example.toml](../config.example.toml) and must be checked against the published schedule again before any new run.

| Profile | Taker fee | What it matches | Source and date |
| --- | --- | --- | --- |
| `retail_default` | 90 bps | Coinbase US Intro tier, no 30 day volume, the rate a new US account pays | the public Coinbase Advanced fee tiers page, read without signing in on 10 October 2026, a rate in force since 16 September 2026 |
| `active_retail` | 30 bps | Coinbase US Advanced 2, 50,000 dollars of 30 day volume, the brief's original placeholder | the same page, 10 October 2026 |
| `low_fee_tier` | 5 bps | Coinbase US VIP3, 20 million dollars of 30 day volume, the brief's low fee placeholder | the same page, 10 October 2026 |

The full US spot schedule read that day, in bps.

| Tier | 30 day volume in dollars | Maker | Taker |
| --- | --- | --- | --- |
| Intro | 0 | 50 | 90 |
| Advanced 1 | 10,000 | 35 | 75 |
| Advanced 2 | 50,000 | 15 | 30 |
| Advanced 3 | 250,000 | 10 | 15 |
| VIP1 | 1 million | 6 | 10 |
| VIP2 | 5 million | 4 | 6 |
| VIP3 | 20 million | 1 | 5 |

Every paper order is a market order, so only the taker column applies. Runs A and C use `retail_default`, runs B and D use `low_fee_tier`, and `active_retail` is kept as a named, sourced scenario. Run E uses `hypothetical_zero`, a taker fee of 0 that no venue charges, so run E is hypothetical and never a realistic cost setting. A few facts sit around these numbers.

* Before 16 September 2026 the US Intro rate was 120 bps. Coinbase's announcement of the change gives Intro taker rates of 90 bps in the US, 50 bps in the EU, the UK and Canada, and 10 bps in the rest of the world, so an account outside the US should set `taker_fee_bps_retail_default` to its own region's rate.
* Coinbase One Premium returns 25 percent of Advanced fees, which would make 90 bps about 67.5. It is a paid subscription and is not modelled.
* A real account climbs tiers on its own volume. A quiet bot trades about 2,000 dollars a day, so it would reach 75 bps in about 5 days and 30 bps in about 25 days. A fixed fee is the rate of a new account and harsher than reality after the first weeks, which is the safe direction.
* The fee also sets the required edge and the death threshold, so it changes every decision, death and parent choice. Repricing one run's trades under another fee is only a rough sensitivity check, and each profile needs its own run.
* The buy and hold benchmark pays one taker fee of the active profile when it buys at launch, so it is charged on the same terms as the bots.

See [section 2.2 of the plan](../PLAN.md#22-do-the-fee-placeholders-match-and-which-profile-should-be-the-default-question-2) and the [Coinbase Advanced fee tiers](https://www.coinbase.com/advanced-vip) page.

## Shorts, the opening fee and funding

Kraken's short selling page says Coinbase US does not offer spot margin shorting. That is a competitor's statement which could not be verified with Coinbase, so it is treated as likely rather than certain. A realistic US retail short therefore means Kraken margin or a perpetual style future, and the modelled short leg is a composite of the active profile's Coinbase spot taker fee and Kraken style margin charges, a mix of two venues that no single account would pay. Both charges are marked as placeholders and were checked against Kraken's published pages on 10 and 11 October 2026. Together they add 2.25 bps to a 30 minute short's round trip, 184.25 bps against 182 for a long at the retail fee, and 4 bps to a 4 hour short in run D, so the mix barely moves costs, and the skill test uses no costs at all.

* `short_open_fee_bps` is 2.0, the top of Kraken's 1 to 2 bps charge to open a BTC margin position. It is charged once on the notional of each short open, booked as funding, and included in the cost estimate for a short. On 500 dollars it is 0.10 dollars. For a 30 minute short it is larger than the hourly charge, so leaving it out would be kinder than the venue it models.
* `short_funding_bps_per_hour` is 0.5, the top of Kraken's rollover range of 0.01 to 0.02 percent per 4 hours, which is 0.25 to 0.5 bps per hour. On perpetual futures in 2025 and 2026 shorts were usually paid rather than charged, on average about 0.05 to 0.08 bps per hour, so 0.5 is conservative. On 500 dollars it is 0.025 dollars an hour.

Funding accrues on the entry notional for the time elapsed since the last accrual, and it is added at every healthy minute and at the close. A short therefore also pays for any minute that is not healthy, Stale or Recovering, and for downtime before a flatten, because in the real world the short would still be open. Rent does not accrue in those times, as the brief requires. Charging funding only in healthy minutes would be kinder than reality. See [section 2.2 of the plan](../PLAN.md#22-do-the-fee-placeholders-match-and-which-profile-should-be-the-default-question-2), the [Kraken fee schedule including margin](https://www.kraken.com/features/fee-schedule), [Kraken US margin fees](https://support.kraken.com/articles/us-margin-fees) and [Kraken short selling](https://www.kraken.com/pro/short-selling).

## Rent

Every bot pays `daily_running_cost`, which is 1.16 dollars a day, charged as that amount divided by 1440 in each healthy minute from birth, including Warmup and the hours before the features become valid. No rent accrues in any minute that is not healthy, Stale or Recovering, or while the system is down.

Rent answers one question. Does the bot beat leaving its capital in cash? The honest anchor is the cost of capital. The three month Treasury yield was 4.23 percent on 8 October 2026 in the Federal Reserve's DGS3MO series, which on 10,000 dollars is 1.159 dollars a day, rounded up to 1.16 so the bots are not flattered. The brief's placeholder of 5 dollars a day is 18.25 percent of start equity a year. The choice is a trade off. For a quiet bot, one that only trades when forced, rent at 1.16 dollars is 6, 16 and 49 percent of its daily burn under the 90, 30 and 5 bps profiles, against 22, 45 and 81 percent at 5 dollars. A higher rent speeds up deaths and evolution, and a lower one keeps the comparison anchored to cash.

All five runs use 1.16 dollars, including run C, whose 500 dollar start makes rent about 85 percent of start equity a year. Fees still dominate there, because a quiet retail bot pays about 18.20 dollars a day in fees and extra slippage for its two forced round trips. The bot that never trades is recorded rather than simulated, as start equity minus the rent of every healthy minute since launch, in the `cash_equity` column of the benchmark table. See [section 2.3 of the plan](../PLAN.md#23-what-daily-running-cost-is-sensible-question-3) and the [FRED DGS3MO series](https://fred.stlouisfed.org/series/DGS3MO).

## The cost estimate and the required edge

At each decision the bot estimates the cost of opening and closing the fixed notional, from the live book at that moment.

```
round_trip_bps = 2 * taker_fee_bps
               + spread_bps
               + 2 * impact_bps                 walk for the notional, beyond the best price, never below zero
               + 2 * extra_slippage_bps
               + short_open_fee_bps + short_funding_bps_per_hour * horizon_minutes / 60      shorts only
required_edge_bps = round_trip_bps + margin_bps
```

The estimate is for a short when the prediction is negative and shorts are allowed, and for a long otherwise. Impact is the walk's average price minus the best ask for a long, or the best bid minus the walk's average for a short, over the mid, in bps. If the visible book cannot fill the notional, the cost is infinite, no order is sent, and the decision row says so. The margin gene starts at 8 bps and evolves within 2 and 60 bps. A signal passes when the absolute prediction is at least the required edge. Because the estimate reads the live book, the required edge widens on its own when spreads widen, and the margin floor stops a bot from evolving into a pure fee burner.

With the measured one cent spread and negligible impact, and the initial margin of 8 bps, the numbers come out as follows.

| Profile | Round trip, long | Required edge, long | Required edge, short at 30 minutes |
| --- | --- | --- | --- |
| `retail_default`, 90 bps | 182 bps | 190 bps | 192.25 bps |
| `active_retail`, 30 bps | 62 bps | 70 bps | 72.25 bps |
| `low_fee_tier`, 5 bps | 12 bps | 20 bps | 22.25 bps |
| `hypothetical_zero`, 0 bps, run E only | about 0.001 bps | about 0.001 bps | about 0.001 bps |

Run D's 4 hour horizon makes a short pay 2 bps to open and 2 bps of funding, so its required edge is 20 bps for a long and 24 bps for a short. Run E has no fee, extra slippage, short charges or margin, so its required edge is only the spread plus the walk through the book on both legs, about 0.001 bps at a one cent spread and under 3 bps at every decision runs A to D recorded. Almost every prediction passes it, so run E's bots trade on their own calls, while their fills still pay half the spread and the walk on each leg. For scale, a 30 minute BTC log return had a standard deviation of about 32 to 34 bps in 2025 and 2026, about 26 bps in the 30 days before the plan was written, and an average absolute size of about 21.5 bps, while the bots' predictions are a few bps. The retail gate is therefore not expected to pass in practice, and every retail trade is expected to be a forced one, an outcome [section 21 of the plan](../PLAN.md#21-controls-parallel-runs-and-success-criteria) fixed before launch. Every decision row stores the prediction, the required edge, the cost estimate, the margin, the gate result and the forced flag. See [section 9](../PLAN.md#9-bots-and-trading-logic) and [section 2.5 of the plan](../PLAN.md#25-fifteen-or-thirty-minute-horizon-question-5), and [`bot.py`](../src/btc_survivor/bot.py).

## Death threshold

A bot dies when its equity falls below what one more round trip and one more day would cost.

```
death_threshold = round_trip_bps / 1e4 * notional + daily_running_cost
```

The round trip is estimated from the live book for the side the bot holds, or for a long if it is flat, without the margin gene. The check runs in every healthy minute, after rent and marking and before the stop loss. With a one cent spread the threshold is about 10.26 dollars under the 90 bps profile, 4.26 dollars under 30 bps and 1.76 dollars under 5 bps, and holding a 30 minute short adds about 0.11 dollars. A bot must therefore lose almost all of its start equity before it dies.

A dead bot leaves its slot at once, and its replacement takes the slot in the same minute. If it holds a position, a closing order goes through the normal broker and is resent each healthy minute until it fills. When it fills the costs apply, any negative cash becomes a recorded writeoff with cash clamped to zero, the bot is marked Dead and its death row is written. A bot with no position finishes at once. The cause of death is whichever of fees, funding, rent, stop loss losses and other trading losses is largest, and all five are stored with the threshold and the final equity, so `btc_survivor explain` can show the numbers. The days this takes under each profile are tabled in [section 3 of the plan](../PLAN.md#3-what-to-expect-from-the-default-experiment), for example about 516 days for a quiet retail bot at 10,000 dollars and about 25 days at run C's 500. See [section 10 of the plan](../PLAN.md#10-execution-and-accounting) and [`account.py`](../src/btc_survivor/account.py).

## How fills reach the books

Cash is settled cash, kept as its own ledger and updated at every event, so the checks against the totals are real checks. Opening a position moves only the fee and, for a short, the opening charge out of cash. Closing settles the gross profit and the closing fee. Rent and funding leave cash as they accrue. At every equity snapshot and after every step of the randomized tests, four identities must hold within one millionth of a dollar.

```
cash   == start_equity + realized_gross_pnl - fees - funding - rent + writeoff
equity == cash + unrealized_pnl      (marked to the exit side of the book)
start_equity - equity == fees + funding + rent + stop_loss_losses + other_trading_losses - writeoff - unrealized_pnl
realized_gross_pnl == -(stop_loss_losses + other_trading_losses)
```

The third line splits a bot's losses into the five parts that decide its cause of death, and the fourth ties the realized gross profit to the stop loss and trading loss totals, so no closed trade is missing from either. A live violation is logged at ERROR and written as an `invariant_violation` feed event, which the dashboard lists with the other recent feed events, and the run continues so the evidence is kept. `tests/test_account.py` checks all four lines after every step of 5,000 seeded random sequences of opens, closes, flips, stops, rent, funding and deaths. See [`account.py`](../src/btc_survivor/account.py) and [`tests/test_account.py`](../tests/test_account.py).

## No principal is reserved

Because opening a position moves only the fee and any short opening charge out of cash, no principal or margin is set aside for it. The broker's checks before a fill look at the book, the position and the quantity and never at the bot's funds, every open uses the same fixed notional, and the only funds check anywhere is the one the config loader makes at start up, that `notional` is no larger than `start_equity`. Negative cash at death becomes a recorded writeoff. In runs A, B, D and E a notional of 500 dollars against about 9,400 dollars or more of equity is harmless, and the largest notional over equity at any open of runs A and B was 0.05 times on 12 October 2026. In run C, with 500 dollars of start equity and a 500 dollar notional, the exposure grows as fees eat the equity.

`scripts/run_c_analysis.py` works out the path of a run C bot that trades only when forced and never gains or loses on the market, from the config alone, with its first open 19 hours after launch and one every 12 hours. An open resets the forced timer in its own decision minute, the timer is due after 720 healthy minutes, and 720 is a whole number of 30 minute horizons, so the bot is already on its phase minute when the timer comes due and forced opens come exactly 12 healthy hours apart. Only a gap in the feed adds a wait. The external review's verifier assumed 12.25 hours, which puts each day in the table below 0.3 to 0.5 days later, and `--period-hours 12.25` reproduces his figures. Days are days of healthy data, and equity after the open counts the open's 4.50 dollar fee and its 1 bp of extra slippage.

| Notional over equity after the open | Healthy day | Equity after the open in dollars |
| --- | --- | --- |
| 2 times | 13.8 | 242.85 |
| 5 times | 21.3 | 97.65 |
| 10 times | 23.8 | 49.25 |
| 24.7 times, at the open before the last | 25.3 | 20.21 |
| 47.5 times, 33.2 times before the fee, at the last open | 25.8 | 10.53, from 15.08 before the open |

The death threshold of 10.26 dollars caps a bot that is still alive at 48.7 times. In this arithmetic the last open leaves the bot 10.53 dollars, just above its threshold, so it holds the position for its 30 minutes at 47.5 times its equity and dies on day 25.8 when the close fee takes it below. Nothing like a margin call exists, so the death rule is the only check on a bot's funds. At 20 dollars of equity a 1 percent stop costs 5 dollars, a quarter of equity, and a 4 percent gap would cost all of it, although a forced position is held only about 30 minutes in every 12 hours. A spot account with no margin could not open even run C's first trade, because 500 dollars plus a 4.50 dollar fee is more than the 500 it holds. Under an illustrative 5 times margin cap a quiet bot would stop trading near day 22 and live on rent until day 92.7 instead of dying on day 25.8, so run C's death schedule depends on this leverage, and its dollar paths are not what a 500 dollar retail account would see.

The same script prints the notional over equity at every recorded open, before and after its fee, with `--every-open` for one line per open, and the recorded death days beside the days a 5 times cap, or any cap given with `--cap`, would have allowed. The first 13 learner opens in run C, by 12 October 2026, were all at about 1 times, at most 1.01 times after the fee. These figures are descriptive and change nothing in the runs. The disclosure is in [the honesty and limits page](honesty_and_limits.md#run-c-trades-on-unreserved-leverage), and the tables are in [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md#run-c-analyses).

## Paper orders never consume liquidity

Every fill walks the book as it stands and leaves it untouched, so orders that arrive together never compete for the same levels. A learner and its twin act at the same moment, and a gap flatten or a restart flatten sends every open position to the book in the same second. In each case every order pays the prices of the untouched book, where a real account's later orders would pay the deeper levels that the earlier ones took. Phase scheduling spreads the learners' own decisions across the horizon, so most fills do not overlap. At 500 dollars against the measured book the walk cost about 0.0006 bps, so the effect is tiny in a quiet market, but it would grow with size and in thin moments, and the simulation cannot show it.

The same gap appears in other forms. Orders never move the price for anyone else, there is no queue, the exchange never rejects or delays an order for its own reasons such as a rate limit or a halted market, every order fills completely or not at all, and fees never change during a run. In these ways paper results are kinder than real trading. They are harsher in two, because a fixed fee ignores the tier a real account would climb to, and the extra 1 bp per leg is far larger than the impact measured at this size. See [the honesty and limits page](honesty_and_limits.md) and [section 20 of the plan](../PLAN.md#20-honest-limits-and-risks).
