# News and scoring

This guide follows a news item from the moment the news process fetches it to the moment it moves S2's news inputs. It covers collection, deduplication and routing, scoring with Haiku, Sonnet and FinBERT under one monthly spend cap, the score streams, news health and how the stock process reads finished scores. Everything here is fixed before launch, and week one before launch may change only cadences, budgets and retention. The design is [section 7](../PLAN_V2.md#7-news-pipeline) and [section 8 of the plan](../PLAN_V2.md#8-language-model-scoring-with-spend-controls), and the overview is [v2_overview.md](v2_overview.md).

## The news process

`btc_survivor_v2 news configs/v2/news.toml` is one long running asyncio process, `v2/news/runner.py`. It polls every enabled channel, runs the Haiku and Sonnet scorers and the FinBERT scorer, writes a heartbeat every 30 seconds and prunes once a day, all through one writer connection to `data/v2/news/news.db` in WAL mode. Scoring starts once `[finbert]` pins the exported model, and from then on the prompt and schema pins, the Anthropic key and the FinBERT files must all match or the process exits with code 2. Before any FinBERT pin is written it collects only, so a restart before the export never stops collection, and the model runtimes load only when scoring starts, so the stock process never loads them. Network calls run with `urllib.request` in worker threads, and FinBERT runs in a worker thread, so polling never stalls. No version 1 module imports the news package, and it imports nothing from the stock modules.

* At start it sets umask 077, installs the redactor before anything logs, refuses a proxy or certificate variable, reads secrets only through `local_secrets`, and takes an exclusive lock on `~/.config/btc_survivor/finnhub_rest.lock` while Finnhub is enabled, so it is the only Finnhub REST client.
* On every start it writes the config hash, the prompt and schema hashes and both model ids to `meta`, with a `config_changed` event when the config hash moved, and starts its receive clock no earlier than the newest item, so receive order survives a wall clock that stepped back.
* A heartbeat that comes far later than due writes a `pause` event, as after the Mac slept. Waits run in slices of at most a minute, because the event loop's clock stops while the Mac sleeps.
* SIGTERM or Ctrl+C lets the tasks finish for up to 20 seconds, writes a `stopped` event and exits with code 0. An expected failure, such as a held lock or a bad config, prints one `error` line and exits with code 2, and a crash exits with code 1.
* `--dry-run` loads the config and the secrets, takes and releases the lock and prints a verdict, opening nothing on the network.

The process is specified in [section 7.1](../PLAN_V2.md#71-the-news-process) and [section 14.3 of the plan](../PLAN_V2.md#143-the-news-process), and run as [v2_operations.md](v2_operations.md#the-news-process) describes.

## Channels and cadences

A channel is one URL pattern on its own schedule. Its cadence depends on the New York time of day, read through `zoneinfo` only to pick the cadence, and every weekday counts as a session and filing day. When a cadence gets faster the channel polls at once.

| Channel | Host | Cadence |
| --- | --- | --- |
| `finnhub_company_SYM`, one per company | Finnhub | 60 s from `09:30` to `16:00` ET, 120 s from `04:00` to `09:30` and `16:00` to `20:00`, 600 s otherwise |
| `finnhub_general` | Finnhub | the same |
| `edgar_atom_8k` | sec.gov | 30 s from `06:00` to `22:00` ET on weekdays, 900 s otherwise |
| `edgar_sub_SYM`, one per company | data.sec.gov | 60 s in the same window, 900 s otherwise |
| `fed_press_all` | federalreserve.gov | 60 s from `07:00` to `19:00` ET on weekdays, 900 s otherwise |
| `fed_press_monetary` | federalreserve.gov | 15 s from `13:58` to `14:20` ET on FOMC statement days only, dormant otherwise |
| `fed_speeches` | federalreserve.gov | 600 s from `07:00` to `21:00` ET on weekdays, 3,600 s otherwise |
| `bls_cpi`, `bls_empsit`, `bls_ppi`, `bls_jolts` | bls.gov | 1,800 s from `07:00` to `19:00` ET on weekdays, 7,200 s otherwise, and 15 s from one minute before to 15 minutes after a scheduled release |

QQQ has no company channel, and its company input pools the seven companies. EDGAR documents are fetched on demand, about three per `8-K`. The channels are defined in [section 7.2 of the plan](../PLAN_V2.md#72-channels-cadences-and-the-rest-budget).

## The calendar

The FOMC and BLS fast windows come from `[calendar]` in `configs/v2/news.toml`, copied on 12 October 2026 from the Federal Reserve's FOMC calendar and the BLS release schedule pages, with the source of each date in a comment.

* FOMC statements on 28 October and 9 December 2026 and 27 January and 17 March 2027, each at `14:00` ET. The Fed's page gives no time, so each takes the time of the last statement, and it calls the 2027 dates tentative.
* CPI, the employment situation and PPI at `08:30` ET, every release from 12 October 2026 through 31 March 2027.
* JOLTS at `10:00` ET on 3 November and 1 December 2026 only. No BLS page read on 12 October listed a later JOLTS release, so later dates are added before launch if BLS lists them by then. If they are still missing at launch, the JOLTS channel polls at 1,800 s on those days instead of 15 s.

`[calendar]` cannot change after launch, because any change alters the news config hash, which fails news health for the rest of the window and breaks criterion 1.6. A moved FOMC date costs only the fast window, because `fed_press_all` keeps polling every 60 s. The rule is in [section 16 of the plan](../PLAN_V2.md#16-build-phases-with-tests).

## Requests and the budget

`v2/news/budget.py` keeps one token bucket per host.

* **Finnhub.** At most 30 calls in any rolling minute and at least 1.1 s apart, against the key's 60 a minute and 30 a second. A 429 pauses every Finnhub channel for 60 s.
* **SEC.** At most 2 requests a second and 60 a minute across sec.gov and data.sec.gov. A 403 or 429 pauses every SEC channel for 900 s.
* **Fed and BLS.** At most 10 a minute each. A 403 or 429 pauses that host for 900 s.

A failed poll comes due again after its cadence times 2 to the power of its consecutive failures, capped at 900 s, and there are no automatic retries. `v2/news/http.py` asks for gzip, caps a body at 8 MB, times out after 10 s and sends `If-None-Match` or `If-Modified-Since` when it can, counting a 304 as success. The contact address goes in the `User-Agent` only for sec.gov, data.sec.gov and bls.gov, the Fed and Finnhub get `btc-survivor/2`, and the Finnhub token travels in the `X-Finnhub-Token` header, never in a URL. Headers are never logged. While the response `Date` headers of the last 20 responses disagree with the local clock by more than 5 s at the median, every poll is recorded as `clock_skew` and counts as failed. The budget is in [section 7.2 of the plan](../PLAN_V2.md#72-channels-cadences-and-the-rest-budget).

## When an item counts

* `received_at` is the stamp of one clock that never goes backwards, taken right after the body is read. Parsing, deduplication, the prefilter, the insert and the commit follow with no `await` between them, so `seq` order equals `received_at` order. The scorers stamp `scored_at` the same way.
* A new `8-K` or monetary policy release is written only once its documents arrive, with the first sighting kept as `detected_at`. A document that fails twice, or cannot be parsed, leaves the item's text marked truncated and writes a `doc_failed` event.
* Vendor times go to `vendor_ts` only, to measure delivery lag and for one guard. An item whose vendor time is more than 24 hours before its receipt gets status `stale_vendor` from every scorer.
* Rows are never updated after insert, only pruned, which `test_v2_news_immutability` checks.

A score then counts for S2 only once the stock process has received its line, as the last sections explain. The rule is in [section 7.3 of the plan](../PLAN_V2.md#73-the-receive-timestamp-rule).

## Seed, backlog, duplicates and the prefilter

Each raw item passes these steps inside its poll.

1. A known `(source, vendor_id)` from the same channel writes nothing, and from another channel writes one `news_dups` row.
2. Every item from a channel's first successful poll gets status `seed`, kept for deduplication and never scored.
3. An item gets status `backlog` when the time since the channel's previous successful poll exceeds the longer of the cadences at the two polls, plus 600 s, plus any budget wait. Filings of kind `8k` are exempt, because a filing is rare and material.
4. The same headline hash from the same source within 24 hours is a duplicate, and so is a token set with a Jaccard index of at least 0.8 against an item of the same source from the last 24 hours. As built, filings skip both steps, because their titles repeat while their accession numbers already tell them apart.
5. The prefilter tags the survivors with symbols through fixed case sensitive aliases, such as Apple, iPhone and Tim Cook for AAPL, and with MACRO through fixed macro keywords, such as FOMC, CPI, payrolls and tariff. An item that matches an alias never also gets MACRO, so one company's news never enters all eight macro inputs. Junk patterns such as stocks to buy are filtered.

A duplicate never changes its original, so tickers are fixed at first receipt. Each relevance tier, `title`, `body`, `vendor`, `macro`, `official` or `filtered`, is stored with the item. The lists are in [section 7.5 of the plan](../PLAN_V2.md#75-seed-backlog-deduplication-and-the-prefilter).

## Routing

The route is set once, at insert.

* `llm` covers Finnhub items with status `new` and tier `title`, `body` or `macro`, and the official kinds `8k`, `fed_press`, `fed_monetary`, `fed_speech` and `bls_release` with status `new`. Haiku and FinBERT score every `llm` item, and Sonnet also scores the `llm` items of the five official kinds and nothing else.
* `none` covers everything else, including `8k_a`, seeds, backlog, duplicates and filtered items.
* `finbert` exists only while `[finnhub] send_to_llm` is false, and then takes the Finnhub items `llm` would have taken.

Only headlines of up to 300 characters, summaries of up to 600 for Finnhub, 1,000 for the Fed and 1,500 for BLS, and the cut filing and FOMC texts are kept, never full articles, images or raw responses. All fetched text is untrusted, and the prompt tells the models to ignore any instruction inside it. Routing is in [section 7.6 of the plan](../PLAN_V2.md#76-routing-and-the-text-kept).

## The request to Haiku and Sonnet

`v2/news/llm_client.py` is the only module that imports `anthropic`, pinned at 1.8.0, and `v2/news/llm_request.py` builds the request.

* **The model.** `claude-haiku-5-5` with `max_tokens` of 4,096, and `claude-sonnet-5-5` with `max_tokens` of 1,024 for its answer. No other model id is accepted anywhere, and Opus is never used.
* **The settings.** Effort low, a JSON schema output format, no temperature, top p or top k, no tools, metadata, stop sequences or fallbacks, and the SDK built with `max_retries=0`, so nothing is ever retried. Haiku always sends `thinking` disabled.
* **Sonnet's thinking.** Anthropic's documentation read on 12 October 2026 says Sonnet 5.5 answers `thinking` disabled with a 400, so by the owner's decision `[llm.sonnet] thinking` chooses in config between disabled and a minimal thinking setting, `between_tools`, the lowest the documentation lists for Sonnet 5.5, or `adaptive` at effort low. Any setting other than disabled adds `thinking_tokens` of room to the request's `max_tokens`, because thinking counts toward it, and the reservation prices that room at the output rate while the cost books every thinking token as output. One live call before launch, made by the assistant session, decides the setting, and until then `[llm.sonnet] enabled` stays false.
* **The prompt.** `v2/news/prompt_v2.txt`, which replaced `prompt_v1.txt` before any live call used it, about 1,400 to 1,600 tokens by an estimate that the first `count_tokens` call replaces, LF only and read as bytes, with the task, the ground rules, the instrument table, one line per level of each field and three fictional, undated examples. It asks a model to judge each item as an analyst would judge the trading that follows the market's first reaction to it, since large US stocks usually price clear news within minutes, and so to give the direction of the move still to come over the next few hours of regular trading. Haiku's single system block carries cache control, and Sonnet's carries none, because its calls are rare and its cache prices are outside the research.
* **The schema.** `v2/news/schema_v2.json`, where each item lists impacts with ticker, event type, whether the item is new, direction from `-1` to 1, surprise from 0 to 3, horizon and confidence from 1 to 5, every field a required enum, and no Bitcoin ticker. The horizon says how long the remaining move lasts, `minutes` when what is left of it is over within minutes, `hours` when it likely continues over the next few hours of trading, and `days` when it builds slowly over days or longer. The config pins `prompt_sha256` and `schema_sha256`, and scoring refuses to start while a pin is empty or disagrees with its file.
* **The user message.** One line of instructions, then one compact JSON object per item with position ids `i01` to `i25`, the source, the EDGAR filer's ticker, the headline cut at 300 characters and the text cut at 700, or at 24,000 for an `8-K` or a monetary policy statement. No date, no stock price and no row field other than these ever enters a request.
* **The size guard.** A request whose exact token count, from the free `count_tokens` endpoint, or whose bytes over 3 without a count, exceed 20,000 is refused. A single item above it ends `failed` with no call.

`v2/news/llm_validate.py` checks each answer against strict Pydantic models that forbid extra fields, then that each id appears once, tickers are unique within an item and an item has at most 10 impacts. Valid items are kept when others fail. The call is in [section 8.1](../PLAN_V2.md#81-placement-and-the-call) and the schema in [section 8.4 of the plan](../PLAN_V2.md#84-output-schema-and-validation).

## Batching

`v2/news/batcher.py` is pure and takes the time as an argument. Finnhub items wait in receive order, ties broken by item seq, and a batch goes when 20 wait or the oldest has waited 60 s, at any hour. Every official item goes alone and at once to each model, so Haiku and Sonnet receive the same user message for it. An item still waiting 300 s after its receipt gets a final `expired` row instead. A split gives the first half `floor(n / 2)` items. Batching is in [section 8.3 of the plan](../PLAN_V2.md#83-eligibility-input-and-batching).

## When a call fails

`v2/news/llm_scorer.py` puts each call on record in `scorer_state` before it is sent, sends it once, books it in the key ledger and writes it with the final rows it decided in one transaction.

| What happened | What the scorer does |
| --- | --- |
| A valid answer with stop reason `end_turn` | one final row per item, scored or irrelevant |
| Some ids missing or invalid | keeps the valid items, requeues each missing item once, then marks it malformed, while invalid items end malformed at once |
| A refusal | splits the batch in halves until a single refused item ends `refused` with its category |
| `max_tokens`, unreadable JSON or a whole answer failing the schema | splits the same way, and a single item ends malformed |
| More than 24 calls for one batch | marks the remaining items malformed |
| A 402, a billing error, or a 429 or 400 naming a spend, credit, billing or usage limit | both models off until `00:00` UTC on the first of the next month |
| Any other 429 | that model paused for 15 minutes or the retry after time, whichever is longer |
| Any other 4xx, such as a 400, 401, 403 or 404, while a 408 counts as a timeout | that model off until restart, with one ERROR line |
| A 5xx, 529, 408, timeout or lost connection | booked at its full reservation, its items failed, no second call |
| `response.model` not the model requested | the answer discarded, booked at its reservation, that model off until restart |
| Refusals in more than 20 percent of a model's last 50 calls | that model paused for 15 minutes with a `scorer_alert` event |
| The process died during a call | at the next start the call becomes `lost`, booked at its reservation, its items failed |
| The monthly cap would be passed | no call, both models capped until the next UTC month, waiting and arriving items `skipped_cap` |

While Haiku is paused, capped or off, every waiting and arriving item routed to it gets a final `skipped_off` or `skipped_cap` row at once, so FinBERT takes over at once. Sonnet's items get the same rows, and a Sonnet error never changes Haiku's state, except a spend limit, which belongs to the whole key. The states and the cap live in `scorer_state` and survive a restart, while off until restart ends with it. Every state change is a `scorer_state` event in `collector_events`. The table is in [section 8.5 of the plan](../PLAN_V2.md#85-failure-handling).

## Spend, the key ledger and the monthly cap

Every process that may call either model with the key, the news process, its probes and `scripts/llm_synthetic_check.py`, reserves against one ledger, `~/.config/btc_survivor/llm_ledger.db`, under an exclusive lock on `llm_ledger.lock` beside it, so two processes can never pass the cap together. `v2/news/ledger.py` works in exact fractions and rounds once.

```text
rates in dollars per million tokens, from [llm.prices], read 11 Oct 2026
  claude-haiku-5-5     input 0.10   output 0.50    cache write 1.25 times input, cache read 0.10 times input
  claude-sonnet-5-5    input 2.00   output 10.00   no cache control, every input token at the base rate
counted = count_tokens of the full request, or request bytes plus reserve_overhead_tokens when counting fails
reserve = (counted * input rate * the largest input multiplier + max_tokens * output rate) / 1e6
          where max_tokens is the request's own, thinking room included
cost    = the usage the answer reports, priced the same way, with thinking tokens as output
```

* **The cap.** A call goes only when the UTC month's spend plus its reservation stays at or under `monthly_usd`, 20 dollars, the same as the Console workspace limit, so the code stops first. Spend counts the cost of every call that returned usage and the reservation of every other call. There is no daily cap of any kind.
* **At the cap** both models stay off until `00:00` UTC on the first of the next month, and FinBERT covers every item.
* **The daily alert.** When one UTC day's spend passes a quarter of the cap, 5 dollars, the news process logs one WARNING, writes one `spend_alert` event and sets `spend_alert` on every heartbeat for the rest of that day, and the dashboard shows a banner. The alert stops nothing.
* **The Console backstop.** The key belongs to a dedicated workspace with a 20 dollar monthly limit and auto reload off. Whether that limit applies to the Max plan's included credits is unverified, which is one reason the cap in code exists.

A reservation is about 0.003 dollars for a full Haiku batch, about 0.014 for a short Sonnet call and about 0.028 for a long one. The expected spend is about 1 to 5 dollars a month for both models together, so the cap is 4 to 20 times the expected spend. The ledger is in [section 8.6](../PLAN_V2.md#86-cost-ledger-and-the-monthly-cap) and the expected cost in [section 8.10 of the plan](../PLAN_V2.md#810-expected-cost).

## FinBERT

`scripts/export_finbert.py` runs once, with the owner's approval, in a throwaway environment outside the repository with torch and transformers pinned, because torch never enters the project environment. It downloads `ProsusAI/finbert` at a pinned commit, about 440 MB, exports ONNX at opset 17, checks the export against PyTorch within 0.0001 on 50 fictional sentences and one full window, and writes `model.onnx`, `tokenizer.json` and `manifest.json` under the gitignored `data/models/finbert`, then prints the three pins for `[finbert]`.

`v2/news/finbert.py`, the only module that imports `onnxruntime`, checks the three files against their pins and the manifest before anything runs, and builds the session from the same bytes it checked. A missing file or any mismatch stops the news process. The session runs on the CPU with one thread and sequential execution, so scores are bitwise repeatable on one machine. It scores the headline and text joined by a period, in windows of up to 510 tokens and at most 16, averaging the probabilities weighted by token count. FinBERT cannot tell which company a text is about, so its impacts take the prefilter's tags, the EDGAR filer or MACRO, and an untagged item ends `untagged`. It scores every routed item as it arrives, with no cost and no cap, and its strength is the positive probability minus the negative one. FinBERT is described in [section 8.7 of the plan](../PLAN_V2.md#87-finbert-and-the-score-streams).

## The score streams

One pure function in `v2/stocks/news_inputs.py` turns the final score rows into streams, and a test shows it gives exactly one primary score per item across every route, status and order of arrival.

* **`news`, the primary stream.** An `llm` item uses its Haiku score when its final Haiku row says scored or irrelevant, visible when that row is received, and otherwise its FinBERT score as a flagged fallback, visible at the later receipt of the two rows. It feeds S2's inputs and the confirmatory predictor.
* **`haiku`** holds Haiku rows only, and **`finbert`** FinBERT rows for every routed item, the baseline.
* **`finbert_matched`** holds FinBERT rows on exactly the items Haiku scored, so `haiku` against `finbert_matched` measures what the model adds on the same items.
* **`sonnet` and `haiku_official`** hold Sonnet and Haiku rows on exactly the official items both scored, each visible at the later receipt of the two.

A model impact counts only when the model marked it new. By the owner's decision of 12 October 2026 its strength folds in the horizon and the surprise, by weights fixed by reasoning before launch and never fitted.

```text
strength = direction * (confidence / 5) * H[horizon] * (2 + surprise) / 5
H        = {minutes: 0.25, hours: 1.0, days: 0.5}
```

The test pairs a score with each 30 minute return over the next session hours, half its weight in the first 2, so the drift per window is about the move still to come over how long it lasts. A move over within minutes is mostly gone before the first evaluation minute, one lasting hours sits where the weight is, and one building over days spreads thin. `(2 + surprise) / 5` keeps a routine item at 0.4 and a major shock at 1, so strength stays from `-1` to 1. FinBERT's strength stays the positive probability minus the negative one. Sonnet never enters the primary stream, so the confirmatory test does not depend on it, and the primary stream mixes FinBERT's scale in only for fallback items, whose share `evaluate` reports. The streams are in [section 8.7 of the plan](../PLAN_V2.md#87-finbert-and-the-score-streams).

## Probes

Each day from `06:00` UTC, between its batches, the Haiku scorer sends the 20 fictional items of `v2/news/probes_v2.json`, written with their answer key before launch, as one reserved call of kind probe. Results go only to `llm_probes`. `probes.agreement` measures the agreement of two runs field by field, and the report that compares each day's run with the first run and the previous one, flagging direction agreement below 80 percent and changing nothing, joins `news-status` when the scorers are wired into the news process. Sonnet runs no probe, because its stream never feeds the confirmatory test. The two probe runs of the synthetic check are not stored in `news.db`, so the baseline is the news process's own first daily probe. Probes are in [section 8.8 of the plan](../PLAN_V2.md#88-lookahead-non-determinism-and-probes).

## News health

News health means the news pipeline is alive. Every heartbeat row is healthy when FinBERT is on, no item routed to a scorer has waited more than `score_wait_seconds`, 600, for its primary score, and the clock check is clean. The states of Haiku and Sonnet never count, because a paused, capped or off Haiku hands every item to FinBERT at once. Each heartbeat also lists the channels in outage, those whose last successful poll is older than three times their cadence or 300 s, whichever is longer, which never touches news health.

On the stock side a session minute is news healthy when the newest heartbeat line received was never more than 120 s old during the minute, and the line in force at the minute's start and every line received in it said healthy and carried the config, prompt and schema hashes recorded at launch, so a minute without a line after an unhealthy one stays unhealthy. Otherwise S2 writes the minute to `news_minutes` with its reason, `no_heartbeat`, `heartbeat_old`, `heartbeat_unhealthy` or `hash_mismatch`. Without news health both runs charge no rent, train nothing, decide nothing, push no label record and write no forecast row, while prices, marks, stops, the session close, epochs and the maturing of pending labels go on. After a stock process restart the first minute waits for a fresh heartbeat, at most 30 s. News health is in [section 7.8 of the plan](../PLAN_V2.md#78-news-health).

## How S2 sees the news

The stock process never opens a model. Once a second its news puller reads new `item_scores` rows with their impacts and item fields, and new `news_heartbeat` rows, from `news.db` opened with `mode=ro`, never immutable, so it sees every later commit. It turns them into recorded lines of kind `news` and `news_heartbeat`, which carry sequence numbers, source, route, scorer, status, scored time, impacts and channels in outage, never headline text. The lines take stamps from the same live clock as the price lines, so the raw recording holds them in the order the engines saw them and a replay needs no `news.db`. The puller retries until `news.db` exists, reopens it after any error or 60 s without a new heartbeat, and starts at the newest score at first launch, stored in `meta` as `news_start_seq`.

S2's two inputs per symbol are computed at each minute boundary from the primary impacts received before it.

```text
w_j     = 2 ** (-a_j / 120)          a_j = session minutes from scored_at_j to the boundary
company = sum(w_j * strength_j) / (sum(w_j) + 1)     over impacts tagged with the symbol
macro   = sum(w_j * strength_j) / (sum(w_j) + 1)     over impacts tagged MACRO
```

The decay counts session minutes, so a score after the close keeps full weight until the open, and impacts older than 600 session minutes are dropped. QQQ's company input takes impacts tagged QQQ or any of the seven companies. S2 stores the inputs at each minute a label record is pushed, and a training sample or a `linear_news` update uses those stored inputs, never the current ones, so nothing learns from news received after its record's minute. S2 writes every score line it read to `news_seen`, with when it counts from and whether it became the primary score. The inputs are in [section 6.3 of the plan](../PLAN_V2.md#63-the-two-news-inputs).

## The news report

`btc_survivor_v2 news-status` opens `news.db` read only and prints counts and verdicts for the last 7 days by default, the status mix of polls and items, bytes received, backlog items and the channel hours that hold them, outage minutes with the longest outage, and two verdicts. The order verdict passes when `received_at` never decreases in `seq` order. The backlog verdict passes when every backlog item follows a restart, a pause, a failed poll or an outage of its channel. `--detail` adds requests per channel per day, backlog per channel per hour and outage minutes per channel per day, and `--detail --samples 50` adds sampled kept and filtered headlines and near duplicate pairs drawn with the report seed. The report is in [section 7.7 of the plan](../PLAN_V2.md#77-week-one).

## Retention

Items, bodies, scores, impacts and calls are kept for good, except filtered items, kept 30 days. Polls, duplicate rows and heartbeats are kept 14 days. Growth is about 1 MB a day for items, 2.5 MB for polls and up to 10 KB per model call, roughly 1 to 2 GB a year, an estimate. The tables are in [section 9.1 of the plan](../PLAN_V2.md#91-the-news-database).

## Limits and open items

* **Sonnet's thinking setting.** Anthropic's documentation read on 12 October 2026 says Sonnet 5.5 rejects `thinking` disabled at every effort level, pointing to a setting that thinks only between tools. A call with a setting Sonnet refuses gets a 400, which turns Sonnet off until restart, so one live call settles the setting before launch, and until then `[llm.sonnet] enabled` is false, Sonnet's items get `skipped_off` rows and the Sonnet comparison is empty. Thinking makes Sonnet's calls cost more, which its reservation covers.
* **Unverified until the first live calls.** Whether integer enums are accepted, whether Haiku's cache reads hit with structured outputs, whether `count_tokens` accepts the full request, the prompt's real token count, the exact wording of the spend limit error, and whether `response.model` comes back exactly as requested, since any suffix turns that model off.
* **The first scoring start.** Haiku gives every item collected since 12 October an `expired` row, while FinBERT scores that whole backlog, stamped at the time it scores it, so news health reports a long wait until FinBERT catches up. The scoring code should therefore start well before launch.
* **Stopping during a call.** The stop grace is 20 s and a model call may take up to 60 s, so a call in flight at SIGTERM is cancelled and booked lost at the next start, rather than finished. That is safe, because its items are failed and FinBERT covers them.
* **One set of scores.** Sampling cannot be fixed, so each item is scored once per scorer and frozen. A fresh rerun would differ by about what the probes measure for Haiku, and Sonnet runs no probe.
* **Mixed scales.** Fallback items carry FinBERT strengths in the primary stream, on a different scale from Haiku's, whose strengths now also weigh horizon and surprise.
* **An untried prompt.** `prompt_v2.txt` has had no live call yet, so whether the models use the minutes horizon and the first reaction framing as intended shows only in the first live synthetic check and the daily probes.
* **Delays.** Received time measures this system, so vendor delay, batching of up to 60 s, outages and filings after `17:30` ET, which EDGAR disseminates the next business day, all make news count later.

The risks are listed in [section 18 of the plan](../PLAN_V2.md#18-risks-and-honest-limits), and how the results are judged in [v2_evaluation.md](v2_evaluation.md).
