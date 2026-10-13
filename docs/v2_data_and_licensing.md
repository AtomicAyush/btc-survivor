# Data and licensing

This guide sets out where version 2 gets its data, what each source's terms allow, the two decisions the owner made about Finnhub data, what never leaves this Mac, how secrets are kept, and the limits a real brokerage account would meet on shorts, margin and day trading that the paper account does not. Vendor facts come from the vendor research of 11 October 2026 unless a date says otherwise, and anything that research could not confirm is marked unverified. The design is in [section 13 of the plan](../PLAN_V2.md#13-dashboards-licensing-and-data-handling), and the overview is [v2_overview.md](v2_overview.md).

## Sources and their terms

| Source | What version 2 takes | Credential | Terms in short |
| --- | --- | --- | --- |
| Finnhub free plan | websocket trade prints for the eight symbols, company news for the seven companies and general news over REST | a data only key in `secrets.env` | personal non professional use, no sharing of the data or results derived from it with any third party without written approval, and one websocket connection per key |
| SEC EDGAR | the `8-K` and `8-K/A` filings of the seven companies, with the main document and Exhibit 99.1 | none, a contact address in the request header | fair access limits of 10 requests a second, kept here at 2 a second and 60 a minute |
| Federal Reserve | press releases, monetary policy releases and speeches from its RSS feeds | none | public content with the reuse terms of its disclaimer page |
| BLS | the CPI, employment situation, PPI and JOLTS release feeds | none, a contact address in the request header | public content with the reuse terms of its linking page |
| Anthropic API | scoring by `claude-haiku-5-5` and `claude-sonnet-5-5` | an API key of a dedicated Console workspace | paid from the included API credits of the owner's Max plan, with a 20 dollar monthly workspace limit |
| FinBERT | the `ProsusAI/finbert` model, exported once to ONNX and run locally | none | code under Apache 2.0, weights with no stated licence and training data under CC BY NC SA 3.0, which suits this personal non commercial project |
| Nasdaq historical quote page | the official daily volume and close for the tape coverage report | none | read by hand in an ordinary browser for personal viewing, and no script ever fetches it |

Every source and limit is listed with its link in [section 20 of the plan](../PLAN_V2.md#20-references).

## Why stock data stays private

* Finnhub's terms bar sharing data or derived results with any third party without written approval.
* The NYSE subscriber agreement counts all information derived from market data as market data.
* From 1 April 2027 the CT Plan replaces the CTA and UTP plans, and its section 4.10 says a non professional's outputs may not be retransmitted, published or furnished to any other person. How Finnhub will implement that is unknown, so continuing past that date needs the terms checked again.
* Delaying prices does not help. Until April 2027 delayed redistribution of Nasdaq listed names needs a UTP licence at 250 dollars a month plus 250 a year.

So prices, candles, simulated fills, profit and loss, forecasts, labels and everything else derived from the stock feed stay on this Mac, under the gitignored `data` folder, together with recordings, databases, logs, exports, filled results and the official figures and coverage records of the tape coverage report. The only results file in the repository is [RESULTS_V2_TEMPLATE.md](RESULTS_V2_TEMPLATE.md), with empty tables. Docs pushed to GitHub carry no feed quality numbers either, which stay under `data/v2/checks`, as question 10 of [section 19 of the plan](../PLAN_V2.md#19-open-questions-for-the-owner) recommends.

## The owner's decision on Finnhub headlines

Finnhub headlines and summaries go to the Anthropic API for scoring by Haiku, by the owner's decision. Finnhub's terms bar sharing its data with third parties without written approval, and sending headlines to the Anthropic API arguably counts. The realistic consequence is that Finnhub could suspend the free key, which also carries the price feed, so prices and company news would stop together and the window would end. The owner accepts that risk.

* Only a headline of up to 300 characters and a summary of up to 600 is sent, never a value from the stock feed or anything derived from one, and only through the dedicated workspace key.
* Sonnet never sees a Finnhub item. Its scorer reads only items of the five official kinds whose source is not Finnhub.
* `[finnhub] send_to_llm = true` in `configs/v2/news.toml` carries the decision. Setting it false routes Finnhub items to FinBERT alone, which changes the news config hash and, after launch, ends the window under criterion 1.6.
* A written request to Finnhub describing this personal, non commercial scoring use would remove the doubt, and question 3 of the plan recommends sending one before launch. Nothing waits for the answer, and an objection turns `send_to_llm` off at once.

The decision is recorded in [section 13.1 of the plan](../PLAN_V2.md#131-why-stock-data-stays-private-and-the-finnhub-decision).

## The owner's decision on stock data in the assistant session

Any AI coding or chat session that builds, runs or checks this system sends what it reads to a third party. The approved plan first kept stock prices and anything derived from them out of such sessions. On 12 October 2026 the owner lifted that rule, so the assistant session may read prices, volumes, per symbol figures, recordings, databases and the private dashboard, run commands with `--detail`, and send dashboard screenshots to the owner. The realistic risk is the same as for headlines, that Finnhub suspends the free key. The rest of the data handling stays as it was.

* By default every version 2 command prints only counts, verdicts, hashes, timings of this system and synthetic figures, and `--detail` adds rows, per symbol figures and samples. `v2/output.py` refuses any other kind of value in a default summary, so a price cannot reach one by accident.
* Apart from the assistant session, no stock data leaves this Mac, whether through a page, a tunnel or a push.
* Logs hold states, counts and timings, never a price, a volume or a fill.
* Tests use synthetic data and hand written Finnhub fixtures only.

The decision is recorded in [section 13.3 of the plan](../PLAN_V2.md#133-stock-data-and-assistant-sessions).

## Local only dashboards

Every dashboard stays on this Mac, for version 1 and version 2 alike. The version 1 dashboard serves `127.0.0.1:8050` and the version 2 dashboard `127.0.0.1:8060`, neither is tunnelled, shared or published, and no tunnel of any kind runs on this Mac. The version 2 dashboard cannot be given another host, refuses port 8050, answers 403 to any Host header other than `127.0.0.1` or `localhost` on its port, which blocks DNS rebinding, refuses every method that could write, and opens every database read only.

The stock process proves this for the record. Ten minutes before each open, and at once when it starts inside that window or inside a session, it runs the local only check of `v2/localcheck.py` in a worker thread. The check lists listening sockets with `lsof` and `netstat`, running processes with `ps` and, when Tailscale is installed, its serve settings, and fails when anything listens on port 8050 or 8060 at an address other than `127.0.0.1`, or a tunnel or remote forward runs, such as `cloudflared`, `ngrok`, `tailscale funnel`, `tailscale serve` or `ssh -R`. Its verdict and counts go to both run databases as a `local_only` event, and `evaluate` fails criterion 1.8 for any failed check and for any session of the window without a passed one. The same check runs by hand as `btc_survivor_v2 check-local`, as [v2_operations.md](v2_operations.md#the-local-only-check) shows, and the rule is in [section 13.2 of the plan](../PLAN_V2.md#132-local-only-with-no-public-page).

## Secrets

* The Finnhub key, the Anthropic key and the contact address live in `~/.config/btc_survivor/secrets.env`, outside the repository, as `FINNHUB_API_KEY`, `ANTHROPIC_API_KEY` and `CONTACT_EMAIL`, one `KEY=value` line each.
* `v2/local_secrets.py` is the only reader. It refuses the file unless the user owns it and nobody else may read it, accepts only those three names, names a missing key without any value, and registers every value with `v2/redact.py` as soon as it reads it.
* `v2/redact.py` masks every registered value in log lines with their tracebacks, uncaught exceptions in any thread and the asyncio loop, recorded close reasons and every stored detail. The dashboard passes every string of every JSON body through it too.
* In the news process only, the Anthropic key goes into the process environment for the SDK, the other `ANTHROPIC_` variables are removed, and the process refuses to start while `ANTHROPIC_LOG` is set. The news process, the stock process and the raw recorder all refuse to start while a proxy or certificate variable such as `HTTPS_PROXY` or `SSL_CERT_FILE` is set, so no key or address can go through another host.
* The Finnhub token travels in the `X-Finnhub-Token` header for REST and in the websocket address built in memory, and is never logged.
* The contact address goes only into the request header for sec.gov, data.sec.gov and bls.gov. A test puts a sentinel in its place and checks that it never reaches a model request, a log line, a database row or a page, and `test_v2_contact_absent` checks that no tracked file holds it.
* `btc_survivor_v2 check-secrets` names the keys present and missing, never a value, and `scripts/v2_secret_scan.py` looks for every value in the repository, `data/v2`, the logs and the push range before every push.

The commands are in [v2_operations.md](v2_operations.md#checks-before-anything-starts), and the design in [section 8.9 of the plan](../PLAN_V2.md#89-credentials-and-redaction).

## Shorts, margin and day trading

The paper account is a simplified margin account. Each learner holds at most one position of about 500 dollars on 10,000 dollars of equity, and may go long or short.

**What the model does with shorts.**

* A short opens the nearest whole number of shares, at least one, because brokers do not short fractional shares. Its notional therefore runs from about 400 to 700 dollars at planning prices and is recorded on the fill.
* Borrow accrues at 1 percent a year on the entry notional over calendar time, about 0.006 bps for a 30 minute short, with no charge to open. Megacaps lend near the general collateral rate, so 1 percent errs harsh, although that rate is outside the research. The model assumes every short can always be borrowed.
* The sale fees fall on the opening leg of a short and the closing leg of a long, never on a purchase.

**The short sale restriction.** SEC Rule 201 trips when a stock falls 10 percent or more below its previous regular session closing price, and then for the rest of that day and the whole next day a short sale may only be displayed or executed at a price above the current national best bid, by the SEC's Rule 201 answers read on 12 October 2026. The model differs from it in three ways, each read from `SymbolGuards` in `v2/stocks/broker.py`.

* It refuses every new short open in the symbol while the restriction holds, which is harsher than the real rule, because without quotes it cannot tell a price above the bid.
* Its reference is the last healthy reference price of the previous session, which stands in for the official close.
* It cannot trigger when the symbol had no healthy reference price in the session before, and a split also trips it, because that reference is not adjusted for splits.

**Splits and dividends.** An overnight move of more than 25 percent keeps the symbol's features invalid until 391 fresh session minutes exist and writes a `split_guard` event. Intraday only trading means no dividend handling is needed.

**Margin.** Shorting needs a margin account. FINRA's investor page on margin, dated 4 June 2026 and read on 12 October 2026, says a margin account must keep at least 2,000 dollars of equity, and that brokers may set higher house requirements and apply margin requirements to positions held during the day, not only at the close. FINRA's margin page says Regulation T lets a broker lend up to 50 percent of the price of a new purchase. One position of 500 dollars on 10,000 dollars of equity never comes near either limit, and the model applies no margin rule at all.

**Day trading.** The plan was written with FINRA's pattern day trader rule, which limits a margin account under 25,000 dollars to three day trades in five business days, against about ten a week for each learner, a little over two round trips a session, forced and own together under the holding rule of 13 October 2026. That rule has since been replaced. FINRA Regulatory Notice `26-10` of 20 April 2026, read on 12 October 2026, says amendments to Rule 4210, filed as `SR-FINRA-2025-017` and approved by the SEC on 14 April 2026, replace the pattern day trader designation, its day trade count and its 25,000 dollar minimum with intraday margin standards, effective 4 June 2026. Under them a broker computes an intraday margin deficit on days with risk increasing trades, the customer must meet it promptly, and a practice of leaving deficits unmet past the fifth business day brings a 90 day freeze on new shorts and new debit balances. Brokers may phase the standards in until 20 October 2027, so in October 2026 a real broker may still apply the old count. With the new standards the paper account's day trading is close to what a small real account may do, and with the old count it is far kinder. Sections 5.3 and 18 of the plan state the new rule and the phase in.

**Why the test does not depend on any of this.** The confirmatory test reads only forecasts and labels, never fills or account values, so neither the short restriction nor any margin or day trading limit can change it. The rules are in [section 5.3 of the plan](../PLAN_V2.md#53-shorts) and its limits in [section 18](../PLAN_V2.md#18-risks-and-honest-limits).

## Fees

There is no commission. The SEC Section 31 fee is 20.60 dollars per million of sale proceeds, in force from 4 April 2026 until 60 days after the fiscal 2027 appropriation. The FINRA trading activity fee is 0.000195 dollars a share sold, capped at 9.79 dollars a trade. A FINRA filing pauses that fee from 1 October to 31 December 2026, and the model charges it throughout, the harsher reading. Each fee rounds up to the next cent per sale, with a minimum of one cent, so a sale of about 500 dollars pays 0.02 dollars of SEC fee and 0.01 of trading activity fee. The CAT fee of 0.000001 dollars a share is left out, and broker waivers are not modelled. These rates were read on 11 October 2026 outside the research and are rechecked against the SEC and FINRA pages before launch, as [section 5.2 of the plan](../PLAN_V2.md#52-fees) requires.

## Risks that remain

* Finnhub could suspend the free key over the headline decision or the assistant session decision, which stops prices and company news together and ends the window.
* The free websocket carries only a partial tape, a small share of consolidated volume, which the owner decided on 12 October 2026 to keep, so reference prices and fills rest on a small share of each symbol's trades. Finnhub could also narrow its tier later, and criterion 1.1 fails the run if the coverage moves by more than a factor of 2 from the gate session's.
* The CT Plan changes the terms for non professionals on 1 April 2027, so any run past that date needs the terms checked again.
* Whether the Console workspace limit applies to the Max plan's included credits is unverified, one reason the cap in code exists.

The full list is in [section 18 of the plan](../PLAN_V2.md#18-risks-and-honest-limits).
