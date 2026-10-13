# BTC Survivor

BTC Survivor runs a small population of tiny neural networks that paper trade Bitcoin against live public market data, around the clock, on a laptop or a Raspberry Pi. No real money is involved and no order is ever sent to an exchange. Each bot pays realistic fees, slippage and a daily running cost, dies when it can no longer cover its costs, and is replaced by a mutated copy of a strong survivor. The design and every decision behind it are in [PLAN.md](PLAN.md).

## Status

The project is being built in phases. Today it has the scaffolding, which means configuration loading and validation, time helpers, logging and the database schema. Trading, learning and the live feed arrive in later phases, described in [section 18 of the plan](PLAN.md#18-build-phases).

## Quick start

Create a virtual environment with Python 3.11 or newer, install the package with its test extra, copy the example configuration, create the database and run the tests.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
cp config.example.toml config.toml
.venv/bin/btc_survivor init-db
.venv/bin/pytest
```

Every setting is explained in [config.example.toml](config.example.toml).

## Version 2

Version 2 adds seven large US technology stocks and the QQQ fund, priced from Finnhub's free websocket, and news scored by language models, and asks one question before launch, whether scored news predicts the next 30 minute return. It lives on the `feat/v2-stocks` branch in its own worktree with the console script `btc_survivor_v2` and its own environments, and leaves version 1 and its runs untouched. Every command prints counts and verdicts by default and rows only with `--detail`, and every dashboard stays on `127.0.0.1`. The design and every decision behind it are in [PLAN_V2.md](PLAN_V2.md), and these guides explain it.

* [v2_overview.md](docs/v2_overview.md) explains what version 2 does and why, from prices and news to the confirmatory test.
* [v2_data_and_licensing.md](docs/v2_data_and_licensing.md) covers the data sources, the cost and fill model, shorts, margin, secrets and licensing.
* [v2_news.md](docs/v2_news.md) covers news collection, the model and FinBERT scoring, spend controls and the score streams.
* [v2_evaluation.md](docs/v2_evaluation.md) covers the `evaluate` command, the criteria, the outcomes, power, the planted news check and the learning rate.
* [v2_operations.md](docs/v2_operations.md) covers running the processes, the coverage report and stale thresholds, the rehearsal, the soak and the secret scan.
* [RESULTS_V2_TEMPLATE.md](docs/RESULTS_V2_TEMPLATE.md) is the empty template the results are written into after the window.

## License

BTC Survivor is released under the [MIT License](LICENSE).
