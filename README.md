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

## License

BTC Survivor is released under the [MIT License](LICENSE).
