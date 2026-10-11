# Contributing to BTC Survivor

Thank you for helping. This project is a careful experiment, so a few rules matter more than usual. The full design is in [PLAN.md](PLAN.md).

## Ground rules

The code must never place, cancel or sign an order on any exchange, and it may only use public market data that needs no credentials. Never add API keys or ask for them. Replay exists to test plumbing and is never used to choose, tune or rank strategies. A test in [tests/test_no_orders.py](tests/test_no_orders.py) scans the source for order placement and credential code.

## Development setup

Use Python 3.11 or newer in a virtual environment and install the package in editable mode with the test extra. Run the whole suite with pytest before every commit. The steps are in the [README](README.md).

## Making changes

Keep modules small and single purpose, and prefer the simplest implementation that meets the plan. Before changing existing code, list the modules and tests that depend on it. Keep code comments to one or two sentences per function, and only where they help. Every change comes with tests, and the suite must pass on Python 3.11 and on the newest supported Python. The module layout is in [section 5 of the plan](PLAN.md#5-module-layout).

## Commits

Work on a feature branch, never push to main, and write every commit message as a single line. Do not add co author lines or any automated attribution to commits, files or docs. The branch rules are in [section 18 of the plan](PLAN.md#18-build-phases).

## Documentation style

Write documentation in plain prose with no emojis, and without colons, semicolons, hyphens or em dashes in the prose. Code, config and file names are exempt. Every section of the README and of the docs ends with links to the relevant docs or external references. The test in [tests/test_docs_style.py](tests/test_docs_style.py) checks these rules.
