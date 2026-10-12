"""The source must contain nothing that could place, cancel or sign an order, no broker client, no credential
handling outside the one secrets reader, model calls in one module and no model id but the two chosen."""

import ast
import re

from helpers import ROOT

API_KEY = r"api_?key"
SECRETS_READER = "src/btc_survivor/v2/local_secrets.py"
LLM_CLIENT = "src/btc_survivor/v2/news/llm_client.py"
MODEL_IDS = {"claude-haiku-5-5", "claude-sonnet-5-5"}
MODEL_ID = re.compile(r"claude-[\w.-]*\w", re.IGNORECASE)
NETWORK_MODULES = {
    "socket", "ssl", "select", "selectors", "socketserver", "asyncio", "http", "urllib", "ftplib", "smtplib",
    "poplib", "imaplib", "telnetlib", "xmlrpc", "webbrowser", "urllib3", "requests", "httpx", "httpx2",
    "aiohttp", "websocket", "websockets", "anthropic", "feed", "finnhub", "finnhub_news", "edgar", "edgar_docs",
    "rss", "llm_client", "dashboard", "subprocess", "multiprocessing", "ctypes",
}

FORBIDDEN = [
    r"place_?order",
    r"cancel_?order",
    r"create_?order",
    r"submit_?order",
    r"edit_?order",
    r"/orders\b",
    r"api_?secret",
    API_KEY,
    r"private_?key",
    r"passphrase",
    r"cb-access",
    r"hmac",
    r"jwt",
    r"(?<![a-z0-9])sign_?(order|request|message)",
    r"brokerage/orders",
    r"alpaca",
    r"trade-api",
    r"paper-api",
    r"time_?in_?force",
    r"order_?type",
    r"tradier",
    r"ibkr",
]


def scan_pattern(relative: str) -> re.Pattern:
    """The forbidden scan for one file, which drops the api_?key pattern for the secrets reader alone."""
    patterns = [p for p in FORBIDDEN if not (p == API_KEY and relative == SECRETS_READER)]
    return re.compile("|".join(patterns), re.IGNORECASE)


def imported_modules(source: str) -> set[str]:
    """Every module an import statement or a literal dynamic import names, relative imports as dotted tails."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}".strip(".") for alias in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if func in ("import_module", "__import__") and isinstance(node.args[0].value, str):
                names.add(node.args[0].value)
    names.discard("")
    return names


def _python_files():
    return sorted(p for d in ("src", "scripts") for p in (ROOT / d).rglob("*.py"))


def _source_files():
    """Every file under src and scripts of any type, skipping bytecode caches and build metadata."""
    files = (p for d in ("src", "scripts") for p in (ROOT / d).rglob("*") if p.is_file())
    return sorted(p for p in files if not any(part == "__pycache__" or part.endswith(".egg-info") for part in p.parts))


def _relative(path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_no_order_placement_or_credential_code_in_source():
    hits = []
    for path in _python_files():
        pattern = scan_pattern(_relative(path))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{_relative(path)}:{number}: {line.strip()}")
    assert not hits, "forbidden order or credential code found\n" + "\n".join(hits)


def test_the_scan_actually_finds_things():
    pattern = scan_pattern("src/btc_survivor/engine.py")
    for sample in ("def place_order(", "client.cancelOrder(", "POST /orders", "API_SECRET =", "import hmac",
                   "def _sign_request(self, body):", "token = build_jwt(key)", "hmac_sha256(secret, msg)"):
        assert pattern.search(sample), sample
    for innocent in ("assign_order = 1", "design_message"):
        assert not pattern.search(innocent), innocent


def test_the_scan_finds_broker_clients():
    pattern = scan_pattern("src/btc_survivor/v2/stocks/broker.py")
    for sample in ("import alpaca_trade_api", "from alpaca.trading.client import TradingClient",
                   "pip install alpaca-trade-api", "https://paper-api.alpaca.markets", "time_in_force='day'",
                   "TimeInForce.DAY", "order_type = 'market'", "OrderType.MARKET", "TradierClient(token)",
                   "from ibkr import Client", "IBKR_GATEWAY"):
        assert pattern.search(sample), sample
    for innocent in ("trader = None", "trade_count += 1", "paper trading only"):
        assert not pattern.search(innocent), innocent


def test_api_key_is_allowed_only_in_the_secrets_reader():
    reader = scan_pattern(SECRETS_READER)
    assert not reader.search("NAMES = ('FINNHUB_API_KEY', 'ANTHROPIC_API_KEY')")
    for sample in ("def place_order(", "import hmac", "API_SECRET =", "import alpaca"):
        assert reader.search(sample), sample
    for other in ("src/btc_survivor/v2/redact.py", "src/btc_survivor/v2/news/local_secrets.py",
                  "src/btc_survivor/local_secrets.py", "scripts/local_secrets.py", "src/btc_survivor/v2/config.py"):
        assert scan_pattern(other).search("ANTHROPIC_API_KEY"), other


def test_the_secrets_reader_imports_no_network_module():
    path = ROOT / SECRETS_READER
    assert path.is_file(), f"{SECRETS_READER} is missing"
    found = sorted(n for n in imported_modules(path.read_text(encoding="utf-8")) if NETWORK_MODULES & set(n.split(".")))
    assert not found, f"{SECRETS_READER} imports network modules {found}"


def test_only_the_model_client_imports_anthropic():
    importers = [_relative(p) for p in _python_files()
                 if any(n.split(".")[0] == "anthropic" for n in imported_modules(p.read_text(encoding="utf-8")))]
    assert set(importers) <= {LLM_CLIENT}, f"only {LLM_CLIENT} may import anthropic, found {importers}"


def test_import_finder_sees_every_form():
    for source in ("import anthropic", "from anthropic import AsyncAnthropic", "import anthropic.types as t",
                   "import importlib\nimportlib.import_module('anthropic')", "__import__('anthropic')"):
        assert any(n.split(".")[0] == "anthropic" for n in imported_modules(source)), source
    for source, network in (("import socket", True), ("from urllib.request import urlopen", True),
                            ("from .news import http", True), ("from ..feed.client import FeedClient", True),
                            ("import os, stat\nfrom . import redact\nfrom pathlib import Path", False)):
        found = any(NETWORK_MODULES & set(n.split(".")) for n in imported_modules(source))
        assert found is network, source


def test_no_model_id_but_haiku_and_sonnet_appears_in_source():
    found = []
    for path in _source_files():
        text = path.read_bytes().decode("utf-8", errors="replace")
        found.extend(f"{_relative(path)}: {m}" for m in MODEL_ID.findall(text) if m not in MODEL_IDS)
    assert not found, "model ids other than the two chosen found\n" + "\n".join(found)


def test_the_model_id_scan_finds_other_ids():
    for sample in ("claude-opus-5-5", "claude-3-opus-20240229", "anthropic.claude-opus-4-1-v1:0",
                   "Claude-Haiku-5-5", "claude-haiku-5-5-20261007", "claude-sonnet-4-5"):
        assert [m for m in MODEL_ID.findall(sample) if m not in MODEL_IDS], sample
    for sample in ('model="claude-haiku-5-5"', "`claude-sonnet-5-5`.", "claude-haiku-5-5, claude-sonnet-5-5"):
        assert MODEL_ID.findall(sample) and set(MODEL_ID.findall(sample)) <= MODEL_IDS, sample
