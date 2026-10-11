"""The source must contain nothing that could place, cancel or sign an order, and no credential handling."""

import re

from helpers import ROOT

FORBIDDEN = [
    r"place_?order",
    r"cancel_?order",
    r"create_?order",
    r"submit_?order",
    r"edit_?order",
    r"/orders\b",
    r"api_?secret",
    r"api_?key",
    r"private_?key",
    r"passphrase",
    r"cb-access",
    r"hmac",
    r"jwt",
    r"(?<![a-z0-9])sign_?(order|request|message)",
    r"brokerage/orders",
]


def test_no_order_placement_or_credential_code_in_source():
    pattern = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
    hits = []
    for path in sorted(p for d in ("src", "scripts") for p in (ROOT / d).rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not hits, "forbidden order or credential code found\n" + "\n".join(hits)


def test_the_scan_actually_finds_things():
    pattern = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
    for sample in ("def place_order(", "client.cancelOrder(", "POST /orders", "API_SECRET =", "import hmac",
                   "def _sign_request(self, body):", "token = build_jwt(key)", "hmac_sha256(secret, msg)"):
        assert pattern.search(sample), sample
    for innocent in ("assign_order = 1", "design_message"):
        assert not pattern.search(innocent), innocent
