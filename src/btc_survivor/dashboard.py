"""A read only local dashboard for one or more runs, served on this machine only.

It serves one self contained page and a JSON view built from the same read only queries as the status command, so
it can never change a run. It binds to 127.0.0.1 and needs nothing beyond the project's own dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from btc_survivor.cli import data
from btc_survivor.cli.evaluate import SKILL_LAGS, forecasts
from btc_survivor.cli.stats import diebold_mariano, skill_test
from btc_survivor.config import Config

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("dashboard.html")
STATE_TTL_S = 5.0  # one computed view is shared by every request for this long, so many viewers cost the same as one
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                               "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
                               "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
CHART_POINTS = 400
RECENT = 25


def run_state(cfg: Config, now_ms: int, name: str | None = None) -> dict:
    """Everything the page shows for one run, or an error entry, so one broken run never blanks the others."""
    name = name or cfg.data_dir.name
    try:
        conn = data.connect(cfg.db_path)
    except FileNotFoundError:
        return {"name": name, "error": f"run {name} has no database yet, has it been started"}
    except sqlite3.Error as exc:
        log.warning("dashboard could not open %s: %s", cfg.db_path, exc)
        return {"name": name, "error": f"the database of run {name} cannot be opened, see the dashboard log"}
    try:
        return _state(conn, cfg, name, now_ms)
    except Exception:  # noqa: BLE001, a report must show the problem rather than drop every run
        log.exception("dashboard could not read %s", cfg.db_path)
        return {"name": name, "error": f"the database of run {name} cannot be read, see the dashboard log"}
    finally:
        conn.close()


def _state(conn, cfg: Config, name: str, now_ms: int) -> dict:
    meta = data.checkpoint_meta(conn)
    live = data.runtime(conn, now_ms)
    rows = data.bots(conn, cfg, meta, live, now_ms)
    return {
        "name": name, "summary": data.summary(conn, cfg, meta, rows, live, now_ms), "bots": rows,
        "config": {"fee_bps": cfg.costs.taker_fee_bps, "start_equity": cfg.trading.start_equity,
                   "rent_per_day": cfg.costs.daily_running_cost, "candle_minutes": 1,
                   "force_hours": cfg.trading.force_trade_after_hours,
                   "notional": cfg.trading.notional, "horizon_minutes": cfg.time.horizon_minutes,
                   "history_minutes": cfg.time.history_minutes_required},
        "history": {"valid": meta["valid_total"] if meta else 0, "labels": meta["label_scale_n"] if meta else 0,
                    "labels_needed": cfg.features.min_samples},
        "equity_curve": equity_curve(conn), "chart_minutes_per_point": chart_step(conn), "skill": skill_so_far(conn, cfg.time.horizon_minutes),
        "decisions": [dict(r) for r in conn.execute(
            "SELECT d.ts, d.bot_id, b.origin, d.action, d.forced, d.prediction_bps, d.required_edge_bps, d.reason "
            "FROM decisions d JOIN bots b USING (bot_id) WHERE d.action != 'none' ORDER BY d.id DESC LIMIT ?",
            (RECENT,))],
        "fills": [dict(r) for r in conn.execute(
            "SELECT filled_at, decided_at, bot_id, leg, side, price, fee, status, reason FROM fills "
            "ORDER BY id DESC LIMIT ?", (RECENT,))],
        "events": [dict(r) for r in conn.execute(
            "SELECT ts, kind, detail FROM feed_events ORDER BY id DESC LIMIT ?", (RECENT,))],
    }


def chart_step(conn) -> int:
    """How many healthy minutes each chart point stands for, one until the run outgrows CHART_POINTS."""
    return max(1, math.ceil(conn.execute("SELECT count(*) FROM benchmark").fetchone()[0] / CHART_POINTS))


def equity_curve(conn) -> list[dict]:
    """The benchmark table thinned to a few hundred points, oldest first, always ending at the newest minute."""
    step = chart_step(conn)
    rows = conn.execute("SELECT ts, btc_mid, buy_hold_equity, matched_equity, living_mean_equity, twin_mean_equity, "
                        "cash_equity FROM benchmark WHERE (rowid - 1) % ? = 0 OR rowid = (SELECT max(rowid) "
                        "FROM benchmark) ORDER BY ts", (step,)).fetchall()
    return [dict(r) for r in rows]


def skill_so_far(conn, horizon: int) -> dict:
    """Descriptive skill on the same rows and with the same statistics as criterion 1. Only the final day counts."""
    out = {}
    for name, rows in forecasts(conn, horizon).items():
        pred = np.array([p for p, _ in rows.values()])
        label = np.array([y for _, y in rows.values()])
        out[name] = {"windows": len(pred), "corr": skill_test(pred, label, SKILL_LAGS)["corr"],
                     "mse_vs_zero": diebold_mariano(label - pred, label, SKILL_LAGS)["mse_ratio"]}
    return out


def clean(value):
    """Replace infinities and NaN with None so the page always receives valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def page_version() -> str:
    """Changes whenever the page file changes, so an open tab can reload itself onto a new version."""
    return str(PAGE.stat().st_mtime_ns)


def run_names(configs: list[Config]) -> list[str]:
    """Tab names from the data folders, numbered when two runs share a folder name so every run can be opened."""
    names, seen = [], Counter(c.data_dir.name for c in configs)
    used: Counter = Counter()
    for c in configs:
        base = c.data_dir.name
        used[base] += 1
        names.append(base if seen[base] == 1 else f"{base} {used[base]}")
    return names


def make_handler(configs: list[Config]):
    names = run_names(configs)
    cache = {"at": -STATE_TTL_S, "body": b""}
    lock = threading.Lock()

    def state_body() -> bytes:
        """The JSON view, recomputed at most once per STATE_TTL_S however many requests arrive."""
        with lock:
            if time.monotonic() - cache["at"] >= STATE_TTL_S:
                now = time.time_ns() // 1_000_000
                state = clean({"now_ms": now, "page_version": page_version(),
                               "runs": [run_state(c, now, n) for c, n in zip(configs, names)]})
                cache["body"] = json.dumps(state, allow_nan=False, default=str).encode()
                cache["at"] = time.monotonic()
            return cache["body"]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            elif self.path == "/api/state":
                self._send(200, "application/json", state_body())
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code: int, kind: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            log.debug("dashboard %s", fmt % args)

    return Handler


def serve(configs: list[Config], port: int) -> None:
    """Serve until Ctrl+C on 127.0.0.1 only, so nothing outside this machine can reach it."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(configs))
    print(f"dashboard for {', '.join(c.data_dir.name for c in configs)} at http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
