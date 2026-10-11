"""A read only local dashboard for one or more runs, served on this machine only.

It serves one self contained page and a JSON view built from the same read only queries as the status command, so
it can never change a run. It binds to 127.0.0.1 and needs nothing beyond the project's own dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
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
CHART_POINTS = 400
RECENT = 25


def run_state(cfg: Config, now_ms: int, name: str | None = None) -> dict:
    """Everything the page shows for one run, or an error entry, so one broken run never blanks the others."""
    name = name or cfg.data_dir.name
    try:
        conn = data.connect(cfg.db_path)
    except FileNotFoundError as exc:
        return {"name": name, "error": str(exc)}
    except sqlite3.Error as exc:
        return {"name": name, "error": f"cannot open {cfg.db_path}: {exc}"}
    try:
        return _state(conn, cfg, name, now_ms)
    except Exception as exc:  # noqa: BLE001, a report must show the problem rather than drop every run
        log.exception("dashboard could not read %s", cfg.db_path)
        return {"name": name, "error": f"cannot read {cfg.db_path}: {type(exc).__name__}: {exc}"}
    finally:
        conn.close()


def _state(conn, cfg: Config, name: str, now_ms: int) -> dict:
    meta = data.checkpoint_meta(conn)
    live = data.runtime(conn, now_ms)
    rows = data.bots(conn, cfg, meta, live, now_ms)
    return {
        "name": name, "summary": data.summary(conn, cfg, meta, rows, live, now_ms), "bots": rows,
        "config": {"fee_bps": cfg.costs.taker_fee_bps, "start_equity": cfg.trading.start_equity,
                   "rent_per_day": cfg.costs.daily_running_cost,
                   "notional": cfg.trading.notional, "horizon_minutes": cfg.time.horizon_minutes,
                   "history_minutes": cfg.time.history_minutes_required},
        "history": {"valid": meta["valid_total"] if meta else 0, "labels": meta["label_scale_n"] if meta else 0,
                    "labels_needed": cfg.features.min_samples},
        "equity_curve": equity_curve(conn), "skill": skill_so_far(conn, cfg.time.horizon_minutes),
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


def equity_curve(conn) -> list[dict]:
    """The benchmark table thinned to a few hundred points, oldest first."""
    count = conn.execute("SELECT count(*) FROM benchmark").fetchone()[0]
    step = max(1, math.ceil(count / CHART_POINTS))
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

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            elif self.path == "/api/state":
                now = time.time_ns() // 1_000_000
                state = clean({"now_ms": now, "page_version": page_version(),
                               "runs": [run_state(c, now, n) for c, n in zip(configs, names)]})
                body = json.dumps(state, allow_nan=False, default=str).encode()
                self._send(200, "application/json", body)
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code: int, kind: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
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
