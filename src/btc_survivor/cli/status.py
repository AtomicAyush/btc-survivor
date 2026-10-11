"""The status command, a table of the living bots, the live price, the benchmarks and a summary line."""

from __future__ import annotations

import time

from btc_survivor.cli import data
from btc_survivor.config import Config

COLUMNS = ("id", "gen", "age h", "state", "equity", "trades", "win", "fees", "rent", "skill", "position",
           "burn/day", "days left", "forced in h")


def fmt(value, spec: str = "", missing: str = "-") -> str:
    return missing if value is None else format(value, spec)


def state_text(bot: dict) -> str:
    if bot["state"] == "warmup" and bot["warmup_minutes"]:
        return f"warmup {min(bot['trained_minutes'] / bot['warmup_minutes'], 1):.0%}"
    return bot["state"]


def bot_line(b: dict) -> list[str]:
    return [str(b["bot_id"]), str(b["generation"]), fmt(b["age_hours"], ".1f"), state_text(b), fmt(b["equity"], ",.2f"),
            str(b["trades"]), fmt(b["win_rate"], ".0%"), fmt(b["fees"], ",.2f"), fmt(b["rent"], ",.2f"),
            fmt(b["skill_corr"], "+.3f"), b["position"], fmt(b["burn_per_day"], ",.2f"), fmt(b["days_left"], ",.0f"),
            fmt(b["forced_in_hours"], ".1f")]


def table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in [list(COLUMNS)] + rows) for i in range(len(COLUMNS))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(COLUMNS, widths))]
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(lines)


def render(cfg: Config, now_ms: int, twins: bool = False) -> str:
    conn = data.connect(cfg.db_path)
    try:
        meta = data.checkpoint_meta(conn)
        live = data.runtime(conn, now_ms)
        rows = data.bots(conn, cfg, meta, live, now_ms)
        s = data.summary(conn, cfg, meta, rows, live, now_ms)
    finally:
        conn.close()
    price = f"BTC {s['price']:,.2f} ({s['price_age_s']:.0f} s old)" if s["price"] else "BTC price unknown"
    out = [f"{cfg.data_dir.name}  feed {s['feed_state']}  {price}  launched {s['launch_ts'] or 'not yet'}, "
           f"{s['healthy_days']:.2f} healthy days", f"stage: {s['stage']}", ""]
    learners = sorted((b for b in rows if b["kind"] == "learner"), key=lambda b: b["slot"])
    out.append(table([bot_line(b) for b in learners]) if learners else "no bots yet")
    if twins:
        out += ["", "coin flip twins", table([bot_line(b) for b in sorted(
            (b for b in rows if b["kind"] == "twin"), key=lambda b: b["slot"])])]
    out.append("")
    if s["buy_hold"] is not None:
        out.append(f"buy and hold {s['buy_hold']:,.2f}, exposure matched {s['matched']:,.2f}, cash {s['cash']:,.2f}"
                   + (f", twin mean {s['twin_mean']:,.2f}" if s["twin_mean"] is not None else ""))
    if s["mean_equity"] is not None:
        out.append(f"living {s['living']} with mean {s['mean_equity']:,.2f} and median {s['median_equity']:,.2f}, "
                   f"{s['deaths']} deaths, all {s['born']} learners ever born hold {s['all_bots_equity']:,.2f} "
                   f"of {s['all_bots_capital']:,.2f} given")
    out += [f"warning: {w}" for w in s["warnings"]]
    if s["checkpoint_age_s"] is not None:
        out.append(f"bot rows come from the checkpoint written {s['checkpoint_age_s']:.0f} s ago, and can be up to "
                   f"{cfg.persistence.checkpoint_every_seconds} s old")
    return "\n".join(out)


def watch(cfg: Config, twins: bool, every_s: float = 60.0) -> None:
    """Redraw the status every minute until Ctrl+C."""
    try:
        while True:
            print("\033[2J\033[H" + render(cfg, time.time_ns() // 1_000_000, twins), flush=True)
            time.sleep(every_s)
    except KeyboardInterrupt:
        pass
