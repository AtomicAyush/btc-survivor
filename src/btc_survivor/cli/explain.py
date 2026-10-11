"""The explain command, one bot's life story from its database rows and the newest checkpoint."""

from __future__ import annotations

from btc_survivor.cli import data
from btc_survivor.clock import MS_PER_HOUR, parse_iso
from btc_survivor.config import Config

PARTS = (("fees", "fees"), ("funding", "funding"), ("rent", "rent"), ("stop losses", "stop_loss"),
         ("trading losses", "other_losses"))


def life(cfg: Config, bot_id: int, now_ms: int) -> dict:
    conn = data.connect(cfg.db_path)
    try:
        bot = conn.execute("SELECT * FROM bots WHERE bot_id = ?", (bot_id,)).fetchone()
        if bot is None:
            raise LookupError(f"no bot {bot_id} in this run")
        death = conn.execute("SELECT * FROM deaths WHERE bot_id = ?", (bot_id,)).fetchone()
        lineage = conn.execute("SELECT * FROM lineage WHERE bot_id = ?", (bot_id,)).fetchone()
        meta = data.checkpoint_meta(conn)
        live = data.runtime(conn, now_ms)
        current = next((b for b in data.bots(conn, cfg, meta, live, now_ms) if b["bot_id"] == bot_id), None)
        state = next((d for d in (meta["bots"] if meta else []) if d["bot_id"] == bot_id), None)
        if death is not None:
            parts = {"fees": death["fees_total"], "funding": death["funding_total"], "rent": death["rent_total"],
                     "stop_loss": death["stop_loss_total"], "other_losses": death["other_losses_total"]}
            equity, writeoff, unrealized = death["final_equity"], death["writeoff"], 0.0
        elif state is not None:
            acct = state["account"]
            parts = {"fees": acct["fees"], "funding": acct["funding"], "rent": acct["rent"],
                     "stop_loss": acct["stop_loss_losses"], "other_losses": acct["other_losses"]}
            equity, writeoff = current["equity"], acct["writeoff"]
            unrealized = equity - acct["cash"]
        else:
            parts, equity, writeoff, unrealized = None, None, 0.0, 0.0
        worst = conn.execute("SELECT id, side, opened_at, closed_at, entry_price, exit_price, pnl_net, exit_reason "
                             "FROM trades WHERE bot_id = ? AND closed_at IS NOT NULL ORDER BY pnl_net LIMIT 5",
                             (bot_id,)).fetchall()
        counts = conn.execute("SELECT action, count(*) AS n, sum(forced) AS forced FROM decisions WHERE bot_id = ? "
                              "GROUP BY action ORDER BY n DESC", (bot_id,)).fetchall()
        trades = conn.execute("SELECT count(*) AS n, sum(pnl_net > 0) AS wins, coalesce(sum(pnl_net), 0) AS net "
                              "FROM trades WHERE bot_id = ? AND closed_at IS NOT NULL", (bot_id,)).fetchone()
    finally:
        conn.close()
    return {"bot": dict(bot), "death": dict(death) if death else None, "lineage": dict(lineage) if lineage else None,
            "current": current, "parts": parts, "equity": equity, "writeoff": writeoff, "unrealized": unrealized,
            "start": cfg.trading.start_equity, "worst": [dict(r) for r in worst],
            "counts": [dict(r) for r in counts], "trades": dict(trades)}


def render(x: dict) -> str:
    b, d = x["bot"], x["death"]
    kind = "coin flip twin" if b["origin"] == "control" else f"{b['origin']} learner"
    lines = [f"bot {b['bot_id']}, a {kind} in slot {b['slot']}, generation {b['generation']}, born {b['born_at']}"]
    if x["lineage"]:
        ln = x["lineage"]
        lines.append(f"  child of bot {ln['parent_id']} (fitness {ln['parent_fitness']:+.3f}), margin factor "
                     f"{ln['margin_factor']:.3f}, learning rate factor {ln['lr_factor']:.3f}")
    lines.append(f"  genes, margin {b['margin_bps']:.2f} bps, learning rate {b['lr']:.6f}")
    t = x["trades"]
    lines.append(f"  {t['n']} closed trades, {t['wins'] or 0} wins, net {t['net']:+,.2f} dollars")
    if x["parts"] is not None:
        total = sum(x["parts"].values()) - x["writeoff"] - x["unrealized"]
        lines.append(f"  equity {x['equity']:,.2f} of {x['start']:,.2f}, losses by part")
        for label, key in PARTS:
            lines.append(f"    {label:15s} {x['parts'][key]:>12,.2f}")
        if x["writeoff"]:
            lines.append(f"    {'writeoff':15s} {-x['writeoff']:>12,.2f}")
        if x["unrealized"]:
            lines.append(f"    {'open position':15s} {-x['unrealized']:>12,.2f}")
        ok = abs((x["start"] - x["equity"]) - total) < 1e-6 * max(1.0, x["start"])
        lines.append(f"    {'sum':15s} {total:>12,.2f}, start minus equity {x['start'] - x['equity']:,.2f}, "
                     f"{'the parts add up' if ok else 'THE PARTS DO NOT ADD UP'}")
    if x["counts"]:
        lines.append("  decisions " + ", ".join(f"{c['action']} {c['n']}" + (f" ({c['forced']} forced)" if c["forced"] else "")
                                               for c in x["counts"]))
    if x["worst"]:
        lines.append("  worst trades")
        for w in x["worst"]:
            lines.append(f"    trade {w['id']} {w['side']} {w['opened_at']} to {w['closed_at']}, {w['entry_price']:,.2f} to "
                         f"{w['exit_price']:,.2f}, net {w['pnl_net']:+,.2f}, closed by {w['exit_reason']}")
    if d:
        hours = (parse_iso(d["died_at"]) - parse_iso(b["born_at"])) / MS_PER_HOUR
        lines.append(f"  died {d['died_at']} after {hours / 24:.1f} days. Equity {d['final_equity']:,.2f} fell below the "
                     f"line of {d['threshold']:,.2f}, one round trip plus a day of rent. The largest loss was {d['cause']}.")
    elif x["current"]:
        c = x["current"]
        lines.append(f"  alive and {c['state']}, position {c['position']}, rolling skill "
                     f"{'-' if c['skill_corr'] is None else format(c['skill_corr'], '+.3f')}")
    return "\n".join(lines)
