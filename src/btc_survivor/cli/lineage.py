"""The lineage command, the chain of parents from a bot back to its random ancestor."""

from __future__ import annotations

from btc_survivor.cli import data
from btc_survivor.clock import MS_PER_DAY, parse_iso
from btc_survivor.config import Config


def chain(cfg: Config, bot_id: int, now_ms: int) -> list[dict]:
    conn = data.connect(cfg.db_path)
    try:
        meta = data.checkpoint_meta(conn)
        living = {b["bot_id"]: b for b in data.bots(conn, cfg, meta, data.runtime(conn, now_ms), now_ms)}
        out, seen, current = [], set(), bot_id
        while current is not None and current not in seen:
            seen.add(current)
            bot = conn.execute("SELECT * FROM bots WHERE bot_id = ?", (current,)).fetchone()
            if bot is None:
                raise LookupError(f"no bot {current} in this run")
            death = conn.execute("SELECT * FROM deaths WHERE bot_id = ?", (current,)).fetchone()
            end = parse_iso(death["died_at"]) if death else (meta["now_ms"] if meta else now_ms)
            out.append({"bot_id": current, "generation": bot["generation"], "origin": bot["origin"],
                        "born_at": bot["born_at"], "lifespan_days": (end - parse_iso(bot["born_at"])) / MS_PER_DAY,
                        "alive": death is None, "cause": death["cause"] if death else None,
                        "equity": death["final_equity"] if death else living.get(current, {}).get("equity")})
            current = bot["parent_id"]
    finally:
        conn.close()
    return out


def render(rows: list[dict]) -> str:
    lines = []
    for i, r in enumerate(rows):
        equity = "-" if r["equity"] is None else f"{r['equity']:,.2f}"
        fate = "alive" if r["alive"] else f"died of {r['cause']}"
        lead = "bot" if i == 0 else "  parent" if i == 1 else "  " * i + "parent"
        lines.append(f"{lead} {r['bot_id']}, generation {r['generation']}, {r['origin']}, born {r['born_at']}, "
                     f"lived {r['lifespan_days']:.1f} days, equity {equity}, {fate}")
    return "\n".join(lines)
