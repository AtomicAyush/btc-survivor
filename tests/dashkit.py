"""The decisions explorer worked out again in plain Python, for the dashboard tests.

Which rows a filter keeps, in what order, with what counts per action and what summary, for decisions and skipped
turns together, and which turns each learner skipped, found minute by minute from the tables with nothing shared
with the dashboard's own code except the reason text a skipped turn carries.
"""

from __future__ import annotations

import urllib.parse

from btc_survivor.cli import data
from btc_survivor.clock import MS_PER_MINUTE, iso, parse_iso

TWIN = "control"
FILTERS = [
    {}, {"bot": "1"}, {"bot": "6"}, {"kind": "learner"}, {"kind": "twin"}, {"action": "close"}, {"action": "flip"},
    {"action": "hold"}, {"action": "none"}, {"action": "open_long,open_short"}, {"forced": "1"}, {"forced": "0"},
    {"gate": "1"}, {"gate": "0"}, {"outcome": "matured"}, {"outcome": "right"}, {"outcome": "wrong"},
    {"outcome": "waiting"}, {"outcome": "missed"}, {"q": "forced"}, {"q": "FORCED BY"}, {"q": "%"}, {"q": "_"},
    {"q": "\\"}, {"q": "coin chose long"}, {"from": "2026-10-03 04:45"}, {"to": "2026-10-03T04:40:00Z"},
    {"from": "2026-10-03 04:30", "to": "2026-10-03 04:50:30.5"}, {"to": "2026-10-03"}, {"from": "2026-10-04"},
    {"hours": "1"},
    {"kind": "learner", "forced": "1", "action": "open_short"},
    {"bot": "1", "gate": "1", "outcome": "right"},
    {"kind": "twin", "q": "coin chose long", "from": "2026-10-03 04:30", "to": "2026-10-03 04:55"},
    {"hours": "1", "outcome": "matured", "action": "close,open_long", "gate": "0"},
    {"kind": "learner", "outcome": "wrong", "forced": "0", "q": "below", "action": "none,close,hold"},
    {"kind": "twin", "outcome": "right"}, {"kind": "twin", "outcome": "wrong"},
    {"kind": "twin", "outcome": "right", "action": "flip,open_short"}, {"outcome": "wrong", "action": "flip"},
]


def decision_rows(cfg) -> tuple[list[dict], str]:
    """Every decision row with its bot's origin, and the run's clock."""
    conn = data.connect(cfg.db_path)
    rows = [dict(r) for r in conn.execute("SELECT d.*, b.origin FROM decisions d JOIN bots b USING (bot_id)")]
    clock = conn.execute("SELECT max(ts) FROM benchmark").fetchone()[0]
    conn.close()
    return rows, clock


def skipped_by_rule(cfg) -> set[tuple[int, str]]:
    """(bot id, minute) for every turn a learner skipped, checked minute by minute.

    A learner is a bot whose origin is not control. It has a turn at every minute whose index modulo the horizon is its
    slot modulo the horizon, from its birth, exclusive, to its death, which is the earlier of died_at and the birth of
    the next learner in its slot, exclusive. Turns count from the run's first decision, but not before its first
    forecast row, up to the newest checkpoint, exclusive, and a turn with no decision row of its own is skipped.
    """
    conn = data.connect(cfg.db_path)
    newest = conn.execute("SELECT ts FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
    first = conn.execute("SELECT min(ts) FROM decisions").fetchone()[0]
    predicted = conn.execute("SELECT min(ts) FROM forecasts").fetchone()[0]
    bots = [dict(r) for r in conn.execute("SELECT * FROM bots")]
    decided = {(r[0], r[1]) for r in conn.execute("SELECT bot_id, ts FROM decisions")}
    conn.close()
    if newest is None or first is None:
        return set()
    start = parse_iso(max(first, predicted or first))
    start += -start % MS_PER_MINUTE
    horizon, out = cfg.time.horizon_minutes, set()
    learners = [b for b in bots if b["origin"] != TWIN]
    for b in learners:
        later = sorted(o["born_at"] for o in learners
                       if o["slot"] == b["slot"] and (o["born_at"], o["bot_id"]) > (b["born_at"], b["bot_id"]))
        ends = [t for t in (b["died_at"], later[0] if later else None) if t]
        end = min(ends) if ends else None
        at = start
        while iso(at) < newest[0]:
            ts = iso(at)
            if ((at // MS_PER_MINUTE) % horizon == b["slot"] % horizon and b["born_at"] < ts
                    and (end is None or ts < end) and (b["bot_id"], ts) not in decided):
                out.add((b["bot_id"], ts))
            at += MS_PER_MINUTE
    return out


def skipped_rows(cfg) -> list[dict]:
    """The skipped turns as explorer rows, after checking the dashboard finds exactly the turns the rule finds."""
    from btc_survivor.dashboard import SKIPPED, skipped_turns

    conn = data.connect(cfg.db_path)
    found = skipped_turns(conn, cfg)
    origins = {r[0]: r[1] for r in conn.execute("SELECT bot_id, origin FROM bots")}
    conn.close()
    assert {(s.bot_id, s.ts) for s in found.values()} == skipped_by_rule(cfg)
    return [{"id": s.id, "bot_id": s.bot_id, "ts": s.ts, "origin": origins[s.bot_id], "action": SKIPPED,
             "reason": s.reason, "prediction_bps": None, "gate_passed": None, "forced": None, "label_bps": None}
            for s in found.values()]


def expected(cfg, params: dict) -> tuple[list[int], dict, dict]:
    """The ids a request should return, newest first, its counts per action and its summary, worked out in Python."""
    from btc_survivor.dashboard import LIST_ACTIONS, SKIPPED, parse_when

    rows, clock = decision_rows(cfg)
    rows += skipped_rows(cfg)
    cutoff = iso(parse_iso(clock) - cfg.time.horizon_minutes * 60_000)

    def call(r):
        """A learner's prediction, or the side a twin's coin chose, long as 1 and short as minus 1."""
        if r["origin"] != TWIN:
            return r["prediction_bps"]
        for side, sign in (("long", 1), ("short", -1)):
            if r["action"] == "open_" + side or (r["action"] == "flip" and r["reason"].endswith("chose " + side)):
                return sign
        return None

    def outcome(r, value):
        if r["action"] == SKIPPED:  # a skipped turn has no label at all
            return False
        label, aim = r["label_bps"], call(r)
        known = label is not None and aim is not None
        return {"matured": label is not None, "right": known and aim * label > 0,
                "wrong": known and aim * label < 0, "waiting": label is None and r["ts"] > cutoff,
                "missed": label is None and r["ts"] <= cutoff}[value]

    tests = {
        "bot": lambda r, v: r["bot_id"] == int(v),
        "kind": lambda r, v: (r["origin"] == TWIN) == (v == "twin"),
        "forced": lambda r, v: r["forced"] == int(v),
        "gate": lambda r, v: r["gate_passed"] == int(v),
        "outcome": outcome,
        "q": lambda r, v: v.lower() in r["reason"].lower(),
        "from": lambda r, v: r["ts"] >= parse_when(v, "from"),
        "to": lambda r, v: r["ts"] <= parse_when(v, "to", end=True),
        "hours": lambda r, v: r["ts"] > iso(parse_iso(clock) - int(v) * 3_600_000),
    }
    kept = [r for r in rows if all(tests[k](r, v) for k, v in params.items() if k != "action")]
    counts = {a: sum(r["action"] == a for r in kept) for a in LIST_ACTIONS}
    if "action" in params:
        kept = [r for r in kept if r["action"] in params["action"].split(",")]
    kept.sort(key=lambda r: (r["ts"], r["id"]), reverse=True)
    summary = {"total": len(kept), "skipped": sum(r["action"] == SKIPPED for r in kept),
               "forced": sum(r["forced"] or 0 for r in kept), "gate_passed": sum(r["gate_passed"] or 0 for r in kept),
               "matured": sum(r["label_bps"] is not None for r in kept),
               "waiting": sum(outcome(r, "waiting") for r in kept), "missed": sum(outcome(r, "missed") for r in kept),
               "right": sum(outcome(r, "right") for r in kept), "wrong": sum(outcome(r, "wrong") for r in kept)}
    return [r["id"] for r in kept], counts, summary


def fetch_all(get, run: str, params: dict, per: int = 200) -> tuple[list[dict], dict]:
    """Every row a filter returns, walking its pages, and the first page's reply."""
    rows, page, first = [], 1, None
    while True:
        code, reply = get("/api/decisions?" + urllib.parse.urlencode({"run": run, **params, "page": page,
                                                                         "per": per}))
        assert code == 200, reply
        first = first or reply
        assert reply["page"] == page and reply["total"] == first["total"]
        rows += reply["rows"]
        if page >= reply["pages"]:
            return rows, first
        page += 1
