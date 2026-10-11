"""Checkpoint encoding: JSON for scalars and generator states, one raw float64 blob for arrays.

A plain byte layout is used instead of npz because zip archives embed the current time, which would make two
identical replays write different bytes. There is no pickle anywhere.
"""

from __future__ import annotations

import json
import zlib
from collections import deque
from dataclasses import asdict

import numpy as np

from btc_survivor.account import Account
from btc_survivor.bot import Bot
from btc_survivor.config import Config
from btc_survivor.controls import TwinBot
from btc_survivor.features import N_FEATURES
from btc_survivor.genes import Genes
from btc_survivor.model import MLP
from btc_survivor.types import Leg, Order

FORMAT_VERSION = 2  # 2 added the linear baseline


class Arrays:
    """Collects named arrays into one blob."""

    def __init__(self) -> None:
        self.items: dict[str, np.ndarray] = {}

    def put(self, key: str, array: np.ndarray) -> None:
        self.items[key] = np.ascontiguousarray(array, dtype=np.float64)

    def encode(self) -> tuple[list[list], bytes]:
        layout, chunks, offset = [], [], 0
        for key in sorted(self.items):
            data = self.items[key].tobytes()
            layout.append([key, list(self.items[key].shape), offset, len(data)])
            chunks.append(data)
            offset += len(data)
        return layout, zlib.compress(b"".join(chunks), 1)


def encode(meta: dict, arrays: Arrays) -> tuple[str, bytes]:
    layout, blob = arrays.encode()
    meta = {**meta, "format": FORMAT_VERSION, "layout": layout}
    return json.dumps(meta, sort_keys=True, separators=(",", ":")), blob


def decode(meta_text: str, blob: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Read a checkpoint back. Raises ValueError for anything unreadable, including a corrupt blob."""
    try:
        meta = json.loads(meta_text)
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format {meta.get('format')!r}")
        raw = zlib.decompress(blob)
        arrays = {}
        for key, shape, offset, length in meta["layout"]:
            if offset + length > len(raw):
                raise ValueError(f"array {key} runs past the end of the blob")
            arrays[key] = np.frombuffer(raw[offset:offset + length], dtype=np.float64).reshape(shape).copy()
    except (zlib.error, KeyError, TypeError) as exc:
        raise ValueError(f"unreadable checkpoint: {exc}") from exc
    return meta, arrays


def rng_from_state(state: dict) -> np.random.Generator:
    bit_generator = getattr(np.random, state["bit_generator"])()
    bit_generator.state = state
    return np.random.Generator(bit_generator)


def dump_order(order: Order | None) -> dict | None:
    if order is None:
        return None
    return {"bot_id": order.bot_id, "legs": [[l.action, l.position_side, l.qty] for l in order.legs],
            "decided_ms": order.decided_ms, "due_ms": order.due_ms, "ref_mid": order.ref_mid,
            "reason": order.reason, "forced": order.forced}


def load_order(d: dict | None) -> Order | None:
    if d is None:
        return None
    legs = [Leg(action, side, qty) for action, side, qty in d["legs"]]
    return Order(d["bot_id"], legs, d["decided_ms"], d["due_ms"], d["ref_mid"], d["reason"], d["forced"])


def dump_bot(bot: Bot, arrays: Arrays) -> dict:
    """Everything that defines a bot. Arrays go into the shared blob under the bot id."""
    key = str(bot.bot_id)
    arrays.put(f"{key}.params", bot.model.params)
    arrays.put(f"{key}.adam_m", bot.opt.m)
    arrays.put(f"{key}.adam_v", bot.opt.v)
    arrays.put(f"{key}.skill_pred", bot.skill.pred)
    arrays.put(f"{key}.skill_label", bot.skill.label)
    return {
        "bot_id": bot.bot_id, "twin": isinstance(bot, TwinBot), "slot": bot.slot, "generation": bot.generation,
        "born_ms": bot.born_ms, "origin": bot.origin, "parent_id": bot.parent_id,
        "genes": [bot.genes.margin_bps, bot.genes.lr], "state": bot.state, "trained_minutes": bot.trained_minutes,
        "alive_minutes": bot.alive_minutes, "minutes_since_open": bot.minutes_since_open,
        "warmup_minutes": bot.warmup_minutes, "pending_preds": sorted(bot.pending_preds.items()),
        "skill_n": bot.skill.n, "skill_head": bot.skill.head, "adam_t": bot.opt.t,
        "hourly_equity": list(bot.hourly_equity), "equity": bot.equity, "death_threshold": bot.death_threshold,
        "account": asdict(bot.account), "pending_order": dump_order(bot.pending_order),
        "rng": bot.rng.bit_generator.state,
    }


def load_bot(cfg: Config, d: dict, arrays: dict[str, np.ndarray]) -> Bot:
    key = str(d["bot_id"])
    rng = rng_from_state(d["rng"])
    if d["twin"]:
        bot: Bot = TwinBot(cfg, d["bot_id"], d["slot"], d["born_ms"], rng)
    else:
        model = MLP(N_FEATURES, cfg.model.hidden_units, params=arrays[f"{key}.params"])
        bot = Bot(cfg, d["bot_id"], d["slot"], d["generation"], d["born_ms"], Genes(*d["genes"]), model, rng,
                  origin=d["origin"], parent_id=d["parent_id"])
    bot.model.params[:] = arrays[f"{key}.params"]
    bot.opt.m, bot.opt.v, bot.opt.t = arrays[f"{key}.adam_m"], arrays[f"{key}.adam_v"], d["adam_t"]
    bot.skill.pred, bot.skill.label = arrays[f"{key}.skill_pred"], arrays[f"{key}.skill_label"]
    bot.skill.n, bot.skill.head = d["skill_n"], d["skill_head"]
    bot.state, bot.trained_minutes, bot.alive_minutes = d["state"], d["trained_minutes"], d["alive_minutes"]
    bot.minutes_since_open, bot.warmup_minutes = d["minutes_since_open"], d["warmup_minutes"]
    bot.pending_preds = {int(t): p for t, p in d["pending_preds"]}
    bot.hourly_equity = deque(d["hourly_equity"], maxlen=bot.hourly_equity.maxlen)
    bot.equity, bot.death_threshold = d["equity"], d["death_threshold"]
    bot.account = Account(**d["account"])
    bot.pending_order = load_order(d["pending_order"])
    return bot
