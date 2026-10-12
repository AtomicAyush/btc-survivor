"""Compare each run's config file with the config text the run stored at launch, read only.

Engine.fresh writes the whole config file's text into the meta table under config_toml, next to master_seed, when a
run is first launched. It is written once and never compared afterwards, because a resume only checks the seed, the
population size and the twin setting (_check_same_run in engine.py). A config edited before a restart would
therefore change a running experiment without any warning, and this script is the check to run before one.

For each config it prints match when the file's text is the stored text, the same settings when only comments or
layout differ, and otherwise every setting that differs, with the launch value and the file's value. A change to
run.log_level is named but does not change the experiment. Checkpoints hold no copy of the config. The file is only
parsed as TOML, never validated, so a file the run would now refuse is still compared key by key.

    python scripts/check_config_integrity.py configs/run_a.toml configs/run_b.toml ... [--copies DIR]

With --copies the database is read from DIR/<run folder>/btc_survivor.db, a copy, instead of the live run. Every
database is opened read only. The exit status is 0 when every run matches or holds the same settings, 1 when a
setting differs and 2 when a database or its stored config is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from pathlib import Path

from btc_survivor.cli import data

HARMLESS = {"run.log_level"}  # read only to choose how much the run logs
MISSING = object()


def flatten(table: dict, prefix: str = "") -> dict[str, object]:
    """Dotted keys for every value, so taker_fee_bps_retail_default under [costs] becomes
    costs.taker_fee_bps_retail_default."""
    out: dict[str, object] = {}
    for key, value in table.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def differences(stored: str, current: str) -> list[tuple[str, object, object]]:
    """Every dotted key whose value differs, with the stored value and the file's, MISSING where one has no such key."""
    a, b = flatten(tomllib.loads(stored)), flatten(tomllib.loads(current))
    return [(key, a.get(key, MISSING), b.get(key, MISSING)) for key in sorted(a.keys() | b.keys())
            if a.get(key, MISSING) != b.get(key, MISSING)]


def database(config_path: Path, current: str, copies: Path | None) -> Path:
    """The run's database, from run.data_dir resolved against the config's folder as the engine does, or the copy
    in copies/<run folder> when copies is given."""
    folder = Path(tomllib.loads(current)["run"]["data_dir"])
    if not folder.is_absolute():
        folder = Path(config_path).resolve().parent / folder
    return (copies / folder.name if copies is not None else folder) / "btc_survivor.db"


def check(config_path: Path, copies: Path | None = None) -> dict:
    """The verdict for one config, with the stored and current hashes and any differing settings."""
    current = Path(config_path).read_text(encoding="utf-8")
    db = database(config_path, current, copies)
    out = {"config": str(config_path), "run": db.parent.name, "db": str(db),
           "file_sha256": hashlib.sha256(current.encode()).hexdigest()}
    try:
        conn = data.connect(db)
    except FileNotFoundError:
        return {**out, "verdict": "missing", "detail": "no database"}
    try:
        stored = data.meta_value(conn, "config_toml")
        seed = data.meta_value(conn, "master_seed")
    finally:
        conn.close()
    if stored is None:
        return {**out, "verdict": "missing", "detail": "the database holds no config_toml in meta"}
    out["stored_sha256"] = hashlib.sha256(stored.encode()).hexdigest()
    out["stored_seed"] = seed
    if stored == current:
        return {**out, "verdict": "match", "differences": []}
    diffs = differences(stored, current)
    if not diffs:
        return {**out, "verdict": "same settings", "differences": []}
    harmless = all(key in HARMLESS for key, _, _ in diffs)
    return {**out, "verdict": "logging only" if harmless else "differs", "differences": diffs}


def _show(value: object) -> str:
    return "absent" if value is MISSING else repr(value)


def render(result: dict) -> str:
    head = f"{result['run']}, {result['config']} against {result['db']}"
    if result["verdict"] == "missing":
        return f"{head}\n   cannot check, {result['detail']}"
    lines = [head, f"   file sha256 {result['file_sha256'][:16]}, stored at launch {result['stored_sha256'][:16]}, "
                   f"stored master_seed {result['stored_seed']}"]
    if result["verdict"] == "match":
        lines.append("   match, the file is the text the run stored at launch")
    elif result["verdict"] == "same settings":
        lines.append("   same settings, the text differs only in comments or layout")
    else:
        lines.append(f"   {'only logging differs' if result['verdict'] == 'logging only' else 'DIFFERS'}, "
                     f"{len(result['differences'])} settings changed since launch")
        for key, was, now in result["differences"]:
            note = ", logging only" if key in HARMLESS else ""
            lines.append(f"   {key} was {_show(was)} at launch, the file says {_show(now)}{note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare run configs with the text each run stored at launch. "
                                                 "Read only.")
    parser.add_argument("configs", nargs="+", type=Path, help="run configs, such as configs/run_a.toml")
    parser.add_argument("--copies", type=Path, help="read DIR/<run folder>/btc_survivor.db instead of the live run")
    args = parser.parse_args(argv)
    status = 0
    for path in args.configs:
        result = check(path, args.copies)
        print(render(result))
        if result["verdict"] == "missing":
            status = 2
        elif result["verdict"] == "differs":
            status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
