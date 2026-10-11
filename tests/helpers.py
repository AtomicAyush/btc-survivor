"""Shared test helpers."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = ROOT / "config.example.toml"


def example_dict() -> dict[str, Any]:
    with EXAMPLE_CONFIG.open("rb") as fh:
        return copy.deepcopy(tomllib.load(fh))


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    return str(v)


def dump_toml(data: dict[str, Any]) -> str:
    """Serialize a two level dict of simple values, enough for config tests."""
    lines: list[str] = []
    for section, table in data.items():
        lines.append(f"[{section}]")
        for key, value in table.items():
            lines.append(f"{key} = {_value(value)}")
        lines.append("")
    return "\n".join(lines)


def write_config(folder: Path, data: dict[str, Any] | None = None) -> Path:
    """Write a config file, by default the example, into folder and return its path."""
    path = folder / "config.toml"
    path.write_text(dump_toml(data if data is not None else example_dict()), encoding="utf-8")
    return path
