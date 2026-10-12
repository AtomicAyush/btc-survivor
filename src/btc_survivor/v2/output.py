"""What version 2 commands print. By default only counts, verdicts, hashes, timings and state words, and rows only
with --detail, which is for the owner's own terminal and never for an assistant session.

The default summary refuses any other type, so a price, a volume or a fill cannot slip into it by accident, and a
title or label holds only letters, spaces and underscores, so no figure can ride in one either.
"""

from __future__ import annotations

import argparse
import numbers
import re
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from btc_survivor.v2.redact import redact

DETAIL_HELP = "also print rows and per symbol figures, only in your own terminal and never in an assistant session"


WORD = re.compile(r"[a-z][a-z_ ]*")


@dataclass(frozen=True)
class Count:
    """A count of this system's rows, events, minutes or checks, never a market figure such as a share volume."""
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, numbers.Integral):
            raise TypeError("a count must be an integer")

    def __str__(self) -> str:
        return str(int(self.value))


@dataclass(frozen=True)
class Hash:
    """A full SHA 256 in hex, so a short run of digits such as a price in cents cannot pass as one."""
    hex: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.hex):
            raise ValueError("a hash must be 64 lower case hex digits")

    def __str__(self) -> str:
        return self.hex


@dataclass(frozen=True)
class Seconds:
    """A timing of this system, such as a run time or a lag, never a market figure."""
    value: float

    def __str__(self) -> str:
        return f"{self.value:.3f} s"


@dataclass(frozen=True)
class Word:
    """A state or verdict word without digits, such as consolidated or inconclusive."""
    text: str

    def __post_init__(self) -> None:
        check_words(self.text)

    def __str__(self) -> str:
        return self.text


def check_words(text: str) -> str:
    """Refuse a state word, title or label that holds anything but lower case letters, spaces and underscores."""
    if not isinstance(text, str) or not WORD.fullmatch(text):
        raise ValueError("a state word, title or label holds only lower case letters, spaces and underscores")
    return text


def format_value(value: Any) -> str:
    """Format a summary value, where a bool is a verdict. A bare integer is refused, because it may be a volume."""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (Count, Hash, Seconds, Word)):
        return str(value)
    raise TypeError(f"a {type(value).__name__} is not a verdict or a Count, Hash, Seconds or Word, "
                    f"so it belongs in the detail rows")


@dataclass(frozen=True)
class Report:
    title: str
    summary: tuple[tuple[str, Any], ...] = ()
    rows: tuple[dict[str, Any], ...] = ()


def table(rows: tuple[dict[str, Any], ...]) -> list[str]:
    """Align rows into columns named by their keys, in order of first appearance."""
    if not rows:
        return ["no rows"]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    cells = [columns, *([str(row.get(column, "")) for column in columns] for row in rows)]
    widths = [max(len(line[i]) for line in cells) for i in range(len(columns))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in cells]


def render(report: Report, detail: bool = False) -> str:
    """The title and summary, then the rows only when detail is set, with any registered secret masked."""
    lines = [check_words(report.title), *(f"{check_words(label)}: {format_value(value)}"
                                         for label, value in report.summary)]
    if detail:
        lines += table(report.rows)
    elif report.rows:
        lines.append(f"{len(report.rows)} rows hidden, pass --detail in your own terminal to see them")
    return redact("\n".join(lines))


def emit(report: Report, detail: bool = False, stream: TextIO | None = None) -> None:
    print(render(report, detail), file=stream or sys.stdout)


def add_detail_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detail", action="store_true", help=DETAIL_HELP)
