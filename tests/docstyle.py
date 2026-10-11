"""Checks for the documentation rules: plain prose without colons, semicolons, hyphens, em dashes or emojis,
and every section ending with links. Code blocks, inline code, link targets and URLs are exempt."""

from __future__ import annotations

import re

_EMOJI = re.compile("[\\U0001F000-\\U0001FAFF\\u2300-\\u23FF\\u25A0-\\u25FF\\u2600-\\u27BF\\u2B00-\\u2BFF\\uFE0F]")
_DASHES = re.compile("[\\u2012-\\u2015]")
_HYPHENS = re.compile("[\\u2010\\u2011\\u2212]")


def _prose(line: str) -> str:
    text = re.sub(r"`[^`]*`", "", line)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*([-*+]|\d+\.)\s", "", text)
    return re.sub(r"^#+\s", "", text)


def _is_table_rule(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("|-: ") and "-" in stripped


def prose_problems(text: str) -> list[str]:
    """Return one message per prose line that breaks the punctuation rules."""
    problems: list[str] = []
    in_code = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or _is_table_rule(line):
            continue
        prose = _prose(line)
        found = []
        if ":" in prose:
            found.append("colon")
        if ";" in prose:
            found.append("semicolon")
        if _DASHES.search(prose):
            found.append("dash")
        if "-" in prose or _HYPHENS.search(prose):
            found.append("hyphen")
        if _EMOJI.search(prose):
            found.append("emoji")
        if found:
            problems.append(f"line {number} has {', '.join(found)}: {line.strip()[:120]}")
    return problems


def section_link_problems(text: str) -> list[str]:
    """Every section with a body must end with a line that holds at least one markdown link."""
    problems: list[str] = []
    in_code = False
    heading = None
    last_line = None
    sections: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            last_line = line
            continue
        if not in_code and line.startswith("#"):
            if heading is not None:
                sections.append((heading, last_line))
            heading, last_line = line.strip(), None
            continue
        if line.strip():
            last_line = line
    if heading is not None:
        sections.append((heading, last_line))
    for heading, last in sections:
        if last is not None and "](" not in last:
            problems.append(f"section '{heading}' does not end with a link")
    return problems
