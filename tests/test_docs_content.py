"""Content checks for the documentation: the replay rule is stated word for word, the README and the docs folder
link to each other completely, and every relative link points at a file and heading that exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from btc_survivor.replay import NEVER_FOR_SELECTION
from helpers import ROOT

REPLAY_SENTENCE = "Replay results are never used to choose, tune, or rank strategies."
README = ROOT / "README.md"
DOCS = sorted((ROOT / "docs").glob("*.md"))
LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
EXTERNAL = ("http://", "https://", "mailto:")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _outside_code(text: str) -> list[str]:
    lines, in_code = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            lines.append(line)
    return lines


def _links(path: Path) -> list[str]:
    return [m.group(1) for line in _outside_code(path.read_text(encoding="utf-8")) for m in LINK.finditer(line)]


def _local_target(source: Path, target: str) -> tuple[Path, str]:
    """The file a relative link points at, resolved from the linking file, and its anchor without the hash."""
    path, _, anchor = target.partition("#")
    return ((source.parent / path).resolve() if path else source.resolve()), anchor


def _slugs(path: Path) -> set[str]:
    """Heading anchors the way GitHub builds them, lower case with punctuation dropped and spaces as hyphens."""
    slugs: set[str] = set()
    seen: dict[str, int] = {}
    for line in _outside_code(path.read_text(encoding="utf-8")):
        heading = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not heading:
            continue
        slug = re.sub(r"[^\w\- ]", "", heading.group(1).strip().lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        slugs.add(slug if count == 0 else f"{slug}-{count}")
    return slugs


def _readme_targets() -> set[Path]:
    return {_local_target(README, t)[0] for t in _links(README) if not t.startswith(EXTERNAL)}


def test_replay_module_constant_is_the_sentence():
    assert NEVER_FOR_SELECTION == REPLAY_SENTENCE


@pytest.mark.parametrize("path", [ROOT / "docs" / "learning.md", README, ROOT / "src" / "btc_survivor" / "replay.py"],
                         ids=lambda p: p.name)
def test_replay_sentence_appears_word_for_word(path):
    assert path.exists(), f"{path.relative_to(ROOT)} is missing"
    assert REPLAY_SENTENCE in _flat(path.read_text(encoding="utf-8")), \
        f"{path.relative_to(ROOT)} must contain the sentence: {REPLAY_SENTENCE}"


PROMISED_DOCS = {"architecture.md", "learning.md", "execution_model.md", "evolution.md", "operations.md",
                 "raspberry_pi.md", "honesty_and_limits.md", "RESULTS_TEMPLATE.md"}


def test_docs_folder_holds_every_doc_the_plan_promises():
    """PLAN section 5 and Phase 9 list these eight files, so a doc that was never written fails here."""
    missing = sorted(PROMISED_DOCS - {p.name for p in DOCS})
    assert not missing, f"docs/ is missing {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_readme_links_every_docs_file(doc):
    assert doc.resolve() in _readme_targets(), f"README.md does not link docs/{doc.name}"


def test_every_docs_file_linked_from_the_readme_exists():
    docs_dir = (ROOT / "docs").resolve()
    linked = {t for t in _readme_targets() if t.parent == docs_dir}
    assert linked, "README.md links no docs files"
    missing = sorted(str(t.relative_to(ROOT)) for t in linked if not t.is_file())
    assert not missing, f"README.md links docs files that do not exist: {missing}"
    assert linked == {p.resolve() for p in DOCS}


@pytest.mark.parametrize("source", [README, *DOCS], ids=lambda p: p.name)
def test_relative_links_resolve(source):
    problems = []
    for target in _links(source):
        if target.startswith(EXTERNAL):
            continue
        path, anchor = _local_target(source, target)
        if not path.exists():
            problems.append(f"{target} points at a missing file")
        elif anchor and path.suffix == ".md" and anchor not in _slugs(path):
            problems.append(f"{target} points at a missing heading")
    assert not problems, f"{source.name}\n" + "\n".join(problems)


def test_slugs_follow_github_rules(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("# A plan\n\n## 11.7 Road map after version 2\n\n```\n# not a heading\n```\n\n## Genes\n\n## Genes\n",
                    encoding="utf-8")
    assert _slugs(page) == {"a-plan", "117-road-map-after-version-2", "genes", "genes-1"}
    assert _links(page) == []
